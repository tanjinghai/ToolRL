import copy
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

from toolrl_pipeline.io import digest, file_hash, loads, write_json
from toolrl_training.approval import StopRun, validate_approval, instance_rate, budget_check
from toolrl_training.checkpoint import publish_checkpoint, validate_checkpoint, save_cpu_state, restore_cpu_state
from toolrl_training.config import ROOT, load, batch_report, grpo_overrides, sft_overrides, verify_data
from toolrl_training.executor import execute, ordered_stages, validate_plan
from toolrl_training.watchdog import TrainingWatchdog, is_final_update


class ConfigTests(unittest.TestCase):
    def test_old_worker_normalization(self):
        for name in ('grpo_smoke', 'grpo_small', 'grpo_expanded', 'grpo_legacy_kl'):
            c = load(ROOT/f'configs/phase3/{name}.json')
            report = batch_report(c)
            self.assertEqual(report['per_rank_micro'], 4)
            self.assertEqual(report['responses_per_update'], 64)
            self.assertEqual(report['accumulation'], 16)
        c = load(ROOT/'configs/phase3/grpo_smoke.json')
        self.assertEqual(batch_report(c)['effective_updates'], 10)
        self.assertEqual(batch_report(dict(c, gpus=2, micro=2, logprob_micro=2))['per_rank_micro'], 4)
        with self.assertRaises(ValueError):
            batch_report(dict(c, gpus=2))
        with self.assertRaises(ValueError):
            batch_report(dict(c, max_model_len=1024))
        with self.assertRaises(ValueError):
            batch_report(dict(c, mini=3))

    def test_kl_and_hydra(self):
        c = load(ROOT/'configs/phase3/grpo_smoke.json')
        paper = grpo_overrides(c, '/model', '/data', '/run')
        self.assertIn('algorithm.kl_ctrl.kl_coef=0.0', paper)
        self.assertIn('actor_rollout_ref.actor.use_kl_loss=false', paper)
        legacy = grpo_overrides(dict(c, kl_mode='legacy-default'), '/model', '/data', '/run')
        self.assertIn('algorithm.kl_ctrl.kl_coef=0.001', legacy)
        s = load(ROOT/'configs/phase3/sft_small.json')
        self.assertIn('data.max_length=3072', sft_overrides(s, '/model', '/data', '/run'))
        with self.assertRaises(ValueError):
            sft_overrides(dict(s, max_length=2048), '/model', '/data', '/run')

    def test_real_frozen_data(self):
        meta = verify_data(ROOT/'dataset/generated/phase2-v1', ROOT/'data_manifests/phase2-v1')
        self.assertEqual(meta['dataset_version'], 'phase2-v1-13c266c7a084')
        with self.assertRaises(ValueError):
            verify_data(ROOT/'dataset/generated/phase2-v1', ROOT/'data_manifests/phase2-v1', 'wrong')

    def test_plan_controlled_comparisons(self):
        plan = load(ROOT/'configs/experiment_plan.yaml')
        self.assertEqual(len(validate_plan(plan)), 14)
        bad = copy.deepcopy(plan)
        next(s for s in bad['stages'] if s['id']=='E3')['config_sha256']='bad'
        with self.assertRaises(ValueError):
            validate_plan(bad)
        with self.assertRaises(ValueError):
            ordered_stages({'stages':[{'id':'a','kind':'fixture','deps':['b'],'timeout_seconds':1}]})
        e2 = next(s for s in plan['stages'] if s['id']=='E2')
        self.assertNotIn('E1', e2['deps'])

    def test_exact_update_count(self):
        for total in (1, 2, 10, 64):
            updated=[]
            for step in range(1, total+2):
                updated.append(step)
                if is_final_update(step,total):
                    break
            self.assertEqual(updated,list(range(1,total+1)))
        with self.assertRaises(ValueError):
            is_final_update(0,10)

    def test_watchdog(self):
        w=TrainingWatchdog()
        for _ in range(2):
            self.assertEqual(w.observe([1,1,2,2],['a','a','b','b']),1)
        with self.assertRaises(FloatingPointError):
            w.observe([1,1,2,2],['a','a','b','b'])
        with self.assertRaises(FloatingPointError):
            TrainingWatchdog().observe([float('nan'),1],['a','a'])
        self.assertEqual(TrainingWatchdog().observe([0,1],['a','a']),0)


class CheckpointTests(unittest.TestCase):
    def test_cpu_resume_optimizer_scheduler_rng_cursor(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'run_A/step_3'
            identity={'data_version':'v1','model_revision':'base1','run_id':'A'}
            random.seed(42)
            save_cpu_state(path, optimizer={'momentum': [1,2]}, scheduler={'step':3},
                           progress={'step':3,'epoch':1,'next_batch':2}, identity=identity)
            expected=random.random()
            random.seed(99)
            restored=restore_cpu_state(path,identity)
            self.assertEqual(random.random(),expected)
            self.assertEqual(restored['optimizer']['momentum'],[1,2])
            self.assertEqual(restored['scheduler']['step'],3)
            self.assertEqual(restored['progress']['next_batch'],2)
            with self.assertRaises(ValueError):
                restore_cpu_state(path,dict(identity,model_revision='other'))
            with self.assertRaises(FileExistsError):
                publish_checkpoint(path,lambda p:write_json(p/'x.json',{}),identity)
            (path/'state.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_checkpoint(path)

    def test_atomic_failure_preserves_prior(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            publish_checkpoint(path/'step_1',lambda p:write_json(p/'x.json',{'ok':True}),{})
            def fail(p):
                write_json(p/'x.json',{'ok':False})
                raise RuntimeError('disk fixture')
            with self.assertRaises(RuntimeError):
                publish_checkpoint(path/'step_2',fail,{})
            validate_checkpoint(path/'step_1')
            self.assertFalse((path/'step_2').exists())
            self.assertEqual(len(list(path.iterdir())),1)


def approval(plan):
    return dict(approved=True, approved_by='CPU fixture only',expires_at_unix=2000,git_commit='a'*40,
                data_version=plan['data_version'],model_revision=plan['model_revision'], plan_sha256=digest(plan),
                gpu_model='A100',gpu_count=2,price_basis='instance',hourly_price_rmb=10,
                historical_spend_rmb=20,committed_other_rmb=0,project_cap_rmb=500,max_run_rmb=100,
                max_wall_seconds=3600,reserve_rmb=10,max_retries=0,allowed_stages=[s['id'] for s in plan['stages']],
                model_manifest_sha256='b'*64)


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.plan=load(ROOT/'configs/experiment_plan.yaml')
        self.a=approval(self.plan)

    def test_no_approval_dryrun_and_price_basis(self):
        out=execute(ROOT/'configs/experiment_plan.yaml','/tmp/toolrl-no-action')
        self.assertEqual(out['mode'],'dry-run')
        self.assertFalse(out['paid_actions_started'])
        self.assertEqual(instance_rate(self.a),10)
        self.assertEqual(instance_rate(dict(self.a,price_basis='per_gpu')),20)
        for key,value in [('approved',False),('historical_spend_rmb',None),('hourly_price_rmb',float('nan')),
                          ('git_commit','b'*40),('data_version','wrong'),('model_revision','other'),
                          ('plan_sha256','wrong')]:
            with self.subTest(key=key),self.assertRaises(StopRun):
                validate_approval(dict(self.a,**{key:value}),self.plan,'a'*40,now=1000)

    def test_budget_includes_idle_and_reserve(self):
        s={'timeout_seconds':60,'cost_cap_rmb':5}
        ledger=budget_check(self.a,0,s,600,now=1800)
        self.assertEqual(ledger['estimated_run_rmb'],5)
        self.assertEqual(ledger['available_rmb'],95)
        for a,now,remaining in [(self.a,3600,0),(dict(self.a,historical_spend_rmb=490),10,600),
                                (dict(self.a,reserve_rmb=101),10,0),(self.a,100,40000)]:
            with self.assertRaises(StopRun):
                budget_check(a,0,s,remaining,now=now)

    def test_whitelist_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            plan={'is_mock':False,'data_version':'v','model_revision':'m','data_root':'ignored','manifests':'ignored',
                  'stages':[dict(id='unauthorized',kind='fixture',deps=[],timeout_seconds=1,cost_cap_rmb=1)]}
            # fixture prohibited even before approval; registered real handler must also be allowlisted.
            plan['stages'][0]['kind']='evaluate'
            pp=Path(d)/'plan.json';write_json(pp,plan)
            a=approval(plan);a.update(allowed_stages=[],expires_at_unix=time.time()+60,
                                    instance_started_at_unix=time.time()-60)
            with patch('toolrl_training.executor.verify_git',return_value='a'*40),patch('toolrl_training.executor.verify_data'),patch('toolrl_training.executor.launch') as launch:
                result=execute(pp,Path(d)/'run',a,True)
                self.assertEqual(result['stages']['unauthorized']['status'],'stopped')
                launch.assert_not_called()


class ExecutorTests(unittest.TestCase):
    def plan(self,d,stages):
        p=Path(d)/'plan.json';write_json(p,dict(is_mock=True,data_version='fixture',model_revision='fixture',stages=stages))
        return p
    def s(self,id,deps=(),**kwargs):
        return dict(id=id,kind='fixture',deps=list(deps),timeout_seconds=4,cost_cap_rmb=0,retries=0,**kwargs)

    def test_retry_dependencies_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.s('A',fail_until_attempt=1);a['retries']=1
            p=self.plan(d,[a,self.s('B',['A'])]);run=Path(d)/'run'
            result=execute(p,run,mock=True)
            self.assertEqual(result['stages']['B']['status'],'completed')
            self.assertEqual(len(result['stages']['A']['attempts']),2)
            log=run/'A/attempt_1/stdout.log'
            self.assertIn('Intentional CPU fixture failure',log.read_text())
            before=file_hash(run/'B/attempt_1/result.json')
            again=execute(p,run,mock=True)
            self.assertEqual(len(again['stages']['A']['attempts']),2)
            self.assertEqual(file_hash(run/'B/attempt_1/result.json'),before)
            (run/'B/attempt_1/fixture.json').write_text('tampered')
            with self.assertRaises(StopRun):
                execute(p,run,mock=True)

    def test_failed_sft_does_not_cancel_independent_e2(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.plan(d,[self.s('E1',fail_until_attempt=5),self.s('E3',['E1']),self.s('E2')])
            result=execute(p,Path(d)/'run',mock=True)
            self.assertEqual(result['stages']['E1']['status'],'failed')
            self.assertEqual(result['stages']['E3']['status'],'skipped')
            self.assertEqual(result['stages']['E2']['status'],'completed')

    def test_timeout_stops_new_work(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.s('A',sleep_seconds=2);s['timeout_seconds']=.2
            p=self.plan(d,[s,self.s('B')]);run=Path(d)/'run'
            result=execute(p,run,mock=True)
            self.assertEqual(result['stages']['A']['status'],'stopped')
            self.assertEqual(result['stages']['B']['status'],'planned')
            self.assertTrue((run/'STOP_AND_BILLING.md').exists())

    def test_recovery_after_failed_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.s('A',fail_until_attempt=1);s['retries']=1
            p=self.plan(d,[s]);run=Path(d)/'run'
            # First invocation interrupted between retry decisions (terminal failed, no live child).
            with patch('toolrl_training.executor.launch',side_effect=StopRun('fixture interruption')):
                result=execute(p,run,mock=True)
            self.assertEqual(result['stages']['A']['status'],'stopped')
            result=execute(p,run,mock=True)
            self.assertEqual(result['stages']['A']['status'],'completed')

    def test_final_mock_cannot_unlock_test(self):
        from toolrl_training.stage import final
        result=final({'is_mock':True,'plan':{},'approval':None,'attempt':'unused'})
        self.assertEqual(result['status'],'skipped')


if __name__=='__main__':unittest.main()

class RetentionTests(unittest.TestCase):
    def test_latest_complete_survives_and_other_runs_untouched(self):
        from toolrl_training.checkpoint import prune_checkpoints
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for run in ('A','B'):
                for step in (1,2):
                    publish_checkpoint(root/run/f'global_step_{step}',lambda p:write_json(p/'state.json',{}),{})
            prune_checkpoints(root/'A',keep=1)
            self.assertFalse((root/'A/global_step_1').exists())
            validate_checkpoint(root/'A/global_step_2')
            self.assertTrue((root/'B/global_step_1').exists())
            with self.assertRaises(ValueError):prune_checkpoints(root/'A',keep=0)
