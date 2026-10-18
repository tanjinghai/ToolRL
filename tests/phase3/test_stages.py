import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from toolrl_pipeline.io import digest,file_hash,write_json,loads,read_jsonl
from toolrl_training.config import ROOT,load
from toolrl_training.executor import execute,artifact_digest
from toolrl_training.stage import train,final,model_path
from toolrl_training.decision import decide_e1


class StageTests(unittest.TestCase):
    def test_e4_manual_gate_and_final_lock_without_model(self):
        plan=load(ROOT/'configs/experiment_plan.yaml')
        plan['e4_review']='/tmp/toolrl-no-real-review-file'
        stage=next(s for s in plan['stages'] if s['id']=='E4')
        payload=dict(plan=plan,stage=stage,approval={'gpu_count':1},is_mock=False,attempt='/tmp/unused')
        self.assertEqual(train(payload)['status'],'stopped')
        with patch('toolrl_training.stage.read_jsonl') as read:
            self.assertEqual(final(dict(payload,approval={}))['status'],'stopped')
            read.assert_not_called()

    def test_model_manifest_revision_hash_and_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'fixture.safetensors').write_bytes(b'CPU fixture, never loaded')
            manifest=dict(revision='r1',files={'fixture.safetensors':file_hash(path/'fixture.safetensors')})
            write_json(path/'MODEL_MANIFEST.json',manifest)
            payload=dict(plan={'model_path':d,'model_revision':'r1'},is_mock=True)
            self.assertEqual(model_path(payload,'base')[0],path)
            payload['plan']['model_revision']='r2'
            with self.assertRaises(ValueError):model_path(payload,'base')
            payload['plan']['model_revision']='r1';(path/'fixture.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):model_path(payload,'base')

    def test_full_mock_evaluation_comparison_badcases_api_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            plan=load(ROOT/'configs/phase3/mock_plan.yaml')
            plan['backup_root']=str(Path(d)/'external-backup')
            # All real 344 validation and 597 API records; no final Test.
            pp=Path(d)/'plan.json';write_json(pp,plan)
            run=Path(d)/'run'
            state=execute(pp,run,mock=True)
            self.assertTrue(all(r['status']=='completed' for k,r in state['stages'].items() if k!='final'))
            self.assertEqual(state['stages']['final']['status'],'skipped')
            e1=Path(state['stages']['E1_eval']['directory'])
            summary=loads((e1/'metrics.json').read_text())
            self.assertEqual(summary['samples'],344)
            self.assertTrue(summary['is_mock'])
            api=loads((e1/'api_bank/metrics.json').read_text())
            self.assertEqual(api['samples'],597)
            self.assertEqual(api['valid_ground_truth_subset']['samples'],479)
            self.assertTrue((e1/'bad_cases.md').is_file())
            for stage_id,previous in [('E1_eval','E0'),('E2_eval','E0'),('E3_eval','E1_eval'),('E3_eval','E2_eval'),('E4_eval','E2_eval'),('E4_eval','E3_eval')]:
                path=Path(state['stages'][stage_id]['directory'])/'comparisons'/previous/'metrics.json'
                self.assertIsNotNone(loads(path.read_text())['comparison'])
            decision=loads((Path(state['stages']['decision_e1']['directory'])/'decision_e1.json').read_text())
            self.assertEqual(decision['source_split'],'validation')
            self.assertEqual(decision['e2'],'independent')
            before={k:r.get('artifact_sha256') for k,r in state['stages'].items()}
            resumed=execute(pp,run,mock=True)
            self.assertEqual({k:r.get('artifact_sha256') for k,r in resumed['stages'].items()},before)
            self.assertFalse((run/'FINAL_TEST_STARTED.json').exists())
            # A changed checkpoint manifest must invalidate completed artifacts, too.
            e1train=Path(state['stages']['E1']['directory'])
            checkpoint=next(e1train.rglob('CHECKPOINT.json'));checkpoint.write_text('{}')
            from toolrl_training.approval import StopRun
            with self.assertRaises(StopRun):execute(pp,run,mock=True)

    def test_decision_rejects_test(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            write_json(path/'metrics.json',{'dataset_file_sha256':'same','generation_config':{}})
            from toolrl_pipeline.io import write_jsonl
            write_jsonl(path/'scored.jsonl',[{'split':'test'}])
            with self.assertRaises(ValueError):decide_e1(path,path,path,{})

class FinalOnceTests(unittest.TestCase):
    def test_frozen_candidate_gate_one_round_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            freeze=root/'frozen.json'
            plan={'data_version':'fixture','data_root':'never-opened','final_freeze':str(freeze)}
            frozen={'data_version':'fixture','plan_sha256':digest(plan),'candidates':[{'source':'base','revision':'r1'}]}
            write_json(freeze,frozen)
            approval={'allow_final_test':True,'final_freeze_sha256':file_hash(freeze)}
            payload={'plan':plan,'approval':approval,'attempt':str(root/'attempt'),'run_dir':str(root/'run'),
                     'is_mock':False,'stage':{'id':'final'}}
            Path(payload['run_dir']).mkdir()
            with patch('toolrl_training.stage.model_path',return_value=(root,'r1')),patch('toolrl_training.stage.evaluate',return_value={'status':'completed'}) as evaluate:
                self.assertEqual(final(payload)['status'],'completed')
                evaluate.assert_called_once()
                payload['run_dir']=str(root/'other_run');Path(payload['run_dir']).mkdir()
                with self.assertRaises(FileExistsError):final(payload)
                self.assertEqual(evaluate.call_count,1)

    def test_candidate_mismatch_before_opening_test(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);freeze=root/'freeze.json'
            plan={'data_version':'fixture','data_root':'never-opened','final_freeze':str(freeze)}
            write_json(freeze,{'data_version':'fixture','plan_sha256':digest(plan),'candidates':[{'source':'base','revision':'r1'}]})
            payload={'plan':plan,'approval':{'allow_final_test':True,'final_freeze_sha256':file_hash(freeze)},
                     'attempt':d,'run_dir':d,'is_mock':False,'stage':{'id':'final'}}
            with patch('toolrl_training.stage.model_path',return_value=(root,'different')),patch('toolrl_training.stage.evaluate') as evaluate:
                with self.assertRaises(ValueError):final(payload)
                evaluate.assert_not_called()
