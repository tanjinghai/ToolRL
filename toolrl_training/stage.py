"""Registered stage handlers, executed in bounded isolated process groups."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from toolrl_pipeline.io import digest, loads, read_jsonl, write_json, write_text
from .checkpoint import inventory, validate_checkpoint, file_hash
from .config import ROOT, DATA_VERSION, load, grpo_overrides, sft_overrides


def completed_directory(payload, stage_id):
    record = payload['state']['stages'][stage_id]
    if record['status'] != 'completed':
        raise ValueError('Required stage incomplete: ' + stage_id)
    return Path(record['directory'])


def model_path(payload, source):
    if source == 'base':
        path = Path(payload['plan']['model_path'])
        manifest = loads((path/'MODEL_MANIFEST.json').read_text())
        if manifest['revision'] != payload['plan']['model_revision']:
            raise ValueError('Base model revision mismatch')
        if not payload['is_mock'] and file_hash(path/'MODEL_MANIFEST.json') != payload['approval']['model_manifest_sha256']:
            raise ValueError('Base model approval hash mismatch')
        if not manifest['files'] or not any(k.endswith(('.safetensors', '.bin')) for k in manifest['files']):
            raise ValueError('Complete model weights are required')
        for rel, expected in manifest['files'].items():
            p = (path/rel).resolve()
            if path.resolve() not in p.parents or p.is_symlink() or file_hash(p) != expected:
                raise ValueError('Model file hash/path failure')
        revision = digest(manifest)
    else:
        record = loads((completed_directory(payload, source)/'model.json').read_text())
        path, revision = Path(record['path']), record['revision']
        if digest(validate_checkpoint(path)) != revision:
            raise ValueError('Exported model identity mismatch')
    return path, revision


def evaluate(payload, model_source=None, dataset=None, output=None, allow_test=False):
    from toolrl_pipeline.evaluation import make_run, generate, score, mock_backend
    stage, plan = payload['stage'], payload['plan']
    out = output or Path(payload['attempt'])
    dataset = Path(dataset or Path(plan['data_root'])/'eval/validation.jsonl')
    records = read_jsonl(dataset)
    if any(r['split'] == 'test' for r in records) and not allow_test:
        raise ValueError('Final Test remains locked')
    source = model_source or stage['model_from']
    if payload['is_mock']:
        model, revision = source, 'mock-' + source
        backend = mock_backend(records, stage.get('mock_mode', 'mixed'))
        backend_name, backend_config = 'mock', {}
    else:
        from toolrl_pipeline.generation_backends import VLLMBackend
        model, revision = model_path(payload, source)
        backend_config = {'max_model_len': 6144, 'tensor_parallel_size': 1,
                          'dtype': 'bfloat16', 'gpu_memory_utilization': .5}
        backend = VLLMBackend(str(model), **backend_config)
        backend_name = 'vllm'
    run = make_run(dataset, Path(payload['run_dir']).name+'-'+stage['id']+'-'+Path(out).name,
                   str(model), revision, backend=backend_name, backend_config=backend_config)
    generate(dataset, out, run, backend)
    previous = stage.get('compare_to')
    comparisons = previous if isinstance(previous, list) else [previous] if previous else []
    score(dataset, out/'predictions.jsonl', out, allow_final_test=allow_test)
    for other in comparisons:
        score(dataset, out/'predictions.jsonl', out/'comparisons'/other,
              previous_results=completed_directory(payload, other)/'scored.jsonl', allow_final_test=allow_test)
    if not allow_test and stage.get('api_bank', False):
        api_dataset = Path(plan['data_root'])/'benchmarks/api_bank.jsonl'
        api_rows = read_jsonl(api_dataset)
        api_backend = mock_backend(api_rows, 'mixed') if payload['is_mock'] else backend
        api_run = make_run(api_dataset, run['run_id']+'-api', str(model), revision,
                           backend=backend_name, backend_config=backend_config)
        generate(api_dataset, out/'api_bank', api_run, api_backend)
        summary = score(api_dataset, out/'api_bank/predictions.jsonl', out/'api_bank')
        if summary['samples'] != 597 or summary['valid_ground_truth_subset']['samples'] != 479:
            raise ValueError('API-Bank expected full597/valid479 denominator mismatch')
    return {'status': 'completed', 'is_mock': payload['is_mock'], 'model_revision': revision}


def train(payload):
    from .config import verify_data
    stage, plan, out = payload['stage'], payload['plan'], Path(payload['attempt'])
    if payload['is_mock']:
        def writer(tmp):
            write_json(tmp/'mock_weights.json', {'model': stage['id'], 'not_real_weights': True})
        from .checkpoint import publish_checkpoint
        export = publish_checkpoint(out/'checkpoints/global_step_1', writer, {'is_mock': True})
    else:
        verify_data(plan['data_root'], plan['manifests'], plan['data_version'])
        if file_hash(ROOT/stage['config']) != stage['config_sha256']:
            raise ValueError('Training configuration changed after plan validation')
        c = load(ROOT/stage['config'])
        if c['gpus'] != payload['approval']['gpu_count']:
            raise ValueError('Training world size differs from approval GPU count')
        # E3's preapproved diagnostic branch uses the same rollout/batch, fewer updates.
        if stage.get('decision_from'):
            decision = loads((completed_directory(payload, stage['decision_from'])/'decision_e1.json').read_text())
            action = decision['e3']
            if action in ('skip', 'manual'):
                return {'status': 'skipped' if action == 'skip' else 'stopped', 'reason': 'E3 validation decision: '+action}
            if action == 'diagnostic':
                c = dict(c, updates=stage['diagnostic_updates'])
        if stage['id'] == 'E4':
            evidence_path = plan.get('e4_review')
            if not evidence_path or not Path(evidence_path).is_file():
                return {'status': 'stopped', 'reason': 'E4 requires real validation bad-case review and explicit approval'}
            evidence = loads(Path(evidence_path).read_text())
            if file_hash(evidence_path) != payload['approval'].get('e4_review_sha256'):
                raise ValueError('E4 evidence not approved')
            probes = loads((completed_directory(payload, 'reward_gate')/'reward_probes.json').read_text())
            badcases = completed_directory(payload, 'E2_eval')/'scored.jsonl'
            if (evidence.get('source_split') != 'validation' or evidence.get('reward_version') != 'strict_matching'
                    or evidence.get('badcase_sha256') != file_hash(badcases)
                    or probes['strict_fullscore_false_positives'] or evidence.get('approved_candidate') is not True):
                raise ValueError('E4 validation/reward agreement gate failed')
            from .reward_probes import audit_validation_rewards
            audit = audit_validation_rewards(Path(plan['data_root'])/'eval/validation.jsonl', badcases)
            write_json(out/'reward_validation_audit.json', audit)
            if audit['fullscore_false_positives']:
                return {'status': 'stopped', 'reason': 'Candidate reward conflicts with independent verifier on real validation',
                        'audit': audit}
        if stage.get('reward_version'):
            c = dict(c, reward_version=stage['reward_version'])
        if c.get('reward_version') == 'legacy' and not payload['approval'].get('allow_legacy_reward_diagnostic', False):
            raise ValueError('Legacy full-score false positives require explicit diagnostic approval')
        model, initialization_revision = model_path(payload, stage['model_from'])
        identity = {'run_id': Path(payload['run_dir']).name, 'stage': stage['id'],
                    'git_commit': payload['approval']['git_commit'], 'data_version': plan['data_version'],
                    'base_model_revision': plan['model_revision'], 'initialization_revision': initialization_revision,
                    'config_sha256': digest(c), 'reward_version': c.get('reward_version'),
                    'reference_initialization': str(model), 'plan_sha256': digest(plan)}
        write_json(out/'run_identity.json', identity)
        os.environ['TOOLRL_RUN_METADATA'] = str(out/'run_identity.json')
        checkpoint_root = out/'checkpoints'
        os.environ['TOOLRL_DATA_ROOT'] = str(Path(plan['data_root']).resolve())
        write_json(out/'resolved_config.json', c)
        if c['kind'] == 'sft':
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node='+str(c['gpus']), '-m', 'verl.trainer.fsdp_sft_trainer']
            cmd += sft_overrides(c, model, plan['data_root'], checkpoint_root)
            export = checkpoint_root/f"global_step_{c['updates']}"
        else:
            cmd = [sys.executable, '-m', 'verl.trainer.main_ppo']
            cmd += grpo_overrides(c, model, plan['data_root'], checkpoint_root)
            export = checkpoint_root/'actor'/f"global_step_{c['updates']}"
        write_json(out/'command.json', cmd)
        subprocess.run(cmd, cwd=ROOT, check=True)
        validate_checkpoint(export)
        for rel in ('config.json', 'tokenizer_config.json', 'training_state.pt'):
            if not (export/rel).is_file():
                raise ValueError('Incomplete trained export: ' + rel)
        if not any(export.glob('*.safetensors')) and not any(export.glob('pytorch_model*.bin')):
            raise ValueError('Missing exported weights')
    manifest = validate_checkpoint(export)
    write_json(out/'model.json', {'path': str(export), 'revision': digest(manifest), 'is_mock': payload['is_mock']})
    return {'status': 'completed', 'is_mock': payload['is_mock']}


def decision(payload):
    from .decision import decide_e1
    result = decide_e1(completed_directory(payload, 'E0'), completed_directory(payload, 'E1_eval'),
                       payload['attempt'], payload['plan']['decision_rules'])
    return {'status': 'completed', 'e3': result['e3'], 'e4': result['e4']}


def reward_gate(payload):
    from .reward_probes import run_probes
    report = run_probes()
    write_json(Path(payload['attempt'])/'reward_probes.json', report)
    return {'status': 'completed' if report['strict_fullscore_false_positives'] == 0 else 'stopped', **report}


def final(payload):
    plan, approval, out = payload['plan'], payload['approval'], Path(payload['attempt'])
    if payload['is_mock']:
        return {'status': 'skipped', 'reason': 'Mock never opens frozen final Test'}
    if not approval.get('allow_final_test'):
        return {'status': 'stopped', 'reason': 'Final Test requires separate explicit frozen-candidates approval'}
    freeze_path = Path(plan['final_freeze'])
    if file_hash(freeze_path) != approval.get('final_freeze_sha256'):
        raise ValueError('Final freeze approval mismatch')
    freeze = loads(freeze_path.read_text())
    if freeze['data_version'] != plan['data_version'] or freeze['plan_sha256'] != digest(plan):
        raise ValueError('Final freeze data/plan mismatch')
    candidates = freeze['candidates']
    if not candidates or len({x['source'] for x in candidates}) != len(candidates):
        raise ValueError('Final candidates must be nonempty and unique')
    # Validate every candidate before opening any Test record.
    for candidate in candidates:
        _, revision = model_path(payload, candidate['source'])
        if revision != candidate['revision']:
            raise ValueError('Final candidate changed after freeze')
    # Claim at the fixed freeze path too, so a different run_dir cannot trigger a second round.
    claim = freeze_path.with_name(freeze_path.name + '.TEST_STARTED.json')
    with claim.open('x') as stream:
        stream.write(freeze_path.read_text())
    lock = Path(payload['run_dir'])/'FINAL_TEST_STARTED.json'
    with lock.open('x') as stream:
        stream.write(freeze_path.read_text())
    for candidate in candidates:
        evaluate(payload, candidate['source'], Path(plan['data_root'])/'eval/test.jsonl',
                 out/candidate['source'], allow_test=True)
    return {'status': 'completed', 'final_one_round': True}


def backup(payload):
    run, out = Path(payload['run_dir']), Path(payload['attempt'])
    destination = Path(payload['plan']['backup_root']).resolve()/run.name
    if run == destination or run in destination.parents or destination.exists():
        raise ValueError('Backup needs a new destination outside the run directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    if shutil.disk_usage(destination.parent).free < required * 1.1:
        raise ValueError('Insufficient backup capacity; original run retained')
    destination.mkdir()
    for item in run.iterdir():
        if item.name in (payload['stage']['id'], 'executor.lock'):
            continue
        target = destination/item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    source_files = {k: v for k, v in inventory(run, exclude_checkpoint=False).items()
                    if not k.startswith(payload['stage']['id']+'/') and k != 'executor.lock'}
    if inventory(destination, exclude_checkpoint=False) != source_files:
        raise ValueError('Backup readback mismatch')
    write_json(out/'backup.json', {'destination': str(destination), 'files': source_files,
                                 'off_instance': 'not asserted; user must choose persistent external storage'})
    return {'status': 'completed', 'backup_path': str(destination)}


def environment(payload):
    subprocess.run([sys.executable, 'scripts/autodl/check_env.py', '--gpu',
                    '--expected-gpu', payload['approval']['gpu_model'],
                    '--gpu-count', str(payload['approval']['gpu_count']),
                    '--output', str(Path(payload['attempt'])/'environment.json')], check=True)
    return {'status': 'completed', 'gpu_checks_executed': True}


def fixture(payload):
    if not payload['is_mock']:
        raise ValueError('Fixture requires mock provenance')
    stage = payload['stage']
    if stage.get('sleep_seconds'):
        time.sleep(stage['sleep_seconds'])
    if stage.get('fail_until_attempt', 0) >= len(payload['state']['stages'][stage['id']]['attempts']):
        raise RuntimeError('Intentional CPU fixture failure')
    write_json(Path(payload['attempt'])/'fixture.json', {'is_mock': True})
    return {'status': 'completed', 'is_mock': True}


HANDLERS = {'evaluate': evaluate, 'train': train, 'decision': decision, 'reward_gate': reward_gate,
            'final': final, 'backup': backup, 'environment': environment, 'fixture': fixture}


def main():
    payload = loads(Path(sys.argv[1]).read_text())
    result = HANDLERS[payload['stage']['kind']](payload)
    write_json(Path(payload['attempt'])/'result.json', result)


if __name__ == '__main__':
    main()
