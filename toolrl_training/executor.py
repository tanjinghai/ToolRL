"""A small stateful DAG executor. Only registered handlers, never shell plans."""
import os
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from toolrl_pipeline.io import digest, file_hash, loads, write_json, write_text
from .approval import StopRun, budget_check, validate_approval, verify_git
from .config import ROOT, load, verify_data
from .checkpoint import inventory

KINDS = {'evaluate', 'train', 'decision', 'reward_gate', 'final', 'backup', 'environment', 'fixture'}


def ordered_stages(plan):
    stages = plan['stages']
    ids = [s['id'] for s in stages]
    if len(ids) != len(set(ids)) or any(not x.replace('_', '').isalnum() for x in ids):
        raise ValueError('Unique alphanumeric stage IDs required')
    pending, ordered = {s['id']: s for s in stages}, []
    while pending:
        ready = [s for s in pending.values() if set(s.get('deps', [])) <= {x['id'] for x in ordered}]
        if not ready:
            raise ValueError('Cyclic or missing stage dependency')
        for stage in ready[:1]:
            if (stage['kind'] not in KINDS or type(stage['timeout_seconds']) not in (int, float)
                    or not math.isfinite(stage['timeout_seconds']) or stage['timeout_seconds'] <= 0
                    or type(stage.get('retries', 0)) is not int or stage.get('retries', 0) < 0
                    or type(stage['cost_cap_rmb']) not in (int, float) or not math.isfinite(stage['cost_cap_rmb'])
                    or stage['cost_cap_rmb'] < 0):
                raise ValueError('Invalid registered stage limits/kind')
            ordered.append(stage)
            pending.pop(stage['id'])
    return ordered


def validate_plan(plan):
    stages = ordered_stages(plan)
    for s in stages:
        if s['kind'] == 'train':
            config_path = ROOT/s['config']
            if s['config_sha256'] != file_hash(config_path):
                raise ValueError('Training config changed: ' + s['id'])
        if s['kind'] == 'fixture' and not plan.get('is_mock'):
            raise ValueError('Fixture handler cannot enter GPU plan')
    # E2/E3 are controlled comparisons. Only E4 may switch reward from the same init.
    by_id = {s['id']: s for s in stages}
    if all(k in by_id and by_id[k]['kind'] == 'train' for k in ('E2', 'E3')):
        a, b = [load(ROOT/by_id[k]['config']) for k in ('E2', 'E3')]
        if a != b or by_id['E2']['model_from'] != 'base' or by_id['E3']['model_from'] != 'E1':
            raise ValueError('E2/E3 must share exact RL configuration, initialized from Base/E1')
    if 'E4' in by_id:
        if by_id['E4'].get('model_from') != 'base':
            raise ValueError('E4 must restart from the E2 base, not continue E2 steps')
        if 'E2' in by_id and load(ROOT/by_id['E4']['config']) != load(ROOT/by_id['E2']['config']):
            raise ValueError('E4 must match E2 initialization budget/config, changing only the registered reward')
    return stages


def artifact_digest(path):
    return digest(inventory(path, exclude_checkpoint=False))


def launch(payload, attempt, timeout, guard):
    """Separate process group; parent can stop a hung Ray/torchrun subprocess tree."""
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt/'payload.json', payload)
    with (attempt/'stdout.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'toolrl_training.stage', str(attempt/'payload.json')],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(attempt/'process.json', {'pid': proc.pid, 'started_at': time.time()})
        start = time.monotonic()
        try:
            while proc.poll() is None:
                if time.monotonic() - start >= timeout:
                    raise StopRun('Stage timeout')
                guard()
                time.sleep(.1)
            if proc.returncode:
                raise RuntimeError(f'Stage subprocess exit={proc.returncode}; see {attempt}/stdout.log')
        finally:
            # Also reap remaining local children on success; a Ray daemon is not a completion signal.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    result = loads((attempt/'result.json').read_text())
    if result.get('status') not in ('completed', 'skipped', 'stopped'):
        raise RuntimeError('Stage missing a terminal result')
    return result


def execute(plan_path, run_dir, approval=None, execute_paid=False, mock=False):
    plan = load(plan_path)
    stages = validate_plan(plan)
    if mock != bool(plan.get('is_mock')):
        if execute_paid or mock:
            raise StopRun('Mock/real plan mismatch')
    if not execute_paid and not mock:
        return {'mode': 'dry-run', 'plan_sha256': digest(plan), 'stages': [s['id'] for s in stages],
                'paid_actions_started': False, 'approval_required': True}
    if mock and execute_paid:
        raise StopRun('Mock cannot authorize GPU execution')
    if execute_paid:
        if approval is None or approval.get('approved') is not True:
            raise StopRun('No explicit approval: use dry-run')
        git_sha = verify_git(ROOT, approval['git_commit'])
        validate_approval(approval, plan, git_sha)
        verify_data(plan['data_root'], plan['manifests'], plan['data_version'])
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir/'executor.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise StopRun('Run locked; inspect old process before manually removing stale lock') from exc
    try:
        state_path = run_dir/'state.json'
        execution_identity = ({k: approval[k] for k in ('git_commit', 'data_version', 'model_revision', 'model_manifest_sha256')}
                              if execute_paid else {'is_mock': True})
        identity = {'plan_sha256': digest(plan), 'approval_sha256': digest(approval), 'is_mock': mock}
        if state_path.exists():
            state = loads(state_path.read_text())
            if state['identity'] != identity:
                if (not execute_paid or state['identity']['plan_sha256'] != identity['plan_sha256']
                        or state.get('execution_identity') != execution_identity):
                    raise StopRun('Run identity changed; create a new run directory')
                state.setdefault('approval_history', []).append(state['identity']['approval_sha256'])
                state['identity'] = identity
        else:
            state = {'identity': identity, 'execution_identity': execution_identity, 'started_at': time.time(), 'stages': {}}
        if execute_paid:
            approved_path = run_dir/'approvals'/f"{digest(approval)}.json"
            if not approved_path.exists():
                write_json(approved_path, approval)
        for s in stages:
            state['stages'].setdefault(s['id'], {'status': 'planned', 'attempts': []})
        write_json(state_path, state)
        for stage in stages:
            record = state['stages'][stage['id']]
            if record['status'] == 'completed':
                if artifact_digest(record['directory']) != record['artifact_sha256']:
                    raise StopRun('Completed artifact changed: ' + stage['id'])
                continue
            if record['status'] == 'skipped':
                continue
            if record['status'] == 'stopped' and record.get('result', {}).get('status') == 'stopped':
                # Manual decisions are resumable only with an explicit renewed authorization.
                if not execute_paid or record.get('stopped_approval_sha256') == identity['approval_sha256']:
                    continue
                if stage['kind'] == 'final' and (run_dir/'FINAL_TEST_STARTED.json').exists():
                    raise StopRun('Final Test already attempted; no automatic second round')
                record['status'] = 'planned'
                record['attempt_limit_offset'] = len(record['attempts'])
            # Backup is a finally-stage, even when normal dependencies fail.
            deps = [state['stages'][x]['status'] for x in stage.get('deps', [])]
            if stage['kind'] != 'backup' and any(x != 'completed' for x in deps):
                record.update(status='skipped', reason='dependency not completed')
                write_json(state_path, state)
                continue
            try:
                if execute_paid:
                    verify_git(ROOT, approval['git_commit'])
                    if stage['id'] not in approval['allowed_stages']:
                        raise StopRun('Stage absent from approval whitelist: ' + stage['id'])
                    if stage.get('retries', 0) > approval['max_retries']:
                        raise StopRun('Retry count exceeds approval')
                    validate_approval(approval, plan, git_sha)
                    # Count initialization/idle since actual instance start, not executor start.
                    started = approval.get('instance_started_at_unix')
                    if type(started) not in (int, float) or not math.isfinite(started) or started <= 0 or started > state['started_at']:
                        raise StopRun('Actual instance start time required, including setup')
                    remaining = sum(s['timeout_seconds'] for s in stages
                                    if state['stages'][s['id']]['status'] == 'planned')
                    ledger = budget_check(approval, started, stage, remaining)
                    write_json(run_dir/'budget.json', ledger)
                else:
                    started = state['started_at']
                if record['status'] == 'running':
                    pidfile = Path(record['directory'])/'process.json'
                    if not pidfile.is_file():
                        raise StopRun('Interrupted stage has unknown process identity; inspect manually')
                    pid = loads(pidfile.read_text())['pid']
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        record.update(status='failed', reason='Prior process exited; bounded retry eligible')
                    else:
                        raise StopRun('Prior stage process still exists; stop and inspect before recovery')
                max_attempts = record.get('attempt_limit_offset', 0) + 1 + stage.get('retries', 0)
                while len(record['attempts']) < max_attempts:
                    attempt = run_dir/stage['id']/f"attempt_{len(record['attempts'])+1}"
                    record['attempts'].append(str(attempt))
                    record.update(status='running', directory=str(attempt))
                    write_json(state_path, state)
                    payload = {'stage': stage, 'plan': plan, 'run_dir': str(run_dir),
                               'attempt': str(attempt), 'state': state, 'approval': approval, 'is_mock': mock}
                    def guard():
                        if execute_paid:
                            validate_approval(approval, plan, git_sha)
                            budget_check(approval, started, stage, 0)
                            elapsed = record.get('runtime_seconds', 0) + time.time() - attempt_start
                            if elapsed/3600 * ledger['instance_hourly_rmb'] > stage['cost_cap_rmb']:
                                raise StopRun('Stage cost cap exceeded')
                    attempt_start = time.time()
                    try:
                        if execute_paid and record.get('runtime_seconds', 0)/3600*ledger['instance_hourly_rmb'] >= stage['cost_cap_rmb']:
                            raise StopRun('Cumulative stage cost cap exhausted')
                        result = launch(payload, attempt, stage['timeout_seconds'], guard)
                        record.update(status=result['status'], result=result, artifact_sha256=artifact_digest(attempt))
                        if result['status'] == 'stopped':
                            record['stopped_approval_sha256'] = identity['approval_sha256']
                        break
                    except StopRun:
                        raise
                    except (RuntimeError, ValueError, OSError) as exc:
                        record.update(status='failed', reason=str(exc))
                        write_json(state_path, state)
                        if len(record['attempts']) >= max_attempts:
                            break
                    finally:
                        record['runtime_seconds'] = record.get('runtime_seconds', 0) + time.time() - attempt_start
                write_json(state_path, state)
                if execute_paid and record['status'] == 'stopped':
                    break
            except StopRun as exc:
                record.update(status='stopped', reason=str(exc))
                write_json(state_path, state)
                # No further paid work on a safety stop; CPU summary is always retained.
                break
        write_text(run_dir/'STOP_AND_BILLING.md', '# 停止与核费\n\n进程退出不代表实例停止计费。'
                   '请立即在 AutoDL 控制台停止实例并核对账单；无已验证自动关机接口。'
                   '\n保留 state.json、budget.json、全部 attempts；恢复需核对最后有效 checkpoint。\n')
        return state
    finally:
        write_text(run_dir/'STOP_AND_BILLING.md', '# 停止与核费\n\n进程退出不代表实例停止计费。请立即在 AutoDL 控制台停止实例并核对账单。\n'
                   '未实现平台自动关机；安全停止时不启动新的付费阶段。当前检查点和日志保留在本 run。\n')
        os.close(fd)
        lock.unlink()
