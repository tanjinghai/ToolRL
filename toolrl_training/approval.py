"""Fail-closed run authorization and instance wall-clock budgeting (no cloud API)."""
import math
import re
import subprocess
import time
from pathlib import Path
from toolrl_pipeline.io import digest


class StopRun(RuntimeError):
    pass


def finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise StopRun(f'Known finite nonnegative {name} required')
    return value


def instance_rate(approval):
    price = finite_nonnegative(approval.get('hourly_price_rmb'), 'hourly price')
    count = approval.get('gpu_count')
    if type(count) is not int or count < 1 or price <= 0:
        raise StopRun('Positive GPU count and actual quoted price required')
    basis = approval.get('price_basis')
    if basis not in ('instance', 'per_gpu'):
        raise StopRun('Specify price_basis instance or per_gpu')
    return price * (count if basis == 'per_gpu' else 1)


def validate_approval(approval, plan, git_sha, now=None):
    now = time.time() if now is None else now
    if approval.get('approved') is not True or not approval.get('approved_by'):
        raise StopRun('No explicit approval: dry-run only')
    expiry = approval.get('expires_at_unix')
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= now:
        raise StopRun('Approval expired')
    if approval.get('git_commit') != git_sha or not re.fullmatch('[0-9a-f]{40}', git_sha):
        raise StopRun('Approved Git commit mismatch')
    for key in ('data_version', 'model_revision'):
        if approval.get(key) != plan.get(key):
            raise StopRun(f'Approval {key} mismatch')
    if approval.get('plan_sha256') != digest(plan):
        raise StopRun('Plan changed after approval')
    if not approval.get('gpu_model') or not approval.get('model_manifest_sha256'):
        raise StopRun('GPU model and model manifest hash required')
    instance_rate(approval)
    for key in ('historical_spend_rmb', 'committed_other_rmb', 'max_run_rmb', 'max_wall_seconds', 'reserve_rmb', 'project_cap_rmb'):
        finite_nonnegative(approval.get(key), key)
    if approval['project_cap_rmb'] > 500 or approval['max_run_rmb'] <= 0 or approval['max_wall_seconds'] <= 0:
        raise StopRun('Invalid project/run caps')
    if type(approval.get('max_retries')) is not int or approval['max_retries'] < 0:
        raise StopRun('Retry allowance required')
    allowed = approval.get('allowed_stages')
    if not isinstance(allowed, list) or not set(allowed) <= {s['id'] for s in plan['stages']}:
        raise StopRun('Invalid stage whitelist')
    return approval


def budget_check(approval, started_at, stage, remaining_seconds, now=None):
    now = time.time() if now is None else now
    elapsed = max(0, now - started_at)  # includes idle/setup and gaps between invocations
    rate = instance_rate(approval)
    spent = elapsed / 3600 * rate
    available = min(approval['max_run_rmb'] - spent,
                    approval['project_cap_rmb'] - approval['historical_spend_rmb'] - approval['committed_other_rmb'] - spent)
    needed = remaining_seconds / 3600 * rate * 1.3 + approval['reserve_rmb']
    stage_cost = stage['timeout_seconds'] / 3600 * rate * 1.3
    if elapsed >= approval['max_wall_seconds'] or elapsed + remaining_seconds > approval['max_wall_seconds'] or needed > available or stage_cost > stage['cost_cap_rmb']:
        raise StopRun('Budget/wall-clock gate rejected stage; preserve/backup and stop billing in platform console')
    return {'elapsed_seconds': elapsed, 'estimated_run_rmb': spent, 'available_rmb': available,
            'needed_rmb': needed, 'instance_hourly_rmb': rate, 'actual_account_bill': 'manual reconciliation required'}


def verify_git(root, expected):
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=root, text=True)
    if sha != expected or dirty:
        raise StopRun('Execute requires approved commit and clean checkout')
    return sha
