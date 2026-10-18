#!/usr/bin/env python3
"""Check copied data, disk, clean code and output layout. Never reads secret files."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolrl_training.config import ROOT, load, verify_data
from toolrl_training.executor import validate_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', default='configs/experiment_plan.yaml')
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--output-root', required=True)
    a = p.parse_args()
    from toolrl_training.approval import verify_git
    verify_git(ROOT, a.expected_commit)
    plan = load(a.plan)
    validate_plan(plan)
    verify_data(plan['data_root'], plan['manifests'], plan['data_version'])
    # Inspect names only, not personal credentials or their contents.
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    forbidden = [x for x in tracked if Path(x).name in ('secret.json', '.env', 'id_rsa', 'id_ed25519') or x.endswith(('.pem', '.key', '.safetensors', '.pth'))]
    if forbidden:
        raise RuntimeError('Forbidden tracked artifact names detected (values not read)')
    out = Path(a.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 128 * 1024**3:
        raise RuntimeError('Insufficient checkpoint/backup disk reserve')
    (out/'cache').mkdir(exist_ok=True)
    (out/'runs').mkdir(exist_ok=True)
    print('Git/data/paths checked; model provenance and approval checked before live stages. GPU training remains unverified.')


if __name__ == '__main__':
    main()
