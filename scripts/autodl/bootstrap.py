#!/usr/bin/env python3
"""Budgeted installation on an already human-approved instance, not a rental API."""
import argparse
from pathlib import Path
import os
import math
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolrl_training.config import ROOT, load
from toolrl_training.approval import validate_approval, verify_git, budget_check, StopRun


def main():
    def interrupted(signum, frame):
        raise StopRun(f'Bootstrap signal {signum}; stop platform billing')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--approval',required=True)
    a=p.parse_args()
    plan,approval=load(a.plan),load(a.approval)
    sha=verify_git(ROOT,approval['git_commit'])
    validate_approval(approval,plan,sha)
    if 'environment' not in approval['allowed_stages']:
        raise StopRun('Environment setup absent from approval whitelist')
    stage=next(s for s in plan['stages'] if s['id']=='environment')
    started=approval.get('instance_started_at_unix')
    if type(started) not in (int,float) or not math.isfinite(started) or started<=0 or started>time.time():
        raise StopRun('Actual instance start required')
    subprocess.run([sys.executable,'scripts/autodl/check_env.py','--host-only'],check=True)
    commands=[[sys.executable,'-m','pip','install','-r','scripts/autodl/requirements-oldstack.txt'],
              [sys.executable,'-m','pip','install','--no-build-isolation','flash-attn==2.6.3'],
              [sys.executable,'-m','pip','install','--no-deps','-e','.'],
              [sys.executable,'-m','pip','check']]
    begin=time.time()
    logdir=ROOT/'runs/bootstrap';logdir.mkdir(parents=True,exist_ok=True)
    with (logdir/f'install_{int(begin)}.log').open('x') as log:
        for cmd in commands:
            budget_check(approval,started,stage,stage['timeout_seconds'])
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while proc.poll() is None:
                    validate_approval(approval,plan,sha)
                    budget_check(approval,started,stage,0)
                    if time.time()-begin>=stage['timeout_seconds']:
                        raise StopRun('Environment setup timeout; stop instance in platform console')
                    time.sleep(.5)
                if proc.returncode:
                    raise RuntimeError('Environment install failed; see bootstrap log and stop billing in platform console')
            finally:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    (logdir/'environment-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    subprocess.run([sys.executable,'scripts/autodl/check_env.py','--gpu','--expected-gpu',approval['gpu_model'],
                    '--gpu-count',str(approval['gpu_count']),'--output',str(logdir/'environment-gpu.json')],check=True)


if __name__=='__main__':main()
