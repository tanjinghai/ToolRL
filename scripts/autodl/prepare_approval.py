#!/usr/bin/env python3
"""Create a concrete reviewable proposal; never sets approved=true."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolrl_pipeline.io import digest, write_json
from toolrl_training.config import ROOT, load
from toolrl_training.executor import validate_plan

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan', default='configs/experiment_plan.yaml')
    p.add_argument('--output', required=True)
    a=p.parse_args()
    plan=load(a.plan)
    validate_plan(plan)
    proposal=load(ROOT/'configs/phase3/approval.example.json')
    proposal.update(plan_sha256=digest(plan), data_version=plan['data_version'],model_revision=plan['model_revision'],
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    write_json(a.output,proposal)
    print('Unapproved proposal saved. Fill actual price/bill/start time/limits and obtain explicit human approval before --execute.')
