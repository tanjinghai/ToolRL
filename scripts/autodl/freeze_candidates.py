#!/usr/bin/env python3
"""Freeze chosen candidates only after real validation. Does not open Test data."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolrl_pipeline.io import digest, loads, write_json
from toolrl_training.config import load
from toolrl_training.stage import model_path

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--approval',required=True)
    p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    plan,state,approval=load(a.plan),loads((Path(a.run_dir)/'state.json').read_text()),load(a.approval)
    if state['identity']['is_mock'] or state['identity']['plan_sha256'] != digest(plan):
        raise ValueError('Only same-plan real validation candidates may be frozen')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    payload=dict(plan=plan,state=state,approval=approval,is_mock=False)
    candidates=[]
    for source in a.sources:
        evaluation='E0' if source=='base' else source+'_eval'
        record=state['stages'][evaluation]
        if record['status']!='completed':
            raise ValueError('Missing real validation evaluation: '+source)
        metrics=loads((Path(record['directory'])/'metrics.json').read_text())
        if metrics['is_mock']:
            raise ValueError('Mock metrics cannot freeze a model')
        path,revision=model_path(payload,source)
        candidates.append(dict(source=source,revision=revision))
    write_json(a.output,dict(data_version=plan['data_version'],plan_sha256=digest(plan),candidates=candidates,
                             policy='One final Test round only; no further tuning from Test'))
