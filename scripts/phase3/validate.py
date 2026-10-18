#!/usr/bin/env python3
"""Reproducible CPU acceptance, including actual torch tensors, no training weights."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from toolrl_pipeline.io import digest,file_hash,loads,write_json
from toolrl_training.config import ROOT,verify_data,load
from toolrl_training.reward_probes import run_probes
from toolrl_training.executor import validate_plan


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    import torch
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError('This acceptance run is specifically CPU only')
    env=dict(os.environ,TOOLRL_TOKENIZER_PATH=str(ROOT/'data_local/tokenizers/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'))
    commands={}
    for phase in ('phase2','phase3'):
        with (out/f'{phase}_tests.txt').open('x') as log:
            result=subprocess.run([sys.executable,'-m','unittest','discover','-s',f'tests/{phase}','-v'],
                                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        text=(out/f'{phase}_tests.txt').read_text()
        if result.returncode or 'skipped=' in text or ' ... skipped ' in text:
            raise RuntimeError(f'{phase} tests failed or skipped; inspect {out}')
        commands[phase]={'exit_code':result.returncode,'log_sha256':file_hash(out/f'{phase}_tests.txt')}
    python_files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','*.py'],cwd=ROOT,text=True).splitlines()
    for rel in python_files:compile((ROOT/rel).read_text(),rel,'exec')
    shell_files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','*.sh'],cwd=ROOT,text=True).splitlines()
    for rel in shell_files:subprocess.run(['bash','-n',str(ROOT/rel)],check=True)
    verify_data(ROOT/'dataset/generated/phase2-v1',ROOT/'data_manifests/phase2-v1')
    for rel in ('configs/experiment_plan.yaml','configs/phase3/smoke_plan.yaml','configs/phase3/mock_plan.yaml'):
        validate_plan(load(ROOT/rel))
    source_files = python_files + shell_files + [str(p.relative_to(ROOT)) for p in (ROOT/'configs').rglob('*') if p.is_file()]
    sources = {rel: file_hash(ROOT/rel) for rel in sorted(set(source_files))}
    write_json(out/'source_hashes.json', sources)
    write_json(out/'reward_probes.json',run_probes())
    write_json(out/'validation.json',dict(cpu_only=True,torch=torch.__version__,torch_cuda=torch.version.cuda,
               tests=commands,tested_source_sha256=digest(sources),compiled_python=len(python_files),shell_syntax=len(shell_files),
               phase2_frozen_hashes_unchanged=True,gpu_training_validated=False,final_test_generated=False,
               model_weights_downloaded=False,paid_api_called=False))
    (out/'cpu_environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(out/'validation.json')


if __name__=='__main__':main()
