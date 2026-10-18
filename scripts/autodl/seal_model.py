#!/usr/bin/env python3
"""Seal already provisioned local weights; does NOT download or verify publisher authenticity."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolrl_pipeline.io import write_json
from toolrl_training.checkpoint import file_hash
from toolrl_training.config import MODEL_REVISION

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--path', required=True)
    p.add_argument('--revision', required=True)
    a=p.parse_args()
    path=Path(a.path).resolve()
    if a.revision != MODEL_REVISION:
        raise ValueError('This plan pins Qwen1.5B immutable revision; use a separately reviewed plan for other models')
    if not all((path/k).is_file() for k in ('config.json','tokenizer_config.json','tokenizer.json')):
        raise ValueError('Missing HF config/tokenizer')
    if not list(path.glob('*.safetensors')) and not list(path.glob('pytorch_model*.bin')):
        raise ValueError('No weights. Provision the approved snapshot first; this script never downloads')
    from toolrl_pipeline.io import loads
    for index in path.glob('*.index.json'):
        for shard in set(loads(index.read_text())['weight_map'].values()):
            if not (path/shard).is_file():
                raise ValueError('Incomplete sharded snapshot: '+shard)
    manifest=path/'MODEL_MANIFEST.json'
    if manifest.exists():
        raise FileExistsError('Model seal immutable; verify existing manifest rather than replace')
    files={}
    for f in sorted(path.rglob('*')):
        if f.is_file():
            files[str(f.relative_to(path))]=file_hash(f)
    write_json(manifest,dict(model_id='Qwen/Qwen2.5-1.5B-Instruct', revision=a.revision,files=files,
                             provenance='user-provisioned snapshot; file integrity checked, origin requires operator verification'))
    print('MODEL_MANIFEST.json sha256:',file_hash(manifest))
