#!/usr/bin/env python3
"""Preflight only: no model download, training, cloud account or credential reads."""
import argparse
import importlib.metadata
import json
import platform
import os
from pathlib import Path
import shutil
import subprocess
import sys


def check(host_only=False, gpu=False, expected_gpu=None, gpu_count=None):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 10):
        raise RuntimeError('Candidate stack requires Linux x86_64 Python 3.10; Mac install rejected')
    if shutil.disk_usage(Path.cwd()).free < 40 * 1024**3:
        raise RuntimeError('Need >=40GiB free; full-state checkpoints + external backup may require much more')
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    if ram < 64 * 1024**3:
        raise RuntimeError('Full optimizer + CPU offload candidate requires >=64GiB host RAM')
    report = {'host_ram_bytes': ram, 'python': platform.python_version(), 'system': platform.platform(), 'gpu_tested': False}
    if host_only:
        return report
    versions = {'torch': '2.4.0', 'vllm': '0.6.3', 'xformers': '0.0.27.post2',
                'transformers': '4.47.1', 'tensordict': '0.5.0', 'numpy': '1.26.4',
                'ray': '2.40.0', 'flash-attn': '2.6.3'}
    report['versions'] = {}
    for name, expected in versions.items():
        actual = importlib.metadata.version(name)
        report['versions'][name] = actual
        if actual.split('+')[0] != expected:
            raise RuntimeError(f'{name}: {actual} != pinned {expected}')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    if gpu:
        import torch
        import vllm
        import ray
        import flash_attn
        from torch.distributed.fsdp import FullyShardedDataParallel
        report['nvcc'] = subprocess.check_output(['nvcc', '--version'], text=True)
        report['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True)
        if not torch.cuda.is_available() or torch.version.cuda != '12.1':
            raise RuntimeError('CUDA available / torch CUDA12.1 required')
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if gpu_count is not None and len(names) != gpu_count:
            raise RuntimeError('GPU count differs from approval')
        if expected_gpu is not None and any(expected_gpu.casefold() not in x.casefold() for x in names):
            raise RuntimeError('GPU model differs from approval')
        if any(torch.cuda.get_device_capability(i)[0] >= 10 for i in range(len(names))):
            raise RuntimeError('Blackwell unsupported by this old-stack candidate')
        for i in range(len(names)):
            with torch.cuda.device(i):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError('BF16 required')
                x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
                if not torch.isfinite(x @ x).all():
                    raise RuntimeError('GPU matrix smoke failed')
        report.update(cuda=torch.version.cuda, devices=names, gpu_tested=True,
                      boundary='imports and CUDA matmul only; FSDP/Ray/vLLM train smoke still required')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host-only', action='store_true')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--expected-gpu')
    p.add_argument('--gpu-count', type=int)
    p.add_argument('--output')
    a = p.parse_args()
    result = check(a.host_only, a.gpu, a.expected_gpu, a.gpu_count)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
