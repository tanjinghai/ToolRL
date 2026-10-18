"""Atomic immutable checkpoint directories and explicit restoration contracts."""
import os
import hashlib
from pathlib import Path
import random
import shutil
import uuid
from toolrl_pipeline.io import loads, write_json


def file_hash(path):
    # Full optimizer files can exceed host RAM; do not use Phase 2's small-file read_bytes helper.
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inventory(directory, exclude_checkpoint=True):
    directory = Path(directory)
    result = {}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink artifact forbidden: {p}')
        if p.is_file() and (not exclude_checkpoint or p.name != 'CHECKPOINT.json'):
            result[str(p.relative_to(directory))] = file_hash(p)
    return result


def publish_checkpoint(destination, writer, metadata):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents concurrent writers from replacing the same step.
    lock = destination.with_suffix('.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex)
    try:
        tmp.mkdir()
        writer(tmp)
        files = inventory(tmp)
        if not files:
            raise ValueError('Empty checkpoint')
        write_json(tmp/'CHECKPOINT.json', {'version': 'toolrl.checkpoint.v1', 'metadata': metadata, 'files': files})
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(tmp, destination)
    finally:
        os.close(fd)
        lock.unlink()
        if tmp.exists():
            shutil.rmtree(tmp)
    return destination


def validate_checkpoint(path, expected=None):
    path = Path(path)
    manifest = loads((path/'CHECKPOINT.json').read_text())
    if manifest['version'] != 'toolrl.checkpoint.v1' or manifest['files'] != inventory(path):
        raise ValueError('Checkpoint integrity failure')
    for key, value in (expected or {}).items():
        if manifest['metadata'].get(key) != value:
            raise ValueError(f'Checkpoint identity mismatch: {key}')
    return manifest


def save_cpu_state(path, *, optimizer, scheduler, progress, identity):
    # CPU fixture format. GPU trainers use torch.save for tensor states, same publisher.
    def writer(tmp):
        write_json(tmp/'state.json', {'optimizer': optimizer, 'scheduler': scheduler,
                                     'progress': progress, 'python_rng': random.getstate()})
    return publish_checkpoint(path, writer, dict(identity, resume_mode='cpu_exact_fixture'))


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def restore_cpu_state(path, identity):
    validate_checkpoint(path, identity)
    state = loads((Path(path)/'state.json').read_text())
    random.setstate(_tuples(state.pop('python_rng')))
    return state


def capture_rng():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def prune_checkpoints(parent, keep=1):
    """Retain latest complete steps in this single run/stage only, after publication.

    Never touches incomplete directories, other runs, or arbitrary paths. The default
    keeps one recoverable full state; earlier final models in other stages are separate.
    """
    import re
    parent = Path(parent)
    if type(keep) is not int or keep < 1:
        raise ValueError('At least one complete checkpoint must survive')
    candidates = []
    for path in parent.iterdir():
        match = re.fullmatch(r'global_step_(\d+)', path.name)
        if match and path.is_dir() and not path.is_symlink():
            validate_checkpoint(path)
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    for _, path in candidates[:-keep]:
        shutil.rmtree(path)


def training_provenance():
    path = os.environ.get('TOOLRL_RUN_METADATA')
    return loads(Path(path).read_text()) if path else {'unmanaged_entry': True}
