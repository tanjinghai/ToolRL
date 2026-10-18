"""Versioned completion-only rewards. The original rlla.py is immutable.

Strict reward shares parsing/schema rules, not the independent verifier's EM.
It gives additive partial credit using an exact global assignment.
"""
from collections import Counter
from contextlib import redirect_stdout
import io
import os
from functools import lru_cache
from toolrl_pipeline.protocol import parse_completion
from toolrl_pipeline.schema import strict_equal, validate_calls

LEGACY_FLAGS = ('REFINEDREWARD', 'INTERMEDIATEREWARD', 'COARSEREWARD', 'CORRECTMAX1',
                'MAX1STEP30MAX3', 'SCHEDULEREWARD', 'WITHLENGTH', 'SCHEDULELENGTH')


def _assignment_value(weights):
    # Independent Hungarian maximum assignment; dummy rows/columns allow missing calls.
    n = len(weights)
    if not n:
        return 0.
    u, v, p, way = [0.]*(n+1), [0.]*(n+1), [0]*(n+1), [0]*(n+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minimum, used = [float('inf')]*(n+1), [False]*(n+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, n+1):
                if not used[j]:
                    cur = -weights[i0-1][j-1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    return sum(weights[p[j]-1][j-1] for j in range(1, n+1))


def strict_matching(completion, ground_truth, tools, metadata=None, step=0):
    gt, pred = parse_completion(ground_truth), parse_completion(completion)
    if not (gt.protocol_valid and gt.json_valid and gt.shape_valid) or validate_calls(gt.calls, tools):
        raise ValueError('Invalid training ground truth')
    order = (metadata or {}).get('call_order', 'parallel' if len(gt.calls) <= 1 else 'unknown')
    if len(gt.calls) > 1 and order not in ('parallel', 'sequential'):
        raise ValueError('Unknown multi-call order')
    valid = pred.protocol_valid and pred.json_valid and pred.shape_valid
    format_score = float(valid and not validate_calls(pred.calls, tools))
    if not gt.calls:
        # Free text semantics are unverifiable. Only penalize unnecessary calls.
        correctness = -3. if pred.calls else 0.
    elif not valid:
        correctness = -3.
    else:
        g, q = gt.calls, pred.calls
        a, b = Counter(x['name'] for x in g), Counter(x['name'] for x in q)
        overlap = sum((a & b).values())
        name_score = overlap / (len(g) + len(q) - overlap) if overlap else 0.
        schemas = {t['name']: t['parameters'].get('properties', {}) for t in tools}
        n = max(len(g), len(q))
        weights = [[0.]*n for _ in range(n)]
        for i, expected in enumerate(g):
            for j, actual in enumerate(q):
                if expected['name'] != actual['name'] or (order == 'sequential' and i != j):
                    continue
                x, y = expected['parameters'], actual['parameters']
                keys = x.keys() & y.keys()
                key_score = len(keys) / len(x.keys() | y.keys()) if x or y else 1.
                values = sum(strict_equal(x[k], y[k], schemas.get(expected['name'], {}).get(k)) for k in keys)
                weights[i][j] = key_score + values
        denominator = 1 + sum(1 + len(x['parameters']) for x in g)
        # Extra calls reduce *all* positive evidence, including parameter matches.
        coverage = min(1., len(g)/max(1, len(q)))
        correctness = 6 * (name_score + _assignment_value(weights)) * coverage / denominator - 3
    return (format_score + correctness, format_score, correctness, 0.)


@lru_cache(maxsize=1)
def _legacy_module():
    # Load this dependency-free legacy module without importing the GPU veRL package.
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    spec = spec_from_file_location('toolrl_original_reward', Path(__file__).resolve().parents[1]/'verl/utils/reward_score/rlla.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy(completion, ground_truth, tools=None, metadata=None, step=0):
    rlla = _legacy_module()
    # Freeze legacy defaults rather than inherit arbitrary shell feature flags.
    previous = {key: os.environ.get(key) for key in LEGACY_FLAGS + ('EXPERIMENT_NAME',)}
    try:
        os.environ.update({key: '0' for key in LEGACY_FLAGS})
        os.environ['EXPERIMENT_NAME'] = 'qwen_phase3_legacy'
        with redirect_stdout(io.StringIO()):
            return rlla.compute_score('<|im_start|>assistant\n' + completion, ground_truth, step)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


REWARDS = {'legacy': legacy, 'strict_matching': strict_matching}


def compute_reward(version, completion, ground_truth, tools, metadata=None, step=0):
    if version not in REWARDS:
        raise ValueError(f'Unknown reward version: {version}')
    return REWARDS[version](completion, ground_truth, tools, metadata, step)
