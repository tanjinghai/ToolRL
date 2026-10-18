"""CPU-testable numerical health rules, called by the live GRPO loop."""
import math
from collections import defaultdict


class TrainingWatchdog:
    def __init__(self, zero_fraction=.9, windows=3):
        self.zero_fraction, self.windows, self.streak = zero_fraction, windows, 0

    def observe(self, scores, groups):
        if len(scores) != len(groups) or not scores or not all(math.isfinite(x) for x in scores):
            raise FloatingPointError('Nonfinite/empty/misaligned reward batch')
        by_group = defaultdict(list)
        for score, group in zip(scores, groups):
            by_group[group].append(score)
        if any(len(v) < 2 for v in by_group.values()):
            raise ValueError('GRPO requires at least two responses per prompt group')
        ratio = sum(max(v)-min(v) <= 1e-8 for v in by_group.values())/len(by_group)
        self.streak = self.streak + 1 if ratio > self.zero_fraction else 0
        if self.streak >= self.windows:
            raise FloatingPointError('Sustained >90% zero-advantage groups; preserve prior checkpoint and stop')
        return ratio


def is_final_update(step, total):
    if type(step) is not int or not 1 <= step <= total:
        raise ValueError('Update step outside 1..total')
    return step == total
