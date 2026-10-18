#!/usr/bin/env python3
"""Default is CPU-only dry-run. GPU execution requires a matching approval file."""
import argparse
import json
import sys
import signal
from toolrl_training.config import load
from toolrl_training.executor import execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='configs/experiment_plan.yaml')
    parser.add_argument('--run-dir', default='runs/phase3-preview')
    parser.add_argument('--approval')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    from toolrl_training.approval import StopRun
    def interrupted(signum, frame):
        raise StopRun(f'Interrupted by signal {signum}; inspect workers and stop platform billing')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        result = execute(args.plan, args.run_dir, load(args.approval) if args.approval else None, args.execute, args.mock)
    except (StopRun, ValueError, OSError) as exc:
        print(json.dumps({'status': 'stopped', 'reason': str(exc)}, ensure_ascii=False))
        sys.exit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(s['status'] in ('failed', 'stopped') for s in result.get('stages', {}).values()):
        sys.exit(2)


if __name__ == '__main__':
    main()
