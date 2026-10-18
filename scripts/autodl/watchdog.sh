#!/usr/bin/env bash
set -euo pipefail
# Independent local wall-clock bound, in addition to the executor's per-stage watchdog.
# GNU timeout kills the process group. It does not stop AutoDL instance billing.
cd "$(dirname "$0")/../.."
timeout --signal=TERM --kill-after=20s "$1" python run_experiments.py --plan "$2" --run-dir "$3" --approval "$4" --execute
