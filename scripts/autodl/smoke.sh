#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Approval must name the smoke plan, including actual instance start/time/price.
python run_experiments.py --plan configs/phase3/smoke_plan.yaml --run-dir "$1" --approval "$2" --execute
