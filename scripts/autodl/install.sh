#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Explicit approval and wall-clock/cost caps apply to setup, too. Linux Python3.10 only.
python scripts/autodl/bootstrap.py --plan "$1" --approval "$2"
