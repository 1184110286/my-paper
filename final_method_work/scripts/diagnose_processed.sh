#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
PROCESSED="${1:-data/processed/cadets_quick}"
python -m malsnif.cli diagnose --processed "$PROCESSED"
