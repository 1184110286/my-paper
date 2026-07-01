#!/usr/bin/env bash
set -euo pipefail

# Quick RGT-HGIDS sanity-check entrypoint.
# For paper claims use run_rgt_hgids_rigorous.sh instead.

export METHOD_NAME="${METHOD_NAME:-RGT-HGIDS}"
export RUN_ENCODERS="${RUN_ENCODERS:-mcbg rgd_bigru}"
export RUN_DATASETS="${RUN_DATASETS:-cadets theia}"
export CADETS_EA_PRESET="${CADETS_EA_PRESET:-calib5m}"
export EPOCHS="${EPOCHS:-5}"
export REDUNDANCY_MODE="${REDUNDANCY_MODE:-target_boundary}"
export TBB_RR_TARGET_COMPRESSION="${TBB_RR_TARGET_COMPRESSION:-0.90}"

TS="$(date +%Y%m%d_%H%M%S)"
export BASE_OUT="${BASE_OUT:-runs/rgt_hgids_quick_${TS}}"

bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
