#!/usr/bin/env bash
set -euo pipefail

# RGT-HGIDS paper-method entrypoint.
# Name: Redundancy-aware Gated Temporal Heterogeneous Graph IDS.
# Pipeline: TBB-RR + RGD-BiGRU-MCBG + ST-HGAN + EHA.
#
# Example:
#   DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_rigorous.sh

export METHOD_NAME="${METHOD_NAME:-RGT-HGIDS}"
export RUN_ENCODERS="${RUN_ENCODERS:-mcbg rgd_bigru}"
export RUN_DATASETS="${RUN_DATASETS:-cadets theia}"
export RIGOR_LEVEL="${RIGOR_LEVEL:-balanced}"
export REDUNDANCY_MODE="${REDUNDANCY_MODE:-target_boundary}"
export TBB_RR_TARGET_COMPRESSION="${TBB_RR_TARGET_COMPRESSION:-0.90}"
export SEMANTIC_ENCODER="${SEMANTIC_ENCODER:-rgd_bigru_mcbg}"

TS="$(date +%Y%m%d_%H%M%S)"
export BASE_OUT="${BASE_OUT:-runs/rgt_hgids_rigorous_${TS}}"

bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
