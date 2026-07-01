#!/usr/bin/env bash
set -euo pipefail

# v3.3: CADETS verdict for MalSnif-aligned MCBG + EA-THGN node-adaptive mechanisms.
# This script removes the original AGF/edge-gate idea and evaluates EHA/ETS/EAW
# inside ST-HGAN message passing.  The key question is whether CADETS reproduces the EA-THGN pattern: EHA/ETS/EHA+ETS reduce FN over non-adaptive ST-HGAN.
#
# Pilot:
#   DEVICE=1 bash scripts/run_cadets_v3_ea_verdict.sh
# Reuse an existing graph cache:
#   DEVICE=1 REUSE_RUN=runs/<previous_cadets_run>/seed_42 bash scripts/run_cadets_v3_ea_verdict.sh
# Three-seed validation:
#   DEVICE=1 SEEDS="42 43 44" bash scripts/run_cadets_v3_ea_verdict.sh

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
PARENT_OUT="${PARENT_OUT:-runs/cadets_v3_ea_verdict_${TS}}"
SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}"
RAW_DIR="${RAW_DIR:-data/raw/darpa_tc/cadets/e3/cdm}"
LABEL_DIR="${LABEL_DIR:-data/raw/darpa_tc/cadets/e3/labels}"
DATASET_NAME="${DATASET_NAME:-cadets_e3_v3_ea_verdict}"
RAW_GLOB="${RAW_GLOB:-ta1-cadets-e3-official*.json*}"
RAW_FILE_SORT="${RAW_FILE_SORT:-cdm_shards}"

# Reuse the same bounded-cache strategy as the stable CADETS runs.  EA-THGN
# ablations are expensive, so default to the existing calibrated 12M cache if
# available and build it only once under runs/_cache/.
CADETS_EA_PRESET="${CADETS_EA_PRESET:-calib5m}"
case "$CADETS_EA_PRESET" in
  smoke)
    MAX_EVENTS="${MAX_EVENTS:-800000}"; WINDOW_EVENTS="${WINDOW_EVENTS:-100000}"; GRAPH_LIMIT_TRAIN="${GRAPH_LIMIT_TRAIN:-6}"; GRAPH_LIMIT_VAL="${GRAPH_LIMIT_VAL:-2}"; GRAPH_LIMIT_TEST="${GRAPH_LIMIT_TEST:-2}" ;;
  calib5m)
    MAX_EVENTS="${MAX_EVENTS:-5000000}"; WINDOW_EVENTS="${WINDOW_EVENTS:-200000}"; GRAPH_LIMIT_TRAIN="${GRAPH_LIMIT_TRAIN:-15}"; GRAPH_LIMIT_VAL="${GRAPH_LIMIT_VAL:-5}"; GRAPH_LIMIT_TEST="${GRAPH_LIMIT_TEST:-5}" ;;
  calib8m)
    MAX_EVENTS="${MAX_EVENTS:-8000000}"; WINDOW_EVENTS="${WINDOW_EVENTS:-200000}"; GRAPH_LIMIT_TRAIN="${GRAPH_LIMIT_TRAIN:-24}"; GRAPH_LIMIT_VAL="${GRAPH_LIMIT_VAL:-8}"; GRAPH_LIMIT_TEST="${GRAPH_LIMIT_TEST:-8}" ;;
  calib12m)
    MAX_EVENTS="${MAX_EVENTS:-12000000}"; WINDOW_EVENTS="${WINDOW_EVENTS:-200000}"; GRAPH_LIMIT_TRAIN="${GRAPH_LIMIT_TRAIN:-36}"; GRAPH_LIMIT_VAL="${GRAPH_LIMIT_VAL:-12}"; GRAPH_LIMIT_TEST="${GRAPH_LIMIT_TEST:-12}" ;;
  full)
    MAX_EVENTS="${MAX_EVENTS:-none}"; WINDOW_EVENTS="${WINDOW_EVENTS:-200000}"; GRAPH_LIMIT_TRAIN="${GRAPH_LIMIT_TRAIN:-0}"; GRAPH_LIMIT_VAL="${GRAPH_LIMIT_VAL:-0}"; GRAPH_LIMIT_TEST="${GRAPH_LIMIT_TEST:-0}" ;;
  *) echo "[ERROR] Unknown CADETS_EA_PRESET=$CADETS_EA_PRESET; use smoke|calib5m|calib8m|calib12m|full" >&2; exit 2 ;;
esac

# Fast but meaningful defaults.  Increase SEEDS/EPOCHS/GRAPH_LIMIT_* for strict runs.
EPOCHS="${EPOCHS:-5}"
DIM="${DIM:-64}"
# Semantic encoder override. Default keeps the original E1_eha_only behavior.
# set SEMANTIC_ENCODER=gdtc_mcbg for E1-GDTC-MCBG, or
# SEMANTIC_ENCODER=rgd_bigru_mcbg for E1-RGD-BiGRU-MCBG, without changing
# the graph/EHA pipeline.
SEMANTIC_ENCODER="${SEMANTIC_ENCODER:-mcbg}"
E1_SEMANTIC_ENCODER="${E1_SEMANTIC_ENCODER:-$SEMANTIC_ENCODER}"
E1_EXPERIMENT_NAME="${E1_EXPERIMENT_NAME:-E1_eha_only}"
GDTC_KERNEL_SIZE="${GDTC_KERNEL_SIZE:-3}"
GDTC_DILATIONS="${GDTC_DILATIONS:-1,2,4}"
GDTC_DROPOUT="${GDTC_DROPOUT:-0.2}"
GDTC_USE_EVENT_WEIGHT_POOLING="${GDTC_USE_EVENT_WEIGHT_POOLING:-1}"
RGD_KERNEL_SIZE="${RGD_KERNEL_SIZE:-3}"
RGD_DILATIONS="${RGD_DILATIONS:-1,2}"
RGD_DROPOUT="${RGD_DROPOUT:-0.2}"
RGD_RESIDUAL_SCALE_INIT="${RGD_RESIDUAL_SCALE_INIT:-0.1}"
RGD_DEPTHWISE_SEPARABLE="${RGD_DEPTHWISE_SEPARABLE:-1}"
RGD_USE_EVENT_WEIGHT_POOLING="${RGD_USE_EVENT_WEIGHT_POOLING:-1}"
HGAN_TOPK="${HGAN_TOPK:-20}"
VAL_EVERY="${VAL_EVERY:-2}"
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-float16}"
CACHE_GRAPHS_IN_MEMORY="${CACHE_GRAPHS_IN_MEMORY:-1}"
TOP_ALERTS_PER_GRAPH="${TOP_ALERTS_PER_GRAPH:-20}"
# essential keeps only history.png and scores_test.png. Use PLOT_MODE=all to
# reproduce the old per-metric png behavior, or PLOT_MODE=none to disable plots.
PLOT_MODE="${PLOT_MODE:-essential}"
PLOT_METRIC_KEYS="${PLOT_METRIC_KEYS:-loss,val_f1,val_mcc,val_average_precision,val_threshold}"
MAX_EVENTS_PER_NODE="${MAX_EVENTS_PER_NODE:-48}"
MAX_EVENTS_PER_EDGE="${MAX_EVENTS_PER_EDGE:-4}"
GRAPH_SIMPLIFY_MODE="${GRAPH_SIMPLIFY_MODE:-leaf}"
GRAPH_SIMPLIFY_RISK_THRESHOLD="${GRAPH_SIMPLIFY_RISK_THRESHOLD:-0.62}"
GRAPH_SIMPLIFY_TOPK_PER_PROCESS="${GRAPH_SIMPLIFY_TOPK_PER_PROCESS:-0}"
GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS="${GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS:-1000000000}"
GRAPH_SIMPLIFY_REPEAT_NORM="${GRAPH_SIMPLIFY_REPEAT_NORM:-8}"
SKIPGRAM_BATCH_SIZE="${SKIPGRAM_BATCH_SIZE:-16384}"
SKIPGRAM_MAX_SENTENCES="${SKIPGRAM_MAX_SENTENCES:-300000}"
SKIPGRAM_MAX_PAIRS="${SKIPGRAM_MAX_PAIRS:-500000}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-val_average_precision}"
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-val_f1_min_recall}"
THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.95}"
NODE_SCOPE="${NODE_SCOPE:-process}"
PATIENCE="${PATIENCE:-5}"
MECHANISM_EPS="${MECHANISM_EPS:-0.001}"

# Experiment switches.  E0-E7 encode the EA-THGN ablation grid:
# EHA=Hops, EAW=Attn-width, ETS=Temp.  Default core set follows the EA-THGN
# paper's observation that structural+temporal adaptivity (EHA+ETS) is often
# strongest, while allowing RUN_ALL_EA=1 for the full 8-combination grid.
RUN_B0="${RUN_B0:-1}"  # MalSnif-style GCN baseline
RUN_B1="${RUN_B1:-1}"  # MCBG semantic-only, no graph propagation
RUN_E0="${RUN_E0:-1}"  # MCBG + ST-HGAN, no adaptive mechanisms
RUN_E1="${RUN_E1:-1}"  # EHA only
RUN_E2="${RUN_E2:-0}"  # EAW only
RUN_E3="${RUN_E3:-1}"  # ETS only
RUN_E4="${RUN_E4:-0}"  # EHA + EAW
RUN_E5="${RUN_E5:-1}"  # EHA + ETS (paper reports strongest interaction on CADETS)
RUN_E6="${RUN_E6:-0}"  # EAW + ETS
RUN_E7="${RUN_E7:-1}"  # all three mechanisms
RUN_ALL_EA="${RUN_ALL_EA:-0}"
if [[ "$RUN_ALL_EA" == "1" ]]; then
  RUN_E0=1; RUN_E1=1; RUN_E2=1; RUN_E3=1; RUN_E4=1; RUN_E5=1; RUN_E6=1; RUN_E7=1
fi

# EA mechanism hyperparameters.
EA_NUM_HEADS="${EA_NUM_HEADS:-4}"
EA_TAU_MIN="${EA_TAU_MIN:-0.1}"
EA_TAU_MAX="${EA_TAU_MAX:-5.0}"
EA_DROPOUT="${EA_DROPOUT:-0.0}"
# GPU guard for exclusive-process lab machines.
GPU_RESERVE_ENABLE="${GPU_RESERVE_ENABLE:-1}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-512}"
GPU_RESERVE_STRICT="${GPU_RESERVE_STRICT:-1}"
GPU_RESERVE_SLEEP="${GPU_RESERVE_SLEEP:-30}"

# Resume / robustness controls.  SKIP_COMPLETED avoids re-running experiments
# whose metrics are already present.  RESUME_EVALUATE_ONLY lets an interrupted
# run evaluate an existing checkpoint without retraining.  GPU_WAIT_* is useful
# when another process temporarily occupies the target GPU.
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
RESUME_EVALUATE_ONLY="${RESUME_EVALUATE_ONLY:-0}"
GPU_WAIT_ENABLE="${GPU_WAIT_ENABLE:-0}"
GPU_WAIT_MIN_FREE_MB="${GPU_WAIT_MIN_FREE_MB:-4400}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-0}"

CADETS_CACHE_ROOT="${CADETS_CACHE_ROOT:-runs/_cache/cadets_e3_${CADETS_EA_PRESET}_events${MAX_EVENTS}_win${WINDOW_EVENTS}}"
if [[ -z "${REUSE_RUN:-}" && -z "${REUSE_PROCESSED_DIR:-}" ]]; then
  export REUSE_PROCESSED_DIR="$CADETS_CACHE_ROOT/processed/graph_cache"
  export REUSE_METADATA_DIR="$CADETS_CACHE_ROOT/analysis/preprocess"
fi

mkdir -p "$PARENT_OUT"
ANALYSIS_ROOT="$PARENT_OUT"
source scripts/lib_gpu_guard.sh
install_gpu_reserve_trap
start_gpu_reserve

cat > "$PARENT_OUT/V3_CADETS_EA_THGN_PLAN.md" <<EOF
# v3 EA-THGN node-adaptive ST-HGAN plan

Purpose: remove the original AGF/edge-gate idea and evaluate EA-THGN-inspired
node-adaptive mechanisms inside a MalSnif-aligned MCBG + ST-HGAN pipeline.

Mechanisms:
- EHA: Elastic Hop Aggregation, node-specific structural depth.
- ETS: Elastic Temporal Softmax, node-specific attention temperature.
- EAW: Elastic Attention Width, node-specific attention-head bandwidth.

Controls:
- dataset_name=$DATASET_NAME
- raw_dir=$RAW_DIR
- label_dir=$LABEL_DIR
- raw_glob=$RAW_GLOB
- device=$DEVICE
- seeds=$SEEDS
- epochs=$EPOCHS
- cadets_ea_preset=$CADETS_EA_PRESET
- max_events=$MAX_EVENTS
- redundancy_mode=${REDUNDANCY_MODE:-prefix_tree}
- graph_simplify_mode=$GRAPH_SIMPLIFY_MODE
- graph_simplify_risk_threshold=$GRAPH_SIMPLIFY_RISK_THRESHOLD
- graph_simplify_topk_per_process=$GRAPH_SIMPLIFY_TOPK_PER_PROCESS
- graph_limits=$GRAPH_LIMIT_TRAIN/$GRAPH_LIMIT_VAL/$GRAPH_LIMIT_TEST
- cache_root=$CADETS_CACHE_ROOT
- dim=$DIM, heads=$EA_NUM_HEADS, hgan_topk=$HGAN_TOPK
- semantic_encoder=$SEMANTIC_ENCODER, e1_semantic_encoder=$E1_SEMANTIC_ENCODER, e1_experiment_name=$E1_EXPERIMENT_NAME
- gdtc_kernel_size=$GDTC_KERNEL_SIZE, gdtc_dilations=$GDTC_DILATIONS, gdtc_event_weight_pooling=$GDTC_USE_EVENT_WEIGHT_POOLING
- rgd_kernel_size=$RGD_KERNEL_SIZE, rgd_dilations=$RGD_DILATIONS, rgd_residual_scale_init=$RGD_RESIDUAL_SCALE_INIT, rgd_depthwise=$RGD_DEPTHWISE_SEPARABLE, rgd_event_weight_pooling=$RGD_USE_EVENT_WEIGHT_POOLING
- run_b0=$RUN_B0, run_b1=$RUN_B1, e0=$RUN_E0, e1=$RUN_E1, e2=$RUN_E2, e3=$RUN_E3, e4=$RUN_E4, e5=$RUN_E5, e6=$RUN_E6, e7=$RUN_E7
- skip_completed=$SKIP_COMPLETED
- resume_evaluate_only=$RESUME_EVALUATE_ONLY
- eval_device=$EVAL_DEVICE
- gpu_wait_enable=$GPU_WAIT_ENABLE
- gpu_wait_min_free_mb=$GPU_WAIT_MIN_FREE_MB

Decision rule:
- E0 > B1 means graph propagation adds value beyond MCBG semantic-only.
- E1/E2/E3 > E0 identify useful single mechanisms.
- E5 > E1/E3/E0 tests the paper-motivated EHA+ETS interaction.
- E7 > E5 tests whether adding EAW on top of EHA+ETS helps or adds noise.
EOF

if [[ "${RAW_GLOB,,}" == *theia* || "${DATASET_NAME,,}" == *theia* ]]; then
  bash scripts/check_theia_data_layout.sh
else
  bash scripts/check_cadets_data_layout.sh
fi

bool_py() {
  case "${1,,}" in
    1|true|yes|y|on) echo "true" ;;
    *) echo "false" ;;
  esac
}

device_arg() {
  if [[ "$1" == "cpu" || "$1" == "CPU" || "$1" == "-1" ]]; then
    echo "cpu"
  else
    echo "$1"
  fi
}

maybe_wait_for_gpu() {
  local dev="$1"
  if [[ "${GPU_WAIT_ENABLE:-0}" != "1" ]]; then
    return 0
  fi
  if [[ "$dev" == "cpu" || "$dev" == "CPU" || "$dev" == "-1" ]]; then
    return 0
  fi
  bash scripts/wait_for_gpu_free.sh "$(device_arg "$dev")" "$GPU_WAIT_MIN_FREE_MB" "$GPU_WAIT_INTERVAL_SECONDS" "$GPU_WAIT_TIMEOUT_SECONDS"
}

resolve_reuse_dirs() {
  local seed_root="$1"
  GRAPH_CACHE="$seed_root/processed/graph_cache"
  PREPROCESS_ANALYSIS="$seed_root/analysis/preprocess"
  if [[ -n "${REUSE_PROCESSED_DIR:-}" ]]; then
    GRAPH_CACHE="$REUSE_PROCESSED_DIR"
  elif [[ -n "${REUSE_RUN:-}" ]]; then
    if [[ -f "$REUSE_RUN/processed/graph_cache/vocab.pkl" ]]; then
      GRAPH_CACHE="$REUSE_RUN/processed/graph_cache"
    elif [[ -f "$REUSE_RUN/processed/vocab.pkl" ]]; then
      GRAPH_CACHE="$REUSE_RUN/processed"
    fi
  fi
  if [[ -n "${REUSE_METADATA_DIR:-}" ]]; then
    PREPROCESS_ANALYSIS="$REUSE_METADATA_DIR"
  elif [[ -n "${REUSE_RUN:-}" ]]; then
    if [[ -f "$REUSE_RUN/analysis/preprocess/metadata.json" ]]; then
      PREPROCESS_ANALYSIS="$REUSE_RUN/analysis/preprocess"
    elif [[ -f "$REUSE_RUN/processed/graph_cache/metadata.json" ]]; then
      PREPROCESS_ANALYSIS="$REUSE_RUN/processed/graph_cache"
    elif [[ -f "$REUSE_RUN/processed/metadata.json" ]]; then
      PREPROCESS_ANALYSIS="$REUSE_RUN/processed"
    fi
  fi
}

write_config() {
  local cfg_path="$1" run_dir="$2" ckpt_dir="$3" model_variant="$4" semantic_encoder="$5" fusion_mode="$6" graph_encoder="$7" edge_gate_mode="$8" time_bias="$9" relation_types="${10}" use_edge_sem="${11}" pruning_mode="${12}" topk="${13}" seed_value="${14}" graph_cache="${15}" preprocess_analysis="${16}" ea_eha="${17:-false}" ea_ets="${18:-false}" ea_eaw="${19:-false}"
  mkdir -p "$(dirname "$cfg_path")" "$run_dir" "$ckpt_dir"
  python - "$cfg_path" "$run_dir" "$ckpt_dir" "$model_variant" "$semantic_encoder" "$fusion_mode" "$graph_encoder" "$edge_gate_mode" "$time_bias" "$relation_types" "$use_edge_sem" "$pruning_mode" "$topk" "$seed_value" "$graph_cache" "$preprocess_analysis" "$ea_eha" "$ea_ets" "$ea_eaw" <<'PY'
import os, sys, yaml
(cfg_path, run_dir, ckpt_dir, model_variant, semantic_encoder, fusion_mode, graph_encoder, edge_gate_mode, time_bias, relation_types, use_edge_sem, pruning_mode, topk, seed_value, graph_cache, preprocess_analysis, ea_eha, ea_ets, ea_eaw) = sys.argv[1:]

def b(x):
    return str(x).lower() in {'1','true','yes','y','on'}

def none_or_int(x):
    return None if str(x).lower() in {'','none','null','all','full'} else int(x)

cfg = {
    'raw_dir': os.environ['RAW_DIR'],
    'processed_dir': graph_cache,
    'metadata_dir': preprocess_analysis,
    'run_dir': run_dir,
    'checkpoint_dir': ckpt_dir,
    'seed': int(seed_value),
    'dataset_name': os.environ.get('DATASET_NAME', 'cadets_e3_v3_ea_verdict'),
    'input_format': 'cdm_json',
    'raw_glob': os.environ.get('RAW_GLOB', '*.json*'),
    'raw_file_sort': os.environ.get('RAW_FILE_SORT', 'cdm_shards'),
    'label_dir': os.environ['LABEL_DIR'],
    'cdm_information_flow': True,
    'node_label_policy': 'process_event_endpoints',
    'process_label_projection': 'adaptive',
    'process_label_min_events': 2,
    'process_label_max_positive_ratio': 0.75,
    'split_ratio': [0.6, 0.2, 0.2],
    'window_events': int(os.environ['WINDOW_EVENTS']),
    'max_events': none_or_int(os.environ['MAX_EVENTS']),
    'filter_selected_events': False,
    'simplify_graph': True,
    'graph_simplify_mode': os.environ.get('GRAPH_SIMPLIFY_MODE', 'leaf'),
    'graph_simplify_risk_threshold': float(os.environ.get('GRAPH_SIMPLIFY_RISK_THRESHOLD', '0.62')),
    'graph_simplify_topk_per_process': int(os.environ.get('GRAPH_SIMPLIFY_TOPK_PER_PROCESS', '0')),
    'graph_simplify_temporal_window_ns': int(os.environ.get('GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS', '1000000000')),
    'graph_simplify_repeat_norm': int(os.environ.get('GRAPH_SIMPLIFY_REPEAT_NORM', '8')),
    'sanitize_paths': True,
    'reduce_sequences': True,
    'redundancy_mode': os.environ.get('REDUNDANCY_MODE', 'prefix_tree'),
    'redundancy_risk_threshold': float(os.environ.get('REDUNDANCY_RISK_THRESHOLD', '2.5')),
    'redundancy_preserve_risk_events': int(os.environ.get('REDUNDANCY_PRESERVE_RISK_EVENTS', '1')),
    'redundancy_repeat_summary': b(os.environ.get('REDUNDANCY_REPEAT_SUMMARY', '1')),
    'redundancy_repeat_min': int(os.environ.get('REDUNDANCY_REPEAT_MIN', '3')),
    'mw_prr_alpha': float(os.environ.get('MW_PRR_ALPHA', '0.2')),
    'mw_prr_attention_beta': float(os.environ.get('MW_PRR_ATTENTION_BETA', '1.0')),
    'btr_rr_max_block_len': int(os.environ.get('BTR_RR_MAX_BLOCK_LEN', '16')),
    'btr_rr_min_gain': int(os.environ.get('BTR_RR_MIN_GAIN', '2')),
    'btr_rr_repeat_cap': int(os.environ.get('BTR_RR_REPEAT_CAP', '32')),
    'btr_rr_alpha': float(os.environ.get('BTR_RR_ALPHA', '0.3')),
    'lz_sr_min_phrase_len': int(os.environ.get('LZ_SR_MIN_PHRASE_LEN', '4')),
    'lz_sr_max_phrase_len': int(os.environ.get('LZ_SR_MAX_PHRASE_LEN', '24')),
    'lz_sr_window': int(os.environ.get('LZ_SR_WINDOW', '512')),
    'lz_sr_min_gain': int(os.environ.get('LZ_SR_MIN_GAIN', '2')),
    'lz_sr_alpha': float(os.environ.get('LZ_SR_ALPHA', '0.25')),
    'frc_rr_cap_size': int(os.environ.get('FRC_RR_CAP_SIZE', '3')),
    'frc_rr_repeat_cap': int(os.environ.get('FRC_RR_REPEAT_CAP', '32')),
    'frc_rr_alpha': float(os.environ.get('FRC_RR_ALPHA', '0.25')),
    'flb_rr_repeat_cap': int(os.environ.get('FLB_RR_REPEAT_CAP', '32')),
    'flb_rr_alpha': float(os.environ.get('FLB_RR_ALPHA', '0.25')),
    'wra_rr_window': int(os.environ.get('WRA_RR_WINDOW', '11')),
    'tbb_rr_target_compression': float(os.environ.get('TBB_RR_TARGET_COMPRESSION', '0.90')),
    'lowercase_tokens': True,
    'word_dim': int(os.environ['DIM']),
    'min_token_freq': 1,
    'skipgram_epochs': 1,
    'skipgram_window': 2,
    'skipgram_negative': 5,
    'skipgram_lr': 0.01,
    'skipgram_batch_size': int(os.environ['SKIPGRAM_BATCH_SIZE']),
    'skipgram_max_sentences': none_or_int(os.environ['SKIPGRAM_MAX_SENTENCES']),
    'skipgram_max_pairs': none_or_int(os.environ['SKIPGRAM_MAX_PAIRS']),
    'freeze_word_embeddings': False,
    'max_tokens_per_event': 12,
    'max_events_per_node': int(os.environ['MAX_EVENTS_PER_NODE']),
    'max_events_per_edge': int(os.environ['MAX_EVENTS_PER_EDGE']),
    'model_variant': model_variant,
    'semantic_encoder': semantic_encoder,
    'hidden_dim': int(os.environ['DIM']),
    'semantic_dim': int(os.environ['DIM']),
    'behavior_dim': int(os.environ['DIM']),
    'mcbg_kernel_sizes': '2,3,5',
    'mcbg_attention_heads': 4,
    'mcbg_dropout': 0.2,
    'gdtc_kernel_size': int(os.environ.get('GDTC_KERNEL_SIZE', '3')),
    'gdtc_dilations': os.environ.get('GDTC_DILATIONS', '1,2,4'),
    'gdtc_dropout': float(os.environ.get('GDTC_DROPOUT', '0.2')),
    'gdtc_use_event_weight_pooling': b(os.environ.get('GDTC_USE_EVENT_WEIGHT_POOLING', '1')),
    'rgd_kernel_size': int(os.environ.get('RGD_KERNEL_SIZE', '3')),
    'rgd_dilations': os.environ.get('RGD_DILATIONS', '1,2'),
    'rgd_dropout': float(os.environ.get('RGD_DROPOUT', '0.2')),
    'rgd_residual_scale_init': float(os.environ.get('RGD_RESIDUAL_SCALE_INIT', '0.1')),
    'rgd_depthwise_separable': b(os.environ.get('RGD_DEPTHWISE_SEPARABLE', '1')),
    'rgd_use_event_weight_pooling': b(os.environ.get('RGD_USE_EVENT_WEIGHT_POOLING', '1')),
    'graph_encoder': graph_encoder,
    'gcn_layers': 2,
    'dropout': 0.2,
    'use_semantics': True,
    'use_edge_weights': True,
    'edge_weight_mode': 'legacy_sigmoid',
    'edge_weight_init_zero': False,
    'hgan_num_relations': 128,
    'hgan_num_time_buckets': 16,
    'hgan_use_node_types': True,
    'hgan_use_relation_types': b(relation_types),
    'hgan_use_time_bias': b(time_bias),
    'hgan_topk': int(topk),
    'hgan_pruning_mode': pruning_mode,
    'hgan_soft_pruning_floor': 0.05,
    'hgan_attention_dropout': 0.1,
    'hgan_use_residual': True,
    'fusion_mode': fusion_mode,
    'edge_gate_mode': edge_gate_mode,
    'edge_gate_hidden_dim': 0,
    'edge_gate_dropout': 0.1,
    'edge_gate_temperature': 1.0,
    'edge_gate_use_edge_semantics': False,
    'ea_use_eha': b(ea_eha),
    'ea_use_ets': b(ea_ets),
    'ea_use_eaw': b(ea_eaw),
    'ea_num_heads': int(os.environ['EA_NUM_HEADS']),
    'ea_tau_min': float(os.environ['EA_TAU_MIN']),
    'ea_tau_max': float(os.environ['EA_TAU_MAX']),
    'ea_dropout': float(os.environ['EA_DROPOUT']),
    'graph_level': False,
    'node_scope': os.environ['NODE_SCOPE'],
    'epochs': int(os.environ['EPOCHS']),
    'lr': 0.001,
    'weight_decay': 0.0,
    'downsample_after_forward': True,
    'downsample_weight': 10,
    'loss_sampling': 'paper',
    'balanced_loss': True,
    'model_selection_metric': os.environ['MODEL_SELECTION_METRIC'],
    'model_selection_tie_breakers': 'val_average_precision,val_mcc,val_balanced_accuracy',
    'threshold_strategy': os.environ['THRESHOLD_STRATEGY'],
    'threshold_min_recall': float(os.environ['THRESHOLD_MIN_RECALL']),
    'threshold': 0.5,
    'allow_unlabeled_training': False,
    'grad_clip': 5.0,
    'patience': int(os.environ['PATIENCE']),
    'val_every': int(os.environ['VAL_EVERY']),
    'cache_graphs_in_memory': b(os.environ['CACHE_GRAPHS_IN_MEMORY']),
    'top_alerts_per_graph': int(os.environ['TOP_ALERTS_PER_GRAPH']),
    'plot_mode': os.environ.get('PLOT_MODE', 'essential'),
    'plot_metric_keys': os.environ.get('PLOT_METRIC_KEYS', 'loss,val_f1,val_mcc,val_average_precision,val_threshold'),
    'max_nodes_per_graph': 120000,
    'max_edges_per_graph': 300000,
    'graph_limit_train': int(os.environ['GRAPH_LIMIT_TRAIN']),
    'graph_limit_val': int(os.environ['GRAPH_LIMIT_VAL']),
    'graph_limit_test': int(os.environ['GRAPH_LIMIT_TEST']),
    'train_progress': True,
    'show_progress': True,
    'use_amp': b(os.environ['USE_AMP']),
    'amp_dtype': os.environ['AMP_DTYPE'],
    'amp_fallback_to_fp32': True,
    'cuda_empty_cache_interval': 1,
    'cuda_empty_cache_after_epoch': True,
    'cuda_empty_cache_after_eval': True,
}
with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

summarize_seed() {
  local analysis_root="$1"
  python - "$analysis_root" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for exp in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {'configs','preprocess','gpu_guard'}):
    mpath=exp/'metrics_test_compact.json'; spath=exp/'train_summary.json'
    if not mpath.exists():
        continue
    mobj=json.load(open(mpath,encoding='utf-8')); metrics=mobj.get('metrics',{})
    summ=json.load(open(spath,encoding='utf-8')) if spath.exists() else {}
    gate=mobj.get('gate_distribution') or {}; att=mobj.get('attention_distribution') or {}
    warnings=list(mobj.get('warnings') or [])
    rows.append({
        'experiment': exp.name,
        'model_variant': summ.get('model_variant'),
        'semantic_encoder': summ.get('semantic_encoder'),
        'fusion_mode': summ.get('fusion_mode'),
        'edge_gate_mode': summ.get('edge_gate_mode'),
        'f1': metrics.get('f1'), 'precision': metrics.get('precision'), 'recall': metrics.get('recall'), 'specificity': metrics.get('specificity'), 'mcc': metrics.get('mcc'),
        'roc_auc': metrics.get('roc_auc'), 'average_precision': metrics.get('average_precision'), 'threshold': metrics.get('threshold'), 'num_samples': metrics.get('num_samples'),
        'prevalence': metrics.get('prevalence'), 'predicted_positive_rate': metrics.get('predicted_positive_rate'),
        'best_f1': metrics.get('best_f1'), 'best_f1_threshold': metrics.get('best_f1_threshold'), 'best_f1_precision': metrics.get('best_f1_precision'), 'best_f1_recall': metrics.get('best_f1_recall'),
        'best_f1_gap': (metrics.get('best_f1') - metrics.get('f1')) if metrics.get('best_f1') is not None and metrics.get('f1') is not None else None,
        'tp': metrics.get('tp'), 'fp': metrics.get('fp'), 'tn': metrics.get('tn'), 'fn': metrics.get('fn'),
        'best_epoch': summ.get('best_epoch'), 'best_val_f1': summ.get('best_val_f1'), 'train_seconds': summ.get('train_total_seconds'),
        'max_cuda_peak_allocated_mb': summ.get('max_cuda_peak_allocated_mb'), 'max_cuda_peak_reserved_mb': summ.get('max_cuda_peak_reserved_mb'),
        'gate_semantic_mean': gate.get('gate_semantic_mean'), 'gate_semantic_std': gate.get('gate_semantic_std'),
        'edge_gate_mean': gate.get('edge_gate_mean'), 'edge_gate_std': gate.get('edge_gate_std'),
        'attention_kept_ratio': att.get('kept_ratio'), 'ets_tau_mean': att.get('ets_tau_mean'), 'eaw_head_mean': att.get('eaw_head_mean'), 'eha_entropy_mean': att.get('eha_entropy_mean'), 'eha_hop_0_mean': att.get('eha_hop_0_mean'), 'eha_hop_1_mean': att.get('eha_hop_1_mean'), 'eha_hop_2_mean': att.get('eha_hop_2_mean'), 'warning_count': len(warnings), 'warnings': ' | '.join(str(w) for w in warnings),
    })
fields=['experiment','model_variant','semantic_encoder','fusion_mode','edge_gate_mode','f1','precision','recall','specificity','mcc','roc_auc','average_precision','threshold','num_samples','prevalence','predicted_positive_rate','best_f1','best_f1_threshold','best_f1_precision','best_f1_recall','best_f1_gap','tp','fp','tn','fn','best_epoch','best_val_f1','train_seconds','max_cuda_peak_allocated_mb','max_cuda_peak_reserved_mb','gate_semantic_mean','gate_semantic_std','edge_gate_mean','edge_gate_std','attention_kept_ratio','ets_tau_mean','eaw_head_mean','eha_entropy_mean','eha_hop_0_mean','eha_hop_1_mean','eha_hop_2_mean','warning_count','warnings']
with open(root/'summary.csv','w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(json.dumps({'summary_csv': str(root/'summary.csv'), 'experiments': [r['experiment'] for r in rows]}, ensure_ascii=False, indent=2))
PY
}

run_one() {
  local name="$1" model_variant="$2" semantic_encoder="$3" fusion_mode="$4" graph_encoder="$5" edge_gate_mode="$6" time_bias="$7" relation_types="$8" use_edge_sem="$9" pruning="${10}" topk="${11}" seed_value="${12}" seed_root="${13}" graph_cache="${14}" preprocess_analysis="${15}" ea_eha="${16:-false}" ea_ets="${17:-false}" ea_eaw="${18:-false}"
  local analysis_root="$seed_root/analysis" ckpt_root="$seed_root/checkpoints" cfg_root="$analysis_root/configs"
  local run_dir="$analysis_root/$name" ckpt_dir="$ckpt_root/$name" cfg="$cfg_root/$name.yaml"
  mkdir -p "$run_dir" "$ckpt_dir" "$cfg_root"
  write_config "$cfg" "$run_dir" "$ckpt_dir" "$model_variant" "$semantic_encoder" "$fusion_mode" "$graph_encoder" "$edge_gate_mode" "$time_bias" "$relation_types" "$use_edge_sem" "$pruning" "$topk" "$seed_value" "$graph_cache" "$preprocess_analysis" "$ea_eha" "$ea_ets" "$ea_eaw"
  if [[ -f "$preprocess_analysis/metadata.json" ]]; then cp -f "$preprocess_analysis/metadata.json" "$run_dir/preprocess_metadata.json"; fi
  echo "========== [$name] seed=$seed_value =========="
  if [[ "$SKIP_COMPLETED" == "1" && -f "$run_dir/metrics_test_compact.json" ]]; then
    echo "[skip] $name seed=$seed_value already has metrics_test_compact.json"
    return 0
  fi

  local trained=0
  if [[ "$RESUME_EVALUATE_ONLY" == "1" && -f "$ckpt_dir/best.pt" && -f "$run_dir/train_summary.json" && ! -f "$run_dir/metrics_test_compact.json" ]]; then
    echo "[resume] $name seed=$seed_value checkpoint exists; skip train and evaluate/analyze only" | tee -a "$run_dir/console.log"
    trained=1
  fi

  if [[ "$trained" != "1" ]]; then
    stop_gpu_reserve
    maybe_wait_for_gpu "$DEVICE"
    python -m malsnif.cli train --config "$cfg" --device "$(device_arg "$DEVICE")" 2>&1 | tee "$run_dir/console.log"
    start_gpu_reserve
  fi

  if [[ ! -f "$run_dir/metrics_test_compact.json" ]]; then
    stop_gpu_reserve
    maybe_wait_for_gpu "$EVAL_DEVICE"
    python -m malsnif.cli evaluate --config "$cfg" --device "$(device_arg "$EVAL_DEVICE")" --checkpoint "$ckpt_dir/best.pt" --split test 2>&1 | tee -a "$run_dir/console.log"
    start_gpu_reserve
  fi
  python -m malsnif.cli analyze-run --run-dir "$run_dir" --out "$run_dir/run_analysis.json" >> "$run_dir/console.log" 2>&1 || true
}

first_reuse="${REUSE_RUN:-}"
for seed_value in $SEEDS; do
  seed_root="$PARENT_OUT/seed_${seed_value}"
  analysis_root="$seed_root/analysis"
  ckpt_root="$seed_root/checkpoints"
  mkdir -p "$analysis_root" "$ckpt_root"
  export RAW_DIR LABEL_DIR DATASET_NAME RAW_GLOB RAW_FILE_SORT WINDOW_EVENTS MAX_EVENTS DIM SEMANTIC_ENCODER E1_SEMANTIC_ENCODER E1_EXPERIMENT_NAME GDTC_KERNEL_SIZE GDTC_DILATIONS GDTC_DROPOUT GDTC_USE_EVENT_WEIGHT_POOLING RGD_KERNEL_SIZE RGD_DILATIONS RGD_DROPOUT RGD_RESIDUAL_SCALE_INIT RGD_DEPTHWISE_SEPARABLE RGD_USE_EVENT_WEIGHT_POOLING MAX_EVENTS_PER_NODE MAX_EVENTS_PER_EDGE GRAPH_SIMPLIFY_MODE GRAPH_SIMPLIFY_RISK_THRESHOLD GRAPH_SIMPLIFY_TOPK_PER_PROCESS GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS GRAPH_SIMPLIFY_REPEAT_NORM REDUNDANCY_MODE REDUNDANCY_RISK_THRESHOLD REDUNDANCY_PRESERVE_RISK_EVENTS REDUNDANCY_REPEAT_SUMMARY REDUNDANCY_REPEAT_MIN MW_PRR_ALPHA MW_PRR_ATTENTION_BETA BTR_RR_MAX_BLOCK_LEN BTR_RR_MIN_GAIN BTR_RR_REPEAT_CAP BTR_RR_ALPHA WRA_RR_WINDOW SKIPGRAM_BATCH_SIZE SKIPGRAM_MAX_SENTENCES SKIPGRAM_MAX_PAIRS GRAPH_LIMIT_TRAIN GRAPH_LIMIT_VAL GRAPH_LIMIT_TEST EPOCHS VAL_EVERY CACHE_GRAPHS_IN_MEMORY TOP_ALERTS_PER_GRAPH PLOT_MODE PLOT_METRIC_KEYS USE_AMP AMP_DTYPE NODE_SCOPE MODEL_SELECTION_METRIC THRESHOLD_STRATEGY THRESHOLD_MIN_RECALL PATIENCE EA_NUM_HEADS EA_TAU_MIN EA_TAU_MAX EA_DROPOUT
  if [[ -n "$first_reuse" ]]; then
    REUSE_RUN="$first_reuse" resolve_reuse_dirs "$seed_root"
  else
    resolve_reuse_dirs "$seed_root"
  fi
  mkdir -p "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" "$analysis_root/configs"
  if [[ ! -f "$GRAPH_CACHE/vocab.pkl" ]]; then
    echo "[preprocess] seed=$seed_value building graph cache: $GRAPH_CACHE"
    pre_cfg="$analysis_root/configs/preprocess.yaml"
    write_config "$pre_cfg" "$analysis_root/preprocess_config" "$ckpt_root/preprocess_config" baseline baseline baseline graphsage none true true true none 0 "$seed_value" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS"
    python -m malsnif.cli preprocess --config "$pre_cfg" 2>&1 | tee "$PREPROCESS_ANALYSIS/console.log"
    first_reuse="$seed_root"
  else
    echo "[preprocess] seed=$seed_value reuse graph cache: $GRAPH_CACHE"
  fi

  if [[ "$RUN_B0" == "1" ]]; then run_one B0_baseline_gcn baseline baseline baseline gcn none true true true none 0 "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" false false false; fi
  if [[ "$RUN_B1" == "1" ]]; then run_one B1_mcbg_semantic_only agf_st_hgan_mcbg "$SEMANTIC_ENCODER" semantic_only st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" false false false; fi
  if [[ "$RUN_E0" == "1" ]]; then run_one E0_mcbg_sthgan_no_adapt ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" false false false; fi
  if [[ "$RUN_E1" == "1" ]]; then run_one "$E1_EXPERIMENT_NAME" ea_st_hgan_mcbg "$E1_SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" true false false; fi
  if [[ "$RUN_E2" == "1" ]]; then run_one E2_eaw_only ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" false false true; fi
  if [[ "$RUN_E3" == "1" ]]; then run_one E3_ets_only ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" false true false; fi
  if [[ "$RUN_E4" == "1" ]]; then run_one E4_eha_eaw ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" true false true; fi
  if [[ "$RUN_E5" == "1" ]]; then run_one E5_eha_ets ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" true true false; fi
  if [[ "$RUN_E6" == "1" ]]; then run_one E6_eaw_ets ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" false true true; fi
  if [[ "$RUN_E7" == "1" ]]; then run_one E7_eha_ets_eaw ea_st_hgan_mcbg "$SEMANTIC_ENCODER" mal_snif_aligned st_hgan none true true true soft "$HGAN_TOPK" "$seed_value" "$seed_root" "$GRAPH_CACHE" "$PREPROCESS_ANALYSIS" true true true; fi

  summarize_seed "$analysis_root"
  bash scripts/collect_analysis_bundle.sh "$seed_root" "$seed_root/analysis_bundle"
done

python -m malsnif.ea_report --root "$PARENT_OUT" --summary "$PARENT_OUT/next_summary.csv" --report "$PARENT_OUT/EA_THGN_DECISION_REPORT.md" --eps "$MECHANISM_EPS"
BUNDLE="$PARENT_OUT/analysis_bundle"
mkdir -p "$BUNDLE"
cp -f "$PARENT_OUT/V3_CADETS_EA_THGN_PLAN.md" "$BUNDLE/" || true
cp -f "$PARENT_OUT/next_summary.csv" "$BUNDLE/" || true
cp -f "$PARENT_OUT/EA_THGN_DECISION_REPORT.md" "$BUNDLE/" || true
for d in "$PARENT_OUT"/seed_*; do
  [[ -d "$d/analysis_bundle" ]] || continue
  mkdir -p "$BUNDLE/$(basename "$d")"
  cp -a "$d/analysis_bundle/." "$BUNDLE/$(basename "$d")/"
done
cat > "$BUNDLE/MANIFEST.txt" <<EOF
created=$(date -Is)
script=scripts/run_cadets_v3_ea_verdict.sh
parent_out=$PARENT_OUT
send_this_directory_for_analysis=true
EOF

echo "[done] parent_out=$PARENT_OUT"
echo "[done] analysis_bundle=$BUNDLE"
