#!/usr/bin/env bash
set -euo pipefail

# Rigorous CADETS E1 WRA-RR redundancy comparison.
# Strategy:
# 1. Run a single-pass strict split search on chronological CADETS events.
# 2. Stop at the earliest event prefix whose train/val/test all contain the
#    required positive-vs-negative process-node mixture.
# 3. Use that rigor-qualified prefix for a paired-seed comparison among:
#    - off
#    - prefix_tree
#    - winnowing_anchor
# 4. Report mean/std, paired deltas, win counts, and conservative verdict rules.
#
# Default lab usage:
#   bash scripts/run_cadets_wra_rr_verdict.sh
#
# Useful overrides:
#   WINDOW_EVENTS=80000 bash scripts/run_cadets_wra_rr_verdict.sh
#   SEEDS="42 43 44" bash scripts/run_cadets_wra_rr_verdict.sh
#   WRA_RR_WINDOW=11 bash scripts/run_cadets_wra_rr_verdict.sh
#   RUN_OFF=0 bash scripts/run_cadets_wra_rr_verdict.sh

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

activate_mal_env() {
  local env_name="${CONDA_ENV_NAME:-mal}"
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
    conda activate "$env_name" >/dev/null 2>&1 || true
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]]; then
    for p in \
      "/d/anaconda3/envs/$env_name" \
      "/c/ProgramData/anaconda3/envs/$env_name" \
      "$HOME/anaconda3/envs/$env_name" \
      "$HOME/miniconda3/envs/$env_name"; do
      if [[ -x "$p/python.exe" || -x "$p/bin/python" || -x "$p/Scripts/python.exe" ]]; then
        export PATH="$p:$p/Scripts:$p/bin:$PATH"
        break
      fi
    done
  fi
}

activate_mal_env
python - <<'PY'
import sys, torch
print("[env] python=", sys.executable)
print("[env] torch=", torch.__version__, "cuda=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[env] gpu=", torch.cuda.get_device_name(0))
PY

export DEVICE="${DEVICE:-0}"
export EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}"
export SEEDS="${SEEDS:-42 43 44 45 46}"
export MAX_EVENTS="${MAX_EVENTS:-none}"
export WINDOW_EVENTS="${WINDOW_EVENTS:-100000}"
export GRAPH_LIMIT_TRAIN="${GRAPH_LIMIT_TRAIN:-0}"
export GRAPH_LIMIT_VAL="${GRAPH_LIMIT_VAL:-0}"
export GRAPH_LIMIT_TEST="${GRAPH_LIMIT_TEST:-0}"
export EPOCHS="${EPOCHS:-5}"
export VAL_EVERY="${VAL_EVERY:-1}"
export DIM="${DIM:-64}"
export HGAN_TOPK="${HGAN_TOPK:-20}"
export MAX_EVENTS_PER_NODE="${MAX_EVENTS_PER_NODE:-48}"
export MAX_EVENTS_PER_EDGE="${MAX_EVENTS_PER_EDGE:-4}"
export USE_AMP="${USE_AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-float16}"
export CACHE_GRAPHS_IN_MEMORY="${CACHE_GRAPHS_IN_MEMORY:-0}"
export TOP_ALERTS_PER_GRAPH="${TOP_ALERTS_PER_GRAPH:-20}"
export PLOT_MODE="${PLOT_MODE:-essential}"
export MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-val_average_precision}"
export THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-val_f1_min_recall}"
export THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.95}"
export NODE_SCOPE="${NODE_SCOPE:-process}"
export PATIENCE="${PATIENCE:-5}"
export GPU_RESERVE_ENABLE="${GPU_RESERVE_ENABLE:-1}"
export GPU_RESERVE_MB="${GPU_RESERVE_MB:-512}"
export GPU_RESERVE_STRICT="${GPU_RESERVE_STRICT:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

export RAW_DIR="${RAW_DIR:-data/raw/darpa_tc/cadets/e3/cdm}"
export LABEL_DIR="${LABEL_DIR:-data/raw/darpa_tc/cadets/e3/labels}"
export RAW_GLOB="${RAW_GLOB:-ta1-cadets-e3-official*.json*}"
export RAW_FILE_SORT="${RAW_FILE_SORT:-cdm_shards}"

export STRICT_REQUIRE_GRAPH_MIX="${STRICT_REQUIRE_GRAPH_MIX:-0}"
export STRICT_REQUIRE_NODE_MIX="${STRICT_REQUIRE_NODE_MIX:-1}"
export STRICT_REQUIRE_MIX_SPLITS="${STRICT_REQUIRE_MIX_SPLITS:-train,val,test}"
export STRICT_MIN_GRAPHS_PER_SPLIT="${STRICT_MIN_GRAPHS_PER_SPLIT:-3}"
export STRICT_CHECK_EVERY_WINDOWS="${STRICT_CHECK_EVERY_WINDOWS:-1}"
export STRICT_MAX_EVENTS_CAP="${STRICT_MAX_EVENTS_CAP:-5000000}"

export GRAPH_SIMPLIFY_MODE="${GRAPH_SIMPLIFY_MODE:-leaf}"
export GRAPH_SIMPLIFY_RISK_THRESHOLD="${GRAPH_SIMPLIFY_RISK_THRESHOLD:-0.62}"
export GRAPH_SIMPLIFY_TOPK_PER_PROCESS="${GRAPH_SIMPLIFY_TOPK_PER_PROCESS:-0}"
export GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS="${GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS:-1000000000}"
export GRAPH_SIMPLIFY_REPEAT_NORM="${GRAPH_SIMPLIFY_REPEAT_NORM:-8}"
# The semantic encoder uses this coefficient as an attention-bias scale for any
# numeric event weight side channel, including WRA-RR.
export MW_PRR_ATTENTION_BETA="${MW_PRR_ATTENTION_BETA:-1.0}"
export WRA_RR_WINDOW="${WRA_RR_WINDOW:-11}"
export RUN_OFF="${RUN_OFF:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${BASE_OUT:-runs/cadets_wra_rr_verdict_${TS}_autostop_win${WINDOW_EVENTS}}"
PRECHECK_OUT="$BASE_OUT/precheck"
EXPERIMENT_OUT="$BASE_OUT/experiment"
mkdir -p "$PRECHECK_OUT" "$EXPERIMENT_OUT"
ABS_BASE_OUT="$(cd "$BASE_OUT" && pwd)"
ABS_PRECHECK_OUT="$(cd "$PRECHECK_OUT" && pwd)"
ABS_EXPERIMENT_OUT="$(cd "$EXPERIMENT_OUT" && pwd)"
BUNDLE_DIR="$BASE_OUT/analysis_bundle"
ABS_BUNDLE_DIR="$ABS_BASE_OUT/analysis_bundle"
ABS_COLLECTED_BUNDLE_DIR="$ABS_BASE_OUT/analysis_bundle/collected"

collect_wra_rr_bundle() {
  if [[ ! -d "$BASE_OUT" ]]; then
    return 0
  fi
  mkdir -p "$BUNDLE_DIR"
  if [[ -f scripts/collect_wra_rr_analysis_bundle.sh ]]; then
    # Use bash explicitly instead of relying on executable bits.  This is
    # important on Windows/Git-Bash and after zip/unzip, where +x may be lost.
    bash scripts/collect_wra_rr_analysis_bundle.sh "$BASE_OUT" "$BUNDLE_DIR" \
      > "$BUNDLE_DIR/collect_wra_rr_analysis_bundle.log" 2>&1 || {
        echo "[WARN] failed to collect WRA-RR analysis bundle; see $BUNDLE_DIR/collect_wra_rr_analysis_bundle.log" >&2
      }
  else
    echo "[WARN] scripts/collect_wra_rr_analysis_bundle.sh not found; no single-folder bundle created" >&2
  fi
}
trap collect_wra_rr_bundle EXIT

cat > "$BASE_OUT/STRICT_EXPERIMENT_PLAN.md" <<EOF
# Rigorous CADETS E1 WRA-RR redundancy experiment (autostop)

- target_model: E1_eha_only
- comparison: off vs prefix_tree vs winnowing_anchor
- fixed_graph_simplify_mode: $GRAPH_SIMPLIFY_MODE
- conda_env: ${CONDA_ENV_NAME:-mal}
- device: $DEVICE
- eval_device: $EVAL_DEVICE
- seeds: $SEEDS
- run_off: $RUN_OFF
- requested_max_events: $MAX_EVENTS
- strict_max_events_cap: $STRICT_MAX_EVENTS_CAP
- window_events: $WINDOW_EVENTS
- graph_limits: $GRAPH_LIMIT_TRAIN / $GRAPH_LIMIT_VAL / $GRAPH_LIMIT_TEST
- threshold_strategy: $THRESHOLD_STRATEGY
- threshold_min_recall: $THRESHOLD_MIN_RECALL
- cache_graphs_in_memory: $CACHE_GRAPHS_IN_MEMORY
- strict_require_graph_mix: $STRICT_REQUIRE_GRAPH_MIX
- strict_require_node_mix: $STRICT_REQUIRE_NODE_MIX
- strict_require_mix_splits: $STRICT_REQUIRE_MIX_SPLITS
- strict_min_graphs_per_split: $STRICT_MIN_GRAPHS_PER_SPLIT
- strict_check_every_windows: $STRICT_CHECK_EVERY_WINDOWS
- graph_simplify_risk_threshold: $GRAPH_SIMPLIFY_RISK_THRESHOLD
- graph_simplify_topk_per_process: $GRAPH_SIMPLIFY_TOPK_PER_PROCESS
- event_weight_attention_beta: $MW_PRR_ATTENTION_BETA
- wra_rr_window: $WRA_RR_WINDOW
- raw_dir: $RAW_DIR
- label_dir: $LABEL_DIR
- base_out: $ABS_BASE_OUT

Rationale:
- keep the current graph simplification fixed while isolating the effect of
  redundancy reduction;
- require split-wise process-node positive/negative mixture, which matches the
  actual supervision target of E1_eha_only on CADETS;
- use a single sequential scan to find the earliest rigor-qualified prefix, then
  compare the three redundancy modes on that same prefix.
EOF

PRECHECK_CFG="$PRECHECK_OUT/precheck.yaml"
PRECHECK_META="$PRECHECK_OUT/cache/analysis/preprocess"
mkdir -p "$PRECHECK_META"

python - "$PRECHECK_CFG" "$PRECHECK_META" <<'PY'
import os, sys, yaml

cfg_path, metadata_dir = sys.argv[1:]

def none_or_int(x):
    return None if str(x).lower() in {"", "none", "null", "all", "full"} else int(x)

requested = none_or_int(os.environ.get("STRICT_MAX_EVENTS_CAP", "none"))
fallback = none_or_int(os.environ.get("MAX_EVENTS", "none"))
cap = requested if requested is not None else fallback

cfg = {
    "raw_dir": os.environ["RAW_DIR"],
    "processed_dir": os.path.join(os.path.dirname(cfg_path), "precheck_unused_processed"),
    "metadata_dir": metadata_dir,
    "run_dir": os.path.join(os.path.dirname(cfg_path), "precheck_run"),
    "checkpoint_dir": os.path.join(os.path.dirname(cfg_path), "precheck_ckpt_unused"),
    "seed": 42,
    "dataset_name": "cadets_e3_wra_rr_strict_precheck_autostop",
    "input_format": "cdm_json",
    "raw_glob": os.environ.get("RAW_GLOB", "*.json*"),
    "raw_file_sort": os.environ.get("RAW_FILE_SORT", "cdm_shards"),
    "label_dir": os.environ["LABEL_DIR"],
    "cdm_information_flow": True,
    "node_label_policy": "process_event_endpoints",
    "process_label_projection": "adaptive",
    "process_label_min_events": 2,
    "process_label_max_positive_ratio": 0.75,
    "split_ratio": [0.6, 0.2, 0.2],
    "window_events": int(os.environ["WINDOW_EVENTS"]),
    "max_events": cap,
    "filter_selected_events": False,
    "simplify_graph": True,
    "graph_simplify_mode": os.environ.get("GRAPH_SIMPLIFY_MODE", "leaf"),
    "graph_simplify_risk_threshold": float(os.environ.get("GRAPH_SIMPLIFY_RISK_THRESHOLD", "0.62")),
    "graph_simplify_topk_per_process": int(os.environ.get("GRAPH_SIMPLIFY_TOPK_PER_PROCESS", "0")),
    "graph_simplify_temporal_window_ns": int(os.environ.get("GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS", "1000000000")),
    "graph_simplify_repeat_norm": int(os.environ.get("GRAPH_SIMPLIFY_REPEAT_NORM", "8")),
    "sanitize_paths": True,
    "reduce_sequences": False,
    "redundancy_mode": "off",
    "redundancy_risk_threshold": float(os.environ.get("REDUNDANCY_RISK_THRESHOLD", "2.5")),
    "redundancy_preserve_risk_events": int(os.environ.get("REDUNDANCY_PRESERVE_RISK_EVENTS", "1")),
    "redundancy_repeat_summary": True,
    "redundancy_repeat_min": int(os.environ.get("REDUNDANCY_REPEAT_MIN", "3")),
    "mw_prr_alpha": 0.2,
    "mw_prr_attention_beta": float(os.environ.get("MW_PRR_ATTENTION_BETA", "1.0")),
    "wra_rr_window": int(os.environ.get("WRA_RR_WINDOW", "11")),
    "lowercase_tokens": True,
    "word_dim": int(os.environ.get("DIM", "64")),
    "min_token_freq": 1,
    "skipgram_epochs": 0,
    "skipgram_window": 2,
    "skipgram_negative": 5,
    "skipgram_lr": 0.01,
    "skipgram_batch_size": 4096,
    "skipgram_max_sentences": 1,
    "skipgram_max_pairs": 1,
    "freeze_word_embeddings": False,
    "max_tokens_per_event": 12,
    "max_events_per_node": int(os.environ.get("MAX_EVENTS_PER_NODE", "48")),
    "max_events_per_edge": int(os.environ.get("MAX_EVENTS_PER_EDGE", "4")),
    "model_variant": "baseline",
    "semantic_encoder": "baseline",
    "hidden_dim": int(os.environ.get("DIM", "64")),
    "semantic_dim": int(os.environ.get("DIM", "64")),
    "behavior_dim": int(os.environ.get("DIM", "64")),
    "graph_encoder": "gcn",
    "graph_level": False,
    "node_scope": os.environ.get("NODE_SCOPE", "process"),
    "epochs": 1,
    "lr": 0.001,
    "weight_decay": 0.0,
    "downsample_after_forward": True,
    "downsample_weight": 10,
    "loss_sampling": "paper",
    "balanced_loss": True,
    "model_selection_metric": "val_average_precision",
    "threshold_strategy": os.environ.get("THRESHOLD_STRATEGY", "val_f1_min_recall"),
    "threshold_min_recall": float(os.environ.get("THRESHOLD_MIN_RECALL", "0.95")),
    "threshold": 0.5,
    "allow_unlabeled_training": False,
    "grad_clip": 5.0,
    "patience": 1,
    "val_every": 1,
    "cache_graphs_in_memory": False,
    "top_alerts_per_graph": 5,
    "plot_mode": "none",
    "max_nodes_per_graph": 120000,
    "max_edges_per_graph": 300000,
    "graph_limit_train": 0,
    "graph_limit_val": 0,
    "graph_limit_test": 0,
    "train_progress": True,
    "show_progress": True,
    "use_amp": False,
    "amp_dtype": "float16",
}

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

REQ_GRAPH_ARGS=()
REQ_NODE_ARGS=()
if [[ "${STRICT_REQUIRE_GRAPH_MIX,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  REQ_GRAPH_ARGS+=(--require-graph-mix)
fi
if [[ "${STRICT_REQUIRE_NODE_MIX,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  REQ_NODE_ARGS+=(--require-node-mix)
fi

echo "[precheck] searching earliest rigorous CADETS prefix..."
python -m malsnif.cli strict-precheck-autostop \
  --config "$PRECHECK_CFG" \
  --required-splits "$STRICT_REQUIRE_MIX_SPLITS" \
  --check-every-windows "$STRICT_CHECK_EVERY_WINDOWS" \
  --min-graphs-per-split "$STRICT_MIN_GRAPHS_PER_SPLIT" \
  "${REQ_GRAPH_ARGS[@]}" \
  "${REQ_NODE_ARGS[@]}" \
  2>&1 | tee "$PRECHECK_OUT/console.log"

echo "[precheck] validating search result..."
set +e
SELECTED_MAX_EVENTS="$(python - "$PRECHECK_META/metadata.json" "$PRECHECK_OUT/STRICT_SPLIT_PRECHECK.md" <<'PY'
import json, os, sys
from pathlib import Path

meta_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
meta = json.load(meta_path.open(encoding="utf-8"))
split_stats = meta.get("split_label_stats", {}) or {}
graph_diag = meta.get("graph_diagnostics", []) or []
warnings = meta.get("validation_warnings", []) or []
search = meta.get("strict_search", {}) or {}

require_graph_mix = str(os.environ.get("STRICT_REQUIRE_GRAPH_MIX", "1")).lower() in {"1", "true", "yes", "y", "on"}
require_node_mix = str(os.environ.get("STRICT_REQUIRE_NODE_MIX", "1")).lower() in {"1", "true", "yes", "y", "on"}
required_splits = [x.strip() for x in os.environ.get("STRICT_REQUIRE_MIX_SPLITS", "train,val,test").split(",") if x.strip()]
min_graphs_per_split = int(os.environ.get("STRICT_MIN_GRAPHS_PER_SPLIT", "3") or 3)

def fmt_bool(v):
    return "PASS" if v else "FAIL"

rows = []
all_ok = True
for split in required_splits:
    s = split_stats.get(split, {}) or {}
    num_graphs = int(s.get("num_graphs", 0) or 0)
    pos_graphs = int(s.get("positive_graphs", 0) or 0)
    neg_graphs = max(num_graphs - pos_graphs, 0)
    pos_proc = int(s.get("positive_process_nodes", 0) or 0)
    neg_proc = int(s.get("negative_process_nodes", 0) or 0)
    enough_graphs = num_graphs >= min_graphs_per_split
    graph_ok = (pos_graphs > 0 and neg_graphs > 0) if require_graph_mix else True
    node_ok = (pos_proc > 0 and neg_proc > 0) if require_node_mix else True
    split_ok = enough_graphs and graph_ok and node_ok
    all_ok = all_ok and split_ok
    rows.append({
        "split": split,
        "num_graphs": num_graphs,
        "positive_graphs": pos_graphs,
        "negative_graphs": neg_graphs,
        "process_nodes": int(s.get("process_nodes", 0) or 0),
        "positive_process_nodes": pos_proc,
        "negative_process_nodes": neg_proc,
        "positive_process_ratio": float(s.get("positive_process_ratio", 0.0) or 0.0),
        "enough_graphs": enough_graphs,
        "graph_mix": graph_ok,
        "node_mix": node_ok,
        "split_ok": split_ok,
    })

selected_max_events = int(search.get("selected_max_events", meta.get("parse_stats", {}).get("raw_events_consumed", 0)) or 0)

with report_path.open("w", encoding="utf-8") as f:
    f.write("# Strict split precheck (autostop)\n\n")
    f.write(f"- metadata: `{meta_path}`\n")
    f.write(f"- required_splits: `{', '.join(required_splits)}`\n")
    f.write(f"- require_graph_mix: `{require_graph_mix}`\n")
    f.write(f"- require_node_mix: `{require_node_mix}`\n")
    f.write(f"- min_graphs_per_split: `{min_graphs_per_split}`\n")
    f.write(f"- overall: `{fmt_bool(all_ok)}`\n")
    f.write(f"- stop_reason: `{search.get('stop_reason')}`\n")
    f.write(f"- selected_max_events: `{selected_max_events}`\n")
    f.write(f"- selected_num_graphs: `{search.get('selected_num_graphs')}`\n\n")
    f.write("| split | graphs | pos_graphs | neg_graphs | process_nodes | pos_proc | neg_proc | pos_proc_ratio | enough_graphs | graph_mix | node_mix | split_ok |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n")
    for r in rows:
        f.write(
            f"| {r['split']} | {r['num_graphs']} | {r['positive_graphs']} | {r['negative_graphs']} | "
            f"{r['process_nodes']} | {r['positive_process_nodes']} | {r['negative_process_nodes']} | "
            f"{r['positive_process_ratio']:.6f} | {fmt_bool(r['enough_graphs'])} | {fmt_bool(r['graph_mix'])} | "
            f"{fmt_bool(r['node_mix'])} | {fmt_bool(r['split_ok'])} |\n"
        )
    hist = search.get("evaluation_history") or []
    if hist:
        f.write("\n## Search checkpoints\n\n")
        f.write("| windows | raw_events_consumed | overall_pass |\n")
        f.write("|---:|---:|---|\n")
        for item in hist:
            f.write(f"| {item.get('windows')} | {item.get('raw_events_consumed')} | {fmt_bool(bool(item.get('overall_pass')))} |\n")
    if warnings:
        f.write("\n## Metadata warnings\n\n")
        for w in warnings:
            f.write(f"- {w}\n")
    if graph_diag:
        f.write("\n## First 12 graph diagnostics\n\n")
        f.write("| graph_index | graph_label | process_nodes | positive_process_nodes | positive_process_ratio | events |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in graph_diag[:12]:
            f.write(
                f"| {row.get('graph_index')} | {row.get('graph_label')} | {row.get('process_nodes')} | "
                f"{row.get('positive_process_nodes')} | {float(row.get('positive_process_ratio', 0.0) or 0.0):.6f} | "
                f"{row.get('events')} |\n"
            )

if not all_ok:
    sys.exit(3)
print(selected_max_events)
PY
)"
PRECHECK_STATUS=$?
set -e

if [[ "$PRECHECK_STATUS" -ne 0 ]]; then
  echo "[precheck] strict condition not satisfied. See $ABS_PRECHECK_OUT/STRICT_SPLIT_PRECHECK.md"
  exit "$PRECHECK_STATUS"
fi

if [[ -z "$SELECTED_MAX_EVENTS" || "$SELECTED_MAX_EVENTS" == "0" ]]; then
  echo "[precheck] failed to derive selected_max_events. See $ABS_PRECHECK_OUT/STRICT_SPLIT_PRECHECK.md"
  exit 3
fi

echo "[precheck] strict split passed with selected_max_events=$SELECTED_MAX_EVENTS"


export RUN_B0=0 RUN_B1=0 RUN_E0=0 RUN_E1=1 RUN_E2=0 RUN_E3=0 RUN_E4=0 RUN_E5=0 RUN_E6=0 RUN_E7=0 RUN_ALL_EA=0

echo "[run] launching WRA-RR verdict on the rigor-qualified prefix..."
MATRIX_FILE="$EXPERIMENT_OUT/run_matrix.tsv"
printf "label\tredundancy_mode\twindow\tbeta\trole\n" > "$MATRIX_FILE"

run_mode() {
  local mode="$1"
  local short="$2"
  local window_param="$3"
  local beta="$4"
  local role="$5"
  local out="$EXPERIMENT_OUT/$short"
  mkdir -p "$out"
  printf "%s\t%s\t%s\t%s\t%s\n" "$short" "$mode" "$window_param" "$beta" "$role" >> "$MATRIX_FILE"
  echo "========== label=$short redundancy_mode=$mode window=${window_param:-na} beta=${beta:-na} =========="
  (
    export PARENT_OUT="$out"
    export REDUNDANCY_MODE="$mode"
    export DATASET_NAME="cadets_e3_wra_rr_${short}"
    export CADETS_CACHE_ROOT="$out/cache"
    export CADETS_EA_PRESET="full"
    export MAX_EVENTS="$SELECTED_MAX_EVENTS"
    export WINDOW_EVENTS="$WINDOW_EVENTS"
    export GRAPH_LIMIT_TRAIN="$GRAPH_LIMIT_TRAIN"
    export GRAPH_LIMIT_VAL="$GRAPH_LIMIT_VAL"
    export GRAPH_LIMIT_TEST="$GRAPH_LIMIT_TEST"
    export EPOCHS="$EPOCHS"
    export VAL_EVERY="$VAL_EVERY"
    export DIM="$DIM"
    export HGAN_TOPK="$HGAN_TOPK"
    export MAX_EVENTS_PER_NODE="$MAX_EVENTS_PER_NODE"
    export MAX_EVENTS_PER_EDGE="$MAX_EVENTS_PER_EDGE"
    export USE_AMP="$USE_AMP"
    export AMP_DTYPE="$AMP_DTYPE"
    export CACHE_GRAPHS_IN_MEMORY="$CACHE_GRAPHS_IN_MEMORY"
    export TOP_ALERTS_PER_GRAPH="$TOP_ALERTS_PER_GRAPH"
    export PLOT_MODE="$PLOT_MODE"
    export MODEL_SELECTION_METRIC="$MODEL_SELECTION_METRIC"
    export THRESHOLD_STRATEGY="$THRESHOLD_STRATEGY"
    export THRESHOLD_MIN_RECALL="$THRESHOLD_MIN_RECALL"
    export NODE_SCOPE="$NODE_SCOPE"
    export PATIENCE="$PATIENCE"
    export RAW_DIR="$RAW_DIR"
    export LABEL_DIR="$LABEL_DIR"
    export RAW_GLOB="$RAW_GLOB"
    export RAW_FILE_SORT="$RAW_FILE_SORT"
    export GRAPH_SIMPLIFY_MODE="$GRAPH_SIMPLIFY_MODE"
    export GRAPH_SIMPLIFY_RISK_THRESHOLD="$GRAPH_SIMPLIFY_RISK_THRESHOLD"
    export GRAPH_SIMPLIFY_TOPK_PER_PROCESS="$GRAPH_SIMPLIFY_TOPK_PER_PROCESS"
    export GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS="$GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS"
    export GRAPH_SIMPLIFY_REPEAT_NORM="$GRAPH_SIMPLIFY_REPEAT_NORM"
    export MW_PRR_ALPHA="0.2"
    export MW_PRR_ATTENTION_BETA="$beta"
    export WRA_RR_WINDOW="$WRA_RR_WINDOW"
    export GPU_RESERVE_ENABLE="$GPU_RESERVE_ENABLE"
    export GPU_RESERVE_MB="$GPU_RESERVE_MB"
    export GPU_RESERVE_STRICT="$GPU_RESERVE_STRICT"
    export SKIP_COMPLETED="$SKIP_COMPLETED"
    unset REUSE_RUN REUSE_PROCESSED_DIR REUSE_METADATA_DIR
    bash scripts/run_cadets_v3_ea_verdict.sh
  )
}

if [[ "$RUN_OFF" == "1" ]]; then
  run_mode "off" "off" "" "$MW_PRR_ATTENTION_BETA" "no_reduction_control"
fi
run_mode "prefix_tree" "prefix_tree" "" "$MW_PRR_ATTENTION_BETA" "malsnif_algorithm1_control"
run_mode "winnowing_anchor" "winnowing_anchor" "$WRA_RR_WINDOW" "$MW_PRR_ATTENTION_BETA" "wra_rr_candidate"

python - "$EXPERIMENT_OUT" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
matrix_path = root / "run_matrix.tsv"
if not matrix_path.exists():
    raise SystemExit(f"missing run matrix: {matrix_path}")

matrix = []
with matrix_path.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        matrix.append(row)

metric_keys = [
    "f1", "precision", "recall", "mcc", "average_precision", "roc_auc",
    "train_seconds", "node_event_reduction_ratio", "simplified_node_events",
    "node_event_weight_mean", "node_event_weight_max", "node_event_high_weight_ratio",
]
primary_metrics = ["f1", "mcc", "recall", "precision", "average_precision", "roc_auc"]

def read_json(path):
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}

def to_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan

def mean(values):
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan

def std(values):
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n <= 1:
        return 0.0 if n == 1 else math.nan
    mu = sum(vals) / n
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))

def first_graph_simplify_mode(meta):
    stats = meta.get("stats") or []
    for item in stats:
        gs = (item or {}).get("graph_simplification") or {}
        mode = gs.get("graph_simplify_mode")
        if mode:
            return mode
    return meta.get("config", {}).get("graph_simplify_mode")

def safe_sum_int(stats, key):
    return sum(int(s.get(key, 0) or 0) for s in stats)

rows = []
for spec in matrix:
    label = spec["label"]
    mode_root = root / label
    meta = read_json(mode_root / "cache" / "analysis" / "preprocess" / "metadata.json")
    stats = meta.get("stats") or []
    before = safe_sum_int(stats, "node_events_before_reduction")
    after = safe_sum_int(stats, "node_events_after_reduction")
    final_node_events = safe_sum_int(stats, "simplified_node_events")
    reduction_ratio = (before - after) / max(before, 1)
    weight_den = max(final_node_events, 1)
    node_event_weight_mean = sum(float(s.get("node_event_weight_mean", 1.0) or 1.0) * int(s.get("simplified_node_events", 0) or 0) for s in stats) / weight_den
    node_event_weight_max = max([float(s.get("node_event_weight_max", 1.0) or 1.0) for s in stats] or [1.0])
    node_event_high_weight_ratio = sum(float(s.get("node_event_high_weight_ratio", 0.0) or 0.0) * int(s.get("simplified_node_events", 0) or 0) for s in stats) / weight_den
    vocab_size = meta.get("vocab_size")
    num_graphs = meta.get("num_graphs")
    graph_simplify_mode = first_graph_simplify_mode(meta)
    for seed_dir in sorted(mode_root.glob("seed_*")):
        summary = seed_dir / "analysis" / "summary.csv"
        if not summary.exists():
            continue
        with summary.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("experiment") != "E1_eha_only":
                    continue
                out = dict(row)
                out.update(spec)
                out["seed"] = seed_dir.name.replace("seed_", "")
                out["num_graphs"] = num_graphs
                out["vocab_size"] = vocab_size
                out["graph_simplify_mode"] = graph_simplify_mode
                out["node_events_before_reduction"] = before
                out["node_events_after_reduction"] = after
                out["node_event_reduction_ratio"] = reduction_ratio
                out["simplified_node_events"] = final_node_events
                out["node_event_weight_mean"] = node_event_weight_mean
                out["node_event_weight_max"] = node_event_weight_max
                out["node_event_high_weight_ratio"] = node_event_high_weight_ratio
                out["mode_root"] = str(mode_root)
                rows.append(out)

fields = [
    "label", "redundancy_mode", "window", "beta", "role", "seed", "experiment",
    "f1", "precision", "recall", "mcc", "average_precision", "roc_auc",
    "tp", "fp", "tn", "fn", "threshold", "best_f1", "best_f1_threshold",
    "best_f1_gap", "train_seconds", "num_graphs", "vocab_size",
    "graph_simplify_mode", "node_events_before_reduction", "node_events_after_reduction",
    "node_event_reduction_ratio", "simplified_node_events", "node_event_weight_mean",
    "node_event_weight_max", "node_event_high_weight_ratio", "mode_root",
]
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
summary_path = root / "summary_wra_rr.csv"
with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

by_label_rows = defaultdict(list)
for row in rows:
    by_label_rows[row["label"]].append(row)

agg_rows = []
for spec in matrix:
    label = spec["label"]
    bucket = by_label_rows.get(label, [])
    if not bucket:
        continue
    item = {k: spec.get(k, "") for k in ["label", "redundancy_mode", "window", "beta", "role"]}
    item.update({
        "n_seeds": len(bucket),
        "num_graphs": bucket[0].get("num_graphs"),
        "vocab_size": bucket[0].get("vocab_size"),
        "graph_simplify_mode": bucket[0].get("graph_simplify_mode"),
        "node_events_before_reduction": bucket[0].get("node_events_before_reduction"),
        "node_events_after_reduction": bucket[0].get("node_events_after_reduction"),
        "simplified_node_events": bucket[0].get("simplified_node_events"),
        "node_event_reduction_ratio": bucket[0].get("node_event_reduction_ratio"),
        "node_event_weight_mean": bucket[0].get("node_event_weight_mean"),
        "node_event_weight_max": bucket[0].get("node_event_weight_max"),
        "node_event_high_weight_ratio": bucket[0].get("node_event_high_weight_ratio"),
    })
    for key in metric_keys:
        vals = [to_float(r.get(key)) for r in bucket]
        item[f"{key}_mean"] = mean(vals)
        item[f"{key}_std"] = std(vals)
    agg_rows.append(item)

agg_fields = [
    "label", "redundancy_mode", "window", "beta", "role", "n_seeds", "num_graphs",
    "vocab_size", "graph_simplify_mode", "node_events_before_reduction",
    "node_events_after_reduction", "simplified_node_events",
    "node_event_reduction_ratio", "node_event_weight_mean", "node_event_weight_max",
    "node_event_high_weight_ratio",
]
for key in metric_keys:
    agg_fields.append(f"{key}_mean")
    agg_fields.append(f"{key}_std")
agg_path = root / "summary_wra_rr_agg.csv"
with agg_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=agg_fields)
    writer.writeheader()
    writer.writerows(agg_rows)

pair_rows = []
prefix_by_seed = {r["seed"]: r for r in by_label_rows.get("prefix_tree", [])}
for spec in matrix:
    label = spec["label"]
    if label == "prefix_tree" or spec.get("redundancy_mode") == "off":
        continue
    for row in by_label_rows.get(label, []):
        base = prefix_by_seed.get(row["seed"])
        if not base:
            continue
        item = {"label": label, "mode": spec.get("redundancy_mode", ""), "window": spec.get("window", ""), "seed": row["seed"]}
        for key in primary_metrics + ["train_seconds"]:
            item[f"delta_{key}"] = to_float(row.get(key)) - to_float(base.get(key))
        item["candidate_f1"] = row.get("f1")
        item["prefix_f1"] = base.get("f1")
        pair_rows.append(item)

pair_fields = ["label", "mode", "window", "seed", "candidate_f1", "prefix_f1"] + [f"delta_{k}" for k in primary_metrics + ["train_seconds"]]
pair_path = root / "paired_vs_prefix.csv"
with pair_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=pair_fields)
    writer.writeheader()
    writer.writerows(pair_rows)

paired_agg = []
by_pair_label = defaultdict(list)
for row in pair_rows:
    by_pair_label[row["label"]].append(row)
for label, bucket in by_pair_label.items():
    item = {"label": label, "mode": bucket[0].get("mode", ""), "window": bucket[0].get("window", ""), "n_paired_seeds": len(bucket)}
    for key in primary_metrics:
        vals = [to_float(r.get(f"delta_{key}")) for r in bucket]
        item[f"delta_{key}_mean"] = mean(vals)
        item[f"delta_{key}_std"] = std(vals)
        item[f"delta_{key}_wins"] = sum(1 for v in vals if not math.isnan(v) and v > 1e-9)
        item[f"delta_{key}_losses"] = sum(1 for v in vals if not math.isnan(v) and v < -1e-9)
        item[f"delta_{key}_ties"] = sum(1 for v in vals if not math.isnan(v) and abs(v) <= 1e-9)
    paired_agg.append(item)

paired_agg_fields = ["label", "mode", "window", "n_paired_seeds"]
for key in primary_metrics:
    paired_agg_fields += [f"delta_{key}_mean", f"delta_{key}_std", f"delta_{key}_wins", f"delta_{key}_losses", f"delta_{key}_ties"]
paired_agg_path = root / "paired_vs_prefix_agg.csv"
with paired_agg_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=paired_agg_fields)
    writer.writeheader()
    writer.writerows(paired_agg)

def fmt(value, digits=6):
    try:
        value = float(value)
    except Exception:
        return str(value) if value is not None else ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"

def verdict_for_pair(item):
    n = int(item.get("n_paired_seeds") or 0)
    f1_delta = float(item.get("delta_f1_mean") or 0.0)
    mcc_delta = float(item.get("delta_mcc_mean") or 0.0)
    recall_delta = float(item.get("delta_recall_mean") or 0.0)
    f1_wins = int(item.get("delta_f1_wins") or 0)
    if n >= 5 and f1_delta >= 0.003 and mcc_delta >= 0.003 and recall_delta >= -0.001 and f1_wins >= math.ceil(0.6 * n):
        return "positive_candidate_signal_requires_confirmation"
    if f1_delta <= -0.003 or mcc_delta <= -0.003 or recall_delta <= -0.005:
        return "negative_or_recall_risk"
    return "tie_or_inconclusive"

report = root / "WRA_RR_VERDICT_REPORT.md"
with report.open("w", encoding="utf-8") as f:
    f.write("# WRA-RR redundancy verdict report\n\n")
    f.write("This report compares off, MalSnif prefix_tree, and WRA-RR winnowing_anchor on the same strict-qualified CADETS prefix and paired seeds. WRA-RR is a one-parameter winnowing representative-anchor reducer; it should not be claimed as an improvement unless paired deltas exceed the predefined rule.\n\n")
    f.write("## Mean +/- std across seeds\n\n")
    f.write("| label | mode | window | seeds | F1 | Precision | Recall | MCC | AP | ROC-AUC | Train(s) | reduction | weight_mean | high_weight_ratio |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in agg_rows:
        f.write(
            f"| {r.get('label')} | {r.get('redundancy_mode')} | {r.get('window')} | {r.get('n_seeds')} | "
            f"{fmt(r.get('f1_mean'))} +/- {fmt(r.get('f1_std'), 4)} | "
            f"{fmt(r.get('precision_mean'))} +/- {fmt(r.get('precision_std'), 4)} | "
            f"{fmt(r.get('recall_mean'))} +/- {fmt(r.get('recall_std'), 4)} | "
            f"{fmt(r.get('mcc_mean'))} +/- {fmt(r.get('mcc_std'), 4)} | "
            f"{fmt(r.get('average_precision_mean'))} +/- {fmt(r.get('average_precision_std'), 4)} | "
            f"{fmt(r.get('roc_auc_mean'))} +/- {fmt(r.get('roc_auc_std'), 4)} | "
            f"{fmt(r.get('train_seconds_mean'), 2)} +/- {fmt(r.get('train_seconds_std'), 2)} | "
            f"{fmt(r.get('node_event_reduction_ratio'))} | {fmt(r.get('node_event_weight_mean'))} | {fmt(r.get('node_event_high_weight_ratio'))} |\n"
        )
    f.write("\n## Paired candidate vs prefix_tree\n\n")
    f.write("| label | mode | window | paired seeds | delta F1 | F1 win/loss/tie | delta MCC | delta Recall | verdict |\n")
    f.write("|---|---|---:|---:|---:|---|---:|---:|---|\n")
    for item in paired_agg:
        f.write(
            f"| {item.get('label')} | {item.get('mode')} | {item.get('window')} | {item.get('n_paired_seeds')} | "
            f"{fmt(item.get('delta_f1_mean'))} +/- {fmt(item.get('delta_f1_std'), 4)} | "
            f"{item.get('delta_f1_wins')}/{item.get('delta_f1_losses')}/{item.get('delta_f1_ties')} | "
            f"{fmt(item.get('delta_mcc_mean'))} | {fmt(item.get('delta_recall_mean'))} | {verdict_for_pair(item)} |\n"
        )
    f.write("\n## Conservative success rule\n\n")
    f.write("A candidate reducer is considered to have a positive signal only if it beats prefix_tree by at least +0.003 mean F1 and +0.003 mean MCC, loses no more than 0.001 mean Recall, and wins F1 on at least 60% of paired seeds. Otherwise the result is tie/inconclusive or negative.\n\n")
    f.write("## Files\n\n")
    f.write(f"- per-seed summary: `{summary_path}`\n")
    f.write(f"- aggregate summary: `{agg_path}`\n")
    f.write(f"- paired deltas: `{pair_path}`\n")
    f.write(f"- paired aggregate: `{paired_agg_path}`\n")

print(json.dumps({
    "summary": str(summary_path),
    "aggregate_summary": str(agg_path),
    "paired": str(pair_path),
    "paired_aggregate": str(paired_agg_path),
    "report": str(report),
    "rows": len(rows),
    "agg_rows": len(agg_rows),
}, ensure_ascii=False, indent=2))
PY

collect_wra_rr_bundle
trap - EXIT

echo "[done] base_out=$ABS_BASE_OUT"
echo "[done] selected_max_events=$SELECTED_MAX_EVENTS"
echo "[done] precheck_report=$ABS_PRECHECK_OUT/STRICT_SPLIT_PRECHECK.md"
echo "[done] experiment_out=$ABS_EXPERIMENT_OUT"
echo "[done] summary=$ABS_EXPERIMENT_OUT/summary_wra_rr.csv"
echo "[done] aggregate_summary=$ABS_EXPERIMENT_OUT/summary_wra_rr_agg.csv"
echo "[done] report=$ABS_EXPERIMENT_OUT/WRA_RR_VERDICT_REPORT.md"
echo "[done] analysis_bundle=$ABS_BUNDLE_DIR"
echo "[done] single_folder_analysis_bundle=$ABS_COLLECTED_BUNDLE_DIR"
echo "[send me] $ABS_COLLECTED_BUNDLE_DIR"
