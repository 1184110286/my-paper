#!/usr/bin/env bash
set -euo pipefail

# Rigorous one-key comparison for E1_eha_only semantic encoders on CADETS/THEIA.
#
# Main comparison:
#   E1_eha_only_mcbg       : original MCBG semantic encoder
#   E1_eha_only_rgd_bigru  : RGD-BiGRU-MCBG semantic encoder
#
# Fixed controls:
#   - redundancy_mode=target_boundary (TBB-RR)
#   - model_variant=ea_st_hgan_mcbg
#   - ST-HGAN + EHA only unchanged
#   - paired seeds and shared graph cache per dataset
#
# Default rigorous run on GPU 1:
#   DEVICE=1 EVAL_DEVICE=1 bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
#
# Faster dry run:
#   EXPERIMENT_LEVEL=smoke DEVICE=1 EVAL_DEVICE=1 bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
#
# Very large run, if your data/GPU budget allows it:
#   EXPERIMENT_LEVEL=full DEVICE=1 EVAL_DEVICE=1 bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
# Needed by CUDA deterministic modes for some matrix ops. Harmless otherwise.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# The project seed helper honors this flag when available; older versions ignore it.
export MALSNIF_DETERMINISTIC="${MALSNIF_DETERMINISTIC:-1}"

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

EXPERIMENT_LEVEL="${EXPERIMENT_LEVEL:-rigorous}"
case "$EXPERIMENT_LEVEL" in
  smoke)
    : "${CADETS_EA_PRESET:=smoke}"
    : "${SEEDS:=42}"
    : "${EPOCHS:=1}"
    : "${PATIENCE:=2}"
    : "${VAL_EVERY:=1}"
    ;;
  medium)
    : "${CADETS_EA_PRESET:=calib8m}"
    : "${SEEDS:=42 43 44}"
    : "${EPOCHS:=10}"
    : "${PATIENCE:=5}"
    : "${VAL_EVERY:=1}"
    ;;
  rigorous)
    : "${CADETS_EA_PRESET:=calib12m}"
    : "${SEEDS:=42 43 44 45 46}"
    : "${EPOCHS:=15}"
    : "${PATIENCE:=8}"
    : "${VAL_EVERY:=1}"
    ;;
  full)
    : "${CADETS_EA_PRESET:=full}"
    : "${SEEDS:=42 43 44 45 46}"
    : "${EPOCHS:=20}"
    : "${PATIENCE:=10}"
    : "${VAL_EVERY:=1}"
    ;;
  *)
    echo "[ERROR] EXPERIMENT_LEVEL must be smoke|medium|rigorous|full, got: $EXPERIMENT_LEVEL" >&2
    exit 2
    ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${BASE_OUT:-runs/e1_rgd_bigru_tbb_rr_theia_cadets_RIGOROUS_${EXPERIMENT_LEVEL}_${TS}}"
RUN_DATASETS="${RUN_DATASETS:-cadets theia}"
# Keep the rigorous default narrow: baseline vs candidate. Set RUN_ENCODERS="mcbg rgd_bigru gdtc" for a reference third arm.
RUN_ENCODERS="${RUN_ENCODERS:-mcbg rgd_bigru}"
DEVICE="${DEVICE:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}"
DIM="${DIM:-64}"
MAX_EVENTS_PER_NODE="${MAX_EVENTS_PER_NODE:-64}"
MAX_EVENTS_PER_EDGE="${MAX_EVENTS_PER_EDGE:-4}"
HGAN_TOPK="${HGAN_TOPK:-20}"
TBB_RR_TARGET_COMPRESSION="${TBB_RR_TARGET_COMPRESSION:-0.90}"
MW_PRR_ATTENTION_BETA="${MW_PRR_ATTENTION_BETA:-1.0}"
RGD_KERNEL_SIZE="${RGD_KERNEL_SIZE:-3}"
RGD_DILATIONS="${RGD_DILATIONS:-1,2}"
RGD_DROPOUT="${RGD_DROPOUT:-0.2}"
RGD_RESIDUAL_SCALE_INIT="${RGD_RESIDUAL_SCALE_INIT:-0.1}"
RGD_DEPTHWISE_SEPARABLE="${RGD_DEPTHWISE_SEPARABLE:-1}"
RGD_USE_EVENT_WEIGHT_POOLING="${RGD_USE_EVENT_WEIGHT_POOLING:-1}"
GDTC_KERNEL_SIZE="${GDTC_KERNEL_SIZE:-3}"
GDTC_DILATIONS="${GDTC_DILATIONS:-1,2,4}"
GDTC_DROPOUT="${GDTC_DROPOUT:-0.2}"
GDTC_USE_EVENT_WEIGHT_POOLING="${GDTC_USE_EVENT_WEIGHT_POOLING:-1}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-val_average_precision}"
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-val_f1_min_recall}"
THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.95}"
NODE_SCOPE="${NODE_SCOPE:-process}"
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-float16}"
CACHE_GRAPHS_IN_MEMORY="${CACHE_GRAPHS_IN_MEMORY:-0}"
PLOT_MODE="${PLOT_MODE:-essential}"
TOP_ALERTS_PER_GRAPH="${TOP_ALERTS_PER_GRAPH:-20}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
GPU_RESERVE_ENABLE="${GPU_RESERVE_ENABLE:-1}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-512}"
GPU_RESERVE_STRICT="${GPU_RESERVE_STRICT:-1}"
GPU_WAIT_ENABLE="${GPU_WAIT_ENABLE:-0}"
GPU_WAIT_MIN_FREE_MB="${GPU_WAIT_MIN_FREE_MB:-4400}"

# Non-inferiority / practical effect thresholds used only for the report verdict.
NI_F1_MARGIN="${NI_F1_MARGIN:-0.002}"
NI_RECALL_MARGIN="${NI_RECALL_MARGIN:-0.005}"
POSITIVE_F1_MARGIN="${POSITIVE_F1_MARGIN:-0.001}"
POSITIVE_RECALL_MARGIN="${POSITIVE_RECALL_MARGIN:-0.000}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

mkdir -p "$BASE_OUT"
ABS_BASE_OUT="$(cd "$BASE_OUT" && pwd)"

python - <<'PY' | tee "$BASE_OUT/ENVIRONMENT.txt"
import json, os, platform, subprocess, sys
try:
    import torch
except Exception as exc:
    torch = None
    torch_error = repr(exc)
else:
    torch_error = None
info = {
    "python": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "torch_version": getattr(torch, "__version__", None),
    "cuda_available": bool(torch and torch.cuda.is_available()),
    "cuda_device_count": torch.cuda.device_count() if torch else None,
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch and torch.cuda.is_available() else [],
    "torch_error": torch_error,
    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "MALSNIF_DETERMINISTIC": os.environ.get("MALSNIF_DETERMINISTIC"),
    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
}
for cmd_name, cmd in {
    "git_rev": ["git", "rev-parse", "HEAD"],
    "git_status_short": ["git", "status", "--short"],
}.items():
    try:
        info[cmd_name] = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5).strip()
    except Exception as exc:
        info[cmd_name] = f"unavailable: {exc}"
print(json.dumps(info, ensure_ascii=False, indent=2))
PY

cat > "$BASE_OUT/E1_RGD_BIGRU_TBB_RR_RIGOROUS_EXPERIMENT_PLAN.md" <<EOF_PLAN
# Rigorous E1-RGD-BiGRU-MCBG TBB-RR experiment plan

Goal: perform a stricter paired-seed comparison between the existing E1_eha_only
MCBG encoder and the RGD-BiGRU-MCBG candidate on CADETS and THEIA.

Why this is stricter than the quick validation run:
- more seeds by default: $SEEDS
- larger preset by default: $CADETS_EA_PRESET
- more epochs/patience: epochs=$EPOCHS, patience=$PATIENCE, val_every=$VAL_EVERY
- paired seed design and shared TBB-RR graph cache per dataset
- fixed model-selection and thresholding policy
- deterministic-mode environment flags recorded in ENVIRONMENT.txt
- aggregate report includes paired deltas, exact sign-test p-values, and bootstrap confidence intervals

Fixed pipeline:
- model_variant: ea_st_hgan_mcbg
- adaptive mechanism: EHA only
- redundancy_mode: target_boundary (TBB-RR)
- graph branch: ST-HGAN + EHA unchanged
- node_scope: $NODE_SCOPE
- model_selection_metric: $MODEL_SELECTION_METRIC
- threshold_strategy: $THRESHOLD_STRATEGY
- threshold_min_recall: $THRESHOLD_MIN_RECALL

Controls:
- experiment_level: $EXPERIMENT_LEVEL
- datasets: $RUN_DATASETS
- encoders: $RUN_ENCODERS
- seeds: $SEEDS
- device: $DEVICE, eval_device: $EVAL_DEVICE
- preset: $CADETS_EA_PRESET
- epochs: $EPOCHS
- dim: $DIM
- max_events_per_node: $MAX_EVENTS_PER_NODE
- max_events_per_edge: $MAX_EVENTS_PER_EDGE
- hgan_topk: $HGAN_TOPK
- tbb_rr_target_compression: $TBB_RR_TARGET_COMPRESSION
- event_weight_attention_beta: $MW_PRR_ATTENTION_BETA
- rgd_kernel_size: $RGD_KERNEL_SIZE
- rgd_dilations: $RGD_DILATIONS
- rgd_dropout: $RGD_DROPOUT
- rgd_residual_scale_init: $RGD_RESIDUAL_SCALE_INIT
- rgd_depthwise_separable: $RGD_DEPTHWISE_SEPARABLE
- rgd_use_event_weight_pooling: $RGD_USE_EVENT_WEIGHT_POOLING
- output: $ABS_BASE_OUT

Primary metrics: F1, Recall, Precision, MCC, Average Precision, ROC-AUC,
false positives, false negatives, training time, and CUDA peak memory.

Conservative report rule:
- non-inferior if mean_delta_f1 >= -$NI_F1_MARGIN and mean_delta_recall >= -$NI_RECALL_MARGIN
- positive if mean_delta_f1 >= $POSITIVE_F1_MARGIN and mean_delta_recall >= $POSITIVE_RECALL_MARGIN with majority paired wins
EOF_PLAN

encoder_to_label() {
  case "$1" in
    mcbg) echo "mcbg" ;;
    rgd_bigru|rgd_bigru_mcbg|e1_rgd_bigru_mcbg) echo "rgd_bigru" ;;
    gdtc|gdtc_mcbg|e1_gdtc_mcbg) echo "gdtc" ;;
    *) echo "$1" | tr -c 'A-Za-z0-9_' '_' ;;
  esac
}

encoder_to_config_name() {
  case "$1" in
    mcbg) echo "mcbg" ;;
    rgd_bigru|rgd_bigru_mcbg|e1_rgd_bigru_mcbg) echo "rgd_bigru_mcbg" ;;
    gdtc|gdtc_mcbg|e1_gdtc_mcbg) echo "gdtc_mcbg" ;;
    *) echo "$1" ;;
  esac
}

run_one_dataset_encoder() {
  local dataset="$1"
  local label="$2"
  local semantic_encoder="$3"
  local raw_dir label_dir raw_glob raw_file_sort dataset_name cache_root

  case "$dataset" in
    cadets)
      raw_dir="${CADETS_RAW_DIR:-data/raw/darpa_tc/cadets/e3/cdm}"
      label_dir="${CADETS_LABEL_DIR:-data/raw/darpa_tc/cadets/e3/labels}"
      raw_glob="${CADETS_RAW_GLOB:-ta1-cadets-e3-official*.json*}"
      raw_file_sort="${CADETS_RAW_FILE_SORT:-cdm_shards}"
      dataset_name="cadets_e3_e1_${label}_tbb_rr_rigorous"
      ;;
    theia)
      raw_dir="${THEIA_RAW_DIR:-data/raw/darpa_tc/theia/e3/cdm}"
      label_dir="${THEIA_LABEL_DIR:-data/raw/darpa_tc/theia/e3/labels}"
      raw_glob="${THEIA_RAW_GLOB:-ta1-theia-e3-official*.json*}"
      raw_file_sort="${THEIA_RAW_FILE_SORT:-cdm_shards}"
      dataset_name="theia_e3_e1_${label}_tbb_rr_rigorous"
      ;;
    *)
      echo "[ERROR] unknown dataset=$dataset; expected cadets or theia" >&2
      exit 2
      ;;
  esac

  cache_root="$ABS_BASE_OUT/_cache/${dataset}_${CADETS_EA_PRESET}_target_boundary_tbb${TBB_RR_TARGET_COMPRESSION}_win${WINDOW_EVENTS:-preset}_max${MAX_EVENTS:-preset}_dim${DIM}"
  mkdir -p "$ABS_BASE_OUT/$dataset/$label"

  echo "========== dataset=$dataset label=$label semantic_encoder=$semantic_encoder =========="
  (
    export PARENT_OUT="$ABS_BASE_OUT/$dataset/$label"
    export RAW_DIR="$raw_dir"
    export LABEL_DIR="$label_dir"
    export RAW_GLOB="$raw_glob"
    export RAW_FILE_SORT="$raw_file_sort"
    export DATASET_NAME="$dataset_name"
    export CADETS_CACHE_ROOT="$cache_root"
    export CADETS_EA_PRESET="$CADETS_EA_PRESET"
    export REDUNDANCY_MODE="target_boundary"
    export TBB_RR_TARGET_COMPRESSION="$TBB_RR_TARGET_COMPRESSION"
    export MW_PRR_ATTENTION_BETA="$MW_PRR_ATTENTION_BETA"
    export SEMANTIC_ENCODER="$semantic_encoder"
    export E1_SEMANTIC_ENCODER="$semantic_encoder"
    export E1_EXPERIMENT_NAME="E1_eha_only_${label}"
    export RGD_KERNEL_SIZE="$RGD_KERNEL_SIZE"
    export RGD_DILATIONS="$RGD_DILATIONS"
    export RGD_DROPOUT="$RGD_DROPOUT"
    export RGD_RESIDUAL_SCALE_INIT="$RGD_RESIDUAL_SCALE_INIT"
    export RGD_DEPTHWISE_SEPARABLE="$RGD_DEPTHWISE_SEPARABLE"
    export RGD_USE_EVENT_WEIGHT_POOLING="$RGD_USE_EVENT_WEIGHT_POOLING"
    export GDTC_KERNEL_SIZE="$GDTC_KERNEL_SIZE"
    export GDTC_DILATIONS="$GDTC_DILATIONS"
    export GDTC_DROPOUT="$GDTC_DROPOUT"
    export GDTC_USE_EVENT_WEIGHT_POOLING="$GDTC_USE_EVENT_WEIGHT_POOLING"
    export RUN_B0=0 RUN_B1=0 RUN_E0=0 RUN_E1=1 RUN_E2=0 RUN_E3=0 RUN_E4=0 RUN_E5=0 RUN_E6=0 RUN_E7=0 RUN_ALL_EA=0
    export SEEDS="$SEEDS"
    export DEVICE="$DEVICE"
    export EVAL_DEVICE="$EVAL_DEVICE"
    export EPOCHS="$EPOCHS"
    export DIM="$DIM"
    export MAX_EVENTS_PER_NODE="$MAX_EVENTS_PER_NODE"
    export MAX_EVENTS_PER_EDGE="$MAX_EVENTS_PER_EDGE"
    export HGAN_TOPK="$HGAN_TOPK"
    export SKIP_COMPLETED="$SKIP_COMPLETED"
    export VAL_EVERY="$VAL_EVERY"
    export PATIENCE="$PATIENCE"
    export MODEL_SELECTION_METRIC="$MODEL_SELECTION_METRIC"
    export THRESHOLD_STRATEGY="$THRESHOLD_STRATEGY"
    export THRESHOLD_MIN_RECALL="$THRESHOLD_MIN_RECALL"
    export NODE_SCOPE="$NODE_SCOPE"
    export USE_AMP="$USE_AMP"
    export AMP_DTYPE="$AMP_DTYPE"
    export CACHE_GRAPHS_IN_MEMORY="$CACHE_GRAPHS_IN_MEMORY"
    export PLOT_MODE="$PLOT_MODE"
    export TOP_ALERTS_PER_GRAPH="$TOP_ALERTS_PER_GRAPH"
    export GRAPH_SIMPLIFY_MODE="${GRAPH_SIMPLIFY_MODE:-leaf}"
    export GRAPH_SIMPLIFY_RISK_THRESHOLD="${GRAPH_SIMPLIFY_RISK_THRESHOLD:-0.62}"
    export GRAPH_SIMPLIFY_TOPK_PER_PROCESS="${GRAPH_SIMPLIFY_TOPK_PER_PROCESS:-0}"
    export GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS="${GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS:-1000000000}"
    export GRAPH_SIMPLIFY_REPEAT_NORM="${GRAPH_SIMPLIFY_REPEAT_NORM:-8}"
    export GPU_RESERVE_ENABLE="$GPU_RESERVE_ENABLE"
    export GPU_RESERVE_MB="$GPU_RESERVE_MB"
    export GPU_RESERVE_STRICT="$GPU_RESERVE_STRICT"
    export GPU_WAIT_ENABLE="$GPU_WAIT_ENABLE"
    export GPU_WAIT_MIN_FREE_MB="$GPU_WAIT_MIN_FREE_MB"
    unset REUSE_RUN REUSE_PROCESSED_DIR REUSE_METADATA_DIR
    bash scripts/run_cadets_v3_ea_verdict.sh
  )
}

for dataset in $RUN_DATASETS; do
  for enc in $RUN_ENCODERS; do
    label="$(encoder_to_label "$enc")"
    semantic_encoder="$(encoder_to_config_name "$enc")"
    run_one_dataset_encoder "$dataset" "$label" "$semantic_encoder"
  done
done

python - "$ABS_BASE_OUT" "$BOOTSTRAP_SAMPLES" "$NI_F1_MARGIN" "$NI_RECALL_MARGIN" "$POSITIVE_F1_MARGIN" "$POSITIVE_RECALL_MARGIN" <<'PY'
import csv
import json
import math
import random
import re
import shutil
import statistics as stats
import sys
from pathlib import Path

root = Path(sys.argv[1])
bootstrap_samples = int(sys.argv[2])
ni_f1_margin = float(sys.argv[3])
ni_recall_margin = float(sys.argv[4])
pos_f1_margin = float(sys.argv[5])
pos_recall_margin = float(sys.argv[6])

METRICS = [
    "f1", "precision", "recall", "specificity", "mcc", "average_precision", "roc_auc",
    "threshold", "tp", "fp", "tn", "fn", "num_samples", "prevalence", "best_f1", "best_f1_threshold",
]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

rows = []
seen = set()
# Only scan canonical run outputs, never analysis_bundle copies.
for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in {"_cache", "analysis_bundle"}]):
    dataset = dataset_dir.name
    for label_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "analysis_bundle"]):
        label = label_dir.name
        for metrics_path in sorted(label_dir.glob("seed_*/analysis/*/metrics_test_compact.json")):
            run_dir = metrics_path.parent
            key = str(run_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            payload = read_json(metrics_path)
            metrics = payload.get("metrics", payload)
            summary_path = run_dir / "train_summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {}
            analysis = read_json(run_dir / "run_analysis.json") if (run_dir / "run_analysis.json").exists() else {}
            seed = summary.get("seed")
            if seed is None:
                m = re.search(r"seed_(\d+)", str(metrics_path))
                seed = int(m.group(1)) if m else None
            row = {
                "dataset": dataset,
                "label": label,
                "experiment": run_dir.name,
                "seed": seed,
                "semantic_encoder": summary.get("semantic_encoder") or metrics.get("semantic_encoder"),
                "train_seconds": summary.get("train_total_seconds", summary.get("train_seconds")),
                "cuda_peak_allocated_mb": summary.get("max_cuda_peak_allocated_mb"),
                "cuda_peak_reserved_mb": summary.get("max_cuda_peak_reserved_mb"),
                "best_epoch": summary.get("best_epoch"),
                "warnings": " | ".join(str(x) for x in analysis.get("warnings", [])) if isinstance(analysis, dict) else "",
                "run_dir": str(run_dir),
                "config": str(run_dir / "config.resolved.yaml"),
                "metrics_file": str(metrics_path),
            }
            for k in METRICS:
                row[k] = metrics.get(k)
            rows.append(row)

summary_csv = root / "E1_RGD_BIGRU_TBB_RR_RIGOROUS_SUMMARY.csv"
fieldnames = [
    "dataset", "label", "experiment", "seed", "semantic_encoder",
    "f1", "precision", "recall", "specificity", "mcc", "average_precision", "roc_auc", "threshold",
    "tp", "fp", "tn", "fn", "num_samples", "prevalence", "best_f1", "best_f1_threshold",
    "best_epoch", "train_seconds", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb", "warnings", "run_dir", "config", "metrics_file",
]
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

# Aggregate per dataset/label.
by = {}
for r in rows:
    by.setdefault((r["dataset"], r["label"]), []).append(r)

agg_rows = []
for (dataset, label), rs in sorted(by.items()):
    out = {"dataset": dataset, "label": label, "runs": len(rs)}
    for metric in ["f1", "precision", "recall", "mcc", "average_precision", "roc_auc", "fp", "fn", "train_seconds", "cuda_peak_allocated_mb"]:
        vals = [fnum(r.get(metric)) for r in rs]
        vals = [v for v in vals if v is not None]
        out[f"mean_{metric}"] = sum(vals) / len(vals) if vals else ""
        out[f"std_{metric}"] = stats.stdev(vals) if len(vals) > 1 else (0.0 if vals else "")
    agg_rows.append(out)

agg_csv = root / "E1_RGD_BIGRU_TBB_RR_RIGOROUS_AGG.csv"
agg_fields = ["dataset", "label", "runs"]
for metric in ["f1", "precision", "recall", "mcc", "average_precision", "roc_auc", "fp", "fn", "train_seconds", "cuda_peak_allocated_mb"]:
    agg_fields += [f"mean_{metric}", f"std_{metric}"]
with agg_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=agg_fields)
    w.writeheader()
    w.writerows(agg_rows)

# Paired deltas vs mcbg.
by_seed = {}
for r in rows:
    by_seed.setdefault((r["dataset"], str(r.get("seed"))), {})[r["label"]] = r
paired_rows = []
for (dataset, seed), item in sorted(by_seed.items()):
    if "mcbg" not in item:
        continue
    for cand_label, cand in sorted(item.items()):
        if cand_label == "mcbg":
            continue
        base = item["mcbg"]
        pr = {"dataset": dataset, "candidate": cand_label, "seed": seed}
        for metric in ["f1", "precision", "recall", "mcc", "average_precision", "roc_auc", "fp", "fn", "train_seconds", "cuda_peak_allocated_mb"]:
            a = fnum(cand.get(metric)); b = fnum(base.get(metric))
            pr[f"delta_{metric}"] = "" if a is None or b is None else a - b
        paired_rows.append(pr)

paired_csv = root / "E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_DELTAS.csv"
paired_fields = ["dataset", "candidate", "seed"] + [f"delta_{m}" for m in ["f1", "precision", "recall", "mcc", "average_precision", "roc_auc", "fp", "fn", "train_seconds", "cuda_peak_allocated_mb"]]
with paired_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=paired_fields)
    w.writeheader()
    w.writerows(paired_rows)


def exact_two_sided_sign_p(vals):
    nz = [v for v in vals if abs(v) > 1e-12]
    n = len(nz)
    if n == 0:
        return 1.0
    wins = sum(v > 0 for v in nz)
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def bootstrap_ci(vals, samples=2000, seed=20260630):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return (None, None, None)
    mean = sum(vals) / len(vals)
    if len(vals) == 1 or samples <= 0:
        return (mean, mean, mean)
    rng = random.Random(seed + len(vals))
    boots = []
    n = len(vals)
    for _ in range(samples):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * (samples - 1))]
    hi = boots[int(0.975 * (samples - 1))]
    return mean, lo, hi

paired_agg_rows = []
for dataset in sorted({r["dataset"] for r in paired_rows}):
    for candidate in sorted({r["candidate"] for r in paired_rows if r["dataset"] == dataset}):
        rs = [r for r in paired_rows if r["dataset"] == dataset and r["candidate"] == candidate]
        out = {"dataset": dataset, "candidate": candidate, "paired_runs": len(rs)}
        f1_vals = [fnum(r.get("delta_f1")) for r in rs]
        f1_vals = [v for v in f1_vals if v is not None]
        out["f1_wins"] = sum(v > 1e-12 for v in f1_vals)
        out["f1_losses"] = sum(v < -1e-12 for v in f1_vals)
        out["f1_ties"] = sum(abs(v) <= 1e-12 for v in f1_vals)
        out["f1_sign_test_p"] = exact_two_sided_sign_p(f1_vals)
        for metric in ["f1", "precision", "recall", "mcc", "average_precision", "roc_auc", "fp", "fn", "train_seconds", "cuda_peak_allocated_mb"]:
            vals = [fnum(r.get(f"delta_{metric}")) for r in rs]
            vals = [v for v in vals if v is not None]
            mean, lo, hi = bootstrap_ci(vals, bootstrap_samples)
            out[f"mean_delta_{metric}"] = mean if mean is not None else ""
            out[f"ci95_low_delta_{metric}"] = lo if lo is not None else ""
            out[f"ci95_high_delta_{metric}"] = hi if hi is not None else ""
        mean_f1 = fnum(out.get("mean_delta_f1")) or 0.0
        mean_recall = fnum(out.get("mean_delta_recall")) or 0.0
        majority = out["f1_wins"] >= math.ceil(max(1, len(rs)) / 2)
        if mean_f1 >= pos_f1_margin and mean_recall >= pos_recall_margin and majority:
            verdict = "positive_candidate_signal"
        elif mean_f1 >= -ni_f1_margin and mean_recall >= -ni_recall_margin:
            verdict = "non_inferior_or_tie"
        else:
            verdict = "recall_or_f1_risk"
        out["verdict"] = verdict
        paired_agg_rows.append(out)

paired_agg_csv = root / "E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_AGG.csv"
paired_agg_fields = ["dataset", "candidate", "paired_runs", "f1_wins", "f1_losses", "f1_ties", "f1_sign_test_p", "verdict"]
for metric in ["f1", "precision", "recall", "mcc", "average_precision", "roc_auc", "fp", "fn", "train_seconds", "cuda_peak_allocated_mb"]:
    paired_agg_fields += [f"mean_delta_{metric}", f"ci95_low_delta_{metric}", f"ci95_high_delta_{metric}"]
with paired_agg_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=paired_agg_fields)
    w.writeheader()
    w.writerows(paired_agg_rows)

report = root / "E1_RGD_BIGRU_TBB_RR_RIGOROUS_REPORT.md"
with report.open("w", encoding="utf-8") as f:
    f.write("# Rigorous E1-RGD-BiGRU-MCBG TBB-RR report\n\n")
    f.write(f"- summary_csv: `{summary_csv}`\n")
    f.write(f"- aggregate_csv: `{agg_csv}`\n")
    f.write(f"- paired_deltas_csv: `{paired_csv}`\n")
    f.write(f"- paired_agg_csv: `{paired_agg_csv}`\n")
    f.write("- baseline: `label=mcbg`\n")
    f.write("- candidate: `label=rgd_bigru`\n")
    f.write("- report method: paired seed deltas, exact sign-test p-value for F1 wins/losses, bootstrap CI for mean deltas\n\n")
    f.write("## Aggregate metrics\n\n")
    f.write("| dataset | label | runs | mean_f1 | mean_recall | mean_precision | mean_mcc | mean_ap | mean_roc_auc | mean_fp | mean_fn | mean_train_s | mean_cuda_mb |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in agg_rows:
        def fmt(key, n=6):
            v = fnum(r.get(key))
            return "" if v is None else f"{v:.{n}f}"
        f.write(
            f"| {r['dataset']} | {r['label']} | {r['runs']} | {fmt('mean_f1')} | {fmt('mean_recall')} | {fmt('mean_precision')} | "
            f"{fmt('mean_mcc')} | {fmt('mean_average_precision')} | {fmt('mean_roc_auc')} | {fmt('mean_fp',2)} | {fmt('mean_fn',2)} | "
            f"{fmt('mean_train_seconds',2)} | {fmt('mean_cuda_peak_allocated_mb',2)} |\n"
        )
    f.write("\n## Paired candidate-minus-baseline deltas\n\n")
    f.write("| dataset | candidate | runs | verdict | F1 wins/losses/ties | sign_p | mean_delta_f1 [95% CI] | mean_delta_recall [95% CI] | mean_delta_mcc [95% CI] | mean_delta_ap [95% CI] | mean_delta_fn | mean_delta_cuda_mb |\n")
    f.write("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in paired_agg_rows:
        def ci(metric):
            m = fnum(r.get(f"mean_delta_{metric}")); lo = fnum(r.get(f"ci95_low_delta_{metric}")); hi = fnum(r.get(f"ci95_high_delta_{metric}"))
            if m is None:
                return ""
            return f"{m:.6f} [{lo:.6f}, {hi:.6f}]"
        def fm(metric, n=6):
            v = fnum(r.get(f"mean_delta_{metric}"))
            return "" if v is None else f"{v:.{n}f}"
        f.write(
            f"| {r['dataset']} | {r['candidate']} | {r['paired_runs']} | {r['verdict']} | "
            f"{r['f1_wins']}/{r['f1_losses']}/{r['f1_ties']} | {fnum(r['f1_sign_test_p']):.4f} | "
            f"{ci('f1')} | {ci('recall')} | {ci('mcc')} | {ci('average_precision')} | {fm('fn',2)} | {fm('cuda_peak_allocated_mb',2)} |\n"
        )
    f.write("\n## Notes for interpretation\n\n")
    f.write("- Treat `average_precision` as a ranking metric across thresholds; thresholded F1/Recall/FP/FN may move differently.\n")
    f.write("- For THEIA, one false negative can change Recall substantially when the test positive count is small.\n")
    f.write("- Use this report together with per-seed `metrics_test_compact.json`, `history.png`, and `scores_test.png` under `analysis_bundle/`.\n")

bundle = root / "analysis_bundle"
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)
for name in [
    "ENVIRONMENT.txt",
    "E1_RGD_BIGRU_TBB_RR_RIGOROUS_EXPERIMENT_PLAN.md",
    "E1_RGD_BIGRU_TBB_RR_RIGOROUS_SUMMARY.csv",
    "E1_RGD_BIGRU_TBB_RR_RIGOROUS_AGG.csv",
    "E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_DELTAS.csv",
    "E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_AGG.csv",
    "E1_RGD_BIGRU_TBB_RR_RIGOROUS_REPORT.md",
]:
    src = root / name
    if src.exists():
        shutil.copy2(src, bundle / name)

# Copy nested child bundles, but exclude caches/checkpoints/raw data by relying on child collection.
for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in {"_cache", "analysis_bundle"}]):
    for label_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        src = label_dir / "analysis_bundle"
        if not src.exists():
            continue
        dst = bundle / dataset_dir.name / label_dir.name
        shutil.copytree(src, dst)

# Extra flat index of the most important files, to make uploads easier to inspect.
flat = bundle / "key_files_flat"
flat.mkdir(exist_ok=True)
for r in rows:
    run = Path(r["run_dir"])
    prefix = f"{r['dataset']}__{r['label']}__seed_{r['seed']}__{r['experiment']}"
    for rel in [
        "console.log", "metrics_test_compact.json", "metrics_test.json", "train_summary.json", "run_analysis.json", "config.resolved.yaml",
        "plots/history.png", "plots/scores_test.png", "attention_summary.json", "gate_summary.json", "preprocess_metadata.json",
    ]:
        src = run / rel
        if src.exists() and src.is_file() and src.stat().st_size <= 20 * 1024 * 1024:
            dst = flat / (prefix + "__" + rel.replace("/", "__"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

(bundle / "MANIFEST.txt").write_text(
    "send_this_directory_for_analysis=true\n"
    f"root={root}\n"
    "contains=environment,plan,summary,aggregate,paired_deltas,paired_agg,report,per-dataset-label seed bundles,key_files_flat\n"
    "excluded=graph_cache,checkpoints,processed,raw,model weights,large arrays\n",
    encoding="utf-8",
)
print(json.dumps({
    "summary_csv": str(summary_csv),
    "aggregate_csv": str(agg_csv),
    "paired_csv": str(paired_csv),
    "paired_agg_csv": str(paired_agg_csv),
    "report": str(report),
    "rows": len(rows),
    "analysis_bundle": str(bundle),
}, ensure_ascii=False, indent=2))
PY

echo "[done] base_out=$ABS_BASE_OUT"
echo "[done] report=$ABS_BASE_OUT/E1_RGD_BIGRU_TBB_RR_RIGOROUS_REPORT.md"
echo "[done] analysis_bundle=$ABS_BASE_OUT/analysis_bundle"
