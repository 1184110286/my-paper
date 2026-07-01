#!/usr/bin/env bash
set -euo pipefail

# One-key E1-GDTC-MCBG validation on CADETS and THEIA with TBB-RR.
#
# What it compares (paired seeds, same TBB-RR graph cache):
#   1) E1_eha_only_mcbg  : existing E1_eha_only semantic branch (MCBG = CNN+BiGRU+MHA)
#   2) E1_eha_only_gdtc  : new E1-GDTC-MCBG semantic branch (gated dilated temporal conv)
#
# Default:
#   bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
#
# Useful overrides:
#   DEVICE=0 SEEDS="42 43 44" EPOCHS=5 bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
#   RUN_DATASETS="cadets" CADETS_EA_PRESET=smoke bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
#   MAX_EVENTS=800000 WINDOW_EVENTS=100000 GRAPH_LIMIT_TRAIN=6 GRAPH_LIMIT_VAL=2 GRAPH_LIMIT_TEST=2 bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh

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

TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${BASE_OUT:-runs/e1_gdtc_tbb_rr_theia_cadets_${TS}}"
RUN_DATASETS="${RUN_DATASETS:-cadets theia}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}"
CADETS_EA_PRESET="${CADETS_EA_PRESET:-calib5m}"
EPOCHS="${EPOCHS:-5}"
DIM="${DIM:-64}"
MAX_EVENTS_PER_NODE="${MAX_EVENTS_PER_NODE:-64}"
MAX_EVENTS_PER_EDGE="${MAX_EVENTS_PER_EDGE:-4}"
HGAN_TOPK="${HGAN_TOPK:-20}"
TBB_RR_TARGET_COMPRESSION="${TBB_RR_TARGET_COMPRESSION:-0.90}"
MW_PRR_ATTENTION_BETA="${MW_PRR_ATTENTION_BETA:-1.0}"
GDTC_KERNEL_SIZE="${GDTC_KERNEL_SIZE:-3}"
GDTC_DILATIONS="${GDTC_DILATIONS:-1,2,4}"
GDTC_DROPOUT="${GDTC_DROPOUT:-0.2}"
GDTC_USE_EVENT_WEIGHT_POOLING="${GDTC_USE_EVENT_WEIGHT_POOLING:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

mkdir -p "$BASE_OUT"
ABS_BASE_OUT="$(cd "$BASE_OUT" && pwd)"

cat > "$BASE_OUT/E1_GDTC_TBB_RR_EXPERIMENT_PLAN.md" <<EOF
# E1-GDTC-MCBG TBB-RR experiment plan

Goal: validate the new E1-GDTC-MCBG semantic encoder against the existing
E1_eha_only MCBG encoder on the same TBB-RR-compressed graph cache.

Fixed pipeline:
- model_variant: ea_st_hgan_mcbg
- adaptive mechanism: EHA only
- redundancy_mode: target_boundary (TBB-RR)
- graph branch: ST-HGAN + EHA unchanged
- comparison: semantic_encoder=mcbg vs semantic_encoder=gdtc_mcbg

Controls:
- datasets: $RUN_DATASETS
- seeds: $SEEDS
- device: $DEVICE, eval_device: $EVAL_DEVICE
- preset: $CADETS_EA_PRESET
- epochs: $EPOCHS
- dim: $DIM
- max_events_per_node: $MAX_EVENTS_PER_NODE
- max_events_per_edge: $MAX_EVENTS_PER_EDGE
- tbb_rr_target_compression: $TBB_RR_TARGET_COMPRESSION
- event_weight_attention_beta: $MW_PRR_ATTENTION_BETA
- gdtc_kernel_size: $GDTC_KERNEL_SIZE
- gdtc_dilations: $GDTC_DILATIONS
- output: $ABS_BASE_OUT

Primary metrics: F1, Recall, Precision, MCC, Average Precision, ROC-AUC,
training time, and CUDA peak memory.  Interpret candidate quality by paired
seed deltas within each dataset.
EOF

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
      dataset_name="cadets_e3_e1_${label}_tbb_rr"
      ;;
    theia)
      raw_dir="${THEIA_RAW_DIR:-data/raw/darpa_tc/theia/e3/cdm}"
      label_dir="${THEIA_LABEL_DIR:-data/raw/darpa_tc/theia/e3/labels}"
      raw_glob="${THEIA_RAW_GLOB:-ta1-theia-e3-official*.json*}"
      raw_file_sort="${THEIA_RAW_FILE_SORT:-cdm_shards}"
      dataset_name="theia_e3_e1_${label}_tbb_rr"
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
    export MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-val_average_precision}"
    export THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-val_f1_min_recall}"
    export THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.95}"
    export NODE_SCOPE="${NODE_SCOPE:-process}"
    export USE_AMP="${USE_AMP:-1}"
    export AMP_DTYPE="${AMP_DTYPE:-float16}"
    export CACHE_GRAPHS_IN_MEMORY="${CACHE_GRAPHS_IN_MEMORY:-0}"
    export PLOT_MODE="${PLOT_MODE:-essential}"
    export TOP_ALERTS_PER_GRAPH="${TOP_ALERTS_PER_GRAPH:-20}"
    export PATIENCE="${PATIENCE:-5}"
    export GRAPH_SIMPLIFY_MODE="${GRAPH_SIMPLIFY_MODE:-leaf}"
    export GRAPH_SIMPLIFY_RISK_THRESHOLD="${GRAPH_SIMPLIFY_RISK_THRESHOLD:-0.62}"
    export GRAPH_SIMPLIFY_TOPK_PER_PROCESS="${GRAPH_SIMPLIFY_TOPK_PER_PROCESS:-0}"
    export GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS="${GRAPH_SIMPLIFY_TEMPORAL_WINDOW_NS:-1000000000}"
    export GRAPH_SIMPLIFY_REPEAT_NORM="${GRAPH_SIMPLIFY_REPEAT_NORM:-8}"
    export GPU_RESERVE_ENABLE="${GPU_RESERVE_ENABLE:-1}"
    export GPU_RESERVE_MB="${GPU_RESERVE_MB:-512}"
    export GPU_RESERVE_STRICT="${GPU_RESERVE_STRICT:-1}"
    unset REUSE_RUN REUSE_PROCESSED_DIR REUSE_METADATA_DIR
    bash scripts/run_cadets_v3_ea_verdict.sh
  )
}

for dataset in $RUN_DATASETS; do
  run_one_dataset_encoder "$dataset" "mcbg" "mcbg"
  run_one_dataset_encoder "$dataset" "gdtc" "gdtc_mcbg"
done

python - "$ABS_BASE_OUT" <<'PY'
import csv, json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in root.glob("*/**/metrics_test_compact.json"):
    # Expected: root/dataset/label/seed_x/analysis/E1_.../metrics_test_compact.json
    parts = metrics_path.relative_to(root).parts
    if len(parts) < 5:
        continue
    dataset, label = parts[0], parts[1]
    run_dir = metrics_path.parent
    cfg_path = run_dir / "config.resolved.yaml"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "train_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rows.append({
        "dataset": dataset,
        "label": label,
        "experiment": run_dir.name,
        "seed": summary.get("seed"),
        "semantic_encoder": summary.get("semantic_encoder"),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "mcc": metrics.get("mcc"),
        "average_precision": metrics.get("average_precision"),
        "roc_auc": metrics.get("roc_auc"),
        "threshold": metrics.get("threshold"),
        "tp": metrics.get("tp"), "fp": metrics.get("fp"), "tn": metrics.get("tn"), "fn": metrics.get("fn"),
        "train_seconds": summary.get("train_seconds"),
        "cuda_peak_allocated_mb": summary.get("max_cuda_peak_allocated_mb"),
        "run_dir": str(run_dir),
        "config": str(cfg_path),
    })

out_csv = root / "E1_GDTC_TBB_RR_SUMMARY.csv"
fieldnames = [
    "dataset", "label", "experiment", "seed", "semantic_encoder", "f1", "precision", "recall", "mcc",
    "average_precision", "roc_auc", "threshold", "tp", "fp", "tn", "fn", "train_seconds",
    "cuda_peak_allocated_mb", "run_dir", "config",
]
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)

def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

by = {}
for r in rows:
    by.setdefault((r["dataset"], r["label"]), []).append(r)

md = root / "E1_GDTC_TBB_RR_REPORT.md"
with md.open("w", encoding="utf-8") as f:
    f.write("# E1-GDTC-MCBG TBB-RR summary\n\n")
    f.write(f"- source_csv: `{out_csv}`\n")
    f.write("- baseline: `label=mcbg`\n")
    f.write("- candidate: `label=gdtc`\n\n")
    f.write("| dataset | label | runs | mean_f1 | mean_recall | mean_mcc | mean_ap | mean_train_seconds |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for key in sorted(by):
        rs = by[key]
        def mean(metric):
            vals = [fnum(x.get(metric)) for x in rs]
            vals = [x for x in vals if x is not None]
            return sum(vals) / len(vals) if vals else float('nan')
        f.write(f"| {key[0]} | {key[1]} | {len(rs)} | {mean('f1'):.6f} | {mean('recall'):.6f} | {mean('mcc'):.6f} | {mean('average_precision'):.6f} | {mean('train_seconds'):.2f} |\n")
    f.write("\n## Paired deltas by seed\n\n")
    f.write("| dataset | seed | delta_f1_gdtc_minus_mcbg | delta_recall | delta_mcc | delta_ap |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    by_seed = {}
    for r in rows:
        by_seed.setdefault((r["dataset"], str(r.get("seed"))), {})[r["label"]] = r
    for (dataset, seed), item in sorted(by_seed.items()):
        if "mcbg" not in item or "gdtc" not in item:
            continue
        def delta(metric):
            a = fnum(item["gdtc"].get(metric)); b = fnum(item["mcbg"].get(metric))
            return float('nan') if a is None or b is None else a - b
        f.write(f"| {dataset} | {seed} | {delta('f1'):.6f} | {delta('recall'):.6f} | {delta('mcc'):.6f} | {delta('average_precision'):.6f} |\n")

print(json.dumps({"summary_csv": str(out_csv), "report": str(md), "rows": len(rows)}, ensure_ascii=False, indent=2))
PY

echo "[done] base_out=$ABS_BASE_OUT"
echo "[done] summary=$ABS_BASE_OUT/E1_GDTC_TBB_RR_SUMMARY.csv"
echo "[done] report=$ABS_BASE_OUT/E1_GDTC_TBB_RR_REPORT.md"
