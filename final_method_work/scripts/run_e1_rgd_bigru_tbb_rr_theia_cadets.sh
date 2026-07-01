#!/usr/bin/env bash
set -euo pipefail

# One-key E1-RGD-BiGRU-MCBG validation on CADETS and THEIA with TBB-RR.
#
# What it compares (paired seeds, same TBB-RR graph cache):
#   1) E1_eha_only_mcbg      : existing E1_eha_only semantic branch (MCBG = CNN+BiGRU+MHA)
#   2) E1_eha_only_rgd_bigru : new RGD-BiGRU-MCBG branch (residual gated dilated CNN + BiGRU)
#
# Default:
#   bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
#
# DEVICE=1 example:
#   DEVICE=1 EVAL_DEVICE=1 SEEDS="42 43 44" EPOCHS=5 bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
#
# Quick smoke example:
#   DEVICE=1 EVAL_DEVICE=1 CADETS_EA_PRESET=smoke SEEDS="42" EPOCHS=1 bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh

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
BASE_OUT="${BASE_OUT:-runs/e1_rgd_bigru_tbb_rr_theia_cadets_${TS}}"
RUN_DATASETS="${RUN_DATASETS:-cadets theia}"
RUN_ENCODERS="${RUN_ENCODERS:-mcbg rgd_bigru}"
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
RGD_KERNEL_SIZE="${RGD_KERNEL_SIZE:-3}"
RGD_DILATIONS="${RGD_DILATIONS:-1,2}"
RGD_DROPOUT="${RGD_DROPOUT:-0.2}"
RGD_RESIDUAL_SCALE_INIT="${RGD_RESIDUAL_SCALE_INIT:-0.1}"
RGD_DEPTHWISE_SEPARABLE="${RGD_DEPTHWISE_SEPARABLE:-1}"
RGD_USE_EVENT_WEIGHT_POOLING="${RGD_USE_EVENT_WEIGHT_POOLING:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

mkdir -p "$BASE_OUT"
ABS_BASE_OUT="$(cd "$BASE_OUT" && pwd)"

cat > "$BASE_OUT/E1_RGD_BIGRU_TBB_RR_EXPERIMENT_PLAN.md" <<EOF
# E1-RGD-BiGRU-MCBG TBB-RR experiment plan

Goal: validate the new RGD-BiGRU-MCBG semantic encoder against the existing
E1_eha_only MCBG encoder on the same TBB-RR-compressed graph cache.

Fixed pipeline:
- model_variant: ea_st_hgan_mcbg
- adaptive mechanism: EHA only
- redundancy_mode: target_boundary (TBB-RR)
- graph branch: ST-HGAN + EHA unchanged
- comparison: semantic_encoder=mcbg vs semantic_encoder=rgd_bigru_mcbg

Controls:
- datasets: $RUN_DATASETS
- encoders: $RUN_ENCODERS
- seeds: $SEEDS
- device: $DEVICE, eval_device: $EVAL_DEVICE
- preset: $CADETS_EA_PRESET
- epochs: $EPOCHS
- dim: $DIM
- max_events_per_node: $MAX_EVENTS_PER_NODE
- max_events_per_edge: $MAX_EVENTS_PER_EDGE
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
Interpret candidate quality by paired seed deltas within each dataset.
EOF

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
    export RGD_KERNEL_SIZE="$RGD_KERNEL_SIZE"
    export RGD_DILATIONS="$RGD_DILATIONS"
    export RGD_DROPOUT="$RGD_DROPOUT"
    export RGD_RESIDUAL_SCALE_INIT="$RGD_RESIDUAL_SCALE_INIT"
    export RGD_DEPTHWISE_SEPARABLE="$RGD_DEPTHWISE_SEPARABLE"
    export RGD_USE_EVENT_WEIGHT_POOLING="$RGD_USE_EVENT_WEIGHT_POOLING"
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
  for enc in $RUN_ENCODERS; do
    label="$(encoder_to_label "$enc")"
    semantic_encoder="$(encoder_to_config_name "$enc")"
    run_one_dataset_encoder "$dataset" "$label" "$semantic_encoder"
  done
done

python - "$ABS_BASE_OUT" <<'PY'
import csv, json, math, re, shutil, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
seen = set()
# Avoid scanning analysis_bundle copies. Use only canonical dataset/label/seed_*/analysis/* paths.
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
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", payload)
            summary_path = run_dir / "train_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            seed = summary.get("seed")
            if seed is None:
                m = re.search(r"seed_(\d+)", str(metrics_path))
                seed = int(m.group(1)) if m else None
            rows.append({
                "dataset": dataset,
                "label": label,
                "experiment": run_dir.name,
                "seed": seed,
                "semantic_encoder": summary.get("semantic_encoder") or metrics.get("semantic_encoder"),
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "mcc": metrics.get("mcc"),
                "average_precision": metrics.get("average_precision", metrics.get("auc_pr", metrics.get("pr_auc"))),
                "roc_auc": metrics.get("roc_auc", metrics.get("auc_roc")),
                "threshold": metrics.get("threshold"),
                "tp": metrics.get("tp"), "fp": metrics.get("fp"), "tn": metrics.get("tn"), "fn": metrics.get("fn"),
                "num_samples": metrics.get("num_samples"),
                "prevalence": metrics.get("prevalence"),
                "best_f1": metrics.get("best_f1"),
                "best_f1_threshold": metrics.get("best_f1_threshold"),
                "train_seconds": summary.get("train_total_seconds", summary.get("train_seconds")),
                "cuda_peak_allocated_mb": summary.get("max_cuda_peak_allocated_mb"),
                "run_dir": str(run_dir),
                "config": str(run_dir / "config.resolved.yaml"),
                "metrics_file": str(metrics_path),
            })

out_csv = root / "E1_RGD_BIGRU_TBB_RR_SUMMARY.csv"
fieldnames = [
    "dataset", "label", "experiment", "seed", "semantic_encoder", "f1", "precision", "recall", "specificity", "mcc",
    "average_precision", "roc_auc", "threshold", "tp", "fp", "tn", "fn", "num_samples", "prevalence",
    "best_f1", "best_f1_threshold", "train_seconds", "cuda_peak_allocated_mb", "run_dir", "config", "metrics_file",
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

md = root / "E1_RGD_BIGRU_TBB_RR_REPORT.md"
with md.open("w", encoding="utf-8") as f:
    f.write("# E1-RGD-BiGRU-MCBG TBB-RR summary\n\n")
    f.write(f"- source_csv: `{out_csv}`\n")
    f.write("- baseline: `label=mcbg`\n")
    f.write("- candidate: `label=rgd_bigru`\n\n")
    f.write("| dataset | label | runs | mean_f1 | mean_recall | mean_precision | mean_mcc | mean_ap | mean_roc_auc | mean_fp | mean_fn | mean_train_seconds |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for key in sorted(by):
        rs = by[key]
        def mean(metric):
            vals = [fnum(x.get(metric)) for x in rs]
            vals = [x for x in vals if x is not None]
            return sum(vals) / len(vals) if vals else float('nan')
        f.write(
            f"| {key[0]} | {key[1]} | {len(rs)} | {mean('f1'):.6f} | {mean('recall'):.6f} | {mean('precision'):.6f} | "
            f"{mean('mcc'):.6f} | {mean('average_precision'):.6f} | {mean('roc_auc'):.6f} | {mean('fp'):.2f} | {mean('fn'):.2f} | {mean('train_seconds'):.2f} |\n"
        )
    f.write("\n## Paired deltas by seed: rgd_bigru minus mcbg\n\n")
    f.write("| dataset | seed | delta_f1 | delta_recall | delta_precision | delta_mcc | delta_ap | delta_roc_auc | delta_fp | delta_fn |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    by_seed = {}
    for r in rows:
        by_seed.setdefault((r["dataset"], str(r.get("seed"))), {})[r["label"]] = r
    for (dataset, seed), item in sorted(by_seed.items()):
        if "mcbg" not in item or "rgd_bigru" not in item:
            continue
        def delta(metric):
            a = fnum(item["rgd_bigru"].get(metric)); b = fnum(item["mcbg"].get(metric))
            return float('nan') if a is None or b is None else a - b
        f.write(
            f"| {dataset} | {seed} | {delta('f1'):.6f} | {delta('recall'):.6f} | {delta('precision'):.6f} | {delta('mcc'):.6f} | "
            f"{delta('average_precision'):.6f} | {delta('roc_auc'):.6f} | {delta('fp'):.2f} | {delta('fn'):.2f} |\n"
        )

bundle = root / "analysis_bundle"
bundle.mkdir(exist_ok=True)
for name in ["E1_RGD_BIGRU_TBB_RR_EXPERIMENT_PLAN.md", "E1_RGD_BIGRU_TBB_RR_SUMMARY.csv", "E1_RGD_BIGRU_TBB_RR_REPORT.md"]:
    src = root / name
    if src.exists():
        shutil.copy2(src, bundle / name)
for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in {"_cache", "analysis_bundle"}]):
    for label_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        src = label_dir / "analysis_bundle"
        if not src.exists():
            continue
        dst = bundle / dataset_dir.name / label_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
(bundle / "MANIFEST.txt").write_text(
    "send_this_directory_for_analysis=true\n"
    f"root={root}\n"
    "contains=experiment_plan,summary_csv,summary_report,per-dataset-label seed bundles with logs metrics configs and plots\n"
    "excluded=graph_cache,checkpoints,processed,raw,model weights,large arrays\n",
    encoding="utf-8",
)
print(json.dumps({"summary_csv": str(out_csv), "report": str(md), "rows": len(rows), "analysis_bundle": str(bundle)}, ensure_ascii=False, indent=2))
PY

echo "[done] base_out=$ABS_BASE_OUT"
echo "[done] summary=$ABS_BASE_OUT/E1_RGD_BIGRU_TBB_RR_SUMMARY.csv"
echo "[done] report=$ABS_BASE_OUT/E1_RGD_BIGRU_TBB_RR_REPORT.md"
echo "[done] analysis_bundle=$ABS_BASE_OUT/analysis_bundle"
