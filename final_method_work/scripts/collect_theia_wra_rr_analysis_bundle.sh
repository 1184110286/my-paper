#!/usr/bin/env bash
set -euo pipefail

# Collect lightweight, analysis-critical artifacts for the THEIA WRA-RR one-key run.
# v4.8.0 behavior: all useful files are flattened into ONE upload folder:
#   <run_root>/analysis_bundle/collected/
# File names are prefixed with source context (root/precheck/experiment/mode/seed)
# so there are no scattered mode/seed subdirectories to package manually.
# Heavyweight graph caches, checkpoints, raw data and generated arrays are excluded.
#
# Usage:
#   bash scripts/collect_wra_rr_analysis_bundle.sh <run_root> [bundle_dir]

RUN_ROOT="${1:?Usage: $0 <run_root> [bundle_dir]}"
BUNDLE_DIR="${2:-$RUN_ROOT/analysis_bundle}"
COLLECTED_DIR="$BUNDLE_DIR/collected"

rm -rf "$COLLECTED_DIR"
mkdir -p "$COLLECTED_DIR"

sanitize_name() {
  local s="$1"
  s="${s//\//__}"
  s="${s// /_}"
  s="${s//:/_}"
  echo "$s"
}

copy_as() {
  local src="$1"
  local name="$2"
  [[ -f "$src" ]] || return 0
  # Refuse large files and known heavyweight outputs.
  case "$src" in
    *graph_cache*|*checkpoints*|*processed*|*raw*|*.pt|*.pth|*.pkl|*.npy|*.npz) return 0 ;;
  esac
  local size
  size=$(wc -c < "$src" 2>/dev/null || echo 0)
  [[ "$size" -le 20971520 ]] || return 0
  cp -a "$src" "$COLLECTED_DIR/$(sanitize_name "$name")"
}

copy_flat_dir() {
  local src_dir="$1"
  local prefix="$2"
  local maxdepth="${3:-1}"
  [[ -d "$src_dir" ]] || return 0
  find "$src_dir" -maxdepth "$maxdepth" -type f \
    \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.log' -o -name '*.txt' -o -name '*.png' \) \
    -size -20M -print0 | while IFS= read -r -d '' p; do
      case "$p" in
        *graph_cache*|*checkpoints*|*processed*|*raw*|*.pt|*.pth|*.pkl|*.npy|*.npz) continue ;;
      esac
      local rel="${p#$src_dir/}"
      copy_as "$p" "${prefix}__${rel}"
    done
}

# Root-level plan and final experiment reports.
copy_flat_dir "$RUN_ROOT" "root" 1
copy_flat_dir "$RUN_ROOT/experiment" "experiment" 1
copy_flat_dir "$RUN_ROOT/precheck" "precheck" 1

# Lightweight preprocess metadata lives under cache/analysis/preprocess, but only
# these small files are useful; full cache and graph_cache are not copied.
copy_as "$RUN_ROOT/precheck/cache/analysis/preprocess/metadata.json" "precheck__preprocess_metadata.json"
copy_as "$RUN_ROOT/precheck/cache/analysis/preprocess/config.preprocess.yaml" "precheck__config.preprocess.yaml"
copy_as "$RUN_ROOT/precheck/cache/analysis/preprocess/console.log" "precheck__preprocess_console.log"

# Determine experiment labels dynamically.  If run_matrix.tsv is unavailable
# because a run was stopped after child experiments but before outer aggregation,
# infer labels from experiment/* directories.
labels=()
if [[ -f "$RUN_ROOT/experiment/run_matrix.tsv" ]]; then
  while IFS=$'\t' read -r label _rest; do
    [[ "$label" == "label" || -z "$label" ]] && continue
    labels+=("$label")
  done < "$RUN_ROOT/experiment/run_matrix.tsv"
fi
if [[ ${#labels[@]} -eq 0 && -d "$RUN_ROOT/experiment" ]]; then
  while IFS= read -r d; do
    labels+=("$(basename "$d")")
  done < <(find "$RUN_ROOT/experiment" -mindepth 1 -maxdepth 1 -type d | sort)
fi
# Stable fallback for minimal or partial runs.
if [[ ${#labels[@]} -eq 0 ]]; then
  labels=(off prefix_tree winnowing_anchor)
fi

for mode in "${labels[@]}"; do
  mode_root="$RUN_ROOT/experiment/$mode"
  [[ -d "$mode_root" ]] || continue

  # Files directly under the mode root and directly under the mode-level
  # analysis_bundle root, such as next_summary.csv and decision reports.
  copy_flat_dir "$mode_root" "mode-${mode}" 1
  copy_flat_dir "$mode_root/analysis_bundle" "mode-${mode}__analysis_bundle" 1

  # Optional mode-level preprocess metadata.
  copy_as "$mode_root/cache/analysis/preprocess/metadata.json" "mode-${mode}__preprocess_metadata.json"
  copy_as "$mode_root/cache/analysis/preprocess/config.preprocess.yaml" "mode-${mode}__config.preprocess.yaml"
  copy_as "$mode_root/cache/analysis/preprocess/console.log" "mode-${mode}__preprocess_console.log"

  # Current outer WRA script launches each mode as a child v3 run, whose useful
  # files live in mode_root/analysis_bundle/seed_*/experiments/...
  for seed_dir in "$mode_root"/seed_* "$mode_root/analysis_bundle"/seed_*; do
    [[ -d "$seed_dir" ]] || continue
    seed_name="$(basename "$seed_dir")"
    if [[ -d "$seed_dir/analysis_bundle" ]]; then
      copy_flat_dir "$seed_dir/analysis_bundle" "mode-${mode}__${seed_name}" 8
    else
      copy_flat_dir "$seed_dir" "mode-${mode}__${seed_name}" 8
    fi
  done

done

# Fallback aggregation for partial/stopped runs where child seed results exist
# but the outer script did not reach its final summary-generation block.  This
# writes aggregate files directly into collected/ so the upload folder remains
# self-contained and useful for analysis.
python3 - "$RUN_ROOT" "$COLLECTED_DIR" <<'PY_FALLBACK' || true
import csv
import json
import math
import statistics as stats
from pathlib import Path
import glob
import os
import re

run_root = Path(os.environ.get('RUN_ROOT_OVERRIDE', '') or __import__('sys').argv[1])
out = Path(__import__('sys').argv[2])

modes = []
exp = run_root / 'experiment'
if exp.exists():
    modes = sorted([p.name for p in exp.iterdir() if p.is_dir()])
preferred = ['off', 'prefix_tree', 'winnowing_anchor']
modes = [m for m in preferred if m in modes] + [m for m in modes if m not in preferred]
rows = []
for mode in modes:
    pattern = exp / mode / 'analysis_bundle' / 'seed_*' / 'experiments' / 'E1_eha_only' / 'metrics_test_compact.json'
    for f in sorted(glob.glob(str(pattern))):
        m = re.search(r'seed_(\d+)', f)
        seed = int(m.group(1)) if m else -1
        with open(f, encoding='utf-8') as fh:
            data = json.load(fh)
        met = data.get('metrics', data)
        row = {'mode': mode, 'seed': seed}
        for k in ['f1','precision','recall','mcc','average_precision','roc_auc','tp','fp','tn','fn','threshold','best_f1','best_f1_threshold','num_samples']:
            row[k] = met.get(k)
        rows.append(row)

if not rows:
    raise SystemExit(0)

def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

fields = ['mode','seed','f1','precision','recall','mcc','average_precision','roc_auc','tp','fp','tn','fn','threshold','best_f1','best_f1_threshold','num_samples']
write_csv(out / 'experiment__summary_theia_wra_rr_fallback.csv', sorted(rows, key=lambda r:(r['mode'],r['seed'])), fields)

agg_rows = []
for mode in modes:
    rs = [r for r in rows if r['mode'] == mode]
    if not rs: continue
    a = {'mode': mode, 'seeds': len(rs)}
    for k in ['f1','precision','recall','mcc','average_precision','roc_auc']:
        vals = [float(r[k]) for r in rs if r.get(k) is not None]
        a[k + '_mean'] = sum(vals)/len(vals) if vals else ''
        a[k + '_std'] = stats.stdev(vals) if len(vals)>1 else 0.0 if vals else ''
    for k in ['tp','fp','tn','fn']:
        vals = [int(r[k]) for r in rs if r.get(k) is not None]
        a[k + '_sum'] = sum(vals) if vals else ''
    agg_rows.append(a)
agg_fields = ['mode','seeds'] + [x for k in ['f1','precision','recall','mcc','average_precision','roc_auc'] for x in [k+'_mean', k+'_std']] + [k+'_sum' for k in ['tp','fp','tn','fn']]
write_csv(out / 'experiment__summary_theia_wra_rr_agg_fallback.csv', agg_rows, agg_fields)

# Paired deltas vs prefix_tree
pref = {r['seed']: r for r in rows if r['mode'] == 'prefix_tree'}
pair_rows = []
for mode in modes:
    if mode == 'prefix_tree': continue
    for r in rows:
        if r['mode'] != mode or r['seed'] not in pref: continue
        p = pref[r['seed']]
        pr = {'candidate': mode, 'seed': r['seed']}
        for k in ['f1','precision','recall','mcc','average_precision','roc_auc']:
            pr['delta_' + k] = float(r[k]) - float(p[k])
        for k in ['tp','fp','tn','fn']:
            pr['delta_' + k] = int(r[k]) - int(p[k])
        pair_rows.append(pr)
pair_fields = ['candidate','seed'] + ['delta_'+k for k in ['f1','precision','recall','mcc','average_precision','roc_auc','tp','fp','tn','fn']]
write_csv(out / 'experiment__paired_vs_prefix_fallback.csv', pair_rows, pair_fields)

pair_agg = []
for mode in sorted({r['candidate'] for r in pair_rows}):
    rs = [r for r in pair_rows if r['candidate'] == mode]
    a = {'candidate': mode, 'paired_seeds': len(rs)}
    f1vals = [float(r['delta_f1']) for r in rs]
    a['f1_win'] = sum(v > 1e-12 for v in f1vals)
    a['f1_loss'] = sum(v < -1e-12 for v in f1vals)
    a['f1_tie'] = sum(abs(v) <= 1e-12 for v in f1vals)
    for k in ['f1','precision','recall','mcc','average_precision','roc_auc','tp','fp','tn','fn']:
        vals = [float(r['delta_'+k]) for r in rs]
        a['delta_'+k+'_mean'] = sum(vals)/len(vals) if vals else ''
    # conservative verdict
    good = (a.get('delta_f1_mean', 0) >= 0.003 and a.get('delta_mcc_mean', 0) >= 0.003 and a.get('delta_recall_mean', 0) >= -0.001 and a.get('f1_win',0) >= math.ceil(0.6*len(rs)))
    risky = a.get('delta_recall_mean',0) < -0.001 or a.get('delta_f1_mean',0) < -0.003
    a['verdict'] = 'positive_sensitivity_signal' if good else ('negative_or_recall_risk' if risky else 'tie_or_inconclusive')
    pair_agg.append(a)
pair_agg_fields = ['candidate','paired_seeds','f1_win','f1_loss','f1_tie','verdict'] + ['delta_'+k+'_mean' for k in ['f1','precision','recall','mcc','average_precision','roc_auc','tp','fp','tn','fn']]
write_csv(out / 'experiment__paired_vs_prefix_agg_fallback.csv', pair_agg, pair_agg_fields)

report = out / 'experiment__THEIA_WRA_RR_VERDICT_REPORT_fallback.md'
with open(report, 'w', encoding='utf-8') as fh:
    fh.write('# THEIA WRA-RR fallback verdict report\n\n')
    fh.write('This report was generated by collect_wra_rr_analysis_bundle.sh because the outer run did not contain final aggregate files.\n\n')
    fh.write('## Aggregate metrics\n\n')
    for a in agg_rows:
        fh.write(f"- {a['mode']}: F1={float(a['f1_mean']):.6f}, MCC={float(a['mcc_mean']):.6f}, Recall={float(a['recall_mean']):.6f}, TP/FP/TN/FN={a['tp_sum']}/{a['fp_sum']}/{a['tn_sum']}/{a['fn_sum']}\n")
    fh.write('\n## Paired deltas vs prefix_tree\n\n')
    for a in pair_agg:
        fh.write(f"- {a['candidate']}: ΔF1={float(a['delta_f1_mean']):+.6f}, ΔMCC={float(a['delta_mcc_mean']):+.6f}, ΔRecall={float(a['delta_recall_mean']):+.6f}, win/loss/tie={a['f1_win']}/{a['f1_loss']}/{a['f1_tie']}, verdict={a['verdict']}\n")
PY_FALLBACK

cat > "$COLLECTED_DIR/MANIFEST.txt" <<EOF_MANIFEST
analysis_bundle_created=$(date -Is)
script=scripts/run_theia_wra_rr_verdict.sh
run_root=$RUN_ROOT
bundle_dir=$BUNDLE_DIR
single_upload_folder=$COLLECTED_DIR
send_this_directory_for_analysis=true
contains=plan, strict precheck report, run matrix, summaries, verdict report, paired deltas, compact metrics, logs, configs, essential plots, top-alert CSVs, lightweight preprocess metadata
excluded=graph cache, checkpoints, raw data, processed cache, full generated embeddings, large arrays, model weights
naming=file names are prefixed by source context: root/precheck/experiment/mode/seed
EOF_MANIFEST

{
  echo "# Single-folder THEIA WRA-RR analysis bundle"
  find "$COLLECTED_DIR" -maxdepth 1 -type f | sed "s#^$COLLECTED_DIR/##" | sort
} > "$COLLECTED_DIR/FILELIST.txt" 2>/dev/null || true

# Compatibility breadcrumb at analysis_bundle root; the actual payload is one folder.
cat > "$BUNDLE_DIR/README_UPLOAD_THIS_FOLDER.txt" <<EOF_README
Upload this single folder for analysis:
$COLLECTED_DIR

All useful metrics/logs/configs/plots were flattened into collected/.
Heavy cache/checkpoint/raw files were excluded.
EOF_README

{
  echo "# Bundle root"
  echo "Payload folder: collected/"
  find "$COLLECTED_DIR" -maxdepth 1 -type f | sed "s#^$BUNDLE_DIR/##" | sort
} > "$BUNDLE_DIR/FILELIST.txt" 2>/dev/null || true

echo "[bundle] THEIA WRA-RR single-folder analysis bundle collected in $COLLECTED_DIR"
