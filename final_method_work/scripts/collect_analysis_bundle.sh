#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="${1:?Usage: $0 <seed_run_root> [bundle_dir]}"
BUNDLE_DIR="${2:-$RUN_ROOT/analysis_bundle}"
mkdir -p "$BUNDLE_DIR"
copy_small() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || return 0
  case "$src" in *graph_cache*|*checkpoints*|*processed*|*raw*|*.pt|*.pth|*.pkl|*.npy|*.npz) return 0 ;; esac
  local size
  size=$(wc -c < "$src" 2>/dev/null || echo 0)
  [[ "$size" -le 20971520 ]] || return 0
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}
copy_tree_small() {
  local src_dir="$1" dst_dir="$2" maxdepth="${3:-4}"
  [[ -d "$src_dir" ]] || return 0
  find "$src_dir" -maxdepth "$maxdepth" -type f \
    \( -name '*.csv' -o -name '*.json' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name '*.log' -o -name '*.txt' -o -name '*.png' \) \
    -size -20M -print0 | while IFS= read -r -d '' p; do
      case "$p" in *graph_cache*|*checkpoints*|*processed*|*raw*|*.pt|*.pth|*.pkl|*.npy|*.npz) continue ;; esac
      local rel="${p#$src_dir/}"
      mkdir -p "$dst_dir/$(dirname "$rel")"
      cp -a "$p" "$dst_dir/$rel"
    done
}
copy_tree_small "$RUN_ROOT/analysis" "$BUNDLE_DIR/analysis" 5
copy_tree_small "$RUN_ROOT/experiments" "$BUNDLE_DIR/experiments" 6
copy_small "$RUN_ROOT/config.resolved.yaml" "$BUNDLE_DIR/config.resolved.yaml"
copy_small "$RUN_ROOT/console.log" "$BUNDLE_DIR/console.log"
copy_small "$RUN_ROOT/train_summary.json" "$BUNDLE_DIR/train_summary.json"
copy_small "$RUN_ROOT/run_analysis.json" "$BUNDLE_DIR/run_analysis.json"
cat > "$BUNDLE_DIR/MANIFEST.txt" <<EOF
created=$(date -Is)
run_root=$RUN_ROOT
bundle_dir=$BUNDLE_DIR
excluded=graph_cache,checkpoints,processed,raw,model weights,large arrays
EOF
echo "[bundle] analysis bundle collected in $BUNDLE_DIR"
