#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_DIR="${RAW_DIR:-data/raw/darpa_tc/theia/e3/cdm}"
LABEL_DIR="${LABEL_DIR:-data/raw/darpa_tc/theia/e3/labels}"
RAW_GLOB="${RAW_GLOB:-ta1-theia-e3-official*.json*}"
MANIFEST="data/raw/darpa_tc/theia/e3/MANIFEST.theia.generated.txt"

mkdir -p "$RAW_DIR" "$LABEL_DIR" "$(dirname "$MANIFEST")"

{
  echo "# THEIA-E3 data layout manifest"
  echo "created=$(date -Is)"
  echo "root=$ROOT_DIR"
  echo "raw_dir=$RAW_DIR"
  echo "label_dir=$LABEL_DIR"
  echo "raw_glob=$RAW_GLOB"
  echo
  echo "## CDM files"
  find "$RAW_DIR" -maxdepth 1 -type f -name "$RAW_GLOB" -printf '%f\t%s bytes\n' | sort || true
  echo
  echo "## Label files"
  find "$LABEL_DIR" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort || true
} > "$MANIFEST"

mapfile -t cdm_files < <(find "$RAW_DIR" -maxdepth 1 -type f -name "$RAW_GLOB" | sort)
if (( ${#cdm_files[@]} == 0 )); then
  echo "[ERROR] No THEIA CDM files found under $RAW_DIR matching $RAW_GLOB" >&2
  echo "Expected examples:" >&2
  echo "  ta1-theia-e3-official.json" >&2
  echo "  ta1-theia-e3-official.json.1" >&2
  echo "  ta1-theia-e3-official.json.2" >&2
  echo "  ta1-theia-e3-official-1.json" >&2
  echo "  ta1-theia-e3-official-1.json.1" >&2
  echo "  ta1-theia-e3-official-2.json" >&2
  exit 2
fi

label_ok=0
for f in \
  "$LABEL_DIR/theia.json" \
  "$LABEL_DIR/theia.txt" \
  "$LABEL_DIR/malicious_uuids.txt" \
  "$LABEL_DIR/malicious_paths.txt" \
  "$LABEL_DIR/malicious_event_types.txt" \
  "$LABEL_DIR/malicious_time_ranges.csv" \
  "$LABEL_DIR/malicious_events.csv"; do
  [[ -s "$f" ]] && label_ok=1 && echo "[OK] Found THEIA label source: $f"
done
if [[ -d "$LABEL_DIR/_raw" ]] && find "$LABEL_DIR/_raw" -type f \( -name '*.json*' -o -name '*.txt' -o -name '*.csv' \) | grep -q .; then
  label_ok=1
  echo "[OK] Found THEIA raw labels under $LABEL_DIR/_raw"
fi
if (( label_ok == 0 )); then
  echo "[ERROR] No effective THEIA label source found in $LABEL_DIR" >&2
  echo "Put one of: theia.json, theia.txt, malicious_uuids.txt, malicious_paths.txt, malicious_event_types.txt, malicious_time_ranges.csv, malicious_events.csv" >&2
  exit 3
fi

echo "[OK] Found ${#cdm_files[@]} THEIA CDM file(s)."
echo "[OK] Manifest written to $MANIFEST"
echo "[OK] THEIA layout appears usable."
