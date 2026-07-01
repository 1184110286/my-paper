#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_DIR="${RAW_DIR:-data/raw/darpa_tc/cadets/e3/cdm}"
LABEL_DIR="${LABEL_DIR:-data/raw/darpa_tc/cadets/e3/labels}"
RAW_GLOB="${RAW_GLOB:-ta1-cadets-e3-official*.json*}"
MANIFEST="data/raw/darpa_tc/cadets/e3/MANIFEST.cadets.generated.txt"

mkdir -p "$RAW_DIR" "$LABEL_DIR" "$(dirname "$MANIFEST")"

{
  echo "# CADETS-E3 data layout manifest"
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
  echo "[ERROR] No CADETS CDM files found under $RAW_DIR matching $RAW_GLOB" >&2
  echo "Expected examples:" >&2
  echo "  ta1-cadets-e3-official.json" >&2
  echo "  ta1-cadets-e3-official.json.1" >&2
  echo "  ta1-cadets-e3-official.json.2" >&2
  echo "  ta1-cadets-e3-official-1.json" >&2
  echo "  ta1-cadets-e3-official-1.json.1" >&2
  echo "  ta1-cadets-e3-official-1.json.2" >&2
  echo "  ta1-cadets-e3-official-1.json.3" >&2
  echo "  ta1-cadets-e3-official-1.json.4" >&2
  echo "  ta1-cadets-e3-official-2.json" >&2
  echo "  ta1-cadets-e3-official-2.json.1" >&2
  exit 2
fi

# At least one usable label source is required for supervised metrics.
label_ok=0
for f in \
  "$LABEL_DIR/cadets.json" \
  "$LABEL_DIR/cadets.txt" \
  "$LABEL_DIR/theia.json" \
  "$LABEL_DIR/theia.txt" \
  "$LABEL_DIR/malicious_uuids.txt" \
  "$LABEL_DIR/malicious_paths.txt" \
  "$LABEL_DIR/malicious_event_types.txt" \
  "$LABEL_DIR/malicious_time_ranges.csv" \
  "$LABEL_DIR/malicious_events.csv"; do
  [[ -s "$f" ]] && label_ok=1 && echo "[OK] Found CADETS label source: $f"
done
if [[ -d "$LABEL_DIR/_raw" ]] && find "$LABEL_DIR/_raw" -type f -name '*.json*' | grep -q .; then
  label_ok=1
  echo "[OK] Found CADETS raw JSON labels under $LABEL_DIR/_raw"
fi
if (( label_ok == 0 )); then
  echo "[ERROR] No effective CADETS label source found in $LABEL_DIR" >&2
  echo "Put one of: cadets.json, cadets.txt, theia.json, theia.txt, malicious_uuids.txt, malicious_paths.txt, malicious_event_types.txt, malicious_time_ranges.csv, malicious_events.csv" >&2
  exit 3
fi

echo "[OK] Found ${#cdm_files[@]} CADETS CDM file(s)."
echo "[OK] Manifest written to $MANIFEST"
echo "[OK] CADETS layout appears usable."
