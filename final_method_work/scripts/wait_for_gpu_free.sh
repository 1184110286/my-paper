#!/usr/bin/env bash
set -euo pipefail

# Wait until the selected GPU has enough free memory before launching a CUDA job.
# This is a defensive helper for shared/exclusive-process lab machines.  It does
# not kill other processes.  It only waits and prints the current nvidia-smi state.
#
# Usage:
#   scripts/wait_for_gpu_free.sh 0 4400 30 7200
# Args:
#   1: GPU id as seen by the current process, e.g. 0 or 1
#   2: minimum free MiB required
#   3: polling interval seconds
#   4: timeout seconds, 0 means wait forever

device="${1:-0}"
min_free_mb="${2:-4400}"
interval="${3:-30}"
timeout="${4:-0}"

if [[ "$device" == "cpu" || "$device" == "CPU" || "$device" == "-1" ]]; then
  exit 0
fi
if [[ "$device" == cuda:* ]]; then
  device="${device#cuda:}"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu-wait] nvidia-smi not found; skip memory wait" >&2
  exit 0
fi

start_ts="$(date +%s)"
while true; do
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$device" 2>/dev/null | head -1 | tr -dc '0-9')"
  if [[ -z "$free_mb" ]]; then
    echo "[gpu-wait] unable to query GPU $device; skip" >&2
    exit 0
  fi
  if (( free_mb >= min_free_mb )); then
    echo "[gpu-wait] cuda:$device free=${free_mb}MiB >= ${min_free_mb}MiB; continue"
    exit 0
  fi
  echo "[gpu-wait] cuda:$device free=${free_mb}MiB < ${min_free_mb}MiB; waiting ${interval}s"
  nvidia-smi -i "$device" || true
  if (( timeout > 0 )); then
    now="$(date +%s)"
    if (( now - start_ts >= timeout )); then
      echo "[gpu-wait] timeout after ${timeout}s waiting for cuda:$device free memory >= ${min_free_mb}MiB" >&2
      exit 88
    fi
  fi
  sleep "$interval"
done
