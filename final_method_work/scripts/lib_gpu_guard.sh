# shellcheck shell=bash
# GPU reservation guard for cluster/lab environments using NVIDIA exclusive-process mode.
# Usage: source this file after ROOT_DIR and ANALYSIS_ROOT are known.

GPU_RESERVE_ENABLE="${GPU_RESERVE_ENABLE:-1}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-512}"
GPU_RESERVE_STRICT="${GPU_RESERVE_STRICT:-1}"
GPU_RESERVE_SLEEP="${GPU_RESERVE_SLEEP:-30}"
GPU_RESERVE_LOG_DIR="${GPU_RESERVE_LOG_DIR:-${ANALYSIS_ROOT:-runs}/gpu_guard}"
GPU_RESERVE_PID=""
GPU_RESERVE_ACTIVE="0"

_gpu_guard_device() {
  local d="${1:-${DEVICE:-0}}"
  if [[ "$d" == cuda:* ]]; then
    echo "${d#cuda:}"
  else
    echo "$d"
  fi
}

start_gpu_reserve() {
  if [[ "${GPU_RESERVE_ENABLE}" != "1" ]]; then
    return 0
  fi
  if [[ "${DEVICE:-}" == "cpu" || "${DEVICE:-}" == "CPU" ]]; then
    echo "[gpu-guard] DEVICE=cpu; skip reservation"
    return 0
  fi
  if [[ "${GPU_RESERVE_ACTIVE}" == "1" ]] && [[ -n "${GPU_RESERVE_PID}" ]] && kill -0 "${GPU_RESERVE_PID}" 2>/dev/null; then
    return 0
  fi
  mkdir -p "$GPU_RESERVE_LOG_DIR"
  local dev heartbeat log
  dev="$(_gpu_guard_device "${DEVICE:-0}")"
  heartbeat="$GPU_RESERVE_LOG_DIR/heartbeat_device_${dev}.json"
  local ready
  ready="$GPU_RESERVE_LOG_DIR/ready_device_${dev}.json"
  log="$GPU_RESERVE_LOG_DIR/gpu_guard_device_${dev}.log"
  echo "[gpu-guard] reserving ~${GPU_RESERVE_MB}MiB on cuda:${dev}; log=${log}"
  python -m malsnif.utils.gpu_guard \
    --device "$dev" \
    --memory-mb "$GPU_RESERVE_MB" \
    --ready-file "$ready" \
    --heartbeat-file "$heartbeat" \
    --heartbeat-seconds "$GPU_RESERVE_SLEEP" \
    >> "$log" 2>&1 &
  GPU_RESERVE_PID=$!
  GPU_RESERVE_ACTIVE="1"
  # Give CUDA a moment to create the context. If it fails in strict mode, stop early.
  sleep 3
  if ! kill -0 "$GPU_RESERVE_PID" 2>/dev/null; then
    GPU_RESERVE_ACTIVE="0"
    GPU_RESERVE_PID=""
    echo "[gpu-guard] reservation process exited early; see ${log}" >&2
    if [[ "${GPU_RESERVE_STRICT}" == "1" ]]; then
      tail -50 "$log" >&2 || true
      exit 2
    fi
  fi
}

stop_gpu_reserve() {
  if [[ -n "${GPU_RESERVE_PID}" ]] && kill -0 "${GPU_RESERVE_PID}" 2>/dev/null; then
    echo "[gpu-guard] releasing reservation pid=${GPU_RESERVE_PID}"
    kill "${GPU_RESERVE_PID}" 2>/dev/null || true
    wait "${GPU_RESERVE_PID}" 2>/dev/null || true
    # Let the driver release the exclusive context before the real job starts.
    sleep 2
  fi
  GPU_RESERVE_PID=""
  GPU_RESERVE_ACTIVE="0"
}

with_gpu_job() {
  # Under EXCLUSIVE_PROCESS, the guard and train/eval cannot hold CUDA contexts
  # concurrently. Release guard, run the real job, then reacquire for CPU gaps.
  stop_gpu_reserve
  "$@"
  local status=$?
  start_gpu_reserve
  return "$status"
}

install_gpu_reserve_trap() {
  trap 'stop_gpu_reserve' EXIT INT TERM
}
