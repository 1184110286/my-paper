from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

_STOP = False


def _handle_stop(signum, frame):  # pragma: no cover - signal path
    global _STOP
    _STOP = True


def _parse_device(value: str) -> int:
    v = str(value).strip().lower()
    if v in {"", "none", "cpu", "-1"}:
        raise ValueError("GPU guard requires a CUDA device, not cpu")
    if v.startswith("cuda:"):
        v = v.split(":", 1)[1]
    return int(v)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hold a CUDA context so a shell pipeline keeps ownership of a GPU "
            "during CPU-only phases on clusters configured with EXCLUSIVE_PROCESS mode."
        )
    )
    parser.add_argument("--device", required=True, help="CUDA device index or cuda:<index>")
    parser.add_argument("--memory-mb", type=int, default=32, help="Small tensor allocation used to keep the context visible")
    parser.add_argument("--ready-file", required=True, help="JSON file written after the GPU context has been acquired")
    parser.add_argument("--heartbeat-file", default=None, help="Optional heartbeat JSON file updated while holding the GPU")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[gpu_guard] failed to import torch: {exc!r}", file=sys.stderr, flush=True)
        return 2

    try:
        dev_idx = _parse_device(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        if dev_idx < 0 or dev_idx >= torch.cuda.device_count():
            raise RuntimeError(f"requested cuda:{dev_idx}, but device_count={torch.cuda.device_count()}")
        torch.cuda.set_device(dev_idx)
        # Initialize the CUDA runtime and allocate a small tensor. In exclusive-process
        # mode, the CUDA context itself is what reserves the device; the allocation is
        # intentionally tiny so it does not materially reduce training capacity.
        torch.cuda.init()
        n = max(1, int(args.memory_mb) * 1024 * 1024 // 4)
        held = torch.empty(n, dtype=torch.float32, device=f"cuda:{dev_idx}")
        held.fill_(0.0)
        torch.cuda.synchronize(dev_idx)
        info = {
            "pid": os.getpid(),
            "device": dev_idx,
            "device_name": torch.cuda.get_device_name(dev_idx),
            "memory_mb": int(args.memory_mb),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "purpose": "MalSnif pipeline GPU reservation during CPU-only phases",
        }
        ready = Path(args.ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(info, ensure_ascii=False), flush=True)
        hb = Path(args.heartbeat_file) if args.heartbeat_file else None
        while not _STOP:
            if hb is not None:
                try:
                    hb.parent.mkdir(parents=True, exist_ok=True)
                    hb.write_text(json.dumps({**info, "heartbeat_at": time.time()}, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            time.sleep(max(0.5, float(args.heartbeat_seconds)))
        # Keep a reference until the signal arrives, then release gracefully.
        del held
        torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        print(f"[gpu_guard] failed to reserve cuda device {args.device}: {exc!r}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
