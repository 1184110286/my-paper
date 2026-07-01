from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import torch

from malsnif.config import Config
from malsnif.data.dataset import preprocess, metadata_output_dir, strict_split_precheck, strict_split_autostop_precheck
from malsnif.train import train
from malsnif.evaluate import evaluate_checkpoint
from malsnif.utils.io import ensure_dir, save_json
from malsnif.data.diagnostics import print_diagnosis, diagnose_metadata
from malsnif.analyze import analyze_run_dir
import json


def _configure_stdio_utf8() -> None:
    """Make Git Bash/Windows console logs readable for Chinese diagnostics."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _device_from_arg(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    try:
        idx = int(device_arg)
        return f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    except Exception:
        return device_arg


def load_cfg(args) -> Config:
    cfg = Config.from_yaml(args.config)
    if getattr(args, "raw", None):
        cfg.raw_dir = args.raw
    if getattr(args, "processed", None):
        cfg.processed_dir = args.processed
    if getattr(args, "run_dir", None):
        cfg.run_dir = args.run_dir
    if getattr(args, "metadata_dir", None):
        cfg.metadata_dir = args.metadata_dir
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    if getattr(args, "label_dir", None):
        cfg.label_dir = args.label_dir
    if getattr(args, "input_format", None):
        cfg.input_format = args.input_format
    if getattr(args, "raw_glob", None):
        cfg.raw_glob = args.raw_glob
    if getattr(args, "max_events", None) is not None:
        cfg.max_events = args.max_events
    if getattr(args, "window_events", None) is not None:
        cfg.window_events = args.window_events
    if getattr(args, "threshold_strategy", None):
        cfg.threshold_strategy = args.threshold_strategy
    if getattr(args, "threshold", None) is not None:
        cfg.threshold = args.threshold
    if getattr(args, "node_scope", None):
        cfg.node_scope = args.node_scope
    if getattr(args, "threshold_min_recall", None) is not None:
        cfg.threshold_min_recall = args.threshold_min_recall
    return cfg


def main(argv=None):
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="MalSnif reproduction CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="parse logs, simplify provenance graphs, build word embeddings")
    p.add_argument("--config", required=True)
    p.add_argument("--raw")
    p.add_argument("--processed")
    p.add_argument("--metadata-dir")
    p.add_argument("--label-dir")
    p.add_argument("--input-format")
    p.add_argument("--raw-glob")
    p.add_argument("--max-events", type=int)
    p.add_argument("--window-events", type=int)

    p = sub.add_parser("strict-precheck", help="lightweight split-rigor check without vocab/cache generation")
    p.add_argument("--config", required=True)
    p.add_argument("--raw")
    p.add_argument("--processed")
    p.add_argument("--metadata-dir")
    p.add_argument("--label-dir")
    p.add_argument("--input-format")
    p.add_argument("--raw-glob")
    p.add_argument("--max-events", type=int)
    p.add_argument("--window-events", type=int)

    p = sub.add_parser("strict-precheck-autostop", help="single-pass strict split search that stops at the earliest rigorous prefix")
    p.add_argument("--config", required=True)
    p.add_argument("--raw")
    p.add_argument("--processed")
    p.add_argument("--metadata-dir")
    p.add_argument("--label-dir")
    p.add_argument("--input-format")
    p.add_argument("--raw-glob")
    p.add_argument("--max-events", type=int)
    p.add_argument("--window-events", type=int)
    p.add_argument("--required-splits", default="train,val,test")
    p.add_argument("--check-every-windows", type=int, default=1)
    p.add_argument("--min-graphs-per-split", type=int, default=None)
    p.add_argument("--require-graph-mix", action="store_true")
    p.add_argument("--require-node-mix", action="store_true")

    p = sub.add_parser("train", help="train MalSnif on processed graphs")
    p.add_argument("--config", required=True)
    p.add_argument("--processed")
    p.add_argument("--metadata-dir")
    p.add_argument("--label-dir")
    p.add_argument("--input-format")
    p.add_argument("--raw-glob")
    p.add_argument("--run-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-events", type=int)
    p.add_argument("--window-events", type=int)
    p.add_argument("--threshold-strategy")
    p.add_argument("--threshold", type=float)
    p.add_argument("--threshold-min-recall", type=float)
    p.add_argument("--node-scope", choices=["auto", "process", "all"])

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--processed")
    p.add_argument("--metadata-dir")
    p.add_argument("--run-dir")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float)
    p.add_argument("--threshold-min-recall", type=float)
    p.add_argument("--node-scope", choices=["auto", "process", "all"])

    p = sub.add_parser("diagnose", help="inspect processed metadata and print sanity checks")
    p.add_argument("--processed", required=True)
    p.add_argument("--metadata-dir")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of human-readable text")

    p = sub.add_parser("analyze-run", help="summarize a run directory and flag metric pathologies")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)

    p = sub.add_parser("run", help="preprocess + train + evaluate")
    p.add_argument("--config", required=True)
    p.add_argument("--raw")
    p.add_argument("--processed")
    p.add_argument("--metadata-dir")
    p.add_argument("--label-dir")
    p.add_argument("--input-format")
    p.add_argument("--raw-glob")
    p.add_argument("--run-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-events", type=int)
    p.add_argument("--window-events", type=int)
    p.add_argument("--threshold-strategy")
    p.add_argument("--threshold", type=float)
    p.add_argument("--threshold-min-recall", type=float)
    p.add_argument("--node-scope", choices=["auto", "process", "all"])

    args = parser.parse_args(argv)
    if args.cmd == "diagnose":
        if getattr(args, "json", False):
            print(json.dumps(diagnose_metadata(args.processed, metadata_dir=getattr(args, "metadata_dir", None)), ensure_ascii=False, indent=2))
        else:
            print_diagnosis(args.processed, metadata_dir=getattr(args, "metadata_dir", None))
        return
    if args.cmd == "analyze-run":
        result = analyze_run_dir(args.run_dir)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        if args.out:
            save_json(result, args.out)
        return
    cfg = load_cfg(args)
    if args.cmd == "preprocess":
        cfg.save(metadata_output_dir(cfg) / "config.preprocess.yaml")
        meta = preprocess(cfg)
        print(f"processed {meta['num_graphs']} graphs -> {cfg.processed_dir}")
    elif args.cmd == "strict-precheck":
        cfg.save(metadata_output_dir(cfg) / "config.precheck.yaml")
        meta = strict_split_precheck(cfg)
        print(f"strict precheck {meta['num_graphs']} graphs -> {metadata_output_dir(cfg)}")
    elif args.cmd == "strict-precheck-autostop":
        cfg.save(metadata_output_dir(cfg) / "config.precheck.yaml")
        required_splits = [x.strip() for x in str(getattr(args, "required_splits", "train,val,test")).split(",") if x.strip()]
        meta = strict_split_autostop_precheck(
            cfg,
            required_splits=required_splits,
            require_graph_mix=bool(getattr(args, "require_graph_mix", False)),
            require_node_mix=bool(getattr(args, "require_node_mix", False)),
            min_graphs_per_split=getattr(args, "min_graphs_per_split", None),
            check_every_windows=getattr(args, "check_every_windows", 1),
        )
        search = meta.get("strict_search", {}) or {}
        print(
            "strict autostop precheck "
            f"{meta['num_graphs']} graphs -> {metadata_output_dir(cfg)} "
            f"(pass={search.get('overall_pass')}, selected_max_events={search.get('selected_max_events')})"
        )
    elif args.cmd == "train":
        ensure_dir(cfg.run_dir)
        cfg.save(Path(cfg.run_dir) / "config.resolved.yaml")
        summary = train(cfg, _device_from_arg(args.device))
        print(summary)
    elif args.cmd == "evaluate":
        ckpt = args.checkpoint or str(Path(getattr(cfg, "checkpoint_dir", None) or (Path(cfg.run_dir) / "checkpoints")) / "best.pt")
        result = evaluate_checkpoint(cfg, ckpt, split=args.split, device_str=_device_from_arg(args.device), out_path=Path(cfg.run_dir) / f"metrics_{args.split}.json", threshold=getattr(args, "threshold", None))
        print(result["metrics"])
    elif args.cmd == "run":
        ensure_dir(cfg.run_dir)
        cfg.save(Path(cfg.run_dir) / "config.resolved.yaml")
        meta = preprocess(cfg)
        save_json(meta, Path(cfg.run_dir) / "preprocess_metadata.json")
        summary = train(cfg, _device_from_arg(args.device))
        selected_threshold = None
        try:
            selected_threshold = float(summary.get("best_threshold"))
        except Exception:
            selected_threshold = None
        result = evaluate_checkpoint(cfg, Path(getattr(cfg, "checkpoint_dir", None) or (Path(cfg.run_dir) / "checkpoints")) / "best.pt", split="test", device_str=_device_from_arg(args.device), out_path=Path(cfg.run_dir) / "metrics_test.json", threshold=selected_threshold)
        print({"train": summary, "test": result["metrics"], "warnings": result.get("warnings", [])})
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main()
