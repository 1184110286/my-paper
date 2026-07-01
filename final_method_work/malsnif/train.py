from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from malsnif.config import Config
from malsnif.data.dataset import ProcessedGraphs, load_vocab
from malsnif.data.masks import resolve_node_scope, node_mask_torch, node_mask_numpy, scope_label_counts
from malsnif.models.malsnif import MalSnifModel
from malsnif.evaluate import evaluate_model, evaluate_checkpoint
from malsnif.utils.io import ensure_dir, save_json
from malsnif.utils.plot import plot_history
from malsnif.utils.seed import set_seed
from malsnif.utils.metrics import binary_metrics
from malsnif.utils.cuda_memory import reset_cuda_peak, cuda_memory_stats, empty_cuda_cache


def _make_grad_scaler(enabled: bool):
    """Create a GradScaler without triggering deprecation warnings on new PyTorch.

    PyTorch 2.x prefers torch.amp.GradScaler('cuda', ...), while older versions
    only expose torch.cuda.amp.GradScaler.  Keep both paths so the project works
    across the lab machines.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device.type, dtype=dtype, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def _looks_like_amp_runtime_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    needles = [
        "half", "bfloat16", "float16", "scalar type", "expected scalar",
        "not implemented for", "must have the same scalar type", "cuda amp",
    ]
    return any(n in msg for n in needles)


def _sample_loss_indices(labels: torch.Tensor, target_mask: torch.Tensor, cfg: Config) -> torch.Tensor:
    idx = torch.where(target_mask)[0]
    if idx.numel() == 0:
        return torch.arange(labels.numel(), device=labels.device)
    target_labels = labels[idx]
    pos = idx[target_labels == 1]
    neg = idx[target_labels == 0]
    sampling = str(getattr(cfg, "loss_sampling", "paper") or "paper").lower()
    if sampling == "all" or not cfg.downsample_after_forward:
        return idx
    if sampling == "balanced":
        if pos.numel() == 0 or neg.numel() == 0:
            return idx
        k = min(pos.numel(), neg.numel())
        psel = pos[torch.randperm(pos.numel(), device=labels.device)[:k]]
        nsel = neg[torch.randperm(neg.numel(), device=labels.device)[:k]]
        return torch.cat([psel, nsel], dim=0)
    # paper-style: keep all positives and at most downsample_weight x positives negatives.
    if pos.numel() == 0:
        k = min(neg.numel(), max(1, cfg.downsample_weight * 8))
        perm = torch.randperm(neg.numel(), device=labels.device)[:k]
        return neg[perm]
    k = min(neg.numel(), int(cfg.downsample_weight * pos.numel()))
    if k <= 0:
        return pos
    perm = torch.randperm(neg.numel(), device=labels.device)[:k]
    return torch.cat([pos, neg[perm]], dim=0)


def _weighted_bce_logits(logits: torch.Tensor, targets: torch.Tensor, cfg: Config) -> torch.Tensor:
    if getattr(cfg, "balanced_loss", False) and targets.numel() > 0:
        pos = targets.sum()
        neg = targets.numel() - pos
        if pos > 0 and neg > 0:
            w_pos = targets.numel() / (2.0 * pos)
            w_neg = targets.numel() / (2.0 * neg)
            if getattr(cfg, "loss_positive_weight", None) is not None:
                w_pos = torch.tensor(float(cfg.loss_positive_weight), dtype=torch.float32, device=targets.device)
            weights = torch.where(targets > 0.5, w_pos, w_neg)
            return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
    return F.binary_cross_entropy_with_logits(logits, targets)


def _loss_for_graph(out: dict, graph: dict, cfg: Config, device, node_scope: str | None = None) -> torch.Tensor:
    if cfg.graph_level:
        target = torch.tensor([float(graph["graph_label"])], dtype=torch.float32, device=device)
        logit = torch.logit(out["graph_prob"].clamp(1e-6, 1 - 1e-6)).view(1)
        return _weighted_bce_logits(logit, target, cfg)
    labels = torch.tensor(graph["node_labels"], dtype=torch.float32, device=device)
    target_mask = node_mask_torch(graph, node_scope or getattr(cfg, "node_scope", "process"), device)
    idx = _sample_loss_indices(labels, target_mask, cfg)
    logits = out["node_logits"][idx]
    return _weighted_bce_logits(logits, labels[idx], cfg)


def _metric_lookup(metric_name: str, row: dict, val_metrics: dict, train_metrics: dict) -> float:
    """Look up a train/validation metric with graceful aliases."""
    metric_name = str(metric_name or "").strip().lower()
    if not metric_name:
        return 0.0
    aliases = {
        "val_auc_pr": "val_average_precision",
        "val_pr_auc": "val_average_precision",
        "val_ap": "val_average_precision",
        "val_auc_roc": "val_roc_auc",
        "train_auc_pr": "train_average_precision",
        "train_pr_auc": "train_average_precision",
        "train_ap": "train_average_precision",
        "train_auc_roc": "train_roc_auc",
    }
    metric_name = aliases.get(metric_name, metric_name)
    if metric_name.startswith("val_"):
        value = val_metrics.get(metric_name[4:])
    elif metric_name.startswith("train_"):
        value = train_metrics.get(metric_name[6:])
    else:
        value = row.get(metric_name, val_metrics.get(metric_name, train_metrics.get(metric_name)))
    if value is None:
        return 0.0
    try:
        value_f = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(value_f):
        return 0.0
    return value_f


def _selection_value(row: dict, val_metrics: dict, train_metrics: dict, cfg: Config) -> float:
    metric_name = str(getattr(cfg, "model_selection_metric", "val_average_precision") or "val_average_precision")
    value = _metric_lookup(metric_name, row, val_metrics, train_metrics)
    if value == 0.0:
        # Historical fallback used by earlier versions.
        value = float(val_metrics.get("mcc", val_metrics.get("balanced_accuracy", val_metrics.get("f1", 0.0))) or 0.0)
    return value


def _selection_tuple(row: dict, val_metrics: dict, train_metrics: dict, cfg: Config) -> tuple[float, ...]:
    """Return lexicographic checkpoint-selection scores.

    Primary metric remains cfg.model_selection_metric.  Secondary metrics are
    tie breakers only.  This keeps the paper-style F1/precision/recall model
    selection while avoiding stale early checkpoints when validation F1 ties but
    AP/MCC improves substantially later.
    """
    primary = _selection_value(row, val_metrics, train_metrics, cfg)
    raw = getattr(cfg, "model_selection_tie_breakers", "") or ""
    if isinstance(raw, (list, tuple)):
        tie_names = list(raw)
    else:
        tie_names = [x.strip() for x in str(raw).split(",") if x.strip()]
    ties = [_metric_lookup(name, row, val_metrics, train_metrics) for name in tie_names]
    # Prefer lower loss if all validation/ranking metrics tie; then prefer later
    # epoch, because equal-threshold F1 often hides better score calibration.
    loss = row.get("loss")
    neg_loss = -float(loss) if loss is not None and np.isfinite(float(loss)) else 0.0
    epoch = float(row.get("epoch", 0) or 0)
    return tuple([primary] + ties + [neg_loss, epoch])


def _selection_is_better(candidate: tuple[float, ...], best: tuple[float, ...] | None, cfg: Config) -> bool:
    if best is None:
        return True
    eps = float(getattr(cfg, "model_selection_epsilon", 1e-9) or 0.0)
    max_len = max(len(candidate), len(best))
    c = list(candidate) + [0.0] * (max_len - len(candidate))
    b = list(best) + [0.0] * (max_len - len(best))
    for cv, bv in zip(c, b):
        if cv > bv + eps:
            return True
        if cv < bv - eps:
            return False
    return False



def _resolve_training_scope(cfg: Config, train_ds: ProcessedGraphs) -> tuple[str, dict | None]:
    """Choose a valid node-level training target.

    On short DARPA CDM prefixes, process-node labels can be single-class even
    though the graph contains labeled file/socket entities.  node_scope=auto
    keeps the process-node target when it has both classes and otherwise falls
    back to all nodes if that target is supervised.  Strict paper configs can
    still set node_scope=process and allow_unlabeled_training=false.
    """
    if cfg.graph_level:
        return "graph", None
    graphs = list(iter(train_ds))
    requested = str(getattr(cfg, "node_scope", "auto") or "auto").lower()
    resolved = resolve_node_scope(graphs, requested)
    diag = scope_label_counts(graphs, resolved)
    cfg.node_scope = resolved
    if requested == "auto" and resolved == "all":
        print("[WARN] node_scope=auto fallback: process-node labels are single-class in train; using all-node supervision for this run.", flush=True)
        print(json.dumps({"node_scope_diagnostics": diag}, ensure_ascii=False), flush=True)
    if not cfg.allow_unlabeled_training and not diag.get("has_both_classes"):
        raise ValueError(
            "Training target nodes contain only one class. "
            f"resolved_node_scope={resolved}, diagnostics={diag}. "
            "For DARPA CDM UUID labels, use node_label_policy=process_event_endpoints, or keep node_scope=auto for diagnostic fallback; "
            "otherwise increase MAX_EVENTS, reduce WINDOW_EVENTS to create more windows, adjust split_ratio, "
            "or set allow_unlabeled_training=true only for smoke tests."
        )
    return resolved, diag

def train(cfg: Config, device_str: str = "cpu") -> dict:
    set_seed(cfg.seed)
    run_dir = ensure_dir(cfg.run_dir)
    ckpt_dir = ensure_dir(getattr(cfg, "checkpoint_dir", None) or (run_dir / "checkpoints"))
    cfg.save(run_dir / "config.resolved.yaml")
    device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
    train_wall_start = time.perf_counter()
    meta_path = Path(cfg.processed_dir) / "metadata.json"
    if meta_path.exists() and not cfg.allow_unlabeled_training:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        if not cfg.graph_level and int(meta.get("parse_stats", {}).get("labeled_events_consumed", 0) or 0) == 0:
            raise ValueError(
                "No positive labels were found in processed_dir; supervised training is blocked to avoid invalid metrics. "
                "Check label_dir / DARPA ground-truth conversion, or set allow_unlabeled_training: true only for smoke tests."
            )
    vocab = load_vocab(cfg.processed_dir)
    model = MalSnifModel(vocab.embeddings, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_ds = ProcessedGraphs(
        cfg.processed_dir, split="train", metadata_dir=getattr(cfg, "metadata_dir", None),
        limit=getattr(cfg, "graph_limit_train", None),
        cache_in_memory=bool(getattr(cfg, "cache_graphs_in_memory", False)),
    )
    val_ds = ProcessedGraphs(
        cfg.processed_dir, split="val", metadata_dir=getattr(cfg, "metadata_dir", None),
        limit=getattr(cfg, "graph_limit_val", None),
        cache_in_memory=bool(getattr(cfg, "cache_graphs_in_memory", False)),
    )

    resolved_node_scope, scope_diag = _resolve_training_scope(cfg, train_ds)
    use_amp_requested = bool(getattr(cfg, "use_amp", False)) and device.type == "cuda"
    use_amp_active = use_amp_requested
    amp_fallback_occurred = False
    amp_fallback_error: str | None = None
    amp_dtype_name = str(getattr(cfg, "amp_dtype", "float16") or "float16").lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = _make_grad_scaler(use_amp_active)
    if use_amp_requested:
        print(json.dumps({"amp": True, "amp_dtype": amp_dtype_name, "amp_fallback_to_fp32": bool(getattr(cfg, "amp_fallback_to_fp32", True))}, ensure_ascii=False), flush=True)
    if getattr(cfg, "graph_limit_train", None):
        print(json.dumps({"fast_graph_limit_train": int(cfg.graph_limit_train), "train_graphs_used": len(train_ds)}, ensure_ascii=False), flush=True)
    if getattr(cfg, "graph_limit_val", None):
        print(json.dumps({"fast_graph_limit_val": int(cfg.graph_limit_val), "val_graphs_used": len(val_ds)}, ensure_ascii=False), flush=True)

    best_score = -1e18
    best_selection_tuple: tuple[float, ...] | None = None
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = float(cfg.threshold)
    bad = 0
    history: list[dict] = []
    empty_cache_calls = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        reset_cuda_peak(device)
        model.train()
        losses: list[float] = []
        train_scores: list[float] = []
        train_labels: list[int] = []
        iterable = train_ds
        if getattr(cfg, "train_progress", True):
            iterable = tqdm(train_ds, total=len(train_ds), desc=f"train epoch {epoch}/{cfg.epochs}", unit="graph", dynamic_ncols=True)
        train_phase_start = time.perf_counter()
        for graph_i, graph in enumerate(iterable, 1):
            opt.zero_grad(set_to_none=True)
            if use_amp_active:
                try:
                    with _autocast_context(device, amp_dtype, enabled=True):
                        out = model(graph, device)
                        loss = _loss_for_graph(out, graph, cfg, device, node_scope=resolved_node_scope)
                    scaler.scale(loss).backward()
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                except RuntimeError as exc:
                    if bool(getattr(cfg, "amp_fallback_to_fp32", True)) and _looks_like_amp_runtime_error(exc):
                        amp_fallback_occurred = True
                        amp_fallback_error = str(exc).split("\n", 1)[0]
                        print(json.dumps({
                            "warning": "AMP failed; falling back to fp32 for the rest of training",
                            "error": amp_fallback_error,
                        }, ensure_ascii=False), flush=True)
                        use_amp_active = False
                        opt.zero_grad(set_to_none=True)
                        out = model(graph, device)
                        loss = _loss_for_graph(out, graph, cfg, device, node_scope=resolved_node_scope)
                        loss.backward()
                        if cfg.grad_clip and cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        opt.step()
                    else:
                        raise
            else:
                out = model(graph, device)
                loss = _loss_for_graph(out, graph, cfg, device, node_scope=resolved_node_scope)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=f"{loss_value:.4f}", refresh=False)
            with torch.no_grad():
                if cfg.graph_level:
                    train_scores.append(float(out["graph_prob"].detach().cpu()))
                    train_labels.append(int(graph["graph_label"]))
                else:
                    p = out["node_probs"].detach().cpu().numpy()
                    m = node_mask_numpy(graph, resolved_node_scope)
                    y = np.asarray(graph["node_labels"], dtype=int)
                    train_scores.extend(p[m].tolist())
                    train_labels.extend(y[m].tolist())
            # Explicitly drop per-graph tensors before optional cache release.
            # This reduces nvidia-smi/reserved memory growth on many heterogeneous
            # graph windows without changing any computation.
            try:
                del out, loss
            except Exception:
                pass
            interval = int(getattr(cfg, "cuda_empty_cache_interval", 0) or 0)
            if interval > 0 and device.type == "cuda" and graph_i % interval == 0:
                if empty_cuda_cache(device, synchronize=False):
                    empty_cache_calls += 1
        train_seconds = time.perf_counter() - train_phase_start
        if getattr(cfg, "train_progress", True):
            print(json.dumps({"stage": "train_metrics", "epoch": epoch, "samples": len(train_labels), "train_seconds": train_seconds}, ensure_ascii=False), flush=True)
        train_metrics = binary_metrics(train_labels, train_scores, threshold=float(cfg.threshold)).to_dict() if train_labels else {}
        val_every = max(1, int(getattr(cfg, "val_every", 1) or 1))
        should_validate = bool(len(val_ds)) and (epoch == 1 or epoch == cfg.epochs or (epoch % val_every == 0))
        if should_validate and getattr(cfg, "train_progress", True):
            print(json.dumps({"stage": "validation_start", "epoch": epoch, "val_graphs": len(val_ds), "val_every": val_every}, ensure_ascii=False), flush=True)
        validation_phase_start = time.perf_counter()
        val_result = evaluate_model(model, val_ds, cfg, device, node_scope=resolved_node_scope) if should_validate else {"metrics": {}, "warnings": [f"validation skipped at epoch {epoch}; val_every={val_every}"]}
        validation_seconds = time.perf_counter() - validation_phase_start if should_validate else 0.0
        if should_validate and bool(getattr(cfg, "cuda_empty_cache_after_eval", False)) and device.type == "cuda":
            if empty_cuda_cache(device, synchronize=False):
                empty_cache_calls += 1
        if bool(getattr(cfg, "cuda_empty_cache_after_epoch", False)) and device.type == "cuda":
            if empty_cuda_cache(device, synchronize=False):
                empty_cache_calls += 1
        val_metrics = val_result["metrics"]
        if getattr(cfg, "train_progress", True):
            print(json.dumps({
                "stage": "validation_metrics_done" if should_validate else "validation_skipped",
                "epoch": epoch,
                "val_samples": val_metrics.get("num_samples"),
                "threshold": val_metrics.get("threshold"),
                "validation_seconds": validation_seconds,
                "val_every": val_every,
            }, ensure_ascii=False), flush=True)
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "node_scope": resolved_node_scope, "validated": bool(should_validate)}
        show_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "average_precision", "predicted_positive", "predicted_negative"]
        row.update({k: train_metrics.get(k) for k in show_keys})
        row.update({f"val_{k}": val_metrics.get(k) for k in show_keys})
        if val_metrics.get("threshold") is not None:
            row["val_threshold"] = val_metrics.get("threshold")
        epoch_seconds = time.perf_counter() - epoch_start
        row.update({
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds,
            "seconds_per_train_graph": train_seconds / max(len(train_ds), 1),
            "seconds_per_val_graph": (validation_seconds / max(len(val_ds), 1)) if should_validate else None,
        })
        row.update(cuda_memory_stats(device))
        row["cuda_empty_cache_calls"] = empty_cache_calls
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if should_validate:
            current_score = _selection_value(row, val_metrics, train_metrics, cfg)
            current_tuple = _selection_tuple(row, val_metrics, train_metrics, cfg)
            val_f1 = float(val_metrics.get("f1", train_metrics.get("f1", 0.0)) or 0.0)
            row["selection_score"] = current_score
            row["selection_tuple"] = list(current_tuple)
            if _selection_is_better(current_tuple, best_selection_tuple, cfg):
                best_score = current_score
                best_selection_tuple = current_tuple
                best_f1 = val_f1
                best_epoch = epoch
                best_threshold = float(val_metrics.get("threshold", cfg.threshold))
                bad = 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": cfg.to_dict(),
                    "threshold": best_threshold,
                    "threshold_diagnostics": val_result.get("threshold_diagnostics"),
                    "selection_metric": cfg.model_selection_metric,
                    "selection_tie_breakers": getattr(cfg, "model_selection_tie_breakers", ""),
                    "selection_score": current_score,
                    "selection_tuple": list(current_tuple),
                    "resolved_node_scope": resolved_node_scope,
                    "node_scope_diagnostics": scope_diag,
                }, ckpt_dir / "best.pt")
            else:
                bad += 1
            if cfg.patience and bad >= cfg.patience:
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_selection_score={best_score:.4f}", flush=True)
                break
    torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"], "config": cfg.to_dict(), "threshold": best_threshold, "resolved_node_scope": resolved_node_scope, "node_scope_diagnostics": scope_diag}, ckpt_dir / "last.pt")
    save_json(history, run_dir / "history.json")
    plot_history(history, run_dir / "plots" / "history.png", mode=getattr(cfg, "plot_mode", "essential"), metric_keys=getattr(cfg, "plot_metric_keys", None))
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_selection_score": best_score,
        "best_selection_tuple": list(best_selection_tuple) if best_selection_tuple is not None else None,
        "best_threshold": best_threshold,
        "model_selection_metric": cfg.model_selection_metric,
        "model_selection_tie_breakers": getattr(cfg, "model_selection_tie_breakers", ""),
        "model_selection_epsilon": getattr(cfg, "model_selection_epsilon", None),
        "threshold_strategy": cfg.threshold_strategy,
        "resolved_node_scope": resolved_node_scope,
        "node_scope_diagnostics": scope_diag,
        "epochs_run": len(history),
        "checkpoint_dir": str(ckpt_dir),
        "train_graphs_used": len(train_ds),
        "val_graphs_used": len(val_ds),
        "graph_limit_train": getattr(cfg, "graph_limit_train", None),
        "graph_limit_val": getattr(cfg, "graph_limit_val", None),
        "val_every": getattr(cfg, "val_every", 1),
        "cache_graphs_in_memory": bool(getattr(cfg, "cache_graphs_in_memory", False)),
        "model_variant": getattr(cfg, "model_variant", "baseline"),
        "semantic_encoder": getattr(cfg, "semantic_encoder", "baseline"),
        "graph_encoder": getattr(cfg, "graph_encoder", "graphsage"),
        "fusion_mode": getattr(cfg, "fusion_mode", "baseline"),
        "hgan_topk": getattr(cfg, "hgan_topk", None),
        "hgan_pruning_mode": getattr(cfg, "hgan_pruning_mode", None),
        "hgan_use_time_bias": getattr(cfg, "hgan_use_time_bias", None),
        "edge_gate_mode": getattr(cfg, "edge_gate_mode", None),
        "edge_gate_use_edge_semantics": getattr(cfg, "edge_gate_use_edge_semantics", None),
        "edge_chunk_size": getattr(cfg, "edge_chunk_size", None),
        "use_amp_requested": use_amp_requested,
        "use_amp": use_amp_active,
        "amp_dtype": amp_dtype_name if use_amp_requested else None,
        "amp_fallback_to_fp32": bool(getattr(cfg, "amp_fallback_to_fp32", True)),
        "amp_fallback_occurred": amp_fallback_occurred,
        "amp_fallback_error": amp_fallback_error,
        "train_total_seconds": time.perf_counter() - train_wall_start,
        "avg_epoch_seconds": float(np.mean([r.get("epoch_seconds", 0.0) for r in history])) if history else None,
        "avg_seconds_per_train_graph": float(np.mean([r.get("seconds_per_train_graph", 0.0) for r in history])) if history else None,
        "max_cuda_peak_allocated_mb": max([float(r.get("cuda_peak_allocated_mb", 0.0) or 0.0) for r in history], default=None),
        "max_cuda_peak_reserved_mb": max([float(r.get("cuda_peak_reserved_mb", 0.0) or 0.0) for r in history], default=None),
        "last_cuda_current_allocated_mb": (history[-1].get("cuda_current_allocated_mb") if history else None),
        "last_cuda_current_reserved_mb": (history[-1].get("cuda_current_reserved_mb") if history else None),
        "cuda_empty_cache_interval": int(getattr(cfg, "cuda_empty_cache_interval", 0) or 0),
        "cuda_empty_cache_after_epoch": bool(getattr(cfg, "cuda_empty_cache_after_epoch", False)),
        "cuda_empty_cache_after_eval": bool(getattr(cfg, "cuda_empty_cache_after_eval", False)),
        "cuda_empty_cache_calls": empty_cache_calls,
    }
    save_json(summary, run_dir / "train_summary.json")
    return summary
