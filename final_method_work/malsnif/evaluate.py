from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
from collections import defaultdict
import numpy as np
import torch
from tqdm.auto import tqdm

from malsnif.config import Config
from malsnif.data.dataset import ProcessedGraphs, load_vocab
from malsnif.data.masks import node_mask_numpy
from malsnif.models.malsnif import MalSnifModel
from malsnif.utils.metrics import binary_metrics, choose_threshold, score_distribution, precision_at_k, metric_warnings
from malsnif.utils.io import save_json
from malsnif.utils.plot import plot_score_histogram
from malsnif.utils.cuda_memory import reset_cuda_peak, cuda_memory_stats, empty_cuda_cache


def _is_auto_threshold(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith("auto")


def _threshold_metric_from_cfg(cfg: Config) -> str:
    strategy = str(getattr(cfg, "threshold_strategy", "fixed") or "fixed").lower()
    if strategy.startswith("val_"):
        strategy = strategy[4:]
    if strategy in {"balanced", "bal"}:
        return "balanced_accuracy"
    if strategy in {"precision_at_recall", "precision_at_recall_095", "precision_at_recall_0.95"}:
        return "precision_at_recall"
    if strategy.endswith("_min_recall"):
        return strategy
    if strategy in {"f1", "mcc", "balanced_accuracy", "precision", "recall", "accuracy"}:
        return strategy
    if isinstance(cfg.threshold, str):
        name = cfg.threshold.lower()
        if name.startswith("auto_"):
            name = name[len("auto_"):]
        if name in {"balanced", "bal"}:
            return "balanced_accuracy"
        if name in {"precision_at_recall", "precision_at_recall_095", "precision_at_recall_0.95"}:
            return "precision_at_recall"
        if name in {"f1", "mcc", "balanced_accuracy", "precision", "recall", "accuracy"}:
            return name
    return "f1"




def _topk_indices_desc(values: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the largest k values in descending order.

    This keeps analysis output identical in meaning to argsort(-scores)[:k] but
    avoids sorting every process node when only a small analyst-facing top-k is
    needed. Metrics still use all scores/labels.
    """
    arr = np.asarray(values)
    if arr.size == 0 or k <= 0:
        return np.asarray([], dtype=int)
    k = min(int(k), int(arr.size))
    if k == arr.size:
        return np.argsort(-arr, kind="mergesort")
    part = np.argpartition(-arr, k - 1)[:k]
    return part[np.argsort(-arr[part], kind="mergesort")]

def collect_scores(model: MalSnifModel, dataset: ProcessedGraphs, cfg: Config, device: torch.device, node_scope: str | None = None) -> dict:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    graph_rows: list[dict] = []
    with torch.no_grad():
        iterable = dataset
        if getattr(cfg, "train_progress", True):
            iterable = tqdm(dataset, total=len(dataset), desc="evaluate", unit="graph", dynamic_ncols=True)
        for gi, graph in enumerate(iterable):
            abs_gi = int(graph.get("_graph_index", gi)) if isinstance(graph, dict) else gi
            out = model(graph, device)
            if cfg.graph_level:
                score = float(out["graph_prob"].detach().cpu())
                label = int(graph["graph_label"])
                scores.append(score)
                labels.append(label)
                graph_rows.append({"graph_index": abs_gi, "score": score, "label": label})
            else:
                probs = out["node_probs"].detach().cpu().numpy()
                mask = node_mask_numpy(graph, node_scope or getattr(cfg, "node_scope", "process"))
                y = np.asarray(graph["node_labels"], dtype=int)
                ps = probs[mask]
                ys = y[mask]
                scores.extend(ps.tolist())
                labels.extend(ys.tolist())
                top_k = int(getattr(cfg, "top_alerts_per_graph", 50) or 0)
                top_idx = _topk_indices_desc(ps, top_k).tolist() if ps.size and top_k > 0 else []
                selected_indices = np.where(mask)[0]
                displays = graph.get("node_displays", []) or []
                total_counts = graph.get("node_total_event_counts", []) or []
                labeled_counts = graph.get("node_labeled_event_counts", []) or []
                token_samples = graph.get("node_event_token_samples", []) or []
                top_alerts = [
                    {
                        "node_index": int(selected_indices[i]),
                        "node_id": graph["node_ids"][int(selected_indices[i])],
                        "node_display": displays[int(selected_indices[i])] if len(displays) > int(selected_indices[i]) else graph["node_ids"][int(selected_indices[i])],
                        "node_type": graph.get("node_types", [])[int(selected_indices[i])] if graph.get("node_types") else "",
                        "event_count": int(total_counts[int(selected_indices[i])]) if len(total_counts) > int(selected_indices[i]) else None,
                        "labeled_event_count": int(labeled_counts[int(selected_indices[i])]) if len(labeled_counts) > int(selected_indices[i]) else None,
                        "score": float(ps[i]),
                        "label": int(ys[i]),
                        "event_token_sample": token_samples[int(selected_indices[i])] if len(token_samples) > int(selected_indices[i]) else "",
                    }
                    for i in top_idx
                ]
                proc_mask = np.asarray(graph.get("process_mask", []), dtype=bool)
                graph_rows.append({
                    "graph_index": abs_gi,
                    "edge_weight_stats": out.get("edge_weight_stats"),
                    "gate_stats": out.get("gate_stats"),
                    "attention_stats": out.get("attention_stats"),
                    "node_scope": node_scope or getattr(cfg, "node_scope", "process"),
                    "num_target_nodes": int(mask.sum()),
                    "positive_target_nodes": int(ys.sum()),
                    "num_process_nodes": int(proc_mask.sum()) if proc_mask.size else None,
                    "positive_process_nodes": int(y[proc_mask].sum()) if proc_mask.size == y.size else None,
                    "graph_label": int(graph["graph_label"]),
                    "top_alerts": top_alerts,
                    "top50_precision": float(sum(x["label"] for x in top_alerts) / max(len(top_alerts), 1)),
                })
            try:
                del out
            except Exception:
                pass
            interval = int(getattr(cfg, "cuda_empty_cache_interval", 0) or 0)
            if interval > 0 and device.type == "cuda" and (gi + 1) % interval == 0:
                empty_cuda_cache(device, synchronize=False)
    if bool(getattr(cfg, "cuda_empty_cache_after_eval", False)) and device.type == "cuda":
        empty_cuda_cache(device, synchronize=False)
    return {"scores": scores, "labels": labels, "graphs": graph_rows}


def evaluate_model(
    model: MalSnifModel,
    dataset: ProcessedGraphs,
    cfg: Config,
    device: torch.device,
    threshold: float | None = None,
    threshold_source: str = "config",
    node_scope: str | None = None,
) -> dict:
    node_scope = node_scope or getattr(cfg, "node_scope", "process")
    collected = collect_scores(model, dataset, cfg, device, node_scope=node_scope)
    scores = collected["scores"]
    labels = collected["labels"]
    threshold_diag: dict[str, Any] | None = None
    if threshold is None:
        strategy = str(getattr(cfg, "threshold_strategy", "fixed") or "fixed").lower()
        if strategy != "fixed" or _is_auto_threshold(cfg.threshold):
            metric = _threshold_metric_from_cfg(cfg)
            threshold, threshold_diag = choose_threshold(labels, scores, metric=metric, min_recall=getattr(cfg, "threshold_min_recall", 0.95), show_progress=bool(getattr(cfg, "train_progress", True)))
            threshold_source = f"auto:{metric}"
        else:
            threshold = float(cfg.threshold)
            threshold_source = "config"
    md = binary_metrics(labels, scores, threshold=threshold).to_dict() if labels else {}
    md["num_samples"] = len(labels)
    md["threshold"] = float(threshold)
    md["threshold_source"] = threshold_source
    md["node_scope"] = node_scope
    all_pos = binary_metrics(labels, np.ones(len(labels)), threshold=0.5).to_dict() if labels else {}
    all_neg = binary_metrics(labels, np.zeros(len(labels)), threshold=0.5).to_dict() if labels else {}
    y_pred = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int).tolist() if scores else []
    warnings: list[str] = []
    if md.get("predicted_negative", 0) == 0 and md.get("negative_count", 0) > 0:
        warnings.append("当前阈值把所有样本都预测为正类；F1/Accuracy 会被正样本占比严重抬高。")
    if md.get("roc_auc") is not None and md["roc_auc"] < 0.5:
        warnings.append("ROC-AUC < 0.5，模型分数方向可能反了，或当前 split/标签构成导致排序学习失败；请优先检查语义 token、标签口径和更大时间范围。")
    if md.get("negative_count", 0) and md.get("positive_count", 0) / max(md.get("num_samples", 1), 1) > 0.8:
        warnings.append("评估集中正样本占比超过 80%；请同时报告 MCC、Balanced Accuracy、ROC-AUC、Precision@K，而不要只看 F1。")
    for w in metric_warnings(md):
        if w not in warnings:
            warnings.append(w)
    graph_labels = [int(g.get("graph_label", 0) or 0) for g in collected.get("graphs", [])]
    if graph_labels and all(x == 1 for x in graph_labels):
        warnings.append(
            "当前评估图窗口全部为 graph_label=1；graph-level 没有负图对照，"
            "node-level 指标也应结合更长良性窗口和 split 诊断解释。"
        )
    gate_rows = [g.get("gate_stats") for g in collected.get("graphs", []) if g.get("gate_stats")]
    if gate_rows:
        # Only warn about gate collapse for actual learnable gates.  Semantic-only
        # and structure-only ablations intentionally report constant pseudo-gates.
        # A vector gate may have a stable graph-level mean while still being very
        # diverse across nodes/dimensions, so also inspect within-graph std.
        learnable_rows = [
            r for r in gate_rows
            if str(r.get("gate_mode", "")).lower() in {"vector", "scalar", "edge_vector", "edge_scalar"}
        ]
        vals = []
        stds = []
        for row in learnable_rows:
            v = row.get("gate_semantic_mean")
            sdev = row.get("gate_semantic_std")
            try:
                if v is not None and np.isfinite(float(v)):
                    vals.append(float(v))
            except Exception:
                pass
            try:
                if sdev is not None and np.isfinite(float(sdev)):
                    stds.append(float(sdev))
            except Exception:
                pass
        if vals:
            mean_v = float(np.mean(vals))
            cross_graph_std = float(np.std(vals)) if len(vals) > 1 else 0.0
            within_mean_std = float(np.mean(stds)) if stds else 0.0
            near_extreme = mean_v < 0.02 or mean_v > 0.98
            collapsed_variation = cross_graph_std < 1e-3 and within_mean_std < 0.02
            if near_extreme and collapsed_variation:
                warnings.append(
                    "learnable gate 几乎塌缩为单分支常数；需要用 A3/A7 消融确认门控不是常数融合。"
                )
    result = {
        "node_scope": node_scope,
        "metrics": md,
        "all_positive_baseline": all_pos,
        "all_negative_baseline": all_neg,
        "threshold_diagnostics": threshold_diag,
        "score_distribution": score_distribution(labels, scores),
        "ranking_metrics": precision_at_k(labels, scores),
        "warnings": warnings,
        "edge_weight_distribution": aggregate_edge_weight_stats(collected.get("graphs", [])),
        "gate_distribution": aggregate_gate_stats(collected.get("graphs", [])),
        "attention_distribution": aggregate_attention_stats(collected.get("graphs", [])),
        "scores": scores,
        "labels": labels,
        "predictions": y_pred,
        "graphs": collected["graphs"],
    }
    return result





def aggregate_gate_stats(graph_rows: list[dict]) -> dict | None:
    rows = [g.get("gate_stats") for g in (graph_rows or []) if g.get("gate_stats")]
    if not rows:
        return None
    numeric_keys = ["gate_semantic_mean", "gate_semantic_std", "gate_semantic_min", "gate_semantic_max", "gate_structure_mean", "edge_gate_mean", "edge_gate_std", "edge_gate_min", "edge_gate_max"]
    out: dict = {"num_graphs": len(rows), "gate_modes": sorted(set(str(r.get("gate_mode", "")) for r in rows))}
    for k in numeric_keys:
        vals = []
        for r in rows:
            try:
                v = r.get(k)
                if v is not None and np.isfinite(float(v)):
                    vals.append(float(v))
            except Exception:
                pass
        if vals:
            out[k] = float(np.mean(vals))
    return out


def aggregate_attention_stats(graph_rows: list[dict]) -> dict | None:
    layer_rows = []
    for g in graph_rows or []:
        att = g.get("attention_stats") or {}
        for row in att.get("layers", []) or []:
            layer_rows.append(row)
    if not layer_rows:
        return None
    total_edges = sum(int(r.get("edges", 0) or 0) for r in layer_rows)
    total_kept = sum(int(r.get("kept_edges", 0) or 0) for r in layer_rows)
    out = {
        "num_layer_rows": len(layer_rows),
        "edges": int(total_edges),
        "kept_edges": int(total_kept),
        "kept_ratio": float(total_kept / max(total_edges, 1)) if total_edges else None,
        "topk_values": sorted(set(int(r.get("topk", 0) or 0) for r in layer_rows)),
        "pruning_modes": sorted(set(str(r.get("pruning_mode", "")) for r in layer_rows)),
    }
    for k in [
        "alpha_min", "alpha_max", "alpha_mean",
        "ets_tau_mean", "ets_tau_std", "ets_tau_min", "ets_tau_max",
        "eaw_head_mean", "eaw_head_std", "eaw_head_min", "eaw_head_max",
    ]:
        vals = []
        for r in layer_rows:
            try:
                v = r.get(k)
                if v is not None and np.isfinite(float(v)):
                    vals.append(float(v))
            except Exception:
                pass
        if vals:
            out[k] = float(np.mean(vals))
    # EHA stats are stored once at encoder level, not per layer.  Preserve them
    # from the graph-level attention_stats dictionaries when present.
    eha_rows = []
    for g in graph_rows or []:
        att = g.get("attention_stats") or {}
        if any(str(k).startswith("eha_") for k in att):
            eha_rows.append(att)
    for k in ["eha_hops", "eha_weight_mean", "eha_weight_std", "eha_entropy_mean", "eha_hop_0_mean", "eha_hop_1_mean", "eha_hop_2_mean", "ea_use_eha", "ea_use_ets", "ea_use_eaw"]:
        vals = []
        for r in eha_rows:
            try:
                v = r.get(k)
                if isinstance(v, bool):
                    vals.append(float(v))
                elif v is not None and np.isfinite(float(v)):
                    vals.append(float(v))
            except Exception:
                pass
        if vals:
            out[k] = float(np.mean(vals))
    return out

def aggregate_edge_weight_stats(graph_rows: list[dict]) -> dict | None:
    stats = [g.get("edge_weight_stats") for g in (graph_rows or []) if g.get("edge_weight_stats")]
    if not stats:
        return None
    total = sum(int(s.get("count", 0) or 0) for s in stats)
    if total <= 0:
        return None
    mean = sum(float(s.get("mean", 0.0)) * int(s.get("count", 0) or 0) for s in stats) / total
    # Combine per-graph population variance by law of total variance.
    second = sum((float(s.get("std", 0.0)) ** 2 + float(s.get("mean", 0.0)) ** 2) * int(s.get("count", 0) or 0) for s in stats) / total
    var = max(0.0, second - mean * mean)
    return {
        "count": int(total),
        "min": min(float(s.get("min", 0.0)) for s in stats),
        "max": max(float(s.get("max", 0.0)) for s in stats),
        "mean": float(mean),
        "std": float(var ** 0.5),
        "num_graphs": len(stats),
    }

def compact_metrics_result(result: dict) -> dict:
    """Return a lightweight metrics JSON without raw per-node arrays.

    Full metrics_test.json keeps scores/labels/predictions/graphs for detailed
    offline analysis.  On full CADETS this can be large, so we also emit a
    compact companion that is easy to inspect and send around.
    """
    graph_summaries = []
    for g in result.get("graphs", []) or []:
        graph_summaries.append({
            "graph_index": g.get("graph_index"),
            "node_scope": g.get("node_scope"),
            "num_target_nodes": g.get("num_target_nodes"),
            "positive_target_nodes": g.get("positive_target_nodes"),
            "num_process_nodes": g.get("num_process_nodes"),
            "positive_process_nodes": g.get("positive_process_nodes"),
            "graph_label": g.get("graph_label"),
            "top50_precision": g.get("top50_precision"),
            "edge_weight_stats": g.get("edge_weight_stats"),
            "gate_stats": g.get("gate_stats"),
            "attention_stats": g.get("attention_stats"),
        })
    return {
        "node_scope": result.get("node_scope"),
        "metrics": result.get("metrics", {}),
        "all_positive_baseline": result.get("all_positive_baseline", {}),
        "all_negative_baseline": result.get("all_negative_baseline", {}),
        "threshold_diagnostics": result.get("threshold_diagnostics"),
        "score_distribution": result.get("score_distribution", {}),
        "ranking_metrics": result.get("ranking_metrics", {}),
        "edge_weight_distribution": result.get("edge_weight_distribution"),
        "gate_distribution": result.get("gate_distribution"),
        # Alias kept for v2 edge-gated experiments: older reports look for gate_distribution,
        # while analysts naturally search for edge_gate_distribution.  They refer to
        # the same aggregate when gate_mode=edge_vector/edge_scalar.
        "edge_gate_distribution": result.get("gate_distribution"),
        "attention_distribution": result.get("attention_distribution"),
        "evaluation_cuda_memory": result.get("evaluation_cuda_memory"),
        "warnings": result.get("warnings", []),
        "graph_subset": result.get("graph_subset", {}),
        "graph_summaries": graph_summaries,
    }


def _normalize_alert_key(row: dict) -> str:
    # UUID-only display names differ across repeated equivalent processes, while
    # event_token_sample usually captures the behavior sequence.  Use the token
    # sample as the primary grouping key and fall back to display/type.
    sample = (row.get("event_token_sample") or "").strip()
    if sample:
        return sample
    return f"{row.get('node_type','')}|{row.get('node_display') or row.get('node_id','')}"


def _write_alert_groups_csv(result: dict, path: str | Path, *, only_predicted: bool = False) -> None:
    """Group top alerts by repeated behavior samples for analyst triage.

    This is analysis-only: it does not affect training/evaluation.  It helps find
    cases where dozens of UUID-distinct process nodes share the same behavior
    pattern, a common provenance-analysis pain point.  When only_predicted=True,
    rows below the selected operating threshold are ignored, producing a concise
    list of groups that actually become alerts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = float((result.get("metrics") or {}).get("threshold", 0.5))
    groups: dict[str, dict] = {}
    for g in result.get("graphs", []) or []:
        for row in g.get("top_alerts", []) or []:
            score = float(row.get("score", 0.0) or 0.0)
            predicted = score >= threshold
            if only_predicted and not predicted:
                continue
            key = _normalize_alert_key(row)
            gid = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
            item = groups.setdefault(gid, {
                "group_id": gid,
                "count": 0,
                "positive_count": 0,
                "above_threshold_count": 0,
                "above_threshold_positive_count": 0,
                "max_score": float("-inf"),
                "score_sum": 0.0,
                "graphs": set(),
                "example_node_id": row.get("node_id", ""),
                "example_node_display": row.get("node_display", ""),
                "example_node_type": row.get("node_type", ""),
                "event_token_sample": row.get("event_token_sample", ""),
            })
            label = int(row.get("label", 0) or 0)
            item["count"] += 1
            item["positive_count"] += label
            item["above_threshold_count"] += int(predicted)
            item["above_threshold_positive_count"] += int(predicted) * label
            item["max_score"] = max(item["max_score"], score)
            item["score_sum"] += score
            item["graphs"].add(str(g.get("graph_index")))
    rows = []
    for item in groups.values():
        count = max(int(item["count"]), 1)
        alert_count = max(int(item["above_threshold_count"]), 1)
        rows.append({
            "group_id": item["group_id"],
            "count": item["count"],
            "positive_count": item["positive_count"],
            "precision": item["positive_count"] / count,
            "above_threshold_count": item["above_threshold_count"],
            "above_threshold_positive_count": item["above_threshold_positive_count"],
            "above_threshold_precision": (item["above_threshold_positive_count"] / alert_count) if item["above_threshold_count"] else "",
            "threshold": threshold,
            "max_score": item["max_score"],
            "mean_score": item["score_sum"] / count,
            "num_graphs": len(item["graphs"]),
            "graph_indices": ";".join(sorted(item["graphs"], key=lambda x: int(x) if x and x.lstrip('-').isdigit() else x)[:50]),
            "example_node_id": item["example_node_id"],
            "example_node_display": item["example_node_display"],
            "example_node_type": item["example_node_type"],
            "event_token_sample": item["event_token_sample"],
        })
    rows.sort(key=lambda r: (-int(r["above_threshold_count"] or 0), -int(r["count"] or 0), -float(r["max_score"] or 0.0)))
    fields = ["group_id", "count", "positive_count", "precision", "above_threshold_count", "above_threshold_positive_count", "above_threshold_precision", "threshold", "max_score", "mean_score", "num_graphs", "graph_indices", "example_node_id", "example_node_display", "example_node_type", "event_token_sample"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def evaluate_checkpoint(
    cfg: Config,
    checkpoint: str | Path,
    split: str = "test",
    device_str: str = "cpu",
    out_path: str | Path | None = None,
    threshold: float | None = None,
) -> dict:
    device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
    vocab = load_vocab(cfg.processed_dir)
    model = MalSnifModel(vocab.embeddings, cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    threshold_source = "override" if threshold is not None else "config"
    if threshold is None and isinstance(state, dict) and "threshold" in state:
        try:
            threshold = float(state["threshold"])
            threshold_source = "checkpoint_val"
        except Exception:
            threshold = None
    limit = None
    if split == "train":
        limit = getattr(cfg, "graph_limit_train", None)
    elif split == "val":
        limit = getattr(cfg, "graph_limit_val", None)
    elif split == "test":
        limit = getattr(cfg, "graph_limit_test", None)
    ds = ProcessedGraphs(
        cfg.processed_dir, split=split, metadata_dir=getattr(cfg, "metadata_dir", None),
        limit=limit, cache_in_memory=bool(getattr(cfg, "cache_graphs_in_memory", False)),
    )
    total_split_graphs = len(ds.meta.get("split", {}).get(split, []))
    node_scope = None
    if isinstance(state, dict):
        node_scope = state.get("resolved_node_scope") or cfg.node_scope
        if node_scope:
            cfg.node_scope = str(node_scope)
    reset_cuda_peak(device)
    result = evaluate_model(model, ds, cfg, device, threshold=threshold, threshold_source=threshold_source, node_scope=node_scope)
    mem_stats = cuda_memory_stats(device)
    if mem_stats:
        result["evaluation_cuda_memory"] = mem_stats
    result["graph_subset"] = {
        "split": split,
        "graph_limit": int(limit) if limit is not None else None,
        "graphs_used": len(ds),
        "total_split_graphs": total_split_graphs,
        "uses_chronological_prefix": bool(limit is not None and int(limit) > 0 and len(ds) < total_split_graphs),
    }
    if out_path:
        out_path = Path(out_path)
        save_json(result, out_path)
        save_json(compact_metrics_result(result), out_path.parent / f"metrics_{split}_compact.json")
        plot_score_histogram(result.get("labels", []), result.get("scores", []), out_path.parent / "plots" / f"scores_{split}.png", threshold=result.get("metrics", {}).get("threshold"), mode=getattr(cfg, "plot_mode", "essential"))
        _write_top_alerts_csv(result, out_path.parent / f"top_alerts_{split}.csv")
        _write_alert_groups_csv(result, out_path.parent / f"top_alert_groups_{split}.csv")
        _write_alert_groups_csv(result, out_path.parent / f"predicted_alert_groups_{split}.csv", only_predicted=True)
    return result


def _write_top_alerts_csv(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["graph_index", "rank", "node_index", "node_id", "node_display", "node_type", "event_count", "labeled_event_count", "score", "label", "event_token_sample"])
        w.writeheader()
        for g in result.get("graphs", []):
            for rank, row in enumerate(g.get("top_alerts", []), 1):
                out = dict(row)
                out["graph_index"] = g.get("graph_index")
                out["rank"] = rank
                w.writerow(out)
