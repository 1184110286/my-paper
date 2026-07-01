from __future__ import annotations

from typing import Iterable
import numpy as np
import torch


def node_mask_numpy(graph: dict, scope: str = "process") -> np.ndarray:
    """Return the node mask used for supervised loss/evaluation.

    scope="process" matches MalSnif's paper-level process-node prediction.
    scope="all" is available as a diagnostic fallback for raw CDM UUID labels
    when a short prefix contains no labeled process nodes but does contain
    labeled files/sockets.
    """
    labels = np.asarray(graph.get("node_labels", []), dtype=int)
    n = int(labels.size)
    scope = str(scope or "process").lower()
    if scope in {"all", "all_nodes", "node", "nodes"}:
        return np.ones(n, dtype=bool)
    pm = np.asarray(graph.get("process_mask", []), dtype=bool)
    if pm.size != n:
        return np.ones(n, dtype=bool) if scope in {"all", "auto"} else np.zeros(n, dtype=bool)
    return pm


def node_mask_torch(graph: dict, scope: str, device) -> torch.Tensor:
    return torch.tensor(node_mask_numpy(graph, scope), dtype=torch.bool, device=device)


def scope_label_counts(graphs: Iterable[dict], scope: str) -> dict:
    total = pos = neg = 0
    graph_count = 0
    for g in graphs:
        graph_count += 1
        labels = np.asarray(g.get("node_labels", []), dtype=int)
        mask = node_mask_numpy(g, scope)
        if mask.size != labels.size:
            continue
        y = labels[mask]
        total += int(y.size)
        pos += int((y == 1).sum())
        neg += int((y == 0).sum())
    return {
        "scope": scope,
        "num_graphs": graph_count,
        "total": total,
        "positives": pos,
        "negatives": neg,
        "positive_ratio": pos / max(total, 1),
        "has_both_classes": bool(pos > 0 and neg > 0),
    }


def resolve_node_scope(graphs: Iterable[dict], requested: str = "auto") -> str:
    requested = str(requested or "process").lower()
    graph_list = list(graphs)
    if requested in {"process", "all"}:
        return requested
    if requested not in {"auto", "auto_process"}:
        return "process"
    process_counts = scope_label_counts(graph_list, "process")
    if process_counts.get("has_both_classes"):
        return "process"
    all_counts = scope_label_counts(graph_list, "all")
    if all_counts.get("has_both_classes"):
        return "all"
    return "process"
