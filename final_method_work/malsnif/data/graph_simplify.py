from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class GraphSimplifyResult:
    node_ids: list[str]
    node_types: list[str]
    node_displays: list[str]
    node_labels: list[int]
    node_event_tokens: list[list[list[str]]]
    node_event_weights: list[list[float]] | None
    node_total_event_counts: list[int]
    node_labeled_event_counts: list[int]
    edge_index: list[tuple[int, int]]
    edge_types: list[str]
    edge_event_tokens: list[list[list[str]]]
    edge_event_weights: list[list[float]] | None
    edge_times_ns: list[int | None]
    stats: dict[str, Any]


_HIGH_RISK_SUFFIXES = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".sh",
    ".bash",
    ".py",
    ".pl",
    ".php",
    ".jar",
    ".class",
    ".bin",
    ".elf",
    ".ko",
    ".service",
    ".timer",
    ".socket",
    ".desktop",
    ".ps1",
    ".bat",
    ".cmd",
    ".vbs",
}

_SUSPICIOUS_TOKENS = {
    "<ip>",
    "tmp",
    "temp",
    "var",
    "dev",
    "shm",
    "proc",
    "shadow",
    "passwd",
    "ssh",
    "authorized_keys",
    "cron",
    "crontab",
    "systemd",
    "autostart",
    "bash",
    "sh",
    "python",
    "perl",
    "php",
    "wget",
    "curl",
    "nc",
    "ncat",
    "chmod",
    "chown",
    "payload",
    "implant",
    "drakon",
    "backdoor",
    "shell",
}

_EVENT_RISK_PRIORS = {
    "EVENT_EXECUTE": 1.0,
    "EVENT_FORK": 0.70,
    "EVENT_CLONE": 0.70,
    "EVENT_LOADLIBRARY": 0.90,
    "EVENT_MMAP": 0.60,
    "EVENT_WRITE": 0.80,
    "EVENT_MODIFY_FILE_ATTRIBUTES": 0.75,
    "EVENT_RENAME": 0.70,
    "EVENT_UNLINK": 0.70,
    "EVENT_CONNECT": 0.80,
    "EVENT_ACCEPT": 0.65,
    "EVENT_SENDTO": 0.65,
    "EVENT_RECVFROM": 0.55,
    "EVENT_READ": 0.20,
    "EVENT_OPEN": 0.18,
    "EVENT_CLOSE": 0.05,
}

_TYPE_RISK_PRIORS = {
    "PROCESS": 0.0,
    "FILE": 0.25,
    "REGISTRY": 0.45,
    "NETWORK": 0.50,
    "SOCKET": 0.50,
    "PIPE": 0.25,
}


def empty_graph_simplify_stats(mode: str = "off") -> dict[str, Any]:
    return {
        "graph_simplify_mode": mode,
        "graph_simplify_candidates": 0,
        "graph_simplify_removed_nodes": 0,
        "graph_simplify_kept_risky_nodes": 0,
        "graph_simplify_preserved_labeled_nodes": 0,
        "graph_simplify_condensed_events": 0,
        "graph_simplify_mean_risk": 0.0,
        "graph_simplify_max_risk": 0.0,
        "graph_simplify_risk_threshold": None,
        "graph_simplify_topk_per_process": 0,
    }


def _is_process_type(node_type: str) -> bool:
    return str(node_type).upper() == "PROCESS"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _event_risk(event_type: str | None) -> float:
    name = str(event_type or "").upper()
    if name in _EVENT_RISK_PRIORS:
        return _EVENT_RISK_PRIORS[name]
    if "EXEC" in name:
        return 1.0
    if "WRITE" in name or "MODIFY" in name or "CREATE" in name:
        return 0.75
    if "CONNECT" in name or "SOCKET" in name or "SEND" in name:
        return 0.70
    if "READ" in name or "OPEN" in name:
        return 0.20
    return 0.35 if name else 0.0


def _semantic_risk(tokens: list[list[str]]) -> float:
    score = 0.0
    flat: list[str] = []
    for event_tokens in tokens:
        flat.extend(str(t).lower() for t in event_tokens)
    for tok in flat:
        if tok in _HIGH_RISK_SUFFIXES:
            score += 0.35
        elif tok in _SUSPICIOUS_TOKENS:
            score += 0.18
        elif tok.startswith(".") and tok not in {".txt", ".log", ".conf", ".cfg", ".json", ".xml"}:
            score += 0.12
    return _clamp01(score)


def _temporal_burst_score(times: list[int], window_ns: int) -> float:
    if len(times) < 2 or window_ns <= 0:
        return 0.0
    span = max(times) - min(times)
    if span <= 0:
        return 1.0
    return _clamp01(1.0 - min(span, window_ns) / max(window_ns, 1))


def _candidate_risk(
    node_idx: int,
    *,
    node_types: list[str],
    incident_edges: list[list[int]],
    edge_types: list[str],
    edge_event_tokens: list[list[list[str]]],
    edge_times_ns: list[int | None],
    edge_event_weights: list[list[float]] | None = None,
    temporal_window_ns: int,
    repeat_norm: int,
) -> float:
    edges = incident_edges[node_idx]
    all_tokens: list[list[str]] = []
    rel_score = 0.0
    times: list[int] = []
    event_count = 0
    for ei in edges:
        rel_score = max(rel_score, _event_risk(edge_types[ei] if ei < len(edge_types) else None))
        for toks in edge_event_tokens[ei] if ei < len(edge_event_tokens) else []:
            all_tokens.append(toks)
            event_count += 1
            if toks:
                rel_score = max(rel_score, _event_risk(toks[0]))
        if ei < len(edge_times_ns) and edge_times_ns[ei] is not None:
            try:
                times.append(int(edge_times_ns[ei]))
            except Exception:
                pass

    semantic_score = _semantic_risk(all_tokens)
    type_score = _TYPE_RISK_PRIORS.get(str(node_types[node_idx]).upper(), 0.20)
    temporal_score = _temporal_burst_score(times, temporal_window_ns)
    repeat_score = math.log1p(max(event_count, 0)) / math.log1p(max(int(repeat_norm or 8), 1))
    repeat_score = _clamp01(repeat_score)

    # Conservative weighted score: keep only leaves with multiple weak signals
    # rather than every network/file leaf. This guards against benign noisy
    # network clients becoming persistent false-positive evidence.
    return _clamp01(
        0.40 * rel_score
        + 0.30 * semantic_score
        + 0.15 * type_score
        + 0.10 * temporal_score
        + 0.05 * repeat_score
    )


def simplify_graph_tokens(
    *,
    cfg: Any,
    node_ids: list[str],
    node_types: list[str],
    node_displays: list[str],
    node_labels: list[int],
    node_event_tokens: list[list[list[str]]],
    node_total_event_counts: list[int],
    node_labeled_event_counts: list[int],
    node_event_weights: list[list[float]] | None = None,
    edge_index: list[tuple[int, int]],
    edge_types: list[str],
    edge_event_tokens: list[list[list[str]]],
    edge_times_ns: list[int | None],
    edge_event_weights: list[list[float]] | None = None,
) -> GraphSimplifyResult:
    mode = str(getattr(cfg, "graph_simplify_mode", "leaf") or "leaf").lower()
    if mode in {"malsnif", "malsnif_leaf", "default", "paper"}:
        mode = "leaf"
    risk_mode = mode in {"risk_aware", "rspc", "risk_semantic", "risk_semantic_preserving"}
    threshold = float(getattr(cfg, "graph_simplify_risk_threshold", 0.62) or 0.62)
    topk_per_process = max(0, int(getattr(cfg, "graph_simplify_topk_per_process", 0) or 0))
    temporal_window_ns = int(getattr(cfg, "graph_simplify_temporal_window_ns", 1_000_000_000) or 0)
    repeat_norm = int(getattr(cfg, "graph_simplify_repeat_norm", 8) or 8)

    n = len(node_ids)
    if node_event_weights is None:
        node_event_weights = [[1.0 for _ in seq] for seq in node_event_tokens]
    if edge_event_weights is None:
        edge_event_weights = [[1.0 for _ in seq] for seq in edge_event_tokens]
    # Keep side-channel weights shape-compatible even for legacy caches.
    for i, seq in enumerate(node_event_tokens):
        if i >= len(node_event_weights):
            node_event_weights.append([1.0 for _ in seq])
        elif len(node_event_weights[i]) < len(seq):
            node_event_weights[i].extend([1.0] * (len(seq) - len(node_event_weights[i])))
        elif len(node_event_weights[i]) > len(seq):
            node_event_weights[i] = node_event_weights[i][: len(seq)]
    for i, seq in enumerate(edge_event_tokens):
        if i >= len(edge_event_weights):
            edge_event_weights.append([1.0 for _ in seq])
        elif len(edge_event_weights[i]) < len(seq):
            edge_event_weights[i].extend([1.0] * (len(seq) - len(edge_event_weights[i])))
        elif len(edge_event_weights[i]) > len(seq):
            edge_event_weights[i] = edge_event_weights[i][: len(seq)]

    incident_processes: list[set[int]] = [set() for _ in range(n)]
    incident_edges: list[list[int]] = [[] for _ in range(n)]
    for ei, (s, d) in enumerate(edge_index):
        incident_edges[s].append(ei)
        incident_edges[d].append(ei)
        if _is_process_type(node_types[s]):
            incident_processes[d].add(s)
        if _is_process_type(node_types[d]):
            incident_processes[s].add(d)

    preserve_labeled_objects = str(getattr(cfg, "node_label_policy", "process_event_endpoints") or "").lower().startswith("matched")
    candidates: list[tuple[int, int, float]] = []
    preserved_labeled: set[int] = set()
    for i, t in enumerate(node_types):
        if _is_process_type(t) or len(incident_processes[i]) != 1:
            continue
        if preserve_labeled_objects and node_labels[i]:
            preserved_labeled.add(i)
            continue
        p = next(iter(incident_processes[i]))
        risk = 0.0
        if risk_mode:
            risk = _candidate_risk(
                i,
                node_types=node_types,
                incident_edges=incident_edges,
                edge_types=edge_types,
                edge_event_tokens=edge_event_tokens,
                edge_times_ns=edge_times_ns,
                temporal_window_ns=temporal_window_ns,
                repeat_norm=repeat_norm,
            )
        candidates.append((i, p, risk))

    keep_risky: set[int] = set()
    if risk_mode:
        by_process: dict[int, list[tuple[float, int]]] = {}
        for i, p, risk in candidates:
            if risk >= threshold:
                keep_risky.add(i)
            by_process.setdefault(p, []).append((risk, i))
        if topk_per_process > 0:
            for rows in by_process.values():
                rows.sort(reverse=True)
                keep_risky.update(i for _, i in rows[:topk_per_process])

    remove: set[int] = set()
    condensed_events = 0
    for i, p, _risk in candidates:
        if i in keep_risky:
            continue
        for ei in incident_edges[i]:
            node_event_tokens[p].extend(edge_event_tokens[ei])
            node_event_weights[p].extend(edge_event_weights[ei] if ei < len(edge_event_weights) else [1.0] * len(edge_event_tokens[ei]))
            condensed_events += len(edge_event_tokens[ei])
            if node_labels[i]:
                node_labels[p] = max(node_labels[p], 1)
        remove.add(i)

    keep = [i for i in range(n) if i not in remove]
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_node_ids = [node_ids[i] for i in keep]
    new_node_types = [node_types[i] for i in keep]
    new_node_displays = [node_displays[i] for i in keep]
    new_node_labels = [node_labels[i] for i in keep]
    new_node_event_tokens = [node_event_tokens[i] for i in keep]
    new_node_event_weights = [node_event_weights[i] for i in keep]
    new_node_total_event_counts = [node_total_event_counts[i] for i in keep]
    new_node_labeled_event_counts = [node_labeled_event_counts[i] for i in keep]
    new_edge_index: list[tuple[int, int]] = []
    new_edge_types: list[str] = []
    new_edge_event_tokens: list[list[list[str]]] = []
    new_edge_event_weights: list[list[float]] = []
    new_edge_times_ns: list[int | None] = []
    seen_edges: dict[tuple[int, int], int] = {}
    for ei, (s, d) in enumerate(edge_index):
        if s in remove or d in remove:
            continue
        ns, nd = old_to_new[s], old_to_new[d]
        key = (ns, nd)
        if key in seen_edges:
            merged_i = seen_edges[key]
            new_edge_event_tokens[merged_i].extend(edge_event_tokens[ei])
            new_edge_event_weights[merged_i].extend(edge_event_weights[ei] if ei < len(edge_event_weights) else [1.0] * len(edge_event_tokens[ei]))
            old_t = new_edge_times_ns[merged_i]
            new_t = edge_times_ns[ei] if ei < len(edge_times_ns) else None
            if new_t is not None and (old_t is None or int(new_t) < int(old_t)):
                new_edge_times_ns[merged_i] = new_t
        else:
            seen_edges[key] = len(new_edge_index)
            new_edge_index.append(key)
            new_edge_types.append(edge_types[ei])
            new_edge_event_tokens.append(list(edge_event_tokens[ei]))
            new_edge_event_weights.append(list(edge_event_weights[ei] if ei < len(edge_event_weights) else [1.0] * len(edge_event_tokens[ei])))
            new_edge_times_ns.append(edge_times_ns[ei] if ei < len(edge_times_ns) else None)

    risks = [risk for _, _, risk in candidates]
    stats = {
        "graph_simplify_mode": mode,
        "graph_simplify_candidates": len(candidates),
        "graph_simplify_removed_nodes": len(remove),
        "graph_simplify_kept_risky_nodes": len(keep_risky),
        "graph_simplify_preserved_labeled_nodes": len(preserved_labeled),
        "graph_simplify_condensed_events": condensed_events,
        "graph_simplify_mean_risk": sum(risks) / max(len(risks), 1),
        "graph_simplify_max_risk": max(risks) if risks else 0.0,
        "graph_simplify_risk_threshold": threshold if risk_mode else None,
        "graph_simplify_topk_per_process": topk_per_process if risk_mode else 0,
    }
    return GraphSimplifyResult(
        node_ids=new_node_ids,
        node_types=new_node_types,
        node_displays=new_node_displays,
        node_labels=new_node_labels,
        node_event_tokens=new_node_event_tokens,
        node_event_weights=new_node_event_weights,
        node_total_event_counts=new_node_total_event_counts,
        node_labeled_event_counts=new_node_labeled_event_counts,
        edge_index=new_edge_index,
        edge_types=new_edge_types,
        edge_event_tokens=new_edge_event_tokens,
        edge_event_weights=new_edge_event_weights,
        edge_times_ns=new_edge_times_ns,
        stats=stats,
    )
