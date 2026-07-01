from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from typing import Any

from malsnif.constants import NODE_TYPE_TO_ID
from malsnif.config import Config
from malsnif.data.events import EventRecord
from malsnif.data.graph_simplify import empty_graph_simplify_stats, simplify_graph_tokens
from malsnif.data.redundancy import reduce_event_sequence_with_weights
from malsnif.data.sanitize import event_to_tokens
from malsnif.data.vocab import Vocabulary


@dataclass
class BuildStats:
    original_nodes: int
    original_edges: int
    simplified_nodes: int
    simplified_edges: int
    node_reduction_ratio: float
    edge_reduction_ratio: float
    original_events: int
    labeled_events: int
    original_positive_events: int
    original_positive_nodes: int
    original_positive_process_nodes: int
    simplified_positive_nodes: int
    simplified_positive_process_nodes: int
    simplified_process_nodes: int
    simplified_node_events: int
    node_events_before_reduction: int
    node_events_after_reduction: int
    node_event_reduction_ratio: float
    redundancy_mode: str
    graph_label: int
    process_label_projection: dict[str, Any] | None = None
    first_event_time_ns: int | None = None
    last_event_time_ns: int | None = None
    event_type_counts: dict[str, int] | None = None
    node_label_policy: str = "process_event_endpoints"
    graph_simplification: dict[str, Any] | None = None
    node_event_weight_mean: float = 1.0
    node_event_weight_max: float = 1.0
    node_event_high_weight_ratio: float = 0.0


def _is_process_type(t: str) -> bool:
    return str(t).upper() == "PROCESS"


def _better_display(candidate: str | None, current: str | None) -> str | None:
    """Prefer human-readable CDM display strings over UUID-only ids.

    The structural node id intentionally remains UUID based, but analysis files
    and top-alert CSVs should expose process command/path/socket context when
    it is available.  This helper is used only for diagnostics; it never changes
    graph structure or model inputs.
    """
    cand = str(candidate or "").strip()
    cur = str(current or "").strip()
    if not cand:
        return cur or None
    def score(x: str) -> int:
        lx = x.lower()
        sc = 0
        if ":" in x or "/" in x or "\\" in x or " " in x or "." in x:
            sc += 2
        if "<guid>" in lx or "subject:" in lx or "file:" in lx or "object:" in lx or "event:" in lx:
            sc -= 4
        if len(x) > 12:
            sc += 1
        return sc
    return cand if score(cand) > score(cur) else (cur or cand)


def _event_endpoint_label_flags(ev: EventRecord, policy: str) -> tuple[bool, bool]:
    """Return whether the source/destination graph endpoint should be labeled.

    `process_event_endpoints` is the default for MalSnif-style process-node
    detection: a positive CDM event labels only the process endpoint(s) involved
    in that event. This keeps supervision on process nodes, which is what MalSnif
    evaluates, while avoiding broad propagation through singleton file/socket nodes.

    `event_endpoints` is the legacy behavior: a positive event labels both graph
    endpoints. It can inflate positives when a DARPA ground-truth file is an
    entity-UUID list.

    `matched_endpoints` labels only endpoints whose UUID actually matched the
    ground-truth UUID set. It is useful as a conservative diagnostic view, but it
    can yield zero process positives when the ground truth mostly lists file or
    socket UUIDs.
    """
    if not ev.tag:
        return False, False
    policy = (policy or "process_event_endpoints").lower()
    if policy in {"event_endpoints", "legacy", "all_event_endpoints"}:
        return True, True
    if policy in {"process_event_endpoints", "process", "process_only", "process_endpoints"}:
        return _is_process_type(ev.src_type), _is_process_type(ev.dst_type)
    match = ev.raw.get("label_match", {}) if isinstance(ev.raw, dict) else {}
    uuid_fields = set(match.get("uuid_fields", []) or []) if isinstance(match, dict) else set()
    src_hit = "src_uuid" in uuid_fields
    dst_hit = "dst_uuid" in uuid_fields
    # For event-level/path/time/event-type labels, keep the event-level fallback.
    fallback = bool(match.get("fallback_event_label", True)) if isinstance(match, dict) else True
    if fallback and not (src_hit or dst_hit):
        return True, True
    return src_hit, dst_hit


class GraphBuilder:
    def __init__(self, cfg: Config, vocab: Vocabulary | None = None):
        self.cfg = cfg
        self.vocab = vocab

    def _node_idx(self, node_map: dict[str, int], node_ids: list[str], node_types: list[str], node_displays: list[str], node_id: str, node_type: str, display: str | None = None) -> int:
        if node_id not in node_map:
            node_map[node_id] = len(node_ids)
            node_ids.append(node_id)
            node_types.append(node_type.upper())
            node_displays.append(str(display or node_id))
        else:
            i = node_map[node_id]
            node_displays[i] = _better_display(display, node_displays[i]) or node_displays[i]
        return node_map[node_id]

    def build_tokens_graph(self, events: list[EventRecord]) -> dict:
        node_map: dict[str, int] = {}
        node_ids: list[str] = []
        node_types: list[str] = []
        node_displays: list[str] = []
        edge_map: dict[tuple[int, int], int] = {}
        edge_index: list[tuple[int, int]] = []
        edge_types: list[str] = []
        edge_times_ns: list[int | None] = []
        node_event_tokens: list[list[list[str]]] = []
        node_event_weights: list[list[float]] = []
        edge_event_tokens: list[list[list[str]]] = []
        edge_event_weights: list[list[float]] = []
        node_labels: list[int] = []
        node_total_event_counts: list[int] = []
        node_labeled_event_counts: list[int] = []

        def ensure_node_arrays() -> None:
            while len(node_event_tokens) < len(node_ids):
                node_event_tokens.append([])
                node_event_weights.append([])
                node_labels.append(0)
                node_total_event_counts.append(0)
                node_labeled_event_counts.append(0)

        for ev in events:
            src_display = ev.raw.get("src_display") if isinstance(ev.raw, dict) else ev.src_id
            dst_display = ev.raw.get("dst_display") if isinstance(ev.raw, dict) else ev.dst_id
            s = self._node_idx(node_map, node_ids, node_types, node_displays, ev.src_id, ev.src_type, src_display)
            d = self._node_idx(node_map, node_ids, node_types, node_displays, ev.dst_id, ev.dst_type, dst_display)
            ensure_node_arrays()
            edge_key = (s, d)
            if edge_key not in edge_map:
                edge_map[edge_key] = len(edge_index)
                edge_index.append(edge_key)
                edge_types.append(ev.edge_type)
                edge_times_ns.append(None)
                edge_event_tokens.append([])
                edge_event_weights.append([])
            eidx = edge_map[edge_key]
            if ev.time is not None:
                try:
                    t_ns = int(float(ev.time))
                    if edge_times_ns[eidx] is None or t_ns < int(edge_times_ns[eidx]):
                        edge_times_ns[eidx] = t_ns
                except Exception:
                    pass
            dst_for_tokens = ev.dst_id
            if ev.raw and isinstance(ev.raw, dict):
                # For read-like events, cdm.py may reverse the structural edge
                # to information-flow direction.  The semantic token should still
                # represent the event object/path rather than the receiving
                # process; cdm.py stores predicate_path/dst_display accordingly.
                dst_for_tokens = ev.raw.get("semantic_object_display") or ev.raw.get("predicate_path") or ev.raw.get("dst_display") or ev.dst_id
            toks = event_to_tokens(ev.edge_type, dst_for_tokens, lowercase=self.cfg.lowercase_tokens, max_tokens=self.cfg.max_tokens_per_event)
            edge_event_tokens[eidx].append(toks)
            edge_event_weights[eidx].append(1.0)
            if _is_process_type(node_types[s]):
                node_event_tokens[s].append(toks)
                node_event_weights[s].append(1.0)
                node_total_event_counts[s] += 1
                if ev.tag:
                    node_labeled_event_counts[s] += 1
            if _is_process_type(node_types[d]):
                node_event_tokens[d].append(toks)
                node_event_weights[d].append(1.0)
                node_total_event_counts[d] += 1
                if ev.tag:
                    node_labeled_event_counts[d] += 1

            policy = str(getattr(self.cfg, "node_label_policy", "process_event_endpoints") or "process_event_endpoints").lower()
            src_label, dst_label = _event_endpoint_label_flags(ev, policy)
            if policy in {"process_event_endpoints", "process", "process_only"}:
                # MalSnif maps process nodes to attack probabilities. CDM labels
                # may point to files/sockets/events; for supervised process-node
                # training, attach a positive event only to the process endpoint(s)
                # directly participating in that event.
                src_label = bool(ev.tag) and _is_process_type(node_types[s])
                dst_label = bool(ev.tag) and _is_process_type(node_types[d])
            if src_label:
                node_labels[s] = max(node_labels[s], 1)
            if dst_label:
                node_labels[d] = max(node_labels[d], 1)

        projection_diag = self._apply_process_label_projection(node_types, node_labels, node_labeled_event_counts)

        original_nodes = len(node_ids)
        original_edges = len(edge_index)
        labeled_events = int(sum(1 for ev in events if ev.tag))
        time_values: list[int] = []
        for ev in events:
            if ev.time is not None:
                try:
                    time_values.append(int(ev.time))
                except Exception:
                    pass
        original_positive_nodes = int(sum(1 for y in node_labels if y))
        original_positive_process_nodes = int(sum(1 for y, t in zip(node_labels, node_types) if y and _is_process_type(t)))
        event_type_counts = dict(Counter(str(ev.edge_type) for ev in events).most_common(20))

        node_events_before_reduction = int(sum(len(x) for x in node_event_tokens))
        node_events_after_reduction = node_events_before_reduction
        redundancy_mode = "off"
        if self.cfg.reduce_sequences:
            redundancy_mode = str(getattr(self.cfg, "redundancy_mode", "prefix_tree") or "prefix_tree")
            for i, seq in enumerate(node_event_tokens):
                if len(seq) > 2:
                    ids = [tuple(t) for t in seq]
                    max_events = self.cfg.max_events_per_node if redundancy_mode.lower() in {"risk_time_prefix_tree", "risk_time", "rt_prefix_tree", "rtprr"} else None
                    reduced_ids, reduced_weights = reduce_event_sequence_with_weights(
                        ids,
                        mode=redundancy_mode,
                        max_events=max_events,
                        risk_threshold=float(getattr(self.cfg, "redundancy_risk_threshold", 2.5) or 2.5),
                        preserve_risk_events=int(getattr(self.cfg, "redundancy_preserve_risk_events", 1) or 0),
                        repeat_summary=bool(getattr(self.cfg, "redundancy_repeat_summary", True)),
                        repeat_min=int(getattr(self.cfg, "redundancy_repeat_min", 3) or 3),
                        mw_prr_alpha=float(getattr(self.cfg, "mw_prr_alpha", 0.2) or 0.2),
                        btr_rr_max_block_len=int(getattr(self.cfg, "btr_rr_max_block_len", 16) or 16),
                        btr_rr_min_gain=int(getattr(self.cfg, "btr_rr_min_gain", 2) or 2),
                        btr_rr_repeat_cap=int(getattr(self.cfg, "btr_rr_repeat_cap", 32) or 32),
                        btr_rr_alpha=float(getattr(self.cfg, "btr_rr_alpha", 0.3) or 0.3),
                        lz_sr_min_phrase_len=int(getattr(self.cfg, "lz_sr_min_phrase_len", 4) or 4),
                        lz_sr_max_phrase_len=int(getattr(self.cfg, "lz_sr_max_phrase_len", 24) or 24),
                        lz_sr_window=int(getattr(self.cfg, "lz_sr_window", 512) or 512),
                        lz_sr_min_gain=int(getattr(self.cfg, "lz_sr_min_gain", 2) or 2),
                        lz_sr_alpha=float(getattr(self.cfg, "lz_sr_alpha", 0.25) or 0.25),
                        frc_rr_cap_size=int(getattr(self.cfg, "frc_rr_cap_size", 3) or 3),
                        frc_rr_repeat_cap=int(getattr(self.cfg, "frc_rr_repeat_cap", 32) or 32),
                        frc_rr_alpha=float(getattr(self.cfg, "frc_rr_alpha", 0.25) or 0.25),
                        flb_rr_repeat_cap=int(getattr(self.cfg, "flb_rr_repeat_cap", 32) or 32),
                        flb_rr_alpha=float(getattr(self.cfg, "flb_rr_alpha", 0.25) or 0.25),
                        wra_rr_window=int(getattr(self.cfg, "wra_rr_window", 11) or 11),
                        tbb_rr_target_compression=float(getattr(self.cfg, "tbb_rr_target_compression", 0.90) or 0.90),
                    )
                    node_event_tokens[i] = [list(t) for t in reduced_ids]
                    node_event_weights[i] = [float(w) for w in reduced_weights]
            node_events_after_reduction = int(sum(len(x) for x in node_event_tokens))

        graph_simplification = empty_graph_simplify_stats("off")
        if self.cfg.simplify_graph:
            simplified = simplify_graph_tokens(
                cfg=self.cfg,
                node_ids=node_ids,
                node_types=node_types,
                node_displays=node_displays,
                node_labels=node_labels,
                node_event_tokens=node_event_tokens,
                node_event_weights=node_event_weights,
                node_total_event_counts=node_total_event_counts,
                node_labeled_event_counts=node_labeled_event_counts,
                edge_index=edge_index,
                edge_types=edge_types,
                edge_event_tokens=edge_event_tokens,
                edge_event_weights=edge_event_weights,
                edge_times_ns=edge_times_ns,
            )
            node_ids = simplified.node_ids
            node_types = simplified.node_types
            node_displays = simplified.node_displays
            node_labels = simplified.node_labels
            node_event_tokens = simplified.node_event_tokens
            node_event_weights = simplified.node_event_weights or [[1.0 for _ in seq] for seq in node_event_tokens]
            node_total_event_counts = simplified.node_total_event_counts
            node_labeled_event_counts = simplified.node_labeled_event_counts
            edge_index = simplified.edge_index
            edge_types = simplified.edge_types
            edge_event_tokens = simplified.edge_event_tokens
            edge_event_weights = simplified.edge_event_weights or [[1.0 for _ in seq] for seq in edge_event_tokens]
            edge_times_ns = simplified.edge_times_ns
            graph_simplification = simplified.stats

        if self.cfg.max_nodes_per_graph is not None and len(node_ids) > self.cfg.max_nodes_per_graph:
            keep = set(range(self.cfg.max_nodes_per_graph))
            old_to_new = {i: i for i in keep}
            edge_keep = [i for i, (s, d) in enumerate(edge_index) if s in keep and d in keep]
            edge_index = [(old_to_new[s], old_to_new[d]) for i, (s, d) in enumerate(edge_index) if i in edge_keep]
            edge_types = [edge_types[i] for i in edge_keep]
            edge_event_tokens = [edge_event_tokens[i] for i in edge_keep]
            edge_event_weights = [edge_event_weights[i] for i in edge_keep]
            edge_times_ns = [edge_times_ns[i] for i in edge_keep]
            node_ids = node_ids[: self.cfg.max_nodes_per_graph]
            node_types = node_types[: self.cfg.max_nodes_per_graph]
            node_displays = node_displays[: self.cfg.max_nodes_per_graph]
            node_labels = node_labels[: self.cfg.max_nodes_per_graph]
            node_event_tokens = node_event_tokens[: self.cfg.max_nodes_per_graph]
            node_event_weights = node_event_weights[: self.cfg.max_nodes_per_graph]
            node_total_event_counts = node_total_event_counts[: self.cfg.max_nodes_per_graph]
            node_labeled_event_counts = node_labeled_event_counts[: self.cfg.max_nodes_per_graph]
        if self.cfg.max_edges_per_graph is not None and len(edge_index) > self.cfg.max_edges_per_graph:
            edge_index = edge_index[: self.cfg.max_edges_per_graph]
            edge_types = edge_types[: self.cfg.max_edges_per_graph]
            edge_event_tokens = edge_event_tokens[: self.cfg.max_edges_per_graph]
            edge_event_weights = edge_event_weights[: self.cfg.max_edges_per_graph]
            edge_times_ns = edge_times_ns[: self.cfg.max_edges_per_graph]

        simplified_nodes = len(node_ids)
        simplified_edges = len(edge_index)
        simplified_positive_nodes = int(sum(1 for y in node_labels if y))
        simplified_positive_process_nodes = int(sum(1 for y, t in zip(node_labels, node_types) if y and _is_process_type(t)))
        simplified_process_nodes = int(sum(1 for t in node_types if _is_process_type(t)))
        flat_event_weights = [float(w) for seq in node_event_weights for w in seq]
        node_event_weight_mean = sum(flat_event_weights) / max(len(flat_event_weights), 1)
        node_event_weight_max = max(flat_event_weights) if flat_event_weights else 1.0
        node_event_high_weight_ratio = sum(1 for w in flat_event_weights if float(w) > 1.000001) / max(len(flat_event_weights), 1)
        stats = BuildStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            simplified_nodes=simplified_nodes,
            simplified_edges=simplified_edges,
            node_reduction_ratio=(original_nodes - simplified_nodes) / max(original_nodes, 1),
            edge_reduction_ratio=(original_edges - simplified_edges) / max(original_edges, 1),
            original_events=len(events),
            labeled_events=labeled_events,
            original_positive_events=labeled_events,
            original_positive_nodes=original_positive_nodes,
            original_positive_process_nodes=original_positive_process_nodes,
            simplified_positive_nodes=simplified_positive_nodes,
            simplified_positive_process_nodes=simplified_positive_process_nodes,
            simplified_process_nodes=simplified_process_nodes,
            simplified_node_events=sum(len(x) for x in node_event_tokens),
            node_events_before_reduction=node_events_before_reduction,
            node_events_after_reduction=node_events_after_reduction,
            node_event_reduction_ratio=(node_events_before_reduction - node_events_after_reduction) / max(node_events_before_reduction, 1),
            redundancy_mode=redundancy_mode,
            graph_label=int(any(node_labels)),
            process_label_projection=projection_diag,
            first_event_time_ns=min(time_values) if time_values else None,
            last_event_time_ns=max(time_values) if time_values else None,
            event_type_counts=event_type_counts,
            node_label_policy=getattr(self.cfg, "node_label_policy", "process_event_endpoints"),
            graph_simplification=graph_simplification,
            node_event_weight_mean=node_event_weight_mean,
            node_event_weight_max=node_event_weight_max,
            node_event_high_weight_ratio=node_event_high_weight_ratio,
        )
        return {
            "node_ids": node_ids,
            "node_types": node_types,
            "node_displays": node_displays,
            "node_type_ids": [NODE_TYPE_TO_ID.get(t.upper(), NODE_TYPE_TO_ID["UNKNOWN"]) for t in node_types],
            "node_event_tokens": node_event_tokens,
            "node_event_weights": node_event_weights,
            "edge_index": edge_index,
            "edge_types": edge_types,
            "edge_times_ns": edge_times_ns,
            "edge_event_tokens": edge_event_tokens,
            "edge_event_weights": edge_event_weights,
            "node_labels": node_labels,
            "process_mask": [_is_process_type(t) for t in node_types],
            "node_total_event_counts": node_total_event_counts,
            "node_labeled_event_counts": node_labeled_event_counts,
            "graph_label": int(any(node_labels)),
            "stats": asdict(stats),
        }

    def _apply_process_label_projection(self, node_types, node_labels, node_labeled_event_counts) -> dict[str, Any]:
        """Optionally recover or shrink process labels from positive CDM events.

        CDM ground truth may be represented as UUID lists at different
        granularities.  Marking a process positive after *any* labeled event is
        useful for quick recall checks, but on CADETS it can label more than 90%
        of processes in a window as positive.  In adaptive mode we therefore keep
        healthy two-class process labels as-is, but when the positive ratio
        exceeds process_label_max_positive_ratio we re-project labels using the
        per-process labeled-event count and choose the least aggressive threshold
        that restores a non-degenerate target.
        """
        mode = str(getattr(self.cfg, "process_label_projection", "none") or "none").lower()
        proc_indices = [i for i, t in enumerate(node_types) if _is_process_type(t)]
        before = int(sum(1 for i in proc_indices if node_labels[i]))
        diag: dict[str, Any] = {
            "mode": mode,
            "applied": False,
            "threshold": None,
            "positive_process_before": before,
            "positive_process_after": before,
            "process_nodes": len(proc_indices),
        }
        if mode in {"", "none", "off", "false"} or not proc_indices:
            return diag

        total = len(proc_indices)
        max_ratio = float(getattr(self.cfg, "process_label_max_positive_ratio", 0.75) or 1.0)
        min_pos = int(getattr(self.cfg, "process_label_min_positive_processes", 1) or 1)
        base_min = int(getattr(self.cfg, "process_label_min_events", 2) or 1)
        counts = {i: int(node_labeled_event_counts[i]) for i in proc_indices}
        before_ratio = before / max(total, 1)
        diag["positive_process_ratio_before"] = before_ratio
        diag["max_labeled_event_count"] = max(counts.values()) if counts else 0

        if mode == "adaptive" and 0 < before < total and before_ratio <= max_ratio:
            diag["reason"] = "skipped_existing_two_class_process_labels"
            return diag

        if mode in {"labeled_events", "all", "event_counts"}:
            thresholds = [base_min]
        else:
            thresholds = sorted(set([base_min, 1, 2, 3, 5, 10, 20] + [c for c in counts.values() if c > 0]))

        chosen: int | None = None
        chosen_pos: list[int] | None = None
        # Keep as many positives as possible while staying under max_ratio.
        best_under: tuple[int, list[int], float] | None = None
        for th in thresholds:
            if mode == "adaptive" and before_ratio > max_ratio:
                projected = [i for i in proc_indices if counts.get(i, 0) >= th]
            else:
                projected = [i for i in proc_indices if node_labels[i] or counts.get(i, 0) >= th]
            pos = len(projected)
            ratio = pos / max(total, 1)
            if mode != "adaptive":
                chosen, chosen_pos = th, projected
                break
            if min_pos <= pos < total and ratio <= max_ratio:
                best_under = (int(th), projected, ratio)
                break

        if mode == "adaptive" and best_under is not None:
            chosen, chosen_pos, _ = best_under
        if chosen is None or chosen_pos is None:
            diag["reason"] = "no_threshold_yielded_two_classes"
            return diag

        if mode == "adaptive" and before_ratio > max_ratio:
            for i in proc_indices:
                node_labels[i] = 0
        for i in chosen_pos:
            node_labels[i] = 1
        after = int(sum(1 for i in proc_indices if node_labels[i]))
        diag.update({
            "applied": after != before,
            "threshold": int(chosen),
            "positive_process_after": after,
            "positive_process_ratio_after": after / max(total, 1),
            "reason": "shrunk_overbroad_process_labels" if before_ratio > max_ratio else "projected_from_labeled_event_counts",
        })
        return diag


def encode_graph_tokens(graph: dict, vocab: Vocabulary, cfg: Config) -> dict:
    # Keep short human-readable event-token samples for analysis/top-alerts.
    # These diagnostics do not affect model inputs and preserve the MalSnif
    # architecture, but they make false positives/true positives auditable.
    def token_sample(nested: list[list[list[str]]], max_seq: int = 5) -> list[str]:
        rows: list[str] = []
        for seq in nested:
            parts = ["/".join(str(x) for x in toks[:6]) for toks in seq[:max_seq]]
            rows.append(" | ".join(parts))
        return rows

    if "node_event_tokens" in graph and "node_event_token_samples" not in graph:
        graph["node_event_token_samples"] = token_sample(graph.get("node_event_tokens", []))

    def enc_nested(nested: list[list[list[str]]], max_events: int) -> list[list[list[int]]]:
        out: list[list[list[int]]] = []
        for seq in nested:
            seq = seq[:max_events]
            if not seq:
                seq = [["<empty>"]]
            out.append([vocab.encode(tokens, max_len=cfg.max_tokens_per_event) for tokens in seq])
        return out

    graph = dict(graph)

    # Store compact relation/time ids for ST-HGAN.  Existing old graph caches may
    # not contain these fields; the model has deterministic fallbacks.
    if "edge_times_ns" in graph and "edge_time_buckets" not in graph:
        times = graph.get("edge_times_ns", []) or []
        finite = []
        for x in times:
            try:
                if x is not None:
                    finite.append(int(x))
            except Exception:
                pass
        if finite:
            lo, hi = min(finite), max(finite)
            span = max(hi - lo, 1)
            buckets = max(1, int(getattr(cfg, "hgan_num_time_buckets", 16) or 16))
            out_b = []
            for x in times:
                try:
                    out_b.append(max(0, min(buckets - 1, int((int(x) - lo) / span * buckets))) if x is not None else 0)
                except Exception:
                    out_b.append(0)
            graph["edge_time_buckets"] = out_b
    def enc_weights(nested_weights: list[list[float]] | None, nested_ids: list[list[list[int]]], max_events: int) -> list[list[float]]:
        out: list[list[float]] = []
        if nested_weights is None:
            nested_weights = []
        for i, seq_ids in enumerate(nested_ids):
            source = nested_weights[i] if i < len(nested_weights) else []
            vals = [float(w) for w in source[:max_events]]
            if not vals:
                vals = [1.0]
            if len(vals) < len(seq_ids):
                vals.extend([1.0] * (len(seq_ids) - len(vals)))
            out.append(vals[: len(seq_ids)])
        return out

    raw_node_weights = graph.pop("node_event_weights", None)
    raw_edge_weights = graph.pop("edge_event_weights", None)
    node_event_ids = enc_nested(graph.pop("node_event_tokens"), cfg.max_events_per_node)
    edge_event_ids = enc_nested(graph.pop("edge_event_tokens"), cfg.max_events_per_edge)
    graph["node_event_ids"] = node_event_ids
    graph["edge_event_ids"] = edge_event_ids
    graph["node_event_weights"] = enc_weights(raw_node_weights, node_event_ids, cfg.max_events_per_node)
    graph["edge_event_weights"] = enc_weights(raw_edge_weights, edge_event_ids, cfg.max_events_per_edge)
    return graph
