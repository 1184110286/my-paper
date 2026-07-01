from pathlib import Path

from malsnif.config import Config
from malsnif.data.events import EventRecord
from malsnif.data.build_graph import GraphBuilder


def _event(src_uuid_hit=False, dst_uuid_hit=False, tag=1):
    uuid_fields = []
    if src_uuid_hit:
        uuid_fields.append("src_uuid")
    if dst_uuid_hit:
        uuid_fields.append("dst_uuid")
    return EventRecord(
        src_id="process:p1",
        src_type="PROCESS",
        dst_id="file:f1",
        dst_type="FILE",
        edge_type="EVENT_WRITE",
        time=1,
        tag=tag,
        raw={"label_match": {"matched": bool(tag), "uuid_fields": uuid_fields, "fallback_event_label": False}, "predicate_path": "/tmp/x"},
    )


def test_matched_endpoint_policy_does_not_label_subject_for_object_only_match():
    cfg = Config(node_label_policy="matched_endpoints", simplify_graph=False, reduce_sequences=False)
    g = GraphBuilder(cfg).build_tokens_graph([_event(src_uuid_hit=False, dst_uuid_hit=True)])
    proc_idx = g["node_ids"].index("process:p1")
    file_idx = g["node_ids"].index("file:f1")
    assert g["node_labels"][file_idx] == 1
    assert g["node_labels"][proc_idx] == 0


def test_legacy_event_endpoint_policy_labels_subject_process():
    cfg = Config(node_label_policy="event_endpoints", simplify_graph=False, reduce_sequences=False)
    g = GraphBuilder(cfg).build_tokens_graph([_event(src_uuid_hit=False, dst_uuid_hit=True)])
    proc_idx = g["node_ids"].index("process:p1")
    assert g["node_labels"][proc_idx] == 1


def test_process_event_endpoint_policy_labels_only_process_endpoint_for_object_match():
    cfg = Config(node_label_policy="process_event_endpoints", simplify_graph=False, reduce_sequences=False)
    g = GraphBuilder(cfg).build_tokens_graph([_event(src_uuid_hit=False, dst_uuid_hit=True)])
    proc_idx = g["node_ids"].index("process:p1")
    file_idx = g["node_ids"].index("file:f1")
    assert g["node_labels"][proc_idx] == 1
    assert g["node_labels"][file_idx] == 0


def test_process_event_endpoint_policy_labels_only_process_endpoint():
    cfg = Config(node_label_policy="process_event_endpoints", simplify_graph=False, reduce_sequences=False)
    g = GraphBuilder(cfg).build_tokens_graph([_event(src_uuid_hit=False, dst_uuid_hit=True)])
    proc_idx = g["node_ids"].index("process:p1")
    file_idx = g["node_ids"].index("file:f1")
    assert g["node_labels"][proc_idx] == 1
    assert g["node_labels"][file_idx] == 0


def test_adaptive_projection_recovers_process_label_from_matched_object_event():
    cfg = Config(
        node_label_policy="matched_endpoints",
        process_label_projection="adaptive",
        simplify_graph=False,
        reduce_sequences=False,
    )
    benign = EventRecord(
        src_id="process:p2", src_type="PROCESS", dst_id="file:f2", dst_type="FILE",
        edge_type="EVENT_WRITE", time=2, tag=0, raw={"label_match": {"matched": False, "uuid_fields": []}}
    )
    g = GraphBuilder(cfg).build_tokens_graph([_event(src_uuid_hit=False, dst_uuid_hit=True), benign])
    proc_idx = g["node_ids"].index("process:p1")
    benign_idx = g["node_ids"].index("process:p2")
    assert g["node_labels"][proc_idx] == 1
    assert g["node_labels"][benign_idx] == 0
    assert g["stats"]["process_label_projection"]["applied"] is True


def test_adaptive_projection_shrinks_overbroad_process_labels():
    events = []
    # 10 processes have at least one labeled event; only 3 have >= 3 labeled events.
    for i in range(10):
        reps = 5 if i < 3 else 1
        for j in range(reps):
            events.append(EventRecord(
                src_id=f"process:p{i}", src_type="PROCESS", dst_id=f"file:f{i}-{j}", dst_type="FILE",
                edge_type="EVENT_WRITE", time=j, tag=1,
                raw={"label_match": {"matched": True, "uuid_fields": ["dst_uuid"], "fallback_event_label": False}, "predicate_path": f"/tmp/x{i}"},
            ))
    cfg = Config(
        node_label_policy="process_event_endpoints",
        process_label_projection="adaptive",
        process_label_max_positive_ratio=0.5,
        process_label_min_events=3,
        simplify_graph=False,
        reduce_sequences=False,
    )
    g = GraphBuilder(cfg).build_tokens_graph(events)
    proc_labels = [y for y, is_proc in zip(g["node_labels"], g["process_mask"]) if is_proc]
    assert sum(proc_labels) == 3
    assert g["stats"]["process_label_projection"]["applied"] is True
    assert g["stats"]["process_label_projection"]["reason"] == "shrunk_overbroad_process_labels"
