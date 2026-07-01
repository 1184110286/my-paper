from malsnif.config import Config
from malsnif.data.build_graph import GraphBuilder
from malsnif.data.events import EventRecord


def test_graph_stats_contains_label_and_time_diagnostics():
    cfg = Config(reduce_sequences=False, simplify_graph=True)
    events = [
        EventRecord("process:p1", "PROCESS", "file:f1", "FILE", "EVENT_WRITE", 10, 0, {"dst_display":"/tmp/a.txt"}),
        EventRecord("process:p1", "PROCESS", "file:f2", "FILE", "EVENT_WRITE", 20, 1, {"dst_display":"/tmp/b.txt"}),
        EventRecord("file:f3", "FILE", "process:p1", "PROCESS", "EVENT_READ", 30, 1, {"dst_display":"/bin/sh"}),
    ]
    g = GraphBuilder(cfg).build_tokens_graph(events)
    st = g["stats"]
    assert st["original_events"] == 3
    assert st["original_positive_events"] == 2
    assert st["first_event_time_ns"] == 10
    assert st["last_event_time_ns"] == 30
    assert st["graph_label"] == 1
    assert st["simplified_positive_process_nodes"] >= 1
    assert st["redundancy_mode"] == "off"
    assert st["node_events_before_reduction"] == st["node_events_after_reduction"]


def test_graph_stats_contains_redundancy_diagnostics():
    cfg = Config(
        reduce_sequences=True,
        redundancy_mode="risk_time_prefix_tree",
        simplify_graph=False,
        max_events_per_node=8,
    )
    events = [
        EventRecord("process:p1", "PROCESS", "file:f1", "FILE", "EVENT_WRITE", 10, 0, {"dst_display":"/tmp/evil.exe"}),
        EventRecord("process:p1", "PROCESS", "file:f2", "FILE", "EVENT_READ", 20, 0, {"dst_display":"/etc/hosts"}),
        EventRecord("process:p1", "PROCESS", "file:f1", "FILE", "EVENT_WRITE", 30, 0, {"dst_display":"/tmp/evil.exe"}),
        EventRecord("process:p1", "PROCESS", "file:f2", "FILE", "EVENT_READ", 40, 0, {"dst_display":"/etc/hosts"}),
    ]
    g = GraphBuilder(cfg).build_tokens_graph(events)
    st = g["stats"]
    assert st["redundancy_mode"] == "risk_time_prefix_tree"
    assert st["node_events_before_reduction"] >= st["node_events_after_reduction"]
    assert "node_event_reduction_ratio" in st


def test_default_leaf_graph_simplification_matches_malsnif_condensation():
    cfg = Config(reduce_sequences=False, simplify_graph=True)
    events = [
        EventRecord("process:p1", "PROCESS", "file:f1", "FILE", "EVENT_WRITE", 10, 0, {"dst_display": "/etc/hosts"}),
    ]

    g = GraphBuilder(cfg).build_tokens_graph(events)
    st = g["stats"]
    gs = st["graph_simplification"]

    assert st["original_nodes"] == 2
    assert st["simplified_nodes"] == 1
    assert st["simplified_edges"] == 0
    assert gs["graph_simplify_mode"] == "leaf"
    assert gs["graph_simplify_candidates"] == 1
    assert gs["graph_simplify_removed_nodes"] == 1
    assert gs["graph_simplify_kept_risky_nodes"] == 0


def test_risk_aware_graph_simplification_keeps_high_risk_singleton_evidence():
    cfg = Config(
        reduce_sequences=False,
        simplify_graph=True,
        graph_simplify_mode="risk_aware",
        graph_simplify_risk_threshold=0.62,
    )
    events = [
        EventRecord("process:p1", "PROCESS", "file:f1", "FILE", "EVENT_EXECUTE", 10, 0, {"dst_display": "/tmp/payload.sh"}),
        EventRecord("process:p1", "PROCESS", "file:f1", "FILE", "EVENT_EXECUTE", 20, 0, {"dst_display": "/tmp/payload.sh"}),
    ]

    g = GraphBuilder(cfg).build_tokens_graph(events)
    st = g["stats"]
    gs = st["graph_simplification"]

    assert st["original_nodes"] == 2
    assert st["simplified_nodes"] == 2
    assert st["simplified_edges"] == 1
    assert gs["graph_simplify_mode"] == "risk_aware"
    assert gs["graph_simplify_candidates"] == 1
    assert gs["graph_simplify_removed_nodes"] == 0
    assert gs["graph_simplify_kept_risky_nodes"] == 1
    assert gs["graph_simplify_max_risk"] >= cfg.graph_simplify_risk_threshold
