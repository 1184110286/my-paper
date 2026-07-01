from malsnif.config import Config
from malsnif.data.build_graph import GraphBuilder
from malsnif.data.events import EventRecord


def test_graph_keeps_human_readable_node_displays():
    cfg = Config(simplify_graph=False, reduce_sequences=False)
    ev = EventRecord(
        src_id="process:11111111-1111-1111-1111-111111111111",
        src_type="PROCESS",
        dst_id="file:22222222-2222-2222-2222-222222222222",
        dst_type="FILE",
        edge_type="EVENT_READ",
        tag=1,
        raw={"src_display": "/usr/bin/python /tmp/mine.py", "dst_display": "/tmp/mine.py"},
    )
    g = GraphBuilder(cfg).build_tokens_graph([ev])
    assert g["node_displays"][0] == "/usr/bin/python /tmp/mine.py"
    assert g["node_displays"][1] == "/tmp/mine.py"

