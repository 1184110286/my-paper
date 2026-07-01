from malsnif.data.redundancy import (
    reduce_event_sequence,
    redundancy_reduction_prefix_tree,
    redundancy_reduction_risk_time_prefix_tree,
)


def test_paper_example():
    seq = list("ABCDEDECDEKEHI")
    assert "".join(redundancy_reduction_prefix_tree(seq)) == "ABCDEKEHI"


def test_no_duplicate():
    assert redundancy_reduction_prefix_tree(list("ABCDE")) == list("ABCDE")


def test_deep_prefix_tree_does_not_hit_python_recursion_limit():
    # The first and last A create a repeated-loop region whose prefix tree is a
    # single long chain.  The old recursive preorder hit RecursionError here on
    # real DARPA CDM process sequences.
    seq = ["A"] + [f"x{i}" for i in range(1500)] + ["A", "tail"]
    reduced = redundancy_reduction_prefix_tree(seq)
    assert reduced
    assert reduced[0] == "A"


def test_risk_time_mode_preserves_extra_high_risk_repeat():
    benign = ("event_read", ".txt", "etc", "hosts")
    risk = ("event_write", ".exe", "tmp", "evil.exe")
    other = ("event_open", "<nosuffix>", "lib")
    tail = ("event_close", "<nosuffix>", "x")
    seq = [benign, risk, other, benign, risk, other, tail]

    base = redundancy_reduction_prefix_tree(seq)
    reduced = redundancy_reduction_risk_time_prefix_tree(seq)

    assert base.count(risk) == 1
    assert reduced.count(risk) == 2
    assert len(reduced) < len(seq)


def test_risk_time_mode_adds_repeat_summary_token():
    benign = ("event_read", ".txt", "etc", "hosts")
    risk = ("event_write", ".exe", "tmp", "evil.exe")
    tail = ("event_close", "<nosuffix>", "x")
    seq = [benign, risk, benign, risk, benign, risk, tail]

    reduced = redundancy_reduction_risk_time_prefix_tree(seq, repeat_min=3)

    summaries = [x for x in reduced if x and x[0] == "<rep>"]
    assert summaries
    assert any(x[1] == "event_write" for x in summaries)


def test_risk_time_mode_respects_budget():
    risk = ("event_write", ".exe", "tmp", "evil.exe")
    seq = [("event_read", ".txt", "etc", f"h{i}") for i in range(20)] + [risk] * 8

    reduced = reduce_event_sequence(seq, mode="risk_time_prefix_tree", max_events=6)

    assert len(reduced) <= 6
    assert risk in reduced
