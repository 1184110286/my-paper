from malsnif.data.redundancy import (
    redundancy_reduction_winnowing_anchor,
    reduce_event_sequence,
    reduce_event_sequence_with_weights,
)


def test_wra_anchor_preserves_short_sequences():
    seq = [("op", ".txt", str(i)) for i in range(5)]
    reduced, weights = redundancy_reduction_winnowing_anchor(seq, window=11)
    assert reduced == seq
    assert weights == [1.0] * len(seq)


def test_wra_anchor_reduces_long_sequence_and_keeps_boundaries():
    seq = [("op", ".txt", str(i)) for i in range(100)]
    reduced, weights = redundancy_reduction_winnowing_anchor(seq, window=11)
    assert len(reduced) < len(seq)
    assert reduced[0] == seq[0]
    assert reduced[-1] == seq[-1]
    assert weights == [1.0] * len(reduced)


def test_wra_anchor_is_deterministic():
    seq = [("write", ".txt", str(i % 7)) for i in range(80)]
    r1, w1 = redundancy_reduction_winnowing_anchor(seq, window=11)
    r2, w2 = redundancy_reduction_winnowing_anchor(seq, window=11)
    assert r1 == r2
    assert w1 == w2


def test_wra_anchor_public_aliases_integrate():
    seq = [("read", ".txt", str(i)) for i in range(80)]
    reduced = reduce_event_sequence(seq, mode="wra_rr", wra_rr_window=11)
    reduced2, weights = reduce_event_sequence_with_weights(seq, mode="winnowing_anchor", wra_rr_window=11)
    assert reduced == reduced2
    assert len(weights) == len(reduced2)
    assert all(w == 1.0 for w in weights)
