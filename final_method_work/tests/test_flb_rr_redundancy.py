from malsnif.data.redundancy import (
    redundancy_reduction_first_last_boundary,
    reduce_event_sequence,
    reduce_event_sequence_with_weights,
)


def test_first_last_boundary_reduces_long_family_run_to_boundaries():
    seq = [("write", ".txt", f"file{i}") for i in range(6)]
    reduced, weights = redundancy_reduction_first_last_boundary(seq, repeat_cap=32, alpha=0.25)
    assert reduced == [seq[0], seq[-1]]
    assert len(weights) == 2
    assert all(w > 1.0 for w in weights)


def test_first_last_boundary_weights_two_event_run_without_changing_tokens():
    seq = [("read", ".txt", "a"), ("read", ".txt", "b")]
    reduced, weights = redundancy_reduction_first_last_boundary(seq)
    assert reduced == seq
    assert len(weights) == 2
    assert min(weights) > 1.0


def test_first_last_boundary_preserves_inserted_family_boundary():
    seq = [
        ("write", ".txt", "a"),
        ("write", ".txt", "b"),
        ("connect", "<nosuffix>", "<ip>"),
        ("write", ".txt", "c"),
        ("write", ".txt", "d"),
        ("write", ".txt", "e"),
    ]
    reduced, weights = redundancy_reduction_first_last_boundary(seq)
    assert ("connect", "<nosuffix>", "<ip>") in reduced
    assert reduced == [seq[0], seq[1], seq[2], seq[3], seq[5]]
    assert weights[2] == 1.0


def test_first_last_boundary_aliases_integrate_with_public_reducers():
    seq = [("regquery", "<nosuffix>", f"k{i}") for i in range(5)]
    reduced = reduce_event_sequence(seq, mode="flb_rr")
    assert reduced == [seq[0], seq[-1]]
    reduced2, weights = reduce_event_sequence_with_weights(seq, mode="first_last_boundary")
    assert reduced2 == reduced
    assert len(weights) == len(reduced2)
    assert max(weights) > 1.0
