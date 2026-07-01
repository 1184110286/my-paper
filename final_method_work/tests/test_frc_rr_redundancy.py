from malsnif.data.redundancy import (
    redundancy_reduction_family_run_cap,
    reduce_event_sequence,
    reduce_event_sequence_with_weights,
)


def test_family_run_cap_keeps_short_runs_and_caps_long_runs():
    seq = [("write", ".txt", f"file{i}") for i in range(6)]
    reduced, weights = redundancy_reduction_family_run_cap(seq, cap_size=3, repeat_cap=32, alpha=0.25)
    assert reduced == [seq[0], seq[2], seq[5]]
    assert len(weights) == 3
    assert all(w > 1.0 for w in weights)


def test_family_run_cap_event_family_boundary_preserves_inserted_variant():
    seq = [
        ("write", ".txt", "a"),
        ("write", ".txt", "b"),
        ("connect", "<nosuffix>", "<ip>"),
        ("write", ".txt", "c"),
        ("write", ".txt", "d"),
        ("write", ".txt", "e"),
        ("write", ".txt", "f"),
    ]
    reduced, weights = redundancy_reduction_family_run_cap(seq, cap_size=3)
    assert ("connect", "<nosuffix>", "<ip>") in reduced
    assert reduced[:3] == seq[:3]
    assert reduced[-3:] == [seq[3], seq[4], seq[6]]
    assert weights[:3] == [1.0, 1.0, 1.0]
    assert all(w > 1.0 for w in weights[-3:])


def test_family_run_cap_aliases_integrate_with_public_reducers():
    seq = [("regquery", "<nosuffix>", f"k{i}") for i in range(5)]
    reduced = reduce_event_sequence(seq, mode="frc_rr", frc_rr_cap_size=3)
    assert len(reduced) == 3
    reduced2, weights = reduce_event_sequence_with_weights(seq, mode="family_run_cap", frc_rr_cap_size=3)
    assert reduced2 == reduced
    assert len(weights) == len(reduced2)
    assert max(weights) > 1.0
