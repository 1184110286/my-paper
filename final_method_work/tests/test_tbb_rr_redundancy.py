from malsnif.data.redundancy import (
    reduce_event_sequence_with_weights,
    redundancy_reduction_target_boundary,
)


def test_tbb_keeps_first_last_per_block_default_target():
    seq = list(range(45))
    reduced, weights = redundancy_reduction_target_boundary(seq, target_compression=0.90)
    assert reduced == [0, 19, 20, 39, 40, 44]
    assert weights == [1.0] * len(reduced)


def test_tbb_short_sequence_unchanged():
    seq = ["a", "b", "c"]
    reduced, weights = redundancy_reduction_target_boundary(seq, target_compression=0.90)
    assert reduced == seq
    assert weights == [1.0, 1.0, 1.0]


def test_tbb_alias_in_weighted_dispatch():
    seq = list(range(60))
    reduced, weights = reduce_event_sequence_with_weights(
        seq,
        mode="target_boundary",
        tbb_rr_target_compression=0.90,
    )
    assert reduced == [0, 19, 20, 39, 40, 59]
    assert weights == [1.0] * len(reduced)


def test_tbb_compression_budget_controls_output_density():
    seq = list(range(1000))
    reduced, _ = redundancy_reduction_target_boundary(seq, target_compression=0.90)
    reduction = 1.0 - len(reduced) / len(seq)
    assert abs(reduction - 0.90) < 0.02
