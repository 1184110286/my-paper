from malsnif.config import Config


def test_edge_weight_default_is_paper_aligned_legacy_sigmoid():
    cfg = Config()
    assert cfg.edge_weight_mode == "legacy_sigmoid"
    assert cfg.edge_weight_init_zero is False
