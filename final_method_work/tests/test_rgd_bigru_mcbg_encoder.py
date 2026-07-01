import numpy as np
import torch

from malsnif.config import Config
from malsnif.models.malsnif import MalSnifModel
from malsnif.models.semantic import RGDBiGRUMCBGEncoder, ResidualGatedDilatedBlock


def test_rgd_bigru_encoder_forward_nested_shape_and_finiteness():
    cfg = Config(semantic_dim=16, behavior_dim=16, word_dim=8, rgd_dilations="1,2", rgd_kernel_size=3)
    emb = np.random.default_rng(10).normal(size=(12, 8)).astype("float32")
    emb[0] = 0.0
    enc = RGDBiGRUMCBGEncoder(emb, cfg)
    nested = [
        [[1, 2, 3], [4, 5], [6]],
        [[2, 2], [3, 4, 5]],
        [],
    ]
    weights = [[1.0, 2.0, 1.5], [0.5, 1.0], []]
    out = enc.forward_nested(nested, max_events=4, max_tokens=5, device=torch.device("cpu"), nested_weights=weights)
    assert out.shape == (3, cfg.behavior_dim)
    assert torch.isfinite(out).all()


def test_residual_gated_block_preserves_sequence_length():
    block = ResidualGatedDilatedBlock(dim=8, kernel_size=3, dilation=2, dropout=0.0, residual_scale_init=0.1)
    x = torch.randn(2, 11, 8)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_model_factory_accepts_rgd_bigru_mcbg_for_e1():
    cfg = Config(
        model_variant="ea_st_hgan_mcbg",
        semantic_encoder="rgd_bigru_mcbg",
        semantic_dim=8,
        behavior_dim=8,
        hidden_dim=8,
        word_dim=8,
        ea_use_eha=True,
        ea_use_ets=False,
        ea_use_eaw=False,
    )
    emb = np.random.default_rng(11).normal(size=(10, 8)).astype("float32")
    emb[0] = 0.0
    model = MalSnifModel(emb, cfg)
    assert isinstance(model.impl.node_encoder, RGDBiGRUMCBGEncoder)


def test_e1_rgd_bigru_model_forward_on_toy_graph():
    from tests.test_agf_model_forward import _toy_graph

    cfg = Config(
        model_variant="ea_st_hgan_mcbg",
        semantic_encoder="rgd_bigru_mcbg",
        hidden_dim=8,
        semantic_dim=8,
        behavior_dim=8,
        word_dim=8,
        hgan_num_relations=16,
        hgan_num_time_buckets=4,
        hgan_topk=2,
        hgan_pruning_mode="soft",
        gcn_layers=1,
        ea_use_eha=True,
        ea_use_ets=False,
        ea_use_eaw=False,
        ea_num_heads=2,
        max_events_per_node=4,
        max_events_per_edge=2,
        max_tokens_per_event=3,
        rgd_dilations="1,2",
    )
    emb = np.random.default_rng(12).normal(size=(16, 8)).astype("float32")
    emb[0] = 0.0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["node_logits"].shape == (3,)
    assert out["model_variant"] == "ea_st_hgan_mcbg"
    assert out["attention_stats"]["ea_use_eha"] is True
