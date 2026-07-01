import numpy as np
import torch

from malsnif.config import Config
from malsnif.models.malsnif import MalSnifModel
from tests.test_agf_model_forward import _toy_graph


def test_ea_st_hgan_forward_with_all_mechanisms():
    cfg = Config(
        model_variant="ea_st_hgan_mcbg",
        semantic_encoder="mcbg",
        hidden_dim=16,
        semantic_dim=16,
        behavior_dim=16,
        word_dim=16,
        mcbg_attention_heads=4,
        hgan_num_relations=16,
        hgan_num_time_buckets=4,
        hgan_topk=2,
        hgan_pruning_mode="soft",
        gcn_layers=2,
        ea_use_eha=True,
        ea_use_ets=True,
        ea_use_eaw=True,
        ea_num_heads=4,
        max_events_per_node=4,
        max_events_per_edge=2,
        max_tokens_per_event=3,
    )
    emb = np.random.default_rng(23).normal(size=(16, 16)).astype("float32")
    emb[0] = 0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["node_logits"].shape == (3,)
    assert out["model_variant"] == "ea_st_hgan_mcbg"
    assert out["gate_stats"] is None
    assert out["attention_stats"] is not None
    assert out["attention_stats"]["ea_use_eha"] is True
    assert out["attention_stats"]["ea_use_ets"] is True
    assert out["attention_stats"]["ea_use_eaw"] is True
    assert "eha_entropy_mean" in out["attention_stats"]
    assert "ets_tau_mean" in out["attention_stats"]["layers"][0]
    assert "eaw_head_mean" in out["attention_stats"]["layers"][0]


def test_ea_st_hgan_no_mechanism_has_no_gate_stats():
    cfg = Config(
        model_variant="ea_st_hgan_mcbg",
        semantic_encoder="mcbg",
        hidden_dim=8,
        semantic_dim=8,
        behavior_dim=8,
        word_dim=8,
        mcbg_attention_heads=2,
        ea_num_heads=2,
        gcn_layers=1,
        max_events_per_node=3,
        max_events_per_edge=2,
        max_tokens_per_event=3,
    )
    emb = np.random.default_rng(24).normal(size=(16, 8)).astype("float32")
    emb[0] = 0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["model_variant"] == "ea_st_hgan_mcbg"
    assert out["gate_stats"] is None
    assert out["attention_stats"]["ea_use_eha"] is False
