import numpy as np
import torch

from malsnif.config import Config
from malsnif.models.malsnif import MalSnifModel
from tests.test_agf_model_forward import _toy_graph


def test_edge_gated_v2_forward_shapes_and_stats():
    cfg = Config(
        model_variant="edge_gated_st_hgan_mcbg",
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
        edge_gate_mode="vector",
        gcn_layers=2,
        max_events_per_node=4,
        max_events_per_edge=2,
        max_tokens_per_event=3,
    )
    emb = np.random.default_rng(10).normal(size=(16, 16)).astype("float32")
    emb[0] = 0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["node_logits"].shape == (3,)
    assert out["node_probs"].shape == (3,)
    assert out["model_variant"] == "edge_gated_st_hgan_mcbg"
    assert out["attention_stats"] is not None
    assert out["gate_stats"] is not None
    assert out["gate_stats"]["gate_mode"] == "edge_vector"
    assert 0.0 <= out["gate_stats"]["edge_gate_mean"] <= 1.0


def test_edge_gated_v2_no_gate_mode_is_fixed_one():
    cfg = Config(
        model_variant="edge_gated_st_hgan_mcbg",
        semantic_encoder="mcbg",
        hidden_dim=8,
        semantic_dim=8,
        behavior_dim=8,
        word_dim=8,
        mcbg_attention_heads=2,
        hgan_num_relations=8,
        hgan_num_time_buckets=4,
        edge_gate_mode="none",
        gcn_layers=1,
        max_events_per_node=3,
        max_events_per_edge=2,
        max_tokens_per_event=3,
    )
    emb = np.random.default_rng(11).normal(size=(16, 8)).astype("float32")
    emb[0] = 0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["gate_stats"]["gate_mode"] == "edge_none"
    assert out["gate_stats"]["edge_gate_mean"] == 1.0
