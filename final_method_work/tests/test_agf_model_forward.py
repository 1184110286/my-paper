import numpy as np
import torch

from malsnif.config import Config
from malsnif.models.malsnif import MalSnifModel


def _toy_graph():
    return {
        "node_ids": ["p1", "f1", "p2"],
        "node_type_ids": [0, 1, 0],
        "node_types": ["PROCESS", "FILE", "PROCESS"],
        "node_event_ids": [
            [[1, 2, 0], [3, 4, 0]],
            [[5, 0, 0]],
            [[2, 6, 0], [7, 8, 0]],
        ],
        "edge_index": [(0, 1), (1, 2), (0, 2)],
        "edge_types": ["EVENT_WRITE", "EVENT_READ", "EVENT_FORK"],
        "edge_time_buckets": [0, 1, 2],
        "edge_event_ids": [
            [[3, 4, 0]],
            [[5, 6, 0]],
            [[7, 8, 0]],
        ],
        "node_labels": [0, 0, 1],
        "process_mask": [True, False, True],
        "graph_label": 1,
    }


def test_agf_st_hgan_mcbg_forward_shapes():
    cfg = Config(
        model_variant="agf_st_hgan_mcbg",
        semantic_encoder="mcbg",
        fusion_mode="agf",
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
        max_events_per_node=4,
        max_events_per_edge=2,
        max_tokens_per_event=3,
    )
    emb = np.random.default_rng(0).normal(size=(16, 16)).astype("float32")
    emb[0] = 0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["node_logits"].shape == (3,)
    assert out["node_probs"].shape == (3,)
    assert out["gate_stats"]["gate_mode"] == "vector"
    assert out["attention_stats"] is not None


def test_baseline_variant_still_forward_shapes():
    cfg = Config(
        model_variant="baseline",
        hidden_dim=16,
        semantic_dim=16,
        behavior_dim=16,
        word_dim=16,
        graph_encoder="graphsage",
        gcn_layers=2,
        max_events_per_node=4,
        max_events_per_edge=2,
        max_tokens_per_event=3,
    )
    emb = np.random.default_rng(1).normal(size=(16, 16)).astype("float32")
    emb[0] = 0
    model = MalSnifModel(emb, cfg)
    out = model(_toy_graph(), torch.device("cpu"))
    assert out["node_logits"].shape == (3,)
    assert out["model_variant"] == "baseline"
