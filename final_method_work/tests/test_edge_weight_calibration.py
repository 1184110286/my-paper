import numpy as np
import torch

from malsnif.config import Config
from malsnif.models.malsnif import MalSnifModel


def test_centered_edge_weight_initializes_to_unweighted_baseline():
    cfg = Config(use_semantics=False, behavior_dim=8, hidden_dim=8, semantic_dim=8, edge_weight_mode="centered_sigmoid", edge_weight_init_zero=True)
    emb = np.zeros((3, 8), dtype=np.float32)
    model = MalSnifModel(emb, cfg)
    feat = torch.randn(5, cfg.behavior_dim)
    weights = model._edge_weights(feat)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_legacy_sigmoid_initialization_matches_previous_half_weight_behavior():
    cfg = Config(use_semantics=False, behavior_dim=8, hidden_dim=8, semantic_dim=8, edge_weight_mode="legacy_sigmoid", edge_weight_init_zero=True)
    emb = np.zeros((3, 8), dtype=np.float32)
    model = MalSnifModel(emb, cfg)
    feat = torch.randn(5, cfg.behavior_dim)
    weights = model._edge_weights(feat)
    assert torch.allclose(weights, torch.full_like(weights, 0.5), atol=1e-6)
