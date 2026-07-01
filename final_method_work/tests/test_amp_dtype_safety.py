import torch

from malsnif.config import Config
from malsnif.models.gcn import WeightedGCNLayer, WeightedGCN


def test_weighted_gcn_layer_accepts_half_nodes_float_edge_weights():
    layer = WeightedGCNLayer(4, 3)
    x = torch.randn(5, 4, dtype=torch.float16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_weight = torch.rand(4, dtype=torch.float32)
    out = layer(x, edge_index, edge_weight)
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_weighted_gcn_accepts_mixed_dtypes_across_layers():
    gcn = WeightedGCN(4, 6, 2, layers=2, dropout=0.0)
    x = torch.randn(5, 4, dtype=torch.float16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_weight = torch.rand(4, dtype=torch.float32)
    out = gcn(x, edge_index, edge_weight)
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


def test_amp_fallback_default_is_enabled():
    cfg = Config(use_amp=True)
    assert cfg.amp_fallback_to_fp32 is True

from malsnif.models.hgan import STHGANEncoder


def test_sthgan_type_projection_is_safe_under_cpu_autocast():
    cfg = Config(
        hgan_num_relations=8,
        hgan_num_time_buckets=4,
        hgan_topk=2,
        hgan_pruning_mode="soft",
        hgan_use_node_types=True,
        hgan_use_relation_types=True,
        hgan_use_time_bias=True,
        gcn_layers=2,
        dropout=0.0,
        hgan_attention_dropout=0.0,
    )
    enc = STHGANEncoder(4, 6, 5, cfg)
    x = torch.randn(5, 4, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    node_type_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    edge_type_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    edge_time_buckets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    edge_weight = torch.ones(4, dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out, stats = enc(x, edge_index, node_type_ids, edge_type_ids, edge_time_buckets, edge_weight=edge_weight)
    assert out.shape == (5, 5)
    assert torch.isfinite(out.float()).all()
    assert stats is not None
