import torch
from malsnif.config import Config
from malsnif.models.graph_encoder import build_graph_encoder
from malsnif.models.graphsage import WeightedGraphSAGE


def test_config_defaults_to_graphsage_backend():
    cfg = Config()
    assert cfg.graph_encoder == "graphsage"
    enc = build_graph_encoder(8, cfg)
    assert isinstance(enc, WeightedGraphSAGE)


def test_graphsage_forward_shape_with_edge_weights_and_chunks():
    cfg = Config(hidden_dim=16, gcn_layers=2, edge_chunk_size=2, graph_encoder="graphsage")
    enc = build_graph_encoder(8, cfg)
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=torch.long)
    edge_weight = torch.rand(edge_index.size(1))
    out = enc(x, edge_index, edge_weight)
    assert out.shape == (5, 16)
    assert torch.isfinite(out).all()


def test_graphsage_half_node_float_edge_dtype_safe():
    cfg = Config(hidden_dim=16, gcn_layers=1, edge_chunk_size=2, graph_encoder="graphsage")
    enc = build_graph_encoder(8, cfg)
    x = torch.randn(5, 8).half()
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=torch.long)
    edge_weight = torch.rand(edge_index.size(1), dtype=torch.float32)
    # CPU linear layers expect matching dtype, so cast module to half to mimic
    # CUDA AMP's half node activations while keeping fp32 edge weights.
    enc = enc.half()
    out = enc(x, edge_index, edge_weight)
    assert out.dtype == torch.float16
    assert out.shape == (5, 16)
