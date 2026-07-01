from __future__ import annotations

from typing import Iterable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from malsnif.config import Config


def parse_int_list(value: str | Iterable[int] | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        out: list[int] = []
        for part in value.replace(";", ",").split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out or list(default)
    return [int(x) for x in value] or list(default)


def _pad_2d(seqs: list[list[int]], pad: int = 0, max_len: int | None = None) -> torch.Tensor:
    if not seqs:
        return torch.zeros((1, 1), dtype=torch.long)
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    max_len = max(max_len, 1)
    out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:max_len] or [pad]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class EventEncoder(nn.Module):
    """Original MalSnif-style event encoder: token embeddings -> GRU."""

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        num, dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num, dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not cfg.freeze_word_embeddings
        self.gru = nn.GRU(dim, cfg.semantic_dim, batch_first=True)
        self.proj = nn.Linear(cfg.semantic_dim, cfg.semantic_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        _, h = self.gru(emb)
        return torch.sigmoid(self.proj(F.relu(h[-1])))


class SequenceEncoder(nn.Module):
    """Original MalSnif-style process/event-sequence encoder: BiLSTM."""

    def __init__(self, cfg: Config):
        super().__init__()
        hidden_each = max(1, cfg.behavior_dim // 2)
        self.bilstm = nn.LSTM(cfg.semantic_dim, hidden_each, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_each * 2, cfg.behavior_dim)

    def forward(self, event_sem: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.bilstm(event_sem)
        if lengths is None:
            last = out[:, -1, :]
        else:
            lengths = torch.clamp(lengths.to(out.device).long(), min=1, max=out.size(1))
            gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            last = out.gather(1, gather_idx).squeeze(1)
        return torch.sigmoid(self.proj(torch.sigmoid(last)))


class HierarchicalLogEncoder(nn.Module):
    """Baseline MalSnif semantic encoder kept for faithful A0 reproduction."""

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_encoder = EventEncoder(embedding_matrix, cfg)
        self.seq_encoder = SequenceEncoder(cfg)

    def forward_nested(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> torch.Tensor:
        item_event_counts = [max(1, min(len(seq), max_events)) for seq in nested_ids]
        flat_events: list[list[int]] = []
        for seq in nested_ids:
            seq = seq[:max_events]
            if not seq:
                seq = [[1]]
            flat_events.extend(seq)
        token_tensor = _pad_2d(flat_events, pad=0, max_len=max_tokens).to(device)
        flat_sem = self.event_encoder(token_tensor)
        num_items = len(nested_ids)
        sem_dim = flat_sem.size(-1)
        padded = torch.zeros((num_items, max(item_event_counts), sem_dim), device=device)
        weight_tensor = torch.ones((num_items, max(item_event_counts)), device=device, dtype=flat_sem.dtype)
        cursor = 0
        for i, cnt in enumerate(item_event_counts):
            weights = None
            if nested_weights is not None and i < len(nested_weights):
                weights = [float(w) for w in nested_weights[i][:cnt]]
            if not weights:
                weights = [1.0] * cnt
            if len(weights) < cnt:
                weights.extend([1.0] * (cnt - len(weights)))
            padded[i, :cnt] = flat_sem[cursor : cursor + cnt]
            weight_tensor[i, :cnt] = torch.tensor(weights[:cnt], dtype=flat_sem.dtype, device=device)
            cursor += cnt
        if nested_weights is not None:
            padded = padded * weight_tensor.unsqueeze(-1)
        lengths = torch.tensor(item_event_counts, dtype=torch.long, device=device)
        return self.seq_encoder(padded, lengths)


class TokenMeanEventEmbedder(nn.Module):
    """Convert each audit event token list into an event vector by masked mean pooling.

    This keeps the original Word2Vec vocabulary/embedding pipeline intact.  MCBG
    then models the sequence of event vectors, rather than introducing a new
    external semantic model.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        num, dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num, dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not cfg.freeze_word_embeddings
        self.output_dim = dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [num_events, max_tokens]
        emb = self.embedding(token_ids)
        mask = (token_ids != 0).to(emb.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom


class SameLengthConv1d(nn.Module):
    """1D convolution over event sequences with output length cropped to input length."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, C]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if y.size(1) > x.size(1):
            y = y[:, : x.size(1), :]
        elif y.size(1) < x.size(1):
            pad = torch.zeros((y.size(0), x.size(1) - y.size(1), y.size(2)), dtype=y.dtype, device=y.device)
            y = torch.cat([y, pad], dim=1)
        return y


class MCBGEncoder(nn.Module):
    """Multi-kernel CNN + BiGRU + multi-head attention semantic encoder.

    This is the semantic branch in AGF-ST-HGAN-MCBG.  It consumes the same nested
    token-id representation as the baseline, so the data pipeline and Word2Vec
    training remain comparable to MalSnif.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_embedder = TokenMeanEventEmbedder(embedding_matrix, cfg)
        word_dim = int(embedding_matrix.shape[1])
        kernels = parse_int_list(getattr(cfg, "mcbg_kernel_sizes", "2,3,5"), [2, 3, 5])
        conv_total = int(getattr(cfg, "mcbg_conv_dim", 0) or cfg.semantic_dim)
        base = max(1, conv_total // max(len(kernels), 1))
        channels = [base for _ in kernels]
        channels[-1] += max(0, conv_total - sum(channels))
        self.convs = nn.ModuleList([SameLengthConv1d(word_dim, ch, k) for ch, k in zip(channels, kernels)])
        self.conv_proj = nn.Linear(sum(channels), cfg.semantic_dim)
        hidden_each = max(1, cfg.behavior_dim // 2)
        self.bigru = nn.GRU(cfg.semantic_dim, hidden_each, batch_first=True, bidirectional=True)
        gru_dim = hidden_each * 2
        if gru_dim != cfg.behavior_dim:
            self.gru_proj = nn.Linear(gru_dim, cfg.behavior_dim)
        else:
            self.gru_proj = nn.Identity()
        heads = int(getattr(cfg, "mcbg_attention_heads", 4) or 1)
        # MultiheadAttention requires divisibility.  Fall back to the largest valid
        # head count <= requested, while keeping the requested value in config logs.
        if cfg.behavior_dim % heads != 0:
            valid = [h for h in range(heads, 0, -1) if cfg.behavior_dim % h == 0]
            heads = valid[0] if valid else 1
        self.attn = nn.MultiheadAttention(cfg.behavior_dim, heads, dropout=float(getattr(cfg, "mcbg_dropout", cfg.dropout)), batch_first=True)
        self.pool_score = nn.Linear(cfg.behavior_dim, 1)
        self.norm = nn.LayerNorm(cfg.behavior_dim)
        self.dropout = nn.Dropout(float(getattr(cfg, "mcbg_dropout", cfg.dropout)))

    def _nested_to_event_tensor(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_event_counts = [max(1, min(len(seq), max_events)) for seq in nested_ids]
        flat_events: list[list[int]] = []
        for seq in nested_ids:
            seq = seq[:max_events]
            if not seq:
                seq = [[1]]
            flat_events.extend(seq)
        token_tensor = _pad_2d(flat_events, pad=0, max_len=max_tokens).to(device)
        flat_event_vec = self.event_embedder(token_tensor)
        num_items = len(nested_ids)
        max_count = max(item_event_counts) if item_event_counts else 1
        event_dim = flat_event_vec.size(-1)
        padded = torch.zeros((num_items, max_count, event_dim), device=device, dtype=flat_event_vec.dtype)
        mask = torch.ones((num_items, max_count), device=device, dtype=torch.bool)
        weight_tensor = torch.ones((num_items, max_count), device=device, dtype=flat_event_vec.dtype)
        cursor = 0
        for i, cnt in enumerate(item_event_counts):
            padded[i, :cnt] = flat_event_vec[cursor : cursor + cnt]
            mask[i, :cnt] = False  # False means not masked for MultiheadAttention.
            weights = None
            if nested_weights is not None and i < len(nested_weights):
                weights = [float(w) for w in nested_weights[i][:cnt]]
            if not weights:
                weights = [1.0] * cnt
            if len(weights) < cnt:
                weights.extend([1.0] * (cnt - len(weights)))
            weight_tensor[i, :cnt] = torch.tensor(weights[:cnt], dtype=flat_event_vec.dtype, device=device)
            cursor += cnt
        return padded, mask, weight_tensor

    def forward_nested(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> torch.Tensor:
        x, key_padding_mask, event_weights = self._nested_to_event_tensor(nested_ids, max_events, max_tokens, device, nested_weights)
        conv_parts = [F.gelu(conv(x)) for conv in self.convs]
        x = self.conv_proj(torch.cat(conv_parts, dim=-1))
        x = self.dropout(F.gelu(x))
        out, _ = self.bigru(x)
        out = self.gru_proj(out)
        # key_padding_mask=True means the timestep is ignored.
        attn_out, _ = self.attn(out, out, out, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.norm(out + self.dropout(attn_out))
        scores = self.pool_score(out).squeeze(-1)
        if nested_weights is not None:
            beta = float(getattr(self.cfg, "mw_prr_attention_beta", 1.0) or 1.0)
            scores = scores + beta * torch.log(event_weights.clamp_min(1e-6))
        scores = scores.masked_fill(key_padding_mask, float("-inf"))
        # Every item has at least one unmasked timestep by construction.
        alpha = torch.softmax(scores, dim=-1)
        return torch.sum(out * alpha.unsqueeze(-1), dim=1)


class SameLengthDilatedConv1d(nn.Module):
    """Length-preserving 1D convolution with explicit dilation padding.

    The project represents every node/edge as an event sequence with shape
    ``[batch, time, dim]``.  This wrapper keeps that public shape stable while
    allowing the temporal receptive field to grow via dilation.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.dilation = max(1, int(dilation))
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=self.kernel_size, dilation=self.dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, C]
        total_pad = self.dilation * (self.kernel_size - 1)
        left = total_pad // 2
        right = total_pad - left
        y = F.pad(x.transpose(1, 2), (left, right))
        y = self.conv(y).transpose(1, 2)
        if y.size(1) > x.size(1):
            y = y[:, : x.size(1), :]
        elif y.size(1) < x.size(1):
            pad = torch.zeros((y.size(0), x.size(1) - y.size(1), y.size(2)), dtype=y.dtype, device=y.device)
            y = torch.cat([y, pad], dim=1)
        return y


class GatedDilatedTemporalBlock(nn.Module):
    """Gated dilated temporal convolution block for audit event sequences."""

    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.value_conv = SameLengthDilatedConv1d(dim, dim, kernel_size, dilation)
        self.gate_conv = SameLengthDilatedConv1d(dim, dim, kernel_size, dilation)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.value_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        y = self.out_proj(value * gate)
        return self.norm(x + self.dropout(y))


class GDTCMCBGEncoder(nn.Module):
    """E1-GDTC-MCBG semantic encoder.

    This is a drop-in replacement for ``MCBGEncoder``.  It keeps the same
    ``forward_nested`` signature and returns ``cfg.behavior_dim`` features, but
    replaces the ordinary multi-kernel CNN + BiGRU + multi-head attention stack
    with gated dilated temporal convolution blocks and evidence-aware pooling.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_embedder = TokenMeanEventEmbedder(embedding_matrix, cfg)
        word_dim = int(embedding_matrix.shape[1])
        self.hidden_dim = int(getattr(cfg, "semantic_dim", 64) or 64)
        self.input_proj = nn.Linear(word_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        dilations = parse_int_list(getattr(cfg, "gdtc_dilations", "1,2,4"), [1, 2, 4])
        kernel_size = int(getattr(cfg, "gdtc_kernel_size", 3) or 3)
        dropout = float(getattr(cfg, "gdtc_dropout", getattr(cfg, "mcbg_dropout", cfg.dropout)) or cfg.dropout)
        self.blocks = nn.ModuleList([
            GatedDilatedTemporalBlock(self.hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.output_proj = nn.Linear(self.hidden_dim, cfg.behavior_dim) if self.hidden_dim != cfg.behavior_dim else nn.Identity()
        self.pool_hidden = nn.Linear(cfg.behavior_dim, cfg.behavior_dim)
        self.pool_score = nn.Linear(cfg.behavior_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.use_event_weight_pooling = bool(getattr(cfg, "gdtc_use_event_weight_pooling", True))

    def _nested_to_event_tensor(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_event_counts = [max(1, min(len(seq), max_events)) for seq in nested_ids]
        flat_events: list[list[int]] = []
        for seq in nested_ids:
            seq = seq[:max_events]
            if not seq:
                seq = [[1]]
            flat_events.extend(seq)
        token_tensor = _pad_2d(flat_events, pad=0, max_len=max_tokens).to(device)
        flat_event_vec = self.event_embedder(token_tensor)
        num_items = len(nested_ids)
        max_count = max(item_event_counts) if item_event_counts else 1
        event_dim = flat_event_vec.size(-1)
        padded = torch.zeros((num_items, max_count, event_dim), device=device, dtype=flat_event_vec.dtype)
        mask = torch.ones((num_items, max_count), device=device, dtype=torch.bool)
        weight_tensor = torch.ones((num_items, max_count), device=device, dtype=flat_event_vec.dtype)
        cursor = 0
        for i, cnt in enumerate(item_event_counts):
            padded[i, :cnt] = flat_event_vec[cursor : cursor + cnt]
            mask[i, :cnt] = False
            weights = None
            if nested_weights is not None and i < len(nested_weights):
                weights = [float(w) for w in nested_weights[i][:cnt]]
            if not weights:
                weights = [1.0] * cnt
            if len(weights) < cnt:
                weights.extend([1.0] * (cnt - len(weights)))
            weight_tensor[i, :cnt] = torch.tensor(weights[:cnt], dtype=flat_event_vec.dtype, device=device)
            cursor += cnt
        return padded, mask, weight_tensor

    def forward_nested(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> torch.Tensor:
        x, key_padding_mask, event_weights = self._nested_to_event_tensor(nested_ids, max_events, max_tokens, device, nested_weights)
        x = self.input_norm(self.input_proj(x))
        x = self.dropout(F.gelu(x))
        for block in self.blocks:
            x = block(x)
        out = self.output_proj(x)
        scores = self.pool_score(torch.tanh(self.pool_hidden(out))).squeeze(-1)
        if nested_weights is not None and self.use_event_weight_pooling:
            beta = float(getattr(self.cfg, "mw_prr_attention_beta", 1.0) or 1.0)
            scores = scores + beta * torch.log(event_weights.clamp_min(1e-6))
        scores = scores.masked_fill(key_padding_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.sum(out * alpha.unsqueeze(-1), dim=1)


class SameLengthDepthwiseSeparableDilatedConv1d(nn.Module):
    """Length-preserving depthwise-separable dilated 1D convolution.

    This keeps the RGD-BiGRU-MCBG encoder parameter-light: the depthwise part
    learns temporal filters per channel and the pointwise part mixes channels.
    """

    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.dilation = max(1, int(dilation))
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=dim,
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self.dilation * (self.kernel_size - 1)
        left = total_pad // 2
        right = total_pad - left
        y = F.pad(x.transpose(1, 2), (left, right))
        y = self.pointwise(self.depthwise(y)).transpose(1, 2)
        if y.size(1) > x.size(1):
            y = y[:, : x.size(1), :]
        elif y.size(1) < x.size(1):
            pad = torch.zeros((y.size(0), x.size(1) - y.size(1), y.size(2)), dtype=y.dtype, device=y.device)
            y = torch.cat([y, pad], dim=1)
        return y


class ResidualGatedDilatedBlock(nn.Module):
    """Residual gated dilated convolution used by RGD-BiGRU-MCBG.

    The learnable residual scale is initialized small so the encoder can fall
    back toward the original BiGRU-driven MCBG path when the convolutional
    enhancement is not useful for a dataset/seed.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        residual_scale_init: float = 0.1,
        depthwise_separable: bool = True,
    ):
        super().__init__()
        conv_cls = SameLengthDepthwiseSeparableDilatedConv1d if depthwise_separable else SameLengthDilatedConv1d
        self.value_conv = conv_cls(dim, kernel_size, dilation) if depthwise_separable else conv_cls(dim, dim, kernel_size, dilation)
        self.gate_conv = conv_cls(dim, kernel_size, dilation) if depthwise_separable else conv_cls(dim, dim, kernel_size, dilation)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.value_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        y = self.out_proj(value * gate)
        return self.norm(x + self.residual_scale * self.dropout(y))


class RGDBiGRUMCBGEncoder(nn.Module):
    """Residual Gated Dilated CNN + BiGRU MCBG encoder.

    This is a conservative E1_eha_only semantic encoder: it replaces only the
    ordinary multi-kernel 1D CNN front-end of MCBG with residual gated dilated
    convolution blocks, then keeps the BiGRU and attention-pooling path.  The
    public ``forward_nested`` signature and output dimension stay identical to
    ``MCBGEncoder`` for low-coupling integration with node and edge encoders.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_embedder = TokenMeanEventEmbedder(embedding_matrix, cfg)
        word_dim = int(embedding_matrix.shape[1])
        self.semantic_dim = int(getattr(cfg, "semantic_dim", 64) or 64)
        self.input_proj = nn.Linear(word_dim, self.semantic_dim)
        self.input_norm = nn.LayerNorm(self.semantic_dim)
        dilations = parse_int_list(getattr(cfg, "rgd_dilations", "1,2"), [1, 2])
        kernel_size = int(getattr(cfg, "rgd_kernel_size", 3) or 3)
        dropout = float(getattr(cfg, "rgd_dropout", getattr(cfg, "mcbg_dropout", cfg.dropout)) or cfg.dropout)
        scale_init = float(getattr(cfg, "rgd_residual_scale_init", 0.1) or 0.1)
        depthwise = bool(getattr(cfg, "rgd_depthwise_separable", True))
        self.blocks = nn.ModuleList([
            ResidualGatedDilatedBlock(
                self.semantic_dim,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
                residual_scale_init=scale_init,
                depthwise_separable=depthwise,
            )
            for d in dilations
        ])
        hidden_each = max(1, cfg.behavior_dim // 2)
        self.bigru = nn.GRU(self.semantic_dim, hidden_each, batch_first=True, bidirectional=True)
        gru_dim = hidden_each * 2
        self.gru_proj = nn.Linear(gru_dim, cfg.behavior_dim) if gru_dim != cfg.behavior_dim else nn.Identity()
        self.skip_proj = nn.Linear(self.semantic_dim, cfg.behavior_dim) if self.semantic_dim != cfg.behavior_dim else nn.Identity()
        heads = int(getattr(cfg, "mcbg_attention_heads", 4) or 1)
        if cfg.behavior_dim % heads != 0:
            valid = [h for h in range(heads, 0, -1) if cfg.behavior_dim % h == 0]
            heads = valid[0] if valid else 1
        self.attn = nn.MultiheadAttention(cfg.behavior_dim, heads, dropout=dropout, batch_first=True)
        self.pool_score = nn.Linear(cfg.behavior_dim, 1)
        self.norm = nn.LayerNorm(cfg.behavior_dim)
        self.attn_norm = nn.LayerNorm(cfg.behavior_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_event_weight_pooling = bool(getattr(cfg, "rgd_use_event_weight_pooling", True))

    def _nested_to_event_tensor(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_event_counts = [max(1, min(len(seq), max_events)) for seq in nested_ids]
        flat_events: list[list[int]] = []
        for seq in nested_ids:
            seq = seq[:max_events]
            if not seq:
                seq = [[1]]
            flat_events.extend(seq)
        token_tensor = _pad_2d(flat_events, pad=0, max_len=max_tokens).to(device)
        flat_event_vec = self.event_embedder(token_tensor)
        num_items = len(nested_ids)
        max_count = max(item_event_counts) if item_event_counts else 1
        event_dim = flat_event_vec.size(-1)
        padded = torch.zeros((num_items, max_count, event_dim), device=device, dtype=flat_event_vec.dtype)
        mask = torch.ones((num_items, max_count), device=device, dtype=torch.bool)
        weight_tensor = torch.ones((num_items, max_count), device=device, dtype=flat_event_vec.dtype)
        cursor = 0
        for i, cnt in enumerate(item_event_counts):
            padded[i, :cnt] = flat_event_vec[cursor : cursor + cnt]
            mask[i, :cnt] = False
            weights = None
            if nested_weights is not None and i < len(nested_weights):
                weights = [float(w) for w in nested_weights[i][:cnt]]
            if not weights:
                weights = [1.0] * cnt
            if len(weights) < cnt:
                weights.extend([1.0] * (cnt - len(weights)))
            weight_tensor[i, :cnt] = torch.tensor(weights[:cnt], dtype=flat_event_vec.dtype, device=device)
            cursor += cnt
        return padded, mask, weight_tensor

    def forward_nested(
        self,
        nested_ids: list[list[list[int]]],
        max_events: int,
        max_tokens: int,
        device,
        nested_weights: list[list[float]] | None = None,
    ) -> torch.Tensor:
        x, key_padding_mask, event_weights = self._nested_to_event_tensor(nested_ids, max_events, max_tokens, device, nested_weights)
        x = self.input_norm(self.input_proj(x))
        x = self.dropout(F.gelu(x))
        for block in self.blocks:
            x = block(x)
        conv_features = x
        gru_out, _ = self.bigru(conv_features)
        out = self.gru_proj(gru_out)
        # Residual skip preserves local attack evidence if the recurrent path
        # over-smooths short suspicious event fragments.
        out = self.norm(out + self.skip_proj(conv_features))
        attn_out, _ = self.attn(out, out, out, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.attn_norm(out + self.dropout(attn_out))
        scores = self.pool_score(out).squeeze(-1)
        if nested_weights is not None and self.use_event_weight_pooling:
            beta = float(getattr(self.cfg, "mw_prr_attention_beta", 1.0) or 1.0)
            scores = scores + beta * torch.log(event_weights.clamp_min(1e-6))
        scores = scores.masked_fill(key_padding_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.sum(out * alpha.unsqueeze(-1), dim=1)


class NullSemanticEncoder(nn.Module):
    """Type-only semantic branch used by structure-only diagnostics."""

    def __init__(self, cfg: Config, num_types: int):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, cfg.behavior_dim)

    def forward_types(self, node_type_ids: torch.Tensor) -> torch.Tensor:
        return self.type_embed(node_type_ids)
