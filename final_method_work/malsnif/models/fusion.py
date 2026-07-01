from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from malsnif.config import Config


class AdaptiveGatedFusion(nn.Module):
    """Node-wise semantic/structure vector gate.

    z_v = sigmoid(W [s, g, |s-g|, s*g])
    f_v = z_v * s + (1-z_v) * g
    """

    def __init__(self, dim: int, cfg: Config, scalar_gate: bool = False):
        super().__init__()
        self.scalar_gate = bool(scalar_gate)
        gate_out = 1 if scalar_gate else dim
        hidden = int(getattr(cfg, "gate_hidden_dim", 0) or dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "gate_dropout", getattr(cfg, "dropout", 0.2)))),
            nn.Linear(hidden, gate_out),
        )
        self.temperature = float(getattr(cfg, "gate_temperature", 1.0) or 1.0)

    def forward(self, semantic: torch.Tensor, structure: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        gate_in = torch.cat([semantic, structure, torch.abs(semantic - structure), semantic * structure], dim=-1)
        z = torch.sigmoid(self.net(gate_in) / max(self.temperature, 1e-6))
        if self.scalar_gate:
            z = z.expand_as(semantic)
        fused = z * semantic + (1.0 - z) * structure
        zd = z.detach().float()
        stats = {
            "gate_mode": "scalar" if self.scalar_gate else "vector",
            "gate_semantic_mean": float(zd.mean().cpu()) if zd.numel() else None,
            "gate_semantic_std": float(zd.std(unbiased=False).cpu()) if zd.numel() else None,
            "gate_semantic_min": float(zd.min().cpu()) if zd.numel() else None,
            "gate_semantic_max": float(zd.max().cpu()) if zd.numel() else None,
            "gate_structure_mean": float((1.0 - zd).mean().cpu()) if zd.numel() else None,
        }
        return fused, z, stats


class StaticConcatFusion(nn.Module):
    """A3 ablation: static late fusion with no adaptive gate."""

    def __init__(self, dim: int, cfg: Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "gate_dropout", getattr(cfg, "dropout", 0.2)))),
            nn.LayerNorm(dim),
        )

    def forward(self, semantic: torch.Tensor, structure: torch.Tensor) -> tuple[torch.Tensor, None, dict]:
        return self.proj(torch.cat([semantic, structure], dim=-1)), None, {"gate_mode": "static_concat"}


class MeanFusion(nn.Module):
    """Simple mean fusion used only as a diagnostic ablation."""

    def forward(self, semantic: torch.Tensor, structure: torch.Tensor) -> tuple[torch.Tensor, None, dict]:
        return 0.5 * (semantic + structure), None, {"gate_mode": "mean"}
