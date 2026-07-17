"""Shared embedding projection: teacher 2048-d -> neuron base_embed_dim 512-d.

Problem H10: distill_neurons.py used a global random orthogonal projection
(_embed_proj = nn.Linear(2048, 512)) that was NEVER saved.  Verification scripts
created a DIFFERENT random projection, so neurons never saw the embedding
distribution they were trained on.  This module trains a single shared
projection by freezing all neurons and optimising only the projection layer.

Usage:
    from taiji.resonance.shared_embed import SharedEmbedProj
    proj = SharedEmbedProj.load("data/shared_proj.pt")
    emb_512 = proj(teacher_emb_2048)  # [B, L, 2048] -> [B, L, 512]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SharedEmbedProj(nn.Module):
    """Frozen-after-training projection from teacher embedding dim to neuron base dim."""

    def __init__(self, src_dim: int = 2048, target_dim: int = 512):
        super().__init__()
        self.src_dim = src_dim
        self.target_dim = target_dim
        self.proj = nn.Linear(src_dim, target_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.proj(emb)

    def save(self, path: str) -> None:
        torch.save(self.proj.state_dict(), path)

    @staticmethod
    def load(path: str, src_dim: int = 2048, target_dim: int = 512) -> "SharedEmbedProj":
        m = SharedEmbedProj(src_dim, target_dim)
        m.proj.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
