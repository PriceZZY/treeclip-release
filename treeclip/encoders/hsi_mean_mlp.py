"""Mean-spectrum + small MLP HSI encoder.

Plan B fallback if SE warm-start fails to beat this baseline (per design spec §3c
validation protocol). Aggregates HSI pixels inside crown mask via mean, feeds through
a small frozen MLP. Note: MLP weights are random-initialized here; for actual use,
either train the MLP downstream (label-supervised) or keep frozen as a "no-foundation
baseline" reference for the warm-start kill criterion.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

from treeclip.encoders.base import Encoder
from treeclip.core.tree_sample import TreeSample


class HSIMeanMLPEncoder(Encoder):
    name = "hsi_mean_mlp"

    def __init__(self, n_bands: int = 369, embedding_dim: int = 128, device: str = "cuda"):
        self.embedding_dim = embedding_dim
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(n_bands, 256),
            nn.GELU(),
            nn.Linear(256, embedding_dim),
        ).to(device)
        self.mlp.eval()
        for p in self.mlp.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, sample: TreeSample) -> torch.Tensor:
        hsi = sample.hsi_crown
        mask = sample.hsi_mask
        if hsi.size == 0 or mask.size == 0 or mask.sum() == 0:
            return torch.zeros(self.embedding_dim)
        pixels = hsi[mask]
        mean_spectrum = pixels.mean(axis=0).astype(np.float32)
        x = torch.from_numpy(mean_spectrum).unsqueeze(0).to(self.device)
        z = self.mlp(x).squeeze(0)
        return z.cpu()
