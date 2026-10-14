"""New 369-band Spectral Adapter for the SE warm-start (design spec §3c, method ii).

Replaces SpectralEarth's original adapter (which expected 224 EnMAP bands).
Architecture mirrors SE: 3 layers of (Conv1D + BN + ReLU) along the band dim,
reducing 369 bands -> 128 channels per pixel.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SpectralAdapter369(nn.Module):
    """Spectral-only adapter: reduces per-pixel spectral dimension via 1x1 conv stack.

    Input:  (B, H, W, bands) — spatial-last HSI patch
    Output: (B, out_channels, H, W) — standard 4D conv layout, ready for ViT patch_embed
    """
    def __init__(self, in_bands: int = 369, out_channels: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_bands, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 192, kernel_size=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, bands = x.shape
        # (B, H, W, bands) -> (B, bands, H*W) for Conv1d
        x = x.reshape(B, H * W, bands).transpose(1, 2)
        y = self.layers(x)  # (B, out, H*W)
        return y.reshape(B, -1, H, W)
