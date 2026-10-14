"""HSI encoder: new 369-band Spectral Adapter + SpectralEarth-warmstart ViT-S backbone.

Architecture (design spec §3c, method ii):
  * Spectral Adapter (~110K params): trained from scratch in warm-start.
  * ViT-S body (~22M params): loaded from SE pretrained checkpoint, fine-tuned at
    LR 1e-5 during warm-start.
  * Both frozen at inference time when this encoder is used downstream.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
from typing import Optional

from treeclip.encoders.base import Encoder
from treeclip.encoders.spectral_adapter import SpectralAdapter369
from treeclip.core.tree_sample import TreeSample


PATCH_SIZE_PX = 32  # input spatial patch (tree-crown-sized)
VIT_PATCH = 4       # ViT patch token size on the 32x32 input -> 8x8 = 64 tokens


def _pad_or_crop_to_size(
    hsi: np.ndarray, mask: np.ndarray, target_size: int = PATCH_SIZE_PX
) -> np.ndarray:
    """Center-pad-or-crop HSI patch to (target_size, target_size, B). Zero-pads outside."""
    H, W, B = hsi.shape if hsi.ndim == 3 else (0, 0, 369)
    if H == 0 or W == 0:
        return np.zeros((target_size, target_size, B if B > 0 else 369), dtype=np.float32)
    out = np.zeros((target_size, target_size, B), dtype=np.float32)
    h_off = (target_size - H) // 2
    w_off = (target_size - W) // 2
    h_src_off = max(0, -h_off)
    w_src_off = max(0, -w_off)
    h_dst_off = max(0, h_off)
    w_dst_off = max(0, w_off)
    h_copy = min(H - h_src_off, target_size - h_dst_off)
    w_copy = min(W - w_src_off, target_size - w_dst_off)
    if h_copy > 0 and w_copy > 0:
        out[h_dst_off : h_dst_off + h_copy, w_dst_off : w_dst_off + w_copy] = (
            hsi[h_src_off : h_src_off + h_copy, w_src_off : w_src_off + w_copy]
        )
    return out


class HSISpectralEarthEncoder(Encoder):
    name = "hsi_se"

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        n_bands: int = 369,
        patch_size_px: int = PATCH_SIZE_PX,
        device: str = "cuda",
    ):
        self.device = device
        self.patch_size = patch_size_px
        self.n_bands = n_bands
        self.adapter = SpectralAdapter369(in_bands=n_bands, out_channels=128).to(device)
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=128,
            num_classes=0,
            img_size=patch_size_px,
            patch_size=VIT_PATCH,
        ).to(device)
        self.embedding_dim = self.vit.num_features  # 384 for ViT-S

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self._load_vit_weights(checkpoint_path)

        self.adapter.eval()
        self.vit.eval()
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def _load_vit_weights(self, ckpt_path: Path) -> None:
        """Load SpectralEarth ViT-S weights, dropping the original 224-band Spectral
        Adapter (we replace it with our 369-band one) and any tensors whose shape is
        bound to differ (patch_embed input chans + pos_embed + head)."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        # SpectralEarth checkpoint keys are prefixed with `vit_core.` for the ViT body
        # and `spectral_adapter.` for their (224-band) adapter. Strip vit_core., drop
        # spectral_adapter.* entirely (we replace it).
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("spectral_adapter"):
                continue
            kk = k
            for pref in ("vit_core.", "backbone.", "module.", "encoder."):
                if kk.startswith(pref):
                    kk = kk[len(pref):]
                    break
            cleaned[kk] = v

        # Drop layers whose shape definitely mismatches:
        #   patch_embed.proj.*: input chans differ (SE was 3 in EnMAP pretrain after adapter,
        #                       we use 128 from our own adapter — actually same 128 — but
        #                       patch_size is 4 vs 16, so spatial conv kernel mismatches)
        #   pos_embed: different patch grid (theirs 14x14, ours 8x8)
        #   head.*: classification head (none here, but be safe)
        compatible = {
            k: v for k, v in cleaned.items()
            if not k.startswith("patch_embed.proj")
            and not k.startswith("head")
            and k != "pos_embed"
        }
        missing, unexpected = self.vit.load_state_dict(compatible, strict=False)
        print(
            f"  loaded SE ViT weights from {ckpt_path.name}: "
            f"loaded={len(compatible)} missing={len(missing)} unexpected={len(unexpected)}"
        )

    @torch.no_grad()
    def encode(self, sample: TreeSample) -> torch.Tensor:
        hsi = sample.hsi_crown
        mask = sample.hsi_mask
        if hsi.size == 0:
            return torch.zeros(self.embedding_dim)
        if mask.shape == hsi.shape[:2]:
            hsi = hsi * mask[..., None]
        patch = _pad_or_crop_to_size(hsi, mask, target_size=self.patch_size)
        x = torch.from_numpy(patch).unsqueeze(0).to(self.device)
        feat = self.adapter(x)
        z = self.vit(feat).squeeze(0)
        return z.cpu()
