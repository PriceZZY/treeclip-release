"""RGB encoder: 224x224 crown crop -> frozen CLIP ViT-B/16 image encoder -> 512-d."""
from __future__ import annotations
import torch
import open_clip
from PIL import Image

from treeclip.encoders.base import Encoder
from treeclip.core.tree_sample import TreeSample


class CLIPRgbEncoder(Encoder):
    name = "clip_rgb"
    embedding_dim = 512

    def __init__(self, device: str = "cuda"):
        self.device = device
        # Use the quickgelu variant: openai's pretrained weights were trained with
        # QuickGELU activation, so this matches exactly and avoids silent inference drift.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained="openai"
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device)
        self.preprocess = preprocess

    @torch.no_grad()
    def encode(self, sample: TreeSample) -> torch.Tensor:
        pil = Image.fromarray(sample.rgb_crop)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        z = self.model.encode_image(x)
        return z.squeeze(0).cpu()
