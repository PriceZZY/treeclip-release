"""Structure encoder: point cloud -> 2 multi-view depth images (224x224 grayscale)
-> replicated to 3 channels -> frozen CLIP ViT-B/16 (quickgelu) -> L2-normalize each
view embedding -> mean -> 512-d.
"""
from __future__ import annotations
import torch
import open_clip
import numpy as np
from PIL import Image

from treeclip.encoders.base import Encoder
from treeclip.encoders.render_views import render_pca_depth_views
from treeclip.core.tree_sample import TreeSample


class CLIPStructEncoder(Encoder):
    name = "clip_struct"
    embedding_dim = 512

    def __init__(self, device: str = "cuda"):
        self.device = device
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
        views = render_pca_depth_views(sample.point_cloud, image_size=224)
        zs = []
        for v in views:
            rgb = np.stack([v, v, v], axis=-1)
            pil = Image.fromarray(rgb)
            x = self.preprocess(pil).unsqueeze(0).to(self.device)
            z = self.model.encode_image(x).squeeze(0)
            z = z / (torch.linalg.norm(z) + 1e-9)
            zs.append(z)
        return torch.stack(zs).mean(dim=0).cpu()
