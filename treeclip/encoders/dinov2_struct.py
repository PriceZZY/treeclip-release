"""Structure encoder (DINOv2 variant): point cloud -> 2 PCA-aligned depth views
(224x224 grayscale) -> replicated to 3 channels -> frozen DINOv2 ViT-B/14 ->
L2-normalize each view embedding -> mean -> 768-d.

Mirrors CLIPStructEncoder EXACTLY (same rendering, same view-fusion, same L2-norm-then-mean)
except the frozen backbone, so the CLIP-vs-DINOv2 comparison isolates the pretrained image
encoder. DINOv2 is purely self-supervised (no text, no labels), so a positive result makes
the unsupervised claim defensible and shows the advantage is not CLIP-specific.

Loading: try timm first (model code ships with timm; weights from HF), fall back to torch.hub
(code from github; weights from fbaipublicfiles). Input kept at 224 to match the CLIP renders.
"""
from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from treeclip.encoders.render_views import render_pca_depth_views
from treeclip.core.tree_sample import TreeSample

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DINOv2StructEncoder:
    name = "dinov2_struct"

    def __init__(self, device: str = "cuda", arch: str = "vit_base_patch14_dinov2.lvd142m"):
        self.device = device
        model = None
        self.kind = None
        try:
            import timm
            model = timm.create_model(arch, pretrained=True, num_classes=0, img_size=224)
            self.kind = f"timm:{arch}"
        except Exception as e_timm:
            self._timm_error = repr(e_timm)
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.kind = "hub:dinov2_vitb14"
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device)
        self.embedding_dim = int(getattr(model, "num_features", 768))
        # renders are already 224x224 uint8; only tensor + ImageNet normalize needed
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @torch.no_grad()
    def encode(self, sample: TreeSample) -> torch.Tensor:
        views = render_pca_depth_views(sample.point_cloud, image_size=224)
        zs = []
        for v in views:
            rgb = np.stack([v, v, v], axis=-1)
            pil = Image.fromarray(rgb)
            x = self.tf(pil).unsqueeze(0).to(self.device)
            z = self.model(x).squeeze(0)
            z = z / (torch.linalg.norm(z) + 1e-9)
            zs.append(z)
        return torch.stack(zs).mean(dim=0).cpu()
