import torch
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.encoders.clip_rgb import CLIPRgbEncoder


def test_clip_rgb_encodes_to_512d():
    enc = CLIPRgbEncoder(device="cpu")
    sample = synthetic_tree()
    z = enc.encode(sample)
    assert z.shape == (512,)
    assert torch.isfinite(z).all()
