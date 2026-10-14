import numpy as np
import torch
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.encoders.hsi_se import HSISpectralEarthEncoder


def test_se_encoder_smoke_random_weights():
    """No SE checkpoint loaded — architecture smoke only."""
    enc = HSISpectralEarthEncoder(checkpoint_path=None, device="cpu")
    sample = synthetic_tree()
    sample.hsi_crown = np.random.RandomState(0).rand(16, 16, 369).astype("float32")
    sample.hsi_mask = np.ones((16, 16), dtype=bool)
    z = enc.encode(sample)
    assert z.shape == (enc.embedding_dim,)
    assert torch.isfinite(z).all()


def test_se_encoder_embedding_dim_is_384():
    """ViT-S has 384-dim feature output."""
    enc = HSISpectralEarthEncoder(checkpoint_path=None, device="cpu")
    assert enc.embedding_dim == 384


def test_se_encoder_handles_tiny_crowns():
    """IDTReeS HSI crowns can be as small as 2x2 pixels — encoder must pad-or-crop."""
    enc = HSISpectralEarthEncoder(checkpoint_path=None, device="cpu")
    sample = synthetic_tree()
    sample.hsi_crown = np.random.RandomState(0).rand(2, 2, 369).astype("float32")
    sample.hsi_mask = np.ones((2, 2), dtype=bool)
    z = enc.encode(sample)
    assert z.shape == (384,)
