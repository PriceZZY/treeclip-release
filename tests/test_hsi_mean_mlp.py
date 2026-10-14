import numpy as np
import torch
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.encoders.hsi_mean_mlp import HSIMeanMLPEncoder


def test_mean_mlp_encoder_returns_fixed_dim():
    enc = HSIMeanMLPEncoder(n_bands=426, embedding_dim=128, device="cpu")
    sample = synthetic_tree()
    z = enc.encode(sample)
    assert z.shape == (128,)


def test_mean_mlp_encoder_handles_real_369_bands():
    """Real IDTReeS HSI has 369 bands (per design spec §B2)."""
    sample = synthetic_tree()
    sample.hsi_crown = np.random.RandomState(0).rand(16, 16, 369).astype("float32")
    sample.hsi_mask = np.random.RandomState(0).randint(0, 2, (16, 16)).astype(bool)
    enc = HSIMeanMLPEncoder(n_bands=369, embedding_dim=128, device="cpu")
    z = enc.encode(sample)
    assert z.shape == (128,)


def test_mean_mlp_encoder_handles_empty_hsi():
    sample = synthetic_tree()
    sample.hsi_crown = np.zeros((0, 0, 369), dtype=np.float32)
    sample.hsi_mask = np.zeros((0, 0), dtype=bool)
    enc = HSIMeanMLPEncoder(n_bands=369, embedding_dim=128, device="cpu")
    z = enc.encode(sample)
    assert z.shape == (128,)
    assert torch.all(z == 0)
