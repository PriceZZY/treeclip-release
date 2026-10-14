import torch
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.encoders.clip_struct import CLIPStructEncoder


def test_struct_encoder_returns_512d():
    enc = CLIPStructEncoder(device="cpu")
    sample = synthetic_tree()
    z = enc.encode(sample)
    assert z.shape == (512,)
    assert torch.isfinite(z).all()


def test_struct_encoder_l2_normalizes_before_mean():
    """The two view embeddings are L2-normalized before averaging, so the resulting
    mean vector has norm in (0, 1]."""
    enc = CLIPStructEncoder(device="cpu")
    sample = synthetic_tree()
    z = enc.encode(sample)
    n = torch.linalg.norm(z).item()
    assert 0.0 < n <= 1.0 + 1e-5
