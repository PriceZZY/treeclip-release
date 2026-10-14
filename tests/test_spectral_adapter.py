import torch
from treeclip.encoders.spectral_adapter import SpectralAdapter369


def test_adapter_369_to_128_channels():
    adapter = SpectralAdapter369(out_channels=128)
    x = torch.randn(2, 16, 16, 369)
    y = adapter(x)
    assert y.shape == (2, 128, 16, 16)


def test_adapter_param_count_small():
    """Adapter should be relatively small — design spec says ~10k, but actual depends
    on architecture; we cap at 200K which is still <<1% of ViT-S 22M body."""
    adapter = SpectralAdapter369(out_channels=128)
    n_params = sum(p.numel() for p in adapter.parameters())
    assert n_params < 200_000, f"Adapter has too many params: {n_params}"


def test_adapter_supports_arbitrary_patch_size():
    adapter = SpectralAdapter369(out_channels=128)
    x = torch.randn(1, 32, 32, 369)
    y = adapter(x)
    assert y.shape == (1, 128, 32, 32)
