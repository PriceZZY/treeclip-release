"""Smoke tests for the three fusion strategies — checks shape/dim handling, not absolute accuracy."""
import numpy as np
from treeclip.encoders.base import EmbeddingBundle
from treeclip.fusion.strategies import (
    fit_late_fusion, fit_early_fusion, fit_xattn_fusion,
    _stack_modalities,
)


def _make_bundle(seed, species, struct_dim=8, rgb_dim=8, spec_dim=6):
    rng = np.random.default_rng(seed)
    return EmbeddingBundle(
        tree_id=f"t{seed}", site="X", species=species,
        z_struct=rng.standard_normal(struct_dim).astype(np.float32),
        z_rgb=rng.standard_normal(rgb_dim).astype(np.float32),
        z_spec=rng.standard_normal(spec_dim).astype(np.float32),
    )


def _make_dataset(n_per_class=10):
    """Two well-separated classes — fusion should achieve high accuracy on this."""
    rng = np.random.default_rng(0)
    bundles, ys = [], []
    for cls in ("A", "B"):
        offset = 5 if cls == "A" else -5
        for i in range(n_per_class):
            seed = ord(cls) * 100 + i
            b = _make_bundle(seed, cls)
            # Shift mean for class separability
            b.z_struct = b.z_struct + offset
            b.z_rgb = b.z_rgb + offset
            b.z_spec = b.z_spec + offset
            bundles.append(b); ys.append(cls)
    return bundles, np.array(ys)


def test_stack_modalities_concat_dims():
    bundles, ys = _make_dataset()
    X = _stack_modalities(bundles, ["z_struct", "z_rgb", "z_spec"])
    assert X.shape == (20, 8 + 8 + 6)


def test_late_fusion_separates_easy_classes():
    bundles, ys = _make_dataset()
    perm = np.random.RandomState(0).permutation(len(ys))
    tr, va = perm[:16], perm[16:]
    res = fit_late_fusion(
        [bundles[i] for i in tr], ys[tr],
        [bundles[i] for i in va], ys[va],
        modalities=["z_struct", "z_rgb", "z_spec"], seed=42, epochs=30, device="cpu",
    )
    assert res.macro_f1 > 0.5


def test_early_fusion_separates_easy_classes():
    bundles, ys = _make_dataset()
    perm = np.random.RandomState(0).permutation(len(ys))
    tr, va = perm[:16], perm[16:]
    res = fit_early_fusion(
        [bundles[i] for i in tr], ys[tr],
        [bundles[i] for i in va], ys[va],
        modalities=["z_struct", "z_rgb", "z_spec"], seed=42, epochs=30, device="cpu",
    )
    assert res.macro_f1 > 0.5


def test_xattn_fusion_runs():
    """Don't assert accuracy on xattn (needs more epochs/data); just smoke."""
    bundles, ys = _make_dataset()
    perm = np.random.RandomState(0).permutation(len(ys))
    tr, va = perm[:16], perm[16:]
    res = fit_xattn_fusion(
        [bundles[i] for i in tr], ys[tr],
        [bundles[i] for i in va], ys[va],
        modalities=["z_struct", "z_rgb", "z_spec"], seed=42, epochs=20, device="cpu",
    )
    assert 0.0 <= res.macro_f1 <= 1.0
