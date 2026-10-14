from pathlib import Path
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.encoders.base import EmbeddingBundle
from treeclip.encoders.precompute_embeddings import precompute_for_cache


def test_precompute_runs_for_synthetic_cache(tmp_path):
    cache_in = tmp_path / "cache" / "OSBS"
    cache_in.mkdir(parents=True)
    for i in range(3):
        synthetic_tree(tree_id=f"t{i}", site="OSBS", species=f"sp_{i}").save_npz(
            cache_in / f"t{i}.npz"
        )
    emb_out = tmp_path / "emb"
    # Only run mean+MLP for speed (no CLIP download needed for this test)
    n = precompute_for_cache(
        cache_root=tmp_path / "cache",
        embedding_root=emb_out,
        device="cpu",
        which=["hsi_mean_mlp"],
    )
    assert n == 3
    paths = sorted((emb_out / "OSBS").glob("*.npz"))
    assert len(paths) == 3
    eb = EmbeddingBundle.load_npz(paths[0])
    assert eb.z_spec is not None


def test_precompute_writes_all_three_modalities(tmp_path):
    """Smoke: all three encoders (clip_rgb, clip_struct, hsi_mean_mlp) populate the bundle."""
    cache_in = tmp_path / "cache" / "OSBS"
    cache_in.mkdir(parents=True)
    synthetic_tree(tree_id="t1", site="OSBS", species="sp_1").save_npz(
        cache_in / "t1.npz"
    )
    emb_out = tmp_path / "emb"
    n = precompute_for_cache(
        cache_root=tmp_path / "cache",
        embedding_root=emb_out,
        device="cpu",
        which=["clip_rgb", "clip_struct", "hsi_mean_mlp"],
    )
    assert n == 1
    eb = EmbeddingBundle.load_npz(emb_out / "OSBS" / "t1.npz")
    assert eb.z_struct is not None and eb.z_struct.shape == (512,)
    assert eb.z_rgb is not None and eb.z_rgb.shape == (512,)
    assert eb.z_spec is not None
