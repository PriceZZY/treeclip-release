"""CLI: walk the cache, run selected encoders per tree, write EmbeddingBundle .npz files."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from treeclip.encoders.base import EmbeddingBundle, Encoder
from treeclip.data.tree_dataset import TreeDataset


VALID = ("clip_rgb", "clip_struct", "hsi_mean_mlp", "hsi_se")


def _build_encoders(
    which: list[str], device: str, se_ckpt: Optional[Path], n_bands: int,
) -> dict[str, Encoder]:
    out: dict[str, Encoder] = {}
    for w in which:
        if w not in VALID:
            raise ValueError(f"unknown encoder {w!r}; valid: {VALID}")
    if "clip_rgb" in which:
        from treeclip.encoders.clip_rgb import CLIPRgbEncoder
        out["clip_rgb"] = CLIPRgbEncoder(device=device)
    if "clip_struct" in which:
        from treeclip.encoders.clip_struct import CLIPStructEncoder
        out["clip_struct"] = CLIPStructEncoder(device=device)
    if "hsi_mean_mlp" in which:
        from treeclip.encoders.hsi_mean_mlp import HSIMeanMLPEncoder
        out["hsi_mean_mlp"] = HSIMeanMLPEncoder(n_bands=n_bands, device=device)
    if "hsi_se" in which:
        from treeclip.encoders.hsi_se import HSISpectralEarthEncoder
        out["hsi_se"] = HSISpectralEarthEncoder(
            checkpoint_path=se_ckpt, n_bands=n_bands, device=device
        )
    return out


def _detect_n_bands(ds: TreeDataset) -> int:
    """Read the first sample to detect HSI band count. Defaults to 369 if no HSI."""
    if len(ds) == 0:
        return 369
    for i in range(min(len(ds), 8)):
        sample = ds[i]
        if sample.hsi_crown.ndim == 3 and sample.hsi_crown.shape[-1] > 0:
            return int(sample.hsi_crown.shape[-1])
    return 369


def precompute_for_cache(
    cache_root: Path,
    embedding_root: Path,
    device: str = "cuda",
    which: Optional[list[str]] = None,
    se_ckpt: Optional[Path] = None,
) -> int:
    """Walk every tree in cache_root, compute selected embeddings, write per-tree
    EmbeddingBundle .npz at embedding_root/{site}/{tree_id}.npz. Returns count."""
    which = which or ["clip_rgb", "clip_struct", "hsi_mean_mlp"]
    ds = TreeDataset(cache_root=cache_root, allowed_ids=None)
    n_bands = _detect_n_bands(ds)
    encs = _build_encoders(which, device, se_ckpt, n_bands)

    n_written = 0
    for i in tqdm(range(len(ds)), desc=f"precompute {embedding_root.name}"):
        sample = ds[i]
        z_struct = encs["clip_struct"].encode(sample).numpy() if "clip_struct" in encs else None
        z_rgb = encs["clip_rgb"].encode(sample).numpy() if "clip_rgb" in encs else None
        if "hsi_se" in encs:
            z_spec = encs["hsi_se"].encode(sample).numpy()
        elif "hsi_mean_mlp" in encs:
            z_spec = encs["hsi_mean_mlp"].encode(sample).numpy()
        else:
            z_spec = None
        eb = EmbeddingBundle(
            tree_id=sample.tree_id, site=sample.site, species=sample.species,
            z_struct=z_struct, z_rgb=z_rgb, z_spec=z_spec,
        )
        out_dir = embedding_root / sample.site
        eb.save_npz(out_dir / f"{sample.tree_id}.npz")
        n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--embedding-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--which", nargs="+",
        default=["clip_rgb", "clip_struct", "hsi_mean_mlp"],
        choices=list(VALID),
    )
    ap.add_argument("--se-ckpt", type=Path, default=None)
    args = ap.parse_args()
    n = precompute_for_cache(
        cache_root=args.cache_root,
        embedding_root=args.embedding_root,
        device=args.device,
        which=args.which,
        se_ckpt=args.se_ckpt,
    )
    print(f"wrote {n} embedding bundles to {args.embedding_root}")


if __name__ == "__main__":
    main()
