"""GATE-S1: render-validation PNGs. Render trees IDENTICALLY via render_pca_depth_views so the
user can eyeball that the pipeline produces sane depth images on the NEW dense-TLS data before any
CLIP number is trusted. Also renders a voxel-downsampled FOR variant to inform the density choice.
CPU-only. Saves runs/render_validation/<set>/.
"""
from __future__ import annotations
import sys, zipfile, io
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import laspy, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from treeclip.core.tree_sample import TreeSample
from treeclip.encoders.render_views import render_pca_depth_views as R

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/render_validation"; OUT.mkdir(parents=True, exist_ok=True)


def voxel_downsample(pts, n_target=4096, vox=0.1):
    """uniform voxel grid downsample (Weiser-like density); fallback random if still large."""
    keys = np.floor(pts / vox).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    p = pts[np.sort(idx)]
    if p.shape[0] > n_target:
        sel = np.random.RandomState(0).choice(p.shape[0], n_target, replace=False)
        p = p[sel]
    return p


def grid(samples, title, path):
    """samples: list of (label, pts). Render 2 views each, save a montage."""
    n = len(samples)
    fig, ax = plt.subplots(n, 2, figsize=(4, 2 * n))
    if n == 1: ax = ax.reshape(1, 2)
    for i, (lab, pts) in enumerate(samples):
        v = R(pts.astype(np.float32), 224)
        for j in range(2):
            ax[i, j].imshow(v[j], cmap="viridis"); ax[i, j].axis("off")
        ax[i, 0].set_title(f"{lab} (n={pts.shape[0]}) v0", fontsize=7, loc="left")
        ax[i, 1].set_title("v1", fontsize=7, loc="left")
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path)


def main():
    # Weiser anchor
    import glob
    wf = sorted(glob.glob(str(ROOT / "data/cache_weiser/*/*.npz")))[:6]
    grid([(f"W{i}", TreeSample.load_npz(p).point_cloud) for i, p in enumerate(wf)],
         "Weiser ANCHOR (ALS, what CLIP was validated on)", OUT / "weiser_anchor.png")

    # FOR-species20K: full density vs voxel-downsampled (pick diverse species)
    D = ROOT / "data/raw/for_species20k"
    meta = pd.read_csv(D / "tree_metadata_dev.csv")
    z = zipfile.ZipFile(D / "dev.zip"); names = set(z.namelist())
    pick = meta.drop_duplicates("species").head(6)
    full, down = [], []
    for row in pick.itertuples(index=False):
        e = f"{int(row.treeID):05d}.laz"
        if e not in names: continue
        las = laspy.read(io.BytesIO(z.read(e)))
        pts = np.vstack([las.x, las.y, las.z]).T
        full.append((str(row.species)[:10], pts))
        down.append((str(row.species)[:10], voxel_downsample(pts)))
    grid(full, "FOR-species20K FULL density (dense TLS)", OUT / "for_full.png")
    grid(down, "FOR-species20K VOXEL-downsampled (Weiser-like budget)", OUT / "for_voxel.png")
    print("DONE. Review PNGs in", OUT)


if __name__ == "__main__":
    main()
