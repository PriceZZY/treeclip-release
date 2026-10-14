"""Per-tree inference cost of the frozen pipeline (PFG round-1, reviewer 2's time-
complexity comment): wall-clock for (i) voxel downsample, (ii) 6-view render + frozen
CLIP encode (GPU), (iii) the matched geometric descriptor, on a fixed random sample of
FOR-species20K clouds. No training occurs anywhere in the pipeline; this measures the
only compute the method actually incurs. Output: runs/protocol_v2/gen/inference_timing.json
"""
from __future__ import annotations
import sys, zipfile, io, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, laspy, torch
from render_validation import voxel_downsample
from data import geo_features
from treeclip.encoders.render_views import render_n_views_pca
import open_clip
from PIL import Image

D = ROOT / "data/raw/for_species20k"
N = 100


def main():
    meta = pd.read_csv(D / "tree_metadata_dev.csv")
    z = zipfile.ZipFile(D / "dev.zip"); names = set(z.namelist())
    rng = np.random.RandomState(0)
    order = rng.permutation(len(meta))
    model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    model.eval().cuda()
    for prm in model.parameters(): prm.requires_grad_(False)
    def encode6(p):
        views = render_n_views_pca(p, 6, 224)
        pil = [Image.fromarray(np.stack([v, v, v], axis=-1)) for v in views]
        x = torch.stack([clip_pre(q) for q in pil]).cuda()
        with torch.no_grad():
            z = model.encode_image(x); z = z / (z.norm(dim=1, keepdim=True) + 1e-9)
        return z.mean(0).cpu().numpy()
    # CUDA warm-up (3 clouds, untimed)
    warm = 0
    rows = list(meta.itertuples(index=False))
    t_load, t_vox, t_enc, t_geo, npts_raw, npts_ds = [], [], [], [], [], []
    for oi in order:
        row = rows[oi]
        e = f"{int(row.treeID):05d}.laz"
        if e not in names: continue
        t0 = time.perf_counter()
        las = laspy.read(io.BytesIO(z.read(e)))
        pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        t1 = time.perf_counter()
        if pts.shape[0] < 3: continue
        p = voxel_downsample(pts); t2 = time.perf_counter()
        _ = encode6(p)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        _ = geo_features(p); t4 = time.perf_counter()
        if warm < 3:
            warm += 1; continue
        t_load.append(t1 - t0); t_vox.append(t2 - t1); t_enc.append(t3 - t2); t_geo.append(t4 - t3)
        npts_raw.append(int(pts.shape[0])); npts_ds.append(int(p.shape[0]))
        if len(t_enc) >= N: break
    def st(v): 
        v = np.array(v) * 1000
        return {"mean_ms": round(float(v.mean()), 2), "median_ms": round(float(np.median(v)), 2),
                "std_ms": round(float(v.std()), 2)}
    per_tree_s = float(np.mean(t_vox) + np.mean(t_enc))
    out = {"n_timed": len(t_enc), "gpu": torch.cuda.get_device_name(0),
           "laz_load": st(t_load), "voxel_downsample": st(t_vox),
           "render6_plus_clip_encode_batched": st(t_enc), "geo_descriptor": st(t_geo),
           "pipeline_per_tree_ms_excl_io": round(per_tree_s * 1000, 2),
           "extrapolated_full_17707_trees_min": round(17707 * per_tree_s / 60, 1),
           "extrapolated_analyzable_13497_min": round(13497 * per_tree_s / 60, 1),
           "median_raw_points": int(np.median(npts_raw)), "median_downsampled_points": int(np.median(npts_ds))}
    json.dump(out, open(ROOT / "runs/protocol_v2/gen/inference_timing.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
