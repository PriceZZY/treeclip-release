"""DECISIVE TEST: is CLIP strong on its HOME TURF (real RGB photos) vs on depth-projections?

If CLIP-on-RGB >> geometry, CLIP has a real reason to exist in the multimodal method (it's a
strong RGB encoder). If CLIP-on-RGB is also weak, then CLIP is a replaceable part and the whole
'TreeCLIP' framing is in question (user's challenge).

Encodes NEON rgb_crop (224x224x3 real photo) with the SAME frozen CLIP image encoder, then the
SAME supervised spatial-CV scoring (leave-region-out / leave-tile-band). NEON only (Weiser/FOR have no RGB).
Saves data/raw/biodiv_3dtrees/.. no — NEON is in data/cache. Output: runs/protocol_v2/gen/clip_rgb_<site>.npz
Usage: python gen_clip_rgb.py OSBS | MLBS
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
from PIL import Image
import torch, open_clip
from treeclip.core.tree_sample import TreeSample

ROOT = Path(__file__).resolve().parent.parent.parent
SITE = sys.argv[1] if len(sys.argv) > 1 else "MLBS"
OUT = ROOT / f"runs/protocol_v2/gen/clip_rgb_{SITE}.npz"


def main():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    model.eval().to("cuda")
    for p in model.parameters(): p.requires_grad_(False)
    X, sp, tid = [], [], []
    t0 = time.time(); n = 0
    paths = sorted((ROOT / "data/cache" / SITE).glob("*.npz"))
    with torch.no_grad():
        for pth in paths:
            t = TreeSample.load_npz(pth)
            if t.species is None or t.rgb_crop.size == 0 or t.rgb_crop.max() == 0:
                continue
            img = Image.fromarray(t.rgb_crop.astype(np.uint8))
            x = preprocess(img).unsqueeze(0).to("cuda")
            z = model.encode_image(x).squeeze(0)
            z = z / (torch.linalg.norm(z) + 1e-9)
            X.append(z.cpu().numpy()); sp.append(t.species); tid.append(t.tree_id); n += 1
            if n % 200 == 0: print(f"  {n} done {(time.time()-t0)/60:.1f}min", flush=True)
    X = np.array(X)
    np.savez_compressed(OUT, X=X, species=np.array(sp), tree_id=np.array(tid))
    print(f"DONE {SITE}: n={n} X={X.shape} normmean={np.linalg.norm(X,axis=1).mean():.3f} saved={OUT.name} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
