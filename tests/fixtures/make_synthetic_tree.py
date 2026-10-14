"""Build a deterministic synthetic TreeSample for testing without real data."""
import numpy as np
from treeclip.core.tree_sample import TreeSample


def synthetic_tree(tree_id="t001", site="OSBS", species="Pinus_palustris", seed=0):
    rng = np.random.default_rng(seed)
    return TreeSample(
        tree_id=tree_id,
        site=site,
        species=species,
        point_cloud=rng.standard_normal((1024, 3)).astype(np.float32),
        rgb_crop=rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        hsi_crown=rng.random((16, 16, 426), dtype=np.float32),
        hsi_mask=rng.integers(0, 2, (16, 16), dtype=bool),
        lidar_density=3.15,
    )
