"""TreeSample: the single-tree training unit, loadable from npz cache."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

SPECIES_NULL_SENTINEL = "__NULL__"


@dataclass
class TreeSample:
    tree_id: str
    site: str
    species: Optional[str]
    point_cloud: np.ndarray   # (N, 3) float32, centered at origin in XY
    rgb_crop: np.ndarray      # (224, 224, 3) uint8
    hsi_crown: np.ndarray     # (H, W, B) float32, raw reflectance
    hsi_mask: np.ndarray      # (H, W) bool, crown pixels
    lidar_density: float      # points / m^2

    def save_npz(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            tree_id=np.array(self.tree_id),
            site=np.array(self.site),
            species=np.array(
                self.species if self.species is not None else SPECIES_NULL_SENTINEL
            ),
            point_cloud=self.point_cloud,
            rgb_crop=self.rgb_crop,
            hsi_crown=self.hsi_crown,
            hsi_mask=self.hsi_mask,
            lidar_density=np.array(self.lidar_density, dtype=np.float64),
        )

    @classmethod
    def load_npz(cls, path: Path | str) -> "TreeSample":
        d = np.load(path, allow_pickle=False)
        species_raw = str(d["species"])
        return cls(
            tree_id=str(d["tree_id"]),
            site=str(d["site"]),
            species=None if species_raw == SPECIES_NULL_SENTINEL else species_raw,
            point_cloud=d["point_cloud"],
            rgb_crop=d["rgb_crop"],
            hsi_crown=d["hsi_crown"],
            hsi_mask=d["hsi_mask"],
            lidar_density=float(d["lidar_density"]),
        )
