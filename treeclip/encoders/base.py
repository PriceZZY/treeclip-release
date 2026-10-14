"""Encoder abstract base class and EmbeddingBundle (per-tree multi-modal embeddings)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from treeclip.core.tree_sample import TreeSample


SPECIES_NULL_SENTINEL = "__NULL__"
_MOD_NULL = np.array([], dtype=np.float32)


class Encoder(ABC):
    """Abstract base: one encoder produces one fixed-dim embedding per TreeSample."""
    name: str
    embedding_dim: int

    @abstractmethod
    def encode(self, sample: TreeSample) -> torch.Tensor:
        """Return a 1-D tensor of shape (embedding_dim,)."""
        ...


@dataclass
class EmbeddingBundle:
    """Per-tree precomputed embeddings, one per modality. Any modality may be None
    if not extracted for this tree (e.g. encoder unavailable, modality empty)."""
    tree_id: str
    site: str
    species: Optional[str]
    z_struct: Optional[np.ndarray]
    z_rgb: Optional[np.ndarray]
    z_spec: Optional[np.ndarray]

    def save_npz(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            tree_id=np.array(self.tree_id),
            site=np.array(self.site),
            species=np.array(self.species if self.species is not None else SPECIES_NULL_SENTINEL),
            z_struct=self.z_struct if self.z_struct is not None else _MOD_NULL,
            z_rgb=self.z_rgb if self.z_rgb is not None else _MOD_NULL,
            z_spec=self.z_spec if self.z_spec is not None else _MOD_NULL,
            has_struct=np.array(self.z_struct is not None),
            has_rgb=np.array(self.z_rgb is not None),
            has_spec=np.array(self.z_spec is not None),
        )

    @classmethod
    def load_npz(cls, path: Path | str) -> "EmbeddingBundle":
        d = np.load(path, allow_pickle=False)
        species_raw = str(d["species"])
        return cls(
            tree_id=str(d["tree_id"]),
            site=str(d["site"]),
            species=None if species_raw == SPECIES_NULL_SENTINEL else species_raw,
            z_struct=d["z_struct"] if bool(d["has_struct"]) else None,
            z_rgb=d["z_rgb"] if bool(d["has_rgb"]) else None,
            z_spec=d["z_spec"] if bool(d["has_spec"]) else None,
        )
