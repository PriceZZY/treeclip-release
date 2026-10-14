"""PyTorch Dataset over per-tree npz cache, with optional id allowlist."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
from torch.utils.data import Dataset
from treeclip.core.tree_sample import TreeSample


class TreeDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        allowed_ids: Optional[Iterable[str]] = None,
    ):
        self.cache_root = Path(cache_root)
        all_paths = sorted(self.cache_root.glob("*/*.npz"))
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            self.paths = [p for p in all_paths if p.stem in allowed]
        else:
            self.paths = all_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> TreeSample:
        return TreeSample.load_npz(self.paths[idx])
