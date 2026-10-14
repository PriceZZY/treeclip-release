"""Deterministic train/val/test split within a single site."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import random
import yaml


@dataclass
class WithinSiteSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def make_within_site_split(
    tree_ids: list[str],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> WithinSiteSplit:
    ids = sorted(tree_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val :]
    split = WithinSiteSplit(train=train, val=val, test=test)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.safe_dump(
                {"train": split.train, "val": split.val, "test": split.test}, f
            )
    return split
