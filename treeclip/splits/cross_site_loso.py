"""Leave-One-Site-Out cross-site split — paper's generalization battleground."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LOSOFold:
    test_site: str
    train: list[str]
    test: list[str]


def make_loso_splits(trees_by_site: dict[str, list[str]]) -> list[LOSOFold]:
    sites = sorted(trees_by_site.keys())
    folds: list[LOSOFold] = []
    for test_site in sites:
        train_ids: list[str] = []
        for s in sites:
            if s == test_site:
                continue
            train_ids.extend(trees_by_site[s])
        folds.append(
            LOSOFold(
                test_site=test_site,
                train=sorted(train_ids),
                test=sorted(trees_by_site[test_site]),
            )
        )
    return folds
