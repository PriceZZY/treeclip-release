from treeclip.splits.cross_site_loso import make_loso_splits


def test_loso_produces_three_folds():
    trees_by_site = {
        "OSBS": [f"o{i}" for i in range(10)],
        "MLBS": [f"m{i}" for i in range(8)],
        "TALL": [f"t{i}" for i in range(12)],
    }
    folds = make_loso_splits(trees_by_site)
    assert len(folds) == 3
    names = {f.test_site for f in folds}
    assert names == {"OSBS", "MLBS", "TALL"}


def test_loso_fold_train_excludes_test_site():
    trees_by_site = {
        "OSBS": ["o1", "o2"],
        "MLBS": ["m1"],
        "TALL": ["t1", "t2", "t3"],
    }
    folds = make_loso_splits(trees_by_site)
    for fold in folds:
        assert set(fold.test) == set(trees_by_site[fold.test_site])
        for sp in trees_by_site:
            if sp == fold.test_site:
                continue
            assert set(trees_by_site[sp]).issubset(set(fold.train))
        assert set(fold.train).isdisjoint(set(fold.test))
