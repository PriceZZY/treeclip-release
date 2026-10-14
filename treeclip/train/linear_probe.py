"""Linear-probe validation: train a logistic regression on embeddings, report macro-F1
on val set. Used as the kill criterion for SE warm-start (design spec §3c, validation protocol)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def linear_probe_macro_f1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> float:
    """Returns macro-F1 of a linear-probe classifier trained on (X_train, y_train) and
    evaluated on (X_val, y_val). Drops rows with any non-finite features."""
    keep_train = np.isfinite(X_train).all(axis=1)
    keep_val = np.isfinite(X_val).all(axis=1)
    Xt, yt = X_train[keep_train], y_train[keep_train]
    Xv, yv = X_val[keep_val], y_val[keep_val]
    if len(Xt) == 0 or len(Xv) == 0:
        return 0.0
    if len(np.unique(yt)) < 2:
        return 0.0
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(Xt, yt)
    yp = clf.predict(Xv)
    return float(f1_score(yv, yp, average="macro", zero_division=0))
