"""Three fusion strategies operating on per-tree EmbeddingBundle.

All three take pre-extracted modality embeddings (z_struct, z_rgb, z_spec) and a
list of species labels, then train a small head on a train split and report
macro-F1 on the val split. Encoders are frozen — fusion learns only the head.

This is the §4 of the design spec, condensed for actual implementation. Designed
so the experiment runner can flip fusion via a single string flag.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score


MODS = ("z_struct", "z_rgb", "z_spec")


def _stack_modalities(bundles, modalities: list[str]) -> np.ndarray:
    """Concatenate selected modalities for each tree. Missing modalities (None)
    are filled with zeros of the modality's expected dimension."""
    # Determine dim of each modality from any present sample
    dims: dict[str, int] = {}
    for m in modalities:
        for b in bundles:
            arr = getattr(b, m)
            if arr is not None and arr.size > 0:
                dims[m] = arr.size
                break
        if m not in dims:
            raise ValueError(f"no sample has modality {m}; can't determine dim")
    out = []
    for b in bundles:
        parts = []
        for m in modalities:
            arr = getattr(b, m)
            if arr is None or arr.size == 0:
                parts.append(np.zeros(dims[m], dtype=np.float32))
            else:
                parts.append(arr.astype(np.float32))
        out.append(np.concatenate(parts))
    return np.stack(out)


@dataclass
class FusionResult:
    macro_f1: float
    accuracy: float


def fit_late_fusion(
    bundles_train, y_train, bundles_val, y_val,
    modalities: list[str],
    seed: int = 42, epochs: int = 50, device: str = "cpu",
) -> FusionResult:
    """Late fusion: per-modality MLP head -> averaged logits.

    Implementation: train one shared classifier head per modality (concat-input not used),
    average their softmax probabilities at inference. Class-balanced via inverse-freq weights.
    """
    classes = sorted(set(y_train))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    yt = np.array([cls_to_idx[c] for c in y_train])
    yv_idx = np.array([cls_to_idx.get(c, -1) for c in y_val])

    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)

    per_mod_probs = []
    for m in modalities:
        Xt = _stack_modalities(bundles_train, [m])
        Xv = _stack_modalities(bundles_val, [m])
        d = Xt.shape[1]
        head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, n_classes),
        ).to(device)
        opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        # class weights
        counts = np.array([(yt == i).sum() for i in range(n_classes)], dtype=np.float32)
        weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
        ce = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
        Xt_t = torch.from_numpy(Xt.astype(np.float32)).to(device)
        yt_t = torch.from_numpy(yt).to(device)
        head.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = head(Xt_t)
            loss = ce(logits, yt_t)
            loss.backward()
            opt.step()
        head.eval()
        Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
        with torch.no_grad():
            probs = torch.softmax(head(Xv_t), dim=-1).cpu().numpy()
        per_mod_probs.append(probs)
    avg = np.mean(per_mod_probs, axis=0)
    yp = np.argmax(avg, axis=-1)
    return FusionResult(
        macro_f1=float(f1_score(yv_idx, yp, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(yv_idx, yp)),
    )


def fit_early_fusion(
    bundles_train, y_train, bundles_val, y_val,
    modalities: list[str],
    seed: int = 42, epochs: int = 50, device: str = "cpu",
) -> FusionResult:
    """Early fusion: concat all modalities into one vector -> single MLP."""
    classes = sorted(set(y_train))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    yt = np.array([cls_to_idx[c] for c in y_train])
    yv_idx = np.array([cls_to_idx.get(c, -1) for c in y_val])

    Xt = _stack_modalities(bundles_train, modalities)
    Xv = _stack_modalities(bundles_val, modalities)
    d = Xt.shape[1]

    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Sequential(
        nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    ).to(device)
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.array([(yt == i).sum() for i in range(n_classes)], dtype=np.float32)
    weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    Xt_t = torch.from_numpy(Xt.astype(np.float32)).to(device)
    yt_t = torch.from_numpy(yt).to(device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(Xt_t)
        loss = ce(logits, yt_t)
        loss.backward()
        opt.step()
    head.eval()
    Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
    with torch.no_grad():
        yp = head(Xv_t).argmax(dim=-1).cpu().numpy()
    return FusionResult(
        macro_f1=float(f1_score(yv_idx, yp, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(yv_idx, yp)),
    )


def fit_xattn_fusion(
    bundles_train, y_train, bundles_val, y_val,
    modalities: list[str],
    seed: int = 42, epochs: int = 80, device: str = "cpu", d_model: int = 256,
) -> FusionResult:
    """Cross-attention fusion: per-modality projection to d_model, then a TransformerEncoder
    layer treats modalities as a sequence of M tokens, mean-pool, then classify."""
    classes = sorted(set(y_train))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    yt = np.array([cls_to_idx[c] for c in y_train])
    yv_idx = np.array([cls_to_idx.get(c, -1) for c in y_val])

    # Build per-modality input tensors
    def stack_one_mod(bundles, mod):
        return _stack_modalities(bundles, [mod])
    Xs_train = [stack_one_mod(bundles_train, m) for m in modalities]
    Xs_val = [stack_one_mod(bundles_val, m) for m in modalities]
    dims = [x.shape[1] for x in Xs_train]

    torch.manual_seed(seed); np.random.seed(seed)

    class XAttnHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.projs = nn.ModuleList([nn.Linear(d, d_model) for d in dims])
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=512,
                batch_first=True, dropout=0.1,
            )
            self.xattn = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.cls = nn.Linear(d_model, n_classes)

        def forward(self, xs):
            tokens = torch.stack([p(x) for p, x in zip(self.projs, xs)], dim=1)
            fused = self.xattn(tokens).mean(dim=1)
            return self.cls(fused)

    head = XAttnHead().to(device)
    opt = optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-4)
    counts = np.array([(yt == i).sum() for i in range(n_classes)], dtype=np.float32)
    weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    Xs_train_t = [torch.from_numpy(x.astype(np.float32)).to(device) for x in Xs_train]
    yt_t = torch.from_numpy(yt).to(device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(Xs_train_t)
        loss = ce(logits, yt_t)
        loss.backward()
        opt.step()
    head.eval()
    Xs_val_t = [torch.from_numpy(x.astype(np.float32)).to(device) for x in Xs_val]
    with torch.no_grad():
        yp = head(Xs_val_t).argmax(dim=-1).cpu().numpy()
    return FusionResult(
        macro_f1=float(f1_score(yv_idx, yp, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(yv_idx, yp)),
    )


FUSION_FNS = {
    "late": fit_late_fusion,
    "early": fit_early_fusion,
    "xattn": fit_xattn_fusion,
}
