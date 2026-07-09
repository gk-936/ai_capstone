"""Deep Feature Reweighting (V5) — confound-agnostic last-layer mitigation.

DFR (Kirichenko et al. 2022): freeze the backbone, retrain ONLY the final linear layer on a
group-balanced set. The backbone already encodes core features; the head just weights the
spurious ones too heavily. Rebalancing the groups breaks the confound<->label correlation and
forces the head to reweight toward core signal.

Pure/near-pure functions kept testable; the model-touching bits import torch lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_balanced_sample(rows: pd.DataFrame, confound_col: str, label_col: str,
                          seed: int = 1337, per_group: int | None = None) -> pd.DataFrame:
    """Equal-sized subset across the NON-EMPTY (confound x label) groups, deterministic per seed.

    Balances across whichever of the four (confound x label) groups actually exist. This matters
    for real confounds where a group is structurally absent — e.g. chest tubes occur almost only
    on pneumothorax-positives, so (tube=1, label=0) is empty; we balance the remaining groups.
    per_group defaults to the smallest non-empty group's size (no upsampling).
    """
    groups = {}
    for c in (0, 1):
        for y in (0, 1):
            g = rows[(rows[confound_col] == c) & (rows[label_col] == y)]
            if len(g) > 0:
                groups[(c, y)] = g
    if len(groups) < 2:
        raise ValueError("need at least 2 non-empty (confound,label) groups to balance")
    smallest = min(len(g) for g in groups.values())
    n = smallest if per_group is None else min(per_group, smallest)
    parts = [g.sample(n=n, random_state=seed) for g in groups.values()]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def extract_features(model, loader, device):
    """Frozen penultimate features (N x D) and labels (N,) for a data loader."""
    import torch

    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = model.forward_features(x)
            pooled = model.forward_head(f, pre_logits=True)  # N x D penultimate
            feats.append(pooled.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def fit_dfr_head(features: np.ndarray, labels: np.ndarray, C: float = 1.0, seed: int = 1337):
    """Fit a logistic-regression head on frozen features (DFR). Returns (weight, bias) as arrays.

    Features are standardized; the standardization is folded back into the returned weight/bias so
    the head can be dropped straight into the model's raw-feature space.
    """
    from sklearn.linear_model import LogisticRegression

    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-6
    z = (features - mu) / sigma
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(z, labels)

    # Map w·((x-mu)/sigma)+b  ->  (w/sigma)·x + (b - w·mu/sigma) in raw feature space.
    w_std = clf.coef_[0]
    b_std = float(clf.intercept_[0])
    weight = w_std / sigma
    bias = b_std - float((w_std * mu / sigma).sum())
    return weight.astype(np.float32), np.float32(bias)


def apply_dfr(model, weight: np.ndarray, bias: float):
    """Replace the model's final linear head with the DFR weight/bias; backbone untouched."""
    import torch

    head = model.classifier  # timm DenseNet final Linear(D, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight, dtype=head.weight.dtype).view(1, -1))
        head.bias.copy_(torch.tensor([bias], dtype=head.bias.dtype))
    return model
