"""Mitigation baselines for the Phase 3 comparison table: ERM and JTT.

All operate on FROZEN backbone features (like DFR), so no backbone retraining and no train-time
group labels are needed.

  ERM  — no mitigation: the model's original head. The floor.
  JTT  — Just Train Twice (Liu et al. 2021): retrain the last layer with examples the ERM model
         misclassifies UPWEIGHTED. Approximates worst-group robustness without group labels.
  DFR  — group-balanced last-layer retrain (in dfr.py).
"""

from __future__ import annotations

import numpy as np


def erm_predictions(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """Binary predictions from the original (ERM) head applied to frozen features."""
    return (features @ weight + bias > 0).astype(int)


def fit_jtt_head(features: np.ndarray, labels: np.ndarray, erm_weight: np.ndarray,
                 erm_bias: float, upweight: float = 20.0, C: float = 1.0, seed: int = 1337):
    """JTT: refit the last layer, upweighting examples the ERM head gets wrong.

    Returns (weight, bias) in raw feature space (standardization folded back), same contract as
    fit_dfr_head so it drops into apply_dfr.
    """
    from sklearn.linear_model import LogisticRegression

    erm_pred = erm_predictions(features, erm_weight, erm_bias)
    wrong = erm_pred != labels
    sample_weight = np.where(wrong, upweight, 1.0)

    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-6
    z = (features - mu) / sigma
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(z, labels, sample_weight=sample_weight)

    w_std = clf.coef_[0]
    b_std = float(clf.intercept_[0])
    weight = w_std / sigma
    bias = b_std - float((w_std * mu / sigma).sum())
    return weight.astype(np.float32), np.float32(bias)


def current_head(model):
    """Return (weight, bias) of the model's current final linear head as numpy arrays."""
    w = model.classifier.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.classifier.bias.detach().cpu().numpy().reshape(-1)[0])
    return w.astype(np.float32), np.float32(b)
