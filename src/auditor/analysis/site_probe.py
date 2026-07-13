"""Experiment A — hospital-detector probe + site-direction projection (pure, torch-free).

A1 fits a probe to classify NIH vs PadChest on frozen features (site is decodable — expected).
A2 is the falsifying control: project the site direction out of the disease-test features and
re-score the FROZEN disease head. If OOD AUC does not move, the disease head is not riding the
site shortcut -> bounds the shortcut slice near zero.
"""

from __future__ import annotations

import numpy as np


def project_out(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return `features` with the component along `direction` removed (row-wise)."""
    u = direction / (np.linalg.norm(direction) + 1e-12)
    return features - np.outer(features @ u, u)


def site_direction(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """Unit vector pointing from mean(feats_b) to mean(feats_a) (a simple, deterministic axis)."""
    d = feats_a.mean(axis=0) - feats_b.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def score_features(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """Linear-head scores in raw feature space."""
    return features @ weight + bias
