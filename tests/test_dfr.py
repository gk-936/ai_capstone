"""V5 DFR tests: group balancing, head fitting, and the standardization fold-back.

Pure-numpy/sklearn — no model or GPU needed. apply_dfr's backbone-untouched property is checked
with a tiny nn.Module.
"""

import numpy as np
import pandas as pd
import pytest

from auditor.mitigation.dfr import fit_dfr_head, group_balanced_sample


def _rows(n_per):
    """Build rows with all 4 (confound,label) groups at controllable sizes."""
    recs = []
    for (c, y), k in n_per.items():
        recs += [{"confound": c, "label": y, "i": i} for i in range(k)]
    return pd.DataFrame(recs)


def test_group_balanced_equalizes_four_groups():
    rows = _rows({(0, 0): 100, (0, 1): 10, (1, 0): 50, (1, 1): 8})
    bal = group_balanced_sample(rows, "confound", "label", seed=1)
    counts = bal.groupby(["confound", "label"]).size()
    assert set(counts.values) == {8}          # all groups equal to the smallest (8)
    assert len(bal) == 32


def test_group_balanced_deterministic():
    rows = _rows({(0, 0): 50, (0, 1): 50, (1, 0): 50, (1, 1): 50})
    a = group_balanced_sample(rows, "confound", "label", seed=7)
    b = group_balanced_sample(rows, "confound", "label", seed=7)
    assert a.equals(b)


def test_group_balanced_empty_group_raises():
    rows = _rows({(0, 0): 10, (0, 1): 10, (1, 0): 10, (1, 1): 0})
    with pytest.raises(ValueError):
        group_balanced_sample(rows, "confound", "label", seed=1)


def test_fit_dfr_head_recovers_separable_boundary():
    # Two clusters separable on feature 0; head should classify them in RAW feature space.
    rng = np.random.RandomState(0)
    n = 200
    pos = rng.normal(loc=[3, 0, 0], scale=0.3, size=(n, 3))
    neg = rng.normal(loc=[-3, 0, 0], scale=0.3, size=(n, 3))
    feats = np.vstack([pos, neg]).astype(np.float32)
    labels = np.array([1] * n + [0] * n)

    weight, bias = fit_dfr_head(feats, labels, seed=0)
    logits = feats @ weight + bias           # apply in raw space (fold-back must be correct)
    preds = (logits > 0).astype(int)
    acc = (preds == labels).mean()
    assert acc > 0.98


def test_apply_dfr_leaves_backbone_untouched():
    import torch
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 3)
            self.classifier = nn.Linear(3, 1)

    from auditor.mitigation.dfr import apply_dfr

    m = Tiny()
    before = m.backbone.weight.detach().clone()
    apply_dfr(m, weight=np.ones(3, dtype=np.float32), bias=np.float32(0.5))
    assert torch.equal(m.backbone.weight, before)                 # backbone unchanged
    assert torch.allclose(m.classifier.weight, torch.ones(1, 3))  # head replaced
    assert torch.allclose(m.classifier.bias, torch.tensor([0.5]))
