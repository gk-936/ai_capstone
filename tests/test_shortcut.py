"""Synthetic-spine tests for V2 confound injection. These must not silently break.

No torch model or real data needed — we test the injection geometry, the correlation rule's
statistics, determinism, and the balanced-subset guarantee directly.
"""

import numpy as np
import pandas as pd
import torch

from auditor.data.nih import SplitIndex
from auditor.evaluation.reliance import balanced_eval_subset
from auditor.shortcut.inject import _WHITE_NORMALIZED, add_corner_tag, tag_decision


# ------------------------------- add_corner_tag geometry ------------------------------

def test_tag_writes_correct_corner_only():
    x = torch.zeros(3, 224, 224)
    out = add_corner_tag(x, corner="top-left", size=20)
    # Tagged region equals the white-normalized value...
    assert torch.allclose(out[:, :20, :20], _WHITE_NORMALIZED.expand(3, 20, 20))
    # ...and everything outside it is untouched.
    assert out[:, 20:, :].abs().sum() == 0
    assert out[:, :, 20:].abs().sum() == 0
    # Original tensor is not mutated.
    assert x.abs().sum() == 0


def test_each_corner_lands_where_expected():
    x = torch.zeros(3, 100, 100)
    checks = {
        "top-right": (slice(0, 10), slice(90, 100)),
        "bottom-left": (slice(90, 100), slice(0, 10)),
        "bottom-right": (slice(90, 100), slice(90, 100)),
    }
    for corner, (ys, xs) in checks.items():
        out = add_corner_tag(x, corner=corner, size=10)
        assert out[:, ys, xs].abs().sum() > 0          # corner is written
        masked = out.clone()
        masked[:, ys, xs] = 0                          # zero out the tagged corner
        assert masked.abs().sum() == 0                 # nothing written outside it


def test_opacity_one_matches_default_hard_overwrite():
    x = torch.rand(3, 50, 50)
    hard = add_corner_tag(x, corner="top-left", size=10)
    blended = add_corner_tag(x, corner="top-left", size=10, opacity=1.0)
    assert torch.allclose(hard, blended)


def test_opacity_zero_is_a_no_op():
    x = torch.rand(3, 50, 50)
    out = add_corner_tag(x, corner="top-left", size=10, opacity=0.0)
    assert torch.allclose(out, x)


def test_opacity_blends_linearly():
    x = torch.zeros(3, 50, 50)
    out = add_corner_tag(x, corner="top-left", size=10, opacity=0.5)
    expected = 0.5 * _WHITE_NORMALIZED.expand(3, 10, 10)
    assert torch.allclose(out[:, :10, :10], expected)
    assert out[:, 10:, :].abs().sum() == 0  # untouched outside the box


def test_bad_shape_and_corner_raise():
    import pytest

    with pytest.raises(ValueError):
        add_corner_tag(torch.zeros(1, 10, 10))          # not 3-channel
    with pytest.raises(ValueError):
        add_corner_tag(torch.zeros(3, 10, 10), corner="middle")


# ------------------------------- correlation rule -------------------------------------

def test_p_one_tags_all_positives_no_negatives():
    for i in range(200):
        name = f"img_{i}.png"
        assert tag_decision(name, label=1, mode="aligned", p=1.0, seed=1) is True
        assert tag_decision(name, label=0, mode="aligned", p=1.0, seed=1) is False


def test_p_half_tags_about_half():
    names = [f"img_{i}.png" for i in range(2000)]
    frac_pos = np.mean([tag_decision(n, 1, "aligned", 0.5, seed=7) for n in names])
    assert 0.45 < frac_pos < 0.55  # ~half of positives tagged at p=0.5


def test_flipped_inverts_correlation():
    # At p=1.0 flipped: negatives always tagged, positives never.
    assert tag_decision("a.png", 0, "flipped", 1.0, seed=3) is True
    assert tag_decision("a.png", 1, "flipped", 1.0, seed=3) is False


def test_removed_never_tags():
    assert tag_decision("a.png", 1, "removed", 0.95, seed=1) is False


def test_decision_is_deterministic():
    a = tag_decision("same.png", 1, "aligned", 0.7, seed=42)
    b = tag_decision("same.png", 1, "aligned", 0.7, seed=42)
    assert a == b


# ------------------------------- balanced subset --------------------------------------

def _fake_index(n_pos: int, n_neg: int) -> SplitIndex:
    rows = pd.DataFrame(
        {
            "image": [f"p{i}.png" for i in range(n_pos)] + [f"n{i}.png" for i in range(n_neg)],
            "patient_id": [f"P{i}" for i in range(n_pos + n_neg)],
            "label": [1] * n_pos + [0] * n_neg,
            "split": ["test"] * (n_pos + n_neg),
        }
    )
    return SplitIndex(df=rows, target="Pneumothorax", n_pos={"test": n_pos})


def test_balanced_subset_is_5050():
    idx = _fake_index(n_pos=50, n_neg=500)
    sub = balanced_eval_subset(idx, "test", seed=1)
    assert (sub["label"] == 1).sum() == (sub["label"] == 0).sum() == 50  # limited by minority


def test_balanced_subset_respects_per_class_cap():
    idx = _fake_index(n_pos=500, n_neg=500)
    sub = balanced_eval_subset(idx, "test", seed=1, per_class=100)
    assert (sub["label"] == 1).sum() == (sub["label"] == 0).sum() == 100
