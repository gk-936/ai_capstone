"""Exp 2 geometry tests: occlusion box placement must be disjoint and in-bounds, the
fill must be confined to the box, and swap pairing must be deterministic and total."""

import numpy as np

from auditor.discovery.occlusion import _occlude, opposite_box, v4b_proxy_box


def test_v4b_proxy_box_is_centered_and_correct_size():
    box = v4b_proxy_box(64, frac=0.5)
    y0, x0, y1, x1 = box
    assert (y1 - y0) == (x1 - x0) == 32
    assert y0 == x0 == 16  # centered: (64-32)/2


def test_opposite_box_is_disjoint_from_top_left_box():
    # V4a's real case: a 20x20 box in a 32x32 crop. Two 20x20 squares cannot both fit
    # disjointly (20+20 > 32), so the irrelevant box must shrink to stay non-overlapping.
    box = (0, 0, 20, 20)
    opp = opposite_box(box, crop_size=32)
    y0, x0, y1, x1 = box
    oy0, ox0, oy1, ox1 = opp
    assert oy0 >= y1 or oy1 <= y0 or ox0 >= x1 or ox1 <= x0  # zero overlap, always
    assert (oy1 - oy0, ox1 - ox0) == (12, 12)  # largest disjoint square: 32-20


def test_opposite_box_matches_when_it_fits():
    # A box small relative to the crop CAN be mirrored at equal size.
    box = (0, 0, 8, 8)
    opp = opposite_box(box, crop_size=32)
    assert (opp[2] - opp[0], opp[3] - opp[1]) == (8, 8)


def test_opposite_box_stays_in_bounds():
    box = (0, 0, 20, 20)
    y0, x0, y1, x1 = opposite_box(box, crop_size=32)
    assert 0 <= y0 <= y1 <= 32
    assert 0 <= x0 <= x1 <= 32


def test_occlude_fills_only_the_box():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = _occlude(img, (0, 0, 20, 20))
    assert (out[:20, :20] == 128).all()
    assert (out[20:, :] == 0).all()
    assert (out[:, 20:] == 0).all()
    assert (img == 0).all()  # original untouched (copy, not in-place)
