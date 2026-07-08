"""V3 geometry tests: tag_box (all corners), peak_in_region, salient_crop, mass_fraction.

Pure-numpy, no model/GPU — these lock the localization logic that the >80% gate depends on.
"""

import numpy as np

from auditor.attribution.gradcam import (
    mass_fraction,
    peak_in_region,
    salient_crop,
    tag_box,
)
from auditor.shortcut.inject import add_corner_tag
import torch


def test_tag_box_all_corners_match_injection():
    """tag_box must describe exactly where add_corner_tag writes, for every corner."""
    h = w = 100
    size = 20
    for corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
        box = tag_box(corner, size, h, w)
        y0, x0, y1, x1 = box
        # Inject a tag, then confirm the bright pixels lie exactly within the box.
        x = torch.zeros(3, h, w)
        tagged = add_corner_tag(x, corner=corner, size=size)
        bright = (tagged.abs().sum(0) > 0).numpy()   # HxW mask of written pixels
        inside = np.zeros((h, w), dtype=bool)
        inside[y0:y1, x0:x1] = True
        assert np.array_equal(bright, inside), f"box mismatch for {corner}"


def test_peak_in_region_true_when_max_inside():
    sal = np.zeros((50, 50), dtype=np.float32)
    sal[5, 7] = 1.0                       # peak in top-left
    assert peak_in_region(sal, (0, 0, 20, 20)) is True
    assert peak_in_region(sal, (30, 30, 50, 50)) is False


def test_peak_in_region_boundary_is_half_open():
    sal = np.zeros((10, 10), dtype=np.float32)
    sal[5, 5] = 1.0
    assert peak_in_region(sal, (0, 0, 6, 6)) is True     # 5 < 6 inside
    assert peak_in_region(sal, (0, 0, 5, 5)) is False    # 5 not < 5


def test_salient_crop_size_and_centering():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    sal = np.zeros((100, 100), dtype=np.float32)
    sal[50, 50] = 1.0
    crop = salient_crop(img, sal, size=20)
    assert crop.shape == (20, 20, 3)


def test_salient_crop_clamps_at_edge():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    sal = np.zeros((100, 100), dtype=np.float32)
    sal[2, 2] = 1.0                       # peak near top-left corner
    crop = salient_crop(img, sal, size=30)
    assert crop.shape == (30, 30, 3)      # still full size, clamped into frame


def test_mass_fraction_all_in_box_is_one():
    sal = np.zeros((10, 10), dtype=np.float32)
    sal[1:3, 1:3] = 1.0
    assert mass_fraction(sal, (0, 0, 5, 5)) == 1.0
    assert mass_fraction(sal, (5, 5, 10, 10)) == 0.0


def test_mass_fraction_zero_saliency_safe():
    sal = np.zeros((10, 10), dtype=np.float32)
    assert mass_fraction(sal, (0, 0, 5, 5)) == 0.0
