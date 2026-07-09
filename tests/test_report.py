"""Phase 2 report tests: markdown assembler + demo montage renders (no GPU)."""

import numpy as np
import torch
import torch.nn as nn

from auditor.report.demo_figure import render_before_after
from auditor.report.run_auditor import _markdown


def test_markdown_has_all_sections_naming_off():
    r = {
        "confound": "tube", "checkpoint": "x",
        "detect": {"reliance_gap": 0.29},
        "mitigate": {"worst_group_before": 0.67, "worst_group_after": 0.83,
                     "id_before": 0.96, "id_after": 0.97, "gap_before": 0.29, "gap_after": 0.14},
        "name": None, "demo_figure": "fig.png",
    }
    md = _markdown(r)
    assert "reliance gap = 0.290" in md
    assert "worst-group acc | 0.670 | 0.830" in md
    assert "skipped" in md  # naming-off
    assert "fig.png" in md


def test_markdown_includes_naming_when_present():
    r = {
        "confound": "tag", "checkpoint": "x",
        "detect": {"reliance_gap": 0.44},
        "mitigate": {"worst_group_before": 0.5, "worst_group_after": 0.61,
                     "id_before": 0.94, "id_after": 0.60, "gap_before": 0.44, "gap_after": 0.0},
        "name": {"precision": 0.98, "recall": 0.90, "control_hallucination": 0.02},
        "demo_figure": "f.png",
    }
    md = _markdown(r)
    assert "precision 0.98" in md and "recall 0.90" in md


def test_demo_montage_renders(tmp_path):
    """A tiny model that mimics the timm interface (forward_features + Grad-CAM target layer)."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential()
            self.features.norm5 = nn.Conv2d(3, 4, 3, padding=1)  # a real conv target layer
            self.classifier = nn.Linear(4, 1)

        def forward(self, x):
            f = self.features.norm5(x)
            return self.classifier(f.mean(dim=(2, 3)))

    m1, m2 = Tiny().eval(), Tiny().eval()
    imgs = [torch.randn(3, 16, 16) for _ in range(2)]
    out = render_before_after(m1, m2, imgs, "features.norm5", tmp_path / "ba.png")
    assert out.endswith("ba.png") and (tmp_path / "ba.png").exists()
