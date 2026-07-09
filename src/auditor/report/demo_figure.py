"""Before/after Grad-CAM demo montage: original (shortcut-reliant) vs DFR-mitigated model.

For N confound-positive images, renders a 3-column grid: [image | CAM before DFR | CAM after DFR].
The story: after mitigation the saliency moves OFF the confound toward real anatomy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from auditor.attribution.gradcam import compute_gradcam
from auditor.attribution.localize_confound import _denormalize


def render_before_after(model_before, model_after, images: list[torch.Tensor],
                        target_layer: str, out_path: str | Path, titles=None) -> str:
    """Render [image | CAM(before) | CAM(after)] rows for each image tensor. Returns out path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["input", "Grad-CAM (before DFR)", "Grad-CAM (after DFR)"]
    for r, x in enumerate(images):
        img_u8 = _denormalize(x)
        cam_before = compute_gradcam(model_before, x, target_layer)
        cam_after = compute_gradcam(model_after, x, target_layer)

        axes[r, 0].imshow(img_u8)
        axes[r, 1].imshow(img_u8)
        axes[r, 1].imshow(cam_before, cmap="jet", alpha=0.45)
        axes[r, 2].imshow(img_u8)
        axes[r, 2].imshow(cam_after, cmap="jet", alpha=0.45)
        for c in range(3):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=9)
        if titles is not None:
            axes[r, 0].set_ylabel(titles[r], fontsize=8)

    fig.suptitle("Auditor: attention shifts off the confound after mitigation", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def collect_positive_images(index, confound: str, cfg, n: int, seed: int) -> list[torch.Tensor]:
    """Return n confound-positive image tensors for the demo (tag: tagged; tube: tube-positive)."""
    from auditor.data.nih import SplitIndex, make_dataset

    test = index.subset("test")
    if confound == "tube":
        rows = test[(test["label"] == 1) & (test["tube"] == 1)].head(n)
        sub = SplitIndex(df=rows.assign(split="demo"), target="Pneumothorax", n_pos={})
        ds = make_dataset(sub, "demo", cfg.data, train=False)
        return [ds[i][0] for i in range(len(ds))]
    # tag: take positives and stamp the tag on them
    from auditor.shortcut.inject import add_corner_tag
    rows = test[test["label"] == 1].head(n)
    sub = SplitIndex(df=rows.assign(split="demo"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "demo", cfg.data, train=False)
    return [add_corner_tag(ds[i][0], corner=cfg.shortcut.corner, size=cfg.shortcut.size)
            for i in range(len(ds))]
