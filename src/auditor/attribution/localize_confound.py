"""V3 experiment: does Grad-CAM on the V2 model localize the injected tag?

Runs the trained (V2) model over N tagged positive test images and the SAME images untagged
(control). Reports peak-in-box hit-rate for both, plus mean mass-fraction, and exports a few
heatmap overlays + salient crops as artifacts (the crops feed V4).
"""

from __future__ import annotations

import numpy as np
import torch

from auditor.attribution.gradcam import (
    compute_gradcam,
    mass_fraction,
    peak_in_region,
    salient_crop,
    tag_box,
)
from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import build_index, make_dataset
from auditor.data.torch_dataset import _MEAN, _STD
from auditor.evaluation.reliance import balanced_eval_subset
from auditor.shortcut.inject import add_corner_tag


def _build_model(backbone: str, checkpoint: str, device: str):
    import timm

    model = timm.create_model(backbone, pretrained=False, num_classes=1)
    state = torch.load(resolve_path(checkpoint), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def _denormalize(x: torch.Tensor) -> np.ndarray:
    """3xHxW normalized tensor -> HxWx3 uint8 image for overlay/export."""
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _save_overlay(path, image_u8: np.ndarray, saliency: np.ndarray, box) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_u8)
    ax.imshow(saliency, cmap="jet", alpha=0.45)
    y0, x0, y1, x1 = box
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   edgecolor="lime", facecolor="none", lw=1.5))
    py, px = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    ax.plot(px, py, "w+", markersize=10, markeredgewidth=2)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=80)
    plt.close(fig)


def run_v3(cfg: Config, logger: RunLogger) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    a = cfg.attribution
    logger.info(f"device={device} checkpoint={a.checkpoint} target_layer={a.target_layer}")

    model = _build_model(cfg.train.backbone, a.checkpoint, device)

    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed)
    balanced = balanced_eval_subset(index, "test", seed=cfg.seed, per_class=a.n_images)
    pos_rows = balanced[balanced["label"] == 1].head(a.n_images).reset_index(drop=True)

    from auditor.data.nih import SplitIndex
    sub_index = SplitIndex(df=pos_rows.assign(split="eval"), target=index.target, n_pos={})
    ds = make_dataset(sub_index, "eval", cfg.data, train=False)

    size = cfg.shortcut.size
    corner = cfg.shortcut.corner

    tagged_hits = untagged_hits = 0
    tagged_mass = []
    n = len(ds)
    for i in range(n):
        x, _ = ds[i]                                   # 3xHxW normalized, UNtagged
        h, w = x.shape[1], x.shape[2]
        box = tag_box(corner, size, h, w)

        x_tag = add_corner_tag(x, corner=corner, size=size)
        sal_tag = compute_gradcam(model, x_tag, a.target_layer)
        sal_untag = compute_gradcam(model, x, a.target_layer)

        if peak_in_region(sal_tag, box):
            tagged_hits += 1
        if peak_in_region(sal_untag, box):
            untagged_hits += 1
        tagged_mass.append(mass_fraction(sal_tag, box))

        if i < a.n_export:
            img_u8 = _denormalize(x_tag)
            _save_overlay(logger.artifact_path(f"overlay_tagged_{i:02d}.png"), img_u8, sal_tag, box)
            crop = salient_crop(img_u8, sal_tag, cfg.attribution.crop_size)
            from PIL import Image
            Image.fromarray(crop).save(logger.artifact_path(f"crop_tagged_{i:02d}.png"))

    metrics = {
        "n_images": n,
        "tagged_peak_in_box": round(tagged_hits / max(n, 1), 4),
        "untagged_peak_in_box": round(untagged_hits / max(n, 1), 4),
        "tagged_mean_mass_fraction": round(float(np.mean(tagged_mass)), 4),
        "target_layer": a.target_layer,
        "checkpoint": a.checkpoint,
    }
    logger.log_metrics(**metrics)
    return metrics
