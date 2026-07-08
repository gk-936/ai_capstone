"""V4a crop export: build the crop set + manifest for the VLM naming step.

Positive crops: salient crops from tagged POSITIVE test images. V3 established that the
saliency peak lands on the injected tag, so the crop centered on that peak contains the tag.
Control crops: crops from UNTAGGED images — confound-free by construction.

`manifest.json` carries perfect ground truth (`has_confound`) because we inject the tag
ourselves. The VLM never sees the manifest — it only sees the crop PNGs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from auditor.attribution.gradcam import compute_gradcam, salient_crop
from auditor.attribution.localize_confound import _build_model, _denormalize
from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import SplitIndex, build_index, make_dataset
from auditor.evaluation.reliance import balanced_eval_subset
from auditor.shortcut.inject import add_corner_tag


def _center_crop(image_u8: np.ndarray, size: int) -> np.ndarray:
    h, w = image_u8.shape[:2]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return image_u8[y0:y0 + size, x0:x0 + size]


def export_crops(cfg: Config, logger: RunLogger) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    d = cfg.discovery

    out_dir = Path(d.out_dir) if d.out_dir else logger.run_dir
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg.train.backbone, d.checkpoint, device)
    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed)

    # Pull enough positives and negatives to make both crop sets from the test split.
    n_need = max(d.n_positive, d.n_control)
    balanced = balanced_eval_subset(index, "test", seed=cfg.seed, per_class=n_need)
    pos_rows = balanced[balanced["label"] == 1].head(d.n_positive).reset_index(drop=True)
    # Controls come from untagged images; use negatives so there is no chance of a tag.
    ctrl_rows = balanced[balanced["label"] == 0].head(d.n_control).reset_index(drop=True)

    manifest = []

    # --- positive crops: tagged, salient-cropped (contain the tag) ---
    sub = SplitIndex(df=pos_rows.assign(split="eval"), target=index.target, n_pos={})
    pos_ds = make_dataset(sub, "eval", cfg.data, train=False)
    for i in range(len(pos_ds)):
        x, _ = pos_ds[i]
        x_tag = add_corner_tag(x, corner=cfg.shortcut.corner, size=cfg.shortcut.size)
        sal = compute_gradcam(model, x_tag, d.target_layer)
        crop = salient_crop(_denormalize(x_tag), sal, d.crop_size)
        cid = f"pos_{i:03d}"
        _save(crops_dir / f"{cid}.png", crop)
        manifest.append({"id": cid, "kind": "positive", "has_confound": True})

    # --- control crops: untagged, center-cropped (no confound) ---
    sub_c = SplitIndex(df=ctrl_rows.assign(split="eval"), target=index.target, n_pos={})
    ctrl_ds = make_dataset(sub_c, "eval", cfg.data, train=False)
    for i in range(len(ctrl_ds)):
        x, _ = ctrl_ds[i]                       # NOT tagged
        crop = _center_crop(_denormalize(x), d.crop_size)
        cid = f"ctrl_{i:03d}"
        _save(crops_dir / f"{cid}.png", crop)
        manifest.append({"id": cid, "kind": "control", "has_confound": False})

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metrics = {
        "n_positive": sum(m["kind"] == "positive" for m in manifest),
        "n_control": sum(m["kind"] == "control" for m in manifest),
        "crops_dir": str(crops_dir),
        "manifest": str(manifest_path),
        "checkpoint": d.checkpoint,
    }
    logger.log_metrics(**metrics)
    return metrics


def _save(path: Path, crop_u8: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(crop_u8).save(path)
