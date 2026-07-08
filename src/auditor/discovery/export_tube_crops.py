"""V4b crop export: salient crops from tube-positive PTX images + tube-negative controls.

Mirrors the V4a exporter but for the real chest-tube confound. Positive crops come from
pneumothorax-positive images that carry an annotated tube (crop centered on the Grad-CAM peak,
which should sit on the tube). Control crops come from tube-negative images. The manifest also
records the NEATX side label so side-consistency can be scored later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from auditor.attribution.gradcam import compute_gradcam, peak_on_side, salient_crop
from auditor.attribution.localize_confound import _build_model, _denormalize
from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import SplitIndex, make_dataset
from auditor.data.neatx import build_tube_index


def _save(path: Path, crop_u8: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(crop_u8).save(path)


def export_tube_crops(cfg: Config, logger: RunLogger, checkpoint: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    d = cfg.discovery
    out_dir = logger.run_dir
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg.train.backbone, checkpoint, device)
    index = build_tube_index(cfg.data, train_enrich=cfg.tube_train_enrich, seed=cfg.seed)
    test = index.subset("test")

    pos = test[(test["label"] == 1) & (test["tube"] == 1)].head(d.n_positive).reset_index(drop=True)
    ctrl = test[test["tube"] == 0].head(d.n_control).reset_index(drop=True)

    manifest = []
    side_hits = side_total = 0

    # Positive (tube) crops — salient, and check peak-on-annotated-side.
    sub = SplitIndex(df=pos.assign(split="eval"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "eval", cfg.data, train=False)
    for i in range(len(ds)):
        x, _ = ds[i]
        sal = compute_gradcam(model, x, d.target_layer)
        crop = salient_crop(_denormalize(x), sal, d.crop_size)
        cid = f"pos_{i:03d}"
        _save(crops_dir / f"{cid}.png", crop)
        side = pos.iloc[i]["side"]
        on_side = peak_on_side(sal, side if isinstance(side, str) else "")
        if on_side is not None:
            side_total += 1
            side_hits += int(on_side)
        manifest.append({"id": cid, "kind": "positive", "has_confound": True,
                         "side": None if not isinstance(side, str) else side})

    # Control (tube-free) crops — salient too, so composition matches positives.
    sub_c = SplitIndex(df=ctrl.assign(split="eval"), target="Pneumothorax", n_pos={})
    ds_c = make_dataset(sub_c, "eval", cfg.data, train=False)
    for i in range(len(ds_c)):
        x, _ = ds_c[i]
        sal = compute_gradcam(model, x, d.target_layer)
        crop = salient_crop(_denormalize(x), sal, d.crop_size)
        cid = f"ctrl_{i:03d}"
        _save(crops_dir / f"{cid}.png", crop)
        manifest.append({"id": cid, "kind": "control", "has_confound": False, "side": None})

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metrics = {
        "n_positive": sum(m["kind"] == "positive" for m in manifest),
        "n_control": sum(m["kind"] == "control" for m in manifest),
        "side_consistency": round(side_hits / side_total, 4) if side_total else None,
        "n_side_scored": side_total,
        "crops_dir": str(crops_dir),
        "manifest": str(out_dir / "manifest.json"),
    }
    logger.log_metrics(**metrics)
    return metrics
