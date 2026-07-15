"""Experiment 2 (image-independence test): does the VLM actually read the crop, or recite
a prior regardless of input?

Three conditions per POSITIVE crop (occlusion only makes sense where the confound actually
is, so this scopes to positive crops — not controls):
  (a) occlude the RELEVANT region (where the confound sits)
  (b) occlude an IRRELEVANT region elsewhere in the crop (same-corner-style placement,
      sized to the largest square genuinely disjoint from the relevant box — see
      `opposite_box`; not always equal area to the relevant box, see its docstring)
  (c) swap in a different same-kind crop's image entirely

Box provenance differs by dataset:
  * V4a (synthetic tag): the relevant box is recomputed EXACTLY by re-running the same
    localization pipeline export_crops.py used (checkpoint forward pass + Grad-CAM + the
    same crop-origin clamping math) — genuine ground truth, not a guess.
  * V4b (real tube): there is no recorded exact box in manifest.json. `salient_crop` centers
    every crop on its own Grad-CAM peak, so the relevant content is expected near the crop
    CENTER — the relevant box here is a PROXY (a fixed centered square), not exact ground
    truth like V4a's. This approximation is a real limitation of this experiment for V4b and
    is surfaced in the scored output, not hidden.

Occlusion fill is flat mid-gray (128) — a neutral value unlikely to itself read as an
artificial bright/white marker to the V4a matcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

Box = tuple[int, int, int, int]  # (y0, x0, y1, x1), half-open, in CROP-local coordinates

_GRAY = 128


def compute_v4a_relevant_boxes(cfg, logger=None) -> dict[str, Box]:
    """Exact crop-local tag box per V4a positive crop id, by re-running the same
    localization pipeline export_crops.py used (deterministic given the same seed/config)."""
    import torch

    from auditor.attribution.gradcam import compute_gradcam, tag_box
    from auditor.attribution.localize_confound import _build_model
    from auditor.common.paths import resolve_device
    from auditor.common.seed import set_seed
    from auditor.data.nih import SplitIndex, build_index, make_dataset
    from auditor.evaluation.reliance import balanced_eval_subset
    from auditor.shortcut.inject import add_corner_tag

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    d = cfg.discovery
    model = _build_model(cfg.train.backbone, d.checkpoint, device)
    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed)

    n_need = max(d.n_positive, d.n_control)
    balanced = balanced_eval_subset(index, "test", seed=cfg.seed, per_class=n_need)
    pos_rows = balanced[balanced["label"] == 1].head(d.n_positive).reset_index(drop=True)
    sub = SplitIndex(df=pos_rows.assign(split="eval"), target=index.target, n_pos={})
    pos_ds = make_dataset(sub, "eval", cfg.data, train=False)

    boxes: dict[str, Box] = {}
    full_box = tag_box(cfg.shortcut.corner, cfg.shortcut.size, cfg.data.image_size, cfg.data.image_size)
    for i in range(len(pos_ds)):
        x, _ = pos_ds[i]
        x_tag = add_corner_tag(x, corner=cfg.shortcut.corner, size=cfg.shortcut.size, opacity=1.0)
        sal = compute_gradcam(model, x_tag, d.target_layer)
        h, w = sal.shape
        peak_y, peak_x = np.unravel_index(int(np.argmax(sal)), sal.shape)
        half = d.crop_size // 2
        y0 = int(np.clip(peak_y - half, 0, max(h - d.crop_size, 0)))
        x0 = int(np.clip(peak_x - half, 0, max(w - d.crop_size, 0)))
        by0, bx0, by1, bx1 = full_box
        local_box = (
            max(0, by0 - y0), max(0, bx0 - x0),
            min(d.crop_size, by1 - y0), min(d.crop_size, bx1 - x0),
        )
        boxes[f"pos_{i:03d}"] = local_box
        if logger and i % 20 == 0:
            logger.info(f"v4a relevant box pos_{i:03d}: {local_box}")
    return boxes


def v4b_proxy_box(crop_size: int, frac: float = 0.5) -> Box:
    """Centered square covering `frac` of the crop side — a PROXY for the relevant region,
    since salient_crop centers every V4b crop on its own Grad-CAM peak but the exact box
    isn't recorded. NOT exact ground truth (unlike V4a's compute_v4a_relevant_boxes)."""
    size = int(crop_size * frac)
    off = (crop_size - size) // 2
    return (off, off, off + size, off + size)


def opposite_box(box: Box, crop_size: int) -> Box:
    """A square in the crop corner diagonally opposite `box`'s centroid, sized to the
    LARGEST square that is genuinely disjoint from `box` in that corner.

    Not necessarily the same area as `box`: when `box` covers more than half the crop in
    either dimension (true for V4a's ~20px tag in a 32px crop — two 20x20 squares cannot
    both fit disjointly in a 32x32 grid), an equal-size mirror would overlap `box`. This
    always returns a box with zero overlap, even if that means a smaller square.
    """
    y0, x0, y1, x1 = box
    h, w = y1 - y0, x1 - x0
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    top_free, left_free = cy >= crop_size / 2, cx >= crop_size / 2
    avail_h = y0 if top_free else crop_size - y1
    avail_w = x0 if left_free else crop_size - x1
    size = max(0, min(avail_h, avail_w, h, w))  # never exceed the original box's own size
    new_y0 = 0 if top_free else crop_size - size
    new_x0 = 0 if left_free else crop_size - size
    return (new_y0, new_x0, new_y0 + size, new_x0 + size)


def _occlude(img: np.ndarray, box: Box) -> np.ndarray:
    y0, x0, y1, x1 = box
    out = img.copy()
    out[y0:y1, x0:x1] = _GRAY
    return out


def build_conditions(crops_dir: str | Path, positive_ids: list[str], relevant_boxes: dict[str, Box],
                     crop_size: int, out_dir: str | Path) -> dict[str, Path]:
    """Write relevant/, irrelevant/, swap/ crop folders + a shared manifest.json.

    Returns {condition_name: crops_subdir}. `positive_ids` fixes iteration order (used for
    the deterministic cyclic swap pairing).
    """
    from PIL import Image

    crops_dir = Path(crops_dir)
    out_dir = Path(out_dir)
    dirs = {c: out_dir / f"crops_{c}" for c in ("relevant", "irrelevant", "swap")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    n = len(positive_ids)
    for i, cid in enumerate(positive_ids):
        img = np.array(Image.open(crops_dir / f"{cid}.png").convert("RGB"))
        rel_box = relevant_boxes[cid]
        irr_box = opposite_box(rel_box, crop_size)

        Image.fromarray(_occlude(img, rel_box)).save(dirs["relevant"] / f"{cid}.png")
        Image.fromarray(_occlude(img, irr_box)).save(dirs["irrelevant"] / f"{cid}.png")

        swap_id = positive_ids[(i + 1) % n]
        swap_img = np.array(Image.open(crops_dir / f"{swap_id}.png").convert("RGB"))
        Image.fromarray(swap_img).save(dirs["swap"] / f"{cid}.png")

        manifest.append({"id": cid, "relevant_box": list(rel_box), "irrelevant_box": list(irr_box),
                         "swap_source": swap_id})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dirs
