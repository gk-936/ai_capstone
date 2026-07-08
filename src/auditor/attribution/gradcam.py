"""Grad-CAM attribution + confound-localization geometry (V3).

The geometry helpers (tag_box, peak_in_region, salient_crop) are pure and unit-tested; they
take a generic box (y0, x0, y1, x1) so the same code works for the real, diffuse device
confound later — nothing here is hardcoded to the corner tag.

compute_gradcam wraps pytorch-grad-cam and needs the model on a device; it is exercised by
the V3 run rather than unit tests.
"""

from __future__ import annotations

import numpy as np
import torch

Box = tuple[int, int, int, int]  # (y0, x0, y1, x1), half-open on y1/x1


def tag_box(corner: str, size: int, h: int, w: int) -> Box:
    """The (y0, x0, y1, x1) region a corner tag of `size` occupies in an HxW image."""
    s = min(size, h, w)
    y0, y1 = (0, s) if "top" in corner else (h - s, h)
    x0, x1 = (0, s) if "left" in corner else (w - s, w)
    return (y0, x0, y1, x1)


def peak_in_region(saliency: np.ndarray, box: Box) -> bool:
    """Pointing game: does the highest-saliency pixel fall inside `box`?"""
    y0, x0, y1, x1 = box
    peak_y, peak_x = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    return bool((y0 <= peak_y < y1) and (x0 <= peak_x < x1))


def peak_on_side(saliency: np.ndarray, side: str) -> bool:
    """V4b coarse localization: does the saliency peak fall on the annotated side?

    side is 'L'/'R'/'BOTH' as in NEATX. NOTE the radiological convention: the patient's LEFT
    ('L') appears on the RIGHT of the image. 'BOTH' or unknown -> not scored (returns None).
    """
    if side not in ("L", "R"):
        return None  # BOTH / NaN — not a clean single-side check
    _, w = saliency.shape
    peak_x = int(np.unravel_index(int(np.argmax(saliency)), saliency.shape)[1])
    on_image_right = peak_x >= w / 2
    # Patient L -> image right; patient R -> image left.
    return on_image_right if side == "L" else (not on_image_right)


def mass_fraction(saliency: np.ndarray, box: Box) -> float:
    """Fraction of total saliency mass inside `box` (supporting metric, not the gate)."""
    y0, x0, y1, x1 = box
    total = float(saliency.sum())
    if total <= 0:
        return 0.0
    return float(saliency[y0:y1, x0:x1].sum()) / total


def salient_crop(image: np.ndarray, saliency: np.ndarray, size: int) -> np.ndarray:
    """Crop `image` (HxW or HxWxC) of side `size` centered on the saliency peak, edge-clamped."""
    h, w = saliency.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    half = size // 2
    y0 = int(np.clip(peak_y - half, 0, max(h - size, 0)))
    x0 = int(np.clip(peak_x - half, 0, max(w - size, 0)))
    return image[y0:y0 + size, x0:x0 + size]


def resolve_target_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Resolve a dotted module path (e.g. 'features.norm5') to the module object."""
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"target layer {name!r} not found; e.g. try one of "
                       f"{[n for n in modules if n.endswith('norm5')]}")
    return modules[name]


def compute_gradcam(model, x: torch.Tensor, target_layer_name: str = "features.norm5") -> np.ndarray:
    """Return an HxW Grad-CAM saliency map in [0,1] for a single image tensor x (3xHxW).

    Uses the model's single positive logit as the target (binary head).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    target_layer = resolve_target_layer(model, target_layer_name)
    device = next(model.parameters()).device
    input_tensor = x.unsqueeze(0).to(device)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # BinaryClassifierOutputTarget(1) drives CAM toward the positive class of a 1-logit head.
        grayscale = cam(input_tensor=input_tensor,
                        targets=[BinaryClassifierOutputTarget(1)])[0]  # HxW in [0,1]
    return grayscale.astype(np.float32)
