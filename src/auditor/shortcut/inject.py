"""Synthetic confound injection (V2). Core of the synthetic spine — heavily tested.

The confound is a bright corner tag. Two subtleties handled here:
  * Images are ImageNet-normalized in the transform, so "white" (pixel 1.0) maps to
    (1.0 - mean)/std per channel in tensor space. We stamp that value so the tag is truly
    max-brightness regardless of normalization.
  * Whether a given image gets tagged is a DETERMINISTIC function of (image name, seed) so
    that repeated evaluation is reproducible — no run-to-run drift in which images carry the
    confound.
"""

from __future__ import annotations

import hashlib

import torch

from auditor.data.torch_dataset import _MEAN, _STD

CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")

# Per-channel tensor value corresponding to a pure-white (1.0) pixel after ImageNet norm.
_WHITE_NORMALIZED = torch.tensor([(1.0 - m) / s for m, s in zip(_MEAN, _STD)]).view(3, 1, 1)


def add_corner_tag(
    x: torch.Tensor, corner: str = "top-left", size: int = 20, value: torch.Tensor | None = None
) -> torch.Tensor:
    """Return a copy of a 3×H×W (normalized) image tensor with a bright square in `corner`."""
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {CORNERS}, got {corner!r}")
    if x.dim() != 3 or x.shape[0] != 3:
        raise ValueError(f"expected 3xHxW tensor, got shape {tuple(x.shape)}")
    val = _WHITE_NORMALIZED.to(x.dtype) if value is None else value

    out = x.clone()
    h, w = x.shape[1], x.shape[2]
    s = min(size, h, w)
    ys = slice(0, s) if "top" in corner else slice(h - s, h)
    xs = slice(0, s) if "left" in corner else slice(w - s, w)
    out[:, ys, xs] = val
    return out


def _hash_unit(image_name: str, seed: int) -> float:
    """Deterministic pseudo-random float in [0,1) from (image_name, seed)."""
    digest = hashlib.sha256(f"{seed}:{image_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def tag_decision(image_name: str, label: int, mode: str, p: float, seed: int) -> bool:
    """Whether this image should carry the tag, deterministically.

    aligned: tag positives w.p. p, negatives w.p. 1-p.
    flipped: tag negatives w.p. p, positives w.p. 1-p.
    removed: never tag.
    """
    if mode == "removed":
        return False
    if mode not in ("aligned", "flipped"):
        raise ValueError(f"mode must be aligned/removed/flipped, got {mode!r}")
    u = _hash_unit(image_name, seed)
    if mode == "aligned":
        prob = p if label == 1 else (1.0 - p)
    else:  # flipped
        prob = p if label == 0 else (1.0 - p)
    return u < prob


class TaggedDataset(torch.utils.data.Dataset):
    """Wraps a NIHDataset, applying the corner tag per (mode, p) with deterministic decisions.

    Image names are read from the base dataset's own rows, so the tag decision is always
    aligned to the image actually loaded (no separate row list to fall out of sync).
    """

    def __init__(self, base, mode: str, p: float, seed: int, corner: str, size: int):
        self.base = base            # underlying NIHDataset (provides rows, transform, images)
        self.mode = mode
        self.p = p
        self.seed = seed
        self.corner = corner
        self.size = size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        name = self.base.rows.iloc[i]["image"]
        if tag_decision(name, int(y.item()), self.mode, self.p, self.seed):
            x = add_corner_tag(x, corner=self.corner, size=self.size)
        return x, y
