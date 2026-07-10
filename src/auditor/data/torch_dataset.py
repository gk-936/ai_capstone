"""Torch Dataset for NIH images. Kept separate so nih.build_index() needs no torch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet stats — timm DenseNet-121 pretrained weights expect these on 3-channel input.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _open_as_8bit_L(path) -> Image.Image:
    """Open an image as 8-bit single-channel ('L'), scaling higher bit depths to 0-255.

    NIH 224px PNGs are already 8-bit 'L' -> returned unchanged. PadChest (and other sites) ship
    16-bit ('I;16'/'I') images; min-max scaling their full range to 0-255 puts them on the same
    intensity representation the model was TRAINED on (NIH 8-bit), so the cross-site OOD gap
    reflects real shift, not bit-depth. The model's transform is untouched (train-eval consistent).
    """
    img = Image.open(path)
    if img.mode in ("I", "I;16", "I;16B", "F"):
        import numpy as np

        arr = np.asarray(img).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo + 1e-6) * 255.0 if hi > lo else arr * 0
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return img.convert("L")


def _build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Grayscale(num_output_channels=3)]  # 'L' -> 3ch for pretrained weights
    if train:
        ops += [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [transforms.Resize((image_size, image_size))]
    ops += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


class NIHDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, images_dir: Path, image_size: int, train: bool):
        self.rows = rows.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tf = _build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        img = _open_as_8bit_L(self.images_dir / row["image"])
        x = self.tf(img)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y
