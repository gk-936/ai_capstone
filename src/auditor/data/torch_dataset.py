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


def _per_image_standardize(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-variance per image, before the ImageNet norm.

    Cross-site fix (V6): NIH (8-bit) and PadChest (16-bit) have very different raw intensity
    distributions. Standardizing each image first removes the bit-depth/acquisition-brightness
    artifact so a cross-site OOD gap reflects real anatomical/population shift, not preprocessing.
    """
    return (x - x.mean()) / (x.std() + 1e-6)


def _build_transform(image_size: int, train: bool, per_image_std: bool = False) -> transforms.Compose:
    ops = [transforms.Grayscale(num_output_channels=3)]  # NIH PNGs are single-channel
    if train:
        ops += [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [transforms.Resize((image_size, image_size))]
    ops += [transforms.ToTensor()]
    if per_image_std:
        ops += [transforms.Lambda(_per_image_standardize)]
    ops += [transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


class NIHDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, images_dir: Path, image_size: int, train: bool,
                 per_image_std: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tf = _build_transform(image_size, train, per_image_std=per_image_std)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        img = Image.open(self.images_dir / row["image"])
        x = self.tf(img)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y
