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


def _build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Grayscale(num_output_channels=3)]  # NIH PNGs are single-channel
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
        img = Image.open(self.images_dir / row["image"])
        x = self.tf(img)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y
