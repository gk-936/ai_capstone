"""NIH ChestX-ray14 data layer.

Two concerns kept deliberately separate:
  * `build_index()` — pure pandas: parse labels, binarize on a target pathology, and assign
    patient-disjoint train/val/test splits. No torch, so it is fast and unit-testable.
  * `NIHDataset` — a torch Dataset wrapping the index for training.

Anti-leakage guardrail: ChestX-ray14 has ~30k patients with multiple images each. Splitting
by image would leak a patient across splits and inflate AUC. We use the OFFICIAL
train_val/test lists (already patient-disjoint), and carve val out of train_val BY PATIENT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from auditor.common.config import DataConfig
from auditor.common.paths import resolve_path

# The 14 findings (plus "No Finding") as they appear in Data_Entry_2017.csv.
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


@dataclass
class SplitIndex:
    """A parsed, split dataframe plus metadata for one training run."""

    df: pd.DataFrame          # columns: image, patient_id, label, split
    target: str
    n_pos: dict               # positives per split, for sanity logging

    def subset(self, split: str) -> pd.DataFrame:
        return self.df[self.df["split"] == split].reset_index(drop=True)


def _read_list(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_index(
    data_cfg: DataConfig,
    target: str = "Pneumothorax",
    val_frac: float = 0.15,
    seed: int = 1337,
    max_per_split: Optional[dict] = None,
) -> SplitIndex:
    """Parse labels and produce a patient-disjoint train/val/test index.

    `max_per_split` (e.g. {"train": 8000, "val": 2000}) caps rows per split for fast spike
    runs; capping is applied AFTER splitting so it never breaks patient-disjointness.
    """
    if target not in PATHOLOGIES:
        raise ValueError(f"Unknown target {target!r}; must be one of {PATHOLOGIES}")

    csv = resolve_path(data_cfg.labels_csv)
    df = pd.read_csv(csv)

    # Binarize multi-label "Finding Labels" (e.g. "Effusion|Mass") on the target.
    labels = df["Finding Labels"].fillna("")
    df = pd.DataFrame(
        {
            "image": df["Image Index"].astype(str),
            "patient_id": df["Patient ID"].astype(str),
            "label": labels.str.split("|").apply(lambda parts: int(target in parts)),
        }
    )

    # Official splits are patient-disjoint; assign train_val vs test by image membership.
    train_val_imgs = _read_list(resolve_path(data_cfg.train_val_list))
    test_imgs = _read_list(resolve_path(data_cfg.test_list))
    in_trainval = df["image"].isin(train_val_imgs)
    in_test = df["image"].isin(test_imgs)

    # Carve val out of train_val BY PATIENT so no patient spans train and val.
    rng = np.random.RandomState(seed)
    tv_patients = df.loc[in_trainval, "patient_id"].unique()
    tv_patients_sorted = np.sort(tv_patients)          # deterministic before shuffle
    rng.shuffle(tv_patients_sorted)
    n_val = int(len(tv_patients_sorted) * val_frac)
    val_patients = set(tv_patients_sorted[:n_val])

    split = np.where(
        in_test,
        "test",
        np.where(df["patient_id"].isin(val_patients), "val", "train"),
    )
    # Rows in neither official list (a couple exist) are dropped by leaving them unassigned.
    split = np.where(in_trainval | in_test, split, "unused")
    df["split"] = split
    df = df[df["split"] != "unused"].reset_index(drop=True)

    if max_per_split:
        keep = []
        for name, cap in ({"train": None, "val": None, "test": None} | max_per_split).items():
            sub = df[df["split"] == name]
            if cap is not None and len(sub) > cap:
                sub = sub.sample(n=cap, random_state=seed)
            keep.append(sub)
        df = pd.concat(keep).reset_index(drop=True)

    n_pos = {s: int(df[df["split"] == s]["label"].sum()) for s in ("train", "val", "test")}
    return SplitIndex(df=df, target=target, n_pos=n_pos)


def assert_patient_disjoint(index: SplitIndex) -> None:
    """Raise if any patient appears in more than one split (the leakage guard)."""
    by_split = {
        s: set(index.df[index.df["split"] == s]["patient_id"]) for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by_split[a] & by_split[b]
        if overlap:
            raise AssertionError(
                f"Patient leakage between {a} and {b}: {len(overlap)} shared patients"
            )


def make_dataset(index: SplitIndex, split: str, data_cfg: DataConfig, train: bool):
    """Factory that builds a torch Dataset for a split (keeps torch import local)."""
    from auditor.data.torch_dataset import NIHDataset

    return NIHDataset(
        rows=index.subset(split),
        images_dir=resolve_path(data_cfg.nih_images),
        image_size=data_cfg.image_size,
        train=train,
    )
