"""NEATX chest-tube annotations layered onto the NIH index (V4b).

NEATX gives image-level tube presence + coarse side (L/R/BOTH) for a subset of NIH images.
This module joins those tube labels to the NIH pneumothorax index and builds a TUBE-ENRICHED
training split so a pneumothorax classifier can learn the natural "tube -> pneumothorax"
shortcut. The reliance gap is then measured on held-out pneumothorax-positives WITH vs WITHOUT
a tube.

Design note (from the data): tube and pneumothorax are ~100% correlated in NEATX, so the
confound is the *device co-occurring with the pathology*, not an injected tag. We control the
shortcut via how strongly TRAIN positives are enriched for tubes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from auditor.common.config import DataConfig
from auditor.common.paths import resolve_path


@dataclass
class TubeIndex:
    """NIH pneumothorax rows annotated with tube presence/side and a train/val/test split."""

    df: pd.DataFrame   # columns: image, patient_id, label(ptx), tube, side, split
    n: dict            # per-split counts for logging

    def subset(self, split: str) -> pd.DataFrame:
        return self.df[self.df["split"] == split].reset_index(drop=True)


def _load_tube_labels(data_cfg: DataConfig) -> pd.DataFrame:
    """Read NEATX aggregated csv -> image, tube(0/1), side(L/R/BOTH/NaN)."""
    csv = resolve_path(data_cfg.tube_csv)
    t = pd.read_csv(csv)
    return pd.DataFrame({
        "image": t["Image Index"].astype(str),
        "tube": t["Drain"].astype(int),
        "side": t["Drain_Location_a1"].astype("string"),
    })


def build_tube_index(
    data_cfg: DataConfig,
    train_enrich: float = 0.9,
    seed: int = 1337,
    val_frac: float = 0.15,
    max_neg_per_split: int | None = 4000,
) -> TubeIndex:
    """Build a pneumothorax index with tube labels and a tube-enriched TRAIN split.

    The training positives are sampled so a fraction `train_enrich` of them carry a tube,
    inducing the shortcut. Val/test keep the natural mix so the reliance gap is honest.
    Patient-disjoint via the NIH official split lists + patient-level val carve.
    """
    labels_csv = resolve_path(data_cfg.labels_csv)
    nih = pd.read_csv(labels_csv)
    df = pd.DataFrame({
        "image": nih["Image Index"].astype(str),
        "patient_id": nih["Patient ID"].astype(str),
        "label": nih["Finding Labels"].fillna("").str.split("|").apply(
            lambda p: int("Pneumothorax" in p)),
    })

    # Join tube labels (images not in NEATX -> tube=0, side=NaN).
    tubes = _load_tube_labels(data_cfg)
    df = df.merge(tubes, on="image", how="left")
    df["tube"] = df["tube"].fillna(0).astype(int)

    # Official patient-disjoint train_val / test membership.
    with open(resolve_path(data_cfg.train_val_list), encoding="utf-8") as f:
        train_val = {ln.strip() for ln in f if ln.strip()}
    with open(resolve_path(data_cfg.test_list), encoding="utf-8") as f:
        test = {ln.strip() for ln in f if ln.strip()}

    rng = np.random.RandomState(seed)
    in_tv = df["image"].isin(train_val)
    tv_patients = np.sort(df.loc[in_tv, "patient_id"].unique())
    rng.shuffle(tv_patients)
    n_val = int(len(tv_patients) * val_frac)
    val_patients = set(tv_patients[:n_val])

    split = np.where(
        df["image"].isin(test), "test",
        np.where(df["patient_id"].isin(val_patients), "val",
                 np.where(in_tv, "train", "unused")))
    df["split"] = split
    df = df[df["split"] != "unused"].reset_index(drop=True)

    df = _enrich_train(df, train_enrich, seed, max_neg_per_split)

    n = {s: _split_counts(df, s) for s in ("train", "val", "test")}
    return TubeIndex(df=df, n=n)


def _enrich_train(df: pd.DataFrame, train_enrich: float, seed: int,
                  max_neg_per_split: int | None) -> pd.DataFrame:
    """Downsample train positives so a `train_enrich` fraction of them carry a tube.

    Non-train splits are left at their natural mix. Optionally cap negatives per split so the
    classes are not wildly imbalanced (pneumothorax is rare).
    """
    keep = []
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0]

        if s == "train" and len(pos) > 0:
            pos_tube = pos[pos["tube"] == 1]
            pos_notube = pos[pos["tube"] == 0]
            # choose counts so tube fraction == train_enrich, keeping as many as possible
            n_tube = len(pos_tube)
            n_notube_target = int(round(n_tube * (1 - train_enrich) / max(train_enrich, 1e-6)))
            n_notube = min(len(pos_notube), n_notube_target)
            pos = pd.concat([pos_tube, pos_notube.sample(n=n_notube, random_state=seed)])

        if max_neg_per_split is not None and len(neg) > max_neg_per_split:
            neg = neg.sample(n=max_neg_per_split, random_state=seed)

        keep.append(pd.concat([pos, neg]))
    return pd.concat(keep).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _split_counts(df: pd.DataFrame, s: str) -> dict:
    sub = df[df["split"] == s]
    pos = sub[sub["label"] == 1]
    return {
        "n": len(sub),
        "pos": int(len(pos)),
        "pos_tube": int((pos["tube"] == 1).sum()),
        "pos_notube": int((pos["tube"] == 0).sum()),
    }


def assert_patient_disjoint(index: TubeIndex) -> None:
    by = {s: set(index.subset(s)["patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by[a] & by[b]
        if overlap:
            raise AssertionError(f"patient leakage {a}/{b}: {len(overlap)} shared")
