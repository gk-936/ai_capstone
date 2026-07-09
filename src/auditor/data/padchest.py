"""PadChest second-site loader (V6 cross-site OOD).

Schema-tolerant: PadChest's labels CSV column names vary slightly across distributions, so we
auto-detect the image-id, labels, and patient-id columns from known candidates rather than
hardcoding. Pneumothorax is a string inside the multi-label `Labels` field.

Kept parallel to data/neatx.py and data/nih.py: pure pandas parsing + a patient-level split,
image resolution deferred to the torch dataset via make_dataset.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

from auditor.common.config import DataConfig
from auditor.common.paths import resolve_path

# Candidate column names across PadChest CSV variants (first match wins).
_IMAGE_COLS = ["ImageID", "ImageId", "image_id", "Image Index", "filename"]
_LABEL_COLS = ["Labels", "labels", "Finding Labels", "findings"]
_PATIENT_COLS = ["PatientID", "Patient ID", "patient_id", "PatientId"]


def _pick(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"PadChest CSV: no {what} column found among {candidates}; "
                   f"actual columns: {list(df.columns)[:15]}")


def _has_pneumothorax(labels_field) -> int:
    """Pneumothorax present? Handles a Python-list-string, a |-joined string, or plain text."""
    if labels_field is None or (isinstance(labels_field, float) and np.isnan(labels_field)):
        return 0
    s = str(labels_field).lower()
    # Try to parse a stringified list (PadChest stores Labels as "['pneumothorax', ...]").
    try:
        parsed = ast.literal_eval(str(labels_field))
        if isinstance(parsed, (list, tuple)):
            s = " ".join(str(x).lower() for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return int("pneumothorax" in s)


@dataclass
class PadChestIndex:
    df: pd.DataFrame   # columns: image, patient_id, label, split
    n: dict

    def subset(self, split: str) -> pd.DataFrame:
        return self.df[self.df["split"] == split].reset_index(drop=True)


def build_padchest_index(data_cfg: DataConfig, labels_csv: str, seed: int = 1337,
                         dfr_frac: float = 0.3, max_rows: int | None = None) -> PadChestIndex:
    """Parse PadChest labels -> pneumothorax binary index with a patient-disjoint dfr/test split.

    dfr_frac of patients go to the 'dfr' split (small target-site adaptation set); the rest are
    'test' (held-out OOD evaluation). max_rows caps for quick runs.
    """
    csv = resolve_path(labels_csv)
    raw = pd.read_csv(csv, low_memory=False)
    img_col = _pick(raw, _IMAGE_COLS, "image-id")
    lab_col = _pick(raw, _LABEL_COLS, "labels")
    pat_col = _pick(raw, _PATIENT_COLS, "patient-id")

    df = pd.DataFrame({
        "image": raw[img_col].astype(str),
        "patient_id": raw[pat_col].astype(str),
        "label": raw[lab_col].apply(_has_pneumothorax),
    })
    df = df.dropna(subset=["image"]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    # Patient-disjoint dfr/test split.
    rng = np.random.RandomState(seed)
    patients = np.sort(df["patient_id"].unique())
    rng.shuffle(patients)
    n_dfr = int(len(patients) * dfr_frac)
    dfr_patients = set(patients[:n_dfr])
    df["split"] = np.where(df["patient_id"].isin(dfr_patients), "dfr", "test")

    n = {s: {"n": int((df["split"] == s).sum()),
             "pos": int(df[(df["split"] == s) & (df["label"] == 1)].shape[0])}
         for s in ("dfr", "test")}
    return PadChestIndex(df=df, n=n)


def assert_patient_disjoint(index: PadChestIndex) -> None:
    a = set(index.subset("dfr")["patient_id"])
    b = set(index.subset("test")["patient_id"])
    if a & b:
        raise AssertionError(f"PadChest patient leakage dfr/test: {len(a & b)} shared")
