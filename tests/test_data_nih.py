"""Data-layer tests.

Synthetic tests (always run) exercise label parsing + patient-disjoint splitting on a tiny
hand-built CSV. Real-data tests run only if the actual NIH files are present, and check the
split against the real dataset (the anti-leakage guard that matters for V1's AUC).
"""

from pathlib import Path

import pandas as pd
import pytest

from auditor.common.config import DataConfig, load_config
from auditor.common.paths import resolve_path
from auditor.data.nih import assert_patient_disjoint, build_index

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ----------------------------- synthetic (no data needed) -----------------------------

def _synthetic_cfg(tmp_path: Path) -> DataConfig:
    """Build a tiny 3-patient dataset: patient A (2 imgs), B (2), C (1)."""
    rows = [
        ("00000A_000.png", "Cardiomegaly|Pneumothorax", "A"),
        ("00000A_001.png", "No Finding", "A"),
        ("00000B_000.png", "Pneumothorax", "B"),
        ("00000B_001.png", "Effusion", "B"),
        ("00000C_000.png", "No Finding", "C"),
    ]
    df = pd.DataFrame(
        {
            "Image Index": [r[0] for r in rows],
            "Finding Labels": [r[1] for r in rows],
            "Patient ID": [r[2] for r in rows],
        }
    )
    csv = tmp_path / "Data_Entry_2017.csv"
    df.to_csv(csv, index=False)

    tv = tmp_path / "train_val.txt"
    te = tmp_path / "test.txt"
    tv.write_text("\n".join(r[0] for r in rows[:4]), encoding="utf-8")  # A,B in train_val
    te.write_text(rows[4][0], encoding="utf-8")                        # C in test

    return DataConfig(
        labels_csv=str(csv),
        train_val_list=str(tv),
        test_list=str(te),
        nih_images=str(tmp_path),
    )


def test_binarizes_multilabel_on_target(tmp_path):
    idx = build_index(_synthetic_cfg(tmp_path), target="Pneumothorax", val_frac=0.0)
    labels = dict(zip(idx.df["image"], idx.df["label"]))
    assert labels["00000A_000.png"] == 1   # Cardiomegaly|Pneumothorax -> positive
    assert labels["00000A_001.png"] == 0   # No Finding -> negative
    assert labels["00000B_000.png"] == 1


def test_test_split_comes_from_official_list(tmp_path):
    idx = build_index(_synthetic_cfg(tmp_path), target="Pneumothorax", val_frac=0.5)
    test_rows = idx.subset("test")
    assert set(test_rows["patient_id"]) == {"C"}


def test_synthetic_split_is_patient_disjoint(tmp_path):
    idx = build_index(_synthetic_cfg(tmp_path), target="Pneumothorax", val_frac=0.5)
    assert_patient_disjoint(idx)  # raises on leakage


def test_unknown_target_raises(tmp_path):
    with pytest.raises(ValueError):
        build_index(_synthetic_cfg(tmp_path), target="NotAPathology")


# ------------------------------- real data (skip if absent) ---------------------------

def _real_cfg() -> DataConfig:
    return load_config(CONFIGS / "base.yaml").data


def _real_data_present() -> bool:
    cfg = _real_cfg()
    return resolve_path(cfg.labels_csv).exists() and resolve_path(cfg.nih_images).exists()


real_only = pytest.mark.skipif(not _real_data_present(), reason="NIH data not present")


@real_only
def test_real_split_patient_disjoint():
    idx = build_index(_real_cfg(), target="Pneumothorax", seed=1337)
    assert_patient_disjoint(idx)  # THE anti-leakage guard on the actual dataset


@real_only
def test_real_has_positives_in_each_split():
    idx = build_index(_real_cfg(), target="Pneumothorax", seed=1337)
    for split in ("train", "val", "test"):
        assert idx.n_pos[split] > 0, f"no positives in {split}"
