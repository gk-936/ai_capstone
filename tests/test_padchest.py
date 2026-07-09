"""V6 PadChest loader tests: schema-tolerance, pneumothorax parsing, patient-disjoint split.

Synthetic CSVs (no images) — validates the parsing that can't wait for the 12.89GB download.
"""

import pandas as pd
import pytest

from auditor.common.config import DataConfig
from auditor.data.padchest import (
    _has_pneumothorax,
    assert_patient_disjoint,
    build_padchest_index,
)


def test_pneumothorax_parsing_variants():
    assert _has_pneumothorax("['pneumothorax', 'effusion']") == 1   # stringified list
    assert _has_pneumothorax("['normal']") == 0
    assert _has_pneumothorax("effusion|pneumothorax") == 1          # pipe-joined
    assert _has_pneumothorax("clear lungs") == 0
    assert _has_pneumothorax(None) == 0
    assert _has_pneumothorax(float("nan")) == 0


def _write_csv(tmp_path, cols):
    """cols: dict of {colname: values}. Writes a PadChest-like CSV, returns its path."""
    p = tmp_path / "padchest_labels.csv"
    pd.DataFrame(cols).to_csv(p, index=False)
    return str(p)


def test_loader_detects_standard_columns(tmp_path):
    csv = _write_csv(tmp_path, {
        "ImageID": [f"img_{i}.png" for i in range(20)],
        "PatientID": [f"P{i // 2}" for i in range(20)],   # 2 imgs/patient
        "Labels": (["['pneumothorax']"] * 8) + (["['normal']"] * 12),
    })
    idx = build_padchest_index(DataConfig(), csv, seed=1)
    assert_patient_disjoint(idx)
    total_pos = idx.n["dfr"]["pos"] + idx.n["test"]["pos"]
    assert total_pos == 8


def test_loader_handles_alternate_column_names(tmp_path):
    # A variant CSV using different-but-known column names.
    csv = _write_csv(tmp_path, {
        "image_id": ["a.png", "b.png", "c.png", "d.png"],
        "patient_id": ["P1", "P1", "P2", "P2"],
        "labels": ["['pneumothorax']", "['normal']", "['normal']", "['pneumothorax']"],
    })
    idx = build_padchest_index(DataConfig(), csv, seed=1)
    assert (idx.n["dfr"]["pos"] + idx.n["test"]["pos"]) == 2


def test_loader_missing_column_raises(tmp_path):
    csv = _write_csv(tmp_path, {"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(KeyError):
        build_padchest_index(DataConfig(), csv, seed=1)


def test_split_is_patient_disjoint(tmp_path):
    csv = _write_csv(tmp_path, {
        "ImageID": [f"i{i}.png" for i in range(40)],
        "PatientID": [f"P{i // 4}" for i in range(40)],
        "Labels": ["['pneumothorax']" if i % 3 == 0 else "['normal']" for i in range(40)],
    })
    idx = build_padchest_index(DataConfig(), csv, seed=3, dfr_frac=0.3)
    assert_patient_disjoint(idx)   # raises on leakage
