"""V4b tests: tube matcher (hallucination guards), side-consistency, and real-data NEATX loader."""

from pathlib import Path

import numpy as np
import pytest

from auditor.attribution.gradcam import peak_on_side
from auditor.common.config import load_config
from auditor.common.paths import resolve_path
from auditor.discovery.score_naming import names_tube, score

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ------------------------------- tube matcher -----------------------------------------

def test_names_tube_hits_device_descriptions():
    for desc in [
        "There is a chest tube in the left hemithorax.",
        "A chest drain is visible.",
        "an intercostal catheter runs across the field",
        "a thin tube/line projecting over the lung",
        "a curved catheter, likely a pigtail drain",
        "a medical device consistent with a thoracostomy tube",
    ]:
        assert names_tube(desc) is True, desc


def test_names_tube_ignores_pure_anatomy():
    for desc in [
        "The lung fields are clear.",
        "Normal ribs and clavicle, no abnormality.",
        "Pulmonary vasculature is unremarkable.",
        "There is a fissure and normal bronchial markings.",
        "",
    ]:
        assert names_tube(desc) is False, desc


# ------------------------------- side consistency -------------------------------------

def test_peak_on_side_radiological_convention():
    sal = np.zeros((10, 10), dtype=np.float32)
    sal[5, 8] = 1.0  # peak on IMAGE right
    # patient L appears on image right -> hit for side 'L'
    assert peak_on_side(sal, "L") is True
    assert peak_on_side(sal, "R") is False
    # BOTH / unknown -> not scored
    assert peak_on_side(sal, "BOTH") is None


# ------------------------------- scoring with tube matcher ----------------------------

def test_score_uses_tube_matcher():
    manifest = [
        {"id": "pos_0", "kind": "positive", "has_confound": True},
        {"id": "pos_1", "kind": "positive", "has_confound": True},
        {"id": "ctrl_0", "kind": "control", "has_confound": False},
    ]
    preds = {"pos_0": "a chest tube", "pos_1": "clear lungs", "ctrl_0": "normal ribs"}
    m = score(manifest, preds, matcher=names_tube)
    assert m["tp"] == 1 and m["fn"] == 1 and m["control_hallucination"] == 0.0


# ------------------------------- real NEATX data (skip if absent) ---------------------

def _tube_present() -> bool:
    cfg = load_config(CONFIGS / "base.yaml").data
    return bool(cfg.tube_csv) and resolve_path(cfg.tube_csv).exists()


@pytest.mark.skipif(not _tube_present(), reason="NEATX tube csv not present")
def test_tube_enriched_split_is_measurable():
    from auditor.data.neatx import assert_patient_disjoint, build_tube_index

    cfg = load_config(CONFIGS / "base.yaml")
    idx = build_tube_index(cfg.data, train_enrich=0.9, seed=1337)
    assert_patient_disjoint(idx)
    # train positives enriched for tubes...
    tr = idx.n["train"]
    assert tr["pos_tube"] / max(tr["pos"], 1) >= 0.85
    # ...and test has BOTH groups so the reliance gap is measurable.
    te = idx.n["test"]
    assert te["pos_tube"] > 0 and te["pos_notube"] > 0
