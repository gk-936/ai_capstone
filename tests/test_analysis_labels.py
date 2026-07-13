import pandas as pd
from auditor.analysis.label_analysis import parse_labels, ptx_label_composition


def test_parse_labels_splits_and_normalizes():
    assert parse_labels("['pneumothorax', 'chest drain tubes']") == ["pneumothorax", "chest drain tubes"]
    assert parse_labels("pneumothorax") == ["pneumothorax"]
    assert parse_labels("") == []
    assert parse_labels(None) == []


def test_ptx_composition_counts_isolated_and_cofindings():
    df = pd.DataFrame({
        "Labels": ["['pneumothorax']",
                   "['pneumothorax', 'chest drain tubes']",
                   "['pneumothorax', 'chest drain tubes']",
                   "['normal']"],
        "ptx": [1, 1, 1, 0],
    })
    comp = ptx_label_composition(df)
    assert comp["n_ptx_positive"] == 3
    assert comp["frac_ptx_isolated"] == round(1 / 3, 4)     # only 1 of 3 is ptx-only
    assert comp["top_cofindings"][0][0] == "chest drain tubes"   # most common co-finding
    assert comp["top_cofindings"][0][1] == 2
