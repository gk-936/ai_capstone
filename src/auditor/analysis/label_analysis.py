"""Experiment B-lite — PadChest ptx label composition from the subset CSV's `Labels` column.

The working subset CSV has no MethodLabel/Projection, so this bounds the label-mismatch slice
qualitatively: how often PadChest's pneumothorax cases carry a co-finding (e.g. chest drains)
that NIH labels separately, which can inflate the cross-site gap. Pure pandas; no model.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd


def parse_labels(cell) -> list[str]:
    """Split a PadChest `Labels` cell (a stringified list) into normalized lowercase tokens."""
    if cell is None or (isinstance(cell, float)) or str(cell).strip() == "":
        return []
    text = str(cell).strip().strip("[]")
    if not text:
        return []
    parts = [p.strip().strip("'\"").lower() for p in text.split(",")]
    return [p for p in parts if p]


def ptx_label_composition(df: pd.DataFrame, labels_col: str = "Labels",
                          ptx_col: str = "ptx") -> dict:
    """Composition of the pneumothorax-positive rows: isolation fraction + top co-findings."""
    pos = df[df[ptx_col] == 1]
    n = len(pos)
    isolated = 0
    cofindings: Counter = Counter()
    for cell in pos[labels_col]:
        tokens = [t for t in parse_labels(cell) if t]
        others = [t for t in tokens if "pneumothorax" not in t]
        if not others:
            isolated += 1
        cofindings.update(others)
    return {
        "n_ptx_positive": int(n),
        "frac_ptx_isolated": round(isolated / n, 4) if n else 0.0,
        "top_cofindings": cofindings.most_common(10),
    }


def run_label_analysis(labels_csv: str, out_path: str) -> dict:
    df = pd.read_csv(labels_csv, low_memory=False)
    comp = ptx_label_composition(df)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)
    return comp
