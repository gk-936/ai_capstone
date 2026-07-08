"""Reliance-gap measurement (V2).

The gap is measured in ACCURACY points on a class-BALANCED subset so the number is honest:
on a ~3-10% prevalent target, plain accuracy is dominated by the majority class and would
hide (or fake) a reliance gap. On a 50/50 subset, chance = 50% and the gap reflects genuine
dependence on the confound.

We evaluate the SAME trained model three ways by re-tagging the same balanced images:
  aligned  — tag follows the train rule (the model's crutch is present)
  removed  — no tags (crutch gone)
  flipped  — tag anti-correlated (crutch actively misleading)
Gate number: gap = acc(aligned) - acc(removed).
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from auditor.common.config import DataConfig
from auditor.data.nih import SplitIndex, make_dataset
from auditor.shortcut.inject import TaggedDataset


def balanced_eval_subset(index: SplitIndex, split: str, seed: int, per_class: int | None = None) -> pd.DataFrame:
    """Return rows with equal positives and negatives from `split` (chance accuracy = 50%)."""
    sub = index.subset(split)
    pos = sub[sub["label"] == 1]
    neg = sub[sub["label"] == 0]
    n = min(len(pos), len(neg))
    if per_class is not None:
        n = min(n, per_class)
    pos = pos.sample(n=n, random_state=seed)
    neg = neg.sample(n=n, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def evaluate_accuracy(model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs = torch.sigmoid(model(x).squeeze(1)).float().cpu()
        preds = (probs >= threshold).int()
        correct += int((preds == y.int()).sum())
        total += len(y)
    return correct / max(total, 1)


def _tagged_loader(base_ds, mode, shortcut_cfg, batch_size):
    ds = TaggedDataset(
        base=base_ds,
        mode=mode,
        p=shortcut_cfg["p"],
        seed=shortcut_cfg["seed"],
        corner=shortcut_cfg["corner"],
        size=shortcut_cfg["size"],
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def reliance_gap(
    model,
    index: SplitIndex,
    data_cfg: DataConfig,
    shortcut_cfg: dict,
    device: str,
    batch_size: int = 16,
    per_class: int | None = 1500,
) -> dict:
    """Train-time confound present at test → measure accuracy aligned/removed/flipped."""
    rows = balanced_eval_subset(index, "test", seed=shortcut_cfg["seed"], per_class=per_class)

    # A NIHDataset over exactly these balanced rows; TaggedDataset re-tags per mode.
    sub_index = SplitIndex(df=rows.assign(split="eval"), target=index.target, n_pos={})
    base_eval = make_dataset(sub_index, "eval", data_cfg, train=False)

    accs = {}
    for mode in ("aligned", "removed", "flipped"):
        loader = _tagged_loader(base_eval, mode, shortcut_cfg, batch_size)
        accs[f"acc_{mode}"] = round(evaluate_accuracy(model, loader, device), 4)

    accs["gap"] = round(accs["acc_aligned"] - accs["acc_removed"], 4)
    accs["n_per_class"] = len(rows) // 2
    return accs
