"""V4b reliance measurement: does the pneumothorax model lean on the chest tube?

Unlike V2 (inject/remove a synthetic tag), we cannot add/remove a real tube. Instead we
compare the model on two held-out groups of pneumothorax-POSITIVE images:
    with-tube   — positives that carry an annotated chest tube
    without-tube — positives with no tube
A model that shortcuts on the device scores well on with-tube and poorly on without-tube.
Reliance gap = acc(with-tube) - acc(without-tube), on a balanced-by-group subset.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from auditor.common.config import DataConfig
from auditor.data.nih import SplitIndex, make_dataset
from auditor.data.neatx import TubeIndex


@torch.no_grad()
def _acc(model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs = torch.sigmoid(model(x).squeeze(1)).float().cpu()
        preds = (probs >= threshold).int()
        correct += int((preds == y.int()).sum())
        total += len(y)
    return correct / max(total, 1)


def _loader_for(rows: pd.DataFrame, data_cfg: DataConfig, batch_size: int) -> DataLoader:
    sub = SplitIndex(df=rows.assign(split="eval"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "eval", data_cfg, train=False)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def tube_reliance_gap(model, index: TubeIndex, data_cfg: DataConfig, device: str,
                      batch_size: int = 16, seed: int = 1337, per_group: int = 400) -> dict:
    """Accuracy on PTX+tube vs PTX+no-tube (held-out test positives). Gap = with - without."""
    test = index.subset("test")
    pos = test[test["label"] == 1]
    with_tube = pos[pos["tube"] == 1]
    without_tube = pos[pos["tube"] == 0]

    n = min(len(with_tube), len(without_tube), per_group)
    with_tube = with_tube.sample(n=n, random_state=seed)
    without_tube = without_tube.sample(n=n, random_state=seed)

    acc_with = _acc(model, _loader_for(with_tube, data_cfg, batch_size), device)
    acc_without = _acc(model, _loader_for(without_tube, data_cfg, batch_size), device)

    return {
        "acc_ptx_with_tube": round(acc_with, 4),
        "acc_ptx_without_tube": round(acc_without, 4),
        "tube_reliance_gap": round(acc_with - acc_without, 4),
        "n_per_group": n,
    }
