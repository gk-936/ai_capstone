"""V5 experiment: DFR mitigation, measured before vs after, for the tag or tube confound.

For each confound we (1) build a group-balanced DFR set over (confound x label), (2) extract
frozen features and fit a new last-layer head, (3) swap it in, and (4) compare confound-
conflicting accuracy and in-distribution accuracy before vs after.

The two confounds differ only in how features/groups are formed:
  tag  — confound applied on the fly (TaggedDataset); groups are tagged vs untagged images.
  tube — confound is a real column in the NEATX-joined index; groups are tube vs no-tube.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import SplitIndex, make_dataset
from auditor.mitigation.dfr import apply_dfr, extract_features, fit_dfr_head, group_balanced_sample
from auditor.train.trainer import _build_model


def _load(cfg: Config, checkpoint: str, device: str):
    from auditor.common.paths import resolve_path

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    return model.to(device).eval()


class _TagFlagged(torch.utils.data.Dataset):
    """Applies the corner tag to selected images (module-level so DataLoader workers can pickle it)."""

    def __init__(self, base, flags, corner, size):
        self.base, self.flags, self.corner, self.size = base, flags, corner, size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        from auditor.shortcut.inject import add_corner_tag

        x, y = self.base[i]
        if self.flags[i]:
            x = add_corner_tag(x, corner=self.corner, size=self.size)
        return x, y


@torch.no_grad()
def _accuracy(model, loader, device, threshold=0.5) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        preds = (torch.sigmoid(model(x).squeeze(1)).cpu() >= threshold).int()
        correct += int((preds == y.int()).sum())
        total += len(y)
    return correct / max(total, 1)


# ------------------------------- TUBE confound ----------------------------------------

def _features_and_groups_tube(model, cfg, device, seed):
    """Group-balanced DFR features for the tube confound (uses the real 'tube' column)."""
    from auditor.data.neatx import build_tube_index

    index = build_tube_index(cfg.data, train_enrich=cfg.tube_train_enrich, seed=seed)
    val = index.subset("val")
    # 4 groups over (tube x label), balanced.
    bal = group_balanced_sample(val, "tube", "label", seed=seed, per_group=cfg.mitigation.per_group)
    sub = SplitIndex(df=bal.assign(split="dfr"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "dfr", cfg.data, train=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=2, pin_memory=True)
    feats, labels = extract_features(model, loader, device)
    return feats, labels, index


def _eval_tube(model, index, cfg, device):
    from auditor.evaluation.tube_reliance import tube_reliance_gap

    gap = tube_reliance_gap(model, index, cfg.data, device, batch_size=cfg.train.batch_size,
                            seed=cfg.seed)
    return {"worst_group_acc": gap["acc_ptx_without_tube"],   # the group the model fails on
            "id_acc": gap["acc_ptx_with_tube"], "gap": gap["tube_reliance_gap"]}


# ------------------------------- TAG confound -----------------------------------------

def _features_and_groups_tag(model, cfg, device, seed):
    """Group-balanced DFR features for the synthetic tag: tagged vs untagged images."""
    from auditor.data.nih import build_index
    from auditor.shortcut.inject import TaggedDataset

    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=seed)
    val = index.subset("val").copy()

    # Balance (tag x label): we control tag as a column by sampling images and marking half tagged.
    pos = val[val["label"] == 1]
    neg = val[val["label"] == 0]
    per = cfg.mitigation.per_group
    take = lambda df: df.sample(n=min(per * 2, len(df)), random_state=seed)  # noqa: E731
    pos, neg = take(pos), take(neg)

    rows, tag_flags = [], []
    for grp in (pos, neg):
        half = len(grp) // 2
        for j, (_, r) in enumerate(grp.iterrows()):
            rows.append(r)
            tag_flags.append(1 if j < half else 0)  # first half tagged, rest not
    import pandas as pd
    bal = pd.DataFrame(rows).reset_index(drop=True)
    bal["_tag"] = tag_flags

    sub = SplitIndex(df=bal.assign(split="dfr"), target="Pneumothorax", n_pos={})
    base = make_dataset(sub, "dfr", cfg.data, train=False)

    ds = _TagFlagged(base, tag_flags, cfg.shortcut.corner, cfg.shortcut.size)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=2, pin_memory=True)
    feats, labels = extract_features(model, loader, device)
    return feats, labels, index


def _eval_tag(model, index, cfg, device):
    from auditor.evaluation.reliance import reliance_gap

    sc = {"corner": cfg.shortcut.corner, "size": cfg.shortcut.size,
          "p": cfg.shortcut.p, "seed": cfg.shortcut.seed}
    g = reliance_gap(model, index, cfg.data, sc, device, batch_size=cfg.train.batch_size)
    return {"worst_group_acc": g["acc_removed"],   # accuracy when the crutch is gone
            "id_acc": g["acc_aligned"], "gap": g["gap"]}


# ------------------------------- driver -----------------------------------------------

def run_v5(cfg: Config, logger: RunLogger, confound: str, checkpoint: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"V5 DFR confound={confound} checkpoint={checkpoint} device={device}")

    model = _load(cfg, checkpoint, device)

    build_feats = _features_and_groups_tube if confound == "tube" else _features_and_groups_tag
    do_eval = _eval_tube if confound == "tube" else _eval_tag

    # Build the group-balanced DFR features once (also yields the index for evaluation).
    feats, labels, index = build_feats(model, cfg, device, cfg.seed)
    logger.info(f"DFR set: {feats.shape[0]} samples, {feats.shape[1]} features")

    before = do_eval(model, index, cfg, device)
    logger.info(f"BEFORE DFR: {before}")

    weight, bias = fit_dfr_head(feats, labels, seed=cfg.seed)
    apply_dfr(model, weight, bias)

    after = do_eval(model, index, cfg, device)
    logger.info(f"AFTER DFR: {after}")

    metrics = {
        "confound": confound,
        "worst_group_before": before["worst_group_acc"],
        "worst_group_after": after["worst_group_acc"],
        "id_before": before["id_acc"],
        "id_after": after["id_acc"],
        "gap_before": before["gap"],
        "gap_after": after["gap"],
    }
    logger.log_metrics(**metrics)
    return metrics
