"""Investigation 4: Label-free OOD-gap attribution.

Generalises the zero-shot source-DFR test from run_v6.py to the synthetic tag confound.
Tests whether reducing the TAG shortcut on the source (NIH) side transfers any benefit to
cross-site performance on PadChest — without using any PadChest labels.

This is a direct analogue to run_v6_source_dfr_zeroshot, which did tube->PadChest.
Here we do: tag-trained model -> source-tag-DFR -> PadChest AUC before/after.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.padchest import build_padchest_index
from auditor.mitigation.dfr import apply_dfr, extract_features, fit_dfr_head
from auditor.mitigation.run_v5 import _TagFlagged
from auditor.report.run_v6 import _padchest_loader
from auditor.train.trainer import _build_model, evaluate


def _tag_balanced_dfr_features(model, cfg, device, seed: int, per_group: int):
    """Build a group-balanced (tagged x label) val set and extract frozen features.

    Mirrors _features_and_groups_tag but pulls directly without the full index object
    so it stays self-contained.
    """
    from auditor.data.nih import build_index, SplitIndex, make_dataset
    import pandas as pd

    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=seed)
    val = index.subset("val").copy()

    pos = val[val["label"] == 1]
    neg = val[val["label"] == 0]
    per = per_group
    rng = np.random.default_rng(seed)

    def take(df):
        n = min(per * 2, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        return df.iloc[idx]

    pos, neg = take(pos), take(neg)

    rows, tag_flags = [], []
    for grp in (pos, neg):
        half = len(grp) // 2
        for j, (_, r) in enumerate(grp.iterrows()):
            rows.append(r)
            tag_flags.append(1 if j < half else 0)

    bal = pd.DataFrame(rows).reset_index(drop=True)
    bal["_tag"] = tag_flags

    sub = SplitIndex(df=bal.assign(split="dfr"), target="Pneumothorax", n_pos={})
    base = make_dataset(sub, "dfr", cfg.data, train=False)
    ds = _TagFlagged(base, tag_flags, cfg.shortcut.corner, cfg.shortcut.size)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=0, pin_memory=True)
    feats, labels = extract_features(model, loader, device)
    return feats, labels


def run_ood_attribution(cfg: Config, logger: RunLogger, checkpoint: str,
                        labels_csv: str, images_dir: str) -> dict:
    """Zero-shot tag source-DFR: mitigate tag shortcut using NIH only, evaluate on PadChest."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Investigation 4: label-free OOD-gap attribution (tag source-DFR), "
                f"checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    bs = cfg.train.batch_size

    # PadChest test set — labels used ONLY for final AUC scoring, never for fitting.
    pc_index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed, dfr_frac=0.0,
                                    max_rows=cfg.data_padchest_max_rows or None)
    pc_test_loader = _padchest_loader(pc_index, "test", images_dir, cfg.data.image_size, bs)

    padchest_before = evaluate(model, pc_test_loader, device)
    logger.info(f"PadChest AUC (tag model, before source-DFR): {padchest_before:.4f}")

    # NIH in-distribution reference (tag model, 8-bit loader)
    from auditor.data.nih import build_index
    from auditor.data.torch_dataset import NIHDataset
    nih_idx = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed,
                          max_per_split={"test": 4000})
    nih_ds = NIHDataset(rows=nih_idx.subset("test"), images_dir=resolve_path(cfg.data.nih_images),
                        image_size=cfg.data.image_size, train=False)
    nih_loader = DataLoader(nih_ds, batch_size=bs, num_workers=0, pin_memory=True)
    nih_auc = evaluate(model, nih_loader, device)
    logger.info(f"NIH test AUC (tag model, reference): {nih_auc:.4f}")

    # SOURCE-side DFR: NIH tag-balanced groups, zero PadChest labels.
    per_group = getattr(cfg.mitigation, "per_group", 200)
    feats, labels = _tag_balanced_dfr_features(model, cfg, device, cfg.seed, per_group)
    logger.info(f"DFR balanced set: {feats.shape[0]} samples (groups over tag x label)")

    m2 = copy.deepcopy(model)
    w, b = fit_dfr_head(feats, labels, seed=cfg.seed)
    apply_dfr(m2, w, b)

    padchest_after = evaluate(m2, pc_test_loader, device)
    logger.info(f"PadChest AUC (after TAG source-DFR, zero-shot): {padchest_after:.4f}")

    ood_drop = float(nih_auc) - float(padchest_before)
    zeroshot_change = float(padchest_after) - float(padchest_before)

    metrics = {
        "mode": "tag_source_dfr_zeroshot",
        "nih_auc_reference": round(float(nih_auc), 4),
        "padchest_before": round(float(padchest_before), 4),
        "padchest_after_source_dfr": round(float(padchest_after), 4),
        "ood_drop": round(ood_drop, 4),
        "zeroshot_change": round(zeroshot_change, 4),
        "n_padchest_test": pc_index.n["test"]["n"],
        "n_dfr_balanced": int(feats.shape[0]),
    }
    (logger.run_dir / "ood_attribution.json").write_text(json.dumps(metrics, indent=2),
                                                         encoding="utf-8")
    logger.log_metrics(**metrics)
    logger.info(f"INVESTIGATION 4 RESULT: {metrics}")
    return metrics
