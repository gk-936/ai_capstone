"""Linear probe: does a shortcut-trained backbone actually encode pneumothorax features?

Freezes the backbone of a tag-trained (p=0.95) model, trains ONLY a fresh linear head on the
FROZEN features of TAG-FREE images, and measures pneumothorax AUC on TAG-FREE test images.

Interpretation:
  * If probe AUC is near the clean V1 baseline (~0.84), the backbone learned real core features
    despite the shortcut -> the "hollow shortcut model" claim is FALSE.
  * If probe AUC is near chance (~0.5), the backbone encodes little beyond the tag -> "hollow"
    would hold.

This is the control that tests a claim asserted (not measured) in earlier notes.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import SplitIndex, build_index, make_dataset
from auditor.mitigation.dfr import extract_features, fit_dfr_head
from auditor.train.trainer import _build_model


def _loader(index, split, cfg, batch_size, cap=None):
    rows = index.subset(split)
    if cap is not None and len(rows) > cap:
        rows = rows.sample(n=cap, random_state=cfg.seed).reset_index(drop=True)
    sub = SplitIndex(df=rows.assign(split="probe"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "probe", cfg.data, train=False)   # make_dataset uses TAG-FREE images
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def run_probe(cfg: Config, logger: RunLogger, checkpoint: str,
              train_cap: int = 15000, test_cap: int = 8000) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"LINEAR PROBE on frozen backbone: {checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed)

    # Fit a fresh linear head on FROZEN features of TAG-FREE train images.
    train_loader = _loader(index, "train", cfg, cfg.train.batch_size, cap=train_cap)
    feats_tr, labels_tr = extract_features(model, train_loader, device)
    logger.info(f"probe train features: {feats_tr.shape}, pos={int(labels_tr.sum())}")
    weight, bias = fit_dfr_head(feats_tr, labels_tr, seed=cfg.seed)

    # Evaluate on FROZEN features of TAG-FREE test images -> AUC.
    test_loader = _loader(index, "test", cfg, cfg.train.batch_size, cap=test_cap)
    feats_te, labels_te = extract_features(model, test_loader, device)
    scores = feats_te @ weight + bias
    probe_auc = float(roc_auc_score(labels_te, scores)) if len(set(labels_te.tolist())) > 1 else float("nan")

    metrics = {
        "checkpoint": checkpoint,
        "probe_auc_tagfree": round(probe_auc, 4),
        "v1_clean_reference": 0.8448,
        "n_train_probe": int(len(labels_tr)), "n_test_probe": int(len(labels_te)),
        # Three-way reading (not binary): chance ~0.50, clean baseline ~0.845.
        "verdict": (
            "near-chance -> hollow (little core signal)" if probe_auc < 0.58 else
            "real but DEGRADED core features (above chance, below clean baseline)"
            if probe_auc < 0.80 else
            "core features intact (comparable to clean baseline)"),
    }
    logger.log_metrics(**metrics)
    logger.info(f"PROBE RESULT: {metrics}")
    return metrics
