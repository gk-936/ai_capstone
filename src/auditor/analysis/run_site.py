"""Experiment A orchestration: hospital-detector probe + site-projection control.

Fits a probe on frozen features (NIH vs PadChest) and reports its AUC (A1). Then projects the
site direction out of the PadChest disease-test features and re-scores the FROZEN disease head,
reporting the OOD-AUC change (A2, the falsifying control). A3 (Investigation 3) fits a FRESH
probe on the projected features to measure residual site decodability after projection.
Reuses the V6 loaders and the DFR feature/head machinery so numbers stay comparable to V6.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from auditor.analysis.site_probe import project_out, score_features
from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.padchest import build_padchest_index
from auditor.mitigation.dfr import extract_features, fit_dfr_head
from auditor.report.run_v6 import _nih_reference_auc, _padchest_loader
from auditor.train.trainer import _build_model


def disease_head_arrays(model) -> tuple[np.ndarray, float]:
    """The frozen classifier head's (weight[D], bias) as numpy, in raw feature space."""
    w = model.classifier.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.classifier.bias.detach().cpu().numpy().reshape(-1)[0])
    return w, b


def _nih_feature_loader(cfg, batch_size):
    """NIH test loader (same 8-bit path as _nih_reference_auc) for feature extraction."""
    from torch.utils.data import DataLoader
    from auditor.data.nih import build_index
    from auditor.data.torch_dataset import NIHDataset

    idx = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed,
                      max_per_split={"test": 4000})
    ds = NIHDataset(rows=idx.subset("test"), images_dir=resolve_path(cfg.data.nih_images),
                    image_size=cfg.data.image_size, train=False)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def run_site_analysis(cfg: Config, logger: RunLogger, checkpoint: str,
                      labels_csv: str, images_dir: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Experiment A (site probe + projection control): checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    bs = cfg.train.batch_size
    pc_index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed, dfr_frac=0.0,
                                    max_rows=cfg.data_padchest_max_rows or None)

    # Frozen features: PadChest test (disease labels) + NIH test.
    pc_loader = _padchest_loader(pc_index, "test", images_dir, cfg.data.image_size, bs)
    pc_feats, pc_labels = extract_features(model, pc_loader, device)
    nih_feats, _ = extract_features(model, _nih_feature_loader(cfg, bs), device)

    # A1: hospital-detector probe (site = NIH:0 vs PadChest:1) with a held-out split for honest AUC.
    rng = np.random.default_rng(cfg.seed)
    X = np.concatenate([nih_feats, pc_feats])
    y = np.concatenate([np.zeros(len(nih_feats)), np.ones(len(pc_feats))]).astype(int)
    perm = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = perm[:cut], perm[cut:]
    w_site, b_site = fit_dfr_head(X[tr], y[tr], seed=cfg.seed)
    site_scores = score_features(X[te], w_site, b_site)
    site_auc = float(roc_auc_score(y[te], site_scores))
    logger.info(f"A1 hospital-detector probe AUC: {site_auc:.4f}")

    # A2: project the site direction out of PadChest disease-test features, re-score disease head.
    w_dis, b_dis = disease_head_arrays(model)
    before = float(roc_auc_score(pc_labels, score_features(pc_feats, w_dis, b_dis)))
    pc_proj = project_out(pc_feats, w_site)          # remove the probe's site axis
    after = float(roc_auc_score(pc_labels, score_features(pc_proj, w_dis, b_dis)))
    logger.info(f"A2 PadChest disease AUC before/after site-projection: {before:.4f} -> {after:.4f}")

    # A3 (Investigation 3): fit a fresh site probe on the PROJECTED combined features.
    # If projection fully removes the site direction, this AUC should collapse to ~0.5.
    X_proj = np.concatenate([
        project_out(nih_feats, w_site),
        project_out(pc_feats, w_site),
    ])
    w_site_proj, b_site_proj = fit_dfr_head(X_proj[tr], y[tr], seed=cfg.seed)
    site_residual_scores = score_features(X_proj[te], w_site_proj, b_site_proj)
    site_residual_auc = float(roc_auc_score(y[te], site_residual_scores))
    logger.info(f"A3 Residual site-probe AUC on projected features: {site_residual_auc:.4f} "
                f"(original: {site_auc:.4f}, random baseline: 0.5000)")

    metrics = {
        "site_probe_auc": round(site_auc, 4),
        "padchest_disease_auc_before": round(before, 4),
        "padchest_disease_auc_after_projection": round(after, 4),
        "projection_change": round(after - before, 4),
        "site_residual_auc_after_projection": round(site_residual_auc, 4),
        "site_decodability_drop": round(site_auc - site_residual_auc, 4),
        "n_nih": int(len(nih_feats)), "n_padchest": int(len(pc_feats)),
    }
    (logger.run_dir / "site_analysis.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.log_metrics(**metrics)
    logger.info(f"EXPERIMENT A RESULT: {metrics}")
    return metrics
