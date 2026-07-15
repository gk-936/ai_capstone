"""Canonical DFR implementation: C tuning on val, 10-draw averaging.
"""
from __future__ import annotations

import json
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import SplitIndex, make_dataset
from auditor.mitigation.dfr import apply_dfr, extract_features, fit_dfr_head, group_balanced_sample
from auditor.mitigation.run_v5 import (
    _eval_tag,
    _eval_tube,
    _features_and_groups_tag,
    _features_and_groups_tube,
    _load,
    _TagFlagged
)

def tune_dfr_c(feats_bal: np.ndarray, labels_bal: np.ndarray, seed: int) -> float:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    # Standard grid for C
    c_grid = [10**i for i in range(-4, 3)]
    
    # Standardize features
    mu = feats_bal.mean(axis=0)
    sigma = feats_bal.std(axis=0) + 1e-6
    z = (feats_bal - mu) / sigma
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_c = 1.0
    best_score = -1.0
    
    for C in c_grid:
        scores = []
        for train_idx, val_idx in cv.split(z, labels_bal):
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
            clf.fit(z[train_idx], labels_bal[train_idx])
            scores.append(clf.score(z[val_idx], labels_bal[val_idx]))
        
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_c = C
            
    return best_c

def extract_full_val_tube(model, cfg, device, seed):
    from auditor.data.neatx import build_tube_index
    index = build_tube_index(cfg.data, train_enrich=cfg.tube_train_enrich, seed=seed)
    val = index.subset("val")
    sub = SplitIndex(df=val.assign(split="val"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "val", cfg.data, train=False)
    # Using 0 workers for Windows
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=0, pin_memory=True)
    feats, labels = extract_features(model, loader, device)
    confounds = val["tube"].values
    return feats, labels, confounds, index

def extract_full_val_tag(model, cfg, device, seed):
    from auditor.data.nih import build_index
    index = build_index(cfg.data, target=cfg.train.target_pathology, seed=seed)
    val = index.subset("val").copy()
    sub = SplitIndex(df=val.assign(split="val"), target="Pneumothorax", n_pos={})
    
    # 1. Untagged feats
    ds_untagged = make_dataset(sub, "val", cfg.data, train=False)
    # Using 0 workers for Windows
    loader_untagged = DataLoader(ds_untagged, batch_size=cfg.train.batch_size, num_workers=0, pin_memory=True)
    feats_untagged, labels_untagged = extract_features(model, loader_untagged, device)
    
    # 2. Tagged feats
    base_ds = make_dataset(sub, "val", cfg.data, train=False)
    flags = [1] * len(base_ds)
    ds_tagged = _TagFlagged(base_ds, flags, cfg.shortcut.corner, cfg.shortcut.size)
    # Using 0 workers for Windows
    loader_tagged = DataLoader(ds_tagged, batch_size=cfg.train.batch_size, num_workers=0, pin_memory=True)
    feats_tagged, labels_tagged = extract_features(model, loader_tagged, device)
    
    feats = np.concatenate([feats_untagged, feats_tagged])
    labels = np.concatenate([labels_untagged, labels_tagged])
    confounds = np.concatenate([np.zeros(len(feats_untagged)), np.ones(len(feats_tagged))])
    
    return feats, labels, confounds, index

def draw_balanced_features(feats: np.ndarray, labels: np.ndarray, confounds: np.ndarray, per_group: int, draw_seed: int):
    groups = {}
    for c in (0, 1):
        for y in (0, 1):
            idx = np.where((confounds == c) & (labels == y))[0]
            if len(idx) > 0:
                groups[(c, y)] = idx
                
    smallest = min(len(g) for g in groups.values())
    n = per_group if per_group is not None else smallest
    n = min(n, smallest)
    
    rng = np.random.default_rng(draw_seed)
    selected = []
    for g in groups.values():
        selected.append(rng.choice(g, size=n, replace=False))
        
    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return feats[selected], labels[selected]

def run_canonical_dfr(cfg: Config, logger: RunLogger, confound: str, checkpoint: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Canonical DFR rerun for {confound} checkpoint={checkpoint} device={device}")
    
    # 1. Load baseline model
    model = _load(cfg, checkpoint, device)
    
    # 2. Extract entire val set
    logger.info("Extracting features over full validation set...")
    if confound == "tube":
        val_feats, val_labels, val_confounds, index = extract_full_val_tube(model, cfg, device, cfg.seed)
        per_group = cfg.mitigation.per_group if cfg.mitigation.per_group else 60
        do_eval = _eval_tube
    else:
        val_feats, val_labels, val_confounds, index = extract_full_val_tag(model, cfg, device, cfg.seed)
        per_group = cfg.mitigation.per_group if cfg.mitigation.per_group else 200
        do_eval = _eval_tag
        
    logger.info(f"Full Val Features shape: {val_feats.shape}, labels: {val_labels.shape}, positive/negative count: {val_labels.sum()}")
    
    # Evaluate BEFORE DFR
    before = do_eval(model, index, cfg, device)
    logger.info(f"BEFORE DFR: {before}")
    
    # 3. Fit standard DFR (untuned C=1.0, 1 draw) for a direct control comparison
    logger.info("Fitting standard DFR (untuned C=1.0, 1 draw)...")
    feats_1, labels_1 = draw_balanced_features(val_feats, val_labels, val_confounds, per_group, cfg.seed)
    w_std, b_std = fit_dfr_head(feats_1, labels_1, C=1.0, seed=cfg.seed)
    
    model_std = copy.deepcopy(model)
    apply_dfr(model_std, w_std, b_std)
    std_res = do_eval(model_std, index, cfg, device)
    logger.info(f"Standard DFR result: {std_res}")
    
    # 4. Tune C on the first balanced draw
    logger.info("Tuning C parameter on first balanced draw via Stratified 5-Fold CV...")
    best_c = tune_dfr_c(feats_1, labels_1, cfg.seed)
    logger.info(f"Best C selected: {best_c}")
    
    # 5. Fit logistic regression on 10 balanced draws and average weight/bias
    logger.info("Fitting and averaging over 10 balanced draws...")
    weights, biases = [], []
    for d in range(10):
        draw_seed = cfg.seed + d * 100
        feats_d, labels_d = draw_balanced_features(val_feats, val_labels, val_confounds, per_group, draw_seed)
        w, b = fit_dfr_head(feats_d, labels_d, C=best_c, seed=draw_seed)
        weights.append(w)
        biases.append(b)
        
    avg_w = np.mean(weights, axis=0)
    avg_b = float(np.mean(biases))
    
    # Apply canonical DFR
    model_can = copy.deepcopy(model)
    apply_dfr(model_can, avg_w, avg_b)
    can_res = do_eval(model_can, index, cfg, device)
    logger.info(f"Canonical DFR result: {can_res}")
    
    metrics = {
        "confound": confound,
        "best_c": best_c,
        "worst_group_before": before["worst_group_acc"],
        "id_before": before["id_acc"],
        "gap_before": before["gap"],
        "worst_group_std_dfr": std_res["worst_group_acc"],
        "id_std_dfr": std_res["id_acc"],
        "gap_std_dfr": std_res["gap"],
        "worst_group_canonical_dfr": can_res["worst_group_acc"],
        "id_canonical_dfr": can_res["id_acc"],
        "gap_canonical_dfr": can_res["gap"]
    }
    
    (logger.run_dir / "canonical_dfr.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    # Write comparison markdown report
    comparison_report = f"""# Investigation 2 — Canonical DFR Refit

* **Confound:** {confound}
* **Tuned C:** {best_c}
* **Number of Draws:** 10

| Metric | Before DFR | Standard DFR (C=1.0, 1 draw) | Canonical DFR (Tuned C, 10-draw avg) |
|---|---|---|---|
| ID Accuracy | {metrics['id_before']:.4f} | {metrics['id_std_dfr']:.4f} | {metrics['id_canonical_dfr']:.4f} |
| Worst-Group Accuracy | {metrics['worst_group_before']:.4f} | {metrics['worst_group_std_dfr']:.4f} | {metrics['worst_group_canonical_dfr']:.4f} |
| Reliance Gap | {metrics['gap_before']:.4f} | {metrics['gap_std_dfr']:.4f} | {metrics['gap_canonical_dfr']:.4f} |
"""
    (logger.run_dir / "canonical_dfr_report.md").write_text(comparison_report, encoding="utf-8")
    logger.log_metrics(**metrics)
    
    logger.info(f"INVESTIGATION 2 RESULT: {metrics}")
    return metrics
