"""Investigation 1 orchestration: severity-stratified tube reliance.
Computes reliance gap on pneumothorax-positive test images stratified by bbox area.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.neatx import build_tube_index
from auditor.data.nih import SplitIndex, make_dataset
from auditor.train.trainer import _build_model

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

def _loader_for(rows: pd.DataFrame, cfg: Config, batch_size: int) -> DataLoader:
    sub = SplitIndex(df=rows.assign(split="eval"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "eval", cfg.data, train=False)
    # Using 0 workers for Windows and interactive/CI safety
    return DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)

def run_severity_analysis(cfg: Config, logger: RunLogger, checkpoint: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Investigation 1: severity-stratified tube reliance, checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    # Load tube index
    idx = build_tube_index(cfg.data, train_enrich=cfg.tube_train_enrich, seed=cfg.seed)
    test = idx.subset("test")
    pos = test[test["label"] == 1]
    
    # Load NIH bboxes
    bbox_path = resolve_path(cfg.data.bbox_csv or "data/BBox_List_2017_Official_NIH.csv")
    bbox = pd.read_csv(bbox_path)
    bbox_ptx = bbox[bbox["Finding Label"] == "Pneumothorax"]

    # Compute box area
    bbox_ptx = bbox_ptx.copy()
    bbox_ptx["area"] = bbox_ptx["w"] * bbox_ptx["h]"]

    # Merge
    overlap = pos.merge(bbox_ptx, left_on="image", right_on="Image Index")
    logger.info(f"Overlapping Pneumothorax positive test images with BBoxes: {len(overlap)}")
    
    # Median split on area
    median_val = overlap["area"].median()
    logger.info(f"Median bbox area: {median_val:.4f}")

    # Stratify
    low_sev = overlap[overlap["area"] < median_val]
    high_sev = overlap[overlap["area"] >= median_val]

    # For each bucket, compute accuracy with-tube vs without-tube
    def run_stratified(df_bucket, prefix: str):
        w_tube = df_bucket[df_bucket["tube"] == 1]
        w_out_tube = df_bucket[df_bucket["tube"] == 0]
        logger.info(f"{prefix} - Total: {len(df_bucket)}, with-tube: {len(w_tube)}, without-tube: {len(w_out_tube)}")
        
        n = min(len(w_tube), len(w_out_tube))
        if n == 0:
            logger.warning(f"Empty category in {prefix}! Cannot calculate balanced reliance gap.")
            return {
                f"acc_{prefix}_with_tube": None,
                f"acc_{prefix}_without_tube": None,
                f"{prefix}_reliance_gap": None,
                f"n_{prefix}_per_group": 0
            }
        
        w_tube_sampled = w_tube.sample(n=n, random_state=cfg.seed)
        w_out_tube_sampled = w_out_tube.sample(n=n, random_state=cfg.seed)

        acc_with = _acc(model, _loader_for(w_tube_sampled, cfg, cfg.train.batch_size), device)
        acc_without = _acc(model, _loader_for(w_out_tube_sampled, cfg, cfg.train.batch_size), device)
        gap = acc_with - acc_without
        
        return {
            f"acc_{prefix}_with_tube": round(acc_with, 4),
            f"acc_{prefix}_without_tube": round(acc_without, 4),
            f"{prefix}_reliance_gap": round(gap, 4),
            f"n_{prefix}_per_group": n
        }

    low_metrics = run_stratified(low_sev, "low_severity")
    high_metrics = run_stratified(high_sev, "high_severity")

    metrics = {
        "median_area": round(float(median_val), 4),
        "total_overlapping_ptx": len(overlap),
        **low_metrics,
        **high_metrics
    }
    
    (logger.run_dir / "severity_analysis.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    # Save a markdown report under the experiments directory
    report_content = f"""# Investigation 1 — Severity-stratified tube reliance

* **Model Checkpoint:** `{checkpoint}`
* **Median BBox Area:** `{metrics['median_area']}` (px^2)
* **Total Overlapping Pneumothorax-positive images:** `{metrics['total_overlapping_ptx']}`

## Low Severity (BBox Area < median)
* **Per group size (balanced):** `{metrics['n_low_severity_per_group']}`
* **Accuracy with-tube:** `{metrics['acc_low_severity_with_tube']}`
* **Accuracy without-tube:** `{metrics['acc_low_severity_without_tube']}`
* **Reliance Gap:** `{metrics['low_severity_reliance_gap']}`

## High Severity (BBox Area >= median)
* **Per group size (balanced):** `{metrics['n_high_severity_per_group']}`
* **Accuracy with-tube:** `{metrics['acc_high_severity_with_tube']}`
* **Accuracy without-tube:** `{metrics['acc_high_severity_without_tube']}`
* **Reliance Gap:** `{metrics['high_severity_reliance_gap']}`
"""
    (logger.run_dir / "severity_report.md").write_text(report_content, encoding="utf-8")
    logger.log_metrics(**{k: v for k, v in metrics.items() if v is not None})
    logger.info(f"INVESTIGATION 1 RESULT: {metrics}")
    return metrics
