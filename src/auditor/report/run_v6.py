"""V6 cross-site OOD: NIH-trained model on PadChest, plus DFR target-site adaptation.

Measures AUC(NIH test) vs AUC(PadChest test) [the OOD drop], then last-layer DFR on a small
PadChest subset and re-measures PadChest AUC [gap narrowed?]. Reuses the trainer's model builder
and AUC eval, and the DFR head fit.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.padchest import assert_patient_disjoint, build_padchest_index
from auditor.mitigation.dfr import apply_dfr, extract_features, fit_dfr_head
from auditor.train.trainer import _build_model, evaluate


def _padchest_loader(index, split, images_dir, image_size, batch_size):
    """Build a loader over PadChest rows (reusing the generic NIHDataset image loader)."""
    from auditor.data.torch_dataset import NIHDataset

    ds = NIHDataset(rows=index.subset(split), images_dir=resolve_path(images_dir),
                    image_size=image_size, train=False)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def run_v6(cfg: Config, logger: RunLogger, checkpoint: str, labels_csv: str,
           images_dir: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"V6 cross-site: checkpoint={checkpoint} site2={images_dir} device={device}")

    # Load the NIH-trained (V1 clean) model.
    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    # PadChest index (patient-disjoint dfr/test).
    index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed,
                                 dfr_frac=0.3, max_rows=cfg.data_padchest_max_rows or None)
    assert_patient_disjoint(index)
    logger.info(f"PadChest split: {index.n}")

    bs = cfg.train.batch_size
    test_loader = _padchest_loader(index, "test", images_dir, cfg.data.image_size, bs)

    # OOD AUC (before adaptation).
    padchest_auc_before = evaluate(model, test_loader, device)
    logger.info(f"PadChest test AUC (OOD, before): {padchest_auc_before:.4f}")

    # DFR adaptation on a small PadChest subset (frozen backbone).
    dfr_loader = _padchest_loader(index, "dfr", images_dir, cfg.data.image_size, bs)
    feats, labels = extract_features(model, dfr_loader, device)
    padchest_auc_after = padchest_auc_before
    if len(set(labels.tolist())) >= 2:
        weight, bias = fit_dfr_head(feats, labels, seed=cfg.seed)
        apply_dfr(model, weight, bias)
        padchest_auc_after = evaluate(model, test_loader, device)
        logger.info(f"PadChest test AUC (after DFR adapt): {padchest_auc_after:.4f}")
    else:
        logger.info("DFR adapt skipped: PadChest dfr subset is single-class")

    metrics = {
        "nih_auc_reference": cfg.v6_nih_auc_reference,   # the known in-dist AUC (e.g. 0.8448)
        "padchest_auc_before": round(float(padchest_auc_before), 4),
        "padchest_auc_after_dfr": round(float(padchest_auc_after), 4),
        "ood_drop": round(cfg.v6_nih_auc_reference - float(padchest_auc_before), 4),
        "recovery": round(float(padchest_auc_after) - float(padchest_auc_before), 4),
        "n_test": index.n["test"]["n"], "n_dfr": index.n["dfr"]["n"],
    }
    (logger.run_dir / "v6_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (logger.run_dir / "v6_report.md").write_text(_markdown(metrics), encoding="utf-8")
    logger.log_metrics(**metrics)
    return metrics


def _markdown(m: dict) -> str:
    return "\n".join([
        "# V6 — cross-site OOD (NIH -> PadChest)",
        "",
        f"- NIH in-distribution AUC (reference): {m['nih_auc_reference']:.3f}",
        f"- PadChest OOD AUC (before adapt): {m['padchest_auc_before']:.3f}",
        f"- **OOD drop:** {m['ood_drop']:.3f}",
        f"- PadChest AUC after DFR adaptation: {m['padchest_auc_after_dfr']:.3f}",
        f"- **Recovery from adaptation:** {m['recovery']:+.3f}",
        "",
        f"(PadChest test n={m['n_test']}, DFR-adapt subset n={m['n_dfr']})",
    ])
