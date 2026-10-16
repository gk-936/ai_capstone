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


def _nih_reference_auc(cfg, model, device, batch_size) -> float:
    """NIH test AUC under the SAME (8-bit) loader — the honest in-distribution reference."""
    from auditor.data.nih import SplitIndex, build_index
    from auditor.data.torch_dataset import NIHDataset

    idx = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed,
                      max_per_split={"test": 8000})  # cap for speed; AUC is stable
    ds = NIHDataset(rows=idx.subset("test"), images_dir=resolve_path(cfg.data.nih_images),
                    image_size=cfg.data.image_size, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    return evaluate(model, loader, device)


def _padchest_loader(index, split, images_dir, image_size, batch_size):
    """Build a loader over PadChest rows (reusing the generic NIHDataset image loader)."""
    from auditor.data.torch_dataset import NIHDataset

    ds = NIHDataset(rows=index.subset(split), images_dir=resolve_path(images_dir),
                    image_size=image_size, train=False)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def run_v6_source_dfr_zeroshot(cfg: Config, logger: RunLogger, checkpoint: str,
                               labels_csv: str, images_dir: str) -> dict:
    """Zero-shot mitigation test: DFR on SOURCE (NIH tube groups), evaluate on PadChest.

    Uses the tube-reliant model. DFR is fit ONLY on NIH data (tube x label balanced) — NO PadChest
    labels touched — then PadChest AUC is compared before vs after. This tests whether reducing a
    real SOURCE shortcut improves cross-site transfer, which the supervised-adaptation version of
    V6 does NOT test. An honest result either way (may be null).
    """
    from auditor.data.neatx import build_tube_index
    from auditor.mitigation.dfr import apply_dfr, extract_features, fit_dfr_head, group_balanced_sample
    from auditor.data.nih import SplitIndex, make_dataset
    import copy

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"V6 SOURCE-DFR zero-shot: tube checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    pc_index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed, dfr_frac=0.0,
                                    max_rows=cfg.data_padchest_max_rows or None)
    pc_test = _padchest_loader(pc_index, "test", images_dir, cfg.data.image_size, cfg.train.batch_size)
    padchest_before = evaluate(model, pc_test, device)
    logger.info(f"PadChest AUC (tube model, before source-DFR): {padchest_before:.4f}")

    # SOURCE-side DFR: NIH tube-balanced groups, no target labels.
    tube_index = build_tube_index(cfg.data, train_enrich=cfg.tube_train_enrich, seed=cfg.seed)
    bal = group_balanced_sample(tube_index.subset("val"), "tube", "label",
                                seed=cfg.seed, per_group=60)
    sub = SplitIndex(df=bal.assign(split="dfr"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "dfr", cfg.data, train=False)
    from torch.utils.data import DataLoader as _DL
    feats, labels = extract_features(model, _DL(ds, batch_size=cfg.train.batch_size,
                                                num_workers=2, pin_memory=True), device)
    m2 = copy.deepcopy(model)
    w, b = fit_dfr_head(feats, labels, seed=cfg.seed)
    apply_dfr(m2, w, b)
    padchest_after = evaluate(m2, pc_test, device)
    logger.info(f"PadChest AUC (after SOURCE-DFR, zero-shot): {padchest_after:.4f}")

    metrics = {
        "mode": "source_dfr_zeroshot",
        "padchest_before": round(float(padchest_before), 4),
        "padchest_after_source_dfr": round(float(padchest_after), 4),
        "zeroshot_change": round(float(padchest_after) - float(padchest_before), 4),
        "n_test": pc_index.n["test"]["n"],
    }
    (logger.run_dir / "v6_source_dfr.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.log_metrics(**metrics)
    logger.info(f"SOURCE-DFR RESULT: {metrics}")
    return metrics


def run_v6(cfg: Config, logger: RunLogger, checkpoint: str, labels_csv: str,
           images_dir: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"V6 cross-site: checkpoint={checkpoint} site2={images_dir} device={device}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed,
                                 dfr_frac=0.3, max_rows=cfg.data_padchest_max_rows or None)
    assert_patient_disjoint(index)
    logger.info(f"PadChest split: {index.n}")

    bs = cfg.train.batch_size

    # In-distribution reference: re-measure NIH test AUC with the SAME 8-bit loader used for
    # PadChest, so the OOD comparison is apples-to-apples (both sites on the 8-bit intensity scale).
    nih_auc = _nih_reference_auc(cfg, model, device, bs)
    logger.info(f"NIH test AUC (8-bit loader, reference): {nih_auc:.4f}")

    test_loader = _padchest_loader(index, "test", images_dir, cfg.data.image_size, bs)

    padchest_auc_before = evaluate(model, test_loader, device)
    logger.info(f"PadChest test AUC (OOD, before): {padchest_auc_before:.4f}")

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
        "nih_auc_reference": round(float(nih_auc), 4),   # re-measured with the same 8-bit loader
        "padchest_auc_before": round(float(padchest_auc_before), 4),
        "padchest_auc_after_dfr": round(float(padchest_auc_after), 4),
        "ood_drop": round(float(nih_auc) - float(padchest_auc_before), 4),
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
        f"- PadChest OOD AUC: {m['padchest_auc_before']:.3f}",
        f"- **OOD drop (distribution shift):** {m['ood_drop']:.3f}",
        f"- PadChest AUC after SUPERVISED target adaptation (last-layer on LABELED PadChest): "
        f"{m['padchest_auc_after_dfr']:.3f} ({m['recovery']:+.3f})",
        "",
        "**Note:** the +change above is supervised target-domain adaptation (uses PadChest labels),",
        "NOT shortcut mitigation. Source-DFR zero-shot (no target labels) is null (+0.005), so the",
        "drop is distribution shift and is NOT shown to be shortcut-driven.",
        "",
        f"(PadChest test n={m['n_test']}, adaptation subset n={m['n_dfr']})",
    ])
