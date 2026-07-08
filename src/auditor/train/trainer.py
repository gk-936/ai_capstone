"""V1 baseline trainer: DenseNet-121 on NIH for one pathology, end-to-end.

Design choices driven by the task and hardware:
  * AMP (mixed precision) — halves memory so batch 32 @224px fits the RTX 3050 (6GB).
  * Gradient accumulation — simulate a larger effective batch on limited VRAM.
  * pos_weight BCE — the target pathology is ~3% prevalent; unweighted loss collapses to
    all-negative. AUC (not accuracy) is the reported metric because prevalence differs
    across splits.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import assert_patient_disjoint, build_index, make_dataset


def _build_model(backbone: str) -> nn.Module:
    import timm

    # Single logit head; grayscale is expanded to 3ch in the transform for pretrained weights.
    return timm.create_model(backbone, pretrained=True, num_classes=1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return ROC-AUC over a loader. Returns nan if a split is single-class."""
    model.eval()
    scores, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        scores.append(torch.sigmoid(logits).float().cpu())
        targets.append(y)
    scores = torch.cat(scores).numpy()
    targets = torch.cat(targets).numpy()
    if len(set(targets.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(targets, scores))


def train_v1(cfg: Config, logger: RunLogger, max_per_split: dict | None = None) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"device={device} amp={cfg.train.amp} backbone={cfg.train.backbone}")

    # --- data ---
    index = build_index(
        cfg.data, target=cfg.train.target_pathology, seed=cfg.seed, max_per_split=max_per_split
    )
    assert_patient_disjoint(index)  # fail loudly if a refactor ever introduces leakage
    sizes = {s: len(index.subset(s)) for s in ("train", "val", "test")}
    logger.info(f"split sizes: {sizes}")
    logger.info(f"positives per split: {index.n_pos}")

    train_ds = make_dataset(index, "train", cfg.data, train=True)
    val_ds = make_dataset(index, "val", cfg.data, train=False)
    test_ds = make_dataset(index, "test", cfg.data, train=False)

    nw = 4
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, num_workers=nw, pin_memory=True)

    # --- model / loss / opt ---
    model = _build_model(cfg.train.backbone).to(device)
    n_pos = max(index.n_pos["train"], 1)
    n_neg = len(index.subset("train")) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)  # rebalance rare positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    use_amp = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_auc = float("-inf")
    best_ckpt = logger.artifact_path("model_best.pth")

    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        val_auc = evaluate(model, val_loader, device)
        logger.info(
            f"epoch {epoch+1}/{cfg.train.epochs} loss={running/max(len(train_loader),1):.4f} "
            f"val_auc={val_auc:.4f} ({time.time()-t0:.0f}s)"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_ckpt)

    # Final test AUC using the best-val checkpoint.
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_auc = evaluate(model, test_loader, device)

    metrics = {
        "target": cfg.train.target_pathology,
        "best_val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "n_train": len(index.subset("train")),
        "n_val": len(index.subset("val")),
        "n_test": len(index.subset("test")),
    }
    logger.log_metrics(**metrics)
    return metrics
