"""V4b trainer: train a pneumothorax classifier on a tube-enriched split to induce the natural
chest-tube shortcut, then measure the tube-reliance gap.

Reuses the V1 trainer's model builder, AMP loop, and AUC eval; swaps in the TubeIndex and the
tube-reliance measurement. The trained checkpoint feeds V4b attribution + naming.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.nih import SplitIndex, make_dataset
from auditor.data.neatx import assert_patient_disjoint, build_tube_index
from auditor.evaluation.tube_reliance import tube_reliance_gap
from auditor.train.trainer import _build_model, evaluate


def _loader(rows, data_cfg, batch_size, train, shuffle):
    sub = SplitIndex(df=rows.assign(split="eval"), target="Pneumothorax", n_pos={})
    ds = make_dataset(sub, "eval", data_cfg, train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


def train_v4b(cfg: Config, logger: RunLogger, train_enrich: float = 0.9) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"device={device} amp={cfg.train.amp} train_enrich={train_enrich}")

    index = build_tube_index(cfg.data, train_enrich=train_enrich, seed=cfg.seed)
    assert_patient_disjoint(index)
    logger.info(f"split counts: {index.n}")

    train_rows = index.subset("train")
    val_rows = index.subset("val")
    train_loader = _loader(train_rows, cfg.data, cfg.train.batch_size, train=True, shuffle=True)
    val_loader = _loader(val_rows, cfg.data, cfg.train.batch_size, train=False, shuffle=False)

    model = _build_model(cfg.train.backbone).to(device)
    n_pos = max(int((train_rows["label"] == 1).sum()), 1)
    n_neg = int((train_rows["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    use_amp = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt = logger.artifact_path("model_best.pth")
    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        val_auc = evaluate(model, val_loader, device)
        logger.info(f"epoch {epoch+1}/{cfg.train.epochs} loss={running/max(len(train_loader),1):.4f} "
                    f"val_auc={val_auc:.4f} ({time.time()-t0:.0f}s)")
        torch.save(model.state_dict(), ckpt)  # keep final epoch (shortcut learned fast)

    # Measure the tube-reliance gap on held-out positives.
    gap = tube_reliance_gap(model, index, cfg.data, device, batch_size=cfg.train.batch_size,
                            seed=cfg.seed)
    logger.info(f"tube reliance: {gap}")

    metrics = {"train_enrich": train_enrich, **gap,
               "n_train": len(train_rows), "checkpoint": str(ckpt)}
    logger.log_metrics(**metrics)
    return metrics
