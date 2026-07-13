"""Experiment C — label-free test-time adaptation: BN-adapt (AdaBN) + TENT.

BN-adapt re-estimates BatchNorm running statistics on the target images (forward-only, no
labels). TENT additionally entropy-minimizes over the BN affine params (light backprop, no
labels). A gain here that source-DFR could not achieve = recoverable covariate shift, not a
source-identifiable shortcut. Reported whichever way it falls.
"""

from __future__ import annotations

import json


def _is_bn(m) -> bool:
    import torch.nn as nn
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def collect_bn_params(model) -> list:
    """The BN affine parameters (weight=gamma, bias=beta) across the model."""
    params = []
    for m in model.modules():
        if _is_bn(m):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def configure_bn_adapt(model):
    """Put BN layers in cumulative-average train mode with reset stats; everything else eval."""
    model.eval()
    for m in model.modules():
        if _is_bn(m):
            m.train()
            m.momentum = None          # cumulative moving average over the target set
            m.reset_running_stats()
    return model


def bn_adapt(model, loader, device):
    """Forward-only BN statistic re-estimation on the target loader (no labels, no grad)."""
    import torch

    configure_bn_adapt(model)
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
    model.eval()
    return model


def configure_tent(model):
    """Freeze all params except BN affine; BN in train mode with running stats disabled."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if _is_bn(m):
            m.train()
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
    return model


def tent_adapt(model, loader, device, steps: int = 1, lr: float = 1e-3):
    """Entropy-minimize the sigmoid output over BN affine params (no labels)."""
    import torch

    configure_tent(model)
    opt = torch.optim.Adam(collect_bn_params(model), lr=lr)
    for _ in range(steps):
        for x, _ in loader:
            out = model(x.to(device))
            p = torch.sigmoid(out)
            ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
            opt.zero_grad()
            ent.backward()
            opt.step()
    model.eval()
    return model


def run_tta(cfg, logger, checkpoint: str, labels_csv: str, images_dir: str,
            method: str) -> dict:
    """method in {'bn', 'tent'}: adapt on the PadChest test images, re-measure OOD AUC."""
    import torch
    from auditor.common.paths import resolve_device, resolve_path
    from auditor.common.seed import set_seed
    from auditor.data.padchest import build_padchest_index
    from auditor.report.run_v6 import _padchest_loader
    from auditor.train.trainer import _build_model, evaluate

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Experiment C TTA ({method}): checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed, dfr_frac=0.0,
                                 max_rows=cfg.data_padchest_max_rows or None)
    loader = _padchest_loader(index, "test", images_dir, cfg.data.image_size, cfg.train.batch_size)

    before = evaluate(model, loader, device)
    if method == "bn":
        bn_adapt(model, loader, device)
    elif method == "tent":
        tent_adapt(model, loader, device, steps=cfg.analysis.tent_steps, lr=cfg.analysis.tent_lr)
    else:
        raise ValueError(f"unknown TTA method: {method!r} (expected 'bn' or 'tent')")
    after = evaluate(model, loader, device)

    metrics = {
        "method": method,
        "padchest_auc_before": round(float(before), 4),
        "padchest_auc_after": round(float(after), 4),
        "tta_gain": round(float(after) - float(before), 4),
    }
    (logger.run_dir / f"tta_{method}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.log_metrics(**metrics)
    logger.info(f"EXPERIMENT C ({method}) RESULT: {metrics}")
    return metrics
