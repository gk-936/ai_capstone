"""Correlation-strength sweep: reliance gap and DFR recovery vs confound strength p.

Pure orchestration — reuses train_v1 (which trains the tagged model AND returns its reliance
gap), then the V5 tag DFR path (features -> fit head -> before/after). One row per p.
"""

from __future__ import annotations

import copy
import json

import numpy as np

from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.mitigation.dfr import apply_dfr, fit_dfr_head
from auditor.mitigation.run_v5 import _eval_tag, _features_and_groups_tag, _load
from auditor.train.trainer import train_v1


def _dfr_recovery_for_checkpoint(cfg: Config, checkpoint: str, device: str) -> dict:
    """Load a tagged model checkpoint and run the tag DFR before/after (reuses V5 tag path)."""
    model = _load(cfg, checkpoint, device)
    feats, labels, index = _features_and_groups_tag(model, cfg, device, cfg.seed)
    before = _eval_tag(model, index, cfg, device)
    weight, bias = fit_dfr_head(feats, labels, seed=cfg.seed)
    apply_dfr(model, weight, bias)
    after = _eval_tag(model, index, cfg, device)
    return {"worst_group_before": before["worst_group_acc"],
            "worst_group_after": after["worst_group_acc"],
            "id_before": before["id_acc"], "id_after": after["id_acc"],
            "gap_before": before["gap"], "gap_after": after["gap"]}


def _markdown(rows: list[dict]) -> str:
    lines = ["| p | reliance gap (trained) | worst-group before->after | ID before->after | gap after DFR |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['p']} | {r['gap_trained']:.3f} | "
            f"{r['worst_group_before']:.3f} -> {r['worst_group_after']:.3f} | "
            f"{r['id_before']:.3f} -> {r['id_after']:.3f} | {r['gap_after']:.3f} |")
    return "\n".join(lines)


def run_sweep(cfg: Config, logger: RunLogger, p_values: list[float]) -> dict:
    device = resolve_device(cfg.device)
    logger.info(f"correlation-strength sweep: p_values={p_values} device={device}")

    rows = []
    for p in p_values:
        set_seed(cfg.seed)
        pcfg = copy.deepcopy(cfg)
        pcfg.shortcut.enabled = True
        pcfg.shortcut.p = float(p)
        pcfg.spike = f"sweep_p{int(round(p * 100))}"

        # Train the tagged model at this p; train_v1 returns the reliance gap and writes model_best.
        sub_logger = RunLogger(pcfg.spike, pcfg, experiments_root=cfg.experiments_root)
        train_metrics = train_v1(pcfg, sub_logger)
        checkpoint = str(sub_logger.artifact_path("model_best.pth"))
        gap_trained = train_metrics.get("gap", float("nan"))
        logger.info(f"p={p}: trained reliance gap={gap_trained}")

        # DFR recovery on that checkpoint.
        rec = _dfr_recovery_for_checkpoint(pcfg, checkpoint, device)
        row = {"p": p, "gap_trained": gap_trained, "checkpoint": checkpoint, **rec}
        rows.append(row)
        logger.info(f"p={p} DFR: worst-group {rec['worst_group_before']:.3f}->"
                    f"{rec['worst_group_after']:.3f}, ID {rec['id_before']:.3f}->"
                    f"{rec['id_after']:.3f}")

    table = _markdown(rows)
    logger.info("\n" + table)
    (logger.run_dir / "sweep_table.md").write_text(table, encoding="utf-8")
    with open(logger.run_dir / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    logger.log_metrics(n_points=len(rows), p_values=list(p_values))
    return {"rows": rows}
