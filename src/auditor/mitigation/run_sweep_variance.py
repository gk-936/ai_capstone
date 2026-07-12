"""Seed-variance for the sweep's DFR-recovery step, on the EXISTING per-p checkpoints.

The correlation sweep (run_sweep.py) trains one model per correlation strength p and reports a
single-seed DFR recovery at each. This module adds seed variance to the *mitigation* step only —
it re-runs DFR + eval at several seeds on the frozen per-p checkpoints (last-layer only, no
retraining), so it is cheap. It does NOT vary the training seed (that needs full retraining);
that limitation is stated explicitly in the output.
"""

from __future__ import annotations

import copy
import json

import numpy as np

from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.mitigation.run_sweep import _dfr_recovery_for_checkpoint, _find_completed_run


def run_sweep_variance(cfg: Config, logger: RunLogger, p_values: list[float],
                       seeds: list[int]) -> dict:
    device = resolve_device(cfg.device)
    logger.info(f"sweep DFR-variance: p={p_values} seeds={seeds} device={device}")

    rows = []
    for p in p_values:
        spike = f"sweep_p{int(round(p * 100))}"
        found = _find_completed_run(cfg, spike)
        if found is None:
            logger.info(f"p={p}: no completed checkpoint found ({spike}) — skipping")
            continue
        checkpoint, gap_trained = found

        per_seed = []
        for seed in seeds:
            scfg = copy.deepcopy(cfg)
            scfg.seed = seed
            rec = _dfr_recovery_for_checkpoint(scfg, checkpoint, device)
            per_seed.append(rec)
            logger.info(f"p={p} seed={seed}: worst-group {rec['worst_group_before']:.3f}->"
                        f"{rec['worst_group_after']:.3f}, ID {rec['id_before']:.3f}->"
                        f"{rec['id_after']:.3f}")

        agg = {"p": p, "gap_trained": gap_trained, "n_seeds": len(per_seed)}
        for k in ("worst_group_after", "id_after", "gap_after"):
            vals = np.array([r[k] for r in per_seed], dtype=float)
            agg[f"{k}_mean"] = round(float(vals.mean()), 4)
            agg[f"{k}_std"] = round(float(vals.std()), 4)
        rows.append(agg)

    table = _markdown(rows)
    logger.info("\n" + table)
    (logger.run_dir / "sweep_variance_table.md").write_text(table, encoding="utf-8")
    with open(logger.run_dir / "sweep_variance.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.log_metrics(n_points=len(rows), seeds=seeds)
    return {"rows": rows}


def _markdown(rows: list[dict]) -> str:
    lines = [
        "| p | trained gap | worst-group after DFR (mean +/- std) | ID after DFR (mean +/- std) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['p']} | {r['gap_trained']:.3f} | "
            f"{r['worst_group_after_mean']:.3f} +/- {r['worst_group_after_std']:.3f} | "
            f"{r['id_after_mean']:.3f} +/- {r['id_after_std']:.3f} |")
    lines.append("")
    lines.append("Note: seeds vary the DFR/eval step only (group-balanced sampling + head fit) on")
    lines.append("frozen per-p checkpoints. Training-seed variance would require full retraining.")
    return "\n".join(lines)
