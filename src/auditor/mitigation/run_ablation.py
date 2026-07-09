"""Phase 3: baseline comparison (ERM / JTT / DFR) x confound x seeds + naming-off framing.

Reuses the V5 per-confound feature/eval builders. For each (confound, seed): build the frozen
features and the group-balanced set once, evaluate ERM (original head), then fit + apply JTT and
DFR heads and evaluate each. Reports worst-group / ID / gap per method, aggregated mean±std
across seeds.

Naming-off ablation: every method here operates on group labels / attribution, WITHOUT any VLM
naming — so the worst-group recovery is achieved with naming removed. The table IS the ablation.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import torch

from auditor.common.config import Config
from auditor.common.paths import resolve_device
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.mitigation.baselines import current_head, fit_jtt_head
from auditor.mitigation.dfr import apply_dfr, fit_dfr_head
from auditor.mitigation.run_v5 import (
    _eval_tag,
    _eval_tube,
    _features_and_groups_tag,
    _features_and_groups_tube,
    _load,
)


def _run_one(cfg: Config, confound: str, checkpoint: str, device: str, seed: int) -> list[dict]:
    """All three methods for one (confound, seed). Returns rows [{method, worst_group, id, gap}]."""
    set_seed(seed)
    build = _features_and_groups_tube if confound == "tube" else _features_and_groups_tag
    do_eval = _eval_tube if confound == "tube" else _eval_tag
    # Tube's confound-x-label groups are small; use a smaller per-group for it.
    cfg.mitigation.per_group = 60 if confound == "tube" else 200

    base_model = _load(cfg, checkpoint, device)
    feats, labels, index = build(base_model, cfg, device, seed)
    erm_w, erm_b = current_head(base_model)

    rows = []
    # ERM: original head, no change.
    erm = do_eval(base_model, index, cfg, device)
    rows.append({"method": "ERM", "confound": confound, "seed": seed,
                 "worst_group": erm["worst_group_acc"], "id": erm["id_acc"], "gap": erm["gap"]})

    # JTT and DFR: fresh copy of the model per method, swap the head, evaluate.
    for name, fit in (("JTT", "jtt"), ("DFR", "dfr")):
        m = copy.deepcopy(base_model)
        if fit == "jtt":
            w, b = fit_jtt_head(feats, labels, erm_w, erm_b, seed=seed)
        else:
            w, b = fit_dfr_head(feats, labels, seed=seed)
        apply_dfr(m, w, b)
        res = do_eval(m, index, cfg, device)
        rows.append({"method": name, "confound": confound, "seed": seed,
                     "worst_group": res["worst_group_acc"], "id": res["id_acc"], "gap": res["gap"]})
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    """Mean±std across seeds for each (confound, method)."""
    out = []
    keys = sorted({(r["confound"], r["method"]) for r in rows})
    for confound, method in keys:
        sel = [r for r in rows if r["confound"] == confound and r["method"] == method]
        agg = {"confound": confound, "method": method, "n_seeds": len(sel)}
        for metric in ("worst_group", "id", "gap"):
            vals = np.array([r[metric] for r in sel], dtype=float)
            agg[f"{metric}_mean"] = round(float(vals.mean()), 4)
            agg[f"{metric}_std"] = round(float(vals.std()), 4)
        out.append(agg)
    return out


def _markdown_table(agg: list[dict]) -> str:
    lines = ["| confound | method | worst-group | ID | reliance gap |",
             "|---|---|---|---|---|"]
    for a in agg:
        lines.append(
            f"| {a['confound']} | {a['method']} | "
            f"{a['worst_group_mean']:.3f} ± {a['worst_group_std']:.3f} | "
            f"{a['id_mean']:.3f} ± {a['id_std']:.3f} | "
            f"{a['gap_mean']:.3f} ± {a['gap_std']:.3f} |")
    return "\n".join(lines)


def run_phase3(cfg: Config, logger: RunLogger, seeds: list[int]) -> dict:
    device = resolve_device(cfg.device)
    logger.info(f"Phase 3 baselines: seeds={seeds} device={device}")

    targets = [("tag", cfg.mitigation.checkpoint_tag), ("tube", cfg.mitigation.checkpoint_tube)]
    all_rows = []
    for confound, ckpt in targets:
        if not ckpt:
            logger.info(f"skip {confound}: no checkpoint configured")
            continue
        for seed in seeds:
            rows = _run_one(cfg, confound, ckpt, device, seed)
            all_rows.extend(rows)
            for r in rows:
                logger.info(f"{confound} seed={seed} {r['method']}: "
                            f"worst_group={r['worst_group']} id={r['id']} gap={r['gap']}")

    agg = _aggregate(all_rows)
    table = _markdown_table(agg)
    logger.info("\n" + table)
    (logger.run_dir / "baseline_table.md").write_text(table, encoding="utf-8")
    with open(logger.run_dir / "baseline_rows.json", "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "aggregate": agg}, f, indent=2)

    logger.log_metrics(n_rows=len(all_rows), seeds=seeds, aggregate=agg)
    return {"aggregate": agg}
