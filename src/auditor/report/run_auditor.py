"""End-to-end auditor: detect -> localize -> [name] -> mitigate -> report + demo figure.

Pure orchestration over already-tested modules. Produces audit_report.{json,md} and the
before/after Grad-CAM figure for one (checkpoint, confound).
"""

from __future__ import annotations

import copy
import json

import torch

from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.mitigation.dfr import apply_dfr, fit_dfr_head
from auditor.mitigation.run_v5 import (
    _eval_tag,
    _eval_tube,
    _features_and_groups_tag,
    _features_and_groups_tube,
    _load,
)
from auditor.report.demo_figure import collect_positive_images, render_before_after


def _naming_metrics(manifest_path, predictions_path, confound):
    """Optional VLM-naming score if a predictions file was supplied; else None."""
    from auditor.discovery.score_naming import load_and_score

    kind = "tube" if confound == "tube" else "tag"
    return load_and_score(manifest_path, predictions_path, confound=kind)


def run_auditor(cfg: Config, logger: RunLogger, confound: str, checkpoint: str,
                predictions: str | None = None, manifest: str | None = None,
                n_demo: int = 3) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"AUDITOR confound={confound} checkpoint={checkpoint} device={device}")

    build = _features_and_groups_tube if confound == "tube" else _features_and_groups_tag
    do_eval = _eval_tube if confound == "tube" else _eval_tag

    model_before = _load(cfg, checkpoint, device)
    feats, labels, index = build(model_before, cfg, device, cfg.seed)

    before = do_eval(model_before, index, cfg, device)
    logger.info(f"detect (before): {before}")

    model_after = copy.deepcopy(model_before)
    weight, bias = fit_dfr_head(feats, labels, seed=cfg.seed)
    apply_dfr(model_after, weight, bias)
    after = do_eval(model_after, index, cfg, device)
    logger.info(f"mitigate (after DFR): {after}")

    report = {
        "confound": confound,
        "checkpoint": checkpoint,
        "detect": {"reliance_gap": before["gap"]},
        "mitigate": {
            "worst_group_before": before["worst_group_acc"],
            "worst_group_after": after["worst_group_acc"],
            "id_before": before["id_acc"], "id_after": after["id_acc"],
            "gap_before": before["gap"], "gap_after": after["gap"],
        },
    }

    if predictions and manifest:
        report["name"] = _naming_metrics(manifest, predictions, confound)
        logger.info(f"name: {report['name']}")
    else:
        report["name"] = None
        logger.info("name: skipped (no predictions supplied) — loop runs naming-off")

    images = collect_positive_images(index, confound, cfg, n_demo, cfg.seed)
    fig_path = render_before_after(
        model_before, model_after, images, cfg.attribution.target_layer,
        out_path=str(logger.run_dir / "before_after_gradcam.png"))
    report["demo_figure"] = fig_path
    logger.info(f"demo figure: {fig_path}")

    with open(logger.run_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (logger.run_dir / "audit_report.md").write_text(_markdown(report), encoding="utf-8")

    logger.log_metrics(**{"confound": confound, **report["mitigate"]})
    return report


def _markdown(r: dict) -> str:
    m = r["mitigate"]
    lines = [
        f"# Auditor report — {r['confound']} confound",
        "",
        f"**Detect:** reliance gap = {r['detect']['reliance_gap']:.3f}",
        "",
        "**Mitigate (DFR):**",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| worst-group acc | {m['worst_group_before']:.3f} | {m['worst_group_after']:.3f} |",
        f"| in-distribution acc | {m['id_before']:.3f} | {m['id_after']:.3f} |",
        f"| reliance gap | {m['gap_before']:.3f} | {m['gap_after']:.3f} |",
        "",
    ]
    if r.get("name"):
        n = r["name"]
        lines += [f"**Name (VLM):** precision {n['precision']:.2f}, recall {n['recall']:.2f}, "
                  f"control-hallucination {n['control_hallucination']:.2f}", ""]
    else:
        lines += ["**Name:** skipped — the loop mitigates without VLM naming.", ""]
    lines += [f"**Demo figure:** `{r['demo_figure']}`"]
    return "\n".join(lines)
