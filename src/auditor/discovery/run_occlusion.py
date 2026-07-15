"""Experiment 2 driver: build the relevant/irrelevant/swap condition crop sets for both
V4a and V4b, under one RunLogger-tracked run for provenance.

Usage:
    python -m auditor.discovery.run_occlusion \\
        --v4a-config configs/v4a_naming.yaml \\
        --v4a-crops experiments/v4a_naming/20260708_105244_da01a4/crops \\
        --v4a-manifest experiments/v4a_naming/20260708_105244_da01a4/manifest.json \\
        --v4b-crops experiments/v4b_tube/20260708_204951_3894b5/crops \\
        --v4b-manifest experiments/v4b_tube/20260708_204951_3894b5/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auditor.discovery.occlusion import build_conditions, compute_v4a_relevant_boxes, v4b_proxy_box


def _positive_ids(manifest_path: str | Path) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [r["id"] for r in manifest if r["kind"] == "positive"]


def build_v4a(cfg, crops_dir: str | Path, manifest_path: str | Path, out_dir: Path, logger) -> dict:
    pos_ids = _positive_ids(manifest_path)
    boxes = compute_v4a_relevant_boxes(cfg, logger=logger)
    dirs = build_conditions(crops_dir, pos_ids, boxes, cfg.discovery.crop_size, out_dir)
    logger.info(f"v4a conditions built: {[str(d) for d in dirs.values()]}")
    return {"n_positive": len(pos_ids), "crop_size": cfg.discovery.crop_size, "dirs": {k: str(v) for k, v in dirs.items()}}


def build_v4b(crops_dir: str | Path, manifest_path: str | Path, crop_size: int, out_dir: Path, logger) -> dict:
    pos_ids = _positive_ids(manifest_path)
    box = v4b_proxy_box(crop_size)
    boxes = {cid: box for cid in pos_ids}
    dirs = build_conditions(crops_dir, pos_ids, boxes, crop_size, out_dir)
    logger.info(f"v4b conditions built (PROXY box {box}): {[str(d) for d in dirs.values()]}")
    return {"n_positive": len(pos_ids), "crop_size": crop_size, "proxy_box": list(box),
            "dirs": {k: str(v) for k, v in dirs.items()}}


def main() -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger

    p = argparse.ArgumentParser()
    p.add_argument("--v4a-config", required=True)
    p.add_argument("--v4a-crops", required=True)
    p.add_argument("--v4a-manifest", required=True)
    p.add_argument("--v4b-crops", required=True)
    p.add_argument("--v4b-manifest", required=True)
    p.add_argument("--v4b-crop-size", type=int, default=64)
    args = p.parse_args()

    cfg = load_config(args.v4a_config)
    logger = RunLogger("exp2_occlusion", cfg, experiments_root=cfg.experiments_root)

    v4a_metrics = build_v4a(cfg, args.v4a_crops, args.v4a_manifest,
                            logger.run_dir / "v4a", logger)
    v4b_metrics = build_v4b(args.v4b_crops, args.v4b_manifest, args.v4b_crop_size,
                            logger.run_dir / "v4b", logger)
    logger.log_metrics(v4a=v4a_metrics, v4b=v4b_metrics)
    logger.info(f"DONE exp2 occlusion build: run_dir={logger.run_dir}")


if __name__ == "__main__":
    main()
