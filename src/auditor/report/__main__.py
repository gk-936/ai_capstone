"""CLI: python -m auditor.report --config configs/auditor_tube.yaml [--predictions ... --manifest ...]"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402

from auditor.common.config import add_config_args, load_config  # noqa: E402
from auditor.common.run_logger import RunLogger  # noqa: E402
from auditor.report.run_auditor import run_auditor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end auditor loop + demo figure")
    add_config_args(parser)
    parser.add_argument("--predictions", default=None, help="Optional VLM predictions.jsonl.")
    parser.add_argument("--manifest", default=None, help="Manifest for the predictions.")
    parser.add_argument("--n-demo", type=int, default=3, help="Images in the demo figure.")
    parser.add_argument("--v6", action="store_true", help="Run V6 cross-site OOD on PadChest.")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)

    if args.v6:
        from auditor.report.run_v6 import run_v6

        metrics = run_v6(cfg, logger, checkpoint=cfg.mitigation.checkpoint,
                         labels_csv=cfg.data.padchest_labels_csv,
                         images_dir=cfg.data.site2_images)
        logger.info(f"DONE V6: {metrics}")
        return

    report = run_auditor(cfg, logger, confound=cfg.mitigation.confound,
                         checkpoint=cfg.mitigation.checkpoint,
                         predictions=args.predictions, manifest=args.manifest,
                         n_demo=args.n_demo)
    logger.info(f"DONE: report at {logger.run_dir}/audit_report.md")


if __name__ == "__main__":
    main()
