"""CLI: python -m auditor.train --config configs/v1_baseline.yaml [--override k=v] [--smoke]"""

from __future__ import annotations

import argparse

from auditor.common.config import add_config_args, load_config
from auditor.common.run_logger import RunLogger
from auditor.train.trainer import train_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 baseline trainer")
    add_config_args(parser)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny capped run to prove the pipeline wires end-to-end (not a real metric).",
    )
    parser.add_argument(
        "--cap-train", type=int, default=None, help="Cap train images/epoch (speed)."
    )
    parser.add_argument("--cap-val", type=int, default=None, help="Cap val images.")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)

    max_per_split = None
    if args.smoke:
        max_per_split = {"train": 200, "val": 100, "test": 100}
        logger.info("SMOKE run: capped splits — result is a wiring check, NOT a V1 metric.")
    elif args.cap_train or args.cap_val:
        max_per_split = {}
        if args.cap_train:
            max_per_split["train"] = args.cap_train
        if args.cap_val:
            max_per_split["val"] = args.cap_val

    metrics = train_v1(cfg, logger, max_per_split=max_per_split)
    logger.info(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
