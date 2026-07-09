"""CLI: python -m auditor.mitigation --config configs/v5_dfr_tag.yaml   (or v5_dfr_tube.yaml)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402

from auditor.common.config import add_config_args, load_config  # noqa: E402
from auditor.common.run_logger import RunLogger  # noqa: E402
from auditor.mitigation.run_v5 import run_v5  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 DFR mitigation")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)
    metrics = run_v5(cfg, logger, confound=cfg.mitigation.confound,
                     checkpoint=cfg.mitigation.checkpoint)
    logger.info(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
