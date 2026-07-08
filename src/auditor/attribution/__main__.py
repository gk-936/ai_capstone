"""CLI: python -m auditor.attribution --config configs/v3_attribution.yaml [--override k=v]"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402

from auditor.attribution.localize_confound import run_v3  # noqa: E402
from auditor.common.config import add_config_args, load_config  # noqa: E402
from auditor.common.run_logger import RunLogger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 attribution localizes the confound")
    add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)
    metrics = run_v3(cfg, logger)
    logger.info(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
