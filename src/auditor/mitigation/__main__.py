"""CLI: python -m auditor.mitigation --config configs/v5_dfr_tag.yaml   (or v5_dfr_tube.yaml)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402

from auditor.common.config import add_config_args, load_config  # noqa: E402
from auditor.common.run_logger import RunLogger  # noqa: E402
from auditor.mitigation.run_v5 import run_v5  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 DFR mitigation / Phase 3 baselines")
    add_config_args(parser)
    parser.add_argument("--phase3", action="store_true",
                        help="Run ERM/JTT/DFR baseline table + variance (naming-off ablation).")
    parser.add_argument("--seeds", default="1337,7,42",
                        help="Comma-separated seeds for --phase3 variance.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run the correlation-strength sweep (gap + DFR recovery vs p).")
    parser.add_argument("--probe", action="store_true",
                        help="Linear probe on the frozen shortcut backbone (tests 'hollow' claim).")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)

    if args.probe:
        from auditor.mitigation.probe import run_probe

        metrics = run_probe(cfg, logger, checkpoint=cfg.mitigation.checkpoint)
        logger.info(f"DONE probe: {metrics}")
        return

    if args.phase3:
        from auditor.mitigation.run_ablation import run_phase3

        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        metrics = run_phase3(cfg, logger, seeds=seeds)
        logger.info(f"DONE phase3: {metrics}")
        return

    if args.sweep:
        from auditor.mitigation.run_sweep import run_sweep

        metrics = run_sweep(cfg, logger, p_values=cfg.sweep_p_values)
        logger.info(f"DONE sweep: {len(metrics['rows'])} points")
        return

    metrics = run_v5(cfg, logger, confound=cfg.mitigation.confound,
                     checkpoint=cfg.mitigation.checkpoint)
    logger.info(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
