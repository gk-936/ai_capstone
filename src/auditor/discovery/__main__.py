"""CLI for V4a discovery.

Export crops (local, needs the V2 checkpoint):
    python -m auditor.discovery export --config configs/v4a_naming.yaml

Score predictions after the Colab run (local):
    python -m auditor.discovery score --manifest <manifest.json> --predictions <predictions.jsonl>
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402


def _cmd_export(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.discovery.export_crops import export_crops

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)
    metrics = export_crops(cfg, logger)
    logger.info(f"DONE export: {metrics}")


def _cmd_score(args) -> None:
    from auditor.discovery.score_naming import load_and_score

    metrics = load_and_score(args.manifest, args.predictions)
    print(json.dumps(metrics, indent=2))
    gate = (
        metrics["precision"] >= 0.90
        and metrics["recall"] >= 0.90
        and metrics["control_hallucination"] < 0.10
    )
    print(f"\nV4a gate (P>=.90 & R>=.90 & halluc<.10): {'PASS' if gate else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V4a discovery: export crops / score naming")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="Export crops + manifest (needs V2 checkpoint).")
    pe.add_argument("--config", required=True)
    pe.add_argument("--override", action="append", default=[])
    pe.set_defaults(func=_cmd_export)

    ps = sub.add_parser("score", help="Score predictions.jsonl against manifest.json.")
    ps.add_argument("--manifest", required=True)
    ps.add_argument("--predictions", required=True)
    ps.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
