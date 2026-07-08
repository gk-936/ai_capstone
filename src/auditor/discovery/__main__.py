"""CLI for V4a discovery.

Export crops (local, needs the V2 checkpoint):
    python -m auditor.discovery export --config configs/v4a_naming.yaml

Name crops via the free Groq vision API (needs GROQ_API_KEY):
    python -m auditor.discovery list-models
    python -m auditor.discovery name-groq --crops <crops/> --manifest <manifest.json> \\
        --out <predictions.jsonl> --model <vision-model-id>

Score predictions (local):
    python -m auditor.discovery score --manifest <manifest.json> --predictions <predictions.jsonl>
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines) so GROQ_API_KEY can live in a gitignored .env."""
    env = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env):
        return
    with open(env, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _cmd_export(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.discovery.export_crops import export_crops

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)
    metrics = export_crops(cfg, logger)
    logger.info(f"DONE export: {metrics}")


def _cmd_export_tube(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.discovery.export_tube_crops import export_tube_crops

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger(cfg.spike, cfg, experiments_root=cfg.experiments_root)
    metrics = export_tube_crops(cfg, logger, checkpoint=args.checkpoint)
    logger.info(f"DONE export-tube: {metrics}")


def _cmd_score(args) -> None:
    from auditor.discovery.score_naming import load_and_score

    metrics = load_and_score(args.manifest, args.predictions, confound=args.confound)
    print(json.dumps(metrics, indent=2))
    # V4a (tag): P>=.90, R>=.90.  V4b (tube): P>=.80, R>=.70.  Both: hallucination <.10.
    p_bar, r_bar = (0.90, 0.90) if args.confound == "tag" else (0.80, 0.70)
    gate = (
        metrics["precision"] >= p_bar
        and metrics["recall"] >= r_bar
        and metrics["control_hallucination"] < 0.10
    )
    tag = "V4a" if args.confound == "tag" else "V4b"
    print(f"\n{tag} gate (P>={p_bar} & R>={r_bar} & halluc<.10): {'PASS' if gate else 'FAIL'}")


def _cmd_list_models(args) -> None:
    _load_dotenv()
    from auditor.discovery.name_groq import list_models

    ids = list_models()
    print("Available Groq models (look for a vision/multimodal one, e.g. *vision* or llama-4*):")
    for m in ids:
        print(" ", m)


def _cmd_name_groq(args) -> None:
    _load_dotenv()
    from auditor.discovery.name_groq import name_all

    name_all(args.crops, args.manifest, args.out, model=args.model, throttle_s=args.throttle)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4a discovery: export / name / score")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="Export crops + manifest (needs V2 checkpoint).")
    pe.add_argument("--config", required=True)
    pe.add_argument("--override", action="append", default=[])
    pe.set_defaults(func=_cmd_export)

    pt = sub.add_parser("export-tube", help="V4b: export tube crops + manifest (needs V4b ckpt).")
    pt.add_argument("--config", required=True)
    pt.add_argument("--checkpoint", required=True, help="V4b tube-reliant model checkpoint.")
    pt.add_argument("--override", action="append", default=[])
    pt.set_defaults(func=_cmd_export_tube)

    pl = sub.add_parser("list-models", help="List Groq model ids (to pick a vision model).")
    pl.set_defaults(func=_cmd_list_models)

    pn = sub.add_parser("name-groq", help="Name crops via Groq vision API -> predictions.jsonl.")
    pn.add_argument("--crops", required=True, help="Path to the crops/ folder.")
    pn.add_argument("--manifest", required=True)
    pn.add_argument("--out", required=True, help="Output predictions.jsonl path.")
    pn.add_argument("--model", required=True, help="Groq vision model id (see list-models).")
    pn.add_argument("--throttle", type=float, default=0.0, help="Seconds to sleep between calls.")
    pn.set_defaults(func=_cmd_name_groq)

    ps = sub.add_parser("score", help="Score predictions.jsonl against manifest.json.")
    ps.add_argument("--manifest", required=True)
    ps.add_argument("--predictions", required=True)
    ps.add_argument("--confound", choices=["tag", "tube"], default="tag",
                    help="Which matcher: 'tag' (V4a) or 'tube' (V4b).")
    ps.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
