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


def _cmd_name_ollama(args) -> None:
    from auditor.discovery.name_ollama import name_all

    name_all(args.crops, args.manifest, args.out, model=args.model, host=args.host)


def _cmd_score(args) -> None:
    from auditor.discovery.score_naming import load_and_score

    metrics = load_and_score(args.manifest, args.predictions, confound=args.confound)
    print(json.dumps(metrics, indent=2))
    # V4a (tag/tag_grounding): P>=.90, R>=.90.  V4b (tube/tube_grounding): P>=.80, R>=.70.
    # Both: hallucination <.10. *_grounding variants are Exp 3's grounding-aware matchers
    # (see score_naming.py) -- same bars, informational only, not a re-litigation of the
    # original V4a/V4b gate (which used the unmodified tag/tube matchers).
    is_v4a = args.confound in ("tag", "tag_grounding")
    p_bar, r_bar = (0.90, 0.90) if is_v4a else (0.80, 0.70)
    gate = (
        metrics["precision"] >= p_bar
        and metrics["recall"] >= r_bar
        and metrics["control_hallucination"] < 0.10
    )
    tag = "V4a" if is_v4a else "V4b"
    print(f"\n{tag} gate (P>={p_bar} & R>={r_bar} & halluc<.10): {'PASS' if gate else 'FAIL'}")


def _cmd_name_multi(args) -> None:
    _load_dotenv()
    import yaml
    from auditor.discovery.name_groq import BLIND_PROMPT
    from auditor.discovery.name_multi import name_all_multi
    from auditor.discovery.providers import build_provider
    from auditor.discovery.score_naming import ABSTENTION_PROMPT

    roster = yaml.safe_load(open(args.roster, encoding="utf-8"))["models"]
    provider = build_provider(roster[args.model])
    prompt = ABSTENTION_PROMPT if args.abstention else BLIND_PROMPT
    name_all_multi(args.crops, args.manifest, args.out, provider, prompt, throttle_s=0.5)


def _cmd_compare(args) -> None:
    _load_dotenv()
    from auditor.discovery.run_comparison import run_comparison

    run_comparison(args.roster, args.crops, args.manifest, args.out_dir)


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

    po = sub.add_parser("name-ollama", help="Name crops via a LOCAL Ollama vision model (no API).")
    po.add_argument("--crops", required=True, help="Path to the crops/ folder.")
    po.add_argument("--manifest", required=True)
    po.add_argument("--out", required=True, help="Output predictions.jsonl path.")
    po.add_argument("--model", required=True, help="Ollama vision tag (e.g. qwen2.5vl:3b).")
    po.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    po.set_defaults(func=_cmd_name_ollama)

    ps = sub.add_parser("score", help="Score predictions.jsonl against manifest.json.")
    ps.add_argument("--manifest", required=True)
    ps.add_argument("--predictions", required=True)
    ps.add_argument("--confound", choices=["tag", "tube", "tube_abstention",
                                           "tag_grounding", "tube_grounding"], default="tag",
                    help="Matcher: 'tag' (V4a), 'tube' (V4b), 'tube_abstention' "
                         "(V4b hedge-suppressing, for the --abstention ablation), or "
                         "'tag_grounding'/'tube_grounding' (Exp 3: CheXagent's mixed "
                         "grounding-token/free-text output -- see score_naming.py).")
    ps.set_defaults(func=_cmd_score)

    pm = sub.add_parser("name-multi", help="Name crops via one roster model (multi-VLM).")
    pm.add_argument("--roster", required=True, help="configs/multi_vlm.yaml")
    pm.add_argument("--model", required=True, help="roster model name (e.g. gemini)")
    pm.add_argument("--crops", required=True)
    pm.add_argument("--manifest", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--abstention", action="store_true", help="use the abstention prompt")
    pm.set_defaults(func=_cmd_name_multi)

    pc = sub.add_parser("compare", help="Run all roster models + build the P/R/halluc table.")
    pc.add_argument("--roster", required=True)
    pc.add_argument("--crops", required=True)
    pc.add_argument("--manifest", required=True)
    pc.add_argument("--out-dir", required=True)
    pc.set_defaults(func=_cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
