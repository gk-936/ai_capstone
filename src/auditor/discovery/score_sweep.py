"""Experiment 1 (difficulty curve): aggregate P/R/hallucination across the opacity sweep.

Reads each v4a_sweep run's config.yaml (for its opacity + manifest path) and matches it to
predictions written by the sweep naming script as predictions_<opacity>_<model>.jsonl. Scores
each with the existing tag matcher (no new matcher needed — the confound identity is
unchanged, only its salience) and writes one table covering every (opacity, model) point.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from auditor.discovery.score_naming import load_and_score


def discover_sweep_runs(sweep_root: str | Path) -> dict[float, Path]:
    """opacity -> run_dir for every experiments/v4a_sweep/<run> directory."""
    sweep_root = Path(sweep_root)
    out: dict[float, Path] = {}
    for run_dir in sorted(sweep_root.iterdir()):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        opacity = cfg["discovery"]["opacity"]
        out[float(opacity)] = run_dir
    return out


def score_sweep(sweep_root: str | Path, naming_dir: str | Path, models: list[str],
                out_path: str | Path) -> list[dict]:
    runs = discover_sweep_runs(sweep_root)
    naming_dir = Path(naming_dir)
    rows = []
    for opacity, run_dir in sorted(runs.items()):
        manifest_path = run_dir / "manifest.json"
        for model in models:
            preds_path = naming_dir / f"predictions_{opacity}_{model}.jsonl"
            if not preds_path.exists():
                print(f"skip opacity={opacity} model={model}: {preds_path} not found")
                continue
            metrics = load_and_score(manifest_path, preds_path, confound="tag")
            rows.append({"opacity": opacity, "model": model, **metrics})
            print(f"opacity={opacity} model={model}: "
                  f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
                  f"halluc {metrics['control_hallucination']:.3f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["| opacity | model | precision | recall | control-hallucination |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['opacity']} | {r['model']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['control_hallucination']:.3f} |")
    table = "\n".join(lines)
    out_path.with_suffix(".md").write_text(table, encoding="utf-8")
    print("\n" + table)
    return rows


if __name__ == "__main__":
    import sys

    sweep_root = sys.argv[1] if len(sys.argv) > 1 else "experiments/v4a_sweep"
    naming_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/v4a_sweep_naming"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "docs/writeup/exp1_difficulty_curve.json"
    score_sweep(sweep_root, naming_dir, ["scout", "pixtral"], out_path)
