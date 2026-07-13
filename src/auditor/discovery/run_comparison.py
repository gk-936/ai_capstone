"""Run the multi-VLM comparison: each roster model over the same crops -> one P/R/halluc row."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from auditor.discovery.name_groq import BLIND_PROMPT
from auditor.discovery.name_multi import name_all_multi
from auditor.discovery.providers import build_provider
from auditor.discovery.score_naming import load_and_score


def _comparison_table(rows: list[dict]) -> str:
    lines = ["| model | precision | recall | control-hallucination |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['control_hallucination']:.3f} |")
    return "\n".join(lines)


def run_comparison(roster_path, crops_dir, manifest_path, out_dir, prompt=BLIND_PROMPT) -> dict:
    roster = yaml.safe_load(Path(roster_path).read_text(encoding="utf-8"))["models"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in roster.items():
        try:
            provider = build_provider(spec)
        except (RuntimeError, ValueError) as e:
            # RuntimeError -> missing keys; ValueError -> bad/unsupported provider name.
            print(f"skip {name}: {e}")   # skip this model, do not fabricate or crash the loop
            continue
        preds = out_dir / f"predictions_{name}.jsonl"
        name_all_multi(crops_dir, manifest_path, preds, provider, prompt, throttle_s=0.5)
        metrics = load_and_score(manifest_path, preds, confound="tube")
        rows.append({"model": name, **{k: metrics[k] for k in
                     ("precision", "recall", "control_hallucination")}})
        print(f"{name}: P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
              f"halluc {metrics['control_hallucination']:.3f}")
    table = _comparison_table(rows)
    (out_dir / "comparison_table.md").write_text(table, encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + table)
    return {"rows": rows}
