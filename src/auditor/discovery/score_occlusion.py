"""Experiment 2 scoring: does the VLM's naming verdict/text change under occlusion or swap,
relative to its own baseline (unmodified-crop) description?

No new ground truth here (unlike P/R scoring) — the question is purely "did the output
change", so this reports per-(dataset, condition, model) CHANGE RATES, not precision/recall.
"""

from __future__ import annotations

import json
from pathlib import Path

from auditor.discovery.score_naming import names_confound, names_tube

_MATCHER = {"v4a": names_confound, "v4b": names_tube}


def _load_predictions(path: str | Path) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row.get("description", "")
    return out


def score_condition(baseline_path: str | Path, condition_path: str | Path,
                    matcher, positive_ids: list[str]) -> dict:
    """Direction matters, not just whether the verdict changed:
      pos_to_neg: named at baseline, NOT named after occlusion/swap — consistent with real
                  grounding (removing/replacing the confound removed the naming).
      neg_to_pos: NOT named at baseline, a NEW device/marker named after occlusion/swap —
                  still fabrication, just relocated onto different pixels.
    A single combined 'verdict_change_rate' conflates these two very different outcomes.
    """
    baseline = _load_predictions(baseline_path)
    condition = _load_predictions(condition_path)

    pos_to_neg = neg_to_pos = unchanged = text_identical = n_scored = missing = 0
    for cid in positive_ids:
        b, c = baseline.get(cid), condition.get(cid)
        if b is None or c is None:
            missing += 1
            continue
        n_scored += 1
        b_flag, c_flag = matcher(b), matcher(c)
        if b_flag and not c_flag:
            pos_to_neg += 1
        elif not b_flag and c_flag:
            neg_to_pos += 1
        else:
            unchanged += 1
        if b.strip() == c.strip():
            text_identical += 1

    verdict_changes = pos_to_neg + neg_to_pos
    return {
        "n_scored": n_scored,
        "n_missing": missing,
        "pos_to_neg": pos_to_neg,
        "neg_to_pos": neg_to_pos,
        "unchanged": unchanged,
        "verdict_change_rate": round(verdict_changes / n_scored, 4) if n_scored else None,
        "text_identical_rate": round(text_identical / n_scored, 4) if n_scored else None,
    }


def score_all(naming_dir: str | Path, occlusion_run_dir: str | Path,
             baselines: dict[str, dict[str, str]], out_path: str | Path) -> list[dict]:
    """baselines = {'v4a': {'scout': path, 'pixtral': path}, 'v4b': {...}}"""
    naming_dir = Path(naming_dir)
    occlusion_run_dir = Path(occlusion_run_dir)
    rows = []
    for ds in ("v4a", "v4b"):
        manifest = json.loads((occlusion_run_dir / ds / "manifest.json").read_text(encoding="utf-8"))
        positive_ids = [r["id"] for r in manifest]
        matcher = _MATCHER[ds]
        for cond in ("relevant", "irrelevant", "swap"):
            for model in ("scout", "pixtral"):
                cond_path = naming_dir / f"predictions_{ds}_{cond}_{model}.jsonl"
                if not cond_path.exists():
                    print(f"skip {ds}/{cond}/{model}: {cond_path} not found")
                    continue
                metrics = score_condition(baselines[ds][model], cond_path, matcher, positive_ids)
                rows.append({"dataset": ds, "condition": cond, "model": model, **metrics})
                print(f"{ds} {cond} {model}: pos->neg {metrics['pos_to_neg']} "
                     f"neg->pos {metrics['neg_to_pos']} unchanged {metrics['unchanged']} "
                     f"(verdict_change {metrics['verdict_change_rate']}, n={metrics['n_scored']})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["| dataset | condition | model | pos->neg | neg->pos | unchanged | "
             "verdict-change-rate | text-identical-rate | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['model']} | "
                     f"{r['pos_to_neg']} | {r['neg_to_pos']} | {r['unchanged']} | "
                     f"{r['verdict_change_rate']} | {r['text_identical_rate']} | {r['n_scored']} |")
    table = "\n".join(lines)
    out_path.with_suffix(".md").write_text(table, encoding="utf-8")
    print("\n" + table)
    return rows


if __name__ == "__main__":
    import sys

    occlusion_run_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/exp2_occlusion/20260714_204431_25c1ea"
    naming_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/exp2_naming"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "docs/writeup/exp2_image_independence.json"

    baselines = {
        "v4a": {
            "scout": "experiments/v4a_naming/20260708_105244_da01a4/predictions.jsonl",
            "pixtral": "experiments/v4a_sweep_naming/predictions_1.0_pixtral.jsonl",
        },
        "v4b": {
            "scout": "experiments/v4b_tube/multi_vlm_compare/predictions_scout.jsonl",
            "pixtral": "experiments/v4b_tube/multi_vlm_compare/predictions_pixtral.jsonl",
        },
    }
    score_all(naming_dir, occlusion_run_dir, baselines, out_path)
