"""Statistical hardening of the V4a-vs-V4b control-hallucination gap.

Pure post-hoc statistics over EXISTING prediction files. No new API calls, no new crops.
Reuses score_naming.score() verbatim for the control-hallucination definition (fraction of
manifest rows with kind == 'control' whose description fires the dataset's matcher), so the
per-crop flags here are exactly the flags the original gate scoring used.

Outputs: bootstrap percentile 95% CIs (n=10000, seed=1337) on each rate and on the
V4a-minus-V4b difference (datasets resampled independently), plus a two-sided Fisher's exact
test per model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from auditor.discovery.score_naming import _MATCHERS

SEED = 1337
N_BOOT = 10000

# Canonical baselines, copied from score_occlusion.py's __main__ block (lines 116-125).
CELLS = {
    ("v4a", "scout"): (
        "experiments/v4a_naming/20260708_105244_da01a4/manifest.json",
        "experiments/v4a_naming/20260708_105244_da01a4/predictions.jsonl",
        "tag",
    ),
    ("v4a", "pixtral"): (
        "experiments/v4a_sweep/20260714_144108_a893f3/manifest.json",
        "experiments/v4a_sweep_naming/predictions_1.0_pixtral.jsonl",
        "tag",
    ),
    ("v4b", "scout"): (
        "experiments/v4b_tube/20260708_204951_3894b5/manifest.json",
        "experiments/v4b_tube/multi_vlm_compare/predictions_scout.jsonl",
        "tube",
    ),
    ("v4b", "pixtral"): (
        "experiments/v4b_tube/20260708_204951_3894b5/manifest.json",
        "experiments/v4b_tube/multi_vlm_compare/predictions_pixtral.jsonl",
        "tube",
    ),
}


def control_flags(manifest_path: str, predictions_path: str, confound: str) -> np.ndarray:
    """Per-control-crop binary flag vector, using score_naming's own matcher + control rule."""
    matcher = _MATCHERS[confound]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    preds = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                preds[row["id"]] = row.get("description", "")
    flags, missing = [], 0
    for row in manifest:
        if row["kind"] != "control":
            continue
        desc = preds.get(row["id"])
        if desc is None:  # score_naming skips missing predictions entirely
            missing += 1
            continue
        flags.append(int(matcher(desc)))
    return np.array(flags, dtype=int), missing


def boot_rates(flags: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(flags)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return flags[idx].mean(axis=1)


def pct_ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = {}
    for key, (man, preds, confound) in CELLS.items():
        flags, missing = control_flags(man, preds, confound)
        data[key] = flags
        print(f"{key}: n={len(flags)} flagged={int(flags.sum())} "
              f"rate={flags.mean():.4f} missing={missing}")

    results = []
    for model in ("scout", "pixtral"):
        a, b = data[("v4a", model)], data[("v4b", model)]
        boot_a, boot_b = boot_rates(a, rng), boot_rates(b, rng)
        boot_gap = boot_a - boot_b
        table = [[int(a.sum()), len(a) - int(a.sum())],
                 [int(b.sum()), len(b) - int(b.sum())]]
        odds, p = fisher_exact(table, alternative="two-sided")
        results.append({
            "model": model,
            "v4a": {"n": len(a), "flagged": int(a.sum()), "rate": float(a.mean()),
                    "ci95": pct_ci(boot_a)},
            "v4b": {"n": len(b), "flagged": int(b.sum()), "rate": float(b.mean()),
                    "ci95": pct_ci(boot_b)},
            "gap_v4a_minus_v4b": {"point": float(a.mean() - b.mean()),
                                  "ci95": pct_ci(boot_gap)},
            "fisher": {"table": table, "odds_ratio": float(odds), "p_two_sided": float(p)},
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
