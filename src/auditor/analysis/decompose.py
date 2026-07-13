"""Combine Experiments A/B/C into the honest OOD-gap decomposition.

Slices are estimated INDEPENDENTLY and may overlap — this is deliberately NOT presented as a
clean additive partition, and the markdown says so.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_decomposition(ood_drop: float, site_projection_change: float,
                        tta_gains: dict, label_note: str) -> tuple[dict, str]:
    best_method = max(tta_gains, key=tta_gains.get) if tta_gains else None
    best_gain = round(float(tta_gains[best_method]), 4) if best_method else 0.0
    d = {
        "ood_drop": round(float(ood_drop), 4),
        "covariate_recoverable_gain": best_gain,
        "covariate_recoverable_method": best_method,
        "shortcut_projection_change": round(float(site_projection_change), 4),
        "label_mismatch_note": label_note,
    }
    md = "\n".join([
        "# Phase 3 — NIH→PadChest OOD-gap decomposition",
        "",
        f"- **Total OOD drop:** {d['ood_drop']:.3f}",
        f"- **C. Covariate-recoverable (label-free TTA, best = {best_method}):** "
        f"{best_gain:+.3f} AUC",
        f"- **A. Shortcut residual (AUC change when site is projected out):** "
        f"{d['shortcut_projection_change']:+.3f} — ≈0 means the disease head does not ride site",
        f"- **B. Label mismatch (B-lite, qualitative):** {label_note}",
        "",
        "**Caveat:** these slices are estimated independently and may overlap. This is NOT a "
        "clean additive partition of the 0.192 gap; it bounds each candidate cause with its own "
        "control and reports each whichever way it fell.",
    ])
    return d, md


def run_decompose(site_json: str, tta_jsons: list[str], label_json: str,
                  ood_drop: float, out_dir: str) -> dict:
    site = json.loads(Path(site_json).read_text(encoding="utf-8"))
    gains = {}
    for p in tta_jsons:
        t = json.loads(Path(p).read_text(encoding="utf-8"))
        gains[t["method"]] = t["tta_gain"]
    label = json.loads(Path(label_json).read_text(encoding="utf-8"))
    note = (f"ptx isolated in {label['frac_ptx_isolated']*100:.0f}% of PadChest cases; "
            f"top co-findings {label['top_cofindings'][:3]}")
    d, md = build_decomposition(ood_drop, site["projection_change"], gains, note)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomposition.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    (out / "decomposition.md").write_text(md, encoding="utf-8")
    _print_safe(md)
    return d


def _print_safe(text: str) -> None:
    """Print unicode even on a legacy (cp1252) Windows console, without crashing."""
    import sys

    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", "replace") + b"\n")
