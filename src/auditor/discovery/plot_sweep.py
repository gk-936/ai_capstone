"""Render the Exp-1 difficulty-curve figure from exp1_difficulty_curve.json (no GPU).

Two panels vs opacity, one line per model (fixed color by model identity, never cycled):
  (left)  precision + recall (solid/dashed)
  (right) control-hallucination

Usage:
    python -m auditor.discovery.plot_sweep <exp1_difficulty_curve.json> [out.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MODEL_COLOR = {"scout": "#2980b9", "pixtral": "#c0392b"}
_MODEL_DISPLAY = {"scout": "Scout", "pixtral": "Pixtral"}


def plot(results_path: str | Path, out_path: str | Path | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    models = sorted({r["model"] for r in rows}, key=lambda m: list(_MODEL_COLOR).index(m))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for model in models:
        color = _MODEL_COLOR[model]
        pts = sorted((r for r in rows if r["model"] == model), key=lambda r: r["opacity"])
        op = [r["opacity"] for r in pts]
        ax1.plot(op, [r["precision"] for r in pts], "o-", color=color, label=f"{_MODEL_DISPLAY[model]} precision")
        ax1.plot(op, [r["recall"] for r in pts], "s--", color=color, label=f"{_MODEL_DISPLAY[model]} recall")
        ax2.plot(op, [r["control_hallucination"] for r in pts], "o-", color=color, label=_MODEL_DISPLAY[model])

    ax1.set_xlabel("Tag opacity")
    ax1.set_ylabel("Score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Precision / recall vs confound salience")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Tag opacity")
    ax2.set_ylabel("Control-hallucination rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(0.10, color="gray", lw=0.8, ls=":", label="V4a/V4b gate (<0.10)")
    ax2.set_title("Fabrication-on-clean-controls vs confound salience")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = str(out_path) if out_path else str(Path(results_path).with_name("exp1_difficulty_curve.png"))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out = plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"figure written: {out}")


if __name__ == "__main__":
    main()
