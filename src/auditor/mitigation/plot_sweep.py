"""Render the correlation-strength sweep figure from sweep_results.json (no GPU).

Two panels vs p:
  (left)  trained reliance gap  — how much the model cheats as correlation strengthens.
  (right) DFR worst-group and ID, before vs after — mitigation recovery + ID retention.

Usage:
    python -m auditor.mitigation.plot_sweep <sweep_results.json> [out.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def plot(results_path: str | Path, out_path: str | Path | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda r: r["p"])
    p = [r["p"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # Left: reliance gap vs p.
    ax1.plot(p, [r["gap_trained"] for r in rows], "o-", color="#c0392b")
    ax1.set_xlabel("correlation strength p")
    ax1.set_ylabel("trained reliance gap")
    ax1.set_title("Shortcut reliance vs strength")
    ax1.axhline(0.0, color="gray", lw=0.6)
    ax1.grid(alpha=0.3)

    # Right: DFR before/after worst-group + ID.
    ax2.plot(p, [r["worst_group_before"] for r in rows], "o--", color="#2980b9",
             label="worst-group (before)")
    ax2.plot(p, [r["worst_group_after"] for r in rows], "o-", color="#2980b9",
             label="worst-group (after DFR)")
    ax2.plot(p, [r["id_before"] for r in rows], "s--", color="#27ae60", label="ID (before)")
    ax2.plot(p, [r["id_after"] for r in rows], "s-", color="#27ae60", label="ID (after DFR)")
    ax2.set_xlabel("correlation strength p")
    ax2.set_ylabel("accuracy")
    ax2.set_title("DFR mitigation vs strength")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = str(out_path) if out_path else str(Path(results_path).with_name("sweep_figure.png"))
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
