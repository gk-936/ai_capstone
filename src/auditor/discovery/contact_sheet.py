"""Contact-sheet artifact: grid a few sample crops per opacity level for visual audit.

Not a metric — a sanity-check image alongside exp1_difficulty_curve.{json,png} so anyone
can eyeball whether a given opacity level actually looks as subtle/obvious as its measured
P/R/hallucination numbers claim, without pulling individual crop files by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from auditor.discovery.score_sweep import discover_sweep_runs


def _load(path: Path, size: int) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    return np.array(Image.fromarray(img).resize((size, size), Image.NEAREST))


def build_contact_sheet(sweep_root: str | Path, out_path: str | Path,
                        n_per_kind: int = 3, tile_size: int = 96) -> str:
    runs = discover_sweep_runs(sweep_root)
    opacities = sorted(runs)
    pad = 4

    rows_imgs = []
    row_labels = []
    for opacity in opacities:
        crops_dir = runs[opacity] / "crops"
        pos_tiles = [_load(crops_dir / f"pos_{i:03d}.png", tile_size) for i in range(n_per_kind)]
        ctrl_tiles = [_load(crops_dir / f"ctrl_{i:03d}.png", tile_size) for i in range(n_per_kind)]
        tiles = pos_tiles + ctrl_tiles
        row = np.full((tile_size, len(tiles) * (tile_size + pad) - pad, 3), 255, dtype=np.uint8)
        for i, tile in enumerate(tiles):
            x0 = i * (tile_size + pad)
            row[:, x0:x0 + tile_size] = tile
        rows_imgs.append(row)
        row_labels.append(f"opacity={opacity}  (first {n_per_kind} positive, then {n_per_kind} control)")

    width = max(r.shape[1] for r in rows_imgs)
    label_h = 20
    sheet = np.full((len(rows_imgs) * (tile_size + label_h + pad), width, 3), 255, dtype=np.uint8)
    for i, row in enumerate(rows_imgs):
        y0 = i * (tile_size + label_h + pad) + label_h
        sheet[y0:y0 + tile_size, :row.shape[1]] = row

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 72, sheet.shape[0] / 72 + 0.3 * len(rows_imgs)))
    ax.imshow(sheet)
    ax.axis("off")
    for i, label in enumerate(row_labels):
        y0 = i * (tile_size + label_h + pad)
        ax.text(0, y0 + label_h - 6, label, fontsize=9, va="bottom")
    fig.tight_layout()
    out_path = str(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    sweep_root = sys.argv[1] if len(sys.argv) > 1 else "experiments/v4a_sweep"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "docs/writeup/exp1_contact_sheet.png"
    print("contact sheet written:", build_contact_sheet(sweep_root, out_path))
