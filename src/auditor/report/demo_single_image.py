"""Live-demo core: one chest X-ray + one checkpoint -> before/after DFR Grad-CAM.

CLI usage:
    python -m auditor.report.demo_single_image --image path/to/xray.png

Optional:
    --config configs/auditor_tube.yaml   (default; picks the tube-shortcut checkpoint + confound)
    --checkpoint path/to/model_best.pth  (override the config's checkpoint)
    --confound tube|tag                  (override the config's confound)
    --out out/demo_single.png            (where to save the figure)

What it does, in order:
  1. loads the checkpoint and the given image with the exact same preprocessing used in training
  2. runs the classifier once: P(pneumothorax) and its Grad-CAM (BEFORE mitigation)
  3. fits a DFR head on the confound's group-balanced set (the same procedure as V5) and swaps
     it into a copy of the model
  4. runs the mitigated model on the same image: P(pneumothorax) and its Grad-CAM (AFTER)
  5. saves a 3-panel [input | Grad-CAM before | Grad-CAM after] figure

This is the single-image version of the batch demo in auditor.report.run_auditor: same
detect -> localize -> mitigate -> localize-again loop, run on whatever image you hand it
instead of a fixed sample from the test set.

`AuditorSession` below does the slow, image-independent part (load checkpoint + fit DFR) once
and exposes a fast per-image `audit()` call — this is what the GUI (demo_gui.py) uses so a
live audience isn't waiting on DFR refitting for every uploaded image.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from auditor.attribution.gradcam import compute_gradcam
from auditor.attribution.localize_confound import _denormalize
from auditor.common.config import Config, load_config
from auditor.common.paths import resolve_device
from auditor.common.seed import set_seed
from auditor.data.torch_dataset import _build_transform, _open_as_8bit_L
from auditor.mitigation.dfr import apply_dfr, fit_dfr_head
from auditor.mitigation.run_v5 import (
    _features_and_groups_tag,
    _features_and_groups_tube,
    _load,
)


def load_single_image(path: str, image_size: int) -> torch.Tensor:
    """Preprocess one arbitrary chest X-ray file with the model's exact eval transform."""
    img = _open_as_8bit_L(path)
    tf = _build_transform(image_size, train=False)
    return tf(img)


@torch.no_grad()
def predict(model, x: torch.Tensor, device: str) -> float:
    model.eval()
    logit = model(x.unsqueeze(0).to(device))
    return float(torch.sigmoid(logit).item())


@dataclass
class AuditResult:
    p_before: float
    p_after: float
    cam_before: np.ndarray
    cam_after: np.ndarray
    img_u8: np.ndarray
    confound: str


class AuditorSession:
    """Loads a checkpoint and fits its DFR-mitigated twin once; audits any number of images fast.

    The slow step (loading the checkpoint, extracting features, fitting the DFR head) does not
    depend on the image being audited — only on the checkpoint and confound — so it happens once
    in __init__. Each call to `.audit(image_path)` after that is just two forward passes plus two
    Grad-CAMs, which is fast enough for a live audience to wait on between images.
    """

    def __init__(self, config_path: str = "configs/auditor_tube.yaml",
                checkpoint: str | None = None, confound: str | None = None,
                overrides: list[str] | None = None):
        self.cfg: Config = load_config(config_path, overrides=overrides or [])
        set_seed(self.cfg.seed)
        self.device = resolve_device(self.cfg.device)
        self.checkpoint = checkpoint or self.cfg.mitigation.checkpoint
        self.confound = confound or self.cfg.mitigation.confound

        self.model_before = _load(self.cfg, self.checkpoint, self.device)

        build_feats = (_features_and_groups_tube if self.confound == "tube"
                      else _features_and_groups_tag)
        feats, labels, _index = build_feats(self.model_before, self.cfg, self.device, self.cfg.seed)
        weight, bias = fit_dfr_head(feats, labels, seed=self.cfg.seed)

        self.model_after = copy.deepcopy(self.model_before)
        apply_dfr(self.model_after, weight, bias)

    def audit(self, image_path: str) -> AuditResult:
        x = load_single_image(image_path, self.cfg.data.image_size)
        p_before = predict(self.model_before, x, self.device)
        cam_before = compute_gradcam(self.model_before, x, self.cfg.attribution.target_layer)
        p_after = predict(self.model_after, x, self.device)
        cam_after = compute_gradcam(self.model_after, x, self.cfg.attribution.target_layer)
        return AuditResult(p_before=p_before, p_after=p_after, cam_before=cam_before,
                           cam_after=cam_after, img_u8=_denormalize(x), confound=self.confound)


def render_figure(result: AuditResult, out_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(result.img_u8)
    axes[0].set_title("input")
    axes[1].imshow(result.img_u8)
    axes[1].imshow(result.cam_before, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM BEFORE DFR\nP={result.p_before:.4f}")
    axes[2].imshow(result.img_u8)
    axes[2].imshow(result.cam_after, cmap="jet", alpha=0.45)
    axes[2].set_title(f"Grad-CAM AFTER DFR\nP={result.p_after:.4f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Auditor demo — confound: {result.confound}", fontsize=13, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_demo(config_path: str, image_path: str, out_path: str,
            checkpoint: str | None = None, confound: str | None = None,
            overrides: list[str] | None = None) -> str:
    print(f"[1/4] loading checkpoint + fitting DFR (config={config_path})")
    session = AuditorSession(config_path, checkpoint=checkpoint, confound=confound,
                             overrides=overrides)
    print(f"[2/4] auditing image: {image_path}")
    result = session.audit(image_path)
    print(f"      P(pneumothorax) BEFORE mitigation: {result.p_before:.3f}")
    print(f"      P(pneumothorax) AFTER mitigation:  {result.p_after:.3f}")
    print(f"[3/4] rendering figure to {out_path}")
    render_figure(result, out_path)
    print("[4/4] done")
    print(f"  P(pneumothorax) before -> after: {result.p_before:.3f} -> {result.p_after:.3f}")
    print(f"  figure saved to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="One-image live demo: Grad-CAM before/after DFR")
    parser.add_argument("--config", default="configs/auditor_tube.yaml",
                        help="Path to a YAML config file (default: the tube-shortcut demo config).")
    parser.add_argument(
        "--override", action="append", default=[], metavar="key.subkey=value",
        help="Override a config value (repeatable).",
    )
    parser.add_argument("--image", required=True, help="Path to any chest X-ray image file.")
    parser.add_argument("--checkpoint", default=None, help="Override the config's checkpoint.")
    parser.add_argument("--confound", default=None, choices=[None, "tube", "tag"],
                        help="Override the config's confound (tube or tag).")
    parser.add_argument("--out", default="out/demo_single.png", help="Output figure path.")
    args = parser.parse_args()

    run_demo(args.config, args.image, args.out,
             checkpoint=args.checkpoint, confound=args.confound, overrides=args.override)


if __name__ == "__main__":
    main()
