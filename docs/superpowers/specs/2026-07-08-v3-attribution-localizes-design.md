# V3 — Attribution Localizes the Confound (design)

**Spike question:** Does Grad-CAM on the V2-cheating model concentrate on the injected tag
region?

**Gate:** peak-in-box hit-rate on tagged positive test images **> 80%** (the "pointing game":
the single highest-saliency pixel falls inside the known confound box).

## Approach
Confound-region-agnostic from the start: the confound is described by a generic box
`(y0, x0, y1, x1)`, so the same code works for the real, diffuse device confound later. For
V2 the box is computed from `(corner, size)`, not hardcoded.

Honesty control: attribute over the tagged positives (gate) AND the SAME positives untagged
(control). If tagged hit-rate is high and untagged hit-rate is low, saliency genuinely tracks
the *tag* rather than "the corner is always bright."

## Components
- `src/auditor/attribution/gradcam.py`
  - `compute_gradcam(model, x, target_layer)` → HxW saliency in [0,1]. DenseNet-121 target
    layer = `model.features.norm5`. Verify a sane map on one image before batch-running.
  - `peak_in_region(saliency, box)` → bool: is argmax(saliency) inside box?
  - `salient_crop(image, saliency, size)` → crop centered on the peak (feeds V4), edge-clamped.
  - `tag_box(corner, size, h, w)` → the `(y0,x0,y1,x1)` for a corner tag (all four corners).
- `src/auditor/attribution/localize_confound.py`
  - Load the V2 checkpoint (path from config/CLI, not hardcoded).
  - Sample N tagged positive test images (default 200) + same images untagged.
  - Compute peak-in-box hit-rate for tagged (gate) and untagged (control).
  - Export ~10 heatmap overlays + salient crops as run artifacts.
- `configs/v3_attribution.yaml` — `extends: v2_shortcut.yaml`; adds
  `attribution: {n_images, crop_size, checkpoint}`.

## Data flow
```
V2 ckpt → Grad-CAM over tagged positives   → peak-in-tag-box hit-rate  (GATE >80%)
       → Grad-CAM over same imgs untagged   → peak-in-corner hit-rate  (CONTROL, expect low)
       → export heatmaps + salient crops (artifacts; feed V4)
```

## Tests (synthetic spine)
- `peak_in_region`: max inside/outside a box → True/False.
- `salient_crop`: correct size, centered on peak, clamped at edges.
- `tag_box`: matches injected tag location for all four corners.
- Grad-CAM itself needs model+GPU → exercised in the run; the geometry around it is unit-tested.

## PASS criterion & skeptic checks
- PASS: tagged peak-in-box > 80%.
- Honesty: tagged high AND untagged low ⇒ saliency follows the tag. If BOTH high, the corner
  is trivially salient — flag and reconsider before declaring a clean PASS.
- If tagged hit-rate < 80% / diffuse → try Grad-CAM++ or a different target layer before FAIL.

## Runtime
One forward + one backward per image; ~200×2 images is a few minutes on the RTX 3050. No training.
