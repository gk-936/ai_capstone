# Investigation 1: Severity-Stratified Tube Reliance

This investigation tests whether the pneumothorax model's tube-reliance accuracy gap varies with the severity/extent of the pneumothorax, using bounding box area ($width \times height$) on the 94 overlapping test images as a proxy for severity.

### Setup and Parameters
* **Model Checkpoint:** `experiments/v4b_tube/20260708_204138_da03d8/artifacts/model_best.pth`
* **Median BBox Area:** `38208.3822` (pixels squared)
* **Total Overlapping Pneumothorax-positive images:** `94`
  * **Low Severity (Area < Median):** 47 images (20 with-tube, 27 without-tube) -> balanced to group size 20
  * **High Severity (Area >= Median):** 47 images (16 with-tube, 31 without-tube) -> balanced to group size 16

### Measured Raw Numbers
* **Low Severity (BBox Area < 38208.3822):**
  * Accuracy with-tube: `1.0000`
  * Accuracy without-tube: `0.6000`
  * **Reliance Gap:** `0.4000` (40.0% accuracy gap)
* **High Severity (BBox Area >= 38208.3822):**
  * Accuracy with-tube: `1.0000`
  * Accuracy without-tube: `0.5625`
  * **Reliance Gap:** `0.4375` (43.75% accuracy gap)

* **Overall V4b tube-reliant test set gap (for reference):** `28.75pp` (reported in PROGRESS.md).

### Findings
1. The reliance gap remains massive and remarkably similar across both severity tiers (40.0pp for low severity vs. 43.75pp for high severity).
2. The model exhibits perfect accuracy (`1.0000`) whenever a tube is present, regardless of the lesion size, while performance drops significantly (to `0.60` and `0.56` respectively) when no tube is present.
3. This is consistent with a strong, severity-invariant shortcut reliance on the presence of the chest tube.
