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

* **Overall V4b tube-reliant test set gap (for reference):** `28.75pp`
  (`experiments/v4b_tube/20260708_204138_da03d8/metrics.json`: with-tube `0.9575`, without-tube
  `0.6700`, `n_per_group=400`).

### Why both strata exceed the pooled 28.75pp

Both severity gaps (40.0pp, 43.75pp) sit *above* the pooled figure they are referenced against.
This is not a contradiction: the two populations differ. The pooled gap is measured over all
tube-index test positives (400/group), whereas the severity strata are restricted to positives
that **also** carry an NIH BBox annotation (94 images), then median-split on box area. That
restriction moves **both** terms of the gap in the widening direction — with-tube accuracy rises
`0.9575 → 1.0000`, and without-tube accuracy falls `0.6700 → 0.6000/0.5625` — so the difference
of the two necessarily exceeds the pooled value. The BBox-annotated subset is a curated,
non-random slice of the test set, and this comparison is between different populations, not a
like-for-like contrast.

### Findings

1. **The measured gap is large in both severity tiers** (40.0pp low, 43.75pp high), and the model
   is perfectly accurate (`1.0000`) whenever a tube is present, in both tiers, regardless of
   lesion size.

2. **This study is underpowered to distinguish severity-invariance from a real severity effect.**
   The strata carry only **20** and **16** images per group. The without-tube 95% CIs (Clopper–
   Pearson) are `[0.361, 0.809]` for low severity and `[0.299, 0.802]` for high — almost entirely
   overlapping. A **single** image flipping in the high-severity stratum moves its gap by 6.25pp
   (`9/16 = 0.4375 → 10/16 = 0.3750`), which is larger than the 3.75pp difference between the two
   tiers. The data is consistent with invariance *and* with a substantial severity effect; it
   does not adjudicate between them. The earlier reading of this result as evidence of a
   "severity-invariant" shortcut is **withdrawn** — that claim requires a sample size this
   investigation does not have.

3. **What can still be concluded.** A pure disease-burden proxy — a model that is merely tracking
   how visible/extensive the pneumothorax is — is not sufficient to explain this result, because
   even the *low-severity* stratum (small lesions, where a burden-tracking model should be
   weakest) shows a large gap with perfect with-tube accuracy. The tube carries signal beyond
   lesion extent. This is directionally consistent with device-keyed reliance, but the strength
   of the effect across severity is unresolved at this n.

4. **What would resolve it.** Severity labels exist only for the 94 BBox-annotated images, which
   is what caps the per-group n here (and is why Investigation 5 is blocked). Adjudicating
   invariance needs a severity-annotated set large enough for non-overlapping CIs, or a
   continuous-severity regression over the pooled 400/group population rather than a median split.
