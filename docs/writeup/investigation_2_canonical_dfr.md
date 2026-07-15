# Investigation 2: Canonical DFR Refit

Tests whether (a) tuning the regularization strength C via cross-validation and (b) averaging weights across 10 random balanced draws produces meaningfully different results from the single-draw C=1.0 DFR already present in the pipeline.

---

## Tube Confound (v4b_tube checkpoint)

* **Best C (5-fold CV on first balanced draw):** `0.001`
* **Val full set:** 4,371 samples, 371 positives

| Metric | Before DFR | Standard DFR (C=1.0, 1 draw) | Canonical DFR (C=0.001, 10-draw avg) |
|---|---|---|---|
| ID Accuracy | 0.9575 | 0.9650 | **0.9975** |
| Worst-group Accuracy | 0.6700 | 0.8425 | **0.9275** |
| Reliance Gap | 0.2875 | 0.1225 | **0.0700** |

---

## Tag Confound (v2_shortcut checkpoint)

* **Best C (5-fold CV on first balanced draw):** `0.001`
* **Val full set:** 26,074 samples (13,037 untagged + 13,037 tagged), 742 positives

| Metric | Before DFR | Standard DFR (C=1.0, 1 draw) | Canonical DFR (C=0.001, 10-draw avg) |
|---|---|---|---|
| ID Accuracy | 0.9397 | 0.5937 | **0.6870** |
| Worst-group Accuracy | 0.5000 | 0.5927 | **0.6813** |
| Reliance Gap | 0.4397 | 0.0010 | **0.0057** |

---

## Findings

**Tube confound:**
* Both DFR variants reduce the gap substantially. Standard DFR at C=1.0 yields gap 0.1225; canonical yields 0.070 — a further ~5.3pp reduction.
* ID accuracy actually *rises* to near-ceiling (0.9975) under canonical DFR, likely because extreme regularization (C=0.001) keeps the head nearly constant and defaults to the dominant class pattern.
* The canonical variant strictly dominates the standard one on all three metrics for this confound.

**Tag confound:**
* The raw C=1.0 standard DFR causes a catastrophic ID accuracy collapse (0.94 → 0.59). The gap nearly vanishes (0.001) but only because the model is now near-random overall — not because confound reliance was untangled.
* Canonical DFR (C=0.001, 10-draw) preserves the near-zero gap (0.0057) while recovering some ID accuracy (0.687), but still well below the pre-DFR baseline (0.94).
* **This is a genuine anomaly:** the tag DFR validation set contains both tagged and untagged versions of every image concatenated together. The logistic regression is near 50/50 — it cannot distinguish the confound axis from the label axis in this symmetric setup, causing the head to collapse to a coin-flip regardless of C or draw count.
