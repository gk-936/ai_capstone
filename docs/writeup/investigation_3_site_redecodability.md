# Investigation 3: Site-Projection Re-Decodability

Tests whether a single projection along the logistic-regression site axis actually removes the site
signal from the feature space, or whether site information is encoded redundantly across many
directions in the 1024-d representation.

---

## Setup

* **Model:** V1 baseline (`experiments/v1_baseline/20260707_232704_9fbb04/artifacts/model_best.pth`)
* **NIH test features:** 4,000 images
* **PadChest V6 features:** 7,572 images

---

## Results

| Step | Metric | Value |
|---|---|---|
| A1 | Hospital-detector probe AUC (raw features) | **0.9993** |
| A2 | PadChest disease AUC before projection | 0.6436 |
| A2 | PadChest disease AUC after projection | 0.6436 |
| A2 | Projection change (ΔDisease AUC) | **0.0000** |
| A3 | Residual site-probe AUC on projected features | **0.9990** |
| A3 | Site decodability drop | **0.0003** |

---

## Findings

1. **A1 confirms strong site decodability.** The hospital-detector probe reaches AUC 0.9993 on raw DenseNet-121 features — site identity (NIH vs PadChest) is almost perfectly encoded in the feature space.

2. **A2: projection has zero effect on disease AUC.** The disease head's PadChest AUC is identical before and after projecting out the probed site direction (0.6436 → 0.6436, Δ = 0.000). This means the disease head is not primarily riding the A1 site axis.

3. **A3: site information survives projection almost completely.** A fresh probe fitted on the projected features achieves AUC 0.9990 — only 0.0003 below the original 0.9993. A single rank-1 projection removes essentially none of the site signal.

**Interpretation:** Site identity is redundantly encoded across many directions in the 1024-d feature space. Rank-1 mean-direction projection is an insufficient intervention — the site signal instantly re-emerges from the remaining 1023 dimensions. To seriously ablate site information would require full subspace projection (PCA of top-k site components) or adversarial training, not a single vector removal.
