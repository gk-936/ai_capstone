# Investigation 4: Label-Free OOD-Gap Attribution (Tag Source-DFR)

Generalises the zero-shot source-DFR test from the tube confound (run_v6_source_dfr_zeroshot)
to the synthetic **tag** confound. The question: does mitigating the tag shortcut on the source
(NIH) domain — using zero PadChest labels — improve cross-site performance on PadChest?

---

## Setup

* **Model:** Tag-trained (`v2_shortcut`, `experiments/v2_shortcut/20260708_060719_0515dc/artifacts/model_best.pth`)
* **DFR balanced set:** 771 samples (groups over tag×label from NIH val)
* **PadChest test n:** 7,572 images

---

## Results

| Metric | Value |
|---|---|
| NIH test AUC (tag model, reference) | 0.7407 |
| PadChest AUC before source-DFR | 0.6345 |
| OOD drop (NIH → PadChest) | **0.1062** |
| PadChest AUC after tag source-DFR | **0.5520** |
| Zero-shot change (after − before) | **−0.0825** |

---

## Findings

Tag source-DFR **does not improve** PadChest AUC. It makes it substantially worse: −8.25pp.

This mirrors (and strengthens) the finding from Investigation 2 (tag confound): DFR on the tag-trained model destabilises the disease head without recovering generalisable signal. The tag shortcut is so dominant in the v2_shortcut model that the feature space has largely collapsed to encoding the tag rather than the underlying pathology. When DFR removes the tag axis from the head, the remaining signal is insufficient to maintain meaningful AUC — either in-distribution (ID drops to ~0.69 from 0.94) or OOD (PadChest drops by further 8.3pp to 0.552).

**Comparison with tube source-DFR (from existing run_v6_source_dfr_zeroshot):** The tube zero-shot change was reported as approximately null (+0.005). The tag zero-shot change is large and **negative** (−0.083). This asymmetry is consistent with the tag shortcut being a far more complete override of the model's representation — the tube model retains some genuine disease signal even after training with the tube enrichment, while the tag model at p=0.95 largely stops learning the disease.
