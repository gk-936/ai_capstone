# V4b — VLM Names the Real Device Confound (design)

**Spike question:** Does prompt-free naming TRANSFER from the synthetic tag to a REAL chest-tube
confound in real chest X-rays?

**Gate (CRITICAL — pivot trigger):** precision >= 0.80 AND recall >= 0.70, control-hallucination
< 10%. Side-consistency (localization) is reported as supporting evidence, not a hard gate.
Failing V4b after V4a passed triggers the DOCUMENTED PIVOT (reframe as synthetic-only study) —
present it and stop, do not improvise.

## Data
NEATX NIH tube annotations (`data/neatx/NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv`):
3,543 images, 1,637 tube-positive, side label (L/R/BOTH). Images already in the NIH folder.
Non-expert labels (2 data-science students) — recorded as a limitation.

**Data reality (checked before building):** in NEATX, tube and pneumothorax are ~100% correlated
(only 2 pneumothorax-negative images carry a tube annotation), because tubes were annotated on
the pneumothorax subset. Among the full NIH pneumothorax-positives (5,298), **31% (1,636) carry
an annotated chest tube**; pneumothorax-negatives essentially never do. So a *controllable*
tube<->label correlation cannot be built. The measurable, scientifically stronger framing is the
NATURAL shortcut below: the chest tube (a treatment device) is a spurious tell for pneumothorax.

## Pipeline (reuses V2->V3->V4a machinery)
1. **Induce tube reliance (natural shortcut)** — train a Pneumothorax classifier on a split where
   TRAIN pneumothorax-positives are ENRICHED for tubes (e.g. 90% of train PTX-positives carry a
   tube), so the model can shortcut "tube -> pneumothorax". Reliance gap = accuracy on
   pneumothorax-positives WITH a tube vs WITHOUT a tube (held-out). A large gap means the model
   leans on the device rather than the collapse. This is a real, documented clinical shortcut, not
   an injected one. If reliance is weak, increase train enrichment before auditing.
2. **Attribute** — Grad-CAM on tube-positive images (confound-region-agnostic code from V3).
   Localization check: does the saliency peak fall on the ANNOTATED side (L/R)? Report
   side-consistency. Export salient crops.
3. **Name** — reuse Groq blind prompt over crops from tube-POSITIVE images (recall) vs
   tube-NEGATIVE control crops (hallucination).
4. **Score** — curated device-term matcher (chest tube/drain, intercostal drain, catheter,
   cannula, line, tube, wire, device), excludes anatomy; manual-audit ~20. Precision / recall /
   control-hallucination vs the gate.

## Data flow
```
NEATX+NIH labels -> tube-enriched PTX train split -> train PTX model -> reliance gap
   (acc on PTX+tube vs PTX+no-tube)
  -> Grad-CAM on tube-positive PTX imgs -> side-consistency + salient crops
  -> Groq blind naming: tube-positive crops vs tube-negative control crops
  -> tube matcher -> P / R / hallucination   (GATE)
```

## Config `configs/v4b_tube.yaml`
NEATX csv path, biased-split correlation strength, crop params; reuse discovery/attribution
settings. Separate DataConfig field for the tube annotations csv.

## Tests (synthetic spine stays green; add V4b units)
- NEATX loader: parses labels + side; correct pos/neg counts; image ids resolve to NIH files.
- Biased-split builder: tube<->PTX correlation matches configured strength; patient-disjoint
  preserved (no patient leak across train/val/test).
- Tube matcher: "chest tube in left hemithorax" -> hit; "clear lung fields" -> miss;
  "rib"/"vessel"/"clavicle" -> miss (hallucination guards).
- Side-consistency: peak on annotated side -> hit; opposite side -> miss.

## PASS criteria & skeptic checks
- PASS: precision >= 0.80, recall >= 0.70, control-hallucination < 0.10.
- PIVOT: if V4b FAILS (V4a having passed), surface the documented pivot and STOP.
- Manual audit validates the matcher before trusting numbers (as in V4a).
- Weak induced reliance -> fix the bias first (pipeline issue, not a V4b fail).
- Expect genuinely harder than V4a: real tubes are thin, varied, overlap anatomy; the crops are
  real. Recall pressure is the likely failure mode (foreshadowed by V4a recall at the bar).

## Runtime
Tube-reliant training on the ~3.5k NEATX-labelled subset is faster than V1. Attribution + Groq
naming: minutes. Fits the RTX 3050 (batch 16, AMP).
