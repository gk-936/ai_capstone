# Phase 3 (partial) — Baselines + Naming Ablation + Variance (design)

**Goal:** Neutralize the "VLM-wrapper" critique with (1) a mitigation-comparison table
(ERM/JTT/DFR), (2) a naming-on/off ablation showing the loop works without the VLM, and
(3) multi-seed variance on headline results. This is rigor, not a pass/fail gate — but with
stated expectations.

## 1. Baselines (`src/auditor/mitigation/baselines.py`)
Three mitigation families on the FROZEN trained backbones (no train-time group labels needed):
- **ERM** — no mitigation = the model's own head = the "before" numbers. The floor.
- **JTT (Just Train Twice)** — upweight examples the ERM model misclassifies, retrain the last
  layer on the reweighted frozen features. No group labels required. `fit_jtt_head()`.
- **DFR** — group-balanced last-layer retrain (already built, V5).

Table: worst-group acc / ID acc / reliance gap per method, per confound (tag + tube).

## 2. Naming-on/off ablation (`run_ablation.py`)
Claim: the detect->localize->MITIGATE loop reaches the same worst-group recovery WITHOUT the VLM
naming step — naming is an interpretability layer, not the engine.
- Naming-ON: attribution crops -> VLM names confound -> mitigate (V4a/V4b path).
- Naming-OFF: attribution localizes / known group labels -> mitigate directly -> same DFR result.
- The real tube already demonstrates this (V5-tube mitigated a confound whose naming FAILED in
  V4b). The ablation formalizes it: worst-group gain is ~identical with naming removed.

## 3. Multi-seed variance
Re-run the DFR retrain + eval across 3 seeds on the fixed backbones (last-layer only, cheap).
Report mean +/- std on headline worst-group-after (tag + tube). Backbones stay single-seed
(documented limitation).

## Components & reuse
- `baselines.py`: `fit_jtt_head(features, labels, erm_head, upweight)` (weighted logreg,
  misclassified examples upweighted) + `erm_metrics` (evaluate the original head). Reuses
  `extract_features` and the eval paths from V5.
- `run_ablation.py`: drives ERM/JTT/DFR x {tag, tube} x 3 seeds -> one metrics table (JSON +
  printed markdown).
- `configs/phase3_baselines.yaml`: tag+tube checkpoints, seed list, per_group.

## Tests
- `fit_jtt_head`: on synthetic data with a minority group ERM misclassifies, JTT upweighting
  improves that group's separation vs plain ERM.
- Seed determinism: same seed -> same DFR head.
- Table assembler: one row per (method, confound, seed) with expected keys.

## Expectations (not a hard gate)
- DFR (and ideally JTT) beat ERM worst-group acc; ERM is the low floor. If they don't, that's an
  honest finding.
- Ablation: naming-off worst-group gain ~= naming-on gain (within noise) -> loop value != VLM.
- Variance: report +/- std; headline gains should exceed their variance to be credible.

## Runtime
All last-layer on frozen backbones. ERM/JTT/DFR x 2 confounds x 3 seeds ~= 18 DFR-scale runs x
~2 min ~= 30-40 min GPU. No backbone retraining.
