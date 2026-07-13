# Phase 3 — Decomposing the NIH→PadChest OOD gap (design)

**Date:** 2026-07-13
**Status:** approved design, pre-implementation
**Depends on:** V1 clean checkpoint, V6 PadChest pipeline (both complete)

## 1. Problem

V6 established a **0.192 cross-site AUC drop** (NIH 0.842 → PadChest 0.650) that is **not
shown to be shortcut-driven**: zero-shot source-DFR (remove source-identified spurious
directions, no target labels) is null (+0.005), while supervised target adaptation (uses
PadChest labels) recovers +0.082. So the mechanism of the gap is left open. Phase 3 answers
**"why the gap, if not the shortcut?"** by decomposing 0.192 into candidate causes, each with
its own falsifying control.

This is characterization, not a pass/fail gate (like the correlation sweep). The deliverable is
one honest sentence + a small decomposition figure, reported **whichever way each control falls**.

## 2. Non-negotiables (from CLAUDE.md + `claims-need-controls`)

- Every candidate cause carries a control that could falsify it; the control is run before the
  claim is written.
- No test-distribution labels in any "mitigation/recovery" claim (source-DFR and all TTA are
  label-free; supervised adaptation stays labelled and is never called mitigation).
- **The standing trap:** "taking it further" must not become "finding the framing where it
  worked." All three experiments and the TTA result are reported even when null. Stratification
  variables are fixed in advance (pre-registered), never chosen post-hoc.

## 3. Shared setup

- **Frozen model:** the V1 clean DenseNet121 checkpoint used as the V6 base
  (`experiments/v1_baseline/20260707_232704_9fbb04/artifacts/model_best.pth`). DenseNet121 has
  BatchNorm → BN-adapt/TENT are applicable.
- **Features:** penultimate 1024-d features, extracted once for a **single frozen,
  patient-disjoint PadChest test split** (fixed seed), reused by all three experiments so every
  number is comparable. NIH reference features extracted from the NIH pneumothorax test split.
- **Reference numbers (already measured):** NIH ID 0.842, PadChest OOD 0.650, gap 0.192,
  source-DFR zero-shot +0.005 (the pre-existing shortcut control).
- **Code layout:** new package `src/auditor/analysis/` — one file per experiment (pure functions),
  a shared `features.py` (extract/cache frozen features), and CLI subcommands under the existing
  discovery/mitigation CLI style, each writing artifacts via `RunLogger`. Dataset paths stay
  config values; nothing hardcoded.

## 4. Experiment A — is the gap a *removable shortcut*?

- **A1 Hospital-detector probe (Zech-style):** logistic regression on frozen features, label =
  site (NIH vs PadChest). Report probe AUC/accuracy. A high value is *expected* and by itself is
  NOT evidence the disease head uses site — it only shows site is decodable.
- **A2 Falsifying control (the real test):** estimate the site direction(s) (probe weight vector /
  class-mean difference / top LDA component), **project them out** of the features, and
  re-evaluate the frozen disease head's OOD AUC on the projected features.
  - *If OOD AUC does not recover* → the disease head is not riding site information → the gap is
    NOT a removable shortcut. Combined with the null source-DFR, this **bounds A's contribution
    near zero**.
  - *If it does recover* → site is a genuine exploited shortcut; report the recovered amount.
- **Optional figure:** Grad-CAM overlap (site-detector attention vs disease-model attention),
  qualitative only, no numeric claim.

## 5. Experiment B — label-definition / prevalence mismatch (B-lite)

The working CSV (`data/padchest_v6_subset.csv`, 7572 rows) carries only
`ImageID/PatientID/Labels/ptx` — no MethodLabel/Projection/age. So Phase 3 ships **B-lite**:
- Analyze the raw `Labels` strings for the pneumothorax cases: how PadChest defines/qualifies
  ptx, co-occurring findings, and localization phrases, versus the NIH NLP-mined definition.
- Report label-composition differences that plausibly inflate the gap (e.g., PadChest ptx cases
  dominated by a co-finding NIH labels separately), as a **bounded, qualified** observation.
- **Upgrade path (B-full), deferred:** if the full PadChest metadata CSV is located, join by
  ImageID → recompute OOD AUC on the physician-labeled subset and stratify by projection/age.
  The loader change to carry extra columns is isolated to `data/padchest.py`; spec supports it,
  implementation does not block on it.

## 6. Experiment C — recoverable covariate shift (label-free TTA)

- **C1 BN-adapt (AdaBN):** forward pass over PadChest test images, replace BN running mean/var
  with target-batch statistics, **no labels, no backprop**; re-evaluate OOD AUC.
- **C2 TENT:** minimize prediction entropy on PadChest test images, updating **only BN affine
  params (γ/β)**, no labels; re-evaluate OOD AUC. Fixed steps/LR, logged.
- **Comparison:** gain vs the null source-DFR (+0.005). A chunk closed by BN-adapt/TENT that
  source-DFR could not close = **recoverable covariate shift, not a source-identifiable
  shortcut.** Reported whether positive or null.

## 7. Combining → the decomposition

Present total gap 0.192 as independently-estimated slices:
- **Covariate-recoverable (C):** BN-adapt/TENT AUC gain.
- **Label-mismatch (B):** qualitative bound from B-lite (quantitative if B-full later).
- **Shortcut-residual (A):** bounded ≈0 by source-DFR + site-projection control.

A small bar figure + one sentence. **Explicitly stated:** the slices are estimated independently
and may overlap — this is NOT a clean additive partition, and the honest framing says so.

## 8. Testing (synthetic-spine style, seeded, no network)

- **site-projection:** projecting out a known direction yields features orthogonal to it (dot ≈ 0)
  and correct shape.
- **BN-stat swap:** after AdaBN on a toy tensor, BN running stats change and forward output
  changes; TENT step updates only γ/β (other params unchanged).
- **label-subset filter (B-lite):** parsing/filtering the `Labels` column yields expected row
  counts on a toy CSV.
- **AUC recompute:** matches sklearn `roc_auc_score` on a toy score/label pair.

## 9. Out of scope

- CANDID-PTX third site (blocked on ethics/DUA — separate track).
- B-full physician-label control (upgrade path only; needs full PadChest metadata).
- Any retraining of the backbone (all experiments operate on the frozen model).

## 10. Machine note

All experiments are frozen-feature probes, BN-stat re-estimation, and CPU label analysis; only
TENT needs light backprop over BN affine params. Feasible on a degraded machine, but the live
runs still wait for machine recovery. The design/spec/plan are complete now.
