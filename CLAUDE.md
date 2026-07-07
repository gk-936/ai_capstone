# CLAUDE.md — Spurious-Correlation Auditor for Chest X-ray Classifiers

## What this project is
An "auditor" system that (1) detects when a chest X-ray classifier relies on a spurious
shortcut, (2) uses a general-purpose VLM to NAME that shortcut in clinical terms without
being told what to look for, (3) mitigates reliance on it, and (4) proves recovery on
out-of-distribution / cross-site data.

## NON-NEGOTIABLE OPERATING PRINCIPLE: VALIDATION-FIRST, GATED
- The project runs as spikes V1–V6, each answering ONE make-or-break question with an
  explicit pass/fail threshold. Build in strict dependency order.
- At the end of EACH spike: run it, report the measured metric vs its threshold, state
  PASS or FAIL, then STOP and wait for my go/no-go. Do not continue autonomously.
- Keep spikes MINIMAL. No premature architecture. Phases 2–4 are LOCKED until green-light.
- V4 (VLM naming) is the critical gate. On FAIL, surface the documented pivot (below) and
  wait — do not push through.
- The synthetic spine (V2, V3, V5) must stay runnable at all times; it is the safety floor.
- Never fabricate or assume a metric. If it wasn't measured, say so. If a result looks too
  clean, suspect leakage/bugs before declaring PASS.

## Gates and thresholds (defaults — confirm with me before treating as final)
- **V1 pipeline sanity:** sane in-distribution AUC on one pathology (sanity, not SOTA).
- **V2 shortcut induction:** reliance gap >20 percentage points (accuracy with the confound
  present vs removed/anti-correlated). If weak, increase correlation strength / salience
  until unambiguous.
- **V3 attribution localizes:** >80% of positive saliency crops contain the confound. If
  diffuse, try alternate attribution methods before failing.
- **V4 VLM naming (CRITICAL GATE) — split into V4a and V4b:**
  - **V4a — synthetic confound (proves the METHOD works).** Ground truth is perfect (the
    injected tag), the confound is large and unambiguous, so expect near-ceiling results.
    PASS: precision >=0.9 AND recall >=0.9, hallucination on confound-free CONTROL crops
    <10%. Failing V4a means the pipeline is broken, not borderline — fix it, do not pivot.
  - **V4b — real device confound e.g. chest tube (proves it TRANSFERS).** Crops are
    genuinely harder. PASS: precision >=0.8 AND recall >=0.7, control-hallucination <10%.
    Failing ONLY V4b (after V4a passes) is the pivot conversation.
  - **Rationale:** precision > recall on purpose — for an auditor, a false alarm (a
    hallucinated shortcut) costs more than a miss, because a tool that fabricates confounds
    is worse than no tool. The control-hallucination bar (<10% fabrication on clean crops)
    is the hardest to relax and is measured separately from precision.

## V4 FAILURE PIVOT
Trigger only if V4a passes but V4b fails (or both fail after pipeline is verified sound).
Drop auto-naming. Reframe as a controlled synthetic-confound mitigation study
(V2/V3/V5/V6 without the naming step) — defensible as a capstone and a smaller paper.
Present the pivot and wait; do not improvise a replacement.

- **V5 mitigation:** worst-group / confound-conflicting accuracy rises with minimal
  in-distribution drop. Fallback: swap mitigation family (masking <-> DFR <-> unlearning).
- **V6 real OOD:** a measurable train -> second-site accuracy drop that mitigation narrows.

**Green-light:** V1–V3 PASS, V4a PASS, V4b PASS, V5–V6 PASS -> proceed to Phase 2.
V4b FAIL -> execute the pivot. Summarize the project's proven ceiling before moving on.

## Environment
- Windows + Python. Quote paths with spaces. pytest runs locally (propose the command;
  I run it if you cannot execute it).
- Config-driven (YAML/argparse), fixed seeds, every run logs config + metrics + artifact
  paths for reproducibility.
- Dataset path is a CONFIG value, never hardcoded. NIH ChestX-ray14 lives in a gitignored
  data/ folder OUTSIDE the repo — never commit images or checkpoints.
- Design so a second site (PadChest / CheXpert / CANDID-PTX, arriving later) plugs in
  without rework.

## How to work
- Plan before coding; propose and wait for confirmation on anything non-trivial.
- Maintain a TODO list; keep PROGRESS.md current; commit at each passing gate with a clear
  message.
- Write tests for the synthetic spine so it cannot silently break.
- Be explicit about assumptions and uncertainty; flag anything unverified.

## Course-outcome mapping (capstone)
- **CO1 Problem + literature:** related-work map (RoentMod, CoBalT, MIMM-X, Zech, DeGrave;
  baselines GroupDRO / JTT / DFR) + novelty framing.
- **CO2 Design + methodology:** the auditor architecture and module design.
- **CO3 Experimentation + GIT:** the spike-driven experiments, baselines, ablations, repo.
- **CO4 Evaluation + discussion + social implications:** metrics, limitations, cross-hospital
  risk, fairness, patient safety.
