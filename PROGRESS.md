# PROGRESS.md

Status legend: **TODO / IN-PROGRESS / PASS / FAIL / BLOCKED**

## Phase 0 — Foundations
- [x] git repo + .gitignore + first commit .......... PASS
- [x] folder structure ............................... PASS
- [x] CLAUDE.md + PROGRESS.md committed ............... PASS
- [x] reproducibility skeleton (config, seeds, logger) PASS (11 tests pass)
- [x] NIH ChestX-ray14 obtained (path in config) ..... PASS (112,120 imgs @224px; paths verified)
- [ ] literature map drafted (CO1) ................... TODO

## Phase 1 — Validation gauntlet (gated; STOP after each spike)

| Spike | Question | Threshold | Status | Measured | Notes |
|-------|----------|-----------|--------|----------|-------|
| V1  | Trains end-to-end, sane AUC? | sane ID AUC (>=0.75) | **PASS** | test AUC **0.8448** (val 0.8269) | full data (73k/13k/25.6k), DenseNet-121, 5 ep, batch 16 @224px; in NIH Pneumothorax lit range 0.80–0.85; patient-disjoint verified; monotone learning curve |
| V2  | Model measurably cheats? | reliance gap >20pp | **PASS** | **gap 44pp** (aligned .94 / removed .50 / flipped .05) | corner tag p=.95, 3 ep; acc_removed=0.500 = pure-shortcut fingerprint (pre-validated); monotone collapse; tag-free AUC dropped 0.84->0.74 as expected |
| V3  | Saliency localizes confound? | >80% of positive crops | TODO | — | |
| V4a | VLM names SYNTHETIC confound? | P>=.90 & R>=.90, ctrl-halluc <10% | TODO | — | proves METHOD; fail = fix pipeline, not pivot |
| V4b | VLM names REAL device confound? | P>=.80 & R>=.70, ctrl-halluc <10% | TODO | — | CRITICAL GATE; fail (after V4a pass) = pivot |
| V5  | Mitigation recovers accuracy? | worst-group up, ID stable | TODO | — | |
| V6  | Real OOD gap exists + narrows? | measurable drop, narrowed | TODO | — | |

**Green-light decision:** ________
(V1–V3 + V4a + V4b + V5–V6 all PASS -> Phase 2 | V4b FAIL -> pivot to synthetic-only study)

## Phase 2 — System build (LOCKED until green-light)
- [ ] promote spikes into modules (trainer, attribution, discovery/VLM-naming, mitigation,
      evaluation/reporting)
- [ ] wire end-to-end auditor loop
- [ ] before/after Grad-CAM demo artifact

## Phase 3 — Experiments + baselines + ablations (LOCKED)
- [ ] synthetic correlation-strength sweep
- [ ] cross-site experiments
- [ ] baselines: ERM / GroupDRO / JTT / DFR
- [ ] contrast vs RoentMod (prompt-free, no generative fine-tuning, lower compute)
- [ ] ablations (naming on/off, mitigation variants); multi-seed variance

## Phase 4 — Evaluation + writeup (LOCKED)
- [ ] consolidate metrics (worst-group acc, ID<->OOD gap, naming P/R, cost vs RoentMod)
- [ ] limitations (naming reliability, incremental framing)
- [ ] prior-art comparison table
- [ ] social-implications section (cross-hospital risk, fairness, patient safety)
- [ ] CO1–CO4 deliverables + paper draft (target: SCSL workshop / ML4H / applied short track)

## Decision log
- 2026-07-07 — framework: PyTorch + timm (DenseNet-121 @ 224px) — CXR standard, best Grad-CAM/DFR ecosystem.
- 2026-07-07 — compute: local RTX 3050 6GB, AMP on, batch-size tuned to fit — for V1/V2/V3/V5.
- 2026-07-07 — V4 VLM: open-weight VLM on Colab/Kaggle (Claude subscription has no API) via a
  file-contract boundary (crops+manifest -> predictions.jsonl -> local scoring), upgradeable to a
  live endpoint later.
- 2026-07-07 — env: Python 3.9.13, pydantic 2 for typed configs; src-layout with tests/conftest.py
  on sys.path (editable install flaky on old pip).
- 2026-07-08 — V1 PASS: full-data DenseNet-121 test AUC 0.8448. Root-caused a mid-epoch CUDA
  "Memory allocation failure" at the AdamW step to VRAM peak on the 6GB laptop GPU (shares memory
  with display); fixed by default batch_size 32->16 (expandable_segments unsupported on Windows,
  no-ops). Bumped torch AMP API to torch.amp.*.
- 2026-07-08 — V2 PASS: reliance gap 44pp (aligned .94 / removed .50 / flipped .05). acc_removed
  landing exactly at 0.500 matches the pure-shortcut fingerprint pre-validated with a synthetic
  tag-detector model before the run — real reliance, not leakage.
