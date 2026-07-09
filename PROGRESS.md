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
| V3  | Saliency localizes confound? | >80% of positive crops | **PASS** | **tagged peak-in-box 100%** (control untagged 0.5%) | Grad-CAM on V2 ckpt, 200 imgs, layer features.norm5; control near-zero proves saliency follows the TAG not the corner; mass-frac 0.14 in a 0.8%-area box |
| V4a | VLM names SYNTHETIC confound? | P>=.90 & R>=.90, ctrl-halluc <10% | **PASS** | **P 0.978 / R 0.90 / halluc 0.02** | Llama-4-Scout (Groq), blind prompt, 100 pos + 100 ctrl crops @32px. Recall exactly at bar: 10 genuine VLM misses. Matcher corrected mid-scoring (rectangular/opaque forms) — verified by full audit of all 23 orig misses; 2 ctrl FPs are real VLM hallucinations, correctly counted |
| V4b | VLM names REAL device confound? | P>=.80 & R>=.70, ctrl-halluc <10% | **FAIL** | **P 0.57 / R 0.66 / halluc 0.50** | Llama-4-Scout on real tube crops. Halluc 50% is the killer: VLM fabricates tubes/central-lines/ETTs on half the tube-FREE controls (47/50 explicit device terms — genuine hallucination, audited, not matcher error). Also misses subtle real tubes (R .66). PIVOT TRIGGERED. |
| V5  | Mitigation recovers accuracy? | worst-group up, ID stable | **PASS** | **tube: 0.67->0.83, ID 0.96->0.97, gap 0.29->0.14** (clean). tag: gap 0.44->0.02 but ID 0.94->0.62 (partial: pure-shortcut model is hollow) | DFR last-layer retrain on group-balanced data. REAL-data mitigation is the anti-wrapper headline. |
| V6  | Real OOD gap exists + narrows? | measurable drop, narrowed | TODO | — | |

**Green-light decision:** V4b FAILED (P.57/R.66/halluc.50) -> PIVOT to synthetic-only study.
V1-V3 + V4a all PASS; V4b (real device naming) fails on hallucination. Per plan: drop
auto-naming as a claim for real confounds; reframe as a controlled synthetic-confound
detect->name->mitigate study (V2/V3/V4a/V5/V6), with V4b reported as an honest negative result
bounding where VLM naming transfers. Awaiting go-ahead to proceed with the pivot (V5 next).

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
- 2026-07-08 — V3 PASS: Grad-CAM peak-in-tag-box 100% on tagged positives vs 0.5% on the same
  images untagged. The near-zero untagged control proves saliency tracks the tag, not the corner.
  Attribution code built confound-region-agnostic (generic box) so it reuses for the real device
  confound later. V3 checkpoint dependency: the V2 model_best.pth.
- 2026-07-08 — V4a PASS (CRITICAL GATE): Llama-4-Scout via Groq free API named the synthetic tag
  from blind prompts. P 0.978 / R 0.90 / ctrl-halluc 0.02. Switched from planned Colab+Qwen2-VL to
  Groq live endpoint (same file contract). Recall exactly at 0.90 (10 real VLM misses on an easy
  confound) — foreshadows V4b will be harder. Keyword matcher was corrected during scoring
  (FAIL 0.77 -> PASS 0.90) after auditing ALL 23 original misses: ~13 were the VLM saying
  'rectangular/opaque white object' that the regex missed (real bug), the rest genuine VLM misses
  kept in the denominator. Fix verified conservative (control-hallucination unchanged at 0.02).
- 2026-07-08 — V4b FAIL -> PIVOT. Real chest-tube confound (NEATX). Detect+localize succeeded
  (tube reliance gap 28.75pp; Grad-CAM side-consistency 96.6%), but NAMING failed the gate:
  P 0.57 / R 0.66 / ctrl-halluc 0.50. Root cause (audited): Llama-4-Scout fabricates devices
  (endotracheal tube / central line / catheter) on 50/100 tube-FREE control crops (47 explicit,
  genuine hallucination not matcher error) and misses subtle real tubes (R .66). The easy
  synthetic box was unfabricatable; real anatomy (vessels/fissures/rib edges) fools a device-
  primed VLM. This is the documented pivot trigger (V4a passed, only V4b failed). Negative result
  is scientifically valuable: it bounds where prompt-free VLM naming transfers. Detect->localize->
  mitigate on synthetic confounds remains fully intact for the pivot study.
  NON-OLLAMA aside: local qwen2.5vl:3b was too unreliable (dropped image ~2/3 of calls) — a
  finding that capable models are required; Ollama path kept in repo for future use.
- 2026-07-09 — V5 PASS (DFR mitigation). REAL tube confound (the paper headline): worst-group
  (PTX-without-tube) 0.67->0.83, ID (PTX-with-tube) 0.96->0.97 (preserved), reliance gap
  0.29->0.14 (halved). Clean DFR win on real data — a VLM wrapper cannot do this. Synthetic tag
  (p=0.95 pure shortcut): gap 0.44->0.02 but ID 0.94->0.62 — a partial/cautionary result showing
  a pure-shortcut model has little real signal to fall back on (informative, kept as the
  companion case; no milder-p retrain needed since tube carries the ID-preservation result).
  Fixed group_balanced_sample to handle the structurally-empty (tube=1,label=0) group.
