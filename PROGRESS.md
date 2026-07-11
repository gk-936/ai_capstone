# PROGRESS.md

Status legend: **TODO / IN-PROGRESS / PASS / FAIL / BLOCKED**

## Phase 0 — Foundations
- [x] git repo + .gitignore + first commit .......... PASS
- [x] folder structure ............................... PASS
- [x] CLAUDE.md + PROGRESS.md committed ............... PASS
- [x] reproducibility skeleton (config, seeds, logger) PASS (11 tests pass)
- [x] NIH ChestX-ray14 obtained (path in config) ..... PASS (112,120 imgs @224px; paths verified)
- [x] literature map drafted (CO1) ................... DONE (docs/writeup/CO1_related_work.md)

## Phase 1 — Validation gauntlet (gated; STOP after each spike)

| Spike | Question | Threshold | Status | Measured | Notes |
|-------|----------|-----------|--------|----------|-------|
| V1  | Trains end-to-end, sane AUC? | sane ID AUC (>=0.75) | **PASS** | test AUC **0.8448** (val 0.8269) | full data (73k/13k/25.6k), DenseNet-121, 5 ep, batch 16 @224px; in NIH Pneumothorax lit range 0.80–0.85; patient-disjoint verified; monotone learning curve |
| V2  | Model measurably cheats? | reliance gap >20pp | **PASS** | **gap 44pp** (aligned .94 / removed .50 / flipped .05) | corner tag p=.95, 3 ep; acc_removed=0.500 = pure-shortcut fingerprint (pre-validated); monotone collapse; tag-free AUC dropped 0.84->0.74 as expected |
| V3  | Saliency localizes confound? | >80% of positive crops | **PASS** | **tagged peak-in-box 100%** (control untagged 0.5%) | Grad-CAM on V2 ckpt, 200 imgs, layer features.norm5; control near-zero proves saliency follows the TAG not the corner; mass-frac 0.14 in a 0.8%-area box |
| V4a | VLM names SYNTHETIC confound? | P>=.90 & R>=.90, ctrl-halluc <10% | **PASS** | **P 0.978 / R 0.90 / halluc 0.02** | Llama-4-Scout (Groq), blind prompt, 100 pos + 100 ctrl crops @32px. Recall exactly at bar: 10 genuine VLM misses. Matcher corrected mid-scoring (rectangular/opaque forms) — verified by full audit of all 23 orig misses; 2 ctrl FPs are real VLM hallucinations, correctly counted |
| V4b | VLM names REAL device confound? | P>=.80 & R>=.70, ctrl-halluc <10% | **FAIL** | **P 0.57 / R 0.66 / halluc 0.50** | Llama-4-Scout on real tube crops. Halluc 50% is the killer: VLM fabricates tubes/central-lines/ETTs on half the tube-FREE controls (47/50 explicit device terms — genuine hallucination, audited, not matcher error). Also misses subtle real tubes (R .66). PIVOT TRIGGERED. |
| V5  | Mitigation recovers accuracy? | worst-group up, ID stable | **PASS** | **tube: 0.67->0.83, ID 0.96->0.97, gap 0.29->0.14** (clean). tag: gap 0.44->0.02 but ID 0.94->0.62 (partial) | DFR last-layer retrain on group-balanced data. The tag-model ID drop reflects DEGRADED (not absent) core features: a linear probe on its frozen backbone recovers pneumothorax AUC 0.690 (chance 0.50, clean V1 0.845) — shortcut reliance partially crowded out core learning, did not hollow it. REAL-data tube mitigation is the anti-wrapper headline. |
| V6  | Real OOD gap exists + narrows? | measurable drop, narrowed | **PASS (drop) / reframed (narrowing)** | **NIH 0.842 -> PadChest 0.650 = OOD drop 0.192. Supervised target adaptation (last-layer fit on LABELED PadChest) -> 0.732 (+0.082). Source-DFR zero-shot (no target labels) -> +0.005 (NULL).** | Real NIH(US)->PadChest(Spain), V1 model, 7,572-img subset (573 PTX-pos). Normalization control: 16-bit->8-bit at load, NIH ref AUC preserved 0.842 (earlier per-image-std broke it to 0.59). HONEST FRAMING: (1) the +0.082 is SUPERVISED TARGET ADAPTATION, not shortcut mitigation — it uses PadChest labels and would help with no shortcut present. (2) The zero-shot source-DFR (reduce a real source shortcut, no target labels) gives +0.005 = null -> the drop is NOT primarily shortcut-driven. (3) The 0.192 drop is DISTRIBUTION SHIFT; candidate causes include scanner/population/label-pipeline differences (NIH-NLP vs PadChest-NLP+manual)/residual intensity/subset selection — shortcuts are ONE unproven candidate. Do NOT claim 'mitigation validated cross-site'. |

**Green-light decision:** V4b FAILED (P.57/R.66/halluc.50) -> PIVOT EXECUTED. V1-V3 + V4a PASS;
V4b (real device naming) fails on hallucination. Pivot done: auto-naming dropped as a claim for
real confounds; project reframed as a controlled detect->localize->mitigate study on synthetic
AND real confounds, with V4b as an honest negative result bounding where VLM naming transfers.
Since the pivot: V5 PASS (real-tube mitigation, the ONE clean cross-setting mitigation result —
no test-distribution labels), Phase-3 baselines+ablation+variance DONE, correlation sweep DONE
(+ linear probe: strong-shortcut backbone is DEGRADED not hollow, AUC 0.690), Phase-2 auditor
loop built, V6 DONE (real cross-site OOD drop 0.192; supervised adaptation closes some of it but
source-DFR zero-shot is null -> drop is distribution shift, not shown to be shortcut-driven). ALL
experiments complete. Remaining: writeup only.

## Phase 2 — System build (unlocked post-pivot)
- [x] promote spikes into modules ................. DONE (src/auditor/{data,train,shortcut,
      attribution,discovery,mitigation,evaluation,report}; 64 tests)
- [x] wire end-to-end auditor loop ............... DONE (report/run_auditor.py, CLI
      `python -m auditor.report`: detect->localize->[name]->mitigate->audit_report.{json,md})
- [x] before/after Grad-CAM demo artifact ........ DONE. Auditor run on the REAL tube model:
      one command produced audit_report (detect gap 0.287 -> mitigate worst-group 0.67->0.83,
      ID 0.96->0.97) + before/after Grad-CAM figure. Post-DFR attention broadens away from the
      confound-associated region. Artifacts: docs/writeup/auditor_before_after_tube.png +
      auditor_report_tube.md.

## Phase 3 — Experiments + baselines + ablations (STARTED — reordered ahead of V6 for paper rigor)
- [x] baselines ERM / JTT / DFR (3 seeds, mean±std) ... DONE
      TUBE (real): ERM 0.672 -> JTT 0.791 / DFR 0.790 worst-group (+0.12, ID held ~0.96); gain >> variance.
      TAG (pure shortcut): ERM 0.500 -> JTT/DFR ~0.61 worst-group but ID collapses 0.94->~0.60
      (degraded-core finding: linear probe recovers AUC 0.690, above chance 0.50 but below clean
      0.845 — shortcut crowds out, not hollows, core features). Two baselines agree -> not a DFR quirk.
- [x] naming-off ablation ................. DONE (by construction: no method calls the VLM, yet
      worst-group recovers on the REAL tube -> the loop's value is NOT the VLM naming).
- [x] multi-seed variance ................. DONE (3 seeds; ± std reported in the table).
- [x] synthetic correlation-strength sweep ... DONE (all 5 points; resumed cleanly for p=0.9/0.95).
      Trained reliance gap rises MONOTONICALLY with p: 0.000 (p=.6) -> 0.010 (.7) -> 0.075 (.8)
      -> 0.294 (.9) -> 0.440 (.95). FINDINGS: (1) ONSET THRESHOLD — below ~p=0.7 the model ignores
      the tag and learns real signal (test AUC ~0.83); reliance then climbs steeply. (2) The
      DFR ID-drop is STRENGTH-DEPENDENT: negligible at low p (no shortcut), large at high p
      (strong-shortcut model has DEGRADED core features, ID 0.94->0.62 at p=.95; linear probe on
      the frozen p=.95 backbone recovers pneumothorax AUC 0.690 vs clean 0.845 — real but reduced,
      NOT hollow). DFR raises worst-group across all p.
      Figure: docs/writeup/sweep_figure.png; table: docs/writeup/sweep_final_table.md. Single seed.
- [x] contrast vs RoentMod (prompt-free, no gen fine-tuning, lower compute) ... DONE (in CO1)
- [x] cross-site experiments ............... DONE (V6: NIH->PadChest drop 0.192; supervised adapt
      +0.082; source-DFR zero-shot null +0.005 -> distribution shift, not shown shortcut-driven)
- [ ] GroupDRO baseline .................... SKIPPED (redundant with ERM/JTT/DFR table; decided)

## Phase 4 — Evaluation + writeup
- [x] consolidate metrics (worst-group acc, ID<->OOD gap, naming P/R, cost vs RoentMod) ... DONE
      (CO3_experiments.md — full spike-by-spike table with final, corrected numbers)
- [x] limitations (naming reliability, incremental framing) ... DONE (CO4_evaluation.md §4)
- [x] prior-art comparison table ... DONE (CO1_related_work.md §4, re-searched through July 2026:
      SpurLens, RaVL, MIMM-X, concurrent arXiv:2604.17110, HalluCXR, NEATX credited as data source)
- [x] social-implications section (cross-hospital risk, fairness, patient safety) ... DONE
      (CO4_evaluation.md §5-6)
- [x] CO1-CO4 deliverables ... DONE, all in docs/writeup/. Two claims corrected post-review before
      finalizing (hollow-shortcut; V6 mitigation) — see CO1 §6, CO4 §2, decision log below.
- [ ] paper draft / venue submission (target: SCSL workshop / ML4H / applied short track) — beyond
      the CO1-CO4 deliverables, not started

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
- 2026-07-09 — Phase 3 (reordered ahead of V6 for paper rigor): ERM/JTT/DFR baseline table with
  3-seed variance; naming-off ablation (no method calls the VLM yet worst-group recovers -> the
  loop's value is not the prompt); correlation-strength sweep (monotone gap-vs-p, onset threshold
  ~p=0.7). GroupDRO baseline SKIPPED as redundant (ERM/JTT/DFR already covers baselines) and heavy.
- 2026-07-09 — Phase 2 built post-pivot: end-to-end auditor loop (report/run_auditor.py) unifying
  detect->localize->[name,optional]->mitigate into one command emitting audit_report + before/after
  Grad-CAM demo figure. Pure orchestration over tested modules; 64 tests pass.
- 2026-07-09 — Phase 2 demo DONE on the REAL tube model: one auditor command -> report (gap 0.287,
  worst-group 0.67->0.83, ID 0.96->0.97) + before/after Grad-CAM (attention broadens off the
  confound post-DFR). Artifacts in docs/writeup/.
- 2026-07-09 — Second site chosen: PadChest (Spain), 224px 12.89GB (Academic Torrents) — NOT the
  1.07TB full set. Cross-site OOD only (NEATX PadChest chest-drains n=9, too few for tube-transfer).
  V6 code built ahead of the download: data/padchest.py (schema-tolerant loader), report/run_v6.py
  (NIH->PadChest AUC drop + DFR target-site adaptation), config/CLI, 5 loader tests.
- 2026-07-09 — REPO-INTEGRITY FIX: the '.gitignore' rule 'data/' (for the dataset folder) also
  matched src/auditor/data/, so the entire data-layer source (nih.py/neatx.py/torch_dataset.py/
  padchest.py + __init__) had been UNTRACKED since Phase 0 — a fresh clone would not run. Anchored
  the rule to /data/ and /experiments/ (repo-root only) and force-added the 5 recovered files;
  verified dataset images/CSVs stay ignored. Repo is now actually reproducible.
- 2026-07-10/11 — V6 cross-site OOD (NIH US -> PadChest Spain). NIH ref AUC 0.842 -> PadChest OOD
  AUC 0.650 (drop 0.192, matches Zech et al. cross-site magnitude). Disk was full (3.3GB) so
  extracted only the 7,572 V6 images (573 PTX-pos + 7k neg) from the 12.79GB tar. NORMALIZATION
  DEBUG: PadChest 16-bit (0-58927) vs NIH 8-bit; first attempt (per-image-std at eval) broke the
  model (NIH 0.83->0.59, train-eval mismatch) — reverted. Correct fix: 16->8-bit at load
  (transform untouched); NIH ref AUC preserved 0.842 (the control that validates the drop).
  MITIGATION CLAIM CORRECTED (post-review): the +0.082 'recovery' was last-layer fit on LABELED
  PadChest = SUPERVISED TARGET ADAPTATION, not shortcut mitigation (would help with no shortcut).
  Ran the honest test — SOURCE-DFR ZERO-SHOT (reduce the real tube shortcut on NIH, no target
  labels, eval PadChest): +0.005 = NULL. So reducing a known source shortcut does NOT improve
  transfer -> the cross-site drop is NOT primarily shortcut-driven. V6 is a DISTRIBUTION-SHIFT
  result (candidate causes: scanner/population/label-pipeline/intensity/subset selection; shortcuts
  one unproven candidate). Dropped the earlier 'mitigation validated across THREE settings' claim.
  Scope: cross-site OOD only (PadChest chest-drains n=9, too few for tube-transfer).
- 2026-07-11 — Exhaustive literature re-search through July 2026 (20 queries, 10 primary-source
  fetches) found no medical-imaging paper doing blind general-VLM naming of a CXR shortcut scored
  on P/R/control-hallucination, and no paper running a zero-shot source-side test of whether a
  known shortcut explains a cross-site OOD gap — both flagged as this project's most defensible
  novel contributions. New citations: SpurLens (arXiv:2503.08884), RaVL (arXiv:2411.04097),
  MIMM-X (arXiv:2511.22990), a concurrent chest-drain-mitigation paper (arXiv:2604.17110, same
  shortcut/different method), HalluCXR (arXiv:2605.20469, corroborates V4b hallucination),
  "VLMs do not always need the image" (arXiv:2606.17710, a limitation for our own VLM step), and
  NEATX credited explicitly as the data source (not just prior art).
- 2026-07-11 — CO1-CO4 writeup completed in docs/writeup/. CO1 rewritten with the new literature
  and final (corrected) results; CO2 documents the actual final architecture; CO3 is the full
  spike-by-spike record with every measured number and the two corrections; CO4 states what the
  evidence does and does not support, the two withdrawn claims and why, limitations (naming
  boundary uncharacterized, VLM image-independence untested, non-expert tube labels, V6 causal
  attribution unresolved, single-seed sweep, hardware scale), and patient-safety/fairness/
  cross-hospital-risk discussion grounded in the measured numbers (tube-absent accuracy 0.67,
  cross-site drop 0.192). 69 tests unaffected (writeup only, no code changes).
