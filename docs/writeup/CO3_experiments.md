# CO3 — Experiments, Spike-by-Spike Results, and Reproducibility

*Spurious-Correlation Auditor for Chest X-ray Classifiers.* This document is the experimental
record: every spike's question, threshold, measured result, and the reasoning behind each
PASS/FAIL/pivot decision, drawn directly from `PROGRESS.md` and the git history. Every number
below was measured and logged by `RunLogger` to a timestamped `experiments/<spike>/<run>/`
directory containing the resolved config and `metrics.json` — nothing here is asserted without
a corresponding run.

## 0. Method: validation-gated spikes

The project ran as a sequence of minimal spikes (V1–V6), each answering one make-or-break
question against an explicit threshold, executed in strict dependency order with a stop-and-report
after each. This produced two properties worth stating explicitly: (1) a failed gate (V4b) did
not block the rest of the program — it triggered a documented pivot and the synthetic/real-tube
spine continued; (2) the discipline of "measure before claiming" was applied inconsistently at
first (two claims were asserted narratively before the control that tests them existed) and was
corrected once a review caught it (§7). Both are part of the honest record, not edited out.

## 1. Foundations

- **Dataset:** NIH ChestX-ray14, 112,120 images at 224×224, patient-disjoint official splits.
  Target pathology: **Pneumothorax** (~3–5% prevalence depending on split — the official test
  split is more pathology-enriched than train, a known property of the release, not a bug).
- **Model:** DenseNet-121 (timm), ImageNet-pretrained, AMP, batch size 16 (tuned to the 6GB
  laptop GPU after a mid-epoch CUDA OOM at batch 32 was root-caused to VRAM pressure from the
  shared display).
- **Reproducibility substrate:** typed YAML configs with inheritance, fixed seeding, and a run
  logger that snapshots the resolved config + metrics + git commit for every run (CO2 §2.1).

## 2. V1 — Pipeline sanity

**Question:** does the pipeline train end-to-end to a sane in-distribution AUC?
**Threshold:** AUC ≥ 0.75 (sanity, not SOTA).
**Result: PASS.** Test AUC **0.8448** (val 0.8269), full 73k/13k/25.6k patient-disjoint split,
5 epochs. In the published NIH-pneumothorax range (0.80–0.85); the learning curve was monotone
and the split was independently verified patient-disjoint before trusting the number.

## 3. V2 — Shortcut induction

**Question:** does the model measurably cheat when a synthetic confound (a 20×20 corner tag,
correlation strength p=0.95) is injected?
**Threshold:** reliance gap (accuracy tag-aligned minus tag-removed) > 20 percentage points.
**Result: PASS.** Gap **44pp** (aligned 0.94, removed 0.50, flipped 0.05). The removed-accuracy
landing at exactly 0.500 matches the "pure shortcut learner" fingerprint that was independently
predicted and verified with a synthetic tag-detector *before* this run, which is why the result
was trusted rather than treated as suspiciously clean.

## 4. V3 — Attribution localizes the confound

**Question:** does Grad-CAM concentrate on the injected tag?
**Threshold:** > 80% of positive-crop saliency peaks land on the confound.
**Result: PASS.** **100%** peak-in-box on tagged positives, versus **0.5%** on the same images
with the tag removed (the untagged control) — ruling out "the corner is always salient" as an
alternative explanation.

## 5. V4a — VLM names the synthetic confound, blind (critical gate)

**Question:** can a general-purpose VLM, given only a saliency crop and a prompt that never
mentions the confound, name it in clinical/plain language?
**Threshold:** precision ≥ 0.90 AND recall ≥ 0.90 AND control-hallucination < 10%.
**Result: PASS.** Precision **0.978**, recall **0.90** (exactly at the bar — 10 genuine misses
on an easy target, foreshadowing V4b), control-hallucination **0.02**. Model: Llama-4-Scout via
Groq (free tier), 100 tagged-positive + 100 confound-free control crops. The scorer's keyword
matcher was corrected mid-analysis after a manual audit of all 23 original misses showed ~13 were
the VLM correctly describing a "rectangular/opaque object" in phrasing the initial regex missed
— a real scorer bug, fixed and re-verified not to have relaxed the control-hallucination check.

## 6. V4b — VLM names a real device confound, blind (the pivot trigger)

**Question:** does naming transfer to a real confound (a chest tube, annotated via NEATX) rather
than a synthetic one?
**Threshold:** precision ≥ 0.80 AND recall ≥ 0.70 AND control-hallucination < 10%.
**Result: FAIL.** Precision **0.57**, recall **0.66**, control-hallucination **0.50**. Detection
and localization both still worked on the real confound (tube reliance gap 28.75pp; Grad-CAM
side-consistency 96.6%) — only naming failed. Audited: 47 of 50 control-crop false positives
explicitly named a device ("endotracheal tube," "central line," "catheter") on images with no
tube present, confirmed as genuine hallucination rather than a scorer artifact. This is
independently corroborated by HalluCXR (arXiv:2605.20469, CO1 §2e), which finds the *same*
qualitative pattern — highest hallucination severity on normal images — in an unrelated VLM
population and task.

**Pivot executed, per the pre-registered plan:** automated naming is dropped as a validated
capability on real device confounds; the project continues as a detect→localize→mitigate study
on both synthetic and real confounds, with V4b reported as a bounded negative result rather than
suppressed.

### 6b. Multi-VLM replication — is the hallucination Scout-specific or general?

**Question:** V4b showed one VLM (Llama-4-Scout) fabricating devices on clean crops. Is that a
quirk of one model, or a property of general-purpose VLMs used as blind auditors? If the latter,
it is a much stronger reason not to trust auto-naming.

**Method:** the *identical* 200 V4b crops (100 tube-positive + 100 confound-free control), the
*same* blind prompt, and the *same* conservative `names_tube` matcher, run through independent
VLM families behind one provider abstraction (`configs/multi_vlm.yaml`; `python -m
auditor.discovery compare`). Two models completed: Llama-4-Scout (Groq, a fresh run) and
Pixtral-12B (Mistral). A third, Gemini (`gemini-flash-latest`), is deferred — Google's free tier
caps `GenerateRequestsPerDayPerProjectPerModel` at ~20 requests/day/model, which cannot cover 200
crops; it is noted as future work, not a result.

| model | precision | recall | **control-hallucination** |
|---|---|---|---|
| Llama-4-Scout (Groq) | 0.545 | 0.660 | **0.550** |
| Pixtral-12B (Mistral) | 0.494 | 0.810 | **0.830** |

**Result: the failure generalizes and worsens.** Both independent VLMs fabricate a device on a
majority of clean control crops (Scout 55%, Pixtral 83%). The Scout re-run (0.55) reproduces
V4b's original 0.50 within run-to-run API/decoding drift; Pixtral is markedly worse. **Audited**
(raw descriptions read against the matcher, per `claims-need-controls`): Scout invents specific
devices on clean anatomy ("endotracheal tube," "central venous catheter"); Pixtral emits
near-identical boilerplate ("linear radiopaque object… looks like a medical device or foreign
body") on 83 of 100 control crops. This is genuine fabrication, not a matcher artifact — and the
matcher is conservative on *positives* (it does not fire on hedged "linear artifacts"/"lines"
phrasing), which depresses recall in the safe direction rather than inflating hallucination.

**Interpretation.** The V4b bounded-negative is not model-specific: across two independent VLM
families, an auditor built on off-the-shelf general VLMs would hallucinate a shortcut on 55–83% of
clean images. This is exactly the false-alarm cost the project's precision-over-recall gate was
designed to catch, now shown to be a cross-model property rather than a single-model accident.

## 7. V5 — Mitigation (Deep Feature Reweighting)

**Question:** does last-layer retraining on confound-balanced data raise confound-conflicting
accuracy without collapsing in-distribution accuracy?
**Threshold:** worst-group accuracy rises with minimal in-distribution drop.

**Real tube confound — the clean result.** Worst-group (pneumothorax-without-tube) accuracy
**0.67 → 0.83**; in-distribution (pneumothorax-with-tube) accuracy **0.96 → 0.97** (preserved,
not traded off); reliance gap **0.29 → 0.14**. This uses **no test-distribution labels** — DFR is
fit on a NIH-held-out group-balanced subset — making it the project's one unambiguous
cross-setting mitigation claim.

**Synthetic tag confound — a partial, informative result.** Reliance gap closes (**0.44 → 0.02**)
but in-distribution accuracy drops **0.94 → 0.62**. The original writeup called this backbone
"hollow." A linear probe run specifically to test that claim (freeze the p=0.95 backbone, fit a
fresh head on confound-free images) measured probe AUC **0.690** — well above chance (0.50),
below the clean baseline (0.845). **The claim was corrected**: the backbone learned real but
*degraded* pneumothorax features; the shortcut crowded out, but did not eliminate, core-feature
learning. This correction is carried through every place the original claim appeared (§7).

**Baseline comparison (Phase 3, ERM/JTT/DFR, 3 seeds, mean ± std):**

| confound | method | worst-group | ID | reliance gap |
|---|---|---|---|---|
| tube | ERM | 0.672 ± 0.022 | 0.959 ± 0.002 | 0.287 ± 0.025 |
| tube | JTT | 0.791 ± 0.047 | 0.940 ± 0.019 | 0.149 ± 0.031 |
| tube | DFR | 0.790 ± 0.036 | 0.964 ± 0.007 | 0.174 ± 0.029 |
| tag  | ERM | 0.500 ± 0.000 | 0.940 ± 0.000 | 0.440 ± 0.000 |
| tag  | JTT | 0.608 ± 0.018 | 0.531 ± 0.016 | −0.077 ± 0.025 |
| tag  | DFR | 0.609 ± 0.022 | 0.603 ± 0.014 | −0.006 ± 0.024 |

On the real tube confound, both JTT and DFR beat the ERM floor by ~12 points, well outside the
±0.02–0.05 seed variance — two independent methods agreeing rules out a single-method artifact.
Because neither JTT nor DFR calls the VLM, this table is simultaneously the **naming-off
ablation**: the mitigation loop's worst-group recovery does not depend on the naming step.

## 8. Correlation-strength sweep

**Question:** how does induced reliance and its mitigability vary with confound strength p?
**Result:** trained reliance gap rises **monotonically** with p (single training seed per point,
resumable run — the sweep was interrupted mid-run for travel and resumed cleanly, skipping the
three already-completed points; the DFR recovery step was subsequently re-run over 3 seeds, below):

| p | reliance gap | worst-group before→after | ID before→after |
|---|---|---|---|
| 0.6 | 0.000 | 0.750 → 0.653 | 0.750 → 0.660 |
| 0.7 | 0.010 | 0.745 → 0.670 | 0.755 → 0.668 |
| 0.8 | 0.075 | 0.722 → 0.672 | 0.796 → 0.639 |
| 0.9 | 0.294 | 0.585 → 0.605 | 0.879 → 0.618 |
| 0.95 | 0.440 | 0.500 → 0.593 | 0.940 → 0.617 |

**Finding 1 — onset threshold:** below p≈0.7 the induced correlation is too weak to be exploited;
the model ignores the tag and learns real signal instead (reliance ≈ 0, test AUC ≈ 0.83, matching
clean V1). Reliance only becomes substantial above that threshold.
**Finding 2 (corrected) — the ID cost of DFR is strength-dependent, not binary:** at low p there
is little shortcut to remove, so DFR mostly just costs accuracy for no gain; at high p, DFR's
larger ID drop reflects the same "degraded, not hollow" phenomenon quantified by the linear probe,
now shown to scale with how dominant the shortcut was allowed to become.

**Finding 3 — the recovery is seed-stable, not a single-seed artifact.** The single-seed sweep
above was criticised (fairly) for reporting one DFR run per point. As a follow-up, the DFR
recovery step was re-run over **3 seeds** on the frozen per-p checkpoints (varying the
group-balanced sampling + head fit only — the cheap, non-retraining part; training-seed variance
would require full retraining and is not claimed):

| p | trained gap | worst-group after DFR (mean ± std, n=3) | ID after DFR (mean ± std) |
|---|---|---|---|
| 0.6 | 0.000 | 0.674 ± 0.016 | 0.670 ± 0.018 |
| 0.7 | 0.010 | 0.683 ± 0.015 | 0.667 ± 0.007 |
| 0.8 | 0.075 | 0.673 ± 0.010 | 0.663 ± 0.025 |
| 0.9 | 0.294 | 0.630 ± 0.017 | 0.655 ± 0.030 |
| 0.95 | 0.440 | 0.609 ± 0.022 | 0.603 ± 0.014 |

Every single-seed value in the table above falls within ≈1 std of these means (e.g. p=0.95:
0.593 vs 0.609 ± 0.022), so the sweep's conclusions do not hinge on the seed. Run artifacts:
`experiments/sweep_tag/20260712_133423_f7026c/sweep_variance.json`.

Full figure: `docs/writeup/sweep_figure.png`; table: `docs/writeup/sweep_final_table.md`.

## 9. Phase 2 — end-to-end auditor loop and demo artifact

One command (`python -m auditor.report`) chains detect → localize → (optional name) → mitigate →
report for a given checkpoint. Run against the real tube model: reliance gap 0.287 detected;
worst-group 0.67 → 0.83 after DFR; ID 0.96 → 0.97; naming skipped by default (demonstrating the
loop does not require it). The before/after Grad-CAM figure
(`docs/writeup/auditor_before_after_tube.png`) shows the model's attention **redistributing away
from the tight, confound-associated hotspot** toward broader anatomical evidence after
mitigation — described honestly as a redistribution, not a dramatic single-spot jump, since the
real tube confound is subtle rather than a stark synthetic marker.

## 10. V6 — Cross-site generalization (NIH → PadChest)

**Question:** does a real cross-hospital distribution shift produce a measurable accuracy drop,
and does mitigation narrow it?

**Setup.** Second site: PadChest (Hospital San Juan, Spain), obtained as the 224×224, 12.89GB
Academic Torrents release (not the 1.07TB original — the resized set is standard in the
literature and was the only version that fit on available disk after other project data). A
7,572-image working subset (573 pneumothorax-positive + 7,000 negative) was extracted from the
release tar because the local disk had only 3.3GB free at download time; one image referenced in
the label CSV was absent from the tar and dropped. The `padchest.py` loader's schema-tolerant
column detection (CO2 §2.2) matched the real PadChest label CSV on first attempt with no changes.

**A real bug, caught and fixed.** PadChest images are 16-bit (raw range 0–58,927); NIH images are
8-bit. An initial attempt to reconcile this by per-image standardizing both sites at evaluation
time **broke the model** (NIH reference AUC collapsed from 0.83 to 0.59) because the model was
never trained under that transform — a train/eval mismatch, not a real result. This was caught
*because* the NIH reference AUC was re-measured after the change, rather than assumed. The
correct fix converts PadChest's 16-bit images to the 8-bit scale the model was trained on **at
load time**, leaving the model's transform untouched; the NIH reference AUC was confirmed
restored to **0.842** before trusting any cross-site number.

**Result — the drop is real:** NIH reference AUC **0.842** → PadChest AUC **0.650**, a drop of
**0.192**, consistent in magnitude with Zech et al.'s cross-hospital findings (CO1 §2a).

**Result — mitigation claim corrected after review.** The initial framing reported a "+0.082
recovery from DFR adaptation" and, combined with V5, claimed "mitigation validated across three
settings." A review correctly identified that this recovery came from **fitting the last layer
on labeled PadChest data** — supervised target-domain adaptation, which would help even if no
shortcut were present, and therefore does not support a mitigation claim. The honest test —
**zero-shot source-DFR**: reduce the real tube shortcut using only NIH (source) data, with **no
PadChest labels touched at all**, then evaluate PadChest — was run and returned **+0.005, a
null result**. Reducing a known source-side shortcut did not improve cross-site transfer.

**Corrected conclusion:** V6 establishes that a real, Zech-magnitude cross-site OOD gap exists
(gate: measurable drop — PASS) but does **not** establish that the gap is shortcut-driven or that
it can be mitigated without target labels (gate: narrows via mitigation — **not supported**; the
supervised-adaptation number is relabeled as adaptation, not mitigation). Candidate causes of the
0.192 drop that this project's design cannot distinguish between: scanner/acquisition
differences, population differences, the two datasets' different label-generation pipelines
(NIH's NLP-only labeler vs. PadChest's NLP+manual process), residual intensity artifacts despite
the 8-bit fix, and the 573-vs-7,000 subset composition. Shortcuts are one unproven candidate among
these, not the established cause.

## 11. Final status summary

| Spike | Threshold | Result |
|---|---|---|
| V1 | AUC ≥ 0.75 | **PASS** — 0.8448 |
| V2 | gap > 20pp | **PASS** — 44pp |
| V3 | localize > 80% | **PASS** — 100% (control 0.5%) |
| V4a | P/R ≥ 0.90, halluc < 10% | **PASS** — 0.978 / 0.90 / 0.02 |
| V4b | P ≥ 0.80, R ≥ 0.70, halluc < 10% | **FAIL** — 0.57 / 0.66 / 0.50 → pivot |
| V5 | worst-group ↑, ID stable | **PASS** (tube, clean) / **partial** (tag, ID cost) |
| Linear probe | (post-hoc control) | 0.690 — degraded, not hollow |
| Sweep | (characterization) | onset ~p=0.7; monotone gap; strength-dependent ID cost |
| V6 | drop exists + narrows | **PASS** (drop, 0.192) / **not supported** (narrowing — null zero-shot) |

69 tests pass across the full codebase. Every commit referenced above is in the project's git
history with a message describing the measured result at the time.

## 12. Reproducibility

- Config-driven: every run's exact configuration is one of the fourteen `configs/*.yaml` files
  (or an `--override` on top of one), logged verbatim alongside its metrics.
- Fixed seeds throughout; the one place stochasticity is reported honestly rather than hidden is
  the correlation sweep and Phase-3 baseline table (single *training* seed per sweep point, but the
  DFR recovery step at each point re-run over 3 seeds with reported std; 3 seeds with reported std
  for the baseline comparison — not claimed as more than that).
- Dataset paths are config values, never hardcoded; NIH, NEATX, and PadChest data live outside
  version control (`.gitignore`), while every line of source that reads them is tracked — a
  repo-integrity bug where the source package `src/auditor/data/` was accidentally caught by the
  dataset-ignore rule was found and fixed mid-project (root-anchoring the `.gitignore` pattern).
