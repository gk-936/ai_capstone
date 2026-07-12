# Extension Plan — pushing the auditor from a 6.5–7 toward an 8+ paper

Written after team + external review converged on the same caps. Three phases, gated per step,
same validation-first discipline as V1–V6. The rule from `claims-need-controls` still holds:
before writing any interpretive claim, run the control that could falsify it.

## Review consensus (what caps it below 8)
1. **Single VLM tested** (Llama-4-Scout) — reviewers will ask if GPT-4V / a specialist CXR-VLM
   fails the same way. → Phase 2.
2. **Non-expert (student) tube labels** (NEATX) — noisier than radiologist ground truth. → CANDID-PTX.
3. **Single-seed correlation sweep; small PadChest subset.** → Phase 1 (partial) + overnight retrain.
4. **Headline is mostly negative/bounding** — a "here's where it breaks" paper. Phase 2 could add a
   positive capability result (a specialist VLM that *does* name real confounds).
5. **V6 causal question left open** — why the OOD gap exists, if not the shortcut. → Phase 3.

## Phase 1 — cheap wins (in progress)

### 1a. CANDID-PTX (expert labels + a THIRD site) — START NOW, long lead time
- 19,237 CXRs, Dunedin Hospital NZ, with **radiologist/trainee chest-tube + pneumothorax
  segmentations** (RLE CSVs). DOI 10.17608/k6.auckland.14173982.
- Access = free online ethics course (globalhealthtrainingcentre.tghn.org) + data-use agreement
  emailed to the corresponding author. Days-to-weeks turnaround.
- **Two payoffs:** (i) radiologist-grade tube ground truth to re-score V4b naming and re-run V5
  mitigation, replacing the non-expert NEATX labels for those specific claims; (ii) a third,
  independent site (US→Spain→NZ) for the Phase-3 cross-site causal work.
- **Action:** register + complete the ethics course, submit the DUA. This is the user's to do;
  the code path (a `data/candid.py` loader) plugs into the existing site-agnostic design.

### 1b. Sweep DFR-recovery variance (cheap — existing checkpoints, last-layer only)
- The Phase-3 table already has 3-seed DFR variance at p=0.95. Extend that to the other four
  sweep checkpoints (p=0.6/0.7/0.8/0.9) by re-running the DFR+eval step at 3 seeds on the frozen
  per-p models. Removes the "single seed" criticism for the *mitigation* curve.
- **Honest scope:** this varies the DFR/eval seed, not the *training* seed. The reliance-gap
  values themselves (which need retraining to re-seed) stay single-seed unless we spend the
  ~5-hour overnight full retrain — offered as optional, not blocking.

## Phase 2 — Multi-VLM naming study (brainstorm before building)
Test V4b (real tube) naming across models to see whether the 0.50 control-hallucination is
Scout-specific or general: a second Groq vision model (free), GPT-4V/4o (needs an OpenAI key),
and an open specialist CXR-VLM on Colab (CheXagent / MAIRA-2 / LLaVA-Rad / CXR-LLaVA — the ones
that could actually pass). Plus abstention/confidence and larger-crop ablations, reported as a
systematic study (not gate-gaming). Either outcome is publishable: general failure = strong
warning; specialist success = a positive capability result that lifts the negative-heavy headline.

## Phase 3 — V6 causal decomposition (brainstorm before building)
Answer "why the NIH→PadChest gap, if not the shortcut?" with separable experiments:
- **Hospital-detector probe** (Zech-style): classify NIH vs PadChest; Grad-CAM the detector to
  see *what* differs.
- **Label-pipeline / prevalence analysis:** evaluate on high-agreement PadChest labels; quantify
  how much of the drop is label-definition noise vs. real shift.
- **Label-free test-time adaptation (TENT/BN-adapt)** vs. the null source-DFR: if generic TTA
  closes the gap but shortcut-removal doesn't, the gap is general covariate shift, not a specific
  learnable shortcut — a clean positive mechanism attached to the existing negative result.
- Adding CANDID (NZ) as a third site turns two points into a trend.

## The standing trap (do not repeat)
"Taking it further" must not become "finding the framing where it worked." If the second VLM
also hallucinates, that is the result. If TTA does not close the gap, that is the result. The
project is rated for reporting its own failures; that must survive the expansion.
