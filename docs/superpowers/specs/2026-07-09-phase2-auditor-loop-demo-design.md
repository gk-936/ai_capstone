# Phase 2 — End-to-End Auditor Loop + Demo Figure (design)

**Goal:** One command runs the full auditor on a (checkpoint, confound) and emits a consolidated
report plus the before/after Grad-CAM demo figure. Pure wiring of existing tested modules — no
new algorithms.

## Component: `src/auditor/report/run_auditor.py` (CLI `python -m auditor.report`)
Given a trained checkpoint + confound type (tag/tube):
1. Detect — reliance gap (reuse reliance.py / tube_reliance.py).
2. Localize — Grad-CAM peak-in-region / side-consistency (reuse attribution/).
3. Name (optional) — if a predictions.jsonl is supplied, fold in VLM naming P/R/hallucination
   (reuse score_naming). Skippable: the loop works naming-off.
4. Mitigate — DFR before/after worst-group + ID (reuse run_v5 feature builders + dfr.py).
5. Report — write audit_report.json + audit_report.md (all metrics in one place).
6. Demo figure — for N confound-positive images: Grad-CAM from the ORIGINAL model vs the
   DFR-MITIGATED model side by side -> before_after_gradcam.png (saliency moves off the confound).

## Data flow
```
(checkpoint, confound) ->
  detect(gap) + localize(CAM) + [name] + mitigate(DFR before/after)
  -> audit_report.{json,md}
  -> N images: CAM(original) vs CAM(DFR) -> before_after_gradcam.png
```

## Components & reuse (nothing new algorithmically)
- run_auditor.py: orchestration over existing reliance/tube_reliance, gradcam, dfr, run_v5
  feature builders, score_naming.
- demo_figure.py: montage via compute_gradcam + _denormalize/overlay helpers.
- configs/auditor_tube.yaml, configs/auditor_tag.yaml: checkpoint + confound + N demo images.

## Tests
- Report assembler: canned metrics -> expected audit_report.json keys.
- Demo montage: renders a PNG from a tiny synthetic model+images (no crash, right shape).
- Underlying detect/localize/mitigate already tested — not re-tested.

## Success (integration, not a gate)
One command yields a coherent report + a figure where post-DFR saliency visibly shifts off the
confound on the tube model (strongest case).

## Runtime
DFR (last-layer) + a handful of Grad-CAM passes. ~5 min GPU, CPU-tolerable for the figure.

## Also
Fix stale PROGRESS checkboxes: CO1, ablations, multi-seed variance, RoentMod contrast are done;
check off the Phase 2 items this delivers.
