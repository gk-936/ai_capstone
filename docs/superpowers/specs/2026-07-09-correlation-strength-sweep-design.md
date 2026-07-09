# Correlation-Strength Sweep (design)

**Goal:** Show how shortcut reliance and DFR mitigation-recovery vary with confound strength `p`
— a curve, not a point. Produces a figure and resolves the tag-case ID-collapse (a milder `p`
model retains real signal, so DFR preserves ID).

## What it does (one script, 5 p values)
For each `p in {0.6, 0.7, 0.8, 0.9, 0.95}`:
1. Train a synthetic-tag pneumothorax model at that `p` (reuse `train_v1` with shortcut.enabled,
   shortcut.p = p).
2. Measure the reliance gap (reuse `reliance_gap`).
3. DFR-mitigate (reuse fit_dfr_head / apply_dfr) -> worst-group + ID before/after.
4. Record one row: p, gap_before, worst_group_before/after, id_before/after, gap_after.

Emits `sweep_results.json`, a printed table, and `sweep_table.md`.

## Component
- `src/auditor/mitigation/run_sweep.py` — `run_sweep(cfg, logger, p_values)`: orchestration only,
  reusing the existing trainer + reliance + DFR (no new modelling logic).
- `configs/sweep_tag.yaml` — extends v2_shortcut; `sweep.p_values` list; epochs 3.
- CLI: `python -m auditor.mitigation --config configs/sweep_tag.yaml --sweep`.

## Data flow
```
for p in [.6,.7,.8,.9,.95]:
   train tag model @p -> reliance_gap -> DFR -> worst-group/ID before/after
-> combined table: gap-vs-p and (worst-group recovery, ID retention)-vs-p
```

## Tests
- sweep-row assembler produces the expected keys per p.
- Training/DFR are already covered by V2/V5 tests; no new core logic to test.

## Expected result (hypothesis, measured honestly)
- Reliance gap rises with p (stronger correlation -> more cheating).
- DFR recovers worst-group across all p; ID retention is better at lower p (milder shortcut ->
  more real signal preserved -> resolves the tag ID-collapse for p<=~0.8).
- A flat/non-monotone curve is reported as-is.

## Config note
`sweep.p_values` needs a config field (list). Add a SweepConfig or reuse a plain list under a new
key. Each p's run writes its own checkpoint (gitignored); combined json is the figure source.

## Runtime
5 x (~15min train + ~3min DFR/eval) ~= 90 min GPU, unattended. Single seed per p (Phase-3 table
already provides variance on the headline; the sweep is about the trend).
