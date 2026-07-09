# V5 — Mitigation via Deep Feature Reweighting (design)

**Spike question:** Does last-layer retraining on confound-balanced data reduce the model's
reliance on the shortcut — raising confound-conflicting accuracy with minimal in-distribution
drop — for BOTH the synthetic tag and the real chest tube?

**Gate:** confound-conflicting group accuracy RISES after DFR AND in-distribution accuracy holds
(a small ID drop is acceptable). Applied to the V2 (tag) model and the V4b (tube) model.

## Why this matters (anti-"VLM-wrapper")
Mitigation is what a VLM wrapper cannot do: it repairs a broken model. Running it on the REAL
tube confound — whose *naming* failed (V4b) but whose *detection+localization* succeeded — shows
the auditor fixes real shortcuts even where automated naming is unreliable. DFR (Kirichenko 2022)
also doubles as a named baseline for Phase 3.

## Method: Deep Feature Reweighting
Freeze the trained backbone; retrain ONLY the final linear layer on a group-balanced set with all
four (confound present/absent) x (label +/-) groups equalized. This breaks the confound<->label
correlation the head exploited, forcing reliance back toward core features. Last-layer-only, so
cheap on the RTX 3050.

## Components
- `src/auditor/mitigation/dfr.py` (confound-agnostic)
  - `extract_features(model, loader, device)` -> frozen penultimate features + labels.
  - `group_balanced_sample(rows, confound_col, label_col, seed, per_group)` -> equal-sized
    4-group subset (deterministic).
  - `fit_dfr_head(features, labels)` -> retrained final layer (logistic regression on frozen
    features per DFR).
  - `apply_dfr(model, weight, bias)` -> model with the swapped head; backbone untouched.
- `src/auditor/mitigation/run_v5.py`
  - Load a trained checkpoint (tag model or tube model via config/CLI).
  - Build the group-balanced DFR set, extract frozen features, fit the new head.
  - Measure confound-conflicting accuracy BEFORE vs AFTER, reusing:
    synthetic -> evaluation/reliance.py (tag aligned/removed/flipped);
    tube -> evaluation/tube_reliance.py (PTX with-tube vs without-tube).
  - Report worst-group acc (before/after), ID acc (before/after), reliance gap (before/after).
- Configs: `configs/v5_dfr_tag.yaml` (extends v2_shortcut), `configs/v5_dfr_tube.yaml`
  (extends v4b_tube). Checkpoint path + DFR params (per-group size, head type).

## Data flow
```
trained ckpt (frozen backbone)
  -> group-balanced set (confound x label equalized) -> extract frozen features -> fit new head
  -> eval before/after: worst-group acc UP, ID acc ~held, reliance gap DOWN
```

## Tests (synthetic spine)
- group_balanced_sample: equal counts across the 4 groups; deterministic per seed.
- fit_dfr_head: on a synthetic separable feature set, recovers a correct linear boundary.
- apply_dfr: swaps the head, backbone params unchanged (identity check).

## PASS criteria & skeptic checks
- PASS: confound-conflicting accuracy rises (both confounds) AND ID accuracy holds.
- Skeptic: if worst-group rises but ID COLLAPSES, it's not a real win — report both, gate on the
  pair.
- If DFR barely moves the needle -> documented fallback: swap mitigation family (masking /
  unlearning). DFR is the chosen first method; masking is the fallback.
- The tube case is the honest hard one — real gains may be modest; report faithfully, even a
  partial narrowing is a legitimate real-data result.

## Runtime
Last-layer only: one forward pass over the balanced set for features (~minutes), fit in seconds.
Cheaper than V1/V4b training. Fits the RTX 3050.
