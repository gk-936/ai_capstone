# V2 — Shortcut Induction (design)

**Spike question:** Does the classifier measurably *cheat* when a controllable synthetic
confound is correlated with the label at train time?

**Gate:** reliance gap = `acc(aligned) − acc(removed)` **> 20 percentage points**.
If weak, increase correlation strength `p` / tag size until unambiguous before failing.

## Confound
A bright white **corner tag** (fixed corner, default top-left, ~20×20 px, value = max).
Chosen because it is salient (forces a gap), trivially localizable (feeds V3), and plainly
nameable by a VLM (feeds V4a: "a white square marker in the corner").

## Correlation structure
Tag applied per a rule with strength `p` (default 0.95) in one of three modes:
- `aligned`  — tag on positives w.p. `p`, on negatives w.p. `1−p` (the train-time rule).
- `removed`  — never tag.
- `flipped`  — tag on negatives w.p. `p`, on positives w.p. `1−p` (anti-correlated).

Tag decisions are **deterministic per image given a seed** so evaluation is reproducible.

## Components
- `src/auditor/shortcut/inject.py`
  - `add_corner_tag(x, corner, size, value)` — stamp bright square on a normalized 3×H×W
    tensor; returns a new tensor, original untouched.
  - `tag_decision(image_name, label, mode, p, seed)` — deterministic bool.
  - `TaggedDataset` — wraps `NIHDataset`, applies the tag per (mode, p).
- `src/auditor/evaluation/reliance.py`
  - `balanced_eval_subset(index, split, seed)` — equal pos/neg rows so accuracy is honest
    (chance = 50%).
  - `evaluate_accuracy(model, loader, threshold=0.5)`.
  - `reliance_gap(model, index, data_cfg, shortcut_cfg, device)` → dict:
    `{acc_aligned, acc_removed, acc_flipped, gap}`, gap = acc_aligned − acc_removed.
- `configs/v2_shortcut.yaml` — `extends: base.yaml`; adds `shortcut: {corner, size, p, seed}`;
  `epochs: 3` (a shortcut is learned fast).
- Training reuses the V1 trainer with the train dataset swapped to
  `TaggedDataset(mode="aligned")`.

## Data flow
```
NIH train → TaggedDataset(aligned, p) → train model
model → balanced test subset, evaluated aligned / removed / flipped → reliance gap
```

## Tests (synthetic spine — must not silently break)
- `add_corner_tag` sets bright pixels in the correct corner and leaves other pixels unchanged.
- Correlation rule: `p=1.0` ⇒ every positive tagged, no negative tagged; `p=0.5` ⇒ ~half
  (statistical check, fixed seed).
- `tag_decision` deterministic per seed.
- `balanced_eval_subset` returns exactly 50/50.

## PASS criterion & skeptic checks
- PASS: gap > 20 pp.
- Gap ≈ 100 pp is plausible for a pure shortcut learner **iff** `acc_removed ≈ 50%` (model at
  chance without its crutch). Verify that signature rather than assuming a bug.
- Gap < 20 pp: raise `p` / `size` before declaring FAIL.

## Runtime
Same envelope as V1 (batch 16, ~8.5 min/epoch). 3 epochs ⇒ ~25 min.
