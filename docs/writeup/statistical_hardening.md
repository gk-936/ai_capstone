# Statistical hardening: V4a vs V4b control-hallucination

Post-hoc statistics over **existing** prediction files. No new API calls, no new crops, no
model runs. Reproduce with:

```
python -m auditor.discovery.stats_hardening
```

## What is being measured

`control_hallucination` is used **exactly as `src/auditor/discovery/score_naming.py:score()`
defines it**: over the manifest rows with `kind == "control"` (all of which have
`has_confound == False`), the fraction whose free-text description makes the dataset's
matcher fire.

- V4a matcher: `names_confound` (bright/artificial qualifier co-occurring with a marker-like
  object word, or an explicit corner+object mention) — selected by `confound="tag"`.
- V4b matcher: `names_tube` (chest-tube / drain / line device vocabulary) — selected by
  `confound="tube"`.

Rows with no prediction are skipped, as in `score()`. **All four cells had 0 missing
predictions**, so no crop was dropped.

## Data (canonical baselines)

The prediction files are the canonical V4a/V4b baselines hardcoded in
`src/auditor/discovery/score_occlusion.py` (lines 116–125). Manifests are the ones the
existing scorers pair with those predictions.

| cell | manifest | predictions | matcher |
|---|---|---|---|
| v4a / scout | `experiments/v4a_naming/20260708_105244_da01a4/manifest.json` | `experiments/v4a_naming/20260708_105244_da01a4/predictions.jsonl` | `tag` |
| v4a / pixtral | `experiments/v4a_sweep/20260714_144108_a893f3/manifest.json` (opacity = 1.0) | `experiments/v4a_sweep_naming/predictions_1.0_pixtral.jsonl` | `tag` |
| v4b / scout | `experiments/v4b_tube/20260708_204951_3894b5/manifest.json` | `experiments/v4b_tube/multi_vlm_compare/predictions_scout.jsonl` | `tube` |
| v4b / pixtral | `experiments/v4b_tube/20260708_204951_3894b5/manifest.json` | `experiments/v4b_tube/multi_vlm_compare/predictions_pixtral.jsonl` | `tube` |

**Consistency check (not a new result).** The four recomputed point estimates reproduce the
already-recorded numbers exactly: `docs/writeup/exp1_difficulty_curve.json` at opacity 1.0
gives 0.02 (scout) and 0.31 (pixtral); `experiments/v4b_tube/multi_vlm_compare/comparison.json`
gives 0.55 (scout) and 0.83 (pixtral).

## Methodology

- **Bootstrap:** non-parametric, **percentile method**, **10,000 resamples**, seed **1337**
  (`numpy.random.default_rng(1337)`). Control crops are resampled with replacement within a
  cell (n resamples of size n).
- **Gap CI:** the **difference is bootstrapped directly** — V4a and V4b control sets are
  resampled **independently** and the difference `rate_v4a - rate_v4b` is taken **per
  resample**; the CI is the 2.5/97.5 percentiles of those 10,000 differences. It is not
  derived from the two marginal CIs.
- **Test:** `scipy.stats.fisher_exact`, **two-sided**, on the 2×2 table
  `[[v4a_flagged, v4a_not_flagged], [v4b_flagged, v4b_not_flagged]]` per model.
- All CI endpoints below are reported as computed (bootstrap percentiles of a proportion over
  n = 100 land on multiples of 0.01).

## Results

### Control-hallucination rate per cell

| dataset | model | flagged / n | point estimate | 95% CI (bootstrap percentile) |
|---|---|---|---|---|
| V4a | scout | 2 / 100 | 0.02 | [0.00, 0.05] |
| V4b | scout | 55 / 100 | 0.55 | [0.45, 0.65] |
| V4a | pixtral | 31 / 100 | 0.31 | [0.22, 0.40] |
| V4b | pixtral | 83 / 100 | 0.83 | [0.75, 0.90] |

### V4a-vs-V4b gap, per model

| model | gap (V4a − V4b) | 95% CI on the difference | Fisher's exact 2×2 | odds ratio | p (two-sided) |
|---|---|---|---|---|---|
| scout | −0.53 | [−0.63, −0.43] | [[2, 98], [55, 45]] | 0.0167 | 1.22e−18 |
| pixtral | −0.52 | [−0.64, −0.40] | [[31, 69], [83, 17]] | 0.0920 | 7.56e−14 |

n per cell: 100 control crops (V4a scout, V4a pixtral, V4b scout, V4b pixtral), 0 missing
predictions in each.

## Statement of significance

For both models, the 95% bootstrap CI on the V4a-minus-V4b control-hallucination difference
excludes 0, and Fisher's exact test rejects independence at the 0.05 level (scout
p = 1.22e−18; pixtral p = 7.56e−14). The gap is statistically significant at the stated level
for scout and for pixtral. No interpretation beyond this is offered here.

## Caveats

- The two V4a cells come from two different export runs (`v4a_naming/20260708_105244_da01a4`
  for scout, `v4a_sweep/20260714_144108_a893f3` at opacity 1.0 for pixtral). Their manifests
  are identical in `(id, kind, has_confound)` for all 200 rows, but the underlying crop pixels
  were not compared. This pairing is inherited from the existing scorers, not chosen here.
- The V4a and V4b control sets are different crops from different exports; the "independent
  resampling" in the gap bootstrap reflects that, but the two datasets are also not matched on
  patient/image, so the gap mixes any dataset difference with the model-behaviour difference.
- Fisher's exact assumes independent crops within and across the two groups; crop-level
  independence was not separately verified (crops may derive from overlapping source images).
- These are the same crops that produced the original V4a/V4b gate decisions. Nothing here is
  an out-of-sample re-test; it quantifies uncertainty on already-reported numbers only.
