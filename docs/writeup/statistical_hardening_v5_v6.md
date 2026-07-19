# Statistical hardening (part 2): V5 mitigation delta and V6 OOD numbers

Companion to `statistical_hardening.md` (which covers only the V4a-vs-V4b naming gap). This
extends the same discipline — quantify uncertainty on already-reported point estimates — to the
two other headline numbers that previously had none: V5's DFR-vs-ERM worst-group delta, and V6's
OOD drop and zero-shot source-DFR null. **Existing-analysis only: no new training, no new API
calls.** V5 reuses the already-logged 3-seed baseline rows; V6 reuses the exact V1/tube
checkpoints and the exact PadChest/NIH test sets already used for the reported run.

## V5 — DFR/JTT vs ERM, tube confound (3-seed baseline table)

Source: `experiments/phase3_baselines/20260709_104724_5a350b/baseline_rows.json`, tube rows only,
worst-group accuracy, seeds {1337, 7, 42}.

| method | seed 1337 | seed 7 | seed 42 | mean |
|---|---|---|---|---|
| ERM | 0.670 | 0.700 | 0.645 | 0.672 |
| JTT | 0.835 | 0.8125 | 0.725 | 0.791 |
| DFR | 0.830 | 0.7975 | 0.7425 | 0.790 |

Paired (by seed) t-test, JTT vs ERM and DFR vs ERM:

| comparison | mean diff | paired t p-value |
|---|---|---|
| JTT − ERM | +0.119 | **0.041** |
| DFR − ERM | +0.118 | **0.030** |

**Honest caveat, stated plainly:** n=3 seeds. A paired t-test at n=3 has almost no power to detect
anything but a large, consistent effect, and the reported p-values should not be read as strong
evidence in the way a larger-n test would provide — they are reported because the project's own
standard (apply the same statistical scrutiny everywhere) requires it, not because n=3 supports a
confident significance claim. The more defensible statement is the one already in the paper: JTT
and DFR both beat ERM in **all three seeds**, by a margin (~12pp) that is large relative to the
observed within-method seed spread (~5-6pp), and two independent methods agree — that qualitative
pattern is the actual evidence, not the p-value on its own.

A bootstrap resampling the 3 seeds (10,000 resamples, seed 1337) gives a 95% CI on DFR−ERM of
[0.098, 0.160] and JTT−ERM of [0.080, 0.165] — both exclude zero, but a bootstrap with only 3
underlying values has a small, discrete resample space (27 possible resamples) and should be read
as a rough robustness check, not a proper-power interval.

**Recommendation for the paper:** report both the point estimates (already there) and this n=3
caveat explicitly next to any significance claim, rather than a bare p-value.

## V6 — OOD drop and zero-shot source-DFR, with bootstrap CIs

Independently re-derived (not just re-read from `metrics.json`) by re-running inference through
the exact same checkpoints (`experiments/v1_baseline/.../model_best.pth` for the OOD-drop number;
`experiments/v4b_tube/20260708_204138_da03d8/.../model_best.pth` for the source-DFR number) over
the exact same NIH/PadChest test sets already used for the reported run, this time saving
per-image scores for a stratified percentile bootstrap (2,000 resamples, seed 1337).

### OOD drop

| quantity | point estimate | 95% CI |
|---|---|---|
| NIH reference AUC | 0.8423 | [0.828, 0.856] |
| PadChest OOD AUC | 0.6503 | [0.619, 0.680] |
| **OOD drop** | **0.1920** | **[0.159, 0.225]** |

Point estimates reproduce `experiments/v6_padchest/20260716_102248_d81853/metrics.json` exactly
(0.8423 / 0.6503 / 0.1920), confirming the re-derivation matches the original run. The drop's CI
excludes zero by a wide margin — this is a real, robust effect, not a borderline one.

### Zero-shot source-DFR (the null result the paper's strongest claim rests on)

| quantity | point estimate | 95% CI |
|---|---|---|
| PadChest AUC before source-DFR | 0.6505 | — |
| PadChest AUC after source-DFR | 0.6551 | — |
| **zero-shot change** | **+0.0046** | **[−0.018, +0.028]** |

Point estimate reproduces `experiments/v6_source_dfr/*/v6_source_dfr.json` (+0.0046, matching the
paper's rounded "+0.005") exactly, confirming the re-derivation is consistent with the original
run.

**This is the result of the hardening pass that most changes how the paper should state its
claim.** The question was whether "+0.005" is a *tight* null (rules out a moderate effect) or a
*wide, uninformative* one. It is closer to the latter: the 95% CI, **[−0.018, +0.028]**, comfortably
includes zero — so the point estimate is not evidence of a real recovery, and the "null" framing is
still correct — but the interval is also wide enough that it does **not** tightly rule out a modest
positive effect (up to roughly +0.03 AUC) either. The honest statement is: *this test cannot
distinguish "no effect" from "a small effect too small for this sample size to detect."* The
current paper phrasing ("reducing a known source-side shortcut does not improve cross-site
transfer... a null result") is directionally correct but should be softened to reflect that the
null is underpowered to rule out a small-to-moderate effect, not just "genuinely zero." This is a
more honest, defensible version of the paper's single most load-bearing negative claim, and it
should replace the unqualified "null" language in the Abstract, §4.3, and Discussion.

## Files

- `docs/writeup/v6_bootstrap_ci.json` — OOD-drop bootstrap (done; excludes zero by a wide margin,
  [0.159, 0.225] — this one *is* a tight, robust effect).
- `docs/writeup/v6_source_dfr_bootstrap_ci.json` — source-DFR null bootstrap (done; wide CI
  including zero, see above — an underpowered null, not a tight one).
- Reproduced with a paired/stratified percentile bootstrap (2,000 resamples, seed 1337), reusing
  the exact checkpoints and test sets from the original runs — no new training, no new API calls.
