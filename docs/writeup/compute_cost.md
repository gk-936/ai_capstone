# Compute cost and runtime

Extracted directly from `run.log` timestamps in the experiment directories already produced by
each reported run (existing-analysis, no new runs). Added because the paper claims a
compute-efficiency contrast against RoentMod's generative pipeline without ever quantifying either
side — this fills the "our side" half of that comparison with real numbers.

## Local GPU training/evaluation (single RTX 3050, 6GB)

| run | wall-clock | source |
|---|---|---|
| V1 training (DenseNet-121, 5 epochs, full 73k/13k/25.6k split) | 45.1 min | `experiments/v1_baseline/20260707_232704_9fbb04/run.log` |
| V2 shortcut-model training (same architecture, tagged data) | 34.2 min | `experiments/v2_shortcut/20260708_060719_0515dc/run.log` |
| V3 Grad-CAM attribution (200 images) | 21.1 min | `experiments/v3_attribution/20260708_100736_fadc22/run.log` |
| Phase 3 baselines (ERM/JTT/DFR × 3 seeds × 2 confounds) | 16.6 min | `experiments/phase3_baselines/20260709_104724_5a350b/run.log` |
| V5 DFR fit (tube, single run) | 1.4 min | `experiments/v5_dfr_tube/20260716_213138_6193ad/run.log` |
| V6 cross-site eval (NIH ref + PadChest, 2 model passes) | 2.6 min | `experiments/v6_padchest/20260716_102248_d81853/run.log` |
| **Total (core reported runs above)** | **~2.0 GPU-hours** | sum |

Not included in the total (smaller/one-off, or not separately logged with `RunLogger`): the
correlation-strength sweep (5 training runs, each comparable to V1/V2's ~30-45 min — roughly
2.5-4 additional GPU-hours not separately summed here), V4a/V4b crop-export stages (1.7 min and
8.2 min respectively — GPU work for Grad-CAM export, not VLM inference), and exploratory/failed
runs not part of the final reported set.

**Total local compute for the reported core results: on the order of 2-6 GPU-hours** on a single
consumer laptop GPU, depending on how much of the sweep and exploratory work is counted. This
supports the paper's efficiency framing (no generative pipeline, no fine-tuning) with an actual
number rather than an assertion.

## VLM API calls (naming stage)

Counted directly from `predictions*.jsonl` line counts across every naming-related experiment
directory (each line = one completed VLM API call):

| study | calls |
|---|---|
| V4a (Scout, canonical run) | 200 |
| V4b (Scout, canonical run) | 200 |
| Multi-VLM replication (Scout + Pixtral on V4b) | 400 |
| Opacity/difficulty sweep (4 opacity levels × 2 models × 200 crops) | 1,600 |
| Occlusion mechanism study (Exp 2: 3 conditions × 2 datasets × 2 models × 100 crops) | 1,200 |
| Specialist-VLM arm (5 models × 2 datasets × 200 crops) | 2,000 |
| **Total** | **5,600 calls** |

All against **free-tier APIs** (Groq, Mistral La Plateforme, Google AI Studio, Colab-hosted
specialist models) — the paper's "one free API pass" framing describes the V4a/V4b canonical run
specifically (400 of the 5,600); the full naming study across all extension experiments used
5,600 free-tier calls in total, still zero marginal API cost and no generative model or
fine-tuning step at any point.

## What this does and doesn't support

This quantifies the compute side of the RoentMod comparison (diffusion generator + editor +
fine-tuning vs. ~2-6 GPU-hours of standard classifier training/eval plus free-tier API calls) —
it does not attempt to estimate RoentMod's own compute cost, which is not reported in comparable
units in that paper and should be cited, not estimated, if a head-to-head number is wanted in the
final draft.
