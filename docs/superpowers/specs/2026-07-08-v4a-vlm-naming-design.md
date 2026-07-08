# V4a — VLM Names the Synthetic Confound (design)

**Spike question:** Can a general-purpose open-weight VLM, given only saliency crops (never
told what to look for), NAME the injected confound?

**Gate (CRITICAL):** precision ≥ 0.90 AND recall ≥ 0.90, control-hallucination < 10%.
Failing V4a means the pipeline is broken — FIX it, do NOT pivot (the pivot is a V4b-only event).

## Boundary — file contract (local ↔ Colab)
```
LOCAL (build)                COLAB (run)                    LOCAL (score)
export_crops.py         →    v4a_vlm_naming.ipynb      →    score_naming.py
  crops/<id>.png               Qwen2-VL-7B-Instruct           predictions.jsonl
  manifest.json                (4-bit on a free T4)             → precision / recall /
  {id, kind, has_confound}     blind "what's out of place?"        control-hallucination
                               → predictions.jsonl
```
Files, not a live API: reproducible, no secrets. Upgradeable to a live endpoint later.

## Components
- `src/auditor/discovery/export_crops.py`
  - Positive crops: salient crops from tagged POSITIVE test images (V3 showed the peak lands
    on the tag, so these contain it). Default 100.
  - Control crops: crops from UNTAGGED images (no confound). Default 100.
  - `crops/<id>.png` + `manifest.json` rows `{id, kind: positive|control, has_confound: bool}`.
    `has_confound` is perfect ground truth — we injected it.
- `src/auditor/discovery/colab/v4a_vlm_naming.ipynb` (run on Colab/Kaggle)
  - transformers + bitsandbytes; Qwen2-VL-7B-Instruct in 4-bit.
  - Blind prompt (never names tag/box/corner): "This is a region a diagnostic model focused on
    in a chest X-ray. Describe anything artificial, non-anatomical, or out of place. Be specific."
  - Writes `predictions.jsonl` rows `{id, description}`.
- `src/auditor/discovery/score_naming.py`
  - Conservative keyword matcher: names the tag iff description mentions a bright/white +
    box/square/rectangle/marker/tag/label/patch, or an explicit corner marker. Tuned to avoid
    firing on clean controls (control-hallucination is the hardest bar).
  - precision, recall, control-hallucination; writes metrics.json; prints a table.
  - Manual audit of ~20 raw descriptions validates the matcher before the number is trusted.
- `configs/v4a_naming.yaml` — crop counts, source checkpoint, crop size, paths.

## Tests (synthetic spine)
- export_crops: manifest has correct has_confound per kind; files exist; counts match config.
- score_naming matcher: "white square in corner"→hit; "clear lung fields"→miss;
  "bright opacity"→miss (guards control-hallucination). Precision/recall/hallucination math
  checked on a synthetic predictions set with known answers.

## PASS criterion & skeptic checks
- PASS: precision ≥ 0.90, recall ≥ 0.90, control-hallucination < 10%.
- Manual audit is the anti-self-deception step — read raw descriptions before trusting the score.
- Low recall → diagnose prompt vs genuine VLM miss. Per CLAUDE.md, V4a fail = fix pipeline.

## What runs where
- Local now: export_crops, score_naming, config, tests, crop generation (no GPU needed).
- Colab (user): the notebook on a free T4; ~200 crops × Qwen2-VL is minutes.
