# CO2 — Auditor Architecture and Module Design

*Spurious-Correlation Auditor for Chest X-ray Classifiers.* This document describes the system
that was actually built, module by module, as it exists in `src/auditor/` at project completion.
Every interface described here is implemented and covered by at least one test (69 tests total).

## 1. Design principles

The system was built spike-by-spike under a validation-gated process (CO3), which shaped three
architectural commitments:

1. **Confound-agnostic core.** The attribution and mitigation machinery (Grad-CAM localization,
   DFR) take a generic confound description (a region box, or a `(confound, label)` column pair)
   rather than being hardcoded to "the corner tag." This is why the same `dfr.py` and
   `gradcam.py` modules serve the synthetic tag (V2/V3), the real chest tube (V4b/V5), and the
   cross-site case (V6) without modification.
2. **File-contract boundaries at the riskiest seam.** The VLM-naming step (V4) is the one
   component with an external, rate-limited, occasionally-unavailable dependency. It is
   decoupled from the rest of the pipeline via a `crops/ + manifest.json → predictions.jsonl`
   file contract, so naming can run against Groq, a local Ollama model, or (originally planned)
   a Colab notebook, interchangeably, without touching the scoring or reporting code.
3. **Reuse over duplication across confounds and sites.** `run_v5.py`'s `_features_and_groups_*`
   /`_eval_*` pairs, and `run_v6.py`'s NIH/PadChest loaders, are the only places that branch on
   *which* confound or site is in play; every downstream computation (feature extraction, DFR
   fitting, AUC) is shared code.

## 2. Module map

```
src/auditor/
├── common/         config system, seeding, run logging, path/device resolution
├── data/           dataset loaders: NIH, NEATX (tube labels), PadChest, torch Dataset
├── train/          V1 baseline trainer; V4b tube-reliant trainer
├── shortcut/        synthetic confound injection (corner tag)
├── attribution/    Grad-CAM + confound-region-agnostic localization geometry
├── discovery/      crop export, VLM naming (Groq/Ollama), naming scorer
├── evaluation/      reliance-gap measurement (tag and tube variants)
├── mitigation/      DFR core, ERM/JTT baselines, correlation sweep, linear probe
└── report/         end-to-end auditor loop, before/after demo figure, V6 cross-site runner
```

### 2.1 `common/` — reproducibility substrate
- **`config.py`** — a single pydantic `Config` tree with sub-configs (`DataConfig`,
  `TrainConfig`, `ShortcutConfig`, `AttributionConfig`, `DiscoveryConfig`, `MitigationConfig`)
  loaded from YAML with `extends:` inheritance and `--override key.path=value` CLI overrides. All
  fourteen `configs/*.yaml` files extend a common `base.yaml`, so every spike config is a diff
  against one source of truth (dataset paths, seed, device).
- **`run_logger.py`** — every invocation gets a timestamped `experiments/<spike>/<run>/`
  directory containing the resolved config, `metrics.json`, a full log, and (for training runs)
  `artifacts/model_best.pth`. This is what makes every number in CO3 traceable to an exact run.
- **`seed.py` / `paths.py`** — deterministic seeding across Python/NumPy/PyTorch, and
  device/path resolution that degrades from CUDA to CPU so tests run without a GPU.

### 2.2 `data/` — one loader per data source, one shared torch Dataset
- **`nih.py`** — parses `Data_Entry_2017.csv`, binarizes on a target pathology, and builds
  **patient-disjoint** train/val/test splits from NIH's official split lists (never image-level
  splitting, which would leak a patient's other images across splits).
- **`neatx.py`** — joins the NEATX chest-drain annotations onto the NIH index and builds a
  **tube-enriched training split**: a configurable fraction of training pneumothorax-positives
  are forced to carry a tube annotation, inducing a *natural* (not injected) shortcut, while
  val/test keep the natural tube/no-tube mix so the reliance gap is measured honestly.
- **`padchest.py`** — a **schema-tolerant** loader: it auto-detects PadChest's `ImageID` /
  `Labels` / `PatientID` columns from a candidate list rather than hardcoding one CSV variant,
  parses the pneumothorax label out of PadChest's stringified-list `Labels` field, and builds a
  patient-disjoint (adaptation-subset / test) split. This tolerance mattered: it worked unmodified
  against the real PadChest CSV on first contact.
- **`torch_dataset.py`** — one `NIHDataset` class used by all three loaders above. Includes
  **`_open_as_8bit_L`**, which converts any image (NIH's native 8-bit, PadChest's 16-bit) to the
  8-bit intensity scale the model was trained on at load time — added after a real cross-site
  normalization bug was caught and fixed (CO3 §V6).

### 2.3 `train/` — model training
- **`trainer.py`** — the V1 baseline trainer: timm DenseNet-121, AMP, `pos_weight`-corrected
  BCE loss for the ~3–5% pneumothorax prevalence, and an optional synthetic-tag injection path
  (`shortcut.enabled`) that turns the same trainer into the V2 shortcut-inducing trainer and,
  with a `p_values` sweep, the correlation-strength sweep trainer.
- **`tube_train.py`** — trains on the NEATX tube-enriched split and reports the tube-reliance
  gap at the end of training, reusing `trainer.py`'s model/AMP/loss code.

### 2.4 `attribution/` — localization, built confound-agnostic
- **`gradcam.py`** — `compute_gradcam` (wraps `pytorch-grad-cam` against a DenseNet-121 layer),
  plus pure geometry functions that take a **generic box** `(y0, x0, y1, x1)`: `peak_in_region`
  (the "pointing game"), `mass_fraction`, `salient_crop`, and `peak_on_side` (a coarse
  left/right check respecting the radiological convention that patient-left appears on the
  image's right). None of these functions know whether the confound is a synthetic tag or a
  real tube — that decision lives entirely in the caller.
- **`localize_confound.py`** — the V3 experiment: runs Grad-CAM over tagged and untagged images
  and reports the localization rate plus an untagged control (to rule out "the corner is always
  salient" as a false explanation).

### 2.5 `discovery/` — the VLM-naming seam
- **`export_crops.py` / `export_tube_crops.py`** — build the `crops/` + `manifest.json` file
  contract: salient crops from confound-positive images (ground truth is exact for the synthetic
  tag; comes from NEATX for the tube) plus confound-free control crops, with perfect
  `has_confound` labels the VLM never sees.
- **`name_groq.py`** — calls a Groq-hosted open-weight vision model (Llama-4-Scout) with a
  **blind prompt** ("describe anything artificial, non-anatomical, or out of place... if normal,
  say so") that never names the target confound. Includes round-robin multi-key rotation and
  automatic retry/backoff on 429s and transient network errors, added after real rate-limit and
  connection-drop failures during the V4b run — the naming step is resumable, so a
  partially-completed batch picks up where it left off rather than restarting.
- **`name_ollama.py`** — an equivalent local path via a self-hosted Ollama vision model, built
  as a fallback when Groq rate limits were biting; kept in the repo though the tested local model
  (qwen2.5vl:3b) proved too unreliable in practice (documented in CO3, not used for a scored run).
- **`score_naming.py`** — turns free-text VLM descriptions into precision / recall /
  control-hallucination against the manifest, via two independent conservative keyword matchers
  (`names_confound` for the tag vocabulary, `names_tube` for the device vocabulary), each tuned to
  avoid firing on generic anatomy language.

### 2.6 `evaluation/` — reliance measurement
- **`reliance.py`** — the synthetic-tag reliance gap: evaluates the same trained model with the
  tag aligned (as trained), removed, and flipped (anti-correlated), on a class-balanced subset so
  the accuracy numbers are meaningful under a rare-disease prevalence.
- **`tube_reliance.py`** — the real-tube analogue: compares accuracy on held-out
  pneumothorax-positive images **with** vs **without** an annotated tube, since a real device
  cannot be synthetically added or removed the way the tag can.

### 2.7 `mitigation/` — DFR core, baselines, sweep, and the probe
- **`dfr.py`** — `group_balanced_sample` (balances whichever `(confound, label)` groups actually
  exist — important because chest tubes co-occur almost exclusively with pneumothorax-positives,
  so one of the four textbook groups is structurally empty), `extract_features` (frozen-backbone
  penultimate features), `fit_dfr_head` (logistic regression on standardized features, with the
  standardization folded back into a raw-space weight/bias), and `apply_dfr` (swaps only the
  final linear layer, leaving the backbone untouched).
- **`baselines.py`** — `fit_jtt_head` (upweights the frozen model's own misclassifications and
  refits the head), giving ERM/JTT/DFR from the same feature-extraction code path.
- **`run_v5.py`** — the single-confound mitigation experiment (before/after worst-group and ID
  accuracy), parameterized over `{tag, tube}`.
- **`run_ablation.py`** — the Phase-3 baseline table: ERM/JTT/DFR × {tag, tube} × several seeds,
  aggregated to mean±std. Because none of these three methods call the VLM, this table doubles as
  the naming-off ablation.
- **`run_sweep.py`** — trains the tag model at a list of correlation strengths `p`, resumable
  (skips any `p` whose run already completed, so an interrupted sweep restarts only the missing
  points), reporting reliance gap and DFR before/after at each `p`.
- **`probe.py`** — the linear-probe control: freezes a shortcut-trained backbone, fits a *fresh*
  head on frozen features of **confound-free** images, and reports the probe AUC. This module was
  added specifically to test (and, as it turned out, falsify) a narrative claim before it was
  allowed to stand (CO4 §1).

### 2.8 `report/` — the end-to-end loop and the cross-site experiment
- **`run_auditor.py`** — the single command that composes the pipeline: load a checkpoint →
  measure the reliance gap (detect) → run DFR and re-measure (mitigate) → optionally fold in a
  VLM naming score if a `predictions.jsonl` is supplied (name) → write `audit_report.{json,md}`.
  Naming is optional by design: the loop's mitigation result does not depend on it, which is the
  direct code-level demonstration behind the naming-off ablation.
- **`demo_figure.py`** — renders the before/after Grad-CAM montage (original vs. DFR-mitigated
  model, same images), the project's primary visual artifact.
- **`run_v6.py`** — the cross-site experiment: measures NIH-reference and PadChest AUC under the
  *same* image loader, runs the supervised-adaptation variant (DFR fit on a small labeled
  PadChest subset), and — added after a review caught the original framing overclaiming — a
  **separate zero-shot variant** (`run_v6_source_dfr_zeroshot`) that fits DFR only on NIH tube
  groups and evaluates PadChest with no target labels touched at all, to test whether reducing a
  known source-side shortcut actually improves cross-site transfer.

## 3. Data flow (the full loop, as `run_auditor.py` executes it)

```
trained checkpoint (+ confound type: tag | tube)
   │
   ├─ DETECT   → evaluation/{reliance,tube_reliance}.reliance_gap(...)
   │              accuracy with vs. without the confound → reliance gap
   │
   ├─ LOCALIZE → attribution/gradcam.compute_gradcam + peak_in_region / peak_on_side
   │              (used directly in V3; folded into the demo figure in Phase 2)
   │
   ├─ NAME (optional) → discovery/export_crops.py → crops/ + manifest.json
   │                      → [Groq | Ollama] blind captioning → predictions.jsonl
   │                      → discovery/score_naming.load_and_score → P / R / control-hallucination
   │
   ├─ MITIGATE → mitigation/dfr.{extract_features, fit_dfr_head, apply_dfr}
   │              group-balanced last-layer retrain → worst-group / ID accuracy, before vs after
   │
   └─ REPORT   → report/run_auditor._markdown → audit_report.{json,md}
                  report/demo_figure.render_before_after → before_after_gradcam.png
```

The cross-site experiment (`run_v6.py`) is the same DETECT/MITIGATE machinery pointed at a second
data source (`data/padchest.py`) instead of a second confound, which is why it required no new
mitigation code — only a new loader and the 8-bit normalization fix.

## 4. What is *not* in the architecture, deliberately

- **No generative model.** Unlike RoentMod (CO1 §2b), nothing in this pipeline synthesizes or
  edits images. The synthetic confound (V2/V3) is a pixel-level corner stamp, not a generative
  counterfactual.
- **No confound-specific code path in `dfr.py` or `gradcam.py`.** Every confound-specific detail
  (what the tag looks like, which CSV has tube labels, which side is "left") lives in `data/` or
  `evaluation/`, never in the shared attribution/mitigation core.
- **No dependency on the VLM for anything except the naming metric itself.** This is enforced by
  construction, not by convention: `run_v5.py` and `run_ablation.py` have no import of
  `discovery/`.

## 5. Test coverage by module

69 tests across the codebase, concentrated on the parts most likely to silently break or most
load-bearing for a claim in CO4: the synthetic-tag injection geometry, the NIH/NEATX/PadChest
loaders' patient-disjointness (the classic leakage trap), the naming scorer's keyword matchers
(explicitly tested to *not* fire on generic anatomy — the control-hallucination guard), the DFR
group-balancing (including the structurally-empty-group case), and the report/demo modules'
end-to-end wiring on synthetic data.
