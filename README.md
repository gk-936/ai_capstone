# Spurious-Correlation Auditor for Chest X-ray Classifiers

An end-to-end auditor for deep chest X-ray classifiers that **detects**, **localizes**, **names**, and **mitigates** shortcut reliance — evaluated on a pneumothorax classifier across synthetic and real confounds, with a cross-site generalization test.

> **Full writeup:** [`docs/capstone_writeup.md`](docs/capstone_writeup.md)

---

## Overview

Deep chest X-ray classifiers can reach high in-distribution accuracy while relying on **spurious shortcuts** — features correlated with the label in training but not causally tied to the pathology. This project builds a validation-gated auditor that runs a four-stage loop:

1. **Detect** — measure reliance gap between shortcut-present and shortcut-removed groups
2. **Localize** — Grad-CAM attribution to pinpoint the confound region
3. **Name** — prompt a general-purpose VLM (no prior hint) to identify the confound in clinical language
4. **Mitigate** — Deep Feature Reweighting (DFR) last-layer retrain to recover worst-group accuracy

The study is structured as six **pre-registered, gated validation spikes (V1–V6)**, each with a pass/fail threshold and a stop-and-report point.

---

## Results

| Spike | Question | Threshold | Result |
|-------|----------|-----------|--------|
| V1 | Trains to a sane AUC? | AUC ≥ 0.75 | **PASS** — test AUC 0.845 |
| V2 | Model measurably cheats? | reliance gap > 20 pp | **PASS** — gap **44 pp** (synthetic tag, p = 0.95) |
| V3 | Grad-CAM localizes the confound? | > 80% peak-in-box | **PASS** — **100%** peak-in-box (control 0.5%) |
| V4a | VLM names a *synthetic* confound? | P ≥ 0.90, R ≥ 0.90, ctrl-halluc < 10% | **PASS** — P 0.978 / R 0.90 / halluc 0.02 |
| V4b | VLM names a *real* device confound? | P ≥ 0.80, R ≥ 0.70, ctrl-halluc < 10% | **FAIL** — halluc **0.50** → pivot |
| V5 | Mitigation recovers accuracy? | worst-group ↑, ID stable | **PASS** — worst-group 0.67 → 0.79 (3-seed mean), ID 0.96 held |
| V6 | Cross-site gap exists / is shortcut-driven? | measurable drop | Drop **0.192** (NIH → PadChest); source-DFR zero-shot **null** (+0.005) |

**Key finding:** Blind VLM naming transfers to a synthetic confound but **fails on a real device confound by fabrication** — VLMs invent devices on tube-free controls at 0.50–0.83 rate across two model families. Mitigation (DFR) is independent of this naming failure and succeeds cleanly on real data without target-distribution labels.

---

## Repository Layout

```
src/auditor/          # Core Python package
  train/              # V1 DenseNet-121 trainer
  shortcut/           # V2 synthetic-tag injection
  attribution/        # V3 Grad-CAM localization
  discovery/          # V4 VLM naming (Groq, Ollama, Colab notebooks)
  evaluation/         # Reliance-gap measurement
  mitigation/         # V5 DFR + baselines (ERM, JTT)
  analysis/           # Phase-3 decomposition, sweep, site analysis
  report/             # End-to-end auditor loop + demo GUI
  data/               # NIH CXR-14, PadChest, NEATX loaders
  common/             # Config, seeding, run logger

configs/              # One YAML per spike/run (extends base.yaml)
tests/                # 128 unit tests covering the full synthetic spine
docs/
  capstone_writeup.md # Full project writeup (CO1–CO4 + Phase 5)
  writeup/            # CO1 related work, CO2 architecture, CO3 experiments, CO4 evaluation
  paper/
    figures/          # Publication-ready figures and generator script
```

---

## Setup

> **Platform:** Windows, Python 3.9, NVIDIA RTX 3050 6 GB

```bash
python -m venv .venv
.venv\Scripts\activate

# Install CUDA-enabled PyTorch first (adjust cu version as needed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install all other dependencies:
pip install -r requirements.txt
pip install -e .
```

`torch`/`torchvision` are pinned separately in the environment and intentionally omitted from `requirements.txt` to avoid overwriting a local CUDA build. Every other dependency is pinned.

The open-weight VLM (Llama-4-Scout, Pixtral-12B, CheXagent-2-3B) used for naming runs via the **Groq free API** or Colab/Kaggle notebooks — not locally on the 6 GB GPU.

---

## Data

None of the datasets are committed. Download them and point `configs/base.yaml` at your copies:

| Dataset | Source | Used in |
|---------|--------|---------|
| NIH ChestX-ray14 (224 × 224) | [Kaggle — khanfashee/nih-chest-x-ray-14-224x224-resized](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) | V1–V5 |
| NEATX chest-tube annotations | [zenodo.org/records/14944064](https://zenodo.org/records/14944064) | V4b, V5 |
| PadChest (224 × 224 subset) | [bimcv.cipf.es/bimcv-projects/padchest](http://bimcv.cipf.es/bimcv-projects/padchest/) | V6 |

Key config keys in `configs/base.yaml`:

| Key | Points at |
|-----|-----------|
| `data.nih_root` | folder with CSVs + split files |
| `data.nih_images` | image folder (the archive double-nests as `images-224/images-224/`) |
| `data.tube_csv` | `NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv` |
| `data.site2_root` | PadChest root (V6; empty by default) |

---

## Running

Every run writes `experiments/<spike>/<timestamp>_<hash>/` with the resolved config, `metrics.json`, artifacts, a log, and the git commit hash.

### Train a spike
```bash
python -m auditor.train --config configs/v1_baseline.yaml
# replace with: v2_shortcut, v3_attribution, v4a_naming, v4b_tube, v5_dfr_tube, ...
```

### VLM naming (V4)
```bash
# 1. Export saliency crops
python -m auditor.discovery export --config configs/v4a_naming.yaml

# 2. Name via Groq API (needs GROQ_API_KEY in .env)
python -m auditor.discovery name-groq \
  --crops <crops/> --manifest <manifest.json> \
  --out predictions.jsonl --model <groq-vision-model-id>

# 3. Score (precision / recall / control-hallucination)
python -m auditor.discovery score \
  --manifest <manifest.json> --predictions predictions.jsonl --confound tag
```

### End-to-end auditor loop
```bash
python -m auditor.report \
  --config configs/auditor_tube.yaml \
  --predictions <predictions.jsonl> --manifest <manifest.json>
```

### Single-image demo
```bash
# Static figure
python -m auditor.report.demo_single_image --image path/to/xray.png --out out/demo.png

# Interactive GUI (Flask)
python -m auditor.report.demo_gui --port 5000
# → open http://127.0.0.1:5000
```

### Regenerate paper figures
```bash
python docs/paper/figures/generate_new_figures.py
# Outputs fig0–fig6 as PNG + PDF from published numbers (matplotlib only, no re-run)
```

### API keys
Copy `.env.example` to `.env` (gitignored) and fill in `GROQ_API_KEY`. For multi-VLM roster runs, also set `GEMINI_API_KEYS`, `MISTRAL_API_KEYS`, `GITHUB_TOKENS`.

---

## Tests

```bash
pytest
```

128 tests cover the full synthetic spine (V2 shortcut injection → V3 attribution → V5 mitigation), plus discovery/analysis/config units. The synthetic spine is the safety floor — it must stay green for any result to be trustworthy.

---

## License

[MIT](LICENSE)
