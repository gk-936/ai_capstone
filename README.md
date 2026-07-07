# Spurious-Correlation Auditor for Chest X-ray Classifiers

An auditor that (1) detects when a chest X-ray classifier relies on a spurious shortcut,
(2) uses a general-purpose VLM to **name** that shortcut in clinical terms without being told
what to look for, (3) mitigates the reliance, and (4) proves recovery on out-of-distribution
data. See [CLAUDE.md](CLAUDE.md) for the operating rules and [PROGRESS.md](PROGRESS.md) for
gate status.

## Status
Phase 0 (foundations). No modelling code yet — the project advances through gated validation
spikes V1–V6; each is built, run, measured against a threshold, and stopped for go/no-go.

## Setup (Windows, RTX 3050 6GB)
```bash
python -m venv .venv
.venv\Scripts\activate
# Install a CUDA torch build first, then the rest:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Data
NIH ChestX-ray14 is **not** committed. Download it, then point `data.nih_root` in
`configs/base.yaml` at the folder. Nothing under `data/` or `experiments/` is tracked by git.

## Running a spike
```bash
python -m auditor.train --config configs/v1_baseline.yaml
```
Every run writes `experiments/<spike>/<timestamp>_<hash>/` with the resolved config,
`metrics.json`, artifacts, a log, and the git commit hash for reproducibility.

## V4 (VLM naming) boundary
Local code exports saliency crops + `manifest.json`; the self-contained notebook in
`src/auditor/discovery/colab/` runs an open-weight VLM on Colab/Kaggle and writes
`predictions.jsonl`; local code scores precision / recall / control-hallucination. The
boundary is files (upgradeable to a live endpoint later).

## Tests
```bash
pytest
```
