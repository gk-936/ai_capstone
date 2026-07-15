# Spurious-Correlation Auditor for Chest X-ray Classifiers

An auditor for chest X-ray classifiers that (1) **detects** reliance on a spurious shortcut,
(2) **localizes** it with Grad-CAM, (3) attempts to **name** it with a general-purpose VLM
given no prior hint, (4) **mitigates** the reliance, and (5) tests **cross-site** recovery.

The project ran as a sequence of gated validation spikes (V1–V6). A pre-registered gate on
the naming step (V4b) **failed** on real device confounds — off-the-shelf VLMs fabricate
devices on a majority of clean crops — triggering a documented pivot. The project's most
defensible contribution is therefore the **bounded negative result**: a measured boundary on
where prompt-free VLM shortcut-naming can be trusted, wrapped around a controlled
detect→localize→mitigate study on both synthetic and real confounds.

See [CLAUDE.md](CLAUDE.md) for the operating rules and [PROGRESS.md](PROGRESS.md) for
spike-by-spike gate status.

## Status
**Core validation gauntlet (V1–V6) complete; extension experiments in progress.** All six
spikes were built, run, measured against a threshold, and recorded. Headline results:

| Spike | Question | Result |
|---|---|---|
| V1 | Trains to a sane AUC? | **PASS** — test AUC 0.845 |
| V2 | Model measurably cheats? | **PASS** — reliance gap 44pp (synthetic tag) |
| V3 | Grad-CAM localizes the confound? | **PASS** — 100% peak-in-box (control 0.5%) |
| V4a | VLM names a synthetic confound? | **PASS** — P 0.98 / R 0.90 / halluc 0.02 |
| V4b | VLM names a real device? | **FAIL** — halluc 0.50 → **pivot** |
| V5 | Mitigation recovers accuracy? | **PASS** (real tube) — worst-group 0.67→0.83, ID preserved |
| V6 | Cross-site gap exists / is shortcut-driven? | Gap **exists** (0.192); shortcut-driven **not supported** (zero-shot null +0.005) |

The V4b failure was replicated across two independent VLM families (Scout 55%, Pixtral 83%
control-hallucination). Extension experiments (naming-difficulty curve, image-independence,
specialist VLM) are underway; see `docs/writeup/`. Writeups CO1–CO4 are in `docs/writeup/`.

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
