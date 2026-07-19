# Reproducibility Statement

This project is engineered so every headline number can be traced from a fixed seed and a
versioned config to a logged metric on disk. The mechanisms below are load-bearing, not
aspirational; the file references let a reviewer verify each claim directly.

## Deterministic seeding
Every run seeds Python, NumPy, and (when installed) PyTorch/CUDA through a single entry
point, `set_seed`, in `src/auditor/common/seed.py`. It sets `PYTHONHASHSEED`, seeds
`random` and `numpy`, calls `torch.manual_seed` / `torch.cuda.manual_seed_all`, and (by
default) forces `cudnn.deterministic = True` with `cudnn.benchmark = False`. The default
seed is `1337`, declared in the config (`Config.seed`, `src/auditor/common/config.py`), and
multi-seed experiments (the ERM/JTT/DFR baseline table) vary it explicitly and report
mean ± std rather than a single draw.

## Config-driven runs; dataset paths are config values, never hardcoded
Runs are driven by typed YAML configs (`src/auditor/common/config.py`), parsed into a
pydantic `Config` model. Configs support `extends:` inheritance (spike configs extend
`configs/base.yaml`, deep-merged) and repeatable `--override key.subkey=value` CLI flags,
attached uniformly via `add_config_args`. All dataset roots — NIH, NEATX tube annotations,
and the PadChest second site — are fields on `DataConfig` (`nih_root`, `nih_images`,
`labels_csv`, `tube_csv`, `site2_root`, `padchest_labels_csv`, …), resolved at load time.
No dataset path is hardcoded in source; `configs/base.yaml` holds the NIH paths and a second
site plugs in through the same config surface with no code change. Paths resolve against the
repo root via `resolve_path` (`src/auditor/common/paths.py`), which also handles Windows
paths containing spaces, and the compute device degrades gracefully from CUDA to CPU via
`resolve_device` so CPU-only replication works.

## Every run logs config + metrics + artifacts + code provenance
`RunLogger` (`src/auditor/common/run_logger.py`) creates one timestamped directory per run,
`experiments/<spike>/<timestamp>_<shorthash>/`, containing:

- `config.yaml` — the fully resolved config actually used (post-merge, post-override);
- `metrics.json` — the measured numbers, rewritten after every `log_metrics` call, which
  are exactly the values the pass/fail gates are checked against;
- `artifacts/` — checkpoints, saliency crops, heatmaps, and figures;
- `run.log` — the console log;
- `git.txt` — the commit hash and a dirty-tree flag, mapping each result to exact code.

This makes each result self-describing: config, metric, artifact, and the commit that
produced it live together. The `experiments/` tree is gitignored (run outputs are not
committed), so reproduction regenerates it rather than reading committed numbers.

## Test suite
The synthetic spine and the module layer are covered by a pytest suite that must stay green
so the pipeline cannot silently break. Verified count on the pinned environment
(`PYTHONPATH=src python -m pytest -q`, run from the repo root): **128 tests pass**. This
matches the current PROGRESS.md figure; earlier docs citing 69/99/132 are superseded.

## What "reproducible" covers
The headline metrics are recorded per-run in `metrics.json` and consolidated, with their
measured values and two documented post-review corrections, in `docs/writeup/CO3_experiments.md`
and the PROGRESS.md gate table. The pipeline is clean-tree reproducible from config + seed:
V1 in-distribution AUC 0.845, the V2 reliance gap (44 pp synthetic tag), V3 saliency
peak-in-box (100% vs 0.5% control), the V4a/V4b naming precision/recall/hallucination
numbers, V5 real-tube DFR mitigation (worst-group 0.67→0.79 three-seed mean / 0.83
single-draw, ID preserved), and the V6 cross-site drop (NIH 0.842 → PadChest 0.650). Numbers
requiring the external NIH ChestX-ray14, NEATX, and PadChest datasets reproduce once those
are downloaded and their paths set in config; the datasets themselves are not redistributed
here (see below). Metrics not measured are reported as such and never fabricated, per the
project's validation-first discipline.

## Not committed (and why the repo still runs)
Dataset images, CSVs, and model checkpoints are never committed: `.gitignore` anchors
`/data/` and `/experiments/` to the repo root and ignores `*.pth`/`*.pt`/`*.ckpt`, `crops/`,
and `predictions.jsonl`. Secrets live in a gitignored `.env` (template: `.env.example`); no
API keys are committed. All Python source under `src/auditor/` — including the entire data
layer (`src/auditor/data/*.py`) — is tracked, so a fresh clone runs once the external
datasets are downloaded and pointed at via config.
