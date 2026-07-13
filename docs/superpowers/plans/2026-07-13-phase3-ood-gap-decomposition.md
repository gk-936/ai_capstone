# Phase 3 — OOD-Gap Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 0.192 NIH→PadChest OOD AUC gap into shortcut (A), label-mismatch (B-lite), and covariate-shift (C) slices, each with a falsifying control, on the frozen V1 DenseNet121.

**Architecture:** A new `src/auditor/analysis/` package with one module per experiment plus a combiner, reusing the existing frozen-feature/AUC/DFR machinery (`auditor.mitigation.dfr`, `auditor.report.run_v6`, `auditor.train.trainer`). Pure functions are torch-free and unit-tested; model-touching orchestration is thin, config-driven, and validated at run time.

**Tech Stack:** Python 3.9, numpy, pandas, scikit-learn, torch, timm (all already in the repo — no new deps).

## Global Constraints

- Python 3.9: every new module starts with `from __future__ import annotations`.
- **No new dependencies.** Only numpy/pandas/scikit-learn/torch/timm.
- **Unit tests must not import torch at module top** (the dev machine's torch import is fragile). Torch-dependent tests use `torch = pytest.importorskip("torch")` inside the test.
- Fixed seeds via `cfg.seed`; dataset paths come from config (`configs/v6_padchest.yaml`), never hardcoded.
- All experiments operate on the **frozen** V1 checkpoint `experiments/v1_baseline/20260707_232704_9fbb04/artifacts/model_best.pth`; no backbone retraining.
- Every orchestration writes artifacts via `RunLogger` (`logger.run_dir`, `logger.log_metrics`).
- Commit at the end of every task with `git -c gc.auto=0 commit` (the dev machine's `gc.auto` is disabled to avoid mmap failures).
- Report every result whichever way its control falls (the standing anti-trap rule).

---

### Task 1: Experiment A — site-probe pure functions

**Files:**
- Create: `src/auditor/analysis/__init__.py` (empty)
- Create: `src/auditor/analysis/site_probe.py`
- Test: `tests/test_analysis_site.py`

**Interfaces:**
- Produces:
  - `project_out(features: np.ndarray, direction: np.ndarray) -> np.ndarray` — remove the component of each row along `direction`.
  - `site_direction(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray` — unit vector from mean(b) to mean(a).
  - `score_features(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray` — linear head scores.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_site.py
import numpy as np
from auditor.analysis.site_probe import project_out, site_direction, score_features


def test_project_out_removes_component_and_keeps_shape():
    feats = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    d = np.array([1.0, 0.0, 0.0])
    out = project_out(feats, d)
    assert out.shape == feats.shape
    assert np.allclose(out @ (d / np.linalg.norm(d)), 0.0, atol=1e-6)  # orthogonal to d
    assert np.allclose(out[:, 1:], feats[:, 1:])                       # other dims untouched


def test_site_direction_points_from_b_to_a_and_is_unit():
    a = np.array([[10.0, 0.0], [12.0, 0.0]])   # mean (11, 0)
    b = np.array([[0.0, 0.0], [2.0, 0.0]])     # mean (1, 0)
    d = site_direction(a, b)
    assert np.allclose(np.linalg.norm(d), 1.0)
    assert d[0] > 0.99                          # points along +axis0


def test_score_features_is_linear_head():
    feats = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([1.0, -1.0])
    assert np.allclose(score_features(feats, w, 0.5), np.array([0.5 - 1.0, 0.5 - 1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_site.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auditor.analysis'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/auditor/analysis/site_probe.py
"""Experiment A — hospital-detector probe + site-direction projection (pure, torch-free).

A1 fits a probe to classify NIH vs PadChest on frozen features (site is decodable — expected).
A2 is the falsifying control: project the site direction out of the disease-test features and
re-score the FROZEN disease head. If OOD AUC does not move, the disease head is not riding the
site shortcut -> bounds the shortcut slice near zero.
"""

from __future__ import annotations

import numpy as np


def project_out(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return `features` with the component along `direction` removed (row-wise)."""
    u = direction / (np.linalg.norm(direction) + 1e-12)
    return features - np.outer(features @ u, u)


def site_direction(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """Unit vector pointing from mean(feats_b) to mean(feats_a) (a simple, deterministic axis)."""
    d = feats_a.mean(axis=0) - feats_b.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def score_features(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """Linear-head scores in raw feature space."""
    return features @ weight + bias
```

Also create the empty package file:

```python
# src/auditor/analysis/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis_site.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/auditor/analysis/__init__.py src/auditor/analysis/site_probe.py tests/test_analysis_site.py
git -c gc.auto=0 commit -m "feat(analysis): site-probe pure functions (project_out/site_direction/score)"
```

---

### Task 2: Experiment A — orchestration + `site` CLI + Phase-3 config

**Files:**
- Create: `src/auditor/analysis/run_site.py`
- Create: `src/auditor/analysis/__main__.py`
- Create: `configs/phase3_analysis.yaml`
- Test: `tests/test_analysis_head.py`

**Interfaces:**
- Consumes: `project_out`, `site_direction`, `score_features` (Task 1); `auditor.mitigation.dfr.extract_features`, `fit_dfr_head`; `auditor.report.run_v6._padchest_loader`; `auditor.data.padchest.build_padchest_index`; `auditor.train.trainer._build_model`.
- Produces:
  - `disease_head_arrays(model) -> tuple[np.ndarray, float]` — the frozen classifier's (weight, bias) as numpy.
  - `run_site_analysis(cfg, logger, checkpoint, labels_csv, images_dir) -> dict` — writes `site_analysis.json`; keys: `site_probe_auc`, `padchest_disease_auc_before`, `padchest_disease_auc_after_projection`, `projection_change`, `n_nih`, `n_padchest`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_head.py
import numpy as np
import pytest


def test_disease_head_arrays_extracts_weight_and_bias():
    torch = pytest.importorskip("torch")
    from auditor.analysis.run_site import disease_head_arrays

    class Dummy:
        pass
    m = Dummy()
    m.classifier = torch.nn.Linear(4, 1)
    with torch.no_grad():
        m.classifier.weight.copy_(torch.arange(4, dtype=torch.float32).view(1, 4))
        m.classifier.bias.copy_(torch.tensor([0.5]))
    w, b = disease_head_arrays(m)
    assert w.shape == (4,)
    assert np.allclose(w, [0, 1, 2, 3])
    assert abs(b - 0.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auditor.analysis.run_site'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/auditor/analysis/run_site.py
"""Experiment A orchestration: hospital-detector probe + site-projection control.

Fits a probe on frozen features (NIH vs PadChest) and reports its AUC (A1). Then projects the
site direction out of the PadChest disease-test features and re-scores the FROZEN disease head,
reporting the OOD-AUC change (A2, the falsifying control). Reuses the V6 loaders and the DFR
feature/head machinery so numbers stay comparable to V6.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from auditor.analysis.site_probe import project_out, score_features
from auditor.common.config import Config
from auditor.common.paths import resolve_device, resolve_path
from auditor.common.run_logger import RunLogger
from auditor.common.seed import set_seed
from auditor.data.padchest import build_padchest_index
from auditor.mitigation.dfr import extract_features, fit_dfr_head
from auditor.report.run_v6 import _nih_reference_auc, _padchest_loader
from auditor.train.trainer import _build_model


def disease_head_arrays(model) -> tuple[np.ndarray, float]:
    """The frozen classifier head's (weight[D], bias) as numpy, in raw feature space."""
    w = model.classifier.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.classifier.bias.detach().cpu().numpy().reshape(-1)[0])
    return w, b


def _nih_feature_loader(cfg, batch_size):
    """NIH test loader (same 8-bit path as _nih_reference_auc) for feature extraction."""
    from torch.utils.data import DataLoader
    from auditor.data.nih import build_index
    from auditor.data.torch_dataset import NIHDataset

    idx = build_index(cfg.data, target=cfg.train.target_pathology, seed=cfg.seed,
                      max_per_split={"test": 4000})
    ds = NIHDataset(rows=idx.subset("test"), images_dir=resolve_path(cfg.data.nih_images),
                    image_size=cfg.data.image_size, train=False)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def run_site_analysis(cfg: Config, logger: RunLogger, checkpoint: str,
                      labels_csv: str, images_dir: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Experiment A (site probe + projection control): checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    bs = cfg.train.batch_size
    pc_index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed, dfr_frac=0.0,
                                    max_rows=cfg.data_padchest_max_rows or None)

    # Frozen features: PadChest test (disease labels) + NIH test.
    pc_loader = _padchest_loader(pc_index, "test", images_dir, cfg.data.image_size, bs)
    pc_feats, pc_labels = extract_features(model, pc_loader, device)
    nih_feats, _ = extract_features(model, _nih_feature_loader(cfg, bs), device)

    # A1: hospital-detector probe (site = NIH:0 vs PadChest:1) with a held-out split for honest AUC.
    rng = np.random.default_rng(cfg.seed)
    X = np.concatenate([nih_feats, pc_feats])
    y = np.concatenate([np.zeros(len(nih_feats)), np.ones(len(pc_feats))]).astype(int)
    perm = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = perm[:cut], perm[cut:]
    w_site, b_site = fit_dfr_head(X[tr], y[tr], seed=cfg.seed)
    site_scores = score_features(X[te], w_site, b_site)
    site_auc = float(roc_auc_score(y[te], site_scores))
    logger.info(f"A1 hospital-detector probe AUC: {site_auc:.4f}")

    # A2: project the site direction out of PadChest disease-test features, re-score disease head.
    w_dis, b_dis = disease_head_arrays(model)
    before = float(roc_auc_score(pc_labels, score_features(pc_feats, w_dis, b_dis)))
    pc_proj = project_out(pc_feats, w_site)          # remove the probe's site axis
    after = float(roc_auc_score(pc_labels, score_features(pc_proj, w_dis, b_dis)))
    logger.info(f"A2 PadChest disease AUC before/after site-projection: {before:.4f} -> {after:.4f}")

    metrics = {
        "site_probe_auc": round(site_auc, 4),
        "padchest_disease_auc_before": round(before, 4),
        "padchest_disease_auc_after_projection": round(after, 4),
        "projection_change": round(after - before, 4),
        "n_nih": int(len(nih_feats)), "n_padchest": int(len(pc_feats)),
    }
    (logger.run_dir / "site_analysis.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.log_metrics(**metrics)
    logger.info(f"EXPERIMENT A RESULT: {metrics}")
    return metrics
```

```python
# src/auditor/analysis/__main__.py
"""CLI for Phase 3 OOD-gap decomposition: site / labels / tta / decompose."""

from __future__ import annotations

import argparse
import json


def _cmd_site(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.analysis.run_site import run_site_analysis

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger("phase3_site", cfg, experiments_root=cfg.experiments_root)
    m = run_site_analysis(cfg, logger, args.checkpoint, args.labels_csv, args.images_dir)
    print(json.dumps(m, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: OOD-gap decomposition")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("site", help="Experiment A: hospital-detector probe + projection control.")
    ps.add_argument("--config", required=True)
    ps.add_argument("--checkpoint", required=True)
    ps.add_argument("--labels-csv", required=True, dest="labels_csv")
    ps.add_argument("--images-dir", required=True, dest="images_dir")
    ps.add_argument("--override", action="append", default=[])
    ps.set_defaults(func=_cmd_site)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

```yaml
# configs/phase3_analysis.yaml
# Phase 3 OOD-gap decomposition. Reuses the V6 PadChest paths + V1 checkpoint.
extends: v6_padchest.yaml
spike: phase3_analysis

analysis:
  tent_steps: 1        # TENT passes over the target set
  tent_lr: 0.001       # Adam LR for BN affine params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis_head.py -v`
Expected: PASS (1 passed; or skipped if torch unavailable)

- [ ] **Step 5: Commit**

```bash
git add src/auditor/analysis/run_site.py src/auditor/analysis/__main__.py configs/phase3_analysis.yaml tests/test_analysis_head.py
git -c gc.auto=0 commit -m "feat(analysis): Experiment A orchestration + site CLI + phase3 config"
```

**Run-time validation (when machine healthy, not part of the test cycle):**
`python -m auditor.analysis site --config configs/phase3_analysis.yaml --checkpoint experiments/v1_baseline/20260707_232704_9fbb04/artifacts/model_best.pth --labels-csv data/padchest_v6_subset.csv --images-dir data/padchest_v6/images-224`

---

### Task 3: Experiment B-lite — label-composition analysis

**Files:**
- Create: `src/auditor/analysis/label_analysis.py`
- Modify: `src/auditor/analysis/__main__.py` (add `labels` subcommand)
- Test: `tests/test_analysis_labels.py`

**Interfaces:**
- Produces:
  - `parse_labels(cell: str) -> list[str]` — split a PadChest `Labels` cell into normalized tokens.
  - `ptx_label_composition(df: pd.DataFrame, labels_col="Labels", ptx_col="ptx") -> dict` — among ptx-positive rows: total, fraction whose only finding is pneumothorax, and the top co-findings.
  - `run_label_analysis(labels_csv: str, out_path: str) -> dict` — read CSV, compute composition, write JSON + return it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_labels.py
import pandas as pd
from auditor.analysis.label_analysis import parse_labels, ptx_label_composition


def test_parse_labels_splits_and_normalizes():
    assert parse_labels("['pneumothorax', 'chest drain tubes']") == ["pneumothorax", "chest drain tubes"]
    assert parse_labels("pneumothorax") == ["pneumothorax"]
    assert parse_labels("") == []
    assert parse_labels(None) == []


def test_ptx_composition_counts_isolated_and_cofindings():
    df = pd.DataFrame({
        "Labels": ["['pneumothorax']",
                   "['pneumothorax', 'chest drain tubes']",
                   "['pneumothorax', 'chest drain tubes']",
                   "['normal']"],
        "ptx": [1, 1, 1, 0],
    })
    comp = ptx_label_composition(df)
    assert comp["n_ptx_positive"] == 3
    assert comp["frac_ptx_isolated"] == round(1 / 3, 4)     # only 1 of 3 is ptx-only
    assert comp["top_cofindings"][0][0] == "chest drain tubes"   # most common co-finding
    assert comp["top_cofindings"][0][1] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auditor.analysis.label_analysis'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/auditor/analysis/label_analysis.py
"""Experiment B-lite — PadChest ptx label composition from the subset CSV's `Labels` column.

The working subset CSV has no MethodLabel/Projection, so this bounds the label-mismatch slice
qualitatively: how often PadChest's pneumothorax cases carry a co-finding (e.g. chest drains)
that NIH labels separately, which can inflate the cross-site gap. Pure pandas; no model.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd


def parse_labels(cell) -> list[str]:
    """Split a PadChest `Labels` cell (a stringified list) into normalized lowercase tokens."""
    if cell is None or (isinstance(cell, float)) or str(cell).strip() == "":
        return []
    text = str(cell).strip().strip("[]")
    if not text:
        return []
    parts = [p.strip().strip("'\"").lower() for p in text.split(",")]
    return [p for p in parts if p]


def ptx_label_composition(df: pd.DataFrame, labels_col: str = "Labels",
                          ptx_col: str = "ptx") -> dict:
    """Composition of the pneumothorax-positive rows: isolation fraction + top co-findings."""
    pos = df[df[ptx_col] == 1]
    n = len(pos)
    isolated = 0
    cofindings: Counter = Counter()
    for cell in pos[labels_col]:
        tokens = [t for t in parse_labels(cell) if t]
        others = [t for t in tokens if "pneumothorax" not in t]
        if not others:
            isolated += 1
        cofindings.update(others)
    return {
        "n_ptx_positive": int(n),
        "frac_ptx_isolated": round(isolated / n, 4) if n else 0.0,
        "top_cofindings": cofindings.most_common(10),
    }


def run_label_analysis(labels_csv: str, out_path: str) -> dict:
    df = pd.read_csv(labels_csv, low_memory=False)
    comp = ptx_label_composition(df)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)
    return comp
```

Add to `src/auditor/analysis/__main__.py` (a new handler + subparser):

```python
def _cmd_labels(args) -> None:
    from auditor.analysis.label_analysis import run_label_analysis

    comp = run_label_analysis(args.labels_csv, args.out)
    print(json.dumps(comp, indent=2))
```

and inside `main()`, after the `site` subparser:

```python
    pl = sub.add_parser("labels", help="Experiment B-lite: PadChest ptx label composition.")
    pl.add_argument("--labels-csv", required=True, dest="labels_csv")
    pl.add_argument("--out", required=True, help="Output JSON path.")
    pl.set_defaults(func=_cmd_labels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/auditor/analysis/label_analysis.py src/auditor/analysis/__main__.py tests/test_analysis_labels.py
git -c gc.auto=0 commit -m "feat(analysis): Experiment B-lite label-composition + labels CLI"
```

---

### Task 4: Experiment C — label-free TTA (BN-adapt + TENT)

**Files:**
- Create: `src/auditor/analysis/tta.py`
- Modify: `src/auditor/analysis/__main__.py` (add `tta` subcommand)
- Test: `tests/test_analysis_tta.py`

**Interfaces:**
- Consumes: `auditor.report.run_v6._padchest_loader`, `auditor.data.padchest.build_padchest_index`, `auditor.train.trainer._build_model`, `auditor.train.trainer.evaluate`.
- Produces:
  - `collect_bn_params(model) -> list` — the BN affine params (γ/β).
  - `configure_bn_adapt(model)` / `bn_adapt(model, loader, device) -> model`.
  - `configure_tent(model)` / `tent_adapt(model, loader, device, steps=1, lr=1e-3) -> model`.
  - `run_tta(cfg, logger, checkpoint, labels_csv, images_dir, method) -> dict` — writes `tta_<method>.json`; keys: `method`, `padchest_auc_before`, `padchest_auc_after`, `tta_gain`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_tta.py
import pytest


def _toy_model(torch):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1))


def test_bn_adapt_updates_running_stats():
    torch = pytest.importorskip("torch")
    from auditor.analysis.tta import bn_adapt

    model = _toy_model(torch)
    bn = model[1]
    before = bn.running_mean.clone()
    x = torch.randn(32, 4) * 5 + 3  # shifted target distribution
    loader = [(x, torch.zeros(32))]
    bn_adapt(model, loader, device="cpu")
    assert not torch.allclose(before, bn.running_mean)  # stats moved toward target


def test_tent_updates_only_bn_affine():
    torch = pytest.importorskip("torch")
    from auditor.analysis.tta import tent_adapt

    model = _toy_model(torch)
    lin0_w = model[0].weight.detach().clone()
    bn_w = model[1].weight.detach().clone()
    x = torch.randn(32, 4)
    loader = [(x, torch.zeros(32))]
    tent_adapt(model, loader, device="cpu", steps=1, lr=0.1)
    assert torch.allclose(lin0_w, model[0].weight)      # backbone weights frozen
    assert not torch.allclose(bn_w, model[1].weight)    # BN affine updated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_tta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auditor.analysis.tta'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/auditor/analysis/tta.py
"""Experiment C — label-free test-time adaptation: BN-adapt (AdaBN) + TENT.

BN-adapt re-estimates BatchNorm running statistics on the target images (forward-only, no
labels). TENT additionally entropy-minimizes over the BN affine params (light backprop, no
labels). A gain here that source-DFR could not achieve = recoverable covariate shift, not a
source-identifiable shortcut. Reported whichever way it falls.
"""

from __future__ import annotations

import json


def _is_bn(m) -> bool:
    import torch.nn as nn
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def collect_bn_params(model) -> list:
    """The BN affine parameters (weight=γ, bias=β) across the model."""
    params = []
    for m in model.modules():
        if _is_bn(m):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def configure_bn_adapt(model):
    """Put BN layers in cumulative-average train mode with reset stats; everything else eval."""
    model.eval()
    for m in model.modules():
        if _is_bn(m):
            m.train()
            m.momentum = None          # cumulative moving average over the target set
            m.reset_running_stats()
    return model


def bn_adapt(model, loader, device):
    """Forward-only BN statistic re-estimation on the target loader (no labels, no grad)."""
    import torch

    configure_bn_adapt(model)
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
    model.eval()
    return model


def configure_tent(model):
    """Freeze all params except BN affine; BN in train mode with running stats disabled."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if _is_bn(m):
            m.train()
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
    return model


def tent_adapt(model, loader, device, steps: int = 1, lr: float = 1e-3):
    """Entropy-minimize the sigmoid output over BN affine params (no labels)."""
    import torch

    configure_tent(model)
    opt = torch.optim.Adam(collect_bn_params(model), lr=lr)
    for _ in range(steps):
        for x, _ in loader:
            out = model(x.to(device))
            p = torch.sigmoid(out)
            ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
            opt.zero_grad()
            ent.backward()
            opt.step()
    model.eval()
    return model


def run_tta(cfg, logger, checkpoint: str, labels_csv: str, images_dir: str,
            method: str) -> dict:
    """method in {'bn', 'tent'}: adapt on the PadChest test images, re-measure OOD AUC."""
    import torch
    from auditor.common.paths import resolve_device, resolve_path
    from auditor.common.seed import set_seed
    from auditor.data.padchest import build_padchest_index
    from auditor.report.run_v6 import _padchest_loader
    from auditor.train.trainer import _build_model, evaluate

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Experiment C TTA ({method}): checkpoint={checkpoint}")

    model = _build_model(cfg.train.backbone)
    model.load_state_dict(torch.load(resolve_path(checkpoint), map_location=device))
    model = model.to(device).eval()

    index = build_padchest_index(cfg.data, labels_csv, seed=cfg.seed, dfr_frac=0.0,
                                 max_rows=cfg.data_padchest_max_rows or None)
    loader = _padchest_loader(index, "test", images_dir, cfg.data.image_size, cfg.train.batch_size)

    before = evaluate(model, loader, device)
    if method == "bn":
        bn_adapt(model, loader, device)
    elif method == "tent":
        tent_adapt(model, loader, device, steps=cfg.analysis.tent_steps, lr=cfg.analysis.tent_lr)
    else:
        raise ValueError(f"unknown TTA method: {method!r} (expected 'bn' or 'tent')")
    after = evaluate(model, loader, device)

    metrics = {
        "method": method,
        "padchest_auc_before": round(float(before), 4),
        "padchest_auc_after": round(float(after), 4),
        "tta_gain": round(float(after) - float(before), 4),
    }
    (logger.run_dir / f"tta_{method}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.log_metrics(**metrics)
    logger.info(f"EXPERIMENT C ({method}) RESULT: {metrics}")
    return metrics
```

Add to `src/auditor/analysis/__main__.py`:

```python
def _cmd_tta(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.analysis.tta import run_tta

    cfg = load_config(args.config, overrides=args.override)
    methods = ["bn", "tent"] if args.method == "both" else [args.method]
    for method in methods:
        logger = RunLogger(f"phase3_tta_{method}", cfg, experiments_root=cfg.experiments_root)
        m = run_tta(cfg, logger, args.checkpoint, args.labels_csv, args.images_dir, method)
        print(json.dumps(m, indent=2))
```

and inside `main()`:

```python
    pt = sub.add_parser("tta", help="Experiment C: label-free TTA (BN-adapt / TENT).")
    pt.add_argument("--config", required=True)
    pt.add_argument("--checkpoint", required=True)
    pt.add_argument("--labels-csv", required=True, dest="labels_csv")
    pt.add_argument("--images-dir", required=True, dest="images_dir")
    pt.add_argument("--method", choices=["bn", "tent", "both"], default="both")
    pt.add_argument("--override", action="append", default=[])
    pt.set_defaults(func=_cmd_tta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis_tta.py -v`
Expected: PASS (2 passed; or skipped if torch unavailable)

- [ ] **Step 5: Commit**

```bash
git add src/auditor/analysis/tta.py src/auditor/analysis/__main__.py tests/test_analysis_tta.py
git -c gc.auto=0 commit -m "feat(analysis): Experiment C label-free TTA (BN-adapt + TENT) + tta CLI"
```

---

### Task 5: Combine — decomposition table + `decompose` CLI

**Files:**
- Create: `src/auditor/analysis/decompose.py`
- Modify: `src/auditor/analysis/__main__.py` (add `decompose` subcommand)
- Test: `tests/test_analysis_decompose.py`

**Interfaces:**
- Consumes: the JSON outputs of Tasks 2–4 (`site_analysis.json`, label composition JSON, `tta_bn.json`, `tta_tent.json`).
- Produces:
  - `build_decomposition(ood_drop, site_projection_change, tta_gains, label_note) -> tuple[dict, str]` — the slice dict + a markdown block. `tta_gains` is `{method: gain}`.
  - `run_decompose(site_json, tta_jsons, label_json, ood_drop, out_dir) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_decompose.py
from auditor.analysis.decompose import build_decomposition


def test_build_decomposition_picks_best_tta_and_bounds_shortcut():
    d, md = build_decomposition(
        ood_drop=0.192,
        site_projection_change=-0.002,           # projecting site out barely moves AUC -> ~0 shortcut
        tta_gains={"bn": 0.03, "tent": 0.05},
        label_note="ptx isolated in 40% of cases; top co-finding chest drains",
    )
    assert d["ood_drop"] == 0.192
    assert d["covariate_recoverable_gain"] == 0.05          # best of bn/tent
    assert d["covariate_recoverable_method"] == "tent"
    assert d["shortcut_projection_change"] == -0.002
    assert "not a clean additive partition" in md.lower()   # honesty caveat present
    assert "0.192" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_decompose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auditor.analysis.decompose'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/auditor/analysis/decompose.py
"""Combine Experiments A/B/C into the honest OOD-gap decomposition.

Slices are estimated INDEPENDENTLY and may overlap — this is deliberately NOT presented as a
clean additive partition, and the markdown says so.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_decomposition(ood_drop: float, site_projection_change: float,
                        tta_gains: dict, label_note: str) -> tuple[dict, str]:
    best_method = max(tta_gains, key=tta_gains.get) if tta_gains else None
    best_gain = round(float(tta_gains[best_method]), 4) if best_method else 0.0
    d = {
        "ood_drop": round(float(ood_drop), 4),
        "covariate_recoverable_gain": best_gain,
        "covariate_recoverable_method": best_method,
        "shortcut_projection_change": round(float(site_projection_change), 4),
        "label_mismatch_note": label_note,
    }
    md = "\n".join([
        "# Phase 3 — NIH→PadChest OOD-gap decomposition",
        "",
        f"- **Total OOD drop:** {d['ood_drop']:.3f}",
        f"- **C. Covariate-recoverable (label-free TTA, best = {best_method}):** "
        f"{best_gain:+.3f} AUC",
        f"- **A. Shortcut residual (AUC change when site is projected out):** "
        f"{d['shortcut_projection_change']:+.3f} — ≈0 means the disease head does not ride site",
        f"- **B. Label mismatch (B-lite, qualitative):** {label_note}",
        "",
        "**Caveat:** these slices are estimated independently and may overlap. This is NOT a "
        "clean additive partition of the 0.192 gap; it bounds each candidate cause with its own "
        "control and reports each whichever way it fell.",
    ])
    return d, md


def run_decompose(site_json: str, tta_jsons: list[str], label_json: str,
                  ood_drop: float, out_dir: str) -> dict:
    site = json.loads(Path(site_json).read_text(encoding="utf-8"))
    gains = {}
    for p in tta_jsons:
        t = json.loads(Path(p).read_text(encoding="utf-8"))
        gains[t["method"]] = t["tta_gain"]
    label = json.loads(Path(label_json).read_text(encoding="utf-8"))
    note = (f"ptx isolated in {label['frac_ptx_isolated']*100:.0f}% of PadChest cases; "
            f"top co-findings {label['top_cofindings'][:3]}")
    d, md = build_decomposition(ood_drop, site["projection_change"], gains, note)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomposition.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    (out / "decomposition.md").write_text(md, encoding="utf-8")
    print(md)
    return d
```

Add to `src/auditor/analysis/__main__.py`:

```python
def _cmd_decompose(args) -> None:
    from auditor.analysis.decompose import run_decompose

    run_decompose(args.site_json, args.tta_json, args.label_json, args.ood_drop, args.out_dir)
```

and inside `main()`:

```python
    pd_ = sub.add_parser("decompose", help="Combine A/B/C into the decomposition table.")
    pd_.add_argument("--site-json", required=True, dest="site_json")
    pd_.add_argument("--tta-json", action="append", default=[], dest="tta_json",
                     help="Repeatable: one per TTA method JSON.")
    pd_.add_argument("--label-json", required=True, dest="label_json")
    pd_.add_argument("--ood-drop", type=float, default=0.192, dest="ood_drop")
    pd_.add_argument("--out-dir", required=True, dest="out_dir")
    pd_.set_defaults(func=_cmd_decompose)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis_decompose.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/auditor/analysis/decompose.py src/auditor/analysis/__main__.py tests/test_analysis_decompose.py
git -c gc.auto=0 commit -m "feat(analysis): OOD-gap decomposition combiner + decompose CLI"
```

---

## Final integration check (after Task 5)

- [ ] Run all Phase-3 unit tests together (torch tests skip cleanly if torch is unavailable):

Run: `python -m pytest tests/test_analysis_site.py tests/test_analysis_head.py tests/test_analysis_labels.py tests/test_analysis_tta.py tests/test_analysis_decompose.py -v`
Expected: all pass or skip; zero failures.

- [ ] Confirm the CLI wires all four subcommands:

Run: `python -m auditor.analysis --help`
Expected: `site`, `labels`, `tta`, `decompose` all listed.

**Live runs (deferred to machine recovery — NOT part of this plan's test cycle):** run `site`, `labels`, `tta --method both`, then `decompose` with the produced JSONs, manually sanity-check each number against its control, and only then fold the decomposition into `docs/writeup/CO4_evaluation.md`.

## Self-Review notes

- **Spec coverage:** A (§4)→Tasks 1–2; B-lite (§5)→Task 3; C both methods (§6)→Task 4; combination + honesty caveat (§7)→Task 5; testing (§8)→each task's test + final check. B-full and CANDID are explicitly out of scope (spec §9).
- **Type consistency:** `projection_change` (run_site) == `site["projection_change"]` (run_decompose); `tta_gain`/`method` (run_tta) == keys read in run_decompose; `frac_ptx_isolated`/`top_cofindings` (label_analysis) == keys read in run_decompose.
- **No placeholders:** every code step is complete and runnable.
