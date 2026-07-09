"""Sweep resumability + plot tests (no GPU, isolated temp dirs — never touch live experiments)."""

import json

from auditor.common.config import Config
from auditor.mitigation.plot_sweep import plot
from auditor.mitigation.run_sweep import _find_completed_run


def _fake_run(root, spike, gap=None):
    rd = root / spike / "20260101_x" / "artifacts"
    rd.mkdir(parents=True)
    (rd / "model_best.pth").write_text("x")
    metrics = {"gap": gap} if gap is not None else {"other": 1}
    (root / spike / "20260101_x" / "metrics.json").write_text(json.dumps(metrics))


def test_find_completed_absent(tmp_path):
    cfg = Config(experiments_root=str(tmp_path))
    assert _find_completed_run(cfg, "sweep_p80") is None


def test_find_completed_present(tmp_path):
    _fake_run(tmp_path, "sweep_p80", gap=0.21)
    cfg = Config(experiments_root=str(tmp_path))
    res = _find_completed_run(cfg, "sweep_p80")
    assert res is not None and res[1] == 0.21


def test_find_completed_incomplete_is_none(tmp_path):
    _fake_run(tmp_path, "sweep_p90", gap=None)  # metrics without a gap = not complete
    cfg = Config(experiments_root=str(tmp_path))
    assert _find_completed_run(cfg, "sweep_p90") is None


def test_plot_renders(tmp_path):
    rows = [
        {"p": 0.6, "gap_trained": 0.0, "worst_group_before": 0.75, "worst_group_after": 0.65,
         "id_before": 0.75, "id_after": 0.66},
        {"p": 0.95, "gap_trained": 0.44, "worst_group_before": 0.50, "worst_group_after": 0.61,
         "id_before": 0.94, "id_after": 0.60},
    ]
    f = tmp_path / "sweep_results.json"
    f.write_text(json.dumps(rows))
    out = plot(f)
    assert out.endswith(".png") and (tmp_path / "sweep_figure.png").exists()
