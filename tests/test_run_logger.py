"""RunLogger writes the expected reproducibility artifacts."""

import json

from auditor.common.config import Config
from auditor.common.run_logger import RunLogger


def test_run_dir_has_all_artifacts(tmp_path):
    cfg = Config(spike="unit_test")
    rl = RunLogger("unit_test", cfg, experiments_root=str(tmp_path))

    rl.log_metrics(auc=0.83, reliance_gap_pp=27.5)

    run_dir = rl.run_dir
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "git.txt").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "artifacts").is_dir()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["auc"] == 0.83
    assert metrics["reliance_gap_pp"] == 27.5


def test_metrics_accumulate(tmp_path):
    cfg = Config(spike="unit_test")
    rl = RunLogger("unit_test", cfg, experiments_root=str(tmp_path))
    rl.log_metrics(a=1)
    rl.log_metrics(b=2)
    metrics = json.loads((rl.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {"a": 1, "b": 2}


def test_artifact_path_is_under_artifacts(tmp_path):
    cfg = Config(spike="unit_test")
    rl = RunLogger("unit_test", cfg, experiments_root=str(tmp_path))
    p = rl.artifact_path("heatmap.png")
    assert p.parent == rl.artifacts_dir
