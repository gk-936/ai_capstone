"""Per-run logging: one timestamped directory holding everything needed to reproduce a run.

Layout: experiments/<spike>/<timestamp>_<shorthash>/
    config.yaml    the fully-resolved config actually used
    metrics.json   measured numbers — the values gates are checked against
    artifacts/     checkpoints, crops, heatmaps, plots
    run.log        console log
    git.txt        commit hash + dirty flag, mapping this result to exact code
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from auditor.common.paths import REPO_ROOT, resolve_path


def _git_state() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        return f"commit: {commit}\ndirty: {bool(dirty)}\n{dirty}\n"
    except Exception as exc:  # not a git repo, or git unavailable
        return f"git state unavailable: {exc}\n"


class RunLogger:
    """Creates and owns a single run's output directory."""

    def __init__(self, spike: str, config: BaseModel, experiments_root: str = "experiments"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short = f"{abs(hash(stamp)) % 0x1000000:06x}"
        self.run_dir = resolve_path(experiments_root) / spike / f"{stamp}_{short}"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Resolved config snapshot
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config.model_dump(), f, sort_keys=False)

        # Exact code provenance
        (self.run_dir / "git.txt").write_text(_git_state(), encoding="utf-8")

        # Console + file logging
        self.logger = logging.getLogger(f"auditor.{spike}.{stamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (fh, ch):
            h.setFormatter(fmt)
            self.logger.addHandler(h)

        self._metrics: dict = {}
        self.info(f"Run dir: {self.run_dir}")

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def log_metrics(self, **metrics) -> None:
        """Accumulate measured metrics and persist metrics.json after each update."""
        self._metrics.update(metrics)
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2)
        self.info(f"metrics: {metrics}")

    def artifact_path(self, name: str) -> Path:
        return self.artifacts_dir / name
