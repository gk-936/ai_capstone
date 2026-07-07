"""Tests for the config loader: inheritance, overrides, and the real config files."""

from pathlib import Path

import pytest

from auditor.common.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_load():
    base = load_config(CONFIGS / "base.yaml")
    assert isinstance(base, Config)
    assert base.seed == 1337
    assert base.data.image_size == 224


def test_extends_deep_merges(tmp_path):
    _write(tmp_path, "base.yaml", "seed: 1\ntrain:\n  epochs: 5\n  lr: 0.1\n")
    child = _write(tmp_path, "child.yaml", "extends: base.yaml\ntrain:\n  epochs: 9\n")
    cfg = load_config(child)
    assert cfg.seed == 1          # inherited
    assert cfg.train.epochs == 9  # overridden
    assert cfg.train.lr == 0.1    # sibling key preserved by deep merge


def test_v1_config_extends_base():
    cfg = load_config(CONFIGS / "v1_baseline.yaml")
    assert cfg.spike == "v1_baseline"
    assert cfg.train.backbone == "densenet121"  # inherited from base


def test_cli_overrides_coerce_types():
    cfg = load_config(
        CONFIGS / "base.yaml",
        overrides=["seed=42", "train.epochs=3", "train.amp=false"],
    )
    assert cfg.seed == 42 and isinstance(cfg.seed, int)
    assert cfg.train.epochs == 3
    assert cfg.train.amp is False


def test_bad_override_raises():
    with pytest.raises(ValueError):
        load_config(CONFIGS / "base.yaml", overrides=["not_a_pair"])
