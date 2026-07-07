"""Typed, YAML-driven config with inheritance and CLI overrides.

A config file may declare `extends: <relative-or-absolute path>` to inherit from a base
(e.g. spike configs extend `base.yaml`). Child keys override parent keys (deep-merged).
`--override key.subkey=value` applies last, on top of the merged result.

The dataset path lives here (`data.nih_root`) — it is never hardcoded in code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    # Dataset roots are config values, resolved at load time. A second site plugs in here.
    nih_root: str = Field(default="", description="Folder holding NIH CSVs + split files.")
    nih_images: str = Field(default="", description="Folder holding the 224px PNG images.")
    labels_csv: str = Field(default="", description="Path to Data_Entry_2017.csv.")
    train_val_list: str = Field(default="", description="Official train/val split list.")
    test_list: str = Field(default="", description="Official test split list.")
    bbox_csv: str = Field(default="", description="Bounding-box CSV (region GT for V3).")
    site2_root: str = Field(default="", description="Optional second-site root (V6).")
    image_size: int = 224


class TrainConfig(BaseModel):
    target_pathology: str = "Pneumothorax"
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-4
    backbone: str = "densenet121"
    amp: bool = True  # mixed precision — keeps the RTX 3050 (6GB) in budget


class Config(BaseModel):
    seed: int = 1337
    device: str = "cuda"  # falls back to cpu via resolve_device() if unavailable
    spike: str = "unnamed"
    experiments_root: str = "experiments"
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    """Load a YAML file, recursively resolving an `extends:` parent."""
    path = path.resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        base = _load_raw(parent_path)
        raw = _deep_merge(base, raw)
    return raw


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for --override values (int/float/bool/str)."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_override(data: dict[str, Any], dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a config file (with `extends`) and apply `key=value` overrides."""
    raw = _load_raw(Path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        _apply_override(raw, key, value)
    return Config(**raw)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach standard --config / --override flags to any spike CLI."""
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.subkey=value",
        help="Override a config value (repeatable).",
    )
