from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "tiny",
    "output_root": "outputs",
    "dataset": {
        "seed": 7,
        "num_entities": 12,
        "time_steps": 18,
        "episodes_per_level": 3,
        "queries_per_episode": 12,
        "memory_write_prob": 0.24,
        "top_k": 12,
        "composite_levels": [],
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    if not merged["dataset"]["composite_levels"]:
        raise ValueError("Config must define at least one composite stress level.")
    return merged


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "config_snapshot.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_path
