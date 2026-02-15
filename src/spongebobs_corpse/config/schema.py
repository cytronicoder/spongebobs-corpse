from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AnalysisSettings:
    seed: int = 42
    confidence_level: float = 0.95
    regression_method: str = "WLS"
    random_time_uncertainty_s: float = 0.01
    random_force_uncertainty_n: float = 0.05


@dataclass(slots=True)
class IOSettings:
    create_final_dir: bool = True
    figure_dpi: int = 400
    manifest_path: str = "batch/outputs/manifest_expected.json"


@dataclass(slots=True)
class AnalysisConfig:
    analysis: AnalysisSettings
    io: IOSettings


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path | None = None) -> AnalysisConfig:
    defaults_path = Path(__file__).with_name("defaults.yaml")
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))

    if config_path is not None:
        override = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        payload = _merge_dict(defaults, override)
    else:
        payload = defaults

    return AnalysisConfig(
        analysis=AnalysisSettings(**payload.get("analysis", {})),
        io=IOSettings(**payload.get("io", {})),
    )
