"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    family: str = "openvla"
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7


@dataclass
class InterventionConfig:
    enabled: bool = False
    dict_name: str = "blank"
    coef: float = 1.0


@dataclass
class LoggingConfig:
    root_dir: str = "logs"
    save_video: bool = True
    save_actions: bool = True


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data = _deep_update(data, overrides)

    return RunConfig(
        model=ModelConfig(**data.get("model", {})),
        env=EnvConfig(**data.get("env", {})),
        intervention=InterventionConfig(**data.get("intervention", {})),
        logging=LoggingConfig(**data.get("logging", {})),
    )


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse key=value overrides into nested dicts (dot notation)."""
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        # basic type coercion
        if raw.lower() in {"true", "false"}:
            value: Any = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides
