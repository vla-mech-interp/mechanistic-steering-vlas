"""Utilities for OpenPI experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_overrides(pairs: list[str]) -> dict:
    overrides: dict = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
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


def load_config(path: Path, overrides: dict | None) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = deep_update(data, overrides)
    return data


def load_intervention_dict(dict_name: str, path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if dict_name not in data:
        raise KeyError(f"Intervention dictionary '{dict_name}' not found in {path}")
    return data[dict_name]


def write_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
