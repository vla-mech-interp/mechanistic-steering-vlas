"""Intervention dictionary loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


def load_intervention_dict(dict_name: str, config_path: str | Path) -> Dict[int, List[str]]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        all_dicts = yaml.safe_load(f) or {}

    if dict_name not in all_dicts:
        available = ", ".join(sorted(all_dicts.keys()))
        raise KeyError(f"Unknown intervention dict '{dict_name}'. Available: {available}")

    raw = all_dicts[dict_name] or {}
    return {int(k): v for k, v in raw.items()}
