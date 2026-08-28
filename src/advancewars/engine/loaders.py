"""Public data loading helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from advancewars.engine.data import BUILTIN_MAPS, RULESETS


def load_map(name: str) -> str:
    """Load a built-in DefendPeace-style map text by name."""
    if name in BUILTIN_MAPS:
        return BUILTIN_MAPS[name]
    path = Path(name)
    if path.exists():
        return path.read_text()
    package = "advancewars.data.maps"
    resource_name = name if name.endswith(".map") else f"{name}.map"
    try:
        return resources.files(package).joinpath(resource_name).read_text()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown map: {name}") from exc


def load_ruleset(name: str = "defendpeace_awbw") -> dict[str, Any]:
    """Load a ruleset dictionary by name."""
    if name in RULESETS:
        return dict(RULESETS[name])
    path = Path(name)
    if path.exists():
        return yaml.safe_load(path.read_text())
    package = "advancewars.data.rulesets"
    resource_name = name if name.endswith(".yaml") else f"{name}.yaml"
    try:
        text = resources.files(package).joinpath(resource_name).read_text()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown ruleset: {name}") from exc
    return yaml.safe_load(text)
