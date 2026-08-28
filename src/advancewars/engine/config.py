"""Experiment configuration for selectable rules subsets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from advancewars.engine.data import UNIT_ALIASES, UNITS


@dataclass(frozen=True)
class GameConfig:
    """Configurable experiment switches for a match.

    ``enabled_units=None`` means the full ruleset unit pool is available.
    """

    name: str = "standard"
    enabled_units: frozenset[str] | None = None
    strict_units: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GameConfig:
        enabled_units = payload.get("enabled_units")
        return cls(
            name=str(payload.get("name", "custom")),
            enabled_units=_normalize_unit_pool(enabled_units),
            strict_units=bool(payload.get("strict_units", False)),
        )

    def with_overrides(
        self,
        *,
        enabled_units: Sequence[str] | None = None,
        strict_units: bool | None = None,
    ) -> GameConfig:
        return GameConfig(
            name=self.name,
            enabled_units=(
                self.enabled_units
                if enabled_units is None
                else _normalize_unit_pool(enabled_units)
            ),
            strict_units=self.strict_units if strict_units is None else strict_units,
        )

    def unit_enabled(self, unit_type: str) -> bool:
        return self.enabled_units is None or unit_type in self.enabled_units

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled_units": (
                None if self.enabled_units is None else sorted(self.enabled_units)
            ),
            "strict_units": self.strict_units,
        }


def load_config(
    config: str | Mapping[str, Any] | GameConfig | None = None,
) -> GameConfig:
    """Load a built-in, file-backed, or inline game config."""
    if config is None:
        return GameConfig()
    if isinstance(config, GameConfig):
        return config
    if isinstance(config, Mapping):
        return GameConfig.from_mapping(config)

    path = Path(config)
    if path.exists():
        payload = yaml.safe_load(path.read_text()) or {}
        return GameConfig.from_mapping(payload)

    package = "advancewars.data.configs"
    resource_name = config if config.endswith(".yaml") else f"{config}.yaml"
    try:
        text = resources.files(package).joinpath(resource_name).read_text()
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise ValueError(f"Unknown config: {config}") from exc
    return GameConfig.from_mapping(yaml.safe_load(text) or {})


def _normalize_unit_pool(enabled_units: Any) -> frozenset[str] | None:
    if enabled_units is None:
        return None
    if not isinstance(enabled_units, Sequence) or isinstance(enabled_units, str):
        raise ValueError("enabled_units must be a list of unit names or null.")

    normalized: list[str] = []
    unknown: list[str] = []
    for raw_unit in enabled_units:
        unit_name = str(raw_unit).lower().replace("-", "_").replace(" ", "_")
        unit_type = UNIT_ALIASES.get(unit_name, unit_name)
        if unit_type not in UNITS:
            unknown.append(str(raw_unit))
        else:
            normalized.append(unit_type)
    if unknown:
        raise ValueError(f"Unknown unit type(s) in config: {', '.join(unknown)}")
    return frozenset(normalized)
