"""Configurable scripted policy parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from advancewars.policies import InfantryDecisionTreePolicy


@dataclass
class InfantryPolicyConfig:
    min_funds_to_build: int = 1000
    attack_base: float = 9000
    attack_damage_weight: float = 14
    attack_cost_weight: float = 0.1
    attack_kill_bonus: float = 2600
    capture_threat_bonus: float = 900
    hq_threat_bonus: float = 2000
    capture_base: float = 7600
    capture_hp_weight: float = 1
    capture_hq_bonus: float = 4500
    capture_factory_bonus: float = 1800
    capture_property_bonus: float = 900
    capture_enemy_property_bonus: float = 1200
    build_base: float = 5200
    build_unit_deficit_weight: float = 120
    build_pressure_penalty: float = 10
    move_progress_weight: float = 180
    move_distance_penalty: float = 20
    move_capture_bonus: float = 1400
    move_hq_bonus: float = 2500
    move_factory_bonus: float = 900
    adjacent_enemy_penalty: float = 350
    nearby_enemy_penalty: float = 80
    wait_penalty: float = 50
    end_turn_score: float = -2000

    def build_policy(self) -> InfantryDecisionTreePolicy:
        return InfantryDecisionTreePolicy(**asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> InfantryPolicyConfig:
        return cls(**(payload or {}))

    @classmethod
    def load(cls, path: str | Path) -> InfantryPolicyConfig:
        payload = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))
