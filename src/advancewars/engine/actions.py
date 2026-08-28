"""Engine action model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from advancewars.engine.coordinates import Coord


class ActionType(str, Enum):
    MOVE = "MOVE"
    ATTACK = "ATTACK"
    CAPTURE = "CAPTURE"
    JOIN = "JOIN"
    LOAD = "LOAD"
    UNLOAD = "UNLOAD"
    LAUNCH = "LAUNCH"
    RESUPPLY = "RESUPPLY"
    REPAIR = "REPAIR"
    DELETE = "DELETE"
    TRANSFORM = "TRANSFORM"
    CO_ABILITY = "CO_ABILITY"
    SWAP_CO = "SWAP_CO"
    WAIT = "WAIT"
    BUILD = "BUILD"
    END_TURN = "END_TURN"


@dataclass(frozen=True)
class Action:
    type: ActionType
    unit_id: int | None = None
    path: tuple[Coord, ...] = ()
    target: Coord | None = None
    build_unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def end_turn(cls) -> Action:
        return cls(ActionType.END_TURN)
