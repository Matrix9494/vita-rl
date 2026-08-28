"""Stable semantic action representation for training interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from advancewars.engine.actions import Action, ActionType
from advancewars.engine.coordinates import Coord
from advancewars.engine.state import GameState


@dataclass(frozen=True)
class SemanticAction:
    """A structured, stable view of a complete game action.

    ``source`` is the acting unit's current coordinate when a unit acts.
    ``destination`` is the movement destination; for current non-composite
    engine actions it is usually the same as ``source``.
    ``target`` is the coordinate affected by the terminal action.
    ``payload`` carries build units, power kind, cargo ids, or similar extras.
    """

    kind: ActionType
    source: Coord | None = None
    destination: Coord | None = None
    target: Coord | None = None
    payload: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def action_to_semantic(action: Action, state: GameState) -> SemanticAction:
    """Convert the current engine action object into a stable semantic action."""
    source = None
    destination = None
    if action.unit_id is not None:
        unit = state.units[action.unit_id]
        source = unit.coord
        destination = unit.coord
    if action.type == ActionType.MOVE and action.path:
        destination = action.path[-1]

    payload: str | int | None = None
    if action.type == ActionType.BUILD:
        payload = action.build_unit
    elif action.type == ActionType.CO_ABILITY:
        payload = action.metadata.get("power", "power")
    elif "cargo_unit_id" in action.metadata:
        payload = int(action.metadata["cargo_unit_id"])

    return SemanticAction(
        kind=action.type,
        source=source,
        destination=destination,
        target=action.target,
        payload=payload,
        metadata=dict(action.metadata),
    )
