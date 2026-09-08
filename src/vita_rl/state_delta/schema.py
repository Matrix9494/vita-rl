"""The canonical, JSON-compatible state-delta protocol definition.

This module intentionally defines only *structural* rules. It does not
interpret user language, tool output, or task semantics.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, TypedDict


GoalStatus = Literal["pending", "active", "blocked", "done", "cancelled"]
ALLOWED_GOAL_STATUSES = frozenset({"pending", "active", "blocked", "done", "cancelled"})
ALLOWED_OPS = frozenset({
    "set", "append", "extend", "delete", "set_status", "noop", "resolve_evidence",
})
LEGAL_STATUS_TRANSITIONS = {
    "pending": {"pending", "active", "blocked", "cancelled"},
    "active": {"active", "blocked", "done", "cancelled"},
    "blocked": {"blocked", "active", "cancelled"},
    "done": {"done"},
    "cancelled": {"cancelled"},
}


class EntityRecord(TypedDict):
    """Entity IDs live in ``/entities/<id>``; records never contain ``id``."""

    type: str
    name: str
    attributes: dict[str, Any]


class GoalRecord(TypedDict):
    type: str
    status: GoalStatus
    slots: dict[str, Any]


class ConstraintRecord(TypedDict):
    goal: str
    strength: Literal["hard", "soft"]
    field: str
    operator: str
    value: Any


class StateDeltaOp(TypedDict, total=False):
    op: Literal["set", "append", "extend", "delete", "set_status", "noop", "resolve_evidence"]
    path: str
    value: Any
    evidence_id: str


class StateDelta(TypedDict):
    ops: list[StateDeltaOp]


# Container-only schemas: these constrain JSON shape, never meaning or values.
GOAL_SLOT_CONTAINER_TYPES: dict[tuple[str, ...], type] = {
    ("items",): list,
    ("requirements",): dict,
    ("requirements", "hard"): list,
    ("requirements", "soft"): list,
    ("selected",): dict,
    ("transaction",): dict,
    ("location",): dict,
    ("timing",): dict,
}


def goal_slot_expected_type(slot_parts: list[str]) -> type | None:
    """Return a lightweight declared container type for a goal-slot pointer."""
    return GOAL_SLOT_CONTAINER_TYPES.get(tuple(slot_parts))


def canonical_schema_description() -> str:
    """Single human-readable schema used by prompts and documentation tests."""
    return (
        '{"entities": {id: {"type": string, "name": string, "attributes": object}}, '
        '"goals": {id: {"type": string, "status": "pending|active|blocked|done|cancelled", '
        '"slots": object}}, "constraints": {id: {"goal": goal_id, '
        '"strength": "hard|soft", "field": string, "operator": string, "value": any}}}'
    )


def empty_canonical_state() -> dict[str, dict[str, Any]]:
    """Return a fresh canonical state. No model-owned fields are implicit."""
    return {"entities": {}, "goals": {}, "constraints": {}}


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy JSON-compatible state before an atomic candidate update."""
    return deepcopy(state)
