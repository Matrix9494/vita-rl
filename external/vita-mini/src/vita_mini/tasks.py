"""Declarative mini tasks, constraints, and scripted user disclosures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .types import DatabaseState


TriggerKind = Literal["initial", "after_agent_turn", "after_tool", "after_successful_tool", "after_order_created"]


@dataclass(frozen=True)
class UserEvent:
    event_id: str
    trigger: TriggerKind
    message: str
    after_turn: int | None = None
    tool_name: str | None = None
    revision_of: str | None = None
    updates: dict[str, Any] = field(default_factory=dict)
    after_event_id: str | None = None
    min_tool_calls: int | None = None


@dataclass(frozen=True)
class Constraint:
    """An executable final-state predicate encoded as data, never prose."""

    constraint_id: str
    field: Literal["product_id", "store_id", "quantity", "address", "paid", "cancelled", "max_price_cents", "tag_absent", "tag_present"]
    expected: Any
    order_selector: Literal["active", "cancelled"] = "active"
    required: bool = True


@dataclass(frozen=True)
class MiniTask:
    task_id: str
    initial_state: DatabaseState
    objectives: tuple[str, ...]
    latent_constraints: tuple[Constraint, ...]
    user_script: tuple[UserEvent, ...]
    difficulty: dict[str, int]
    oracle_solution: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def initial_message(self) -> str:
        return next(event.message for event in self.user_script if event.trigger == "initial")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
