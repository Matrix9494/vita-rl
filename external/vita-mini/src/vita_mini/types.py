"""State and result types for deterministic vita-mini delivery episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Store:
    store_id: str
    name: str
    tags: tuple[str, ...]
    supports_dine_in: bool


@dataclass(frozen=True)
class Product:
    product_id: str
    store_id: str
    name: str
    price_cents: int
    tags: tuple[str, ...]
    inventory: int


@dataclass
class DeliveryOrder:
    order_id: str
    user_id: str
    store_id: str
    product_id: str
    quantity: int
    address: str
    delivery_time: str
    total_cents: int
    status: Literal["unpaid", "paid", "cancelled"]


@dataclass
class DatabaseState:
    stores: dict[str, Store]
    products: dict[str, Product]
    orders: dict[str, DeliveryOrder]
    locations: dict[str, str]
    current_time: str
    next_order_number: int = 1


@dataclass
class UserState:
    user_id: str
    latent_constraints: dict[str, Any]
    revealed_constraints: dict[str, Any] = field(default_factory=dict)
    current_preferences: dict[str, Any] = field(default_factory=dict)
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    next_user_event: int = 0


@dataclass
class InteractionState:
    agent_turns: int = 0
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    user_events: list[dict[str, Any]] = field(default_factory=list)
    fired_event_ids: set[str] = field(default_factory=set)


@dataclass
class EnvironmentState:
    database: DatabaseState
    user: UserState
    task_id: str | None = None
    interaction: InteractionState = field(default_factory=InteractionState)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    reward: float
    success: bool
    constraint_score: float
    tool_validity: float
    constraints: dict[str, bool]
    failed_conditions: list[str]
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
