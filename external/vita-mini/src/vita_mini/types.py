"""JSON-safe data types used by the deterministic mini environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Product:
    """An immutable catalogue entry; prices are stored in integer cents."""

    product_id: str
    name: str
    category: str
    price_cents: int
    inventory: int
    tags: tuple[str, ...] = ()


@dataclass
class CartLine:
    product_id: str
    quantity: int


@dataclass
class Order:
    order_id: str
    user_id: str
    lines: list[CartLine]
    total_cents: int
    delivery_address: str
    status: Literal["submitted", "cancelled"]
    created_at: str


@dataclass(frozen=True)
class MiniTask:
    """A static task and its machine-checkable success conditions."""

    task_id: str
    user_message: str
    goals: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ToolResult:
    """The only value returned across the agent/environment tool boundary."""

    ok: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    reward: float
    success: bool
    failed_conditions: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

