"""The standalone deterministic environment and rule-based evaluator."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .tasks import get_task, list_tasks
from .tools import StorefrontTools, ToolError
from .types import CartLine, EvaluationResult, Order, Product, ToolResult


def _initial_state() -> dict[str, Any]:
    catalog = {
        "cold-brew": Product("cold-brew", "Cold Brew Coffee", "drinks", 425, 12, ("coffee", "cold", "caffeine")),
        "green-tea": Product("green-tea", "Green Tea", "drinks", 300, 10, ("tea", "hot", "caffeine")),
        # The one seeded submitted order already reserves one bagel, so this
        # is the seven-unit remaining inventory visible to an agent.
        "bagel": Product("bagel", "Plain Bagel", "bakery", 250, 7, ("bread", "breakfast")),
    }
    return {
        "account": {"user_id": "mini-user-001", "display_name": "Mini User", "email": "mini@example.test"},
        "catalog": catalog,
        "cart": [],
        "delivery_address": None,
        "orders": {
            "mini-order-0001": Order(
                order_id="mini-order-0001", user_id="mini-user-001",
                lines=[CartLine("bagel", 1)], total_cents=250,
                delivery_address="1 Mini Way", status="submitted", created_at="2026-01-01T09:00:00Z",
            )
        },
        "next_order_number": 2,
        "logical_time": "2026-01-01T09:00:00Z",
    }


class MiniEnvironment:
    """A stateful storefront with a narrow, LLM-safe tool boundary."""

    def __init__(self) -> None:
        self._state = _initial_state()
        self._tools = StorefrontTools(self._state)
        self._task_id: str | None = None

    def reset(self, task_id: str | None = None) -> dict[str, Any]:
        """Reset all mutable state and return the static initial observation."""
        self._state = _initial_state()
        self._tools = StorefrontTools(self._state)
        self._task_id = task_id
        task = get_task(task_id) if task_id is not None else None
        return {
            "task_id": task.task_id if task else None,
            "user_message": task.user_message if task else None,
            "account": self._tools.get_account(),
            "logical_time": self._state["logical_time"],
        }

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return the complete agent-facing tool set in OpenAI function format."""
        return deepcopy(self._tools.schemas())

    @staticmethod
    def agent_policy() -> str:
        """Return the system policy supplied to every supported harness.

        ``{time}`` is substituted by the harness, matching the existing
        VitaBench harness contract without depending on VitaBench itself.
        """
        return """You are a storefront assistant in a deterministic tool-use environment.
Complete the user's request by calling the available tools. Treat tool results
as the source of truth; do not claim that an order was created or cancelled
until the corresponding tool confirms it. Use a concise final response after
the requested state transition succeeds.

Current logical time: {time}"""

    def tool_names(self) -> list[str]:
        return self._tools.names

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Invoke one agent tool without exposing implementation exceptions."""
        try:
            return ToolResult(ok=True, result=self._tools.invoke(name, arguments or {}))
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))

    def evaluate(self) -> EvaluationResult:
        """Evaluate the selected static task using only explicit state conditions."""
        if self._task_id is None:
            raise ValueError("reset(task_id) must be called before evaluate()")
        task = get_task(self._task_id)
        failures = [self._describe_goal(goal) for goal in task.goals if not self._goal_met(goal)]
        return EvaluationResult(
            reward=1.0 if not failures else 0.0,
            success=not failures,
            failed_conditions=failures,
            state=self.snapshot(),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copied, JSON-safe state for tests and reproducibility."""
        return {
            "account": deepcopy(self._state["account"]),
            "catalog": [self._tools.get_product(product_id) for product_id in sorted(self._state["catalog"])],
            "cart": self._tools.get_cart(),
            "orders": self._tools.list_orders(),
            "logical_time": self._state["logical_time"],
        }

    @staticmethod
    def tasks() -> list[dict[str, str]]:
        """List static task ids and their user messages without model generation."""
        return [{"task_id": task.task_id, "user_message": task.user_message} for task in list_tasks()]

    def _goal_met(self, goal: dict[str, Any]) -> bool:
        goal_type = goal["type"]
        if goal_type == "submitted_product_quantity":
            return any(
                order.status == "submitted"
                and sum(line.quantity for line in order.lines if line.product_id == goal["product_id"]) == goal["quantity"]
                for order in self._state["orders"].values()
            )
        if goal_type == "order_address":
            return any(order.status == "submitted" and order.delivery_address == goal["address"] for order in self._state["orders"].values())
        if goal_type == "order_status":
            order = self._state["orders"].get(goal["order_id"])
            return order is not None and order.status == goal["status"]
        raise ValueError(f"Unsupported goal type {goal_type!r}")

    @staticmethod
    def _describe_goal(goal: dict[str, Any]) -> str:
        goal_type = goal["type"]
        if goal_type == "submitted_product_quantity":
            return f"a submitted order must contain {goal['quantity']} x {goal['product_id']}"
        if goal_type == "order_address":
            return f"a submitted order must use address {goal['address']!r}"
        if goal_type == "order_status":
            return f"order {goal['order_id']} must have status {goal['status']!r}"
        return f"unsupported goal: {goal}"
