"""Deterministic, hidden-state delivery environment with scripted users."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .generator import built_in_tasks
from .tasks import Constraint, MiniTask, UserEvent
from .tools import DeliveryTools, ToolError
from .types import EnvironmentState, EvaluationResult, InteractionState, ToolResult, UserState


class MiniEnvironment:
    """A research-oriented delivery environment; it never invokes an LLM."""

    def __init__(self, tasks: dict[str, MiniTask] | None = None) -> None:
        self._tasks = tasks or built_in_tasks()
        self._task: MiniTask | None = None
        self._state: EnvironmentState | None = None
        self._tools: DeliveryTools | None = None

    def reset(self, task_id: str | None = None) -> dict[str, Any]:
        if task_id is None:
            task_id = sorted(self._tasks)[0]
        try:
            self._task = self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"Unknown task {task_id!r}; available: {', '.join(sorted(self._tasks))}") from exc
        database = deepcopy(self._task.initial_state)
        latent = {constraint.constraint_id: constraint.expected for constraint in self._task.latent_constraints}
        self._state = EnvironmentState(
            database=database,
            user=UserState(user_id="mini-user-001", latent_constraints=latent),
            task_id=self._task.task_id,
            interaction=InteractionState(),
        )
        self._tools = DeliveryTools(database, self._state.user.user_id)
        initial = self._consume_event(self._initial_event())
        return {
            "task_id": self._task.task_id,
            "user_message": initial["message"],
            "logical_time": database.current_time,
        }

    @staticmethod
    def agent_policy() -> str:
        return """You are a delivery assistant in a deterministic tool-use environment.
Complete the user's currently stated requests using tools. Information can be
revealed or revised later, so treat newer user instructions as superseding
earlier preferences. Tool-confirmed data is authoritative. Before a final
answer, inspect and pay or cancel orders only when the user has requested it.

Current logical time: {time}"""

    def openai_tools(self) -> list[dict[str, Any]]:
        return deepcopy(self._require_tools().schemas())

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        state = self._require_state()
        state.interaction.tool_calls += 1
        try:
            result = self._require_tools().invoke(name, arguments or {})
            return ToolResult(ok=True, result=result)
        except ToolError as exc:
            state.interaction.invalid_tool_calls += 1
            return ToolResult(ok=False, error=str(exc))

    def next_user_event(
        self, *, agent_turn: int | None = None, tool_name: str | None = None, tool_success: bool | None = None
    ) -> list[dict[str, Any]]:
        """Reveal every currently triggered user event, exactly once.

        The runner calls this after agent turns and tool calls. The returned
        messages are agent-visible; latent constraints and the rest of state
        remain private to evaluation.
        """
        state = self._require_state()
        if agent_turn is not None:
            state.interaction.agent_turns = max(state.interaction.agent_turns, agent_turn)
        events = []
        for event in self._require_task().user_script:
            if event.trigger == "initial" or event.event_id in state.interaction.fired_event_ids:
                continue
            if self._event_matches(event, state.interaction.agent_turns, tool_name, tool_success):
                events.append(self._consume_event(event))
        return events

    def evaluate(self) -> EvaluationResult:
        state = self._require_state()
        task = self._require_task()
        checks = {constraint.constraint_id: self._constraint_met(constraint) for constraint in task.latent_constraints}
        required = [constraint.constraint_id for constraint in task.latent_constraints if constraint.required]
        active_required = [constraint for constraint in task.latent_constraints if constraint.required and constraint.order_selector == "active"]
        active_orders = [order for order in state.database.orders.values() if order.status != "cancelled"]
        one_complete_active_order = any(
            all(self._order_matches(order, constraint) for constraint in active_required)
            for order in active_orders
        )
        success = one_complete_active_order and all(checks[constraint_id] for constraint_id in required)
        constraint_score = sum(checks.values()) / len(checks) if checks else 1.0
        validity = 1.0 - state.interaction.invalid_tool_calls / max(1, state.interaction.tool_calls)
        return EvaluationResult(
            reward=1.0 if success else 0.0,
            success=success,
            constraint_score=constraint_score,
            tool_validity=validity,
            constraints=checks,
            failed_conditions=[constraint_id for constraint_id, met in checks.items() if not met],
            state=self.snapshot(),
        )

    def snapshot(self) -> dict[str, Any]:
        """Executor-only complete state snapshot for reproducibility and reward audit."""
        state = self._require_state()
        return {
            "database": {
                "stores": {key: _json(value) for key, value in state.database.stores.items()},
                "products": {key: _json(value) for key, value in state.database.products.items()},
                "orders": {key: _json(value) for key, value in state.database.orders.items()},
                "locations": dict(state.database.locations),
                "current_time": state.database.current_time,
            },
            "user": _json(state.user),
            "interaction": _json(state.interaction),
        }

    @staticmethod
    def tasks() -> list[dict[str, Any]]:
        return [{"task_id": task_id} for task_id in sorted(built_in_tasks())]

    def _constraint_met(self, constraint: Constraint) -> bool:
        state = self._require_state()
        orders = list(state.database.orders.values())
        if constraint.order_selector == "active":
            orders = [order for order in orders if order.status != "cancelled"]
        else:
            orders = [order for order in orders if order.status == "cancelled"]
        if not orders:
            return False
        return any(self._order_matches(order, constraint) for order in orders)

    def _order_matches(self, order: Any, constraint: Constraint) -> bool:
        product = self._require_state().database.products[order.product_id]
        if constraint.field == "product_id": return order.product_id == constraint.expected
        if constraint.field == "store_id": return order.store_id == constraint.expected
        if constraint.field == "quantity": return order.quantity == constraint.expected
        if constraint.field == "address": return order.address == constraint.expected
        if constraint.field == "paid": return (order.status == "paid") == bool(constraint.expected)
        if constraint.field == "cancelled": return (order.status == "cancelled") == bool(constraint.expected)
        if constraint.field == "max_price_cents": return order.total_cents <= int(constraint.expected)
        if constraint.field == "tag_absent": return str(constraint.expected) not in product.tags
        if constraint.field == "tag_present": return str(constraint.expected) in product.tags
        raise ValueError(f"Unsupported constraint field {constraint.field!r}")

    def _event_matches(self, event: UserEvent, agent_turn: int, tool_name: str | None, tool_success: bool | None) -> bool:
        if event.trigger == "after_agent_turn":
            return agent_turn >= (event.after_turn or 1)
        if event.trigger == "after_tool":
            return event.tool_name is None or event.tool_name == tool_name
        if event.trigger == "after_successful_tool":
            return bool(tool_success) and (event.tool_name is None or event.tool_name == tool_name)
        if event.trigger == "after_order_created":
            return bool(tool_success) and tool_name == "create_order"
        return False

    def _consume_event(self, event: UserEvent) -> dict[str, Any]:
        state = self._require_state()
        state.interaction.fired_event_ids.add(event.event_id)
        if event.revision_of:
            state.user.revision_history.append({"event_id": event.event_id, "revision_of": event.revision_of, "updates": dict(event.updates)})
        state.user.revealed_constraints.update(event.updates)
        state.user.current_preferences.update(event.updates)
        record = {"event_id": event.event_id, "message": event.message, "updates": dict(event.updates), "revision_of": event.revision_of}
        state.interaction.user_events.append(record)
        return record

    def _initial_event(self) -> UserEvent:
        return next(event for event in self._require_task().user_script if event.trigger == "initial")

    def _require_state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Call reset(task_id) before using vita-mini")
        return self._state

    def _require_tools(self) -> DeliveryTools:
        self._require_state()
        assert self._tools is not None
        return self._tools

    def _require_task(self) -> MiniTask:
        if self._task is None:
            raise RuntimeError("Call reset(task_id) before using vita-mini")
        return self._task


def _json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value
