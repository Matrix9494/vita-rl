"""Built-in deterministic tasks.  Add data here instead of LLM user prompts."""

from __future__ import annotations

from .types import MiniTask


TASKS: dict[str, MiniTask] = {
    "buy_coffee": MiniTask(
        task_id="buy_coffee",
        user_message="Please order two cold brew coffees for delivery to 1 Mini Way.",
        goals=(
            {"type": "submitted_product_quantity", "product_id": "cold-brew", "quantity": 2},
            {"type": "order_address", "address": "1 Mini Way"},
        ),
    ),
    "cancel_order": MiniTask(
        task_id="cancel_order",
        user_message=(
            "I changed my mind. Please cancel my existing order mini-order-0001."
        ),
        goals=(
            {"type": "order_status", "order_id": "mini-order-0001", "status": "cancelled"},
        ),
    ),
}


def get_task(task_id: str) -> MiniTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task {task_id!r}. Available tasks: {available}") from exc


def list_tasks() -> list[MiniTask]:
    return [TASKS[task_id] for task_id in sorted(TASKS)]

