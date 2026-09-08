"""Feasible, seeded procedural delivery-task generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .tasks import Constraint, MiniTask, UserEvent
from .types import DatabaseState, DeliveryOrder, Product, Store


@dataclass(frozen=True)
class DifficultyConfig:
    num_constraints: int = 8
    num_objectives: int = 1
    target_horizon: int = 7
    max_retention_distance: int = 4
    num_revisions: int = 1
    num_distractors: int = 2
    num_tools: int = 9


class TaskGenerator:
    """Creates a solvable delivery task from a seed and difficulty controls."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(self, task_id: str | None = None, difficulty: DifficultyConfig | None = None) -> MiniTask:
        difficulty = difficulty or DifficultyConfig()
        rng = random.Random(self.seed)
        target_store = "store-target"
        target_product = "product-herbal-tea"
        stores = {
            target_store: Store(target_store, "Garden Delivery", ("tea", "healthy", "delivery"), True),
        }
        products = {
            target_product: Product(target_product, target_store, "Herbal Tea", rng.randrange(375, 501, 25), ("tea", "caffeine-free", "non-spicy"), 20),
            "product-cold-brew": Product("product-cold-brew", target_store, "Cold Brew", rng.randrange(375, 501, 25), ("coffee", "caffeine", "cold"), 20),
        }
        for index in range(difficulty.num_distractors):
            store_id = f"store-distractor-{index}"
            product_id = f"product-distractor-{index}"
            stores[store_id] = Store(store_id, f"Distractor Cafe {index}", ("coffee", "delivery"), bool(index % 2))
            products[product_id] = Product(product_id, store_id, "Spicy Coffee", rng.randrange(250, 501, 25), ("coffee", "caffeine", "spicy"), 20)
        database = DatabaseState(
            stores=stores, products=products, orders={},
            locations={"home": "1 Home Street", "office": "2 Office Plaza"},
            current_time="2026-01-01 09:00:00", next_order_number=1,
        )
        constraints = [
            Constraint("correct_product", "product_id", target_product),
            Constraint("correct_store", "store_id", target_store),
            Constraint("quantity_two", "quantity", 2),
            Constraint("office_address", "address", "office"),
            Constraint("replacement_paid", "paid", True),
            Constraint("caffeine_free", "tag_absent", "caffeine"),
            Constraint("non_spicy", "tag_absent", "spicy"),
            Constraint("under_budget", "max_price_cents", 1000),
        ][: max(1, difficulty.num_constraints)]
        script = [
            UserEvent("initial", "initial", "Please order two cold brews for delivery to home."),
            UserEvent("revise-drink", "after_successful_tool", "Actually, I cannot have caffeine. Please use a non-spicy herbal tea instead.", tool_name="search_products", revision_of="drink", updates={"product_id": target_product, "tag_absent": ["caffeine", "spicy"]}),
            UserEvent("revise-address", "after_order_created", "Please send it to the office, not home, and then pay for the final order.", revision_of="address", updates={"address": "office", "paid": True}),
        ]
        if difficulty.num_revisions == 0:
            script = [script[0], UserEvent("address", "after_agent_turn", "Please send it to the office and pay after checking the order.", after_turn=2, updates={"address": "office", "paid": True})]
        return MiniTask(
            task_id=task_id or f"delivery-{self.seed:06d}", initial_state=database,
            objectives=tuple("create and pay a delivery order" for _ in range(max(1, difficulty.num_objectives))), latent_constraints=tuple(constraints),
            user_script=tuple(script), difficulty={name: getattr(difficulty, name) for name in difficulty.__dataclass_fields__},
            oracle_solution=(
                {"tool": "search_products", "arguments": {"query": "coffee"}},
                {"tool": "search_products", "arguments": {"query": "herbal tea"}},
                {"tool": "create_order", "arguments": {"store_id": target_store, "product_id": target_product, "quantity": 2, "address": "home", "delivery_time": "2026-01-01 12:00:00"}},
                {"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "address": "office"}},
                {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}},
            ),
        )


def built_in_tasks() -> dict[str, MiniTask]:
    """Small static set backed by stable generator seeds for reproducible tests."""
    return {
        "delivery_revision": TaskGenerator(seed=7).generate("delivery_revision"),
    }
