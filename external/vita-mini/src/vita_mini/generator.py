"""Feasible, seeded procedural delivery-task generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .tasks import Constraint, MiniTask, UserEvent
from .types import DatabaseState, Product, Store


@dataclass(frozen=True)
class DifficultyConfig:
    num_constraints: int = 8
    num_objectives: int = 1
    target_horizon: int = 7
    max_retention_distance: int = 4
    num_revisions: int = 2
    num_distractors: int = 4
    num_tools: int = 9


@dataclass(frozen=True)
class _Catalog:
    slug: str
    initial_name: str
    initial_tags: tuple[str, ...]
    final_name: str
    final_tags: tuple[str, ...]
    forbidden_tags: tuple[str, str]
    revision_reason: str


_CATALOGS = (
    _Catalog("tea", "Cold Brew", ("coffee", "caffeine", "cold"), "Herbal Tea", ("tea", "caffeine-free", "non-spicy"), ("caffeine", "spicy"), "I cannot have caffeine"),
    _Catalog("fruit", "Spicy Chicken Wrap", ("wrap", "chicken", "spicy"), "Fresh Fruit Bowl", ("fruit", "vegan", "gluten-free", "non-spicy"), ("meat", "spicy"), "I need a vegan, gluten-free meal"),
    _Catalog("oat", "Whole Milk Latte", ("latte", "dairy", "caffeine"), "Oat Milk Latte", ("latte", "oat", "vegan", "caffeine-free"), ("dairy", "caffeine"), "I need to avoid both dairy and caffeine"),
    _Catalog("soup", "Chicken Chili", ("chili", "chicken", "spicy"), "Lentil Soup", ("soup", "lentil", "vegan", "gluten-free", "non-spicy"), ("meat", "spicy"), "I need a mild plant-based option"),
    _Catalog("water", "Energy Drink", ("energy", "caffeine", "sugar"), "Sparkling Water", ("water", "caffeine-free", "sugar-free", "non-spicy"), ("caffeine", "sugar"), "I need a caffeine-free, sugar-free drink"),
)
_TARGET_STORES = (
    ("store-willow", "Willow Market", ("healthy", "delivery", "pantry")),
    ("store-harbor", "Harbor Pantry", ("fresh", "delivery", "local")),
    ("store-sunrise", "Sunrise Kitchen", ("prepared", "delivery", "healthy")),
    ("store-maple", "Maple Provisions", ("market", "delivery", "seasonal")),
)
_SOURCE_STORES = (
    ("store-corner", "Corner Counter", ("coffee", "delivery")),
    ("store-metro", "Metro Bites", ("quick", "delivery")),
    ("store-river", "River Cafe", ("cafe", "delivery")),
)
_FLOW_NAMES = (
    "replace",
    "modify",
    "direct",
    "reschedule",
    "memory_delayed",
    "memory_accumulate",
    "memory_partial",
)
_MEMORY_FLOWS = frozenset(_FLOW_NAMES[4:])


class TaskGenerator:
    """Creates reproducible tasks with varied catalogues and action graphs.

    ``delivery_revision`` deliberately remains a stable fixture for examples
    and regression tests. Procedural ``generated:<seed>`` tasks vary both the
    product domain and interaction pattern, not merely prices or identifiers.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(self, task_id: str | None = None, difficulty: DifficultyConfig | None = None) -> MiniTask:
        difficulty = difficulty or DifficultyConfig()
        if task_id == "delivery_revision":
            return self._canonical_delivery_revision(task_id, difficulty)
        return self._procedural_task(task_id, difficulty)

    def _procedural_task(self, task_id: str | None, difficulty: DifficultyConfig) -> MiniTask:
        rng = random.Random(self.seed)
        catalog = rng.choice(_CATALOGS)
        target_store_id, target_store_name, target_store_tags = rng.choice(_TARGET_STORES)
        source_store_id, source_store_name, source_store_tags = rng.choice(_SOURCE_STORES)
        flow = rng.choice(_FLOW_NAMES)
        if difficulty.num_revisions == 0:
            flow = "direct"
        final_quantity, initial_quantity = rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
        if flow == "memory_partial":
            initial_quantity = rng.choice(tuple(quantity for quantity in (1, 2, 3) if quantity != final_quantity))
        initial_address = rng.choice(("home", "office", "studio", "campus"))
        final_address = rng.choice(tuple(address for address in ("home", "office", "studio", "campus") if address != initial_address))
        delivery_hour = rng.randrange(10, 15)
        delivery_time = f"2026-01-01 {delivery_hour:02d}:00:00"
        revised_delivery_time = f"2026-01-01 {delivery_hour + rng.choice((1, 2)):02d}:30:00"
        final_price, initial_price = rng.randrange(200, 451, 25), rng.randrange(200, 501, 25)
        budget = final_price * final_quantity + rng.choice((0, 50, 100, 150))
        final_product_id, initial_product_id = f"product-{catalog.slug}-final", f"product-{catalog.slug}-initial"
        initial_store_id = target_store_id if flow == "modify" else source_store_id
        initial_store_name = target_store_name if flow == "modify" else source_store_name

        stores = {target_store_id: Store(target_store_id, target_store_name, target_store_tags, True)}
        if initial_store_id != target_store_id:
            stores[source_store_id] = Store(source_store_id, source_store_name, source_store_tags, True)
        products = {
            final_product_id: Product(final_product_id, target_store_id, catalog.final_name, final_price, catalog.final_tags, rng.randrange(12, 25)),
            initial_product_id: Product(initial_product_id, initial_store_id, catalog.initial_name, initial_price, catalog.initial_tags, rng.randrange(12, 25)),
        }
        self._add_distractors(rng, stores, products, difficulty.num_distractors)
        if difficulty.num_distractors:
            self._add_near_match(rng, products, catalog, final_price)
        database = DatabaseState(
            stores=stores, products=products, orders={},
            locations={"home": "1 Home Street", "office": "2 Office Plaza", "studio": "3 Studio Lane", "campus": "4 Campus Way"},
            current_time="2026-01-01 09:00:00", next_order_number=1,
        )
        constraints = [
            Constraint("correct_product", "product_id", final_product_id), Constraint("correct_store", "store_id", target_store_id),
            Constraint(f"quantity_{final_quantity}", "quantity", final_quantity), Constraint(f"{final_address}_address", "address", final_address),
            Constraint("final_order_paid", "paid", True), Constraint(f"no_{catalog.forbidden_tags[0]}", "tag_absent", catalog.forbidden_tags[0]),
            Constraint(f"no_{catalog.forbidden_tags[1]}", "tag_absent", catalog.forbidden_tags[1]), Constraint("within_budget", "max_price_cents", budget),
        ][: max(1, difficulty.num_constraints)]
        if flow in _MEMORY_FLOWS:
            script, oracle_solution, memory_requirements = self._memory_flow(
                flow, catalog, final_product_id, target_store_id, target_store_name, initial_quantity, final_quantity,
                final_address, delivery_time, revised_delivery_time, budget,
            )
        else:
            script, oracle_solution = self._flow(
                flow, catalog, initial_product_id, final_product_id, initial_store_id, target_store_id, initial_store_name, target_store_name,
                initial_quantity, final_quantity, initial_address, final_address, delivery_time, revised_delivery_time, budget, difficulty.num_revisions,
            )
            memory_requirements = []
        difficulty_values = {name: getattr(difficulty, name) for name in difficulty.__dataclass_fields__}
        difficulty_values["scenario_index"] = _FLOW_NAMES.index(flow)
        metadata = {
            "task_family": flow,
            "memory_requirements": memory_requirements,
            "retention_distance": max((requirement["retention_distance"] for requirement in memory_requirements), default=0),
        }
        return MiniTask(
            task_id=task_id or f"delivery-{self.seed:06d}", initial_state=database,
            objectives=tuple("create and pay a delivery order" for _ in range(max(1, difficulty.num_objectives))),
            latent_constraints=tuple(constraints), user_script=tuple(script), difficulty=difficulty_values,
            oracle_solution=tuple(oracle_solution), metadata=metadata,
        )

    @staticmethod
    def _add_distractors(rng: random.Random, stores: dict[str, Store], products: dict[str, Product], count: int) -> None:
        distractors = (("Spicy Mocha", ("coffee", "caffeine", "spicy")), ("Sugared Soda", ("drink", "sugar", "caffeine")), ("Creamy Pasta", ("pasta", "dairy", "meat")), ("Hot Wings", ("chicken", "spicy", "meal")))
        for index in range(count):
            name, tags = distractors[index % len(distractors)]
            store_id, product_id = f"store-distractor-{index}", f"product-distractor-{index}"
            stores[store_id] = Store(store_id, f"Market Stall {index + 1}", ("delivery", tags[0]), bool(index % 2))
            products[product_id] = Product(product_id, store_id, name, rng.randrange(150, 501, 25), tags, rng.randrange(8, 22))

    @staticmethod
    def _add_near_match(rng: random.Random, products: dict[str, Product], catalog: _Catalog, final_price: int) -> None:
        """Add a same-name decoy that violates the final dietary constraints."""
        products[f"product-{catalog.slug}-near-match"] = Product(
            f"product-{catalog.slug}-near-match",
            "store-distractor-0",
            catalog.final_name,
            max(150, final_price - 25),
            (catalog.final_tags[0], *catalog.forbidden_tags),
            rng.randrange(8, 22),
        )

    @staticmethod
    def _memory_requirement(
        constraint: str,
        value: object,
        disclosure_event: str,
        disclosure_index: int,
        required_event: str,
        required_index: int,
        *,
        persistent: bool = True,
        superseded_by: str | None = None,
        required_action: str | None = "create_order",
    ) -> dict[str, object]:
        return {
            "constraint": constraint,
            "value": value,
            "disclosure_event": disclosure_event,
            "last_disclosure_event": disclosure_event,
            "disclosure_event_index": disclosure_index,
            "required_event": required_event,
            "required_event_index": required_index,
            "required_action": required_action,
            "persistent": persistent,
            "superseded_by": superseded_by,
            "db_recoverable_before_use": False,
            "retention_distance": required_index - disclosure_index,
        }

    @classmethod
    def _memory_flow(
        cls,
        flow: str,
        catalog: _Catalog,
        final_product_id: str,
        target_store_id: str,
        target_store_name: str,
        initial_quantity: int,
        final_quantity: int,
        final_address: str,
        delivery_time: str,
        revised_delivery_time: str,
        budget: int,
    ) -> tuple[list[UserEvent], list[dict[str, object]], list[dict[str, object]]]:
        """Build delayed-commitment tasks whose decisive order is created last.

        The events deliberately expose preferences in pieces.  Before
        ``finalize`` no order exists, so quantity, dietary, and budget
        requirements cannot be recovered through the delivery database.
        """
        budget_dollars = f"${budget / 100:.2f}"
        first_forbidden, second_forbidden = catalog.forbidden_tags
        disclosed_quantity = initial_quantity if flow == "memory_partial" else final_quantity
        if flow == "memory_accumulate":
            initial = UserEvent(
                "persistent-preferences",
                "initial",
                f"Before we settle on an item, please keep these requirements in mind: I need {final_quantity} unit(s), and the final total must stay at or under {budget_dollars}. Please look through the options first; do not place an order yet.",
                updates={"quantity": final_quantity, "max_price_cents": budget},
            )
            requirements = [
                cls._memory_requirement("max_price_cents", budget, "persistent-preferences", 0, "finalize", 3),
            ]
        else:
            initial = UserEvent(
                "persistent-preferences",
                "initial",
                f"Before we settle on an item, please keep these requirements in mind: I need {disclosed_quantity} unit(s), it must avoid {first_forbidden} and {second_forbidden}, and the final total must stay at or under {budget_dollars}. Please look through the options first; do not place an order yet.",
                updates={"quantity": disclosed_quantity, "tag_absent": [first_forbidden, second_forbidden], "max_price_cents": budget},
            )
            requirements = [
                cls._memory_requirement("tag_absent", first_forbidden, "persistent-preferences", 0, "finalize", 3 if flow != "memory_delayed" else 2),
                cls._memory_requirement("tag_absent", second_forbidden, "persistent-preferences", 0, "finalize", 3 if flow != "memory_delayed" else 2),
                cls._memory_requirement("max_price_cents", budget, "persistent-preferences", 0, "finalize", 3 if flow != "memory_delayed" else 2),
            ]

        if flow == "memory_delayed":
            script = [
                initial,
                UserEvent(
                    "choose-product",
                    "after_tool",
                    f"I have decided on the {catalog.final_name} from {target_store_name}. Please check that option next; I will give delivery details once we have settled it.",
                    updates={"product_id": final_product_id, "store_id": target_store_id},
                ),
                UserEvent(
                    "finalize",
                    "after_tool",
                    f"Send the option we settled on to {final_address} for {delivery_time}, then place and pay for the order.",
                    after_event_id="choose-product",
                    min_tool_calls=2,
                    updates={"address": final_address, "paid": True},
                ),
            ]
            requirements.append(cls._memory_requirement("quantity", final_quantity, "persistent-preferences", 0, "finalize", 2))
            requirements.append(cls._memory_requirement("product_id", final_product_id, "choose-product", 1, "finalize", 2))
            oracle = [
                {"tool": "search_products", "arguments": {"query": "delivery"}},
                {"tool": "search_products", "arguments": {"query": catalog.final_name, "store_id": target_store_id}},
                {"tool": "create_order", "arguments": {"store_id": target_store_id, "product_id": final_product_id, "quantity": final_quantity, "address": final_address, "delivery_time": delivery_time}},
                {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}},
            ]
        elif flow == "memory_accumulate":
            script = [
                initial,
                UserEvent(
                    "add-dietary-preference",
                    "after_tool",
                    f"One more lasting preference: please avoid {first_forbidden} and {second_forbidden}. Continue looking; do not order yet.",
                    updates={"tag_absent": [first_forbidden, second_forbidden]},
                ),
                UserEvent(
                    "choose-product",
                    "after_tool",
                    f"Use the {catalog.final_name} from {target_store_name}.",
                    after_event_id="add-dietary-preference",
                    min_tool_calls=2,
                    updates={"product_id": final_product_id, "store_id": target_store_id},
                ),
                UserEvent(
                    "finalize",
                    "after_tool",
                    f"Now send it to {final_address} for {revised_delivery_time}, then place and pay for the order.",
                    after_event_id="choose-product",
                    min_tool_calls=3,
                    updates={"address": final_address, "paid": True},
                ),
            ]
            requirements.extend((
                cls._memory_requirement("quantity", final_quantity, "persistent-preferences", 0, "finalize", 3),
                cls._memory_requirement("tag_absent", first_forbidden, "add-dietary-preference", 1, "finalize", 3),
                cls._memory_requirement("tag_absent", second_forbidden, "add-dietary-preference", 1, "finalize", 3),
                cls._memory_requirement("product_id", final_product_id, "choose-product", 2, "finalize", 3),
            ))
            oracle = [
                {"tool": "search_products", "arguments": {"query": "delivery"}},
                {"tool": "search_products", "arguments": {"query": catalog.final_name, "store_id": target_store_id}},
                {"tool": "get_product", "arguments": {"product_id": final_product_id}},
                {"tool": "create_order", "arguments": {"store_id": target_store_id, "product_id": final_product_id, "quantity": final_quantity, "address": final_address, "delivery_time": revised_delivery_time}},
                {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}},
            ]
        else:
            script = [
                initial,
                UserEvent(
                    "revise-quantity",
                    "after_tool",
                    f"Actually, make that {final_quantity} unit(s). The other requirements still stand, but do not order yet.",
                    revision_of="quantity",
                    updates={"quantity": final_quantity},
                ),
                UserEvent(
                    "choose-product",
                    "after_tool",
                    f"Use the {catalog.final_name} from {target_store_name}.",
                    after_event_id="revise-quantity",
                    min_tool_calls=2,
                    updates={"product_id": final_product_id, "store_id": target_store_id},
                ),
                UserEvent(
                    "finalize",
                    "after_tool",
                    f"Please send it to {final_address} for {revised_delivery_time}, then place and pay for the final order.",
                    after_event_id="choose-product",
                    min_tool_calls=3,
                    updates={"address": final_address, "paid": True},
                ),
            ]
            requirements.extend((
                cls._memory_requirement("quantity", initial_quantity, "persistent-preferences", 0, "revise-quantity", 1, persistent=False, superseded_by="revise-quantity", required_action=None),
                cls._memory_requirement("quantity", final_quantity, "revise-quantity", 1, "finalize", 3),
                cls._memory_requirement("product_id", final_product_id, "choose-product", 2, "finalize", 3),
            ))
            oracle = [
                {"tool": "search_products", "arguments": {"query": "delivery"}},
                {"tool": "search_products", "arguments": {"query": catalog.final_name, "store_id": target_store_id}},
                {"tool": "get_product", "arguments": {"product_id": final_product_id}},
                {"tool": "create_order", "arguments": {"store_id": target_store_id, "product_id": final_product_id, "quantity": final_quantity, "address": final_address, "delivery_time": revised_delivery_time}},
                {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}},
            ]
        return script, oracle, requirements

    @staticmethod
    def _flow(flow: str, catalog: _Catalog, initial_product_id: str, final_product_id: str, initial_store_id: str, target_store_id: str, initial_store_name: str, target_store_name: str, initial_quantity: int, final_quantity: int, initial_address: str, final_address: str, delivery_time: str, revised_delivery_time: str, budget: int, num_revisions: int) -> tuple[list[UserEvent], list[dict[str, object]]]:
        budget_dollars = f"${budget / 100:.2f}"
        final_requirements = f"avoid {catalog.forbidden_tags[0]} and {catalog.forbidden_tags[1]}"
        initial = UserEvent("initial", "initial", f"Please order {initial_quantity} unit(s) of {catalog.initial_name} for delivery to {initial_address}. Choose a delivery time after the current logical time.")
        if flow == "replace":
            script = [
                initial,
                UserEvent("revise-item", "after_order_created", f"Change of plans: {catalog.revision_reason}. Cancel that order and replace it with {final_quantity} unit(s) of {catalog.final_name} from a suitable store. It must {final_requirements} and cost no more than {budget_dollars} total.", revision_of="item", updates={"product_id": final_product_id, "store_id": target_store_id, "quantity": final_quantity, "tag_absent": list(catalog.forbidden_tags), "max_price_cents": budget}),
                UserEvent("finalize", "after_successful_tool", f"Send the replacement to {final_address} and pay once the final order is correct.", tool_name="cancel_order", revision_of="address", updates={"address": final_address, "paid": True}),
            ]
            oracle = [
                {"tool": "search_products", "arguments": {"query": catalog.initial_name}},
                {"tool": "create_order", "arguments": {"store_id": initial_store_id, "product_id": initial_product_id, "quantity": initial_quantity, "address": initial_address, "delivery_time": delivery_time}},
                {"tool": "search_products", "arguments": {"query": catalog.final_name}},
                {"tool": "cancel_order", "arguments": {"order_id": "mini-order-0001"}},
                {"tool": "create_order", "arguments": {"store_id": target_store_id, "product_id": final_product_id, "quantity": final_quantity, "address": final_address, "delivery_time": delivery_time}},
                {"tool": "pay_order", "arguments": {"order_id": "mini-order-0002"}},
            ]
        elif flow == "modify":
            if num_revisions >= 2:
                script = [
                    initial,
                    UserEvent("revise-line", "after_order_created", f"Please update that same order to {final_quantity} unit(s) of {catalog.final_name}. {catalog.revision_reason}, so it must {final_requirements} and cost at most {budget_dollars}.", revision_of="line", updates={"product_id": final_product_id, "quantity": final_quantity, "tag_absent": list(catalog.forbidden_tags), "max_price_cents": budget}),
                    UserEvent("finalize", "after_successful_tool", f"Now send the updated order to {final_address}, reschedule it for {revised_delivery_time}, and then pay.", tool_name="modify_order", revision_of="delivery", updates={"address": final_address, "paid": True}),
                ]
                final_modify = {"order_id": "mini-order-0001", "address": final_address, "delivery_time": revised_delivery_time}
            else:
                script = [
                    initial,
                    UserEvent("revise-line", "after_order_created", f"Please update that same order to {final_quantity} unit(s) of {catalog.final_name}. {catalog.revision_reason}, so it must {final_requirements} and cost at most {budget_dollars}. Send it to {final_address}, then pay.", revision_of="line", updates={"product_id": final_product_id, "quantity": final_quantity, "address": final_address, "tag_absent": list(catalog.forbidden_tags), "max_price_cents": budget, "paid": True}),
                ]
                final_modify = None
            oracle = [
                {"tool": "search_products", "arguments": {"query": catalog.initial_name}},
                {"tool": "create_order", "arguments": {"store_id": initial_store_id, "product_id": initial_product_id, "quantity": initial_quantity, "address": initial_address, "delivery_time": delivery_time}},
                {"tool": "search_products", "arguments": {"query": catalog.final_name, "store_id": target_store_id}},
                {"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "product_id": final_product_id, "quantity": final_quantity, **({} if final_modify else {"address": final_address})}},
            ]
            if final_modify:
                oracle.append({"tool": "modify_order", "arguments": final_modify})
            oracle.append({"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}})
        elif flow == "reschedule":
            initial = UserEvent("initial", "initial", f"Please order {final_quantity} unit(s) of {catalog.final_name} from {target_store_name} for delivery to {initial_address}. It must {final_requirements} and cost no more than {budget_dollars}. Choose a future delivery time.")
            if num_revisions >= 2:
                script = [
                    initial,
                    UserEvent("reschedule", "after_order_created", f"Keep the delivery address for now, but reschedule the order for {revised_delivery_time}.", revision_of="delivery"),
                    UserEvent("finalize", "after_successful_tool", f"Please move the rescheduled order to {final_address}, then pay.", tool_name="modify_order", revision_of="address", updates={"address": final_address, "paid": True}),
                ]
                first_modify = {"order_id": "mini-order-0001", "delivery_time": revised_delivery_time}
            else:
                script = [initial, UserEvent("reschedule", "after_order_created", f"Please move the order to {final_address}, reschedule it for {revised_delivery_time}, and then pay.", revision_of="delivery", updates={"address": final_address, "paid": True})]
                first_modify = {"order_id": "mini-order-0001", "address": final_address, "delivery_time": revised_delivery_time}
            oracle = [
                {"tool": "search_products", "arguments": {"query": catalog.final_name}},
                {"tool": "create_order", "arguments": {"store_id": target_store_id, "product_id": final_product_id, "quantity": final_quantity, "address": initial_address, "delivery_time": delivery_time}},
                {"tool": "modify_order", "arguments": first_modify},
            ]
            if num_revisions >= 2:
                oracle.append({"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "address": final_address}})
            oracle.append({"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}})
        else:
            initial = UserEvent("initial", "initial", f"Please order {final_quantity} unit(s) of {catalog.final_name} from {target_store_name} for delivery to {initial_address}. It must {final_requirements} and cost no more than {budget_dollars}. Choose a future delivery time.")
            script = [initial]
            oracle = [{"tool": "search_products", "arguments": {"query": catalog.final_name}}, {"tool": "create_order", "arguments": {"store_id": target_store_id, "product_id": final_product_id, "quantity": final_quantity, "address": final_address if num_revisions == 0 else initial_address, "delivery_time": delivery_time}}]
            if num_revisions >= 2:
                script.extend((
                    UserEvent("reschedule", "after_order_created", f"Please reschedule it for {revised_delivery_time}; keep the address unchanged for now.", revision_of="delivery"),
                    UserEvent("finalize", "after_successful_tool", f"Please send it to {final_address} instead, then pay for the final order.", tool_name="modify_order", revision_of="address", updates={"address": final_address, "paid": True}),
                ))
                oracle.extend((
                    {"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "delivery_time": revised_delivery_time}},
                    {"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "address": final_address}},
                    {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}},
                ))
            elif num_revisions:
                script.append(UserEvent("finalize", "after_order_created", f"Please send it to {final_address} instead, then pay for the final order.", revision_of="address", updates={"address": final_address, "paid": True}))
                oracle.extend(({"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "address": final_address}}, {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}}))
            else:
                script[0] = UserEvent("initial", "initial", initial.message + " Please pay for the order now.", updates={"paid": True})
                oracle.append({"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}})
        return script, oracle

    def _canonical_delivery_revision(self, task_id: str, difficulty: DifficultyConfig) -> MiniTask:
        """Keep the original cold-brew fixture stable for documentation/tests."""
        rng = random.Random(self.seed)
        target_store, target_product = "store-target", "product-herbal-tea"
        stores = {target_store: Store(target_store, "Garden Delivery", ("tea", "healthy", "delivery"), True)}
        products = {
            target_product: Product(target_product, target_store, "Herbal Tea", rng.randrange(375, 501, 25), ("tea", "caffeine-free", "non-spicy"), 20),
            "product-cold-brew": Product("product-cold-brew", target_store, "Cold Brew", rng.randrange(375, 501, 25), ("coffee", "caffeine", "cold"), 20),
        }
        self._add_distractors(rng, stores, products, difficulty.num_distractors)
        database = DatabaseState(stores=stores, products=products, orders={}, locations={"home": "1 Home Street", "office": "2 Office Plaza"}, current_time="2026-01-01 09:00:00", next_order_number=1)
        constraints = [
            Constraint("correct_product", "product_id", target_product), Constraint("correct_store", "store_id", target_store), Constraint("quantity_two", "quantity", 2), Constraint("office_address", "address", "office"), Constraint("replacement_paid", "paid", True), Constraint("caffeine_free", "tag_absent", "caffeine"), Constraint("non_spicy", "tag_absent", "spicy"), Constraint("under_budget", "max_price_cents", 1000),
        ][: max(1, difficulty.num_constraints)]
        script = [
            UserEvent("initial", "initial", "Please order two cold brews for delivery to home. If a delivery time is needed, choose a reasonable time after the current logical time."),
            UserEvent("revise-drink", "after_successful_tool", "Actually, I cannot have caffeine. Please use a non-spicy herbal tea instead, and keep the final total at or under $10.", tool_name="search_products", revision_of="drink", updates={"product_id": target_product, "tag_absent": ["caffeine", "spicy"], "max_price_cents": 1000}),
            UserEvent("revise-address", "after_order_created", "Please send it to the office, not home, and then pay for the final order.", revision_of="address", updates={"address": "office", "paid": True}),
        ]
        return MiniTask(task_id=task_id, initial_state=database, objectives=tuple("create and pay a delivery order" for _ in range(max(1, difficulty.num_objectives))), latent_constraints=tuple(constraints), user_script=tuple(script), difficulty={name: getattr(difficulty, name) for name in difficulty.__dataclass_fields__}, oracle_solution=(
            {"tool": "search_products", "arguments": {"query": "coffee"}}, {"tool": "search_products", "arguments": {"query": "herbal tea"}}, {"tool": "create_order", "arguments": {"store_id": target_store, "product_id": target_product, "quantity": 2, "address": "home", "delivery_time": "2026-01-01 12:00:00"}}, {"tool": "modify_order", "arguments": {"order_id": "mini-order-0001", "address": "office"}}, {"tool": "pay_order", "arguments": {"order_id": "mini-order-0001"}},
        ))


def built_in_tasks() -> dict[str, MiniTask]:
    """Small stable fixture set; generated tasks supply the diversity."""
    return {"delivery_revision": TaskGenerator(seed=7).generate("delivery_revision")}
