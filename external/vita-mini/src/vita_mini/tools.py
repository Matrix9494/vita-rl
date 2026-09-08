"""Business tools and their OpenAI-compatible schema definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .types import CartLine, Order, Product


class ToolError(ValueError):
    """An expected, agent-visible failure caused by invalid tool use."""


def _function_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


class StorefrontTools:
    """Stateful tools available to an agent during a storefront episode."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self._handlers: dict[str, Callable[..., Any]] = {
            "get_account": self.get_account,
            "update_account_profile": self.update_account_profile,
            "search_catalog": self.search_catalog,
            "get_product": self.get_product,
            "get_cart": self.get_cart,
            "add_to_cart": self.add_to_cart,
            "remove_from_cart": self.remove_from_cart,
            "set_delivery_address": self.set_delivery_address,
            "checkout": self.checkout,
            "list_orders": self.list_orders,
            "get_order": self.get_order,
            "cancel_order": self.cancel_order,
            "get_current_time": self.get_current_time,
        }

    @property
    def names(self) -> list[str]:
        return list(self._handlers)

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._handlers:
            raise ToolError(f"Unknown tool {name!r}")
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object")
        try:
            return self._handlers[name](**arguments)
        except TypeError as exc:
            # TypeError is usually an unknown/missing tool argument.  Do not
            # expose an implementation traceback to an LLM.
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        string = {"type": "string"}
        positive_int = {"type": "integer", "minimum": 1}
        return [
            _function_schema("get_account", "Get the active customer account." , {}),
            _function_schema(
                "update_account_profile", "Update one or both mutable account fields.",
                {"display_name": string, "email": {**string, "format": "email"}},
            ),
            _function_schema(
                "search_catalog", "Find available products by case-insensitive words in their name, category, or tags.",
                {"query": {**string, "minLength": 1}, "category": string}, ["query"],
            ),
            _function_schema("get_product", "Get one product and its current inventory.", {"product_id": string}, ["product_id"]),
            _function_schema("get_cart", "Get the active cart, line items, and subtotal.", {}),
            _function_schema(
                "add_to_cart", "Add an in-stock product to the active cart.",
                {"product_id": string, "quantity": positive_int}, ["product_id", "quantity"],
            ),
            _function_schema("remove_from_cart", "Remove a product line from the active cart.", {"product_id": string}, ["product_id"]),
            _function_schema("set_delivery_address", "Set the non-empty delivery address used at checkout.", {"address": {**string, "minLength": 1}}, ["address"]),
            _function_schema("checkout", "Submit the non-empty cart as one order. A delivery address is required.", {}),
            _function_schema("list_orders", "List every order owned by the active customer.", {}),
            _function_schema("get_order", "Get an order owned by the active customer.", {"order_id": string}, ["order_id"]),
            _function_schema("cancel_order", "Cancel a submitted order and restore its inventory.", {"order_id": string}, ["order_id"]),
            _function_schema("get_current_time", "Get the fixed logical time for this episode.", {}),
        ]

    def get_account(self) -> dict[str, Any]:
        return deepcopy(self._state["account"])

    def update_account_profile(
        self, display_name: str | None = None, email: str | None = None
    ) -> dict[str, Any]:
        if display_name is None and email is None:
            raise ToolError("Provide display_name, email, or both")
        if display_name is not None:
            if not isinstance(display_name, str) or not display_name.strip():
                raise ToolError("display_name must be a non-empty string")
            self._state["account"]["display_name"] = display_name.strip()
        if email is not None:
            if not isinstance(email, str) or "@" not in email or not email.strip():
                raise ToolError("email must be a valid non-empty email address")
            self._state["account"]["email"] = email.strip()
        return self.get_account()

    def search_catalog(self, query: str, category: str | None = None) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        words = set(query.lower().split())
        normalized_category = category.lower().strip() if category else None
        matches = []
        for product in self._products().values():
            searchable = " ".join((product.name, product.category, *product.tags)).lower()
            if words.issubset(set(searchable.split())) and (
                normalized_category is None or product.category.lower() == normalized_category
            ):
                matches.append(self._product_view(product))
        return matches

    def get_product(self, product_id: str) -> dict[str, Any]:
        return self._product_view(self._product(product_id))

    def get_cart(self) -> dict[str, Any]:
        lines = []
        subtotal_cents = 0
        for line in self._state["cart"]:
            product = self._product(line.product_id)
            subtotal_cents += product.price_cents * line.quantity
            lines.append({**self._product_view(product), "quantity": line.quantity, "line_total_cents": product.price_cents * line.quantity})
        return {"lines": lines, "subtotal_cents": subtotal_cents, "delivery_address": self._state["delivery_address"]}

    def add_to_cart(self, product_id: str, quantity: int) -> dict[str, Any]:
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise ToolError("quantity must be a positive integer")
        product = self._product(product_id)
        existing = next((line for line in self._state["cart"] if line.product_id == product_id), None)
        requested = quantity + (existing.quantity if existing else 0)
        if requested > product.inventory:
            raise ToolError(f"Only {product.inventory} unit(s) of {product_id} are available")
        if existing:
            existing.quantity = requested
        else:
            self._state["cart"].append(CartLine(product_id=product_id, quantity=quantity))
        return self.get_cart()

    def remove_from_cart(self, product_id: str) -> dict[str, Any]:
        for index, line in enumerate(self._state["cart"]):
            if line.product_id == product_id:
                self._state["cart"].pop(index)
                return self.get_cart()
        raise ToolError(f"Product {product_id!r} is not in the cart")

    def set_delivery_address(self, address: str) -> dict[str, Any]:
        if not isinstance(address, str) or not address.strip():
            raise ToolError("address must be a non-empty string")
        self._state["delivery_address"] = address.strip()
        return {"delivery_address": self._state["delivery_address"]}

    def checkout(self) -> dict[str, Any]:
        if not self._state["cart"]:
            raise ToolError("Cannot checkout an empty cart")
        if not self._state["delivery_address"]:
            raise ToolError("Set a delivery address before checkout")
        cart = self.get_cart()
        for line in self._state["cart"]:
            product = self._product(line.product_id)
            if line.quantity > product.inventory:
                raise ToolError(f"Inventory changed; only {product.inventory} unit(s) of {product.product_id} remain")
        for line in self._state["cart"]:
            self._state["catalog"][line.product_id] = Product(
                **{**product_to_dict(self._product(line.product_id)), "inventory": self._product(line.product_id).inventory - line.quantity}
            )
        order = Order(
            order_id=f"mini-order-{self._state['next_order_number']:04d}",
            user_id=self._state["account"]["user_id"],
            lines=deepcopy(self._state["cart"]),
            total_cents=cart["subtotal_cents"],
            delivery_address=self._state["delivery_address"],
            status="submitted",
            created_at=self._state["logical_time"],
        )
        self._state["next_order_number"] += 1
        self._state["orders"][order.order_id] = order
        self._state["cart"] = []
        return self._order_view(order)

    def list_orders(self) -> list[dict[str, Any]]:
        return [self._order_view(order) for order in self._state["orders"].values()]

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._order_view(self._order(order_id))

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order.status == "cancelled":
            raise ToolError(f"Order {order_id} is already cancelled")
        order.status = "cancelled"
        for line in order.lines:
            product = self._product(line.product_id)
            self._state["catalog"][line.product_id] = Product(
                **{**product_to_dict(product), "inventory": product.inventory + line.quantity}
            )
        return self._order_view(order)

    def get_current_time(self) -> dict[str, str]:
        return {"logical_time": self._state["logical_time"]}

    def _products(self) -> dict[str, Product]:
        return self._state["catalog"]

    def _product(self, product_id: str) -> Product:
        if not isinstance(product_id, str) or product_id not in self._products():
            raise ToolError(f"Unknown product {product_id!r}")
        return self._products()[product_id]

    def _order(self, order_id: str) -> Order:
        if not isinstance(order_id, str) or order_id not in self._state["orders"]:
            raise ToolError(f"Unknown order {order_id!r}")
        order = self._state["orders"][order_id]
        if order.user_id != self._state["account"]["user_id"]:
            raise ToolError(f"Order {order_id!r} does not belong to the active customer")
        return order

    @staticmethod
    def _product_view(product: Product) -> dict[str, Any]:
        return product_to_dict(product)

    def _order_view(self, order: Order) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "lines": [
                {"product_id": line.product_id, "quantity": line.quantity}
                for line in order.lines
            ],
            "total_cents": order.total_cents,
            "delivery_address": order.delivery_address,
            "status": order.status,
            "created_at": order.created_at,
        }


def product_to_dict(product: Product) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "name": product.name,
        "category": product.category,
        "price_cents": product.price_cents,
        "inventory": product.inventory,
        "tags": list(product.tags),
    }
