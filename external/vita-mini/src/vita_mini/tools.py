"""Deterministic delivery-domain tools and OpenAI function schemas."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from .types import DatabaseState, DeliveryOrder, Product


class ToolError(ValueError):
    """Expected, agent-visible tool failure."""


def _schema(name: str, description: str, properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(required), "additionalProperties": False,
    }}}


class DeliveryTools:
    """The entire agent-visible API for one deterministic delivery database."""

    def __init__(self, database: DatabaseState, user_id: str) -> None:
        self.database = database
        self.user_id = user_id
        self._handlers: dict[str, Callable[..., Any]] = {
            "search_stores": self.search_stores,
            "search_products": self.search_products,
            "get_store": self.get_store,
            "get_product": self.get_product,
            "create_order": self.create_order,
            "modify_order": self.modify_order,
            "cancel_order": self.cancel_order,
            "pay_order": self.pay_order,
            "get_order": self.get_order,
        }

    def schemas(self) -> list[dict[str, Any]]:
        string = {"type": "string"}
        positive = {"type": "integer", "minimum": 1}
        return [
            _schema("search_stores", "Search delivery stores by words in their name or tags.", {"query": {**string, "minLength": 1}}, ("query",)),
            _schema("search_products", "Search products by words in product name or tags. Optionally restrict to one store.", {"query": {**string, "minLength": 1}, "store_id": string}, ("query",)),
            _schema("get_store", "Get one store's details and product identifiers.", {"store_id": string}, ("store_id",)),
            _schema("get_product", "Get one product's details, tags, price, and current inventory.", {"product_id": string}, ("product_id",)),
            _schema("create_order", "Create one unpaid delivery order. Product, store, quantity, address, and delivery time must be explicit.", {"store_id": string, "product_id": string, "quantity": positive, "address": {**string, "minLength": 1}, "delivery_time": {**string, "minLength": 1}}, ("store_id", "product_id", "quantity", "address", "delivery_time")),
            _schema("modify_order", "Modify an existing unpaid order. Supply at least one field to change.", {"order_id": string, "product_id": string, "quantity": positive, "address": {**string, "minLength": 1}, "delivery_time": {**string, "minLength": 1}}, ("order_id",)),
            _schema("cancel_order", "Cancel an existing unpaid or paid order and restore inventory.", {"order_id": string}, ("order_id",)),
            _schema("pay_order", "Pay an existing unpaid order after checking its final details.", {"order_id": string}, ("order_id",)),
            _schema("get_order", "Get one of the active user's orders.", {"order_id": string}, ("order_id",)),
        ]

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._handlers:
            raise ToolError(f"Unknown tool {name!r}")
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object")
        try:
            return self._handlers[name](**arguments)
        except TypeError as exc:
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc

    def search_stores(self, query: str) -> list[dict[str, Any]]:
        words = _words(query)
        return [self._store_view(store_id) for store_id, store in sorted(self.database.stores.items())
                if words.issubset(_words(" ".join((store.name, *store.tags))))]

    def search_products(self, query: str, store_id: str | None = None) -> list[dict[str, Any]]:
        words = _words(query)
        if store_id is not None:
            self._store(store_id)
        return [self._product_view(product) for product_id, product in sorted(self.database.products.items())
                if (store_id is None or product.store_id == store_id)
                and words.issubset(_words(" ".join((product.name, *product.tags))))]

    def get_store(self, store_id: str) -> dict[str, Any]:
        return self._store_view(store_id)

    def get_product(self, product_id: str) -> dict[str, Any]:
        return self._product_view(self._product(product_id))

    def create_order(self, store_id: str, product_id: str, quantity: int, address: str, delivery_time: str) -> dict[str, Any]:
        product = self._validate_line(store_id, product_id, quantity)
        self._validate_address_time(address, delivery_time)
        self._set_inventory(product_id, product.inventory - quantity)
        order = DeliveryOrder(
            order_id=f"mini-order-{self.database.next_order_number:04d}", user_id=self.user_id,
            store_id=store_id, product_id=product_id, quantity=quantity, address=address.strip(),
            delivery_time=delivery_time.strip(), total_cents=product.price_cents * quantity, status="unpaid",
        )
        self.database.next_order_number += 1
        self.database.orders[order.order_id] = order
        return self._order_view(order)

    def modify_order(self, order_id: str, product_id: str | None = None, quantity: int | None = None, address: str | None = None, delivery_time: str | None = None) -> dict[str, Any]:
        order = self._order(order_id)
        if order.status != "unpaid":
            raise ToolError("Only unpaid orders can be modified")
        if all(value is None for value in (product_id, quantity, address, delivery_time)):
            raise ToolError("Provide product_id, quantity, address, or delivery_time")
        new_product_id = product_id if product_id is not None else order.product_id
        new_quantity = quantity if quantity is not None else order.quantity
        # Restore old inventory before validating the replacement line.
        old_product = self._product(order.product_id)
        self._set_inventory(order.product_id, old_product.inventory + order.quantity)
        try:
            replacement = self._validate_line(order.store_id, new_product_id, new_quantity)
        except Exception:
            self._set_inventory(order.product_id, old_product.inventory)
            raise
        if address is not None:
            self._validate_address_time(address, order.delivery_time)
            order.address = address.strip()
        if delivery_time is not None:
            self._validate_address_time(order.address, delivery_time)
            order.delivery_time = delivery_time.strip()
        self._set_inventory(new_product_id, replacement.inventory - new_quantity)
        order.product_id, order.quantity = new_product_id, new_quantity
        order.total_cents = replacement.price_cents * new_quantity
        return self._order_view(order)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order.status == "cancelled":
            raise ToolError(f"Order {order_id} is already cancelled")
        product = self._product(order.product_id)
        self._set_inventory(product.product_id, product.inventory + order.quantity)
        order.status = "cancelled"
        return self._order_view(order)

    def pay_order(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order.status != "unpaid":
            raise ToolError(f"Only unpaid orders can be paid; status is {order.status}")
        order.status = "paid"
        return self._order_view(order)

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._order_view(self._order(order_id))

    def _validate_line(self, store_id: str, product_id: str, quantity: int) -> Product:
        self._store(store_id)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise ToolError("quantity must be a positive integer")
        product = self._product(product_id)
        if product.store_id != store_id:
            raise ToolError("product_id does not belong to store_id")
        if product.inventory < quantity:
            raise ToolError(f"Only {product.inventory} unit(s) of {product_id} are available")
        return product

    def _validate_address_time(self, address: str, delivery_time: str) -> None:
        if not isinstance(address, str) or address.strip() not in self.database.locations:
            raise ToolError("address must be one of the known location names")
        if not isinstance(delivery_time, str) or delivery_time <= self.database.current_time:
            raise ToolError("delivery_time must be after the current logical time")

    def _store(self, store_id: str):
        if not isinstance(store_id, str) or store_id not in self.database.stores:
            raise ToolError(f"Unknown store {store_id!r}")
        return self.database.stores[store_id]

    def _product(self, product_id: str) -> Product:
        if not isinstance(product_id, str) or product_id not in self.database.products:
            raise ToolError(f"Unknown product {product_id!r}")
        return self.database.products[product_id]

    def _order(self, order_id: str) -> DeliveryOrder:
        if not isinstance(order_id, str) or order_id not in self.database.orders:
            raise ToolError(f"Unknown order {order_id!r}")
        order = self.database.orders[order_id]
        if order.user_id != self.user_id:
            raise ToolError("Order does not belong to the active user")
        return order

    def _set_inventory(self, product_id: str, inventory: int) -> None:
        product = self._product(product_id)
        self.database.products[product_id] = Product(**{**asdict(product), "inventory": inventory})

    def _store_view(self, store_id: str) -> dict[str, Any]:
        store = self._store(store_id)
        return {**asdict(store), "tags": list(store.tags), "product_ids": sorted(product.product_id for product in self.database.products.values() if product.store_id == store_id)}

    @staticmethod
    def _product_view(product: Product) -> dict[str, Any]:
        return {**asdict(product), "tags": list(product.tags)}

    @staticmethod
    def _order_view(order: DeliveryOrder) -> dict[str, Any]:
        return asdict(order)


def _words(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        raise ToolError("query must be a non-empty string")
    return set(text.lower().replace("-", " ").split())
