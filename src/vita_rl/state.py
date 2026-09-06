"""Observable-only structured task tracking for VitaBench harnesses.

This module deliberately receives only messages visible to the agent: user
text, assistant tool calls, and tool responses. It never receives task,
simulator, or evaluator objects. The tracker is incremental: each event is
applied to the state it already holds.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ConstraintStatus = Literal["known", "satisfied", "violated", "unknown"]
SubgoalStatus = Literal["pending", "active", "done", "failed"]


@dataclass
class Constraint:
    id: str
    domain: str
    subject: str
    field: str
    desired_value: Any
    source_turn: int
    source_text: str
    status: ConstraintStatus = "known"
    evidence: list[str] = field(default_factory=list)


@dataclass
class Entity:
    entity_id: str
    entity_type: str
    domain: str
    name: str | None = None
    observed_attributes: dict[str, Any] = field(default_factory=dict)
    discovered_turn: int = 0
    valid_tool_namespace: str = ""


@dataclass
class Transaction:
    transaction_id: str
    transaction_type: str
    entity_id: str | None
    items: list[str] = field(default_factory=list)
    quantities: dict[str, int] = field(default_factory=dict)
    date_time: str | None = None
    location: str | None = None
    status: str = "pending_confirmation"
    payment_status: str = "unknown"
    latest_evidence_turn: int = 0


@dataclass
class Subgoal:
    id: str
    domain: str
    description: str
    status: SubgoalStatus = "pending"


@dataclass
class OpenQuestion:
    subject: str
    field: str
    reason: str
    source_turn: int
    source_text: str = ""


@dataclass
class TerminationState:
    unresolved_constraints: int = 0
    violated_constraints: int = 0
    unresolved_subgoals: int = 0
    failed_transactions: int = 0
    open_questions: int = 0
    can_stop: bool = False


@dataclass
class TaskState:
    """Long-lived, typed state derived only from agent-visible events."""

    turn: int = 0
    constraints: dict[str, Constraint] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    transactions: dict[str, Transaction] = field(default_factory=dict)
    subgoals: dict[str, Subgoal] = field(default_factory=dict)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    termination: TerminationState = field(default_factory=TerminationState)
    debug_traces: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot without recursively copying traces."""
        payload = asdict(self)
        payload.pop("debug_traces", None)
        return payload

    @staticmethod
    def _normal(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    @staticmethod
    def _domain_for_text(text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ("hotel", "flight", "train", "attraction", "ticket", "room", "seat")):
            return "ota"
        if any(word in lowered for word in ("restaurant", "reservation", "book a table", "shop")):
            return "instore"
        if any(word in lowered for word in ("delivery", "deliver", "order", "store", "food", "hat")):
            return "delivery"
        return "unknown"

    @staticmethod
    def _domain_for_tool(name: str) -> str:
        lowered = name.lower()
        if "delivery" in lowered:
            return "delivery"
        if "instore" in lowered or "in_store" in lowered:
            return "instore"
        if any(token in lowered for token in ("hotel", "attraction", "flight", "train", "ota_")):
            return "ota"
        return "unknown"

    @staticmethod
    def _entity_type(argument_name: str) -> str | None:
        return {
            "store_id": "store", "shop_id": "shop", "hotel_id": "hotel",
            "attraction_id": "attraction", "flight_id": "flight", "train_id": "train",
            "product_id": "product", "room_id": "room", "ticket_id": "ticket",
            "seat_id": "seat",
        }.get(argument_name)

    def _upsert_constraint(
        self, *, domain: str, subject: str, field_name: str, desired_value: Any,
        source_turn: int, source_text: str,
    ) -> Constraint:
        """Add a requirement without deleting an unrelated earlier one."""
        key = f"{domain}:{self._normal(subject)}:{field_name}"
        existing = self.constraints.get(key)
        if existing is not None:
            if existing.desired_value != desired_value:
                explicitly_superseded = bool(re.search(
                    r"\b(?:instead|replace|change|update|rather than)\b", source_text,
                    re.IGNORECASE,
                ))
                if explicitly_superseded:
                    existing.evidence.append(
                        f"turn {source_turn}: superseded {existing.desired_value!r} with {desired_value!r}"
                    )
                    existing.desired_value = desired_value
                    existing.source_turn = source_turn
                    existing.source_text = source_text
                    existing.status = "known"
                else:
                    # Two independently stated requirements must coexist. A
                    # later user turn is not evidence that an earlier one was
                    # withdrawn.
                    key = f"{key}:turn{source_turn}"
                    constraint = Constraint(
                        id=key, domain=domain, subject=subject, field=field_name,
                        desired_value=desired_value, source_turn=source_turn,
                        source_text=source_text,
                    )
                    self.constraints[key] = constraint
                    return constraint
            return existing
        constraint = Constraint(
            id=key, domain=domain, subject=subject, field=field_name,
            desired_value=desired_value, source_turn=source_turn, source_text=source_text,
        )
        self.constraints[key] = constraint
        return constraint

    def _upsert_subgoal(self, domain: str, description: str) -> Subgoal:
        identifier = f"{domain}:{self._normal(description)}"
        if identifier not in self.subgoals:
            self.subgoals[identifier] = Subgoal(
                id=identifier, domain=domain, description=description
            )
        return self.subgoals[identifier]

    def _add_open_question(self, text: str, turn: int) -> None:
        if any(question.source_text == text for question in self.open_questions):
            return
        self.open_questions.append(OpenQuestion(
            subject="user_request", field="unparsed_requirement",
            reason="No deterministic extraction rule matched this request.",
            source_turn=turn, source_text=text,
        ))

    def _extract_user_constraints(self, text: str, turn: int) -> None:
        """Conservatively extract explicit request fields; never guess values."""
        domain = self._domain_for_text(text)
        found = False
        quantity_match = re.search(
            r"\b(?:need|want|order|place(?:\s+the\s+order\s+for)?|book|reserve|buy|get)\s+(\d+)\s+([a-z][a-z0-9 _-]{1,60}?)(?=\s*(?:,|\.|;|for\b|to\b|with\b|on\b|at\b|$))",
            text, re.IGNORECASE,
        )
        if quantity_match:
            quantity = int(quantity_match.group(1))
            subject = quantity_match.group(2).strip().rstrip("s")
            self._upsert_constraint(domain=domain, subject=subject, field_name="quantity", desired_value=quantity, source_turn=turn, source_text=text)
            self._upsert_constraint(domain=domain, subject=subject, field_name="item", desired_value=subject, source_turn=turn, source_text=text)
            self._upsert_subgoal(domain, f"{subject} request")
            found = True

        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if date_match:
            subject = "hotel" if "hotel" in text.lower() else "booking"
            self._upsert_constraint(domain=domain, subject=subject, field_name="date", desired_value=date_match.group(1), source_turn=turn, source_text=text)
            self._upsert_subgoal(domain, f"{subject} booking")
            found = True
        month_date = re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b",
            text, re.IGNORECASE,
        )
        if month_date and not date_match:
            self._upsert_constraint(domain=domain, subject="booking", field_name="date", desired_value=month_date.group(0), source_turn=turn, source_text=text)
            self._upsert_subgoal(domain, "booking")
            found = True
        time_match = re.search(r"\b(before|after|around|at)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", text, re.IGNORECASE)
        if time_match:
            self._upsert_constraint(domain=domain, subject="schedule", field_name=f"time_{time_match.group(1).lower()}", desired_value=time_match.group(2).strip(), source_turn=turn, source_text=text)
            found = True
        party_match = re.search(r"\b(?:for|party of)\s+(\d+)\s+(?:people|persons|guests)\b", text, re.IGNORECASE)
        if party_match:
            self._upsert_constraint(domain=domain, subject="reservation", field_name="party_size", desired_value=int(party_match.group(1)), source_turn=turn, source_text=text)
            self._upsert_subgoal(domain, "reservation")
            found = True
        address_match = re.search(r"\b(?:deliver(?:y)?\s+(?:it\s+)?to|address(?:\s+is)?|to)\s+([A-Z][^,.!?;]{3,120})", text)
        if address_match:
            self._upsert_constraint(domain=domain, subject="delivery", field_name="address", desired_value=address_match.group(1).strip(), source_turn=turn, source_text=text)
            self._upsert_subgoal(domain, "delivery supplies")
            found = True
        type_match = re.search(r"\b([a-z][a-z -]{1,30})\s+(room|ticket|seat)\b", text, re.IGNORECASE)
        if type_match and any(marker in text.lower() for marker in ("need", "want", "book", "reserve")):
            kind = type_match.group(2).lower()
            self._upsert_constraint(domain=domain, subject=kind, field_name=f"{kind}_type", desired_value=type_match.group(1).strip(), source_turn=turn, source_text=text)
            self._upsert_subgoal(domain, f"{kind} booking")
            found = True
        preference_match = re.search(r"\b(?:from|at)\s+([A-Z][A-Za-z0-9 &' -]{2,80})\b", text)
        if preference_match and any(noun in text.lower() for noun in ("hotel", "store", "shop", "restaurant")):
            self._upsert_constraint(domain=domain, subject="provider", field_name="preference", desired_value=preference_match.group(1).strip(), source_turn=turn, source_text=text)
            found = True
        required_attribute = re.search(r"\b(?:with|must have|must be|without|avoid)\s+([a-z][a-z0-9 -]{2,80})", text, re.IGNORECASE)
        if required_attribute:
            requirement = required_attribute.group(0).strip()
            self._upsert_constraint(domain=domain, subject=f"attribute:{requirement}", field_name="required_attribute", desired_value=requirement, source_turn=turn, source_text=text)
            found = True
        if re.search(r"\b(?:pay|paid|payment)\b", text, re.IGNORECASE):
            self._upsert_constraint(domain=domain, subject="transaction", field_name="payment_status", desired_value="paid", source_turn=turn, source_text=text)
            found = True
        if re.search(r"\b(?:cancel|cancellation)\b", text, re.IGNORECASE):
            self._upsert_constraint(domain=domain, subject="transaction", field_name="cancellation", desired_value="cancelled", source_turn=turn, source_text=text)
            found = True
        if not found and text.strip():
            self._add_open_question(text, turn)

    def _register_entities(self, arguments: dict[str, Any], domain: str, turn: int) -> None:
        for argument_name, value in arguments.items():
            entity_type = self._entity_type(argument_name)
            if entity_type is None or not isinstance(value, str) or not value:
                continue
            entity_domain = "delivery" if entity_type == "store" else domain
            existing = self.entities.get(value)
            if existing is None:
                self.entities[value] = Entity(
                    entity_id=value, entity_type=entity_type, domain=entity_domain,
                    discovered_turn=turn, valid_tool_namespace=entity_domain,
                )
            else:
                existing.observed_attributes.setdefault("seen_in", []).append(argument_name)

    def _register_entities_from_content(self, content: str, domain: str, turn: int) -> None:
        """Register IDs exposed by a successful, agent-visible tool result."""
        for argument_name, entity_type in (
            ("store_id", "store"), ("shop_id", "shop"), ("hotel_id", "hotel"),
            ("attraction_id", "attraction"), ("flight_id", "flight"),
            ("train_id", "train"), ("product_id", "product"), ("room_id", "room"),
            ("ticket_id", "ticket"), ("seat_id", "seat"),
        ):
            for entity_id in re.findall(rf"\b{argument_name}\s*[:=]\s*['\"]?([^,\s\)\]'\"]+)", content):
                entity_domain = "delivery" if entity_type == "store" else domain
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        entity_id=entity_id, entity_type=entity_type,
                        domain=entity_domain, discovered_turn=turn,
                        valid_tool_namespace=entity_domain,
                    )

    @staticmethod
    def _is_success(result: dict[str, Any]) -> bool:
        content = str(result.get("content") or "").lower()
        # VitaBench's environment occasionally returns assertion text with
        # ``error=False``. Treat those explicit rejection phrases as failures:
        # only a confirmed tool result may create/update world state.
        return not result.get("error") and not any(token in content for token in (
            "error:", "failed", "not found", "cannot ", "must be ",
            "invalid", "does not ", "insufficient", "no available", "already exists",
        ))

    @staticmethod
    def _order_id(content: str) -> str | None:
        match = re.search(r"(?:order_id|book_id|reservation_id)\s*[:=]\s*['\"]?([^,'\s)]+)", content)
        if match:
            return match.group(1)
        match = re.search(r"\b(?:order|booking|reservation)\s+([A-Za-z0-9_-]+)\b", content, re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _confirmed_items(content: str, fallback_items: list[str], fallback_quantities: dict[str, int]) -> tuple[list[str], dict[str, int]]:
        """Prefer product names/quantities present in a successful tool repr."""
        pairs = re.findall(
            r"(?:\bproduct_name|\bname)\s*[:=]\s*([^,\]\)]+).*?quantity\s*[:=]\s*(\d+)",
            content, flags=re.IGNORECASE | re.DOTALL,
        )
        if not pairs:
            return fallback_items, fallback_quantities
        items = [name.strip().strip("'\"") for name, _ in pairs]
        quantities = {item: int(quantity) for item, quantity in pairs}
        return items, quantities

    def _transaction_from_create(self, name: str, arguments: dict[str, Any], result: dict[str, Any], turn: int) -> None:
        domain = self._domain_for_tool(name)
        transaction_type = name.removeprefix("create_").removesuffix("_order")
        goal = self._upsert_subgoal(domain, transaction_type.replace("_", " "))
        if not self._is_success(result):
            goal.status = "failed"
            return
        raw = str(result.get("content") or "")
        transaction_id = self._order_id(raw) or f"{transaction_type}:{turn}"
        entity_id = next((arguments[key] for key in ("store_id", "shop_id", "hotel_id", "attraction_id", "flight_id", "train_id") if arguments.get(key)), None)
        item_values = arguments.get("product_ids") or [arguments[key] for key in ("product_id", "room_id", "ticket_id", "seat_id") if arguments.get(key)]
        items = [str(item) for item in item_values]
        quantities: dict[str, int] = {}
        counts = arguments.get("product_cnts")
        if isinstance(counts, list):
            quantities.update({item: int(count) for item, count in zip(items, counts)})
        elif items and isinstance(arguments.get("quantity"), int):
            quantities[items[0]] = int(arguments["quantity"])
        elif items:
            quantities[items[0]] = 1
        if isinstance(arguments.get("customer_count"), int):
            quantities["party_size"] = int(arguments["customer_count"])
        items, quantities = self._confirmed_items(raw, items, quantities)
        transaction = Transaction(
            transaction_id=transaction_id, transaction_type=transaction_type, entity_id=entity_id,
            items=items, quantities=quantities,
            date_time=str(arguments.get("date") or arguments.get("time") or arguments.get("dispatch_time") or "") or None,
            location=str(arguments.get("address") or "") or None,
            status="created", payment_status="unpaid", latest_evidence_turn=turn,
        )
        status_match = re.search(r"status\s*=\s*['\"]([^'\"]+)", raw)
        if status_match:
            transaction.status = status_match.group(1)
            transaction.payment_status = "paid" if transaction.status == "paid" else transaction.status
        self.transactions[transaction_id] = transaction
        goal.status = "done" if transaction.status not in ("failed", "cancelled") else "failed"
        self._update_subgoals_from_transaction(transaction)

    def _update_subgoals_from_transaction(self, transaction: Transaction) -> None:
        """Mark only clearly corresponding user subgoals as complete."""
        item_words = {self._normal(item) for item in transaction.items}
        for goal in self.subgoals.values():
            if goal.domain != self._domain_for_tool(
                "delivery" if transaction.transaction_type == "delivery" else transaction.transaction_type
            ):
                continue
            description = self._normal(goal.description)
            type_matches = transaction.transaction_type in description
            item_matches = any(item and item in description for item in item_words)
            if type_matches or item_matches:
                goal.status = "done" if transaction.status not in ("failed", "cancelled") else "failed"

    def _update_existing_transaction(self, name: str, arguments: dict[str, Any], result: dict[str, Any], turn: int) -> None:
        identifier = str(arguments.get("order_id") or arguments.get("book_id") or arguments.get("reservation_id") or "")
        if not identifier or identifier not in self.transactions:
            return
        transaction = self.transactions[identifier]
        transaction.latest_evidence_turn = turn
        if not self._is_success(result):
            if "failed" in str(result.get("content") or "").lower():
                transaction.status = "failed"
            return
        lowered = name.lower()
        if "pay_" in lowered:
            transaction.payment_status = "paid"
            transaction.status = "paid"
        elif "cancel" in lowered:
            transaction.status = "cancelled"
        elif "modify" in lowered and arguments.get("time"):
            transaction.date_time = str(arguments["time"])
            if isinstance(arguments.get("customer_count"), int):
                transaction.quantities["party_size"] = int(arguments["customer_count"])

    def _apply_tool_result(self, result: dict[str, Any], action: dict[str, Any] | None, turn: int) -> None:
        if action is None:
            return
        name = str(action.get("name") or "")
        arguments = dict(action.get("arguments") or {})
        if self._is_success(result):
            self._register_entities(arguments, self._domain_for_tool(name), turn)
            self._register_entities_from_content(
                str(result.get("content") or ""), self._domain_for_tool(name), turn
            )
        if name.startswith("create_") or name in {"instore_book", "instore_reservation"}:
            self._transaction_from_create(name, arguments, result, turn)
        else:
            self._update_existing_transaction(name, arguments, result, turn)

    def _matching_transactions(self, constraint: Constraint) -> list[Transaction]:
        allowed = {
            "delivery": {"delivery"}, "ota": {"hotel", "attraction", "flight", "train"},
            "instore": {"instore_product", "instore_book", "instore_reservation", "instore"},
        }.get(constraint.domain)
        candidates = list(self.transactions.values())
        if allowed is not None:
            candidates = [txn for txn in candidates if txn.transaction_type in allowed]
        subject = self._normal(constraint.subject)
        if constraint.field in ("quantity", "item") and subject not in {"", "item"}:
            candidates = [txn for txn in candidates if any(subject in self._normal(item) or self._normal(item) in subject for item in txn.items)]
        return candidates

    def _reconcile_constraint(self, constraint: Constraint) -> None:
        transactions = self._matching_transactions(constraint)
        if not transactions:
            return
        desired = constraint.desired_value
        matches: list[bool] = []
        for transaction in transactions:
            actual: Any = None
            if constraint.field == "quantity":
                subject = self._normal(constraint.subject)
                actual = next((count for item, count in transaction.quantities.items() if subject in self._normal(item) or self._normal(item) in subject), None)
            elif constraint.field == "date":
                actual = transaction.date_time
            elif constraint.field == "address":
                actual = transaction.location
            elif constraint.field == "party_size":
                actual = transaction.quantities.get("party_size")
            elif constraint.field == "payment_status":
                actual = transaction.payment_status
            elif constraint.field == "cancellation":
                actual = transaction.status
            elif constraint.field == "item":
                actual = " ".join(transaction.items)
            if actual is None:
                continue
            if constraint.field == "date":
                matches.append(str(actual).lower().startswith(str(desired).lower()))
            elif isinstance(desired, str):
                matches.append(self._normal(desired) in self._normal(actual))
            else:
                matches.append(actual == desired)
            constraint.evidence = [f"transaction {transaction.transaction_id}: {constraint.field}={actual!r}"]
        if matches:
            constraint.status = "satisfied" if any(matches) else "violated"

    def reconcile(self) -> None:
        for constraint in self.constraints.values():
            self._reconcile_constraint(constraint)
        self.termination.unresolved_constraints = sum(c.status in ("known", "unknown") for c in self.constraints.values())
        self.termination.violated_constraints = sum(c.status == "violated" for c in self.constraints.values())
        self.termination.unresolved_subgoals = sum(g.status in ("pending", "active") for g in self.subgoals.values())
        self.termination.failed_transactions = sum(t.status == "failed" for t in self.transactions.values())
        self.termination.open_questions = len(self.open_questions)
        self.termination.can_stop = not any((self.termination.unresolved_constraints, self.termination.violated_constraints, self.termination.unresolved_subgoals, self.termination.failed_transactions, self.termination.open_questions))

    def _trace(self, before: dict[str, Any], event: dict[str, Any]) -> None:
        self.debug_traces.append({
            "turn": self.turn, "state_before_update": before, "event": deepcopy(event),
            "state_after_update": self.snapshot(), "rendered_state": self.render(),
            "constraint_counts": {
                "satisfied": sum(c.status == "satisfied" for c in self.constraints.values()),
                "violated": self.termination.violated_constraints,
                "unresolved": self.termination.unresolved_constraints,
            }, "can_stop": self.termination.can_stop,
        })

    def observe(self, observation: dict[str, Any], pending_action: dict[str, Any] | None = None) -> None:
        """Apply one new user/tool observation to the existing state."""
        before = self.snapshot()
        self.turn += 1
        if observation.get("kind") == "user":
            self._extract_user_constraints(str(observation.get("content") or ""), self.turn)
        elif observation.get("kind") == "tool":
            calls_by_id = {str(call.get("id") or ""): call for call in (pending_action or {}).get("tool_calls", [])}
            for result in observation.get("results") or []:
                matching_call = calls_by_id.get(str(result.get("id") or ""))
                if matching_call is None and len(calls_by_id) == 1:
                    matching_call = next(iter(calls_by_id.values()))
                self._apply_tool_result(result, matching_call, self.turn)
        self.reconcile()
        self._trace(before, observation)

    def record_action(self, action: dict[str, Any]) -> None:
        """Record an assistant action and activate its associated subgoal."""
        before = self.snapshot()
        for call in action.get("tool_calls") or []:
            name = str(call.get("name") or "")
            if name.startswith("create_") or name in {"instore_book", "instore_reservation"}:
                self._upsert_subgoal(self._domain_for_tool(name), name.replace("create_", "").replace("_", " ")).status = "active"
        self.reconcile()
        self._trace(before, {"kind": "assistant_action", **deepcopy(action)})

    def render(self) -> str:
        """Render a compact stable view, never a raw-history summary."""
        lines = ["[CURRENT TASK STATE]", "Constraints"]
        if self.constraints:
            for constraint in sorted(self.constraints.values(), key=lambda item: item.id):
                lines.append(f"- {constraint.domain} / {constraint.subject} / {constraint.field} = {constraint.desired_value!r} [{constraint.status.upper()}]")
                if constraint.evidence:
                    lines.append(f"  evidence: {constraint.evidence[-1]}")
        else:
            lines.append("- none")
        lines.append("Entities")
        if self.entities:
            lines.extend(f"- {entity.entity_id}: {entity.domain}/{entity.entity_type} (tools: {entity.valid_tool_namespace})" for entity in sorted(self.entities.values(), key=lambda item: item.entity_id))
        else:
            lines.append("- none")
        lines.append("Transactions")
        if self.transactions:
            lines.extend(f"- {txn.transaction_id}: {txn.transaction_type}, {txn.status}, payment={txn.payment_status}, items={txn.items}, quantities={txn.quantities}" for txn in sorted(self.transactions.values(), key=lambda item: item.transaction_id))
        else:
            lines.append("- none")
        lines.append("Subgoals")
        if self.subgoals:
            lines.extend(f"- {goal.description}: {goal.status.upper()}" for goal in sorted(self.subgoals.values(), key=lambda item: item.id))
        else:
            lines.append("- none")
        lines.append("Open Questions")
        if self.open_questions:
            lines.extend(f"- {question.subject}/{question.field}: {question.reason}" for question in self.open_questions)
        else:
            lines.append("- none")
        lines.extend(("Termination", f"- unresolved constraints: {self.termination.unresolved_constraints}", f"- violated constraints: {self.termination.violated_constraints}", f"- unresolved subgoals: {self.termination.unresolved_subgoals}", f"- failed transactions: {self.termination.failed_transactions}", f"- open questions: {self.termination.open_questions}", f"- can_stop: {'YES' if self.termination.can_stop else 'NO'}"))
        return "\n".join(lines)


@dataclass
class AgentWorkingState:
    """Harness runtime metadata plus a persistent :class:`TaskState`."""

    turn: int = 0
    latest_observation: dict[str, Any] | None = None
    latest_user_request: str | None = None
    latest_tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    tool_error_count: int = 0
    task_state: TaskState = field(default_factory=TaskState)

    def observe(self, observation: dict[str, Any]) -> None:
        self.turn += 1
        self.latest_observation = observation
        if observation.get("kind") == "user":
            self.latest_user_request = str(observation.get("content") or "")
        if observation.get("kind") == "tool":
            self.latest_tool_results = list(observation.get("results") or [])
            self.tool_error_count += sum(bool(result.get("error")) for result in self.latest_tool_results)
        self.task_state.observe(observation, self.pending_action)
        self.pending_action = None

    def record_action(self, action: dict[str, Any]) -> None:
        self.pending_action = action
        self.task_state.record_action(action)

    def as_prompt(self) -> str:
        runtime = json.dumps({"turn": self.turn, "pending_action": self.pending_action, "tool_error_count": self.tool_error_count}, ensure_ascii=False, sort_keys=True)
        return f"{self.task_state.render()}\nRuntime: {runtime}"


# Preserve the original public placeholder alias for existing callers.
State = dict[str, Any]
