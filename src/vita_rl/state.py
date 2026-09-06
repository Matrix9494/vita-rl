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
    tags: list[str] = field(default_factory=list)
    score: float | None = None
    location: str | None = None
    coordinates: tuple[float, float] | None = None
    parent_entity_id: str | None = None
    discovered_turn: int = 0
    last_updated_turn: int = 0
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
class ActiveGoal:
    domain: str
    intent: str
    object_type: str
    target_description: str
    fulfillment_mode: str
    destination: str | None
    status: Literal["active", "completed", "abandoned", "failed"]
    source_turn: int
    source_text: str


@dataclass
class UnparsedGoalEvidence:
    source_turn: int
    source_text: str
    reason: str


@dataclass
class WorldFact:
    key: str
    value: Any
    fact_type: str
    source: str
    source_turn: int
    confidence: Literal["observed", "derived"] = "observed"


@dataclass
class Relation:
    subject_id: str
    relation: str
    object_id: str
    source_turn: int


@dataclass
class Candidate:
    candidate_id: str
    domain: str
    store_id: str | None
    product_id: str | None
    suitability: dict[str, ConstraintStatus] = field(default_factory=dict)
    status: Literal["discovered", "under_consideration", "selected", "rejected"] = "discovered"
    evidence: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class ExecutionState:
    selected_store_id: str | None = None
    selected_product_id: str | None = None
    current_order_id: str | None = None
    current_order_status: str = "not_created"
    payment_status: str = "not_started"
    cancellation_status: str = "not_requested"
    destination: str | None = None
    dispatch_time: str | None = None
    delivery_time_minutes: float | None = None
    last_successful_action: str | None = None
    last_failed_action: str | None = None


@dataclass
class ProgressState:
    completed_steps: list[str] = field(default_factory=list)
    current_step: str = "understand request"
    unresolved_requirements: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_relevant_facts_needed: list[str] = field(default_factory=list)


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
    active_goal: ActiveGoal | None = None
    constraints: dict[str, Constraint] = field(default_factory=dict)
    world_facts: dict[str, WorldFact] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    transactions: dict[str, Transaction] = field(default_factory=dict)
    subgoals: dict[str, Subgoal] = field(default_factory=dict)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    unparsed_goal_evidence: list[UnparsedGoalEvidence] = field(default_factory=list)
    execution: ExecutionState = field(default_factory=ExecutionState)
    progress: ProgressState = field(default_factory=ProgressState)
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

    def _domain_for_text(self, text: str) -> str:
        """Resolve domain from durable task context before loose keywords."""
        lowered = text.lower()
        # Explicit user switches are the only lexical evidence that can
        # override a currently active task.
        if any(phrase in lowered for phrase in (
            "dine-in", "dine in", "book a table", "table reservation",
            "in-store shopping", "in store shopping", "pick up in store",
        )):
            return "instore"
        if any(word in lowered for word in ("hotel", "flight", "train", "attraction", "ticket", "room", "seat")):
            return "ota"
        # A fresh purchase/delivery request is an explicit task switch even
        # when the user omits the word "delivery" (for example, "I need 3
        # sun hats").  A bare provider noun is not enough to switch domains.
        if any(word in lowered for word in ("delivery", "deliver")) or re.search(
            r"\b(?:need|want|order|buy|get)\s+\d+\b", lowered
        ):
            return "delivery"
        # An in-flight action is stronger context than the wording of a
        # follow-up.  In particular, "restaurant" must not silently turn a
        # delivery task into an in-store task.
        active_transaction_domains = {
            "delivery" if transaction.transaction_type == "delivery" else self._domain_for_tool(transaction.transaction_type)
            for transaction in self.transactions.values()
            if transaction.status in ("pending_confirmation", "created", "unpaid", "paid")
        }
        active_transaction_domains.discard("unknown")
        if len(active_transaction_domains) == 1:
            return active_transaction_domains.pop()
        active_domains = {goal.domain for goal in self.subgoals.values() if goal.status in ("pending", "active")}
        if len(active_domains) == 1:
            return active_domains.pop()
        if self.active_goal and self.active_goal.status == "active":
            return self.active_goal.domain
        if any(word in lowered for word in ("delivery", "deliver", "order", "store", "food", "hat")):
            return "delivery"
        if any(word in lowered for word in ("restaurant", "reservation", "shop")):
            return "instore"
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

    def _set_fact(
        self, key: str, value: Any, fact_type: str, source: str, turn: int,
        confidence: Literal["observed", "derived"] = "observed",
    ) -> None:
        """Store one compact, idempotent, agent-observed world fact."""
        existing = self.world_facts.get(key)
        if existing is None or existing.value != value:
            self.world_facts[key] = WorldFact(
                key=key, value=value, fact_type=fact_type, source=source,
                source_turn=turn, confidence=confidence,
            )
        elif existing is not None:
            existing.source_turn = turn

    def _add_relation(self, subject_id: str, relation: str, object_id: str, turn: int) -> None:
        if not subject_id or not object_id:
            return
        if not any(
            item.subject_id == subject_id and item.relation == relation and item.object_id == object_id
            for item in self.relations
        ):
            self.relations.append(Relation(subject_id, relation, object_id, turn))

    def _set_active_goal(self, text: str, domain: str, turn: int) -> None:
        """Create or refine a persistent task goal using explicit user text."""
        lowered = text.lower()
        if re.search(r"\b(?:no further help|i'll handle it|i will handle it|stop|never mind|nevermind)\b", lowered):
            if self.active_goal and self.active_goal.status == "active":
                self.active_goal.status = "abandoned"
            return
        asks_for_task = bool(re.search(r"\b(?:need|want|order|place|book|reserve|buy|get|deliver|recommend|help)\b", lowered))
        if not asks_for_task:
            return
        fulfillment = "delivery" if any(word in lowered for word in ("delivery", "deliver", "department", "work address")) else domain
        object_type = "food" if any(word in lowered for word in ("food", "eat", "meal", "rice noodle", "noodle", "restaurant")) else "request"
        if "rice noodle" in lowered:
            target = "rice noodles"
        elif "mild" in lowered and object_type == "food":
            target = "mild meal"
        else:
            target = object_type
        destination = "user's department" if "department" in lowered else None
        # After completion, do not reopen a goal merely because a user says
        # "I don't want …" while discussing the same order. A new completed
        # goal needs a clear fulfillment verb, not only a preference word.
        if self.active_goal is None or self.active_goal.status != "active":
            if self.active_goal is not None and not re.search(
                r"\b(?:need|order|place|book|reserve|buy|deliver)\b", lowered
            ):
                return
            self.active_goal = ActiveGoal(
                domain=fulfillment if fulfillment != "unknown" else domain,
                intent="obtain" if any(word in lowered for word in ("order", "need", "want", "get")) else "recommend",
                object_type=object_type, target_description=target,
                fulfillment_mode=fulfillment, destination=destination,
                status="active", source_turn=turn, source_text=text,
            )
            return
        # Same task: enrich it, do not erase its original intent.
        if destination:
            self.active_goal.destination = destination
        if "rice noodle" in lowered:
            self.active_goal.target_description = "rice noodles"
        if fulfillment != "unknown":
            self.active_goal.fulfillment_mode = fulfillment
            self.active_goal.domain = fulfillment

    def _upsert_constraint(
        self, *, domain: str, subject: str, field_name: str, desired_value: Any,
        source_turn: int, source_text: str,
    ) -> Constraint:
        """Add a requirement without deleting an unrelated earlier one."""
        key = f"{domain}:{self._normal(subject)}:{field_name}"
        existing = self.constraints.get(key)
        if existing is not None:
            if existing.desired_value == desired_value:
                # Repeated/paraphrased requirements are corroborating
                # evidence, not new constraints.
                if existing.source_text != source_text:
                    existing.evidence.append(f"reaffirmed turn {source_turn}")
                    existing.source_turn = source_turn
                    existing.source_text = source_text
                return existing
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

    def _add_open_question(self, subject: str, field_name: str, reason: str, text: str, turn: int) -> None:
        if any(question.subject == subject and question.field == field_name for question in self.open_questions):
            return
        self.open_questions.append(OpenQuestion(
            subject=subject, field=field_name, reason=reason,
            source_turn=turn, source_text=text,
        ))

    def _add_unparsed_evidence(self, text: str, turn: int, reason: str) -> None:
        if text.strip() and not any(item.source_text == text for item in self.unparsed_goal_evidence):
            self.unparsed_goal_evidence.append(UnparsedGoalEvidence(turn, text, reason))

    def _extract_user_constraints(self, text: str, turn: int) -> None:
        """Conservatively extract explicit request fields; never guess values."""
        domain = self._domain_for_text(text)
        self._set_active_goal(text, domain, turn)
        if self.active_goal and self.active_goal.status == "active":
            domain = self.active_goal.domain
        lowered = text.lower()
        found = False
        # Compact, high-value delivery/food constraints. Each independent
        # user requirement gets its own stable key so paraphrases update it.
        # Food category means a dish type, never a generic noun in a
        # restriction such as "high-purine foods" or "don't risk food".
        food_match = re.search(
            r"\b([a-z][a-z -]{0,35}(?:rice noodles?|noodles?|porridge))\b",
            text, re.IGNORECASE,
        )
        if food_match:
            category = re.sub(r"^(?:some|a|an|the)\s+", "", food_match.group(1).strip(), flags=re.IGNORECASE)
            if "rice noodle" in category.lower():
                category = "rice noodles"
            self._upsert_constraint(domain=domain, subject="food", field_name="food_category", desired_value=category, source_turn=turn, source_text=text)
            found = True
        if re.search(r"\b(?:mild|non-spicy|not spicy)\b", lowered):
            self._upsert_constraint(domain=domain, subject="food", field_name="mild", desired_value="required", source_turn=turn, source_text=text)
            found = True
        if "fried" in lowered and re.search(r"\b(?:avoid|without|no|not|nothing)\b", lowered):
            self._upsert_constraint(domain=domain, subject="food", field_name="fried", desired_value="forbidden", source_turn=turn, source_text=text)
            found = True
        if re.search(r"\bhigh[- ]purine\b", lowered) and re.search(r"\b(?:avoid|without|no|not|nothing)\b", lowered):
            self._upsert_constraint(domain=domain, subject="food", field_name="high_purine", desired_value="forbidden", source_turn=turn, source_text=text)
            found = True
        if re.search(r"\b(?:no|not)\s+(?:a\s+)?small\s+delivery[- ]only|not[^.]{0,35}delivery[- ]only", lowered):
            self._upsert_constraint(domain=domain, subject="provider", field_name="delivery_only", desired_value="forbidden", source_turn=turn, source_text=text)
            found = True
        if re.search(r"\b(?:something|try)\s+(?:new|different)|tired of (?:my )?usual", lowered):
            self._upsert_constraint(domain=domain, subject="preference", field_name="novelty", desired_value="preferred", source_turn=turn, source_text=text)
            found = True
        if any(word in lowered for word in ("deliver", "delivery", "department", "work address")):
            self._upsert_constraint(domain=domain, subject="fulfillment", field_name="mode", desired_value="delivery", source_turn=turn, source_text=text)
            found = True
        quantity_match = re.search(
            r"\b(?:need|want|order|place(?:\s+the\s+order\s+for)?|book|reserve|buy|get)\s+(\d+)\s+([a-z][a-z0-9 _-]{1,60}?)(?=\s*(?:,|\.|;|for\b|to\b|with\b|on\b|\s+at\b|$))",
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
        time_match = re.search(r"\b(before|after|around)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", text, re.IGNORECASE)
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
            self.execution.destination = address_match.group(1).strip()
            if self.active_goal:
                self.active_goal.destination = self.execution.destination
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
            # Retain only independently actionable attributes which do not
            # already have a normalized field above.  Pronouns ("avoid
            # those") and workflow phrases ("with the delivery") are not
            # requirements and must not poison completion state.
            if any(marker in requirement.lower() for marker in (
                "low sugar", "sugar-free", "gluten", "vegan", "vegetarian",
                "dairy", "halal", "kosher", "fried items",
            )):
                self._upsert_constraint(
                    domain=domain, subject=f"attribute:{requirement}",
                    field_name="required_attribute", desired_value=requirement,
                    source_turn=turn, source_text=text,
                )
                found = True
        if re.search(r"\b(?:pay|paid|payment)\b", text, re.IGNORECASE):
            self._upsert_constraint(domain=domain, subject="transaction", field_name="payment_status", desired_value="paid", source_turn=turn, source_text=text)
            found = True
        if re.search(r"\b(?:cancel|cancellation)\b", text, re.IGNORECASE):
            self._upsert_constraint(domain=domain, subject="transaction", field_name="cancellation", desired_value="cancelled", source_turn=turn, source_text=text)
            found = True
        # Parser coverage is not an open question. Preserve a bounded,
        # task-relevant evidence item that is rendered to the agent instead.
        if text.strip() and (not found or len(self.unparsed_goal_evidence) < 3):
            self._add_unparsed_evidence(text, turn, "Retained because some user intent may not be normalized.")

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
                    discovered_turn=turn, last_updated_turn=turn, valid_tool_namespace=entity_domain,
                )
            else:
                seen_in = existing.observed_attributes.setdefault("seen_in", [])
                if argument_name not in seen_in:
                    seen_in.append(argument_name)
                existing.last_updated_turn = turn

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
                        last_updated_turn=turn, valid_tool_namespace=entity_domain,
                    )

    @staticmethod
    def _comma_value(value: str) -> str:
        return value.strip().strip("'\"")

    @staticmethod
    def _tags(content: str) -> list[str]:
        match = re.search(r"tags\s*=\s*\[([^\]]*)\]", content, re.IGNORECASE)
        if not match:
            return []
        return [item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()]

    def _upsert_entity_from_product(self, content: str, domain: str, turn: int) -> None:
        """Parse the public ``StoreProduct(...)`` representation losslessly."""
        match = re.search(
            r"StoreProduct\(store_name=(.*?),\s*store_id=([^,\)]+),\s*product_name=(.*?),\s*product_id=([^,\)]+),\s*attributes=(.*?),\s*quantity=([0-9]+),\s*price=([0-9.]+),\s*tags=\[([^\]]*)\]",
            content, re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return
        store_name, store_id, product_name, product_id, attributes, quantity, price, tags = match.groups()
        store_id, product_id = self._comma_value(store_id), self._comma_value(product_id)
        product = self.entities.setdefault(product_id, Entity(
            entity_id=product_id, entity_type="product", domain=domain,
            discovered_turn=turn, last_updated_turn=turn, valid_tool_namespace=domain,
        ))
        product.name = self._comma_value(product_name)
        product.parent_entity_id = store_id
        product.tags = [item.strip().strip("'\"") for item in tags.split(",") if item.strip()]
        product.observed_attributes.update({
            "attributes": self._comma_value(attributes), "quantity": int(quantity), "price": float(price),
        })
        product.last_updated_turn = turn
        store = self.entities.setdefault(store_id, Entity(
            entity_id=store_id, entity_type="store", domain="delivery" if domain == "delivery" else domain,
            discovered_turn=turn, last_updated_turn=turn, valid_tool_namespace="delivery" if domain == "delivery" else domain,
        ))
        store.name = self._comma_value(store_name)
        store.last_updated_turn = turn
        self._add_relation(product_id, "belongs_to_store", store_id, turn)
        candidate_id = f"{domain}:{store_id}:{product_id}"
        candidate = self.candidates.setdefault(candidate_id, Candidate(
            candidate_id=candidate_id, domain=domain, store_id=store_id, product_id=product_id,
        ))
        candidate.status = "under_consideration"
        evidence = f"product {product.name} from {store.name or store_id}"
        if evidence not in candidate.evidence:
            candidate.evidence.append(evidence)

    def _upsert_entity_from_store(self, content: str, domain: str, turn: int) -> None:
        match = re.search(
            r"Store\(name=(.*?),\s*store_id=([^,\)]+),\s*score=([0-9.]+),\s*location=(.*?),.*?longitude\s*[:=]\s*([0-9.]+),\s*latitude\s*[:=]\s*([0-9.]+),\s*tags=\[([^\]]*)\]",
            content, re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return
        name, store_id, score, location, longitude, latitude, tags = match.groups()
        store_id = self._comma_value(store_id)
        store = self.entities.setdefault(store_id, Entity(
            entity_id=store_id, entity_type="store", domain="delivery" if domain == "delivery" else domain,
            discovered_turn=turn, last_updated_turn=turn, valid_tool_namespace="delivery" if domain == "delivery" else domain,
        ))
        store.name = self._comma_value(name)
        store.score = float(score)
        store.location = self._comma_value(location)
        store.coordinates = (float(longitude), float(latitude))
        store.tags = [item.strip().strip("'\"") for item in tags.split(",") if item.strip()]
        store.last_updated_turn = turn
        self._set_fact(f"store_coordinates:{store_id}", store.coordinates, "coordinates", "tool", turn)

    def _extract_world_facts(self, name: str, arguments: dict[str, Any], content: str, turn: int) -> None:
        """Extract compact facts only from the current agent-visible tool result."""
        user_id = re.search(r"['\"]?user id['\"]?\s*[:=]\s*['\"]?([^,'\s}\]]+)", content, re.IGNORECASE)
        if user_id:
            self._set_fact("user_id", user_id.group(1), "user_id", name, turn)
        for label, fact_key in (("home address", "home_address"), ("work address", "work_address")):
            match = re.search(rf"['\"]?{label}['\"]?\s*[:=]\s*['\"]([^'\"]+)", content, re.IGNORECASE)
            if match:
                self._set_fact(fact_key, match.group(1), "address", name, turn)
                if fact_key == "work_address" and self.active_goal and self.active_goal.destination == "user's department":
                    self.execution.destination = match.group(1)
        if name == "address_to_longitude_latitude":
            coordinates = re.search(r"\[\s*['\"]?([0-9.-]+)['\"]?\s*,\s*['\"]?([0-9.-]+)['\"]?\s*\]", content)
            address = str(arguments.get("address") or "destination")
            if coordinates:
                value = (float(coordinates.group(1)), float(coordinates.group(2)))
                self._set_fact(f"coordinates:{self._normal(address)}", value, "coordinates", name, turn)
                if self.execution.destination is None or self._normal(address) in self._normal(self.execution.destination):
                    self._set_fact("destination_coordinates", value, "coordinates", name, turn)
        if name == "longitude_latitude_to_distance":
            distance = re.search(r"[-+]?[0-9]*\.?[0-9]+", content)
            if distance:
                self._set_fact("selected_store_distance_m", float(distance.group(0)), "distance_m", name, turn, "derived")
        if name == "delivery_distance_to_time":
            duration = re.search(r"[-+]?[0-9]*\.?[0-9]+", content)
            if duration:
                minutes = float(duration.group(0))
                self._set_fact("estimated_delivery_minutes", minutes, "duration_minutes", name, turn, "derived")
                self.execution.delivery_time_minutes = minutes

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
        selected_product_id = next(
            (item for item in items if item in self.entities and self.entities[item].entity_type == "product"),
            None,
        )
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
        self.execution.current_order_id = transaction_id
        self.execution.current_order_status = transaction.status
        self.execution.payment_status = transaction.payment_status
        self.execution.destination = transaction.location or self.execution.destination
        self.execution.dispatch_time = transaction.date_time
        self.execution.last_successful_action = name
        for item in items:
            if item in self.entities and self.entities[item].entity_type == "product":
                self.execution.selected_product_id = item
        if selected_product_id:
            self.execution.selected_product_id = selected_product_id
        if entity_id:
            self.execution.selected_store_id = str(entity_id)
        selected_candidate = self.candidates.get(
            f"{domain}:{self.execution.selected_store_id}:{self.execution.selected_product_id}"
        )
        if selected_candidate:
            selected_candidate.status = "selected"
        goal.status = "done" if transaction.status not in ("failed", "cancelled") else "failed"
        self._update_subgoals_from_transaction(transaction)
        if self.active_goal and self.active_goal.domain == domain:
            self.active_goal.status = "completed" if transaction.status not in ("failed", "cancelled") else "failed"

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
            self.execution.last_failed_action = name
            return
        lowered = name.lower()
        if "pay_" in lowered:
            transaction.payment_status = "paid"
            transaction.status = "paid"
            self.execution.payment_status = "paid"
        elif "cancel" in lowered:
            transaction.status = "cancelled"
            self.execution.cancellation_status = "cancelled"
        elif "modify" in lowered and arguments.get("time"):
            transaction.date_time = str(arguments["time"])
            if isinstance(arguments.get("customer_count"), int):
                transaction.quantities["party_size"] = int(arguments["customer_count"])
        self.execution.current_order_status = transaction.status
        self.execution.last_successful_action = name

    def _apply_tool_result(self, result: dict[str, Any], action: dict[str, Any] | None, turn: int) -> None:
        if action is None:
            return
        name = str(action.get("name") or "")
        arguments = dict(action.get("arguments") or {})
        content = str(result.get("content") or "")
        domain = self._domain_for_tool(name)
        if self._is_success(result):
            self._register_entities(arguments, domain, turn)
            self._register_entities_from_content(content, domain, turn)
            self._upsert_entity_from_product(content, domain, turn)
            self._upsert_entity_from_store(content, domain, turn)
            self._extract_world_facts(name, arguments, content, turn)
            self.execution.last_successful_action = name
        else:
            self.execution.last_failed_action = name
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
        # First reconcile desired state against observed candidate/entity facts.
        # This never promotes an unsupported negative attribute to satisfied.
        candidate_matches: list[bool] = []
        candidate_evidence: list[str] = []
        for candidate in self.candidates.values():
            if candidate.domain != constraint.domain:
                continue
            product = self.entities.get(candidate.product_id or "")
            store = self.entities.get(candidate.store_id or "")
            actual: str | None = None
            if constraint.field == "food_category" and product and product.name:
                actual = product.name
            elif constraint.field == "mild" and product:
                actual = " ".join(product.tags)
            elif constraint.field in {"fried", "high_purine"} and product:
                actual = " ".join([*product.tags, str(product.observed_attributes.get("attributes", ""))])
            elif constraint.field == "delivery_only" and store:
                actual = " ".join(store.tags)
            if actual is None:
                continue
            if constraint.field == "food_category":
                matched = self._normal(constraint.desired_value) in self._normal(actual)
            elif constraint.field == "mild":
                matched = "mild" in self._normal(actual)
            elif constraint.field == "delivery_only":
                matched = "dineinavailable" in self._normal(actual)
            elif constraint.field == "fried":
                normal_actual = self._normal(actual)
                if any(marker in normal_actual for marker in ("notfried", "nonfried", "unfried")):
                    matched = True
                elif "fried" in normal_actual:
                    matched = False
                else:
                    candidate.suitability[constraint.field] = "unknown"
                    continue
            elif constraint.field == "high_purine":
                normal_actual = self._normal(actual)
                if any(marker in normal_actual for marker in ("lowpurine", "nopurine", "nonpurine")):
                    matched = True
                elif "highpurine" in normal_actual:
                    matched = False
                else:
                    candidate.suitability[constraint.field] = "unknown"
                    continue
            else:
                matched = False
            candidate.suitability[constraint.field] = "satisfied" if matched else "violated"
            candidate_matches.append(matched)
            candidate_evidence.append(f"candidate {candidate.candidate_id}: {constraint.field}={actual!r}")
        if candidate_matches:
            constraint.status = "satisfied" if any(candidate_matches) else "violated"
            constraint.evidence = [candidate_evidence[-1]]
            return

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
        self._refresh_progress()
        self.termination.unresolved_constraints = sum(c.status in ("known", "unknown") for c in self.constraints.values())
        self.termination.violated_constraints = sum(c.status == "violated" for c in self.constraints.values())
        self.termination.unresolved_subgoals = sum(g.status in ("pending", "active") for g in self.subgoals.values())
        self.termination.failed_transactions = sum(t.status == "failed" for t in self.transactions.values())
        self.termination.open_questions = len(self.open_questions)
        goal_done = self.active_goal is not None and self.active_goal.status in ("completed", "abandoned")
        self.termination.can_stop = bool(goal_done and not any((
            self.termination.unresolved_constraints, self.termination.violated_constraints,
            self.termination.unresolved_subgoals, self.termination.failed_transactions,
            self.termination.open_questions,
        )))

    def _refresh_progress(self) -> None:
        """Derive advisory progress from observed state; never select actions."""
        completed: list[str] = []
        if self.candidates:
            completed.append("candidate found")
        if self.execution.destination or "work_address" in self.world_facts:
            completed.append("address known")
        if self.execution.current_order_id:
            completed.append("order created")
            for goal in self.subgoals.values():
                if goal.domain == "delivery" and goal.description == "delivery supplies":
                    goal.status = "done"
        if self.execution.payment_status == "paid":
            completed.append("payment completed")
        self.progress.completed_steps = completed
        self.progress.current_step = "fulfill active goal" if self.active_goal else "understand request"
        self.progress.unresolved_requirements = [
            constraint.field for constraint in self.constraints.values()
            if constraint.status in ("known", "unknown", "violated")
        ]
        self.progress.missing_information = []
        if self.active_goal and self.active_goal.fulfillment_mode == "delivery" and not self.execution.destination:
            self.progress.missing_information.append("delivery destination")
        self.progress.blockers = []
        if self.execution.last_failed_action:
            self.progress.blockers.append(f"last failed action: {self.execution.last_failed_action}")
        self.progress.next_relevant_facts_needed = [
            item for item in self.progress.unresolved_requirements
            if item in {"fried", "high_purine", "novelty", "food_category", "mild", "delivery_only"}
        ]

    def _trace(self, before: dict[str, Any], event: dict[str, Any]) -> None:
        after = self.snapshot()
        self.debug_traces.append({
            "turn": self.turn, "state_before_update": before, "event": deepcopy(event),
            "state_after_update": after, "state_diff": self._state_diff(before, after),
            "rendered_state": self.render(),
            "constraint_counts": {
                "satisfied": sum(c.status == "satisfied" for c in self.constraints.values()),
                "violated": self.termination.violated_constraints,
                "unresolved": self.termination.unresolved_constraints,
            }, "can_stop": self.termination.can_stop,
        })

    @staticmethod
    def _state_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
        """Small, inspectable change summary alongside full trace snapshots."""
        groups = {
            "goal": ("active_goal",), "constraints": ("constraints",),
            "facts": ("world_facts",), "entities": ("entities", "relations"),
            "candidates": ("candidates",), "execution": ("execution",),
            "progress": ("progress",), "termination": ("termination",),
        }
        return {
            group: [field_name for field_name in fields if before.get(field_name) != after.get(field_name)]
            for group, fields in groups.items()
            if any(before.get(field_name) != after.get(field_name) for field_name in fields)
        }

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
        """Render bounded decision state, not a transcript or hidden context."""
        def compact(value: Any, limit: int = 180) -> str:
            rendered = str(value).replace("\n", " ")
            return rendered if len(rendered) <= limit else f"{rendered[:limit - 1]}…"

        lines = ["[CURRENT TASK STATE]", "Active Goal"]
        if self.active_goal:
            goal = self.active_goal
            destination = f"; destination={compact(goal.destination, 110)}" if goal.destination else ""
            lines.append(
                f"- {goal.status.upper()}: {goal.domain}/{goal.intent} {goal.target_description} "
                f"({goal.fulfillment_mode}){destination}"
            )
        else:
            lines.append("- none")

        lines.append("Constraints (desired)")
        if self.constraints:
            for constraint in sorted(self.constraints.values(), key=lambda item: item.id):
                lines.append(f"- {constraint.domain} / {constraint.subject} / {constraint.field} = {compact(repr(constraint.desired_value), 120)} [{constraint.status.upper()}]")
                if constraint.evidence:
                    lines.append(f"  observed: {compact(constraint.evidence[-1], 150)}")
        else:
            lines.append("- none")

        lines.append("Known Facts (observed/derived)")
        if self.world_facts:
            for fact in sorted(self.world_facts.values(), key=lambda item: item.key)[:10]:
                lines.append(f"- {fact.key}: {compact(fact.value, 150)} ({fact.confidence})")
            if len(self.world_facts) > 10:
                lines.append(f"- +{len(self.world_facts) - 10} more facts")
        else:
            lines.append("- none")

        lines.append("Entities")
        if self.entities:
            for entity in sorted(self.entities.values(), key=lambda item: item.entity_id)[:12]:
                details = []
                if entity.name:
                    details.append(f"name={compact(entity.name, 70)}")
                if entity.tags:
                    details.append(f"tags={compact(', '.join(entity.tags), 70)}")
                if entity.score is not None:
                    details.append(f"score={entity.score}")
                if entity.location:
                    details.append(f"location={compact(entity.location, 70)}")
                lines.append(
                    f"- {entity.entity_id}: {entity.domain}/{entity.entity_type} "
                    f"(tools: {entity.valid_tool_namespace})" + (f"; {', '.join(details)}" if details else "")
                )
            if len(self.entities) > 12:
                lines.append(f"- +{len(self.entities) - 12} more entities")
        else:
            lines.append("- none")

        lines.append("Candidates")
        if self.candidates:
            for candidate in sorted(self.candidates.values(), key=lambda item: item.candidate_id)[:8]:
                suitability = ", ".join(f"{key}={value}" for key, value in sorted(candidate.suitability.items())) or "unverified"
                lines.append(
                    f"- {candidate.candidate_id}: {candidate.status}; "
                    f"store={candidate.store_id or '-'} product={candidate.product_id or '-'}; {suitability}"
                )
                if candidate.rejection_reasons:
                    lines.append(f"  rejected because: {compact('; '.join(candidate.rejection_reasons), 150)}")
        else:
            lines.append("- none")

        lines.append("Execution")
        lines.append(
            f"- order_id={self.execution.current_order_id or 'none'}; order_status={self.execution.current_order_status}; "
            f"payment={self.execution.payment_status}; cancellation={self.execution.cancellation_status}"
        )
        lines.append(
            f"- selected store={self.execution.selected_store_id or 'none'}; "
            f"product={self.execution.selected_product_id or 'none'}; "
            f"destination={compact(self.execution.destination, 120) if self.execution.destination else 'none'}"
        )
        if self.execution.delivery_time_minutes is not None:
            lines.append(f"- observed delivery estimate={self.execution.delivery_time_minutes:g} minutes")

        lines.append("Valid IDs")
        ids_by_type: dict[str, list[str]] = {}
        for entity in self.entities.values():
            ids_by_type.setdefault(entity.entity_type, []).append(entity.entity_id)
        if ids_by_type:
            for entity_type, identifiers in sorted(ids_by_type.items()):
                lines.append(f"- {entity_type}: {', '.join(sorted(identifiers)[:8])}")
        else:
            lines.append("- none observed")
        if self.transactions:
            lines.append(f"- order: {', '.join(sorted(self.transactions))}")

        lines.append("Action Preconditions (advisory)")
        lines.append("- create delivery order: observed store ID, product ID, and destination required")
        lines.append(f"- pay/cancel/get order: observed order ID required; currently {self.execution.current_order_id or 'none'}")

        lines.append("Transactions")
        if self.transactions:
            lines.extend(f"- {txn.transaction_id}: {txn.transaction_type}, {txn.status}, payment={txn.payment_status}, items={compact(txn.items, 100)}, quantities={txn.quantities}" for txn in sorted(self.transactions.values(), key=lambda item: item.transaction_id))
        else:
            lines.append("- none")
        lines.append("Subgoals")
        if self.subgoals:
            lines.extend(f"- {goal.description}: {goal.status.upper()}" for goal in sorted(self.subgoals.values(), key=lambda item: item.id))
        else:
            lines.append("- none")
        lines.append("Progress")
        lines.append(f"- current: {self.progress.current_step}")
        if self.progress.completed_steps:
            lines.append(f"- completed: {', '.join(self.progress.completed_steps)}")
        if self.progress.unresolved_requirements:
            lines.append(f"- unresolved: {', '.join(self.progress.unresolved_requirements)}")
        if self.progress.missing_information:
            lines.append(f"- missing: {', '.join(self.progress.missing_information)}")
        if self.progress.blockers:
            lines.append(f"- blockers: {', '.join(self.progress.blockers)}")
        if self.progress.next_relevant_facts_needed:
            lines.append(f"- next facts: {', '.join(self.progress.next_relevant_facts_needed)}")

        lines.append("Unparsed Goal Evidence")
        if self.unparsed_goal_evidence:
            for evidence in self.unparsed_goal_evidence[-3:]:
                lines.append(f"- turn {evidence.source_turn}: {compact(evidence.source_text, 260)}")
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
