import inspect

from vita_rl import state as state_module
from vita_rl.state import AgentWorkingState


def _action(call_id, name, arguments):
    return {"content": "", "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def _tool(call_id, name, content, error=False):
    return {"kind": "tool", "results": [{"id": call_id, "name": name, "content": content, "error": error}]}


def _created_delivery(state, order_id, quantity):
    state.record_action(_action("create", "create_delivery_order", {
        "store_id": "S123", "product_ids": ["sun_hat"], "product_cnts": [quantity],
        "address": "Jinxiu Garden", "dispatch_time": "2026-09-22 12:00:00",
    }))
    state.observe(_tool("create", "create_delivery_order", f"Order(order_id='{order_id}', status='unpaid')"))


def test_quantity_constraint_persists_across_unrelated_events():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "I need 3 sun hats."})
    key = "delivery:sunhat:quantity"
    assert state.task_state.constraints[key].desired_value == 3

    state.observe({"kind": "user", "content": "What shops are nearby?"})
    state.record_action(_action("lookup", "get_delivery_store_info", {"store_id": "S123"}))
    state.observe(_tool("lookup", "get_delivery_store_info", "Store(store_id='S123')"))

    assert state.task_state.constraints[key].desired_value == 3
    assert state.task_state.constraints[key].source_turn == 1


def test_place_order_phrase_and_product_name_are_extracted_precisely():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Please place the order for 1 Black Pepper Chicken Curry Omurice."})
    key = "delivery:blackpepperchickencurryomurice:quantity"
    assert state.task_state.constraints[key].desired_value == 1
    state.record_action(_action("create", "create_delivery_order", {
        "store_id": "S123", "product_ids": ["P1"], "product_cnts": [1],
        "address": "Mingshi Garden", "dispatch_time": "2026-09-22 12:00:00",
    }))
    state.observe(_tool("create", "create_delivery_order", "Order(order_id:D1, status:unpaid, products:[StoreProduct(store_name=A Store, product_name=Black Pepper Chicken Curry Omurice, product_id=P1, quantity=1)])"))
    assert state.task_state.transactions["D1"].items == ["Black Pepper Chicken Curry Omurice"]
    assert state.task_state.constraints[key].status == "satisfied"


def test_distinct_constraints_are_additive_without_explicit_supersession():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "I need food without fried items."})
    state.observe({"kind": "user", "content": "I also need food with low sugar."})
    values = {constraint.desired_value for constraint in state.task_state.constraints.values()}
    assert "without fried items" in values
    assert "with low sugar" in values


def test_quantity_mismatch_is_reconciled_as_violated():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "I need 3 sun hats."})
    _created_delivery(state, "D1", 1)

    constraint = state.task_state.constraints["delivery:sunhat:quantity"]
    assert constraint.status == "violated"
    assert "quantity=1" in constraint.evidence[-1]


def test_hotel_date_persists_through_later_unrelated_turns():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Please book a hotel on 2026-09-22."})
    state.observe({"kind": "user", "content": "Thanks, what is the weather?"})

    constraint = state.task_state.constraints["ota:hotel:date"]
    assert constraint.desired_value == "2026-09-22"
    assert constraint.source_turn == 1


def test_tool_confirmed_entity_has_domain_type_and_namespace():
    state = AgentWorkingState()
    state.record_action(_action("hotel", "get_ota_hotel_info", {}))
    state.observe(_tool("hotel", "get_ota_hotel_info", "Hotel Info: Hotel(hotel_id='H456')"))

    entity = state.task_state.entities["H456"]
    assert (entity.domain, entity.entity_type, entity.valid_tool_namespace) == ("ota", "hotel", "ota")


def test_transaction_lifecycle_created_paid_cancelled_updates_same_record():
    state = AgentWorkingState()
    _created_delivery(state, "D1", 1)
    state.record_action(_action("pay", "pay_delivery_order", {"order_id": "D1"}))
    state.observe(_tool("pay", "pay_delivery_order", "Payment successful"))
    state.record_action(_action("cancel", "cancel_delivery_order", {"order_id": "D1"}))
    state.observe(_tool("cancel", "cancel_delivery_order", "Order D1 has been cancelled."))

    transaction = state.task_state.transactions["D1"]
    assert transaction.status == "cancelled"
    assert transaction.payment_status == "paid"


def test_only_confirmed_tool_result_creates_transaction_and_uses_observed_product_name():
    state = AgentWorkingState()
    state.record_action(_action("create", "create_delivery_order", {
        "store_id": "S123", "product_ids": ["P1"], "product_cnts": [3],
        "address": "Jinxiu Garden", "dispatch_time": "2026-09-22 12:00:00",
    }))
    state.observe(_tool("create", "create_delivery_order", "dispatch_time 2025-09-22 10:00:00 must be in the future"))
    assert state.task_state.transactions == {}

    state.record_action(_action("create2", "create_delivery_order", {
        "store_id": "S123", "product_ids": ["P1"], "product_cnts": [3],
        "address": "Jinxiu Garden", "dispatch_time": "2026-09-22 12:00:00",
    }))
    state.observe(_tool("create2", "create_delivery_order", "Order(order_id:D2, status:unpaid, products:[StoreProduct(product_name=Sun Hat, product_id=P1, quantity=3)])"))
    transaction = state.task_state.transactions["D2"]
    assert transaction.items == ["Sun Hat"]
    assert transaction.quantities == {"Sun Hat": 3}


def test_subgoals_persist_when_one_domain_is_done_and_another_is_pending():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Book a hotel on 2026-09-22."})
    state.record_action(_action("hotel", "create_hotel_order", {"hotel_id": "H456", "room_id": "R1"}))
    state.observe(_tool("hotel", "create_hotel_order", "Order(order_id='H1', status='unpaid')"))
    state.observe({"kind": "user", "content": "I need 3 sun hats."})

    hotel_goal = state.task_state.subgoals["ota:hotelbooking"]
    delivery_goal = state.task_state.subgoals["delivery:sunhatrequest"]
    assert hotel_goal.status == "done"
    assert delivery_goal.status == "pending"
    assert "hotel booking: DONE" in state.task_state.render()
    assert "sun hat request: PENDING" in state.task_state.render()


def test_termination_requires_no_violations_or_pending_subgoals():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "I need 3 sun hats."})
    _created_delivery(state, "D1", 1)
    assert state.task_state.termination.can_stop is False
    assert state.task_state.termination.violated_constraints == 1

    _created_delivery(state, "D2", 3)
    assert state.task_state.termination.violated_constraints == 0
    assert state.task_state.termination.unresolved_subgoals == 0
    assert state.task_state.termination.can_stop is True


def test_each_event_has_inspectable_before_after_trace():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "I need 3 sun hats."})
    trace = state.task_state.debug_traces[-1]
    assert {"turn", "state_before_update", "event", "state_after_update", "rendered_state", "constraint_counts", "can_stop"} <= set(trace)
    assert trace["state_after_update"]["constraints"]["delivery:sunhat:quantity"]["desired_value"] == 3


def test_tracker_has_no_hidden_task_or_rubric_access_path():
    source = inspect.getsource(state_module)
    forbidden_paths = (
        "task.instructions", "user_scenario.user_profile", "evaluation_criteria",
        "ExpectedState", "overall_rubrics", "state_rubrics",
    )
    assert not any(path in source for path in forbidden_paths)
