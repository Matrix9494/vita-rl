"""Regression coverage for compact, stateful task tracking behavior."""

from vita_rl.state import AgentWorkingState


def _action(call_id, name, arguments):
    return {"content": "", "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def _tool(call_id, name, content, error=False):
    return {"kind": "tool", "results": [{"id": call_id, "name": name, "content": content, "error": error}]}


PRODUCT = (
    "StoreProduct(store_name=Jintugo Bean Sprout Crossing Bridge Rice Noodles, "
    "store_id=S85277471199200385_S98778, product_name=Free-range chicken crossing "
    "bridge rice noodles (clear soup), product_id=S26110520013660896_P88296, "
    "attributes=, quantity=9999, price=25.0, tags=['Easily digestible', 'Mild'])"
)
STORE = (
    "Store(name=Jintugo Bean Sprout Crossing Bridge Rice Noodles, "
    "store_id=S85277471199200385_S98778, score=5.0, location=No. 46 Yuanxi Road, "
    "Wuhua District, Kunming, Yunnan Province, Jintugo Bean Sprout Crossing Bridge "
    "Rice Noodle Restaurant longitude:102.70636,latitude:25.055897, "
    "tags=['Dine-in available', 'Quick service'])"
)


def test_initial_delivery_goal_and_source_survive_tool_observation():
    state = AgentWorkingState()
    request = "I need something mild to eat delivered to my department."
    state.observe({"kind": "user", "content": request})
    state.record_action(_action("search", "delivery_product_search_recommend", {"keywords": ["mild"]}))
    state.observe(_tool("search", "delivery_product_search_recommend", PRODUCT))

    goal = state.task_state.active_goal
    assert goal and goal.domain == "delivery" and goal.status == "active"
    assert goal.source_text == request
    assert request in state.task_state.render()


def test_active_delivery_goal_is_not_switched_by_restaurant_wording():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Please deliver mild rice noodles to my department."})
    state.observe({"kind": "user", "content": "Could this restaurant offer a different one?"})
    assert state.task_state.active_goal and state.task_state.active_goal.domain == "delivery"
    assert all(constraint.domain == "delivery" for constraint in state.task_state.constraints.values())


def test_unparsed_evidence_is_visible_but_not_a_fake_open_question():
    state = AgentWorkingState()
    text = "Make it comforting but do not make me decide ten things."
    state.observe({"kind": "user", "content": text})
    assert state.task_state.unparsed_goal_evidence[-1].source_text == text
    assert state.task_state.open_questions == []
    assert text in state.task_state.render()


def test_food_restrictions_are_independent_and_negative_claims_stay_unverified():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Deliver mild rice noodles, no fried food or high purine soup."})
    constraints = state.task_state.constraints
    assert constraints["delivery:food:mild"].desired_value == "required"
    assert constraints["delivery:food:fried"].desired_value == "forbidden"
    assert constraints["delivery:food:high_purine"].desired_value == "forbidden"
    state.record_action(_action("search", "delivery_product_search_recommend", {}))
    state.observe(_tool("search", "delivery_product_search_recommend", PRODUCT))
    assert constraints["delivery:food:mild"].status == "satisfied"
    assert constraints["delivery:food:high_purine"].status == "known"


def test_restriction_words_do_not_become_fake_food_categories_or_reopen_order_goal():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Deliver rice noodles; avoid high-purine foods."})
    assert {item.desired_value for item in state.task_state.constraints.values()} >= {"rice noodles", "forbidden"}
    assert not any(
        item.field == "food_category" and "purine" in str(item.desired_value)
        for item in state.task_state.constraints.values()
    )
    state.record_action(_action("create", "create_delivery_order", {
        "store_id": "S1", "product_ids": ["rice_noodles"], "product_cnts": [1],
    }))
    state.observe(_tool("create", "create_delivery_order", "Order(order_id='D1', status='unpaid')"))
    assert state.task_state.active_goal and state.task_state.active_goal.status == "completed"
    state.observe({"kind": "user", "content": "I don't want any risks with the food."})
    assert state.task_state.active_goal and state.task_state.active_goal.status == "completed"


def test_world_facts_entities_relations_and_candidates_are_idempotent():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Deliver mild rice noodles to my department."})
    action = _action("search", "delivery_product_search_recommend", {})
    state.record_action(action)
    state.observe(_tool("search", "delivery_product_search_recommend", PRODUCT))
    state.record_action(_action("store", "get_delivery_store_info", {"store_id": "S85277471199200385_S98778"}))
    state.observe(_tool("store", "get_delivery_store_info", f"{STORE}, products={PRODUCT}"))
    state.record_action(action)
    state.observe(_tool("search", "delivery_product_search_recommend", PRODUCT))

    task = state.task_state
    assert task.entities["S26110520013660896_P88296"].parent_entity_id == "S85277471199200385_S98778"
    assert task.entities["S85277471199200385_S98778"].coordinates == (102.70636, 25.055897)
    assert len(task.relations) == 1
    assert len(task.candidates) == 1
    assert "store_coordinates:S85277471199200385_S98778" in task.world_facts


def test_execution_uses_only_observed_ids_and_tracks_lifecycle():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "Please deliver mild rice noodles to my department."})
    state.record_action(_action("search", "delivery_product_search_recommend", {}))
    state.observe(_tool("search", "delivery_product_search_recommend", PRODUCT))
    state.record_action(_action("create", "create_delivery_order", {
        "store_id": "S85277471199200385_S98778", "product_ids": ["S26110520013660896_P88296"],
        "address": "Yunnan University Affiliated Hospital", "dispatch_time": "2026-09-22 12:00:00",
    }))
    state.observe(_tool("create", "create_delivery_order", "Order(order_id='D100', status='unpaid')"))
    assert state.task_state.execution.current_order_id == "D100"
    assert state.task_state.execution.selected_product_id == "S26110520013660896_P88296"
    state.record_action(_action("pay", "pay_delivery_order", {"order_id": "D100"}))
    state.observe(_tool("pay", "pay_delivery_order", "Payment successful"))
    state.record_action(_action("cancel", "cancel_delivery_order", {"order_id": "D100"}))
    state.observe(_tool("cancel", "cancel_delivery_order", "Order D100 has been cancelled."))
    assert state.task_state.execution.payment_status == "paid"
    assert state.task_state.execution.cancellation_status == "cancelled"
    assert "D100" in state.task_state.render()


def test_failed_create_cannot_invent_order_or_select_argument_ids():
    state = AgentWorkingState()
    state.record_action(_action("bad", "create_delivery_order", {
        "store_id": "not-observed", "product_ids": ["also-not-observed"],
    }))
    state.observe(_tool("bad", "create_delivery_order", "Error: Store not found"))
    assert state.task_state.execution.current_order_id is None
    assert state.task_state.transactions == {}
    assert "not-observed" not in state.task_state.entities


def test_completion_and_trace_diff_are_explicit():
    state = AgentWorkingState()
    state.observe({"kind": "user", "content": "I need 1 sun hat."})
    trace = state.task_state.debug_traces[-1]
    assert "goal" in trace["state_diff"] and "constraints" in trace["state_diff"]
    state.record_action(_action("create", "create_delivery_order", {
        "store_id": "S1", "product_ids": ["sun_hat"], "product_cnts": [1],
    }))
    state.observe(_tool("create", "create_delivery_order", "Order(order_id='D1', status='unpaid')"))
    assert state.task_state.active_goal and state.task_state.active_goal.status == "completed"
    assert state.task_state.termination.can_stop is True
