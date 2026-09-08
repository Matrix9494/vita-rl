"""CPU-only invariants for the state-delta transaction engine."""

from __future__ import annotations

from copy import deepcopy
import json

from vita_rl.state_delta import apply_state_delta, empty_canonical_state
from vita_rl.state_delta.updaters import LLMStateUpdater, NoOpStateUpdater, ReplayStateUpdater


def _goal(status="active", **slots):
    return {"type": "trip", "status": status, "slots": slots}


def _entity(name, **attributes):
    return {"type": "location", "name": name, "attributes": attributes}


def _apply(state, delta, observation=None, unresolved_evidence=None):
    return apply_state_delta(
        state=state,
        delta=delta,
        metadata={},
        observation=observation or {"kind": "user", "content": "observation"},
        turn_idx=57,
        unresolved_evidence=unresolved_evidence,
    )


def test_explicit_correction_replaces_only_target_slot():
    state = empty_canonical_state()
    state["goals"]["g3"] = _goal(departure_city="Harbin", arrival_city="Dalian")
    before = deepcopy(state)

    result = _apply(state, {"ops": [{"op": "set", "path": "/goals/g3/slots/departure_city", "value": "Beijing"}]})

    assert result.rejected_ops == []
    assert result.state["goals"]["g3"]["slots"] == {
        "departure_city": "Beijing", "arrival_city": "Dalian"
    }
    assert state == before  # pure function: input was never mutated
    assert result.metadata["/goals/g3/slots/departure_city"] == {
        "source_turn": 57, "source_type": "explicit_user"
    }


def test_slot_isolation_does_not_cross_contaminate_entities():
    state = empty_canonical_state()
    state["entities"] = {
        "home": _entity("home", room="Room 1203"),
        "work": _entity("work", floor="5th Floor", address="Old Building"),
    }
    result = _apply(state, {"ops": [{"op": "set", "path": "/entities/work/attributes/address", "value": "Building D"}]})

    assert result.rejected_ops == []
    assert result.state["entities"]["home"]["attributes"] == {"room": "Room 1203"}
    assert result.state["entities"]["work"]["attributes"] == {
        "floor": "5th Floor", "address": "Building D"
    }


def test_soft_constraint_stays_soft_without_a_model_hard_edit():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal()
    delta = {"ops": [{"op": "set", "path": "/constraints/c1", "value": {
        "goal": "g1", "strength": "soft", "field": "location", "operator": "near", "value": "Yuanda"
    }}]}
    result = _apply(state, delta, {"kind": "user", "content": "Anything near Yuanda would be convenient."})

    assert result.rejected_ops == []
    assert result.state["constraints"]["c1"]["strength"] == "soft"


def test_invalid_delta_is_atomic_and_retains_observation():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(destination="Dalian")
    before = deepcopy(state)
    observation = {"kind": "user", "content": "My aunt is coming from Beijing."}
    result = _apply(state, {"ops": [
        {"op": "set", "path": "/goals/g1/slots/departure_city", "value": "Beijing"},
        {"op": "explode", "path": "/goals/g1", "value": "bad"},
    ]}, observation)

    assert result.state == before
    assert result.metadata == {}
    assert result.accepted_ops == []
    assert result.rejected_ops
    assert result.unresolved_evidence == [{
        "id": "e1", "turn": 57, "source": "user", "observation": observation,
        "reason": "unsupported operation: 'explode'",
    }]


def test_set_is_idempotent_and_malformed_reference_cannot_create_goal():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal()
    delta = {"ops": [{"op": "set", "path": "/goals/g1/slots/departure_city", "value": "Beijing"}]}
    once = _apply(state, delta).state
    twice = _apply(once, delta).state
    assert twice == once

    invalid = _apply(empty_canonical_state(), {"ops": [{
        "op": "set", "path": "/goals/accidental/slots/city", "value": "Beijing"
    }]})
    assert invalid.state == empty_canonical_state()
    assert invalid.rejected_ops


def test_path_matching_redundant_goal_id_is_stripped_but_mismatch_is_rejected():
    accepted = _apply(empty_canonical_state(), {"ops": [{
        "op": "set", "path": "/goals/g1", "value": {
            "id": "g1", "type": "delivery", "status": "active", "slots": {}
        }
    }]})
    assert accepted.state["goals"]["g1"] == {"type": "delivery", "status": "active", "slots": {}}
    assert "id" not in accepted.state["goals"]["g1"]

    rejected = _apply(empty_canonical_state(), {"ops": [{
        "op": "set", "path": "/goals/g1", "value": {
            "id": "g2", "type": "delivery", "status": "active", "slots": {}
        }
    }]})
    assert rejected.state == empty_canonical_state()
    assert "must exactly match" in rejected.error


def test_goal_status_transition_is_legal_and_does_not_complete_other_goal():
    state = empty_canonical_state()
    state["goals"] = {"g1": _goal("active"), "g2": _goal("active")}
    result = _apply(state, {"ops": [{"op": "set_status", "path": "/goals/g1", "value": "done"}]})
    assert result.rejected_ops == []
    assert result.state["goals"]["g1"]["status"] == "done"
    assert result.state["goals"]["g2"]["status"] == "active"


def test_append_requires_existing_list_and_rejects_existing_nonlist_atomically():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(tags=[])
    accepted = _apply(state, {"ops": [{"op": "append", "path": "/goals/g1/slots/tags", "value": "urgent"}]})
    assert accepted.state["goals"]["g1"]["slots"]["tags"] == ["urgent"]
    missing = _apply(state, {"ops": [{"op": "append", "path": "/goals/g1/slots/missing", "value": "urgent"}]})
    assert missing.state == state
    assert "path does not exist" in missing.error
    state["goals"]["g1"]["slots"]["scalar"] = "not-a-list"
    rejected = _apply(state, {"ops": [{"op": "append", "path": "/goals/g1/slots/scalar", "value": "urgent"}]})
    assert rejected.state == state
    assert rejected.rejected_ops


def test_status_wrapper_is_normalized_but_other_status_objects_are_rejected():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal("pending")
    accepted = _apply(state, {"ops": [{
        "op": "set_status", "path": "/goals/g1", "value": {"status": "active"}
    }]})
    assert accepted.state["goals"]["g1"]["status"] == "active"
    rejected = _apply(state, {"ops": [{
        "op": "set_status", "path": "/goals/g1", "value": {"status": "active", "reason": "x"}
    }]})
    assert rejected.state == state


def test_append_adds_one_element_and_rejects_a_list_value():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(items=[])
    accepted = _apply(state, {"ops": [{
        "op": "append", "path": "/goals/g1/slots/items", "value": {"quantity": 1},
    }]})
    assert accepted.state["goals"]["g1"]["slots"]["items"] == [{"quantity": 1}]

    rejected = _apply(state, {"ops": [{
        "op": "append", "path": "/goals/g1/slots/items", "value": [{"quantity": 1}],
    }]})
    assert rejected.state == state
    assert "exactly one element" in rejected.error


def test_extend_concatenates_a_list_and_requires_a_list_value():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(items=[])
    result = _apply(state, {"ops": [{
        "op": "extend", "path": "/goals/g1/slots/items",
        "value": [{"quantity": 1}, {"quantity": 2}],
    }]})
    assert result.state["goals"]["g1"]["slots"]["items"] == [{"quantity": 1}, {"quantity": 2}]

    rejected = _apply(state, {"ops": [{
        "op": "extend", "path": "/goals/g1/slots/items", "value": {"quantity": 3},
    }]})
    assert rejected.state == state
    assert "extend value must be a list" in rejected.error


def test_set_preserves_json_type_and_never_parses_json_strings():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(items=[], selected={"x": 1}, note="hello", count=1, flag=True)
    for path, value in (
        ("/goals/g1/slots/items", '[{"quantity": 1}]'),
        ("/goals/g1/slots/selected", "{}"),
        ("/goals/g1/slots/note", ["hello"]),
        ("/goals/g1/slots/count", "1"),
        ("/goals/g1/slots/flag", 1),
    ):
        rejected = _apply(state, {"ops": [{"op": "set", "path": path, "value": value}]})
        assert rejected.state == state
        assert "preserve existing JSON type" in rejected.error


def test_nested_path_mutation_and_invalid_list_traversal():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(items=[{"quantity": 9999}])
    result = _apply(state, {"ops": [{
        "op": "set", "path": "/goals/g1/slots/items/0/quantity", "value": 1,
    }]})
    assert result.state["goals"]["g1"]["slots"]["items"][0]["quantity"] == 1

    for path in ("/goals/g1/slots/items/1/quantity", "/goals/g1/slots/items/0/quantity/value"):
        rejected = _apply(state, {"ops": [{"op": "set", "path": path, "value": 1}]})
        assert rejected.state == state
        assert rejected.rejected_ops


def test_existing_goal_subtree_replacement_is_rejected():
    state = empty_canonical_state()
    state["goals"]["g1"] = _goal(address="hospital", item="soup")
    result = _apply(state, {"ops": [{
        "op": "set", "path": "/goals/g1",
        "value": _goal(quantity=1),
    }]})
    assert result.state == state
    assert "only when the goal does not exist" in result.error


def test_entity_schema_is_canonical_and_rejects_embedded_id():
    accepted = _apply(empty_canonical_state(), {"ops": [{
        "op": "set", "path": "/entities/U000001",
        "value": {"type": "user_profile", "name": "User U000001", "attributes": {}},
    }]})
    assert accepted.rejected_ops == []
    rejected = _apply(empty_canonical_state(), {"ops": [{
        "op": "set", "path": "/entities/U000001",
        "value": {"id": "U000001", "type": "user_profile", "name": "User", "attributes": {}},
    }]})
    assert rejected.state == empty_canonical_state()
    assert "no embedded id" in rejected.error


def test_resolve_evidence_is_explicit_and_only_removes_named_id():
    state = empty_canonical_state()
    evidence = [{"id": "e7", "turn": 2, "source": "user", "observation": {"kind": "user", "content": "x"}, "reason": "bad"}]
    result = _apply(state, {"ops": [{"op": "resolve_evidence", "evidence_id": "e7"}]}, unresolved_evidence=evidence)
    assert result.unresolved_evidence == []
    unknown = _apply(state, {"ops": [{"op": "resolve_evidence", "evidence_id": "e8"}]}, unresolved_evidence=evidence)
    assert unknown.unresolved_evidence[0]["id"] == "e7"
    assert unknown.unresolved_evidence[1]["id"] == "e8"


def test_updater_abstractions_support_no_model_and_replay(tmp_path):
    noop = NoOpStateUpdater().update(
        state={}, metadata={}, observation={}, unresolved_evidence=[], turn_idx=1
    )
    assert noop.delta == {"ops": [{"op": "noop"}]}

    replay_path = tmp_path / "gold.jsonl"
    replay_path.write_text(json.dumps({"turn": 2, "delta": {"ops": [{"op": "noop"}]}}) + "\n")
    replay = ReplayStateUpdater(replay_path)
    assert replay.update(turn_idx=2).delta == {"ops": [{"op": "noop"}]}
    assert "missing_replay_delta" in replay.update(turn_idx=3).delta

    calls = []
    llm = LLMStateUpdater(lambda system, user: calls.append((system, user)) or '{"ops":[{"op":"noop"}]}')
    proposal = llm.update(
        state=empty_canonical_state(), metadata={}, observation={"kind": "user", "content": "hello"},
        unresolved_evidence=[], turn_idx=1,
    )
    assert proposal.delta == {"ops": [{"op": "noop"}]}
    assert "internal state-update component" in calls[0][0]
