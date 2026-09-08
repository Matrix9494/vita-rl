import json

import pytest

vita = pytest.importorskip("vita")

from vita.data_model.message import AssistantMessage, ToolMessage, UserMessage
from vita_rl import harness
from vita_rl.harness import VitaRLStateDeltaAgent
from vita_rl.state_delta.updaters import StateDeltaProposal


class FixedUpdater:
    def __init__(self, delta):
        self.delta = delta
        self.calls = []

    def update(self, **kwargs):
        self.calls.append(kwargs)
        return StateDeltaProposal(raw_delta=json.dumps(self.delta), delta=self.delta, updater_name="fixed")


class SequencedUpdater:
    def __init__(self, deltas):
        self.deltas = iter(deltas)

    def update(self, **kwargs):
        delta = next(self.deltas)
        return StateDeltaProposal(raw_delta=json.dumps(delta), delta=delta, updater_name="sequence")


def _agent(updater):
    return VitaRLStateDeltaAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english", state_updater=updater,
    )


def test_delta_harness_runs_updater_before_action_and_traces(monkeypatch, tmp_path):
    captured = {}

    def fake_generate(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AssistantMessage(role="assistant", content="I will search.")

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_STATE_DELTA_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    updater = FixedUpdater({"ops": [{"op": "set", "path": "/goals/g1", "value": {
        "type": "delivery", "status": "active", "slots": {"item": "lunch"}
    }}]})
    agent = _agent(updater)
    response, state = agent.generate_next_message(
        UserMessage(role="user", content="Find me lunch."), agent.get_init_state()
    )

    assert response.content == "I will search."
    assert updater.calls[0]["observation"] == {"kind": "user", "content": "Find me lunch."}
    assert state.canonical_state["goals"]["g1"]["slots"]["item"] == "lunch"
    assert len(captured["messages"]) == 2
    assert "canonical_semantic_state" in captured["messages"][0].content
    assert "observed_task_state" not in captured["messages"][0].content
    assert "Find me lunch." in captured["messages"][0].content  # latest observation only
    assert captured["messages"][1].content == "Find me lunch."
    record = json.loads((tmp_path / "trace.jsonl").read_text())
    assert record["accepted_ops"] and record["agent_action"]["content"] == "I will search."
    assert record["updater_thinking_trace"] is None
    assert record["action_thinking_trace"] is None
    assert "observed_task_state_after_update" not in record
    assert record["termination_decision"]["accepted"] is False


def test_default_llm_updater_is_a_separate_no_tool_call(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["tools"] is None:
            return AssistantMessage(role="assistant", content=json.dumps({"ops": [{
                "op": "set", "path": "/goals/g1", "value": {
                    "type": "delivery", "status": "active", "slots": {"item": "lunch"}
                }
            }]}))
        return AssistantMessage(role="assistant", content="I will search.")

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_STATE_DELTA_UPDATER", "llm")
    agent = VitaRLStateDeltaAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english",
    )
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Find lunch."), agent.get_init_state()
    )

    assert len(calls) == 2
    assert calls[0]["tools"] is None
    assert calls[0]["enable_think"] is True
    assert "internal state-update component" in calls[0]["messages"][0].content
    assert calls[1]["tools"] == []
    assert state.canonical_state["goals"]["g1"]["status"] == "active"


def test_delta_trace_persists_updater_and_action_thinking(monkeypatch, tmp_path):
    calls = 0

    def fake_generate(**kwargs):
        nonlocal calls
        calls += 1
        if kwargs["tools"] is None:
            return AssistantMessage(
                role="assistant",
                content='{"ops":[{"op":"noop"}]}',
                raw_data={"message": {"reasoning_content": "updater reasoning"}},
            )
        return AssistantMessage(
            role="assistant",
            content="I will search.",
            raw_data={"message": {"reasoning": "action reasoning"}},
        )

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_STATE_DELTA_UPDATER", "llm")
    monkeypatch.setenv("VITA_STATE_DELTA_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    agent = VitaRLStateDeltaAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english",
    )
    agent.generate_next_message(
        UserMessage(role="user", content="Find lunch."), agent.get_init_state()
    )

    record = json.loads((tmp_path / "trace.jsonl").read_text())
    assert calls == 2
    assert record["updater_thinking_trace"] == "updater reasoning"
    assert record["action_thinking_trace"] == "action reasoning"


def test_delta_agent_always_enables_thinking_for_action_and_updater():
    agent = _agent(FixedUpdater({"ops": [{"op": "noop"}]}))
    assert agent.enable_think is True


def test_delta_harness_rejected_edit_preserves_evidence_and_rejects_stop(monkeypatch):
    def fake_generate(**kwargs):
        return AssistantMessage(role="assistant", content="###STOP###")

    monkeypatch.setattr(harness, "generate", fake_generate)
    agent = _agent(FixedUpdater({"ops": [{"op": "set", "path": "not/a/pointer", "value": "bad"}]}))
    response, state = agent.generate_next_message(
        ToolMessage(id="x", name="search", role="tool", content="result", error=False), agent.get_init_state()
    )

    assert response.content == "###STOP###"
    assert state.canonical_state == {"entities": {}, "goals": {}, "constraints": {}}
    assert len(state.unresolved_evidence) == 1
    assert state.last_termination_decision["requested"] is True
    assert state.last_termination_decision["accepted"] is False
    assert agent.is_stop(response) is False


def test_delta_harness_only_accepts_stop_after_terminal_llm_delta_with_tool_provenance(monkeypatch):
    calls = 0

    def fake_generate(**kwargs):
        nonlocal calls
        calls += 1
        return AssistantMessage(role="assistant", content="continue" if calls == 1 else "###STOP###")

    monkeypatch.setattr(harness, "generate", fake_generate)
    agent = _agent(SequencedUpdater([
        {"ops": [{"op": "set", "path": "/goals/g1", "value": {
            "type": "delivery", "status": "active", "slots": {}
        }}]},
        {"ops": [{"op": "set_status", "path": "/goals/g1", "value": "done"}]},
    ]))
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Please complete this."), agent.get_init_state()
    )
    response, state = agent.generate_next_message(
        ToolMessage(id="x", name="create_order", role="tool", content="confirmed", error=False), state
    )
    assert state.last_termination_decision["accepted"] is True
    assert agent.is_stop(response) is True


def test_no_cpu_extraction_of_dietary_constraints(monkeypatch):
    monkeypatch.setattr(
        harness, "generate", lambda **kwargs: AssistantMessage(role="assistant", content="continue")
    )
    agent = _agent(FixedUpdater({"ops": [{"op": "noop"}]}))
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="No fried food and no high-purine food."),
        agent.get_init_state(),
    )
    assert state.canonical_state == {"entities": {}, "goals": {}, "constraints": {}}


def test_no_address_regex_fallback_after_malformed_delta(monkeypatch):
    monkeypatch.setattr(
        harness, "generate", lambda **kwargs: AssistantMessage(role="assistant", content="continue")
    )
    agent = _agent(FixedUpdater({"not": "a delta"}))
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Deliver to Yunnan University Affiliated Hospital."),
        agent.get_init_state(),
    )
    assert state.canonical_state == {"entities": {}, "goals": {}, "constraints": {}}
    assert state.unresolved_evidence[0]["observation"]["content"] == "Deliver to Yunnan University Affiliated Hospital."


def test_tool_result_is_raw_context_not_automatic_semantic_state(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AssistantMessage(role="assistant", content="continue")

    monkeypatch.setattr(harness, "generate", fake_generate)
    agent = _agent(FixedUpdater({"ops": [{"op": "noop"}]}))
    _, state = agent.generate_next_message(
        ToolMessage(
            id="order-1",
            name="create_delivery_order",
            role="tool",
            error=False,
            content="Order(order_id:OT0107ef5c87, status:unpaid)",
        ),
        agent.get_init_state(),
    )
    assert state.canonical_state == {"entities": {}, "goals": {}, "constraints": {}}
    assert "OT0107ef5c87" in captured["messages"][0].content


def test_no_quantity_or_closing_language_parsing(monkeypatch):
    monkeypatch.setattr(
        harness, "generate", lambda **kwargs: AssistantMessage(role="assistant", content="continue")
    )
    agent = _agent(SequencedUpdater([
        {"ops": [{"op": "set", "path": "/goals/g1", "value": {
            "type": "delivery", "status": "active", "slots": {}
        }}]},
        {"ops": [{"op": "noop"}]},
        {"ops": [{"op": "noop"}]},
    ]))
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="I need lunch."), agent.get_init_state()
    )
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="I want one serving."), state
    )
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="That's all, thanks."), state
    )
    assert "quantity" not in state.canonical_state["goals"]["g1"]["slots"]
    assert state.canonical_state["goals"]["g1"]["status"] == "active"


def test_valid_llm_quantity_edit_is_applied_mechanically(monkeypatch):
    monkeypatch.setattr(
        harness, "generate", lambda **kwargs: AssistantMessage(role="assistant", content="continue")
    )
    agent = _agent(SequencedUpdater([
        {"ops": [{"op": "set", "path": "/goals/g1", "value": {
            "type": "delivery", "status": "active", "slots": {}
        }}]},
        {"ops": [{"op": "set", "path": "/goals/g1/slots/quantity", "value": 1}]},
    ]))
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="I need lunch."), agent.get_init_state()
    )
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="I want one serving."), state
    )
    assert state.canonical_state["goals"]["g1"]["slots"]["quantity"] == 1


def test_model_proposed_done_without_tool_provenance_cannot_stop(monkeypatch):
    monkeypatch.setattr(
        harness, "generate", lambda **kwargs: AssistantMessage(role="assistant", content="###STOP###")
    )
    agent = _agent(FixedUpdater({"ops": [{"op": "set", "path": "/goals/g1", "value": {
        "type": "delivery", "status": "done", "slots": {}
    }}]}))
    response, state = agent.generate_next_message(
        UserMessage(role="user", content="Please deliver lunch."), agent.get_init_state()
    )
    assert state.last_termination_decision["accepted"] is False
    assert state.last_termination_decision["unverified_terminal_goals"] == {"g1": "done"}
    assert agent.is_stop(response) is False


def test_action_input_excludes_unresolved_evidence_but_updater_receives_it(monkeypatch, tmp_path):
    captured = {}

    def fake_generate(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AssistantMessage(role="assistant", content="continue")

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_STATE_DELTA_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    updater = FixedUpdater({"ops": [{"op": "noop"}]})
    agent = _agent(updater)
    state = agent.get_init_state()
    state.unresolved_evidence = [{
        "id": "e4", "turn": 1, "source": "user",
        "observation": {"kind": "user", "content": "recovery-only raw text"}, "reason": "bad edit",
    }]
    _, state = agent.generate_next_message(UserMessage(role="user", content="latest only"), state)

    assert updater.calls[0]["unresolved_evidence"][0]["id"] == "e4"
    assert "recovery-only raw text" not in captured["messages"][0].content
    record = json.loads((tmp_path / "trace.jsonl").read_text())
    assert record["updater_input"]["unresolved_evidence"][0]["id"] == "e4"
    assert "unresolved_evidence" not in record["action_input"]
    assert record["action_input"]["newest_raw_observation"]["content"] == "latest only"


def test_terminal_stop_is_not_permanently_blocked_by_unresolved_evidence(monkeypatch):
    monkeypatch.setattr(
        harness, "generate", lambda **kwargs: AssistantMessage(role="assistant", content="###STOP###")
    )
    agent = _agent(FixedUpdater({"ops": [{"op": "set_status", "path": "/goals/g1", "value": "done"}]}))
    state = agent.get_init_state()
    state.canonical_state["goals"]["g1"] = {"type": "delivery", "status": "active", "slots": {}}
    state.unresolved_evidence = [{"id": "e1", "turn": 1, "source": "user", "observation": {}, "reason": "bad"}]
    response, state = agent.generate_next_message(
        ToolMessage(id="x", name="create_order", role="tool", content="confirmed", error=False), state
    )
    assert state.last_termination_decision["accepted"] is True
    assert state.last_termination_decision["unresolved_evidence_warning"] is True
    assert agent.is_stop(response) is True
