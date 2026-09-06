import pytest


vita = pytest.importorskip("vita")

from vita.data_model.message import AssistantMessage, ToolMessage, UserMessage
from vita_rl import harness
from vita_rl.harness import VitaRLStatefulAgent


def test_stateful_agent_updates_state_before_next_action(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AssistantMessage(role="assistant", content="I will search.")

    monkeypatch.setattr(harness, "generate", fake_generate)
    agent = VitaRLStatefulAgent(
        tools=[],
        domain_policy="Current time: {time}",
        llm="test-model",
        time="2025-01-01 12:00:00",
        language="english",
    )
    state = agent.get_init_state()
    response, state = agent.generate_next_message(
        UserMessage(role="user", content="Find me lunch."), state
    )

    assert response.content == "I will search."
    assert state.working_state.turn == 1
    assert state.working_state.latest_user_request == "Find me lunch."
    assert state.working_state.pending_action == {
        "content": "I will search.",
        "tool_calls": [],
    }
    state_prompt = captured["messages"][0].content
    assert "[CURRENT TASK STATE]" in state_prompt
    assert '"pending_action": null' in state_prompt
    assert len(captured["messages"]) == 2
    assert captured["messages"][1].content == "Find me lunch."


def test_stateful_agent_observation_resolves_prior_action(monkeypatch):
    prompts = []

    def fake_generate(**kwargs):
        prompts.append(kwargs["messages"][0].content)
        return AssistantMessage(role="assistant", content=f"action {len(prompts)}")

    monkeypatch.setattr(harness, "generate", fake_generate)
    agent = VitaRLStatefulAgent(
        tools=[],
        domain_policy="Current time: {time}",
        llm="test-model",
        time="2025-01-01 12:00:00",
        language="english",
    )
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Find lunch."), agent.get_init_state()
    )
    assert state.working_state.pending_action["content"] == "action 1"

    _, state = agent.generate_next_message(
        ToolMessage(
            id="call-1", name="search", role="tool", content="One result", error=False
        ),
        state,
    )

    # The second action sees the new tool observation, while its own action is
    # recorded only after generation for the following transition.
    assert "[CURRENT TASK STATE]" in prompts[1]
    assert '"pending_action": null' in prompts[1]
    assert state.working_state.turn == 2
    assert state.working_state.pending_action["content"] == "action 2"


def test_stateful_agent_sends_only_latest_normalized_tool_observation(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AssistantMessage(role="assistant", content="next action")

    monkeypatch.setattr(harness, "generate", fake_generate)
    agent = VitaRLStatefulAgent(
        tools=[],
        domain_policy="Current time: {time}",
        llm="test-model",
        time="2025-01-01 12:00:00",
        language="english",
    )
    state = agent.get_init_state([
        UserMessage(role="user", content="Old request."),
        AssistantMessage(role="assistant", content="Old action."),
    ])

    _, state = agent.generate_next_message(
        ToolMessage(
            id="call-1", name="search", role="tool", content="Newest result", error=False
        ),
        state,
    )

    assert len(captured["messages"]) == 2
    assert "Old request." not in captured["messages"][1].content
    assert "Old action." not in captured["messages"][1].content
    assert captured["messages"][1].content.startswith("[LATEST TOOL OBSERVATION]\n")
    assert '"content": "Newest result"' in captured["messages"][1].content
    # The transcript is still kept locally, but no longer becomes model input.
    assert len(state.messages) == 4
