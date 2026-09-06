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
    assert '"latest_user_request": "Find me lunch."' in state_prompt


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
    assert '"name": "search"' in prompts[1]
    assert '"pending_action": null' in prompts[1]
    assert state.working_state.turn == 2
    assert state.working_state.pending_action["content"] == "action 2"
