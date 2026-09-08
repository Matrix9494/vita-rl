import json

import pytest

vita = pytest.importorskip("vita")

from vita.data_model.message import AssistantMessage, UserMessage
from vita_rl import harness
from vita_rl.harness import VitaRLStandardAgent


def test_standard_harness_persists_action_thinking_trace(monkeypatch, tmp_path):
    def fake_generate(**kwargs):
        assert kwargs["enable_think"] is True
        return AssistantMessage(
            role="assistant",
            content="I will search.",
            raw_data={"message": {"reasoning_content": "action reasoning"}},
        )

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv(
        "VITA_STANDARD_THINKING_TRACE_PATH", str(tmp_path / "standard.jsonl")
    )
    agent = VitaRLStandardAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english", enable_think=True,
    )
    response, _ = agent.generate_next_message(
        UserMessage(role="user", content="Find lunch."), agent.get_init_state()
    )

    assert response.content == "I will search."
    record = json.loads((tmp_path / "standard.jsonl").read_text())
    assert record == {
        "turn": 1,
        "raw_observation": {"kind": "user", "content": "Find lunch."},
        "agent_action": {"content": "I will search.", "tool_calls": []},
        "action_thinking_trace": "action reasoning",
    }
