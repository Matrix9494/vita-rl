import json
from pathlib import Path

import pytest

vita = pytest.importorskip("vita")

from vita.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.utils.llm_utils import _template_token_count, format_messages
from vita_rl import harness
from vita_rl.harness import VitaRLRecentTurnsAgent, VitaRLStandardAgent, VitaRLSummaryAgent


def _summary_agent():
    return VitaRLSummaryAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english",
    )


def _standard_agent():
    return VitaRLStandardAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english",
    )


def test_before_first_compaction_matches_standard_native_transcript(monkeypatch):
    standard_inputs = []
    summary_inputs = []

    def fake_generate(**kwargs):
        assert kwargs["tools"] == []
        target = summary_inputs if len(summary_inputs) < 2 else standard_inputs
        target.append(kwargs["messages"])
        return AssistantMessage(role="assistant", content="native assistant response")

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_SUMMARY_WINDOW_SIZE", "10")

    summary = _summary_agent()
    summary_state = summary.get_init_state()
    for content in ("first user message", "second user message"):
        _, summary_state = summary.generate_next_message(
            UserMessage(role="user", content=content), summary_state
        )

    standard = _standard_agent()
    standard_state = standard.get_init_state()
    for content in ("first user message", "second user message"):
        _, standard_state = standard.generate_next_message(
            UserMessage(role="user", content=content), standard_state
        )

    assert summary_state.summary == ""
    assert summary_inputs == standard_inputs


def test_compacts_complete_native_turns_and_preserves_tool_roles(monkeypatch, tmp_path):
    calls = []
    action_responses = iter(
        [
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(
                    id="call-1", name="lookup_store", arguments={"query": "tea", "limit": 2}
                )],
            ),
            AssistantMessage(role="assistant", content="The tool result is noted."),
            AssistantMessage(role="assistant", content="A second turn is complete."),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="call-2", name="lookup_menu", arguments={"store": "Tea House"})],
            ),
            AssistantMessage(role="assistant", content="The second tool result is noted."),
        ]
    )

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["tools"] is None:
            return AssistantMessage(
                role="assistant", content=f"summary {sum(c['tools'] is None for c in calls)}",
                raw_data={"message": {"reasoning_content": "summary reasoning"}},
            )
        response = next(action_responses)
        response.raw_data = {"message": {"reasoning_content": "action reasoning"}}
        return response

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_SUMMARY_WINDOW_SIZE", "1")
    monkeypatch.setenv("VITA_SUMMARY_TRACE_PATH", str(tmp_path / "summary.jsonl"))

    agent = _summary_agent()
    state = agent.get_init_state()
    _, state = agent.generate_next_message(UserMessage(role="user", content="Find tea."), state)
    tool_result = ToolMessage(
        role="tool", id="call-1", name="lookup_store", content="Tea House", error=False
    )
    _, state = agent.generate_next_message(
        MultiToolMessage(role="tool", tool_messages=[tool_result]), state
    )

    # Tool results remain role=tool and the preceding assistant tool call is
    # still native before that complete user turn is compacted.
    second_action_messages = calls[1]["messages"]
    assert [message.role for message in second_action_messages] == [
        "system", "user", "assistant", "tool"
    ]
    assert second_action_messages[-2].tool_calls[0].name == "lookup_store"
    assert second_action_messages[-2].tool_calls[0].arguments == {"query": "tea", "limit": 2}
    assert second_action_messages[-1] is tool_result

    # Finishing turn two triggers compaction, but only after the whole turn is
    # complete. The first complete turn is what reaches the summarizer.
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Now choose one."), state
    )
    summary_call = calls[3]
    summary_input = json.loads(summary_call["messages"][1].content)
    assert [item["role"] for item in summary_input["recent_history"]] == [
        "user", "assistant", "tool", "assistant"
    ]
    assert summary_input["recent_history"][1]["tool_calls"][0]["arguments"] == {
        "query": "tea", "limit": 2
    }
    assert state.summary == "summary 1"
    assert len(state.completed_turns) == 1
    assert [message.role for message in state.recent_messages] == ["user", "assistant"]

    # The next action receives one combined leading system message, not a
    # second system message after the transcript. Its recent suffix is native.
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Check that shop's menu."), state
    )
    action_after_compaction = calls[4]["messages"]
    assert [message.role for message in action_after_compaction] == [
        "system", "user", "assistant", "user"
    ]
    assert "Current time:" in action_after_compaction[0].content
    assert "<rolling_history_summary>" in action_after_compaction[0].content

    second_tool_result = ToolMessage(
        role="tool", id="call-2", name="lookup_menu", content="Jasmine tea", error=False
    )
    _, state = agent.generate_next_message(
        MultiToolMessage(role="tool", tool_messages=[second_tool_result]), state
    )
    tool_continuation = calls[5]["messages"]
    assert [message.role for message in tool_continuation] == [
        "system", "user", "assistant", "user", "assistant", "tool"
    ]
    assert tool_continuation[-2].tool_calls[0].name == "lookup_menu"
    assert tool_continuation[-1] is second_tool_result

    records = [json.loads(line) for line in (tmp_path / "summary.jsonl").read_text().splitlines()]
    assert records[2]["summary_input_history"][1]["role"] == "assistant"
    assert records[2]["summary_input_history"][2]["role"] == "tool"
    assert records[2]["completed_turns_after"] == 1
    assert all(record["action_thinking_trace"] == "action reasoning" for record in records)


def test_compacted_native_tool_continuation_is_qwen_template_valid(monkeypatch):
    """Guard the exact failure seen in the first full k=20 attempt."""
    captured_actions = []
    responses = iter([
        AssistantMessage(role="assistant", tool_calls=[ToolCall(id="a", name="x", arguments={})]),
        AssistantMessage(role="assistant", content="done"),
        AssistantMessage(role="assistant", content="next"),
        AssistantMessage(role="assistant", tool_calls=[ToolCall(id="b", name="y", arguments={})]),
        AssistantMessage(role="assistant", content="final"),
    ])

    def fake_generate(**kwargs):
        if kwargs["tools"] is None:
            return AssistantMessage(role="assistant", content="summary")
        captured_actions.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_SUMMARY_WINDOW_SIZE", "1")
    monkeypatch.setenv(
        "VITA_INPUT_TOKENIZER_PATH", str(Path(__file__).parents[1] / "models/Qwen3.5-4B")
    )
    agent = _summary_agent()
    state = agent.get_init_state()
    _, state = agent.generate_next_message(UserMessage(role="user", content="one"), state)
    _, state = agent.generate_next_message(
        MultiToolMessage(role="tool", tool_messages=[ToolMessage(
            role="tool", id="a", name="x", content="x result", error=False
        )]), state
    )
    _, state = agent.generate_next_message(UserMessage(role="user", content="two"), state)
    _, state = agent.generate_next_message(UserMessage(role="user", content="three"), state)
    _, state = agent.generate_next_message(
        MultiToolMessage(role="tool", tool_messages=[ToolMessage(
            role="tool", id="b", name="y", content="y result", error=False
        )]), state
    )

    # The final action continues from a native tool result after compaction.
    prompt = format_messages(captured_actions[-1])
    assert _template_token_count(prompt, None, {"enable_thinking": True}) > 0


def test_bootstrap_compacts_old_complete_turns_without_dropping_assistants(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["tools"] is None
        return AssistantMessage(role="assistant", content="bootstrapped summary")

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_SUMMARY_WINDOW_SIZE", "1")
    history = [
        UserMessage(role="user", content="first"),
        AssistantMessage(role="assistant", tool_calls=[ToolCall(id="call-2", name="search", arguments={"q": "dessert"})]),
        ToolMessage(role="tool", id="call-2", name="search", content="result", error=False),
        AssistantMessage(role="assistant", content="first done"),
        UserMessage(role="user", content="second"),
        AssistantMessage(role="assistant", content="second done"),
    ]

    state = _summary_agent().get_init_state(history)

    payload = json.loads(calls[0]["messages"][1].content)
    assert [item["role"] for item in payload["recent_history"]] == ["user", "assistant", "tool", "assistant"]
    assert payload["recent_history"][1]["tool_calls"][0]["name"] == "search"
    assert state.summary == "bootstrapped summary"
    assert state.recent_messages == history[-2:]
    assert state.messages == history


def test_recent_turns_harness_keeps_only_complete_native_turns(monkeypatch):
    action_inputs = []

    def fake_generate(**kwargs):
        action_inputs.append(kwargs["messages"])
        return AssistantMessage(role="assistant", content="done")

    monkeypatch.setattr(harness, "generate", fake_generate)
    monkeypatch.setenv("VITA_RECENT_TURNS_WINDOW_SIZE", "1")
    agent = VitaRLRecentTurnsAgent(
        tools=[], domain_policy="Current time: {time}", llm="test-model",
        time="2025-01-01 12:00:00", language="english",
    )
    state = agent.get_init_state()
    _, state = agent.generate_next_message(UserMessage(role="user", content="first"), state)
    _, state = agent.generate_next_message(UserMessage(role="user", content="second"), state)
    _, state = agent.generate_next_message(UserMessage(role="user", content="third"), state)

    # The third action has exactly the previous completed turn and current
    # native user message; no summary system message is introduced.
    assert [message.role for message in action_inputs[-1]] == ["system", "user", "assistant", "user"]
    assert [message.content for message in action_inputs[-1][1:]] == ["second", "done", "third"]
    assert [message.content for message in state.recent_messages] == ["third", "done"]
