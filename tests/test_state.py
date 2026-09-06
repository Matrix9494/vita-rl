from vita_rl.state import AgentWorkingState


def test_working_state_records_action_then_observation_transition():
    state = AgentWorkingState()
    state.record_action({"content": "I will search.", "tool_calls": []})
    assert state.pending_action is not None

    state.observe({"kind": "user", "content": "Please find lunch."})

    assert state.turn == 1
    assert state.pending_action is None
    assert state.latest_user_request == "Please find lunch."
    assert '"turn": 1' in state.as_prompt()


def test_working_state_records_tool_errors():
    state = AgentWorkingState()
    state.observe(
        {
            "kind": "tool",
            "results": [
                {"name": "search", "content": "not found", "error": True},
                {"name": "lookup", "content": "ok", "error": False},
            ],
        }
    )

    assert state.tool_error_count == 1
    assert state.latest_tool_results[0]["name"] == "search"
