"""Regression tests for the non-interactive deterministic VitaBench treatment."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("vita")

from vita.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from vita.data_model.simulation import TerminationReason
from vita.user.user_simulator import UserSimulator
from vita_rl.autonomous_user_runner import (
    AUTONOMOUS_EXECUTION_DIRECTIVE,
    AutonomousDeterministicOrchestrator,
)
from vita_rl.deterministic_user import (
    DETERMINISTIC_USER_NAME,
    PROFILE_CONTEXT_HEADER,
    REVIEW_REMINDER,
    ZERO_USAGE,
    DeterministicTaskUser,
)


class ScriptedAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def get_init_state(self, message_history=None):
        del message_history
        return SimpleNamespace(system_messages=[SystemMessage(role="system", content="agent")])

    def generate_next_message(self, message, state):
        self.inputs.append(message)
        return self.responses.pop(0), state

    @staticmethod
    def is_stop(message):
        return "###STOP###" in (message.content or "")

    def set_seed(self, seed):
        del seed


class LoopingAgent(ScriptedAgent):
    def __init__(self):
        super().__init__([])

    def generate_next_message(self, message, state):
        self.inputs.append(message)
        return _tool_call(), state


def _tool_call():
    return AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id="call-1", name="lookup", arguments={})],
    )


class FakeEnvironment:
    def __init__(self):
        self.tools = SimpleNamespace(db=SimpleNamespace(time="2025-01-01 00:00:00"))

    def get_response(self, call):
        return ToolMessage(
            role="tool", id=call.id, name=call.name, content="observation", requestor="assistant"
        )


def _orchestrator(agent, user, max_steps=10):
    return AutonomousDeterministicOrchestrator(
        domain="delivery",
        agent=agent,
        user=user,
        environment=FakeEnvironment(),
        task=SimpleNamespace(id="controlled-test", message_history=[]),
        max_steps=max_steps,
    )


def test_one_time_disclosure_is_async_and_has_no_model_usage():
    user = DeterministicTaskUser(instructions="Complete the public task.", llm="ignored")
    state = asyncio.run(user.get_init_state())
    initial, state = asyncio.run(user.initial_message(state))

    assert inspect.iscoroutinefunction(DeterministicTaskUser.get_init_state)
    assert inspect.iscoroutinefunction(DeterministicTaskUser.generate_next_message)
    assert initial.content == "Complete the public task."
    assert initial.raw_data == {"implementation": DETERMINISTIC_USER_NAME, "llm_called": False}
    assert initial.cost == 0.0 and initial.usage == ZERO_USAGE
    assert user.llm is None and user.llm_args == {}
    assert state.instructions_sent is True
    with pytest.raises(RuntimeError, match="non-interactive"):
        asyncio.run(user.generate_next_message(AssistantMessage(role="assistant", content="done"), state))


def test_initial_disclosure_includes_only_supplied_public_profile():
    user = DeterministicTaskUser(
        instructions="Deliver dinner.",
        persona="{'home address': '1 Public Lane', 'dietary restrictions': 'No alcohol'}",
    )
    state = asyncio.run(user.get_init_state())
    initial, _ = asyncio.run(user.initial_message(state))

    assert initial.content == (
        "Deliver dinner.\n\n"
        f"{PROFILE_CONTEXT_HEADER}\n"
        "{'home address': '1 Public Lane', 'dietary restrictions': 'No alcohol'}"
    )
    assert "evaluation" not in initial.content.lower()


def test_review_is_bounded_fact_free_and_has_no_model_usage():
    user = DeterministicTaskUser(instructions="Complete the task.")
    state = asyncio.run(user.get_init_state())
    review, state = asyncio.run(user.review_message(state))

    assert review.content == REVIEW_REMINDER
    assert review.usage == ZERO_USAGE and review.cost == 0.0
    assert review.raw_data["llm_called"] is False
    assert review.raw_data["message_kind"] == "fact_free_review"
    with pytest.raises(RuntimeError, match="No deterministic review"):
        asyncio.run(user.review_message(state))


def test_authored_user_history_does_not_falsely_mark_controlled_disclosure_sent():
    user = DeterministicTaskUser(instructions="Complete the public task.")
    state = asyncio.run(
        user.get_init_state([UserMessage(role="user", content="An authored prior turn")])
    )

    assert state.instructions_sent is False
    assert state.messages[0].content == "An authored prior turn"


def test_autonomous_tool_loop_has_no_synthetic_user_turn_between_tools():
    agent = ScriptedAgent([
        _tool_call(),
        AssistantMessage(role="assistant", content="###STOP###"),
        AssistantMessage(role="assistant", content="Reviewed. ###STOP###"),
    ])
    orchestrator = _orchestrator(agent, DeterministicTaskUser(instructions="Use tools."))
    orchestrator.initialize()
    orchestrator.step()  # initial user -> agent tool call
    orchestrator.step()  # agent tool call -> environment
    orchestrator.step()  # environment observation -> agent first completion
    orchestrator.step()  # bounded review -> agent stop

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_STOP
    assert [message.role for message in orchestrator.trajectory] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert len(orchestrator.agent_state.system_messages) == 1
    assert AUTONOMOUS_EXECUTION_DIRECTIVE in orchestrator.agent_state.system_messages[0].content
    assert isinstance(agent.inputs[0], UserMessage)
    assert isinstance(agent.inputs[1], ToolMessage)
    assert sum(isinstance(message, UserMessage) for message in orchestrator.trajectory) == 2


def test_one_review_occurs_before_agent_stop():
    agent = ScriptedAgent([
        AssistantMessage(role="assistant", content="Finished. ###STOP###"),
        AssistantMessage(role="assistant", content="Reviewed. ###STOP###"),
    ])
    simulation = _orchestrator(agent, DeterministicTaskUser(instructions="Finish."), max_steps=10).run()

    assert simulation.termination_reason == TerminationReason.AGENT_STOP.value
    assert [message.role for message in simulation.messages] == ["user", "assistant", "user", "assistant"]
    assert simulation.messages[2].content == REVIEW_REMINDER


def test_true_tool_loop_still_hits_max_steps_safety_bound():
    simulation = _orchestrator(LoopingAgent(), DeterministicTaskUser(instructions="Loop."), max_steps=3).run()

    assert simulation.termination_reason == TerminationReason.MAX_STEPS.value
    assert sum(message.role == "user" for message in simulation.messages) == 1


def test_input_validation_and_stock_user_behavior_are_unchanged():
    with pytest.raises(ValueError, match="non-empty"):
        DeterministicTaskUser(instructions="  ")
    assert not inspect.iscoroutinefunction(UserSimulator.generate_next_message)
