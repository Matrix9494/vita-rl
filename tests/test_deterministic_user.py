"""Focused coverage for the no-LLM VitaBench deterministic user treatment."""

import pytest

pytest.importorskip("vita")

from vita.data_model.message import AssistantMessage, UserMessage
from vita_rl.deterministic_user import (
    DETERMINISTIC_USER_NAME,
    REMINDER,
    ZERO_USAGE,
    DeterministicTaskUser,
    register_deterministic_task_user,
)


def test_empty_history_discloses_instructions_once_then_repeats_reminder():
    user = DeterministicTaskUser(instructions="Book mild noodles.", llm="must-not-be-called")
    state = user.get_init_state()

    first, state = user.generate_next_message(
        AssistantMessage(role="assistant", content="How can I help?"), state
    )
    second, state = user.generate_next_message(
        AssistantMessage(role="assistant", content="Anything else?"), state
    )

    assert first.content == "Book mild noodles."
    assert second.content == REMINDER
    assert first.cost == second.cost == 0.0
    assert first.usage == second.usage == ZERO_USAGE
    assert first.raw_data["llm_called"] is False
    assert user.is_stop(second) is False


def test_authored_user_history_is_preserved_without_instruction_injection():
    previous = UserMessage(role="user", content="I already described the issue.")
    user = DeterministicTaskUser(instructions="Do not repeat this.")
    state = user.get_init_state([previous])
    reply, state = user.generate_next_message(
        AssistantMessage(role="assistant", content="I need more information."), state
    )

    assert state.instructions_sent is True
    assert state.messages[0].content == previous.content
    assert reply.content == REMINDER


def test_registration_is_idempotent_and_uses_the_expected_name():
    register_deterministic_task_user()
    register_deterministic_task_user()
    from vita.registry import registry

    assert registry.get_user_constructor(DETERMINISTIC_USER_NAME) is DeterministicTaskUser
