"""A public-instruction-only, non-LLM user for controlled VitaBench runs.

This is deliberately a benchmark treatment, not a replacement for VitaBench's
LLM user simulator.  It sees the same public ``instructions`` and supplied
conversation history as an ordinary user implementation, but neither calls a
model nor receives a task's environment or evaluation criteria.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from pydantic import Field

from vita.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from vita.registry import registry
from vita.user.base import (
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)


DETERMINISTIC_USER_NAME = "vita_rl_deterministic_task_user"
REMINDER = "Please proceed using my original request."
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class DeterministicTaskUserState(UserState):
    """Conversation state with an explicit disclosure marker.

    ``messages`` remains the upstream user-facing transcript.  The marker is
    separate so an empty task history can disclose the instructions exactly
    once, while an authored history containing a user message is left intact.
    """

    instructions_sent: bool = False
    system_messages: list[SystemMessage] = Field(default_factory=list)


class DeterministicTaskUser(BaseUser):
    """Scripted VitaBench user that discloses only public task instructions."""

    def __init__(
        self,
        persona: Optional[str] = None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict[str, Any]] = None,
        language: Optional[str] = None,
        **_: Any,
    ) -> None:
        # VitaBench passes persona/language to its LLM user.  Accept them for
        # constructor compatibility but deliberately do not use either one.
        del persona, language
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> DeterministicTaskUserState:
        history = list(message_history or [])
        assert all(is_valid_user_history_message(message) for message in history), (
            "Invalid user message history. User history must contain only "
            "user messages, non-tool assistant messages, or user-requested tools."
        )
        return DeterministicTaskUserState(
            messages=deepcopy(history),
            instructions_sent=any(isinstance(message, UserMessage) for message in history),
        )

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        del message
        # This controlled user never ends a task; VitaBench's configured
        # maximum number of steps is the intentional termination boundary.
        return False

    def set_seed(self, seed: int) -> None:
        # There is no stochastic model call to seed.
        del seed

    def generate_next_message(
        self, message: ValidUserInputMessage, state: DeterministicTaskUserState
    ) -> tuple[UserMessage, DeterministicTaskUserState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        if not state.instructions_sent:
            response = str(self.instructions or "")
            state.instructions_sent = True
        else:
            response = REMINDER

        user_message = UserMessage(
            role="user",
            content=response,
            cost=0.0,
            usage=deepcopy(ZERO_USAGE),
            raw_data={"implementation": DETERMINISTIC_USER_NAME, "llm_called": False},
        )
        state.messages.append(user_message)
        return user_message, state


def register_deterministic_task_user() -> None:
    """Register the controlled user once, without altering VitaBench sources."""
    users = registry.get_users()
    if DETERMINISTIC_USER_NAME in users:
        existing = registry.get_user_constructor(DETERMINISTIC_USER_NAME)
        if existing is not DeterministicTaskUser:
            raise RuntimeError(
                f"User name {DETERMINISTIC_USER_NAME!r} is already registered by {existing}"
            )
        return
    registry.register_user(DeterministicTaskUser, DETERMINISTIC_USER_NAME)
