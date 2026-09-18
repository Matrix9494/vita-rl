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
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class DeterministicTaskUserState(UserState):
    """Conversation state with an explicit disclosure marker.

    ``messages`` remains the upstream user-facing transcript. The explicit
    marker distinguishes our initial disclosure from arbitrary authored user
    turns already present in a task history.
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
        # VitaBench passes persona/language/model settings to its LLM user.
        # Accept these solely for constructor compatibility. This treatment
        # deliberately retains only public instructions and never has a model.
        del persona, language, llm, llm_args
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(
                "DeterministicTaskUser requires non-empty public task instructions"
            )
        super().__init__(instructions=instructions, llm=None, llm_args=None)

    async def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> DeterministicTaskUserState:
        history = list(message_history or [])
        assert all(is_valid_user_history_message(message) for message in history), (
            "Invalid user message history. User history must contain only "
            "user messages, non-tool assistant messages, or user-requested tools."
        )
        instructions_sent = any(
            isinstance(message, UserMessage)
            and (message.raw_data or {}).get("implementation") == DETERMINISTIC_USER_NAME
            for message in history
        )
        return DeterministicTaskUserState(
            messages=deepcopy(history),
            instructions_sent=instructions_sent,
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

    async def initial_message(
        self, state: DeterministicTaskUserState
    ) -> tuple[UserMessage, DeterministicTaskUserState]:
        """Disclose public instructions once for the autonomous runner."""
        if state.instructions_sent:
            raise RuntimeError("Controlled task instructions were already disclosed")
        user_message = UserMessage(
            role="user",
            content=self.instructions,
            cost=0.0,
            usage=deepcopy(ZERO_USAGE),
            raw_data={"implementation": DETERMINISTIC_USER_NAME, "llm_called": False},
        )
        state.instructions_sent = True
        state.messages.append(user_message)
        return user_message, state

    async def generate_next_message(
        self, message: ValidUserInputMessage, state: DeterministicTaskUserState
    ) -> tuple[UserMessage, DeterministicTaskUserState]:
        del message, state
        raise RuntimeError(
            "DeterministicTaskUser is non-interactive and must run through "
            "the vita_rl autonomous deterministic-user orchestrator"
        )


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
