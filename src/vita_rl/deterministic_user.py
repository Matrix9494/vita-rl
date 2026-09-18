"""A public-task-context, non-LLM user for controlled VitaBench runs.

This is deliberately a benchmark treatment, not a replacement for VitaBench's
LLM user simulator. It sees the public ``instructions``, supplied user profile,
and conversation history that VitaBench supplies to a user implementation, but
neither calls a model nor receives task environment state or evaluation criteria.
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
PROFILE_CONTEXT_HEADER = "User profile (use only to resolve details unspecified in the request):"
REVIEW_REMINDER = (
    "Before finishing, do not merely restate or assert that the work is correct. Use "
    "the tools to inspect the actual task-visible state and verify it line by line "
    "against my original request and public profile. If any requested action, quantity, "
    "location, product attribute, constraint, timing, or status is incomplete or "
    "incorrect, use the tools now to correct it. Otherwise, finish the task."
)


class DeterministicTaskUserState(UserState):
    """Conversation state with an explicit disclosure marker.

    ``messages`` remains the upstream user-facing transcript. The explicit
    marker distinguishes our initial disclosure from arbitrary authored user
    turns already present in a task history.
    """

    instructions_sent: bool = False
    review_turns_remaining: int = 2
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
        # VitaBench passes a public user profile to every user implementation.
        # Unlike environment state or evaluator criteria, it is user-owned
        # context and can resolve an otherwise ambiguous request (for example,
        # a home versus work delivery destination). This treatment never calls
        # a model and never receives non-user task state.
        del language, llm, llm_args
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(
                "DeterministicTaskUser requires non-empty public task instructions"
            )
        super().__init__(instructions=instructions, llm=None, llm_args=None)
        self.persona = persona.strip() if isinstance(persona, str) else ""

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
        """Disclose the request and only its public user-owned context once."""
        if state.instructions_sent:
            raise RuntimeError("Controlled task instructions were already disclosed")
        content = self.instructions
        if self.persona:
            content = f"{content}\n\n{PROFILE_CONTEXT_HEADER}\n{self.persona}"
        user_message = UserMessage(
            role="user",
            content=content,
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

    async def review_message(
        self, state: DeterministicTaskUserState
    ) -> tuple[UserMessage, DeterministicTaskUserState]:
        """Give a bounded generic, fact-free execution review request.

        The message is invariant across tasks: it neither reads environment or
        evaluator state nor derives task-specific corrections.
        """
        if state.review_turns_remaining < 1:
            raise RuntimeError("No deterministic review turns remain")
        user_message = UserMessage(
            role="user",
            content=REVIEW_REMINDER,
            cost=0.0,
            usage=deepcopy(ZERO_USAGE),
            raw_data={"implementation": DETERMINISTIC_USER_NAME, "llm_called": False,
                      "message_kind": "fact_free_review"},
        )
        state.review_turns_remaining -= 1
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
