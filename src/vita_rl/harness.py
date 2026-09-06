"""Root-owned, baseline-equivalent VitaBench agent harnesses.

The standard VitaBench ``LLMAgent`` is intentionally reimplemented here so
new harnesses can evolve in vita-rl without carrying uncommitted patches in
the external VitaBench checkout.  The baseline implementation below preserves
the standard agent's system prompt, history handling, tool schema, and LLM
generation call exactly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel
from vita.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from vita.agent.llm_agent import LLMAgent, LLMAgentState
from vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool
from vita.utils.llm_utils import generate
from vita.utils.utils import get_now, get_weekday

from vita_rl.state import AgentWorkingState


STANDARD_HARNESS_NAME = "vita_rl_standard"
STATEFUL_HARNESS_NAME = "vita_rl_stateful"


class VitaRLStandardAgent(LLMAgent):
    """A source-controlled replica of VitaBench's current ``LLMAgent``.

    This class remains an ``LLMAgent`` subclass so VitaBench's existing runner
    constructs it normally.  Future experimental harnesses should subclass
    this class rather than alter the external checkout.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time=None,
        enable_think: bool = False,
        language: str = None,
    ):
        # ``VitaRLStandardAgent`` subclasses LLMAgent so the external runner
        # recognizes it. Calling ``super()`` here would invoke LLMAgent's own
        # initializer a second time; the upstream implementation instead
        # initializes LocalAgent directly.
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.time = time + " " + get_weekday(time, language)
        self.enable_think = enable_think

    @property
    def system_prompt(self) -> str:
        if self.time is not None:
            return self.domain_policy.format(time=self.time)
        return self.domain_policy.format(time=get_now("%Y-%m-%d %H:%M:%S"))

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(message) for message in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int) -> None:
        if self.llm is None:
            raise ValueError("LLM is not set")
        current_seed = self.llm_args.get("seed")
        if current_seed is not None:
            logger.warning(
                f"Seed is already set to {current_seed}, resetting it to {seed}"
            )
        self.llm_args["seed"] = seed


def register_standard_harness() -> None:
    """Register the source-controlled baseline before VitaBench builds its CLI."""
    from vita.registry import registry

    registered = registry.get_agents()
    if STANDARD_HARNESS_NAME in registered:
        existing = registry.get_agent_constructor(STANDARD_HARNESS_NAME)
        if existing is not VitaRLStandardAgent:
            raise RuntimeError(
                f"Agent name {STANDARD_HARNESS_NAME!r} is already registered by {existing}"
            )
        return
    registry.register_agent(VitaRLStandardAgent, STANDARD_HARNESS_NAME)


class StatefulHarnessState(LLMAgentState):
    """VitaBench history plus the root-owned working state."""

    working_state: AgentWorkingState


class VitaRLStatefulAgent(VitaRLStandardAgent):
    """A state-transition harness over the baseline VitaBench agent policy.

    VitaBench invokes an agent after a user or environment message arrives.
    This method therefore realizes the requested transition as:

    ``previous action -> new observation -> state update -> next action``.

    The next action is conditioned on both the ordinary full history and the
    explicit state view. No external VitaBench source is changed.
    """

    _STATE_PROTOCOL = """\
## Stateful Harness Protocol
You are selecting the next action from the explicit working state below.
The state was updated from the latest user or tool observation after the prior
action. Use tool-confirmed facts over assumptions. If a tool result is an
error, treat the corresponding goal as unresolved. Before ending, verify that
all user constraints are satisfied by actual tool-confirmed state.

<working_state>
{working_state}
</working_state>
"""

    @staticmethod
    def _observation(message: ValidAgentInputMessage) -> dict[str, Any]:
        if isinstance(message, UserMessage):
            return {"kind": "user", "content": message.content or ""}
        tool_messages = (
            message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        )
        assert all(isinstance(tool_message, ToolMessage) for tool_message in tool_messages)
        return {
            "kind": "tool",
            "results": [
                {
                    "name": tool_message.name,
                    "content": tool_message.content or "",
                    "error": tool_message.error,
                }
                for tool_message in tool_messages
            ],
        }

    @staticmethod
    def _action(message: AssistantMessage) -> dict[str, Any]:
        return {
            "content": message.content or "",
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments}
                for call in (message.tool_calls or [])
            ],
        }

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> StatefulHarnessState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(message) for message in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )

        working_state = AgentWorkingState()
        for history_message in message_history:
            if isinstance(history_message, AssistantMessage):
                working_state.record_action(self._action(history_message))
            elif isinstance(history_message, UserMessage):
                working_state.observe(self._observation(history_message))
            elif isinstance(history_message, ToolMessage):
                working_state.observe(self._observation(history_message))

        return StatefulHarnessState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
            working_state=working_state,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: StatefulHarnessState
    ) -> tuple[AssistantMessage, StatefulHarnessState]:
        # This is the state-update phase: a new observation resolves the
        # previous pending action before the next action is requested.
        state.working_state.observe(self._observation(message))
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Qwen's chat template requires all system text to be at the start of
        # the transcript. Merge our state protocol into VitaBench's leading
        # policy message instead of inserting a system message after history.
        leading_system = state.system_messages[0]
        state_message = SystemMessage(
            role="system",
            content=(leading_system.content or "")
            + "\n\n"
            + self._STATE_PROTOCOL.format(
                working_state=state.working_state.as_prompt()
            ),
        )
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=[state_message] + state.system_messages[1:] + state.messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        state.working_state.record_action(self._action(assistant_message))
        return assistant_message, state


def register_stateful_harness() -> None:
    """Register the root-owned state-transition harness with VitaBench."""
    from vita.registry import registry

    registered = registry.get_agents()
    if STATEFUL_HARNESS_NAME in registered:
        existing = registry.get_agent_constructor(STATEFUL_HARNESS_NAME)
        if existing is not VitaRLStatefulAgent:
            raise RuntimeError(
                f"Agent name {STATEFUL_HARNESS_NAME!r} is already registered by {existing}"
            )
        return
    registry.register_agent(VitaRLStatefulAgent, STATEFUL_HARNESS_NAME)
