"""Message and tool protocol used by the root-owned harnesses.

The existing VitaBench objects remain the default so its runner and tests are
unchanged.  ``VITA_RL_PROTOCOL=mini`` selects the deliberately small local
implementation used by :mod:`vita_rl.mini_runner`; that path has no import of
the external VitaBench package.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


_mode = os.environ.get("VITA_RL_PROTOCOL", "auto").lower()
if _mode not in {"auto", "vita", "mini"}:
    raise ValueError("VITA_RL_PROTOCOL must be auto, vita, or mini")


if _mode != "mini":
    try:
        from vita.agent.base import (  # type: ignore[no-redef]
            LocalAgent,
            ValidAgentInputMessage,
            is_valid_agent_history_message,
        )
        from vita.agent.llm_agent import LLMAgent, LLMAgentState  # type: ignore[no-redef]
        from vita.data_model.message import (  # type: ignore[no-redef]
            AssistantMessage,
            Message,
            MultiToolMessage,
            SystemMessage,
            ToolMessage,
            UserMessage,
        )
        from vita.environment.tool import Tool  # type: ignore[no-redef]
        from vita.utils.llm_utils import generate  # type: ignore[no-redef]
        from vita.utils.utils import get_now, get_weekday  # type: ignore[no-redef]
    except ImportError:
        if _mode == "vita":
            raise
        _mode = "mini"


if _mode == "mini":
    class SystemMessage(BaseModel):
        role: str = "system"
        content: str | None = None
        turn_idx: int | None = None
        timestamp: str | None = None


    class ToolCall(BaseModel):
        id: str = ""
        name: str
        arguments: dict[str, Any]
        requestor: str = "assistant"


    class ParticipantMessage(BaseModel):
        role: str
        content: str | None = None
        tool_calls: list[ToolCall] | None = None
        turn_idx: int | None = None
        timestamp: str | None = None
        cost: float | None = None
        usage: dict[str, Any] | None = None
        raw_data: dict[str, Any] | None = None

        def is_tool_call(self) -> bool:
            return bool(self.tool_calls)


    class AssistantMessage(ParticipantMessage):
        role: str = "assistant"


    class UserMessage(ParticipantMessage):
        role: str = "user"


    class ToolMessage(BaseModel):
        id: str
        name: str
        role: str = "tool"
        content: str | None = None
        requestor: str = "assistant"
        error: bool = False
        turn_idx: int | None = None
        timestamp: str | None = None


    class MultiToolMessage(BaseModel):
        role: str = "tool"
        tool_messages: list[ToolMessage]


    Message = SystemMessage | AssistantMessage | UserMessage | ToolMessage | MultiToolMessage
    ValidAgentInputMessage = UserMessage | ToolMessage | MultiToolMessage

    class LLMAgentState(BaseModel):
        system_messages: list[SystemMessage]
        messages: list[Any]


    class LocalAgent:
        STOP_TOKEN = "###STOP###"

        def __init__(self, tools: list[Any], domain_policy: str):
            self.tools = tools
            self.domain_policy = domain_policy

        @classmethod
        def is_stop(cls, message: AssistantMessage) -> bool:
            return cls.STOP_TOKEN in (message.content or "")


    class LLMAgent(LocalAgent):
        """Compatibility parent; concrete behavior is owned by vita_rl.harness."""


    class Tool:
        """A tiny schema carrier sufficient for a harness/model call."""

        def __init__(self, name: str, openai_schema: dict[str, Any]):
            self.name = name
            self._openai_schema = openai_schema

        @property
        def openai_schema(self) -> dict[str, Any]:
            return self._openai_schema


    def is_valid_agent_history_message(message: Any) -> bool:
        return isinstance(message, (AssistantMessage, UserMessage, ToolMessage))


    def get_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)


    def get_weekday(time: str | None, language: str | None = None) -> str:
        del language
        if not time:
            return ""
        try:
            return datetime.fromisoformat(time.replace("Z", "+00:00")).strftime(" %A")
        except ValueError:
            return ""


    def generate(*args: Any, **kwargs: Any) -> AssistantMessage:
        del args, kwargs
        raise RuntimeError(
            "The standalone harness protocol needs an injected generate_fn. "
            "Use vita_rl.mini_runner.run_mini_episode()."
        )


USING_VITABENCH = _mode != "mini"


__all__ = [
    "AssistantMessage",
    "LLMAgent",
    "LLMAgentState",
    "LocalAgent",
    "Message",
    "MultiToolMessage",
    "SystemMessage",
    "Tool",
    "ToolMessage",
    "UserMessage",
    "USING_VITABENCH",
    "ValidAgentInputMessage",
    "generate",
    "get_now",
    "get_weekday",
    "is_valid_agent_history_message",
]
