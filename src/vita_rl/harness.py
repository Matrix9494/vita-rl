"""Root-owned, baseline-equivalent VitaBench agent harnesses.

The standard VitaBench ``LLMAgent`` is intentionally reimplemented here so
new harnesses can evolve in vita-rl without carrying uncommitted patches in
the external VitaBench checkout.  The baseline implementation below preserves
the standard agent's system prompt, history handling, tool schema, and LLM
generation call exactly.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger
from pydantic import Field
from vita_rl.harness_protocol import (
    AssistantMessage,
    LLMAgent,
    LLMAgentState,
    LocalAgent,
    Message,
    MultiToolMessage,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
    ValidAgentInputMessage,
    generate,
    get_now,
    get_weekday,
    is_valid_agent_history_message,
)

from vita_rl.state import AgentWorkingState
from vita_rl.state_delta import (
    LLMStateUpdater,
    NoOpStateUpdater,
    ReplayStateUpdater,
    StateUpdater,
    apply_state_delta,
    empty_canonical_state,
)
from vita_rl.state_delta.prompts import action_state_protocol


STANDARD_HARNESS_NAME = "vita_rl_standard"
STATEFUL_HARNESS_NAME = "vita_rl_stateful"
STATE_DELTA_HARNESS_NAME = "vita_rl_state_delta"


def _positive_window_size(*environment_names: str, default: int) -> int:
    """Read one positive history-window setting with legacy-name fallback."""
    for environment_name in environment_names:
        raw_value = os.environ.get(environment_name)
        if raw_value is None:
            continue
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{environment_name} must be a positive integer") from exc
        if value < 1:
            raise ValueError(f"{environment_name} must be a positive integer")
        return value
    return default


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
        generate_fn: Callable[..., AssistantMessage] | None = None,
    ):
        # ``VitaRLStandardAgent`` subclasses LLMAgent so the external runner
        # recognizes it. Calling ``super()`` here would invoke LLMAgent's own
        # initializer a second time; the upstream implementation instead
        # initializes LocalAgent directly.
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.time = time + " " + get_weekday(time, language)
        # Every LLM call made by a harness reads this one policy bit.  In
        # particular, auxiliary calls (summary/delta proposals) do not
        # silently turn reasoning on or off relative to action generation.
        self.enable_think = bool(enable_think)
        self._generate_fn = generate_fn

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
        self._append_event_to_transcript(state, message)
        assistant_message = self._generate_action(state.system_messages + state.messages)
        state.messages.append(assistant_message)
        self._save_standard_thinking_trace(
            message=message,
            assistant_message=assistant_message,
            turn=len([item for item in state.messages if isinstance(item, AssistantMessage)]),
        )
        return assistant_message, state

    def _generate(
        self, *, messages: list[Message], tools: list[Tool] | None
    ) -> AssistantMessage:
        """Run one harness-owned LLM request under the common call policy."""
        return (self._generate_fn or generate)(
            model=self.llm,
            tools=tools,
            messages=messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )

    def _generate_action(self, messages: list[Message]) -> AssistantMessage:
        return self._generate(messages=messages, tools=self.tools)

    def _generate_auxiliary(self, messages: list[Message]) -> AssistantMessage:
        """Generate non-action text without exposing environment tools."""
        return self._generate(messages=messages, tools=None)

    @staticmethod
    def _append_event_to_transcript(
        state: LLMAgentState, message: ValidAgentInputMessage
    ) -> None:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    @staticmethod
    def _action(message: AssistantMessage) -> dict[str, Any]:
        return {
            "content": message.content or "",
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in (message.tool_calls or [])
            ],
        }

    @staticmethod
    def _observation(
        message: ValidAgentInputMessage, *, include_tool_ids: bool = True
    ) -> dict[str, Any]:
        """Preserve an incoming user/tool event for an audit or state prompt."""
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
                    **({"id": tool_message.id} if include_tool_ids else {}),
                    "name": tool_message.name,
                    "content": tool_message.content or "",
                    "error": tool_message.error,
                }
                for tool_message in tool_messages
            ],
        }

    @classmethod
    def _prompt_observation(cls, message: ValidAgentInputMessage) -> UserMessage:
        """Encode one event where a native tool role would lack its call context."""
        observation = cls._observation(message)
        if observation["kind"] == "user":
            return UserMessage(role="user", content=observation["content"])
        return UserMessage(
            role="user",
            content=(
                "[LATEST TOOL OBSERVATION]\n"
                + json.dumps(observation["results"], ensure_ascii=False, sort_keys=True)
            ),
        )

    @staticmethod
    def _thinking_trace(message: AssistantMessage) -> str | None:
        """Return provider-preserved reasoning without interpreting it."""
        raw_data = message.raw_data
        if not isinstance(raw_data, dict):
            return None
        raw_message = raw_data.get("message")
        if not isinstance(raw_message, dict):
            return None
        reasoning = raw_message.get("reasoning_content") or raw_message.get("reasoning")
        return reasoning if isinstance(reasoning, str) else None

    def _save_standard_thinking_trace(
        self,
        *,
        message: ValidAgentInputMessage,
        assistant_message: AssistantMessage,
        turn: int,
    ) -> None:
        """Append an opt-in action/reasoning audit record for the baseline.

        This is deliberately a side-channel audit sink: it neither changes
        the standard harness's model context nor supplies state on a later
        run. The full VitaBench result continues to retain its usual message
        transcript independently.
        """
        trace_path = os.environ.get("VITA_STANDARD_THINKING_TRACE_PATH")
        if not trace_path:
            return
        record = {
            "turn": turn,
            "raw_observation": self._observation(message),
            "agent_action": self._action(assistant_message),
            "action_thinking_trace": self._thinking_trace(assistant_message),
        }
        self._append_jsonl(trace_path, record)

    @staticmethod
    def _append_jsonl(trace_path: str, record: dict[str, Any]) -> None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

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
    """Local audit transcript plus the root-owned working state.

    ``messages`` is retained for callback compatibility and post-run
    inspection.  It is deliberately *not* the model context for this
    harness; see :meth:`VitaRLStatefulAgent._prompt_observation`.
    """

    working_state: AgentWorkingState


class VitaRLStatefulAgent(VitaRLStandardAgent):
    """A state-transition harness over the baseline VitaBench agent policy.

    VitaBench invokes an agent after a user or environment message arrives.
    This method therefore realizes the requested transition as:

    ``previous action -> new observation -> state update -> next action``.

    The next action is conditioned on an explicit state view and exactly one
    latest observation.  Older observations and actions are represented only
    by the structured state, never copied into the model context. No external
    VitaBench source is changed.
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
        # The original stateful tracker intentionally does not treat tool-call
        # IDs as semantic state.  Keep that prompt contract while sharing the
        # representation logic with the baseline and delta harness.
        return VitaRLStandardAgent._observation(message, include_tool_ids=False)

    @staticmethod
    def _save_new_traces(state: StatefulHarnessState, trace_start: int) -> None:
        """Append state transitions to an optional, job-scoped JSONL trace.

        The launcher sets ``VITA_STATE_TRACE_PATH``. The tracker always keeps
        its in-memory traces, while this opt-in sink makes a real trajectory
        inspectable after VitaBench finishes without altering VitaBench output.
        """
        trace_path = os.environ.get("VITA_STATE_TRACE_PATH")
        if not trace_path:
            return
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for trace in state.working_state.task_state.debug_traces[trace_start:]:
                handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")

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
        trace_start = len(state.working_state.task_state.debug_traces)
        state.working_state.observe(self._observation(message))
        self._append_event_to_transcript(state, message)

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
        assistant_message = self._generate_action(
            # The state system message plus exactly this new observation is
            # the entire model window.  ``state.messages`` is intentionally
            # excluded: it is retained only as a local audit transcript.
            messages=[state_message, self._prompt_observation(message)],
        )
        state.messages.append(assistant_message)
        state.working_state.record_action(self._action(assistant_message))
        self._save_new_traces(state, trace_start)
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


SUMMARY_HARNESS_NAME = "vita_rl_summary"
RECENT_TURNS_HARNESS_NAME = "vita_rl_recent_turns"


class NativeTurnHarnessState(LLMAgentState):
    """Full transcript plus a bounded prompt suffix split at whole turns.

    A turn starts with a user message and includes every assistant/tool
    exchange until the assistant gives a response without tool calls.  Native
    roles and tool-call IDs remain untouched in both fields.
    """

    completed_turns: list[list[Message]] = Field(default_factory=list)
    active_turn: list[Message] = Field(default_factory=list)

    @property
    def recent_messages(self) -> list[Message]:
        return [message for turn in self.completed_turns for message in turn] + self.active_turn


class NativeTurnContextAgent(VitaRLStandardAgent):
    """Shared native-transcript bookkeeping for bounded-history harnesses."""

    @staticmethod
    def _native_messages(message: ValidAgentInputMessage) -> list[Message]:
        return list(message.tool_messages) if isinstance(message, MultiToolMessage) else [message]

    @staticmethod
    def _is_turn_complete(message: Message) -> bool:
        return isinstance(message, AssistantMessage) and not message.tool_calls

    def _append_native_message(self, state: NativeTurnHarnessState, message: Message) -> None:
        if isinstance(message, UserMessage) and state.active_turn:
            # Preserve an unusual incomplete segment rather than dropping it.
            state.completed_turns.append(state.active_turn)
            state.active_turn = []
        state.active_turn.append(message)
        if self._is_turn_complete(message):
            state.completed_turns.append(state.active_turn)
            state.active_turn = []

    def _restore_native_history(
        self, state: NativeTurnHarnessState, message_history: list[Message]
    ) -> None:
        for history_message in message_history:
            self._append_native_message(state, history_message)


class SummaryHarnessState(NativeTurnHarnessState):
    """Transcript plus unconstrained rolling natural-language memory.

    ``summary`` is model-authored prose, not a CPU-defined task schema. The
    completed_turns and active_turn retain a verbatim native transcript. A
    turn starts with a user message and includes every assistant/tool exchange
    until the assistant gives a response without tool calls. Compaction only
    moves completed turns into ``summary``; it never splits a turn.
    """

    summary: str = ""
    event_turn: int = 0
    transition_log: list[dict[str, Any]] = Field(default_factory=list)


class VitaRLSummaryAgent(NativeTurnContextAgent):
    """A bounded-context agent with model-written rolling history summaries.

    The action model receives a model-written summary of the old transcript
    prefix plus a verbatim suffix of the latest
    ``VITA_SUMMARY_WINDOW_SIZE`` completed native interaction turns (and an
    in-progress turn when one exists). The CPU only chooses whole-turn
    cutoffs; it never extracts semantic fields or prescribes summary content.
    """

    _SUMMARY_SYSTEM = """\
You maintain an unconstrained, compact history summary for another agent.
Update the existing summary using the supplied recent native interaction
history. The history may contain user messages, assistant responses, assistant
tool calls with IDs and arguments, and tool results. Keep facts, user requests,
constraints, decisions, tool-confirmed outcomes, identifiers, unfinished work,
and failures that matter for the next action. Discard superseded or irrelevant
detail. Do not invent facts. Return only the replacement natural-language
summary; do not use JSON, headings, or explain the summarization process.
"""

    _SUMMARY_CONTEXT = """\
<rolling_history_summary>
{summary}
</rolling_history_summary>
The native messages following this context are the recent interaction history.
"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.summary_window_size = _positive_window_size(
            "VITA_HISTORY_WINDOW_SIZE", "VITA_SUMMARY_WINDOW_SIZE", default=5
        )

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        """Serialize native messages without changing their role or tool calls."""
        if isinstance(message, MultiToolMessage):
            return {
                "role": "tool",
                "tool_messages": [
                    VitaRLSummaryAgent._serialize_message(tool_message)
                    for tool_message in message.tool_messages
                ],
            }
        return message.model_dump(mode="json")

    @classmethod
    def _serialize_messages(cls, messages: list[Message]) -> list[dict[str, Any]]:
        return [cls._serialize_message(message) for message in messages]

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> SummaryHarnessState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(message) for message in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        state = SummaryHarnessState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )
        self._restore_native_history(state, message_history)
        # A resume may end at a completed natural-language assistant response.
        # Such a turn is eligible for compaction; a pending tool-call turn is
        # deliberately retained in full until it receives its tool result.
        self._compact(state)
        return state

    def _summarize(
        self, summary: str, history: list[Message]
    ) -> AssistantMessage:
        summary_input = json.dumps(
            {
                "current_summary": summary,
                "recent_history": self._serialize_messages(history),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return self._generate_auxiliary(
            [
                SystemMessage(role="system", content=self._SUMMARY_SYSTEM),
                UserMessage(role="user", content=summary_input),
            ]
        )

    def _compact(
        self, state: SummaryHarnessState
    ) -> tuple[list[Message], AssistantMessage | None]:
        """Summarize only old *complete turns*, retaining a full-turn suffix.

        A failed or blank summary never causes the prefix to be discarded. This
        avoids silently dropping messages merely because compaction did not
        produce usable memory.
        """
        if len(state.completed_turns) <= self.summary_window_size:
            return [], None

        prefix_turns = state.completed_turns[: -self.summary_window_size]
        prefix = [message for turn in prefix_turns for message in turn]
        response = self._summarize(state.summary, prefix)
        if response.content:
            state.summary = response.content
            state.completed_turns = state.completed_turns[-self.summary_window_size :]
        return prefix, response

    @classmethod
    def _save_summary_trace(cls, record: dict[str, Any]) -> None:
        trace_path = os.environ.get("VITA_SUMMARY_TRACE_PATH")
        if not trace_path:
            return
        cls._append_jsonl(trace_path, record)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SummaryHarnessState
    ) -> tuple[AssistantMessage, SummaryHarnessState]:
        incoming_messages = self._native_messages(message)
        state.event_turn += 1
        summary_before = state.summary

        self._append_event_to_transcript(state, message)
        for incoming_message in incoming_messages:
            self._append_native_message(state, incoming_message)

        # Before the first compaction this exactly matches the baseline
        # message structure: domain policy followed by the native transcript.
        leading_system = state.system_messages[0]
        action_messages: list[Message] = [leading_system]
        if state.summary:
            # Qwen's native chat template permits exactly one system message,
            # and it must be first. Keep the summary in that first message
            # while preserving every remaining interaction event natively.
            action_messages[0] = SystemMessage(
                role="system",
                content=(leading_system.content or "")
                + "\n\n"
                + self._SUMMARY_CONTEXT.format(summary=state.summary),
            )
        action_messages.extend(state.recent_messages)
        assistant_message = self._generate_action(action_messages)
        state.messages.append(assistant_message)
        self._append_native_message(state, assistant_message)
        # A compaction boundary occurs only after this complete interaction
        # turn has ended. Tool-call turns remain intact until their terminal
        # assistant response arrives.
        summary_input_history, summary_response = self._compact(state)
        record = {
            "turn": state.event_turn,
            "window_size": self.summary_window_size,
            "raw_input_history": self._serialize_messages(incoming_messages),
            "summary_before": summary_before,
            "summary_after": state.summary,
            "summary_updated": summary_response is not None,
            "summary_input_history": self._serialize_messages(summary_input_history),
            "recent_history_after": self._serialize_messages(state.recent_messages),
            "completed_turns_after": len(state.completed_turns),
            "active_turn_after": self._serialize_messages(state.active_turn),
            "summary_thinking_trace": (
                self._thinking_trace(summary_response) if summary_response else None
            ),
            "agent_action": self._action(assistant_message),
            "action_thinking_trace": self._thinking_trace(assistant_message),
        }
        state.transition_log.append(record)
        self._save_summary_trace(record)
        return assistant_message, state


def register_summary_harness() -> None:
    """Register the rolling-summary harness with VitaBench."""
    from vita.registry import registry

    registered = registry.get_agents()
    if SUMMARY_HARNESS_NAME in registered:
        existing = registry.get_agent_constructor(SUMMARY_HARNESS_NAME)
        if existing is not VitaRLSummaryAgent:
            raise RuntimeError(
                f"Agent name {SUMMARY_HARNESS_NAME!r} is already registered by {existing}"
            )
        return
    registry.register_agent(VitaRLSummaryAgent, SUMMARY_HARNESS_NAME)


class RecentTurnsHarnessState(NativeTurnHarnessState):
    """Full native transcript plus a bounded, verbatim recent-turn view."""


class VitaRLRecentTurnsAgent(NativeTurnContextAgent):
    """Standard agent context truncated to the latest complete native turns.

    Unlike ``vita_rl_summary``, this ablation uses no model-written memory:
    old turns are simply absent from the action prompt. ``state.messages``
    remains the complete native transcript for the runner and diagnostics.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.recent_turns_window_size = _positive_window_size(
            "VITA_HISTORY_WINDOW_SIZE",
            "VITA_RECENT_TURNS_WINDOW_SIZE",
            # This last fallback maintains the command-line compatibility used
            # by the initial paired summary/recent-turn experiments.
            "VITA_SUMMARY_WINDOW_SIZE",
            default=10,
        )

    def _trim_complete_turns(self, state: RecentTurnsHarnessState) -> None:
        if len(state.completed_turns) > self.recent_turns_window_size:
            state.completed_turns = state.completed_turns[-self.recent_turns_window_size :]

    def _append_native_message(self, state: RecentTurnsHarnessState, message: Message) -> None:
        super()._append_native_message(state, message)
        # Truncation starts only after a whole native interaction turn.
        self._trim_complete_turns(state)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> RecentTurnsHarnessState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(message) for message in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        state = RecentTurnsHarnessState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )
        self._restore_native_history(state, message_history)
        return state

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: RecentTurnsHarnessState
    ) -> tuple[AssistantMessage, RecentTurnsHarnessState]:
        self._append_event_to_transcript(state, message)
        for incoming_message in self._native_messages(message):
            self._append_native_message(state, incoming_message)

        assistant_message = self._generate_action(
            [state.system_messages[0], *state.recent_messages]
        )
        state.messages.append(assistant_message)
        self._append_native_message(state, assistant_message)
        return assistant_message, state


def register_recent_turns_harness() -> None:
    """Register the native recent-turn truncation ablation."""
    from vita.registry import registry

    registered = registry.get_agents()
    if RECENT_TURNS_HARNESS_NAME in registered:
        existing = registry.get_agent_constructor(RECENT_TURNS_HARNESS_NAME)
        if existing is not VitaRLRecentTurnsAgent:
            raise RuntimeError(
                f"Agent name {RECENT_TURNS_HARNESS_NAME!r} is already registered by {existing}"
            )
        return
    registry.register_agent(VitaRLRecentTurnsAgent, RECENT_TURNS_HARNESS_NAME)


class StateDeltaHarnessState(LLMAgentState):
    """Transcript plus the sole semantic state for the delta experiment.

    ``canonical_state`` can change only through an accepted LLM/oracle delta.
    The harness retains raw observations in the transcript, trace, and
    unresolved-evidence buffer, but never parses them into semantic state.
    """

    canonical_state: dict[str, Any]
    state_metadata: dict[str, dict[str, Any]]
    unresolved_evidence: list[dict[str, Any]]
    turn: int = 0
    transition_log: list[dict[str, Any]] = Field(default_factory=list)
    last_termination_decision: dict[str, Any] = Field(default_factory=dict)


class VitaRLStateDeltaAgent(VitaRLStandardAgent):
    """LLM/oracle semantic delta -> CPU structural validation -> commit.

    The CPU does not inspect user language or tool text to infer entities,
    constraints, IDs, transactions, goals, or completion. It decides only
    whether an edit is structurally legal and applies accepted edits
    mechanically and atomically.
    """

    def __init__(self, *args: Any, state_updater: StateUpdater | None = None, **kwargs: Any):
        # Delta was introduced as a reasoning-first experiment.  Preserve that
        # default for direct construction, while an explicit caller/CLI value
        # remains the sole policy used by both updater and action requests.
        # The sixth positional argument of the baseline constructor is also
        # ``enable_think``; do not add a keyword when a legacy caller used it.
        if len(args) < 6:
            kwargs.setdefault("enable_think", True)
        super().__init__(*args, **kwargs)
        # ``LLMStateUpdater`` consumes only an updater's final JSON string.
        # Retain its reasoning until the surrounding transition is recorded.
        self._updater_thinking_trace: str | None = None
        self._last_stop_accepted = False
        self.state_updater = state_updater or self._configured_updater()

    def _configured_updater(self) -> StateUpdater:
        mode = os.environ.get("VITA_STATE_DELTA_UPDATER", "llm").lower()
        if mode == "llm":
            return LLMStateUpdater(self._generate_delta)
        if mode == "noop":
            return NoOpStateUpdater()
        if mode == "replay":
            path = os.environ.get("VITA_STATE_DELTA_REPLAY_PATH")
            if not path:
                raise ValueError("VITA_STATE_DELTA_REPLAY_PATH is required for replay updater")
            return ReplayStateUpdater(path)
        raise ValueError("VITA_STATE_DELTA_UPDATER must be llm, noop, or replay")

    def _generate_delta(self, system: str, user: str) -> str:
        """Dedicated model call; no tools or action-policy history are sent."""
        proposal = self._generate_auxiliary(
            [
                SystemMessage(role="system", content=system),
                UserMessage(role="user", content=user),
            ]
        )
        self._updater_thinking_trace = self._thinking_trace(proposal)
        return proposal.content or ""

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> StateDeltaHarnessState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(message) for message in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        # History is intentionally not replayed into semantic state: doing so
        # would require a CPU interpretation of prior language/tool output.
        # Preserve raw events as unresolved evidence for an updater to recover
        # with future LLM/oracle deltas.
        bootstrap_evidence = [
            {"id": f"e{index}", "turn": 0, "source": "resume_history", "observation": self._observation(message),
             "reason": "history was not replayed into the delta database"}
            for index, message in enumerate(message_history, start=1)
            if not isinstance(message, AssistantMessage)
        ]
        return StateDeltaHarnessState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
            canonical_state=empty_canonical_state(),
            state_metadata={},
            unresolved_evidence=bootstrap_evidence,
        )

    def _stop_decision(self, state: StateDeltaHarnessState, requested: bool) -> dict[str, Any]:
        nonterminal_goals = {
            goal_id: goal["status"]
            for goal_id, goal in state.canonical_state["goals"].items()
            if goal["status"] not in {"done", "cancelled"}
        }
        unverified_terminal_goals = {
            goal_id: goal["status"]
            for goal_id, goal in state.canonical_state["goals"].items()
            if goal["status"] in {"done", "cancelled"}
            and not self._has_terminal_evidence(state, goal_id, goal["status"])
        }
        accepted = bool(
            requested
            and state.canonical_state["goals"]
            and not nonterminal_goals
            and not unverified_terminal_goals
        )
        return {
            "requested": requested,
            "accepted": accepted,
            "reason": (
                "all canonical goals are terminal with accepted provenance" if accepted else
                "no canonical goal has been recorded" if not state.canonical_state["goals"] else
                "active/pending/blocked canonical goals remain" if nonterminal_goals else
                "terminal canonical goals lack accepted provenance" if unverified_terminal_goals else
                "agent did not request stop"
            ),
            "nonterminal_goals": nonterminal_goals,
            "unverified_terminal_goals": unverified_terminal_goals,
            "unresolved_evidence_count": len(state.unresolved_evidence),
            "unresolved_evidence_warning": bool(requested and state.unresolved_evidence),
        }

    @staticmethod
    def _has_terminal_evidence(
        state: StateDeltaHarnessState, goal_id: str, status: str
    ) -> bool:
        """Require CPU-side provenance, not an LLM's status proposal alone."""
        source = (state.state_metadata.get(f"/goals/{goal_id}/status") or {}).get("source_type")
        return source == "tool_observation" or (status == "cancelled" and source == "explicit_user")

    @classmethod
    def _save_trace(cls, record: dict[str, Any]) -> None:
        trace_path = os.environ.get("VITA_STATE_DELTA_TRACE_PATH")
        if not trace_path:
            return
        cls._append_jsonl(trace_path, record)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: StateDeltaHarnessState
    ) -> tuple[AssistantMessage, StateDeltaHarnessState]:
        observation = self._observation(message)
        turn = state.turn + 1
        state_before = deepcopy(state.canonical_state)
        metadata_before = deepcopy(state.state_metadata)
        unresolved_before = deepcopy(state.unresolved_evidence)
        # Avoid carrying a prior LLM trace into replay/no-op/custom updates.
        self._updater_thinking_trace = None
        proposal = self.state_updater.update(
            state=state_before,
            metadata=metadata_before,
            observation=observation,
            unresolved_evidence=unresolved_before,
            turn_idx=turn,
        )
        update = apply_state_delta(
            state=state_before,
            delta=proposal.delta,
            metadata=metadata_before,
            observation=observation,
            turn_idx=turn,
            unresolved_evidence=unresolved_before,
        )
        state.canonical_state = update.state
        state.state_metadata = update.metadata
        state.unresolved_evidence = update.unresolved_evidence
        state.turn = turn
        self._append_event_to_transcript(state, message)

        leading_system = state.system_messages[0]
        action_system = SystemMessage(
            role="system",
            content=(leading_system.content or "") + "\n\n" + action_state_protocol(
                state=state.canonical_state,
                observation=observation,
            ),
        )
        assistant_message = self._generate_action(
            [
                action_system,
                self._prompt_observation(message),
            ]
        )
        state.messages.append(assistant_message)
        action_thinking_trace = self._thinking_trace(assistant_message)
        decision = self._stop_decision(
            state, self.STOP_TOKEN in (assistant_message.content or "")
        )
        self._last_stop_accepted = decision["accepted"]
        state.last_termination_decision = decision
        record = {
            "turn": turn,
            "raw_observation": observation,
            "state_before_update": state_before,
            "metadata_before_update": metadata_before,
            "updater_input": {
                "canonical_semantic_state": state_before,
                "newest_raw_observation": observation,
                "unresolved_evidence": unresolved_before,
            },
            "raw_model_delta": proposal.raw_delta,
            "updater_thinking_trace": self._updater_thinking_trace,
            "parsed_delta": proposal.delta,
            "updater": proposal.updater_name,
            "accepted_ops": update.accepted_ops,
            "rejected_ops": update.rejected_ops,
            "validation_error": update.error,
            "state_after_update": deepcopy(state.canonical_state),
            "metadata_after_update": deepcopy(state.state_metadata),
            "unresolved_evidence": deepcopy(state.unresolved_evidence),
            "action_input": {
                "canonical_semantic_state": deepcopy(state.canonical_state),
                "newest_raw_observation": observation,
            },
            "agent_action": self._action(assistant_message),
            "action_thinking_trace": action_thinking_trace,
            "termination_decision": decision,
        }
        state.transition_log.append(record)
        self._save_trace(record)
        return assistant_message, state

    def is_stop(self, message: AssistantMessage) -> bool:
        """Let VitaBench terminate only after the harness accepts the request."""
        if self.STOP_TOKEN not in (message.content or ""):
            return False
        return self._last_stop_accepted


def register_state_delta_harness() -> None:
    """Register the experimental delta harness without changing prior agents."""
    from vita.registry import registry

    registered = registry.get_agents()
    if STATE_DELTA_HARNESS_NAME in registered:
        existing = registry.get_agent_constructor(STATE_DELTA_HARNESS_NAME)
        if existing is not VitaRLStateDeltaAgent:
            raise RuntimeError(
                f"Agent name {STATE_DELTA_HARNESS_NAME!r} is already registered by {existing}"
            )
        return
    registry.register_agent(VitaRLStateDeltaAgent, STATE_DELTA_HARNESS_NAME)
