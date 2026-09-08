"""Run registered deterministic tool environments through vita-rl harnesses.

This module intentionally owns the episode loop that VitaBench normally
provides: it gives the initial static user message to a harness, executes its
tool calls against a selected environment, and feeds structured tool messages
back.  It never creates a GPT user or invokes an LLM evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vita_rl.environments import tool_environment_registry


class Generator(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EnvironmentEpisodeResult:
    """Audit record for one fully deterministic tool-environment episode."""

    environment: str
    task_id: str
    harness: str
    reward: float
    success: bool
    termination_reason: str
    num_agent_turns: int
    num_tool_calls: int
    num_tool_errors: int
    final_assistant_response: str
    evaluation: dict[str, Any]
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SchemaTool:
    """Schema-only bridge from an environment to a harness tool list."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema
        self.name = str(schema["function"]["name"])

    @property
    def openai_schema(self) -> dict[str, Any]:
        return self._schema


def _load_standalone_protocol() -> tuple[Any, Any]:
    """Load the existing harness module under its VitaBench-free protocol.

    Selection happens before importing either protocol or harness, which makes
    ``python -m vita_rl.environment_runner`` safe in an environment where VitaBench
    happens to be installed.  A process that previously imported the
    VitaBench-backed harness remains usable, but is not considered a
    standalone mini run.
    """
    protocol = sys.modules.get("vita_rl.harness_protocol")
    if protocol is None:
        os.environ.setdefault("VITA_RL_PROTOCOL", "mini")
        from vita_rl import harness_protocol as protocol  # pylint: disable=import-outside-toplevel
    if getattr(protocol, "USING_VITABENCH", False):
        raise RuntimeError(
            "vita_rl.harness_protocol was already loaded with VitaBench. "
            "Run a tool environment in a fresh process or set VITA_RL_PROTOCOL=mini "
            "before importing vita_rl.harness."
        )
    from vita_rl import harness  # pylint: disable=import-outside-toplevel
    return harness, protocol


def _harness_constructor(name: str, harness: Any) -> type:
    constructors = {
        "vita_rl_standard": harness.VitaRLStandardAgent,
        "vita_rl_stateful": harness.VitaRLStatefulAgent,
        "vita_rl_summary": harness.VitaRLSummaryAgent,
        "vita_rl_recent_turns": harness.VitaRLRecentTurnsAgent,
        "vita_rl_state_delta": harness.VitaRLStateDeltaAgent,
    }
    try:
        return constructors[name]
    except KeyError as exc:
        raise ValueError(f"Unknown harness {name!r}; choose from {', '.join(constructors)}") from exc


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json")
    elif hasattr(message, "dict"):
        value = message.dict()
    else:
        value = {"role": getattr(message, "role", "unknown"), "content": getattr(message, "content", None)}
    if "tool_messages" in value:
        return {
            "role": "tool",
            "tool_messages": [_message_to_dict(tool_message) for tool_message in value["tool_messages"]],
        }
    return value


def run_tool_environment_episode(
    *,
    environment_name: str,
    task_id: str,
    harness_name: str,
    model: str,
    generate_fn: Generator,
    llm_args: dict[str, Any] | None = None,
    max_steps: int = 30,
    max_errors: int = 5,
    enable_think: bool = False,
    state_updater: Any | None = None,
) -> EnvironmentEpisodeResult:
    """Run one registered environment task through any root-owned harness."""
    if max_steps < 1 or max_errors < 1:
        raise ValueError("max_steps and max_errors must be positive")
    harness, protocol = _load_standalone_protocol()
    env = tool_environment_registry.create(environment_name)
    observation = env.reset(task_id)
    tools = [SchemaTool(schema) for schema in env.openai_tools()]
    constructor = _harness_constructor(harness_name, harness)
    kwargs: dict[str, Any] = {
        "tools": tools,
        "domain_policy": env.agent_policy(),
        "llm": model,
        "llm_args": dict(llm_args or {}),
        "time": observation["logical_time"].replace("T", " ").replace("Z", ""),
        "enable_think": enable_think,
        "generate_fn": generate_fn,
    }
    if harness_name == "vita_rl_state_delta" and state_updater is not None:
        kwargs["state_updater"] = state_updater
    agent = constructor(**kwargs)
    state = agent.get_init_state()
    incoming: Any = protocol.UserMessage(role="user", content=observation["user_message"])
    num_tool_calls = 0
    num_tool_errors = 0
    final_response = ""
    termination_reason = "max_steps"

    for agent_turn in range(1, max_steps + 1):
        assistant, state = agent.generate_next_message(incoming, state)
        final_response = assistant.content or ""
        calls = assistant.tool_calls or []
        if not calls:
            termination_reason = "agent_stop" if agent.is_stop(assistant) else "assistant_final"
            break
        tool_messages = []
        for call in calls:
            num_tool_calls += 1
            result = env.call_tool(call.name, call.arguments)
            num_tool_errors += int(not result.ok)
            tool_messages.append(
                protocol.ToolMessage(
                    role="tool",
                    id=call.id or f"tool-call-{agent_turn}-{num_tool_calls}",
                    name=call.name,
                    content=json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
                    error=not result.ok,
                )
            )
        if num_tool_errors >= max_errors:
            termination_reason = "max_tool_errors"
            break
        incoming = protocol.MultiToolMessage(role="tool", tool_messages=tool_messages)

    evaluation = env.evaluate()
    messages = [_message_to_dict(message) for message in state.messages]
    return EnvironmentEpisodeResult(
        environment=environment_name,
        task_id=task_id,
        harness=harness_name,
        reward=evaluation.reward,
        success=evaluation.success,
        termination_reason=termination_reason,
        num_agent_turns=len([message for message in state.messages if isinstance(message, protocol.AssistantMessage)]),
        num_tool_calls=num_tool_calls,
        num_tool_errors=num_tool_errors,
        final_assistant_response=final_response,
        evaluation=evaluation.to_dict(),
        messages=messages,
    )


class OpenAICompatibleGenerator:
    """Minimal OpenAI-chat-completions generator for Qwen/SGLang endpoints."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 600):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def __call__(
        self,
        *,
        model: str,
        messages: list[Any],
        tools: list[Any] | None,
        enable_think: bool = False,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _, protocol = _load_standalone_protocol()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_openai_message(message) for message in messages],
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = [tool.openai_schema for tool in tools]
            payload["tool_choice"] = "auto"
        for key in ("temperature", "max_tokens", "top_p", "seed"):
            if kwargs.get(key) is not None:
                payload[key] = kwargs[key]
        if "max_tokens" not in payload and kwargs.get("max_new_tokens") is not None:
            payload["max_tokens"] = kwargs["max_new_tokens"]
        # SGLang forwards this field to Qwen chat templates when supported.
        if enable_think:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        request = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # nosec B310: caller owns endpoint
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach model endpoint {self.base_url}: {exc.reason}") from exc
        try:
            raw_choice = body["choices"][0]
            raw_message = raw_choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model response has no choices[0].message") from exc
        calls = []
        for raw_call in raw_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Model returned non-JSON function arguments") from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("Model tool-call arguments must decode to an object")
            calls.append(protocol.ToolCall(
                id=str(raw_call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=arguments,
            ))
        return protocol.AssistantMessage(
            role="assistant",
            content=raw_message.get("content"),
            tool_calls=calls or None,
            usage=body.get("usage"),
            raw_data=raw_choice,
        )


def _openai_message(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", None)
    if role == "assistant":
        calls = getattr(message, "tool_calls", None)
        return {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in calls or []
            ] or None,
        }
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": getattr(message, "id"),
            "name": getattr(message, "name"),
            "content": getattr(message, "content", None),
        }
    return {"role": role, "content": getattr(message, "content", None)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic tool environment through a vita-rl harness.")
    parser.add_argument("--environment", default="vita-mini", choices=tool_environment_registry.names())
    parser.add_argument("--task-id", default="buy_coffee")
    parser.add_argument("--harness", default="vita_rl_standard", choices=(
        "vita_rl_standard", "vita_rl_stateful", "vita_rl_summary",
        "vita_rl_recent_turns", "vita_rl_state_delta",
    ))
    parser.add_argument("--model", required=True, help="Qwen model identifier accepted by the OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("ENVIRONMENT_BASE_URL", "http://127.0.0.1:30000/v1/chat/completions"))
    parser.add_argument("--api-key", default=os.environ.get("ENVIRONMENT_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--enable-think", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON episode-record destination.")
    args = parser.parse_args()
    result = run_tool_environment_episode(
        environment_name=args.environment,
        task_id=args.task_id,
        harness_name=args.harness,
        model=args.model,
        generate_fn=OpenAICompatibleGenerator(args.base_url, args.api_key),
        llm_args={"temperature": args.temperature, "max_tokens": args.max_tokens},
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        enable_think=args.enable_think,
    )
    rendered = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
