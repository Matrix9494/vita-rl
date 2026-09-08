"""Legacy VitaBench backend behind the neutral runtime dispatch boundary."""

from __future__ import annotations

import json
import threading
from typing import Any


_LOCK = threading.RLock()


class VitaBenchProxyGenerator:
    """VitaBench's message bridge to the Dressage proxy."""

    def __init__(self, request: Any) -> None:
        self.request = request
        self.turn_count = 0

    def __call__(self, model, messages, tools=None, tool_choice=None, enable_think=False, **kwargs):
        del enable_think
        import requests
        from vita.data_model.message import AssistantMessage, ToolCall
        from vita.utils.llm_utils import format_messages

        self.turn_count += 1
        body = {"model": model or self.request.agent_model, "messages": format_messages(messages), "stream": False}
        if tools:
            body["tools"] = [tool.openai_schema for tool in tools]
            body["tool_choice"] = tool_choice or "auto"
        for key in ("temperature", "max_tokens", "top_p", "seed"):
            if kwargs.get(key) is not None:
                body[key] = kwargs[key]
        if "max_tokens" not in body and self.request.sampling_params.get("max_new_tokens"):
            body["max_tokens"] = int(self.request.sampling_params["max_new_tokens"])
        response = requests.post(
            f"{self.request.dressage_proxy_url.rstrip('/')}/v1/chat/completions",
            json=body,
            headers={
                "X-Session-Id": self.request.session_id,
                "X-Instance-Id": self.request.instance_id,
                "X-Turn-Id": f"{self.request.session_id}:agent:{self.turn_count:04d}",
            },
            timeout=(10, 900),
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("Dressage proxy response has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("invalid proxy tool-call JSON") from exc
            calls.append(ToolCall(id=str(call.get("id") or ""), name=str(function.get("name") or ""), arguments=arguments))
        if not message.get("content") and not calls:
            raise RuntimeError("Dressage proxy returned an empty assistant message")
        return AssistantMessage(
            role="assistant",
            content=message.get("content"),
            tool_calls=calls or None,
            cost=0.0,
            usage=payload.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0},
            raw_data=choice,
        )


def run_vitabench_episode(request: Any):
    """Preserve upstream VitaBench behavior behind the shared API."""
    from vita.agent import llm_agent
    from vita.config import models
    from vita.data_model.message import AssistantMessage
    from vita.run import get_tasks, run_task
    from vita_rl.runtime_server import EpisodeResponse

    task = get_tasks(request.domain, [request.task_id], language=request.language)[0]
    generator = VitaBenchProxyGenerator(request)
    agent_args = dict(request.sampling_params)
    if "max_new_tokens" in agent_args and "max_tokens" not in agent_args:
        agent_args["max_tokens"] = agent_args["max_new_tokens"]
    with _LOCK:
        original = llm_agent.generate
        llm_agent.generate = generator
        try:
            simulation = run_task(
                domain=request.domain,
                task=task,
                agent="llm_agent",
                user="user_simulator",
                llm_agent=request.agent_model,
                llm_args_agent=agent_args,
                llm_user="gpt-4.1",
                llm_args_user=dict(models["gpt-4.1"]),
                llm_evaluator="gpt-4.1",
                llm_args_evaluator=dict(models["gpt-4.1"]),
                max_steps=request.max_steps,
                max_errors=request.max_errors,
                max_retries=0,
                language=request.language,
            )
        finally:
            llm_agent.generate = original
    if simulation.reward_info is None:
        raise RuntimeError("VitaBench evaluator returned no reward")
    final = next(
        (str(message.content) for message in reversed(simulation.messages) if isinstance(message, AssistantMessage) and message.content),
        "",
    )
    return EpisodeResponse(
        task_id=simulation.task_id,
        completed=True,
        reward=float(simulation.reward_info.reward),
        termination_reason=str(simulation.termination_reason),
        num_agent_turns=generator.turn_count,
        proxy_turns=generator.turn_count,
        simulation={
            "environment": "vitabench",
            "simulation_id": simulation.id,
            "duration_seconds": round(float(simulation.duration), 3),
            "message_count": len(simulation.messages),
            "agent_cost": simulation.agent_cost,
            "user_cost": simulation.user_cost,
        },
        final_assistant_response=final,
    )
