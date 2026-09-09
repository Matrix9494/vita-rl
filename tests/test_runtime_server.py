"""Tests for the environment-neutral Dressage runtime API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from vita_rl.runtime_server import EpisodeRequest, _sglang_normalized_tool_schema


def test_dressage_tool_schema_matches_sglang_openai_normalization():
    source = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Looks up a value.",
            "parameters": {"type": "object", "properties": {}},
            # BFCL supplies this non-OpenAI extension. SGLang discards it
            # before rendering a normal chat-completions request.
            "response": {"type": "object"},
        },
    }
    assert _sglang_normalized_tool_schema(source) == {
        "type": "function",
        "function": {
            "description": "Looks up a value.",
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        },
        "defer_loading": None,
    }


def test_legacy_vitabench_payload_normalizes_to_neutral_request():
    request = EpisodeRequest.from_dict({
        "domain": "delivery",
        "language": "chinese",
        "task_id": "10711001",
        "session_id": "session",
        "instance_id": "instance",
        "agent_model": "proxy-model",
        "dressage_proxy_url": "http://proxy/",
    })
    assert request.environment == "vitabench"
    assert request.environment_args == {"domain": "delivery", "language": "chinese"}
    assert "domain" not in request.to_dict()


def test_runtime_dispatches_vita_mini_without_vitabench_import():
    """Exercise the exact HTTP-runtime dispatch seam with a Dressage-like generator."""
    repository = Path(__file__).parents[1]
    script = r'''
import sys
from vita_rl import harness_protocol
from vita_rl import runtime_server

assert harness_protocol.USING_VITABENCH is False

class ScriptedDressageProxy:
    def __init__(self, request):
        self.turn_count = 0

    def __call__(self, *, tools, **_kwargs):
        self.turn_count += 1
        calls = [
            ("search_products", {"query": "coffee"}),
            None,
            ("search_products", {"query": "herbal tea"}),
            ("create_order", {"store_id": "store-target", "product_id": "product-herbal-tea", "quantity": 2, "address": "home", "delivery_time": "2026-01-01 12:00:00"}),
            None,
            ("modify_order", {"order_id": "mini-order-0001", "address": "office"}),
            ("pay_order", {"order_id": "mini-order-0001"}),
        ]
        if self.turn_count <= len(calls):
            action = calls[self.turn_count - 1]
            if action is None:
                return harness_protocol.AssistantMessage(role="assistant", content="Waiting for more details.")
            name, arguments = action
            return harness_protocol.AssistantMessage(
                role="assistant",
                tool_calls=[harness_protocol.ToolCall(
                    id=f"dressage-{self.turn_count}", name=name, arguments=arguments
                )],
            )
        return harness_protocol.AssistantMessage(role="assistant", content="Completed.")

runtime_server.DressageProxyGenerator = ScriptedDressageProxy
request = runtime_server.EpisodeRequest.from_dict({
    "environment": "vita-mini",
    "task_id": "delivery_revision",
    "session_id": "session",
    "instance_id": "instance",
    "agent_model": "proxy-model",
    "dressage_proxy_url": "http://dressage-proxy",
    "environment_args": {"harness": "vita_rl_standard"},
})
response = runtime_server.run_environment_episode(request)
assert response.completed and response.reward == 1.0
assert response.proxy_turns == 8
assert response.simulation["environment"] == "vita-mini"
assert "vita" not in sys.modules
'''
    environment = dict(os.environ)
    environment["VITA_RL_PROTOCOL"] = "mini"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(repository / "external" / "vita-mini" / "src")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
