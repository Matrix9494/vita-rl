"""Tests for the environment-neutral Dressage runtime API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from vita_rl.runtime_server import EpisodeRequest


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
            ("add_to_cart", {"product_id": "cold-brew", "quantity": 2}),
            ("set_delivery_address", {"address": "1 Mini Way"}),
            ("checkout", {}),
        ]
        if self.turn_count <= len(calls):
            name, arguments = calls[self.turn_count - 1]
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
    "task_id": "buy_coffee",
    "session_id": "session",
    "instance_id": "instance",
    "agent_model": "proxy-model",
    "dressage_proxy_url": "http://dressage-proxy",
    "environment_args": {"harness": "vita_rl_standard"},
})
response = runtime_server.run_environment_episode(request)
assert response.completed and response.reward == 1.0
assert response.proxy_turns == 4
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
