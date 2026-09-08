"""Regression coverage for the standalone vita-mini/harness integration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_mini_runs_every_harness_without_importing_vitabench():
    """Use a child process because protocol selection is intentionally import-time."""
    repository = Path(__file__).parents[1]
    script = r'''
import sys
from vita_rl import harness_protocol
from vita_rl.mini_runner import run_mini_episode
from vita_rl.state_delta import NoOpStateUpdater

assert harness_protocol.USING_VITABENCH is False
assert "vita" not in sys.modules

class ScriptedGenerator:
    def __init__(self):
        self.step = 0

    def __call__(self, *, tools, **_kwargs):
        if tools is None:
            return harness_protocol.AssistantMessage(
                role="assistant", content='{"ops":[{"op":"noop"}]}'
            )
        self.step += 1
        calls = [
            ("add_to_cart", {"product_id": "cold-brew", "quantity": 2}),
            ("set_delivery_address", {"address": "1 Mini Way"}),
            ("checkout", {}),
        ]
        if self.step <= len(calls):
            name, arguments = calls[self.step - 1]
            return harness_protocol.AssistantMessage(
                role="assistant",
                tool_calls=[harness_protocol.ToolCall(
                    id=f"call-{self.step}", name=name, arguments=arguments
                )],
            )
        return harness_protocol.AssistantMessage(
            role="assistant", content="Your coffee order is submitted."
        )

for name in (
    "vita_rl_standard", "vita_rl_stateful", "vita_rl_summary",
    "vita_rl_recent_turns", "vita_rl_state_delta",
):
    result = run_mini_episode(
        task_id="buy_coffee",
        harness_name=name,
        model="scripted-qwen",
        generate_fn=ScriptedGenerator(),
        state_updater=NoOpStateUpdater(),
    )
    assert result.success, (name, result)
    assert result.reward == 1.0, (name, result)
    assert result.num_tool_calls == 3, (name, result)

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
