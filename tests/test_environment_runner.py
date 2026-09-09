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
from vita_rl.environment_runner import run_tool_environment_episode
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
            ("search_products", {"query": "coffee"}),
            None,
            ("search_products", {"query": "herbal tea"}),
            ("create_order", {"store_id": "store-target", "product_id": "product-herbal-tea", "quantity": 2, "address": "home", "delivery_time": "2026-01-01 12:00:00"}),
            None,
            ("modify_order", {"order_id": "mini-order-0001", "address": "office"}),
            ("pay_order", {"order_id": "mini-order-0001"}),
        ]
        if self.step <= len(calls):
            action = calls[self.step - 1]
            if action is None:
                return harness_protocol.AssistantMessage(role="assistant", content="I will wait for the next detail.")
            name, arguments = action
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
    result = run_tool_environment_episode(
        environment_name="vita-mini",
        task_id="delivery_revision",
        harness_name=name,
        model="scripted-qwen",
        generate_fn=ScriptedGenerator(),
        state_updater=NoOpStateUpdater(),
    )
    assert result.success, (name, result)
    assert result.reward == 1.0, (name, result)
    assert result.num_tool_calls == 5, (name, result)
    assert len(result.user_events) == 3, (name, result.user_events)

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


def test_generated_batch_records_five_replayable_procedural_tasks():
    repository = Path(__file__).parents[1]
    script = r'''
from vita_rl import harness_protocol
from vita_rl.environment_runner import run_generated_environment_batch

class ScriptedGenerator:
    def __init__(self): self.step = 0
    def __call__(self, *, tools, **_kwargs):
        if tools is None:
            return harness_protocol.AssistantMessage(role="assistant", content="summary")
        self.step += 1
        calls = [
            ("search_products", {"query": "coffee"}), None,
            ("search_products", {"query": "herbal tea"}),
            ("create_order", {"store_id": "store-target", "product_id": "product-herbal-tea", "quantity": 2, "address": "home", "delivery_time": "2026-01-01 12:00:00"}),
            None, ("modify_order", {"order_id": "mini-order-0001", "address": "office"}),
            ("pay_order", {"order_id": "mini-order-0001"}),
        ]
        action = calls[(self.step - 1) % len(calls)]
        if action is None:
            return harness_protocol.AssistantMessage(role="assistant", content="waiting")
        name, arguments = action
        return harness_protocol.AssistantMessage(role="assistant", tool_calls=[
            harness_protocol.ToolCall(id=f"call-{self.step}", name=name, arguments=arguments)
        ])

result = run_generated_environment_batch(
    environment_name="vita-mini", num_environments=5, generation_seed=99,
    harness_name="vita_rl_summary", summary_window_size=3, model="scripted-qwen",
    generate_fn=ScriptedGenerator(), max_steps=100,
)
assert result.summary_window_size == 3
assert len(result.episodes) == 5
assert len({episode.task_id for episode in result.episodes}) == 5
assert all(episode.task_id.startswith("generated:") for episode in result.episodes)
assert all(episode.num_agent_turns <= 100 for episode in result.episodes)
'''
    environment = dict(os.environ)
    environment["VITA_RL_PROTOCOL"] = "mini"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(repository / "external" / "vita-mini" / "src")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=repository, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_bfcl_keeps_all_tool_substeps_in_the_current_user_turn():
    """Match BFCL's native ``while True`` execution for one fixed task.

    The official checker groups all calls made before an empty model response
    under the same user turn.  This guards against advancing the adapter after
    every tool result, which would silently make correct trajectories fail.
    """
    repository = Path(__file__).parents[1]
    script = r'''
import ast
from vita_rl import harness_protocol
from vita_rl.bfcl_environment import BFCLMultiTurnBaseEnvironment
from vita_rl.environment_runner import run_tool_environment_episode

env = BFCLMultiTurnBaseEnvironment()
env.reset("multi_turn_base_0")
ground_truth = env._ground_truth["multi_turn_base_0"]
tools = env.openai_tools()

def decode(call):
    node = ast.parse(call, mode="eval").body
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    properties = next(
        item["function"]["parameters"]["properties"]
        for item in tools
        if item["function"]["name"] == node.func.id
    )
    arguments = {
        key: ast.literal_eval(value)
        for key, value in zip(properties, node.args)
    }
    arguments.update({
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in node.keywords
    })
    return node.func.id, arguments

class GroundTruthGenerator:
    def __init__(self):
        self.actions = [
            action
            for user_turn in ground_truth
            for action in ([(decode(call)) for call in user_turn] + [None])
        ]
        self.index = 0

    def __call__(self, **_kwargs):
        action = self.actions[self.index]
        self.index += 1
        if action is None:
            return harness_protocol.AssistantMessage(role="assistant", content="done")
        name, arguments = action
        return harness_protocol.AssistantMessage(
            role="assistant",
            tool_calls=[harness_protocol.ToolCall(
                id=f"call-{self.index}", name=name, arguments=arguments
            )],
        )

result = run_tool_environment_episode(
    environment_name="bfcl-multi-turn-base",
    task_id="multi_turn_base_0",
    harness_name="vita_rl_standard",
    model="scripted",
    generate_fn=GroundTruthGenerator(),
    max_steps=20,
)
assert result.success, result.evaluation
assert result.reward == 1.0
assert [event["event_id"] for event in result.user_events] == [
    "multi_turn_base_0:user_turn:1",
    "multi_turn_base_0:user_turn:2",
    "multi_turn_base_0:user_turn:3",
]
assert [len(turn) for turn in result.evaluation["state"]["model_calls"]] == [
    len(user_turn) for user_turn in ground_truth
]
'''
    environment = dict(os.environ)
    environment["VITA_RL_PROTOCOL"] = "mini"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(repository / "src"),
            str(repository / "external" / "gorilla" / "berkeley-function-call-leaderboard"),
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=repository, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_openai_generator_explicitly_disables_qwen_thinking_when_requested():
    repository = Path(__file__).parents[1]
    script = r'''
import json
from unittest.mock import patch
from vita_rl.environment_runner import OpenAICompatibleGenerator

captured = {}
class Response:
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def read(self):
        return b'{"choices":[{"message":{"role":"assistant","content":"done"}}]}'

def fake_urlopen(request, timeout):
    captured["payload"] = json.loads(request.data.decode("utf-8"))
    return Response()

with patch("vita_rl.environment_runner.urlopen", fake_urlopen):
    OpenAICompatibleGenerator("http://example.invalid")(
        model="qwen", messages=[], tools=None, enable_think=False,
    )
assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
'''
    environment = dict(os.environ)
    environment["VITA_RL_PROTOCOL"] = "mini"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(repository / "external" / "vita-mini" / "src")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=repository, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
