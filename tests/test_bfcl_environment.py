"""Parity tests for the native BFCL multi-turn environment adapter."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


def test_bfcl_environment_uses_native_tools_and_exact_checker(tmp_path: Path):
    repository = Path(__file__).parents[1]
    script = r'''
import ast
import json
from vita_rl.bfcl_environment import BFCLMultiTurnBaseEnvironment, bfcl_task_splits

def decode(call, tools):
    node = ast.parse(call, mode="eval").body
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    properties = next(
        item["function"]["parameters"]["properties"]
        for item in tools
        if item["function"]["name"] == node.func.id
    )
    args = {
        key: ast.literal_eval(value)
        for key, value in zip(properties, node.args)
    }
    args.update({keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords})
    return node.func.id, args

train_ids, eval_ids = bfcl_task_splits()
assert len(train_ids) == 160
assert len(eval_ids) == 40
assert set(train_ids).isdisjoint(eval_ids)
assert len(set(train_ids) | set(eval_ids)) == 200

env = BFCLMultiTurnBaseEnvironment()
observation = env.reset("multi_turn_base_0")
assert "final_report.pdf" in observation["user_message"]
tools = env.openai_tools()
assert any(tool["function"]["name"] == "mv" for tool in tools)
for turn in env._ground_truth["multi_turn_base_0"]:
    for call in turn:
        name, arguments = decode(call, tools)
        result = env.call_tool(name, arguments)
        assert result.ok, (call, result)
    env.next_user_event(agent_turn=1)
evaluation = env.evaluate()
assert evaluation.success
assert evaluation.reward == 1.0
assert evaluation.state["checker"]["valid"] is True

failed = BFCLMultiTurnBaseEnvironment()
failed.reset("multi_turn_base_0")
assert failed.evaluate().reward == 0.0
print(json.dumps({"tools": len(tools), "train": len(train_ids), "eval": len(eval_ids)}))
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
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_bfcl_manifest_writer_is_disjoint_and_runtime_compatible(tmp_path: Path):
    repository = Path(__file__).parents[1]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(repository / "src"),
            str(repository / "external" / "gorilla" / "berkeley-function-call-leaderboard"),
        ]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "build_bfcl_grpo_manifests.py"),
            "--train-output",
            str(train_path),
            "--eval-output",
            str(eval_path),
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    evaluation = [json.loads(line) for line in eval_path.read_text().splitlines()]
    train_ids = {item["metadata"]["task_id"] for item in train}
    eval_ids = {item["metadata"]["task_id"] for item in evaluation}
    assert len(train) == 160 and len(evaluation) == 40
    assert train_ids.isdisjoint(eval_ids)
    assert all(
        item["metadata"]["environment"] == "bfcl-multi-turn-base"
        and item["metadata"]["environment_args"]["enable_think"] is False
        for item in train + evaluation
    )
