#!/usr/bin/env python3
"""Verify and compare paired deterministic-user VitaBench delivery runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("run_id", "task_set", "task_language", "selection_seed", "task_count", "tasks", "aggregate")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing summary fields: {missing}")
    if payload["task_set"] != "delivery" or payload["task_language"] != "english":
        raise ValueError(f"{path}: expected English delivery, got {payload['task_set']}/{payload['task_language']}")
    if payload["task_count"] != 100 or len(payload["tasks"]) != 100:
        raise ValueError(f"{path}: expected exactly 100 task records")
    user = payload.get("user_simulator") or {}
    if user.get("implementation") != "vita_rl_deterministic_task_user" or user.get("llm_calls") is not False:
        raise ValueError(f"{path}: not a deterministic one-shot user run")
    ids = [record.get("task_id") for record in payload["tasks"]]
    if len(set(ids)) != 100 or any(not task_id for task_id in ids):
        raise ValueError(f"{path}: task IDs are missing or duplicated")
    return payload


def metrics(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["tasks"]
    aggregate = summary["aggregate"]
    successes = sum(record.get("success") is True for record in records)
    return {
        "run_id": summary["run_id"],
        "agent": (summary.get("role_models") or {}).get("agent"),
        "successes": successes,
        "success_rate": successes / len(records),
        "mean_reward": sum(float(record.get("reward") or 0.0) for record in records) / len(records),
        "termination_reasons": dict(Counter(record.get("termination_reason") for record in records)),
        "tool_error_count": sum(int(record.get("tool_error_count") or 0) for record in records),
        "agent_steps": sum(int(record.get("agent_steps") or 0) for record in records),
        "prompt_tokens": sum(int(record.get("prompt_tokens_total") or 0) for record in records),
        "output_tokens": sum(int(record.get("output_tokens_total") or 0) for record in records),
        "wall_clock_seconds": sum(float(record.get("wall_clock_seconds") or 0.0) for record in records),
        "summary_aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-summary", type=Path, required=True)
    parser.add_argument("--terra-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qwen = load_summary(args.qwen_summary)
    terra = load_summary(args.terra_summary)
    qwen_ids = [task["task_id"] for task in qwen["tasks"]]
    terra_ids = [task["task_id"] for task in terra["tasks"]]
    if qwen["selection_seed"] != terra["selection_seed"] or Counter(qwen_ids) != Counter(terra_ids):
        raise ValueError("Runs do not have the same seeded 100-task delivery manifest")

    qwen_by_id = {task["task_id"]: task for task in qwen["tasks"]}
    terra_by_id = {task["task_id"]: task for task in terra["tasks"]}
    terra_wins = qwen_wins = ties = 0
    for task_id in qwen_ids:
        delta = float(terra_by_id[task_id].get("reward") or 0.0) - float(qwen_by_id[task_id].get("reward") or 0.0)
        if delta > 0:
            terra_wins += 1
        elif delta < 0:
            qwen_wins += 1
        else:
            ties += 1

    qwen_metrics = metrics(qwen)
    terra_metrics = metrics(terra)
    comparable = ("successes", "success_rate", "mean_reward", "tool_error_count", "agent_steps", "prompt_tokens", "output_tokens", "wall_clock_seconds")
    comparison = {
        key: terra_metrics[key] - qwen_metrics[key] for key in comparable
    }
    output = {
        "comparison_label": "paired deterministic-one-shot-autonomous VitaBench English delivery evaluation",
        "manifest_verified": True,
        "selection_seed": qwen["selection_seed"],
        "task_count": 100,
        "qwen": qwen_metrics,
        "terra": terra_metrics,
        "terra_minus_qwen": comparison,
        "per_task_reward_comparison": {"terra_wins": terra_wins, "qwen_wins": qwen_wins, "ties": ties},
        "source_summaries": {"qwen": str(args.qwen_summary), "terra": str(args.terra_summary)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "qwen_successes": qwen_metrics["successes"], "terra_successes": terra_metrics["successes"]}))


if __name__ == "__main__":
    main()
