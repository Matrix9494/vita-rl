#!/usr/bin/env python3
"""Create reproducible, Base-only diagnostics for a BFCL GRPO run."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def update(rollout_id: int) -> int:
    return 0 if rollout_id == 0 else rollout_id + 1


def phase(value: int) -> str:
    if value <= 20:
        return "1-20"
    if value <= 40:
        return "21-40"
    if value <= 60:
        return "41-60"
    if value <= 80:
        return "61-80"
    return "81-100"


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def eval_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points, raw = [], []
    for row in rows:
        outcomes = row.get("task_outcomes", [])
        values = [float(item.get("terminal_reward") or 0) for item in outcomes]
        item = {
            "update": update(int(row["rollout_id"])),
            "rollout_id": int(row["rollout_id"]),
            "category": "multi_turn_base",
            "correct": int(sum(value > 0 for value in values)),
            "total": len(values),
            "accuracy": _mean(values) or 0.0,
        }
        points.append(item)
        for outcome in outcomes:
            raw.append({"update": item["update"], "category": "multi_turn_base", **outcome})
    return sorted(points, key=lambda value: value["update"]), raw


def transitions(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_update: dict[int, dict[str, int]] = {}
    for outcome in raw:
        by_update.setdefault(int(outcome["update"]), {})[str(outcome["task_id"])] = int(float(outcome.get("terminal_reward") or 0) > 0)
    if not by_update:
        return []
    baseline = by_update.get(0, {})
    transitions_out = []
    previous = 0
    for current in sorted(key for key in by_update if key != 0):
        current_tasks = by_update[current]
        reference = baseline if previous == 0 else by_update[previous]
        common = sorted(set(reference) & set(current_tasks))
        gains = [task for task in common if not reference[task] and current_tasks[task]]
        losses = [task for task in common if reference[task] and not current_tasks[task]]
        transitions_out.append({
            "from_update": previous,
            "to_update": current,
            "compared_tasks": len(common),
            "both_correct": sum(reference[task] and current_tasks[task] for task in common),
            "both_wrong": sum(not reference[task] and not current_tasks[task] for task in common),
            "failure_to_success": gains,
            "success_to_failure": losses,
            "net_correct_change": len(gains) - len(losses),
        })
        previous = current
    return transitions_out


def group_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries, coverage, all_groups = [], Counter(), []
    for row in rows:
        value = update(int(row["rollout_id"]))
        groups = row.get("groups", [])
        p_hist, reward_values, variances, response_tokens = Counter(), [], [], []
        truncations = excluded = group_size_mismatch = 0
        for group in groups:
            members = group.get("members", [])
            rewards = [float(member.get("terminal_reward") or 0) for member in members]
            successes = int(round(sum(rewards)))
            p_hist[f"{successes}/8"] += 1
            reward_values.extend(rewards)
            variances.append(float(group.get("reward_variance") or 0))
            if len(members) != 8:
                group_size_mismatch += 1
            for member in members:
                coverage[str(member.get("task_id"))] += 1
                response_tokens.append(float(member.get("response_tokens") or 0))
                truncations += int(bool(member.get("truncated")))
                excluded += int(member.get("fully_excluded_segments") or 0)
        summary = {
            "update": value, "phase": phase(value), "num_groups": len(groups),
            "group_size_mismatch": group_size_mismatch,
            "mean_terminal_reward": _mean(reward_values),
            "mean_group_reward": _mean([float(group.get("mean_terminal_reward") or 0) for group in groups]),
            "mean_group_reward_variance": _mean(variances),
            "zero_variance_groups": sum(value == 0 for value in variances),
            "mixed_groups": sum(0 < int(key.split("/")[0]) < 8 for key, count in p_hist.items() for _ in range(count)),
            "all_zero_groups": p_hist["0/8"], "all_one_groups": p_hist["8/8"],
            "p_histogram": {f"{key}/8": p_hist[f"{key}/8"] for key in range(9)},
            "mean_response_tokens": _mean(response_tokens),
            "truncated_trajectories": truncations, "fully_excluded_segments": excluded,
        }
        summaries.append(summary)
        all_groups.append(summary)
    phase_rows = []
    for name in ("1-20", "21-40", "41-60", "61-80", "81-100"):
        current = [row for row in all_groups if row["phase"] == name]
        if current:
            phase_rows.append({"phase": name, "updates": len(current), "mean_terminal_reward": _mean([row["mean_terminal_reward"] for row in current if row["mean_terminal_reward"] is not None]), "mean_group_variance": _mean([row["mean_group_reward_variance"] for row in current if row["mean_group_reward_variance"] is not None])})
    return summaries, {
        "sampling": "cyclic shuffled 160-task train manifest; no replacement within each epoch, reshuffled between epochs",
        "expected_prompt_appearances": 400,
        "observed_prompt_appearances": sum(coverage.values()),
        "unique_task_ids": len(coverage),
        "task_appearance_histogram": dict(sorted(coverage.items())),
        "phase_statistics": phase_rows,
    }


def training_metrics(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    metric_re = re.compile(r"(?:rollout|step)\s+(\d+):\s+(\{.*\})")
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = metric_re.search(line)
        if not match:
            continue
        try:
            metrics = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if isinstance(metrics, dict):
            rows.append({"index": int(match.group(1)), **metrics})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row if not isinstance(row[field], (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_root
    out = args.output_dir or root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    eval_points, eval_raw = eval_summary(jsonl(root / "bfcl-heldout-eval.jsonl"))
    groups, coverage = group_summary(jsonl(root / "bfcl-train-groups.jsonl"))
    report = {
        "scope": {"environment": "bfcl-multi-turn-base", "train_split": 160, "heldout_split": 40, "categories": ["multi_turn_base"], "unavailable_categories": ["multi_turn_long_context", "multi_turn_miss_param", "multi_turn_miss_func"]},
        "evaluation_points": eval_points, "paired_transitions": transitions(eval_raw),
        "training_group_statistics": groups, "coverage": coverage,
        "raw_paths": {"heldout_per_task": str(root / "bfcl-heldout-eval.jsonl"), "train_per_group": str(root / "bfcl-train-groups.jsonl")},
        "training_metrics": training_metrics(root / "logs" / "grpo.log"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    write_csv(out / "heldout_accuracy.csv", eval_points)
    write_csv(out / "training_group_statistics.csv", groups)
    write_csv(out / "training_metrics.csv", report["training_metrics"])
    lines = ["# BFCL GRPO diagnostics", "", "Scope: deterministic 160/40 `multi_turn_base` split only. Long-context, missing-parameter, and missing-function categories are intentionally unavailable for this run.", "", "## Held-out exact-match", "", "| Update | Correct | Total | Accuracy |", "|---:|---:|---:|---:|"]
    lines += [f"| {row['update']} | {row['correct']} | {row['total']} | {row['accuracy']:.3f} |" for row in eval_points]
    lines += ["", "Raw records and full statistics: `report.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(out), "evaluations": len(eval_points), "training_updates_logged": len(groups)}, sort_keys=True))


if __name__ == "__main__":
    main()
