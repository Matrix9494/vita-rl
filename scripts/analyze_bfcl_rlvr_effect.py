#!/usr/bin/env python3
"""Compare a BFCL RLVR checkpoint with its frozen update-0 baseline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def score(outcome: dict[str, Any]) -> int:
    return int(float(outcome.get("terminal_reward") or 0.0) > 0.0)


def update_from_rollout(rollout_id: int) -> int:
    return 0 if rollout_id == 0 else rollout_id + 1


def failure_kind(outcome: dict[str, Any]) -> str:
    if score(outcome):
        return "success"
    simulation = outcome.get("simulation") or {}
    checker = ((simulation.get("evaluation") or {}).get("state") or {}).get("checker") or {}
    error_type = checker.get("error_type")
    if error_type:
        return str(error_type)
    failed = (simulation.get("evaluation") or {}).get("failed_conditions") or []
    if failed:
        return str(failed[0])
    return str(outcome.get("termination_reason") or "unknown_failure")


def compact(outcome: dict[str, Any]) -> dict[str, Any]:
    simulation = outcome.get("simulation") or {}
    return {
        "task_id": outcome.get("task_id"),
        "reward": score(outcome),
        "failure_kind": failure_kind(outcome),
        "termination_reason": outcome.get("termination_reason"),
        "agent_turns": outcome.get("num_agent_turns"),
        "tool_calls": simulation.get("num_tool_calls"),
        "tool_errors": simulation.get("num_tool_errors"),
    }


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def mean_metric(outcomes: list[dict[str, Any]], key: str) -> float | None:
    values = [item[key] for item in outcomes if isinstance(item.get(key), (int, float))]
    return mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--best-update", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_root
    rows = [json.loads(line) for line in (root / "bfcl-heldout-eval.jsonl").read_text().splitlines() if line]
    evaluations = {update_from_rollout(int(row["rollout_id"])): row for row in rows}
    selected_update = args.best_update
    if selected_update is None and (root / "best-heldout.json").exists():
        selected_update = update_from_rollout(json.loads((root / "best-heldout.json").read_text())["best_rollout_id"])
    if selected_update is None:
        selected_update = max(evaluations)
    baseline = {str(item["task_id"]): item for item in evaluations[0]["task_outcomes"]}
    selected = {str(item["task_id"]): item for item in evaluations[selected_update]["task_outcomes"]}
    shared = sorted(set(baseline) & set(selected))
    gains = [task for task in shared if not score(baseline[task]) and score(selected[task])]
    losses = [task for task in shared if score(baseline[task]) and not score(selected[task])]
    stable_success = [task for task in shared if score(baseline[task]) and score(selected[task])]
    stable_failure = [task for task in shared if not score(baseline[task]) and not score(selected[task])]
    records = {
        "gains": [{"baseline": compact(baseline[task]), "selected": compact(selected[task])} for task in gains],
        "losses": [{"baseline": compact(baseline[task]), "selected": compact(selected[task])} for task in losses],
        "remaining_failures": [compact(selected[task]) for task in stable_failure + losses],
    }
    curve = []
    for point, row in sorted(evaluations.items()):
        outcomes = row["task_outcomes"]
        curve.append({"update": point, "correct": sum(score(item) for item in outcomes), "total": len(outcomes), "accuracy": mean(score(item) for item in outcomes)})
    selected_compact = [compact(item) for item in selected.values()]
    report = {
        "scope": {"environment": "bfcl_multi_turn_base", "heldout_tasks": len(shared), "baseline_update": 0, "selected_update": selected_update, "selected_checkpoint": str(json.loads((root / "best-heldout.json").read_text()).get("checkpoint")) if (root / "best-heldout.json").exists() else None},
        "paired_effect": {"baseline_correct": sum(score(baseline[task]) for task in shared), "selected_correct": sum(score(selected[task]) for task in shared), "absolute_change": (sum(score(selected[task]) for task in shared) - sum(score(baseline[task]) for task in shared)) / len(shared), "gains": len(gains), "losses": len(losses), "stable_success": len(stable_success), "stable_failure": len(stable_failure), "exact_two_sided_mcnemar_p": exact_mcnemar(len(gains), len(losses))},
        "selected_failure_kinds": dict(Counter(failure_kind(item) for item in selected.values() if not score(item))),
        "selected_behavior": {"all_tasks": {key: mean_metric(selected_compact, key) for key in ("agent_turns", "tool_calls", "tool_errors")}, "successes": {key: mean_metric([item for item in selected_compact if item["reward"]], key) for key in ("agent_turns", "tool_calls", "tool_errors")}, "failures": {key: mean_metric([item for item in selected_compact if not item["reward"]], key) for key in ("agent_turns", "tool_calls", "tool_errors")}},
        "evaluation_curve": curve,
        "cases": records,
    }
    output = args.output_dir or root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "rlvr_effect.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    paired = report["paired_effect"]
    lines = ["# BFCL RLVR effect analysis", "", f"Update 0 baseline versus selected update {selected_update} on the same 40 held-out `multi_turn_base` tasks.", "", "## Paired result", "", f"- Baseline: {paired['baseline_correct']}/40 ({paired['baseline_correct'] / 40:.1%})", f"- Selected: {paired['selected_correct']}/40 ({paired['selected_correct'] / 40:.1%})", f"- Net: {paired['selected_correct'] - paired['baseline_correct']:+d}/40 ({paired['absolute_change']:+.1%})", f"- Gains / regressions / stable successes / stable failures: {paired['gains']} / {paired['losses']} / {paired['stable_success']} / {paired['stable_failure']}", f"- Exact two-sided McNemar p-value: {paired['exact_two_sided_mcnemar_p']:.4f}", "", "## Selected remaining failure modes", ""]
    lines += [f"- `{kind}`: {count}" for kind, count in report["selected_failure_kinds"].items()]
    lines += ["", "## Task IDs", "", f"- Gains: {', '.join(gains) or 'none'}", f"- Regressions: {', '.join(losses) or 'none'}", f"- Remaining failures: {', '.join(item['task_id'] for item in records['remaining_failures']) or 'none'}", "", "The JSON sidecar includes per-case turns, tool calls/errors, termination reasons, and checker failure types. Raw tool traces and model responses remain in `bfcl-heldout-eval.jsonl`."]
    (output / "rlvr_effect.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"selected_update": selected_update, **paired}, sort_keys=True))


if __name__ == "__main__":
    main()
