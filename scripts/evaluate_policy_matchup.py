"""Evaluate two infantry policy configs on a map split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from advancewars.training.policy_config import InfantryPolicyConfig
from advancewars.training.rollout import run_match


def _score(metrics: dict[str, float | int]) -> float:
    return (
        float(metrics["unit_value"])
        + 0.5 * float(metrics["funds"])
        + 15 * float(metrics["income"])
        + 1000 * float(metrics["property_count"])
        + 500 * float(metrics["unit_count"])
    )


def _summarize(games: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    raw = Counter(game["winner_label"] for game in games)
    raw_by_side: dict[str, Counter[str]] = defaultdict(Counter)
    adjudicated: Counter[str] = Counter()
    adjudicated_by_side: dict[str, Counter[str]] = defaultdict(Counter)
    metric_diffs: dict[str, list[float]] = defaultdict(list)

    for game in games:
        side = f"current_p{game['current_player_id']}"
        raw_by_side[side][game["winner_label"]] += 1

        current_metrics = game["final_metrics"]["A"]
        baseline_metrics = game["final_metrics"]["B"]
        for key in ("income", "property_count", "unit_count", "unit_value", "funds"):
            metric_diffs[key].append(
                float(current_metrics[key]) - float(baseline_metrics[key])
            )

        label = game["winner_label"]
        if label == "draw":
            current_score = _score(current_metrics)
            baseline_score = _score(baseline_metrics)
            label = (
                "draw"
                if abs(current_score - baseline_score) < 1e-9
                else ("A" if current_score > baseline_score else "B")
            )
            game["report_adjudicated_score"] = {
                "A": current_score,
                "B": baseline_score,
                "diff_A_minus_B": current_score - baseline_score,
            }
        game["report_adjudicated_winner_label"] = label
        adjudicated[label] += 1
        adjudicated_by_side[side][label] += 1

    return {
        "comparison": args.name,
        "split": str(args.split),
        "test_maps": len(json.loads(args.split.read_text())),
        "games": len(games),
        "max_steps": args.max_steps,
        "max_turns": args.max_turns,
        "raw_hard_results": dict(raw),
        "raw_hard_results_by_current_side": {
            key: dict(value) for key, value in sorted(raw_by_side.items())
        },
        "report_adjudicated_results": dict(adjudicated),
        "report_adjudicated_results_by_current_side": {
            key: dict(value) for key, value in sorted(adjudicated_by_side.items())
        },
        "average_metric_diff_current_minus_baseline": {
            key: sum(values) / len(values) if values else 0
            for key, values in sorted(metric_diffs.items())
        },
        "policy_files": {
            "current": str(args.current_policy),
            "baseline": str(args.baseline_policy),
        },
        "games_detail": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-map-dir", type=Path, required=True)
    parser.add_argument("--converted-map-dir", type=Path, required=True)
    parser.add_argument("--current-policy", type=Path, required=True)
    parser.add_argument("--baseline-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-jsonl", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-turns", type=int, default=80)
    args = parser.parse_args()

    maps = json.loads(args.split.read_text())
    args.converted_map_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)

    current = InfantryPolicyConfig.load(args.current_policy)
    baseline = InfantryPolicyConfig.load(args.baseline_policy)

    games: list[dict[str, Any]] = []
    with args.progress_jsonl.open("w") as progress:
        for index, entry in enumerate(maps):
            for current_player_id in (0, 1):
                game = run_match(
                    map_entry=entry,
                    raw_map_dir=args.raw_map_dir,
                    converted_map_dir=args.converted_map_dir,
                    policy_a=current,
                    policy_b=baseline,
                    seed=args.seed_base + index,
                    max_steps=args.max_steps,
                    max_turns=args.max_turns,
                    a_player_id=current_player_id,
                )
                game["current_player_id"] = current_player_id
                game["baseline_player_id"] = 1 - current_player_id
                games.append(game)
                progress.write(json.dumps(game, ensure_ascii=False) + "\n")
                progress.flush()
                print(
                    f"[{len(games):03d}/{len(maps) * 2}] "
                    f"map={entry['map_id']} current_p={current_player_id} "
                    f"winner={game['winner_label']} steps={game['steps']}",
                    flush=True,
                )

    summary = _summarize(games, args)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("--- summary ---", flush=True)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "games_detail"},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
