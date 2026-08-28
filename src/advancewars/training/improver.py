"""Generate Codex improvement requests from self-play diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_codex_improvement_request(
    *,
    iteration_dir: str | Path,
    iteration: int,
    learner_label: str,
    diagnostics: dict[str, Any],
    games: list[dict[str, Any]],
    max_examples: int = 12,
) -> Path:
    """Write the markdown prompt consumed by the Codex CLI improver step."""

    output = Path(iteration_dir) / "codex_improvement_request.md"
    output.write_text(
        build_codex_improvement_request(
            iteration_dir=Path(iteration_dir),
            iteration=iteration,
            learner_label=learner_label,
            diagnostics=diagnostics,
            games=games,
            max_examples=max_examples,
        )
    )
    return output


def build_codex_improvement_request(
    *,
    iteration_dir: Path,
    iteration: int,
    learner_label: str,
    diagnostics: dict[str, Any],
    games: list[dict[str, Any]],
    max_examples: int = 12,
) -> str:
    opponent = "B" if learner_label == "A" else "A"
    learner_policy = iteration_dir / f"next_policy_{learner_label}.yaml"
    opponent_policy = iteration_dir / f"next_policy_{opponent}.yaml"
    summary_path = iteration_dir / "summary.json"
    diagnostics_path = iteration_dir / "diagnostics.json"

    learner_stats = _learner_stats(games, learner_label)
    examples = _rank_examples(games, learner_label)[:max_examples]

    lines = [
        "# Codex Self-Play Policy Improvement Request",
        "",
        f"Iteration: `{iteration}`",
        f"Learner this iteration: `{learner_label}`",
        f"Opponent: `{opponent}`",
        "",
        "## Goal",
        "",
        (
            "Improve the learner's deterministic scripted Advance Wars policy using "
            "the training games from this iteration. This is not a neural-network "
            "training step. You are Codex acting as the policy improver."
        ),
        "",
        "## Important Files",
        "",
        f"- Learner next policy config: `{learner_policy}`",
        f"- Opponent next policy config: `{opponent_policy}`",
        "- Shared policy implementation: `src/advancewars/policies/infantry_decision_tree.py`",
        f"- Iteration summary: `{summary_path}`",
        f"- Diagnostics JSON: `{diagnostics_path}`",
        f"- Per-game records: `{iteration_dir / 'games'}`",
        "",
        "## Rules For This Improvement Step",
        "",
        "- Do not run the 20-map test split in this iteration step.",
        "- Prefer small, explainable improvements to the learner policy.",
        "- Keep the policy deterministic.",
        "- Do not change engine rules or map conversion unless a clear bug is found.",
        "- Preserve the PettingZoo structured action interface.",
        "- If changing config, edit only the learner's `next_policy_*.yaml`.",
        (
            "- If changing shared Python policy code, remember it affects both A and B; "
            "prefer logic that is controlled by config values when possible."
        ),
        "- Run focused tests after edits, at least `python3 -m pytest tests/test_training_loop.py tests/test_infantry_policy.py`.",
        "",
        "## Aggregate Diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2, sort_keys=True)[:12000],
        "```",
        "",
        "## Learner Summary",
        "",
        _format_kv(learner_stats),
        "",
        "## Highest-Priority Game Examples",
        "",
    ]

    if examples:
        for index, example in enumerate(examples, start=1):
            lines.extend(
                [
                    f"### Example {index}: {example['map_name']} (`{example['map_id']}`)",
                    "",
                    _format_kv(example),
                    "",
                ]
            )
    else:
        lines.append("No decisive loss examples were found; inspect draw and timeout games.")
        lines.append("")

    lines.extend(
        [
            "## Suggested Work Pattern",
            "",
            "1. Inspect the aggregate diagnostics and the example game JSON files.",
            "2. Identify one or two concrete failure modes.",
            "3. Modify the learner config and/or policy code to address those modes.",
            "4. Run the focused tests listed above.",
            "5. Leave a concise note in `codex_improvement_notes.md` inside this iteration directory describing what changed and why.",
            "",
        ]
    )
    return "\n".join(lines)


def _learner_stats(games: list[dict[str, Any]], learner_label: str) -> dict[str, Any]:
    total = max(1, len(games))
    wins = sum(1 for game in games if game["winner_label"] == learner_label)
    losses = sum(
        1 for game in games if game["winner_label"] not in {learner_label, "draw"}
    )
    draws = sum(1 for game in games if game["winner_label"] == "draw")
    timeout_like = sum(1 for game in games if not game.get("done", False))
    metric_diffs = [_metric_diff(game, learner_label) for game in games]
    return {
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / total, 4),
        "timeout_or_step_limited_games": timeout_like,
        "avg_income_diff": round(_avg(diff["income_diff"] for diff in metric_diffs), 2),
        "avg_property_diff": round(
            _avg(diff["property_diff"] for diff in metric_diffs), 2
        ),
        "avg_unit_value_diff": round(
            _avg(diff["unit_value_diff"] for diff in metric_diffs), 2
        ),
    }


def _rank_examples(
    games: list[dict[str, Any]], learner_label: str
) -> list[dict[str, Any]]:
    examples = []
    for index, game in enumerate(games):
        diffs = _metric_diff(game, learner_label)
        score = (
            diffs["unit_value_diff"] / 1000
            + diffs["income_diff"] / 1000
            + diffs["property_diff"] * 2
            - (50 if game["winner_label"] != learner_label else 0)
        )
        examples.append(
            {
                "game_file": str(
                    Path("games") / f"{index:03d}_{game['map']['map_id']}.json"
                ),
                "map_id": game["map"]["map_id"],
                "map_name": game["map"]["name"],
                "winner": game["winner_label"],
                "steps": game["steps"],
                "turn": game["turn"],
                "a_player_id": game["a_player_id"],
                **diffs,
                "priority_score": round(score, 2),
            }
        )
    return sorted(examples, key=lambda item: item["priority_score"])


def _metric_diff(game: dict[str, Any], learner_label: str) -> dict[str, float]:
    opponent = "B" if learner_label == "A" else "A"
    mine = game["final_metrics"][learner_label]
    other = game["final_metrics"][opponent]
    return {
        "income_diff": float(mine["income"] - other["income"]),
        "property_diff": float(mine["property_count"] - other["property_count"]),
        "unit_count_diff": float(mine["unit_count"] - other["unit_count"]),
        "unit_value_diff": float(mine["unit_value"] - other["unit_value"]),
    }


def _avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _format_kv(payload: dict[str, Any]) -> str:
    return "\n".join(f"- `{key}`: {value}" for key, value in payload.items())
