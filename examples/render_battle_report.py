"""Render a saved rollout JSON as one concatenated battle-report image."""

from __future__ import annotations

import argparse

from advancewars.render.battle_report import render_battle_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory",
        nargs="?",
        default="runs/heuristic_duel_rollout.json",
    )
    parser.add_argument("--output", default="runs/heuristic_duel_report.png")
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=48)
    parser.add_argument("--include-initial", action="store_true")
    args = parser.parse_args()

    output = render_battle_report(
        args.trajectory,
        args.output,
        columns=args.columns,
        tile_size=args.tile_size,
        include_initial=args.include_initial,
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
