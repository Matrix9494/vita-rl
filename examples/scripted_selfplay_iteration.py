"""Run one alternating scripted-policy self-play iteration."""

from __future__ import annotations

import argparse

from advancewars.training import run_iteration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--dataset-dir", default="datasets/awbw_maps")
    parser.add_argument("--output-root", default="runs/scripted_selfplay")
    parser.add_argument("--game-count", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_iteration(
        iteration=args.iteration,
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        game_count=args.game_count,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    print(f"learner={summary['metadata']['learner']}")
    print(f"wins={summary['diagnostics']['wins']}")
    print(f"codex_request={summary['metadata']['codex_request']}")


if __name__ == "__main__":
    main()
