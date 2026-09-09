#!/usr/bin/env python3
"""Write deterministic, disjoint BFCL GRPO prompt manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from vita_rl.bfcl_environment import (
    BFCL_ENVIRONMENT_NAME,
    BFCL_SPLIT_SEED,
    bfcl_task_splits,
)


def prompt_record(task_id: str) -> dict:
    return {
        "prompt": "BFCL V4 multi-turn function-calling task",
        "label": "",
        "metadata": {
            "environment": BFCL_ENVIRONMENT_NAME,
            "task_id": task_id,
            "agent_model": "proxy-model",
            "max_steps": 20,
            "max_errors": 5,
            "environment_args": {
                "harness": "vita_rl_standard",
                "enable_think": False,
            },
            "reward_fn": "environment",
        },
    }


def write_manifest(path: Path, task_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task_id in task_ids:
            handle.write(json.dumps(prompt_record(task_id), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=BFCL_SPLIT_SEED)
    parser.add_argument("--bfcl-source-root", type=Path)
    args = parser.parse_args()
    train_ids, eval_ids = bfcl_task_splits(
        source_root=args.bfcl_source_root, seed=args.seed
    )
    write_manifest(args.train_output, train_ids)
    write_manifest(args.eval_output, eval_ids)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "train_tasks": len(train_ids),
                "eval_tasks": len(eval_ids),
                "train_output": str(args.train_output),
                "eval_output": str(args.eval_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
