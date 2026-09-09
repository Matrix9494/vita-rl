#!/usr/bin/env python3
"""Select the native checkpoint with the strongest BFCL held-out reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("No BFCL held-out evaluation records were written.")
    best = max(
        records,
        key=lambda record: (
            float(record["mean_terminal_reward"]),
            int(record["rollout_id"]),
        ),
    )
    step = int(best["rollout_id"])
    candidate = args.checkpoint_root / f"iter_{step:07d}"
    checkpoint = candidate if candidate.is_dir() else args.reference_checkpoint
    payload = {
        "best_rollout_id": step,
        "best_heldout_reward": float(best["mean_terminal_reward"]),
        "checkpoint": str(checkpoint),
        "selection_rule": "maximum held-out mean terminal reward; latest ties win",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
