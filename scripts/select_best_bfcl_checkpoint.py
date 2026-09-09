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
    # Eval IDs are zero-based rollout IDs.  ID 0 is the reference evaluation;
    # a trained checkpoint is eligible only when a native checkpoint exists at
    # the same ID.  Never assign an unmatched evaluation to the reference (or
    # a later checkpoint): that would mislabel the checkpoint's held-out score.
    observed_best = max(
        records,
        key=lambda record: (
            float(record["mean_terminal_reward"]),
            int(record["rollout_id"]),
        ),
    )
    candidates = []
    for record in records:
        step = int(record["rollout_id"])
        checkpoint = args.reference_checkpoint if step == 0 else args.checkpoint_root / f"iter_{step:07d}"
        if checkpoint.is_dir():
            candidates.append((record, checkpoint))
    if not candidates:
        raise SystemExit("No checkpoint-aligned BFCL held-out evaluation records were written.")
    best, checkpoint = max(
        candidates,
        key=lambda item: (
            float(item[0]["mean_terminal_reward"]),
            int(item[0]["rollout_id"]),
        ),
    )
    step = int(best["rollout_id"])
    payload = {
        "best_rollout_id": step,
        "best_heldout_reward": float(best["mean_terminal_reward"]),
        "checkpoint": str(checkpoint),
        "selection_rule": "maximum checkpoint-aligned held-out mean terminal reward; latest ties win",
        "best_observed_heldout_reward": float(observed_best["mean_terminal_reward"]),
        "best_observed_rollout_id": int(observed_best["rollout_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
