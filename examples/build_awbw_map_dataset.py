"""Build a small local AWBW map dataset split."""

from __future__ import annotations

import argparse

from advancewars.eval import build_default_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/awbw_maps")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    args = parser.parse_args()

    manifest = build_default_dataset(
        args.output_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        timeout=args.timeout,
        delay_seconds=args.delay_seconds,
    )
    print(f"saved={args.output_dir}")
    print(f"train={len(manifest['train'])}")
    print(f"test={len(manifest['test'])}")


if __name__ == "__main__":
    main()
