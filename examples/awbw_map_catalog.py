"""Print AWBW public map category counts."""

from __future__ import annotations

import argparse

from advancewars.eval import (
    AWBW_CATEGORIES_URL,
    fetch_categories,
    format_category_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=AWBW_CATEGORIES_URL)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()

    categories = fetch_categories(args.url, timeout=args.timeout)
    print(format_category_summary(categories))


if __name__ == "__main__":
    main()
