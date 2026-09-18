#!/usr/bin/env python3
"""Validate and summarize a Vessl BFCL V4 native-harness evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


GROUPS = {
    "non_live": {"simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"},
    "live": {"live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"},
    "multi_turn": {"multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"},
    "memory": {"memory_kv", "memory_vector", "memory_rec_sum"},
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_or_jsonl(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def trace_json_stream(path: Path) -> tuple[list[dict], int]:
    """Read normal JSONL and older adjacent-object traces without losing rows."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    parse_errors = 0
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        try:
            row, position = decoder.raw_decode(text, position)
            if not isinstance(row, dict):
                parse_errors += 1
            else:
                rows.append(row)
        except json.JSONDecodeError:
            parse_errors += 1
            next_record = text.find('{"id"', position + 1)
            if next_record < 0:
                break
            position = next_record
    return rows, parse_errors


def contains_inference_error(value: object) -> bool:
    if isinstance(value, str):
        return "Error during inference:" in value
    if isinstance(value, dict):
        return any(contains_inference_error(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_inference_error(item) for item in value)
    return False


def group_for(category: str) -> str:
    for group, categories in GROUPS.items():
        if category in categories:
            return group
    raise ValueError(f"Unknown category: {category}")


def model_summary(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.bfcl_root))
    from bfcl_eval.utils import (  # pylint: disable=import-outside-toplevel
        get_directory_structure_by_category,
        get_file_name_by_category,
        load_dataset_entry,
    )

    categories: dict[str, dict] = {}
    model_dir = args.run_root / "result" / args.model.replace("/", "_")
    score_dir = args.run_root / "score" / args.model
    total_expected = total_correct = total_scored = total_errors = 0
    for category in args.categories:
        expected_ids = {entry["id"] for entry in load_dataset_entry(category, include_prereq=False)}
        directory = get_directory_structure_by_category(category)
        result_path = model_dir / directory / get_file_name_by_category(category, is_result_file=True)
        score_path = score_dir / directory / get_file_name_by_category(category, is_score_file=True)
        if not result_path.is_file() or not score_path.is_file():
            raise SystemExit(f"Missing official BFCL result or score for {category}")
        results = jsonl(result_path)
        actual_ids = [entry.get("id") for entry in results]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
            raise SystemExit(f"Result IDs for {category} do not exactly match its official dataset")
        error_count = sum(contains_inference_error(entry.get("result")) for entry in results)
        if error_count:
            raise SystemExit(f"{category} has {error_count} unresolved inference-error placeholders")
        score_rows = json_or_jsonl(score_path)
        if not isinstance(score_rows, list) or not score_rows:
            raise SystemExit(f"Unexpected score format for {category}")
        score = score_rows[0]
        if score.get("total_count") != len(expected_ids):
            raise SystemExit(f"Score total for {category} does not match its official dataset")
        categories[category] = {
            "group": group_for(category),
            "expected_count": len(expected_ids),
            "generated_count": len(results),
            "correct_count": score.get("correct_count"),
            "total_count": score.get("total_count"),
            "accuracy": score.get("accuracy"),
            "inference_error_count": error_count,
            "result_file": str(result_path),
            "score_file": str(score_path),
        }
        total_expected += len(expected_ids)
        total_correct += score["correct_count"]
        total_scored += score["total_count"]
        total_errors += error_count
    if total_expected != args.expected_total or total_scored != args.expected_total:
        raise SystemExit(f"Expected {args.expected_total} scored records, got expected={total_expected} scored={total_scored}")

    trace_rows, trace_parse_errors = trace_json_stream(args.trace) if args.trace.is_file() else ([], 0)
    group_totals: dict[str, dict[str, int | float | None]] = defaultdict(lambda: {"correct_count": 0, "total_count": 0})
    for entry in categories.values():
        aggregate = group_totals[entry["group"]]
        aggregate["correct_count"] += entry["correct_count"]
        aggregate["total_count"] += entry["total_count"]
    for aggregate in group_totals.values():
        aggregate["accuracy"] = aggregate["correct_count"] / aggregate["total_count"]
    prompt_tokens = sum(((row.get("usage") or {}).get("prompt_tokens") or 0) for row in trace_rows)
    output_tokens = sum(((row.get("usage") or {}).get("completion_tokens") or 0) for row in trace_rows)
    payload = {
        "benchmark": "BFCL V4 official harness",
        "harness": "native BFCL OpenAI function-calling agentic loop",
        "model_label": args.model_label,
        "model_registry_name": args.model,
        "run_root": str(args.run_root),
        "categories": categories,
        "groups": dict(group_totals),
        "aggregate": {
            "correct_count": total_correct,
            "total_count": total_scored,
            "accuracy": total_correct / total_scored,
            "inference_error_count": total_errors,
            "raw_generation_count": len(trace_rows),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "trace_parse_errors": trace_parse_errors,
        },
        "trace_file": str(args.trace),
        "trace": {"valid": trace_parse_errors == 0, "parse_errors": trace_parse_errors},
        "validation": {"accepted": True, "expected_total": args.expected_total},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"summary={args.output}")


def comparison_summary(args: argparse.Namespace) -> None:
    base = json.loads(args.base.read_text(encoding="utf-8"))
    grpo = json.loads(args.grpo.read_text(encoding="utf-8"))
    if not base.get("validation", {}).get("accepted") or not grpo.get("validation", {}).get("accepted"):
        raise SystemExit("Both per-model summaries must be accepted before comparison")
    if base["aggregate"]["total_count"] != grpo["aggregate"]["total_count"]:
        raise SystemExit("Base and GRPO scored different totals")
    per_category = {}
    for category, base_record in base["categories"].items():
        grpo_record = grpo["categories"].get(category)
        if not grpo_record:
            raise SystemExit(f"GRPO summary missing {category}")
        per_category[category] = {
            "base_accuracy": base_record["accuracy"],
            "grpo_accuracy": grpo_record["accuracy"],
            "accuracy_delta": grpo_record["accuracy"] - base_record["accuracy"],
        }
    payload = {
        "benchmark": "BFCL V4 official harness",
        "scope": "all scoring categories except web_search and format_sensitivity",
        "models": {"base": str(args.base), "grpo100_best_heldout": str(args.grpo)},
        "aggregate": {
            "total_count": base["aggregate"]["total_count"],
            "base_accuracy": base["aggregate"]["accuracy"],
            "grpo_accuracy": grpo["aggregate"]["accuracy"],
            "accuracy_delta": grpo["aggregate"]["accuracy"] - base["aggregate"]["accuracy"],
        },
        "per_category": per_category,
        "validation": {"accepted": True},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"comparison={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--bfcl-root", type=Path, required=True)
    model.add_argument("--run-root", type=Path, required=True)
    model.add_argument("--model", required=True)
    model.add_argument("--model-label", required=True)
    model.add_argument("--trace", type=Path, required=True)
    model.add_argument("--expected-total", type=int, default=4906)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("categories", nargs="+")
    comparison = subparsers.add_parser("comparison")
    comparison.add_argument("--base", type=Path, required=True)
    comparison.add_argument("--grpo", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "model":
        model_summary(args)
    else:
        comparison_summary(args)


if __name__ == "__main__":
    main()
