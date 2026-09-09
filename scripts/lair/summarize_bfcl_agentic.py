#!/usr/bin/env python3
"""Write a compact, stable summary beside a native BFCL V4 agentic run."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    """Read BFCL scores (JSON) and native result files (JSONL)."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.run_root / "result" / args.model.replace("/", "_")
    score_dir = args.run_root / "score" / args.model
    categories = {}
    for category in args.categories:
        result_files = list(model_dir.rglob(f"BFCL_v4_{category}_result.json"))
        score_files = list(score_dir.rglob(f"BFCL_v4_{category}_score.json"))
        results = sum(len(load_json(path)) for path in result_files)
        score = load_json(score_files[0])[0] if len(score_files) == 1 else None
        categories[category] = {
            "result_files": [str(path) for path in result_files],
            "score_file": str(score_files[0]) if len(score_files) == 1 else None,
            "generated_entries": results,
            "accuracy": score.get("accuracy") if score else None,
            "correct_count": score.get("correct_count") if score else None,
            "total_count": score.get("total_count") if score else None,
        }

    scored = [value for value in categories.values() if value["total_count"]]
    total = sum(value["total_count"] for value in scored)
    correct = sum(value["correct_count"] for value in scored)
    raw_count = sum(1 for _ in args.trace.open(encoding="utf-8")) if args.trace.exists() else 0
    summary = {
        "benchmark": "BFCL V4 official harness",
        "harness": "native BFCL OpenAI function-calling agentic loop",
        "agent_model": "Qwen3.5-4B local SGLang",
        "model_registry_name": args.model,
        "run_root": str(args.run_root),
        "immediate_model_raw_trace": str(args.trace),
        "immediate_raw_generations": raw_count,
        "categories": categories,
        "aggregate": {
            "correct_count": correct,
            "total_count": total,
            "accuracy": correct / total if total else None,
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={args.output}")


if __name__ == "__main__":
    main()
