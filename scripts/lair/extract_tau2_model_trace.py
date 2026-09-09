#!/usr/bin/env python3
"""Extract every raw agent/user generation from a tau2 result JSON.

The tau2 message record contains a normalized ``content`` field and the full
LiteLLM response object under ``raw_data``.  This extractor deliberately keeps
both, rather than relying on a provider-specific schema.  When SGLang exposes
Qwen reasoning, the normalized ``reasoning_content`` field is recovered while
the unmodified response remains available as ``raw_response`` for diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def _first_value(value: Any, keys: tuple[str, ...]) -> Any | None:
    """Return the first nonempty value below one of ``keys`` in JSON data."""
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if candidate not in (None, "", [], {}):
                return candidate
        for candidate in value.values():
            found = _first_value(candidate, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _first_value(candidate, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def _raw_message(raw_response: dict[str, Any]) -> dict[str, Any]:
    """Handle both LiteLLM's flattened and OpenAI's choices-based responses."""
    message = raw_response.get("message")
    if isinstance(message, dict):
        return message
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for simulation_index, simulation in enumerate(result.get("simulations", [])):
        for message_index, message in enumerate(simulation.get("messages", [])):
            role = message.get("role")
            if role not in {"assistant", "user"}:
                continue
            raw_response = message.get("raw_data") or {}
            raw_message = _raw_message(raw_response)
            content = raw_message.get("content", message.get("content"))
            records.append(
                {
                    "simulation_index": simulation_index,
                    "task_id": simulation.get("task_id"),
                    "simulation_id": simulation.get("id"),
                    "message_index": message_index,
                    "model_role": "agent" if role == "assistant" else "user_simulator",
                    "role": role,
                    "timestamp": message.get("timestamp"),
                    "content": content,
                    "reasoning_content": _first_value(raw_response, REASONING_KEYS),
                    "tool_calls": raw_message.get("tool_calls", message.get("tool_calls")),
                    "usage": message.get("usage"),
                    "raw_response": raw_response,
                }
            )

    payload = {
        "source_result": str(args.result),
        "format": "one record per immediate tau2 model generation",
        "records": records,
        "num_records": len(records),
        "num_reasoning_records": sum(record["reasoning_content"] is not None for record in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"trace={args.output}")
    print(f"model_generations={payload['num_records']}")
    print(f"reasoning_generations={payload['num_reasoning_records']}")


if __name__ == "__main__":
    main()
