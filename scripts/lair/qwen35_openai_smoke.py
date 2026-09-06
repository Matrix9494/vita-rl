#!/usr/bin/env python3
"""Fail-closed OpenAI-compatible tool and reasoning checks for Qwen3.5."""

import argparse
import json
from pathlib import Path

import requests


def request(base_url: str, payload: dict) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        json=payload,
        timeout=(10, 180),
    )
    response.raise_for_status()
    return response.json()


def message(response: dict) -> dict:
    try:
        return response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed OpenAI response: {response}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI /v1 base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tool_response = request(
        args.base_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call the echo function exactly once with text set to ping. "
                        "Do not answer in prose."
                    ),
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Return the supplied text.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    tool_message = message(tool_response)
    tool_calls = tool_message.get("tool_calls") or []
    tool_content = tool_message.get("content") or ""
    if not tool_calls:
        raise RuntimeError(
            "Tool smoke failed: response did not contain structured message.tool_calls; "
            f"content={tool_content!r}"
        )
    if "<tool_call" in tool_content.lower() or "<function=" in tool_content.lower():
        raise RuntimeError(
            "Tool smoke failed: tool XML leaked into message.content instead of "
            "being returned through message.tool_calls"
        )

    # Replay the exact OpenAI assistant-tool-call + tool-result history that
    # VitaBench sends on its next agent turn.  This catches incompatibilities
    # that are invisible in a single isolated function-call response.
    continuation_response = request(
        args.base_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call the echo function exactly once with text set to ping. "
                        "Do not answer in prose."
                    ),
                },
                {
                    "role": "assistant",
                    "content": tool_message.get("content"),
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"],
                    "name": "echo",
                    "content": "ping",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Return the supplied text.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    continuation_message = message(continuation_response)
    continuation_content = continuation_message.get("content") or ""
    if not continuation_content:
        raise RuntimeError(
            "Tool continuation smoke failed: no final assistant content after tool result"
        )

    reasoning_response = request(
        args.base_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Briefly compute 7 times 8, then give only the answer.",
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    reasoning_message = message(reasoning_response)
    reasoning_content = reasoning_message.get("reasoning_content")
    final_content = reasoning_message.get("content") or ""
    if not reasoning_content:
        raise RuntimeError(
            "Reasoning smoke failed: response did not expose message.reasoning_content"
        )
    if not final_content:
        raise RuntimeError("Reasoning smoke failed: response did not contain final content")
    if "<think" in final_content.lower() or "</think" in final_content.lower():
        raise RuntimeError(
            "Reasoning smoke failed: raw thinking tags leaked into message.content"
        )

    result = {
        "tool": {
            "tool_choice": "auto",
            "tool_calls": tool_calls,
            "content": tool_content,
            "continuation_content": continuation_content,
        },
        "reasoning": {
            "reasoning_content": reasoning_content,
            "content": final_content,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"qwen35_openai_smoke={output}")
    print(f"tool_calls={len(tool_calls)}")
    print("reasoning_separated=true")


if __name__ == "__main__":
    main()
