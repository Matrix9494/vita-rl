#!/usr/bin/env python3
"""Install Vita-RL's local Qwen3.5 BFCL adapter into an isolated checkout.

The upstream BFCL submodule is intentionally left unmodified in the primary
worktree.  This overlay is applied only to the disposable Vessl evaluation
worktree and is recorded there with the source and resulting file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent


HANDLER_SOURCE = dedent(
    r'''
    """BFCL adapter for a locally served Qwen3.5 function-calling model."""

    import json
    import os
    import threading
    from pathlib import Path
    from typing import Any

    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )


    _TRACE_LOCK = threading.Lock()


    class Qwen35LocalFCHandler(OpenAICompletionsHandler):
        """Use the existing SGLang OpenAI-compatible Qwen3.5 endpoint."""

        @staticmethod
        def _thinking_enabled() -> bool:
            return os.getenv("BFCL_QWEN_ENABLE_THINKING", "false").strip().lower() in {
                "1", "true", "yes", "on"
            }

        @staticmethod
        def _float_env(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        @staticmethod
        def _int_env(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def _build_client_kwargs(self) -> dict:
            base_url = os.getenv("BFCL_QWEN_BASE_URL")
            if not base_url:
                raise RuntimeError("BFCL_QWEN_BASE_URL must point at the local SGLang /v1 endpoint")
            return {
                "base_url": base_url,
                "api_key": os.getenv("BFCL_QWEN_API_KEY", "EMPTY"),
            }

        def _query_FC(self, inference_data: dict):
            messages: list[dict] = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(messages), "tools": tools}
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self._float_env("BFCL_QWEN_TOP_P", 0.8),
                "presence_penalty": self._float_env("BFCL_QWEN_PRESENCE_PENALTY", 1.5),
                "max_tokens": self._int_env("BFCL_QWEN_MAX_TOKENS", 8192),
                "store": False,
                "extra_body": {
                    "top_k": self._int_env("BFCL_QWEN_TOP_K", 20),
                    "min_p": self._float_env("BFCL_QWEN_MIN_P", 0.0),
                    "repetition_penalty": self._float_env("BFCL_QWEN_REPETITION_PENALTY", 1.0),
                    "chat_template_kwargs": {
                        "enable_thinking": self._thinking_enabled(),
                    },
                },
            }
            if tools:
                kwargs["tools"] = tools
            return self.generate_with_backoff(**kwargs)

        def _parse_query_response_FC(self, api_response: Any) -> dict:
            self._record_immediate_response(api_response)
            response_data = super()._parse_query_response_FC(api_response)
            reasoning = getattr(api_response.choices[0].message, "reasoning_content", None)
            if reasoning:
                response_data["reasoning_content"] = reasoning
                history = response_data["model_responses_message_for_chat_history"]
                history = history.model_dump(exclude_none=True) if hasattr(history, "model_dump") else dict(history)
                history["reasoning_content"] = reasoning
                response_data["model_responses_message_for_chat_history"] = history
            return response_data

        @staticmethod
        def _record_immediate_response(api_response: Any) -> None:
            trace_path = os.getenv("BFCL_IMMEDIATE_TRACE_PATH")
            if not trace_path:
                return
            payload = api_response.model_dump(mode="json") if hasattr(api_response, "model_dump") else api_response
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _TRACE_LOCK, path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\\n")
    '''
).lstrip()

IMPORT = "from bfcl_eval.model_handler.api_inference.qwen35_local import Qwen35LocalFCHandler\n"
REGISTRATION = dedent(
    '''
        "qwen35-4b-local-FC": ModelConfig(
            model_name="qwen35-4b-local",
            display_name="Qwen3.5-4B (Local FC)",
            url="https://huggingface.co/Qwen/Qwen3.5-4B",
            org="Qwen",
            license="Apache-2.0",
            model_handler=Qwen35LocalFCHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=False,
        ),
    '''
).lstrip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bfcl_root.resolve()
    if root.name != "berkeley-function-call-leaderboard" or not (root / "bfcl_eval").is_dir():
        raise SystemExit(f"Not a BFCL checkout: {root}")

    handler = root / "bfcl_eval/model_handler/api_inference/qwen35_local.py"
    model_config = root / "bfcl_eval/constants/model_config.py"
    original_hash = sha256(model_config)
    handler.parent.mkdir(parents=True, exist_ok=True)
    handler.write_text(HANDLER_SOURCE, encoding="utf-8")

    config = model_config.read_text(encoding="utf-8")
    if IMPORT not in config:
        anchor = "from bfcl_eval.model_handler.api_inference.qwen import ("
        if anchor not in config:
            raise SystemExit("BFCL model_config import anchor was not found")
        config = config.replace(anchor, IMPORT + anchor, 1)
    if '"qwen35-4b-local-FC": ModelConfig(' not in config:
        anchor = "api_inference_model_map = {\n"
        if anchor not in config:
            raise SystemExit("BFCL model_config registry anchor was not found")
        config = config.replace(anchor, anchor + REGISTRATION, 1)
    model_config.write_text(config, encoding="utf-8")

    manifest = {
        "overlay": "vita-rl-bfcl-qwen35-local-v1",
        "bfcl_root": str(root),
        "model_config_original_sha256": original_hash,
        "model_config_overlay_sha256": sha256(model_config),
        "handler_sha256": sha256(handler),
    }
    (root / ".vita-rl-qwen35-overlay.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
