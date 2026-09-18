#!/usr/bin/env python3
"""Install the OpenRouter GPT-5.6 Terra BFCL adapter in an isolated checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent


HANDLER_SOURCE = dedent(
    r'''
    """BFCL OpenAI-completions adapter for OpenRouter's GPT-5.6 Terra."""

    import fcntl
    import json
    import os
    import threading
    from pathlib import Path
    from typing import Any

    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler


    _TRACE_LOCK = threading.Lock()


    class OpenRouterTerraFCHandler(OpenAICompletionsHandler):
        """Use the runner-owned localhost proxy without persisting its raw credential."""

        def _build_client_kwargs(self) -> dict:
            base_url = os.getenv("BFCL_TERRA_BASE_URL")
            if not base_url:
                raise RuntimeError("BFCL_TERRA_BASE_URL must point at the localhost OpenRouter proxy")
            return {"base_url": base_url, "api_key": os.getenv("BFCL_TERRA_API_KEY", "vita-rl-local-proxy")}

        def _query_FC(self, inference_data: dict):
            messages: list[dict] = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(messages), "tools": tools}
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": self.model_name,
                "reasoning_effort": os.getenv("BFCL_TERRA_REASONING_EFFORT", "high"),
                "max_tokens": int(os.getenv("BFCL_TERRA_MAX_TOKENS", "8192")),
                "store": False,
            }
            if tools:
                kwargs["tools"] = tools
            return self.generate_with_backoff(**kwargs)

        def _parse_query_response_FC(self, api_response: Any) -> dict:
            self._record_immediate_response(api_response)
            return super()._parse_query_response_FC(api_response)

        @staticmethod
        def _record_immediate_response(api_response: Any) -> None:
            trace_path = os.getenv("BFCL_IMMEDIATE_TRACE_PATH")
            if not trace_path:
                return
            payload = api_response.model_dump(mode="json") if hasattr(api_response, "model_dump") else api_response
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _TRACE_LOCK, path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\\n")
                    handle.flush()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    '''
).lstrip()

IMPORT = "from bfcl_eval.model_handler.api_inference.openrouter_terra import OpenRouterTerraFCHandler\n"
REGISTRATION = dedent(
    '''
        "openrouter-gpt-5.6-terra-FC": ModelConfig(
            model_name="openai/gpt-5.6-terra",
            display_name="GPT-5.6 Terra via OpenRouter (FC)",
            url="https://openrouter.ai/",
            org="OpenAI via OpenRouter",
            license="Proprietary",
            model_handler=OpenRouterTerraFCHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
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

    handler = root / "bfcl_eval/model_handler/api_inference/openrouter_terra.py"
    model_config = root / "bfcl_eval/constants/model_config.py"
    original_hash = sha256(model_config)
    handler.write_text(HANDLER_SOURCE, encoding="utf-8")
    config = model_config.read_text(encoding="utf-8")
    if IMPORT not in config:
        anchor = "from bfcl_eval.model_handler.api_inference.qwen import ("
        if anchor not in config:
            raise SystemExit("BFCL model_config import anchor was not found")
        config = config.replace(anchor, IMPORT + anchor, 1)
    if '"openrouter-gpt-5.6-terra-FC": ModelConfig(' not in config:
        anchor = "api_inference_model_map = {\n"
        if anchor not in config:
            raise SystemExit("BFCL model_config registry anchor was not found")
        config = config.replace(anchor, anchor + REGISTRATION, 1)
    model_config.write_text(config, encoding="utf-8")
    manifest = {
        "overlay": "vita-rl-bfcl-openrouter-terra-v1",
        "bfcl_root": str(root),
        "model_config_original_sha256": original_hash,
        "model_config_overlay_sha256": sha256(model_config),
        "handler_sha256": sha256(handler),
    }
    (root / ".vita-rl-terra-overlay.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
