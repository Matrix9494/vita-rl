#!/usr/bin/env bash
# Official BFCL V4 multi-turn evaluation for GPT-5.6 Terra via OpenRouter.
# This runs only in an isolated Vessl worktree. The raw OpenRouter credential
# is held solely by the localhost proxy process and is never persisted.
set -euo pipefail

source /root/setup_env.sh

REPO="${BFCL_EVAL_REPO:?BFCL_EVAL_REPO is required}"
RUN_ID="${BFCL_EVAL_RUN_ID:-vessl-bfcl-v4-terra-multiturn-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${BFCL_EVAL_OUTPUT_ROOT:-/root/outputs/vita-rl/bfcl/${RUN_ID}}"
PYTHON="${BFCL_PYTHON:-/root/venvs/dressage-cu129-py312/bin/python}"
MODEL_ID="openrouter-gpt-5.6-terra-FC"
MODEL_NAME="openai/gpt-5.6-terra"
REASONING_EFFORT="${BFCL_TERRA_REASONING_EFFORT:-high}"
MAX_TOKENS="${BFCL_TERRA_MAX_TOKENS:-8192}"
CONCURRENCY="${BFCL_CONCURRENCY:-20}"
BFCL_ROOT="$REPO/external/gorilla/berkeley-function-call-leaderboard"
OVERLAY="$REPO/scripts/vessl/bfcl_terra_overlay.py"
SUMMARIZER="$REPO/scripts/vessl/summarize_bfcl_v4.py"
EXPECTED_TOTAL=800
CATEGORIES=(multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context)

[[ -e "$REPO/.git" ]] || { echo "Missing isolated repository: $REPO" >&2; exit 2; }
[[ -d "$BFCL_ROOT" ]] || { echo "Missing initialized BFCL submodule: $BFCL_ROOT" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Missing BFCL Python: $PYTHON" >&2; exit 2; }
[[ -f "$OVERLAY" && -f "$SUMMARIZER" ]] || { echo "Missing Terra BFCL support scripts" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required in-memory" >&2; exit 2; }
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "BFCL_CONCURRENCY must be positive" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BFCL_PROJECT_ROOT="$RUN_ROOT"
export BFCL_IMMEDIATE_TRACE_PATH="$RUN_ROOT/immediate_model_trace.jsonl"
export BFCL_TERRA_REASONING_EFFORT="$REASONING_EFFORT"
export BFCL_TERRA_MAX_TOKENS="$MAX_TOKENS"

OPENROUTER_PID=""
OPENROUTER_PORT_FILE=""
cleanup() {
    local code=$?
    [[ -n "$OPENROUTER_PID" ]] && kill "$OPENROUTER_PID" 2>/dev/null || true
    [[ -z "$OPENROUTER_PORT_FILE" ]] || rm -f "$OPENROUTER_PORT_FILE"
    exit "$code"
}
trap cleanup EXIT INT TERM

OPENROUTER_PORT_FILE="$(mktemp /tmp/vita-rl-openrouter.XXXXXX)"
"$PYTHON" "$REPO/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" \
    >"$RUN_ROOT/logs/openrouter-proxy.log" 2>&1 &
OPENROUTER_PID=$!
for _ in $(seq 1 100); do
    [[ -s "$OPENROUTER_PORT_FILE" ]] && break
    kill -0 "$OPENROUTER_PID" 2>/dev/null || { echo "OpenRouter proxy exited early" >&2; exit 1; }
    sleep 0.1
done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter proxy startup timed out" >&2; exit 1; }
export BFCL_TERRA_BASE_URL="http://127.0.0.1:$(<"$OPENROUTER_PORT_FILE")/v1"
export BFCL_TERRA_API_KEY="vita-rl-local-proxy"
unset OPENROUTER_API_KEY

"$PYTHON" "$OVERLAY" --bfcl-root "$BFCL_ROOT" >"$RUN_ROOT/overlay-manifest.json"
PYTHONPATH="$BFCL_ROOT" "$PYTHON" - <<'PY'
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

categories = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
assert MODEL_CONFIG_MAPPING["openrouter-gpt-5.6-terra-FC"].model_name == "openai/gpt-5.6-terra"
assert parse_test_category_argument(categories) == sorted(categories)
assert sum(len(load_dataset_entry(category, include_prereq=False)) for category in categories) == 800
print("bfcl_terra_adapter_and_multiturn_manifest=ok")
PY

"$PYTHON" - "$RUN_ROOT/terra-tool-smoke.json" <<'PY'
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

payload = {
    "model": "openai/gpt-5.6-terra",
    "messages": [{"role": "user", "content": "Call echo exactly once with text ping."}],
    "tools": [{"type": "function", "function": {"name": "echo", "description": "Echo text.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}],
    "reasoning_effort": os.environ["BFCL_TERRA_REASONING_EFFORT"], "max_tokens": 256,
}
request = Request(os.environ["BFCL_TERRA_BASE_URL"] + "/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Authorization": "Bearer vita-rl-local-proxy", "Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=600) as response:
    reply = json.loads(response.read())
tool_calls = ((reply.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
if reply.get("model") != payload["model"] or not tool_calls:
    raise SystemExit("Terra OpenAI tool smoke did not return the requested model and tool call")
Path(sys.argv[1]).write_text(json.dumps({"model": reply.get("model"), "tool_call_count": len(tool_calls),
    "finish_reason": ((reply.get("choices") or [{}])[0]).get("finish_reason"), "usage": reply.get("usage") or {},
    "reasoning_effort": payload["reasoning_effort"]}, indent=2) + "\n")
PY

"$PYTHON" - "$RUN_ROOT/manifest.json" "$REPO" "$BFCL_ROOT" "$RUN_ID" "$MODEL_NAME" "$REASONING_EFFORT" "$MAX_TOKENS" "$CONCURRENCY" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

output, repo, bfcl_root, run_id, model, effort, max_tokens, concurrency = sys.argv[1:]
revision = lambda path: subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
Path(output).write_text(json.dumps({"run_id": run_id, "model": model,
    "scope": "BFCL V4 all official multi-turn scoring categories", "categories": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
    "expected_total": 800, "inference": {"reasoning_effort": effort, "max_tokens": int(max_tokens), "temperature": "provider_default_not_sent", "concurrency": int(concurrency)},
    "provenance": {"vita_rl_commit": revision(repo), "bfcl_submodule_commit": revision(bfcl_root)}}, indent=2) + "\n")
PY

cd "$BFCL_ROOT"
for category in "${CATEGORIES[@]}"; do
    if grep -qxF "$category" "$RUN_ROOT/completed-categories.txt" 2>/dev/null; then
        echo "[terra] category=$category already completed; retaining official artifacts"
        continue
    fi
    echo "[terra] category=$category"
    PYTHONPATH="$BFCL_ROOT" "$PYTHON" -m bfcl_eval generate --model "$MODEL_ID" --test-category "$category" \
        --temperature 0.0 --num-threads "$CONCURRENCY" --include-input-log
    PYTHONPATH="$BFCL_ROOT" "$PYTHON" -m bfcl_eval evaluate --model "$MODEL_ID" --test-category "$category"
    printf '%s\n' "$category" >> "$RUN_ROOT/completed-categories.txt"
done

"$PYTHON" "$SUMMARIZER" model --bfcl-root "$BFCL_ROOT" --run-root "$RUN_ROOT" --model "$MODEL_ID" \
    --model-label "gpt-5.6-terra" --trace "$BFCL_IMMEDIATE_TRACE_PATH" --expected-total "$EXPECTED_TOTAL" \
    --output "$RUN_ROOT/summary.json" "${CATEGORIES[@]}"
"$PYTHON" - "$RUN_ROOT/summary.json" "$RUN_ROOT/manifest.json" "$RUN_ROOT/terra-tool-smoke.json" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

summary_path, manifest_path, smoke_path, run_id = map(Path, sys.argv[1:4])
payload = json.loads(summary_path.read_text())
payload.update({"run_id": sys.argv[4], "manifest_file": str(manifest_path), "terra_tool_smoke_file": str(smoke_path),
                "raw_completion_trace_file": str(summary_path.parent / "immediate_model_trace.jsonl")})
payload["validation"]["all_multiturn_categories"] = True
summary_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
echo "BFCL Terra multi-turn evaluation accepted: $RUN_ROOT/summary.json"
