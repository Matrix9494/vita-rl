#!/usr/bin/env bash
# Evaluate base Qwen3.5-4B and the BFCL GRPO-100 checkpoint with the native
# official BFCL V4 harness. This script runs only from an isolated Vessl
# worktree; it never changes the long-lived /root/projects/vita-rl checkout.
set -euo pipefail

source /root/setup_env.sh

REPO="${BFCL_EVAL_REPO:?BFCL_EVAL_REPO is required}"
RUN_ID="${BFCL_EVAL_RUN_ID:-vessl-bfcl-v4-qwen35-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${BFCL_EVAL_OUTPUT_ROOT:-/root/outputs/vita-rl/bfcl/${RUN_ID}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/root/venvs/dressage-cu129-py312/bin/python}"
MODEL_ID="qwen35-4b-local-FC"
TEMPERATURE="${BFCL_TEMPERATURE:-0.7}"
CONCURRENCY="${BFCL_CONCURRENCY:-30}"
PORT="${SGLANG_PORT:-30100}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
BFCL_ROOT="$REPO/external/gorilla/berkeley-function-call-leaderboard"
OVERLAY="$REPO/scripts/vessl/bfcl_qwen35_overlay.py"
SUMMARIZER="$REPO/scripts/vessl/summarize_bfcl_v4.py"
EXPECTED_TOTAL=4906

BASE_MODEL="${BFCL_BASE_MODEL:-/root/models/Qwen3.5-4B}"
GRPO_MODEL="${BFCL_GRPO100_MODEL:-/root/checkpoints/vita-rl/bfcl-multiturn-base-grpo100-diagnostics-20260910T000201Z/hf-best-heldout}"
CATEGORIES=(
    simple_python simple_java simple_javascript multiple parallel parallel_multiple irrelevance
    live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance
    multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context
    memory_kv memory_vector memory_rec_sum
)

[[ -e "$REPO/.git" ]] || { echo "Missing isolated repository: $REPO" >&2; exit 2; }
[[ -d "$BFCL_ROOT" ]] || { echo "Missing initialized BFCL submodule: $BFCL_ROOT" >&2; exit 2; }
[[ -x "$SGLANG_PYTHON" ]] || { echo "Missing Vessl inference Python: $SGLANG_PYTHON" >&2; exit 2; }
[[ -f "$OVERLAY" && -f "$SUMMARIZER" ]] || { echo "Missing BFCL Vessl support scripts" >&2; exit 2; }
[[ "$TEMPERATURE" == "0.7" ]] || { echo "This evaluation is pinned to Qwen non-thinking temperature 0.7" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs"
"$REPO/scripts/vessl/bootstrap_sglang_runtime.sh"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BFCL_QWEN_BASE_URL="$BASE_URL"
export BFCL_QWEN_API_KEY=EMPTY
export BFCL_QWEN_ENABLE_THINKING=false
export BFCL_QWEN_MAX_TOKENS=8192
export BFCL_QWEN_TOP_P=0.8
export BFCL_QWEN_TOP_K=20
export BFCL_QWEN_MIN_P=0.0
export BFCL_QWEN_PRESENCE_PENALTY=1.5
export BFCL_QWEN_REPETITION_PENALTY=1.0

"$SGLANG_PYTHON" - <<'PY'
import torch

if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected exactly two Vessl H100s, found {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(2)]
if any("H100" not in name for name in names):
    raise SystemExit(f"Expected H100 GPUs, found {names}")
print({"gpus": names})
PY

"$SGLANG_PYTHON" "$OVERLAY" --bfcl-root "$BFCL_ROOT" | tee "$RUN_ROOT/overlay-manifest.log"
PYTHONPATH="$BFCL_ROOT" "$SGLANG_PYTHON" - <<'PY'
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.utils import parse_test_category_argument

assert "qwen35-4b-local-FC" in MODEL_CONFIG_MAPPING
categories = parse_test_category_argument([
    "simple_python", "live_simple", "multi_turn_base", "memory_kv"
])
assert categories == ["live_simple", "memory_kv", "multi_turn_base", "simple_python"]
print("bfcl_adapter_and_categories=ok")
PY

SERVER_PID=""
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
cleanup() {
    local exit_code=$?
    stop_server
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

write_manifest() {
    local model_label="$1"
    local model_path="$2"
    local model_root="$3"
    "$SGLANG_PYTHON" - "$model_label" "$model_path" "$model_root/manifest.json" "$REPO" "$BFCL_ROOT" "$RUN_ID" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

label, model_path, output, repo, bfcl_root, run_id = sys.argv[1:]
model = Path(model_path)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def revision(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
files = {name: digest(model / name) for name in ("config.json", "generation_config.json", "model.safetensors.index.json")}
overlay_path = Path(bfcl_root) / ".vita-rl-qwen35-overlay.json"
payload = {
    "run_id": run_id,
    "model_label": label,
    "model_path": str(model),
    "model_file_sha256": files,
    "provenance": {
        "vita_rl_commit": revision(repo),
        "bfcl_submodule_commit": revision(str(Path(bfcl_root).parents[1])),
        "bfcl_overlay": json.loads(overlay_path.read_text()),
    },
    "inference": {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0,
        "thinking": False, "max_tokens": 8192,
    },
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

smoke_server() {
    local output="$1"
    "$SGLANG_PYTHON" - "$BASE_URL" "$output" <<'PY'
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

base_url, output = sys.argv[1:]
payload = {
    "model": "qwen35-4b-local",
    "messages": [{"role": "user", "content": "Call echo exactly once with text ping."}],
    "tools": [{"type": "function", "function": {
        "name": "echo", "description": "Echo text.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }}],
    "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "max_tokens": 256,
    "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = Request(f"{base_url}/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=300) as response:
    reply = json.loads(response.read())
message = reply["choices"][0]["message"]
if not message.get("tool_calls"):
    raise SystemExit("Non-thinking Qwen tool smoke returned no structured tool call")
if message.get("reasoning_content"):
    raise SystemExit("Non-thinking Qwen tool smoke unexpectedly returned reasoning content")
Path(output).write_text(json.dumps({
    "model": reply.get("model"), "tool_call_count": len(message["tool_calls"]),
    "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "thinking": False},
}, indent=2) + "\n")
PY
}

run_model() {
    local label="$1"
    local model_path="$2"
    local model_root="$RUN_ROOT/$label"
    local trace="$model_root/immediate_model_trace.jsonl"
    local server_log="$model_root/logs/sglang.log"
    [[ -d "$model_path" ]] || { echo "Missing $label model: $model_path" >&2; exit 2; }
    mkdir -p "$model_root/logs"
    write_manifest "$label" "$model_path" "$model_root"
    export BFCL_PROJECT_ROOT="$model_root"
    export BFCL_IMMEDIATE_TRACE_PATH="$trace"

    "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path "$model_path" --served-model-name qwen35-4b-local \
        --host 127.0.0.1 --port "$PORT" --dp-size 2 --mem-fraction-static 0.32 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
        >"$server_log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null 2>&1 && break
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "SGLang exited early; see $server_log" >&2; exit 1; }
        sleep 2
    done
    curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null || { echo "SGLang startup timed out; see $server_log" >&2; exit 1; }
    smoke_server "$model_root/qwen-nonthinking-tool-smoke.json"

    cd "$BFCL_ROOT"
    for category in "${CATEGORIES[@]}"; do
        echo "[$label] category=$category"
        PYTHONPATH="$BFCL_ROOT" "$SGLANG_PYTHON" -m bfcl_eval generate --model "$MODEL_ID" --test-category "$category" --temperature "$TEMPERATURE" \
            --num-threads "$CONCURRENCY" --include-input-log
        PYTHONPATH="$BFCL_ROOT" "$SGLANG_PYTHON" -m bfcl_eval evaluate --model "$MODEL_ID" --test-category "$category"
        printf '%s\n' "$category" >> "$model_root/completed-categories.txt"
    done
    "$SGLANG_PYTHON" "$SUMMARIZER" model --bfcl-root "$BFCL_ROOT" --run-root "$model_root" \
        --model "$MODEL_ID" --model-label "$label" --trace "$trace" --expected-total "$EXPECTED_TOTAL" \
        --output "$model_root/summary.json" "${CATEGORIES[@]}"
    stop_server
}

run_model base "$BASE_MODEL"
run_model grpo100_best_heldout "$GRPO_MODEL"
"$SGLANG_PYTHON" "$SUMMARIZER" comparison \
    --base "$RUN_ROOT/base/summary.json" --grpo "$RUN_ROOT/grpo100_best_heldout/summary.json" \
    --output "$RUN_ROOT/summary.json"
echo "BFCL evaluation accepted: $RUN_ROOT/summary.json"
