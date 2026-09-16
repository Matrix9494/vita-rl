#!/usr/bin/env bash
# Run the fixed 100-task VitaBench delivery evaluation with the LLM state-delta
# harness on Vessl.  OPENROUTER_API_KEY must be supplied only in this process
# environment by the remote submit wrapper; this script never reads a key file.
set -euo pipefail

source /root/setup_env.sh

REPO="${VITA_EVAL_REPO:-${VITA_RL_ROOT:?VITA_RL_ROOT is required}}"
VITA_ROOT="${VITA_ROOT:?VITA_ROOT is required}"
VITA_PYTHON="${VITA_PYTHON:-${VITA_VENV:?VITA_VENV is required}/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/root/venvs/dressage-cu129-py312/bin/python}"
MODEL="${QWEN_MODEL:?QWEN_MODEL is required}"
RUN_ID="${VITA_EVAL_RUN_ID:-vessl-vitabench-delivery-state-delta-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${VITA_EVAL_OUTPUT_ROOT:-/root/outputs/vita-rl/vitabench/${RUN_ID}}"
TASK_SET=delivery
TASK_COUNT=100
SELECTION_SEED=20260906
MAX_STEPS=100
CONCURRENCY="${VITA_MAX_CONCURRENCY:-30}"
PORT="${SGLANG_PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

RESULT="$RUN_ROOT/result.json"
SUMMARY="$RUN_ROOT/summary.json"
SELECTION="$RUN_ROOT/task-selection.json"
SGLANG_SMOKE="$RUN_ROOT/sglang-smoke.json"
STATE_DELTA_TRACE="$RUN_ROOT/state-delta-trace.jsonl"
SGLANG_LOG="$RUN_ROOT/logs/sglang.log"
OPENROUTER_LOG="$RUN_ROOT/logs/openrouter-proxy.log"

[[ -d "$REPO" ]] || { echo "Missing evaluation repository: $REPO" >&2; exit 2; }
[[ -d "$VITA_ROOT" ]] || { echo "Missing VitaBench checkout: $VITA_ROOT" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
[[ -x "$VITA_PYTHON" ]] || { echo "Missing VitaBench Python: $VITA_PYTHON" >&2; exit 2; }
[[ -x "$SGLANG_PYTHON" ]] || { echo "Missing SGLang Python: $SGLANG_PYTHON" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required in-memory" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs"
export VITA_RL_ROOT="$REPO"
export VITA_INPUT_TOKENIZER_PATH="$MODEL"
export VITA_STATE_DELTA_UPDATER=llm
export VITA_STATE_DELTA_TRACE_PATH="$STATE_DELTA_TRACE"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"

SGLANG_PID=""
OPENROUTER_PID=""
OPENROUTER_PORT_FILE=""
VITA_MODEL_CONFIG=""
cleanup() {
    local exit_code=$?
    [[ -n "$SGLANG_PID" ]] && kill "$SGLANG_PID" 2>/dev/null || true
    [[ -n "$OPENROUTER_PID" ]] && kill "$OPENROUTER_PID" 2>/dev/null || true
    [[ -z "$OPENROUTER_PORT_FILE" ]] || rm -f "$OPENROUTER_PORT_FILE"
    [[ -z "$VITA_MODEL_CONFIG" ]] || rm -f "$VITA_MODEL_CONFIG"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

"$SGLANG_PYTHON" - <<'PY'
import torch
import sglang

if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected exactly two Vessl H100s, found {torch.cuda.device_count()}")
print({"sglang": getattr(sglang, "__version__", "unknown"), "gpus": [torch.cuda.get_device_name(i) for i in range(2)]})
PY
PYTHONPATH="$VITA_ROOT/src" "$VITA_PYTHON" -c 'import vita; print("vitabench_import=ok")'

# The proxy inherits the key; benchmark and model processes receive only a
# disposable localhost credential.  The runner's own copy is removed directly
# after the proxy has become ready.
OPENROUTER_PORT_FILE="$(mktemp /tmp/vita-rl-openrouter.XXXXXX)"
"$VITA_PYTHON" "$REPO/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" \
    >"$OPENROUTER_LOG" 2>&1 &
OPENROUTER_PID=$!
for _ in $(seq 1 100); do
    [[ -s "$OPENROUTER_PORT_FILE" ]] && break
    kill -0 "$OPENROUTER_PID" 2>/dev/null || { echo "OpenRouter proxy exited early" >&2; exit 1; }
    sleep 0.1
done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter proxy startup timed out" >&2; exit 1; }

VITA_MODEL_CONFIG="$(mktemp /tmp/vita-rl-models.XXXXXX.yaml)"
"$VITA_PYTHON" - "$VITA_MODEL_CONFIG" "$(<"$OPENROUTER_PORT_FILE")" "$BASE_URL/chat/completions" <<'PY'
import json
import sys
from pathlib import Path

path, proxy_port, agent_endpoint = sys.argv[1:]
config = {
    "default": {
        "base_url": f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        "temperature": 0.0,
        "headers": {"Authorization": "Bearer vita-rl-local-proxy", "Content-Type": "application/json"},
    },
    "models": [{
        "name": "gpt-4.1", "max_tokens": 4096,
    }, {
        "name": "qwen35-4b-local",
        "base_url": agent_endpoint,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "n": 1,
        "max_tokens": 8192,
        "max_input_tokens": 65536,
        "chat_template_kwargs": {"enable_thinking": False},
        "headers": {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
    }],
}
Path(path).write_text(json.dumps(config, indent=2) + "\n")
PY
export VITA_MODEL_CONFIG_PATH="$VITA_MODEL_CONFIG"
unset OPENROUTER_API_KEY

cd "$VITA_ROOT"
PYTHONPATH="$VITA_ROOT/src" "$VITA_PYTHON" - "$SELECTION" <<'PY'
import json
import random
import sys
from vita.run import load_tasks

tasks = load_tasks("delivery", language="english")
selected = random.Random(20260906).sample(tasks, 100)
payload = {"task_set": "delivery", "selection_seed": 20260906, "task_count": 100,
           "task_ids": [task.id for task in selected]}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
mapfile -t TASK_IDS < <("$VITA_PYTHON" - "$SELECTION" <<'PY'
import json
import sys
for task_id in json.load(open(sys.argv[1], encoding="utf-8"))["task_ids"]:
    print(task_id)
PY
)
[[ "${#TASK_IDS[@]}" -eq "$TASK_COUNT" ]] || { echo "Task selection was incomplete" >&2; exit 1; }

"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --dp-size 2 \
    --mem-fraction-static 0.32 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    >"$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
for _ in $(seq 1 240); do
    curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null 2>&1 && break
    kill -0 "$SGLANG_PID" 2>/dev/null || { echo "SGLang exited early; see $SGLANG_LOG" >&2; exit 1; }
    sleep 2
done
curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null || {
    echo "SGLang startup timed out; see $SGLANG_LOG" >&2; exit 1;
}

"$VITA_PYTHON" "$REPO/scripts/lair/qwen35_openai_smoke.py" \
    --base-url "$BASE_URL" --model qwen35-4b-local --output "$SGLANG_SMOKE"

PYTHONPATH="$REPO/src:$VITA_ROOT/src" "$VITA_PYTHON" -m vita_rl.vita_cli run \
    --domain delivery --task-set-name delivery --task-ids "${TASK_IDS[@]}" \
    --agent vita_rl_state_delta --agent-llm qwen35-4b-local \
    --user-llm gpt-4.1 --evaluator-llm gpt-4.1 --max-steps "$MAX_STEPS" \
    --num-trials 1 --max-concurrency "$CONCURRENCY" --language english --save-to "$RESULT"

[[ -s "$STATE_DELTA_TRACE" ]] || { echo "State-delta trace is empty: $STATE_DELTA_TRACE" >&2; exit 1; }

REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
VITABENCH_COMMIT="$(git -C "$VITA_ROOT" rev-parse HEAD)"
"$VITA_PYTHON" - "$RESULT" "$SUMMARY" "$SELECTION" "$SGLANG_SMOKE" "$STATE_DELTA_TRACE" \
    "$REPO_COMMIT" "$VITABENCH_COMMIT" "$RUN_ID" "$CONCURRENCY" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

(result_path, summary_path, selection_path, smoke_path, trace_path,
 repo_commit, vitabench_commit, run_id, concurrency) = sys.argv[1:]
result = json.loads(Path(result_path).read_text())
simulations = result.get("simulations", [])
if len(simulations) != 100:
    raise SystemExit(f"Expected 100 completed tasks, got {len(simulations)}")
records = []
for simulation in simulations:
    messages = simulation.get("messages", [])
    usages = [message.get("usage") or {} for message in messages]
    records.append({
        "task_id": simulation.get("task_id"),
        "reward": (simulation.get("reward_info") or {}).get("reward"),
        "success": (simulation.get("reward_info") or {}).get("reward") == 1.0,
        "agent_steps": max(0, len(messages) - 1),
        "wall_clock_seconds": simulation.get("duration"),
        "prompt_tokens_total": sum(usage.get("prompt_tokens", 0) for usage in usages),
        "output_tokens_total": sum(usage.get("completion_tokens", 0) for usage in usages),
        "tool_error_count": sum(bool(message.get("error")) for message in messages if message.get("role") == "tool"),
        "termination_reason": simulation.get("termination_reason"),
    })
traces = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
if not traces:
    raise SystemExit("State-delta trace was empty")
summary = {
    "benchmark": "VitaBench",
    "run_id": run_id,
    "role_models": {"agent": "qwen35-4b-local", "user_simulator": "gpt-4.1", "evaluator": "gpt-4.1"},
    "agent_inference": {"temperature": 0.0, "thinking": False, "top_p": 1.0, "top_k": 1},
    "harness": "vita_rl_state_delta",
    "state_delta_updater": "llm",
    "task_set": "delivery",
    "selection_seed": 20260906,
    "task_count": 100,
    "max_steps": 100,
    "requested_concurrency": int(concurrency),
    "result_file": result_path,
    "task_selection_file": selection_path,
    "sglang_smoke_file": smoke_path,
    "state_delta_trace_file": trace_path,
    "provenance": {"vita_rl_commit": repo_commit, "vitabench_commit": vitabench_commit},
    "aggregate": {
        "successes": sum(record["success"] for record in records),
        "mean_reward": sum((record["reward"] or 0.0) for record in records) / len(records),
        "termination_reasons": dict(Counter(record["termination_reason"] for record in records)),
        "tool_error_count": sum(record["tool_error_count"] for record in records),
        "agent_steps": sum(record["agent_steps"] for record in records),
        "prompt_tokens": sum(record["prompt_tokens_total"] for record in records),
        "output_tokens": sum(record["output_tokens_total"] for record in records),
    },
    "state_delta_diagnostics": {
        "trace_records": len(traces),
        "updater_calls": sum(trace.get("updater") == "llm" for trace in traces),
        "validation_errors": sum(bool(trace.get("validation_error")) for trace in traces),
        "rejected_ops": sum(len(trace.get("rejected_ops") or []) for trace in traces),
        "unresolved_evidence_final": len(traces[-1].get("unresolved_evidence") or []),
        "stop_request_rejections": sum(
            bool((trace.get("termination_decision") or {}).get("requested"))
            and not bool((trace.get("termination_decision") or {}).get("accepted"))
            for trace in traces
        ),
        "fallback_behavior": "none",
    },
    "tasks": records,
}
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
print(f"summary={summary_path}")
PY
