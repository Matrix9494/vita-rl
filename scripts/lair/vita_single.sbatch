#!/usr/bin/env bash
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=vita_single
#SBATCH --output=/u/dz13/vita-rl/logs/vita_single-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/vita_single-%j.err

# Run a bounded genuine VitaBench delivery diagnostic against a local SGLang
# endpoint. It defaults to task 10711001; set VITA_TASK_IDS to a
# space-separated fixed subset when deliberately expanding the diagnostic.
set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
MODEL="$REPO/models/Qwen3.5-4B"
VITABENCH="$REPO/external/vitabench"
PORT="${SGLANG_PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
TASK_IDS="${VITA_TASK_IDS:-${VITA_TASK_ID:-10711001}}"
MAX_STEPS="${VITA_MAX_STEPS:-100}"
AGENT_THINKING="${VITA_AGENT_THINKING:-false}"
AGENT_IMPLEMENTATION="${VITA_AGENT_IMPLEMENTATION:-vita_rl_standard}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
SGLANG_LOG="$REPO/logs/sglang-${JOB_TAG}.log"
RESULT="$REPO/outputs/vita_single-${JOB_TAG}.json"
SUMMARY="$REPO/outputs/vita_diagnostic-${JOB_TAG}.json"
SGLANG_SMOKE="$REPO/outputs/sglang_qwen35_smoke-${JOB_TAG}.json"
OPENROUTER_LOG="$REPO/logs/openrouter-${JOB_TAG}.log"

mkdir -p "$REPO/logs" "$REPO/outputs"
source "$VENV/bin/activate"
export VITA_INPUT_TOKENIZER_PATH="$MODEL"
export HF_HOME=/data/user/dz13/vita-rl/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"

SGLANG_PID=""
OPENROUTER_PID=""
OPENROUTER_PORT_FILE=""
VITA_MODEL_CONFIG=""
cleanup() {
    local exit_code=$?
    if [[ -n "$SGLANG_PID" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
    if [[ -n "$OPENROUTER_PID" ]] && kill -0 "$OPENROUTER_PID" 2>/dev/null; then
        kill "$OPENROUTER_PID" 2>/dev/null || true
        wait "$OPENROUTER_PID" 2>/dev/null || true
    fi
    [[ -z "$OPENROUTER_PORT_FILE" ]] || rm -f "$OPENROUTER_PORT_FILE"
    [[ -z "$VITA_MODEL_CONFIG" ]] || rm -f "$VITA_MODEL_CONFIG"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }
[[ -d "$VITABENCH" ]] || { echo "Missing VitaBench checkout: $VITABENCH" >&2; exit 1; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
    echo "OPENROUTER_API_KEY is required: this launcher follows the VESSL role split (local Qwen agent, gpt-4.1 user/evaluator)." >&2
    exit 2
}

# Match VESSL's baseline role assignment without storing the OpenRouter secret
# in VitaBench's model configuration or result metadata.  The relay inherits
# the key, then the benchmark process has the key removed from its environment.
OPENROUTER_PORT_FILE="$(mktemp /tmp/vita-rl-openrouter.XXXXXX)"
python "$REPO/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" \
    >"$OPENROUTER_LOG" 2>&1 &
OPENROUTER_PID=$!
for _ in $(seq 1 100); do
    [[ -s "$OPENROUTER_PORT_FILE" ]] && break
    if ! kill -0 "$OPENROUTER_PID" 2>/dev/null; then
        echo "OpenRouter relay exited early; see $OPENROUTER_LOG" >&2
        exit 1
    fi
    sleep 0.1
done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter relay did not become ready; see $OPENROUTER_LOG" >&2; exit 1; }

VITA_MODEL_CONFIG="$(mktemp /tmp/vita-rl-models.XXXXXX.yaml)"
python - "$VITA_MODEL_CONFIG" "$(<"$OPENROUTER_PORT_FILE")" "$BASE_URL/chat/completions" <<'PY'
import json
import sys
from pathlib import Path

path, proxy_port, agent_endpoint = sys.argv[1:]
config = {
    "default": {
        "base_url": f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        "temperature": 0.0,
        "headers": {
            "Authorization": "Bearer vita-rl-local-proxy",
            "Content-Type": "application/json",
        },
    },
    "models": [
        {"name": "gpt-4.1", "max_tokens": 4096},
        {
            "name": "qwen35-4b-local",
            "base_url": agent_endpoint,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "n": 1,
            "max_tokens": 8192,
            "max_input_tokens": 65536,
            "chat_template_kwargs": {"enable_thinking": False},
            "headers": {
                "Authorization": "Bearer EMPTY",
                "Content-Type": "application/json",
            },
        },
    ],
}
Path(path).write_text(json.dumps(config, indent=2) + "\n")
PY
export VITA_MODEL_CONFIG_PATH="$VITA_MODEL_CONFIG"
unset OPENROUTER_API_KEY

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside this Slurm allocation")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

sglang serve --model-path "$MODEL" \
    --served-model-name qwen35-4b-local \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    >"$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

for _ in $(seq 1 180); do
    if curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null; then
        break
    fi
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "SGLang exited early; see $SGLANG_LOG" >&2
        exit 1
    fi
    sleep 2
done
curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null || {
    echo "SGLang did not become healthy; see $SGLANG_LOG" >&2
    exit 1
}

python "$REPO/scripts/lair/qwen35_openai_smoke.py" \
    --base-url "$BASE_URL" \
    --model qwen35-4b-local \
    --output "$SGLANG_SMOKE"

VITA_THINK_ARGS=()
AGENT_MODEL=qwen35-4b-local
if [[ "$AGENT_THINKING" == "true" ]]; then
    VITA_THINK_ARGS+=(--enable-think)
elif [[ "$AGENT_THINKING" != "false" ]]; then
    echo "VITA_AGENT_THINKING must be true or false, got: $AGENT_THINKING" >&2
    exit 2
fi

cd "$VITABENCH"
# Intentional shell splitting: the documented VITA_TASK_IDS interface is a
# whitespace-separated sequence of VitaBench task identifiers.
# shellcheck disable=SC2206
TASK_ID_ARGS=( $TASK_IDS )
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python -m vita_rl.vita_cli run \
    --domain delivery \
    --task-ids "${TASK_ID_ARGS[@]}" \
    --agent "$AGENT_IMPLEMENTATION" \
    --agent-llm "$AGENT_MODEL" \
    --user-llm gpt-4.1 \
    --evaluator-llm gpt-4.1 \
    --max-steps "$MAX_STEPS" \
    --num-trials 1 \
    --max-concurrency 1 \
    "${VITA_THINK_ARGS[@]}" \
    --language english \
    --save-to "$RESULT"

python - "$RESULT" "$SUMMARY" "$SGLANG_SMOKE" <<'PY'
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
sglang_smoke_path = sys.argv[3]
payload = json.loads(result.read_text())
simulations = payload.get("simulations", [])
if not simulations:
    raise SystemExit(f"No simulations were saved to {result}")

records = []
for simulation in simulations:
    usages = [message.get("usage") or {} for message in simulation.get("messages", [])]
    finish_reasons = [
        (message.get("raw_data") or {}).get("finish_reason")
        for message in simulation.get("messages", [])
    ]
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    termination = simulation.get("termination_reason")
    records.append({
        "task_id": simulation.get("task_id"),
        "reward": reward,
        "success": reward == 1.0,
        "agent_steps": max(0, len(simulation.get("messages", [])) - 1),
        "wall_clock_seconds": simulation.get("duration"),
        "prompt_tokens_total": sum(usage.get("prompt_tokens", 0) for usage in usages),
        "prompt_tokens_max_request": max((usage.get("prompt_tokens", 0) for usage in usages), default=0),
        "output_tokens_total": sum(usage.get("completion_tokens", 0) for usage in usages),
        "output_tokens_max_request": max((usage.get("completion_tokens", 0) for usage in usages), default=0),
        "generation_hit_max_tokens": "length" in finish_reasons,
        "termination_reason": termination,
        "failure_class": (
            "successful_completion" if reward == 1.0 else
            "max_steps_termination" if termination == "max_steps" else
            "tool_call_or_agent_formatting_failure" if termination == "invalid_agent_message" else
            "infrastructure_or_evaluator_failure" if reward is None else
            "task_not_completed"
        ),
    })

assistant_tool_calls = [
    tool_call
    for simulation in simulations
    for message in simulation.get("messages", [])
    if message.get("role") == "assistant"
    for tool_call in (message.get("tool_calls") or [])
]
tool_messages = [
    message
    for simulation in simulations
    for message in simulation.get("messages", [])
    if message.get("role") == "tool"
]
user_reasoning_leaks = [
    {
        "content": message.get("content"),
        "reasoning_content": ((message.get("raw_data") or {}).get("message") or {}).get("reasoning_content"),
    }
    for simulation in simulations
    for message in simulation.get("messages", [])
    if message.get("role") == "user"
    and (
        "<think" in (message.get("content") or "").lower()
        or "</think" in (message.get("content") or "").lower()
        or ((message.get("raw_data") or {}).get("message") or {}).get("reasoning_content")
    )
]

summary = {
    "result_file": str(result),
    "sglang_smoke_file": sglang_smoke_path,
    "effective_settings": {
        "agent_model": "qwen35-4b-local",
        "user_simulator_model": "gpt-4.1",
        "evaluator_model": "gpt-4.1",
        "agent_thinking": "${AGENT_THINKING}" == "true",
        "user_simulator_thinking": "provider_default (gpt-4.1)",
        "evaluator_thinking": "provider_default (gpt-4.1)",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "n": 1,
        "agent_max_tokens": 8192,
        "user_evaluator_max_tokens": 4096,
        "max_input_tokens": 65536,
        "max_steps": payload.get("info", {}).get("max_steps"),
    },
    "tasks": records,
    "tool_use_verification": {
        "assistant_tool_call_count": len(assistant_tool_calls),
        "tool_message_count": len(tool_messages),
        "user_reasoning_leak_count": len(user_reasoning_leaks),
        "user_reasoning_leaks": user_reasoning_leaks,
    },
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"result={result}")
print(f"diagnostic_summary={summary_path}")
for record in records:
    print(json.dumps(record, sort_keys=True))
print(json.dumps(summary["tool_use_verification"], sort_keys=True))

if not assistant_tool_calls:
    raise SystemExit("VitaBench smoke failed: no structured assistant.tool_calls were recorded")
if not tool_messages:
    raise SystemExit("VitaBench smoke failed: no ToolMessage/tool result was recorded")
if user_reasoning_leaks:
    raise SystemExit("VitaBench smoke failed: user simulator reasoning leaked into a UserMessage")
PY
