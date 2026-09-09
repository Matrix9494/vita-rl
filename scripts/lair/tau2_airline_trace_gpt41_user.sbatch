#!/usr/bin/env bash
# Airline-only tau2 diagnostic: full-history local Qwen3.5-4B agent with an
# OpenRouter gpt-4.1 user simulator.  The proxy reads the local key at runtime;
# tau2 receives only a localhost URL and placeholder API key.
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=tau2_airline_gpt41
#SBATCH --output=/u/dz13/vita-rl/logs/tau2_airline_gpt41-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/tau2_airline_gpt41-%j.err

set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
TAU2_ROOT="$REPO/external/tau2-bench"
TAU2_DATA_DIR="$TAU2_ROOT/data"
MODEL="$REPO/models/Qwen3.5-4B"
OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-$REPO/key}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
PORT="${SGLANG_PORT:-$((42000 + ${SLURM_JOB_ID:-0} % 10000))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
CONCURRENCY=30
RUN_TAG="tau2_airline_full_history_qwen35_4b_gpt41_user_trace_${JOB_TAG}"
RESULT="$TAU2_DATA_DIR/simulations/${RUN_TAG}_airline.json"
ARTIFACT_DIR="$REPO/outputs/tau2"
EXPORTED_RESULT="$ARTIFACT_DIR/${RUN_TAG}_airline.json"
RAW_TRACE="$ARTIFACT_DIR/${RUN_TAG}_immediate_model_trace.jsonl"
TRACE="$ARTIFACT_DIR/${RUN_TAG}_transcript_trace.json"
SUMMARY="$ARTIFACT_DIR/${RUN_TAG}_summary.json"
SGLANG_LOG="$REPO/logs/sglang-tau2-airline-gpt41-${JOB_TAG}.log"
OPENROUTER_LOG="$REPO/logs/openrouter-tau2-airline-gpt41-${JOB_TAG}.log"
SGLANG_PID=""
OPENROUTER_PID=""
OPENROUTER_PORT_FILE=""

mkdir -p "$REPO/logs" "$ARTIFACT_DIR" "$TAU2_DATA_DIR/simulations"
[[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
git -C "$TAU2_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Missing tau2 source: $TAU2_ROOT" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }
[[ -r "$OPENROUTER_KEY_FILE" ]] || { echo "OpenRouter key file is unreadable: $OPENROUTER_KEY_FILE" >&2; exit 1; }

source "$VENV/bin/activate"
export TAU2_DATA_DIR
export HF_HOME=/data/user/dz13/vita-rl/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export TAU2_IMMEDIATE_TRACE_PATH="$RAW_TRACE"
# This applies only to Qwen's reasoning-only responses. The OpenRouter user
# path has no chat-template thinking flag.
export TAU2_REASONING_FALLBACK=1

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
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

python - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected two allocated H100s, found {torch.cuda.device_count()}")
print(f"gpus={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
PY

# The key is inherited only by the proxy process; it is removed before tau2 is
# invoked so neither LiteLLM metadata nor a trajectory can serialize it.
export OPENROUTER_API_KEY="$(<"$OPENROUTER_KEY_FILE")"
OPENROUTER_PORT_FILE="$(mktemp /tmp/tau2-openrouter.XXXXXX)"
python "$REPO/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" \
    >"$OPENROUTER_LOG" 2>&1 &
OPENROUTER_PID=$!
for _ in $(seq 1 100); do
    [[ -s "$OPENROUTER_PORT_FILE" ]] && break
    if ! kill -0 "$OPENROUTER_PID" 2>/dev/null; then
        echo "OpenRouter proxy exited early; see $OPENROUTER_LOG" >&2
        exit 1
    fi
    sleep 0.1
done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter proxy did not become ready" >&2; exit 1; }
OPENROUTER_BASE_URL="http://127.0.0.1:$(<"$OPENROUTER_PORT_FILE")/v1"
unset OPENROUTER_API_KEY

# Fail before allocating Qwen weights for a missing/invalid OpenRouter key.
python - "$OPENROUTER_BASE_URL" <<'PY'
import sys
import requests

response = requests.post(
    f"{sys.argv[1]}/chat/completions",
    headers={"Authorization": "Bearer tau2-local-proxy", "Content-Type": "application/json"},
    json={"model": "openai/gpt-4.1", "messages": [{"role": "user", "content": "Reply READY."}], "max_tokens": 8, "temperature": 0.0},
    timeout=(10, 120),
)
response.raise_for_status()
if not response.json().get("choices"):
    raise SystemExit("OpenRouter gpt-4.1 health check returned no choices")
print("openrouter_gpt41_ready=true")
PY

sglang serve --model-path "$MODEL" \
    --served-model-name qwen35-4b-local \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dp-size 2 \
    --max-running-requests "$CONCURRENCY" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    >"$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

for _ in $(seq 1 240); do
    if curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null; then
        break
    fi
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "SGLang exited early; see $SGLANG_LOG" >&2
        exit 1
    fi
    sleep 2
done
curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null || exit 1

AGENT_MODEL="openai/qwen35-4b-local"
AGENT_ARGS="{\"api_base\":\"${BASE_URL}\",\"api_key\":\"EMPTY\",\"temperature\":0.0,\"max_tokens\":8192,\"chat_template_kwargs\":{\"enable_thinking\":true}}"
# The second openai/ prefix preserves OpenRouter's vendor-qualified model ID
# after LiteLLM strips its own provider prefix.
USER_MODEL="openai/openai/gpt-4.1"
USER_ARGS="{\"api_base\":\"${OPENROUTER_BASE_URL}\",\"api_key\":\"tau2-local-proxy\",\"temperature\":0.0,\"max_tokens\":4096}"

cd "$TAU2_ROOT"
tau2 run \
    --domain airline \
    --agent llm_agent \
    --agent-llm "$AGENT_MODEL" \
    --agent-llm-args "$AGENT_ARGS" \
    --user user_simulator \
    --user-llm "$USER_MODEL" \
    --user-llm-args "$USER_ARGS" \
    --num-trials 1 \
    --max-steps 200 \
    --max-errors 10 \
    --max-concurrency "$CONCURRENCY" \
    --seed 300 \
    --log-level ERROR \
    --save-to "${RUN_TAG}_airline"

cp "$RESULT" "$EXPORTED_RESULT"
python "$REPO/scripts/lair/extract_tau2_model_trace.py" --result "$EXPORTED_RESULT" --output "$TRACE"
python - "$EXPORTED_RESULT" "$RAW_TRACE" "$TRACE" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

result_path, raw_trace_path, trace_path, summary_path = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
trace = json.loads(trace_path.read_text(encoding="utf-8"))
raw_records = [json.loads(line) for line in raw_trace_path.read_text(encoding="utf-8").splitlines()]
simulations = result["simulations"]
rewards = [(item.get("reward_info") or {}).get("reward", 0.0) for item in simulations]
summary = {
    "benchmark": "tau2-bench v0.2.0",
    "domain": "airline",
    "history_harness": "tau2 native llm_agent (standard full history)",
    "agent_model": "openai/qwen35-4b-local",
    "user_simulator_model": "openai/gpt-4.1 via localhost OpenRouter proxy",
    "concurrency": 30,
    "agent_thinking_enabled": True,
    "reasoning_only_fallback": True,
    "result_file": str(result_path),
    "immediate_model_raw_trace": str(raw_trace_path),
    "transcript_trace": str(trace_path),
    "tasks": len(result["tasks"]),
    "simulations": len(simulations),
    "successes": sum(reward == 1.0 for reward in rewards),
    "mean_reward": sum(rewards) / len(rewards),
    "transcript_model_generations": trace["num_records"],
    "immediate_raw_generations": len(raw_records),
    "qwen_reasoning_only_fallback_retries": sum(record["retry_without_thinking"] for record in raw_records),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"summary={summary_path}")
PY
