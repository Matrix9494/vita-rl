#!/usr/bin/env bash
# Airline-only tau2 diagnostic.  This deliberately uses the benchmark's
# native full-history llm_agent but enables Qwen's exposed reasoning field and
# writes every immediate raw model response to a separate audit trace.
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=tau2_airline_trace
#SBATCH --output=/u/dz13/vita-rl/logs/tau2_airline_trace-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/tau2_airline_trace-%j.err

set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
TAU2_ROOT="$REPO/external/tau2-bench"
TAU2_DATA_DIR="$TAU2_ROOT/data"
MODEL="$REPO/models/Qwen3.5-4B"
JOB_TAG="${SLURM_JOB_ID:-manual}"
PORT="${SGLANG_PORT:-$((41000 + ${SLURM_JOB_ID:-0} % 10000))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
CONCURRENCY=30
RUN_TAG="tau2_airline_full_history_trace_qwen35_4b_${JOB_TAG}"
RESULT="$TAU2_DATA_DIR/simulations/${RUN_TAG}_airline.json"
ARTIFACT_DIR="$REPO/outputs/tau2"
EXPORTED_RESULT="$ARTIFACT_DIR/${RUN_TAG}_airline.json"
RAW_TRACE="$ARTIFACT_DIR/${RUN_TAG}_immediate_model_trace.jsonl"
TRACE="$ARTIFACT_DIR/${RUN_TAG}_transcript_trace.json"
SUMMARY="$ARTIFACT_DIR/${RUN_TAG}_summary.json"
SGLANG_LOG="$REPO/logs/sglang-tau2-airline-trace-${JOB_TAG}.log"
SGLANG_PID=""

mkdir -p "$REPO/logs" "$ARTIFACT_DIR" "$TAU2_DATA_DIR/simulations"
[[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
git -C "$TAU2_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Missing tau2 source: $TAU2_ROOT" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }

source "$VENV/bin/activate"
export TAU2_DATA_DIR
export HF_HOME=/data/user/dz13/vita-rl/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export TAU2_IMMEDIATE_TRACE_PATH="$RAW_TRACE"
export TAU2_REASONING_FALLBACK=1

cleanup() {
    local exit_code=$?
    if [[ -n "$SGLANG_PID" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

python - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected two allocated H100s, found {torch.cuda.device_count()}")
print(f"gpus={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
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

# tau2 passes these through LiteLLM directly to SGLang.  Qwen returns its
# exposed reasoning in message.reasoning_content, which tau2 stores raw_data.
LLM_MODEL="openai/qwen35-4b-local"
# Reasoning consumes output tokens before Qwen emits its visible reply.  The
# native tau2 validator requires that final reply, so the diagnostic budget is
# deliberately larger than the standard non-thinking benchmark budget.
LLM_ARGS="{\"api_base\":\"${BASE_URL}\",\"api_key\":\"EMPTY\",\"temperature\":0.0,\"max_tokens\":8192,\"chat_template_kwargs\":{\"enable_thinking\":true}}"

cd "$TAU2_ROOT"
tau2 run \
    --domain airline \
    --agent llm_agent \
    --agent-llm "$LLM_MODEL" \
    --agent-llm-args "$LLM_ARGS" \
    --user user_simulator \
    --user-llm "$LLM_MODEL" \
    --user-llm-args "$LLM_ARGS" \
    --num-trials 1 \
    --max-steps 200 \
    --max-errors 10 \
    --max-concurrency "$CONCURRENCY" \
    --seed 300 \
    --log-level ERROR \
    --save-to "${RUN_TAG}_airline"

# Preserve tau2's native result for resume, and publish an immutable project
# artifact next to the trace and summary for experiment comparison.
cp "$RESULT" "$EXPORTED_RESULT"
python "$REPO/scripts/lair/extract_tau2_model_trace.py" --result "$EXPORTED_RESULT" --output "$TRACE"
python - "$EXPORTED_RESULT" "$RAW_TRACE" "$TRACE" "$SUMMARY" "$BASE_URL" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
raw_trace_path = Path(sys.argv[2])
trace_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
base_url = sys.argv[5]
result = json.loads(result_path.read_text(encoding="utf-8"))
trace = json.loads(trace_path.read_text(encoding="utf-8"))
simulations = result["simulations"]
rewards = [(item.get("reward_info") or {}).get("reward", 0.0) for item in simulations]
summary = {
    "benchmark": "tau2-bench v0.2.0",
    "domain": "airline",
    "history_harness": "tau2 native llm_agent (standard full history)",
    "agent_model": "openai/qwen35-4b-local",
    "user_simulator_model": "openai/qwen35-4b-local",
    "concurrency": 30,
    "thinking_enabled": True,
    "reasoning_only_fallback": True,
    "result_file": str(result_path),
    "immediate_model_raw_trace": str(raw_trace_path),
    "transcript_trace": str(trace_path),
    "tasks": len(result["tasks"]),
    "simulations": len(simulations),
    "successes": sum(reward == 1.0 for reward in rewards),
    "mean_reward": sum(rewards) / len(rewards),
    "model_generations": trace["num_records"],
    "generations_with_exposed_reasoning": trace["num_reasoning_records"],
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"summary={summary_path}")
PY
