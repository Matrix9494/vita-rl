#!/usr/bin/env bash
# Official BFCL V4 agentic memory evaluation for the local Qwen3.5-4B model.
# The benchmark supplies the multi-turn tool loop, memory backends, and scorer;
# this launcher only manages the allocated SGLang service and artifacts.
#
# Web-search categories run separately because their official WebSearchAPI
# requires SERPAPI_API_KEY.
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=bfcl_v4_memory
#SBATCH --output=/u/dz13/vita-rl/logs/bfcl_v4_memory-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/bfcl_v4_memory-%j.err

set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
BFCL_ROOT="$REPO/external/gorilla/berkeley-function-call-leaderboard"
MODEL_PATH="$REPO/models/Qwen3.5-4B"
MODEL_ID=qwen35-4b-local-FC
JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_TAG="bfcl_v4_agentic_memory_qwen35_4b_${JOB_TAG}"
RUN_ROOT="$REPO/outputs/bfcl/$RUN_TAG"
RAW_TRACE="$RUN_ROOT/immediate_model_trace.jsonl"
SUMMARY="$RUN_ROOT/summary.json"
PORT="${SGLANG_PORT:-$((43000 + ${SLURM_JOB_ID:-0} % 10000))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
CONCURRENCY="${BFCL_CONCURRENCY:-30}"
SGLANG_LOG="$REPO/logs/sglang-bfcl-v4-memory-${JOB_TAG}.log"
SGLANG_PID=""

mkdir -p "$REPO/logs" "$RUN_ROOT"
[[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
[[ -d "$REPO/external/gorilla/.git" && -d "$BFCL_ROOT/bfcl_eval" ]] || { echo "Missing official BFCL source: $BFCL_ROOT" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model checkpoint: $MODEL_PATH" >&2; exit 1; }

source "$VENV/bin/activate"
export HF_HOME=/data/user/dz13/vita-rl/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export BFCL_PROJECT_ROOT="$RUN_ROOT"
export BFCL_QWEN_BASE_URL="$BASE_URL"
export BFCL_QWEN_API_KEY=EMPTY
export BFCL_QWEN_MAX_TOKENS="${BFCL_QWEN_MAX_TOKENS:-8192}"
export BFCL_IMMEDIATE_TRACE_PATH="$RAW_TRACE"

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

sglang serve --model-path "$MODEL_PATH" \
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
curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null

cd "$BFCL_ROOT"
bfcl generate \
    --model "$MODEL_ID" \
    --test-category memory_kv,memory_vector,memory_rec_sum \
    --temperature 0.0 \
    --num-threads "$CONCURRENCY" \
    --include-input-log

bfcl evaluate \
    --model "$MODEL_ID" \
    --test-category memory_kv,memory_vector,memory_rec_sum

python "$REPO/scripts/lair/summarize_bfcl_agentic.py" \
    --run-root "$RUN_ROOT" \
    --model "$MODEL_ID" \
    --categories memory_kv memory_vector memory_rec_sum \
    --trace "$RAW_TRACE" \
    --output "$SUMMARY"
