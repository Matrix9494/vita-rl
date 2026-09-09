#!/usr/bin/env bash
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=vita-mini-k3
#SBATCH --output=outputs/slurm-vita-mini-k3-%j.out
#SBATCH --error=outputs/slurm-vita-mini-k3-%j.err

# Seeded vita-mini tasks, evaluated by local Qwen3.5-4B using the same
# two-H100 / data-parallel SGLang setup as the prior VitaBench evaluation.
set -euo pipefail

REPO_ROOT="/u/dz13/vita-rl"
PYTHON="$REPO_ROOT/.venv-eval/bin/python"
SGLANG="$REPO_ROOT/.venv-eval/bin/sglang"
MODEL_PATH="$REPO_ROOT/models/Qwen3.5-4B"
SERVED_MODEL_NAME="qwen35-4b-local"
PORT="${SGLANG_PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
LOG_PATH="$REPO_ROOT/outputs/sglang-vita-mini-k3-${SLURM_JOB_ID}.log"
MAX_CONCURRENCY="${VITA_MAX_CONCURRENCY:-30}"
NUM_ENVIRONMENTS="${VITA_NUM_ENVIRONMENTS:-5}"
GENERATION_SEED="${VITA_GENERATION_SEED:-20260908}"
OUTPUT_PATH="${VITA_OUTPUT_PATH:-$REPO_ROOT/outputs/vita_mini_summary_k3_${NUM_ENVIRONMENTS}_seed${GENERATION_SEED}.json}"

mkdir -p "$REPO_ROOT/outputs"
cd "$REPO_ROOT"
# SGLang/FlashInfer JIT invokes Ninja by executable name.  The supported
# executable is installed in this project-local evaluation virtualenv.
export PATH="$REPO_ROOT/.venv-eval/bin:$PATH"

cleanup() {
  if [[ -n "${SGLANG_PID:-}" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
    kill "$SGLANG_PID" || true
    wait "$SGLANG_PID" || true
  fi
}
trap cleanup EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/external/vita-mini/src" \
  "$SGLANG" serve --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "$PORT" \
  --dp-size 2 --max-running-requests "$MAX_CONCURRENCY" \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  >"$LOG_PATH" 2>&1 &
SGLANG_PID=$!

MODELS_URL="http://127.0.0.1:${PORT}/v1/models"
for _ in $(seq 1 180); do
  if curl -fsS "$MODELS_URL" >/dev/null; then
    break
  fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "SGLang failed to start; see $LOG_PATH" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "$MODELS_URL" >/dev/null

PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/external/vita-mini/src" \
  VITA_RL_PROTOCOL=mini \
  "$PYTHON" -m vita_rl.environment_runner \
  --environment vita-mini --num-environments "$NUM_ENVIRONMENTS" --generation-seed "$GENERATION_SEED" \
  --harness vita_rl_summary --summary-window-size 3 --max-steps 100 \
  --max-concurrency "$MAX_CONCURRENCY" --model "$SERVED_MODEL_NAME" \
  --base-url "$BASE_URL" --temperature 1.0 --max-tokens 8192 \
  --output "$OUTPUT_PATH"
