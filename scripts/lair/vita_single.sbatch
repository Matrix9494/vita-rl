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

# Run exactly one genuine VitaBench delivery task against a local SGLang
# endpoint. Override SGLANG_PORT, VITA_TASK_ID, or VITA_MAX_STEPS at submit
# time only when a deliberate experiment needs different values.
set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
MODEL="$REPO/models/Qwen3.5-4B"
VITABENCH="$REPO/external/vitabench"
PORT="${SGLANG_PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
TASK_ID="${VITA_TASK_ID:-10711001}"
MAX_STEPS="${VITA_MAX_STEPS:-100}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
SGLANG_LOG="$REPO/logs/sglang-${JOB_TAG}.log"
RESULT="$REPO/outputs/vita_single-${JOB_TAG}.json"

mkdir -p "$REPO/logs" "$REPO/outputs"
source "$VENV/bin/activate"
export VITA_MODEL_CONFIG_PATH="$REPO/configs/eval/local_sglang.yaml"
export HF_HOME=/data/user/dz13/vita-rl/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"

SGLANG_PID=""
cleanup() {
    local exit_code=$?
    if [[ -n "$SGLANG_PID" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }
[[ -d "$VITABENCH" ]] || { echo "Missing VitaBench checkout: $VITABENCH" >&2; exit 1; }

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

python - <<PY
from openai import OpenAI

client = OpenAI(base_url="${BASE_URL}", api_key="EMPTY")
response = client.chat.completions.create(
    model="qwen35-4b-local",
    messages=[{"role": "user", "content": "Reply with exactly: ready"}],
    max_tokens=16,
    temperature=0.0,
)
content = response.choices[0].message.content
if not content:
    raise RuntimeError("SGLang returned an empty test response")
print(f"sglang_test_response={content!r}")
PY

cd "$VITABENCH"
vita run \
    --domain delivery \
    --task-ids "$TASK_ID" \
    --agent-llm qwen35-4b-local \
    --user-llm qwen35-4b-local \
    --evaluator-llm qwen35-4b-local \
    --max-steps "$MAX_STEPS" \
    --num-trials 1 \
    --max-concurrency 1 \
    --language english \
    --save-to "$RESULT"

python - "$RESULT" <<'PY'
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
payload = json.loads(result.read_text())
simulations = payload.get("simulations", [])
if not simulations:
    raise SystemExit(f"No simulations were saved to {result}")
reward = simulations[0].get("reward_info")
if reward is None:
    raise SystemExit(f"No evaluator reward was saved to {result}")
print(f"result={result}")
print(f"reward={reward.get('reward')}")
PY
