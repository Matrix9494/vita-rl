#!/usr/bin/env bash
# Configurable two-H100 VitaBench batch. The local SGLang server uses two
# data-parallel replicas and admits at most 20 running requests; VitaBench
# launches at most 20 simulations concurrently. By default it remains a
# 40-task cross-domain diagnostic. Set VITA_TASK_SET and VITA_TASK_COUNT to
# run any complete 100-task set; the full benchmark is four such batches.
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=vita40_dp2
#SBATCH --output=/u/dz13/vita-rl/logs/vita40-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/vita40-%j.err

set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
MODEL="$REPO/models/Qwen3.5-4B"
VITABENCH="$REPO/external/vitabench"
PORT="${SGLANG_PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
TASK_SET="${VITA_TASK_SET:-cross_domain}"
TASK_COUNT="${VITA_TASK_COUNT:-40}"
SELECTION_SEED="${VITA_SELECTION_SEED:-20260906}"
MAX_STEPS="${VITA_MAX_STEPS:-300}"
CONCURRENCY="${VITA_MAX_CONCURRENCY:-20}"
AGENT_IMPLEMENTATION="${VITA_AGENT_IMPLEMENTATION:-vita_rl_standard}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
SGLANG_LOG="$REPO/logs/sglang-vita40-${JOB_TAG}.log"
RESULT="$REPO/outputs/vita40-${JOB_TAG}.json"
SUMMARY="$REPO/outputs/vita40_summary-${JOB_TAG}.json"
SELECTION="$REPO/outputs/vita40_tasks-${JOB_TAG}.json"
SGLANG_SMOKE="$REPO/outputs/sglang_qwen35_smoke-${JOB_TAG}.json"
OPENROUTER_LOG="$REPO/logs/openrouter-vita40-${JOB_TAG}.log"

case "$TASK_SET" in
    delivery|ota|instore)
        DOMAIN="$TASK_SET"
        ;;
    cross_domain)
        DOMAIN="delivery,ota,instore"
        ;;
    *)
        echo "VITA_TASK_SET must be one of delivery, ota, instore, cross_domain; got: $TASK_SET" >&2
        exit 2
        ;;
esac

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

# Keep the actual API key in the relay only. VitaBench sees a disposable
# localhost endpoint and a placeholder header, just as in VESSL.
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

if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected two allocated GPUs, found {torch.cuda.device_count()}")
print(f"torch={torch.__version__}")
print(f"gpus={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
PY

# Persist the sampled IDs before launching work. VitaBench's --num-tasks takes
# the first N tasks, so sampling must be explicit to produce a random subset.
cd "$VITABENCH"
python - "$SELECTION" "$TASK_SET" "$TASK_COUNT" "$SELECTION_SEED" <<'PY'
import json
import random
import sys
from vita.run import load_tasks

path, task_set, task_count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tasks = load_tasks(task_set, language="english")
if task_count > len(tasks):
    raise SystemExit(f"Requested {task_count} tasks but {task_set} only has {len(tasks)}")
selected = random.Random(seed).sample(tasks, task_count)
payload = {
    "task_set": task_set,
    "selection_seed": seed,
    "task_count": task_count,
    "task_ids": [task.id for task in selected],
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(json.dumps(payload))
PY
mapfile -t TASK_IDS < <(python - "$SELECTION" <<'PY'
import json
import sys
for task_id in json.load(open(sys.argv[1], encoding="utf-8"))["task_ids"]:
    print(task_id)
PY
)
[[ "${#TASK_IDS[@]}" -eq "$TASK_COUNT" ]] || { echo "Task selection was incomplete" >&2; exit 1; }

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
curl --fail --silent --show-error --max-time 5 "$BASE_URL/models" >/dev/null || {
    echo "SGLang did not become healthy; see $SGLANG_LOG" >&2
    exit 1
}

python "$REPO/scripts/lair/qwen35_openai_smoke.py" \
    --base-url "$BASE_URL" \
    --model qwen35-4b-local \
    --output "$SGLANG_SMOKE"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python -m vita_rl.vita_cli run \
    --domain "$DOMAIN" \
    --task-set-name "$TASK_SET" \
    --task-ids "${TASK_IDS[@]}" \
    --agent "$AGENT_IMPLEMENTATION" \
    --agent-llm qwen35-4b-local \
    --user-llm gpt-4.1 \
    --evaluator-llm gpt-4.1 \
    --max-steps "$MAX_STEPS" \
    --num-trials 1 \
    --max-concurrency "$CONCURRENCY" \
    --enable-think \
    --language english \
    --save-to "$RESULT"

python - "$RESULT" "$SUMMARY" "$SELECTION" "$SGLANG_SMOKE" "$CONCURRENCY" <<'PY'
import json
import sys
from pathlib import Path

result_path, summary_path, selection_path, smoke_path = map(Path, sys.argv[1:5])
concurrency = int(sys.argv[5])
result = json.loads(result_path.read_text())
simulations = result.get("simulations", [])
if not simulations:
    raise SystemExit(f"No simulations were saved to {result_path}")

records = []
for simulation in simulations:
    usages = [message.get("usage") or {} for message in simulation.get("messages", [])]
    finish_reasons = [(message.get("raw_data") or {}).get("finish_reason") for message in simulation.get("messages", [])]
    reward = (simulation.get("reward_info") or {}).get("reward")
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
    })

summary = {
    "result_file": str(result_path),
    "task_selection_file": str(selection_path),
    "sglang_smoke_file": str(smoke_path),
    "role_models": {
        "agent": "qwen35-4b-local",
        "user_simulator": "gpt-4.1",
        "evaluator": "gpt-4.1",
    },
    "agent_inference": {
        "thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": 8192,
        "max_input_tokens": 65536,
    },
    "user_evaluator_inference": {"max_tokens": 4096, "temperature": 0.0},
    "requested_concurrency": concurrency,
    "requested_sglang_data_parallel_size": 2,
    "requested_sglang_max_running_requests": concurrency,
    "tasks": records,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"parallel_summary={summary_path}")
PY
