#!/usr/bin/env bash
# Official tau2-bench v0.2.0, full task suite (50 airline + 114 retail +
# 114 telecom).  The benchmark's native `llm_agent` is intentionally used:
# it supplies the original policy prompt and the complete conversation/tool
# transcript at every turn.  No compressed-history vita-rl harness is used.
#
# Agent and user simulator both use the local Qwen3.5-4B endpoint.  This keeps
# this formal run self-contained: no remote-model key is written to a config,
# Slurm environment, trajectory, or log.  Change TAU2_USER_MODEL/ARGS only if
# comparing against a separately documented user-simulator configuration.
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=tau2_full_q35_4b
#SBATCH --output=/u/dz13/vita-rl/logs/tau2_full-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/tau2_full-%j.err

set -euo pipefail

REPO=/u/dz13/vita-rl
VENV="$REPO/.venv-eval"
TAU2_ROOT="$REPO/external/tau2-bench"
MODEL="$REPO/models/Qwen3.5-4B"
TAU2_DATA_DIR="$TAU2_ROOT/data"
JOB_TAG="${SLURM_JOB_ID:-manual}"
PORT="${SGLANG_PORT:-$((31000 + ${SLURM_JOB_ID:-0} % 10000))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
CONCURRENCY="${TAU2_MAX_CONCURRENCY:-12}"
MAX_STEPS="${TAU2_MAX_STEPS:-200}"
MAX_ERRORS="${TAU2_MAX_ERRORS:-10}"
RUN_TAG="${TAU2_RUN_TAG:-tau2_full_history_qwen35_4b_${JOB_TAG}}"
SGLANG_LOG="$REPO/logs/sglang-tau2-${JOB_TAG}.log"
SGLANG_SMOKE="$REPO/outputs/sglang_tau2_qwen35_smoke-${JOB_TAG}.json"
SUMMARY="$REPO/outputs/tau2_full_history_summary-${JOB_TAG}.json"
# tau2's DATA_DIR is the repository's ``data/`` directory, and its runner
# writes runs directly below ``data/simulations`` (not ``data/tau2``).
RESULT_DIR="$TAU2_DATA_DIR/simulations"

mkdir -p "$REPO/logs" "$REPO/outputs" "$RESULT_DIR"
[[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
git -C "$TAU2_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Missing tau2 source checkout: $TAU2_ROOT" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }

source "$VENV/bin/activate"
export TAU2_DATA_DIR
export HF_HOME=/data/user/dz13/vita-rl/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

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

python - <<'PY'
import torch
import tau2

if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected two allocated H100s, found {torch.cuda.device_count()}")
print(f"tau2={tau2.__file__}")
print(f"torch={torch.__version__}")
print(f"gpus={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
PY

# Data parallelism creates two Qwen replicas; tau2's thread pool feeds both
# agent and user-simulator requests through the same OpenAI-compatible server.
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

# `openai/...` tells LiteLLM to use the local OpenAI-compatible SGLang server.
# Both JSON objects are immutable across all three domains, a requirement for a
# comparable full-suite result.  Thinking is disabled for reproducible tool
# behavior and to match the benchmark's standard non-reasoning agent path.
AGENT_MODEL="openai/qwen35-4b-local"
USER_MODEL="openai/qwen35-4b-local"
LLM_ARGS="{\"api_base\":\"${BASE_URL}\",\"api_key\":\"EMPTY\",\"temperature\":0.0,\"max_tokens\":1024,\"chat_template_kwargs\":{\"enable_thinking\":false}}"

cd "$TAU2_ROOT"
for DOMAIN in airline retail telecom; do
    tau2 run \
        --domain "$DOMAIN" \
        --agent llm_agent \
        --agent-llm "$AGENT_MODEL" \
        --agent-llm-args "$LLM_ARGS" \
        --user user_simulator \
        --user-llm "$USER_MODEL" \
        --user-llm-args "$LLM_ARGS" \
        --num-trials 1 \
        --max-steps "$MAX_STEPS" \
        --max-errors "$MAX_ERRORS" \
        --max-concurrency "$CONCURRENCY" \
        --seed 300 \
        --log-level ERROR \
        --save-to "${RUN_TAG}_${DOMAIN}"
done

python - "$SUMMARY" "$SGLANG_SMOKE" "$RUN_TAG" "$RESULT_DIR" "$BASE_URL" "$CONCURRENCY" "$MAX_STEPS" "$MAX_ERRORS" <<'PY'
import json
import sys
from pathlib import Path

summary_path, smoke_path, run_tag, result_dir, base_url, concurrency, max_steps, max_errors = sys.argv[1:]
result_dir = Path(result_dir)
domains = {}
all_runs = []
for domain in ("airline", "retail", "telecom"):
    path = result_dir / f"{run_tag}_{domain}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    simulations = payload["simulations"]
    successes = sum((run.get("reward_info") or {}).get("reward") == 1.0 for run in simulations)
    domains[domain] = {
        "result_file": str(path),
        "tasks": len(payload["tasks"]),
        "simulations": len(simulations),
        "successes": successes,
        "mean_reward": sum((run.get("reward_info") or {}).get("reward", 0.0) for run in simulations) / len(simulations),
    }
    all_runs.extend(simulations)
if sum(item["tasks"] for item in domains.values()) != 278:
    raise SystemExit(f"Expected 278 full-suite tasks, got {domains}")
payload = {
    "benchmark": "tau2-bench",
    "tau2_source_commit": "f8de30c298689cbe0117d76a378e7315a17e5bd8",
    "history_harness": "tau2 native llm_agent (standard full history)",
    "agent_model": "openai/qwen35-4b-local",
    "user_simulator_model": "openai/qwen35-4b-local",
    "agent_and_user_llm_args": {
        "api_base": base_url,
        "temperature": 0.0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "num_trials": 1,
    "seed": 300,
    "max_steps": int(max_steps),
    "max_errors": int(max_errors),
    "max_concurrency": int(concurrency),
    "sglang_smoke_file": smoke_path,
    "domains": domains,
    "full_suite_tasks": sum(item["tasks"] for item in domains.values()),
    "full_suite_simulations": len(all_runs),
    "full_suite_successes": sum((run.get("reward_info") or {}).get("reward") == 1.0 for run in all_runs),
    "full_suite_mean_reward": sum((run.get("reward_info") or {}).get("reward", 0.0) for run in all_runs) / len(all_runs),
}
Path(summary_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"tau2_summary={summary_path}")
PY
