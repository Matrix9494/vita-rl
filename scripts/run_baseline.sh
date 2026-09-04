#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_baseline.sh [options]

Options:
  --domain NAME          VitaBench domain (default: delivery)
  --task-id ID           Run one task; omit to use VitaBench's task selection
  --output NAME           VitaBench save-to name (default: vita_rl_baseline)
  --agent-llm NAME        Agent model name (default: /workspace/models/Qwen3.5-4B)
  --user-llm NAME         User model name (default: gpt-4.1)
  --evaluator-llm NAME    Evaluator model name (default: gpt-4.1)
  --max-steps N           Maximum simulation steps (default: 300)
  --num-trials N          Number of trials (default: 1)
  --language NAME         chinese or english (default: english)
  --help                  Show this help

Runtime overrides are VITA_ROOT, VITA_VENV, QWEN_MODEL, SGLANG_BASE_URL,
and VITA_MODEL_CONFIG_PATH. OPENROUTER_API_KEY must be injected externally
unless VITA_MODEL_CONFIG_PATH points to a complete external configuration.
EOF
}

DOMAIN="delivery"
TASK_ID=""
OUTPUT="vita_rl_baseline"
AGENT_LLM="${QWEN_MODEL:-/workspace/models/Qwen3.5-4B}"
USER_LLM="gpt-4.1"
EVALUATOR_LLM="gpt-4.1"
MAX_STEPS="300"
NUM_TRIALS="1"
LANGUAGE="english"

while (($#)); do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --agent-llm) AGENT_LLM="$2"; shift 2 ;;
    --user-llm) USER_LLM="$2"; shift 2 ;;
    --evaluator-llm) EVALUATOR_LLM="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

VITA_ROOT="${VITA_ROOT:-/workspace/projects/vitabench}"
VITA_VENV="${VITA_VENV:-/workspace/venvs/vita}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:30000/v1/chat/completions}"

[[ -d "$VITA_ROOT" ]] || { echo "VitaBench checkout not found: $VITA_ROOT" >&2; exit 1; }
[[ -x "$VITA_VENV/bin/vita" ]] || { echo "VitaBench CLI not found: $VITA_VENV/bin/vita" >&2; exit 1; }
[[ -d "$AGENT_LLM" ]] || { echo "Local agent model not found: $AGENT_LLM" >&2; exit 1; }

if [[ -n "${VITA_MODEL_CONFIG_PATH:-}" ]]; then
  MODEL_CONFIG="$VITA_MODEL_CONFIG_PATH"
  [[ -f "$MODEL_CONFIG" ]] || { echo "Model config not found: $MODEL_CONFIG" >&2; exit 1; }
  CLEANUP_CONFIG=0
else
  [[ -n "${OPENROUTER_API_KEY:-}" ]] || {
    echo "OPENROUTER_API_KEY is not set; inject GiveMeANode secret 'openrouter' or set VITA_MODEL_CONFIG_PATH." >&2
    exit 1
  }
  MODEL_CONFIG="$(mktemp "${TMPDIR:-/tmp}/vita-rl-models.XXXXXX.yaml")"
  CLEANUP_CONFIG=1
  trap 'rm -f "$MODEL_CONFIG"' EXIT
  python - "$MODEL_CONFIG" "$AGENT_LLM" "$SGLANG_BASE_URL" "$OPENROUTER_API_KEY" <<'PY'
import json
import sys
from pathlib import Path

path, agent_model, agent_endpoint, api_key = sys.argv[1:]
Path(path).write_text(
    f"""default:
  base_url: https://openrouter.ai/api/v1/chat/completions
  temperature: 0.0
  headers:
    Authorization: Bearer {api_key}
    Content-Type: application/json
models:
  - name: gpt-4.1
    max_tokens: 4096
  - name: {json.dumps(agent_model)}
    base_url: {json.dumps(agent_endpoint)}
    chat_template_kwargs:
      enable_thinking: false
    max_tokens: 4096
"""
)
PY
fi

if [[ "$CLEANUP_CONFIG" -eq 1 ]]; then
  export VITA_MODEL_CONFIG_PATH="$MODEL_CONFIG"
fi

cd "$VITA_ROOT"
args=(
  run
  --domain "$DOMAIN"
  --agent llm_agent
  --agent-llm "$AGENT_LLM"
  --user user_simulator
  --user-llm "$USER_LLM"
  --evaluator-llm "$EVALUATOR_LLM"
  --num-trials "$NUM_TRIALS"
  --max-steps "$MAX_STEPS"
  --max-concurrency 1
  --language "$LANGUAGE"
  --save-to "$OUTPUT"
)
if [[ -n "$TASK_ID" ]]; then
  args+=(--task-ids "$TASK_ID")
fi

"$VITA_VENV/bin/vita" "${args[@]}"
