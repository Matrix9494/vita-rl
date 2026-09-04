#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VITA_ROOT="${VITA_ROOT:-/workspace/projects/vitabench}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/Qwen3.5-4B}"
VITA_VENV="${VITA_VENV:-/workspace/venvs/vita}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:30000/v1/chat/completions}"

for path in "$VITA_ROOT" "$QWEN_MODEL" "$VITA_VENV"; do
  [[ -e "$path" ]] || { echo "Required external path not found: $path" >&2; exit 1; }
done

command -v curl >/dev/null || { echo "curl is required to check SGLang" >&2; exit 1; }
SGLANG_MODELS_URL="${SGLANG_BASE_URL%/v1/chat/completions}/v1/models"
curl --fail --silent --show-error "$SGLANG_MODELS_URL" >/dev/null || {
  echo "SGLang endpoint is unavailable: $SGLANG_MODELS_URL" >&2
  exit 1
}

exec "$SCRIPT_DIR/run_baseline.sh" \
  --domain delivery \
  --task-id 10711001 \
  --agent-llm "$QWEN_MODEL" \
  --user-llm gpt-4.1 \
  --evaluator-llm gpt-4.1 \
  --num-trials 1 \
  --max-steps 300 \
  --output smoke_qwen35_4b \
  --language english
