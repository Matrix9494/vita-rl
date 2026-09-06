#!/usr/bin/env bash
# Restore only ephemeral development/baseline runtime state after a VESSL resume.
set -euo pipefail

source /root/setup_env.sh

RUNTIME_DIR=/root/.vita_rl_runtime
LOG_DIR=/root/logs/vita-rl
SGLANG_PID_FILE="${RUNTIME_DIR}/sglang.pid"
SGLANG_LOG="${LOG_DIR}/sglang.log"
# The 4B hybrid model's default server allocation is overly aggressive on a
# single H100 and has caused its detokenizer subprocess to be killed during
# CUDA-graph capture. Keep inference headroom by default; callers that need a
# larger KV cache can explicitly override this.
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.32}"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR"
echo "=== vita-rl resume ==="

check_path() {
    [[ -e "$1" ]] || { echo "[FAIL] missing: $1" >&2; exit 1; }
}

check_path /root/Dressage
check_path /root/Dressage/slime
check_path /root/Megatron-LM
check_path "$VITA_RL_ROOT"
check_path "$VITA_ROOT"
check_path "$VITA_VENV"
check_path "$QWEN_MODEL"
echo "[OK] persistent paths"

python - <<'PY'
import torch
import sglang
import dressage
import vita_rl

assert torch.cuda.is_available()
print("[OK] GPU:", torch.cuda.get_device_name(0))
print("[OK] torch:", torch.__version__)
print("[OK] sglang:", getattr(sglang, "__version__", "installed"))
PY

SGLANG_MODELS_URL="${SGLANG_BASE_URL%/v1/chat/completions}/v1/models"
if curl -fsS "$SGLANG_MODELS_URL" >/dev/null 2>&1; then
    echo "[OK] SGLang already running"
else
    echo "[START] SGLang"
    if [[ -f "$SGLANG_PID_FILE" ]]; then
        OLD_PID="$(<"$SGLANG_PID_FILE")"
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
            rm -f "$SGLANG_PID_FILE"
        fi
    fi

    nohup python -m sglang.launch_server \
        --model-path "$QWEN_MODEL" \
        --host 127.0.0.1 \
        --port 30000 \
        --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
        >"$SGLANG_LOG" 2>&1 &
    echo $! > "$SGLANG_PID_FILE"

    for _ in $(seq 1 180); do
        if curl -fsS "$SGLANG_MODELS_URL" >/dev/null 2>&1; then
            echo "[OK] SGLang ready"
            break
        fi
        if ! kill -0 "$(<"$SGLANG_PID_FILE")" 2>/dev/null; then
            echo "[FAIL] SGLang exited; see $SGLANG_LOG" >&2
            exit 1
        fi
        sleep 2
    done

    curl -fsS "$SGLANG_MODELS_URL" >/dev/null || {
        echo "[FAIL] SGLang startup timeout; see $SGLANG_LOG" >&2
        exit 1
    }
fi

"$VITA_VENV/bin/vita" --help >/dev/null
echo "[OK] VitaBench"

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[OK] OpenRouter key loaded"
else
    echo "[WARN] OPENROUTER_API_KEY not loaded"
fi

echo
echo "=== stack ready ==="
echo "SGLang: $SGLANG_BASE_URL"
echo "vita-rl: $VITA_RL_ROOT"
echo "VitaBench: $VITA_ROOT"
echo "SGLang log: $SGLANG_LOG"
