#!/usr/bin/env bash
# Rollout-only VitaBench -> Dressage smoke test; deliberately does not run GRPO.
set -euo pipefail
source /root/setup_env.sh
VITA_PORT=${VITA_PORT:-9010}; PROXY_PORT=${PROXY_PORT:-8800}
VITA_URL="http://127.0.0.1:${VITA_PORT}"; DRESSAGE_PROXY_URL=${DRESSAGE_PROXY_URL:-"http://127.0.0.1:${PROXY_PORT}"}
LOG_DIR=/root/logs/vita-rl; RUNTIME_DIR=/root/.vita_rl_runtime; mkdir -p "$LOG_DIR" "$RUNTIME_DIR"
VITA_PID="$RUNTIME_DIR/vita-server.pid"; PROXY_PID="$RUNTIME_DIR/dressage-proxy.pid"; started_vita=0; started_proxy=0
OPENROUTER_PID=""; OPENROUTER_PORT_FILE=""; VITA_MODEL_CONFIG=""
cleanup() {
  if (( started_vita )); then kill "$(<"$VITA_PID")" 2>/dev/null || true; rm -f "$VITA_PID"; fi
  if (( started_proxy )); then kill "$(<"$PROXY_PID")" 2>/dev/null || true; rm -f "$PROXY_PID"; fi
  [[ -n "$OPENROUTER_PID" ]] && kill "$OPENROUTER_PID" 2>/dev/null || true
  [[ -n "$OPENROUTER_PORT_FILE" ]] && rm -f "$OPENROUTER_PORT_FILE"
  [[ -n "$VITA_MODEL_CONFIG" ]] && rm -f "$VITA_MODEL_CONFIG"
}; trap cleanup EXIT
"$VITA_RL_ROOT/scripts/resume_stack.sh"
if ! curl -fsS "$DRESSAGE_PROXY_URL/health" >/dev/null 2>&1; then
  router="${SGLANG_BASE_URL%%/v1/*}"
  PYTHONPATH="/root/Dressage:/root/Dressage/slime:/root/Megatron-LM" nohup python3 -m dressage.proxy.server --sglang-router-url "$router" --tokenizer-path "$QWEN_MODEL" --host 127.0.0.1 --port "$PROXY_PORT" --model-mask-type qwen3_5 --model-tool-call-type qwen3_5 --model-reasoning-type qwen3 --tool-call-parse-backend sglang_api --reasoning-parse-backend sglang_api --token-build-mode tito --tito-model qwen3_5 >"$LOG_DIR/dressage-proxy.log" 2>&1 & echo $! >"$PROXY_PID"; started_proxy=1
fi
for _ in $(seq 1 60); do curl -fsS "$DRESSAGE_PROXY_URL/health" >/dev/null 2>&1 && break; sleep 1; done; curl -fsS "$DRESSAGE_PROXY_URL/health" >/dev/null
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 1; }
OPENROUTER_PORT_FILE="$(mktemp /tmp/vita-rl-openrouter.XXXXXX)"
"$VITA_VENV/bin/python" "$VITA_RL_ROOT/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" >/dev/null 2>&1 & OPENROUTER_PID=$!
for _ in $(seq 1 100); do [[ -s "$OPENROUTER_PORT_FILE" ]] && break; sleep 0.1; done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter proxy failed to start" >&2; exit 1; }
VITA_MODEL_CONFIG="$(mktemp /tmp/vita-rl-models.XXXXXX.yaml)"
"$VITA_VENV/bin/python" - "$VITA_MODEL_CONFIG" "$(<"$OPENROUTER_PORT_FILE")" <<'PY'
import json, sys
from pathlib import Path
path, port = sys.argv[1:]
Path(path).write_text("default:\n  base_url: " + json.dumps(f"http://127.0.0.1:{port}/v1/chat/completions") + "\n  headers:\n    Authorization: Bearer vita-rl-local-proxy\n    Content-Type: application/json\nmodels:\n  - name: gpt-4.1\n    max_tokens: 4096\n")
PY
if ! curl -fsS "$VITA_URL/health" >/dev/null 2>&1; then PYTHONPATH="$VITA_RL_ROOT/src:$VITA_ROOT/src" VITA_MODEL_CONFIG_PATH="$VITA_MODEL_CONFIG" nohup env -u OPENROUTER_API_KEY "$VITA_VENV/bin/python" -m vita_rl.vita_server --port "$VITA_PORT" >"$LOG_DIR/vita-server.log" 2>&1 & echo $! >"$VITA_PID"; started_vita=1; fi
for _ in $(seq 1 30); do curl -fsS "$VITA_URL/health" >/dev/null 2>&1 && break; sleep 1; done; curl -fsS "$VITA_URL/health" >/dev/null
export DRESSAGE_PROXY_URL VITA_RUNTIME_URL="$VITA_URL"
PYTHONPATH="$VITA_RL_ROOT/src:/root/Dressage:/root/Dressage/slime:/root/Megatron-LM" python3 - <<'PY'
import asyncio
from types import SimpleNamespace
from slime.utils.types import Sample
from vita_rl.dressage_adapter import generate
s=Sample(index=0,prompt="Vita delivery smoke",metadata={"vita_domain":"delivery","vita_task_id":"10711001","vita_language":"chinese","vita_agent_model":"proxy-model"})
segments=asyncio.run(generate(SimpleNamespace(rollout_max_response_len=2048,reward_key=None),s,{"temperature":0.0,"max_new_tokens":2048}))
last=segments[-1]
assert "vita_reward" in last.metadata and sum(last.loss_mask or []) > 0
print({"task_id":last.metadata["vita_task_id"],"reward":last.metadata["vita_reward"],"proxy_turns":last.metadata["vita_proxy_turns"],"segments":len(segments),"trainable_loss_tokens":sum(last.loss_mask or [])})
PY
