#!/usr/bin/env bash
# One real VitaBench -> Dressage -> Slime GRPO iteration on one H100.
#
# The script produces both the native Megatron checkpoint used by Slime and a
# standalone Hugging Face-format checkpoint.  Set HF_REPO_ID to upload the
# latter after authenticating with `hf auth login` in the VESSL workspace.
set -euo pipefail

source /root/setup_env.sh

RUN_ID="${RUN_ID:-vita-grpo-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/root/checkpoints/vita-rl/${RUN_ID}}"
REF_LOAD="${REF_LOAD:-${RUN_ROOT}/reference_torch_dist}"
CKPT_SAVE="${CKPT_SAVE:-${RUN_ROOT}/megatron}"
HF_OUTPUT="${HF_OUTPUT:-${RUN_ROOT}/hf}"
PROMPT_DATA="${PROMPT_DATA:-${RUN_ROOT}/prompts.jsonl}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
VITA_PORT="${VITA_PORT:-9010}"
PROXY_PORT="${PROXY_PORT:-8800}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}') }"
MASTER_ADDR="${MASTER_ADDR// /}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"
[[ -n "$MASTER_ADDR" ]] || { echo "Unable to resolve MASTER_ADDR" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required for VitaBench user/evaluator calls" >&2; exit 2; }

RUNTIME_DIR=/root/.vita_rl_runtime
mkdir -p "$RUNTIME_DIR"
VITA_PID=""; PROXY_PID=""; OPENROUTER_PID=""; OPENROUTER_PORT_FILE=""; VITA_MODEL_CONFIG=""

cleanup() {
  [[ -n "$VITA_PID" ]] && kill "$VITA_PID" 2>/dev/null || true
  [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2>/dev/null || true
  [[ -n "$OPENROUTER_PID" ]] && kill "$OPENROUTER_PID" 2>/dev/null || true
  [[ -n "$OPENROUTER_PORT_FILE" ]] && rm -f "$OPENROUTER_PORT_FILE"
  [[ -n "$VITA_MODEL_CONFIG" ]] && rm -f "$VITA_MODEL_CONFIG"
  ray stop --force >/dev/null 2>&1 || true
  # Return the interactive VESSL workspace to its normal inference-ready state.
  "$VITA_RL_ROOT/scripts/resume_stack.sh" >>"$LOG_DIR/resume-stack.log" 2>&1 || true
}
trap cleanup EXIT

echo "[1/7] stopping the interactive SGLang server before model conversion"
SGLANG_PID_FILE="$RUNTIME_DIR/sglang.pid"
if [[ -s "$SGLANG_PID_FILE" ]]; then
  old_sglang_pid="$(<"$SGLANG_PID_FILE")"
  if kill -0 "$old_sglang_pid" 2>/dev/null; then kill "$old_sglang_pid"; fi
  rm -f "$SGLANG_PID_FILE"
fi
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1 || break
  sleep 1
done

echo "[2/7] converting the base model to Slime/Megatron torch_dist"
if [[ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]]; then
  (
    cd /root/Dressage/slime
    source scripts/models/qwen3.5-4B.sh
    PYTHONPATH=/root/Megatron-LM:/root/Dressage/slime python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$QWEN_MODEL" \
      --save "$REF_LOAD"
  ) 2>&1 | tee "$LOG_DIR/convert-hf-to-torch-dist.log"
fi

echo "[3/7] writing the Vita prompt group"
cat >"$PROMPT_DATA" <<JSONL
{"prompt":"Vita delivery GRPO smoke", "label":"", "metadata":{"vita_domain":"delivery","vita_task_id":"10711001","vita_language":"chinese","vita_agent_model":"proxy-model","vita_max_steps":20,"reward_fn":"vita_smoke_dense"}}
JSONL

echo "[4/7] starting proxy, Vita runtime, and the local OpenRouter relay"
router="http://${MASTER_ADDR}:${SGLANG_PORT}"
PYTHONPATH="/root/Dressage:/root/Dressage/slime:/root/Megatron-LM" \
  python3 -m dressage.proxy.server \
  --sglang-router-url "$router" --tokenizer-path "$QWEN_MODEL" \
  --host 0.0.0.0 --port "$PROXY_PORT" \
  --model-mask-type qwen3_5 --model-tool-call-type qwen3_5 \
  --tool-call-parse-backend sglang_api --model-reasoning-type qwen3 \
  --reasoning-parse-backend sglang_api --token-build-mode tito --tito-model qwen3_5 \
  >"$LOG_DIR/dressage-proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null

OPENROUTER_PORT_FILE="$(mktemp /tmp/vita-rl-openrouter.XXXXXX)"
"$VITA_VENV/bin/python" "$VITA_RL_ROOT/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" >"$LOG_DIR/openrouter-proxy.log" 2>&1 &
OPENROUTER_PID=$!
for _ in $(seq 1 100); do [[ -s "$OPENROUTER_PORT_FILE" ]] && break; sleep 0.1; done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter relay did not start" >&2; exit 1; }
VITA_MODEL_CONFIG="$(mktemp /tmp/vita-rl-models.XXXXXX.yaml)"
"$VITA_VENV/bin/python" - "$VITA_MODEL_CONFIG" "$(<"$OPENROUTER_PORT_FILE")" <<'PY'
import json
import sys
from pathlib import Path
path, port = sys.argv[1:]
Path(path).write_text(
    "default:\n"
    f"  base_url: {json.dumps(f'http://127.0.0.1:{port}/v1/chat/completions')}\n"
    "  headers:\n    Authorization: Bearer vita-rl-local-proxy\n    Content-Type: application/json\n"
    "models:\n  - name: gpt-4.1\n    max_tokens: 4096\n"
)
PY
PYTHONPATH="$VITA_RL_ROOT/src:$VITA_ROOT/src" VITA_MODEL_CONFIG_PATH="$VITA_MODEL_CONFIG" \
  env -u OPENROUTER_API_KEY "$VITA_VENV/bin/python" -m vita_rl.vita_server --host 0.0.0.0 --port "$VITA_PORT" \
  >"$LOG_DIR/vita-server.log" 2>&1 &
VITA_PID=$!
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${VITA_PORT}/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:${VITA_PORT}/health" >/dev/null

echo "[5/7] starting one-GPU Ray and submitting one GRPO rollout/update iteration"
export PYTHONPATH="/root/Megatron-LM:/root/Dressage:/root/Dressage/slime:$VITA_RL_ROOT/src:${PYTHONPATH:-}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus 1 --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  >"$LOG_DIR/ray-head.log" 2>&1

RUNTIME_ENV_JSON="$(cat <<JSON
{"env_vars":{"PYTHONPATH":"/root/Megatron-LM:/root/Dressage:/root/Dressage/slime:$VITA_RL_ROOT/src","DRESSAGE_PROXY_URL":"http://${MASTER_ADDR}:${PROXY_PORT}","VITA_RUNTIME_URL":"http://${MASTER_ADDR}:${VITA_PORT}","DRESSAGE_REWARD_MODULES":"vita_rl.reward","no_proxy":"127.0.0.1,localhost,${MASTER_ADDR}","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}
JSON
)"

cd /root/Dressage/slime
source scripts/models/qwen3.5-4B.sh
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="$RUNTIME_ENV_JSON" -- \
  python3 -m train \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$QWEN_MODEL" --ref-load "$REF_LOAD" --save "$CKPT_SAVE" --save-interval 1 --no-save-optim --no-load-optim \
  --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync \
  --custom-generate-function-path vita_rl.dressage_adapter.generate \
  --custom-rm-path dressage.reward.custom_rm.custom_rm \
  --data-source-path dressage.rollout.data_source.DressageDataSource \
  --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process \
  --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data \
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata \
  --num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 4 --rollout-max-response-len 1024 --rollout-temperature 1.0 --global-batch-size 4 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 4096 --log-probs-chunk-size 512 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.32 --sglang-router-port "$SGLANG_PORT" --router-policy consistent_hashing \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \
  2>&1 | tee "$LOG_DIR/grpo.log"

echo "[6/7] converting the saved Slime checkpoint back to Hugging Face format"
native_checkpoint="$(find "$CKPT_SAVE" -type d -name 'iter_*' | sort | tail -n 1)"
[[ -n "$native_checkpoint" ]] || { echo "No Slime checkpoint was written" >&2; exit 1; }
PYTHONPATH=/root/Megatron-LM:/root/Dressage/slime python3 tools/convert_torch_dist_to_hf.py \
  --input-dir "$native_checkpoint" --output-dir "$HF_OUTPUT" --origin-hf-dir "$QWEN_MODEL" \
  2>&1 | tee "$LOG_DIR/convert-torch-dist-to-hf.log"
[[ -f "$HF_OUTPUT/config.json" ]] || { echo "Hugging Face checkpoint conversion did not produce config.json" >&2; exit 1; }

echo "[7/7] validating the Hugging Face checkpoint"
HF_OUTPUT="$HF_OUTPUT" python3 - <<'PY'
import os
from transformers import AutoConfig
path = os.environ["HF_OUTPUT"]
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
print({"hf_checkpoint": path, "model_type": config.model_type})
PY

if [[ -n "${HF_REPO_ID:-}" ]]; then
  echo "Uploading ${HF_OUTPUT} to ${HF_REPO_ID}"
  hf upload "$HF_REPO_ID" "$HF_OUTPUT" .
else
  echo "HF_REPO_ID is unset; retained local Hugging Face checkpoint at ${HF_OUTPUT}"
fi
