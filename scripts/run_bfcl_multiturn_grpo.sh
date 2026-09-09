#!/usr/bin/env bash
# Full-parameter, CPU-offloaded GRPO for BFCL V4 multi_turn_base on Vessl.
set -euo pipefail
source /root/setup_env.sh

RUN_ID="${RUN_ID:-bfcl-multi-turn-base-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/root/checkpoints/vita-rl/$RUN_ID}"
REF_LOAD="${REF_LOAD:-$RUN_ROOT/reference_torch_dist}"
CKPT_SAVE="${CKPT_SAVE:-$RUN_ROOT/megatron}"
FINAL_HF_OUTPUT="${FINAL_HF_OUTPUT:-$RUN_ROOT/hf-final}"
BEST_HF_OUTPUT="${BEST_HF_OUTPUT:-$RUN_ROOT/hf-best-heldout}"
PROMPT_DATA="${PROMPT_DATA:-$RUN_ROOT/train-prompts.jsonl}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-$RUN_ROOT/eval-prompts.jsonl}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
EVAL_METRICS="${EVAL_METRICS:-$RUN_ROOT/bfcl-heldout-eval.jsonl}"
BEST_SELECTION="${BEST_SELECTION:-$RUN_ROOT/best-heldout.json}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
RESUME_FROM="${RESUME_FROM:-}"
VITA_PORT="${VITA_PORT:-9011}"
PROXY_PORT="${PROXY_PORT:-8801}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
BFCL_ROOT="${BFCL_SOURCE_ROOT:-$VITA_RL_ROOT/external/gorilla/berkeley-function-call-leaderboard}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
[[ -n "$MASTER_ADDR" && -d "$BFCL_ROOT" ]] || { echo "Missing address or BFCL source" >&2; exit 2; }

VITA_PID=""
PROXY_PID=""
cleanup() {
  [[ -n "$VITA_PID" ]] && kill "$VITA_PID" 2>/dev/null || true
  [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  "$VITA_RL_ROOT/scripts/resume_stack.sh" >>"$LOG_DIR/resume-stack.log" 2>&1 || true
}
trap cleanup EXIT

echo "[1/8] BFCL checkout and deterministic manifests"
bash "$VITA_RL_ROOT/scripts/setup_bfcl_harness.sh"
PYTHONPATH="$VITA_RL_ROOT/src:$BFCL_ROOT" "$VITA_VENV/bin/python" "$VITA_RL_ROOT/scripts/build_bfcl_grpo_manifests.py" --train-output "$PROMPT_DATA" --eval-output "$EVAL_PROMPT_DATA" --bfcl-source-root "$BFCL_ROOT" | tee "$LOG_DIR/manifest.log"

echo "[2/8] BFCL runtime dependency check"
PYTHONPATH="$VITA_RL_ROOT/src:$BFCL_ROOT" "$VITA_VENV/bin/python" -c 'import bfcl_eval; from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_checker; import overrides, openai, tenacity'

echo "[3/8] stop interactive inference and convert reference"
RUNTIME_DIR=/root/.vita_rl_runtime
if [[ -s "$RUNTIME_DIR/sglang.pid" ]]; then
  kill "$(<"$RUNTIME_DIR/sglang.pid")" 2>/dev/null || true
  rm -f "$RUNTIME_DIR/sglang.pid"
fi
if [[ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]]; then
  ( cd /root/Dressage/slime; source scripts/models/qwen3.5-4B.sh; PYTHONPATH=/root/Megatron-LM:/root/Dressage/slime python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$QWEN_MODEL" --save "$REF_LOAD" ) 2>&1 | tee "$LOG_DIR/convert-hf-to-torch-dist.log"
fi

echo "[4/8] start no-thinking Dressage proxy and BFCL runtime"
PYTHONPATH=/root/Dressage:/root/Dressage/slime:/root/Megatron-LM python3 -m dressage.proxy.server --sglang-router-url "http://$MASTER_ADDR:$SGLANG_PORT" --tokenizer-path "$QWEN_MODEL" --host 0.0.0.0 --port "$PROXY_PORT" --model-mask-type qwen3_5 --model-tool-call-type qwen3_5 --tool-call-parse-backend sglang_api --model-reasoning-type qwen3 --reasoning-parse-backend sglang_api --token-build-mode tito --tito-model qwen3_5 >"$LOG_DIR/dressage-proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null
PYTHONPATH="$VITA_RL_ROOT/src:$BFCL_ROOT" BFCL_SOURCE_ROOT="$BFCL_ROOT" "$VITA_VENV/bin/python" -m vita_rl.runtime_server --host 0.0.0.0 --port "$VITA_PORT" >"$LOG_DIR/vita-server.log" 2>&1 &
VITA_PID=$!
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:$VITA_PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:$VITA_PORT/health" >/dev/null

echo "[5/8] start one-H100 Ray"
export PYTHONPATH="/root/Megatron-LM:/root/Dressage:/root/Dressage/slime:$VITA_RL_ROOT/src:$BFCL_ROOT:${PYTHONPATH:-}"
export no_proxy="127.0.0.1,localhost,$MASTER_ADDR"
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus 1 --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265 >"$LOG_DIR/ray-head.log" 2>&1
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM:/root/Dressage:/root/Dressage/slime:$VITA_RL_ROOT/src:$BFCL_ROOT\",\"DRESSAGE_PROXY_URL\":\"http://$MASTER_ADDR:$PROXY_PORT\",\"ENVIRONMENT_RUNTIME_URL\":\"http://$MASTER_ADDR:$VITA_PORT\",\"DRESSAGE_REWARD_MODULES\":\"vita_rl.reward\",\"BFCL_SOURCE_ROOT\":\"$BFCL_ROOT\",\"BFCL_EVAL_METRICS_PATH\":\"$EVAL_METRICS\",\"no_proxy\":\"127.0.0.1,localhost,$MASTER_ADDR\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}"
LOAD_ARGS=(--ref-load "$REF_LOAD" --no-load-optim --no-load-rng)
[[ -n "$RESUME_FROM" ]] && LOAD_ARGS=(--load "$RESUME_FROM")

echo "[6/8] submit GRPO: $NUM_ROLLOUT updates, 4 prompts x 8 samples"
cd /root/Dressage/slime
source scripts/models/qwen3.5-4B.sh
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="$RUNTIME_ENV_JSON" -- python3 -m train \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate "${MODEL_ARGS[@]}" --hf-checkpoint "$QWEN_MODEL" "${LOAD_ARGS[@]}" --save "$CKPT_SAVE" --save-interval "$SAVE_INTERVAL" \
  --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync --eval-function-path vita_rl.bfcl_eval_rollout.generate_rollout --custom-generate-function-path vita_rl.dressage_adapter.generate --custom-rm-path dressage.reward.custom_rm.custom_rm --data-source-path dressage.rollout.data_source.DressageDataSource --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data --custom-eval-rollout-log-function-path vita_rl.bfcl_eval_logging.log_eval_rollout_data \
  --prompt-data "$PROMPT_DATA" --eval-prompt-data bfcl-heldout "$EVAL_PROMPT_DATA" --apply-chat-template --input-key prompt --label-key label --metadata-key metadata --eval-input-key prompt --eval-label-key label --eval-interval "$EVAL_INTERVAL" --n-samples-per-eval-prompt 1 --eval-temperature 0.0 --eval-top-k 1 --eval-max-response-len 4096 \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32 --rollout-max-response-len 4096 --rollout-temperature 0.8 --rollout-top-p 1.0 --rollout-shuffle --dataloader-type cyclic --seed 300 --rollout-seed 300 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 --optimizer adam --lr 1e-6 --lr-decay-style constant --lr-warmup-iters "$LR_WARMUP_ITERS" --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0 --use-precision-aware-optimizer --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu 4096 --log-probs-chunk-size 512 --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.32 --sglang-router-port "$SGLANG_PORT" --router-policy consistent_hashing --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash 2>&1 | tee "$LOG_DIR/grpo.log"

echo "[7/8] convert final and selected held-out checkpoint"
native_final="$(find "$CKPT_SAVE" -type d -name 'iter_*' | sort | tail -n 1)"
[[ -n "$native_final" ]] || { echo "No native final checkpoint" >&2; exit 1; }
PYTHONPATH=/root/Megatron-LM:/root/Dressage/slime python3 tools/convert_torch_dist_to_hf.py --input-dir "$native_final" --output-dir "$FINAL_HF_OUTPUT" --origin-hf-dir "$QWEN_MODEL" 2>&1 | tee "$LOG_DIR/convert-final-to-hf.log"
PYTHONPATH="$VITA_RL_ROOT/src" "$VITA_VENV/bin/python" "$VITA_RL_ROOT/scripts/select_best_bfcl_checkpoint.py" --metrics "$EVAL_METRICS" --checkpoint-root "$CKPT_SAVE" --reference-checkpoint "$REF_LOAD" --output "$BEST_SELECTION" | tee "$LOG_DIR/select-best-heldout.log"
best_native="$("$VITA_VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$BEST_SELECTION")"
PYTHONPATH=/root/Megatron-LM:/root/Dressage/slime python3 tools/convert_torch_dist_to_hf.py --input-dir "$best_native" --output-dir "$BEST_HF_OUTPUT" --origin-hf-dir "$QWEN_MODEL" 2>&1 | tee "$LOG_DIR/convert-best-to-hf.log"
echo "[8/8] complete; primary artifact: $BEST_HF_OUTPUT"
