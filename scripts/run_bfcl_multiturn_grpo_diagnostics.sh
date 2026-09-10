#!/usr/bin/env bash
# 100-update Base-only characterization run; the generic launcher does all training.
set -euo pipefail
VITA_RL_ROOT="${VITA_RL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:-bfcl-multi-turn-base-grpo100-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/root/checkpoints/vita-rl/$RUN_ID}"
export VITA_RL_ROOT RUN_ID RUN_ROOT
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export TRAIN_GROUPS="${TRAIN_GROUPS:-$RUN_ROOT/bfcl-train-groups.jsonl}"

bash "$VITA_RL_ROOT/scripts/run_bfcl_multiturn_grpo.sh"
"${VITA_VENV:-/root/venvs/vita}/bin/python" "$VITA_RL_ROOT/scripts/analyze_bfcl_grpo_diagnostics.py" --run-root "$RUN_ROOT"
