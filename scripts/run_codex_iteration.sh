#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/u/dz13/advancewars}"
ITERATION="${1:?usage: scripts/run_codex_iteration.sh <iteration> [game_count]}"
GAME_COUNT="${2:-200}"
MAX_STEPS="${MAX_STEPS:-250}"
MAX_TURNS="${MAX_TURNS:-40}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/scripted_selfplay}"
DATASET_DIR="${DATASET_DIR:-datasets/awbw_maps}"
CODEX_MODEL="${CODEX_MODEL:-}"

cd "$ROOT"
export PYTHONPATH=src

ITER_DIR="$OUTPUT_ROOT/iter_$(printf "%04d" "$ITERATION")"
mkdir -p "$ITER_DIR"

echo "[iteration $ITERATION] running $GAME_COUNT self-play games"
python3 examples/scripted_selfplay_iteration.py \
  --iteration "$ITERATION" \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --game-count "$GAME_COUNT" \
  --max-steps "$MAX_STEPS" \
  --max-turns "$MAX_TURNS" \
  --seed "$SEED" \
  2>&1 | tee "$ITER_DIR/rollout.log"

REQUEST="$ITER_DIR/codex_improvement_request.md"
if [[ ! -f "$REQUEST" ]]; then
  echo "missing Codex request: $REQUEST" >&2
  exit 1
fi

echo "[iteration $ITERATION] running Codex improver"
CODEX_ARGS=(
  -a never
)
if [[ -n "$CODEX_MODEL" ]]; then
  CODEX_ARGS+=(-m "$CODEX_MODEL")
fi
CODEX_ARGS+=(
  exec
  -C "$ROOT"
  --skip-git-repo-check
  -s danger-full-access
)
CODEX_ARGS+=(-)

codex "${CODEX_ARGS[@]}" < "$REQUEST" 2>&1 | tee "$ITER_DIR/codex_improvement.log"

echo "[iteration $ITERATION] done"
