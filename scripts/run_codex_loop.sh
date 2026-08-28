#!/usr/bin/env bash
set -euo pipefail

START_ITERATION="${1:-0}"
ITERATIONS="${2:-1}"
GAME_COUNT="${3:-200}"
ROOT="${ROOT:-/u/dz13/advancewars}"

cd "$ROOT"

for ((offset = 0; offset < ITERATIONS; offset++)); do
  iteration=$((START_ITERATION + offset))
  scripts/run_codex_iteration.sh "$iteration" "$GAME_COUNT"
done
