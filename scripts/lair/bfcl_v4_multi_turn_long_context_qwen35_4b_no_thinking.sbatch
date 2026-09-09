#!/usr/bin/env bash
# Official BFCL V4 multi_turn_long_context, with Qwen thinking disabled.
# This delegates to the validated base launcher so model/server/scorer settings
# cannot drift between the BFCL multi-turn comparisons.
#SBATCH --account=llmreason
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=bfcl_v4_mt_long_nt
#SBATCH --output=/u/dz13/vita-rl/logs/bfcl_v4_mt_long_nt-%j.out
#SBATCH --error=/u/dz13/vita-rl/logs/bfcl_v4_mt_long_nt-%j.err

set -euo pipefail

REPO=/u/dz13/vita-rl
JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_ROOT="$REPO/outputs/bfcl/bfcl_v4_multi_turn_long_context_qwen35_4b_no_thinking_${JOB_TAG}"
mkdir -p "$RUN_ROOT"

export BFCL_TEST_CATEGORY=multi_turn_long_context
export BFCL_QWEN_ENABLE_THINKING=false
export BFCL_TEMPERATURE=0.0
export BFCL_RUN_TAG="bfcl_v4_multi_turn_long_context_qwen35_4b_no_thinking_${JOB_TAG}"
export BFCL_RUN_ROOT="$RUN_ROOT"

exec bash "$REPO/scripts/lair/bfcl_v4_multi_turn_base_qwen35_4b.sbatch"
