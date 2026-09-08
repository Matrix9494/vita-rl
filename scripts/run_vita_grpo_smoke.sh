#!/usr/bin/env bash
# Compatibility entry point for existing Vessl jobs. New jobs use
# run_environment_grpo_smoke.sh with ENVIRONMENT=vita-mini or vitabench.
exec "$(dirname "$0")/run_environment_grpo_smoke.sh" "$@"
