#!/usr/bin/env bash
# Submit or fetch a Vessl GPT-5.6 Terra VitaBench upper-bound evaluation.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEY_FILE="${OPENROUTER_KEY_FILE:-$REPO/key}"
COMMIT="${VITA_EVAL_COMMIT:-$(git -C "$REPO" rev-parse HEAD)}"
TASK_SET="${VITA_TASK_SET:-delivery}"
TASK_LANGUAGE="${VITA_TASK_LANGUAGE:-english}"
TASK_COUNT="${VITA_TASK_COUNT:-100}"
CONCURRENCY="${VITA_MAX_CONCURRENCY:-20}"
TASK_IDS_OVERRIDE="${VITA_TASK_IDS:-}"
RECOVERY_OF="${VITA_RECOVERY_OF:-}"
USER_IMPLEMENTATION="${VITA_USER_IMPLEMENTATION:-user_simulator}"
USER_LLM="${VITA_USER_LLM:-gpt-4.1}"
RUN_ID="${VITA_EVAL_RUN_ID:-vessl-vitabench-${TASK_SET}-terra-upper-bound-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_WORKTREE="/root/projects/vita-rl-vita-eval-${COMMIT:0:12}"
REMOTE_ROOT="/root/outputs/vita-rl/vitabench/${RUN_ID}"
LOCAL_PARENT="$REPO/outputs/vessl_vitabench"
REMOTE_RUNNER="$REMOTE_WORKTREE/scripts/vessl/run_vitabench_terra_upper_bound.sh"

if [[ "${1:-}" == "--fetch" ]]; then
    mkdir -p "$LOCAL_PARENT"
    scp -r "vessl-vita:${REMOTE_ROOT}" "$LOCAL_PARENT/"
    echo "Fetched $REMOTE_ROOT to $LOCAL_PARENT/$RUN_ID"
    exit 0
fi

[[ -s "$KEY_FILE" ]] || { echo "Missing or empty key file: $KEY_FILE" >&2; exit 2; }
key_mode="$(stat -c '%a' "$KEY_FILE")"
(( (8#$key_mode & 077) == 0 )) || { echo "Refusing insecure key file mode: $key_mode" >&2; exit 2; }
git -C "$REPO" diff --check

ssh vessl-vita bash -s -- "$COMMIT" "$REMOTE_WORKTREE" <<'REMOTE'
set -euo pipefail
commit="$1"
worktree="$2"
git -C /root/projects/vita-rl fetch --quiet origin main
git -C /root/projects/vita-rl cat-file -e "${commit}^{commit}"
if [[ -e "$worktree" ]]; then
    [[ "$(git -C "$worktree" rev-parse HEAD)" == "$commit" ]] || {
        echo "Existing Vessl worktree has the wrong commit: $worktree" >&2
        exit 2
    }
else
    git -C /root/projects/vita-rl worktree add --detach "$worktree" "$commit"
fi
REMOTE

# Transfer the raw secret only via stdin. It is never a remote argument, file,
# Vessl metadata value, or log entry.
remote_command=$(printf 'IFS= read -r OPENROUTER_API_KEY; export OPENROUTER_API_KEY; mkdir -p %q; nohup env VITA_EVAL_REPO=%q VITA_EVAL_OUTPUT_ROOT=%q VITA_EVAL_RUN_ID=%q VITA_TASK_SET=%q VITA_TASK_LANGUAGE=%q VITA_TASK_COUNT=%q VITA_MAX_CONCURRENCY=%q VITA_TASK_IDS=%q VITA_RECOVERY_OF=%q VITA_USER_IMPLEMENTATION=%q VITA_USER_LLM=%q %q > %q 2>&1 & pid=$!; unset OPENROUTER_API_KEY; printf "pid=%%s run_root=%%s\\n" "$pid" %q' \
    "$REMOTE_ROOT" "$REMOTE_WORKTREE" "$REMOTE_ROOT" "$RUN_ID" "$TASK_SET" "$TASK_LANGUAGE" "$TASK_COUNT" "$CONCURRENCY" "$TASK_IDS_OVERRIDE" "$RECOVERY_OF" "$USER_IMPLEMENTATION" "$USER_LLM" "$REMOTE_RUNNER" "$REMOTE_ROOT/driver.log" "$REMOTE_ROOT")
tr -d '\r\n' < "$KEY_FILE" | ssh vessl-vita "bash -lc $(printf '%q' "$remote_command")"
echo "Vessl Terra evaluation started. Poll driver.log remotely, then run: $0 --fetch"
