#!/usr/bin/env bash
# Submit or fetch the single-task Vessl deterministic-user VitaBench smoke.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEY_FILE="${OPENROUTER_KEY_FILE:-$REPO/key}"
COMMIT="${VITA_EVAL_COMMIT:-$(git -C "$REPO" rev-parse HEAD)}"
RUN_ID="${VITA_EVAL_RUN_ID:-vessl-vitabench-delivery-deterministic-user-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_WORKTREE="/root/projects/vita-rl-vita-eval-${COMMIT:0:12}"
REMOTE_ROOT="/root/outputs/vita-rl/vitabench/${RUN_ID}"
LOCAL_PARENT="$REPO/outputs/vessl_vitabench"
REMOTE_RUNNER="$REMOTE_WORKTREE/scripts/vessl/run_vitabench_deterministic_user_smoke.sh"

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

# The secret traverses SSH stdin only. It is never passed as an argument,
# written remotely, added to git, or included in Vessl job metadata/logs.
remote_command=$(printf 'IFS= read -r OPENROUTER_API_KEY; export OPENROUTER_API_KEY; mkdir -p %q; nohup env VITA_EVAL_REPO=%q VITA_EVAL_OUTPUT_ROOT=%q VITA_EVAL_RUN_ID=%q %q > %q 2>&1 & pid=$!; unset OPENROUTER_API_KEY; printf "pid=%%s run_root=%%s\\n" "$pid" %q' \
    "$REMOTE_ROOT" "$REMOTE_WORKTREE" "$REMOTE_ROOT" "$RUN_ID" "$REMOTE_RUNNER" "$REMOTE_ROOT/driver.log" "$REMOTE_ROOT")
tr -d '\r\n' < "$KEY_FILE" | ssh vessl-vita "bash -lc $(printf '%q' "$remote_command")"
echo "Vessl deterministic-user smoke started. Poll driver.log remotely, then run: $0 --fetch"
