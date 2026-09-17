#!/usr/bin/env bash
# Submit/fetch the resumable Vessl BFCL V4 Qwen3.5-4B comparison.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT="${BFCL_EVAL_COMMIT:-$(git -C "$REPO" rev-parse HEAD)}"
RUN_ID="${BFCL_EVAL_RUN_ID:-vessl-bfcl-v4-qwen35-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_WORKTREE="/root/projects/vita-rl-bfcl-eval-${COMMIT:0:12}"
REMOTE_ROOT="/root/outputs/vita-rl/bfcl/${RUN_ID}"
LOCAL_PARENT="$REPO/outputs/vessl_bfcl"
REMOTE_RUNNER="$REMOTE_WORKTREE/scripts/vessl/run_bfcl_v4_qwen35.sh"

if [[ "${1:-}" == "--fetch" ]]; then
    mkdir -p "$LOCAL_PARENT/$RUN_ID"
    scp -r "vessl-vita:${REMOTE_ROOT}/." "$LOCAL_PARENT/$RUN_ID/"
    echo "Fetched $REMOTE_ROOT to $LOCAL_PARENT/$RUN_ID"
    exit 0
fi

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
git -C "$worktree" submodule update --init --checkout external/gorilla
REMOTE

remote_command=$(printf 'mkdir -p %q; nohup env BFCL_EVAL_REPO=%q BFCL_EVAL_OUTPUT_ROOT=%q BFCL_EVAL_RUN_ID=%q bash %q > %q 2>&1 & pid=$!; printf "pid=%%s run_root=%%s\\n" "$pid" %q' \
    "$REMOTE_ROOT" "$REMOTE_WORKTREE" "$REMOTE_ROOT" "$RUN_ID" "$REMOTE_RUNNER" "$REMOTE_ROOT/driver.log" "$REMOTE_ROOT")
ssh vessl-vita "bash -lc $(printf '%q' "$remote_command")"
echo "Vessl BFCL evaluation started. Poll $REMOTE_ROOT/driver.log, then run: $0 --fetch"
