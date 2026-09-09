#!/usr/bin/env bash
# Fetch the pinned upstream BFCL harness and apply vita-rl's local Qwen adapter.
# This intentionally does not install Python dependencies; activate the desired
# venv and install BFCL separately when an evaluation will be run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GORILLA_ROOT="$REPO/external/gorilla"
BFCL_COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
PATCH_FILE="$REPO/patches/bfcl/qwen35_local_adapter.patch"

[[ -f "$PATCH_FILE" ]] || { echo "Missing adapter patch: $PATCH_FILE" >&2; exit 1; }

if [[ ! -e "$GORILLA_ROOT" ]]; then
    mkdir -p "$(dirname "$GORILLA_ROOT")"
    git clone https://github.com/ShishirPatil/gorilla.git "$GORILLA_ROOT"
fi

git -C "$GORILLA_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Expected a Git checkout at $GORILLA_ROOT" >&2
    exit 1
}

if [[ -n "$(git -C "$GORILLA_ROOT" status --porcelain)" ]]; then
    if git -C "$GORILLA_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE" 2>/dev/null; then
        echo "BFCL Qwen adapter is already applied."
        exit 0
    fi
    echo "Refusing to modify a dirty Gorilla checkout: $GORILLA_ROOT" >&2
    exit 1
fi

git -C "$GORILLA_ROOT" fetch --tags origin
git -C "$GORILLA_ROOT" checkout --detach "$BFCL_COMMIT"

if git -C "$GORILLA_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "BFCL Qwen adapter is already applied."
elif git -C "$GORILLA_ROOT" apply --unidiff-zero --check "$PATCH_FILE"; then
    git -C "$GORILLA_ROOT" apply --unidiff-zero "$PATCH_FILE"
    echo "Applied BFCL Qwen adapter."
else
    echo "Adapter patch does not apply to pinned BFCL checkout." >&2
    exit 1
fi

printf 'BFCL source ready at %s\n' "$GORILLA_ROOT/berkeley-function-call-leaderboard"
