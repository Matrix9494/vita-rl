#!/usr/bin/env bash
# Fetch the pinned tau2-bench source and apply vita-rl's diagnostic trace patch.
# Python dependencies remain the responsibility of the active evaluation venv.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAU2_ROOT="$REPO/external/tau2-bench"
TAU2_COMMIT=f8de30c298689cbe0117d76a378e7315a17e5bd8
PATCH_FILE="$REPO/patches/tau2/diagnostic_trace.patch"

[[ -f "$PATCH_FILE" ]] || { echo "Missing tau2 patch: $PATCH_FILE" >&2; exit 1; }

if [[ ! -e "$TAU2_ROOT" ]]; then
    mkdir -p "$(dirname "$TAU2_ROOT")"
    git clone https://github.com/sierra-research/tau2-bench.git "$TAU2_ROOT"
fi

git -C "$TAU2_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Expected a Git checkout at $TAU2_ROOT" >&2
    exit 1
}

if [[ -n "$(git -C "$TAU2_ROOT" status --porcelain)" ]]; then
    if git -C "$TAU2_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE"; then
        echo "tau2 diagnostic trace patch is already applied."
        exit 0
    fi
    echo "Refusing to modify a dirty tau2 checkout: $TAU2_ROOT" >&2
    exit 1
fi

git -C "$TAU2_ROOT" fetch --tags origin
git -C "$TAU2_ROOT" checkout --detach "$TAU2_COMMIT"

if git -C "$TAU2_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE"; then
    echo "tau2 diagnostic trace patch is already applied."
elif git -C "$TAU2_ROOT" apply --unidiff-zero --check "$PATCH_FILE"; then
    git -C "$TAU2_ROOT" apply --unidiff-zero "$PATCH_FILE"
    echo "Applied tau2 diagnostic trace patch."
else
    echo "Diagnostic trace patch does not apply to pinned tau2 checkout." >&2
    exit 1
fi

printf 'tau2 source ready at %s\n' "$TAU2_ROOT"
