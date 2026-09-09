#!/usr/bin/env bash
# Apply the portable response-preserving truncation fix to the pinned Dressage
# checkout used by Vessl.  The patch is intentionally idempotent.
set -euo pipefail

VITA_RL_ROOT="${VITA_RL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DRESSAGE_ROOT="${DRESSAGE_ROOT:-/root/Dressage}"
PATCH_FILE="$VITA_RL_ROOT/patches/dressage/preserve_response_tokens.patch"

[[ -d "$DRESSAGE_ROOT/.git" && -f "$PATCH_FILE" ]] || {
  echo "Dressage checkout or response-preserving patch is missing" >&2
  exit 2
}

if git -C "$DRESSAGE_ROOT" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "Dressage response-preserving token cap patch is already applied."
elif git -C "$DRESSAGE_ROOT" apply --check "$PATCH_FILE"; then
  git -C "$DRESSAGE_ROOT" apply "$PATCH_FILE"
  echo "Applied Dressage response-preserving token cap patch."
else
  echo "Dressage response-preserving token cap patch does not apply." >&2
  exit 1
fi
