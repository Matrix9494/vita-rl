#!/usr/bin/env bash
# Apply the portable Dressage fixes required by the pinned Vessl checkout.
# Each patch is intentionally idempotent.
set -euo pipefail

VITA_RL_ROOT="${VITA_RL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DRESSAGE_ROOT="${DRESSAGE_ROOT:-/root/Dressage}"
PATCH_FILES=(
  "$VITA_RL_ROOT/patches/dressage/preserve_response_tokens.patch"
  "$VITA_RL_ROOT/patches/dressage/qwen35_disable_thinking.patch"
)

[[ -d "$DRESSAGE_ROOT/.git" ]] || {
  echo "Dressage checkout is missing" >&2
  exit 2
}

for patch_file in "${PATCH_FILES[@]}"; do
  [[ -f "$patch_file" ]] || {
    echo "Dressage patch is missing: $patch_file" >&2
    exit 2
  }
  patch_name="$(basename "$patch_file")"
  if git -C "$DRESSAGE_ROOT" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "Dressage patch is already applied: $patch_name"
  elif git -C "$DRESSAGE_ROOT" apply --check "$patch_file"; then
    git -C "$DRESSAGE_ROOT" apply "$patch_file"
    echo "Applied Dressage patch: $patch_name"
  else
    echo "Dressage patch does not apply: $patch_name" >&2
    exit 1
  fi
done
