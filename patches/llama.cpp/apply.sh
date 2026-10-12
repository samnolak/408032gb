#!/usr/bin/env bash
# Apply our llama.cpp patches to a COPY of a pinned checkout (never in place, AGENTS.md "Upstream pins").
#   patches/llama.cpp/apply.sh <llama.cpp checkout at ec7630a or 79e2e74> <new directory>
# The copy is `git clone --shared` of the checkout (objects borrowed, the source is not touched), detached at the
# source's HEAD, with every patches/llama.cpp/<short sha>/*.patch applied by `git apply` to the working tree (not
# committed: HEAD stays the pin, so tools that check the pin still see it; `git -C <copy> diff` shows the patch).
# An existing <new directory> is reused if it is a copy at the same commit with exactly these patches applied
# (checked with `git apply --reverse --check` and `git diff` against the patch files); anything else is an error.
set -euo pipefail
SRC=$(cd "${1:?llama.cpp checkout}" && pwd)
DST=${2:?new directory for the patched copy}
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$SRC" rev-parse HEAD)
case "$SHA" in
  ec7630a640789c393694fb194f1bbbf0369fc62d) DIR=$HERE/ec7630a ;;
  79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6) DIR=$HERE/79e2e74 ;;
  *) echo "apply.sh: $SRC is at $SHA, not a pin with patches here" >&2; exit 1 ;;
esac
mapfile -t PATCHES < <(ls "$DIR"/*.patch)
[ "${#PATCHES[@]}" -gt 0 ] || { echo "apply.sh: no patches in $DIR" >&2; exit 1; }
patched_ok() {   # $1 = copy: at SHA, and its working-tree diff is exactly the patches
  [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" = "$SHA" ] || return 1
  for p in "${PATCHES[@]}"; do git -C "$1" apply --reverse --check "$p" 2>/dev/null || return 1; done
  local tmp; tmp=$(mktemp -d)
  git clone -q --shared --no-checkout "$1" "$tmp/c" && git -C "$tmp/c" -c advice.detachedHead=false checkout -q --detach "$SHA" \
    && for p in "${PATCHES[@]}"; do git -C "$tmp/c" apply "$p"; done
  local same=0
  [ "$(git -C "$1" diff)" = "$(git -C "$tmp/c" diff)" ] || same=1
  rm -rf "$tmp"
  return $same
}
if [ -e "$DST" ]; then
  if patched_ok "$DST"; then
    echo "apply.sh: reusing $DST ($SHA + ${#PATCHES[@]} patch(es))"
    exit 0
  fi
  echo "apply.sh: $DST exists and is not $SHA with exactly the patches of $DIR" >&2
  exit 1
fi
git clone -q --shared --no-checkout "$SRC" "$DST"
git -C "$DST" -c advice.detachedHead=false checkout -q --detach "$SHA"
for p in "${PATCHES[@]}"; do
  git -C "$DST" apply "$p"
  echo "apply.sh: applied $(basename "$p")"
done
git -C "$DST" diff --stat
