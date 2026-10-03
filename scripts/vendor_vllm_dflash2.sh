#!/usr/bin/env bash
# Vendor the DFlash2 reference implementation from vLLM (Apache-2.0), byte-exact, at one pinned commit (ADR-011).
# GitHub is reachable from the agent sandbox, so this runs anywhere; git itself verifies the blob ids.
#   scripts/vendor_vllm_dflash2.sh [commit]
set -euo pipefail
PIN=${1:-bc21cba9673cfc2256a2726b4bc32f062237cd35}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=$ROOT/third_party/vllm-dflash2
FILES=(
  LICENSE
  vllm/model_executor/models/qwen3_dflash.py
  vllm/model_executor/models/qwen3_dflash2.py
  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
  vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
  tests/v1/spec_decode/test_dflash2.py
)
WORK=$(mktemp -d)
git clone -q --filter=blob:none --no-checkout https://github.com/vllm-project/vllm "$WORK/vllm"
git -C "$WORK/vllm" cat-file -e "$PIN^{commit}"
rm -rf "$DEST" && mkdir -p "$DEST"
{
  echo "{"
  echo " \"repo\": \"vllm-project/vllm\", \"revision\": \"$PIN\", \"license\": \"Apache-2.0\","
  echo " \"files\": ["
} > "$DEST/PIN.json"
: > "$DEST/SHA256SUMS"
n=0
for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  git -C "$WORK/vllm" show "$PIN:$f" > "$DEST/$f"
  blob=$(git -C "$WORK/vllm" rev-parse "$PIN:$f")
  [ "$(git hash-object "$DEST/$f")" = "$blob" ] || { echo "blob id mismatch: $f" >&2; exit 1; }
  sha=$(sha256sum "$DEST/$f" | cut -d' ' -f1)
  echo "$sha  $f" >> "$DEST/SHA256SUMS"
  n=$((n + 1)); sep=","; [ "$n" -eq "${#FILES[@]}" ] && sep=""
  echo "  {\"path\": \"$f\", \"size\": $(stat -c %s "$DEST/$f"), \"sha256\": \"$sha\", \"git_blob\": \"$blob\"}$sep" >> "$DEST/PIN.json"
done
echo " ]" >> "$DEST/PIN.json"; echo "}" >> "$DEST/PIN.json"
rm -rf "$WORK"
echo "vendored $n files of vllm-project/vllm @ $PIN into third_party/vllm-dflash2 (git blob ids verified)"
