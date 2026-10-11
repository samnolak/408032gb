#!/usr/bin/env bash
# N17: tools/tinygen/mimo2_tiny.py writes the same GGUFs before and after its N17 options (--mtp-layers, --head-dims) for
# the default, --seed 3 and the N16 MTP model (n16.gguf: --fused-qkv --sinks swa --value-scale 0.75 --mtp), and
# --mtp-layers 1 is --mtp; sha256 with the gguf-py of each llama.cpp pin.
# usage: 2026-10-11-n17-genhash.sh <repo> <commit before the change> <out dir> <gguf-py>...
set -euo pipefail
REPO=$1; BEFORE=$2; OUT=$3; shift 3
mkdir -p "$OUT"
git -C "$REPO" show "$BEFORE:tools/tinygen/mimo2_tiny.py" > "$OUT/mimo2_tiny_before.py"
N16="--fused-qkv --sinks swa --value-scale 0.75 --mtp"
for gp in "$@"; do
  tag=$(git -C "$gp/.." rev-parse --short HEAD)
  for args in "" "--seed 3" "$N16"; do
    n=$(echo "default$args" | tr -d ' -.')
    PYTHONPATH="$gp" python3 "$OUT/mimo2_tiny_before.py" "$OUT/before-$tag-$n.gguf" $args >/dev/null
    PYTHONPATH="$gp" python3 "$REPO/tools/tinygen/mimo2_tiny.py" "$OUT/after-$tag-$n.gguf" $args >/dev/null
  done
  PYTHONPATH="$gp" python3 "$REPO/tools/tinygen/mimo2_tiny.py" "$OUT/after-$tag-mtplayers1.gguf" \
    --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 1 >/dev/null
done
cd "$OUT" && sha256sum before-*.gguf after-*.gguf
