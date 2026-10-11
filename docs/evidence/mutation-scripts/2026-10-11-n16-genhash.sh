#!/usr/bin/env bash
# N16: tools/tinygen/mimo2_tiny.py writes the same default GGUF before and after its N16 options (sha256), with the
# gguf-py of each llama.cpp pin. usage: 2026-10-11-n16-genhash.sh <repo> <commit before the change> <out dir> <gguf-py>...
set -euo pipefail
REPO=$1; BEFORE=$2; OUT=$3; shift 3
mkdir -p "$OUT"
git -C "$REPO" show "$BEFORE:tools/tinygen/mimo2_tiny.py" > "$OUT/mimo2_tiny_before.py"
for gp in "$@"; do
  tag=$(git -C "$gp/.." rev-parse --short HEAD)
  for args in "" "--seed 3"; do
    n=$(echo "default$args" | tr -d ' -')
    PYTHONPATH="$gp" python3 "$OUT/mimo2_tiny_before.py" "$OUT/before-$tag-$n.gguf" $args >/dev/null
    PYTHONPATH="$gp" python3 "$REPO/tools/tinygen/mimo2_tiny.py" "$OUT/after-$tag-$n.gguf" $args >/dev/null
  done
done
cd "$OUT" && sha256sum before-*.gguf after-*.gguf
