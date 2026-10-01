#!/usr/bin/env bash
# Build the ggml parity harness against a CPU build of ggml at the pinned llama.cpp commit.
#   tests/ggml_parity/build.sh <llama.cpp checkout> <ggml cmake build dir>
set -euo pipefail
src=$1; bld=$2; here=$(cd "$(dirname "$0")" && pwd)
cc -O2 -std=c11 -I"$src/ggml/include" "$here/gdn_harness.c" \
   "$bld/src/libggml.a" "$bld/src/libggml-cpu.a" "$bld/src/libggml-base.a" \
   -lstdc++ -lm -lpthread -o "$here/gdn_harness"
echo "built $here/gdn_harness  (export GGML_GDN_HARNESS=$here/gdn_harness)"
