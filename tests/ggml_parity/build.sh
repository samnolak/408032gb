#!/usr/bin/env bash
# Build the ggml parity harness against a CPU build of ggml at the pinned llama.cpp commit.
#   tests/ggml_parity/build.sh <llama.cpp checkout> <ggml cmake build dir>
set -euo pipefail
src=$1; bld=$2; here=$(cd "$(dirname "$0")" && pwd)
# a ggml built with OpenMP references GOMP_* symbols; link libgomp only then
# (grep without -q: with pipefail, grep -q exits early, nm gets SIGPIPE and the test turns false)
omp=""
if nm "$bld/src/libggml-cpu.a" 2>/dev/null | grep "U GOMP_" >/dev/null; then
  omp="-fopenmp"
fi
cc -O2 -std=c11 -I"$src/ggml/include" "$here/gdn_harness.c" \
   "$bld/src/libggml.a" "$bld/src/libggml-cpu.a" "$bld/src/libggml-base.a" \
   -lstdc++ -lm -lpthread $omp -o "$here/gdn_harness"
echo "built $here/gdn_harness  (export GGML_GDN_HARNESS=$here/gdn_harness)"
