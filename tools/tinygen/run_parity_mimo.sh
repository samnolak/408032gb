#!/usr/bin/env bash
# MiMo-V2.x-Flash G1 on CPU: tiny mimo2 model -> real llama.cpp graph -> dumps -> oracle parity.
#   tools/tinygen/run_parity_mimo.sh <llama.cpp checkout @ec7630a> [build dir]
set -euo pipefail
LLAMA=${1:?llama.cpp checkout}; BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=$(mktemp -d)
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/mimo2_tiny.py" "$WORK/mimo.gguf"
TOK=$(python3 -c "import random; random.seed(3); print(','.join(str(random.randrange(3,96)) for _ in range(40)))")
DUMP_DIR="$WORK/dump" DUMP_TOKENS="$TOK" DUMP_PREFIXES="attn_norm,Qcur,Kcur,Vcur,kqv_out,attn_out,ffn_moe,l_out" \
  "$BUILD/dump-tensors" -m "$WORK/mimo.gguf" -c 512 -b 512 -ub 512 -t 1 -fa off -ctk f32 -ctv f32 >"$WORK/run.log" 2>&1
cd "$ROOT" && MIMO_TINY_DUMP="$WORK/dump" MIMO_TINY_GGUF="$WORK/mimo.gguf" python3 -m unittest tests.test_mimo_tiny_parity -v
