#!/usr/bin/env bash
# G1 on CPU, no GPU needed: tiny GLM-5.3-Flash-shaped model -> real llama.cpp glm5-next graph
# -> tensor dumps -> oracle parity. Usage:
#   tools/tinygen/run_parity.sh <llama.cpp checkout @ec7630a> [build dir]
set -euo pipefail
LLAMA=${1:?llama.cpp checkout}; BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=$(mktemp -d)
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/tiny.gguf" --layers 12
TOK=$(python3 -c "import random; random.seed(2); print(','.join(str(random.randrange(3,96)) for _ in range(64)))")
# f32 caches, no flash attention: the reference precision contract (tests/test_glm_tiny_parity.py)
DUMP_DIR="$WORK/dump" DUMP_TOKENS="$TOK" \
DUMP_PREFIXES="kda_,hc_,attn_norm,ffn_norm,ffn_moe_,ffn_shexp,ffn_out,l_out,indexer,kq_mask,q_absorbed,q_resid,kv_cmpr,kqv_out,attn_out" \
  "$BUILD/dump-tensors" -m "$WORK/tiny.gguf" -c 512 -b 512 -ub 512 -t 1 -fa off -ctk f32 -ctv f32 >"$WORK/run.log" 2>&1
GLM_TINY_DUMP="$WORK/dump" GLM_TINY_GGUF="$WORK/tiny.gguf" python3 -m unittest tests.test_glm_tiny_parity -v
echo "artifacts: $WORK"
