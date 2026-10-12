#!/usr/bin/env bash
# N2 on CPU, no GPU needed: tiny qwen4exp model (architecture of the reassessment's candidate #1) -> real
# llama.cpp qwen4exp graph -> tensor dumps -> oracle parity (ref/qwen4exp.py). Usage:
#   tools/tinygen/run_parity_qwen4exp.sh <llama.cpp checkout @ec7630a> [build dir]
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=$(mktemp -d)
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$WORK/qwen4exp.gguf"
# 64 ids in [3, 96); EOS (id 2) at 20 and at 41-42 exercises the PLE n-gram reset, single and repeated
TOK=$(python3 -c "import random; random.seed(5); t=[random.randrange(3,96) for _ in range(64)]; t[20]=t[41]=t[42]=2; print(','.join(map(str,t)))")
# f32 caches, no flash attention: the reference precision contract (tests/test_qwen4exp_tiny_parity.py)
DUMP_DIR="$WORK/dump" DUMP_TOKENS="$TOK" \
DUMP_PREFIXES="hc_,ple_,linear_attn_,z-,beta_sigmoid,alpha-,a_softplus,gate,conv_output,q_conv_predelta,k_conv_predelta,v_conv_predelta,attn_,final_output,indexer_,kq_mask_qsa,Qcur,Kcur,Vcur,ffn_,shared_expert,l_last,result_" \
  "$BUILD/dump-tensors" -m "$WORK/qwen4exp.gguf" -c 512 -b 512 -ub 512 -t 1 -fa off -ctk f32 -ctv f32 >"$WORK/run.log" 2>&1 \
  || { tail -n 20 "$WORK/run.log"; exit 1; }
echo "$TOK" > "$WORK/dump/tokens.txt"
cd "$ROOT" && QWEN4EXP_TINY_DUMP="$WORK/dump" QWEN4EXP_TINY_GGUF="$WORK/qwen4exp.gguf" \
  python3 -m unittest tests.test_qwen4exp_tiny_parity -v
echo "artifacts: $WORK"
