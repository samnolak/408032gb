#!/usr/bin/env bash
# N10 / gap 2 of docs/reassessment/CORRECTNESS.md on CPU, no GPU needed: quantized weights. The N2/N6 tiny qwen4exp
# architecture with every quantizable matrix of input width 256 (tools/tinygen/qwen4exp_tiny.py --quant-dims, so that
# k-quants apply), quantized by the pinned llama.cpp's own llama-quantize to Q8_0 and to Q4_K_M, run by the real
# qwen4exp graph at the f32 cache / flash-attention-off contract of N2/N6 and dumped graph by graph; compared with
# ref/qwen4exp.py on the dequantized weights with the CPU activation quantization emulated (ref/quant.py):
# tests/test_qwen4exp_quant_parity.py, tests/test_quant_ref.py. The same for the G1 / N9 tiny glm5-next
# (tools/tinygen/glm5next_tiny.py --quant-dims --layers 12, the N9 token sequence) under <work>/glm5next:
# tests/test_glm5next_quant_parity.py.
# The three PLE gammas ple_norm_{key,query,conv}.weight are kept F32 (--tensor-type ple_norm=f32): their names miss
# llama-quantize's "_norm.weight" exclusion (src/llama-quant.cpp:300), so it quantizes these 2-D tensors, and the CPU
# backend then aborts in the ggml_mul that applies them ("binary_op: unsupported types: dst: f32, src0: f32, src1:
# q8_0", ggml/src/ggml-cpu/binary-ops.cpp:135) at both pins; see docs/evidence/2026-10-10-quant-fa-parity.md.
# Runs per type: pf64 (one 64-token ubatch: activation rows quantized four at a time by the repacked Q4_K path),
# ub10 (8 rows in fours + 2 single rows), decode (48 + 16 x 1: one row, the gemv path); the token sequence of N6.
# Q4_K_M also pf64-nr and decode-nr with --no-repack (common/arg.cpp:2421-2428): the Q4_K weights stay in the plain
# layout, whose activations go through quantize_row_q8_K row by row (Q8_0, Q5_K, Q6_K are not repacked on x86).
# Usage:
#   tools/tinygen/run_parity_quant.sh <llama.cpp checkout @ec7630a, or the R1 pin of tools/engines/pins.json> [build dir]
# QUANT_WORK=<dir> keeps the artifacts there.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${QUANT_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors llama-quantize >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" --quant-dims "$WORK/qwen4exp-f32.gguf"
mkdir -p "$WORK/glm5next"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" --quant-dims --layers 12 \
  "$WORK/glm5next/glm5next-f32.gguf"
ONES16=$(python3 -c "print(','.join(['1']*16))")
TOK=$(python3 -c "import random; random.seed(6); t=[random.randrange(3,96) for _ in range(64)]
for p in (15, 33, 47, 55, 56, 59, 61): t[p] = 2
print(','.join(map(str,t)))")
TOK_G=$(python3 -c "import random; random.seed(9); print(','.join(str(random.randrange(3,96)) for _ in range(64)))")
# N6's prefixes plus beta- (the ssm_beta product before its sigmoid); glm5-next: the inputs and outputs of its products
PFX_G="inp_embd,attn_norm,ffn_norm,kda_,q_resid,q_absorbed,kqv_out,attn_out,ffn_,result_"
PFX="hc_,ple_,linear_attn_,z-,beta-,beta_sigmoid,alpha-,a_softplus,gate,conv_output,q_conv_predelta,k_conv_predelta,v_conv_predelta,attn_,final_output,indexer_,kq_mask_qsa,Qcur,Kcur,Vcur,ffn_,shared_expert,l_last,result_,conv_state_at,state_predelta,new_state"
run() {   # run <dir> <name> <-ub> [DUMP_CALLS] [extra flags]; tokens $T, prefixes $P
  local d="$1/$2"
  env DUMP_DIR="$d" DUMP_TOKENS="$T" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$P" DUMP_CALLS="${4:-}" DUMP_N_RS_SEQ=0 \
    "$BUILD/dump-tensors" -m "$1/model.gguf" -c 512 -b 512 -ub "$3" -t 1 -ctk f32 -ctv f32 -fa off ${5:-} \
    >"$1/$2.log" 2>&1 || { tail -n 20 "$1/$2.log"; exit 1; }
  echo "$T" > "$d/tokens.txt"
  echo "${1#"$WORK"/} $2: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', len(m['calls']), 'calls')" "$d/manifest.json")"
}
quantize_and_run() {   # quantize_and_run <dir> <f32 gguf> <type> [llama-quantize options]
  rm -rf "${1:?}"; mkdir -p "$1"
  "$BUILD/quantize/llama-quantize" ${4:-} "$2" "$1/model.gguf" "$3" 1 \
    >"$1/quantize.log" 2>&1 || { tail -n 20 "$1/quantize.log"; exit 1; }
  echo "$T" > "$1/tokens.txt"
  run "$1" pf64   512                    # one 64-token ubatch
  run "$1" ub10   10                     # 7 ubatches (10 x 6 + 4)
  run "$1" decode 512 "48,$ONES16"       # prefill 48, 16 x 1
  if [ "$3" = Q4_K_M ]; then
    run "$1" pf64-nr   512 ""            --no-repack
    run "$1" decode-nr 512 "48,$ONES16"  --no-repack
  fi
}
for QT in Q8_0 Q4_K_M; do
  T=$TOK P=$PFX quantize_and_run "$WORK/$QT" "$WORK/qwen4exp-f32.gguf" "$QT" "--tensor-type ple_norm=f32"
  T=$TOK_G P=$PFX_G quantize_and_run "$WORK/glm5next/$QT" "$WORK/glm5next/glm5next-f32.gguf" "$QT"
done
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && QWEN4EXP_QUANT_DUMP="$WORK" GLM5NEXT_QUANT_DUMP="$WORK/glm5next" \
  python3 -m unittest tests.test_qwen4exp_quant_parity tests.test_glm5next_quant_parity tests.test_quant_ref -v
echo "artifacts: $WORK"
