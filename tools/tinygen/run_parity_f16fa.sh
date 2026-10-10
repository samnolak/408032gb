#!/usr/bin/env bash
# N10 / gap 3 of docs/reassessment/CORRECTNESS.md on CPU, no GPU needed: the R1 default precision of the KV cache and
# the attention (-ctk f16 -ctv f16 -fa on, scripts/r1/configs/llamacpp-server.json) on the N6 tiny qwen4exp model and
# the G1/N9 tiny glm5-next model, run by the real llama.cpp graphs and dumped graph by graph like N6/N9, then compared
# with the precision-aware oracles (ref/qwen4exp.py, ref/glm5next.py with prec=..., flash attention emulated by
# ref/fattn.py): tests/test_qwen4exp_f16fa_parity.py, tests/test_glm5next_f16fa_parity.py, tests/test_fattn_ref.py.
# Runs per model: pf64 (one 64-token ubatch: the prefill shape of N2/G1, the tiled CPU flash-attention path), ub16,
# ub10, (glm5-next: ub18), decode (48 + 16 x 1) and rollback (n_rs_seq 3), the token sequences of N6 and N9.
# Usage:
#   tools/tinygen/run_parity_f16fa.sh <llama.cpp checkout @ec7630a, or the R1 pin of tools/engines/pins.json> [build dir]
# PARITY_KV=f16|f32 (default f16) and PARITY_FA=on|off (default on) choose -ctk/-ctv and -fa (the other combinations
# split the effect of the cache type from the effect of flash attention); F16FA_WORK=<dir> keeps the artifacts there.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${F16FA_WORK:-$(mktemp -d)}
export PARITY_KV=${PARITY_KV:-f16} PARITY_FA=${PARITY_FA:-on}
case "$PARITY_KV/$PARITY_FA" in f16/on|f16/off|f32/on|f32/off) ;; *) echo "PARITY_KV=f16|f32, PARITY_FA=on|off"; exit 2;; esac
FLAGS="-ctk $PARITY_KV -ctv $PARITY_KV -fa $PARITY_FA"
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$WORK/qwen4exp.gguf"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/glm5next.gguf" --layers 12
ONES14=$(python3 -c "print(','.join(['1']*14))"); ONES16=$(python3 -c "print(','.join(['1']*16))")
# the token sequences of tools/tinygen/run_parity_qwen4exp_decode.sh (N6) and run_parity_glm5next_decode.sh (N9)
TOK_Q=$(python3 -c "import random; random.seed(6); t=[random.randrange(3,96) for _ in range(64)]
for p in (15, 33, 47, 55, 56, 59, 61): t[p] = 2
print(','.join(map(str,t)))")
TOK_G=$(python3 -c "import random; random.seed(9); print(','.join(str(random.randrange(3,96)) for _ in range(64)))")
rollback_tokens() {   # the draft at 48..51 carries two wrong tokens at 50-51, rolled back, then 50..63 one by one
  python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]
d=t[48:50]+[(x-3+17)%93+3 for x in t[50:52]]
print(','.join(map(str,t[:48]+d+t[50:])))" "$1"
}
# N6's and N9's prefixes plus the attention output (kqv_out; glm5-next fattn_mla), the softmax a run without flash
# attention names (kq_soft_max: absent when flash attention ran) and the caches the attention reads: the K/V cache
# (qwen4exp), the MLA latent cache (glm5-next) and the indexer k-pool cache (both)
PFX_Q="hc_,ple_,linear_attn_,z-,beta_sigmoid,alpha-,a_softplus,gate,conv_output,q_conv_predelta,k_conv_predelta,v_conv_predelta,attn_,final_output,indexer_,kq_mask_qsa,Qcur,Kcur,Vcur,ffn_,shared_expert,l_last,result_,conv_state_at,state_predelta,new_state,kqv_out,kq_soft_max,cache_k_l,cache_v_l,cache_idx_k_l"
PFX_G="hc_,attn_norm,ffn_norm,kda_,new_state,q_resid,kv_cmpr,q_absorbed,indexer_,kq_mask_dsa,kq_soft_max,kv_gathered,kqv_,attn_out,ffn_moe_topk,ffn_moe_weights_scaled,ffn_moe_out,ffn_shexp,ffn_out,l_out,result_,fattn_mla,cache_k_l,cache_idx_k_l"
run() {   # run <dir> <model> <prefixes> <name> <tokens> <-ub> [DUMP_CALLS] [DUMP_N_RS_SEQ]
  local d="$1/$4"
  env DUMP_DIR="$d" DUMP_TOKENS="$5" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$3" DUMP_CALLS="${7:-}" DUMP_N_RS_SEQ="${8:-0}" \
    "$BUILD/dump-tensors" -m "$2" -c 512 -b 512 -ub "$6" -t 1 $FLAGS \
    >"$1/$4.log" 2>&1 || { tail -n 20 "$1/$4.log"; exit 1; }
  echo "$5" > "$d/tokens.txt"
  echo "$(basename "$1") $4: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', len(m['calls']), 'calls, n_rs_seq', m['n_rs_seq'])" "$d/manifest.json") ($FLAGS)"
}
for M in q g; do
  rm -rf "$WORK/$M"; mkdir -p "$WORK/$M"
  if [ $M = q ]; then MODEL="$WORK/qwen4exp.gguf"; PFX=$PFX_Q; TOK=$TOK_Q; else MODEL="$WORK/glm5next.gguf"; PFX=$PFX_G; TOK=$TOK_G; fi
  echo "$TOK" > "$WORK/$M/tokens.txt"
  run "$WORK/$M" "$MODEL" "$PFX" pf64     "$TOK"                     512                 # one 64-token ubatch
  run "$WORK/$M" "$MODEL" "$PFX" ub16     "$TOK"                     16                  # 4 ubatches aligned with the pools
  run "$WORK/$M" "$MODEL" "$PFX" ub10     "$TOK"                     10                  # 7 ubatches, pools straddle
  [ $M = g ] && run "$WORK/$M" "$MODEL" "$PFX" ub18 "$TOK"           18                  # 18+18+18+10
  run "$WORK/$M" "$MODEL" "$PFX" decode   "$TOK"                     512 "48,$ONES16"    # prefill 48, 16 x 1
  run "$WORK/$M" "$MODEL" "$PFX" rollback "$(rollback_tokens "$TOK")" 512 "48,4,r2,$ONES14" 3
done
# the commit decides pin-dependent details of the expected dumps (see tests/test_*_decode_parity.py)
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && QWEN4EXP_F16FA_DUMP="$WORK/q" QWEN4EXP_F16FA_GGUF="$WORK/qwen4exp.gguf" \
  GLM5NEXT_F16FA_DUMP="$WORK/g" GLM5NEXT_F16FA_GGUF="$WORK/glm5next.gguf" \
  python3 -m unittest tests.test_qwen4exp_f16fa_parity tests.test_glm5next_f16fa_parity tests.test_fattn_ref -v
echo "artifacts: $WORK"
