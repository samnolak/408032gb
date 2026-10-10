#!/usr/bin/env bash
# N6 on CPU, no GPU needed: the N2 tiny qwen4exp model run by the real llama.cpp graph as several ubatches, as a
# token-by-token decode, and with a rollback, dumped graph by graph -> tests/test_qwen4exp_decode_parity.py
# (oracle: ref/qwen4exp.py forward and its incremental path forward_chunk); and the same model with an MTP block,
# driven like common/speculative.cpp -> tests/test_qwen4exp_mtp_parity.py (oracle: mtp_forward_chunk). Usage:
#   tools/tinygen/run_parity_qwen4exp_decode.sh <llama.cpp checkout @ec7630a> [build dir]
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=$(mktemp -d)
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$WORK/qwen4exp.gguf"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$WORK/qwen4exp_mtp.gguf" --mtp
# 64 ids in [3, 96); EOS (id 2) at 15 (last token of the first 16-token ubatch), 33, 47 (last prefill token before
# the decode), 55-56 (repeated, decoded one by one), 59 (read by the last 10-token ubatch) and 61: every run has a
# graph whose first token reads an EOS predecessor from an earlier graph
TOK=$(python3 -c "import random; random.seed(6); t=[random.randrange(3,96) for _ in range(64)]
for p in (15, 33, 47, 55, 56, 59, 61): t[p] = 2
print(','.join(map(str,t)))")
# rollback run: the draft at 48..51 carries two wrong tokens at 50-51 (shifted by 17 inside [3, 96)), which are
# rolled back, then 50..63 are decoded one by one
TOK_RB=$(python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]
d=t[48:50]+[(x-3+17)%93+3 for x in t[50:52]]
print(','.join(map(str,t[:48]+d+t[50:])))" "$TOK")
TOK56=$(python3 -c "import sys; print(','.join(sys.argv[1].split(',')[:56]))" "$TOK")
ONES8=$(python3 -c "print(','.join(['1']*8))"); ONES14=$(python3 -c "print(','.join(['1']*14))")
ONES16=$(python3 -c "print(','.join(['1']*16))")
mkdir -p "$WORK/n6"; echo "$TOK" > "$WORK/n6/tokens.txt"
# N2's prefixes plus the carried state: conv_state_at (GDN and PLE conv histories), state_predelta, new_state; for
# the MTP run also mtp_* and h_nextn
PFX="hc_,ple_,linear_attn_,z-,beta_sigmoid,alpha-,a_softplus,gate,conv_output,q_conv_predelta,k_conv_predelta,v_conv_predelta,attn_,final_output,indexer_,kq_mask_qsa,Qcur,Kcur,Vcur,ffn_,shared_expert,l_last,result_,conv_state_at,state_predelta,new_state,mtp_,h_nextn"
# f32 caches, no flash attention: the reference precision contract (tests/test_qwen4exp_tiny_parity.py)
run() {   # run <name> <model> <tokens> <-ub> [DUMP_CALLS] [DUMP_N_RS_SEQ]; extra env in $EXTRA
  local d="$WORK/n6/$1"
  env DUMP_DIR="$d" DUMP_TOKENS="$3" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$PFX" DUMP_CALLS="${5:-}" DUMP_N_RS_SEQ="${6:-0}" \
    ${EXTRA:-} "$BUILD/dump-tensors" -m "$2" -c 512 -b 512 -ub "$4" -t 1 -fa off -ctk f32 -ctv f32 \
    >"$WORK/n6/$1.log" 2>&1 || { tail -n 20 "$WORK/n6/$1.log"; exit 1; }
  echo "$3" > "$d/tokens.txt"
  echo "$1: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', len(m['calls']), 'calls, n_rs_seq', m['n_rs_seq'])" "$d/manifest.json")"
}
M="$WORK/qwen4exp.gguf"
run ub16     "$M" "$TOK"    16                             # one call, 4 ubatches aligned with the k-pools
run ub10     "$M" "$TOK"    10                             # one call, 7 ubatches, pools straddle ubatches
run decode   "$M" "$TOK"    512 "48,$ONES16"               # prefill 48, then one token per llama_process call
run rollback "$M" "$TOK_RB" 512 "48,4,r2,$ONES14" 3        # 4-token draft, roll back 2, decode the rest
# MTP: target calls 48 (3 ubatches of 16) + 8 x 1, each mirrored by the MTP context, then 4 greedy draft steps
EXTRA="DUMP_MTP=1 DUMP_MTP_DRAFT=4" run mtp "$WORK/qwen4exp_mtp.gguf" "$TOK56" 16 "48,$ONES8"
echo "mtp (MTP context): $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', [(c['kind'], c['n']) for c in m['calls']])" "$WORK/n6/mtp/mtp/manifest.json")"
# the commit decides one detail of the expected dumps (padding rows of indexer_pool_k_new, see the test)
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && QWEN4EXP_DECODE_DUMP="$WORK/n6" QWEN4EXP_TINY_GGUF="$WORK/qwen4exp.gguf" \
  QWEN4EXP_MTP_GGUF="$WORK/qwen4exp_mtp.gguf" \
  python3 -m unittest tests.test_qwen4exp_decode_parity tests.test_qwen4exp_mtp_parity tests.test_qwen4exp_chunked_ref -v
echo "artifacts: $WORK"
