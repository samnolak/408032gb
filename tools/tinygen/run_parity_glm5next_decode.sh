#!/usr/bin/env bash
# N9 on CPU, no GPU needed: the G1 tiny glm5-next model (GLM-5.3-Flash shape, 12 layers: KDA, DSA at 3/7/11, a dense
# lead layer, MoE) run by the real llama.cpp graph as several ubatches, as a token-by-token decode and with a rollback,
# dumped graph by graph -> tests/test_glm5next_decode_parity.py (oracle: ref/glm5next.py forward and its incremental
# path forward_chunk / cache_truncate); where the llama.cpp checkout has a glm5-next MTP graph (not at ec7630a), the same
# model with an MTP block, driven like common/speculative.cpp -> tests/test_glm5next_mtp_parity.py (oracle:
# mtp_forward_chunk); tests/test_glm5next_chunked_ref.py checks the oracle's own continuity. Usage:
#   tools/tinygen/run_parity_glm5next_decode.sh <llama.cpp checkout @ec7630a, or the R1 pin of tools/engines/pins.json> [build dir]
# GLM5NEXT_DECODE_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${GLM5NEXT_DECODE_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
# the G1 model of tools/tinygen/run_parity.sh (seed 0, 12 layers)
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/glm5next.gguf" --layers 12
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/glm5next_mtp.gguf" --layers 12 --mtp
# 64 ids in [3, 96); the model has no token-dependent logic besides the embedding, so any ids do
TOK=$(python3 -c "import random; random.seed(9); print(','.join(str(random.randrange(3,96)) for _ in range(64)))")
# rollback run: the draft at 48..51 carries two wrong tokens at 50-51 (shifted by 17 inside [3, 96)), which are
# rolled back, then 50..63 are decoded one by one
TOK_RB=$(python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]
d=t[48:50]+[(x-3+17)%93+3 for x in t[50:52]]
print(','.join(map(str,t[:48]+d+t[50:])))" "$TOK")
TOK56=$(python3 -c "import sys; print(','.join(sys.argv[1].split(',')[:56]))" "$TOK")
ONES8=$(python3 -c "print(','.join(['1']*8))"); ONES14=$(python3 -c "print(','.join(['1']*14))")
ONES16=$(python3 -c "print(','.join(['1']*16))")
rm -rf "$WORK/n9"; mkdir -p "$WORK/n9"; echo "$TOK" > "$WORK/n9/tokens.txt"
# every named tensor the tests read: mHC, KDA, the KDA state written back, DSA (both decode paths of ec7630a), MoE,
# head; for the MTP run also mtp_* and h_nextn
PFX="hc_,attn_norm,ffn_norm,kda_,new_state,q_resid,kv_cmpr,q_absorbed,indexer_,kq_mask_dsa,kq_soft_max_gathered,kv_gathered,kqv_,attn_out,ffn_moe_topk,ffn_moe_weights_scaled,ffn_moe_out,ffn_shexp,ffn_out,l_out,result_,mtp_,h_nextn"
# f32 caches, no flash attention: the reference precision contract (tests/test_glm_tiny_parity.py)
run() {   # run <name> <tokens> <-ub> [DUMP_CALLS] [DUMP_N_RS_SEQ]; model in $MODEL, extra env in $EXTRA
  local d="$WORK/n9/$1"
  env DUMP_DIR="$d" DUMP_TOKENS="$2" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$PFX" DUMP_CALLS="${4:-}" DUMP_N_RS_SEQ="${5:-0}" \
    ${EXTRA:-} "$BUILD/dump-tensors" -m "${MODEL:-$WORK/glm5next.gguf}" -c 512 -b 512 -ub "$3" -t 1 -fa off -ctk f32 -ctv f32 \
    >"$WORK/n9/$1.log" 2>&1 || { tail -n 20 "$WORK/n9/$1.log"; exit 1; }
  echo "$2" > "$d/tokens.txt"
  echo "$1: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', len(m['calls']), 'calls, n_rs_seq', m['n_rs_seq'])" "$d/manifest.json")"
}
run ub16     "$TOK"    16                             # one call, 4 ubatches aligned with the k-pools
run ub10     "$TOK"    10                             # one call, 7 ubatches, pools straddle ubatches
run ub18     "$TOK"    18                             # one call, 18+18+18+10: > 16 tokens (ec7630a scatter path), straddling
run decode   "$TOK"    512 "48,$ONES16"               # prefill 48, then one token per llama_process call
run rollback "$TOK_RB" 512 "48,4,r2,$ONES14" 3        # 4-token draft, roll back 2, decode the rest
# MTP: target calls 48 (3 ubatches of 16) + 8 x 1, each mirrored by the MTP context, then 4 greedy draft steps; only
# where llama.cpp has a glm5-next MTP graph (ec7630a throws "GLM5-Next NextN graph not implemented yet")
if grep -q "graph_mtp::graph_mtp" "$LLAMA/src/models/glm5-next.cpp"; then
  MODEL="$WORK/glm5next_mtp.gguf" EXTRA="DUMP_MTP=1 DUMP_MTP_DRAFT=4" run mtp "$TOK56" 16 "48,$ONES8"
  echo "mtp (MTP context): $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', [(c['kind'], c['n']) for c in m['calls']])" "$WORK/n9/mtp/mtp/manifest.json")"
else
  echo "mtp: not run, this llama.cpp has no glm5-next MTP graph (src/models/glm5-next.cpp)"
fi
# the commit decides which DSA decode path and which k-pool padding the dumps show (see the test)
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && GLM5NEXT_DECODE_DUMP="$WORK/n9" GLM5NEXT_TINY_GGUF="$WORK/glm5next.gguf" \
  GLM5NEXT_MTP_GGUF="$WORK/glm5next_mtp.gguf" \
  python3 -m unittest tests.test_glm5next_decode_parity tests.test_glm5next_mtp_parity tests.test_glm5next_chunked_ref -v
echo "artifacts: $WORK"
