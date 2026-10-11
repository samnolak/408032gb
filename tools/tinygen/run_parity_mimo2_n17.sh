#!/usr/bin/env bash
# N17 (follow-up of gap 12 of docs/reassessment/CORRECTNESS.md, docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md) on
# CPU, no GPU needed: the two mimo2 (MiMo-V2.x-Flash) gaps the real MiMo-V2.6-Flash config made relevant
# (third_party/hf/mimo-v2.6-flash-chat/config.json: num_nextn_predict_layers 3; the R1 default precision).
# One tiny model, n17.gguf = tools/tinygen/mimo2_tiny.py --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 3 (the
# N16 trunk and MTP block 6 of n16.gguf, plus MTP blocks 7 and 8), run by the real mimo2 graph (-c 512 -t 1) and dumped
# graph by graph (tools/dump/dump_tensors.cpp DUMP_PER_GRAPH / DUMP_CALLS / DUMP_MTP) over the 300-token sequence of N16:
#   mtp3         f32 caches, flash attention off, -ub 64: the three MTP heads (nextn_layer_offset 0, 1, 2) driven like
#                common/speculative.cpp chain_heads with llama-server's seq_rm steps: target 240, 8 x 1, then drafts of
#                3 steps ("d3"), verify batches of 4 and rejections ("r<n>": both contexts cut, the pending h moved to
#                the last kept position), 6 x 1, and a final draft (DUMP_MTP_DRAFT=3); both SWA caches wrap at 256
#   fa/pf64      -ctk f16 -ctv f16 -fa on (the R1 default, scripts/r1/configs/llamacpp-server.json), one 64-token
#                ubatch (-ub 512): the tiled CPU flash attention
#   fa/ub64      the same precision, one call of 300 tokens, -ub 64: 4 tiled ubatches and one of 44 rows (one_chunk),
#                the SWA ring wraps, the global layers read 512 cells in the last ubatch
#   fa/decode    prefill 240 (-ub 64), 60 one-token calls: one_chunk per token, the split-KV path on the global layers
#                once they read 512 cells (position >= 256), the SWA ring wraps during the decode
#   fa/rollback  the N16 rollback plan at that precision (rollbacks by 2 and by 10, inside and across the window)
#   fa/mtp3      the mtp3 plan at that precision (the MTP context takes the same cache types and flash attention)
#   fa/hd192     n17hd.gguf = the same options with --head-dims 192,128 (MiMo-V2.6-Flash's K / V head sizes, no MTP
#                blocks), 64 + 16 x 1 at that precision (-ub 512): the AVX2 SIMD loops of the CPU flash attention
# -> tests/test_mimo2_mtp_heads_parity.py, tests/test_mimo2_f16fa_parity.py (oracles: ref/mimo2.py with prec, ref/fattn.py
# with sinks), tests/test_mimo2_heads_ref.py and tests/test_fattn_sinks_ref.py (numpy-only properties of the same code).
# Usage:
#   tools/tinygen/run_parity_mimo2_n17.sh <llama.cpp checkout @ec7630a, or the R1 pin of tools/engines/pins.json> [build dir]
# MIMO2_N17_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${MIMO2_N17_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
rm -rf "$WORK/n17"; mkdir -p "$WORK/n17/fa"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/mimo2_tiny.py" "$WORK/n17/n17.gguf" \
  --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 3
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/mimo2_tiny.py" "$WORK/n17/n17hd.gguf" \
  --fused-qkv --sinks swa --value-scale 0.75 --head-dims 192,128
# the 300 ids of N16 (tools/tinygen/run_parity_mimo2_decode.sh)
TOK=$(python3 -c "import random; random.seed(16); print(','.join(str(random.randrange(3,96)) for _ in range(300)))")
echo "$TOK" > "$WORK/n17/tokens.txt"
seq_py() { python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]; w=lambda xs: [(x-3+17)%93+3 for x in xs]
$1
print(','.join(map(str,s)))" "$TOK"; }
ones() { python3 -c "import sys; print(','.join(['1']*int(sys.argv[1])))" "$1"; }
# mtp3: verify batches of 4 after each 3-step draft; the rejected positions carry wrong ids (shifted by 17 in [3, 96))
TOK_MTP=$(seq_py "s=t[:250]+w(t[250:252])+t[250:253]+w(t[253:254])+t[253:260]+w(t[260:263])+t[260:266]")
PLAN_MTP="240,$(ones 8),d3,4,r2,d3,4,r1,d3,4,1,1,d3,4,r3,$(ones 6)"
TOK_RB=$(seq_py "s=t[:256]+t[256:258]+w(t[258:260])+t[258:264]+t[264:266]+w(t[266:276])+t[266:]")
TOK64=$(seq_py "s=t[:64]")
PFX="embd,attn_norm,wqkv,Qcur,Kcur,Vcur,kq,attn_out,ffn_inp,ffn_norm,ffn_up,ffn_gate,ffn_swiglu,ffn_out,ffn_moe_logits,ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_out,l_out,result_,h_nextn,mtp_"
# with flash attention: no kq / kq_soft_max nodes; the caches (cache_k_l<il> / cache_v_l<il> views) instead
PFX_FA="embd,attn_norm,wqkv,Qcur,Kcur,Vcur,kq_soft_max,kqv_out,attn_out,cache_k_l,cache_v_l,ffn_inp,ffn_norm,ffn_up,ffn_gate,ffn_swiglu,ffn_out,ffn_moe_logits,ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_out,l_out,result_,h_nextn"
run() {   # run <name> <tokens> <-ub> <flags> [DUMP_CALLS]; extra env in $EXTRA, prefixes $P, model $MODEL (n17.gguf)
  local d="$WORK/n17/$1"
  env DUMP_DIR="$d" DUMP_TOKENS="$2" DUMP_PER_GRAPH=1 DUMP_PREFIXES="${P:-$PFX}" DUMP_CALLS="${5:-}" ${EXTRA:-} \
    "$BUILD/dump-tensors" -m "$WORK/n17/${MODEL:-n17.gguf}" -c 512 -b 512 -ub "$3" -t 1 $4 \
    >"$WORK/n17/$(echo "$1" | tr / _).log" 2>&1 || { tail -n 20 "$WORK/n17/$(echo "$1" | tr / _).log"; exit 1; }
  echo "$2" > "$d/tokens.txt"
  echo "$1: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', len(m['calls']), 'calls')" "$d/manifest.json") ($4)"
}
F32="-fa off -ctk f32 -ctv f32"; F16FA="-fa on -ctk f16 -ctv f16"
EXTRA="DUMP_MTP=1 DUMP_MTP_DRAFT=3" run mtp3 "$TOK_MTP" 64 "$F32" "$PLAN_MTP"
echo "mtp3 (MTP context): $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); k=[c['kind'] for c in m['calls']]; print(len(m['graphs']), 'graphs,', {x: k.count(x) for x in sorted(set(k))}, 'heads', sorted({c['head'] for c in m['calls'] if 'head' in c}))" "$WORK/n17/mtp3/mtp/manifest.json")"
P="$PFX,cache_k_l,cache_v_l" EXTRA="DUMP_MTP=1 DUMP_MTP_DRAFT=3" run fa/mtp3 "$TOK_MTP" 64 "$F16FA" "$PLAN_MTP"
P=$PFX_FA run fa/pf64     "$TOK64"  512 "$F16FA"
P=$PFX_FA run fa/ub64     "$TOK"    64  "$F16FA"
P=$PFX_FA run fa/decode   "$TOK"    64  "$F16FA" "240,$(ones 60)"
P=$PFX_FA run fa/rollback "$TOK_RB" 64  "$F16FA" "240,$(ones 16),4,r2,$(ones 6),12,r10,$(ones 34)"
P=$PFX_FA MODEL=n17hd.gguf run fa/hd192 "$(seq_py "s=t[:80]")" 512 "$F16FA" "64,$(ones 16)"
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && MIMO2_N17_DUMP="$WORK/n17" \
  python3 -m unittest tests.test_mimo2_mtp_heads_parity tests.test_mimo2_f16fa_parity tests.test_mimo2_heads_ref tests.test_fattn_sinks_ref -v
echo "artifacts: $WORK"
