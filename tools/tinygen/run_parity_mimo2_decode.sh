#!/usr/bin/env bash
# N16 (gap 12 of docs/reassessment/CORRECTNESS.md) on CPU, no GPU needed: MiMo-V2.x-Flash (llama.cpp "mimo2") beyond the
# single G1 prefill. Two tiny models from tools/tinygen/mimo2_tiny.py (same trunk weights):
#   split.gguf  the G1 model (default options: split attn_q/k/v, sinks on every layer, no value scale, no MTP block)
#   n16.gguf    --fused-qkv --sinks swa --value-scale 0.75 --mtp: the fused attn_qkv path, sinks on the sliding-window
#               layers only, the attention value scale, and an MTP block (layer 6)
# run by the real mimo2 graph on the CPU (f32 caches, flash attention off, -c 512), dumped graph by graph
# (tools/dump/dump_tensors.cpp DUMP_PER_GRAPH / DUMP_CALLS / DUMP_MTP) over one 300-token sequence; the SWA cache then
# has 256 cells (pad(min(512, 8 + n_ubatch), 256), src/llama-kv-cache-iswa.cpp:73), so every run reaching position 256
# wraps it:
#   ub64      one call, -ub 64: 64 x 4 + 44 (the fifth ubatch lands in cells 0..43)
#   ub13      one call, -ub 13: 23 x 13 + 1 (ubatches longer than the window 8; one straddles the wrap)
#   ub5       one call, -ub 5: 60 x 5 (the window spans two or three ubatches)
#   decode    prefill 240 (-ub 64), then 60 one-token calls (the SWA cache wraps at 256 during the decode)
#   rollback  prefill 240, 16 x 1, a 4-token draft (2 wrong) rolled back by 2 (inside the window), 6 x 1, a 12-token
#             draft (10 wrong) rolled back by 10 (more than the window), 34 x 1
#   evict     prefill 280 (the SWA cache wrapped past positions 0..23), roll back to position 20, a call of 24 and 4 x 1:
#             llama.cpp accepts the rollback although the SWA cache no longer holds positions 13..19 (expected to
#             differ from the full-sequence oracle; the cell-accurate incremental oracle matches)
#   split     the G1 model: prefill 240, 20 x 1 (the split-qkv path and sinks on the global layers)
#   mtp       the MTP block driven like common/speculative.cpp: target 240 + 24 x 1, each call mirrored by the MTP
#             context (its own SWA cache wraps at 256 too), then 4 greedy draft steps
# and n16.gguf quantized by the pinned llama-quantize to MXFP4_MOE (routed experts MXFP4, the other matrices Q8_0):
#   mxfp4/pf64, mxfp4/decode (48 + 16 x 1), and both with --no-repack (MXFP4 is repacked to mxfp4_8x8 on AVX2 by
#   default, ggml/src/ggml-cpu/repack.cpp:5094-5098)
# -> tests/test_mimo2_decode_parity.py, tests/test_mimo2_mtp_parity.py (oracle: ref/mimo2.py), tests/test_mimo2_chunked_ref.py
# (the oracle's own continuity), tests/test_mimo2_quant_parity.py and tests/test_mxfp4_ref.py (ref/quant.py MXFP4).
# Usage:
#   tools/tinygen/run_parity_mimo2_decode.sh <llama.cpp checkout @ec7630a, or the R1 pin of tools/engines/pins.json> [build dir]
# MIMO2_DECODE_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${MIMO2_DECODE_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  -DG0_QUANTIZE=ON >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors llama-quantize >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
rm -rf "$WORK/n16"; mkdir -p "$WORK/n16"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/mimo2_tiny.py" "$WORK/n16/split.gguf"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/mimo2_tiny.py" "$WORK/n16/n16.gguf" \
  --fused-qkv --sinks swa --value-scale 0.75 --mtp
# 300 ids in [3, 96); the model has no token-dependent logic besides the embedding
TOK=$(python3 -c "import random; random.seed(16); print(','.join(str(random.randrange(3,96)) for _ in range(300)))")
echo "$TOK" > "$WORK/n16/tokens.txt"
seq_py() { python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]; w=lambda xs: [(x-3+17)%93+3 for x in xs]
$1
print(','.join(map(str,s)))" "$TOK"; }
# rollback: draft 256..259 with wrong ids at 258-259, draft 264..275 with wrong ids at 266..275 (shifted by 17 in [3, 96))
TOK_RB=$(seq_py "s=t[:256]+t[256:258]+w(t[258:260])+t[258:264]+t[264:266]+w(t[266:276])+t[266:]")
TOK_EV=$(seq_py "s=t[:280]+t[20:48]")
TOK260=$(seq_py "s=t[:260]"); TOK264=$(seq_py "s=t[:264]"); TOK64=$(seq_py "s=t[:64]")
ones() { python3 -c "import sys; print(','.join(['1']*int(sys.argv[1])))" "$1"; }
PFX="embd,attn_norm,wqkv,Qcur,Kcur,Vcur,kq,attn_out,ffn_inp,ffn_norm,ffn_up,ffn_gate,ffn_swiglu,ffn_out,ffn_moe_logits,ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_out,l_out,result_,h_nextn,mtp_"
PFX_Q="embd,attn_norm,wqkv,kqv_out,attn_out,ffn_,result_"
# f32 caches, no flash attention: the reference precision contract of G1
run() {   # run <name> <model> <tokens> <-ub> [DUMP_CALLS] [extra dump-tensors flags]; extra env in $EXTRA, prefixes $P
  local d="$WORK/n16/$1"
  env DUMP_DIR="$d" DUMP_TOKENS="$3" DUMP_PER_GRAPH=1 DUMP_PREFIXES="${P:-$PFX}" DUMP_CALLS="${5:-}" ${EXTRA:-} \
    "$BUILD/dump-tensors" -m "$2" -c 512 -b 512 -ub "$4" -t 1 -fa off -ctk f32 -ctv f32 ${6:-} \
    >"$WORK/n16/$(echo "$1" | tr / _).log" 2>&1 || { tail -n 20 "$WORK/n16/$(echo "$1" | tr / _).log"; exit 1; }
  echo "$3" > "$d/tokens.txt"
  echo "$1: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', len(m['calls']), 'calls')" "$d/manifest.json")"
}
M=$WORK/n16/n16.gguf
run ub64     "$M" "$TOK"    64
run ub13     "$M" "$TOK"    13
run ub5      "$M" "$TOK"    5
run decode   "$M" "$TOK"    64 "240,$(ones 60)"
run rollback "$M" "$TOK_RB" 64 "240,$(ones 16),4,r2,$(ones 6),12,r10,$(ones 34)"
run evict    "$M" "$TOK_EV" 64 "280,r260,24,$(ones 4)"
run split    "$WORK/n16/split.gguf" "$TOK260" 64 "240,$(ones 20)"
EXTRA="DUMP_MTP=1 DUMP_MTP_DRAFT=4" run mtp "$M" "$TOK264" 64 "240,$(ones 24)"
echo "mtp (MTP context): $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['graphs']), 'graphs,', [(c['kind'], c['n']) for c in m['calls']][-6:])" "$WORK/n16/mtp/mtp/manifest.json")"
# MXFP4: the routed experts (3-D, ne[2] > 1) to MXFP4, every other matrix to Q8_0 (src/llama-quant.cpp:512-523)
mkdir -p "$WORK/n16/mxfp4"
"$BUILD/quantize/llama-quantize" "$M" "$WORK/n16/mxfp4/model.gguf" MXFP4_MOE 1 >"$WORK/n16/mxfp4/quantize.log" 2>&1 \
  || { tail -n 20 "$WORK/n16/mxfp4/quantize.log"; exit 1; }
QM=$WORK/n16/mxfp4/model.gguf
# -lv 4: the log lists the model buffers, which tests/test_mimo2_quant_parity.py reads (CPU_REPACK only without -nr)
P=$PFX_Q run mxfp4/pf64      "$QM" "$TOK64" 512 ""               "-lv 4"
P=$PFX_Q run mxfp4/decode    "$QM" "$TOK64" 512 "48,$(ones 16)" "-lv 4"
P=$PFX_Q run mxfp4/pf64-nr   "$QM" "$TOK64" 512 ""               "-lv 4 --no-repack"
P=$PFX_Q run mxfp4/decode-nr "$QM" "$TOK64" 512 "48,$(ones 16)" "-lv 4 --no-repack"
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && MIMO2_DECODE_DUMP="$WORK/n16" \
  python3 -m unittest tests.test_mimo2_decode_parity tests.test_mimo2_mtp_parity tests.test_mimo2_chunked_ref tests.test_mimo2_quant_parity tests.test_mxfp4_ref -v
echo "artifacts: $WORK"
