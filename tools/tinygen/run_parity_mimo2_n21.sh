#!/usr/bin/env bash
# N21 (docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md) on CPU, no GPU needed: the mimo2 MTP-heads finding of N17
# (head 0 of a multi-head MTP context reads K/V of other positions after a process call that wraps the SWA ring) and
# its fix, patches/llama.cpp/0001-mtp-heads-slot-replay.patch (common/speculative.cpp process() places each ubatch once
# for all heads; llama_memory_set_slot_replay). The model is N17's n17.gguf (tools/tinygen/mimo2_tiny.py --fused-qkv
# --sinks swa --value-scale 0.75 --mtp-layers 3), -c 512 -ub 64 -t 1, two plans:
#   mtp3   N17's plan (240, 8 x 1, d3, 4, r2, d3, 4, r1, d3, 4, 1, 1, d3, 4, r3, 6 x 1, DUMP_MTP_DRAFT=3): the verify
#          call at 253..256 wraps the 256-cell MTP SWA ring (head 0 then reads foreign K/V at 257..263);
#   mtp3b  40, 4 x 1, 260 (5 ubatches: head 0's fourth wraps the ring and takes the cells of positions 0..43, which the
#          first ubatch's window needs before heads 1 and 2 run it), 3 x 1, d3, 4, r1, 2 x 1, DUMP_MTP_DRAFT=3.
# Runs (every one with DUMP_MTP=1, f32 caches, flash attention off unless named fa):
#   <plan>             the upstream order replayed by tools/dump/dump_tensors.cpp (as N17)
#   spec/<plan>        the same plan through llama.cpp's own common/speculative.cpp (DUMP_MTP_SPEC=1): at an
#                      unpatched pin the upstream code, with the patch the fixed one
# and, only when the checkout carries the patch (src/llama-ext.h defines LLAMA_EXT_HAS_SLOT_REPLAY):
#   fix/<plan>         the patched order replayed (DUMP_MTP_FIX=1)
#   fix/fa-mtp3        the same at the R1 default precision (-ctk f16 -ctv f16 -fa on)
#   spec-off/<plan>    the patched common/speculative.cpp with LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1 (the upstream order)
# -> tests/test_mimo2_mtp_fix_parity.py (the fixed runs against ref/mimo2.py's fixed process and the full-sequence
# oracle; spec runs bit for bit against the replay of the same order), tests/test_mimo2_mtp_fix_ref.py (numpy only).
# Usage:
#   tools/tinygen/run_parity_mimo2_n21.sh <llama.cpp checkout: @ec7630a, @79e2e74, or a copy with the patch> [build dir]
# MIMO2_N21_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${MIMO2_N21_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PATCHED=false
grep -q 'LLAMA_EXT_HAS_SLOT_REPLAY' "$LLAMA/src/llama-ext.h" && PATCHED=true
rm -rf "$WORK/n21"; mkdir -p "$WORK/n21"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/mimo2_tiny.py" "$WORK/n21/n17.gguf" \
  --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 3
# N16's ids extended to 320 (random.seed(16): the first 300 are N16's and N17's)
TOK=$(python3 -c "import random; random.seed(16); print(','.join(str(random.randrange(3,96)) for _ in range(320)))")
echo "$TOK" > "$WORK/n21/tokens.txt"
seq_py() { python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]; w=lambda xs: [(x-3+17)%93+3 for x in xs]
$1
print(','.join(map(str,s)))" "$TOK"; }
ones() { python3 -c "import sys; print(','.join(['1']*int(sys.argv[1])))" "$1"; }
# mtp3: N17's tokens and plan (tools/tinygen/run_parity_mimo2_n17.sh)
TOK_MTP=$(seq_py "s=t[:250]+w(t[250:252])+t[250:253]+w(t[253:254])+t[253:260]+w(t[260:263])+t[260:266]")
PLAN_MTP="240,$(ones 8),d3,4,r2,d3,4,r1,d3,4,1,1,d3,4,r3,$(ones 6)"
# mtp3b: the verify call at 307..310 carries one wrong id
TOK_B=$(seq_py "s=t[:310]+w(t[310:311])+t[310:312]")
PLAN_B="40,$(ones 4),260,$(ones 3),d3,4,r1,$(ones 2)"
PFX="embd,attn_norm,wqkv,Qcur,Kcur,Vcur,kq,attn_out,ffn_inp,ffn_norm,ffn_up,ffn_gate,ffn_swiglu,ffn_out,ffn_moe_logits,ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_out,l_out,result_,h_nextn,mtp_"
run() {   # run <name> <tokens> <flags> <DUMP_CALLS>; extra env in $EXTRA, prefixes $P
  local d="$WORK/n21/$1"
  env DUMP_DIR="$d" DUMP_TOKENS="$2" DUMP_PER_GRAPH=1 DUMP_PREFIXES="${P:-$PFX}" DUMP_CALLS="$4" \
    DUMP_MTP=1 DUMP_MTP_DRAFT=3 ${EXTRA:-} \
    "$BUILD/dump-tensors" -m "$WORK/n21/n17.gguf" -c 512 -b 512 -ub 64 -t 1 $3 \
    >"$WORK/n21/$(echo "$1" | tr / _).log" 2>&1 || { tail -n 20 "$WORK/n21/$(echo "$1" | tr / _).log"; exit 1; }
  echo "$2" > "$d/tokens.txt"
  echo "$1: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); k=[c['kind'] for c in m['calls']]; print(len(m['graphs']), 'MTP graphs,', {x: k.count(x) for x in sorted(set(k))})" "$d/mtp/manifest.json") (${EXTRA:-replay})"
}
F32="-fa off -ctk f32 -ctv f32"; F16FA="-fa on -ctk f16 -ctv f16"
run mtp3  "$TOK_MTP" "$F32" "$PLAN_MTP"
run mtp3b "$TOK_B"   "$F32" "$PLAN_B"
EXTRA="DUMP_MTP_SPEC=1" run spec/mtp3  "$TOK_MTP" "$F32" "$PLAN_MTP"
EXTRA="DUMP_MTP_SPEC=1" run spec/mtp3b "$TOK_B"   "$F32" "$PLAN_B"
if $PATCHED; then
  EXTRA="DUMP_MTP_FIX=1" run fix/mtp3  "$TOK_MTP" "$F32" "$PLAN_MTP"
  EXTRA="DUMP_MTP_FIX=1" run fix/mtp3b "$TOK_B"   "$F32" "$PLAN_B"
  P="$PFX,cache_k_l,cache_v_l" EXTRA="DUMP_MTP_FIX=1" run fix/fa-mtp3 "$TOK_MTP" "$F16FA" "$PLAN_MTP"
  EXTRA="DUMP_MTP_SPEC=1 LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1" run spec-off/mtp3  "$TOK_MTP" "$F32" "$PLAN_MTP"
  EXTRA="DUMP_MTP_SPEC=1 LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1" run spec-off/mtp3b "$TOK_B"   "$F32" "$PLAN_B"
fi
echo "{\"patched\": $PATCHED}" > "$WORK/n21/n21.json"
echo "patched: $PATCHED"
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD 2>/dev/null || echo unknown)
cd "$ROOT" && MIMO2_N21_DUMP="$WORK/n21" python3 -m unittest tests.test_mimo2_mtp_fix_parity tests.test_mimo2_mtp_fix_ref -v
echo "artifacts: $WORK"
