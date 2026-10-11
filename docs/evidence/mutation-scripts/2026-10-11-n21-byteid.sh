#!/usr/bin/env bash
# N21: tools/dump/dump_tensors.cpp of this tree against the one of <base commit> (the branch point, e71b241) on the modes
# N21 did not add: N17's three-head MTP plan (mtp3) and N21's several-ubatch plan (mtp3b) with DUMP_MTP=1 (the replay of
# the upstream order), f32 and f16 / flash attention, plus N19's byte-identity modes (qwen4exp default, per-graph,
# rollback, one-head MTP; docs/evidence/mutation-scripts/2026-10-11-n19-byteid.sh). Every output file must be
# byte-identical. The base binary is compiled from `git show <base>:tools/dump/dump_tensors.cpp` with the compile flags
# and link line of <build dir>, so both link the same llama.cpp libraries.
# usage: 2026-10-11-n21-byteid.sh <repo> <base commit> <build dir> <llama.cpp checkout> <out dir>
set -euo pipefail
REPO=$1; BASE=$2; B=$(cd "$3" && pwd); L=$(cd "$4" && pwd); OUT=$5
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
bash "$REPO/docs/evidence/mutation-scripts/2026-10-11-n19-byteid.sh" "$REPO" "$BASE" "$B" "$L" "$OUT/n19"
OLD=$OUT/n19/dump-tensors-base; NEW=$B/dump-tensors
PYTHONPATH="$L/gguf-py" python3 "$REPO/tools/tinygen/mimo2_tiny.py" "$OUT/n17.gguf" --fused-qkv --sinks swa \
  --value-scale 0.75 --mtp-layers 3 >/dev/null
TOK=$(python3 -c "import random; random.seed(16); print(','.join(str(random.randrange(3,96)) for _ in range(320)))")
seq_py() { python3 -c "import sys; t=[int(x) for x in sys.argv[1].split(',')]; w=lambda xs: [(x-3+17)%93+3 for x in xs]
$1
print(','.join(map(str,s)))" "$TOK"; }
ones() { python3 -c "import sys; print(','.join(['1']*int(sys.argv[1])))" "$1"; }
TOK_MTP=$(seq_py "s=t[:250]+w(t[250:252])+t[250:253]+w(t[253:254])+t[253:260]+w(t[260:263])+t[260:266]")
PLAN_MTP="240,$(ones 8),d3,4,r2,d3,4,r1,d3,4,1,1,d3,4,r3,$(ones 6)"
TOK_B=$(seq_py "s=t[:310]+w(t[310:311])+t[310:312]")
PLAN_B="40,$(ones 4),260,$(ones 3),d3,4,r1,$(ones 2)"
PFX="embd,attn_norm,wqkv,Qcur,Kcur,Vcur,kq,attn_out,ffn_inp,ffn_norm,ffn_up,ffn_gate,ffn_swiglu,ffn_out,ffn_moe_logits,ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_out,l_out,result_,h_nextn,mtp_,cache_k_l,cache_v_l"
mode() { # mode <name> <tokens> <plan> <flags...>
  local name=$1 tok=$2 plan=$3; shift 3
  for b in base new; do
    BIN=$OLD; [ $b = new ] && BIN=$NEW
    (cd "$OUT" && env DUMP_DIR="$OUT/$name-$b" DUMP_TOKENS="$tok" DUMP_PREFIXES="$PFX" DUMP_PER_GRAPH=1 DUMP_CALLS="$plan" \
      DUMP_MTP=1 DUMP_MTP_DRAFT=3 "$BIN" -m "$OUT/n17.gguf" -c 512 -b 512 -ub 64 -t 1 "$@" > "$OUT/$name-$b.log" 2>&1) \
      || echo "$name: $b exit status $?"
  done
  echo "$name: $(find "$OUT/$name-new" -type f | wc -l) files; diff -r: $(diff -r "$OUT/$name-base" "$OUT/$name-new" > /dev/null && echo identical || echo DIFFERENT)"
}
mode mtp3 "$TOK_MTP" "$PLAN_MTP" -fa off -ctk f32 -ctv f32
mode mtp3b "$TOK_B" "$PLAN_B" -fa off -ctk f32 -ctv f32
mode fa-mtp3 "$TOK_MTP" "$PLAN_MTP" -fa on -ctk f16 -ctv f16
