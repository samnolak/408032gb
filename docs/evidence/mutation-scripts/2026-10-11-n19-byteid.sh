#!/usr/bin/env bash
# N19: tools/dump/dump_tensors.cpp of this tree against the one of <base commit> on the plans that use no several-sequence
# step (the N2 default call and the N6 per-graph, rollback and MTP modes): every output file must be byte-identical.
# The base binary is compiled from `git show <base>:tools/dump/dump_tensors.cpp` with the compile flags and link line of
# <build dir> (a tools/g0 build of this tree with target dump-tensors), so both link the same llama.cpp libraries.
# usage: 2026-10-11-n19-byteid.sh <repo> <base commit> <build dir> <llama.cpp checkout> <out dir>
set -euo pipefail
REPO=$1; BASE=$2; B=$(cd "$3" && pwd); L=$(cd "$4" && pwd); OUT=$5
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
git -C "$REPO" show "$BASE:tools/dump/dump_tensors.cpp" > "$OUT/dump_tensors_base.cpp"
F=$B/CMakeFiles/dump-tensors.dir
DEFS=$(sed -n 's/^CXX_DEFINES = //p' "$F/flags.make"); INCS=$(sed -n 's/^CXX_INCLUDES = //p' "$F/flags.make")
c++ $DEFS $INCS -O3 -DNDEBUG -std=gnu++17 -c "$OUT/dump_tensors_base.cpp" -o "$OUT/base.o"
LINK=$(sed -e 's/^[^ ]* -O3 -DNDEBUG //' -e 's/"CMakeFiles[^"]*"//' -e 's/-o dump-tensors//' "$F/link.txt")
(cd "$B" && eval c++ -O3 -DNDEBUG "$OUT/base.o" -o "$OUT/dump-tensors-base" $LINK)
NEW=$B/dump-tensors; OLD=$OUT/dump-tensors-base
PYTHONPATH="$L/gguf-py" python3 "$REPO/tools/tinygen/qwen4exp_tiny.py" "$OUT/qwen4exp.gguf" >/dev/null
PYTHONPATH="$L/gguf-py" python3 "$REPO/tools/tinygen/qwen4exp_tiny.py" "$OUT/mtp.gguf" --mtp >/dev/null
TOK=$(python3 -c "import random; random.seed(6); t=[random.randrange(3,96) for _ in range(64)]
for p in (15, 33, 47, 55, 56, 59, 61): t[p] = 2
print(','.join(map(str,t)))")
TOK56=$(python3 -c "import sys; print(','.join(sys.argv[1].split(',')[:56]))" "$TOK")
PFX="hc_,ple_,linear_attn_,conv_output,attn_,final_output,indexer_,kq_mask_qsa,Qcur,Kcur,Vcur,ffn_,l_last,result_,conv_state_at,state_predelta,new_state,mtp_,h_nextn"
mode() { # mode <name> <model> <ub> <tokens> [env...]
  local name=$1 m=$2 ub=$3 tok=$4; shift 4
  for b in base new; do
    BIN=$OLD; [ $b = new ] && BIN=$NEW
    (cd "$OUT" && env DUMP_DIR="$OUT/$name-$b" DUMP_TOKENS="$tok" DUMP_PREFIXES="$PFX" "$@" "$BIN" -m "$m" -c 512 -b 512 \
      -ub "$ub" -t 1 -fa off -ctk f32 -ctv f32 > "$OUT/$name-$b.log" 2>&1) || echo "$name: $b exit status $?"
  done
  echo "$name: $(find "$OUT/$name-new" -type f | wc -l) files; diff -r: $(diff -r "$OUT/$name-base" "$OUT/$name-new" > /dev/null && echo identical || echo DIFFERENT)"
}
echo "llama.cpp $(git -C "$L" rev-parse --short HEAD); base dump_tensors.cpp @$BASE vs $(git -C "$REPO" rev-parse --short HEAD)"
mode default "$OUT/qwen4exp.gguf" 512 "$TOK"
mode ub10 "$OUT/qwen4exp.gguf" 10 "$TOK" DUMP_PER_GRAPH=1
mode rollback "$OUT/qwen4exp.gguf" 512 "$TOK" DUMP_PER_GRAPH=1 DUMP_CALLS="48,4,r2,1,1,1,1,1,1,1,1,1,1,1,1" DUMP_N_RS_SEQ=3
mode mtp "$OUT/mtp.gguf" 16 "$TOK56" DUMP_PER_GRAPH=1 DUMP_CALLS="48,1,1,1,1,1,1,1,1" DUMP_MTP=1 DUMP_MTP_DRAFT=4
