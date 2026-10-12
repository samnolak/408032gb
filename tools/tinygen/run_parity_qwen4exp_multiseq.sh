#!/usr/bin/env bash
# N19 (gap 10 of docs/reassessment/CORRECTNESS.md) on CPU, no GPU needed: several sequences in one llama.cpp qwen4exp
# context, seq_cp, seq_rm and rollback edges. The N2/N6 tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0) runs the
# nine plans of tools/tinygen/qwen4exp_multiseq_plans.py through the real qwen4exp graph (f32 caches, flash attention
# off, -c 512 -b 512 -t 1), dumped graph by graph by tools/dump/dump_tensors.cpp (DUMP_PER_GRAPH, the several-sequence
# DUMP_CALLS steps, DUMP_KV_UNIFIED, DUMP_N_RS_SEQ):
#   multi_uni / multi_str    3 sequences (-np 3), one KV stream (kv_unified) / one stream per sequence: an unequal
#                            prefill in one call, interleaved one-token decode, an idle sequence, other batch orders
#   cp_uni / cp_str          seq_cp 0->1 and 0->2, both continue, drop 1 and 2, copy again
#   rb_uni / rb_str          n_rs_seq 3, -ub 8: rollbacks of 1-3 positions per sequence, a refused rollback > n_rs_seq,
#                            a refused second (pending) rollback, a rollback inside a two-ubatch call
#   stale                    rollbacks deeper than the last call's last ubatch (accepted by llama.cpp; stale snapshots)
#   cp_pending               seq_cp while a rollback is pending (the copy gets the state before the rollback)
#   cp_shared_rb             a rollback of one of two sequences sharing a recurrent cell (ec7630a accepts, 79e2e74 not)
# -> tests/test_qwen4exp_multiseq_parity.py (oracle: ref/qwen4exp.py MultiSeqContext and forward) and
#    tests/test_qwen4exp_multiseq_ref.py (the oracle's own properties, numpy only).
# Usage:
#   tools/tinygen/run_parity_qwen4exp_multiseq.sh <llama.cpp checkout @ec7630a, or the R1 pin of tools/engines/pins.json> [build dir]
# QWEN4EXP_MULTISEQ_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${QWEN4EXP_MULTISEQ_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
rm -rf "$WORK/n19"; mkdir -p "$WORK/n19"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$WORK/n19/qwen4exp.gguf"
PLANS="$ROOT/tools/tinygen/qwen4exp_multiseq_plans.py"
# what the tests read: the carried states, every GDN / PLE / attention / QSA / MoE component, the attention
# probabilities over all cells (cross-sequence leakage) and the head
PFX="hc_init,hc_combine,ple_,linear_attn_out,conv_output,attn_,final_output,indexer_,kq_mask_qsa,kq_soft_max,Qcur,Kcur,Vcur,ffn_moe_topk,ffn_out,l_last,result_,conv_state_at,state_predelta,new_state"
for run in $(python3 "$PLANS" --list); do
  mapfile -t L < <(python3 "$PLANS" "$run")
  read -r NP KVU RS UB <<< "${L[2]}"
  d="$WORK/n19/$run"
  env DUMP_DIR="$d" DUMP_TOKENS="${L[1]}" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$PFX" DUMP_CALLS="${L[0]}" DUMP_KV_UNIFIED="$KVU" \
    DUMP_N_RS_SEQ="$RS" "$BUILD/dump-tensors" -m "$WORK/n19/qwen4exp.gguf" -c 512 -b 512 -ub "$UB" -np "$NP" -t 1 \
    -fa off -ctk f32 -ctv f32 >"$WORK/n19/$run.log" 2>&1 || { tail -n 20 "$WORK/n19/$run.log"; exit 1; }
  echo "${L[1]}" > "$d/tokens.txt"
  echo "$run: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); c=m['calls']
print(len(m['graphs']), 'graphs,', len(c), 'steps, seq_rm ok:', [x['ok'] for x in c if x['kind'] == 'seq_rm'])" "$d/manifest.json") (${L[3]})"
done
# the commit decides which pin's behaviour the tests expect (tools/tinygen/qwen4exp_multiseq_plans.py RUNS)
export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
cd "$ROOT" && QWEN4EXP_MULTISEQ_DUMP="$WORK/n19" QWEN4EXP_TINY_GGUF="$WORK/n19/qwen4exp.gguf" \
  python3 -m unittest tests.test_qwen4exp_multiseq_parity tests.test_qwen4exp_multiseq_ref -v
echo "artifacts: $WORK"
