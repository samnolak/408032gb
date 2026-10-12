#!/usr/bin/env bash
# N20 on CPU, no GPU needed: the recurrent rollback guard (patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch)
# on llama.cpp qwen4exp. Applies the patch to a copy of the pinned checkout (patches/llama.cpp/apply.sh, at
# <build dir>-src, reused when it is already that copy), builds tools/dump/dump_tensors.cpp against it, runs the N2/N6
# tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0) through the nine N19 plans (tools/tinygen/qwen4exp_multiseq_plans.py)
# and the four N20 plans (tools/tinygen/qwen4exp_rs_guard_plans.py) exactly as run_parity_qwen4exp_multiseq.sh does,
# then the tests with QWEN4EXP_RS_GUARD=1:
#   tests.test_qwen4exp_multiseq_parity   N19's 63 dump tests; the guard changes only stale (both rollbacks refused, no
#                                         divergence), cp_pending (no divergence) and cp_shared_rb at ec7630a (refused)
#   tests.test_qwen4exp_rs_guard_parity   the four N20 runs (7 dump tests each) and the guard's refusal messages
#   tests.test_qwen4exp_multiseq_ref, tests.test_qwen4exp_rs_guard_ref   the oracle's own properties (numpy only)
# QWEN4EXP_RS_GUARD_MODE=upstream builds against the checkout as given (it must not carry the patch) and runs the same
# plans and tests without QWEN4EXP_RS_GUARD: the tests then expect the pin's own (silently wrong) behaviour.
# QWEN4EXP_RS_GUARD_MODE=combined builds against the checkout as given, which must be at the pin (HEAD) and carry the N20
# patch AND every patch named in QWEN4EXP_RS_GUARD_EXTRA_PATCHES (space-separated paths, e.g. the N21
# patches/llama.cpp/0001-mtp-heads-slot-replay.patch), checked with `git apply --reverse --check`; plans and tests as in
# patched mode (CI job n20-n21-combined of parity-r1-pin.yml: N20 and N21 on one copy).
# Usage:
#   tools/tinygen/run_parity_qwen4exp_rs_guard.sh <llama.cpp checkout @ec7630a or @79e2e74> [build dir]
# One build dir per mode (the patched copy and the checkout are different llama.cpp source trees).
# QWEN4EXP_RS_GUARD_SRC=<dir> puts the patched copy there instead of <build dir>-src (apply.sh reuses or creates it).
# QWEN4EXP_RS_GUARD_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
SRC=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools-rs-guard}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${QWEN4EXP_RS_GUARD_WORK:-$(mktemp -d)}
MODE=${QWEN4EXP_RS_GUARD_MODE:-patched}
mkdir -p "$WORK" "$BUILD"; BUILD=$(cd "$BUILD" && pwd)
SHA=$(git -C "$SRC" rev-parse HEAD)
PATCH=$ROOT/patches/llama.cpp/${SHA:0:7}/0001-recurrent-rollback-guard.patch
[ -f "$PATCH" ] || { echo "no N20 patch for llama.cpp $SHA" >&2; exit 1; }
case "$MODE" in
  patched)
    LLAMA=${QWEN4EXP_RS_GUARD_SRC:-$BUILD-src}
    bash "$ROOT/patches/llama.cpp/apply.sh" "$SRC" "$LLAMA"
    GUARD=1 ;;
  combined)
    for P in "$PATCH" ${QWEN4EXP_RS_GUARD_EXTRA_PATCHES:-}; do
      git -C "$SRC" apply --reverse --check "$P" || { echo "combined mode: $SRC does not carry $P" >&2; exit 1; }
    done
    LLAMA=$SRC
    GUARD=1 ;;
  upstream)
    if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
      echo "QWEN4EXP_RS_GUARD_MODE=upstream: $SRC carries the N20 patch" >&2; exit 1
    fi
    LLAMA=$SRC
    GUARD=0 ;;
  *) echo "QWEN4EXP_RS_GUARD_MODE must be patched, combined or upstream" >&2; exit 1 ;;
esac
echo "mode $MODE: llama.cpp $SHA, sources $LLAMA ($(git -C "$LLAMA" diff --stat | tail -n 1 | sed 's/^ *//'))"
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
export LLAMA_CPP_GGUF_PY="$SRC/gguf-py"
rm -rf "$WORK/n20"; mkdir -p "$WORK/n20"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$WORK/n20/qwen4exp.gguf"
# as run_parity_qwen4exp_multiseq.sh: the carried states, every GDN / PLE / attention / QSA / MoE component, the
# attention probabilities over all cells and the head
PFX="hc_init,hc_combine,ple_,linear_attn_out,conv_output,attn_,final_output,indexer_,kq_mask_qsa,kq_soft_max,Qcur,Kcur,Vcur,ffn_moe_topk,ffn_out,l_last,result_,conv_state_at,state_predelta,new_state"
for PLANS in "$ROOT/tools/tinygen/qwen4exp_multiseq_plans.py" "$ROOT/tools/tinygen/qwen4exp_rs_guard_plans.py"; do
  for run in $(python3 "$PLANS" --list); do
    mapfile -t L < <(python3 "$PLANS" "$run")
    read -r NP KVU RS UB <<< "${L[2]}"
    d="$WORK/n20/$run"
    env DUMP_DIR="$d" DUMP_TOKENS="${L[1]}" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$PFX" DUMP_CALLS="${L[0]}" DUMP_KV_UNIFIED="$KVU" \
      DUMP_N_RS_SEQ="$RS" "$BUILD/dump-tensors" -m "$WORK/n20/qwen4exp.gguf" -c 512 -b 512 -ub "$UB" -np "$NP" -t 1 \
      -fa off -ctk f32 -ctv f32 >"$WORK/n20/$run.log" 2>&1 || { tail -n 20 "$WORK/n20/$run.log"; exit 1; }
    echo "${L[1]}" > "$d/tokens.txt"
    echo "$run: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); c=m['calls']
print(len(m['graphs']), 'graphs,', len(c), 'steps, seq_rm ok:', [x['ok'] for x in c if x['kind'] == 'seq_rm'])" "$d/manifest.json"), guard refusals logged: $(grep -c 'refusing a rollback' "$WORK/n20/$run.log" || true)"
  done
done
# the commit decides which pin's behaviour the tests expect; QWEN4EXP_RS_GUARD=1: the patched behaviour
export PARITY_LLAMA_COMMIT="$SHA"
if [ "$GUARD" = 1 ]; then export QWEN4EXP_RS_GUARD=1; else unset QWEN4EXP_RS_GUARD; fi
cd "$ROOT" && QWEN4EXP_MULTISEQ_DUMP="$WORK/n20" QWEN4EXP_TINY_GGUF="$WORK/n20/qwen4exp.gguf" \
  python3 -m unittest tests.test_qwen4exp_multiseq_parity tests.test_qwen4exp_rs_guard_parity tests.test_qwen4exp_multiseq_ref tests.test_qwen4exp_rs_guard_ref -v
echo "artifacts: $WORK"
