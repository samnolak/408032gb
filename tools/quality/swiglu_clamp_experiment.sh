#!/usr/bin/env bash
# SwiGLU clamp semantics of GLM-5.3-Flash on real weights (N24 owner decision 3). Two builds of llama.cpp @79e2e74 that
# differ only in the clamped SwiGLU:
#   pre   stock: silu(min(g, L)) * clamp(u, -L, L)   = HF transformers 5.17-5.19 modeling_glm5_next.py, vLLM @bc21cba
#   post  patches/llama.cpp/n24-swiglu-clamp-after-silu.patch: min(silu(g), L) * clamp(u, -L, L) = ik_llama.cpp, exllamav3
# Everything else (weights, graph, kernels, flags, corpus, chunks) is equal, so every difference below is the clamp.
#
# Steps (each prints its exact commands; nothing is downloaded; outputs only under the given directories):
#   build  <llama.cpp checkout containing 79e2e74> <work dir> [extra cmake args, e.g. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89]
#            -> <work>/src-{pre,post} (git archive of 79e2e74, post patched), <work>/build-{pre,post}/bin/{llama-perplexity,
#               llama-server,llama-tokenize}, <work>/build-pre-dump/bin/dump-tensors (tools/g0), <work>/variants.txt
#   kld    <work dir> <out dir> <model.gguf (first shard)> <corpus.txt> [<reference.gguf, BF16 or Q8_0, first shard>]
#            with a reference: per variant v, the reference run with build-v saves full-vocabulary logits
#            (llama-perplexity --kl-divergence-base ref-v.kld), then the model with build-pre and build-post is compared
#            with both reference files (--kl-divergence): 2 reference runs + 4 comparisons.
#            without a reference: the model with build-pre saves q-pre.kld, the model with build-post is compared with it
#            (size of the clamp effect on this GGUF) and both report perplexity on the corpus.
#   gate   <work dir> <out dir> <model.gguf> <corpus.txt> [limit, default 10]
#            dump-tensors (stock build) on the first CTX tokens of the corpus, prefixes ffn_moe_gate, ffn_gate, ffn_moe_up,
#            ffn_up; tools/quality/swiglu_gate_stats.py counts gate pre-activations above L (the only place the two
#            semantics differ) and the largest difference of the two SwiGLU outputs.
# Environment: RUN_ARGS (engine flags for every llama run, e.g. "-ngl 99 -ot exps=CPU -t 32 -fa on"), CTX (512),
#   CHUNKS (32), BATCH (= CTX: one sequence per batch; glm5-next has recurrent KDA state).
# Reading the result (docs/evidence/2026-10-11-n24-owner-decisions.md section 3):
#   * against a reference produced by the SAME variant the KLD measures the quantisation, not the clamp;
#   * a reference from HF transformers / vLLM would be "pre" by construction (both implement pre), so it can only show
#     which llama.cpp build reproduces those implementations, not what GLM-5.3-Flash was trained with;
#   * the discriminating number is the perplexity on natural text of the same weights under the two semantics, together
#     with the gate statistics: if no gate exceeds L the two semantics are identical on this corpus (no evidence either way).
set -euo pipefail
PIN=79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PATCH="$ROOT/patches/llama.cpp/n24-swiglu-clamp-after-silu.patch"
CTX=${CTX:-512}; CHUNKS=${CHUNKS:-32}; BATCH=${BATCH:-$CTX}; RUN_ARGS=${RUN_ARGS:-}
say() { printf '+ %s\n' "$*" >&2; }
run() { say "$*"; "$@"; }

cmd=${1:-}; shift || true
case "$cmd" in
build)
  LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); WORK=${2:?work dir}; shift 2
  mkdir -p "$WORK"
  for v in pre post; do
    [ -e "$WORK/src-$v" ] && { echo "$WORK/src-$v exists; use a new work dir" >&2; exit 2; }
    mkdir -p "$WORK/src-$v"
    say "git -C $LLAMA archive $PIN | tar -x -C $WORK/src-$v"
    git -C "$LLAMA" archive "$PIN" | tar -x -C "$WORK/src-$v"
    if [ "$v" = post ]; then (cd "$WORK/src-$v" && run patch -p1 --forward < "$PATCH"); fi
    run cmake -S "$WORK/src-$v" -B "$WORK/build-$v" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF "$@"
    run cmake --build "$WORK/build-$v" -j"$(nproc)" --target llama-perplexity llama-server llama-tokenize
  done
  run cmake -S "$ROOT/tools/g0" -B "$WORK/build-pre-dump" -DLLAMA_CPP_DIR="$WORK/src-pre" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF "$@"
  run cmake --build "$WORK/build-pre-dump" -j"$(nproc)" --target dump-tensors
  {
    echo "pin $PIN; patch $(sha256sum "$PATCH" | cut -d' ' -f1) $PATCH"
    for v in pre post; do
      for f in ggml/src/ggml-cpu/ops.cpp ggml/src/ggml-cuda/unary.cuh; do
        echo "$v $(sha256sum "$WORK/src-$v/$f" | cut -d' ' -f1) $f"
      done
      grep -n "N24 experiment" "$WORK/src-$v/ggml/src/ggml-cpu/ops.cpp" "$WORK/src-$v/ggml/src/ggml-cuda/unary.cuh" \
        | sed "s|^|$v marker |" || echo "$v no N24 marker (stock)"
    done
  } | tee "$WORK/variants.txt"
  ;;
kld)
  WORK=${1:?work dir}; OUT=${2:?out dir}; MODEL=${3:?model gguf}; CORPUS=${4:?corpus}; REF=${5:-}
  mkdir -p "$OUT"
  ppl() { # <variant> <model> <log> [extra args]
    local v=$1 m=$2 log=$3; shift 3
    say "$WORK/build-$v/bin/llama-perplexity -m $m -f $CORPUS -c $CTX -b $BATCH --chunks $CHUNKS $RUN_ARGS $* > $log 2>&1"
    # shellcheck disable=SC2086
    "$WORK/build-$v/bin/llama-perplexity" -m "$m" -f "$CORPUS" -c "$CTX" -b "$BATCH" --chunks "$CHUNKS" $RUN_ARGS "$@" \
      > "$log" 2>&1 || { echo "llama-perplexity failed, see $log" >&2; tail -n 20 "$log" >&2; exit 1; }
    grep -E "Final estimate|Mean +KLD|Same top p|RMS Δp|Mean +ln\(PPL\(Q\)/PPL\(base\)\)" "$log" | sed "s|^|$(basename "$log"): |" || true
  }
  if [ -n "$REF" ]; then
    for r in pre post; do
      ppl "$r" "$REF" "$OUT/ref-$r.log" --kl-divergence-base "$OUT/ref-$r.kld"
      for v in pre post; do
        ppl "$v" "$MODEL" "$OUT/q-$v-vs-ref-$r.log" --kl-divergence-base "$OUT/ref-$r.kld" --kl-divergence
      done
    done
  else
    ppl pre "$MODEL" "$OUT/q-pre.log" --kl-divergence-base "$OUT/q-pre.kld"
    ppl post "$MODEL" "$OUT/q-post-vs-q-pre.log" --kl-divergence-base "$OUT/q-pre.kld" --kl-divergence
  fi
  sha256sum "$CORPUS" | tee "$OUT/corpus.sha256"
  ;;
gate)
  WORK=${1:?work dir}; OUT=${2:?out dir}; MODEL=${3:?model gguf}; CORPUS=${4:?corpus}; LIMIT=${5:-10}
  mkdir -p "$OUT/dump"
  say "$WORK/build-pre/bin/llama-tokenize -m $MODEL -f $CORPUS --ids --log-disable"
  TOK=$("$WORK/build-pre/bin/llama-tokenize" -m "$MODEL" -f "$CORPUS" --ids --log-disable 2>/dev/null \
        | tr -d '[] ' | tr ',' '\n' | head -n "$CTX" | paste -sd, -)
  [ -n "$TOK" ] || { echo "tokenizer gave no ids" >&2; exit 1; }
  say "DUMP_DIR=$OUT/dump DUMP_TOKENS=<first $CTX ids> DUMP_PREFIXES=ffn_moe_gate,ffn_moe_up,ffn_gate,ffn_up $WORK/build-pre-dump/bin/dump-tensors -m $MODEL -c $CTX -b $CTX -ub $CTX $RUN_ARGS"
  # shellcheck disable=SC2086
  DUMP_DIR="$OUT/dump" DUMP_TOKENS="$TOK" DUMP_PREFIXES="ffn_moe_gate,ffn_moe_up,ffn_gate,ffn_up" \
    "$WORK/build-pre-dump/bin/dump-tensors" -m "$MODEL" -c "$CTX" -b "$CTX" -ub "$CTX" $RUN_ARGS > "$OUT/dump.log" 2>&1
  run python3 "$ROOT/tools/quality/swiglu_gate_stats.py" "$OUT/dump" --limit "$LIMIT" --out "$OUT/gate_stats.json"
  ;;
*)
  sed -n '2,32p' "$0" >&2; exit 2;;
esac
