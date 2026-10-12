#!/usr/bin/env bash
# N12 on CPU, no GPU needed (gap 5 of docs/reassessment/CORRECTNESS.md): the G1/N9 tiny GLM model written in the
# dialect ik_llama.cpp is run on in R1 (tools/tinygen/glm5next_tiny.py --dialect ik: architecture "glm5next"), run by
# ik_llama.cpp at the pin of tools/engines/pins.json, dumped graph by graph with tools/dump/ik_dump_tensors.cpp, and
# compared with ref/glm5next_ik.py (ref/glm5next.py plus ik's documented deviations) by tests/test_ik_glm5next_parity.py.
# Usage:
#   LLAMA_CPP_GGUF_PY=<llama.cpp checkout>/gguf-py tools/tinygen/run_parity_ik_glm5next.sh <ik_llama.cpp checkout> [build dir]
# The ik checkout must be at 5194a9e1451b885e24dd5a065f650864e8807c59 (checked). IK_GLM5NEXT_WORK=<dir> keeps the
# artifacts in <dir> instead of a fresh temporary directory.
set -euo pipefail
IK=$(cd "${1:?ik_llama.cpp checkout}" && pwd); BUILD=${2:-build-ik}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${IK_GLM5NEXT_WORK:-$(mktemp -d)}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
: "${LLAMA_CPP_GGUF_PY:?set LLAMA_CPP_GGUF_PY to a llama.cpp gguf-py (the writer and the test read GGUF with it)}"
PIN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['engines']['ik_llama.cpp']['sha'])" \
  "$ROOT/tools/engines/pins.json")
HEAD=$(git -C "$IK" rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "ik_llama.cpp checkout is at $HEAD, the pin is $PIN"; exit 1; }
mkdir -p "$WORK"
# pins.json flags without the four CUDA ones (tools/ik/CMakeLists.txt)
cmake -S "$ROOT/tools/ik" -B "$BUILD" -DIK_LLAMA_DIR="$IK" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target ik-dump-tensors >/dev/null
grep -E "^GGML_(AVX2|AVX512|NATIVE|CUDA):" "$BUILD/CMakeCache.txt" | tr '\n' ' '; echo
DUMP="$(cd "$BUILD" && pwd)/ik-dump-tensors"
# the G1 / N9 model (seed 0, 12 layers: KDA, DSA at 3/7/11, a dense lead layer, MoE) in both dialects
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/ik.gguf" --layers 12 --dialect ik
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/llama.gguf" --layers 12
# the 64 ids of tools/tinygen/run_parity_glm5next_decode.sh
TOK=$(python3 -c "import random; random.seed(9); print(','.join(str(random.randrange(3,96)) for _ in range(64)))")
ONES16=$(python3 -c "print(','.join(['1']*16))")
rm -rf "$WORK/n12"; mkdir -p "$WORK/n12"; echo "$TOK" > "$WORK/n12/tokens.txt"
PFX="inp_embd,hc_init,hc_attn_pre-,attn_norm-,linear_attn_out-,new_state-,kv_compressed-,kv_cache-,Qcur-"
PFX="$PFX,dsa_indexer_k-,dsa_indexer_gate-,dsa_cached_all-,dsa_indexer_k_pooled-,dsa_indexer_q-,dsa_indexer_weights-"
PFX="$PFX,dsa_indexer_score-,dsa_top_k-,kqv_out-,attn_out-,ffn_norm-,ffn_out-,ffn_moe_out-,ffn_shexp-,l_out-,hc_collapse"
run() {   # run <name> <-ub> <DUMP_CALLS or ""> <ik flags...>; model in $MODEL
  local name=$1 ub=$2 calls=$3; shift 3
  # run from $WORK/n12: ik's logger writes llama.log into the current directory
  (cd "$WORK/n12" && env DUMP_DIR="$WORK/n12/$name" DUMP_TOKENS="$TOK" DUMP_PREFIXES="$PFX" DUMP_CALLS="$calls" \
    "$DUMP" -m "${MODEL:-$WORK/ik.gguf}" -c 512 -b 512 -ub "$ub" -t 1 "$@" >"$WORK/n12/$name.log" 2>&1) \
    || { tail -n 20 "$WORK/n12/$name.log"; exit 1; }
  echo "$* " > "$WORK/n12/$name/flags.txt"
  echo "$name: $(tail -n 1 "$WORK/n12/$name.log") [$*]"
}
# ik keeps no f32 MLA latent for both K and V on the CPU: with -fa off and -mla 2/3 (default 3) the latent takes the
# -ctv type, which must be f16 or bf16 (src/llama.cpp:1496-1497, 9257-9259); with -mla 1 -fa off K can be f32 but V is
# read from a transposed -ctv copy (:1513-1516; not run here); with -fa on an f32 K/V cache aborts
# (ggml/src/iqk/iqk_flash_attn.cpp:280-298). The closest to the reference contract used here: -fa off, -ctk f32,
# -ctv f16 (the latent in f16, default -mla 3), indexer cache f32 (-ictk).
REF=(-fa off -ctk f32 -ctv f16 -ictk f32)
run p64      64  ""             "${REF[@]}" --dsa      # one 64-token ubatch, DSA k-pool selection on (opt-in, --dsa)
run ub16     16  ""             "${REF[@]}" --dsa      # one call, 4 ubatches
run decode   512 "48,$ONES16"   "${REF[@]}" --dsa      # prefill 48, then 16 calls of one token
run dense    64  ""             "${REF[@]}"            # without --dsa: dense MLA (ik's default)
run ictk16   64  ""             -fa off -ctk f32 -ctv f16 -ictk f16 --dsa   # indexer cache f16 (ik's default -ictk)
# the precision of the R1 template (scripts/r1/configs/ik-llamacpp-server.json: -fa on, -ctk/-ctv f16, -ictk default)
run fa16     64  ""             -fa on -ctk f16 -ctv f16 --dsa
run fa16dec  512 "48,$ONES16"   -fa on -ctk f16 -ctv f16 --dsa
# the same weights under the llama.cpp name "glm5-next" (ik's alias): logits only
MODEL="$WORK/llama.gguf" PFX="" run alias 64 "" "${REF[@]}" --dsa
export PARITY_IK_COMMIT="$HEAD"
cd "$ROOT" && IK_GLM5NEXT_DUMP="$WORK/n12" IK_GLM5NEXT_GGUF="$WORK/ik.gguf" \
  python3 -m unittest tests.test_ik_glm5next_parity -v
echo "artifacts: $WORK"
