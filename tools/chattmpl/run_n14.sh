#!/usr/bin/env bash
# N14 driver (gap 6 of docs/reassessment/CORRECTNESS.md): the vendored GLM-5.3-Flash and Qwen3.8-Flash-Next chat
# templates through llama-server's Jinja engine and chat code at the R1 pin, on CPU.
#
#   tools/chattmpl/run_n14.sh <llama.cpp checkout> <llama.cpp build> <N11 tinygen dir> <work dir>
#
# <llama.cpp checkout>  at the R1 pin (tools/engines/pins.json "llama.cpp"); its gguf-py writes the tiny models
# <llama.cpp build>     a shared-library CPU build of that checkout with bin/llama-server and bin/libllama-common.so
#                       (e.g. cmake -B build -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON; cmake --build build
#                       --target llama-server)
# <N11 tinygen dir>     tools/tinygen of branch claude/n11-r1-tools-cpu-server (text_vocab.py and the generators):
#                       git archive origin/claude/n11-r1-tools-cpu-server tools/tinygen | tar -x -C <dir>
#                       and pass <dir>/tools/tinygen
# <work dir>            scratch: models, harness, server logs, results.json (llama-server writes logs there)
#
# Steps: compile tools/chattmpl/chat_parse_harness.cpp against the build's libllama-common; write the bytes-only
# vocabulary tiny models (tools/chattmpl/bytes_vocab_model.py); run tools/chattmpl/run_n14.py (render, prefix,
# parse, live); run tests/test_chat_templates.py against the same binaries.
set -euo pipefail
if [ $# -ne 4 ]; then
    sed -n '2,20p' "$0"
    exit 2
fi
LLAMA=$(cd "$1" && pwd)
BUILD=$(cd "$2" && pwd)
TINYGEN=$(cd "$3" && pwd)
WORK=$4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

PIN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['engines']['llama.cpp']['sha'])" "$REPO/tools/engines/pins.json")
HEAD=$(git -C "$LLAMA" rev-parse HEAD)
echo "llama.cpp checkout $HEAD (pins.json llama.cpp: $PIN)"
if [ "$HEAD" != "$PIN" ]; then
    echo "the checkout is not at the R1 pin; the expectations of tests/test_chat_templates.py are for $PIN" >&2
    exit 1
fi
"$BUILD/bin/llama-server" --version 2>&1 | grep -E "^version" || true

echo "== harness"
g++ -std=c++17 -O1 -I"$LLAMA/common" -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/vendor" \
    "$REPO/tools/chattmpl/chat_parse_harness.cpp" -o "$WORK/chat-parse-harness" \
    -L"$BUILD/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$BUILD/bin"

echo "== models"
mkdir -p "$WORK/models"
for a in qwen4exp glm5next; do
    PYTHONPATH="$LLAMA/gguf-py" python3 "$REPO/tools/chattmpl/bytes_vocab_model.py" "$TINYGEN" "$a" \
        "$WORK/models/${a}_bytes.gguf"
done
sha256sum "$WORK/models/"*_bytes.gguf

echo "== run_n14.py"
cd "$WORK"
python3 "$REPO/tools/chattmpl/run_n14.py" --server-bin "$BUILD/bin/llama-server" \
    --glm-gguf "$WORK/models/glm5next_bytes.gguf" --qwen-gguf "$WORK/models/qwen4exp_bytes.gguf" \
    --harness "$WORK/chat-parse-harness" --out "$WORK/out"

echo "== tests/test_chat_templates.py"
cd "$REPO"
CHATTMPL_SERVER_BIN="$BUILD/bin/llama-server" CHATTMPL_HARNESS="$WORK/chat-parse-harness" \
CHATTMPL_GLM_GGUF="$WORK/models/glm5next_bytes.gguf" CHATTMPL_QWEN_GGUF="$WORK/models/qwen4exp_bytes.gguf" \
CHATTMPL_WORK="$WORK/test-out" python3 -m unittest -v tests.test_chat_templates
