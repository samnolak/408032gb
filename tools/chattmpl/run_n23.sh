#!/usr/bin/env bash
# N23 driver: the corrected GLM-5.3-Flash chat template (templates/glm-5.3-flash/chat_template.jinja) through
# llama-server's Jinja engine and chat code at the R1 pin, on CPU, against the official HF template under jinja2.
#
#   tools/chattmpl/run_n23.sh <llama.cpp checkout> <llama.cpp build> <N11 tinygen dir> <work dir>
#
# Arguments as tools/chattmpl/run_n14.sh (same pin check, harness, bytes-vocabulary tiny models). Steps: harness,
# models, tools/chattmpl/run_n23.py (equal, render, prefix, parse, live), then tests/test_glm_template_n23.py and
# tests/test_chat_templates.py (N14, unchanged expectations) against the same binaries.
set -euo pipefail
if [ $# -ne 4 ]; then
    sed -n '2,10p' "$0"
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
    echo "the checkout is not at the R1 pin; the expectations of tests/test_glm_template_n23.py are for $PIN" >&2
    exit 1
fi
"$BUILD/bin/llama-server" --version 2>&1 | grep -E "^version" || true
sha256sum "$REPO/templates/glm-5.3-flash/chat_template.jinja" "$REPO/third_party/hf/glm-5.3-flash-chat/chat_template.jinja"

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

echo "== run_n23.py"
cd "$WORK"
python3 "$REPO/tools/chattmpl/run_n23.py" --server-bin "$BUILD/bin/llama-server" \
    --glm-gguf "$WORK/models/glm5next_bytes.gguf" --harness "$WORK/chat-parse-harness" --out "$WORK/out" \
    | grep -v "got exception: .*type must be string, but is null"

echo "== tests/test_glm_template_n23.py tests/test_chat_templates.py"
cd "$REPO"
CHATTMPL_SERVER_BIN="$BUILD/bin/llama-server" CHATTMPL_HARNESS="$WORK/chat-parse-harness" \
CHATTMPL_GLM_GGUF="$WORK/models/glm5next_bytes.gguf" CHATTMPL_QWEN_GGUF="$WORK/models/qwen4exp_bytes.gguf" \
CHATTMPL_WORK="$WORK/test-out" python3 -m unittest -v tests.test_glm_template_n23 tests.test_chat_templates
