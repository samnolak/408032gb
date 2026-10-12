#!/usr/bin/env bash
# N11 on CPU, no GPU needed: the measurement tools of R1 (tools/bench, tools/quality/logprob_compare.py, tools/gateway,
# tools/kv/slot_persist.py, scripts/r1/run_r1.py + tools/verify/pack.py) against the real llama-server of R1 (llama.cpp
# pinned in tools/engines/pins.json) on a tiny random-weight qwen4exp with a text vocabulary
# (tools/tinygen/qwen4exp_tiny.py --text-vocab) -> tools/srvcheck/r1_tools_smoke.py. Checks formats and protocols only:
# no score, no speed. Any failed check fails this script.
# Usage: tools/srvcheck/run_n11.sh <llama.cpp checkout at the pins.json commit> [build dir] [output dir]
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-$LLAMA/build-n7}; OUT=${3:-$(mktemp -d)}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['engines']['llama.cpp']['sha'])" "$ROOT/tools/engines/pins.json")
HEAD=$(git -C "$LLAMA" rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "llama.cpp checkout is $HEAD, pins.json says $PIN" >&2; exit 1; }
echo "llama.cpp $HEAD (pins.json)"
cmake -S "$LLAMA" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_OPENSSL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target llama-server >/dev/null
SERVER="$BUILD/bin/llama-server"; "$SERVER" --version 2>&1 | head -n 1
mkdir -p "$OUT"
PYTHONPATH="$LLAMA/gguf-py" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$OUT/qwen4exp_text.gguf" --text-vocab --ctx 262144
sha256sum "$OUT/qwen4exp_text.gguf"
t0=$(date +%s)
rc=0
python3 "$ROOT/tools/srvcheck/r1_tools_smoke.py" --server "$SERVER" --model "$OUT/qwen4exp_text.gguf" \
  --workdir "$OUT/work" --out "$OUT/n11-smoke.json" || rc=$?
echo "smoke wall time $(( $(date +%s) - t0 )) s (CPU, tiny model: not a measurement)"
echo "artifacts: $OUT"
exit $rc
