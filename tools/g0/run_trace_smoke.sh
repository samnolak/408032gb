#!/usr/bin/env bash
# CPU smoke test of the per-token routing tracer and the cache simulator on the tiny glm5-next model (random weights:
# checks the pipeline, says nothing about real routing). Usage:
#   tools/g0/run_trace_smoke.sh <llama.cpp checkout @ec7630a> [build dir]
set -euo pipefail
LLAMA=${1:?llama.cpp checkout}; BUILD=${2:-build-tools}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=$(mktemp -d)
cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target g0-expert-trace >/dev/null
PYTHONPATH="$LLAMA/gguf-py" python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$WORK/tiny.gguf" --layers 12 >/dev/null
mkdir -p "$WORK/traces" "$WORK/out" "$WORK/out0"
python3 - "$WORK/traces" <<'PY'
import random, sys
for i in range(5):
    r = random.Random(i)
    open(f"{sys.argv[1]}/p{i}.ids", "w").write(",".join(str(r.randrange(3, 96)) for _ in range(300)))
PY
G0_TRACES_DIR="$WORK/traces" G0_OUT_DIR="$WORK/out" G0_GEN_TOKENS=40 "$BUILD/g0-expert-trace" -m "$WORK/tiny.gguf" -c 512 -b 128 -ub 128 -t 2 2>&1 | grep -E "prompt \+"
G0_TRACES_DIR="$WORK/traces" G0_OUT_DIR="$WORK/out0" G0_GEN_TOKENS=0 "$BUILD/g0-expert-trace" -m "$WORK/tiny.gguf" -c 512 -b 128 -ub 128 -t 2 >/dev/null 2>&1
python3 - "$ROOT/tools/g0" "$WORK" <<'PY'
import sys
sys.path.insert(0, sys.argv[1]); import cachesim as cs
w = sys.argv[2]
for i in range(5):
    g, p = cs.read_trace(f"{w}/out/p{i}.trace"), cs.read_trace(f"{w}/out0/p{i}.trace")
    assert g["n_prompt"] == p["n_tokens"] == 300 and g["n_tokens"] == 340, (g["n_prompt"], g["n_tokens"])
    assert all(g["per_layer"][l][:300] == p["per_layer"][l] for l in p["layer_ids"]), "prompt part differs"
print("trace checks OK: 5 files, prompt part of generation traces == prompt-only traces")
PY
python3 "$ROOT/tools/g0/cachesim.py" "$WORK/out" --n-experts 8 --capacity 0.25,0.5,0.75,1.0 --holdout-every 5
echo "artifacts: $WORK"
