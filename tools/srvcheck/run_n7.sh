#!/usr/bin/env bash
# N7 on CPU, no GPU needed: the real llama-server of R1 (llama.cpp pinned in tools/engines/pins.json) on the tiny
# hybrid-memory models qwen4exp (gated delta net + QSA attention, Qwen3.8-Flash-Next graph) and glm5-next (KDA +
# MLA/DSA, GLM-5.3-Flash graph), each also with an MTP block -> tools/srvcheck/hybrid_server_checks.py.
#   gate:            f32 caches, flash attention off (the precision contract of the CPU parity runs), --tol 1e-3;
#                    any failed check fails this script;
#   characterization: the server's defaults for KV type and flash attention (what an R1 run starts from); recorded,
#                    never fails the script (docs/evidence/2026-10-10-n7-server-hybrid-cpu.md says why).
# Usage: tools/srvcheck/run_n7.sh <llama.cpp checkout at the pins.json commit> [build dir] [output dir]
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
export PYTHONPATH="$LLAMA/gguf-py"
python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$OUT/qwen4exp.gguf"
python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" "$OUT/qwen4exp_mtp.gguf" --mtp
python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$OUT/glm5next.gguf"
python3 "$ROOT/tools/tinygen/glm5next_tiny.py" "$OUT/glm5next_mtp.gguf" --mtp
unset PYTHONPATH
CHECK="$ROOT/tools/srvcheck/hybrid_server_checks.py"
GATE="-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup"
DEFAULTS="-c 1024 -np 1 -t 2 --no-warmup"
fail=0
for m in qwen4exp glm5next; do
  echo "== $m gate ($GATE)"
  python3 "$CHECK" --server "$SERVER" --model "$OUT/$m.gguf" --mtp-model "$OUT/${m}_mtp.gguf" --extra "$GATE" \
    --workdir "$OUT/$m-gate" --out "$OUT/$m-gate.json" || fail=1
done
for m in qwen4exp glm5next; do
  echo "== $m characterization ($DEFAULTS)"
  python3 "$CHECK" --server "$SERVER" --model "$OUT/$m.gguf" --mtp-model "$OUT/${m}_mtp.gguf" --extra "$DEFAULTS" \
    --workdir "$OUT/$m-defaults" --out "$OUT/$m-defaults.json" || true
done
echo "artifacts: $OUT"
exit $fail
