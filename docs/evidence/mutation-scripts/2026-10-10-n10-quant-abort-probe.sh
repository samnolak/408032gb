#!/usr/bin/env bash
# N10 gap 2: what llama.cpp does with the tiny qwen4exp model (--quant-dims) quantized by its own llama-quantize
# without --tensor-type ple_norm=f32: the per-tensor type of the three PLE gammas, then one 64-token prefill.
# usage: docs/evidence/mutation-scripts/2026-10-10-n10-quant-abort-probe.sh <llama.cpp checkout> <g0 build dir> <scratch>
set -u
LLAMA=$1; BUILD=$2; OUT=$3; ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUT"
PYTHONPATH="$LLAMA/gguf-py" python3 "$ROOT/tools/tinygen/qwen4exp_tiny.py" --quant-dims "$OUT/f32.gguf" >/dev/null
"$BUILD/quantize/llama-quantize" "$OUT/f32.gguf" "$OUT/q8_0.gguf" Q8_0 1 >"$OUT/quantize.log" 2>&1
echo "llama-quantize rc=$?; types of the PLE gammas:"
python3 -c "
import sys; sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
for t in GGUFReader(sys.argv[2]).tensors:
    if 'ple_norm' in t.name: print(' ', t.name, t.tensor_type.name, [int(x) for x in t.shape])" "$LLAMA/gguf-py" "$OUT/q8_0.gguf"
env DUMP_DIR="$OUT/dump" DUMP_TOKENS="5,6,7,8,9,10,11,12" DUMP_PER_GRAPH=1 DUMP_PREFIXES="result_" \
  "$BUILD/dump-tensors" -m "$OUT/q8_0.gguf" -c 512 -b 512 -ub 512 -t 1 -ctk f32 -ctv f32 -fa off >"$OUT/run.log" 2>&1
echo "dump-tensors rc=$?"
grep -m1 "binary_op" "$OUT/run.log"
