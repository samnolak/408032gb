#!/usr/bin/env bash
# Seconds-long self-test of the trace pipeline on a tiny random-weight qwen3moe GGUF (tools/tinygen/qwen3moe_tiny.py),
# run before any real model is downloaded. qwen3moe is the graph of the first real proxy model: it keeps only the
# output rows before the last layer's FFN (src/models/qwen3moe.cpp:123-125 @ec7630a), which broke the first
# traces-real run (no trace written). Checks:
#   1. g0-expert-trace and g0-expert-probe succeed with prompts longer than one batch (-b 64, 150..224 tokens);
#   2. tools/g0/predict_eval.py self-checks pass: router logits recomputed from the GGUF, selection rule, and every
#      probe row equal to the trace row of the same token and layer;
#   3. the routing of a -b 64 run equals a -b 1 run (each token its own batch and output) on >= 99% of the
#      (token, layer) rows; the rest are near-ties that batch-size numerics may flip.
# Usage: scripts/traces/smoke_tiny.sh <llama.cpp checkout> <g0 build dir> <work dir>
set -euo pipefail
LLAMA=$(cd "${1:?llama.cpp}" && pwd); G0=$(cd "${2:?g0 build}" && pwd); WORK=${3:?work dir}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$WORK" && mkdir -p "$WORK"/{ids,tr64,tr1,probes,train}
logged() { local log=$1; shift; "$@" > "$log" 2>&1 || { echo "FAILED ($?): $*" >&2; tail -n 40 "$log" >&2; exit 1; }; }
PYTHONPATH="$LLAMA/gguf-py" python3 "$ROOT/tools/tinygen/qwen3moe_tiny.py" "$WORK/tiny.gguf"
python3 - "$WORK/ids" <<'PY'
import random, sys
r = random.Random(3)
for i in range(3):
    open(f"{sys.argv[1]}/c{i:02d}.ids", "w").write(",".join(str(r.randrange(5, 90)) for _ in range(150 + 37 * i)))
PY
for b in 64 1; do
  logged "$WORK/trace-b$b.log" env G0_TRACES_DIR="$WORK/ids" G0_OUT_DIR="$WORK/tr$b" G0_GEN_TOKENS=8 \
    "$G0/g0-expert-trace" -m "$WORK/tiny.gguf" -c 512 -b "$b" -ub "$b" -t 2
done
grep -E "generated tokens" "$WORK/trace-b64.log" | sed 's/^.* I //; s/ -> .*//'
logged "$WORK/probe.log" env G0_TRACES_DIR="$WORK/ids" G0_OUT_DIR="$WORK/probes" G0_GEN_TOKENS=8 G0_PROBE_TAIL=100 \
  "$G0/g0-expert-probe" -m "$WORK/tiny.gguf" -c 512 -b 64 -ub 64 -t 2
grep -E "recorded" "$WORK/probe.log" | sed 's/^.* I //; s/ -> .*//'
cp "$WORK/tr64/c00.trace" "$WORK/train/"
# full output to a file, then show its first line: piping into head would close the pipe early and, under pipefail,
# fail the step with BrokenPipeError when Python writes unbuffered (PYTHONUNBUFFERED=1)
PYTHONPATH="$LLAMA/gguf-py" python3 "$ROOT/tools/g0/predict_eval.py" "$WORK/probes" --gguf "$WORK/tiny.gguf" \
  --train-traces "$WORK/train" --check-traces "$WORK/tr64" -o "$WORK/predict.json" > "$WORK/predict.txt"
head -n 1 "$WORK/predict.txt"
python3 - "$ROOT/tools/g0" "$WORK" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import cachesim
same = total = 0
for f in sorted(os.listdir(os.path.join(sys.argv[2], "tr64"))):
    a = cachesim.read_trace(os.path.join(sys.argv[2], "tr64", f))
    b = cachesim.read_trace(os.path.join(sys.argv[2], "tr1", f))
    assert (a["n_tokens"], a["n_prompt"], list(a["layer_ids"])) == (b["n_tokens"], b["n_prompt"], list(b["layer_ids"])), f
    for l in a["per_layer"]:
        x, y = np.sort(np.asarray(a["per_layer"][l]), -1), np.sort(np.asarray(b["per_layer"][l]), -1)
        same += int((x == y).all(-1).sum())
        total += len(x)
print(f"-b 64 vs -b 1: {same}/{total} (token, layer) rows select the same experts")
sys.exit(0 if same >= 0.99 * total else 1)
PY
echo "smoke ok"
