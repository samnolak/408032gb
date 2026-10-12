#!/usr/bin/env bash
# N12 probe (not a mutation check): N11's R1 tools (branch claude/n11-r1-tools-cpu-server, extracted with git archive,
# never merged here) against ik_llama.cpp's llama-server on a tiny glm5next model with a text vocabulary (N11's
# tools/tinygen/glm5next_tiny.py --arch glm5next --text-vocab): /tokenize, tools/bench/prompts.py --verify (the /tokenize
# dialect R1 relies on) and tools/quality/logprob_compare.py between two ik servers, "ref" (-fa off, f32 indexer cache)
# and "cand" (the CPU-applicable flags of scripts/r1/configs/ik-llamacpp-server.json at a smaller -c). Corpus: N11's
# (tools/bench/prompts.py Filler, seed "n11-corpus"). Nothing here is a measurement.
# usage: 2026-10-11-n12-ik-logprob.sh <n11 tree> <ik bin dir> <model> <work dir>
set -euo pipefail
N11=$1; BIN=$2; MODEL=$3; W=$4
mkdir -p "$W/corpus"; W=$(cd "$W" && pwd)
python3 - "$N11" "$W/corpus" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "tools", "bench"))
from prompts import Filler
for k in ("prose", "code", "logs"):
    open(os.path.join(sys.argv[2], k + ".txt"), "w", encoding="utf-8").write(Filler(k, "n11-corpus").text(400))
PY
wait_ok() { for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 0.5; done; return 1; }
# started from $W: ik's logger writes llama.log into the current directory
cd "$W"
"$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port 18481 -c 32768 -np 1 --jinja -ncmoe 17 -fa on --dsa \
  -ctk f16 -ctv f16 -t 2 --threads-batch 2 -b 2048 -ub 1024 -muge > "$W/cand.log" 2>&1 & CAND=$!
"$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port 18482 -c 8192 -np 1 -fa off -ctk f32 -ctv f16 -ictk f32 \
  --dsa -t 2 > "$W/ref.log" 2>&1 & REF=$!
trap 'kill $CAND $REF 2>/dev/null; wait $CAND $REF 2>/dev/null || true' EXIT
wait_ok 18481 && wait_ok 18482 && echo "both servers healthy"
echo "--- POST /tokenize {content: 'Hello world'}"; curl -s http://127.0.0.1:18481/tokenize -d '{"content": "Hello world"}'; echo
echo "--- POST /tokenize with_pieces"; curl -s http://127.0.0.1:18481/tokenize -d '{"content": "Hello world", "with_pieces": true}'; echo
echo "--- prompts.py --verify"
(cd "$N11" && python3 tools/bench/prompts.py --url http://127.0.0.1:18481 --targets 256,1024 --verify 2>&1 | tail -n 6) || echo "prompts.py rc=$?"
echo "--- logprob_compare ref (-fa off) vs cand (template CPU flags)"
(cd "$N11" && python3 tools/quality/logprob_compare.py --ref http://127.0.0.1:18482 --cand http://127.0.0.1:18481 \
  --corpus "$W"/corpus/*.txt --prefixes 40 --top-k 20 --max-chars 1500 --out "$W/lp.json" 2>&1 | tail -n 12) \
  || echo "logprob_compare rc=$?"
python3 - "$W/lp.json" <<'PY' || true
import json, sys
j = json.load(open(sys.argv[1]))
print("summary", json.dumps(j["summary"]))
kinds = {}
for e in j["errors"]:
    k = "incomplete UTF-8 (known llama-server limitation, N11)" if "incomplete UTF-8" in e["error"] else e["error"][:120]
    kinds[k] = kinds.get(k, 0) + 1
print("errors", len(j["errors"]), kinds)
PY
