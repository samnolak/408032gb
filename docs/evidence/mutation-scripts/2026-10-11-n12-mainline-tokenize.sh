#!/usr/bin/env bash
# N12 probe (not a mutation check): the /tokenize and prompts.py --verify part of 2026-10-11-n12-ik-logprob.sh against
# llama.cpp's llama-server at the R1 pin (79e2e74), on the same tiny byte-BPE weights under llama.cpp's name glm5-next
# (N11's tools/tinygen/glm5next_tiny.py --layers 12 --text-vocab --ctx 4096), to tell ik's behaviour from the tool's.
# usage: 2026-10-11-n12-mainline-tokenize.sh <n11 tree> <llama-server @79e2e74> <gguf-py> <work dir>
set -uo pipefail
N11=$1; SRV=$2; GGUF_PY=$3; W=$4
mkdir -p "$W"
"$SRV" --version 2>&1 | head -n 1
(cd "$N11" && PYTHONPATH="$GGUF_PY" python3 tools/tinygen/glm5next_tiny.py "$W/glm5-next_text.gguf" --layers 12 \
  --text-vocab --ctx 4096)
(cd "$W" && "$SRV" -m "$W/glm5-next_text.gguf" --host 127.0.0.1 --port 18491 -c 8192 -np 1 -t 2 > "$W/server.log" 2>&1) &
PID=$!
trap 'pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:18491/health >/dev/null && break; sleep 0.5; done
echo "--- POST /tokenize {content: 'Hello world'}"; curl -s http://127.0.0.1:18491/tokenize -d '{"content": "Hello world"}'; echo
echo "--- POST /tokenize with_pieces"; curl -s http://127.0.0.1:18491/tokenize -d '{"content": "Hello world", "with_pieces": true}'; echo
echo "--- prompts.py --verify"
(cd "$N11" && python3 tools/bench/prompts.py --url http://127.0.0.1:18491 --targets 256,1024 --verify 2>&1 | tail -n 6)
echo "prompts.py rc=$?"
