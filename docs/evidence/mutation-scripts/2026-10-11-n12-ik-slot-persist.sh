#!/usr/bin/env bash
# N12 probe (not a mutation check): N11's tools/kv/slot_persist.py (branch claude/n11-r1-tools-cpu-server, extracted
# with git archive) against ik_llama.cpp's llama-server with the CPU-applicable flags of
# scripts/r1/configs/ik-llamacpp-server.json plus --slot-save-path, on the tiny byte-BPE glm5next model of
# 2026-10-11-n12-ik-logprob.sh; prompt = the first 3000 characters of that script's prose corpus file.
# usage: 2026-10-11-n12-ik-slot-persist.sh <n11 tree> <ik bin dir> <model> <corpus prose.txt> <work dir>
set -euo pipefail
N11=$1; BIN=$2; MODEL=$3; PROSE=$4; W=$5
mkdir -p "$W/slots"; head -c 3000 "$PROSE" > "$W/prompt.txt"
(cd "$W" && "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port 18483 -c 32768 -np 1 --jinja -ncmoe 17 -fa on \
  --dsa -ctk f16 -ctv f16 -t 2 --threads-batch 2 -b 2048 -ub 1024 -muge --slot-save-path "$W/slots" \
  > "$W/server.log" 2>&1) &
PID=$!
trap 'pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:18483/health >/dev/null && break; sleep 0.5; done
(cd "$N11" && python3 tools/kv/slot_persist.py --url http://127.0.0.1:18483 --prompt-file "$W/prompt.txt" --gen 16 \
  --repeats 2 --out "$W/slot.json" 2>&1 | tail -n 5)
python3 - "$W/slot.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
for x in j["runs"]:
    print({k: x[k] for k in ("same_continuation", "compared")}, "prompt_n cold", x["cold"]["prompt_n"], "warm",
          x["warm"]["prompt_n"], "| restore", json.dumps(x["restore"]["response"])[:110])
print("errors", j["errors"])
PY
