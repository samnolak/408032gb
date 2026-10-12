#!/usr/bin/env bash
# Real-model routing traces + expert-prediction probe for one model of data/traces/real/REQUEST.json (CPU only).
#   1. download the GGUF (scripts/traces/fetch_hf_gguf.py, revision and sha256 recorded)
#   2. build the agent-style corpus (scripts/traces/build_agent_prompts.py) and render it with the model's chat
#      template through llama-server (scripts/traces/render_ids.py) -> token ids
#   3. tools/g0 g0-expert-trace on every conversation (prompt + G0_GEN_TOKENS greedy tokens) -> *.trace
#   4. g0-expert-probe on the held-out conversations (every holdout_every-th) -> *.probe (not kept: large)
#   5. tools/g0/predict_eval.py (self-checks + predictors, trained on the other traces) and tools/g0/cachesim.py
#   6. package into data/traces/real/<name>/ (gzip traces, ids, manifests, results); the model file is deleted
# Usage: scripts/traces/run_real_traces.sh <model name> <llama.cpp checkout @REQUEST llama_cpp> <g0 build dir> \
#          <llama-server binary> <work dir>
set -euo pipefail
NAME=${1:?model name}; LLAMA=$(cd "${2:?llama.cpp}" && pwd); G0=$(cd "${3:?g0 build}" && pwd)
SERVER=${4:?llama-server}; WORK=${5:?work dir}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); REQ="$ROOT/data/traces/real/REQUEST.json"
OUT="$ROOT/data/traces/real/$NAME"
q() { python3 -c "import json,sys; r=json.load(open('$REQ')); m=[x for x in r['models'] if x['name']=='$NAME'][0]; print(eval(sys.argv[1]))" "$1"; }
PIN=$(q "r['llama_cpp']"); HEAD=$(git -C "$LLAMA" rev-parse HEAD)
[ "$PIN" = "$HEAD" ] || { echo "llama.cpp is $HEAD, REQUEST.json says $PIN" >&2; exit 1; }
GEN=$(q "r['gen_tokens']"); HOLD=$(q "r['probe']['holdout_every']"); TAIL=$(q "r['probe']['tail']")
NPROMPTS=$(q "r['corpus']['n']"); MAXC=$(q "r['corpus']['max_chars']")
rm -rf "$WORK" && mkdir -p "$WORK"/{model,prompts,ids,traces,train,test,probes} "$OUT"
NT=$(nproc)
# run a step with its full output in a log; on failure show the log's tail and stop (never filter errors away)
logged() { local log=$1; shift; "$@" > "$log" 2>&1 || { echo "FAILED ($?): $*" >&2; tail -n 40 "$log" >&2; exit 1; }; }

echo "== $NAME: download"
python3 "$ROOT/scripts/traces/fetch_hf_gguf.py" --repo "$(q "m['hf_repo']")" --regex "$(q "m['file_regex']")" \
  --dir "$WORK/model" > "$WORK/fetch.json"
cat "$WORK/fetch.json"
MODEL=$(python3 -c "import json; print(json.load(open('$WORK/fetch.json'))['path'])")
read -r ARCH NEXP K < <(PYTHONPATH="$LLAMA/gguf-py" python3 - "$MODEL" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); f = lambda k: r.get_field(k).contents() if r.get_field(k) else 0
a = f("general.architecture"); print(a, f(f"{a}.expert_count"), f(f"{a}.expert_used_count"))
PY
)
echo "arch=$ARCH n_expert=$NEXP k=$K"

echo "== $NAME: corpus and chat-template rendering"
python3 "$ROOT/scripts/traces/build_agent_prompts.py" --root "$ROOT" --out "$WORK/prompts" --n "$NPROMPTS" --max-chars "$MAXC"
"$SERVER" -m "$MODEL" --host 127.0.0.1 --port 18765 -c 16384 -np 1 -t "$NT" --no-warmup --jinja > "$WORK/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:18765/health >/dev/null && break; sleep 1; done
python3 "$ROOT/scripts/traces/render_ids.py" --url http://127.0.0.1:18765 --prompts "$WORK/prompts" --out "$WORK/ids"
kill "$SPID"; wait "$SPID" || true

echo "== $NAME: routing traces (prompt + $GEN generated tokens)"
logged "$WORK/trace.log" env G0_TRACES_DIR="$WORK/ids" G0_OUT_DIR="$WORK/traces" G0_GEN_TOKENS="$GEN" \
  "$G0/g0-expert-trace" -m "$MODEL" -c 8192 -b 512 -ub 512 -t "$NT"
grep -E "generated tokens" "$WORK/trace.log" | sed 's/ -> .*//' | tee "$OUT/trace.txt"
i=0
for f in "$WORK/ids"/*.ids; do   # bash globs are sorted
  b=$(basename "$f" .ids)
  if [ $((i % HOLD)) -eq $((HOLD - 1)) ]; then cp "$f" "$WORK/test/"; else cp "$WORK/traces/$b.trace" "$WORK/train/"; fi
  i=$((i + 1))
done

echo "== $NAME: expert-prediction probe on held-out conversations"
logged "$WORK/probe.log" env G0_TRACES_DIR="$WORK/test" G0_OUT_DIR="$WORK/probes" G0_GEN_TOKENS="$GEN" \
  G0_PROBE_TAIL="$TAIL" "$G0/g0-expert-probe" -m "$MODEL" -c 8192 -b 512 -ub 512 -t "$NT"
grep -E "recorded" "$WORK/probe.log" | sed 's/ -> .*//' | tee "$OUT/probe.txt"
PYTHONPATH="$LLAMA/gguf-py" python3 "$ROOT/tools/g0/predict_eval.py" "$WORK/probes" --gguf "$MODEL" \
  --train-traces "$WORK/train" --check-traces "$WORK/traces" -o "$OUT/predict.json" | tee "$OUT/predict.txt"

echo "== $NAME: cache policies"
python3 "$ROOT/tools/g0/cachesim.py" "$WORK/traces" --n-experts "$NEXP" --capacity 0.1,0.25,0.5,0.75 \
  --holdout-every "$HOLD" -o "$OUT/cachesim.json" | tee "$OUT/cachesim.txt"

echo "== $NAME: package"
mkdir -p "$OUT/traces" "$OUT/ids"
for t in "$WORK/traces"/*.trace; do gzip -9 -n -c "$t" > "$OUT/traces/$(basename "$t").gz"; done
cp "$WORK/ids"/*.ids "$WORK/ids/RENDER.json" "$OUT/ids/"
cp "$WORK/prompts/INDEX.json" "$OUT/CORPUS.json"
python3 - "$WORK/fetch.json" "$OUT/MANIFEST.json" "$ARCH" "$NEXP" "$K" "$PIN" "$GEN" "$HOLD" "$TAIL" "$NT" <<'PY'
import json, sys, platform, subprocess
fetch = json.load(open(sys.argv[1])); fetch.pop("path", None)
cpu = subprocess.run(["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | cut -c1-200"], capture_output=True,
                     text=True).stdout
json.dump({"model": fetch, "arch": sys.argv[3], "n_expert": int(sys.argv[4]), "k": int(sys.argv[5]),
           "llama_cpp": sys.argv[6], "gen_tokens": int(sys.argv[7]), "holdout_every": int(sys.argv[8]),
           "probe_tail": int(sys.argv[9]), "threads": int(sys.argv[10]),
           "trace_args": "-c 8192 -b 512 -ub 512 (CPU, GGML_NATIVE=ON build)", "host": platform.platform(),
           "cpu": cpu.strip().splitlines(),
           "label": "REAL model (proxy family), CONSTRUCTED agent-style conversations from repository files; "
                    "greedy generation; CPU numerics"}, open(sys.argv[2], "w"), indent=1)
PY
rm -rf "$WORK/model"
du -sh "$OUT"
echo "done: $OUT"
