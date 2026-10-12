#!/usr/bin/env bash
# N26a rule 3: llama-server @79e2e74 (CPU build) direct vs through tools/gateway/gateway.py, tiny qwen4exp with the real vocabulary (N23).
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
W=$S/n26a/gw-real2; mkdir -p "$W"
BIN=$S/llama.cpp/build-cpu/bin/llama-server
echo "\$ git -C /home/user/wt-n26a log --oneline -1 -- tools/gateway/gateway.py"; git -C /home/user/wt-n26a log --oneline -1 -- tools/gateway/gateway.py
echo "\$ git -C /home/user/wt-n26a status --short tools/gateway/gateway.py   # (empty = the committed file)"; git -C /home/user/wt-n26a status --short tools/gateway/gateway.py
echo "\$ llama-server --version"; "$BIN" --version 2>&1 | head -1
"$BIN" -m $S/n23/qlive/out/qwen4exp_realvocab.gguf --host 127.0.0.1 --port 18601 -c 2048 -np 1 --jinja -t 2 > "$W/server.log" 2>&1 &
SP=$!
for i in $(seq 1 90); do c=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://127.0.0.1:18601/health); [ "$c" = 200 ] && break; sleep 1; done
echo "llama-server -m qwen4exp_realvocab.gguf -c 2048 -np 1 --jinja: GET /health -> $c"
python3 /home/user/wt-n26a/tools/gateway/gateway.py --backend http://127.0.0.1:18601 --listen 127.0.0.1:18602 --log-file "$W/gw.jsonl" > "$W/gw.log" 2>&1 &
GP=$!
for i in $(seq 1 30); do c=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://127.0.0.1:18602/health); [ "$c" = 200 ] && break; sleep 0.5; done
B='{"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1","reasoning_content":null},{"role":"user","content":"Q2"}],"max_tokens":2,"temperature":0}'
echo "\$ B='$B'"
echo "\$ curl -s http://127.0.0.1:18601/v1/chat/completions -d \"\$B\"     # direct"
curl -s --noproxy '*' -o "$W/direct.json" -w 'HTTP %{http_code}\n' http://127.0.0.1:18601/v1/chat/completions -H 'Content-Type: application/json' -d "$B"; cat "$W/direct.json"; echo
echo "\$ curl -s http://127.0.0.1:18602/v1/chat/completions -d \"\$B\"     # through the gateway"
curl -s --noproxy '*' -o "$W/gw.json" -w 'HTTP %{http_code}\n' http://127.0.0.1:18602/v1/chat/completions -H 'Content-Type: application/json' -d "$B"; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('object', 'usage')}))" "$W/gw.json"
sleep 1
echo "\$ python3 -c '...' gw.jsonl   # fields of the gateway's request log"
python3 -c "import json,sys; r=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(json.dumps({k: r[k] for k in ('path','status','dropped_null_fields','injected_defaults','error')}))" "$W/gw.jsonl"
kill $GP $SP; wait 2>/dev/null
