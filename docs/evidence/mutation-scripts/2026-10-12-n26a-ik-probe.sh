#!/usr/bin/env bash
# N26a: ik_llama.cpp @5194a9e CPU build (N12) with --chat-template-file = the repo's GLM template; tiny glm5next GGUF of N12.
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
W=$S/n26a/ik-tpl2; mkdir -p "$W"
BIN=$S/n12/ik-build/bin/llama-server
TPL=/home/user/wt-n26a/templates/glm-5.3-flash/chat_template.jinja
echo "\$ llama-server --version"; "$BIN" --version 2>&1 | head -2
echo "\$ sha256sum chat_template.jinja"; sha256sum "$TPL"
MSG='{"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1","reasoning_content":"R1"},{"role":"user","content":"Q2"},{"role":"assistant","content":"A2"},{"role":"user","content":"Q3"}]}'
for mode in template-file gguf-builtin; do
  EX=(); [ $mode = template-file ] && EX=(--chat-template-file "$TPL")
  echo "\$ llama-server -m ik.gguf --port 18591 -c 2048 --parallel 1 --jinja ${EX[*]} -t 2   # mode $mode"
  "$BIN" -m $S/n12/work-ci/ik.gguf --host 127.0.0.1 --port 18591 --ctx-size 2048 --parallel 1 --jinja "${EX[@]}" -t 2 > "$W/server-$mode.log" 2>&1 &
  PID=$!
  for i in $(seq 1 90); do c=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://127.0.0.1:18591/health); [ "$c" = 200 ] && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
  echo "GET /health -> $c"
  echo "\$ POST /apply-template $MSG"
  curl -s --noproxy '*' http://127.0.0.1:18591/apply-template -H 'Content-Type: application/json' -d "$MSG"; echo
  kill $PID; wait $PID 2>/dev/null
  grep -c "error" "$W/server-$mode.log" | sed 's/^/server log lines with "error": /'
done
