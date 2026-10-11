#!/usr/bin/env bash
# N14 mutation check: does tests/test_chat_templates.py fail when the reference renderer, the prefix analysis, the
# driver, the server client or the parse harness is broken? Each mutation runs on a fresh `git archive HEAD` copy of
# the repository; the whole test module runs with the server environment of tools/chattmpl/run_n14.sh.
# Paths of the session sandbox (N=, S=, WT=): replace them with your own to repeat.
#   MUTDIR=<scratch subdir> bash docs/evidence/mutation-scripts/2026-10-11-n14-mutations.sh
set -uo pipefail
N=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n14
S=$N/..
WT=/home/user/wt-n14
LLAMA=$S/llama.cpp                 # checkout at 79e2e74
BUILD=$S/llama.cpp/build-cpu       # shared CPU build of it
WORK=$N/final                      # models from run_n14.sh
MUTDIR=${MUTDIR:-$N/mut}
mkdir -p "$MUTDIR"
echo "HEAD $(git -C "$WT" rev-parse --short HEAD)"

fresh() {  # $1 = name -> prints the copy's path
    local d=$MUTDIR/$1
    rm -rf "$d"; mkdir -p "$d"
    git -C "$WT" archive HEAD | tar -x -C "$d"
    echo "$d"
}

harness() {  # $1 = copy -> builds its harness
    g++ -std=c++17 -O1 -I"$LLAMA/common" -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/vendor" \
        "$1/tools/chattmpl/chat_parse_harness.cpp" -o "$1/chat-parse-harness" \
        -L"$BUILD/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$BUILD/bin"
}

run_tests() {  # $1 = copy -> exit status of the test module; log in $1/test.log
    (cd "$1" && CHATTMPL_SERVER_BIN="$BUILD/bin/llama-server" CHATTMPL_HARNESS="$1/chat-parse-harness" \
        CHATTMPL_GLM_GGUF="$WORK/models/glm5next_bytes.gguf" CHATTMPL_QWEN_GGUF="$WORK/models/qwen4exp_bytes.gguf" \
        CHATTMPL_WORK="$1/test-out" python3 -m unittest -v tests.test_chat_templates > "$1/test.log" 2>&1)
}

mutate() {  # $1 name, $2 file, $3 python replacement old, $4 new
    local d; d=$(fresh "$1")
    python3 - "$d/$2" "$3" "$4" <<'EOF' || { echo "$1 NOT APPLIED"; return; }
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
if old not in s:
    sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
EOF
    harness "$d"
    run_tests "$d"; local st=$?
    local failed
    failed=$(grep -E "^(FAIL|ERROR): " "$d/test.log" \
        | sed -E 's/^(FAIL|ERROR): [a-z_0-9]+ \(tests\.test_chat_templates\.([A-Za-z0-9]+)\.([a-z_0-9]+)\)$/\2.\3/' \
        | sort -u | tr '\n' ' ')
    if [ $st -ne 0 ]; then echo "$1 KILLED  ($failed)"; else echo "$1 SURVIVED"; fi
}

d=$(fresh base); harness "$d"; run_tests "$d"; echo "base exit $? ($(tail -1 "$d/test.log"))"

mutate M1-tojson-ascii tools/chattmpl/hf_render.py \
    'def tojson(x, ensure_ascii=False,' 'def tojson(x, ensure_ascii=True,'
mutate M2-no-trim-blocks tools/chattmpl/hf_render.py \
    'trim_blocks=True, lstrip_blocks=True' 'trim_blocks=False, lstrip_blocks=False'
mutate M3-lcp-short tools/chattmpl/analysis.py \
    'while i < n and a[i] == b[i]:' 'while i < n - 1 and a[i] == b[i]:'
# M4 is expected to survive: a GBNF string literal accepts a raw newline (src/llama-grammar.cpp:180-181 @79e2e74,
# parse_char decodes any unescaped byte), so the mutated grammar is the same grammar; M4b breaks the quote escape
mutate M4-gbnf-newline tools/chattmpl/run_n14.py \
    'elif ch == "\n":' 'elif False:'
mutate M4b-gbnf-quote tools/chattmpl/run_n14.py \
    "elif ch == '\"':" 'elif False:'
mutate M5-no-cache-prompt tools/chattmpl/run_n14.py \
    '"cache_prompt": True' '"cache_prompt": False'
mutate M6-glm-preserve-kwarg tools/chattmpl/run_n14.py \
    '"preserve_off": ({"clear_thinking": True}, {"preserve_reasoning": False}, None),' \
    '"preserve_off": ({"clear_thinking": True}, {"preserve_reasoning": True}, None),'
mutate M7-harness-no-reasoning tools/chattmpl/chat_parse_harness.cpp \
    'req.value("reasoning_format", std::string("deepseek"))' 'req.value("reasoning_format", std::string("none"))'
mutate M8-client-drops-kwargs tools/chattmpl/server.py \
    'body["chat_template_kwargs"] = kwargs' 'pass'
