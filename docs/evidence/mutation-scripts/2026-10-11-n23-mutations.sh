#!/usr/bin/env bash
# N23 mutation check: do the N23 tests fail when the corrected template, the drivers, the token-level tool, the LFS
# fetcher or the vendoring of LFS pins is broken? Each mutation runs on a fresh `git archive HEAD` copy of the
# repository and runs only the test module(s) that cover the mutated file:
#   template / run_n14 / convs   tests.test_glm_template_n23 with the llama-server environment of tools/chattmpl/run_n23.sh
#   token_prefix                 tests.test_token_level with llama-tokenize and the vocab-only GGUFs (llama side)
#   hf_lfs_fetch / vendor        tests.test_hf_lfs
# Paths of the session sandbox (S=, WT=, ...): replace them with your own to repeat.
#   MUTDIR=<scratch subdir> bash docs/evidence/mutation-scripts/2026-10-11-n23-mutations.sh
#   ONLY='^M(7|8|9|1[0-3])-' ... runs only the mutations whose names match (and no base runs)
set -uo pipefail
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
WT=/home/user/wt-n23
BUILD=$S/llama.cpp/build-cpu                  # shared CPU build of llama.cpp 79e2e74 (llama-server)
WORK=$S/n23/run1                              # models and harness from tools/chattmpl/run_n23.sh
TOKBIN=$S/n15/build-tok/bin/llama-tokenize    # llama-tokenize of 79e2e74
VOCAB=$S/n23/n15                              # vocab-glm.gguf, vocab-qwen.gguf (N15's vocab_gguf.py on N15's headers)
MUTDIR=${MUTDIR:-$S/n23/mut}
ONLY=${ONLY:-}
mkdir -p "$MUTDIR"
echo "HEAD $(git -C "$WT" rev-parse --short HEAD)"

fresh() {  # $1 = name -> prints the copy's path
    local d=$MUTDIR/$1
    rm -rf "$d"; mkdir -p "$d"
    git -C "$WT" archive HEAD | tar -x -C "$d"
    echo "$d"
}

run_tests() {  # $1 = copy, $2 = group -> exit status; log in $1/test.log
    case $2 in
        template) (cd "$1" && CHATTMPL_SERVER_BIN="$BUILD/bin/llama-server" CHATTMPL_HARNESS="$WORK/chat-parse-harness" \
                     CHATTMPL_GLM_GGUF="$WORK/models/glm5next_bytes.gguf" CHATTMPL_WORK="$1/test-out" \
                     python3 -m unittest -v tests.test_glm_template_n23 > "$1/test.log" 2>&1) ;;
        token) (cd "$1" && TOKCHECK_LLAMA_TOKENIZE="$TOKBIN" TOKCHECK_VOCAB_GLM="$VOCAB/vocab-glm.gguf" \
                  TOKCHECK_VOCAB_QWEN="$VOCAB/vocab-qwen.gguf" python3 -m unittest -v tests.test_token_level \
                  > "$1/test.log" 2>&1) ;;
        lfs) (cd "$1" && python3 -m unittest -v tests.test_hf_lfs > "$1/test.log" 2>&1) ;;
    esac
}

mutate() {  # $1 name, $2 group, $3 file, $4 old, $5 new
    if [ -n "$ONLY" ] && ! [[ $1 =~ $ONLY ]]; then return; fi
    local d; d=$(fresh "$1")
    python3 - "$d/$3" "$4" "$5" <<'EOF' || { echo "$1 NOT APPLIED"; return; }
import sys
p, old, new = sys.argv[1:]
s = open(p, encoding="utf-8").read()
if old not in s:
    sys.exit(1)
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
EOF
    run_tests "$d" "$2"; local st=$?
    local failed
    failed=$(grep -E "^(FAIL|ERROR): " "$d/test.log" \
        | sed -E 's/^(FAIL|ERROR): [a-z_0-9]+ \(tests\.[a-z_0-9]+\.([A-Za-z0-9]+)\.([a-z_0-9]+)\)$/\2.\3/' \
        | sort -u | tr '\n' ' ')
    if [ $st -ne 0 ]; then echo "$1 KILLED  ($failed)"; else echo "$1 SURVIVED ($(tail -1 "$d/test.log"))"; fi
}

if [ -z "$ONLY" ]; then
    for g in template token lfs; do
        d=$(fresh "base-$g"); run_tests "$d" "$g"; echo "base $g exit $? ($(tail -1 "$d/test.log"))"
    done
fi

T=templates/glm-5.3-flash/chat_template.jinja
mutate M1-no-reset template $T '{%- set reasoning_content = none %}
' ''
mutate M2-is-defined template $T 'and reasoning_content is not none -%}' 'and reasoning_content is defined -%}'
# M3 renders the same bytes everywhere (an empty string reset falls into the same <think></think> output); only the
# structural test of the file can see it
mutate M3-reset-empty-string template $T '{%- set reasoning_content = none %}' "{%- set reasoning_content = '' %}"
mutate M4-tag-in-header template $T 'task N23).' 'task N23, <think>).'
mutate M5-server-loads-official template tools/chattmpl/run_n14.py \
    'return pathlib.Path(t["template"]) if t.get("template") else' 'return pathlib.Path(t["template"]) if False else'
# M6 is expected to survive: the reference becomes our file, which renders like the official one under jinja2 (that is
# the property JinjaEquality checks directly)
mutate M6-reference-is-ours template tools/chattmpl/run_n14.py \
    'if "reference" in t:' 'if False:'
mutate M7-conv-no-missing-turn template tools/chattmpl/convs.py \
    '_asst("a2", "R2"), {"role": "user", "content": "u3"}, _asst("a3"), {"role": "user", "content": "u4"}]},' \
    '_asst("a2", "R2"), {"role": "user", "content": "u3"}, _asst("a3", "R3"), {"role": "user", "content": "u4"}]},'
mutate M8-held-drops-a-token token tools/chattmpl/token_prefix.py 'held = tp + tt' 'held = tp + tt[1:]'
mutate M9-llama-with-bos token tools/chattmpl/token_prefix.py '"--ids", "--log-disable", "--no-bos",' '"--ids", "--log-disable",'
mutate M10-cache-size-only lfs scripts/hf_lfs_fetch.py 'return h.hexdigest() == e["sha256"]' 'return True'
mutate M11-vendor-no-sha-check lfs scripts/vendor_hf_files.py 'if len(data) != size or sha != oid:' 'if len(data) != size:'
mutate M12-vendor-any-lfs lfs scripts/vendor_hf_files.py 'if os.path.basename(path) not in LFS_ALLOWED:' 'if False:'
mutate M13-fetch-no-length-cap lfs scripts/hf_lfs_fetch.py 'if n > e["size"]:' 'if False:'
