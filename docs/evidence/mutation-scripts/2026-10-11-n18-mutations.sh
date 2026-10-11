#!/usr/bin/env bash
# N18 mutation check: does tests/test_gguf_headers.py fail when the new code paths are broken - named repos of a
# discover request (scripts/gguf_headers_fetch.py), the separate-MTP-file analysis, the required-keys parser and the
# mimo2 config map / probes (tools/gguf_headers/analyze.py, gguf_branches.py)? Same mechanism as
# 2026-10-11-n15-mutations.sh: each mutation runs on a fresh `git archive HEAD` copy of the worktree; the whole test module
# runs with gguf-py of the CI pin and both llama.cpp source checkouts.
# Paths of the session sandbox (N=, S=, WT=): replace them with your own to repeat.
#   MUTDIR=<scratch subdir> bash docs/evidence/mutation-scripts/2026-10-11-n18-mutations.sh
set -uo pipefail
N=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n18
S=$N/..
WT=/home/user/wt-n18
MUTDIR=${MUTDIR:-$N/mut}
mkdir -p "$MUTDIR"
echo "HEAD $(git -C "$WT" rev-parse --short HEAD)"

fresh() {  # $1 = name -> prints the copy's path
    local d=$MUTDIR/$1
    rm -rf "$d"; mkdir -p "$d"
    git -C "$WT" archive HEAD | tar -x -C "$d"
    echo "$d"
}

run_tests() {  # $1 = copy -> exit status of the test module; log in $1/test.log
    (cd "$1" && LLAMA_CPP_GGUF_PY="$S/llama-ec7630a/gguf-py" LLAMA_CPP_SRC_79E2E74="$S/llama.cpp" \
        timeout 600 python3 -m unittest -v tests.test_gguf_headers > "$1/test.log" 2>&1)
}

mutate() {  # $1 name, $2 file, then pairs: old new [old new ...]
    local name=$1 file=$2; shift 2
    local d; d=$(fresh "$name")
    python3 - "$d/$file" "$@" <<'EOF' || { echo "$name NOT APPLIED"; return; }
import sys
p, pairs = sys.argv[1], sys.argv[2:]
s = open(p).read()
for old, new in zip(pairs[::2], pairs[1::2]):
    if old not in s:
        sys.exit(1)
    s = s.replace(old, new, 1)
open(p, "w").write(s)
EOF
    run_tests "$d"; local st=$?
    local failed
    failed=$(grep -E "^(FAIL|ERROR): " "$d/test.log" \
        | sed -E 's/^(FAIL|ERROR): [a-z_0-9]+ \(tests\.test_gguf_headers\.([A-Za-z0-9]+)\.([a-z_0-9]+)\)$/\2.\3/' \
        | sort -u | tr '\n' ' ')
    if [ $st -ne 0 ]; then echo "$name KILLED  ($failed)"; else echo "$name SURVIVED"; fi
}

d=$(fresh base); run_tests "$d"; echo "base exit $? ($(grep -E '^Ran ' "$d/test.log"), $(tail -1 "$d/test.log"))"

F=scripts/gguf_headers_fetch.py
A=tools/gguf_headers/analyze.py
B=tools/gguf_headers/gguf_branches.py
mutate M20-named-revision-unchecked $F 'if rev != "main" and sha != rev:' 'if False:'
mutate M21-named-listed-again-by-search $F 'REPO_RE.match(str(row["id"])) and row["id"] not in named:' \
    'REPO_RE.match(str(row["id"])):'
mutate M22-named-unavailable-raises $F \
    'row.update(status="repo_unavailable", http_status=e.code, http_body=body, revision=None)' 'raise'
mutate M23-duplicate-repos-accepted $F 'elif len({x["repo"] for x in named}) != len(named):' 'elif False:'
mutate M24-optional-counted-missing $A 'return {"missing": sorted(ref - have - optional),' \
    'return {"missing": sorted(ref - have),'
mutate M25-no-fused-split-alternatives $A 'ALTERNATIVES = [' 'ALTERNATIVES = [] and ['
mutate M26-globals-unchecked $A 'if hits(a) and name not in m.names:' 'if False:'
mutate M27-no-mtp-only-handling-ignored $A 'if not anchors["detect"]:' 'if False:'
mutate M28-missing-graph-ignored $A 'if anchors["no_graph"]:' 'if False:'
mutate M29-tokenizer-unchecked $A 'if "tokenizer.ggml.model" not in m.kv:' 'if False:'
mutate M30-extra-tensors-ignored $A 'if r["extra"]:' 'if False:'
mutate M31-lax-key-regex $A '([^;{\n]*)\);")' '([^;]*)\);")'
mutate M32-conditional-generic-keys-required $A \
    'if not req or (generic and depth > 1) or (kv, depth > 1) in seen:' 'if not req or (kv, depth > 1) in seen:'
mutate M33-pattern-without-mtp-layers $A 'return [int(x) for x in v] + [1] * n' 'return [int(x) for x in v]'
mutate M34-kv-array-without-mtp-layers $A '[cfg.get("swa_num_key_value_heads")] * n' '[]'
mutate M35-sinks-label-inverted $B \
    "{'with' if 'attn_sinks.weight' in _suffixes(m, i) else 'without'}" \
    "{'without' if 'attn_sinks.weight' in _suffixes(m, i) else 'with'}"
mutate M36-mtp-qkv-unchecked $B '"MTP layer with fused attn_qkv" if "attn_qkv.weight" in sx else' \
    '"MTP layer with fused attn_qkv" if True else'
mutate M37-value-scale-inverted $B '_f(m.a("attention.value_scale")) != 1.0 else {"no value scale"}' \
    '_f(m.a("attention.value_scale")) == 1.0 else {"no value scale"}'
mutate M38-trunk-probe-ignored $B 'has_trunk = trunk_probe in m.names' 'has_trunk = True'
mutate M39-tiny-workdir-not-created $A '    os.makedirs(workdir, exist_ok=True)
    out, skipped = {}, []' '    out, skipped = {}, []'
