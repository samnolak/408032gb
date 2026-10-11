#!/usr/bin/env bash
# N15 mutation check: does tests/test_gguf_headers.py fail when the header parser's offset / alignment logic, the
# fetcher's Range, cap, pin and pending logic, or the analyzer's comparisons are broken? Each mutation runs on a fresh
# `git archive HEAD` copy of the worktree; the whole test module runs with gguf-py of the CI pin and both llama.cpp
# source checkouts (so the tinygen and anchor classes run too).
# Paths of the session sandbox (N=, S=, WT=): replace them with your own to repeat.
#   MUTDIR=<scratch subdir> bash docs/evidence/mutation-scripts/2026-10-11-n15-mutations.sh
set -uo pipefail
N=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n15
S=$N/..
WT=/home/user/wt-n15
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

P=tools/gguf_headers/gguf_header.py
F=scripts/gguf_headers_fetch.py
A=tools/gguf_headers/analyze.py
mutate M1-no-data-padding $P \
    'h.data_offset = align_up(c.p, h.alignment) if h.n_tensors > 0 else c.p' 'h.data_offset = c.p'
mutate M2-pad-without-tensors $P \
    'h.data_offset = align_up(c.p, h.alignment) if h.n_tensors > 0 else c.p' 'h.data_offset = align_up(c.p, h.alignment)'
mutate M3-default-alignment-16 $P 'DEFAULT_ALIGNMENT = 32' 'DEFAULT_ALIGNMENT = 16'
mutate M4-ignore-alignment-key $P '            h.alignment = e["value"]' '            pass'
mutate M5-no-contiguity-check $P 'if t["offset"] != run:' 'if False:'
mutate M6-inline-big-arrays $P 'big = len(v) > max_inline' 'big = False'
mutate M7-keep-infos-end-only $F 'return bytes(buf[:h.data_offset]), h, info' 'return bytes(buf[:h.infos_end]), h, info'
mutate M8-no-cap $F \
    'if e.need > cap:' 'if e.need > cap + size:' \
    'want = min(max(e.need, 2 * len(buf)), cap, size)' 'want = min(max(e.need, 2 * len(buf)), size)'
mutate M9-accept-200 $F 'if r.status != 206:' 'if False:'
mutate M10-no-total-check $F ' or m.group(3) != str(total):' ':'
mutate M11-no-revision-pin-check $F 'if SHA_RE.match(requested) and sha != requested:' 'if False:'
mutate M12-refetch-done-requests $F \
    'if not os.path.exists(os.path.join(out_dir(p, out_root), result_name(req))):' 'if True:'
mutate M13-no-run-cap $F 'if total > RUN_CAP:' 'if False:'
mutate M14-no-hard-cap $F 'not MIN_INITIAL <= cap <= HARD_CAP' 'not MIN_INITIAL <= cap'
mutate M15-etag-inverted $F '(etag == lfs_sha)' '(etag != lfs_sha)'
mutate M16-template-always-equal $A 'byte_equal=emb == ref)' 'byte_equal=True)'
mutate M17-groups-threshold $A 'if (g or 0) > 1 else' 'if (g or 0) > 8 else'
mutate M18-no-summary-rederivation $A ' and derived == committed)' ')'
# round 2: the size check of a metadata-only shard (writer padding after the KV block)
mutate M19-no-metadata-shard-padding $P \
    'if summary["n_tensors"] == 0 and size != off and size == align_up(off, summary["alignment"]):' 'if False:'
