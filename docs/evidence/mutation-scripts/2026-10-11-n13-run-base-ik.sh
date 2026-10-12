#!/usr/bin/env bash
# Baseline of the N12 ik driver (job ik-glm5next), plain, from a scratch cwd.
set -u
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
N=$S/n13; WT=/home/user/wt-n13
mkdir -p "$N/base" "$N/tmp-ik" "$N/cwd-ik"
export TMPDIR=$N/tmp-ik PYTHONPATH=$WT LLAMA_CPP_GGUF_PY=$S/llama.cpp/gguf-py
cd "$N/cwd-ik"
t0=$(date +%s)
"$WT/tools/tinygen/run_parity_ik_glm5next.sh" "$S/n12/ik-src" "$N/build-ik" > "$N/base/ik-run_parity_ik_glm5next.log" 2>&1; rc=$?
echo "ik run_parity_ik_glm5next rc=$rc wall_s=$(( $(date +%s) - t0 )) $(grep -E '^Ran |^OK|^FAILED' "$N/base/ik-run_parity_ik_glm5next.log" | tr '\n' ' ')" | tee -a "$N/base/summary-ik.txt"
find "$N/tmp-ik" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
