#!/usr/bin/env bash
# Baseline: every llama.cpp parity driver of ci.yml / parity-r1-pin.yml at both pins, plain (no checker), from a
# scratch cwd. Logs: $N/base/<pin>-<driver>.log, rc and wall time in $N/base/summary.txt.
set -u
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
N=$S/n13; WT=/home/user/wt-n13
mkdir -p "$N/base" "$N/tmp" "$N/cwd"
export TMPDIR=$N/tmp PYTHONPATH=$WT
cd "$N/cwd"
for pin in ${PINS:-ec r1}; do
  if [ "$pin" = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  for d in ${DRIVERS:-run_parity run_parity_mimo run_parity_qwen4exp run_parity_qwen4exp_decode run_parity_glm5next_decode run_parity_f16fa run_parity_quant}; do
    t0=$(date +%s)
    "$WT/tools/tinygen/$d.sh" "$L" "$N/build-$pin" > "$N/base/$pin-$d.log" 2>&1; rc=$?
    echo "$pin $d rc=$rc wall_s=$(( $(date +%s) - t0 )) $(grep -E '^Ran |^OK|^FAILED' "$N/base/$pin-$d.log" | tr '\n' ' ')" | tee -a "$N/base/summary.txt"
    find "$N/tmp" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  done
done
