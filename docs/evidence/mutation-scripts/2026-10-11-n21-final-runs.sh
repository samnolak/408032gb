#!/usr/bin/env bash
# N21 final runs on the committed HEAD of the worktree (outputs pasted into docs/evidence/2026-10-11-n21-mimo-mtp-heads-
# fix.md): the N21 step at both unpatched pins and on the patched copy of 79e2e74 (the CI job n21-patched: HEAD stays
# the pin, patches/llama.cpp/0001 applied with git apply), every other CI parity step whose driver uses
# tools/dump/dump_tensors.cpp or ref/mimo2.py at both pins, and the N16 / N17 steps on the patched copy; all through
# tools/ci/check_unittest.py, from the worktree root (as CI), work directories under <S>/n21/tmp (TMPDIR), removed after
# each step.
set -uo pipefail
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
WT=/home/user/wt-n21
export TMPDIR=$S/n21/tmp
cd $WT
echo "worktree HEAD $(git -C $WT rev-parse HEAD); status: $(git -C $WT status --short | wc -l) changed or untracked files"
pin() { case $1 in ec) echo $S/llama-ec7630a;; r1) echo $S/llama.cpp;; r1p) echo $S/n21/llama-r1-patched;; esac; }
step() {  # step <id> <driver> <pin> [work env var]; ONLY=<regex> runs only the steps whose id matches
  local id=$1 drv=$2 p=$3 wv=${4:-}
  if [ -n "${ONLY:-}" ] && ! [[ $id =~ $ONLY ]]; then return 0; fi
  local L; L=$(pin $p); local B=$S/n21/build-$p; local W=$S/n21/tmp/work-$id-$p
  rm -rf $W; mkdir -p $W
  local t0; t0=$(date +%s)
  if [ -n "$wv" ]; then
    env $wv=$W python3 tools/ci/check_unittest.py --step $id --checkout $L -- $drv $L $B > $S/n21/final-$id-$p.log 2>&1
  else
    python3 tools/ci/check_unittest.py --step $id --checkout $L -- $drv $L $B > $S/n21/final-$id-$p.log 2>&1
  fi
  local rc=$?
  echo "$id @$p ($(git -C $L rev-parse --short HEAD)$( [ $p = r1p ] && echo ' + patches/llama.cpp/0001')) wrapper_rc=$rc wall_s=$(( $(date +%s) - t0 )): $(grep -E '^(Ran [0-9]+ tests|OK|FAILED)' $S/n21/final-$id-$p.log | tr '\n' ' ')$(grep -E '^::(notice|error)::' $S/n21/final-$id-$p.log | tail -1)"
  rm -rf $W $S/n21/tmp/tmp.*
}
step n21-mimo2-mtp-fix tools/tinygen/run_parity_mimo2_n21.sh ec MIMO2_N21_WORK
step n21-mimo2-mtp-fix tools/tinygen/run_parity_mimo2_n21.sh r1 MIMO2_N21_WORK
step n21-mimo2-mtp-fix-patched tools/tinygen/run_parity_mimo2_n21.sh r1p MIMO2_N21_WORK
step n17-mimo2-mtp-heads-fa tools/tinygen/run_parity_mimo2_n17.sh r1p MIMO2_N17_WORK
step n16-mimo2-decode tools/tinygen/run_parity_mimo2_decode.sh r1p MIMO2_DECODE_WORK
for p in ec r1; do
  step g1-glm tools/tinygen/run_parity.sh $p
  step g1-mimo tools/tinygen/run_parity_mimo.sh $p
  step n2-qwen4exp tools/tinygen/run_parity_qwen4exp.sh $p
  step n6-qwen4exp-decode tools/tinygen/run_parity_qwen4exp_decode.sh $p
  step n9-glm5next-decode tools/tinygen/run_parity_glm5next_decode.sh $p
  step n10-f16fa tools/tinygen/run_parity_f16fa.sh $p F16FA_WORK
  step n10-quant tools/tinygen/run_parity_quant.sh $p
  step n16-mimo2-decode tools/tinygen/run_parity_mimo2_decode.sh $p MIMO2_DECODE_WORK
  step n17-mimo2-mtp-heads-fa tools/tinygen/run_parity_mimo2_n17.sh $p MIMO2_N17_WORK
  step n19-qwen4exp-multiseq tools/tinygen/run_parity_qwen4exp_multiseq.sh $p
done
echo FINAL-DONE
