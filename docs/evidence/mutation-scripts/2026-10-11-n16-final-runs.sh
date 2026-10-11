#!/usr/bin/env bash
# N16 final runs on the committed HEAD of the worktree (outputs pasted into the evidence)
set -uo pipefail
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
WT=/home/user/wt-n16
cd $S/n16/cwd
echo "worktree HEAD $(git -C $WT rev-parse HEAD); status: $(git -C $WT status --short | wc -l) changed files"
for p in ec r1; do
  L=$S/llama-ec7630a; [ $p = r1 ] && L=$S/llama.cpp
  rm -rf $S/n16/work-$p
  t0=$(date +%s)
  MIMO2_DECODE_WORK=$S/n16/work-$p python3 $WT/tools/ci/check_unittest.py --step n16-mimo2-decode --checkout $L -- \
    $WT/tools/tinygen/run_parity_mimo2_decode.sh $L $S/n16/build-$p > $S/n16/final-wrapped-$p.log 2>&1
  echo "$p wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
done
echo "== pin dump comparison"
python3 $WT/docs/evidence/mutation-scripts/2026-10-11-n16-pindiff.py $S/n16/work-ec/n16 $S/n16/work-r1/n16
python3 $WT/docs/evidence/mutation-scripts/2026-10-11-n16-pindiff.py $S/n16/work-ec/n16/mxfp4 $S/n16/work-r1/n16/mxfp4
for p in ec r1; do
  L=$S/llama-ec7630a; [ $p = r1 ] && L=$S/llama.cpp
  echo "== bites $p ($(git -C $L rev-parse --short HEAD))"
  bash $WT/docs/evidence/mutation-scripts/2026-10-11-n16-bites.sh $WT $S/n16/work-$p $L $S/n16/bites-$p
  echo "== evict probe $p"
  python3 $WT/docs/evidence/mutation-scripts/2026-10-11-n16-evict-probe.py $WT $S/n16/work-$p/n16 $L/gguf-py
done
for p in ec r1; do
  L=$S/llama-ec7630a; [ $p = r1 ] && L=$S/llama.cpp
  rm -rf $S/n16/mut/$p
  python3 $WT/docs/evidence/mutation-scripts/2026-10-11-n16-mimo2-mutations.py $WT $S/n16/work-$p $L $S/n16/mut/$p > $S/n16/final-mut-$p.log 2>&1
  echo "mutation $p rc=$?"
done
echo FINAL-DONE
