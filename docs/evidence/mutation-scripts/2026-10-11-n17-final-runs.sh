#!/usr/bin/env bash
# N17 final runs on the committed HEAD of the worktree (outputs pasted into docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md):
# the N17 CI step at both pins from empty work directories, the N16 and N10 steps at both pins (the dump tool and
# ref/fattn.py changed under them), the pin dump comparison, the one-head compatibility of the dump tool, the bites, the
# probes and the mutation rounds.
set -uo pipefail
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
WT=/home/user/wt-n17
M=$WT/docs/evidence/mutation-scripts
cd $S/n17/cwd
echo "worktree HEAD $(git -C $WT rev-parse HEAD); status: $(git -C $WT status --short | wc -l) changed or untracked files"
pin() { if [ "$1" = ec ]; then echo $S/llama-ec7630a; else echo $S/llama.cpp; fi; }
for p in ec r1; do
  L=$(pin $p)
  rm -rf $S/n17/work-$p
  t0=$(date +%s)
  MIMO2_N17_WORK=$S/n17/work-$p python3 $WT/tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --checkout $L -- \
    $WT/tools/tinygen/run_parity_mimo2_n17.sh $L $S/n17/build-$p > $S/n17/final-wrapped-$p.log 2>&1
  echo "n17 $p wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
done
for p in ec r1; do
  L=$(pin $p)
  rm -rf $S/n17/n16work-$p
  MIMO2_DECODE_WORK=$S/n17/n16work-$p python3 $WT/tools/ci/check_unittest.py --step n16-mimo2-decode --checkout $L -- \
    $WT/tools/tinygen/run_parity_mimo2_decode.sh $L $S/n17/build-$p > $S/n17/final-n16-$p.log 2>&1
  echo "n16 $p wrapper_rc=$?"
  rm -rf $S/n17/n16work-$p
  rm -rf $S/n17/n10work-$p
  F16FA_WORK=$S/n17/n10work-$p python3 $WT/tools/ci/check_unittest.py --step n10-f16fa --checkout $L -- \
    $WT/tools/tinygen/run_parity_f16fa.sh $L $S/n17/build-$p > $S/n17/final-n10-$p.log 2>&1
  echo "n10 $p wrapper_rc=$?"
  rm -rf $S/n17/n10work-$p
done
python3 $M/2026-10-11-n17-pindiff.py $S/n17/work-ec/n17 $S/n17/work-r1/n17
echo "== one-head compatibility of the dump tool (ec7630a)"
bash $M/2026-10-11-n17-n16compat.sh $S/n17/build-ec-old/dump-tensors $S/n17/build-ec/dump-tensors $S/llama-ec7630a $WT $S/n17/n16compat
for p in ec r1; do
  L=$(pin $p)
  echo "== bites $p ($(git -C $L rev-parse --short HEAD))"
  bash $M/2026-10-11-n17-bites.sh $WT $S/n17/work-$p $L $S/n17/bites-$p
  echo "== foreign probe $p"
  python3 $M/2026-10-11-n17-foreign-probe.py $WT $S/n17/work-$p/n17 $L
  echo "== flash-attention op probe $p"
  python3 $M/2026-10-11-n17-fa-op-probe.py $WT $S/n17/work-$p/n17 $L
done
for p in ec r1; do
  L=$(pin $p)
  rm -rf $S/n17/mut/$p; mkdir -p $S/n17/mut/$p
  python3 $M/2026-10-11-n17-mutations.py $WT $S/n17/work-$p $L $S/n17/mut/$p > $S/n17/final-mut-$p.log 2>&1
  echo "mutation $p rc=$?"
done
echo FINAL-DONE
