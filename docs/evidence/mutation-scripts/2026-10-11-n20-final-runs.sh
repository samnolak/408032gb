#!/usr/bin/env bash
# N20 final runs (docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md section 5), one pin per call:
#   bash docs/evidence/mutation-scripts/2026-10-11-n20-final-runs.sh <repo> <llama.cpp checkout at the pin> <work dir> <ec|r1>
# <work dir> holds llama-<p>-guard (patched copy, patches/llama.cpp/apply.sh), build-<p>-guard (its build),
# build-<p>-up (build of the unpatched checkout). Steps:
#   1 the N20 step through tools/ci/check_unittest.py, patched (as CI runs it, the copy and build dir given)
#   2 the N20 driver in upstream mode (the unpatched checkout): the same 13 plans, the pin's own behaviour expected
#   3 byte identity of the dumps of the 13 plans, patched vs upstream
#   4 earlier steps on the patched build through the checker: n2-qwen4exp, n6-qwen4exp-decode, n9-glm5next-decode
#   5 the N19 step unchanged on the patched build (no QWEN4EXP_RS_GUARD): which tests fail is the guard's effect
set -uo pipefail
WT=$(cd "${1:?repo}" && pwd); L=$(cd "${2:?llama.cpp checkout}" && pwd); W=$(cd "${3:?work dir}" && pwd); P=${4:?ec|r1}
COPY=$W/llama-$P-guard; B=$W/build-$P-guard; BU=$W/build-$P-up
cd "$W"
echo "repo $(git -C "$WT" rev-parse --short HEAD) ($(git -C "$WT" status --short | wc -l) changed files); llama.cpp $(git -C "$L" rev-parse HEAD) ($(git -C "$L" status --short | wc -l) changed files); copy $(git -C "$COPY" rev-parse --short HEAD), $(git -C "$COPY" diff --stat | tail -n 1)"
step() {   # step <label> <log> <command...>
  local label=$1 log=$2; shift 2
  local t0; t0=$(date +%s)
  "$@" > "$log" 2>&1
  echo "$label rc=$? wall_s=$(( $(date +%s) - t0 ))"
}
echo "== 1 n20-qwen4exp-rs-guard (patched) through the checker"
rm -rf "$W/w-$P-guard"
step "n20 patched $P" "$W/n20-$P-guard.log" env QWEN4EXP_RS_GUARD_SRC="$COPY" QWEN4EXP_RS_GUARD_WORK="$W/w-$P-guard" \
  python3 "$WT/tools/ci/check_unittest.py" --step n20-qwen4exp-rs-guard --checkout "$L" -- \
  "$WT/tools/tinygen/run_parity_qwen4exp_rs_guard.sh" "$L" "$B"
grep -E "^(mode|[a-z_]+: [0-9]+ graphs|Ran |OK|FAILED|::)" "$W/n20-$P-guard.log"
echo "== 2 the N20 driver, upstream mode (unpatched checkout) through the checker"
rm -rf "$W/w-$P-up"
step "n20 upstream $P" "$W/n20-$P-up.log" env QWEN4EXP_RS_GUARD_MODE=upstream QWEN4EXP_RS_GUARD_WORK="$W/w-$P-up" \
  python3 "$WT/tools/ci/check_unittest.py" --step n20-qwen4exp-rs-guard --checkout "$L" -- \
  "$WT/tools/tinygen/run_parity_qwen4exp_rs_guard.sh" "$L" "$BU"
grep -E "^(mode|[a-z_]+: [0-9]+ graphs|Ran |OK|FAILED|::)" "$W/n20-$P-up.log"
echo "== 3 dumps patched vs upstream, per plan"
for d in "$W/w-$P-guard/n20"/*/; do
  run=$(basename "$d")
  if diff -rq "$d" "$W/w-$P-up/n20/$run" > /dev/null; then echo "$run: identical ($(find "$d" -type f | wc -l) files)"
  else echo "$run: differs ($(diff -rq "$d" "$W/w-$P-up/n20/$run" | wc -l) files)"; fi
done
echo "== 4 earlier steps on the patched build (its copy as the checkout) through the checker"
for st in n2-qwen4exp:run_parity_qwen4exp.sh n6-qwen4exp-decode:run_parity_qwen4exp_decode.sh \
          n9-glm5next-decode:run_parity_glm5next_decode.sh; do
  s=${st%%:*}; drv=${st##*:}
  rm -rf "$W/tmp"; mkdir -p "$W/tmp"
  step "$s patched $P" "$W/$s-$P-guard.log" env TMPDIR="$W/tmp" GLM5NEXT_DECODE_WORK="$W/tmp/w" PYTHONPATH="$WT" \
    python3 "$WT/tools/ci/check_unittest.py" --step "$s" --checkout "$COPY" -- "$WT/tools/tinygen/$drv" "$COPY" "$B"
  tail -n 1 "$W/$s-$P-guard.log"
done
rm -rf "$W/tmp"
echo "== 5 the N19 step unchanged on the patched build (QWEN4EXP_RS_GUARD unset)"
rm -rf "$W/w19-$P-guard"
step "n19 on the patched build $P" "$W/n19-$P-guard.log" env -u QWEN4EXP_RS_GUARD QWEN4EXP_MULTISEQ_WORK="$W/w19-$P-guard" \
  "$WT/tools/tinygen/run_parity_qwen4exp_multiseq.sh" "$COPY" "$B"
grep -E "^(FAIL|ERROR):|^Ran |^OK|^FAILED" "$W/n19-$P-guard.log"
rm -rf "$W/w19-$P-guard"
