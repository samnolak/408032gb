#!/usr/bin/env bash
# Mutations of the N13 mechanism on git-archive copies of HEAD: each must make tests/test_check_unittest.py fail.
set -u
N=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n13
M=$N/${MUTDIR:-mut}; mkdir -p "$M"
fresh() { local d="$M/$1"; mkdir -p "$d"; git -C /home/user/wt-n13 archive HEAD | tar -x -C "$d"; echo "$d"; }
run() {   # run <name> <description>; the copy is $M/<name>
  local out rc
  out=$(cd "$M/$1" && python3 -m unittest tests.test_check_unittest 2>&1); rc=$?
  echo "$1 rc=$rc | $(echo "$out" | grep -E '^Ran |^OK|^FAILED' | tr '\n' ' ')| $(echo "$out" | grep -E '^(FAIL|ERROR):' | sed -E 's/ \(tests\.test_check_unittest\.[A-Za-z]+\.[a-z_]+\)//' | sort -u | tr '\n' ' ')| $2"
}
d=$(fresh baseline); run baseline "no mutation"
d=$(fresh unwrap_g1)
sed -i 's|run: python3 tools/ci/check_unittest.py --step g1-glm --checkout ../llama.cpp -- tools/tinygen/run_parity.sh|run: tools/tinygen/run_parity.sh|' "$d/.github/workflows/ci.yml"
run unwrap_g1 "ci.yml: the G1 step runs its driver without the checker"
d=$(fresh new_step)
printf '      - name: a new parity step without the checker\n        run: tools/tinygen/run_parity_mimo.sh ../llama.cpp ../build-tools\n' >> "$d/.github/workflows/parity-r1-pin.yml"
run new_step "parity-r1-pin.yml: a new driver step appended to job ik-glm5next without the checker"
d=$(fresh no_ec_n9)
python3 - "$d/tools/ci/parity_expectations.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); del d["steps"]["n9-glm5next-decode"]["expect"]["ec7630a640789c393694fb194f1bbbf0369fc62d"]; json.dump(d, open(p, "w"))
PY
run no_ec_n9 "expectations: the ec7630a entry of n9-glm5next-decode removed"
d=$(fresh pin_bump)
sed -i 's/79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6/0123456789abcdef0123456789abcdef01234567/' "$d/tools/engines/pins.json"
run pin_bump "tools/engines/pins.json: llama.cpp R1 pin changed, expectations not updated"
d=$(fresh mtp_reason)
sed -i 's/no MTP run under .+ \\\\(llama\\\\.cpp ec7630a/no MTP run under .+ \\\\(llama\\\\.cpp 79e2e74/' "$d/tools/ci/parity_expectations.json"
grep -c 'llama\\\\.cpp 79e2e74' "$d/tools/ci/parity_expectations.json" > /dev/null || echo "mtp_reason: sed did not apply"
run mtp_reason "expectations: the MTP skip reason regex names the wrong commit"
d=$(fresh modules)
sed -i 's/"modules": \["tests.test_qwen4exp_decode_parity", "tests.test_qwen4exp_mtp_parity", "tests.test_qwen4exp_chunked_ref"\]/"modules": ["tests.test_qwen4exp_decode_parity", "tests.test_qwen4exp_chunked_ref"]/' "$d/tools/ci/parity_expectations.json"
run modules "expectations: n6 lists one module fewer than its driver runs"
d=$(fresh no_count)
sed -i 's/    if result\["ran"\] != exp\["ran"\]:/    if False:/' "$d/tools/ci/check_unittest.py"
run no_count "checker: Ran N not compared"
d=$(fresh no_skipcount)
sed -i 's/    if got_k != len(want):/    if False:/' "$d/tools/ci/check_unittest.py"
run no_skipcount "checker: skipped=K not compared"
d=$(fresh any_id)
sed -i 's/hit = next((k for k in free if tid == want\[k\]\["test"\]/hit = next((k for k in free if True/' "$d/tools/ci/check_unittest.py"
run any_id "checker: a skip matches an expected entry by reason only"
d=$(fresh no_status)
sed -i 's/    if result\["status"\] != "OK":/    if False:/' "$d/tools/ci/check_unittest.py"
run no_status "checker: status not compared"
d=$(fresh no_unattributed)
sed -i 's/    if len(result\["skips"\]) != got_k:/    if False:/' "$d/tools/ci/check_unittest.py"
run no_unattributed "checker: skip lines not counted against skipped=K"
d=$(fresh env_ignored)
sed -i 's/    if env.get(var) and env\[var\] != sha:/    if False:/' "$d/tools/ci/check_unittest.py"
run env_ignored "checker: PARITY_LLAMA_COMMIT disagreeing with the checkout is ignored"
d=$(fresh rc_lost)
sed -i 's/                return rc$/                return 0/' "$d/tools/ci/check_unittest.py"
run rc_lost "checker: a failing command is reported but the step returns 0"
