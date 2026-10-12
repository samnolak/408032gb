#!/usr/bin/env bash
# Mutations of the N13 audit fixes (FAIL 1, MINOR 3) on git-archive copies of HEAD: each must make
# tests/test_check_unittest.py fail.
set -u
N=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n13
M=$N/${MUTDIR:-mut-audit-final}; mkdir -p "$M"
fresh() { local d="$M/$1"; mkdir -p "$d"; git -C /home/user/wt-n13 archive HEAD | tar -x -C "$d"; echo "$d"; }
run() {   # run <name> <description>; the copy is $M/<name>
  local out rc
  out=$(cd "$M/$1" && python3 -m unittest tests.test_check_unittest 2>&1); rc=$?
  echo "$1 rc=$rc | $(echo "$out" | grep -E '^Ran |^OK|^FAILED' | tail -2 | tr '\n' ' ')| $(echo "$out" | grep -E '^(FAIL|ERROR):' | sed -E 's/ \(tests\.test_check_unittest\.[A-Za-z]+\.[a-z_]+\)//' | sort -u | tr '\n' ' ')| $2"
}
must() { grep -qF -- "$2" "$1" || { echo "mutation anchor missing in $1: $2"; exit 1; }; }
d=$(fresh baseline); run baseline "no mutation"
d=$(fresh binding_off)
must "$d/tools/ci/check_unittest.py" '    target, what = command_target(cmd), "the driver'"'"'s checkout"'
sed -i 's/^    target, what = command_target(cmd), "the driver'"'"'s checkout"$/    return None/' "$d/tools/ci/check_unittest.py"
run binding_off "checker: the command is not bound to the step's commit (check_command_commit returns None)"
d=$(fresh gguf_binding_off)
must "$d/tools/ci/check_unittest.py" 'if target is None and step["pin"] == "llama.cpp" and env.get("LLAMA_CPP_GGUF_PY"):'
sed -i 's/if target is None and step\["pin"\] == "llama.cpp" and env.get("LLAMA_CPP_GGUF_PY"):/if False:/' "$d/tools/ci/check_unittest.py"
run gguf_binding_off "checker: inline commands not bound through LLAMA_CPP_GGUF_PY"
d=$(fresh g1_other_checkout)
must "$d/.github/workflows/ci.yml" '--step g1-glm --checkout ../llama.cpp -- tools/tinygen/run_parity.sh ../llama.cpp ../build-tools'
sed -i 's|--step g1-glm --checkout ../llama.cpp -- tools/tinygen/run_parity.sh ../llama.cpp ../build-tools|--step g1-glm --checkout ../llama.cpp -- tools/tinygen/run_parity.sh ../llama-other ../build-tools|' "$d/.github/workflows/ci.yml"
run g1_other_checkout "ci.yml: the G1 driver runs on ../llama-other while the step checks ../llama.cpp"
d=$(fresh gguf_other_checkout)
must "$d/.github/workflows/ci.yml" 'LLAMA_CPP_GGUF_PY=../llama.cpp/gguf-py python3 tools/ci/check_unittest.py --step gguf-audit'
sed -i 's|LLAMA_CPP_GGUF_PY=../llama.cpp/gguf-py python3 tools/ci/check_unittest.py --step gguf-audit|LLAMA_CPP_GGUF_PY=../llama-other/gguf-py python3 tools/ci/check_unittest.py --step gguf-audit|' "$d/.github/workflows/ci.yml"
run gguf_other_checkout "ci.yml: the GGUF-audit step reads gguf-py of ../llama-other"
d=$(fresh ik_tool_step)
printf '      - name: a new ik driver outside tools/tinygen, unwrapped\n        run: tools/ik/run_something.sh ../ik_llama.cpp ../build-ik\n' >> "$d/.github/workflows/parity-r1-pin.yml"
run ik_tool_step "parity-r1-pin.yml: an unwrapped tools/ik/*.sh step appended to job ik-glm5next"
d=$(fresh renamed_step)
must "$d/.github/workflows/parity-r1-pin.yml" '      - name: llama.cpp at the R1 engine pin (tools/engines/pins.json)'
sed -i 's|      - name: llama.cpp at the R1 engine pin (tools/engines/pins.json)|      - name: llama.cpp at the R1 engine pin, renamed|' "$d/.github/workflows/parity-r1-pin.yml"
run renamed_step "parity-r1-pin.yml: an allow-listed step renamed (stale entry, unlisted step)"
