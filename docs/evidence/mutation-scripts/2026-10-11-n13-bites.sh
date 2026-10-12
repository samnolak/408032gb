#!/usr/bin/env bash
# "Prove it bites": copies of the drivers with the environment of their unittest line broken, run through the checker
# exactly like the CI steps. Variants:
#   unset  the dump variable is removed from the unittest line (the env plumbing broke)
#   empty  the dump variable points at a fresh empty directory (the dumps went elsewhere)
#   nomtp  run_parity_glm5next_decode.sh with its MTP run disabled (the pin-dependent skip forced at both pins)
# The copies differ from tools/tinygen/<driver>.sh only by ROOT (fixed to the worktree) and the break; the diff of each
# copy is saved next to it. before_rc = what the step returned without the checker (the command's own status: the
# checker returns it unchanged when non-zero, check_unittest.py main()), after_rc = what the step returns now.
set -u
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
N=$S/n13; WT=/home/user/wt-n13; CHK="python3 $WT/tools/ci/check_unittest.py"
B=$N/bite; mkdir -p "$B/copies" "$B/logs" "$N/tmp" "$N/cwd"
export TMPDIR=$N/tmp PYTHONPATH=$WT
cd "$N/cwd"
declare -A VAR=(
  [run_parity]='GLM_TINY_DUMP="$WORK/dump"'
  [run_parity_mimo]='MIMO_TINY_DUMP="$WORK/dump"'
  [run_parity_qwen4exp]='QWEN4EXP_TINY_DUMP="$WORK/dump"'
  [run_parity_qwen4exp_decode]='QWEN4EXP_DECODE_DUMP="$WORK/n6"'
  [run_parity_glm5next_decode]='GLM5NEXT_DECODE_DUMP="$WORK/n9"'
  [run_parity_f16fa]='GLM5NEXT_F16FA_DUMP="$WORK/g"'
  [run_parity_quant]='GLM5NEXT_QUANT_DUMP="$WORK/glm5next"'
  [run_parity_ik_glm5next]='IK_GLM5NEXT_DUMP="$WORK/n12"'
)
declare -A STEP=([run_parity]=g1-glm [run_parity_mimo]=g1-mimo [run_parity_qwen4exp]=n2-qwen4exp
  [run_parity_qwen4exp_decode]=n6-qwen4exp-decode [run_parity_glm5next_decode]=n9-glm5next-decode
  [run_parity_f16fa]=n10-f16fa [run_parity_quant]=n10-quant [run_parity_ik_glm5next]=n12-ik-glm5next)
ROOTLINE='\$(cd "\$(dirname "\$0")/\.\./\.\." && pwd)'
mkcopy() {   # mkcopy <driver> <variant> -> path of the copy
  local d=$1 v=$2 src="$WT/tools/tinygen/$1.sh" dst="$B/copies/$2-$1.sh" pat
  pat=$(printf '%s' "${VAR[$d]}" | sed 's/[$]/\\$/g; s/[.]/\\./g')
  [ "$(grep -c -- "${VAR[$d]}" "$src")" = 1 ] || { echo "pattern ${VAR[$d]} not unique in $src" >&2; exit 1; }
  sed -e "s|^ROOT=$ROOTLINE|ROOT=$WT|" "$src" > "$dst"
  case $v in
    unset) sed -i "s|$pat ||" "$dst" ;;
    empty) sed -i "s|$pat|${VAR[$d]%%=*}=\"\$(mktemp -d)\"|" "$dst" ;;
    nomtp) sed -i 's|^if grep -q "graph_mtp::graph_mtp" "\$LLAMA/src/models/glm5-next.cpp"; then|if false; then|' "$dst" ;;
  esac
  chmod +x "$dst"
  diff -u "$src" "$dst" > "$dst.diff"
  [ "$(grep -c '^[-+][^-+]' "$dst.diff")" -ge 4 ] || { echo "copy $dst: the break did not apply" >&2; cat "$dst.diff" >&2; exit 1; }
  echo "$dst"
}
row() {   # row <pin> <step> <variant> <after rc> <t0> <log>
  local before
  before=$(grep -oP '::error::.*the command exited with status \K[0-9]+' "$6" || echo 0)
  echo "$1 $2 $3 before_rc=$before after_rc=$4 wall_s=$(( $(date +%s) - $5 )) | $(grep -E '^Ran |^OK|^FAILED' "$6" | tr '\n' ' ')| $(grep -E '^::(notice|error)::' "$6" | head -3 | cut -c1-260 | tr '\n' ' ')" | tee -a "$B/summary.txt"
  find "$N/tmp" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}
for pin in ${PINS:-ec r1}; do
  if [ "$pin" = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  for d in ${DRIVERS:-run_parity run_parity_mimo run_parity_qwen4exp run_parity_qwen4exp_decode run_parity_glm5next_decode run_parity_f16fa run_parity_quant}; do
    for v in ${VARIANTS:-unset empty}; do
      c=$(mkcopy "$d" "$v") || exit 1; t0=$(date +%s); log="$B/logs/$pin-$v-${STEP[$d]}.log"
      $CHK --step "${STEP[$d]}" --checkout "$L" -- "$c" "$L" "$N/build-$pin" > "$log" 2>&1
      row "$pin" "${STEP[$d]}" "$v" $? "$t0" "$log"
    done
  done
  if [ -z "${DRIVERS:-}" ]; then
    c=$(mkcopy run_parity_glm5next_decode nomtp) || exit 1; t0=$(date +%s); log="$B/logs/$pin-nomtp-n9-glm5next-decode.log"
    $CHK --step n9-glm5next-decode --checkout "$L" -- "$c" "$L" "$N/build-$pin" > "$log" 2>&1
    row "$pin" n9-glm5next-decode nomtp $? "$t0" "$log"
  fi
done
if [ -z "${DRIVERS:-}" ]; then
  L=$S/llama-ec7630a
  # the two inline steps of ci.yml: unset = the variable dropped from the step's command line
  t0=$(date +%s); log="$B/logs/ec-unset-ggml-gdn.log"
  LLAMA_CPP_GGUF_PY=$L/gguf-py $CHK --step ggml-gdn --checkout "$L" -- python3 -m unittest tests.test_kda_vs_ggml tests.test_fitplan -v > "$log" 2>&1
  row ec ggml-gdn "unset(GGML_GDN_HARNESS)" $? "$t0" "$log"
  t0=$(date +%s); log="$B/logs/ec-empty-ggml-gdn.log"; e=$(mktemp -d)
  GGML_GDN_HARNESS=$e/gdn_harness LLAMA_CPP_GGUF_PY=$L/gguf-py $CHK --step ggml-gdn --checkout "$L" -- python3 -m unittest tests.test_kda_vs_ggml tests.test_fitplan -v > "$log" 2>&1
  row ec ggml-gdn "empty(GGML_GDN_HARNESS=<empty dir>/gdn_harness)" $? "$t0" "$log"
  t0=$(date +%s); log="$B/logs/ec-unset-gguf-audit.log"
  $CHK --step gguf-audit --checkout "$L" -- python3 -m unittest tests.test_roofline tests.test_cachesim tests.test_gguf_quant_audit tests.test_offline_weights -v > "$log" 2>&1
  row ec gguf-audit "unset(LLAMA_CPP_GGUF_PY)" $? "$t0" "$log"
  t0=$(date +%s); log="$B/logs/ec-empty-gguf-audit.log"; e=$(mktemp -d)
  LLAMA_CPP_GGUF_PY=$e $CHK --step gguf-audit --checkout "$L" -- python3 -m unittest tests.test_roofline tests.test_cachesim tests.test_gguf_quant_audit tests.test_offline_weights -v > "$log" 2>&1
  row ec gguf-audit "empty(LLAMA_CPP_GGUF_PY=<empty dir>)" $? "$t0" "$log"
  # the ik job
  for v in unset empty; do
    c=$(mkcopy run_parity_ik_glm5next "$v") || exit 1; t0=$(date +%s); log="$B/logs/ik-$v-n12-ik-glm5next.log"
    LLAMA_CPP_GGUF_PY=$S/llama.cpp/gguf-py $CHK --step n12-ik-glm5next --checkout "$S/n12/ik-src" -- "$c" "$S/n12/ik-src" "$N/build-ik" > "$log" 2>&1
    row ik n12-ik-glm5next "$v" $? "$t0" "$log"
  done
fi
