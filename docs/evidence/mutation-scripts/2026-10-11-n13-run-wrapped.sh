#!/usr/bin/env bash
# Every parity step of ci.yml (cpu-parity) and parity-r1-pin.yml (parity, ik-glm5next) as the workflows now run it:
# python3 tools/ci/check_unittest.py --step <id> --checkout <checkout> -- <driver or unittest command>, from a scratch
# cwd, at ec7630a (ci.yml) and 79e2e74 / ik 5194a9e (parity-r1-pin.yml). Logs: $N/wrapped/<pin>-<step>.log.
set -u
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
N=$S/n13; WT=/home/user/wt-n13; CHK="python3 $WT/tools/ci/check_unittest.py"
OUT=$N/${OUTDIR:-wrapped}
mkdir -p "$OUT" "$N/tmp" "$N/cwd"
export TMPDIR=$N/tmp PYTHONPATH=$WT
cd "$N/cwd"
note() {   # note <pin> <step> <rc> <t0> <log>
  echo "$1 $2 wrapper_rc=$3 wall_s=$(( $(date +%s) - $4 )) | $(grep -E '^Ran |^OK|^FAILED' "$5" | tr '\n' ' ')| $(grep -E '^::(notice|error)::' "$5" | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
  find "$N/tmp" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}
for pin in ${PINS:-ec r1}; do
  if [ "$pin" = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  for pair in run_parity:g1-glm run_parity_mimo:g1-mimo run_parity_qwen4exp:n2-qwen4exp \
              run_parity_qwen4exp_decode:n6-qwen4exp-decode run_parity_glm5next_decode:n9-glm5next-decode \
              run_parity_f16fa:n10-f16fa run_parity_quant:n10-quant; do
    d=${pair%%:*}; st=${pair#*:}; t0=$(date +%s)
    $CHK --step "$st" --checkout "$L" -- "$WT/tools/tinygen/$d.sh" "$L" "$N/build-$pin" > "$OUT/$pin-$st.log" 2>&1
    note "$pin" "$st" $? "$t0" "$OUT/$pin-$st.log"
  done
  if [ "$pin" = ec ]; then
    t0=$(date +%s)
    GGML_GDN_HARNESS=$WT/tests/ggml_parity/gdn_harness LLAMA_CPP_GGUF_PY=$L/gguf-py \
      $CHK --step ggml-gdn --checkout "$L" -- python3 -m unittest tests.test_kda_vs_ggml tests.test_fitplan -v > "$OUT/ec-ggml-gdn.log" 2>&1
    note ec ggml-gdn $? "$t0" "$OUT/ec-ggml-gdn.log"
    t0=$(date +%s)
    LLAMA_CPP_GGUF_PY=$L/gguf-py $CHK --step gguf-audit --checkout "$L" -- \
      python3 -m unittest tests.test_roofline tests.test_cachesim tests.test_gguf_quant_audit tests.test_offline_weights -v > "$OUT/ec-gguf-audit.log" 2>&1
    note ec gguf-audit $? "$t0" "$OUT/ec-gguf-audit.log"
  else
    t0=$(date +%s)
    LLAMA_CPP_GGUF_PY=$L/gguf-py $CHK --step n12-ik-glm5next --checkout "$S/n12/ik-src" -- \
      "$WT/tools/tinygen/run_parity_ik_glm5next.sh" "$S/n12/ik-src" "$N/build-ik" > "$OUT/ik-n12-ik-glm5next.log" 2>&1
    note ik n12-ik-glm5next $? "$t0" "$OUT/ik-n12-ik-glm5next.log"
  fi
done
