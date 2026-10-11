#!/usr/bin/env bash
# N16: the CI step n16-mimo2-decode fails through tools/ci/check_unittest.py when its tests did not all run: no dump
# environment, MIMO2_DECODE_DUMP pointing at an empty directory, a dump directory missing the mtp run, or one missing a
# quant run. The step's own test command (the driver's unittest line) is run on saved dumps through the checker's
# command mode with --commit. usage: 2026-10-11-n16-bites.sh <repo> <driver work dir (holds n16/)> <llama.cpp> <scratch>
set -uo pipefail
REPO=$1; WORK=$2; LLAMA=$3; T=$4
SHA=$(git -C "$LLAMA" rev-parse HEAD)
MODS="tests.test_mimo2_decode_parity tests.test_mimo2_mtp_parity tests.test_mimo2_chunked_ref tests.test_mimo2_quant_parity tests.test_mxfp4_ref"
rm -rf "$T"; mkdir -p "$T/empty"
cp -al "$WORK/n16" "$T/no-mtp"; rm -rf "$T/no-mtp/mtp"
cp -al "$WORK/n16" "$T/no-quant-run"; rm -rf "$T/no-quant-run/mxfp4/decode-nr"
case_() {  # case_ <label> <env...>
  local label=$1; shift
  (cd "$REPO" && env -u MIMO2_DECODE_DUMP -u LLAMA_CPP_GGUF_PY -u PARITY_LLAMA_COMMIT "$@" \
     python3 tools/ci/check_unittest.py --step n16-mimo2-decode --commit "$SHA" -- python3 -m unittest $MODS -v \
     > "$T/$label.log" 2>&1)
  echo "$label: checker rc=$? | $(grep -E '^Ran |^OK|^FAILED' "$T/$label.log" | tr '\n' ' ')| $(grep -o '::error::.*' "$T/$label.log" | head -2 | cut -c1-160 | tr '\n' ' ')"
}
case_ complete MIMO2_DECODE_DUMP="$WORK/n16" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
case_ no-env
case_ empty-dir MIMO2_DECODE_DUMP="$T/empty" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
case_ no-mtp-run MIMO2_DECODE_DUMP="$T/no-mtp" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
case_ no-quant-run MIMO2_DECODE_DUMP="$T/no-quant-run" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
rm -rf "$T/no-mtp" "$T/no-quant-run"
