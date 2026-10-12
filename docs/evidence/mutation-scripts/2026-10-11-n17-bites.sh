#!/usr/bin/env bash
# N17: the CI step n17-mimo2-mtp-heads-fa fails through tools/ci/check_unittest.py when its tests did not all run: no dump
# environment, MIMO2_N17_DUMP pointing at an empty directory, a dump directory without the MTP context of mtp3, or one
# without the fa/hd192 run. The step's own test command (the driver's unittest line) runs on saved dumps through the
# checker's command mode with --commit. usage: 2026-10-11-n17-bites.sh <repo> <driver work dir (holds n17/)> <llama.cpp> <scratch>
set -uo pipefail
REPO=$1; WORK=$2; LLAMA=$3; T=$4
SHA=$(git -C "$LLAMA" rev-parse HEAD)
MODS="tests.test_mimo2_mtp_heads_parity tests.test_mimo2_f16fa_parity tests.test_mimo2_heads_ref tests.test_fattn_sinks_ref"
rm -rf "$T"; mkdir -p "$T/empty"
cp -al "$WORK/n17" "$T/no-mtp"; rm -rf "$T/no-mtp/mtp3/mtp"
cp -al "$WORK/n17" "$T/no-hd192"; rm -rf "$T/no-hd192/fa/hd192"
case_() {  # case_ <label> <env...>
  local label=$1; shift
  (cd "$REPO" && env -u MIMO2_N17_DUMP -u LLAMA_CPP_GGUF_PY -u PARITY_LLAMA_COMMIT "$@" \
     python3 tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --commit "$SHA" -- python3 -m unittest $MODS -v \
     > "$T/$label.log" 2>&1)
  echo "$label: checker rc=$? | $(grep -E '^Ran |^OK|^FAILED' "$T/$label.log" | tr '\n' ' ')| $(grep -o '::error::.*' "$T/$label.log" | head -2 | cut -c1-160 | tr '\n' ' ')"
}
case_ complete MIMO2_N17_DUMP="$WORK/n17" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
case_ no-env
case_ empty-dir MIMO2_N17_DUMP="$T/empty" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
case_ no-mtp-context MIMO2_N17_DUMP="$T/no-mtp" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
case_ no-hd192-run MIMO2_N17_DUMP="$T/no-hd192" LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
rm -rf "$T/no-mtp" "$T/no-hd192"
