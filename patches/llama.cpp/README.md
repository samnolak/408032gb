# Patches on ggml-org/llama.cpp

Upstream stays untouched (AGENTS.md "Upstream pins"); a patch is applied to a copy of the checkout:
`git -C <copy of llama.cpp @base> apply <patch>`. None of these patches is used by CI today.

| patch | base | what | verified |
|---|---|---|---|
| 0001-mtp-heads-slot-replay | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin) | `common/speculative.cpp` draft-mtp `process()`: each ubatch of a target batch is placed once for all NextN heads (heads 1..H-1 re-use head 0's KV cells through the new staging API `llama_memory_set_slot_replay`, `src/llama-ext.h`; `llama_kv_cache::prepare()` honours it). Fixes the N17 finding (head 0 of a multi-head MTP context reads K/V of other positions after a batch that wraps the SWA ring) and the multi-ubatch variant found in N21. `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` restores the upstream order for A/B runs. | CPU only, tiny mimo2 model with 3 MTP heads: `tools/tinygen/run_parity_mimo2_n21.sh <patched copy> <build>`, docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md. `git apply --check`: applies at 79e2e74; at ec7630a `src/llama-ext.h` does not apply (other context), the other 7 files do. GPU, llama-server binary and real MiMo weights: NOT RUN |
