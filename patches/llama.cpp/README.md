# Patches on ggml-org/llama.cpp

Upstream stays untouched (AGENTS.md "Upstream pins"); a patch is applied to a copy of the checkout, never in place.
Two layouts exist (merged in claude/n25-integration, docs/evidence/2026-10-11-n25-integration.md):

* `patches/llama.cpp/<short sha>/*.patch` (N20): one directory per pin; `patches/llama.cpp/apply.sh <checkout> <new dir>`
  applies every patch of the checkout's pin to a `git clone --shared` copy (HEAD stays the pin). CI applies the N20
  guard in step `n20-qwen4exp-rs-guard` of job `cpu-parity` (`.github/workflows/ci.yml`, ec7630a) and of job `parity`
  (`.github/workflows/parity-r1-pin.yml`, R1 pin), through `tools/tinygen/run_parity_qwen4exp_rs_guard.sh`, in its own
  build directory `../build-tools-rs-guard`.
* `patches/llama.cpp/0001-mtp-heads-slot-replay.patch` (N21): `git -C <copy of llama.cpp @base> apply <patch>`. CI
  applies it in job `n21-patched` of `.github/workflows/parity-r1-pin.yml` (to a copy of the R1 pin, step
  `n21-mimo2-mtp-fix-patched`). apply.sh does not pick it up (it reads only the pin directories).
* `patches/llama.cpp/n24-swiglu-clamp-after-silu.patch` (N24): an experiment patch, NOT for production (its own header);
  applied (`patch -p1`, to a copy) only by `tools/quality/swiglu_clamp_experiment.sh`, never by CI or apply.sh;
  `docs/evidence/mutation-scripts/2026-10-11-n24-swiglu-patch-check.sh` compares a stock and a patched build it is
  given; docs/evidence/2026-10-11-n24-owner-decisions.md section 3.4.

Every other step runs the unpatched pins. The N20 and N21 patches touch disjoint files (N20: `src/llama-memory-recurrent.{cpp,h}`;
N21: `common/speculative.cpp`, `src/llama-context.cpp`, `src/llama-ext.h`, `src/llama-kv-cache.{cpp,h}`,
`src/llama-kv-cache-iswa.{cpp,h}`, `src/llama-memory.h`); no CI job applies both together.

| patch | base | what | verified |
|---|---|---|---|
| 79e2e74/0001-recurrent-rollback-guard, ec7630a/0001-recurrent-rollback-guard | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin), `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) | `src/llama-memory-recurrent.cpp` / `.h`: `mem_cell::rs_n` counts the snapshot planes a cell's rows hold; `seq_rm` refuses (returns false, logs "refusing a rollback") a rollback `r >= rs_n`; `seq_cp` gives the destination the source's pending `rs_idx`; `s_copy` resets the pending rollback of every sequence of the cell (N20 audit FAIL 1). The ec7630a patch also adds 79e2e74's shared-cell refusal. Details: docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md section 3. | CPU only, tiny qwen4exp model: `tools/tinygen/run_parity_qwen4exp_rs_guard.sh <checkout> <build>` (patched copy by apply.sh), docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md. GPU and real weights: NOT RUN |
| 0001-mtp-heads-slot-replay | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin) | `common/speculative.cpp` draft-mtp `process()`: each ubatch of a target batch is placed once for all NextN heads (heads 1..H-1 re-use head 0's KV cells through the new staging API `llama_memory_set_slot_replay`, `src/llama-ext.h`; `llama_kv_cache::prepare()` honours it). Fixes the N17 finding (head 0 of a multi-head MTP context reads K/V of other positions after a batch that wraps the SWA ring) and the multi-ubatch variant found in N21. `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` restores the upstream order for A/B runs (only a value starting with `1` counts; `0`, `true`, `yes` or empty keep the fix). | CPU only, tiny mimo2 model with 3 MTP heads: `tools/tinygen/run_parity_mimo2_n21.sh <patched copy> <build>`, docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md. `git apply --check`: applies at 79e2e74; at ec7630a `src/llama-ext.h` does not apply (other context), the other 7 files do. GPU, llama-server binary and real MiMo weights: NOT RUN |
