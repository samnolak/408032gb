# Patches on sergqwer/strata-glm @ed37419 (ADR-007)

Applied in order by CI (job `strata-glm-sm89`) and by hand with `git -C strata-glm apply <patch>`.
Upstream stays untouched; every change lives here.

| patch | what | verified |
|---|---|---|
| 0001 | Linux: `_fseeki64` (MSVC) -> `fseeko` | syntax on Linux; CI build |
| 0002 | G4 stage 1: layer-pipeline stages, `--layer-split K1,K2,..` `--gpus D0,D1,..` | **RUN on 1×3090 sm_86** (2026-10-03): bit-exact single-GPU split (`--layer-split 14,25,35 --gpus 0,0,0,0`), logits byte-identical, same tokens; synthetic checkpoint (`glm-synthetic`), real weights and 4 GPUs NOT RUN |
| 0003 | Linux disk tier: `DiskReader` via `O_DIRECT` `pread`, buffered fallback on `EINVAL` | functional test on real files (`tests/strata_glm/run_diskreader_test.sh`, CI): mirrors, unaligned offsets, file tail, unaligned buffer fallback, priorities, missing file |
| 0004 | Linux RAM tier: `TieredExpertSource` on posix (`O_DIRECT` `pread` windows, buffered readers) | by grishberg; RUN on 1×3090 sm_86 (2026-10-03): bit-exact against the disk-only run on a synthetic pack; real weights NOT RUN |
| 0005 | `src/program/generate.cpp`: comment of the small-tier gate corrected (no behaviour change) | by grishberg; from Grigory-Rylov/strata-glm-3090 b7ee7fb |
| 0006 | `tools/glm_pack.py` runs on Linux (`/proc` private-memory guard, language-model prefix resolved) | by grishberg; from Grigory-Rylov/strata-glm-3090 13bbdc1; NOT RUN by us (needs the safetensors checkpoint) |

## 0002: how to check it (needs a GPU, in this order)

1. **One stage = original.** Without the new flags nothing changes: same tokens as unpatched strata-glm.
2. **Bit-exact split on one GPU.** All stages on the same card must give identical tokens and logits.
   Use the exact mode (`--skip-disk 0`, `--skip-ram 0`): by default an unpredicted disk-only expert under 10%
   of the routing weight is skipped, and which experts are disk-only depends on the tiers, which differ
   between one engine and four stages. One 32 GB card plus 128 GB RAM does not hold all 163 GiB of experts,
   so this check reads from disk: it needs patch 0003 on Linux.
   ```
   strata-glm --pack P --profile P/expert-profile-boot.bin --tokens t.txt --chunk 8192 --max-new 64 \
       --skip-disk 0 --dump-logits a.f32 > ref.log                                      # reference
   strata-glm ... --skip-disk 0 --layer-split 14,25,35 --vram-experts 300 --dump-logits b.f32 > split.log
   cmp a.f32 b.f32 && diff <(grep output: ref.log) <(grep output: split.log)
   ```
   `--vram-experts` per stage keeps the stages from taking each other's VRAM on one card.
   Expected: identical tokens. Logits: identical if the expert kernels' results do not depend on where a
   blob is read from (VRAM or pinned RAM through UVA); **CONFIRMED identical** on 1×3090 (2026-10-03,
   synthetic checkpoint, `--dense-fp4 all` to fit 24 GiB).
3. **Four GPUs.** `--layer-split 14,25,35 --gpus 0,1,2,3`: same tokens as step 2.

Known limits of stage 1 (documented, to be done in later patches): no expert prediction across a stage
boundary (the next stage's first layer runs unpredicted); stages run one after another (no overlap of one
stage's compute with another's); the disk tier still exists only on Windows (not needed when VRAM + RAM
hold every expert, see tools/fitplan.py).


`ed37419` + patches 0001-0006 equals `Grigory-Rylov/strata-glm-3090` @`b115f38` for `src/` and `tools/` (tree hashes
`93cd5ece7e2f44912a06dc06b1d8baa69c376ba7`, `58eb5268d3b804277147b5e9d25fc4cc60f2b284`; CONFIRMED 2026-10-03). The fork's README/PROBLEMS are its own.
