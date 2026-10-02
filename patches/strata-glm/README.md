# Patches on sergqwer/strata-glm @ed37419 (ADR-007)

Applied in order by CI (job `strata-glm-sm89`) and by hand with `git -C strata-glm apply <patch>`.
Upstream stays untouched; every change lives here.

| patch | what | verified |
|---|---|---|
| 0001 | Linux: `_fseeki64` (MSVC) -> `fseeko` | syntax on Linux; CI build |
| 0002 | G4 stage 1: layer-pipeline stages, `--layer-split K1,K2,..` `--gpus D0,D1,..` | syntax, no new warnings; **NOT RUN on a GPU** |

## 0002: how to check it (needs a GPU, in this order)

1. **One stage = original.** Without the new flags nothing changes: same tokens as unpatched strata-glm.
2. **Bit-exact split on one GPU.** All stages on the same card must give identical tokens and logits:
   ```
   strata-glm --pack P --profile P/expert-profile-boot.bin --tokens t.txt --chunk 8192 --max-new 64 \
       --dump-logits a.f32                                      # reference
   strata-glm ... --layer-split 14,25,35 --vram-experts 300 --dump-logits b.f32   # 4 stages, GPU 0
   cmp a.f32 b.f32 && diff <(grep output: ref.log) <(grep output: split.log)
   ```
   `--vram-experts` per stage keeps the stages from taking each other's VRAM on one card.
   Expected difference: none in tokens. Logits identical unless the tiers move experts at different times
   (tier placement does not change the math: the same expert computes the same result from VRAM or RAM).
3. **Four GPUs.** `--layer-split 14,25,35 --gpus 0,1,2,3`: same tokens as step 2.

Known limits of stage 1 (documented, to be done in later patches): no expert prediction across a stage
boundary (the next stage's first layer runs unpredicted); stages run one after another (no overlap of one
stage's compute with another's); the disk tier still exists only on Windows (not needed when VRAM + RAM
hold every expert, see tools/fitplan.py).
