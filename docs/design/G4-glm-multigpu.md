# G4 design: GLM-5.3-Flash on 4x RTX 4080 Super with strata-glm (engine-integrator, 2026-10-02)

Base: sergqwer/strata-glm @ed37419 (ADR-007). Labels per AGENTS.md. Nothing here is measured yet.

## What exists (CONFIRMED by reading the code)

| piece | where | single-GPU assumption |
|---|---|---|
| whole engine state | `src/glm/glm_main.cpp` `struct Engine` (line 399), 85 `cudaMalloc`/host-alloc sites | one device, one set of streams |
| model constants | `glm_main.cpp:299-304`: 45 layers, 3 dense, 4096, hc 4, vocab 154880; eps, clamp 10, scale 2.5 | match our CONFIRMED config |
| expert tiers | static VRAM part + VRAM LRU + pinned RAM + disk (`DiskReader`, `Lru`, `CpuExperts`) | VRAM tier sized for one card |
| disk tier | `DiskReader` is `_WIN32` only; `open()` returns false elsewhere (lines 65-80) | not needed here (below) |
| multi-GPU | none in `src/glm` (0 hits for cudaSetDevice / peer access) | — |
| Strata layer split (Qwen path) | `src/program/generate.cpp:1117+`, `--layer-split`, `--split-device`, docs/MULTI_GPU.md | proven bit-exact hand-off on one card |

## Target placement (PROVISIONAL, tools/fitplan.py --quant NVFP4)

12,096 routed experts x 14.16 MB = 171 GB. Per card ~27 GiB is left for experts after dense weights,
caches and reserve; 4 cards hold 65-69% of the experts (8.0-8.5K), RAM holds the other 51-58 GiB.
**No disk tier.** The author's limits (13% in VRAM, disk waits) do not apply; ours are PCIe and per-card HBM bandwidth.

Stage split by expert count, 45 layers = 3 dense + 42 MoE:

| stage / GPU | layers | MoE layers | notes |
|---|---|---|---|
| 0 | 0-13 | 11 | embedding, 3 dense layers |
| 1 | 14-24 | 11 | |
| 2 | 25-34 | 10 | |
| 3 | 35-44 | 10 | final norm, lm_head (154880 x 4096) |

Exact cut points: chosen by measured free VRAM per card in G4 (Strata's `auto` does the same).

## Design

1. **Engine per stage.** Split `Engine` into a stage object owning layers `[lo, hi)`: their dense weights only
   (not a copy of all, unlike Strata's Qwen path), their KDA states, their DSA latent and indexer caches, their
   VRAM expert tier and their streams. One host thread per stage, `cudaSetDevice` once per thread.
2. **Hand-off.** Between stages the residual is the mHC state: 4 streams x 4096 fp32 = 64 KiB per token.
   Decode: 64 KiB per token per boundary. Prompt chunk of 8192 tokens: 512 MiB per boundary. Use
   `cudaMemcpyPeerAsync` (P2P works on this server, CONFIRMED by the customer) with a pinned-RAM fallback.
3. **RAM tier.** One pinned host arena; each stage registers only its layers' experts and copies them over its own
   PCIe link. The static VRAM selection per stage comes from the boot profile restricted to the stage's layers.
4. **Prefetch across a boundary.** strata-glm predicts layer l+1's experts from layer l's streams. At the last
   layer of a stage the next layer's router lives on the next card: replicate the 42 router matrices
   (288 x 4096 fp32 = 4.5 MiB each) on the previous stage.
5. **Overlap.** Single-stream decode keeps one card computing at a time; the other three use that time to copy
   their predicted RAM experts over their own links (4 independent PCIe links instead of one).
6. **Bit-exactness check first.** Like Strata's `--split-device 0`: all four stages on one card must give
   tokens identical to the unsplit engine before any speed work.

## Tasks (owner, routing)

| # | task | owner | routing | acceptance |
|---|---|---|---|---|
| 1 | Linux build + run on one 4080 with the real checkpoint | infra-ops | Terra | `strata-glm --tokens` runs; teacher-forced KL vs author's BF16 reference within their table |
| 2 | Stage object: layers [lo, hi) on one device; everything else unchanged | engine-integrator | Sol | 4 stages on GPU0 = unsplit engine, tokens identical — **written: patches/strata-glm/0002, compiles; NOT RUN** |
| 3 | Real split on 4 GPUs with P2P hand-off | engine-integrator | Sol | tokens identical to task 2 — **in 0002 (`--gpus`, cudaMemcpyPeer); NOT RUN** |
| 4 | Per-stage tiers + router replication for prefetch | engine-integrator | Sol | VRAM hit rate logged per stage |
| 5 | Measurements: decode/prefill tok/s, hit rate, PCIe use per card | perf-bench | Terra | numbers with commands in docs/evidence |

Open questions (UNKNOWN until the server): PCIe generation and width per card on the H12D-8D; real hit rate
with 65-69% of experts resident; whether the per-card 736 GB/s or the PCIe links bound decode.
