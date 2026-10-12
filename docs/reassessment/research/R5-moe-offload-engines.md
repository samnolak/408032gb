# R5. Hierarchical / MoE-offload engines: code-level landscape and the techniques worth porting (2026-10-10)

Role: engine landscape researcher (hierarchical / MoE-offload engines). Only this file and `matrix/R5.json` were written; nothing
was committed. All clones are in the session scratchpad (`.../scratchpad/research/r5`), SHAs below come from `git log -1` /
`git ls-remote` run on 2026-10-10.

**Labels.** VERIFIED = I read the code/doc at the recorded SHA (`repo@sha:path:line` or function) or ran the command.
REPORTED = stated by a README/paper/issue/author; hardware/quant/context given where known. HYPOTHESIS = my inference (input named).
UNKNOWN. DISPROVED (with source). NOT FOUND (with the searches tried). Mapping to AGENTS.md: VERIFIED = CONFIRMED,
REPORTED/HYPOTHESIS = PROVISIONAL.

**Target (from BRIEF, unverified):** 4x RTX 4080 Super 32 GB (sm_89, P2P not proven), EPYC Zen 3 AVX2, 128 GB DDR4-3200
(4 or 8 channels UNKNOWN), 4 NVMe (*correction 2026-10-12: customer plans 3 x 1 TB NVMe; `docs/evidence/2026-10-12-n26b-nvme-3x1tb.md`*). Models: GLM-5.3-Flash UD-Q4_K_XL (~186 GiB; experts 177,528 MiB, dense 8,550 MiB per
`PROBLEMS.md` of this repo, PROVISIONAL) and Qwen3.8-Flash-Next UD-Q4_K_XL (111.3 GB, fits 128 GB VRAM).

---

## 0. Ten findings

1. **A GLM-5.3-Flash engine for up to 16 NVIDIA GPUs already exists and R2 did not know it: `mw00/project-maya` ("Maya")**
   (MIT, Strata-derived, created 2026-10-06, 215 commits, 289 stars). VRAM / RAM / NVMe expert tiers, CPU lane on AVX2,
   per-route min-max split CPU vs PCIe, layer split over N GPUs with one shared CPU pool, **exact pipelined speculative
   decode across a head and a tail GPU group** (MTP block), KV-latent INT8 option. Its GPU expert kernels accept Q4_K/Q5_K/Q6_K
   (VERIFIED, `glm_fast.cu:925-933`), CPU lane uses ggml-cpu `vec_dot` for any ggml type (VERIFIED, `native_expert.cpp:20-62`).
   Shipped quants are 2-3.5 bit; UD-Q4_K_XL end-to-end is UNKNOWN ("other quants are experimental: they run", README).
2. **Strata upstream moved again: `fb58e0d` → `61b3fb5` (+235 commits, v0.1.42, 2026-10-10). Still Qwen3.8-Flash-Next only**
   (`gguf_reader.hpp:591` guard, VERIFIED). **Its CUDA serve now defaults to a lossy routing change:
   `STRATA_ROUTE_TAIL_SKIP=7`** — a missed expert ranked ≥7 by every token of a verify window is neither copied nor computed,
   no renormalisation (VERIFIED, `expert_source.cpp:3592-3612`, `generate.cpp:8068-8085`). Anyone benchmarking Strata 0.1.42 for
   us must set `STRATA_ROUTE_TAIL_SKIP=0` (exact routing requirement).
3. **The core decode algorithm is the same in four independent codebases (Strata, Maya, glm-flash-lite, FreeToken):** compute
   VRAM hits on the GPU, compute RAM misses on the CPU **in place**, send a share f of the misses over PCIe, with
   f = L/(L + c·R) chosen so both lanes finish together (Strata c = 1.85 fitted on 3 boxes; FreeToken c = 1 from
   overlapped bandwidth probes; Maya and glm-flash-lite solve a min-max per route). VERIFIED in all four (§3.1). llama.cpp
   mainline still has **no CPU lane in `--moe-cache-mib`** (D1 H3; no new commit since `c811cb8`, VERIFIED log to `abee0c8`).
4. **The 4-GPU pipeline bubble has two working remedies in code, neither in llama.cpp:** (a) Maya's head/tail speculative
   pipelining (REPORTED 4x T4: 15.0 → 22.6 tok/s; exact output by construction, VERIFIED `glm_fast_path.cu:3753-3760`);
   (b) Strata's helper-GPU expert caches (`--expert-cache-device1..3` + `--remote-expert-opt`): GPU0 runs dense, GPUs 1-3 hold
   disjoint expert sets and return one weighted partial sum per token through pinned host memory, no P2P (REPORTED 2x4090
   +63% mixed / +132% code; 3-4 helpers "not validated").
5. **GLM-5.3-Flash routing facts from real traces (lt-Strata, Q2/Q3 packs):** adjacent-token reuse 23% (8x chance); distinct
   experts in a window of 2/3/4/8/16 tokens = 1.78/2.46/3.10/5.3/8.7x one token; ideal per-layer LRU hit 41% at 10% coverage,
   58% at 20%, 76% at 40%, 84% at 60% (incl. cold misses; ceiling 88.5% at 100%) (VERIFIED as data:
   `lt-strata@e4dbb18:tools/sim/data/lru_curve.json`, `docs/GLM_DECODE_RESEARCH.md:61-68`). This is direct input for
   `tools/g0/cachesim.py` and ROOFLINE (hit ≈ 0.9-0.95 at our ~60% coverage — HYPOTHESIS).
6. **Next-layer-router prefetch is mostly a net loss in decode when PCIe already carries a miss share:** Strata Foresight slots
   0.91-1.00x on 4 boxes (REPORTED, opt-in only); Maya `PREFETCH_N=1` 19.64 → 17.0 tok/s, +3% only after tuning rank ≤2
   (REPORTED). It pays for NVMe tiers (glm-flash-lite +13%, Strata SSD lane +14%) — i.e. for us only for DeepSeek-V4.1 Engram /
   NVMe spill, not for GLM UD-Q4_K_XL whose experts fit VRAM+RAM.
7. **New measured GLM-5.3-Flash numbers on single consumer GPUs (all REPORTED, all < 4 bit):** PolyStrata 3.0-bit on 1x5090 +
   Xeon 8470Q (DDR5, cloud, 25 vCPU) 40-42 tok/s plain, 47-65 with MTP + lookup, vs llama.cpp 26 on the same box; Strata-DS4
   (DeepSeek-V4-Flash Q2) on 4090 + Ryzen 5700X **DDR4-3200, AVX2**: 23.3-24.0 vs llama.cpp `--moe-expert-cache` 16.3 on the same
   box; Maya-L (~3.5 bpw, 156 GB) on 9 GPUs (8x5060Ti + 3090) 30.7 tok/s at 95.8% VRAM hits. No GLM ≥4-bit number on a
   4x 24-32 GB box exists in any of these repos (NOT FOUND).
8. **lt-Strata glm53f (base B) added a calibrated GLM CPU+GPU simulator (`tools/sim`) and a literature survey;** its own
   conclusion for one 16 GB GPU: lossless ≤ 26 tok/s; 50 tok/s only with a stack of lossy levers (affinity routing, expert
   skip, tapering). Still ≤ 2 GPUs (`glm_decode.cpp:8957`, VERIFIED) and still no Q4_K expert execution.
9. **Lossy levers are now default or prominent in several engines** (Strata tail skip; KTransformers expert deferral;
   lt-Strata affinity/REAP; Maya `MTP_KEEP` affects drafts only and is lossless). CPU-vs-GPU expert placement itself changes
   the last bits (Strata: KL ≈ 0.0006-0.0015 run-to-run; Maya/PolyStrata say the same) — exact *routing* is achievable,
   bit-exact *outputs* across runs are not, once the split is timing-driven.
10. **KTransformers remains unusable for GLM on this CPU as documented** (FP8, AVX-512, ≥350 GB RAM; VERIFIED tutorial at
    `5d048e5`); its AVX2 path exists for BF16/FP8/GPTQ_INT4/RAWINT4 (`kt-kernel/operators/avx2/*`, VERIFIED) but no GLM-5.3 INT4
    checkpoint path is documented (UNKNOWN). A vLLM fork runs GLM-5.3-Flash FP8 on sm_89 (8x4090 48 GB, TP8, all in VRAM) —
    this **disproves R2's HYPOTHESIS "GLM-5.3-Flash will likely not start on sm_89 in vLLM"** for that fork, but it does not fit
    128 GB (§2.6).

---

## 1. Strata upstream (Niko1221/Strata) at `61b3fb5` (2026-10-10) vs `fb58e0d` (2026-10-08)

`git rev-list --count fb58e0d..61b3fb5` = 235 (VERIFIED). Tag v0.1.42 (`cdbf458`). Model support unchanged: `docs/MODELS.md:3`
"Strata runs one model, Qwen3.8-Flash-Next"; `include/strata/artifact/gguf_reader.hpp:591` rejects any other architecture
(VERIFIED).

### 1.1 Subsystems (code read)

| subsystem | where | what it does (VERIFIED unless marked) |
|---|---|---|
| decode window / doorbell | `src/core/verify.cpp` `Verifier::service` (3056), `publish_plan` (2261), `fetch_dma` (2233); `src/kernels/cuda/elementwise.cu:186-432` | a captured CUDA graph per window; per MoE layer the GPU publishes `(x, ids, weights)` to mapped pinned memory and spins on a host flag; the host thread runs the pool callback, publishes a GPU plan (flag A), issues the PCIe DMA share on a copy stream (flag B via `cudaLaunchHostFunc`), then raises the layer flag |
| miss planner (CPU / PCIe / VRAM / peer / helper) | `src/core/expert_source.cpp:3525` `expert_pool_dispatch_multi` | distinct experts of the window in routing order; resident → GPU group; peer/helper → that GPU; last `pcie_num/256` of the misses → PCIe staging (if pinned); rest → CPU jobs; tail skip (below) |
| PCIe share | `src/program/generate.cpp:1611-1630` `pcie_frac_for_gbps`, `struct PcieAuto` (`kPoolFactor = 1.85`) | 0.1.42: share = L/(L + 1.85·R), R = expert bytes / measured CPU-pool time per miss, refined 3 times in the first ~10 s, single GPU only |
| route tail skip (**lossy, default ON on CUDA serve**) | `expert_source.cpp:3592-3612`, `generate.cpp:8068-8085` | missed expert whose minimum rank in the window is ≥ 7 → contribution zero, no renormalisation. REPORTED KL vs off 0.003-0.006, top-1 97.5-98.2%, decode ×1.13-1.16 on single 12-16 GB cards, ×1.0 when hit rate 99.7% (`docs/DETAILS.md:1560-1611`) |
| router look-ahead | `include/strata/core/expert_source.hpp:235` `RouterLookahead`, `expert_source.cpp:2241` `run` | thread applies layer l+1's BF16 router to layer l's MoE input, warms pages (file tier) or feeds Foresight slots; `STRATA_LOOKAHEAD_STATS` scores recall (new, 2026-10-09; **no published numbers**) |
| adaptive VRAM tier | `src/core/expert_cache.cpp`; `tools/replay_cache_policy.py` (new) | decayed-frequency admission, swaps on a job thread; offline replay of `--dump-routing` traces through the policy (new) |
| CPU kernels | `src/kernels/cpu/iq_avx2.cpp`, `kq_avx2.cpp`, `q2_avx2.cpp`, `native_expert.cpp`, `pool.cpp` | AVX2 i-quant/K-quant rows, AVX-VNNI forms, AVX1 fallbacks (new); pool pins one worker per physical core |
| multi-GPU | `docs/MULTI_GPU.md`; `generate.cpp` `SplitDrive` (`kMax = 8`), `drive_pool_split` | layer split (auto placement over all layouts for ≤ 4 cards), each stage its own cache, one shared CPU pool; helper caches CUDA1-3 (`docs/SECOND_GPU.md`); peer device; data-parallel replicas (`8d276c1`, opt-in) |
| 262K / KV | `docs/DETAILS.md:1784-1800`, `docs/KV_PREFETCH.md` | INT8 (and K8V4) KV; from 64K only the most-read KV part stays in VRAM, the rest streams from RAM; opt-in `STRATA_KV_PREFETCH` overlaps the upload with prefill |
| n-gram / SSD tier | `src/ngram/ple_reader.cpp`; `--ple-io direct` | the 28.8 GB PLE table read with O_DIRECT, a few rows per token (Qwen-specific) |
| MTP / lookup | `src/core/mtp.cpp`, `src/spec/*` | own MTP layer drafts ≤ 3, prompt-lookup drafter ≤ 5, cost-gated |

### 1.2 What is new since `fb58e0d` and matters to us

- **Tail skip default ON (0.1.42)** — see above; contradicts "exact routing". `STRATA_ROUTE_TAIL_SKIP=0` restores 0.1.41 bytes
  (REPORTED, `DETAILS.md:1569-1571`).
- **PCIe share from measured CPU speed** — best shares measured 0.15-0.20 (RTX 3060), 0.20 (P100) vs old default 0.55; +30% decode on
  the 3060 (REPORTED, `DETAILS.md:1612-1636`). Lesson for us: **the CPU lane should take most misses; PCIe is a minority lane in decode.**
- **Prefill CPU share default ON for chunks < 1024 tokens (single GPU)**: −20…−34% prompt time on 512-1000-token chunks (REPORTED); changes
  bits, `auto` is not repeatable run to run (#1684).
- `--aux-cpus`: moving non-pool threads off pool cores gave +10% decode on a 22-core Xeon (REPORTED, `DETAILS.md:1670-1712`).
- Foresight swap slots (next-layer prediction into VRAM): **net loss** 0.91-1.00x on RTX 3060 / P100 / 5070 (REPORTED, `DETAILS.md:1650-1660`).

### 1.3 Benchmarks relevant to our hardware class (`bench/results/` at `61b3fb5`, all REPORTED)

| report | hardware | model / quant | numbers |
|---|---|---|---|
| `2026-10-04-community-rtx-4080s-iq3s` | **1x RTX 4080 SUPER 32 GB**, i7-13790F (AVX2), 96 GB, Gen4 x16 (24.0 GB/s probe) | Qwen3.8-FN IQ3_S, 524K ctx | 4 pool workers 28.7% faster than 15 (hybrid Intel CPU); 524K costs +0.9…6.3% at a 44K prompt; decode 91.3 tok/s (R2 row 16) |
| `2026-09-29-rtx3090-epyc-milan` | 2x3090, **EPYC 7453 (Zen 3, AVX2)** KVM guest, 165 GiB | Qwen3.8-FN IQ3_XXS / Coder IQ1_M | 262K cells measured; Coder 2-GPU 98.6 tok/s |
| `2026-10-02-community-rtx4000ada-layer-split` | EPYC 7313P (Zen 3) | layer split | (not read in detail) |
| `2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool` | 2x3080 20 GB (Gen3 x16), Xeon E5-2696 v4 (AVX2), 128 GB DDR4-2400 4ch | **Qwen3.8-FN UD-Q4_K_XL**, 2x262K lanes | decode 81-91 tok/s at 4K-127K prompts, prefill 2,996 tok/s at 104K; **fork build** (noon-at-cgn `repo/w7` `428d991`), not upstream |
| `docs/UNSLOTH_Q4.md:31-35` | 3090 + 3080 Ti (x4), 91 GB | Qwen3.8-FN UD-Q4_K_XL | 45 tok/s split vs 38 one card |

**Implication (HYPOTHESIS):** Qwen3.8-Flash-Next UD-Q4_K_XL on 4x32 GB fits VRAM entirely, so offload techniques are irrelevant for it;
the 80-90 tok/s on 2x20 GB + 4-ch DDR4 AVX2 (with ~half the experts on CPU) is a lower bound for a 4-card layer split.

---

## 2. The Strata family and other GLM-capable offload engines

### 2.1 Forks and derivatives (SHA, date, status)

| project | SHA (date) | status / what it is | relevant for us |
|---|---|---|---|
| **mw00/project-maya** | `10b7b07` (2026-10-10) | NEW. GLM-5.3-Flash engine, Strata-derived, MIT; v1.0.31; ≤16 NVIDIA GPUs, AMD experimental | **highest** (§2.2) |
| arpahmad/n1oss | `ce5e8e6` (2026-10-08) | NEW, 3 commits; Maya repackaged "for 3090 clusters x4/x8/x16"; same numbers as Maya | none beyond Maya |
| VecSzn/PolyStrata | `27f9ec9` (2026-10-10) | NEW. Second engine `strata-poly` for GLM (single GPU), MTP draft + lookup, adaptive swaps | data point (§0.7) |
| Indras-Mirror/Strata-DS4 | `924516a` (2026-10-10) | NEW. DeepSeek-V4-Flash on Strata: predicted prefetch (~62-70% recall), exclusive VRAM LRU, PCIe share, CPU pool | technique evidence (§3.4, §3.5) |
| Indras-Mirror/Strata-GLM | `152c49f` (2026-10-10) | NEW, 0 stars; README at that SHA mostly repeats the DS4 text (UNKNOWN whether GLM numbers exist) | low |
| OpenMOSE/Strata-GLM-AMD | `6c71ff0` (2026-10-10) | NEW. GLM on 2x W7900 (ROCm), 3-bit GSQ-RCO, Q8_0 caches, 30-34 tok/s with draft (REPORTED) | AMD-only |
| daraskme/Strata-GLM | `9e0b779` (2026-10-08) | NEW. Maya-based, single GPU; adds Q4_K/Q5_K/Q6_K expert decode checks; RTX PRO 6000 96 GB + i9-12900KS + **DDR4-3200 128 GiB**, model "OrcaRouter Uncensored Q4_K_M" (a GLM-5.3-Flash derivative per the README; identity not verified), ctx 262,144: **15.9 tok/s after an 8,180-token prompt, 17.1 after 260,090** (REPORTED; author: "50 tokens/s not reached") | evidence that a Maya-line engine ran a 4-bit K-quant GLM at 260K on DDR4 |
| lighttransport/Strata glm53f (base B) | `e4dbb18` (2026-10-10) | +23 non-merge commits by Syoyo Fujita since `27deb91` (VERIFIED `git log --no-merges --author`): `tools/sim`, B550 work, exact CPU cache prototype, regression tests; merged upstream 0.1.41 | simulator + trace data (§0.5, §0.8); engine still ≤ 2 GPUs, no Q4_K |
| sergqwer/strata-glm (base A) | `ed37419` (2026-10-01) | unchanged since pin | — |
| Grigory-Rylov/strata-glm-3090 | `794cd69` (2026-10-03) | unchanged since R2 | — |
| 0xSero/glm53-flash-offload = sybil-solutions/glm-flash-lite | `6769b27` (2026-10-09) | renamed; ExLlamaV3 + own C++/CUDA host engine (`kernels/nv2`), EXL3 3.05 bpw, single GPU | technique evidence (§3.1, §3.4) |
| noon-at-cgn/Strata | `82f46a8` (HEAD) | fork with shared KV pool, batch-MTP on split (PRs #1614/#1636/#1637 upstream) | Qwen only |

### 2.2 Project Maya in depth (VERIFIED at `10b7b07` unless marked)

- **Model path.** `src/core/glm_model.cu` (2,796 lines), `glm_fast_path.cu` (4,465), `glm_prefill.cu` (2,867); KDA, NoPE-MLA + DSA, mHC (4 streams
  pinned: `glm_model.cu:470`), MTP/NextN, vision. Reads `glm5next` and `glm5-next` GGUFs (README; `tools/gguf_fix_arch.py`).
- **Expert formats.** GPU routed-expert kernels: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS
  (`glm_fast.cu:2900-2944`); Q4_K/Q5_K/Q6_K use a slower per-row dot ("no shared-activation form yet", `glm_fast.cu:926`). CPU lane: any type
  ggml-cpu has a `vec_dot` for (`src/kernels/cpu/native_expert.cpp:31-62`), AVX2/AVX-512 i-quant fast paths for ≥ 2 tokens.
- **Tiers.** VRAM cache (usage-file seeded, background promotion `STRATA_GLM_PROMOTE`, frequency gate `PROMOTE_MIN`), pinned RAM tier sized from
  free RAM (`STRATA_GLM_RAM_RESIDENT=1` = every non-VRAM expert in RAM, no disk), NVMe reads with O_DIRECT (README table; `STRATA_GLM_DROP_CACHE`).
- **Lane split.** `Glm5Model::fast_cpu_lane_setup` (`glm_fast_path.cu:1473`): times one expert on the CPU pool (c) and one over PCIe (p) at start,
  then for each count f = 0..8 of RAM-tier experts in a route picks k = argmin max(p·(f−k), 0.04 ms + c·k); the coldest experts go to the CPU.
  Calibration tool `--calibrate` measures real output speed and stores `STRATA_GLM_PCIE_SHARE` / `CPU_LANE` per machine.
- **Multi-GPU.** Layer split; `auto` prices every placement for 2-4 GPUs (README `STRATA_GLM_SPLIT`); 3+ GPUs share one CPU pool (9-GPU box:
  25.7 → 30.7 tok/s, CHANGELOG:179-183, REPORTED). Activations hop through pinned host memory (`Glm5Model::hop_to_next`, no P2P needed).
- **Pipelined speculative decode** (`glm_fast_path.cu:3753-3830`, `spec_ready`, `spec_head_last`, `spec_kda_copy`, `spec_head`): see §3.2.
- **Long context.** DSA indexer keys kept in a ring of `8192+64` positions on the fast path with pooled keys for the whole context
  (`glm_model.cu:2463`, `dsa_pool_`); CHANGELOG claims "the same output" (REPORTED; HYPOTHESIS: correct because the indexer scores
  pooled blocks, but must be checked against llama.cpp `glm5-next` dumps at > 8K). KV-latent INT8 opt-in (544 B/token/layer).
- **Measured (REPORTED):** 2x V100 32 GB PCIe3 + 30 GB RAM, Maya-S (2-bit experts): up to 40 tok/s; 1x V100: 19; 2x CMP 170HX all-in-VRAM
  Maya-M: 61.3; 4090 + 192 GB DDR5 Maya-L: 16.4; 9 GPUs Maya-L: 30.7. **No measurement at ≥ 4 bit, none on 4080, none on EPYC DDR4.**
- **Limits.** Quants offered are 2-3.5 bit (Maya-S/S24/M/L); UD-Q4_K_XL is "experimental"; no tensor parallelism; helper-GPU expert path of
  Strata is not wired into the GLM fast path (no `remote` in `glm_fast_path.cu`, VERIFIED grep); a single conversation engine with SSD slots.

### 2.3 glm-flash-lite (ex 0xSero/glm53-flash-offload) — host engine (VERIFIED `6769b27`)

- `kernels/nv2/nv2_host.cpp:1-12` describes the design: one controller thread polls per-layer requests from the GPU (mapped ring), owns all
  residency decisions, `plan_and_reply` (line 422) assigns each routed expert a lane: `VRAM`, `RAM` zero-copy (GPU reads pinned host memory
  directly), `CPU` (AVX2 kernel computes in place), `NVC` (NVMe→CPU), `NVG` (NVMe→GPU), greedy **min-max** with costs `g0, thit, tzc, ca, cb, ctok`.
- Exclusive tiers: an expert evicted from VRAM moves to RAM; VRAM and RAM never hold the same expert ("~6,100 different experts instead of
  ~4,800" at 55 GiB, README).
- Measured time split (REPORTED, v4.6, C1, 55.8 ms/token): admissions copy 31%, non-MoE GPU 20%, MoE kernel 18%, waiting for CPU lane 18%,
  NVMe 10%; "every lane at its hardware floor about 41 tok/s".
- Hardware: 1x3090, **EPYC 7443P (Zen 3) 8-channel DDR4**, 4x NVMe RAID0; 28.2 tok/s all-RAM, 19.5 at 55 GiB + NVMe, EXL3 3.05 bpw.
  The CPU lane is "near-exact" (decode KL 0.005); bit-exact mode `nvme-exact` 8.3 tok/s.

### 2.4 FreeToken (FlashML-org) and offshoots

- HEAD `ee9789f` (2026-10-10); still one GPU (`python/freetoken/server/args.py:834` "tensor parallelism is not supported yet", VERIFIED).
- `moe/offload_kernels.py:43` `ensure_experts_hybrid`: per step fetch `round(fraction·misses)` over PCIe into the LRU slots, rest → CPU (`-1` ids),
  Q16 fixed point, CUDA-graph safe. `moe/bench_profile.py:156` `load_hybrid_fetch_fraction`: fraction = pcie_ov / (pcie_ov + cpu_ov) from
  `ft bench bw` overlapped probes (VERIFIED).
- `moe/cpu_executor.py:31-51`: GPU↔CPU handshake with **`cuStreamWriteValue64` / `cuStreamWaitValue64` stream memops** (no spin kernel, no
  host-func per layer: "~30-50 µs callback latency … ~6 ms/step on a 75-layer model" avoided) (VERIFIED as code comment + flag).
- CPU formats: bf16, nvfp4, mxfp4, ds_fp4, q4_0 (`cpu_executor.py` `_WFMT_IDS`); **no Q4_K, GGUF only for gemma4** (`models/gguf/config.py:19-21`).
- freetoken-ox-boost `fdb99c6` (2026-10-02): GLM-5.3-Flash NVFP4 on RTX PRO 6000, ~38 tok/s, misses ~31/token (REPORTED); found and fixed a KDA
  prefix-cache snapshot bug (missing ×64-boundary state writer → ~10% corrupted cache hits) — a correctness warning for any KDA prefix cache.
- New offshoots (code exists, not read in depth): `surajbhan/freeToken-rs` (Rust port, q4_0), `Waltarb/nvmoe` (`344c5ec`, 3-tier with NVMe, no license).

### 2.5 KTransformers (`5d048e5`, 2026-10-10, Apache-2.0)

- AVX2 MoE operators exist: `kt-kernel/operators/avx2/{bf16,fp8,gptq_int4,gptq_int4_avxvnni,rawint4,mxfp4,mxfp8}-moe.hpp` and the LLAMAFILE
  backend (`operators/llamafile/moe.hpp`, ggml quant types incl. K-quants via llamafile sgemm) (VERIFIED listing). `kt-kernel/README.md:189-193`
  still lists LLAMAFILE as the AVX2 backend and RAWINT4/FP8/BF16 as AVX-512 (VERIFIED).
- GLM-5.3-Flash tutorial: FP8 weights 306 GiB, ≥350 GB RAM, "AVX-512 FP8 CPU expert kernel", SM89/SM120, TP4 via SGLang (VERIFIED,
  `doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md:9-19`). Not runnable on our server as documented.
- Expert deferral is in code: `kt-kernel/python/experts_base.py:492` `select_deferred_experts` keeps the top `protected_k` by score immediate,
  defers the rest one layer (lossy) (VERIFIED).
- Placement strategies `frequency`/`dynamic-expert-update`: 4x4090 TP4 Qwen3-Next-80B FP8, 50% GPU experts 76.2 / 81.2 tok/s vs 0% 52.7
  (REPORTED, `experts-sched-Tutorial.md:178-190`; ShareGPT load, not batch 1).

### 2.6 Adjacent: yhfgyyf/vllm-deepseek-v4-sm89 (`3f3bbae`, 2026-09-16)

vLLM fork for sm_89/sm_120: GLM-5.3-Flash FP8 on **8x RTX 4090 48 GB**, TP8, `NCCL_P2P_DISABLE=1`, MTP 5, `--max-model-len 262144`
(README §6.1, community issue #74, REPORTED); DeepSeek-V4.1-Flash with **Engram CPU offload** (`--engram-config '{"cpu_offload":true}'`).
All weights in VRAM — not an expert-offload engine and 306 GiB does not fit 128 GB. Relevant to R2's sm_89 blocker only.

### 2.7 Other engines (brief, VERIFIED HEAD; capability text REPORTED unless noted)

| project | SHA (date) | verdict for us |
|---|---|---|
| EfficientMoE/MoE-Infinity | `0f96d56` (2026-10-08), Apache-2.0 | unchanged since R2; GLM via transformers eager attention; default build sm_80/90 |
| Neroued/ninfer, Wallawalla47/Infernix | `81c8ce0` (2026-10-07), `a3edb45` (2026-10-09), Apache-2.0 | unchanged since R2: **sm_120a only** (R2 VERIFIED `CMakeLists.txt:3-12`) |
| Tiiny-AI/PowerInfer | `8bd56d6` (2026-05-11), MIT | stale; no target models |
| efeslab/fiddler | `227715b` (2024-04-28), Apache-2.0 | abandoned (Mixtral prototype) |
| JustVugg/colibri | `bf24429` (2026-10-06), Apache-2.0, ~41k stars | pure C disk-streaming engine for GLM-5.3-Flash, MiMo-V2.6, DS-V4.1 etc.; GLM-5.3-Flash "about 20 s per token warm" on a laptop; 6x5090 GLM-5.2 9 tok/s (REPORTED). Not a speed candidate |
| danveloper/flash-moe | `3601d41` (2026-03-19), no LICENSE file | C/Metal, Qwen3.5-397B from SSD on Mac (REPORTED 4.4 tok/s); Apple-only |
| ek15072809/Swap-MoE | `50459bb` (2026-08-29), MIT | llama.cpp patch: expert streaming from SSD via mmap (`src/llama-expert-stream.cpp`) |
| yalun753/moe-l2 | `3b17ede` (2026-10-01), Apache-2.0 | llama.cpp-based LRU expert cache for 8 GB cards |
| davetha/vllm-expert-cache | `813b97f` (2026-09-08), Apache-2.0 | vLLM plugin, device-side LRU/LFU expert cache, PCIe streaming only |
| ongunm/llama-moe-cache (FATE) | `77c8767` (2026-04-07), **AGPL-3.0** | cross-layer gate prefetch in llama.cpp; license blocks code reuse |
| ggml-org/llama.cpp | `abee0c8` (2026-10-10) | no MoE-cache change after `c811cb8` (VERIFIED `git log --since=2026-10-05`) |
| ikawrakow/ik_llama.cpp | `89bba35` (2026-10-10) | no expert-cache commit since 2026-09-20 (VERIFIED log grep); CPU affinity flags added (`afdbc5e`, `c75ad6b`) |

### 2.8 Research systems from the task list

| system | code? | SHA / searches | note |
|---|---|---|---|
| Speculating Experts (arXiv 2603.19289) | yes: axonn-ai/yalis branch `offload_prefetch` | `b1970f7` (2026-02-28), Apache-2.0; `yalis/offloading/manager.py` (`prefetch_layer`, `fetch_layer`, `PrefetchMode`) | quasi-hidden-state next-layer prediction, −14% TPOT vs on-demand (REPORTED); PyTorch, PCIe-only |
| MoE-SpAc (2603.09983) | yes: lshAlgorithm/MoE-SpAc | `77bb9cf` (2026-03-23), MIT (llama.cpp fork; +690 lines in `ggml-backend.cpp`) | SD-as-lookahead utility, online integer split CPU/GPU; +42% vs SD baseline (REPORTED) |
| AdapMoE (ICCAD'24) | yes: PKU-SEC-Lab/AdapMoE | `2cfe8df` (2025-04-28), no LICENSE | adaptive gating = lossy; stale |
| DAOP (DATE'25) | yes: ecolab-nus/DAOP | `08fef33` (2025-06-22), Apache-2.0 | approximate pre-calculation = lossy; stale |
| FlashMoE (2601.17063) | NOT FOUND | WebSearch "FlashMoE arXiv 2601.17063 SSD expert offloading" → paper only | ML cache policy for SSD tier, +51% hit vs LRU/LFU (REPORTED) |
| Flash-MoE (C/Metal) | yes | see §2.7 | Apple-only |
| ZipMoE (2601.21198) | NOT FOUND | WebSearch "ZipMoE expert offloading compression github" → paper only | |
| HOBBIT, ProMoE, Klotski | NOT FOUND | GitHub repo search "HOBBIT mixed precision expert offloading", "ProMoE prefetch moe in:name,description", "Klotski moe inference" → 0 hits | HOBBIT lossy; Klotski throughput-oriented |
| MoE-Lightning | NOT FOUND | repo search "MoE-Lightning" → unrelated | throughput-oriented |
| ExpertFlow (2410.17954) | NOT FOUND (paper code) | repo search "ExpertFlow moe" → unrelated Apple/llama.cpp forks | |
| SpecPrefetch (2607.24787), SSD-LLaMA | NOT FOUND | repo search "DAOP moe offloading OR SpecPrefetch OR SSD-LLaMA" → DAOP only | |
| Swap-MoE | yes | §2.7 | |

---

## 3. The most promising techniques for our server — algorithm, limits, port, cost, offline tests

Framing (HYPOTHESIS, from `docs/reassessment/ROOFLINE.md` rev. 3 and the numbers above): GLM-5.3-Flash UD-Q4_K_XL with ~105-110 GiB of
VRAM for experts gives coverage ≈ 0.6 and hit ≈ 0.9-0.95 (lt-Strata curve, §0.5); the remaining ~65-70 GiB of experts fit in 128 GB RAM if
tiers are exclusive, so **decode needs no NVMe**. Misses ≈ 17-34 per token × 14.7 MiB ≈ 0.25-0.5 GB/token from DRAM — 2-7 ms on 4-8
channels, i.e. smaller than the GPU side. The GPU side (dense ~8.4 GiB + hits ~4.5 GiB per token, read serially by one card at a time in a
layer split) is ≈ 20-25 ms → ~40-45 tok/s. **So on this server the binding constraint for GLM is the layer-split bubble and the dense part,
not the CPU tier.** Techniques are ranked by how much they attack that.

### T1. Pipelined speculative decode across GPU groups (Maya) — attacks the 3-of-4-idle bubble, exact

Call sequence (VERIFIED, `glm_fast_path.cu:3753-3830`; head = first `n/2` parts, `STRATA_GLM_SPEC_HEAD` overrides):
```
round(p, token t_p, draft d_{p+1} from NextN):
  head.spec_kda_copy(save)                 # KDA S + conv states of head layers -> kda_bak_
  tail.layers(p, t_p)            ||  head.layers(p+1, d_{p+1})      # both GPU groups busy
  t_{p+1} = sample(tail.head_logits)       # tail, NextN, sampler see confirmed tokens only
  if t_{p+1} == d_{p+1}: keep head's work (token cost = slower half)
  else: head.spec_kda_copy(restore); head.layers(p+1, t_{p+1})      # redo, exact
  hop_to_next: residual rows -> pinned host -> next part (no P2P)
```
- Why it fits us: 4 cards → 2+2 groups; each group holds half the layers' experts; a token costs ≈ max(head, tail) instead of head+tail when
  drafts are accepted. REPORTED 4x T4: 15.0 → 22.6 tok/s (×1.5); 2-GPU RTX 3090 + 5060 Ti: +3-5% (asymmetric cards). Drafts with missing
  experts compute only the top-`MTP_KEEP` (=2) missing ones — draft quality only, output unaffected (README).
- Limits: gain ≤ 2x by construction and only at the MTP acceptance rate (GLM MTP-1 acceptance UNKNOWN on our text; R3/ADR-011 data);
  MLA/DSA caches of the speculative position must also be rolled back or overwritten (Maya re-runs the position; VERIFIED only for KDA copy);
  CPU lane is shared, so both groups' misses contend for DRAM.
- Port: (a) run Maya as-is on the server (cheapest: it is MIT, multi-GPU, GLM); (b) in llama.cpp this needs two concurrent `ggml` graph
  executions on disjoint device sets with state snapshot of recurrent layers — no such facility today (HYPOTHESIS: weeks–months, conflicts
  with the scheduler). Cost: (a) days; (b) ≥ 1-2 months.
- Offline tests: exactness is testable on CPU-only/any 2 CUDA devices with a tiny GLM-shaped model: token stream with `STRATA_GLM_NO_SPEC=1`
  vs default must be identical for greedy decode (Maya's own claim); on the 3x3090 `epyc-ai` host with the real GGUF (it reads `glm5next`).

### T2. Min-max lane planner with CPU-in-place miss compute (Strata / Maya / glm-flash-lite / FreeToken) — required for GLM on any engine

Algorithm (VERIFIED in four codebases; Strata variant, `expert_source.cpp:3525-3700`):
```
per MoE layer l, window of T tokens (k routes each):
  D = distinct experts of the window, in routing order
  for e in D: kind[e] = VRAM if resident[l][e] else PEER/HELPER if held else MISS
  (lossy, Strata default) drop MISS e whose min rank over the window >= 7     # we must disable this
  m = round(f * |MISS|)          # f = L/(L + 1.85 R) (Strata) | pcie/(pcie+cpu) (FreeToken) | argmin_k max(p(f-k), 0.04+c k) (Maya)
  last m misses -> PCIe staging (pinned blob -> cudaMemcpyAsync on copy stream -> flag B)
  publish GPU plan (flag A): VRAM groups first, PCIe groups after flag B
  CPU pool: quantize activations once per token, one job per distinct missed expert, rows written to h_ymiss
  raise layer flag; GPU adds CPU rows + its own rows
```
glm-flash-lite adds a zero-copy lane (GPU reads a pinned RAM expert in place, no admission) and NVMe lanes, in one greedy pass sorted by
route count (`nv2_host.cpp:422-470`).
- Limits: GPU↔host round trip per layer (doorbell spin ~10-50 µs; FreeToken's stream memops avoid an SM-resident spin); CPU and GPU round
  differently → last-bit differences, timing-driven split is not repeatable (Strata KL floor 0.0006-0.0015); DMA steals DRAM bandwidth from
  the CPU lane when the CPU is DRAM-bound (Strata paper §6 p.9, R3 §1.8). On our server the CPU lane is under the GPU time (framing above), so
  its value is "misses off the critical path", not raw speed — unless VRAM coverage is lower than assumed (long context KV, DFlash drafter).
- Port into llama.cpp: this is D1 H3. It requires splitting one `MUL_MAT_ID` between the CUDA backend (hits) and the CPU backend (misses)
  in the same graph step, with an async handshake. Minimal design (HYPOTHESIS): keep `--moe-cache-mib` slots; at the cache `copy` hook
  (`src/llama-moe-cache.cpp` ~515-560 @10a60cf30) mark misses `-1` like FreeToken, launch a host task (stream memop) that computes them from
  the host copy with ggml-cpu `vec_dot` (Q4_K works, as Maya proves), and add a `GGML_OP_ADD` of the CPU partial sum. Cost 3-6 weeks + parity
  test (`tools/quality/logprob_compare.py`, top-1 ≥ 0.99). Alternative: use Maya (already has it, GLM-specific).
- Offline tests (no target GPU): CPU expert GEMV rate for Q4_K/Q5_K on a Zen 2/3 box with 1…48 threads (ggml `vec_dot` vs Strata
  `kq_avx2.cpp` vs Maya `native_expert.cpp`) → c per expert; with a measured PCIe probe p (any Gen4 GPU) compute the Maya plan table; planner
  unit test with recorded routes (lt-Strata traces or our `tools/g0` traces).

### T3. Helper-GPU expert parallelism (Strata `--expert-cache-device1..3` + `--remote-expert-opt`) — attacks the bubble for decode without TP

Mechanism (REPORTED `docs/SECOND_GPU.md:1-40, 87-125`; dispatch VERIFIED in `expert_pool_dispatch_multi`: `helper_holds`, `remote[r]->begin`,
kind 2):
```
CUDA0: all dense layers, KV/state, MTP, its own expert cache
CUDA1..3: disjoint expert sets (ranked profile, round-robin or per-layer placement), each computes its rows
per MoE layer: CUDA0 publishes x + ids; helpers compute their experts concurrently with the CPU pool;
               each helper reduces to one weighted partial sum per token on-device; returns via pinned host buffer;
               CUDA0 waits for all helpers, adds CPU rows, continues
```
- Why it fits us: no P2P, no TP; GPUs 1-3 work in parallel on every MoE layer instead of idling in a layer pipeline. Ceiling (HYPOTHESIS): one
  card reads the dense part (~8.4 GiB → ~14 ms at 590 GB/s effective) while helpers read ~1/3 of the hit bytes each → ≈ 55-65 tok/s before
  speculation, minus 45 host round trips per token (≈ 1-2 ms). But CUDA0 must hold dense + 262K KV + MTP, leaving it few expert slots.
- Limits: validated only for 2 cards (REPORTED +63% mixed / +132% code on 2x4090; "three/four cards have not been validated"); a layer waits for
  its slowest helper; summation order changes bits; Strata implements it only for Qwen3.8-FN; Maya's GLM fast path does not have it.
- Port: into Maya (GLM) — reuse Strata's `remote_experts.cpp` (present in Maya's tree, unused by `glm_fast_path.cu`) → 4-8 weeks (HYPOTHESIS);
  into llama.cpp — an "expert-parallel MoE" split mode does not exist; would be a new `-sm` mode (months).
- Offline tests: Qwen3.8-FN on Strata with 2 helpers on any 3-GPU box (the `epyc-ai` 3x3090 host qualifies) measures scaling 1→2 helpers;
  round-trip latency of pinned-host handoff (microbench, any 2 CUDA GPUs).

### T4. Exclusive VRAM/RAM tiers with frequency-gated admission — makes GLM UD-Q4_K_XL fit with no NVMe in decode

Evidence (REPORTED, code VERIFIED where noted): glm-flash-lite "an expert evicted from VRAM moves into RAM; VRAM and RAM never hold the same
expert" (+27% more distinct experts at 55 GiB); Strata-DS4 exclusive VRAM LRU (miss takes the LRU slot of its layer): hit 70 → 84%, decode
18.8 → 22.3 tok/s (DDR4-3200, AVX2, 4090); Maya `PROMOTE_MIN=6` stopped churn on a flat route table ("4090 D with 128 GB: decode 1.9x" with
`RAM_RESIDENT=1`); lt-Strata trace fact "a recency cache with only 16 promotions per token and no frequency gate hits under 10%, which is why
Strata's decayed-frequency admission is right for a slow PCIe link" (`GLM_DECODE_RESEARCH.md:68`).
- For us: 128 GiB VRAM + ~110 GB usable RAM ≥ 173 GiB of experts + 8.4 GiB dense only if RAM holds **just** the non-VRAM experts (exclusive).
  A full RAM copy (llama.cpp `--moe-cache-mib`: experts live in host RAM, GPU slots are copies) needs 173 GiB of RAM → does **not** fit 128 GB
  (HYPOTHESIS from sizes; llama.cpp would then mmap from NVMe for the remainder). **This is a concrete argument against stock llama.cpp
  `--moe-cache-mib` for GLM UD-Q4_K_XL on 128 GB RAM**, and for `-ot`/static placement (VRAM-resident experts never loaded to RAM) or an
  exclusive-tier engine (Maya, glm-flash-lite).
- Port: llama.cpp static split already exclusive (`-ot` regex per expert block is per *layer tensor*, not per expert — HYPOTHESIS: expert-level
  exclusivity needs repacked GGUF tensors or a loader patch); Maya: built in. Cost: llama.cpp per-expert loader patch 2-4 weeks.
- Offline tests: `tools/g0/cachesim.py` with an `exclusive` flag (RAM-hit vs NVMe-miss accounting) on traces; memory arithmetic from GGUF
  headers (`gguf-dump`) of the real UD-Q4_K_XL files on `epyc-ai`.

### T5. Next-layer router lookahead prefetch — conditional, mainly for an NVMe tier

Algorithm (VERIFIED: Strata `RouterLookahead::submit/run`, Maya `STRATA_GLM_PREFETCH_*` in README + `FastState::pred_*`,
glm-flash-lite `k_lookahead.py` + `r.pf_key` in `plan_and_reply`): apply layer l+1's router to layer l's MoE input (cos-sim of adjacent
residuals is high), take top-k, and start copies (VRAM) or reads (NVMe→RAM) of predicted non-resident experts.
- Measured: decode with a PCIe miss lane — Strata Foresight 0.91-1.00x, Maya N=1 −13% (→ +3% with rank ≤ 2 and 4 blocks); "first guesses are
  right almost always, its 5th-8th mostly not" (Maya README); Strata-DS4 recall 62-70%; NVMe tier — glm-flash-lite misses 40 → 22/token, +13%;
  Strata SSD lane +14%; Speculating Experts −14% TPOT (PCIe-only offload, REPORTED). Negative prior result: learned route predictors did not
  beat "repeat previous token" (R3 §1.5).
- For us: only for DeepSeek-V4.1-Flash (Engram + experts spill to NVMe) or if GLM must spill. Port cost small inside Maya/Strata (exists);
  in llama.cpp ~2-3 weeks (prefetch hook beside `--moe-cache-mib`). Offline: recall@k from hidden-state dumps (`tools/g0/predict_eval.py`).

### Not recommended (lossy or wrong regime)

Strata `STRATA_ROUTE_TAIL_SKIP` (default!), `STRATA_ROUTE_RESIDENT`, lt-Strata affinity/REAP-50/expert skip/verifier tapering, KTransformers
expert deferral, HOBBIT low-precision misses, AdapMoE/DAOP approximations — all change routing or math. Throughput systems (MoE-Lightning,
Klotski) — batch-oriented. Block drafters (DFlash2) with CPU-resident experts — lt-Strata's simulator and B550 runs show the verify window's
distinct expert bytes grow almost as fast as accepted tokens (5.3x bytes for 5.4 tokens at block 8, `GLM_SIM.md:201-206`), so they pay only
when hit rate is high — which on our server it should be (HYPOTHESIS; re-check after E1).

---

## 4. What changes for the project (HYPOTHESIS unless marked)

1. **Add Maya as an engine candidate for GLM-5.3-Flash** next to llama.cpp/ik_llama (R2 direction A/B). It is the only engine found with GLM +
   ≥3 NVIDIA GPUs + CPU lane + exact speculative pipelining, MIT, actively maintained (≥ 10 commits/day). Risks: young (4 days), quants it
   tunes are < 4 bit, Q4_K expert kernels are the slow per-row path, no TP, single-conversation server, author is one person + contributors.
2. **Do not use Strata 0.1.42 defaults in any accuracy comparison** (tail skip; prefill CPU share). Pin `STRATA_ROUTE_TAIL_SKIP=0`,
   `STRATA_PREFILL_CPU_SHARE=0` for parity runs.
3. **D1 H3 (CPU lane in llama.cpp) has four reference implementations** to copy from (Strata dispatch, Maya plan, FreeToken memops,
   glm-flash-lite zero-copy lane), but on our server its payoff is lower than in their single-GPU setups because ~90% of experts are in VRAM.
   The pipeline bubble (T1/T3) and dense-part TP are the bigger levers.
4. **Feed lt-Strata's GLM trace statistics into ROOFLINE/cachesim now** (union ratios, LRU curve) instead of the uniform-routing assumption,
   and mark them as Q2/Q3-pack traces of coding fixtures (distribution shift UNKNOWN).

## 5. Experiments runnable without the target server

| # | experiment | where | decides |
|---|---|---|---|
| X1 | Maya on `epyc-ai` (3x3090, EPYC, 251 GiB) with the existing `GLM-5.3-Flash-UD-Q4_K_XL` GGUF: load, `--bench`, `STRATA_GLM_NO_SPEC=1` vs spec on 2 and 3 GPUs, greedy token-stream identity, logits vs llama.cpp `glm5-next` dumps | 3090 host | does UD-Q4_K_XL run in Maya; exactness of T1; T1 gain on 3 cards |
| X2 | CPU expert microbench Q4_K/Q5_K/Q6_K (ggml `vec_dot`, Strata `kq_avx2`, Maya `native_expert`) 1…N threads, NPS1/NPS4 | any Zen 2/3 box (e.g. `epyc-ai`) | c in the lane planner; worker count (4080S report: fewer workers faster on hybrid Intel) |
| X3 | `tools/g0/cachesim.py` with lt-Strata `lru_curve.json` as a sanity reference + an `exclusive` RAM-tier mode | this sandbox | T4 sizing; hit at 0.55-0.65 coverage |
| X4 | Strata Qwen3.8-FN with 1 and 2 helper GPUs (`--expert-cache-device1/2 auto --remote-expert-opt`) | 3090 host | T3 scaling beyond 2 cards |
| X5 | pinned-host hop / doorbell / stream-memop round-trip latency microbench | any 2 CUDA GPUs | per-layer overhead budget for T2/T3 (45 layers × RTT) |
| X6 | Maya DSA indexer ring (`8192+64`) vs full-context reference at 32K/128K | 3090 host or CPU reference path | long-context correctness of Maya |

## 6. Negative findings

- Strata upstream: still single-architecture; tail skip lossy by default; helper GPUs 3-4 unvalidated; Foresight net loss.
- lt-Strata: ≤ 2 GPUs, no Q4_K execution, author's direction is lossy (affinity, REAP); simulator does not model a 4-GPU split.
- Maya: no ≥ 4-bit or 4080/EPYC measurement; GLM fast path lacks helper-GPU EP; Q4_K expert kernel not optimised.
- n1oss: re-branded Maya (3 commits). Indras-Mirror/Strata-GLM: README content is DS4's (UNKNOWN GLM results).
- FreeToken: still single GPU, no Q4_K/GGUF for GLM. KTransformers: GLM path needs AVX-512 FP8 + 350 GB RAM.
- llama.cpp `--moe-cache-mib`: no CPU lane, needs every expert in host RAM (does not fit 128 GB for GLM UD-Q4_K_XL, T4).
- Research prototypes: HOBBIT, ProMoE, MoE-Lightning, Klotski, ExpertFlow, FlashMoE, ZipMoE, SpecPrefetch, SSD-LLaMA — code NOT FOUND;
  AdapMoE, DAOP — stale and lossy; llama-moe-cache — AGPL-3.0.
- No independent confirmation found for any number in §0.7 (each is a single author's run on one box).

## 7. Sources

Repositories (clone/ls-remote 2026-10-10; SHA, commit date):
- Niko1221/Strata `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` (2026-10-10 14:49 +0200); base of diff `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (2026-10-08) — https://github.com/Niko1221/Strata
- mw00/project-maya `10b7b078c11f455ff16f3ea3bbea5170c879e767` (2026-10-10) — https://github.com/mw00/project-maya
- lighttransport/Strata branch glm53f `e4dbb18a0cc4de20359c2c74b6ea419776d2e8f5` (2026-10-10 17:17 +0900); previous `27deb91` (2026-10-07) — https://github.com/lighttransport/Strata/tree/glm53f
- sergqwer/strata-glm `ed37419fccd0c52e07d26d526a29c2098f547843` (2026-10-01) — https://github.com/sergqwer/strata-glm
- Grigory-Rylov/strata-glm-3090 `794cd69ae270389c67ed201b855b14aafb5eb875` (2026-10-03) — https://github.com/Grigory-Rylov/strata-glm-3090
- 0xSero/glm53-flash-offload = sybil-solutions/glm-flash-lite `6769b27652a18487758caac45a2b6999dc0a4184` (2026-10-09) — https://github.com/sybil-solutions/glm-flash-lite
- arpahmad/n1oss `ce5e8e6df8290cb51984f9abb3bed67f4032ded2` (2026-10-08) — https://github.com/arpahmad/n1oss
- VecSzn/PolyStrata `27f9ec9dd4d0832525dbedfe396261876cc3855d` (README, docs/GLM.md via raw) — https://github.com/VecSzn/PolyStrata
- Indras-Mirror/Strata-DS4 `924516a06fe2472633514a6eb812fc678d288dff` (README via raw) — https://github.com/Indras-Mirror/Strata-DS4
- Indras-Mirror/Strata-GLM `152c49f4d37c55d62179104a51816346203aff17` — https://github.com/Indras-Mirror/Strata-GLM
- OpenMOSE/Strata-GLM-AMD `6c71ff0d84f1543747ef9bc8451d56bcba3cccf7` — https://github.com/OpenMOSE/Strata-GLM-AMD
- daraskme/Strata-GLM `9e0b779d963c2b1ccc1925864a2247c92e16b43a` — https://github.com/daraskme/Strata-GLM
- noon-at-cgn/Strata `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253` (HEAD; report cites `428d991` on `repro/w7`) — https://github.com/noon-at-cgn/Strata
- FlashML-org/FreeToken `ee9789ff6cdcc2e9c5e9467aba630d38cd230fa6` (2026-10-10) — https://github.com/FlashML-org/FreeToken
- Cerynitius/freetoken-ox-boost `fdb99c6daae162094c8e2a80a5c177136c5e52f6` (2026-10-02) — https://github.com/Cerynitius/freetoken-ox-boost
- kvcache-ai/ktransformers `5d048e516507e1603e0ed19645611b4501476a50` (2026-10-10) — https://github.com/kvcache-ai/ktransformers
- yhfgyyf/vllm-deepseek-v4-sm89 `3f3bbaea916707c851ca027b87b97fb6605998b8` (2026-09-16) — https://github.com/yhfgyyf/vllm-deepseek-v4-sm89
- JustVugg/colibri `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850` (2026-10-06) — https://github.com/JustVugg/colibri
- ggml-org/llama.cpp `abee0c8476b585b0d081d580b5ce21d0addc22c4` (2026-10-10) — https://github.com/ggml-org/llama.cpp
- ikawrakow/ik_llama.cpp `89bba35c8841618502b2be8273fb5265b405aca1` (2026-10-10) — https://github.com/ikawrakow/ik_llama.cpp
- EfficientMoE/MoE-Infinity `0f96d563db2dda5f1834bf667f6fcc44f45b2b9d` (2026-10-08); Tiiny-AI/PowerInfer `8bd56d69906c9d2dba4d3bf6899763401e01a9a4` (2026-05-11); efeslab/fiddler `227715bfd6e8c731b29548eab01d9919c4fe9564` (2024-04-28); Neroued/ninfer `81c8ce093b2c1646a87566a8e59d807fcf0ec95c` (2026-10-07); Wallawalla47/Infernix `a3edb450f373d1044a1589cf4e742a4e7121c14f` (2026-10-09)
- axonn-ai/yalis branch offload_prefetch `b1970f7881129d92448e2f83b0702fea48644b92` (2026-02-28); lshAlgorithm/MoE-SpAc `77bb9cf1e3593b1e1ea0ec543b3e338e7edd1129` (2026-03-23); PKU-SEC-Lab/AdapMoE `2cfe8df43012097170e71658b943d7a86f3f3ced` (2025-04-28); ecolab-nus/DAOP `08fef3330fe73249b757d26c2c00cd0dfecb840d` (2025-06-22); ek15072809/Swap-MoE `50459bb422a01c0ba2e1da08b4d932f652f3e413` (2026-08-29); danveloper/flash-moe `3601d41882a346d85041f7955c66429d248e8cd9` (2026-03-19); ongunm/llama-moe-cache `77c8767d26bd6285b2fe351c58143ee4d6b72fa6` (2026-04-07); yalun753/moe-l2 `3b17ede0001d51dcf8ca1d8f3b0f74fbffb740c6` (2026-10-01); davetha/vllm-expert-cache `813b97ffc8c7646dc7a790a6ef956efcd45f9ece` (2026-09-08); Waltarb/nvmoe `344c5ec77cf4c9b7e62b2c2829baee539d548330` (2026-09-24)

Pages / papers (REPORTED, search snippets unless stated):
- Speculating Experts: https://arxiv.org/html/2603.19289v1 ; https://www.alphaxiv.org/abs/2603.19289
- MoE-SpAc: https://arxiv.org/abs/2603.09983 ; SP-MoE: https://arxiv.org/abs/2510.10302
- FlashMoE (SSD cache policy): https://arxiv.org/abs/2601.17063v1 ; ZipMoE: https://arxiv.org/abs/2601.21198v1
- Flash-MoE (Mac): https://github.com/Anemll/flash-moe ; https://lilting.ch/en/articles/flash-moe-qwen35-397b-metal-inference
- SpecMD: https://arxiv.org/html/2602.03921v1 ; SPICE: https://arxiv.org/pdf/2608.21240 ; MoE-SpeQ: https://arxiv.org/html/2511.14102v1
- vllm-deepseek-v4-sm89 issue #74 (GLM-5.3-Flash on 8x4090 48 GB): https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74
- lt-Strata's own survey sources (llamaperf, Unsloth docs, AcceptMoE, EcoSpec, DraftExpert, SeqMoE, Cacheable by Design?): listed in
  `lt-strata@e4dbb18:docs/GLM_DECODE_RESEARCH.md:70-92` (not re-checked here).
- In-repo: `docs/reassessment/research/R2-engines.md`, `R3-techniques.md`, `README.md` (corrections), `docs/reassessment/ROOFLINE.md`,
  `D1-moe-cache-policy.md`, `tools/g0/cachesim.py` (all @`b6f4fe3`).
