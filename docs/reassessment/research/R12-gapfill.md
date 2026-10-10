# R12. Gap fill for the coverage audit: listed projects that R1–R10 did not cover or could not find (2026-10-10)

Role: research coverage auditor (deliverables A and B). Scope: the items of the customer's list (USER_SPEC §3.2) that
`COVERAGE.md` found **not covered** or **NOT FOUND** in R1–R10, plus one existence conflict between R5 and R8.
Matrix rows: `matrix/R12-gapfill.json` (8 rows, merged into `MATRIX.md`). Clones in the session scratch dir
`research/r12/`; every SHA below is from `git log -1 --format='%H %ci'` or `git ls-remote` run on 2026-10-10.
No builds, no GPU, nothing committed.

**Labels** (BRIEF rule 3): VERIFIED = read in a clone at the recorded SHA (`repo@sha:path:line` or function) or a command
run here; REPORTED = README / paper / search snippet / third-party summary (source named; hardware given); HYPOTHESIS =
my inference (basis given); UNKNOWN; DISPROVED; NOT FOUND (searches listed). AGENTS.md mapping: VERIFIED = CONFIRMED,
REPORTED/HYPOTHESIS = PROVISIONAL.

## 0. Findings

1. **"NInfer Extreme" exists: `Acelogic/ninfer-4090-extreme` @`c90f85e` (2026-10-09, Apache-2.0).** It is an NInfer
   derivative whose own engine (`src/flashnext/`) runs **Qwen3.8-Flash-Next on one sm_89 GPU** with a Strata-style
   VRAM expert cache, CPU-computed misses and attention KV paged through pinned RAM. The build is **sm_89 only**
   (`CMakeLists.txt:3-12`, VERIFIED). This corrects the R2/R5 shorthand "NInfer = sm_120a only, not applicable": true for
   upstream `Neroued/ninfer`, not for the fork family. **But it does not fit our server as is:** its CPU expert kernels
   are AVX-512 only (`src/flashnext/CMakeLists.txt:37-42` `-march=x86-64-v4 -mavx512vnni`; 223 lines with `_mm512` in
   `cpu_experts.cpp`, VERIFIED) and the EPYC Zen 3 has AVX2 only; its GGUF reader knows F32/F16/BF16/Q8_0/Q6_K/IQ4_NL/
   IQ3_S/IQ4_XS and **no Q4_K/Q5_K** (`src/flashnext/gguf.h:17-26`, VERIFIED), so the UD-Q4_K_XL file does not load and
   the UD-IQ4_XS file it uses has IQ3_S gate/up experts (< 4 bit, `research/README.md` correction); one GPU only.
2. **Its measurements are the first public numbers of a Strata-class engine on an Ada card at 250K depth** (REPORTED,
   `docs/maintainer/flash-next-engine.md` §4.5, 1× RTX 4090 + Ryzen 9 7950X, DDR5 55–62 GB/s, UD-IQ4_XS): 45.1 tok/s
   plain and 44.9 tok/s with MTP at a 250,139-token prompt, 69.3% expert hits, median step 18.3 ms; MTP acceptance falls
   with depth (94.5% short, 75% after 32K, 53% at 250K), so **MTP stops paying at depth** on this engine.
3. **SSD-LLaMA is a paper, arXiv 2609.18110 (2026-09-16), with no code found** (REPORTED; searches in §5). Its idea is
   implemented, by the SGLang team's own account, as **SGLang "SSD Expert Pack"**, which is in SGLang main
   (`sgl-project/sglang@0acd93c:python/sglang/srt/layers/moe/expert_pack.py`, VERIFIED) — a GPU-only demand-fetch NVMe
   expert tier for DeepSeek-V4-Flash and Kimi-K3, ~2 tok/s class on one RTX 5090 (REPORTED). Neither R7 nor R10
   mentions it.
4. **SpecPrefetch has code: `wei390/SpecPrefetch` @`54c3295`** (R5 recorded NOT FOUND). It is a trained next-layer
   expert predictor for DeepSeek-VL2-Tiny only, PyTorch, no runtime and **no LICENSE file** (VERIFIED). Recall@8
   against the teacher's top-6: 0.92–0.97 (VERIFIED as file content `eval/eval_results.json`; REPORTED as a
   measurement).
5. **"Swap-MoE" is ambiguous.** R5 took `ek15072809/Swap-MoE` (a 2026 llama.cpp mmap-streaming patch, one line in R5).
   The research system **SwapMoE (ACL 2024, arXiv 2308.15030)** has no code: `MobileLLM/SwapMoE@f08ebc8` holds only
   README + LICENSE and `fqt111/SwapMoE@c73df29` only a README with images (VERIFIED, file listings). It is lossy
   ("virtual experts", Rouge-2 −0.041, REPORTED abstract) and therefore excluded.
6. **Cache-aware routing** (a listed direction) was not studied in R1–R10 (only "lt-Strata affinity routing" was named as
   lossy, R5 §0.8, §3). Code exists: `pmerolla/fomoe@ca31caf` `src/car.c:36 car_evaluate` substitutes the best-scoring
   cached expert for an uncached one when the score ratio ≥ τ and renormalises (VERIFIED); REPORTED 5.1 → 8.8 tok/s at
   +3.5% WikiText PPL (Qwen3.5-397B Q4_K_M, 2× RX 9060 XT). A pre-registered quality study
   (`arjvnv/cache-aware-moe-generation@d22dc89`, REPORTED README) finds damage visible in generated text at bias values
   inside the "deployable" band and a 22× cost difference between fused and host-synchronised implementations.
   **Verdict (HYPOTHESIS): changes routing → violates the "correct math" requirement; keep excluded.**
7. **ProMoE existence conflict resolved:** `promoe-opensource/promoe` exists at `022f8b2` (2024-10-30), as R8 said
   (VERIFIED by clone); R5's "code NOT FOUND" is **DISPROVED**. No LICENSE in the top-level tree (VERIFIED `git ls-tree`).
8. **R8's "no engine examined uses io_uring" stays true for its sample but is not universal:** fomoe submits per-drive
   io_uring reads (`src/nvme_io.c` `drive_io_thread`, one ring per drive, VERIFIED) and SSD-LLaMA reports io_uring direct
   parallel reads (REPORTED).

## 1. NInfer-4090 Extreme — what the code does

Lineage (README "Credits", REPORTED): Neroued/ninfer (sm_120a) → Don-Chad/ninfer-3090 (sm_86) → sergiuszm, UDPSendToFailed
(`ninfer-4090`, ls-remote HEAD `5c60b7c`) → JGamboa/ninfer-4090-windows → Acelogic (adds `src/flashnext/`: 8,027 lines of
`.cpp` and 4,317 lines of `.cu`, VERIFIED `wc -l`).

Placement (VERIFIED, `flash-next-engine.md` §3.1 and code): every dense weight in VRAM in its GGUF format; an expert cache
of ~6,700 of 24,576 experts in a `cuMemCreate`/`cuMemMap` arena whose top can be unmapped and remapped (`vram.cpp`); every
routed expert also in pinned RAM in a lossless repack (Q4L/Q4X/IQ4L/Q8_0); above a 32K window the attention K/V lives in
pinned RAM with a VRAM page cache (`cuda/kv_stream.cu`).

Decode step (VERIFIED, `engine.cpp:1721 host_experts`, §3.3):
```
one CUDA graph per step of T = 1..4 tokens
per layer l: GPU: mixer + attention/DeltaNet + router -> ExpertLink[l] (mapped host memory): ids, x, weights, on_cpu[]
             GPU: computes cached experts + shared expert; reads some misses itself zero-copy from pinned RAM
             host thread: spin on ExpertLink[l].req == seq (queues background swaps while spinning)
                          CpuExperts::run(l, T, x, ids, weights, on_cpu) -> out   (one read per distinct expert)
                          raise flag; GPU one-block kernel waits, adds the CPU partial sum
```
Decode-time cache adaptation (VERIFIED, `engine.cpp:2335` `AdaptParams`; parameters REPORTED as tuned offline by replaying
`NINFER_FN_ROUTING_DUMP` traces):
```
after every step: use[l,e] += pairs of the step, decayed with half-life 32 steps
score = use + prior (5 tokens' worth of decayed long-run routing)
per layer: an uncached expert with a use replaces a cached one scoring >= 1.5 lower; <= 192 swaps per step
swap = evict from the map before the next step (the CPU computes it meanwhile) -> DMA from the pinned copy on its own
       stream -> GPU conversion (byte-identical) -> enters the map at the first step after the copy landed
```
REPORTED effect (32K-token code prompt, 256 decoded tokens): best static cache chosen from the prompt 64%, cache that
follows the last steps 75%, oracle static cache chosen from the decode itself 81%. Same family as Strata `adapt()`
(R8 §9.3) — a second, independent implementation of the policy `tools/g0/cachesim.py` should gain as an arm.

Numerics (REPORTED, §3.2): FP32 activations on the GPU, exact dequantisation; CPU experts use 8-bit activations
(1.1–1.4% relative error per expert output); logits within 1.6e-4 relative of an FP32 reference. Placement changes only
low bits — consistent with R5 §0.9.

What it would take for us (HYPOTHESIS): Qwen3.8-Flash-Next UD-Q4_K_XL fits 4×32 GB VRAM (R5 §1.3), so the offload part is
not needed; the engine would need multi-GPU, Q4_K/Q5_K readers and an AVX2 CPU path. As a donor it offers the KV paging
and the adaptation policy (Apache-2.0). Testable now: compile for sm_89 in CI (Linux build path exists in CMake, untested
here); policy replay offline; AVX-512 kernels need an AVX-512 box.

## 2. SGLang SSD Expert Pack — what the code does

`ExpertPackStore` (`expert_pack.py:232`, VERIFIED): a pack file `SGLANG-EXPERTPACK-v1` with a header carrying
model-identity, source-blob and config sha256 (`ExpertPackHeader`, refused on mismatch), one contiguous record per
(layer, expert) holding gate/up/down. `acquire(layer, topk_ids)` (`:507`): copy the ids to the host (a host sync per MoE
layer), count frequency, for each distinct miss pick a victim (`_victim_slot :464`: free slot, else the lowest-frequency
LRU entry not in the active top-k), `os.preadv` the record in 4 aligned splits on a thread pool into pinned staging, copy
to the GPU slot on a transfer stream, make the compute stream wait on the copy events. No prediction, no CPU expert compute,
O_DIRECT optional (on by default in the 5090 launchers, blog source `:160`). Pack builders exist for DeepSeek-V4 and Kimi-K3
(`model_loader/expert_pack/prepare_*_pack.py`), not for GLM-5.3-Flash. Reported by the authors (blog source file
`lm-sys.github.io blog/2026-08-29-sglang-ssd-expert-pack.md:250-253, 280, 286`, read via raw.githubusercontent; 1× RTX 5090,
Intel Ultra5 230F, 32 GB RAM, PCIe 5.0 SSD, 200-token completions): DeepSeek-V4-Flash decode 1.99 / 1.85 tok/s
(Alpaca / MMLU) against 0.29 / 0.28 for their baseline, unique-key VRAM hit 54% / 46%, 1.66–2.04 GB read from the SSD per
generated token; Kimi-K3 hit 17–23% and 22–28 GB per token. The blog states that the SGLang path is GPU-centric and does not
reproduce the paper's CPU expert execution or CUDA decompression (`:50`). For us (HYPOTHESIS): no decode value (R8 §3: NVMe must not feed weights per token); the identity-checked pack
header is a good pattern for our own NVMe artefacts (prefix caches, Engram tables).

SSD-LLaMA itself (REPORTED, third-party summary `uni042/llm-paper-summary`, file
`papers/inference/01-offload-hierarchical-memory/2026-2609.18110-ssd-llama-ssd-native-trillion-parameter-moe.md`,
"code: null"): llama.cpp-based; per-expert pack; io_uring direct parallel reads; H2D starts per completed read; rANS
lossless compression decoded on the GPU (8.89 → 5.93 GB/token of SSD traffic); RAM/VRAM caches by frequency and recency;
CPU/GPU split chosen by measured token latency (on the RTX 5090 every CPU split was worse). The authors state that predictive
prefetch hides only about 5% on GLM-5.2. Evaluated on short benchmarks with 16–32 GB RAM.

## 3. Cache-aware routing — algorithm and why it stays out

`fomoe src/car.c:36 car_evaluate` (VERIFIED): for each routed expert E not in VRAM/RAM, find the cached, not-yet-selected
expert C with the highest router score; if `s_C / s_E >= τ`, substitute C for E (renormalise unless `QMOE_CAR_DAMPEN`),
with a per-layer substitution budget; a background thread later loads the substituted experts ("backfill") so the cache
does not drift. It is lossy by construction (README marks it "Experimental"). Skliar et al. (TMLR 2025, cited by the arjvnv
study, REPORTED) report >50% fewer misses at 0.1–3% perplexity cost; the arjvnv study shows perplexity and especially
repetition metrics understate the damage in generated text. Our requirement is exact routing (R5 §3 "Not recommended",
README corrections), so no further work.

## 4. Corrections to earlier files

| file / claim | correction | label |
|---|---|---|
| R5 §2.8 "SpecPrefetch … NOT FOUND" (code) | code at `wei390/SpecPrefetch@54c3295` (no licence) | VERIFIED |
| R5 §2.8 "HOBBIT, ProMoE, Klotski NOT FOUND" | ProMoE exists: `promoe-opensource/promoe@022f8b2` (R8 already had it) | DISPROVED for ProMoE |
| R5 §2.8 "SSD-LLaMA NOT FOUND" | paper arXiv 2609.18110 identified; code still NOT FOUND; SGLang SSD Expert Pack implements the idea | REPORTED / VERIFIED |
| R2 §1.3, R5 §2.7 "NInfer/Infernix: sm_120a only — not applicable" | upstream yes; `Acelogic/ninfer-4090-extreme` is sm_89-only and runs Qwen3.8-Flash-Next (AVX-512 host, single GPU) | VERIFIED |
| R8 §0.6 "No engine examined uses io_uring" | fomoe (VERIFIED) and SSD-LLaMA (REPORTED) do | VERIFIED |

## 5. Searches (for every NOT FOUND)

- "NInfer Extreme": WebSearch `"NInfer Extreme" inference engine` (generic results only); GitHub repo search `ninfer
  extreme` (0), code search `"NInfer Extreme"` (0), repo search `NInfer` (209 repos; ports to sm_70/80/86/89, none named
  Extreme); WebSearch extended `NInfer Extreme Qwen3.8 GitHub fork` → `Acelogic/ninfer-4090-extreme` (found; README
  title "NInfer-4090 Extreme").
- SSD-LLaMA code: WebSearch `"SSD-LLaMA" MoE SSD offloading`, `"SSD-LLaMA" paper` (extended) → arXiv 2609.18110; GitHub
  repo search `SSD-LLaMA` (20 unrelated repos), repo search "SSD-native inference trillion-parameter MoE consumer PC" (0),
  code search `"2609.18110"` (72 hits, all paper digests/notes) and `"SSD-LLaMA"` (88 hits: digests and the lmsys blog
  `lm-sys/lm-sys.github.io blog/2026-08-29-sglang-ssd-expert-pack.md`) → **no code repository**.
- SpecPrefetch: WebSearch `SpecPrefetch … 2607.24787 code github` → HF card `jinwei001/SpecPrefetch_deepseekvl2` naming
  `github.com/wei390/SpecPrefetch`; `git ls-remote` + clone confirmed.
- SwapMoE: WebSearch `SwapMoE … github` (paper only); GitHub repo search `SwapMoE` → `MobileLLM/SwapMoE`,
  `fqt111/SwapMoE` (both cloned: no source code).
- HOBBIT (re-check): repo search `HOBBIT mixed precision expert offloading` (0) — still NOT FOUND (agrees with R5/R8).
- Cache-aware routing: repo search `cache-aware routing MoE` → `pmerolla/fomoe`, `arjvnv/cache-aware-moe-generation`.
- colibri routing statistics for GLM-5.3-Flash (to feed cachesim): `docs/routing-telemetry.md@bf24429` describes a usage-file
  format only; `docs/glm53-flash.md:117` "correctness only: no GPU has run the real weights" (VERIFIED) — no data.

## 6. Not verified

- Whether `ninfer-4090-extreme` builds on Linux (CMake has non-MSVC flags; not built: > 20 min and no AVX-512 here).
- Any NInfer-Extreme number on another box; the server-vs-`fn_generate` gap its author lists (46 vs 77 tok/s).
- The lmsys SSD Expert Pack numbers beyond the blog's own source file (no independent run found).
- SSD-LLaMA details beyond a third-party summary (arxiv.org blocked).
- The descriptions of the other NInfer ports (`QilinWan/Ninfer-Blackwell-Ada` "RTX 4090/4090D/4080 SUPER/4070 support",
  `alanthinker/ninfer-4090-yarn`, `Astrangemaninhere/ninfer-fusion` "NVMe cold tiers") — GitHub search descriptions only,
  dense Qwen3.8-27B focus (REPORTED).

## Sources

Repositories (cloned or ls-remote 2026-10-10; SHA, commit date, licence):

| repo | SHA | date | licence |
|---|---|---|---|
| https://github.com/Acelogic/ninfer-4090-extreme | c90f85e813847a629da243461c439037f39c811a | 2026-10-09 01:58 -0400 | Apache-2.0 |
| https://github.com/UDPSendToFailed/ninfer-4090 (ls-remote) | 5c60b7c9b455231795c09da21a9fbb6aa53f08e5 | — | not checked |
| https://github.com/mylordmonkeyman/ninfer-flash-next-v100-backport (README only) | 000fbf54a4ae417c8c6d4827ebfe1b337d271f66 | 2026-09-19 | Apache-2.0 |
| https://github.com/Neroued/ninfer (ls-remote) | 81c8ce093b2c1646a87566a8e59d807fcf0ec95c | — (= R2) | Apache-2.0 (R2) |
| https://github.com/sgl-project/sglang (sparse: expert_pack files) | 0acd93c35c2ae84bc2c5129d5d92916c77c45a70 | 2026-10-10 12:29 -0700 | Apache-2.0 |
| https://github.com/wei390/SpecPrefetch | 54c3295b832de9bc4b4f5e12584e15385fd41f96 | 2026-08-25 10:26 +0800 | none (no LICENSE) |
| https://github.com/MobileLLM/SwapMoE | f08ebc81a00160a6eb1e6da15de7b48ba999659b | 2024-10-09 | Apache-2.0 (README only) |
| https://github.com/fqt111/SwapMoE | c73df29e39b44609011d953bdeef0af55d241878 | 2024-08-13 | none (README only) |
| https://github.com/promoe-opensource/promoe | 022f8b22dc11c5fdfbbcee831f6628c14bccf6dd | 2024-10-30 15:22 +0000 | none in top-level tree |
| https://github.com/pmerolla/fomoe | ca31caffb0bacccb70800e72fc6b697ecbd0ca10 | 2026-03-31 | MIT |
| https://github.com/arjvnv/cache-aware-moe-generation | d22dc896f7fc63ac6aff3d8d66075bf6e5cbeb58 | 2026-09-07 | Apache-2.0 |
| https://github.com/JustVugg/colibri (sparse: two docs) | bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850 | 2026-10-06 | Apache-2.0 (R6) |
| https://github.com/scale-snu/SSD-offloading (ls-remote only; code of arXiv 2508.06978, not read) | 99f3903d67cda89471881a531dd57420abc6e8be | — | not checked |

Files read (raw.githubusercontent.com):
https://raw.githubusercontent.com/uni042/llm-paper-summary/main/papers/inference/01-offload-hierarchical-memory/2026-2609.18110-ssd-llama-ssd-native-trillion-parameter-moe.md ;
https://raw.githubusercontent.com/lm-sys/lm-sys.github.io/main/blog/2026-08-29-sglang-ssd-expert-pack.md (blog source:
SSD-LLaMA relation `:9, :42, :50`, hardware `:17`, numbers `:250-253, :280, :286`).

URLs from search results (REPORTED, pages not opened unless stated): https://arxiv.org/abs/2609.18110 ;
https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/ ; https://www.winzheng.com/en/article/sglang-ssd-expert-pack-consumer-hardware ;
https://digg.com/ai/2lsgezve ; https://arxiv.org/abs/2607.24787 ; https://huggingface.co/jinwei001/SpecPrefetch_deepseekvl2 ;
https://arxiv.org/abs/2308.15030 ; https://aclanthology.org/2024.acl-long.363 ; https://arxiv.org/abs/2508.06978 ;
https://www.alphaxiv.org/abs/2609.18063 (Edge0) ; https://arxiv.org/pdf/2608.21240 (SPICE) ; https://pith.science/paper/2609.18110 .
