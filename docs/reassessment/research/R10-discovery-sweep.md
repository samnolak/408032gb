# R10. Discovery sweep: what the 2026-10-09 research missed (2026-03 … 2026-10)

Role: discovery researcher (breadth). Date of all access: 2026-10-10. Repository read-only; this file and
`matrix/R10.json` are the only outputs. No builds, no GPU.

**Labels.** VERIFIED = I read it in a clone at the recorded SHA (`repo@sha:path:line` or function) or ran the command.
REPORTED = stated by a README / issue / Discord summary / search snippet; hardware, quant and context are given.
HYPOTHESIS = my inference, input stated. UNKNOWN. DISPROVED = contradicted by a primary source.
Mapping to AGENTS.md: VERIFIED = CONFIRMED; REPORTED and HYPOTHESIS = PROVISIONAL.

**Access.** `git clone --depth 1` and `git ls-remote` to github.com and raw.githubusercontent.com work. WebFetch and curl to
every non-GitHub host failed (`getaddrinfo ENOTFOUND` / proxy `CONNECT 403`: x.com, note.com, helix.ml, bearblog,
reddit mirrors). GitHub MCP `search_repositories` / `search_code` work; `issue_read` is refused for repositories outside
this session ("not configured for this session"), so GitHub issue *contents* are known only from search snippets or from
repos that copy them. Every number from a page I could not open is marked "(snippet)".

Clones live in the scratch dir `.../scratchpad/research/r10/<owner>_<repo>` (2.5 GB total). SHAs are from
`git log -1 --format='%H %ci'` in each clone (list in §8).

---

## 0. Ten-line summary

1. **The biggest miss is a vLLM path on consumer cards.** Three community vLLM forks run our candidate models on
   non-Hopper GPUs, including sm_89: `wtdcode/vllm-backport` (ships `-sm89` images; GLM-5.3-Flash AWQ-W4A16,
   Qwen3.8-Flash-Next AWQ-W4A16, DeepSeek-V4.1-Flash with Engram in pinned host RAM), `yhfgyyf/vllm-deepseek-v4-sm89`
   (SM89 Triton path for the DSA indexer, VERIFIED in code) and `Morrowmake/vllm-cmp170hx` (Ampere DSA kernels).
   R2's "GLM-5.3-Flash will not start on sm_89 in vLLM" is true for upstream at `ed7ae70`, **not for these forks**.
2. **Qwen3.8-Flash-Next already meets the customer's numbers on 4 consumer GPUs (REPORTED, medium-high).**
   `halt95/qwen38-flash-next-3090s`: 4×RTX 3090 (220 W cap), INT4 g128 AutoRound experts (≥4 bit), FP8 KV,
   TP2×PP2+EP, MTP K=3: **155.5 tok/s median at 261K** (ctx_tg, thinking on), 117.6 tok/s thinking off at 261,120,
   prefill 5,236 tok/s at 261K, step time 20.0 ms at 261K; three full 262K sessions resident. Runs in a Proxmox LXC;
   also runs without P2P (slower prefill). Independent family results: 2×3090+128 GB DDR4 vLLM overlay 103 tok/s after
   256K; 4×V100 SGLang port 113 tok/s.
3. **GLM-5.3-Flash at ≥4 bit with CPU offload has one strong new data point.** `anoane/glm53-flash-exl3-recipe`:
   EXL3 **4.05 bpw**, one 96 GB card + dual-channel DDR5 (57.5 GB/s), ExLlamaV3 CPU expert tier with **profile-based
   static placement** (patch 0001): **39.7–40.2 tok/s prose, 33.7–34.0 code at 262K**, no speculation. It publishes a
   validated cost model `step_ms ≈ 10.8 + 0.77 × cold_percent` (R² 0.9994).
4. **Expert-offload inside vLLM exists** (`DominikBucko/qwen38-flash-next-2x3090`, Apache-2.0): device-side LRU expert
   map in Triton, tiered VMM expert pools, AVX2 INT4 CPU expert kernel, streamed prefill staging; 2×3090 + Zen 3
   Threadripper PRO, 8-ch DDR4-3200 (our CPU class): Qwen3.8-FN 103.1 tok/s after a 260K prompt. Only for
   `compressed-tensors` WNA16 MoE today; GLM-5.3 W4A16 AutoRound checkpoints exist.
5. **TP without P2P has a ready recipe.** `Cedriceuh/freetoken-rdna3` adds to FreeToken (upstream is still single-GPU
   and has no speculation, VERIFIED at `ee9789f`): TP with uneven split, a **graph-capturable host-memory all-reduce**
   (~12 µs per decode message inside a graph, REPORTED), MTP speculation, an EXL3 reader and a RAM KV tier. Written for
   ROCm, two ranks; the kernel already has CUDA branches.
6. **MiMo-V2.6-Flash has its first real offload numbers**: `SSHdotCodes/mimo-v2.6-flash-pro6000` (llama.cpp + 11 patch
   sets): 1×RTX PRO 6000 96 GB + 128 GB DDR5, exact NVFP4, DFlash: **71.2 tok/s after 131K, 45.1 at 520K** (REPORTED).
   The techniques are per-expert residency from a routing profile and next-layer prediction fused into the router.
   They are Blackwell-FP4-specific in kernels, not in idea.
7. **P2P on Ada inside KVM with small BAR1 is demonstrated** (`xiaoyanzi191/rtx4090-small-bar-p2p`, 7×4090 48 GB, 256 MiB
   BAR1, 580.173.02 patch) and **P2P inside a Proxmox Q35 guest** (club-3090 `docs/PCIE_P2P.md`, 3090, 610.57.04-p2p-v3).
   Measured gain is small for decode (+2.5…7 %), larger for prefill (+16…22 %).
8. **Negative evidence that matters for GLM:** LRU caching on GLM's 288-expert routing gave 9 % hit vs 5.6 % random
   (pctablet505); CPU co-execution *lost* (0.895×) when host RAM was only 1.15× PCIe; MTP with offload *lost* in three
   independent GLM runs because verifying k+1 rows multiplies cold-expert traffic. Profile-based static placement and a
   fast CPU tier are what worked.
9. **Popular but not useful for us:** colibri (41k★, CPU/disk-first), TensorFold (Mac-first), flashrt (sm_90+ only,
   Q2), AFD engines (RDMA + DGX Spark), most DGX-Spark/RTX-PRO recipes (model resident in 256+ GB).
10. Recommended deep looks, in order: halt95 (Qwen path), vllm-backport + yhfgyyf (sm_89 vLLM base), DominikBucko overlay
   (vLLM expert tiering + AVX2), anoane placement profiles (ExLlamaV3 GLM ≥4 bit), freetoken-rdna3 host all-reduce +
   MTP, glm-flash-lite/dsv41-flash-offload mechanisms, MiMo pro6000 runtime, P2P-in-VM routes, sglang-sxm2, GLM
   llama.cpp patch set (GLM53-BLACKWELL). Details in §2.

---

## 1. Method and coverage

- WebSearch, about 30 queries in English, Chinese and Russian (standard and extended). Model + hardware queries
  ("GLM-5.3-Flash 4x 4090", "Qwen3.8-Flash-Next 4x 3090 vLLM", "MiMo-V2.6-Flash 4x 3090", "DeepSeek-V4.1-Flash Engram
  NVMe"), technique queries ("expert prefetch ik_llama", "P2P 4090 driver patch", "moe-cache-mib"), papers, and
  `本地部署 GLM-5.3-Flash 4090`, `专家卸载 推理引擎`, `GLM-5.3-Flash локальный запуск 4090`. The Chinese and Russian queries
  returned only spec pages, no measurements.
- GitHub MCP repository search with targeted queries: `glm53 in:name` (308 repos), `mimo-v2.6 … in:name`,
  `expert offload created:>2026-06-01`, `moe inference consumer gpu created:>2026-05-01`, `tensorfold`, and others.
  This surfaced most of the new material; web search is weak on GitHub repos less than six weeks old.
- The `local-inference-lab/rtx6kpro` wiki (`d24d7ce`, 2026-10-10) contains daily summaries of a ~large RTX-PRO/Spark
  Discord. I grepped them for consumer-GPU reports; they led to several repos below.
- Awesome lists (xlite-dev/Awesome-LLM-Inference, MoE-Inf and JustQJ awesome-moe-inference via raw.githubusercontent):
  no 2026 MoE-offload or consumer-GPU engine entries. Checked, not useful.
- 37 repositories cloned and read (READMEs, docs, and the code paths named in §2). 

---

## 2. New and worth a deep look (ranked)

Ranking criterion: expected contribution to ≥50 tok/s single-stream at 262K, ≥4-bit, on 4×4080S-32G + 128 GB DDR4 +
EPYC Zen 3 AVX2, weighted by how much is real code and how independently confirmed it is.

### 2.1 halt95/qwen38-flash-next-3090s — Qwen3.8-Flash-Next on 4×3090 with vLLM (rank 1)

- Identity: `28cbb7293f134c244b443fb43bbf6e74ff4d8e24` (2026-10-06), Apache-2.0 for the repo; checkpoint under
  `LICENSE.qwen-community-1.0` (VERIFIED, files present). Engine = "public vLLM `v0.30.0` tag plus 97 commits" shipped as
  a git bundle in releases; 6 patches visible in `patches/` (FP8 E4M3 QSA KV reader for sm_86, KV absmax collector,
  `fused_gdn_decode_post_conv_mtp`, FP8 KV clipping counter, PLE offload home device, routed-experts capturer) (VERIFIED,
  `patches/0001…0006`).
- Quant: Intel AutoRound **INT4 g128 routed experts**, RadixArk FP8 n-gram table (~48 GiB in host RAM), MTP draft experts
  INT4, GDN projections INT8, attention BF16 (`docs/reference.md:367-369`, VERIFIED text). Meets "≥4 bit" for experts.
- Layout: TP2 × PP2 + EP, MTP K=3, FP8 KV with static scales, `--block-size 4096`, KV pool 924,993 tokens
  (`docs/reference.md:25-26, 297`). P2P via aikitoria patch (`reference.md:289`); "without peer-to-peer … the release
  ran unchanged … NCCL moves the traffic through host memory … about 4,000 tok/s [prefill] against 5,410",
  short-prompt decode 127–141 tok/s incl. TTFT (`reference.md:100-105`, v2.2.0, different host with x8 links).
- Numbers (REPORTED, `benchmarks/2026-10-06/BENCH-CARD.md:31-39, 113-121`), 4×RTX 3090 @220 W, EPYC 7532 (Zen 2),
  container 96 GB:

  | depth | prefill tok/s | decode tok/s (median, thinking on, MTP) | step ms |
  |---|---:|---:|---:|
  | 10K | 5,219 | 152.1 | 19.36 |
  | 131K (ladder) | — | 161.8 (on) / 117.6 (off) | 19.85 |
  | 200K | 5,324 | 149.7 | 19.92 |
  | 261K | 5,236 | 155.5 (ladder 174.3 on / 117.6 off) | 20.04 |

- Independent confirmation: Discord summary "Eric measured 4×RTX 3090 (96 GB, bf16 ngram, 256 GB host RAM): prefill
  3800–4500 tok/s, decode 150+ tok/s at C1 with MTP" (rtx6kpro `daily-summaries/2026-09/2026-09-21.md:32`, REPORTED,
  context unknown; may be the same person). Same model family on other hardware: §2.4 (2×3090) and §2.10 (4×V100).
- What it means for us (HYPOTHESIS): a step of ~20 ms ≈ 50 target forwards/s on 3090s (936 GB/s). The 4080S has 736 GB/s
  (0.79×), FP8 tensor cores, and 32 GB per card (more KV and fewer PP stages needed). Expect roughly 0.75–0.85× →
  **~115–145 tok/s with MTP at 262K, ~90–100 tok/s thinking off**, if the stack builds for sm_89 and P2P works; minus
  some percent without P2P. The binary artifacts are sm_86-only (`build-artifacts-sm86-…`); a rebuild for `8.9` is
  needed. Main risk: a 97-commit vLLM fork that we would have to carry.
- Testable without target HW: build for sm_89 in CI (compile only), unit tests in `tests/`. Real numbers need the server.

### 2.2 wtdcode/vllm-backport (lazymio) — vLLM for older cards, with sm_89 images (rank 2)

- `29e66dad461b11caae6039f510eaa7913f9dfb19` (2026-09-27), Apache-2.0. Images `lazymio/vllm-backport:latest-sm89`
  (README "Docker Usage" table) and `torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'`
  (`docker/Dockerfile:292`, VERIFIED).
- Models: DeepSeek-V4-Flash-0731 (native FP4), Qwen3.8-Flash-Next (FP8, AWQ W4A16), **GLM-5.3-Flash (FP8, AWQ W4A16)**,
  DSV4-Vision, **DeepSeek-V4.1-Flash with Engram** (README status table). Model code present: `vllm/models/glm5next`,
  `qwen4_exp`, `deepseek_v4_1`; `vllm/model_executor/models/mimo_v2.py` (VERIFIED, `ls`).
- Engram / PLE tables in pinned CPU memory with UVA lookup: `EngramConfig.cpu_offload` (`vllm/config/engram.py:37-47`,
  VERIFIED); a paired LMCache fork for KV.
- GLM-5.3-Flash recipe: TP4, MTP-3, `FULL_AND_PIECEWISE` CUDA graphs with `NCCL_ALGO=Ring NCCL_PROTO=Simple` and
  `--disable-custom-all-reduce`; "Verified with MTP-3 on 4×A100-80GB" (REPORTED, README). No tok/s published.
- Limits: **no expert offload** (grep of `expert.*offload` in `vllm/` found nothing, VERIFIED). GLM AWQ W4A16 is about
  170–180 GB (REPORTED snippet, HF card: 312B routed params INT4, rest BF16) → does not fit 128 GB VRAM. So for GLM this is
  a *kernel base* (DSA/KDA/MLA on sm_80/86/89 without DeepGEMM), not a solution by itself. For Qwen3.8-FN it is a
  second ready path (AWQ W4A16 + `VLLM_PLE_CPU_OFFLOAD` + TP4 + EP, README §Qwen3.8).
- Also used as the base of `sybil-solutions/dsv41-flash-offload` (§2.7).

### 2.3 yhfgyyf/vllm-deepseek-v4-sm89 — SM89 sparse-indexer path in vLLM (rank 3, paired with 2.2)

- `3f3bbaea916707c851ca027b87b97fb6605998b8` (2026-09-16), Apache-2.0; vLLM `0.28.1rc1.dev517`, FlashInfer 0.6.18 fork.
- VERIFIED: `_has_cuda_indexer_mqa_backend()` returns true for `has_deep_gemm() or (HAS_TRITON and not use_fp4_cache and
  current_platform.is_device_capability((8, 9)))` (`vllm/model_executor/layers/sparse_attn_indexer.py:61-66`); the
  error otherwise reads "requires DeepGEMM, or an active Triton backend on SM89 with FP8 query cache" (`:1235-1237`).
  SM89 metadata builders for DSV4 / V4.1 indexers (`vllm/v1/attention/backends/mla/indexer.py:89, 765, 1433, 1463`).
- REPORTED: GLM-5.3-Flash FP8 launched by a community user on 8×RTX 4090 48 GB (issue #74, README changelog 2026-08-31;
  issue body not readable here), TP8, `FLASHINFER_MLA_SPARSE_SM120` backend, MTP-5, 262,144 context. DSV4.1 on SM120
  4×PRO 6000: 267–272 tok/s C1 at 8–32K with DSpark, **80 tok/s at 131K** (DSpark acceptance 4.6 %) (README §3.2).
- Separate REPORTED (snippet): `ykk648/deepseek-v4.1-flash-8x4090`: 120–149 tok/s with DSpark on 8×4090 48 GB.
- For us: the SM89 indexer path is exactly what R2 flagged as a blocker. FP8 weights (306 GiB) cannot fit; combine with a
  W4A16 checkpoint and an expert-offload layer (2.4). HYPOTHESIS: medium porting cost (two forks, same vLLM era).

### 2.4 DominikBucko/qwen38-flash-next-2x3090 — expert tiering and an AVX2 CPU tier inside vLLM (rank 4)

- `939905b0dab8c6eb3852ff35cbbd42d098af2451` (2026-10-02), Apache-2.0. A 27-file overlay on a pinned vLLM image
  (`repro.lock.json`: `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece…`, `0.1.dev20073+g8e685d198`).
- VERIFIED components under `runtime/vllm-overlay/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/`:
  - `lru_map.py:update_kernel`: device-side LRU expert→slot map in Triton (protected slots, ages, clock, misses → victim
    slots), so the cache update stays inside the CUDA graph;
  - `tiered_vmm.py:allocate_tiered/apply_tiered` (VMM-backed hot pools), `tiered_runtime.py`, `staging_copy.py`,
    `stream_stage.py` (prefill streams cold experts 1–2 layers ahead), `gpu_share.py`;
  - `cpu_experts.py:_Service.submit/wait_add` and `cpu_moe.cpp` (882 lines): **AVX2+F16C INT4 nibble dequant**
    (`cpu_moe.cpp:33-70`, `_mm256_cvtph_ps`, `_mm256_srli_epi32` masks) for GPTQ/AutoRound-packed experts;
  - `ple_offload/` worker process for the n-gram table.
- Host (REPORTED, `docs/hardware.md:22-32`): 2×3090 @300 W, **Threadripper PRO 5975WX (Zen 3), 8×16 GB DDR4-3200, 8
  channels**, P2P passed. This is the closest CPU/RAM class to our server in any report found.
- Numbers (REPORTED, README "fast 256K"): 260,096-token prompt + 2,048 out: **103.1 tok/s best, 92.6–103.1 over 3 runs**;
  131K: 104.5; agent-128K profile 109.8; 64 GB-RAM profile 84–89 at 131K. "Decode speed depends on free RAM": 91–95 with
  another job filling RAM.
- For us: this is the missing piece that makes a vLLM path viable for GLM-5.3-Flash at ≥4 bit (experts ≈150+ GB > 128 GB
  VRAM). Port = make GLM's `FusedMoE` use the same `compressed_tensors_moe_wna16` tiered method (checkpoint:
  `Intel/GLM-5.3-Flash-W4A16-AutoRound`, used by `zebgop-ops/glm53flash-pp`). Risk: the overlay is model-specific in
  places (PLE, GDN); GLM routing is far sparser than Qwen's (288/8 vs 512/10 but much flatter, see §4), so LRU will be
  weaker than profile placement (§2.5). HYPOTHESIS cost: 3–6 engineer-weeks.

### 2.5 anoane/glm53-flash-exl3-recipe — ExLlamaV3 expert-placement profiles, GLM-5.3-Flash 4.05 bpw (rank 5)

- `116e6a2e0ada95b5f3b592581ac0cd1ff2788adf` (2026-09-01). **No LICENSE file** (VERIFIED, `ls`) → licence UNKNOWN; the
  same change is on branch `moe-expert-profiles` of `anoane/exllamav3-anemone` (HEAD `c6d2e3e`, ls-remote), commit
  `1149e2f` (README). Patch `0001-moe-expert-placement-profiles.patch` touches `exllamav3/model_init.py` and
  `exllamav3/modules/block_sparse_mlp_cpu.py` (README).
- Mechanism (REPORTED, README): `-mcs N` puts the tail N experts of every MoE layer on the CPU; a **profile** (27 shipped,
  built from domain corpora at long context) renumbers experts so the hot ones are resident; `-mcpm static`.
- Numbers (REPORTED, README:22-36): RTX PRO 6000 96 GB + Ryzen 9 9950X3D, 128 GB DDR5 dual-channel (57.5 GB/s), EXL3
  4.05 bpw (~165 GB), `-cs 270336 -mcs 128`: **prose 39.71–40.24 tok/s, code 33.66–34.01 tok/s at 262,144**; 32K: 32.9;
  before (stock `-mcs 180`, no profile): 16.8–17.4. 1M context: 33.0. No speculation.
- Cost model (README:50-56): `step_ms ≈ 10.8 + 0.77 × cold_percent`, slope = 42 layers × 8 × 12.62 MB / 100 / 53 GB/s;
  at 256K: CPU-expert stall 38.0 ms, all GPU kernels 14.5 ms, attention/DSA/KV 0.95 ms; attention grows +0.137 ms over a
  64× context increase.
- Its own negatives: MTP acceptance 0.462 at 32K "does not pay" (multiplies cold-expert bytes); quantized KV blocked on the
  DSA path ("asserts `qc is None`", fp16 KV only); non-uniform per-layer residency 1.005×; streaming experts to VRAM
  instead of CPU compute ~1.0×; CPU kernel already at 52 GB/s of 57.5.
- Same author, same box, other engines (REPORTED, `anoane/02-glm53-flash-nvfp4-freetoken` README:107-120, 22.8K prompt):
  ExLlamaV3 CPU tail without profile 17.6 tok/s; FreeToken NVFP4 22.2–22.7 tok/s; FreeToken MTP "12 tok/s with corrupted
  output", off.
- For us (HYPOTHESIS, from the cost model): we can keep ~100 GB of the ~151 GB experts resident (vs ~85 of 151 there),
  and our 8-ch DDR4 could give 2× its CPU bandwidth if the AVX2 kernel scales (UNKNOWN: ExLlamaV3's CPU kernel was
  measured at 52 GB/s with AVX-512 on Zen 5). Cold rate and CPU bandwidth both improve; GPU time worsens (4080S 736 GB/s
  vs 1.8 TB/s, and ExLlamaV3 CPU offload is layer-split only, `doc/env_vars.md:315-325@151539c` per R2). Net: roughly
  35–55 tok/s at 262K without speculation; whether MTP/DFlash2 helps depends on the cold rate (§4).
  Testable now: the profile builder and the cost model run on CPU-side traces; our `tools/g0/cachesim.py` can replay them.

### 2.6 Cedriceuh/freetoken-rdna3 — TP without P2P, MTP and EXL3 on top of FreeToken (rank 6)

- `a4469ae01d60180df72b427d88358a04b44661e5` (2026-10-10), Apache-2.0, fork of FreeToken at `0d652e7` (2026-09-26)
  with every change listed (`docs/rdna3/changes-from-upstream.md`, VERIFIED).
- Upstream FreeToken is still single-GPU: `parser.error("tensor parallelism is not supported yet: --gpu takes one
  entry")` (`FlashML-org/FreeToken@ee9789ff:python/freetoken/server/args.py:834`, VERIFIED 2026-10-10) and has no
  `spec_decode.py` (VERIFIED, `ls`). R2's statement stands for upstream.
- What the fork adds (VERIFIED by file and doc): per-rank sharding for `qwen4_exp` (attention, GDN heads, experts, PLE),
  uneven split (`distributed/split.py`), **host-memory all-reduce** (`kernel/host_allreduce.py`,
  `kernel/csrc/jit/host_allreduce.cuh`), MTP head + verify/rollback (`spec_decode.py`, 274 lines; `models/qwen4_exp/mtp.py`),
  EXL3 codec and Triton kernels, a RAM KV tier (`kvcache/host_kv_pool.py`).
- Host all-reduce algorithm (`host_allreduce.cuh` header, VERIFIED): both ranks map the same shared host pages
  (fine-grained/uncached); one block per rank: (1) copy local tensor into own slot, (2) publish a sequence number,
  (3) spin on the peer's sequence number (timeout trap), (4) add the peer's slot in fp32 in fixed order. The sequence
  counter lives in device memory and is bumped by the kernel, so the launch has fixed arguments and is CUDA-graph
  capturable; slots double-buffered by parity. Host side is HIP-only today (`host_allreduce.py:90`, `hipHostRegister`);
  the kernel already has CUDA branches (`%globaltimer`).
- Numbers (REPORTED, README): Qwen3.8-FN on RX 7900 XTX + XT, MTP on, 9K→248K: NVFP4 72–86 tok/s, EXL3 3.05 bpw 83–107,
  **EXL3 4.05 bpw 76–87**; 4 agents with 82–117K contexts: 35 s instead of 599 s thanks to the RAM KV tier.
- For us: the all-reduce idea is the cheapest way to make TP/EP decode tolerable without P2P. Port: `cudaHostRegister`
  (mapped, portable) + `cudaHostGetDevicePointer`, generalise to 4 ranks (4 slots × 2 parities, reduce in rank order).
  HYPOTHESIS: ~1 week incl. tests; decode win depends on how many all-reduces per token (GLM: 1–2 per layer × 45 layers).

### 2.7 sybil-solutions/glm-flash-lite (= 0xSero/glm53-flash-offload) and sybil-solutions/dsv41-flash-offload (rank 7)

- glm-flash-lite `6769b27652a18487758caac45a2b6999dc0a4184` (2026-10-09), MIT. **Same SHA as R2's
  `0xSero/glm53-flash-offload`** (both `ls-remote` give `6769b27`), i.e. the repo was renamed/transferred; it now has
  280★. R2 row 8 numbers still hold; new since R2:
  - NVMe tier: 55 GiB RAM cap + 117 GB 4K-aligned O_DIRECT expert store on 4×NVMe RAID0: **20.0 tok/s at 93,437 tokens**
    (lab run 2026-10-09, "accepted"); 16 GiB cap: 14.7 (REPORTED, README table). Still **3.05 bpw** (below our floor).
  - Mechanisms (VERIFIED in docs and code names): no-duplicate VRAM/RAM tiers, victim ring drained by a copy engine,
    host controller thread with per-expert "landed" flags in mapped memory, **layer-ahead prefetch** ("the next layer's
    router runs on the current layer's input"; NVMe misses 40 → 22/token, +13 %), **decode lookahead** that launches
    forward(k+1) from the device-side token before the host sees token k (`glm53/k_lookahead.py:1-30`; ~2.1 ms/token
    bubble removed, bitwise identical), AVX2 CPU lane computing ~150 cold experts/token in place
    (`docs/how-it-works.md:100-116`; not bit-exact: KL 0.0047).
  - Profile (README "Where the time goes", v4.6, 55.8 ms/token): PCIe copies 31 %, non-MoE GPU 20 %, MoE kernel 18 %, CPU
    wait 18 %, NVMe 10 %; "every lane at its hardware floor about 41 [tok/s]".
- dsv41-flash-offload `d890a1c897f1c9d6d981473ec106224aeb9cbb4d` (2026-10-07), MIT, created 2026-10-02 (new).
  DeepSeek-V4.1-Flash EXL3 **3.0 bpw** on 1×3090 + 503 GiB DDR4 8-ch (EPYC 7443P): **decode 20.59 tok/s at 261,000**,
  16.4–20.9 over 8K–261K, prefill 589–813 tok/s, Engram mmapped on NVMe, exact prefix caching (D130/D141; README
  tables). Stack: vllm-backport + `vllm_exl3` plugin + Ampere patches, sm_86.
- For us: mechanisms are directly reusable ideas; the repos themselves are single-GPU and below 4 bit. DSV4.1 needs ≥256
  GiB host RAM in this design (README "At a glance") → does not fit our 128 GB.

### 2.8 SSHdotCodes/mimo-v2.6-flash-pro6000 — MiMo-V2.6-Flash with an expert pool (rank 8)

- `464bf2676aa9bd106cc9833684c0d993e825c005` (2026-10-06), MIT. Engine = llama.cpp `58367713a6…` + 11 patch sets
  (`sources.lock.json`, VERIFIED): a published 45-patch MiMo NVFP4 runtime (FP8 KV, DFlash, expert pool, tool parser),
  then speed v2–v6: next-layer expert prefetch, lossless NVFP4C re-encode, in-kernel expert ready flags, **per-expert
  residency** (hot-first renumbering, pinned + dynamic VMM pools from a routing profile), grouped kernels for 9–512-token
  expert batches, speculative rejection sampling, user-message checkpoint gate; v6 fuses the next layer's expert
  prediction into the router matmul.
- Numbers (REPORTED, README): RTX PRO 6000 96 GB + Ryzen 9950X3D + 128 GB DDR5 (Gen5 x16), exact NVFP4, 524,288 ctx:
  coding 77.0 tok/s, tool agent 70.7, **after a cold 131K prompt 71.2 tok/s**; v5 at **519,882 tokens 45.1 tok/s**;
  cold 262K prefill 82.4 s. 95.9 GiB of experts pinned; "128 GB RAM is the minimum".
- For us: the only engine with measured MiMo-V2.6 offload numbers. The kernels rely on Blackwell FP4/FP8 block-scaled
  tensor cores (v5.2 notes); on sm_89 the experts would need a W4A16 (Marlin-style) path. The *residency and prediction*
  logic is portable C++ in llama.cpp. HYPOTHESIS: MiMo on 4×4080S would land well below 71 tok/s at equal cold rate,
  because per-card bandwidth is 2.4× lower; it gains from 128 GB of VRAM (most experts resident).

### 2.9 P2P on Ada and inside VMs: xiaoyanzi191/rtx4090-small-bar-p2p and club-3090 PCIE_P2P.md (rank 9)

- `xiaoyanzi191/rtx4090-small-bar-p2p@dbd0cdf15c76548626ab224dbd9e34097d866ea3` (2026-09-16), MIT (driver patch keeps
  NVIDIA MIT/GPLv2). `patches/580.173.02-p2p-dynamic-align64k-budget.patch`, `tests/owner_verify.cu`,
  `tests/peer_matrix.cu` (VERIFIED, files). Tested on **7×RTX 4090 48 GB inside a KVM guest with 256 MiB BAR1 per GPU**:
  fixes a peer-PTE alignment bug (payload landed 0x150000 bytes early) and a BAR1 budget heuristic; 42 directed pairs ×
  630 cases zero errors; NCCL 2/4/7-GPU all-reduce pass; 2-GPU 8 MiB all-reduce 18.99 GB/s P2P vs 13.57 SHM; end-to-end
  **+2.5–2.6 % at concurrency 1** (REPORTED, README).
- club-3090 `docs/PCIE_P2P.md:481-560` (`noonghunna/club-3090@c8b6a73`, Apache-2.0): Proxmox + Q35/OVMF + VFIO, 3090,
  aikitoria `610.57.04-p2p-v3` commit `94b69eb…`, OVMF MMIO64 256 GiB, `iommu=pt`, no `x-nv-gpudirect-clique`; matched
  A/B with custom all-reduce off: prefill +21.9 % (10K) / +16.3 % (90K), decode +6.3 % / +7.1 % (REPORTED).
- For us: our cards are modded 4080S (BAR1 size UNKNOWN; small BAR is plausible on modded VBIOS) in a Proxmox VM — both
  findings are directly on our risk list (R4 §4). Use `owner_verify.cu` as the acceptance test; do not trust
  `topo -p2p` alone. Expected gain for decode is single-digit percent; P2P is not the main lever.

### 2.10 dg1kjd/sglang-sxm2 — SGLang ported to Volta with GLM/Qwen/DSV4.1 (rank 10)

- `73abfe1dd7e7aae7a62b222a201e48f532fb298e` (2026-10-07), Apache-2.0; credits `haohervchb/sglang-V100` (TileLang attention
  backend, QSA and GDN kernels, NVFP4 path for hardware without FP4, TurboMind sm70 integration) (README:181).
- Numbers (REPORTED): Qwen3.8-FN NVFP4 on **4×V100-SXM2-32GB TP4 (NVLink)**, MTP 3/4, ctx 262,144: **112.8 tok/s**
  (six short chat prompts, `docs/v100/Qwen3.8-Flash-Next.md:43-57`), "about 47 target forwards per second"; GLM-5.3-Flash
  NVFP4 8×V100: ~175 tok/s (README:44); DSV4.1 with Engram in host RAM + experts spilled to host: 7–11 tok/s.
  PCIe V100 variant documented (`SGLANG_CUSTOM_AR_ALLOW_PCIE=1`) but unmeasured.
- For us: evidence that the SGLang model code can be carried to a pre-Hopper arch with Triton/TileLang kernels; the radix
  cache "replays an approximate linear-attention state for a cached prefix", so greedy output can differ cold vs warm
  (`docs/v100/Qwen3.8-Flash-Next.md:172`) — a correctness caveat for hybrid models.

### 2.11 Also worth a look (not top-10)

| item | why | label |
|---|---|---|
| `abtraore/GLM53-BLACKWELL@64ef4e6` (no licence file) | 5 llama.cpp patches for glm5next: `can_reuse` for the kpool graph input (+37 %: the 45-layer graph was rebuilt on the host every token), NextN MTP head (stubbed in the branch), **`ssm_a` placement fix (34 tensors sat in host memory and were copied every graph)**, kpool mask fill, n_kv reuse pad. 4×5090 UD-IQ1_S: 43 → 130–138 tok/s, 14.5 → 58 at 85K | REPORTED; whether mainline `79e2e74` has these fixes is UNKNOWN — check before E1 |
| `Morrowmake/vllm-cmp170hx@ab60b72` + recipe `cb3311e` (MIT) | Ampere (sm_80) kernels for GLM-5.3 DSA in vLLM; TP4 or PP4; P2P content check at startup; W4A16, 262K, DFlash2: 482 tok/s C1 TP4 (cards unlocked to 64 GB → model resident) | REPORTED; not our memory regime |
| `zebgop-ops/glm53flash-pp@1a14495` (Apache-2.0) | Python-only overlay on the official `vllm-openai:glm53-flash` image for sm_80, `Intel/GLM-5.3-Flash-W4A16-AutoRound`, PP4 over Gen2 x4 **without P2P**: 66–70 tok/s C1, 63.9 at 512K | REPORTED; resident in 256 GB |
| `kaka86mm/glm53-flash-170hx@218b47f` (Apache-2.0) | PP4 + EXL3 4bpw + offline Marlin INT4 sidecar + DFlash2 k=7 adaptive; FULL_AND_PIECEWISE graphs "the single largest decode lever"; KV tier 1.58M GPU + 3M in pinned `/dev/shm` | REPORTED; resident |
| `hughmadden/glm53f-afd@983f7f1` (MIT), `tpurtell/ds41rt@3067d06` (MIT) | Attention–FFN disaggregation: attention+KV on one GPU, routed experts width-sharded across ranks; GLM no-drafter 52.7 tok/s, DFlash2 79–195 | REPORTED; needs RoCE + GB10; the *split* idea (attention GPU + expert GPUs) is a candidate layout for 4 cards |
| `pctablet505/glm53-flash-single-gpu@865edb6` | the best negative-results table for GLM offload (§4) | REPORTED |
| `neron82/helios-glm53flash@700eebe` (MIT) | from-scratch C++/CUDA GLM engine for 2×3090 with asymmetric PCIe and AVX2-only Ryzen 3700X, 27 GB/s RAM: 17.8 tok/s at 246K | REPORTED; ~2 bpw |
| `DominikBucko/qwen38-flash-next-3090@10b24c2` | single 3090 + 64 GB: 43–51 tok/s at 128K | REPORTED (ls-remote only) |
| `UnsignedChad/windows-freetoken-mtp@4caf402` | MTP over FreeToken offload, Qwen3.6-35B, α=0.894: **projected** 1.8× | REPORTED projection, not end-to-end |
| rtx6kpro daily summaries | turboderp DSV4-Flash-0731 EXL3 3.04 bpw ~100 tok/s on 4090+PRO 6000 (2026-08-23); custom llama.cpp on 5090 + Optane PMEM ~50 tok/s on two Qwen3.8FN streams (2026-09-30) | REPORTED (Discord summaries) |

---

## 3. Best REPORTED single-stream decode at long context, hardware close to ours

None of these is on 4×4080S-32G. "≥4b" = routed experts at ≥4 bits. Confidence: H = repeated runs + published method and
raw files; M = one source with method; L = snippet / summary / unknown method. All REPORTED unless noted.

### 3.1 Qwen3.8-Flash-Next (125B + 51B n-gram)

| # | hardware | quant (≥4b?) | engine | context | decode tok/s | spec | source | conf. |
|---|---|---|---|---|---|---|---|---|
| Q1 | 4×RTX 3090 220 W, EPYC 7532, P2P patched | INT4 g128 AutoRound (yes), FP8 KV | vLLM v0.30.0+97 (halt95 v2.5.1) | 261K | **155.5** median (139.5–160.8); 117.6 thinking off | MTP K=3 | halt95 `BENCH-CARD.md:37-39,121` | H |
| Q2 | same, no P2P, x8 links, 2 NUMA nodes | same | same (v2.2.0) | 24-token prompt | 127–141 incl. TTFT | MTP | halt95 `reference.md:100-105` | L |
| Q3 | 2×RTX 3090, TR PRO 5975WX (Zen 3), 8-ch DDR4-3200 128 GB | INT4 AutoRound + FP8 PLE (yes) | vLLM + tiering overlay | 260,096 | **103.1** best, 92.6–103.1 | MTP-3 | DominikBucko README "fast 256K" | M |
| Q4 | 4×V100-SXM2 32 GB, NVLink | NVFP4 (yes) | SGLang sm70 port | short prompts, ctx 262K | 112.8 | MTP 3/4 | sglang-sxm2 `docs/v100/Qwen3.8-Flash-Next.md:47-55` | M |
| Q5 | RX 7900 XTX + XT (AMD) | EXL3 4.05 bpw (yes) | FreeToken-rdna3 TP2 | 9K→248K | 76–87 | MTP | freetoken-rdna3 README | M |
| Q6 | 6×RTX 4090, no NVLink | UD-Q4_K_XL (yes) | llama.cpp fork lukolszewski | 250K | 48.7 | — | R2 row 14 | M |
| Q7 | 4×RTX 3090 | bf16 n-gram, quant UNKNOWN | UNKNOWN (likely vLLM) | UNKNOWN | 150+ | MTP | rtx6kpro 2026-09-21:32 | L |
| Q8 | 1×RTX 5080 16 GB (sm_120), DDR5 | Q2_0 (**no**) | flashrt | 250K | 104.9 (91.9 no MTP) | MTP | flashrt README | M |

### 3.2 GLM-5.3-Flash (320B / 18B active)

| # | hardware | quant (≥4b?) | engine | context | decode tok/s | spec | source | conf. |
|---|---|---|---|---|---|---|---|---|
| G1 | 1×RTX PRO 6000 96 GB, Ryzen 9950X3D, DDR5 2-ch 57.5 GB/s | EXL3 4.05 bpw (yes) | ExLlamaV3 + placement profiles | 262,144 | **39.7–40.2 prose, 33.7–34.0 code** | none | anoane recipe README:29-36 | M |
| G2 | 1×RTX PRO 6000, PCIe Gen5, DDR5-3600 | NVFP4 (yes) | FreeToken v0.1.2 + ox-boost | 128K / after 232K | 36.8 / 25.1 | none | R2 row 11 | M |
| G3 | same class (anoane box) | NVFP4 (yes) | FreeToken + b12x | 22.8K | 22.2–22.7 | none | anoane/02 README:111-117 | M |
| G4 | 1×RTX 3090, EPYC 7443P 8-ch DDR4 | EXL3 3.05 bpw (**no**) | glm-flash-lite (ExLlamaV3 + tiers) | 93,437 | 28.6 (all RAM); 20.0 (55 GiB + NVMe) | none | glm-flash-lite README tables | H |
| G5 | 4×RTX 3090 TP4, all resident | EXL3 2.05 bpw (**no**) | ExLlamaV3/TabbyAPI | ~300K | 84.8 (105 short) | MTP k=3 | club-3090 issue #1615 (snippet) | L |
| G6 | 4×RTX 5090 | UD-IQ1_S (**no**) | llama.cpp + 5 patches | 86K | 58 (130–138 short) | MTP n=4 | GLM53-BLACKWELL README | M |
| G7 | 4×RTX 3090 | IQ4_XS (mixed, see R2 README) | llama.cpp club-3090 moecache | 90K | 16.0 | none | R2 row 1 | M |
| G8 | 2×L40S (sm_89), EPYC 9845 (AVX-512) | FP8 (yes) | CPU expert offload, engine UNKNOWN | UNKNOWN | 18.3 | UNKNOWN | HF zai-org/GLM-5.3-Flash discussion 38 (snippet) | L |
| G9 | 1×RTX PRO 6000 | NVFP4 (yes) | vLLM PR #53906 + hot cache | 304,800 prefilled | ~15–16 | MTP-2 | pctablet505 README | M |
| G10 | 2×3090, Ryzen 3700X AVX2, DDR4 27 GB/s | EXL3 ~2 bpw (**no**) | helios (custom) | 246K | 17.8 | UNKNOWN | helios README | M |
| G11 | 4×3090 | UNKNOWN | UNKNOWN | 32K / 64–80K | ~11 / ~5 | UNKNOWN | x.com superalesha (snippet; page not loadable) | L |
| G12 | 4×CMP 170HX unlocked 64 GB (256 GB VRAM), Gen2 x4, no P2P | W4A16 AutoRound (yes) | vLLM overlay, PP4 | 512K | 63.9 (66–70 short) | MTP | zebgop-ops RESULTS.md:19,42 | M — resident, not our regime |

### 3.3 MiMo-V2.6-Flash (309B / 15B active)

| # | hardware | quant | engine | context | decode tok/s | spec | source | conf. |
|---|---|---|---|---|---|---|---|---|
| M1 | 1×RTX PRO 6000 96 GB, 9950X3D, 128 GB DDR5, Gen5 | NVFP4 exact (yes) | llama.cpp + 11 patch sets | 131K / 520K | **71.2 / 45.1** | DFlash | SSHdotCodes README | M |
| M2 | Mac Studio M3 Ultra 256 GB | MLX 4-bit (yes) | MLX | UNKNOWN | 59.4 | UNKNOWN | openclawdc.com (snippet) | L |
| M3 | 2×DGX Spark | UNKNOWN | vLLM + DFlash | UNKNOWN | ~38 (peak 80 code) | DFlash | NVIDIA forum / contextstudios (snippet) | L |
| M4 | 4 DGX Spark + RTX 5090 (AFD) | UNKNOWN | mimo26f-afd | UNKNOWN | ~100 | UNKNOWN | rtx6kpro 2026-09-26:42 | L |
| — | consumer multi-GPU (3090/4090/4080) | — | — | — | **no report found** | — | searches §1 | UNKNOWN |

### 3.4 DeepSeek-V4.1-Flash (552B + 196B Engram) and V4-Flash

| # | hardware | quant | engine | context | decode tok/s | spec | source | conf. |
|---|---|---|---|---|---|---|---|---|
| D1 | 1×RTX 3090, EPYC 7443P, 503 GiB DDR4 8-ch | EXL3 3.0 bpw (**no**) | vllm-backport + vllm_exl3 + AVX2 tier | 261,000 | **20.59** | none | dsv41-flash-offload README (D119) | H |
| D2 | 8×RTX 4090 48 GB (384 GB VRAM) | native FP4/FP8 | vLLM sm89 fork | random input | 120–149 | DSpark | ykk648/deepseek-v4.1-flash-8x4090 (snippet) | L |
| D3 | 4×RTX PRO 6000 | native | yhfgyyf vLLM fork | 131K | 80.0 | DSpark (acc. 4.6 %) | yhfgyyf README §3.2 | M |
| D4 | 8×V100 | official + Engram in host RAM | SGLang sm70 | UNKNOWN | 7–11 | MTP | sglang-sxm2 README | M |
| D5 (V4-Flash-0731) | RTX 4090 + RTX PRO 6000 | EXL3 3.04 bpw (no) | ExLlamaV3 | UNKNOWN | ~100 | UNKNOWN | rtx6kpro 2026-08-23:5 | L |

**Reading (HYPOTHESIS).** For Qwen3.8-FN the target (≥50 tok/s at 262K, ≥4 bit) is exceeded 2–3× on 3090-class hardware
by vLLM with MTP (Q1, Q3). For GLM-5.3-Flash at ≥4 bit the best single-stream number with offload is ~40 tok/s at 262K
(G1) without speculation, on one Blackwell card with 2.4× the bandwidth of a 4080S. No report reaches 50 tok/s at 262K
for GLM ≥4 bit with experts partly in host RAM. MiMo has one comparable data point (M1, 71 at 131K, Blackwell FP4).

---

## 4. Concrete algorithms worth porting (from code read)

**A. Profile-based static expert placement + CPU tier (anoane; also MiMo pro6000 "per-expert residency").**
```
offline: run corpus at long context → per (layer, expert) routing counts → order experts hot-first per layer
load:    renumber experts so ids [0, R) are hot; resident = R per layer (uniform R is ~optimal: +0.5 % for per-layer R)
decode:  router → topk ids → ids < R: GPU grouped GEMV from VRAM; ids ≥ R: CPU kernel on pinned RAM copy,
         overlapped with the GPU part of the same layer → GPU adds CPU partial sum
model:   step_ms = gpu_ms + (layers·topk·expert_bytes / cpu_GBps) · cold_fraction    (fit R² 0.9994 on one box)
```
Our use: ExLlamaV3 (pinned `151539c` in tools/engines/pins.json) + patch 0001; or the same idea in llama.cpp via
`-ot` regexes generated from the profile. Feed the profile into `tools/g0/cachesim.py` to compare with LRU.

**B. Device-side LRU expert map (DominikBucko `lru_map.py:update_kernel`).** One Triton kernel per MoE layer takes the
step's requested global ids, marks slots used this step as protected, picks victims by (age, slot) order among
unprotected slots, writes `cache_map`, `slot_global_ids`, `slot_ages`, emits (miss_local_id, slot) pairs for the copy
engine, and advances a device clock — no host round-trip, graph-capturable.

**C. Host-memory all-reduce without P2P (freetoken-rdna3 `host_allreduce.cuh`).** Per call: copy → publish seq →
spin on peers' seq (system-scope volatile load, timeout trap) → fixed-order fp32 sum; seq in device memory;
double-buffered slots. Generalise to 4 ranks; use for decode-sized messages only (default max 256 KiB), NCCL above.

**D. Layer-ahead router prefetch and decode lookahead (glm-flash-lite).** Run layer l+1's router on layer l's input to
start reading its likely cold experts one layer early (NVMe misses 40 → 22/token); launch forward(k+1) on the device-side
sampled token before the host processes token k (removes ~2.1 ms/token on a 3090; only when one job, no stop/grammar
dependency).

**E. Fused next-layer expert prediction (MiMo pro6000 v6).** Compute the next layer's prediction as a second channel of
the router matmul + fused top-k (replaces ~27 µs/layer of kernels), and issue its prefetch after the current layer's
expert kernels are launched.

---

## 5. Checked and not useful (with reason)

| item | SHA / date | reason |
|---|---|---|
| JustVugg/colibri (Apache-2.0, 40,970★) | `bf24429` 2026-10-06 | CPU/disk-first C engine; GLM-5.3-Flash CUDA path "correctness only: no GPU has run the real weights" (`docs/glm53-flash.md:117`); best GPU numbers 9.0–9.2 tok/s (GLM-5.2 on 6×5090). Interesting only for routing telemetry (`docs/routing-telemetry.md`) and its note "MTP … 32 % loss around 85 % expert hit" (README:613) |
| ashhart/TensorFold (Apache-2.0, 1,215★) | `9041e20` 2026-10-10 | Zig engine; GLM-5.3-Flash on Metal (two M5 Ultras) only; CUDA = Nemotron on GB10/3090 (README:59-77) |
| BFinn/flashrt (Apache-2.0) | `90e7a55` 2026-10-03 | Excellent Qwen3.8-FN numbers but CMake refuses < sm_90 (PDL) (`CMakeLists.txt:98-106`), Q2_0 only, one GPU. Its claim that Strata's cache path "is NOT CORRECT" refers to Strata 0.1.6; Strata later measured cache parity (`Niko1221/Strata bench/results/2026-09-27-cache-parity/README.md`, via code search) |
| optim-enterprises-bv/titan-engine (MIT) | `12bda9c` 2026-10-04 | mistral.rs fork, one laptop GPU, Qwen3.6/Qwen3-Next; none of our models |
| yairpatch/flyweight (Apache-2.0) | `a65e4a4` 2026-10-10 | one NVIDIA card; Qwen4exp/DSV4 GGUF but no GLM5-Next/MiMo; no multi-GPU |
| vcruz305/vllm-exl3 | `223e246` 2026-09-25 | **AGPL-3.0-only**; target "one GB10, TP=1"; CPU-offload planner without executor (README table) |
| davetha/vllm-expert-cache (Apache-2.0) | `813b97f` 2026-09-08 | AMD/hipcc only; Qwen3-30B on MI210 |
| Waltarb/nvmoe, yalun753/moe-l2, ek15072809/Swap-MoE, ap03906101/moe-hotcache, biggiehorizon/llamacppMoEMaster, bdrazn/blobpager | 2026-08…10 | single small GPU llama.cpp patches; GLM only at IQ2 (nvmoe 1.8–8.9 tok/s); ideas already covered by llama.cpp `--moe-cache-mib` and R3 |
| AFD engines: hughmadden/glm53f-afd, mimo26f-afd, tpurtell/ds41rt, squarrier/…-AFD | 2026-09/10 | need RoCE RDMA and DGX Spark/Blackwell ranks; concept noted in §2.11 |
| local-inference-lab/rtx6kpro, DGX-Spark and RTX-PRO recipes (glm53 in:name search, ~40 repos) | 2026-08…10 | model resident in 192–512 GB of Blackwell/GB10 memory; sm_120/121 kernels (b12x, FP4) |
| hyd998877/vllm_170hx_glm53-flash-exl3-optimized, Flun/glm53-flash-cmp170hx-exl3, liumorrisclaw/cmp170hx-glm53-lab | 2026-09/10 | 64 GB-unlocked CMP 170HX, model resident; only useful as "no-P2P PP" evidence |
| konradish/local-llm-benchmarks | `cc15dd6` 2026-10-03 | 4080 16 GB, small models; Qwen3.8-FN only on a 2080 via Strata (39.2 tok/s at 64K, IQ3) |
| jiafeimao1990/qwen-flash-consumer-lab | `d99f311` 2026-09-19 | 2×5060 Ti 16 GB, Windows, ExLlamaV3 1.5.0 patch; 96K max |
| sybil-solutions/qwen38-flash-next-b70-offload | 2026-10-06 | Intel Arc B70 / SYCL |
| Awesome-LLM-Inference, awesome-moe-inference (two lists) | raw HEAD 2026-10-10 | no 2026 MoE-offload / consumer-GPU engine entries |
| Papers: ATSInfer (arXiv 2607.10183), OSDI'26 "Cloud-Grade SLOs for Local MoE" (2606.10493), FluxMoE (2604.02715), Speculating Experts (2603.19289), DALI (2602.03495) | abstracts only | no code repository found by repo search; numbers are for DeepSeek-V3 / small models; R3 owns this area |
| x.com "I failed to make GLM-5.3 Flash usable on 4x RTX 3090" | — | still not loadable; snippet: ~11 tok/s at 32K, ~5 at 64–80K (row G11) |
| Reddit "4×RTX 4080 Super 32GB, no NVLink, gpt-oss-120b" (2026-09) | — | snippet only, no numbers; the one known public mention of our exact GPU set |

---

## 6. Corrections and updates to earlier research files

| file / claim | update | label |
|---|---|---|
| R2 §1 item 4, §7 item 6: GLM-5.3-Flash "will most likely not start on sm_89 in vLLM" | True for upstream `ed7ae70`. **Not true for community forks**: yhfgyyf has an SM89 Triton indexer branch (`sparse_attn_indexer.py:61-66`); vllm-backport ships `-sm89` images with GLM-5.3-Flash support; Morrowmake runs it on sm_80 | VERIFIED (code) / REPORTED (runs) |
| R2 §2.1 `0xSero/glm53-flash-offload` | now `sybil-solutions/glm-flash-lite`, same HEAD `6769b27`; NVMe tier added; sibling `dsv41-flash-offload` | VERIFIED (ls-remote) |
| R2 §1 item 7, R1: Qwen3.8-FN "46–104 tok/s on comparable GPUs" | 4×3090 vLLM W4A16 + MTP: 155.5 at 261K (Q1); 2×3090 + RAM: 103 at 260K (Q3) | REPORTED |
| R2 FreeToken "one GPU, no speculation" | still true upstream at `ee9789f` (2026-10-10); the rdna3 fork adds TP, host all-reduce and MTP | VERIFIED |
| R2 §4 "any MiMo-V2.6-Flash numbers on consumer GPUs — not found" | still none on consumer GPUs; first offload numbers on one RTX PRO 6000 (M1) | REPORTED |
| R4 §4.2/4.5 P2P | add small-BAR1 Ada fix (580.173.02) and the Proxmox-guest route with measured A/B (§2.9) | REPORTED |

---

## 7. What I could not verify

- Contents of GitHub issues (club-3090 #1543/#1615, yhfgyyf #74, llama.cpp #26563 `--expert-hot-s`): MCP `issue_read`
  is refused for these repos; only snippets.
- Any non-GitHub page (HF model cards, Discord, x.com, note.com, helix.ml, bearblog): DNS/proxy refusal. The
  `wtdcode/GLM-5.3-Flash-AWQ-W4A16` size (~170–180 GB) is a snippet-based estimate.
- Whether llama.cpp mainline `79e2e74` already contains GLM53-BLACKWELL's `can_reuse`/`ssm_a` fixes.
- Whether ExLlamaV3's CPU expert kernel reaches high bandwidth with AVX2 only on Zen 3 (anoane measured AVX-512 on Zen 5).
- Whether halt95's and DominikBucko's vLLM trees build for sm_89 (no build run; >20 min).
- Exact quant mix of `Intel/GLM-5.3-Flash-W4A16-AutoRound` (no HF access).

---

## 8. Sources

Repositories (all `git clone --depth 1` on 2026-10-10 unless "ls-remote"; SHA and commit date from `git log -1`):

| repo | SHA | commit date | licence |
|---|---|---|---|
| https://github.com/halt95/qwen38-flash-next-3090s | 28cbb7293f134c244b443fb43bbf6e74ff4d8e24 | 2026-10-06 | Apache-2.0 (+ Qwen community licence file) |
| https://github.com/wtdcode/vllm-backport | 29e66dad461b11caae6039f510eaa7913f9dfb19 | 2026-09-27 | Apache-2.0 |
| https://github.com/yhfgyyf/vllm-deepseek-v4-sm89 | 3f3bbaea916707c851ca027b87b97fb6605998b8 | 2026-09-16 | Apache-2.0 |
| https://github.com/DominikBucko/qwen38-flash-next-2x3090 | 939905b0dab8c6eb3852ff35cbbd42d098af2451 | 2026-10-02 | Apache-2.0 |
| https://github.com/anoane/glm53-flash-exl3-recipe | 116e6a2e0ada95b5f3b592581ac0cd1ff2788adf | 2026-09-01 | none found |
| https://github.com/anoane/02-glm53-flash-nvfp4-freetoken | 1399d7650a0d8532a74a6dd93a48956ac0ac0d4c | 2026-09-22 | MIT |
| https://github.com/anoane/exllamav3-anemone (ls-remote) | c6d2e3eea40dcd15947f55974bde3555a550a157 | — | UNKNOWN |
| https://github.com/Cedriceuh/freetoken-rdna3 | a4469ae01d60180df72b427d88358a04b44661e5 | 2026-10-10 | Apache-2.0 |
| https://github.com/FlashML-org/FreeToken | ee9789ff6cdcc2e9c5e9467aba630d38cd230fa6 | 2026-10-10 | Apache-2.0 |
| https://github.com/sybil-solutions/glm-flash-lite | 6769b27652a18487758caac45a2b6999dc0a4184 | 2026-10-09 | MIT |
| https://github.com/sybil-solutions/dsv41-flash-offload | d890a1c897f1c9d6d981473ec106224aeb9cbb4d | 2026-10-07 | MIT |
| https://github.com/SSHdotCodes/mimo-v2.6-flash-pro6000 | 464bf2676aa9bd106cc9833684c0d993e825c005 | 2026-10-06 | MIT |
| https://github.com/xiaoyanzi191/rtx4090-small-bar-p2p | dbd0cdf15c76548626ab224dbd9e34097d866ea3 | 2026-09-16 | MIT |
| https://github.com/noonghunna/club-3090 | c8b6a737988640bd0985e7d8ab6d0571f81a49fc | 2026-10-10 | Apache-2.0 |
| https://github.com/dg1kjd/sglang-sxm2 | 73abfe1dd7e7aae7a62b222a201e48f532fb298e | 2026-10-07 | Apache-2.0 |
| https://github.com/abtraore/GLM53-BLACKWELL | 64ef4e680349fd2dd14d6e471fe0d8c8c3f7b569 | 2026-09-04 | none found |
| https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe | cb3311eb980ac67e890c185f98dddda57e2abd48 | 2026-10-06 | MIT |
| https://github.com/Morrowmake/vllm-cmp170hx (ls-remote) | ab60b723ada254a442a4ba5ff27bf837aa27ef83 | — | UNKNOWN |
| https://github.com/zebgop-ops/glm53flash-pp | 1a14495ec1a7460e560181e8e11dd9af581b99bc | 2026-09-12 | Apache-2.0 |
| https://github.com/kaka86mm/glm53-flash-170hx | 218b47ff321b40875ac9c285a39f41425c6f0fd7 | 2026-09-19 | Apache-2.0 |
| https://github.com/Flun/glm53-flash-cmp170hx-exl3 | 5933112fdb858c960dfa1a68f507c3a98a9f1201 | 2026-10-07 | MIT |
| https://github.com/pctablet505/glm53-flash-single-gpu | 865edb6ec46f9c7cd72ee2ff137355a80b783f8d | 2026-09-02 | MIT (docs) |
| https://github.com/loktar00/glm53-flash-llamacpp-3090-recipe | 9d2ce467bdb885397dbd2c79bbf9abd40fb34faf | 2026-08-31 | not checked |
| https://github.com/neron82/helios-glm53flash | 700eebe89f8164f7a25321f88ad6bfe24cce294a | 2026-10-07 | MIT |
| https://github.com/hughmadden/glm53f-afd | 983f7f197755012b59a1b78285f4e2bab7d15b1f | 2026-10-06 | MIT |
| https://github.com/tpurtell/ds41rt | 3067d0684a0b572d88a17d38ca3f7d10599619b4 | 2026-09-25 | MIT |
| https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks | dc49f5f180a9f91f3c45d1e225499dd1437b058c | 2026-10-07 | not checked |
| https://github.com/local-inference-lab/rtx6kpro | d24d7ce6b4eee06552d2d4b096a6418a4666c51e | 2026-10-10 | not checked |
| https://github.com/JustVugg/colibri | bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850 | 2026-10-06 | Apache-2.0 |
| https://github.com/ashhart/TensorFold | 9041e20a423ef23cf060fa034d90ae5ffd86d768 | 2026-10-10 | Apache-2.0 |
| https://github.com/BFinn/flashrt | 90e7a558fd05b1e8890c8bb473d87de524095d19 | 2026-10-03 | Apache-2.0 |
| https://github.com/optim-enterprises-bv/titan-engine | 12bda9cf1bb5fc3fd62d64310082a1b54794972c | 2026-10-04 | MIT |
| https://github.com/yairpatch/flyweight | a65e4a4a3e5434b7c5a9d3d5c615e37d2b68fc0e | 2026-10-10 | Apache-2.0 |
| https://github.com/vcruz305/vllm-exl3 | 223e246ff8da552d2ace976c4bcbed804a8ec04c | 2026-09-25 | AGPL-3.0-only |
| https://github.com/davetha/vllm-expert-cache | 813b97ffc8c7646dc7a790a6ef956efcd45f9ece | 2026-09-08 | Apache-2.0 |
| https://github.com/Waltarb/nvmoe | 344c5ec77cf4c9b7e62b2c2829baee539d548330 | 2026-09-24 | MIT (README badge) |
| https://github.com/UnsignedChad/windows-freetoken-mtp | 4caf402d1e1e0d95654067c0b197adc8af93436c | 2026-08-23 | Apache-2.0 |
| https://github.com/yalun753/moe-l2 | 3b17ede0001d51dcf8ca1d8f3b0f74fbffb740c6 | 2026-10-01 | not checked |
| https://github.com/ek15072809/Swap-MoE | 50459bb422a01c0ba2e1da08b4d932f652f3e413 | 2026-08-29 | MIT |
| https://github.com/ap03906101/moe-hotcache | beda4496714f03e74bded372402cba5224e29baf | 2026-09-02 | MIT |
| https://github.com/biggiehorizon/llamacppMoEMaster | d7df49d1aa33f9c515145aed58f760fbcc1ae5cb | 2026-08-14 | MIT |
| https://github.com/bdrazn/blobpager | d7c9297ff054a3b54827c8f5eb2f77d01d8126a1 | 2026-09-21 | MIT |
| https://github.com/konradish/local-llm-benchmarks | cc15dd6a04ea50d4d281d40948f6a3e56da4edbb | 2026-10-03 | MIT |
| https://github.com/jiafeimao1990/qwen-flash-consumer-lab | d99f3115a6808e3c070aabd20a66943113f58690 | 2026-09-19 | MIT |
| ls-remote only: Cerynitius/freetoken-ox-boost `fdb99c6` (unchanged since R2), lukolszewski/llama.cpp-multigpu `ee5d54a` (unchanged), aikitoria/open-gpu-kernel-modules `c8bff54`, DominikBucko/qwen38-flash-next-3090 `10b24c2`, zebgop-ops/qwen38-flashnext-pp `2f551bb`, hyd998877/vllm_170hx_glm53-flash-exl3-optimized `5bd3f5f`, local-inference-lab… | | | |

Code search (GitHub MCP): `"NOT CORRECT" repo:Niko1221/Strata` → `bench/results/2026-09-27-cache-parity/README.md`.

Raw files: https://raw.githubusercontent.com/xlite-dev/Awesome-LLM-Inference/main/README.md ,
https://raw.githubusercontent.com/MoE-Inf/awesome-moe-inference/main/README.md ,
https://raw.githubusercontent.com/JustQJ/awesome-moe-inference/main/README.md ,
https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/glm-5.3-flash.md .

Web pages known only from search snippets (not loadable here): https://x.com/superalesha/article/2096976602127044995 ;
https://helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1 ; https://note.com/cute_agapan9087/n/na5ba75fcf758 ;
https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/38 ; https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16 ;
https://github.com/noonghunna/club-3090/issues/1615 ; https://github.com/ykk648/deepseek-v4.1-flash-8x4090 ;
https://openclawdc.com/blog/can-i-run-mimo-v2-6-flash-locally/ ; https://www.contextstudios.ai/local-ai/vcruz305--mimo-v26-flash-rl-exl3-rtx6000 ;
https://reddit.sentinel-team.org/posts/1w1fe3g/snapshots/2026-09-04T21%3A20%3A12.3645Z ;
https://am17an.bearblog.dev/prefetching-weights-in-llamacpp/ ; https://github.com/ggml-org/llama.cpp/pull/26563 ;
https://arxiv.org/abs/2607.10183 ; https://arxiv.org/abs/2606.10493 ; https://arxiv.org/html/2604.02715v1 ;
https://arxiv.org/abs/2603.19289 ; https://www.mindstudio.ai/blog/qwen-3-8-flash-next-local-agent-setup/ ;
https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870 .
