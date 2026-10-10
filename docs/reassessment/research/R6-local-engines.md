# R6. Local single-node engines: llama.cpp, ik_llama.cpp, ExLlamaV3 in depth, plus the wider landscape

Role: engine landscape researcher (local single-node engines). Date of all access: 2026-10-10.
Scope: what changed since the CI pins in `tools/engines/pins.json` (llama.cpp `79e2e74`, ik_llama.cpp `5194a9e`,
ExLlamaV3 `151539c`). Also the local engines that R2 (2026-10-09) did not cover or covered from READMEs only.
The names the owner asked about are checked in §8.
Builds on R2 (`R2-engines.md`) and its corrections (`README.md` in this folder). R2 facts are not repeated unless they changed.

**Labels** (BRIEF rule 3):
- VERIFIED: I read the code at the recorded SHA (`repo@sha:path:line` or function), or I ran the command.
- REPORTED: a README, PR, issue or commit message says so. The hardware is given.
- HYPOTHESIS: my inference, with its basis.
- UNKNOWN.
- DISPROVED.

Repo labels (AGENTS.md): VERIFIED = CONFIRMED; REPORTED and HYPOTHESIS = PROVISIONAL.
No GPU exists here. Nothing below was run on a GPU, and no number in this report was measured by me.

---

## 0. Summary for the lead (top findings)

1. **llama.cpp moved only 12 commits since the pin `79e2e74`** (HEAD `abee0c8`, 2026-10-10). None of them touches the MoE cache, `-sm tensor`, the meta backend or the CUDA all-reduce.
   The important news is in open PRs (VERIFIED by fetching their heads):
   - **#30277** (ggerganov, draft, 2026-10-10) removes `GLM5_NEXT` from the `-sm tensor` deny list. But it **mirrors the KDA and MLA attention on every GPU** and splits only the FFN and output.
   - **#29963** keeps pipeline parallelism on with host experts. The author reports ×1.5–2 prefill and unchanged generation.
   - **#27861** (draft, 2026-08-28) is the **exact hybrid we want**: cached experts are computed on the GPU, misses on the CPU, in the same layer, with throttled asynchronous inserts. It covers decode with `n_tokens == 1` only and has no maintainer review.
2. **The merged `--moe-cache-mib` is a synchronous upload cache, not a CPU/GPU hybrid** (VERIFIED, `src/llama-moe-cache.cpp@abee0c8`, identical at the pin).
   - Per MoE layer, the scheduler copy callback reads the router ids back to the host and calls `ggml_backend_synchronize` (`:549-550`). It then runs an LRU plan on the CPU and uploads misses with `set_async` on the same compute stream (`:568`).
   - That is about 42 host round-trips per token for GLM-5.3-Flash. Misses are never computed on the CPU. Batches above 32 tokens bypass the cache.
   - With the default `mmap`, host experts sit in pageable memory: the loader refuses the pinned host buffer under mmap (`src/llama-model-loader.cpp:1269-1277`).
   - The scheduler turns pipeline parallelism off when the cache is enabled (`src/llama-context.cpp:434-440`).
3. **llama.cpp `-sm tensor` on 4 GPUs without P2P uses NCCL or a generic fallback; the host-staged all-reduce exists only for exactly 2 GPUs** (VERIFIED).
   - The internal all-reduce stages through pinned host memory and needs no P2P (`ggml/src/ggml-cuda/allreduce.cu:13-37`). It refuses `n_devices != 2` (`:402-406`).
   - With 4 GPUs the chain is NCCL (Linux default) → meta-backend butterfly over `cudaMemcpyPeerAsync`.
   - `GLM5_NEXT` is still excluded at HEAD (`src/llama-arch.cpp:1240`). `QWEN4EXP`, `MIMO2` and `DEEPSEEK4` are not excluded (`10f340d` re-enabled qwen4exp).
   - The MoE cache throws under `-sm tensor` (`llama-moe-cache.cpp:234-236`). CPU-resident experts do work with `-sm tensor` (`e60eff9`).
4. **ik_llama.cpp: 1 commit since the pin** (HEAD `89bba35`).
   - `-sm graph` still excludes glm5next, qwen4exp and deepseek4/41 (`src/llama.cpp:4216-4250`). MiMo2 is capped at 4 GPUs (`:4575-4581`), which is exactly our count.
   - Open **PR #2644** (2026-10-10) ports the mainline `--moe-cache-mib`. The author reports +19–24% on Qwen3.8-FN with 1–2 RTX 3090.
5. **ExLlamaV3 is the most complete design for our problem in an existing engine. `master` = pin `151539c`; `dev` = `d334b03` is 40+ commits ahead.**
   - **Native TP backend with no P2P and no NCCL** (VERIFIED, `model_tp_backend.py:221-250,458-475`, `exllamav3_ext/parallel/all_reduce_cpu.cu`). GPUs write zero-copy into mapped pinned shared memory, a CPU helper process sums with AVX2/AVX-512, and GPUs read back. The decode path is CUDA-graph capturable.
   - Reported: 6×RTX 3090 Gen3 decodes Qwen3.8-Flash-Next 6.05 bpw at 76 tok/s with native TP (commit `8114c49`).
   - **`-mcs/--moe_cpu_split`**: a per-layer expert split. Hot experts sit on the GPU, the tail on a CPU worker process, and both run concurrently in the same layer.
     The handoff uses `cuStreamWaitValue32` memops, so the host never synchronizes. Hot/cold placement is dynamic and swaps between generations.
   - **But the split works only in layer-split mode, not with TP** (`exllamav3/model_init.py:214-217`).
   - Its CPU kernels decode EXL3 trellis. Reported on AVX2: 15.6–21 GB/s on a 20-core Xeon (commit `94c05ba`).
   - GLM-5.3-Flash ≥4 bpw (~160 GB) does not fit 128 GB VRAM. So for GLM, ExLlamaV3 means layer split plus a CPU tail (HYPOTHESIS, arithmetic in §4.6).
6. **Two engines R2 missed, both with GLM-5.3-Flash on CUDA:**
   - **fastllm** (ztxz16, Apache-2.0, HEAD `a2ff521`, 2026-10-10).
     - GLM-5.3-Flash KDA with MTP, Qwen4-Exp, DeepSeek-V4/V4.1 with Engram.
     - Multi-GPU TP with an NCCL fallback; CPU/NUMA/disk expert tiers.
     - A **graph-capturable CUDA LRU expert cache whose misses a CUDA kernel pulls from mapped pinned host memory**. An "eager hybrid" leaves a selected expert on the CPU when its estimated CPU completion beats a cache refill (`src/devices/cuda/moe/fastllm-moe-cache.cu:1-9`, VERIFIED).
     - GLM-5.3 throughput numbers are not published ("No publishable complete GLM-5 / GLM-5.3-Flash throughput table", `docs/glm5_en.md`).
   - **DwarfStar = antirez/ds4** (MIT, 23.7k★, HEAD `fc80bd6`).
     - Model-specific: DeepSeek-V4/V4.1 Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next. Ada sm_89 is an explicit target.
     - Peer copies are validated at init with a pinned-host bounce fallback (`ds4_cuda.cu:2799-2900`).
     - But on NVIDIA its GLM path needs full VRAM residency: GLM streaming is enabled only for Metal and ROCm (`ds4.c:455-462`). Its own GLM Q4 file is 178 GiB.
7. **None of the "specialized" GLM/MiMo engines found fits 4×Ada with ≥4 bit and expert offload.** helios-glm53flash is EXL3 2.05 bpw on 2×3090. glm53f-afd, mimo26f-afd and spark-engine target DGX Spark/RDMA. sglang-sxm2 targets 8×V100 with full residency. warp is a CPU/Metal SSD engine. All are REPORTED/VERIFIED in §7.
8. **Names asked about:**
   - DwarfStar = antirez/ds4 (found).
   - Pulsar = giannisanni/pulsar (found, last commit 2026-09-01).
   - Colibrì = JustVugg/colibri (found).
   - Splash = incoai/splash (found, **Apple-silicon only**).
   - llamAmpere = JakeATX/llamAmpere (found; sm_86 llama.cpp fork for dense Qwen3.8-27B).
   - gufo = gufo-org/gufo (found, **Strix Halo gfx1151 only**).
   - machx-inference-engine = Red-Weasel/machx-inference-engine (found, **Intel Arc SYCL only**).
   - None is NOT FOUND.

**What I could not verify:**
- Any number on 4080/Ada with offload for GLM-5.3-Flash ≥4 bit. None was found.
- Whether NCCL initializes and how fast its SHM transport is inside a Proxmox VM without P2P.
- The EXL3 CPU-kernel speed on Zen 3.
- fastllm's GLM-5.3 speed.
- PR #30277's performance (none published).

---

## 1. Repositories read (SHA, date)

All VERIFIED by `git log -1 --format='%H %ci'` in my clones (scratch dir `research/r6`) or `git ls-remote --symref`, 2026-10-10.

| repo | HEAD | commit date | vs pin |
|---|---|---|---|
| ggml-org/llama.cpp (master) | `abee0c8476b585b0d081d580b5ce21d0addc22c4` | 2026-10-10 20:00 +0200 | pin `79e2e74` + 12 commits (`git log --oneline 79e2e74..HEAD`, list in §2.1) |
| ikawrakow/ik_llama.cpp (main) | `89bba35c8841618502b2be8273fb5265b405aca1` | 2026-10-10 08:32 +0200 | pin `5194a9e` + 1 commit (`89bba35 server: reset slot.sparams …`) |
| turboderp-org/exllamav3 (master) | `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` | 2026-10-07 18:47 +0200 | **= pin** |
| turboderp-org/exllamav3 (dev) | `d334b036b351f9952ff0112264d2ef9c80b5ba58` | 2026-10-10 16:41 +0200 | ahead of master (§4.7) |
| antirez/ds4 (DwarfStar) | `fc80bd695da76ee14bcb86820bf900a5cdfda806` | 2026-10-07 21:43 +0200 | new |
| ztxz16/fastllm | `a2ff521f61820e1972a84e494cda58c0c2572c5f` | 2026-10-10 11:12 +0800 | new |
| lukolszewski/llama.cpp-multigpu (multigpu) | `ee5d54afd20d94aaea3ed097837eb079b96aa674` | 2026-10-07 18:50 +0200 | in R2 as a source only |
| EricLBuehler/mistral.rs | `3f2515e9b5adc2ac44949c128c50a13b15721294` | 2026-10-01 00:18 -0400 | same as R2 |
| aphrodite-engine/aphrodite-engine | `82b0cd8c2c1f24ad897871f2aac140d37bca1642` | 2026-09-11 22:24 +0200 | new |
| InternLM/lmdeploy | `822dddb8b3bc895eaf6cffb45a5514768e7ecb84` | 2026-10-09 11:58 +0800 | same as R2 |
| mlc-ai/mlc-llm | `8978ea9aec626d64096aaa92fbc89bda2e3ea802` | 2026-10-07 04:57 +0530 | new |
| mozilla-ai/llamafile (= Mozilla-Ocho/llamafile) | `a3ccf098ed9c7c77687f5430b0143156d3bcc45a` | 2026-10-08 20:53 +0100 | new |
| JakeATX/llamAmpere | `54cef46c266455a8c5e625527f40d74350ce1e74` | 2026-10-10 04:12 -0500 | new |
| incoai/splash | `8a2b42069923d483151372b615f6182c72c2b6cd` | 2026-10-09 23:34 -0700 | new |
| gufo-org/gufo | `665fc182445dc2685f2a9b2b4314aa7d92a858c2` | 2026-10-10 19:07 +0200 | new |
| Red-Weasel/machx-inference-engine | `eab562917b7c3d08bd5909ec659e7d4241e1185d` | 2026-10-07 03:45 -0500 | new |
| JustVugg/colibri | `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850` | 2026-10-06 22:02 +0200 | new |
| uv-genai/colibri (copy) | `5e7d6ecb85b8b815fa9be1c6d5b5b6f6bbff6ad9` | 2026-07-14 19:46 +0800 | stale copy |
| s-b-repo/peregrine | `869a5edd3d13848c42bd1bb728139c85c8b12961` | 2026-08-23 20:15 +0200 | new |
| giannisanni/pulsar | `18f990094e1091ee2ef3078f20be418f8c162f03` | 2026-09-01 14:58 -0300 | unchanged since R2 |
| sqliteai/warp | `09fcff352ca55223b08ee222d15054b90546c6a9` | 2026-09-18 09:20 +0200 | new |
| dg1kjd/sglang-sxm2 | `73abfe1dd7e7aae7a62b222a201e48f532fb298e` | 2026-10-07 19:51 +0000 | new |
| Indras-Mirror/Strata-GLM | `152c49f4d37c55d62179104a51816346203aff17` | 2026-10-10 22:26 +1100 | new |
| neron82/helios-glm53flash | `700eebe89f8164f7a25321f88ad6bfe24cce294a` | 2026-10-07 10:28 +0200 | new |
| 0xSero/glm53-flash-offload | `6769b27652a18487758caac45a2b6999dc0a4184` | 2026-10-09 09:03 -0400 | = R2 |

PR heads fetched with `git fetch origin pull/N/head` (VERIFIED SHAs):

| PR | head SHA | date |
|---|---|---|
| llama.cpp #27861 | `bccbacdb8945680f1cfc7e6bffd1e59014705750` | 2026-08-28 |
| llama.cpp #30277 | `bcb110179a1513c983c8f90f50fa3623951b76c6` | 2026-10-10 |
| llama.cpp #29963 | `f81e182b340c452adb4725cd6a44c5d6734baea1` | 2026-10-09 |
| llama.cpp #30205 | `8f7f1ae0541dc7cedf505b2adb9b8f9bc514481c` | 2026-10-09 |
| llama.cpp #30059 | `03550d187baf5d9f020880f127a0699b823d4395` | 2026-10-06 |
| llama.cpp #26414 | `f794cbdd73acc083063bcd8bc9782386f221f09a` | 2026-09-24 |
| llama.cpp #28696 | `2076412598e99127ea2554467b2b7530856895d3` | 2026-09-22 |

---

## 2. llama.cpp (deep)

### 2.1 Delta since the pin `79e2e74` → `abee0c8` (VERIFIED)

`git log --oneline 79e2e74..HEAD` lists 12 commits. The ones touching our path:
- `781dbc5` "spec: properly handle mtmd input for mtp"
- `1bb2b9f` "CUDA/HIP: fix race in flash_attn_ext_f16_process_tile"
- `1623d8c` "cuda: always use MMVQ for MUL_MAT_ID on sm_60" (not our arch)

`git diff --stat 79e2e74 HEAD` shows no change in any of these files: `src/llama-moe-cache.cpp`, `src/llama-arch.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/allreduce.cu`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `src/llama-context.cpp`, `ggml/src/ggml-cuda/fattn-common.cuh`.
Small changes landed in `src/llama-graph.cpp` (18 lines), `src/models/glm5-next.cpp` (13) and `common/arg.cpp` (2).
**So the pin is still current for engine behaviour.** All line numbers below are at `abee0c8`.

The last four weeks for our models (VERIFIED, `git log` after `--shallow-since=2026-09-10`):

| commit | date | change |
|---|---|---|
| `649dcb1` | 09-30 | GLM-5.3-Flash (GLM5-Next) support |
| `b9acf13` | 10-07 | GLM5Next MTP |
| `3109914` | 10-06 | "remove the gather path of the glm5-next sparse attention" |
| `05af0d2` | 09-30 | glm5-next: unique scatter rows for dead indexer slots |
| `c061df1` | 10-01 | Qwen4Exp MTP |
| `10f340d` | 10-01 | re-enable `-sm tensor` for qwen4exp |
| `3cf0325` | 09-20 | sparse FA for qwen4 |
| `bfd73a8` | 09-22 | convert: MiMo-V2.6 |
| `22bdcc4` | 09-30 | mimo: DFlash |
| `d6cf9ac` | 10-07 | MoE cache (#29887) |
| `c811cb8` | 10-08 | MoE cache over multiple GPUs (#30112) |
| `033df86` | 10-08 | server: preserve context checkpoints across slot save/restore |
| `e60eff9` | 10-09 | meta: handle host views ("fixes the assert hit by KV cache views when using --split-mode tensor with partial offload") |
| `5503b04` | 10-01 | meta: clear inactive AllReduce shards with FILL |
| `38a5b42` | 09-15 | HIP: enable AllReduce for ROCm |
| `2f3fd02` | 09-16 | "Enable CUDA graph for MTP draft" |

DeepSeek-V4.1 is still only an open PR, **#28696** (head 2026-09-22). Mainline has no `deepseek41` arch.

### 2.2 `--moe-cache-mib`: the algorithm as implemented (VERIFIED)

Files: `src/llama-moe-cache.{h,cpp}`, `src/llama-graph.cpp:2422-2458` (`build_moe_cache_slots`), `src/llama-graph.cpp:1586-1600` (`build_lora_mm_id`), `src/llama-context.cpp:2641-2718` (`sched_copy_experts`).

```
setup (llama_moe_cache::impl ctor, llama-moe-cache.cpp:221-452)
  throw if split_mode == TENSOR                                   # :234-236
  for each layer whose expert tensors are ALL host weights (-ot/-ncmoe/-cmoe):   # :241-264
      group by (device that owns the layer, identical expert layout)
  budget per device = size * tensor_split share (or free VRAM share)            # :280-308
  per group: n_slots = max s with sum_t pad(nb[2]*(s+1)) <= budget*host_bytes_g/host_bytes_dev
             if n_slots < n_expert_used: group not cached                       # :313-329
  banks: one device tensor [ne0, ne1, n_slots+1] per expert tensor (gate/up/down)
  slot_map: host I32 [n_expert] per layer, -1 = not cached

graph build (build_moe_cache_slots)
  if n_tokens > 32 (max_batch, :196) or n_tokens*k > n_slots: no cache for this ubatch  # :462-473
  slots = get_rows(slot_map /*host weight*/, selected_experts)   -> forces a new scheduler split
  mul_mat_id(bank_view, x, slots) on the GPU in place of mul_mat_id(host_experts, x, ids)

scheduler copy callback (copy(), :515-583), once per cached layer per ubatch:
  ids  <- tensor_get_async(selected_experts); ggml_backend_synchronize(backend)   # :549-550  HOST STALL
  lru.plan(il, ids): touch hits; for sorted misses take LRU head, evict, assign   # :74-119
  for each miss run (consecutive expert & slot): tensor_set_async(bank, host+e*size, slot*size)  # :568
  tensor_set_async(slot_map_dev, slot_map_host)                                   # :580
  (the GPU split then runs mul_mat_id over the bank)
large batches (prefill, op offload): copy_experts() copies cached experts D2D from banks,
  uploads the rest from host, read-only (no LRU update)                           # :480-513, llama-context.cpp:2697-2716
```

Properties for our server (VERIFIED unless labelled):
- **No CPU computation of misses.** With the cache, every routed expert of a cached layer runs on the GPU.
  CPU-only compute happens only for layers whose group is not cached, or for batches the cache rejects (`get_slot_map` returns nullptr). Those fall back to the normal `-ot` path, where the CPU computes small batches.
- **Synchronous per layer.**
  - The host blocks on `ggml_backend_synchronize` to read the router output (`:550`), then plans the LRU and enqueues uploads on the same stream before the expert matmul.
  - No prefetch, no second copy stream, no overlap of layer *l* misses with layer *l−1* compute.
  - GLM-5.3-Flash has 42 MoE layers (`models/glm-5.3-flash.json`): about 42 host round-trips per decoded token. HYPOTHESIS: ≥10–30 µs each, plus GPU idle while the host plans; the 4080 cannot hide it.
- **Pageable source by default.**
  - Upload sources are the host expert tensors (`w->data`). With `mmap` on (the default), the loader replaces the pinned host buffer type with the plain CPU buffer type (`src/llama-model-loader.cpp:1269-1277`).
  - So `cudaMemcpyAsync` copies from pageable memory. HYPOTHESIS: staged and slower; R2/PR reviewers say the same (#29887 thread, REPORTED).
  - `--no-mmap` lets the CPU weights use the first GPU's host (pinned) buffer type (`src/llama-model.cpp:1121-1129`). Pinning ~100+ GB of experts on a 128 GB host is itself a risk (HYPOTHESIS).
- **Multi-GPU (#30112):**
  - Each GPU caches only the layers assigned to it by `-sm layer` (`:247-251`). The budget is split like `--tensor-split` (`:280-301`).
  - Pipeline parallelism is disabled when the cache is on (`src/llama-context.cpp:434-440`; it is also disabled by any `-ot`).
  - One decoded token therefore walks GPU0 → GPU1 → GPU2 → GPU3 sequentially, with 42 host syncs in total.
- **Policy:** plain LRU per group (one LRU per device and layout, shared by its layers). No frequency admission.
  A reviewer reported LRU underperforming LFU-style admission (#29887 thread, REPORTED). Our `tools/g0/cachesim.py` already models this policy.
- **Speculation:**
  - The cache serves the target only: the draft gets `moe_cache_size = 0` (`common/speculative.cpp:2569`).
  - MTP verification batches (≤32 tokens) use it.
  - Reviewers warn that n-gram/MTP drafts multiply expert transfers (REPORTED, #29887).

Reported effect (PR #29887 thread, single GPU, Qwen3.8-Flash-Next Q4_0):

| GPU | baseline tg | with cache | hit rate |
|---|---|---|---|
| RTX 4090 | 25.0 | 40.7 tok/s | 77% |
| RTX 5090 | 30.8 | 67.8 tok/s | 89% |

On **2×3090** one user measured the cache **25–34% slower** than without it. That was before or around #30112; REPORTED, conditions incomplete.
On R9700: 11.0 → 27.6 tok/s at 67% of experts cached.

### 2.3 `-sm tensor` (meta backend) and all-reduce without P2P (VERIFIED)

**How it runs.**
- `ggml/src/ggml-backend-meta.cpp` splits each weight across devices, builds per-device subgraphs, and all-reduces at the end of each subgraph (`:2464-2497`).
- All-reduce goes to the backend's `ggml_backend_comm_allreduce_tensor`, falling back to a butterfly of `ggml_backend_tensor_copy_async` plus `ADD` (`:2350-2459`).
- The butterfly copy is `cudaMemcpyPeerAsync` (`ggml-cuda.cu:2571-2620`). Without peer access the CUDA driver stages it through host memory (HYPOTHESIS, standard driver behaviour; not measured).

**CUDA comm selection** (`ggml-cuda.cu:1155-1243`): `GGML_CUDA_ALLREDUCE` unset → Linux tries NCCL → `internal` → `none` (butterfly).
- NCCL path: FP32 for small tensors, BF16 above 262144 elements for ≥4 GPUs (`:998-1068`). The thresholds were tuned "on RTX 4090s connected via 16x PCIe 4.0" (comment `:1016`).
- **The internal host-staged all-reduce supports exactly 2 GPUs** (`allreduce.cu:400-406`).
  - Design (`allreduce.cu:13-60`): each GPU kernel writes its BF16 shard into mapped pinned host memory, spins on a host flag, then reads the peer shard. Large tensors use copy-engine D2H+H2D chunks.
  - It needs no P2P and no `hostNativeAtomicSupported`.
- **With 4×4080 the choice is NCCL (if it initializes in the VM) or the butterfly over driver-staged peer copies.**
  NCCL without P2P uses its SHM transport (HYPOTHESIS from NCCL design; UNKNOWN in a Proxmox VM).

**Model coverage at HEAD** (`src/llama-arch.cpp:1205-1243`, `llm_arch_supports_sm_tensor`):
- Excluded: `GLM5_NEXT`, `KIMI_K3`, `KIMI_LINEAR`, `BAILINGMOE3`, `DEEPSEEK2`, `DEEPSEEK32`, `GLM_DSA`, `MINIMAX_M3`, and others.
- **Not excluded:** `QWEN4EXP` (re-enabled in `10f340d`), `MIMO2`, `DEEPSEEK4`.
- Support for MIMO2/DEEPSEEK4 is "default true", **not tested by me**.

**Draft PR #30277 (GLM-5.3-Flash `-sm tensor`)**, VERIFIED from the commit message at `bcb1101` (`src/llama-arch.cpp` hunk removes GLM5_NEXT, KIMI_LINEAR, KIMI_K3, BAILINGMOE3):
> "the KDA attention and indexer tensors have no rule at all, so mirror the attention part … their MLA layers use an absorbed latent, so neither can be head-split. FFN and output still split".

Consequences (HYPOTHESIS from that design):
1. Attention, indexer and KDA are computed redundantly on all 4 GPUs. Only the FFN and output projections get the 4× bandwidth split.
2. The MLA latent cache and KDA state are **replicated** on every GPU. At 262K that is ~8 GB per GPU (`models/glm-5.3-flash.json` kv 30720 B/token, PROVISIONAL), not 2 GB.
3. The routed experts can stay host-resident (`e60eff9` fixes partial offload with `-sm tensor`) but cannot use the MoE cache.
4. No benchmarks are in the PR. Only `test-llama-archs` 517/517 at 1–4 devices (REPORTED).

**Other split facts:**
- `-sm tensor` forces flash attention (`src/llama-context.cpp:3918-3928`).
- Backend sampling is disabled under `-sm tensor` (`:1305-1308`).
- `--fit` refuses `-sm tensor` (`common/fit.cpp:183-184`).

### 2.4 `-ot` / `-ncmoe` / CPU path (VERIFIED)

- `-ncmoe N` adds buffer overrides for the first N layers' expert tensors (`common/arg.cpp:2770-2778`). The CPU backend then computes their `MUL_MAT_ID`.
- For batches ≥ `GGML_OP_OFFLOAD_MIN_BATCH` (default 32, `ggml-cuda.cu:6037`) the op is offloaded to the GPU.
  The scheduler then uploads **only the experts used** (`sched_copy_experts`, `llama-context.cpp:2651-2716`). It reads `ids` with a host sync and merges consecutive experts.
- CPU and GPU splits run in scheduler order. A CPU split blocks until its inputs are copied from the GPU. HYPOTHESIS from the `ggml_backend_sched` design: no intra-layer CPU∥GPU overlap. That is exactly the gap PR #27861 fills.

### 2.5 Speculation, KV, FA for MLA/DSA, CUDA graphs, checkpoints (VERIFIED)

- **Speculative types** (`common/common.h:175-187`): draft, EAGLE3, MTP, DFlash, DSpark, and 5 n-gram variants.
  - GLM5-Next MTP: `b9acf13`. Qwen4Exp MTP: `c061df1`. MiMo DFlash: `22bdcc4`.
  - CUDA graphs for the MTP draft: `2f3fd02`.
- **KV cache types** (`common/arg.cpp:304-314`): f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1.
  FP8 KV is only an open draft (#28392, REPORTED title).
- **GLM5-Next attention** (`src/models/glm5-next.cpp:1028-1088`):
  - Nope-only MLA with absorbed `wk_b`. The cache holds the latent only (`cpy_k(kv_cmpr)`).
  - The DSA selection is a scatter mask over all `n_kv` cells (`build_kpool_select`, `:889-1026`). Attention is `build_attn_mha(..., inp_kpool->n_sel, ...)`.
  - **CUDA FA honours the `n_kv_max` hint by compacting the mask into gathered indices and attending only `n_kv_max` keys** (`fattn-common.cuh:1101-1141`, `fattn.cu:14-149`). So DSA is sparse in the FA kernel; the gather path was removed in `3109914`.
  - Building and compacting the mask is still O(n_kv) per query token (HYPOTHESIS: small but growing with context, similar to what the lukolszewski fork fixed for qwen4exp; §6.1).
- **CUDA graphs** (`ggml-cuda.cu:2644-2672, 4639-4680`):
  - Captured per split, keyed by the first node. Disabled when `MUL_MAT_ID` needs a sync.
  - With the MoE cache every cached layer is its own split boundary with a host callback. So graphs cover small fragments only (HYPOTHESIS).
- **Hybrid-memory checkpoints in the server:** `-ctxcp/--ctx-checkpoints`, `-cms/--checkpoint-min-step`, `-cram/--cache-ram` (`common/arg.cpp:1700-1725`).
  `033df86` keeps checkpoints across slot save/restore.

### 2.6 Open PRs that matter (VERIFIED head SHAs; content REPORTED from PR pages or VERIFIED from diffs)

| PR | status | what | relevance |
|---|---|---|---|
| **#27861** "GPU-resident LRU cache for host-offloaded MoE expert weights" | draft since 2026-08-28; no maintainer review; commenters call it abandoned | **Hit on GPU, miss on CPU, same layer.** CPU `mul_mat_id` skips cached ids via `src[3]` table and zeroes their rows (`ggml-cpu.c` hunk). A second GPU `mul_mat_id` chain over cached slices; outputs summed (`llama-graph.cpp` hunk). Async throttled inserts by a worker thread; mapping published after the copy completes. `n_tokens == 1` and SiLU/SwiGLU only. | The exact Strata-style hybrid inside ggml. Author: 2×3090 Qwen3.8-FN 15.5 → 19–20 tok/s. Others: +15–40% (REPORTED). Needs rebase on `d6cf9ac`. |
| **#30277** "meta: enable tensor split for KDA hybrid models" | draft 2026-10-10, by ggerganov | GLM5_NEXT `-sm tensor` (§2.3) | TP of FFN/output only; attention mirrored |
| **#29963** "pipeline parallelism with MoE experts in host RAM" | ready for review 2026-10-09 | Each GPU uploads the experts for its own layers; consecutive ubatches overlap across devices | Author: prefill ×1.5–2 on 3×RTX PRO 4000 + EPYC 9274F, **generation unchanged**, and `--moe-cache-mib` *slower* in their tests (REPORTED) |
| **#28414** `--prefetch-experts-slots` | draft | Lookahead H2D upload of whole host expert tensors during prefill, on a second stream | Prefill only; author says not for multi-GPU; blocking review comments (REPORTED) |
| **#30205** comma-separated `--moe-cache-mib` per device | open | per-GPU budgets | convenience |
| **#26414** `--pin-hot-experts` | open | pin hottest experts in RAM (mlock) | low for us (RAM is the cold tier) |
| **#30059** "LLama-wackMall-fusion expert cache" | open | another expert store (`moe-store`) | not evaluated |
| **#28696** DeepSeek V4.1 | open since 09-10 | adds `deepseek41` | needed for DS-V4.1 on mainline |
| **#25051** vulkan TP all-reduce | open | — | not CUDA |

### 2.7 What limits single-stream decode on 4×4080 without P2P in llama.cpp (HYPOTHESIS, from the code above)

1. **`-sm layer`:**
   - One GPU is busy at a time. The dense part of GLM is read sequentially (R2 §5.0: ≈13.8 ms/token at Q8_0 dense, ≈72 tok/s ceiling even with free experts).
   - 3 activation hops per token through `cudaMemcpyPeerAsync`, staged through the host without P2P.
2. **Experts in RAM:**
   - Either CPU compute serialized with the GPU splits (`-ot`), or synchronous PCIe uploads per layer (`--moe-cache-mib`).
   - Never both at once, and never overlapped with the next layer.
3. **`-sm tensor`:**
   - ~2 all-reduces per layer through NCCL-SHM or the butterfly, since the internal no-P2P all-reduce is 2-GPU only.
   - For GLM, only after #30277, with replicated attention and KV. No MoE cache.
4. **MoE-cache host syncs (~42/token)** make CUDA graphs ineffective for the MoE part.

---

## 3. ik_llama.cpp (deep)

HEAD `89bba35` = pin `5194a9e` + 1 commit (server sampling reset). The 2026-09-14…10-06 history is in R2.

- **`-sm graph`** support list: `is_model_split_supported`, `src/llama.cpp:4216-4250` (VERIFIED).
  - Includes MIMO2, GLM_DSA, DEEPSEEK2, QWEN35MOE, MINIMAX_M2/M3. **Excludes GLM5NEXT, QWEN4EXP, DEEPSEEK4/41** (QWEN3NEXT is commented out).
  - Unsupported archs silently fall back to `layer` with a warning (`:4566-4572`).
  - MiMo2 is capped at 4 GPUs (`:4575-4581`).
  - KDA and DeltaNet have `SPLIT_MODE_GRAPH` handling for the state (`src/llama-kda.cpp:112`, `src/llama-delta-net.cpp:471`). HYPOTHESIS: groundwork exists, but the glm5next graph is not enabled.
- **Graph-mode all-reduce** (`ggml/src/ggml-cuda/reduce.cu`, VERIFIED):
  - NCCL only for selected cases (`:146-175`, author comment: NCCL underperformed).
  - Otherwise a ring of `cudaMemcpyPeerAsync` with reduction in f16/bf16 or **Q8_0** (`k_add` for `block_q8_0`, `:28-44`).
  - The fast small-token kernel path requires `ctx.p2p_enabled` (`:448`). Without P2P, small decode reductions go through the peer-copy ring.
  - README warns of gibberish with graph mode plus partial offload (`README.md:22`).
- **Expert offload** (VERIFIED, `common/common.cpp`):
  - `-cmoe`, `-ncmoe N`, `-ot`, `-muge` (`:2473`), `-ooae` offload-only-active-experts (default on, `-no-ooae` at `:2783`).
  - `-prexp` prefetch-experts into the page cache, `-dexp` defer experts, `-dple` keep PLE/Engram tables on file (help `:3563-3570`), `--numa`.
  - **No GPU expert cache at HEAD.** Open **PR #2644** (2026-10-10, SamuelOliveirads) ports mainline #29887 + #30112.
    REPORTED on TR PRO 5965WX + 2×3090, Qwen3.8-FN UD-IQ4_XS: 39.03 → 48.26 tok/s on 1 GPU, 37.01 → 44.17 on 2 GPUs `-sm layer`; graph mode +1.3% on Qwen3.6-35B.
- **KV types** (`common/common.cpp:4519-4550`): f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, q6_0, **q8_KV**, plus Hadamard-rotated K/V (`-khad/-vhad`, `:2477-2484`).
- **GLM5-Next graph** (`src/graphs/build_glm5next.cpp`, 689 lines): DSA top-k over k-pools; the flash path uses a sparse top-k mask (`:224-266`).
  MTP for glm5next is merged (`d741de5`). The "complete MTP" PR #2399 is still open (REPORTED title).
- **IQK AVX2 CPU kernels:** the README states AVX2 and CUDA (Turing+) are the "fully functional and performant" backends (`README.md:13`).
  No AVX2 tok/s number is in the README. Measured GLM numbers on EPYC hosts are in R2 §4, rows 3–6 (REPORTED).
- **Long context:** checkpoints spill to disk (`912e9eb`, R2). Open #2641 enforces `--cache-ram` by trimming checkpoints (REPORTED title).

Single-stream limits on our box (HYPOTHESIS): the same as llama.cpp `-sm layer` for GLM/Qwen4/DS4. `-sm graph` is only available for MiMo2. Graph-mode decode reductions without P2P go through the peer-copy ring.

---

## 4. ExLlamaV3 (deep)

`master` = pin `151539c` (v1.6.0). `dev` = `d334b03`. MIT. TabbyAPI (server) is AGPL-3.0 (R2).

### 4.1 Architectures (VERIFIED, `exllamav3/architecture/`)

- `glm5_next.py`: arch string `Glm5NextForConditionalGeneration`, KDA through `GatedDeltaNet`, MLA with DSA k-pool indexer, mHC, `BlockSparseMLP`. Plus `glm5_next_mtp.py`. `"supports_tp": True` (`:350`).
- `mimo_v2.py` (`MiMoV2ForCausalLM`, TP `:389`, +mtp, vision). V2.6 compatibility: UNKNOWN; the arch string is the same family.
- `qwen4_exp.py`: QSA indexer and PLE n-gram table. TP at `:311`.
- `deepseek_v4.py` (+mtp, vision). TP is the default capability (`model/model.py:27`).
- **No DeepSeek-V4.1/Engram** class on master or dev (`git grep DeepseekV41` on `origin/dev`: no hits).

### 4.2 TP/EP without P2P/NCCL (VERIFIED)

`exllamav3/model/model_tp_backend.py` has three backends: `TPBackendNull` (warmup), `TPBackendNCCL` (torch.distributed; delegates broadcast and gather to native), and **`TPBackendNative`** (`:221`).

Native setup (docstring `:233-246`, code `:247-394`):
- The master process creates 5 POSIX shared-memory regions: G (globals), B (bulk), R (reduce), S (small gather), LL (low-latency broadcast).
- Every CUDA worker process registers them with `cudaHostRegister(PORTABLE|MAPPED)` and gets device aliases.
- `all_reduce()` (`:458-475`) always calls `ext.pg_all_reduce_cpu`. The pure-GPU ring `pg_all_reduce` (`parallel/all_reduce.cu`, ring through host buffer) is commented out.

`pg_all_reduce_cpu` (`exllamav3_ext/parallel/all_reduce_cpu.cu`):

```
decode-size payload (single chunk, :437-530), CUDA-graph capturable:
  GPU r: send kernel  -> zero-copy store of its partial (fp16/bf16/fp32 wire) into R[slot r][stage]
                         -> write stage flag (system-scope)
  CPU helper process (run_cpu_reduce_jobs, :84-146): spin on job queue; when all ranks flagged,
                         sum the R slots (AVX2/AVX-512, threaded for big chunks) into R[accumulator]
                         -> release flag
  GPU r: recv kernel  -> wait flag; read accumulator back (zero-copy load) into the local tensor
prefill-size payload (:267-436): cooperative multi-block send/recv pipelines, CPU accumulate and
  both copy directions overlap; throttled by CPU release counters
```

So the all-reduce costs two PCIe crossings per GPU plus a CPU add, and needs no peer access.
**Reported:** 6×RTX 3090 PCIe Gen3, TR PRO 3945WX, Qwen3.8-Flash-Next EXL3 6.05 bpw, native TP: decode 76.1 tok/s at temp 1.0, cold prefill 1238 → 1458 tok/s after threading the AVX2 accumulate. AVX2 single-thread accumulate ≈10 GB/s, 4 threads 15.7 GB/s (commit `8114c49`, in master).

### 4.3 CPU MoE: two modes (VERIFIED)

**`-mcl/--moe_cpu_offload N`** (`doc/env_vars.md:315-325`):
- Routed experts of the first N MoE layers run on a spawned CPU worker process with weights in RAM.
- Layer-split only. Requires mul1-codebook experts, K ≤ 8 and uniform biases.
- "The parent's forward pass never blocks on the CPU." During prefill, hot experts are streamed to the GPU while the CPU does the tail.

**`-mcs/--moe_cpu_split N`** (`exllamav3/model_init.py:63, 214-217`): "run the TAIL N routed experts of every eligible block-sparse MoE layer on the CPU, overlapping the CPU GEMMs with each layer's own GPU expert compute. Dynamic hot/cold expert placement is on by default".
It asserts `not args.tensor_parallel`.

The algorithm, from `exllamav3/modules/block_sparse_mlp_cpu.py` and `exllamav3/model/moe_cpu_host.py`:

```
load: cpu_maybe_split_load (:134-170): keep the first E-N experts of the layer on the GPU, register
      the tail N with the CPU worker (weights rehomed into a hugepage arena, swizzled for AVX-512)
      optional static order from EXL3_MOE_CPU_SPLIT_STATS (per-layer selection counts)
decode step, per MoE layer (cpu_split_submit :230 → submit_issue_fused :1218):
  1. ONE GPU kernel (moe_split_issue): translate router ids through split_map (dynamic placement),
     update hit histogram, write sel/x/w zero-copy into a pinned job slot; if no selected expert is
     CPU-resident, skip the payload entirely
  2. cuStreamWriteValue32(data_ready, seq)          # stream memop, no SM, no host sync (moe_handoff.cu:94-96)
  3. GPU computes its resident experts + shared expert            ║ CPU worker computes tail experts
  4. cuStreamWaitValue32(done, seq); moe_split_collect_add       # fold CPU partial into output (:1280)
  5. cuStreamWriteValue32(consumed, seq)
dynamic placement (_split_swap_tick :606, _split_sweep_layer :633):
  every EXL3_MOE_CPU_SWAP_INTERVAL (128) steps mark a sweep pending; run it when the generator
  queue drains (between generations): per layer, swap hottest CPU expert with coldest GPU expert
  while hits_cpu/hits_gpu > EXL3_MOE_CPU_SWAP_HYST (2.0), budget EXL3_MOE_CPU_SWAP_MAX (64) swaps,
  re-read promoted weights from the checkpoint, decay counts
```

This is the Strata idea (hot set in VRAM, cold tail computed on the CPU, concurrent within a layer) with:
- an exact route: placement changes timing, not math;
- no host synchronization on the decode path.

**It is the strongest reusable design found in this sweep.**

Limits, all VERIFIED:
- Layer-split only. Not combinable with TP.
- mul1-codebook EXL3 experts only.
- Exclusive with `-mcl`.

CPU kernel speed, REPORTED (commit `94c05ba`, Xeon Gold 6148, 20 cores, cold experts from DRAM):
- AVX2 tier, 4096×2048 trellis GEMV: K3 15.6 GB/s, K4 ~21 GB/s.
- The AVX-512BW tier was added on top. End to end in TabbyAPI, GLM-5.3-Flash 3.05 bpw with 232/288 experts per layer on the CPU went 4.3–4.9 → 7.9 tok/s.
- `dev` adds "planar AVX2 MoE layouts" (`cc2ebee`, `2e37599`, `145b2ca`, `77ddae2`, 2026-10-09/10) with no published end-to-end number.

### 4.4 EXL3 quantization, cache, long context

- EXL3 is a QTIP variant at 2–8 bpw (README `:264-272`). Cache quantization is 2–8 bit.
- Quality versus GGUF at ≥4 bpw for GLM: UNKNOWN. Only Llama-3.1 KLD plots are in `doc/`.
- An EXL3 quant must be produced by `convert` from safetensors on a GPU (R2/pins). Community EXL3 GLM-5.3-Flash quants exist at 4.0 bpw (JANGQ-AI), "Q4" (0xSero) and K2 (REPORTED, search results). Not inspected.
- Cache modules: `exllamav3/cache/{mla,dsa,qsa,recurrent,quant}.py`.
- `dev` fixes long-context drift: "RoPE: Use more accurate sin/cos to avoid drift at long contexts" (`e6871da`).
- dev also replaces checkpointed history states for recurrent drafts ("Replace history states with single initial recurrent state…", `3a8e2fb`).

### 4.5 dev branch delta (VERIFIED `git log master..origin/dev`)

- Hybrid draft mode (`e72a906`).
- Recurrent mode for drafts in GDN/KDA/Mamba/SWA (`7de419b`).
- QSA top-k bound fix (`03d0042`, `74d48b7` for MLA).
- QCache fixed staging buffer (`fde1b37`).
- TP device-group barriers (`706f695`).
- Planar AVX2 CPU-MoE (§4.3).
- No change to the "layer-split only" restriction (`model_init.py` assertion unchanged on dev: HYPOTHESIS, not diffed line by line).

### 4.6 What it would mean on our server (HYPOTHESIS, arithmetic)

**GLM-5.3-Flash at 4.0 bpw:**
- 12,384 experts × 25.17 M params × 0.5 B ≈ 156 GB of experts, plus the dense part. This exceeds 128 GB VRAM, so TP is impossible and the split is mandatory.
- With ~95 GB VRAM left for experts, about 39% of experts sit on the CPU.
- Uniform routing gives ≈1.6 GB of CPU expert bytes per token. At 40–60 GB/s effective EXL3-AVX2 throughput on 48 Zen 3 cores (scaled ×2–3 from the 20-core Xeon), that is 27–40 ms → **25–37 tok/s ceiling from the CPU tail alone**, before dynamic-placement gains.
- Same order as R2 §5.0 and `ROOFLINE.md`. The EXL3 CPU decode may be compute-bound rather than DRAM-bound on AVX2.

**Qwen3.8-Flash-Next:**
- It fits VRAM at 4–6 bpw, so native TP over 4 GPUs applies.
- The 6×3090 Gen3 report (76 tok/s) is the best public evidence that host-staged TP works for single-stream decode on PCIe-only GeForce cards.

### 4.7 What limits single-stream decode on 4 GPUs without P2P (HYPOTHESIS)

- **TP mode:**
  - One CPU-reduce round-trip per all-reduce (2 PCIe crossings plus a CPU spin-wait) times ~2 per layer.
  - The Python/torch multiprocess orchestration, partly hidden by CUDA graphs.
- **Layer split with `-mcs`:**
  - GPUs work sequentially (as in llama.cpp).
  - Per layer: max(GPU resident experts, CPU tail) plus the flag-wait latency.
  - The CPU tail throughput of trellis decode on AVX2.

---

## 5. Other engines with GLM-5.3-Flash or our models on CUDA

### 5.1 fastllm (ztxz16/fastllm), Apache-2.0, HEAD `a2ff521` (VERIFIED unless noted)

- **Models** (README core-capabilities list and `src/models/`): `glm5_next.cpp` (3931 lines, GLM-5.3-Flash KDA, paged cache, MTP), `glm5_moe_dsa.cpp`, `qwen4exp`, `deepseekv4.cpp`, `deepseekv41.cpp` (+Engram, DSpark), `kimi_k3.cpp`. MiMo: not found by tree grep.
- **Devices** (`src/devices/`): cpu (`avx2.cpp`, `avx512*.cpp`, `amx.cpp`), cuda, multicuda (TP), numas, disk.
  `--device cuda --moe_device numa|cpu` puts routed experts on the CPU (`docs/glm5_en.md`).
- **CUDA expert cache** (`src/devices/cuda/moe/fastllm-moe-cache.cu:1-9`):
  > "Graph-capturable GGUF, FP8 and compact NVFP4 expert cache… A device LRU cache stores complete expert records… Misses are pulled directly from mapped pinned host memory by a CUDA kernel. Eager hybrid decode can leave selected experts on NUMA when their estimated CPU completion is earlier than a cache refill."

  Admission is limited to "at most one cold CPU route" per step (`:2571-2574`). Misses on the CPU path never demand-fill (`:2490`).
  This is a cost-model hybrid in the spirit of FreeToken's q*, with zero-copy miss fetch.
- **GLM-5.3 with TP:**
  - `Glm5NextModel::RunMoeWithPrefix` (`src/models/glm5_next.cpp:2380-2530`): TP ranks compute their slice of resident experts.
  - NUMA-resident routed experts are computed by rank 0, with event-based waits instead of a host drain on decode.
  - `FastllmCudaMergeMOEHybridOnDevices` merges the result.
  - GGUF-resident GLM experts must live on one CUDA device (assert at `:2483-2484`).
- **All-reduce** (`src/devices/multicuda/fastllm-custom-allreduce.cu:1-6`): adapted from vLLM custom all-reduce (peer reads) with an **NCCL fallback**. Without P2P it will use NCCL (HYPOTHESIS).
- **GGUF:** Q4_K CUDA MMQ and grouped MoE kernels (`src/devices/cuda/gguf_mmq/`). The NUMA device handles Q4_K/Q5_K (`numasdevice.cpp:6607`).
- **Benchmarks for GLM-5.3:** "No publishable complete GLM-5 / GLM-5.3-Flash throughput table is stored in the repository" (`docs/glm5_en.md`). Only kernel micro-benchmark JSONs (`docs/benchmarks/glm53_*_20261003.json`).
- **Verdict** (HYPOTHESIS): the only engine found that combines GLM-5.3-Flash, multi-GPU TP, CPU/NUMA experts and an asynchronous GPU expert cache in one process. Worth an E1-style run. Largely Chinese docs; Python packaging (`ftllm`).

### 5.2 DwarfStar = antirez/ds4, MIT, HEAD `fc80bd6` (VERIFIED unless noted)

- **Model-specific builds:** DeepSeek V4/V4.1 Flash, GLM 5.2/5.3, GLM 5.3 Flash, DeepSeek V4 PRO, Qwen3.8 Flash Next (`README.md`). Its own GGUFs only.
  GLM-5.3-Flash files: `glm53-q2` 90 GiB, `glm53-q4` 178 GiB, `glm53-fp8` 305 GiB with "inference not implemented" (`docs/MODELS.md:147-171`).
- **CUDA on Ada** is explicit (`docs/CUDA_MULTI_GPU.md`: "Ada Lovelace is supported, including 48 GB L40S").
- **Multi-GPU:**
  - "Normal layer placement" for GLM. Startup "refuses a layout that would require unsupported CPU execution".
  - "Flash tensor parallelism" pairs GPUs and halves routed experts per pair, for DeepSeek Flash only.
  - Reported: 8×L40S, ~126 tok/s aggregate with 16 sessions (not single-stream).
- **Peer copies:** NxN peer matrix validated at init with multi-size round-trip probes; on failure, a pinned-host bounce (`ds4_cuda.cu:2799-2900, 3509-3530`). The comment cites silent `cudaMemcpyPeer` corruption on RTX 6000 Ada.
  This is a useful safety idea for our modded 4080s: DS4_FORCE_HOST_BOUNCE / DS4_FORCE_CUDA_PEER.
- **SSD/RAM streaming for GLM: Metal and ROCm only** (`ds4.c:455-462`, `ds4_backend_supports_glm_streaming_full_layers`).
  On NVIDIA CUDA, GLM-5.3-Flash Q4 (178 GiB) cannot run on 128 GB VRAM (HYPOTHESIS from the two facts).
- Engram for DS-V4.1 (`ds4_engram.c`). CUDA DS-V4.1 is text-only with Q2 SSD streaming on Spark (`docs/MODELS.md:48-49`).
- **Verdict:** not usable for GLM ≥4 bit on our box. The peer-validation and bounce code is worth copying (MIT).

### 5.3 Specialized GLM-5.3 / MiMo engines found (README level; VERIFIED existence and SHA)

| repo | what | fits us? |
|---|---|---|
| neron82/helios-glm53flash `700eebe` (MIT) | from-scratch C++20/CUDA, **EXL3 2.05 bpw**, 2×RTX 3090 | no (≤3 bit) |
| hughmadden/glm53f-afd (created 2026-09-28, updated 2026-10-06; not cloned) | Rust/CUDA, attention on one 5090, experts on 4×DGX Spark over RDMA, EXL3 K4 | no (disaggregated hardware) |
| hughmadden/mimo26f-afd (MIT) | same design for MiMo-V2.6-Flash | no |
| fly9i/spark-engine | EXL3 engine for GB10, GLM-5.3 TP2 | no (sm_121) |
| dg1kjd/sglang-sxm2 `73abfe1` (Apache-2.0) | SGLang fork: GLM-5.3-Flash NVFP4 TP8 on **8×V100 32 GB** (sm70 decode kernels, two-level all-reduce, FP16 KV). Reported ~175 tok/s decode with MTP | Shows SGLang *can* run GLM-5.3 off-Hopper (vs R2's DeepGEMM blocker). But it needs 256 GB VRAM, full residency and no offload |
| sqliteai/warp `09fcff3` (Apache-2.0, 2.5k★) | C engine, CPU/Metal, experts from NVMe; GLM-5.3-Flash 3.86 tok/s on an M5 Pro | no CUDA ("no … CUDA … dependency") |
| bricesommers/apus-glm5.3-flash | C11, FP8 from NVMe, AVX2/NEON/Metal | CPU/SSD class |
| MindLab-Research/ferrite | Rust, sm_100a kernels | no (Blackwell) |
| Indras-Mirror/Strata-GLM `152c49f` (MIT) | Strata fork; README is **Strata-DS4** (DeepSeek-V4-Flash Q2). Reported 23.3–24.0 tok/s on a 4090 + Ryzen 5700X AVX2 versus llama.cpp 16.3 tok/s | Q2; GLM-5.3 3-bit with REAP pruning (description); not ≥4 bit |
| 0xSero/glm53-flash-offload `6769b27` | ExLlamaV3 + plugin, 3.05 bpw, 28.2 tok/s on a 3090 + EPYC 7443P (R2 row 8) | ≤3 bit |

---

## 6. Other local engines (requested list)

### 6.1 llama.cpp-multigpu (lukolszewski), MIT, `ee5d54a`

A downstream patch set of 47 commits on upstream `df03399` (`docs/multigpu/patches.md`, VERIFIED). Relevant to single-stream long-context decode:
- patch 1: qwen4exp decode depth-decay at long context;
- patch 2: compact-gather attention for quantized KV;
- patch 7: KQ mask built on the device instead of the host;
- patch 8: QSA bias on the device;
- patch 12: rotating pinned staging of scheduler inputs;
- patch 30: no pipeline drain on re-plan.

Reported 6×4090 Gen4 x8, Qwen3.8-FN UD-Q4_K_XL at 250K: upstream 21.0 → **48.7 tok/s** (R2 row 14). Validated on sm_86 only (patches.md).
Qwen4exp-specific. For GLM5-Next the same host-mask issue is plausible (HYPOTHESIS).

### 6.2 Summary table

| engine | status / license | our models | MoE offload / multi-GPU without NVLink | verdict |
|---|---|---|---|---|
| mistral.rs `3f2515e` | active, MIT | Qwen4Exp only (`mistralrs-core/src/vision_models/qwen4_exp/`, `gguf/qwen4exp.rs`) | NCCL TP, "P2P layer mapping", ring (README); no CPU-expert path found in tree | not for GLM/MiMo |
| Aphrodite `82b0cd8` | last commit 2026-09-11, **AGPL-3.0** | none of GLM5-Next/MiMo-V2/Qwen4Exp/DS-V4 in `model_executor/models` (tree grep) | vLLM fork; `cpu_moe.py` is the CPU *platform* backend | no |
| LMDeploy `822dddb` | active, Apache-2.0 | DeepSeek-V4 (PyTorch engine), GLM-5/5.2 (`glm_moe_dsa.py`), Qwen3-Next; no GLM-5.3-Flash/MiMo-V2/Qwen4Exp | TP; no CPU experts found | no |
| MLC-LLM `8978ea9` | active, Apache-2.0 | none (tree grep for glm5/mimo/qwen4/deepseek_v4: no hits) | — | no |
| llamafile `a3ccf09` | active, Apache-2.0 | inherits llama.cpp, no model files of ours | — | no added value over llama.cpp |
| llamAmpere `54cef46` | active, MIT (llama.cpp fork) | Qwen3.8-27B (dense) focus; SJ-KVaRN KV codec (4/4: 262K on one 24 GB card) | single GPU sm_86 | KV-compression idea only |
| machx `eab5629` | active, Apache-2.0 | GLM-5.3-Flash UD-Q4_K_XL with host-resident experts, MiMo-V2.6, DS-V4.1 | **Intel Arc SYCL only**. Reported on 2×Arc Pro B70: GLM-5.3-Flash 14.0–14.4 tok/s at 16K; MiMo-V2.6 16.7 plain / 23–24 with drafter | not portable; useful as a host-expert data point |
| gufo `665fc18` | active, MIT | Qwen3.8-FN, DS-V4-Flash | **Strix Halo gfx1151 only** | no |
| Splash `8a2b420` | active, Apache-2.0 | Qwen3.8-27B, Qwen3.6-35B-A3B | **Apple silicon only** (README badge, macOS 26.4) | no |
| Colibrì (JustVugg) `bf24429` | active, Apache-2.0 | GLM-5.2/5.3, GLM-5.3-Flash, DS-V4/V4.1, MiMo-V2.6, Qwen3.8-FN (README list) | CPU-first, SSD streaming, optional CUDA tier. Reported "router-lookahead prefetch: next layer's routing is 71.6% predictable" (uv-genai copy README) | slow (≤9 tok/s even on 6×5090 for GLM-5.2); lookahead idea is relevant |
| peregrine `869a5ed` | stale since 2026-08-23, no license file | GLM-5.2 | 3-lane CPU∥GPU∥io_uring scheduler (README) | no |
| Pulsar `18f9900` | no commits since 2026-09-01, MIT | no GLM-5.3/MiMo/DS-V4.1 (R2) | SSD streaming | no |

---

## 7. Most promising technologies: concrete algorithm and what it takes here

### T1. Intra-layer GPU-hit / CPU-miss split with asynchronous handoff

Sources: ExLlamaV3 `-mcs` (§4.3), llama.cpp PR #27861, fastllm eager hybrid.

**Call sequence to port into ggml/llama.cpp** (HYPOTHESIS for the port; each step mirrors code read above):

```
per MoE layer l, decode ubatch (n_tokens <= 8):
  GPU: router → ids (device)
  GPU: kernel K1: slot = map[l][ids]; split ids into hit list (slot>=0) and miss list;
       write x, miss ids, weights into a pinned job slot (zero-copy); cuStreamWriteValue32(ready[l])
  CPU worker pool (spinning on ready flags, NUMA-pinned): GEMV gate/up/down for miss experts from RAM
       (ggml Q4_K AVX2 / IQK kernels), write partial y_cpu into pinned slot; set done[l]
  GPU: mul_mat_id over cache banks for hit list + shared expert          (concurrent with CPU)
  GPU: cuStreamWaitValue32(done[l]); y = y_gpu + y_cpu                    (no host sync)
  async admission: histogram on GPU; between steps or generations, copy hottest misses into
       LRU/LFU victims on a separate copy stream; publish map after copy completes (as #27861)
```

**What it takes:**
- A new llama.cpp backend path. ggml's scheduler cannot express "two backends on one op concurrently", so #27861 builds two chains and sums them.
- The handoff via stream memops avoids the `ggml_backend_synchronize` in `llama_moe_cache::copy`.
- Risks:
  - CPU and GPU partials round differently. ExLlamaV3 and Strata-DS4 note non-bit-exact but valid results; parity tests need a tolerance.
  - The scheduler's split model.
  - Maintenance against upstream.
- **Testable without target hardware:**
  - On CPU only, the numerics of the split (hit set on a "fake GPU" = second CPU backend).
  - Cache policy with `tools/g0/cachesim.py` on routing traces.

### T2. Host-staged all-reduce for 4 GPUs without P2P

Sources: ExLlamaV3 native backend, llama.cpp `allreduce.cu` (2 GPUs only).

**Option a:** generalize `ggml_cuda_ar_pipeline` from 2 to N ranks.
- Each GPU writes its shard into its own pinned slot, spins until all N tokens arrive, and reads the N−1 peer slots, summing locally. That is N−1 reads per GPU per call.
- Or use ExLlamaV3's CPU-sum variant: N writes, 1 CPU sum, N reads.

**Option b:** keep NCCL and check that it initializes inside the VM (NCCL_P2P_DISABLE=1, SHM transport).

**What it takes:** a patch to `allreduce.cu` (`ggml_cuda_ar_pipeline_init` refuses `n != 2`) and `ggml-cuda.cu` comm init.
Testable only on ≥2 GPUs. The 3×3090 host (`epyc-ai`) can test N=3.

Value for GLM (HYPOTHESIS):
- Only after #30277. Then only the FFN/output are split, and experts are on the CPU anyway.
- The gain is bounded by the dense FFN share. Attention and indexer stay replicated, so the "TP of the dense part" lever from R2 §5.0 is only partly available.

### T3. Asynchronous expert cache with zero-copy miss fetch and cost-model admission

Source: fastllm (§5.1). A kernel reads missing experts directly from mapped pinned host memory: no `cudaMemcpy` launch and no host sync.
"Leave on CPU if CPU finishes before the refill" is the q* idea.
Port cost and risk are similar to T1. fastllm is Apache-2.0, so code can be borrowed with a NOTICE.

### T4. Validated P2P with automatic host bounce

Source: DwarfStar `ds4_cuda.cu:2799-2900`.
Cheap to add to our bench tooling (G0/E0). It probes byte-exactness of `cudaMemcpyPeer` on the modded cards before trusting P2P.

### T5. Pipeline overlap with host experts

Source: llama.cpp PR #29963. Prefill only, per the author's report (×1.5–2). It disables nothing we need.
Worth testing in E1 once merged.

---

## 8. Names the owner asked about

| name | result | how found |
|---|---|---|
| DwarfStar | **found**: antirez/ds4, "DwarfStar" (README logo/title), MIT, `fc80bd6` 2026-10-07, 23,759★ | GitHub MCP `search_repositories "DwarfStar"` → topics `dwarfstar`; then `ds4 user:antirez` |
| Pulsar | **found**: giannisanni/pulsar `18f9900` (2026-09-01), MIT; no commits since R2 | R2 + `git ls-remote` |
| Colibrì | **found**: JustVugg/colibri `bf24429` (2026-10-06), Apache-2.0; copies/forks uv-genai/colibri, colibri-rus, peregrine, iliria | `search_repositories "colibri inference engine"` |
| Splash | **found**: incoai/splash `8a2b420` (2026-10-09), Apache-2.0, Inco AI; Apple silicon only | WebSearch "Inco AI Splash" → repo name; `git ls-remote` |
| llamAmpere | **found**: JakeATX/llamAmpere `54cef46` (2026-10-10), llama.cpp fork for RTX 3090 (sm_86) | `search_repositories "llamAmpere"` |
| gufo | **found**: gufo-org/gufo `665fc18` (2026-10-10), MIT, Strix Halo only | `search_repositories "gufo inference"` |
| machx-inference-engine | **found**: Red-Weasel/machx-inference-engine `eab5629` (2026-10-07), Apache-2.0, Intel Arc SYCL | `search_repositories "machx-inference-engine"` |

---

## 9. Negative findings

1. llama.cpp `--moe-cache-mib`:
   - no CPU compute of misses;
   - a per-layer host sync;
   - no prefetch;
   - pageable source under default mmap;
   - disables pipeline parallelism;
   - incompatible with `-sm tensor`;
   - plain LRU (all VERIFIED, §2.2).
2. llama.cpp's no-P2P internal all-reduce is **2-GPU only** (`allreduce.cu:402`).
3. llama.cpp `-sm tensor` for GLM-5.3-Flash is not in master. The draft mirrors attention and KV on all GPUs.
4. ik_llama `-sm graph` still excludes glm5next, qwen4exp and deepseek4/41, and has no GPU expert cache at HEAD.
5. ExLlamaV3:
   - CPU split/offload and TP are mutually exclusive;
   - no DeepSeek-V4.1;
   - the server is AGPL (TabbyAPI);
   - CPU kernels decode EXL3 trellis (AVX2 15.6–21 GB/s reported on 20 Xeon cores);
   - GLM ≥4 bpw does not fit VRAM for TP.
6. DwarfStar: GLM on NVIDIA is VRAM-resident only; GLM FP8 inference is not implemented.
7. fastllm: no published GLM-5.3 throughput; docs mostly Chinese; GGUF GLM resident experts limited to one CUDA device.
8. Splash, gufo and machx are hardware-locked to Apple, Strix Halo and Intel Arc respectively.
   Aphrodite, LMDeploy, MLC-LLM and llamafile do not run GLM-5.3-Flash.
   mistral.rs runs only Qwen4Exp of our list.
9. No public benchmark exists for any of our models on 4×4080/4090 with expert offload at ≥4 bit.

## 10. Experiments this suggests (all runnable before the server exists, except where marked)

| # | experiment | where | decides |
|---|---|---|---|
| X1 | Instrument `llama_moe_cache::copy` (time from `synchronize` to return, hit rate) on Qwen3.6-35B-A3B with `--moe-cache-mib` | any CUDA GPU (3×3090 host) | Size of the host-sync overhead per layer (T1 motivation) |
| X2 | Rebase #27861 onto `abee0c8`; parity test (CPU-only backends) of the split sum vs stock `mul_mat_id` | CPU here | Feasibility of T1 in ggml |
| X3 | ExLlamaV3 `-mcs` with dynamic swap on a small EXL3 MoE (Qwen3.6-35B) vs llama.cpp `--moe-cache-mib` at equal VRAM | 3×3090 host | Whether the async split beats the sync cache; transfer of the result to GLM |
| X4 | NCCL init and all-reduce latency inside the Proxmox VM with P2P off; ExLlamaV3 native TP all-reduce latency | **target server** (E0/E5) | Whether `-sm tensor` / native TP is worth it on 4 GPUs |
| X5 | fastllm `ftllm server` GLM-5.3-Flash GGUF with `--moe_device numa` on 1–3 GPUs | 3×3090 host (has the UD-Q4_K_XL GGUF) | The first data point of a cost-model hybrid on our model |
| X6 | P2P validity probe (DwarfStar style) | target server (G0) | Trust in peer copies on modded 4080s |

---

## Sources

Repositories (all cloned or `ls-remote`d 2026-10-10; SHA and commit date in §1):
- https://github.com/ggml-org/llama.cpp @`abee0c8476b585b0d081d580b5ce21d0addc22c4`
  - code: `src/llama-moe-cache.{h,cpp}`, `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-arch.cpp`, `src/llama-model.cpp`, `src/llama-model-loader.cpp`, `src/models/glm5-next.cpp`, `common/arg.cpp`, `common/common.h`, `common/speculative.cpp`, `common/fit.cpp`
  - ggml: `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/{ggml-cuda.cu,allreduce.cu,fattn.cu,fattn-common.cuh}`
  - PR heads: #27861 `bccbacd`, #30277 `bcb1101`, #29963 `f81e182`, #30205 `8f7f1ae`, #30059 `03550d1`, #26414 `f794cbd`, #28696 `2076412`
- https://github.com/ikawrakow/ik_llama.cpp @`89bba35c8841618502b2be8273fb5265b405aca1`: `src/llama.cpp`, `src/graphs/build_glm5next.cpp`, `src/llama-kda.cpp`, `ggml/src/ggml-cuda/reduce.cu`, `common/common.cpp`, `README.md`
- https://github.com/turboderp-org/exllamav3 master @`151539c77abc7ab7425d30da7a4e8e3c5c154e7b`, dev @`d334b036b351f9952ff0112264d2ef9c80b5ba58`
  - `exllamav3/model/{model_tp_backend.py,moe_cpu_host.py,model.py}`, `exllamav3/modules/block_sparse_mlp_cpu.py`, `exllamav3/model_init.py`
  - `exllamav3/exllamav3_ext/parallel/{all_reduce.cu,all_reduce_cpu.cu}`, `exllamav3_ext/cpu/moe_handoff.cu`, `exllamav3/architecture/*`, `doc/env_vars.md`, `README.md`
  - commits `8114c49`, `94c05ba`, `cc2ebee`, `2e37599`, `e6871da`
- https://github.com/antirez/ds4 @`fc80bd695da76ee14bcb86820bf900a5cdfda806`: `README.md`, `docs/CUDA_MULTI_GPU.md`, `docs/MODELS.md`, `docs/SSD_STREAMING.md`, `docs/PERFORMANCE.md`, `ds4_cuda.cu`, `ds4.c`
- https://github.com/ztxz16/fastllm @`a2ff521f61820e1972a84e494cda58c0c2572c5f`: `README.md`, `docs/glm5_en.md`, `docs/version.md`, `src/models/glm5_next.cpp`, `src/devices/cuda/moe/fastllm-moe-cache.cu`, `src/devices/multicuda/fastllm-custom-allreduce.cu`, `src/devices/numas/numasdevice.cpp`
- https://github.com/lukolszewski/llama.cpp-multigpu @`ee5d54afd20d94aaea3ed097837eb079b96aa674`: `README.md`, `docs/multigpu/patches.md`
- https://github.com/EricLBuehler/mistral.rs @`3f2515e9b5adc2ac44949c128c50a13b15721294`
- https://github.com/aphrodite-engine/aphrodite-engine @`82b0cd8c2c1f24ad897871f2aac140d37bca1642`
- https://github.com/InternLM/lmdeploy @`822dddb8b3bc895eaf6cffb45a5514768e7ecb84`
- https://github.com/mlc-ai/mlc-llm @`8978ea9aec626d64096aaa92fbc89bda2e3ea802`
- https://github.com/mozilla-ai/llamafile @`a3ccf098ed9c7c77687f5430b0143156d3bcc45a`
- https://github.com/JakeATX/llamAmpere @`54cef46c266455a8c5e625527f40d74350ce1e74`
- https://github.com/incoai/splash @`8a2b42069923d483151372b615f6182c72c2b6cd`
- https://github.com/gufo-org/gufo @`665fc182445dc2685f2a9b2b4314aa7d92a858c2`
- https://github.com/Red-Weasel/machx-inference-engine @`eab562917b7c3d08bd5909ec659e7d4241e1185d`
- https://github.com/JustVugg/colibri @`bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850`; https://github.com/uv-genai/colibri @`5e7d6ecb85b8b815fa9be1c6d5b5b6f6bbff6ad9`
- https://github.com/s-b-repo/peregrine @`869a5edd3d13848c42bd1bb728139c85c8b12961`
- https://github.com/giannisanni/pulsar @`18f990094e1091ee2ef3078f20be418f8c162f03`
- https://github.com/sqliteai/warp @`09fcff352ca55223b08ee222d15054b90546c6a9`
- https://github.com/dg1kjd/sglang-sxm2 @`73abfe1dd7e7aae7a62b222a201e48f532fb298e`
- https://github.com/Indras-Mirror/Strata-GLM @`152c49f4d37c55d62179104a51816346203aff17`
- https://github.com/neron82/helios-glm53flash @`700eebe89f8164f7a25321f88ad6bfe24cce294a`
- https://github.com/0xSero/glm53-flash-offload @`6769b27652a18487758caac45a2b6999dc0a4184`
- Existence and description only (GitHub search, not cloned): hughmadden/glm53f-afd, hughmadden/mimo26f-afd, fly9i/spark-engine, bricesommers/apus-glm5.3-flash, MindLab-Research/ferrite.

Web pages (REPORTED content):
- https://github.com/ggml-org/llama.cpp/pull/29887 (MoE cache, benchmarks, review thread)
- https://github.com/ggml-org/llama.cpp/pull/29963
- https://github.com/ggml-org/llama.cpp/pull/30277
- https://github.com/ggml-org/llama.cpp/pull/28414
- https://github.com/ggml-org/llama.cpp/pull/27861
- https://github.com/ikawrakow/ik_llama.cpp/pull/2644
- Splash (secondary): https://hermes-ai.net/news/inco-ai-s-splash-runs-qwen3-8-27b-twice-as-fast-on-apple-silicon/ , https://trendshift.io/repositories/246624 (lmstudio.ai and aiwiki.ai did not resolve: ENOTFOUND)
- EXL3 GLM-5.3 quants (search snippets only): https://huggingface.co/JANGQ-AI/GLM-5.3-Flash-EXL3 , https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4 , https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2

Repo files used: `tools/engines/pins.json`, `models/glm-5.3-flash.json`, `docs/reassessment/research/R2-engines.md`, `docs/reassessment/research/README.md` (all @`b6f4fe3`).
