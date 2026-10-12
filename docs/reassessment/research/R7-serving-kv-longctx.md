# R7 — Serving engines (vLLM / SGLang / TRT-LLM / Dynamo) and KV / long-context infrastructure for GLM-5.3-Flash and Qwen3.8-Flash-Next on 4×RTX 4080 Super 32 GB (sm_89)

Research sweep 2026-10-10, KV / long-context researcher and serving-engine researcher (subagent R7). Builds on
`research/R2-engines.md` §2–3 and `research/R3-techniques.md` §7 (sweep of 2026-10-09), `ROOFLINE.md` (rev. 3),
`models/specs/*.json` and `docs/evidence/2026-10-10-n7-server-hybrid-cpu.md` (branch `claude/n7-server-hybrid-cpu`).
R2/R3 read vLLM `ed7ae70` and SGLang `46c2585` (2026-10-09); this sweep re-reads both at HEAD of 2026-10-10 and adds a
fork that R2 missed (SGLang-KT, which has an sm_89 path for GLM-5.3-Flash).

Labels (BRIEF rule 3): **VERIFIED** = code read at the recorded SHA (`repo@sha:path:line`) or command run here;
**REPORTED** = README/paper/issue/search snippet (URL given, hardware stated); **HYPOTHESIS** = my inference (basis given);
**UNKNOWN**; **DISPROVED**. Arithmetic derived from configs is labeled **PROVISIONAL (formula)** as the task requests; it is
HYPOTHESIS-class, not a measurement. Nothing here ran on a GPU. No engine was built.

---

## 0. Top findings (read this first)

1. **Stock vLLM and stock SGLang cannot run GLM-5.3-Flash on sm_89** — not a missing flag, three hard gates in code
   (VERIFIED): vLLM's GLM indexer raises without DeepGEMM and DeepGEMM is gated to SM90/SM100/SM120
   (`vllm@187a0eb:vllm/models/glm5next/nvidia/sparse_indexer.py:699-702`, `vllm/platforms/cuda.py:689-695`); every vLLM
   sparse-MLA backend is SM9x/10x/12x only and the sm_8x priority list contains only those three
   (`vllm/platforms/cuda.py:136-153`, `flashmla_sparse.py:203-204`, `flashattn_mla_sparse.py:77-78`,
   `flashinfer_mla_sparse_sm90.py:189-190`). SGLang upstream defaults DSA to `flashmla_sparse`/`fa3` on CUDA < SM100,
   rejects `triton` on CUDA, and its k-pool indexer calls `deep_gemm.fp8_(paged_)mqa_logits` on CUDA except TileLang on
   SM90 (`sglang@f9cee8d:python/sglang/srt/arg_groups/overrides.py:668-683,536-542`;
   `layers/attention/dsa/dsa_indexer_kpool.py:943-954,1055`). TRT-LLM's GLM k-pool backend requires FlashMLA
   (`TensorRT-LLM@7179c5b:tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/backend.py:50,499-500`).
2. **New since R2: the KTransformers fork of SGLang (`kvcache-ai/sglang@18c0de7`) has a real SM86/SM89 path for
   GLM-5.3-Flash** — a Triton k-pool indexer that consumes E4M3 Q/K on SM89
   (`python/sglang/srt/layers/attention/nsa/glm5_next_indexer_triton.py:21-37`), a Triton sparse-MLA decode with
   BF16 query and scaled-FP8 latent KV tuned for (8,9) (`nsa/glm5_next_sparse_attention.py:1-13,38-39`) and GLM-only
   KDA Triton kernels with SM86/SM89 configs (`linear/kernels/glm5_next_kda_ops.py`) — VERIFIED by reading. **But its
   CPU-expert path for GLM requires `--kt-method FP8`** ("the current {method} KT kernel discards swiglu_limit and would
   produce incorrect activations", `python/sglang/srt/models/glm5_next_moe.py:178-185`, VERIFIED), and the official KT
   tutorial sizes the FP8 model at ~306 GiB with ≥350 GB RAM (`ktransformers@5d048e5:doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md:12-16`).
   With 128 GB RAM + 128 GB VRAM this does not fit (HYPOTHESIS: 306 GiB > 256 GB total). Its Triton kernels are the most
   useful *donor code* (Apache-2.0) for an sm_89 DSA path; the engine as a whole is not a route for us.
3. **No serving engine in this set computes MoE experts on the CPU except KTransformers.** vLLM has only weight
   streaming (`cpu_offload_gb` UVA, layer-group prefetch — `vllm/config/offload.py:17-79`, no expert compute: grep for
   CPU-expert code returns nothing, VERIFIED); SGLang upstream has the same streaming plus the `KTEPWrapperMethod` hook
   with a **static** split (expert ids `< num_gpu_experts` on GPU, the rest on CPU, CPU work only on TP rank 0 —
   `layers/moe/kt_ep_wrapper.py:100-115,325-372`, VERIFIED); TRT-LLM has none (only the Qwen4-Exp PLE table can live in
   pinned host memory, `_torch/modules/qwen4_exp/ple.py:52-64`, VERIFIED). Streaming ~40% of GLM's experts over PCIe per
   token is the slow path ROOFLINE §4 p.3 already rules out. **vLLM no longer has a GGUF loader** (`grep -rli gguf vllm/`
   finds only two unrelated model files; no `gguf_loader.py` in `model_executor/model_loader/`, VERIFIED) — our
   UD-Q4_K_XL files cannot be served by vLLM at all.
4. **Custom all-reduce on 4 PCIe GPUs without NVLink is disabled in both vLLM and SGLang regardless of P2P**
   (`vllm/distributed/device_communicators/custom_all_reduce.py:270-276`;
   `sglang/.../custom_all_reduce_utils.py:492-498`, VERIFIED) → NCCL. New: both engines gained an **opt-in FlashInfer
   PCIe-IPC all-reduce** (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`, `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`), REPORTED by its
   authors at 5.3 µs vs 205 µs NCCL for 12 KiB on 8×SM120 (`sglang/.../pcie_ipc_ar.py:13-27`). It uses CUDA IPC peer
   pointers with plain PTX loads/stores (`flashinfer@ee4de7d:flashinfer/jit/comm.py:268-279`) → HYPOTHESIS: needs working
   GPU P2P, which GeForce drivers normally disable (BRIEF). Test it in E5; do not plan on it.
5. **Classic KV compression and sparse-attention retrofits barely matter for our two models.** GLM-5.3-Flash stores
   14.4 KiB/token in bf16 (3.61 GiB at 262K) and Qwen3.8-Flash-Next 27.8 KiB/token (6.94 GiB) — PROVISIONAL, formulas
   in §3.1 — and both already select top-2048 positions with a trained indexer. kvpress, SnapKV, ShadowKV, Quest,
   InfiniGen, MagicPIG, DuoAttention, RetroInfer, TriAttention are all dense-GQA, HF-transformers or vLLM-dense methods
   (§3.3); none supports MLA-latent + indexer or linear-attention layers. Saving even all of GLM's KV buys ~1.8 GiB with FP8
   ≈ 120 extra resident experts (~1% of 12 096) — PROVISIONAL. **What matters is recurrent-state checkpointing,
   prefix reuse across agent turns and NVMe persistence** (§4).
6. **Restore beats re-prefill by two to three orders of magnitude at 262K.** Full session snapshot (KV + state):
   GLM ~4.0 GB, Qwen ~7.6 GB (PROVISIONAL) → ~0.8 s / ~1.5 s from one NVMe at an assumed 5 GB/s, 0.2–0.4 s from RAM, vs
   37–655 s re-prefill at 7 000–400 tok/s (§4.3). Break-even prefix length for an NVMe load is tens of tokens (GLM: ~13 at
   400 tok/s, ~230 at 7 000 tok/s). The real risk is not speed but **prefix divergence from chat-template re-rendering**
   (reasoning stripped from older turns, §4.1) and **checkpoint RAM**: llama.cpp keeps up to 32 checkpoints per slot by
   default, 145.6 MiB each for GLM in fp32 → 4.55 GiB RAM per slot (PROVISIONAL) on a box whose RAM holds experts.
7. **llama.cpp (the engine R2 picks for R1) already does the right checkpoint placement** at HEAD: mid-prompt checkpoints
   are skipped unless the batch starts a user message or is near the prompt end, spacing ≥ `--checkpoint-min-step`
   (default 8192), last user message always checkpointed, checkpoints appended to slot save files
   (`llama.cpp@abee0c8:tools/server/server-context.cpp:4030-4061,2521-2585,2586`; `common/common.h:637-640`, VERIFIED).
   Whether the pinned `79e2e74` already has the span-aware rule is UNKNOWN (only its prefix reuse/slot save were
   verified on CPU in N7).

---

## 1. Part 1 — serving engines at source level

### 1.1 Snapshots read

| repo | SHA (HEAD, `git log -1`) | date | what was read |
|---|---|---|---|
| vllm-project/vllm | `187a0eb98aa42341d703f83421d693fa7585581b` | 2026-10-10 16:36 UTC | `vllm/models/{glm5next,qwen4_exp}`, `vllm/platforms/cuda.py`, `vllm/v1/attention/backends/mla/*`, `vllm/distributed/device_communicators/*`, `vllm/config/{cache,offload}.py`, `vllm/distributed/kv_transfer/kv_connector/v1/{offloading*,hisparse}`, `vllm/v1/kv_offload/**`, `docs/design/hisparse.md`, `docs/features/kv_offloading_usage.md` |
| sgl-project/sglang | `f9cee8d2b2d96a626db1968ba81e4d72bb31794a` | 2026-10-10 17:03 UTC | `models/{glm5_next,qwen4_exp}.py`, `layers/attention/{dsa,qsa,dsa_backend.py}`, `arg_groups/{overrides.py,fields/*.py,hisparse_hook.py}`, `layers/moe/kt_ep_wrapper.py`, `mem_cache/{mamba_checkpoint_pool,hicache_storage}.py`, `mem_cache/sparsity/*`, `distributed/device_communicators/{custom_all_reduce*,pcie_ipc_ar}.py` |
| kvcache-ai/sglang (SGLang-KT fork) | `18c0de72e59359baba793f7a5675f983aa656fc9` | 2026-09-30 | `models/glm5_next*.py`, `layers/attention/nsa/glm5_next_*.py`, `layers/attention/linear/kernels/glm5_next_kda_ops.py` |
| kvcache-ai/ktransformers | `5d048e516507e1603e0ed19645611b4501476a50` | 2026-10-10 | `doc/en/kt-kernel/{GLM-5.3-Flash-Tutorial,AVX2-Tutorial}.md`, `.gitmodules`, `kt-kernel/python/utils/amx.py` |
| NVIDIA/TensorRT-LLM (sparse checkout `tensorrt_llm/_torch`) | `7179c5b01b9d19acef69a6526330eb72f3b70e10` | 2026-10-10 | `_torch/models/modeling_{glm5_next,qwen4_exp}.py`, `_torch/attention/backends/sparse/{glm_kpool,qsa}`, `_torch/kv_cache_compression/*`, `_torch/modules/qwen4_exp/ple.py` |
| flashinfer-ai/flashinfer | `ee4de7dfd6d66a872345eee6e371923e6425f4a5` | 2026-10-10 | `flashinfer/jit/comm.py`, `flashinfer/comm/{cuda_ipc,pcie_ipc_*}.py`, `csrc/pcie_ipc_all_reduce.cu` |
| ai-dynamo/dynamo | `b208989fbe1cfbfde64d11334d8af4ff04fe1a99` | 2026-10-10 | `docs/fern/.../backends/vllm/native-kv-offloading.md`, `lib/` layout |
| ggml-org/llama.cpp (sparse) | `abee0c8476b585b0d081d580b5ce21d0addc22c4` | 2026-10-10 | `tools/server/server-context.cpp`, `common/{arg.cpp,common.h}`, `models/templates/*` |

### 1.2 Model files (does the engine know the architecture?)

| engine | GLM-5.3-Flash (`glm5next`) | Qwen3.8-Flash-Next (`qwen4exp`) |
|---|---|---|
| vLLM @187a0eb | `Glm5NextForCausalLM`/`…ForConditionalGeneration`/`Glm5NextMTP` (`model_executor/models/registry.py:120,421-423,686`); `SupportsPP`, `MixtureOfExperts`, `IsHybrid`, `SupportsEagle3` (`models/glm5next/common/model.py:1047-1049`) — VERIFIED | `Qwen4ExpForCausalLM`/`…ForConditionalGeneration`/`Qwen4ExpMTP` (`registry.py:112-114,594-596,690`), `SupportsPP`, `IsHybrid` (`models/qwen4_exp/nvidia/model.py:615-623`) — VERIFIED |
| SGLang @f9cee8d | `models/glm5_next.py`, `glm5_next_nextn.py` — VERIFIED (file list) | `models/qwen4_exp.py`, `qwen4_exp_mtp.py`, `qwen4_exp_ple_table.py` — VERIFIED |
| SGLang-KT @18c0de7 | `models/glm5_next{,_dsa,_moe,_norm}.py` with SM86/SM89 kernels — VERIFIED | not checked |
| TRT-LLM @7179c5b | `modeling_glm5_next.py` + `attention/backends/sparse/glm_kpool/` — VERIFIED | `modeling_qwen4_exp.py` + `sparse/qsa/` — VERIFIED |

### 1.3 sm_89 code paths

**GLM-5.3-Flash.** Three kernels decide: indexer logits (FP8 MQA over pooled keys), sparse MLA over 2051 selected latents,
KDA chunked prefill / recurrent decode.

| engine | indexer | sparse MLA | KDA | verdict on sm_89 |
|---|---|---|---|---|
| vLLM | `SparseAttnIndexerKpool.__init__` raises `"requires DeepGEMM to be installed"` if `has_deep_gemm()` is false (`nvidia/sparse_indexer.py:699-702`); the kernels are `deep_gemm.fp8_fp4_(paged_)mqa_logits` (`:329,580`); `support_deep_gemm()` = SM90 or SM100/SM120 family (`platforms/cuda.py:689-695`) | sm_8x priority list = `FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA` + sparse tail `FLASH_ATTN_MLA_SPARSE`(cap 9), `FLASHMLA_SPARSE`(9,10), `FLASHINFER_MLA_SPARSE_SM90`(9) (`cuda.py:136-153`) | prefill: FlashKDA only for major ∈ {9,10,12}, else Triton `chunk_kda_with_fused_gate` (`glm5next/common/kda.py:146-160`) | **blocked** (VERIFIED gates; the DeepGEMM python package may import on sm_89, but its kernels and the sparse backends are not built/selected for 8.9 — HYPOTHESIS for the import, VERIFIED for backend selection) |
| SGLang upstream | ragged: `deep_gemm.fp8_mqa_logits` on CUDA (`dsa_indexer_kpool.py:943`); paged: TileLang only when `arch_major == 9` (`:948-954`), else `deep_gemm.fp8_paged_mqa_logits` (`:1055`) | auto: `flashmla_sparse` prefill, `fa3` decode for major < 10 (`overrides.py:678-683`); `triton` DSA backend is ROCm-only (`:536-542`); `triton_sparse_mla` requires SM90+ (`dsa_backend.py:359-362`) | Triton (fla) | **blocked** (VERIFIED) |
| SGLang-KT | Triton scorer, `use_glm5_next_triton_indexer` true for (8,6),(8,9); SM89 consumes E4M3 Q/K (`nsa/glm5_next_indexer_triton.py:1-37`) | Triton decode, BF16 Q / scaled-FP8 KV, config (16,8,2) for (8,9) (`nsa/glm5_next_sparse_attention.py:1-13,30-40,153-161`) | GLM-only Triton gate + recurrent decode with SM86/SM89 configs (`linear/kernels/glm5_next_kda_ops.py:1-30`) | **runs by design** (VERIFIED code exists; not executed here). Authors' tutorial claims "NVIDIA SM89 and SM120" (`GLM-5.3-Flash-Tutorial.md:15`, REPORTED) |
| TRT-LLM | own TopK reuse (`modeling_glm5_next.py:948-953`) | `flash_mla_sparse_fwd` required (`glm_kpool/backend.py:50,499-500`); native path only SM100/103 (`:612-614`) | — | **blocked** (VERIFIED: FlashMLA is SM90+; vLLM's own gate for it is `capability.major in [9, 10]`) |

Independent analogue (REPORTED): `renning22/glm-5.2-4090@1540871` ports SGLang's DSA stack (indexer GEMM, top-k/page
transform, sparse MLA decode) to sm_89 with Triton + non-WGMMA TileLang and reports ~24 tok/s single-stream for GLM-5.2
FP8 on 32×4090-48GB (TP8×PP4, 4 nodes) — README; GLM-5.2, not Flash; different hardware and scale.

**Qwen3.8-Flash-Next.** vLLM's QSA backend `QWEN4_EXP_QSA_TRITON` is Triton (`nvidia/ops/qsa.py:1-3`, `@triton.jit`
kernels) with SM90/SM120 *tuning tables* only (`ops/qsa.py:19-28,519-559`) and requires `flash_attn_varlen_func`
(`nvidia/qsa.py:252-253`); FA2 in vLLM's build supports sm_80+ (HYPOTHESIS: standard FA2 coverage). QSA accepts BF16 or
FP8-e4m3 main KV, no other KV quantization, no context parallelism (`qsa.py:255-259,355-375`). Hyperconnection has SM90
fast paths with fallback (`nvidia/hyperconnection.py:115`). → **Qwen on vLLM sm_89 is plausible but untested**
(HYPOTHESIS). The PLE n-gram table can live in pinned host memory (`nvidia/ngram_embedding.py:210-211`, VERIFIED);
community W4A16 checkpoints exist (REPORTED: `VnimanieAI/Qwen3.8-Flash-Next-W4A16` claims 4×24 GB at 96K and 8×24 GB at
262K, PLE tables "102 GB" in host RAM; `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` — HF pages via search only). A 102 GB host
table on a 128 GB RAM box is marginal (HYPOTHESIS). SGLang's QSA is TileLang with torch fallback
(`layers/attention/qsa/mqa.py:3,18-19`); sm_89 behaviour UNKNOWN. TRT-LLM QSA indexer picks exact radix top-k when the
CuTe-DSL GVR prerequisites are missing (`sparse/qsa/indexer.py:178-200`); sm_89 end-to-end UNKNOWN.

### 1.4 CPU offload of MoE experts

| engine | mechanism | computes experts on CPU? | evidence |
|---|---|---|---|
| vLLM | `cpu_offload_gb` (UVA zero-copy: GPU kernels read host weights over PCIe every forward) or prefetch backend (`offload_group_size`, `offload_num_in_group`, async prefetch of whole layers); segment matching on param names such as `experts.w13_weight` | **no** | `vllm/config/offload.py:17-79` VERIFIED; no CPU MoE path found by grep (`ktransformers|expert_offload|cpu_experts|offload_experts` → none) |
| SGLang upstream | `cpu_offload_gb`, `offload_group_size/num_in_group/prefetch_step`, `offload_mode` (`arg_groups/fields/exec_.py:950-965`); KT hook `KTEPWrapperMethod` | **only via `kt_kernel`** (KTransformers). Split is static by expert id (`mask_cpu_expert_ids`: `topk_ids >= num_gpu_experts → -1`), submit → GPU → sync → add, CPU part on `tp_rank == 0` only | `layers/moe/kt_ep_wrapper.py:100-115,117-135,325-372` VERIFIED |
| SGLang "EPLB" | `srt/eplb/*` (expert distribution recorder, location updater, redundant experts) | no — balances experts **across GPUs** (EP), not CPU offload | file list VERIFIED; semantics HYPOTHESIS from names/R2 |
| SGLang-KT | as above + layerwise prefill; for GLM-5.3-Flash only `--kt-method FP8` | yes (FP8 only for GLM) | `models/glm5_next_moe.py:178-185` VERIFIED; RAM ≥350 GB (tutorial) |
| TRT-LLM | none for experts; PLE host offload for Qwen4-Exp (`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD`) | no | `_torch/modules/qwen4_exp/ple.py:52-64,364-402` VERIFIED |

Cost of the vLLM/SGLang streaming path for GLM (PROVISIONAL): the experts that do not fit VRAM (~39% of expert bytes
at 4.89 bpw, ROOFLINE §2 PP column "RAM 0.39") must cross PCIe in full per forward under UVA/prefetch, because these
mechanisms move layers, not the 8 routed experts: 0.39 × (12 096 experts × 25.17 M params × 4.89/8 B) ≈ 72 GB per token
→ ~3 s/token at 25 GB/s. Even an engine that streamed only routed experts would move ~0.39 × 42 × 8 × 15.4 MB ≈ 2.0 GB
per token → ~80 ms (≤ 12 tok/s). Matches ROOFLINE §4 p.3 ("compute misses on the CPU, don't copy them").

### 1.5 All-reduce on 4 PCIe GPUs without NVLink (and without proven P2P)

- **vLLM**: `CustomAllreduce.__init__` returns disabled when `same_node and world_size > 2 and not fully_connected`
  ("not supported on more than two PCIe-only GPUs", `custom_all_reduce.py:270-276`); `fully_connected` is NVML
  `NVML_P2P_CAPS_INDEX_NVLINK` (`platforms/cuda.py:809-818`). For 2 GPUs it additionally runs `_can_p2p`
  (`custom_all_reduce.py:86-99,277-291`). → on our 4×4080: NCCL (VERIFIED). NCCL without P2P uses its SHM/host
  transport (HYPOTHESIS, standard NCCL behaviour; `NCCL_P2P_DISABLE` style tuning appears in the 4090 port README).
- **SGLang**: identical rule (`custom_all_reduce_utils.py:492-498`, VERIFIED).
- **FlashInfer PCIe-IPC all-reduce** (new): vLLM `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` (default 0, `envs.py:275,1951`),
  world sizes 2/4/8, bf16/fp16 (`flashinfer_pcie_ipc_all_reduce.py:28-29`); SGLang `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`
  with a decode-sized workspace (`pcie_ipc_ar.py:30-60,98-114`). Authors' numbers (REPORTED, 8×SM120, two NUMA islands,
  hidden 6144 bf16): 12 KiB 5.3 µs vs NCCL 205.1 µs; TP8 decode TPOT 13.6 vs 21.1 ms (`pcie_ipc_ar.py:13-49`).
  FlashInfer builds it for any SM ("plain PTX loads/stores and CUDA IPC", `flashinfer/jit/comm.py:268-279`) and maps peer
  slabs from `cudaIpcOpenMemHandle` (`flashinfer/comm/cuda_ipc.py:131-144`; `csrc/pcie_ipc_all_reduce.cu:140-175`).
  **HYPOTHESIS:** remote loads/stores through IPC pointers need CUDA peer access between the devices; with P2P disabled by
  the GeForce driver the workspace either fails to open or the kernels fault → falls back to NCCL. Also only TP decode
  benefits; GLM can't use vLLM/SGLang TP on sm_89 anyway (1.3). Relevance: the *algorithm* (staged one-in/one-out
  pushes, neighbour-ordered) is a reference for E5 if P2P turns out to work (R4).

### 1.6 TP / EP / PP, prefix caching for hybrids, KV FP8 (summary)

| capability | vLLM @187a0eb | SGLang @f9cee8d |
|---|---|---|
| TP / PP / EP | TP, PP (`SupportsPP` on both models), EP via `MixtureOfExperts`; QSA forbids PCP/DCP (`qsa.py:368-374`) | TP, PP, EP, DP-attention; KT EP wrapper |
| prefix cache for KDA/GDN | `mamba_cache_mode="align"`: state cached only at the last token of a scheduler step that lands on `i·block_size`; `none` when prefix caching is off (`config/cache.py:189-196`); `prefix_cache_retention_interval` controls retained Mamba/SWA checkpoints (`:159-165`); `enable_mamba_shared_prefix_checkpoint` (`:197-202`); KDA has a prefill checkpoint alignment hook (`glm5next/common/kda.py:171-175`) — VERIFIED | radix + `MambaCheckpointPool`: cached states stored **int8** per (head, k-channel), conv window native, dequantized once on hit, "~2× more cached states", composes with HiCache (`mem_cache/mamba_checkpoint_pool.py:15-46`) — VERIFIED |
| granularity of hybrid prefix hits | unified block size N such that an attention page ≥ a Mamba page; LMCache reports N = 400 for Qwen3.8-Flash-Next at TP8 (`LMCache@1fde7be:docs/source/recipes/qwen3_8_flash_next.rst`) | radix node per request boundary (HYPOTHESIS from radix design) |
| KV tiers | `OffloadingConnector` CPU + secondary tiers (FS, object, KVCR adapter `v1/kv_offload/tiering/{fs,obj,kvcr}`), Mamba-align aware (`offloading/scheduler.py:147-173`); `--kv-offloading-size` (`engine/arg_utils.py:831,1405`); HiSparse (sparse-MLA KV host pool + GPU hot LRU, `docs/design/hisparse.md`, experimental) | HiCache L1 GPU / L2 host / L3 `file, mooncake, hf3fs, nixl, aibrix, …` (`arg_groups/fields/memory.py:98,145-165`); pools `KV, MAMBA, SWA, INDEXER` (`mem_cache/hicache_storage.py:53-59`) |
| KV FP8 | `CacheDType` includes fp8/fp8_e4m3/fp8_ds_mla/nvfp4_ds_mla/turboquant_*/int8_per_token_head… (`config/cache.py:39-59`); QSA: bf16 or fp8-e4m3 only | DSA fp8_e4m3 → `flashmla_kv` (Hopper) / `trtllm` (Blackwell); TileLang DSA on CUDA requires bf16 KV (`overrides.py:544-552`) |

Implication for us (HYPOTHESIS): even for Qwen3.8-Flash-Next on vLLM, hybrid prefix hits land on N-token boundaries:
PROVISIONAL N ≈ 1 596 (TP1), 798 (TP4), 399 (TP8) tokens for Qwen — formula in §3.1; the TP8 value reproduces LMCache's
reported 400, which corroborates the state-size formula. For agent turns this is fine (lose < N tokens per hit).

### 1.7 TensorRT-LLM, NVIDIA Dynamo, NIXL

- **TRT-LLM** @7179c5b: both model files exist; GLM path requires FlashMLA (blocked on sm_89, 1.3); no CPU-expert
  offload (1.4). New KV features: `kv_cache_compression/triattention` (periodic physical KV eviction by trigonometric
  scoring; "KV pools must be read with kv_layout=HND", `triattention.py:16-22`) and `quantization_for_cold_page`
  (NVFP4 cold pages) — both for dense attention pools (HYPOTHESIS from module names/docstrings). Not a route for us.
- **NVIDIA Dynamo** @b208989: orchestration layer (frontend, KV-aware router, KVBM block manager, planner) on top of
  vLLM/SGLang/TRT-LLM. It passes vLLM's KV offloading through and routes on tiered prefix overlap; "Models with
  sliding-window or Mamba/SSM layers ✅ … router sees only full-attention KV cache groups"
  (`docs/fern/.../backends/vllm/native-kv-offloading.md:21-25`, VERIFIED text). For one node and one user it adds
  nothing the engine lacks (HYPOTHESIS).
- **NIXL** @a88b4dd: transfer library (UCX, GDS and other plugins; Apache-2.0, README) used by vLLM/SGLang/Dynamo for P/D
  transfer and KV tiers. Single-node relevance: only as an SSD (GDS/POSIX) backend under HiCache or vLLM tiers.
- **Mooncake** @559f83c: distributed KV store / RDMA transfer engine; integrated as SGLang HiCache backend and vLLM
  connector (README news, REPORTED). Multi-node design; on one box a local file tier is simpler (HYPOTHESIS).

### 1.8 Verdict for the 4×4080 server (Part 1)

| model | vLLM | SGLang upstream | SGLang-KT | TRT-LLM | what to do |
|---|---|---|---|---|---|
| GLM-5.3-Flash (≥4-bit, 262K, 128 GB VRAM + 128 GB RAM) | blocked: DeepGEMM + no sparse-MLA backend on sm_89; no GGUF; no CPU experts | blocked (DeepGEMM / FlashMLA / FA3) | kernels OK on sm_89, but CPU experts FP8-only → ~306 GiB, does not fit | blocked (FlashMLA) | stay on llama.cpp / ik_llama / Strata-family (R2). Use SGLang-KT's Triton indexer / sparse-MLA / KDA kernels as reference or donor code (Apache-2.0) if we write our own sm_89 DSA path |
| Qwen3.8-Flash-Next | plausible (Triton QSA, FA2, PLE host table, community W4A16) but untested on sm_89; experts must fit VRAM (no CPU experts); TP over NCCL | UNKNOWN | — | UNKNOWN | keep as a comparison point only if an NVLink-less TP4 run on a W4A16 checkpoint fits; primary remains Strata (ROOFLINE: fits VRAM at Q4) |

---

## 2. Part 2 — KV / long-context infrastructure

### 2.1 Tiered KV stores and persistence

| system | SHA | tiers / format | integration point | hybrid (recurrent state) support | applies to us? |
|---|---|---|---|---|---|
| **LMCache** (dev) | `1fde7be` (2026-10-10) | L1 CPU, L2 local disk, GDS, Mooncake, NIXL, S3, Valkey, … (`docs/source/mp/l2_storage/*`); `kv_codec/asym_k16_v8.py`; CacheGen compression | vLLM `LMCacheMPConnector` (separate server process); SGLang storage backend | Mamba/GDN/KDA: recurrent state as an **opaque page**; needs `--mamba-cache-mode align`, `--separate-object-groups`, chunk = N (`docs/source/mp/hybrid_models.rst`); "not bit-exact … GDN backends do not support batch-invariant mode"; GLM-5.3-Flash **not covered** ("A -Flash variant … keeps a per-request kpool-tail scratch group (KpoolTailSpec) … not covered by this validation", `recipes/glm5_3.rst:111-116`); Qwen3.8-Flash-Next validated with vLLM on 8 GPUs, N = 400 (`recipes/qwen3_8_flash_next.rst`) — VERIFIED text | only behind vLLM; GLM blocked; Qwen possible |
| **py-kvcache** — exists (VERIFIED) | `atlarge-research/py-kvcache@3abba7a` (2026-07-26), Apache-2.0 | vLLM `OffloadingConnector` spec; files named by vLLM block hash; io_uring direct I/O; DRAM LRU/ARC write-through; **break-even gate** (`break_even.py:should_load`: decline a disk/RAM load when prefix < threshold for that medium); cost-model load planner; preloading while queued | requires the author's vLLM fork `t348575/vllm` (README) | inherits vLLM's (none of its own) | idea only; paper REPORTED 2.0× faster disk loads than LMCache at 80K tokens on H100 (arXiv 2609.11744 abstract via search) |
| **vLLM OffloadingConnector** | vLLM @187a0eb | CPU pinned primary + secondary `fs`/`obj`/`kvcr`/`p2p` tiers; GPU↔CPU `cudaMemcpyAsync`; per-request `max_load_tokens` (`docs/features/kv_offloading_usage.md`) | built in (`--kv-offloading-size`) | `MambaSpec` → "depends on a single state", hits aligned to N (`offloading/scheduler.py:147-173`) | only if vLLM is the engine |
| **vLLM HiSparse** (experimental) | vLLM @187a0eb | host pool of sparse-MLA KV + GPU hot rows with LRU; TP ranks share one pinned host pool for replicated MLA KV (`docs/design/hisparse.md`) | `MultiConnector` + sparse-MLA backends (`flashmla_sparse.py:372`) | n/a | no (sparse-MLA backends are SM9x+) |
| **SGLang HiCache** | SGLang @f9cee8d | L1 GPU / L2 host / L3 storage (`file`, `mooncake`, `hf3fs`, `nixl`, …); pools KV/MAMBA/SWA/INDEXER with `TRAILING_PAGES` hit policy for state pools (`hicache_storage.py:53-59,95,136,194-204`) | `--enable-hierarchical-cache`, `--hicache-storage-backend` | yes (int8 Mamba checkpoints compose with it) | only if SGLang is the engine (GLM blocked) |
| **Mooncake Store** | `559f83c` | distributed DRAM/SSD pool over RDMA/TCP | SGLang HiCache backend, vLLM connector (REPORTED) | via host engine | no (single node) |
| **llama.cpp server** | `abee0c8` (HEAD); R1 pin `79e2e74` | `--cache-ram` prompt cache in RAM (default 8192 MiB), `--ctx-checkpoints` (default 32 per slot), `--checkpoint-min-step` (default 8192), `--slot-save-path` + `POST /slots/{id}?action=save|restore` with checkpoints appended to the file (`common/common.h:637-640`, `server-context.cpp:2586-2652`) | built in | checkpoints = `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` state (recurrent/SWA part) (`server-context.cpp:2575`); N7 verified prefix extend, slot save/restore across restart, two slots, MTP on tiny glm5-next/qwen4exp (CPU) | **yes — primary path (R1)** |
| **Strata** (R3 §7.5) | `fb58e0d` (pins.json) | session files; message-boundary cache `STRATA_CACHE_MESSAGE_BOUNDARY=1`; KV prefetch from RAM (`docs/KV_PREFETCH.md`) | own server | yes (GDN) | yes for Qwen (R2) |

### 2.2 KV checkpoint / restore of recurrent state — what exists

- **Exact snapshot at token boundaries** (vLLM align, SGLang radix + int8 pool, llama.cpp checkpoints, LMCache opaque
  pages, Strata message boundary): the only exact method; cost = state bytes per checkpoint (§3.1).
- **Int8 cached state** (SGLang): one rounding of S at store, never re-enters the recurrence as a quant/dequant loop
  (`mamba_checkpoint_pool.py:24-36`). GLM: 145.6 → 44.6 MiB per checkpoint (PROVISIONAL). Not bit-exact; needs a
  logit/KLD check (E11/E12).
- **DASC** — decay-aware state compression of GDN/KDA checkpoints by per-head retention (arXiv 2608.30386; REPORTED via
  search snippet only; code not found in this sweep).
- **Tail-Replay** — keep full-attention KV, drop recurrent checkpoints, rebuild linear-attention state by replaying a
  short recent suffix (REPORTED, abstract via search). **HYPOTHESIS: approximate** (the true state depends on the whole
  prefix; only exact if decay makes old contributions negligible) → not acceptable for our "correct math" requirement
  without a KLD gate.
- **Marconi** (MLSys 2025) — admission/eviction for hybrid prefix caches weighting reuse likelihood × FLOPs saved per
  byte; up to 34.4× token hit rate vs baselines (REPORTED abstract). Relevant idea: with few, large checkpoints, admit
  only at reusable points (message boundaries) — which llama.cpp HEAD already approximates.
- **Continuum** (arXiv 2511.02230, ICLR 2026) — pin a request's KV for a TTL during tool calls (REPORTED). For a
  single-user box this reduces to "never evict the active session during a tool call" (HYPOTHESIS).

### 2.3 KV compression and sparse attention — applicability to OUR models

| method | SHA / source | what it needs | GLM-5.3-Flash (11 DSA/MLA + 34 KDA) | Qwen3.8-FN (12 QSA + 36 GDN) |
|---|---|---|---|---|
| kvpress (NVIDIA, 40+ presses incl. SnapKV, DuoAttention, KVzip, DMS) | `NVIDIA/kvpress@7136c22` (2026-10-05) | HF transformers; `SUPPORTED_MODELS` = Llama, Mistral, Phi3, Qwen2, Qwen3, Gemma3 (`presses/base_press.py:27-34`) | no (MLA latent, indexer, KDA) | no |
| SnapKV | `FasterDecoding/SnapKV@e216ddc` (2025-07) | prefill-time eviction by observation window; Llama/Mistral monkeypatches | no: the DSA indexer already picks 2048 per query; evicting latents also removes candidates from the indexer → changes model semantics (HYPOTHESIS) | no (same) |
| ShadowKV | `bytedance/ShadowKV@e51904c` (2025-05) | low-rank pre-RoPE keys on GPU, values on CPU, Llama-3/Qwen2 GQA, A100 | no (latent already 512-d, no RoPE in GLM latent) | no |
| Quest | `mit-han-lab/Quest@01c1623` (2025-07); also in SGLang `mem_cache/sparsity/algorithms/quest_algorithm.py` | per-page min/max keys, query-aware page top-k; Llama/Mistral; 4090/Ada6000 (README) | redundant: DSA is a trained, finer top-k | redundant: QSA block-sparse |
| InfiniGen | `snu-comparch/InfiniGen@3754f52` (2024-07, stale) | speculative KV prefetch from CPU for OPT/Llama | no | no |
| MagicPIG | `Infini-AI-Lab/MagicPIG@ac9aa36` (2024-12, stale) | LSH sampling on **AVX-512** CPU; "Only Llamas are supported" | no (and no AVX-512 on 7H43) | no |
| DuoAttention | `mit-han-lab/duo-attention@fe93c31` (2025-02) | trained retrieval-vs-streaming head masks; Llama/Mistral | no | no |
| RetroInfer | `microsoft/RetrievalAttention@03f912c` (2026-09) | KV as vector index (GPU–CPU co-execution), Llama-3-8B-1048K demo (README) | no | no |
| TriAttention | `WeianMao/triattention@a4bc3c8` (2026-08); in TRT-LLM | periodic KV eviction with offline calibration; vLLM/SGLang/MLX/llama.cpp-HIP ports (README) | no evidence for MLA/DSA (UNKNOWN) | UNKNOWN |
| FP8/INT8 KV | engines | per-block scales, fp32 accumulation | **yes, small win**: latent FP8 + FP8 indexer key halves 3.61 → 1.86 GiB at 262K (PROVISIONAL); SGLang-KT already stores scaled-FP8 latents on SM89 | QSA fp8-e4m3: 6.94 → ~4.0 GiB (PROVISIONAL) |

Conclusion (HYPOTHESIS): for our two models the only KV-side levers are (a) FP8/INT8 storage of latent/KV with fp32
accumulation (quality gate E12, indexer keys checked separately — R3 §7.3), and (b) persistence/prefix reuse. For
**MiMo-V2.6-Flash** (9 full-attention global layers, ~5.6 GiB KV read per token at 262K, ROOFLINE §4 p.4) Quest/ShadowKV-
style retrofits *would* matter — out of scope here, flagged for whoever evaluates MiMo.

---

## 3. Bytes per token and per checkpoint at 262 144 tokens (PROVISIONAL, formulas)

Inputs: `models/specs/glm-5.3-flash.json` (11 indexed layers, `kv_elems_per_token` 512, `aux_bytes_per_token` 320, pool
4; 34 linear layers, `state_bytes` 4 489 216) and `models/specs/qwen3.8-flash-next.json` (12 indexed, 1024 elems,
aux 320; 36 linear, `state_bytes` 3 145 728); Qwen GDN 16 k-heads / 48 v-heads (`docs/strata-map.md:11`), conv kernel 4
assumed. Script: `scratchpad/research/r7_calc.py` (output pasted below).

### 3.1 Formulas

- GLM KV/token (bf16) = 11 × (512·2 [latent, `qk_rope` 0] + 128·2 [indexer key f16] + 128·2/4 [pooled key share]) = 14 784 B.
- GLM KV/token (FP8) = 11 × ((512 + 4·4) + (128 + 4) + (128 + 4)/4) ≈ 7 623 B (fp8 latent + 4 fp32 group scales as in
  SGLang-KT's 128-group scaling; fp8 indexer key + scale, 132 B as in SGLang's `head_dim_with_sf = 132`).
- GLM KDA state per layer = 64·128·128·4 (fp32 S) + 3·8192·3·4 (q/k/v short-conv windows) = 4 489 216 B; ×34.
- Qwen KV/token (bf16) = 12 × (2 KV heads · 256 · 2 [K,V] · 2 B + 320) = 28 416 B.
- Qwen GDN state per layer = 48·128·128·4 + 3·(16·128·2 + 48·128)·4 = 3 268 608 B; ×36.
- vLLM unified block N (per rank) ≈ state bytes per layer per rank / attention bytes per token per layer per rank;
  GLM latent is shared by all heads and replicated under TP (spec `kv_heads` note), Qwen's 2 KV heads → 1 per rank at TP ≥ 2.

### 3.2 Results (literal output)

```
$ python3 -I scratchpad/research/r7_calc.py
GLM KV/token bf16 B 14784 fp8 B 7623.0
GLM KV@262K bf16 GiB 3.609 fp8 GiB 1.861
GLM KDA state fp32 MB 152.6 MiB 145.6 | bf16 MiB 77.6 | int8 MiB 44.6
GLM full snapshot @262K bf16+fp32 GiB 3.752
GLM llama.cpp-layout per token B 21504 @262K GiB 5.25
QWEN KV/token bf16 B 28416 @262K GiB 6.938; fp8-ish GiB 3.984
QWEN GDN state fp32 MiB 112.2 int8 MiB 32.1
QWEN snapshot @262K GiB 7.047
TP1: GLM N~4384 tokens ; QWEN N~1596 tokens
TP4: GLM N~1096 tokens ; QWEN N~798 tokens
TP8: GLM N~548 tokens ; QWEN N~399 tokens
GLM spec bf16 GB 4.03  NVMe1@5GB/s 0.81s  NVMe3@14GB/s 0.29s  RAM->GPU@20GB/s 0.20s
GLM llama.cpp GB 5.79  NVMe1@5GB/s 1.16s  NVMe3@14GB/s 0.41s  RAM->GPU@20GB/s 0.29s
QWEN bf16 GB 7.57  NVMe1@5GB/s 1.51s  NVMe3@14GB/s 0.54s  RAM->GPU@20GB/s 0.38s
re-prefill 262K at 400 tok/s: 655 s
re-prefill 262K at 1000 tok/s: 262 s
re-prefill 262K at 3000 tok/s: 87 s
re-prefill 262K at 7000 tok/s: 37 s
32 ckpts GLM fp32 GiB 4.55 int8 GiB 1.39 ; QWEN fp32 GiB 3.51
GLM break-even prefix tokens at R=400, NVMe 5GB/s: 13
GLM break-even prefix tokens at R=1000, NVMe 5GB/s: 33
GLM break-even prefix tokens at R=7000, NVMe 5GB/s: 232
```

| quantity | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| KV/token, bf16 | 14.4 KiB | 27.8 KiB |
| KV at 262K, bf16 / FP8 | 3.61 / 1.86 GiB | 6.94 / ~3.98 GiB |
| recurrent state per checkpoint, fp32 / bf16 / int8 | 145.6 / 77.6 / 44.6 MiB | 112.2 / — / 32.1 MiB |
| full snapshot at 262K (KV bf16 + state fp32) | 3.75 GiB (= ROOFLINE §2 "3.75 GiB", consistent) | 7.05 GiB (ROOFLINE: 7.0) |
| llama.cpp storage (3×128 f16 indexer columns, 12 DSA layers incl. MTP) | 5.25 GiB by this formula; R1 §3 / spec note "up to ~8 GB" (the formula's dtype assumptions may be low) | — |
| measured anchor | ik_llama on the 3090 host: 1598 MiB at 131K with q8_0 KV + f16 indexer (`PROBLEMS.md:65@eb59224`, PROVISIONAL per R3) → ~3.2 GiB at 262K | none |

Per decoded token the recurrent state is also read and written: GLM 2 × 152.6 MB ≈ 0.31 GB/token in fp32 (PROVISIONAL),
which ROOFLINE already includes as "state" bytes; an fp32→bf16 working state would halve it (quality risk, R1 note "keep
the state in fp32").

---

## 4. Part 3 — what it means for agent sessions

### 4.1 Prefix stability under chat-template re-rendering (VERIFIED templates; GLM-5.3-Flash/Qwen3.8 templates themselves UNKNOWN — not readable here)

- **Qwen3.5 template** (vLLM test fixture `rust/src/chat/tests/templates/qwen35.jinja:67-101`, proxy for the Qwen line):
  `last_query_index` = last user message that is *not* a `<tool_response>`; assistant messages **after** it are rendered
  with `<think>…</think>`, assistant messages **before** it without reasoning. Generation prompt with thinking off is
  `<think>\n\n</think>\n\n` (`:149-153`).
  - Within a tool loop: append-only → full prefix reuse (VERIFIED by reading the template logic).
  - When a new real user message arrives: every assistant message of the previous turn is re-rendered **without** its
    reasoning → the token sequence diverges at the first assistant message after the previous user query. Reusable
    prefix = up to and including the previous user query + `<|im_start|>assistant\n`.
- **GLM-4.7-Flash template** (`llama.cpp@abee0c8:models/templates/GLM-4.7-Flash.jinja:32-56`): `last_user_index` counts
  role `user` only (tool results are role `tool`), reasoning kept for assistant messages after it, **or for all when
  `clear_thinking` is false** ("preserved thinking"); stripped turns render as `</think>` + content. → set
  `chat_template_kwargs: {"clear_thinking": false}` for GLM if the 5.3-Flash template has the same knob (HYPOTHESIS by
  lineage) — history becomes append-only and every turn reuses the full prefix.
- **Client side (HYPOTHESIS):** many OpenAI-compatible agent clients do not send `reasoning_content` back. Then even inside
  a tool loop the re-rendered assistant message lacks the thinking tokens that are in the cache → divergence at every
  step at the start of the last assistant message. Mitigation is exactly a checkpoint at the start of each assistant
  message (= end of the preceding user/tool message), which llama.cpp HEAD and Strata's message-boundary cache provide.
- **Server-side tool parsing/normalisation** (whitespace trimming, JSON re-serialisation of tool calls) can also change
  tokens of earlier assistant messages (HYPOTHESIS) — the N7 harness checks token-id prompts only; add a chat-level test.

### 4.2 Checkpoint placement (hybrid models cannot roll back recurrent state)

To resume at divergence point d with the newest state checkpoint at c ≤ d, the engine must re-run **all layers** for
tokens c..L_new (layers interleave, so KDA/GDN inputs depend on full-attention outputs); DSA/QSA KV for 0..d is reused.
Cost = (L_new − c)/R. Placement that minimises (d − c) for agent traffic, in priority order:

1. start of the last user/tool message (= end of everything the client will resend unchanged);
2. end of the system prompt + tool schema (shared by every session of the same agent);
3. start of each assistant message (covers clients that drop reasoning);
4. periodic every Δ tokens for very long single messages (pasted files), Δ = 8–16K.

llama.cpp HEAD implements 1, 4 and "near prompt end" (`server-context.cpp:4030-4061`, VERIFIED); 2 and 3 are covered only
if they coincide with user-message starts. RAM budget (PROVISIONAL): 32 GLM checkpoints = 4.55 GiB fp32 per slot
(1.39 GiB if int8 like SGLang); with experts in RAM, set `--ctx-checkpoints 8–12` and keep `--checkpoint-min-step` at
8192, or implement int8 checkpoints (patch, ~100 lines, quality gate E11).

Concrete call sequence for a session store on top of llama-server (HYPOTHESIS design, uses only VERIFIED endpoints):

```
on request(session_id, messages):
    toks = tokenize(render(messages, clear_thinking=False))          # GLM; Qwen: native template
    if slot(session_id) is resident: send /completion (cache_prompt) # engine finds LCP + newest checkpoint <= LCP
    elif file = store.lookup(session_id, fingerprint):               # fingerprint = sha(model file, quant, template,
        POST /slots/0?action=restore {filename: file}                 #   llama.cpp SHA, KV dtype, ctx size)
        send /completion                                             # reuse = LCP(file tokens, toks)
    else: send /completion                                           # cold prefill
    after response: POST /slots/0?action=save {filename: session_id.bin}   # async, NVMe; checkpoints appended
    verify (sampled): logits of first generated token vs cold run on a 1% sample (E11)
```

### 4.3 Restore-from-NVMe vs re-prefill at 262K — cost model

Parameters: S_kv(L) = b·L (b from §3.2), S_state = state bytes, k = number of checkpoints written in the file, BW_nvme
(single 990 Pro: Samsung spec 7.45 GB/s seq read, REPORTED; 5 GB/s assumed inside the VM, HYPOTHESIS until E13),
BW_ram→gpu ≈ 20–25 GB/s (R4), t0 = open + metadata (~1–2 ms), R(L) = prefill rate (ROOFLINE bound ≥ ~5 000 tok/s for
GLM ≈ "TTFT ~50 s"; observed 400 tok/s on base A / 3090, R3 §7 table; Qwen 744–7 403 tok/s on 6×4090 llama.cpp at 250K,
R2 §4 row 14 — REPORTED, other hardware).

- T_restore(L) = t0 + (b·L + (k+1)·S_state)/BW + (L_new − c)/R
- T_prefill(L) = L/R(L)
- restore wins when L > t0·R/(1 − b·R/BW) — tens to hundreds of tokens (§3.2: 13 / 33 / 232 tokens at R = 400 / 1000 /
  7000 tok/s, NVMe 5 GB/s). py-kvcache encodes the same gate as measured curves (`break_even.py:should_load`).

| at L = 262 144 | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| snapshot size (spec layout) | 4.03 GB | 7.57 GB |
| restore, 1 NVMe @5 GB/s / 3 NVMe striped @14 GB/s / RAM→GPU @20 GB/s | 0.81 / 0.29 / 0.20 s | 1.51 / 0.54 / 0.38 s |
| re-prefill @400 / 1000 / 3000 / 7000 tok/s | 655 / 262 / 87 / 37 s | same |
| ratio | 45–3 200× | 25–1 700× |

(all PROVISIONAL; replaced by E11/K3 `tools/kv/slot_persist.py` measurements on the server.)

Practical consequences (HYPOTHESIS): (1) persist every session after every response — at ≤ 8 GB per 262K session, 3 TB
of NVMe holds hundreds of sessions; (2) keep the hot session resident (`--cache-ram`, or simply never erase the slot
during tool calls — Continuum's point); (3) the 262K TTFT problem for agents reduces to the *first* fill of a context
and to template-induced divergence, not to every turn; (4) restore files are model/quant/template/engine specific —
never share across versions (LMCache: "cache entries must not be shared across engines with different attention
backends or kernel block sizes").

### 4.4 What to test first (adds to E11/E12, no GPU needed for some)

1. Chat-level N7 extension (CPU, tiny models): render real multi-turn agent transcripts with the GLM and Qwen templates,
   with and without `reasoning_content` echoed and with `clear_thinking` false/true; assert reuse ≥ (prompt − last turn).
2. Checkpoint RAM: `--ctx-checkpoints` × state size on the tiny models, logged sizes vs formula (§3.1).
3. On the server: K3 at 32K/128K/262K, file sizes vs §3.2, restore time from RAM page cache and from cold NVMe
   (`echo 3 > drop_caches`), first-token logits equal to cold prefill.
4. If SGLang-KT is ever revisited: its sm_89 Triton DSA kernels against `ref/` oracles (parity-oracle skill) — they are
   the only third-party sm_89 GLM DSA implementation found.

---

## 5. Could not verify / negatives

- GLM-5.3-Flash and Qwen3.8-Flash-Next chat templates (HF blocked) — used GLM-4.7-Flash and Qwen3.5 templates as proxies.
- Whether `cudaIpcOpenMemHandle`-based FlashInfer PCIe-IPC all-reduce works with P2P disabled — UNKNOWN (no GPU).
- Whether vLLM Qwen4-Exp actually runs on sm_89 — HYPOTHESIS; no run, no report found.
- SGLang-KT GLM path on sm_89 — code read, not run; RAM sizing makes it moot for 128 GB.
- DASC and Tail-Replay code — not found (searched by title; only abstracts via search).
- `kvcr` (vLLM tier adapter) — external package, not read.
- llama.cpp pin `79e2e74` checkpoint placement rule — not re-read (HEAD `abee0c8` read instead).
- `py-kvcache` is NOT on PyPI (`pip download py-kvcache` → "No matching distribution"); it exists on GitHub only.

---

## Sources

Repositories (URL, SHA, commit date):
1. https://github.com/vllm-project/vllm — `187a0eb98aa42341d703f83421d693fa7585581b`, 2026-10-10 16:36:03 +0000
2. https://github.com/sgl-project/sglang — `f9cee8d2b2d96a626db1968ba81e4d72bb31794a`, 2026-10-10 17:03:40 +0000
3. https://github.com/kvcache-ai/sglang (SGLang-KT) — `18c0de72e59359baba793f7a5675f983aa656fc9`, 2026-09-30 21:29:17 +0800
4. https://github.com/kvcache-ai/ktransformers — `5d048e516507e1603e0ed19645611b4501476a50`, 2026-10-10 16:16:45 +0800
5. https://github.com/NVIDIA/TensorRT-LLM — `7179c5b01b9d19acef69a6526330eb72f3b70e10`, 2026-10-10 04:22:14 -0500 (sparse checkout)
6. https://github.com/flashinfer-ai/flashinfer — `ee4de7dfd6d66a872345eee6e371923e6425f4a5`, 2026-10-10 18:28:30 +0000
7. https://github.com/LMCache/LMCache (branch dev) — `1fde7bec33cd629277da9b2291049fa706584ab0`, 2026-10-10 09:58:48 +0000
8. https://github.com/atlarge-research/py-kvcache — `3abba7a502d553f6e7e2e58b92086487e3395d7e`, 2026-07-26 17:38:25 +0200
9. https://github.com/ai-dynamo/dynamo — `b208989fbe1cfbfde64d11334d8af4ff04fe1a99`, 2026-10-10 11:27:18 -0400
10. https://github.com/ai-dynamo/nixl — `a88b4ddc604afb77066c625f1777a087f4d69616` (ls-remote; README via raw.githubusercontent)
11. https://github.com/kvcache-ai/Mooncake — `559f83cb73eb870d208a8d4f333b1a24ee992f26` (ls-remote; README via raw)
12. https://github.com/ggml-org/llama.cpp — `abee0c8476b585b0d081d580b5ce21d0addc22c4`, 2026-10-10 20:00:56 +0200 (sparse)
13. https://github.com/NVIDIA/kvpress — `7136c224105ba7b046a6e538742c3589698680b8`, 2026-10-05
14. https://github.com/FasterDecoding/SnapKV — `e216ddc84c5bd210378cbdbbba12ba02102aa640`, 2025-07-10
15. https://github.com/bytedance/ShadowKV — `e51904cdeab7d4d34013370f09f2cf5fcd655e15`, 2025-05-01
16. https://github.com/mit-han-lab/Quest — `01c1623bf9395009520874e989e29f683203b357`, 2025-07-10
17. https://github.com/snu-comparch/InfiniGen — `3754f52d131c30ab7a310e3c4cf356e8e3326ac3`, 2024-07-09
18. https://github.com/Infini-AI-Lab/MagicPIG — `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`, 2024-12-16
19. https://github.com/mit-han-lab/duo-attention — `fe93c314ae87306ef6629dc16713250b4718ffe7`, 2025-02-10
20. https://github.com/microsoft/RetrievalAttention (RetroInfer) — `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`, 2026-09-14
21. https://github.com/WeianMao/triattention — `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`, 2026-08-04
22. https://github.com/renning22/glm-5.2-4090 — `1540871f740cb2cd4ef1d14dfb720bba672e928c`, 2026-06-23
23. Repository docs read: /home/user/wt-research @`b6f4fe3` — ROOFLINE.md, research/R2-engines.md, research/R3-techniques.md,
    research/README.md, models/specs/{glm-5.3-flash,qwen3.8-flash-next}.json, docs/strata-map.md, tools/kv/slot_persist.py;
    /home/user/408032gb docs/evidence/2026-10-10-n7-server-hybrid-cpu.md.

URLs (search results; REPORTED):
- py-kvcache paper: https://arxiv.org/abs/2609.11744 ; review https://www.themoonlight.io/en/review/building-py-kvcache-a-performance-characterization-of-external-kv-caching-for-vllm-with-nvme-ssds
- Marconi: https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html
- DASC (arXiv 2608.30386): https://www.emergentmind.com/papers/2608.30386 ; Tail-Replay: R3 §7.5 (arXiv 2608.30310) and the same search
- Continuum: https://arxiv.org/abs/2511.02230v1 , https://iclr.cc/virtual/2026/10012473
- Qwen3.8-Flash-Next 4-bit checkpoints: https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16 , https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 , https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- GLM on Ada / DSA: https://github.com/renning22/glm-5.2-4090 ; https://discuss.vllm.ai/t/seeking-help-deepseek-v4-flash-fails-to-deploy-on-8x4090-48gb-vram-per-card/2731 ; https://deploy.helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1
- Other KV libraries seen while verifying py-kvcache: https://gittrend.io/repo/ovg-project/kvcached , https://pypi.org/project/kachedb/
