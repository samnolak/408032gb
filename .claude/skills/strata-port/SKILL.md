---
name: strata-port
description: Map of the Strata codebase (pinned c499bd1) and the checklist for porting a new model architecture into it. Use when reading Strata code or planning a backend.
---

# Porting into Strata

Full map: docs/strata-map.md. The short version:

| subsystem | Strata files | reuse |
|---|---|---|
| geometry + shape checks | include/strata/core/layout.hpp | pattern reuse; per-arch struct |
| GGUF reader | include/strata/artifact/gguf_reader.hpp | reuse; arch guard must be generalized |
| expert source / cache / CPU pool | include/strata/core/expert_source.hpp, expert_cache.hpp, src/kernels/cpu/expert.cpp | reuse after runtime geometry (G2) |
| expert GEMV (CUDA) | src/kernels/cuda/s2_expert_grouped.cu, native_mmvq.cu, iq_kernels.cu | H/FF are constexpr today: refactor in G2 |
| router | include/strata/kernels/router_top10.hpp | generic in k; GLM needs sigmoid + noaux_tc bias |
| linear attention | include/strata/kernels/gdn.hpp, fused_gdn.hpp | base for KDA (per-channel decay) |
| sparse attention selection | include/strata/kernels/qsa.hpp, src/kernels/cuda/qsa_select.cu | base for GLM k-pool indexer (4 -> 32 heads) |
| residual streams | include/strata/kernels/gr.hpp | GLM/DeepSeek need Sinkhorn mHC |
| n-gram table on SSD | include/strata/kernels/ngram.hpp, src/ngram/ple_reader.cpp | base for DeepSeek Engram |
| MTP + verify | src/core/mtp.cpp, src/core/verify.cpp | reuse |
| multi-GPU | docs/MULTI_GPU.md, include/strata/core/remote_experts.hpp | pipeline only; TP/EP is ours |

## Porting checklist

1. Fact sheet in docs/models/ and geometry in models/ (model-analyst).
2. Op-by-op diff against qwen4exp: same / parametrize / new.
3. For each "new" op: oracle in ref/ first, then kernel, then dump parity.
4. Load-time shape assertions for every tensor the kernels touch.
5. Token-level parity against llama.cpp before any performance work.
