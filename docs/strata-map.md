# Strata code map (pinned c499bd102e7a4135c0de389dcfe38c399759ccc8)

Strata: ~91K lines in src/, include/, serve/,
tools/ (CONFIRMED: `wc -l`, docs/evidence/2026-10-01-recon.md). MIT license.

## Architecture lock-in (CONFIRMED)

| where | what is fixed |
|---|---|
| include/strata/artifact/gguf_reader.hpp:12 | guard: `general.architecture == "qwen4exp"` |
| include/strata/core/layout.hpp `ModelGeometry` | n_embd 2560, 48 layers, `qsa_interval` 4, GDN 16 k-heads / 48 v-heads, QSA 24/2 heads x 256, hc 4, 512 experts x 640 |
| src/kernels/cuda/s2_expert_grouped.cu:37-38 | `constexpr int H = 2560; FF = 640;` |
| include/strata/core/expert_source.hpp:18-20 | expert blob geometry fixed by the artifact; other widths refused |
| include/strata/kernels/ngram.hpp | PLE n-gram constants (n_embd 2560, head dim 160, 3-grams, 8 heads per n-gram) |
| src/program/generate.cpp:1749-1755 | only `expert_count`, `expert_used_count`, RoPE keys read from GGUF |

## Subsystems

| subsystem | files | GLM-5.3-Flash | MiMo-V2.6-Flash | DeepSeek-V4.1-Flash |
|---|---|---|---|---|
| expert source, cache, CPU pool | core/expert_source.*, core/expert_cache.hpp, kernels/cpu/expert.cpp | reuse after G2 | reuse after G2 | reuse after G2 |
| pinned host arena | src/core/pinned.cu | reuse | reuse | reuse |
| expert GEMV i-quants (CUDA) | kernels/cuda/native_mmvq.cu, iq_kernels.cu | runtime H/FF (G2) | + MXFP4 dequant | + MXFP4 dequant |
| router | kernels/router_top10.hpp | sigmoid + noaux_tc bias + scale 2.5 | softmax? UNKNOWN | sqrtsoftplus (config) |
| linear attention | kernels/gdn.hpp, fused_gdn.hpp | KDA = per-channel decay (ref/kda.py) | none | none |
| sparse attention | kernels/qsa.hpp, cuda/qsa_select.cu, qsa_prompt_attn.cu | k-pool indexer: reuse selection, 32 heads; attention core is MLA-nope (new) | none | CSA2 two-level indexer (new) |
| full / sliding attention | kernels/cuda/qsa.cu (GQA 24/2) | — | SWA 128 + global GQA with sinks (new, simple) | SWA + compressed (new) |
| residual streams | kernels/gr.hpp, cuda/gr.cu, fused_gr.cu | mHC + Sinkhorn 20 iters (new op on top) | plain residual | mHC 4 streams (shared with GLM) |
| n-gram memory on SSD | kernels/ngram.hpp, src/ngram/ple_reader.cpp | — | — | Engram: 2-,3-,4-gram hashes, layers 1 and 14 |
| MTP / speculation | core/mtp.cpp, core/verify.cpp | 1 MTP layer (DSA + MoE) | 5-layer DFlash drafter, 7 tokens | DSpark (target layers 37-39) |
| multi-GPU | docs/MULTI_GPU.md, core/remote_experts.hpp | pipeline; TP/EP is G6 | same | same |
| server / API | serve/server.py | reuse | reuse | reuse |

## GLM-5.3-Flash op-by-op diff vs qwen4exp (from llama.cpp graphs @ec7630a)

Shared builders (CONFIRMED, `src/models/qwen4exp.cpp` and `glm5-next.cpp`):
`build_inp_kpool`, `build_input_k_idxs`, `build_delta_net_base`, `build_gdn_l2_norm`,
`build_recurrent_attn`, `build_rs`, `build_moe_ffn`, `build_norm`, `build_ffn`.

| op | qwen4exp | glm5-next | action |
|---|---|---|---|
| linear-attn input | fused qkv + one conv | separate q/k/v convs (`glm5_conv1d`) | parametrize |
| linear-attn gate | scalar per head | per channel, `lower_bound * sigmoid(-(A*(f_b(f_a x)+dt)))` | new gate, same recurrence (`ggml_gated_delta_net` accepts both) |
| linear-attn output | norm-gated | RMSNorm(o) * sigmoid(g_b(g_a x)) | parametrize |
| sparse-attn selection | `build_qsa_sel` | `build_kpool_select` (key, gate and APE pooled per 4) | parametrize heads 4 -> 32, add pool gate/APE |
| sparse-attn core | GQA 24 q / 2 kv, d 256 | MLA, q_lora 1536, kv_lora 512, nope 256, rope 0, 64 heads | new kernel |
| residual | `build_hc_mix` / `build_hc_combine` | `build_hc_pre` / `build_hc_post` / `build_hc_sinkhorn` | new (shared with deepseek4) |
| MoE | 512 x 640, top-10, shared expert | 288 x 2048, top-8, 1 shared, first 3 layers dense | runtime geometry (G2) + router variant |
| PLE / n-gram | yes | no | drop |
