# GLM-5.3-Flash — fact sheet (first target)

Source: HF zai-org/GLM-5.3-Flash config.json (retrieved 2026-10-01) unless noted. All CONFIRMED.

| item | value |
|---|---|
| layers | 45 + 1 MTP (`num_nextn_predict_layers` 1, layer 45) |
| hidden | 4096; vocab 154880 |
| attention pattern | `linear_attention` x3 then `deepseek_sparse_attention`, repeated: KDA on 34 layers, DSA on 3,7,...,43 |
| KDA | 64 heads x 128, short conv 4, `gate_lower_bound` -5.0 |
| DSA (MLA) | 64 heads, q_lora 1536, kv_lora 512, qk nope 256, **qk rope 0**, v 256, `mla_use_nope` |
| indexer | 32 heads x 128, top-k 2048, k-pool 4 with compress gate + APE, always select tail |
| residual | mHC: `hc_mult` 4, Sinkhorn 20 iterations, `hc_eps` 1e-6 |
| MoE | 288 routed, top-8, 1 shared, expert FF 2048; layers 0-2 dense (FF 12288) |
| router | sigmoid scoring, `noaux_tc` with `e_score_correction_bias`, routed scale 2.5, norm top-k, fp32 |
| SwiGLU | `swiglu_limit` 10.0 (clamp) |
| weights | FP8 E4M3, 128x128 blocks; hc_*, KDA small projections, A_log/dt_bias, norms, lm_head, router kept high precision |
| vision | 24-layer ViT, 1024 hidden (optional; skip for text) |
| llama.cpp | arch `glm5next`, src/models/glm5-next.cpp @ec7630a; GGUF quants exist (DevQuasar) |

Numerics risks (PROVISIONAL): swiglu clamp must be applied before the down projection in every
expert path (GPU and CPU); Sinkhorn in fp32; KDA decay exp(g) with g in (-5, 0) underflows
nowhere in fp32 but accumulates over long contexts — keep the state in fp32 like Strata's GDN.
