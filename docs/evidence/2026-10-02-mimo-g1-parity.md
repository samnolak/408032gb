# MiMo-V2.x-Flash G1 evidence, 2026-10-02 (sandbox CPU, no GPU)

Reference: llama.cpp ec7630a, CPU build, real `mimo2` graph, tiny random model from tools/tinygen/mimo2_tiny.py
(6 layers: dense + 5 MoE; global layers 0 and 5, sliding-window layers 1-4 with window 8; 4 query heads, 1 KV head
on global and 2 on SWA layers; K dim 24, V dim 16; partial NEOX RoPE on 8 dims, base 10000 / 5000 on SWA;
attention sinks; 8 experts, top-2). 40 tokens, `-fa off -ctk f32 -ctv f32`. Script: tools/tinygen/run_parity_mimo.sh.

```
$ tools/tinygen/run_parity_mimo.sh <llama.cpp> /tmp/g0cfg
test_attention_swa_sinks_gqa ... ok
test_plain_swiglu ... ok
test_rope ... ok
test_router ... ok
test_sinks_matter ... ok
Ran 5 tests
OK
```

Mutation check of the attention oracle on layer 1 (relative error against kqv_out):
correct 1.1e-07; no window 6.1e-01; window 9 instead of 8: 4.5e-01; tiled GQA (h % n_kv) instead of grouped (h // r): 9.1e-01.

## Findings (CONFIRMED)

1. **GQA grouping**: query head h reads KV head h // (n_head / n_kv) (ggml mul_mat broadcast). The delta-net op
   (GLM KDA) uses the tiled h % H_k instead: two different conventions in one codebase.
2. **Window**: key p0 is hidden from query p1 when p1 - p0 >= n_swa (window includes the token itself).
3. **Sinks**: one logit per head added to the softmax denominator only.
4. **Routing**: sigmoid + bias for selection, normalised weights, **no scaling** unless the GGUF carries
   `expert_weights_scale` (the graph passes hparams.expert_weights_scale; 0 = none). Whether the real MiMo GGUF
   sets it: UNKNOWN - check the first real GGUF's metadata.
5. **No SwiGLU clamp** for mimo2 (the clamp path is GLM5_NEXT / DEEPSEEK4 only).
