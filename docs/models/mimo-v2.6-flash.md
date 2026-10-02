# MiMo-V2.6-Flash — fact sheet (second target)

| item | value | label |
|---|---|---|
| params | 309B total / 15B active | CONFIRMED (model card) |
| layers | 48: 1 dense + 47 MoE | CONFIRMED (vLLM recipe) |
| attention | 39 SWA (window 128) + 9 global; 64 q heads, 8 SWA KV heads, 4 global KV heads | CONFIRMED (config via MiaAI-Lab deploy notes) |
| head dims | QK 192, V 128 | PROVISIONAL (V2-Flash report) |
| MoE | 256 routed, top-8, no shared expert | CONFIRMED |
| MTP | 5-layer DFlash-style drafter, 7 tokens per pass | CONFIRMED (vLLM recipe) |
| weights | experts MXFP4 (~150 GiB), compute FP8 | CONFIRMED (recipe); size reproduced by tests/test_fitplan.py |
| attention sinks | per-head sinks tensor | CONFIRMED (llama.cpp src/models/mimo2.cpp) |
| numerics | layer 47: some expert intermediates exceed fp16 range | PROVISIONAL (EXL3 builder note) — keep expert intermediates fp32/bf16 |
| llama.cpp | arch `mimo2`; V2.6 support specifically UNKNOWN until a GGUF is loaded |

Parity facts (CONFIRMED on the llama.cpp mimo2 graph, docs/evidence/2026-10-02-mimo-g1-parity.md): grouped GQA
(h // r), window rule p1 - p0 < n_swa, sinks in the denominator only, partial NEOX RoPE with its own base on SWA
layers, no SwiGLU clamp, routed weights unscaled unless the GGUF sets `expert_weights_scale` (UNKNOWN for real files).
