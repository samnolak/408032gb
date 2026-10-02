# DeepSeek-V4.1-Flash — fact sheet (third target)

| item | value | label |
|---|---|---|
| params | 552B backbone + 196B Engram; 8B active prefill / 16B decode | CONFIRMED (model card) |
| layers | 40, Causal Encoder-Decoder 20 + 20; hidden 5120 | CONFIRMED |
| MoE | 384 routed, top-6, 1 shared, expert FF 2304, scoring sqrtsoftplus | CONFIRMED (config.json) |
| residual | mHC, 4 streams, Sinkhorn | CONFIRMED (NeMo docs, MLX build) |
| attention | SWA + CSA2 with a two-level indexer, compression ratios 1 and 2 | CONFIRMED (vLLM recipe) |
| Engram | layers 1 and 14; 384,006,168 and 384,016,682 rows x 256; 8 heads x 2-,3-,4-gram hashes (24 rows per layer per token); FP8 + E8M0/32 = 189.1 GiB | CONFIRMED (inference/config.json @2cba9e4; corrected 2026-10-02: the tables differ in size) |
| speculation | DSpark: target layers 37-39, 128 experts top-3 | CONFIRMED (config.json) |
| weights | routed experts MXFP4 = 268.9 GiB; does not fit VRAM+RAM here | CONFIRMED + tests/test_fitplan.py |
| llama.cpp | no deepseek41 graph at ec7630a; references: DeepSeek inference/, SGLang, DwarfStar (Metal), PipeNetwork MLX | CONFIRMED |

Engram is the same kind of table Strata already streams from SSD for Qwen3.8 (hashed n-grams,
prime-sized tables, a few rows per token): include/strata/kernels/ngram.hpp is the starting point.
NVIDIA NeMo docs state Engram reuses the Qwen3.8 Flash Next row-owner embedding.
