# GLM-5.3-Flash-DFlash2: fact sheet (draft model for speculative decoding)

Sources: vLLM @`bc21cba9673cfc2256a2726b4bc32f062237cd35` (Apache-2.0), read in the sandbox 2026-10-03;
model cards and pull-request descriptions found by web search. The vendor weights are gated, so their own
`config.json` has not been read: every value of that config is PROVISIONAL.

## What it is (CONFIRMED by reading the vLLM code at the pin)
| fact | source |
|---|---|
| DFlash2 = DFlash + a grouped convolution around attention and around the MLP of every drafter layer + a candidate selector | `vllm/model_executor/models/qwen3_dflash2.py`: `DFlash2Qwen3DecoderLayer` (:229), `DFlashGroupedConv` (:170), `CandidateSelector` (:314) |
| block size = 1 + number of speculative tokens | `qwen3_dflash2.py:256` |
| the drafter never runs its layers over the context: the target's hidden states at `target_layer_ids` are concatenated, projected by `fc`, RMS-normed, projected to K and V of every drafter layer, K RMS-normed, and written into the drafter's KV cache | `qwen3_dflash.py`: `combine_hidden_states` (:845), `_project_context_kv` (:580), `_normalize_context_k` (:620), `precompute_and_store_context_kv` (:632) |
| the drafted block is one forward pass of the drafter layers over the block positions (layers may be non-causal; per-layer causality comes from the config) | `qwen3_dflash.py`: `forward` (:709), `_dflash_layer_causal` (:85) |
| grouped convolution: within a block, position p mixes its own hidden state with those of positions p-1 .. p-(taps-1) of the same block, coefficients = learned base + an input-dependent delta | `qwen3_dflash2.py`: `_grouped_conv` (:144) |
| candidate selector: top-k candidate tokens per block position; edge score = unary logit + (predecessor_codebook[previous token] * projected hidden) . successor_codebook[candidate]; a walk from the anchor token picks one token per position | `qwen3_dflash2.py`: `_score_edges` (:290); `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`: `_selector_walk_kernel` (:16) |
| config keys the code reads: `conv_kernel_size`, `conv_group_size`, `selector_rank`, `selector_top_k`, `mask_token_id`, `target_hidden_size`, `use_aux_hidden_state`, `sliding_window`, `input_embedding_scale`, `output_multiplier`, `final_logit_softcapping` | grep of the three files |

## The vendor drafter `incoai/GLM-5.3-Flash-DFlash2` (PROVISIONAL: cards and PR texts, not its config)
| fact | source |
|---|---|
| 5 sliding-window layers, window 2048, block size 8, hence 7 drafted tokens per step | description of vLLM PR #56983; community cards ("num_speculative_tokens must be 7") |
| target hidden states taken at layers 5, 14, 24, 33, 42; selector rank 256, top-16 | card cfontes/GLM-5.3-Flash-DFlash2-TR3-v3 ("unchanged from the vendor drafter") |
| about 2.2 GB on disk; shares the target's embedding and output head | card cfontes/glm-5.3-flash-dflash2-tp4 |
| gated: manual approval; licence CC-BY-NC-ND-4.0 | model card (web search 2026-10-02) |
| published for SGLang; vLLM support of DFlash2 merged 2026-08-21 (PR #52816), the GLM pairing is PR #55423 (draft) and #56983 | tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark docs; PR pages |

Community-trained drafters exist too (canada-quant/GLM-5.3-Flash-DFlash2-E/F/G: 8 full-attention layers, about 1.84B
parameters, 6.2 GB bf16, 9 taps). Their licence: UNKNOWN (not read).

## Other people's measurements (PROVISIONAL, their hardware, never to be quoted as ours)
- 2x DGX Spark, NVFP4 target, vLLM TP2: 46.9 tok/s with DFlash2 against 21.8 tok/s with MTP-4; acceptance 74.1% (420 of 567).
- EXL3-quantised target: the acceptance ceiling is set by quantisation noise in the tapped hidden states, not by the
  drafter (card cfontes/...-TR3-v3). Relevant to ADR-010: a Q2 target may accept far fewer tokens.

## UNKNOWN
The vendor drafter's hidden size, head counts, convolution kernel and group sizes, mask token id, embedding scale; whether
our GGUF targets keep the tapped hidden states close enough for useful acceptance; acceptance and tok/s on our hardware.
