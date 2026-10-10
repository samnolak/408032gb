# N2: qwen4exp tiny-model CPU parity, 2026-10-10 (sandbox CPU, no GPU)

Task N2 of the hardware-free track (docs/reassessment/ROADMAP.md on branch `claude/relaxed-dijkstra-1d7i5o`; this
branch is cut from `main` = `eb59224`, which does not contain docs/reassessment/): a CPU reference for the
architecture of the reassessment's candidate #1, which llama.cpp implements as `qwen4exp`.

Reference: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (pin checked below), CPU build of
`tools/g0` target `dump-tensors`, the real `qwen4exp` graph (src/models/qwen4exp.cpp, 1458 lines; "G:<line>" below
means a line of that file at the pin; every cited line was printed from the pinned checkout before citing).
Oracle: ref/qwen4exp.py (float64; reuses ref/kda.py gated_delta_rule). Tests: tests/test_qwen4exp_tiny_parity.py
(dump parity, 17 tests), tests/test_qwen4exp_ref.py (oracle properties, 16 tests).
Script: tools/tinygen/run_parity_qwen4exp.sh. CI: step appended to `cpu-parity` in .github/workflows/ci.yml
(NOT RUN in GitHub Actions: the branch is not pushed).

## Tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0)

| part | value |
|---|---|
| layers | 8: gated delta net on 0-2 and 4-6, full attention on 3 and 7 (full_attention_interval 4, recurrent_layers array) |
| width | n_embd 64, vocab 96, rms eps 1e-6, all tensors f32 |
| hyper-connections | hc 4 streams, low rank 8, mixer before attention and before MoE in every layer, final mixer = output norm |
| full attention | 4 query heads, 2 KV heads, head dim 16, partial rotary 8 (4 pairs), IMRoPE sections [1,1,1,1], base 1000, gated query |
| QSA | layer 3: compress ratio 4 (kpool), indexer 4 heads x 16, budget top_k 8 = 2 pools + tail; layer 7: ratio 0 -> dense causal path |
| gated delta net | 2 key heads, 4 value heads, head dim 16, conv kernel 4 |
| MoE | 8 experts, top-2, expert ff 32, shared expert ff 48 with its sigmoid gate |
| PLE | layer 1; 3-grams, 2 heads per n-gram (4 heads x 16 = n_embd), conv kernel 3 (dilation 3), EOS id 2, head vocab sizes 37/41/43/47, 45-bit multipliers |
| MTP | none (nextn_predict_layers absent) |

Input: 64 token ids (`random.seed(5)`, ids in [3, 96)) with EOS (id 2) at positions 20, 41, 42, so the PLE n-gram
window reset is exercised for a single and a repeated EOS. Run flags `-c 512 -b 512 -ub 512 -t 1 -fa off -ctk f32
-ctv f32` (the precision contract of tests/test_glm_tiny_parity.py): one ubatch, prefill from an empty cache.

## Tolerances

Error metric per compared tensor: max|got - want| / max|want| (normwise relative; elementwise relative error is
ill-conditioned near zero), plus max abs and the argmax index, printed for every layer and component at the end of
the parity run. f32 paths: 1e-4 (agent contract: fp32 1e-4). Exact (0) for copies: the embedding gather and the PLE
table gather (which checks the host-side hash). Integer outputs (MoE top-k ids, QSA cell selection) must be equal
as sets; for QSA a different selection is accepted only if the oracle's own scores make it another valid top-k
(Q.qsa_tie_variant: same tail, whole visible pools, budget, boundary scores equal within f32 rounding); the run
reports how many tokens needed that (0 here). No tolerance was loosened to pass: the largest observed error is
7.4e-7 (end-to-end logits), more than 100x inside 1e-4.

## Findings (CONFIRMED by the run below unless labelled)

1. **Executed CPU paths.** The fused ops ran, not the unfused graph branches: `hc_gate-<il> (reshaped)` exists in
   the dump (only the fused `ggml_dsv4_hc_pre_gated` branch reshapes the gate, G:340-342) and `q_conv_predelta-0`
   has ne `[16,2,64,1]`, i.e. 2 key heads with no `ggml_repeat_4d` (G:1116-1120 is skipped when the fused gated
   delta net is on). There is no command-line switch for these: `cparams.fused_gdn_*` / `fused_dsv4_hc_*` are set
   true in src/llama-context.cpp:233-243 and only a device mismatch disables them (lines 562-580).
2. **cb() renames.** Calling cb twice on one tensor keeps the later name: `kqv_out` survives as `attn_pregate`
   (G:920, 1003), `v_conv` as `v_conv_predelta` (G:1107, 1124), the second `hc_combine` as `l_last` (G:495, 498).
   A dump consumer must ask for the surviving names (tools/tinygen/run_parity_qwen4exp.sh does).
3. **Routing.** softmax over the experts, top-k, weights renormalised over the chosen experts, **no scaling**:
   `grep -c EXPERT_WEIGHTS src/models/qwen4exp.cpp` = 0, so `hparams.expert_weights_scale` keeps its default 0
   (src/llama-hparams.h:124) and build_moe_ffn skips the scale (src/llama-graph.cpp:2171); no
   `ffn_moe_weights_scaled` tensor appears in the dump. Shared expert: SwiGLU, times sigmoid(w . x) per token.
4. **Gated delta net.** Output gate is sigmoid(z), not silu (G:641-645); value head h reads key head h % Hk (fused op,
   ggml/src/ggml-cpu/ops.cpp:11200), checked with Hk = 2 < Hv = 4; q, k L2-normalised with eps = rms_norm_epsilon.
5. **Hyper-connections.** mixed = mean over streams of rms_norm(stream) * gamma * sigmoid(gate), gate from a
   low-rank silu bottleneck scaled by 1/hc; combine weight 2*sigmoid(inject/hc) per stream.
6. **IMRoPE positions differ between the indexer query and the pooled indexer key.** Text queries/keys use M-RoPE
   positions (p, p, p, 0) (src/llama-graph.cpp:131-141); a pooled indexer key is rotated with its first member's
   (p0, y, x, p0) (src/llama-memory-hybrid-idx.cpp:931-937), and for a token batch the cell's y = x = p0
   (src/llama-batch.cpp:817-823, src/llama-kv-cache.cpp:1133-1139), so (p0, p0, p0, p0). The two differ only for a
   rotary pair in the 4th IMRoPE section. The tiny model puts pair 3 there on purpose and the dump confirms the
   asymmetry (mutants M2 and M12 below fail). For the converter's default sections [11, 11, 10, 0] with 64 rotary
   dims no pair falls in the 4th section (command below), so the difference is moot for that configuration. Whether
   the real model's sections leave a pair in the 4th section: UNKNOWN (needs the real GGUF's
   `qwen4exp.rope.dimension_sections` and `rope.dimension_count`). Whether the reference implementation (outside
   llama.cpp) rotates pooled keys the same way: UNKNOWN (not available offline).
7. **QSA.** pooled key = imrope(rms_norm(mean of the pool's raw indexer keys) * k_norm); score = sum over indexer
   heads of relu(q . k) / sqrt(idx_dim), visible only once the pool is complete; selection = top
   min(n_pool, top_k/kpool) pools + the incomplete tail kpool*floor((t+1)/kpool) .. t; the causal mask is added.
   53 of 64 tokens on layer 3 attend to fewer cells than causal attention would (reported in the table), 0 ties.
8. **PLE.** The host-side hash (uint64 multiply/xor, `% vocab + offset`) matches the oracle exactly for all 64 x 4
   rows, including the EOS reset and the sequence start (missing predecessors read as EOS).

## Coverage: worst error per component (from the final run below)

Aggregated from the per-component table of the final run (`python3 aggregate.py final_run.log`, a scratchpad
helper that takes the maximum over layers; layer -1 = global tensors and the final mixer). Every row passed
TOL = 1e-4; exact rows passed with 0.

| component | layers | max abs | max rel (worst layer) |
|---|---|---|---|
| attn_pregate = kqv_out (qsa) | 3 | 1.31e-07 | 1.22e-07 (L3) |
| gate_sigmoid | 3,7 | 6.62e-08 | 7.88e-08 (L7) |
| attn_gated | 3,7 | 4.88e-08 | 8.50e-08 (L3) |
| attn_output (wo) | 3,7 | 4.69e-08 | 8.20e-08 (L3) |
| attn_pregate = kqv_out (dense causal) | 7 | 1.11e-07 | 8.17e-08 (L7) |
| Qcur_full | 3,7 | 2.15e-07 | 1.21e-07 (L7) |
| Qcur_reshaped (q half) | 3,7 | 2.15e-07 | 1.21e-07 (L7) |
| gate_reshaped (gate half) | 3,7 | 1.79e-07 | 1.01e-07 (L7) |
| Qcur_normed | 3,7 | 4.56e-07 | 1.41e-07 (L3) |
| Kcur (proj) | 3,7 | 1.56e-07 | 8.44e-08 (L3) |
| Kcur_normed | 3,7 | 4.17e-07 | 1.45e-07 (L7) |
| Vcur | 3,7 | 1.62e-07 | 1.00e-07 (L3) |
| Qcur (IMRoPE) | 3,7 | 1.23e-06 | 4.00e-07 (L3) |
| Kcur (IMRoPE) | 3,7 | 1.02e-06 | 3.53e-07 (L7) |
| hc_init (exact) | -1 | 0.00e+00 | 0.00e+00 (L-1) |
| e2e l_last | 0,1,2,3,4,5,6,7 | 4.73e-06 | 5.82e-07 (L5) |
| e2e result_norm | -1 | 1.28e-06 | 6.95e-07 (L-1) |
| e2e logits | -1 | 1.49e-06 | 7.44e-07 (L-1) |
| q l2 norm | 0,1,2,4,5,6 | 8.66e-08 | 9.19e-08 (L1) |
| k l2 norm | 0,1,2,4,5,6 | 1.02e-07 | 1.09e-07 (L4) |
| v split | 0,1,2,4,5,6 | 0.00e+00 | 0.00e+00 (L0) |
| gdn scan (attn_output) | 0,1,2,4,5,6 | 1.44e-08 | 1.63e-07 (L1) |
| final_output (gated norm) | 0,1,2,4,5,6 | 4.04e-07 | 1.48e-07 (L2) |
| linear_attn_out | 0,1,2,4,5,6 | 2.05e-07 | 1.02e-07 (L4) |
| linear_attn_qkv_mixed | 0,1,2,4,5,6 | 2.25e-07 | 1.15e-07 (L6) |
| z | 0,1,2,4,5,6 | 2.39e-07 | 1.14e-07 (L4) |
| beta_sigmoid | 0,1,2,4,5,6 | 8.86e-08 | 1.14e-07 (L0) |
| alpha | 0,1,2,4,5,6 | 1.64e-07 | 1.17e-07 (L1) |
| a_softplus | 0,1,2,4,5,6 | 1.62e-07 | 1.02e-07 (L2) |
| gate (log decay) | 0,1,2,4,5,6 | 3.78e-07 | 1.14e-07 (L5) |
| conv_output_raw | 0,1,2,4,5,6 | 3.68e-07 | 1.20e-07 (L0) |
| conv_output_silu | 0,1,2,4,5,6 | 4.70e-07 | 2.09e-07 (L4) |
| hc_combine (attn) | 0,1,2,3,4,5,6,7 | 6.12e-07 | 8.31e-08 (L1) |
| hc_combine (ffn) = l_last | 0,1,2,3,4,5,6,7 | 5.17e-07 | 8.61e-08 (L0) |
| hc_norm[0] | 0,1,2,3,4,5,6,7,-1 | 5.34e-07 | 1.18e-07 (L3) |
| hc_gate[0] | 0,1,2,3,4,5,6,7,-1 | 1.76e-07 | 1.92e-07 (L3) |
| hc_mixed[0] | 0,1,2,3,4,5,6,7,-1 | 2.80e-07 | 1.46e-07 (L0) |
| hc_inject[0] | 0,1,2,3,4,5,6,7 | 5.26e-07 | 2.04e-07 (L7) |
| hc_norm[1] | 0,1,2,3,4,5,6,7 | 5.31e-07 | 1.18e-07 (L0) |
| hc_gate[1] | 0,1,2,3,4,5,6,7 | 1.72e-07 | 2.43e-07 (L1) |
| hc_mixed[1] | 0,1,2,3,4,5,6,7 | 2.28e-07 | 1.25e-07 (L1) |
| hc_inject[1] | 0,1,2,3,4,5,6,7 | 4.54e-07 | 1.74e-07 (L2) |
| result_output | -1 | 1.80e-07 | 8.99e-08 (L-1) |
| logits.bin | -1 | 1.80e-07 | 8.99e-08 (L-1) |
| ffn_moe_gate | 0,1,2,3,4,5,6,7 | 1.70e-07 | 9.52e-08 (L3) |
| ffn_moe_up | 0,1,2,3,4,5,6,7 | 1.66e-07 | 9.66e-08 (L0) |
| ffn_moe_swiglu | 0,1,2,3,4,5,6,7 | 1.07e-07 | 9.11e-08 (L4) |
| ffn_moe_down | 0,1,2,3,4,5,6,7 | 1.26e-07 | 1.81e-07 (L3) |
| ffn_moe_out (routed) | 0,1,2,3,4,5,6,7 | 9.88e-08 | 2.57e-07 (L1) |
| ffn_shexp | 0,1,2,3,4,5,6,7 | 1.42e-07 | 2.19e-07 (L7) |
| shared_expert_gate_sigmoid | 0,1,2,3,4,5,6,7 | 7.37e-08 | 9.84e-08 (L6) |
| ffn_out (routed + gated shared) | 0,1,2,3,4,5,6,7 | 1.24e-07 | 2.61e-07 (L2) |
| ffn_moe_logits | 0,1,2,3,4,5,6,7 | 1.84e-07 | 1.18e-07 (L2) |
| ffn_moe_probs (softmax) | 0,1,2,3,4,5,6,7 | 5.50e-08 | 1.34e-07 (L6) |
| ffn_moe_weights_norm | 0,1,2,3,4,5,6,7 | 9.40e-08 | 1.29e-07 (L2) |
| ple_embd = table[hash rows] (exact) | -1 | 0.00e+00 | 0.00e+00 (L-1) |
| ple_gate | 1 | 1.04e-07 | 1.25e-07 (L1) |
| ple_gated_value | 1 | 4.55e-07 | 1.28e-07 (L1) |
| ple_conv_out | 1 | 7.74e-07 | 1.53e-07 (L1) |
| ple out via hc_norm | 1 | 7.48e-07 | 1.74e-07 (L1) |
| indexer_k_raw | 3 | 1.66e-07 | 1.07e-07 (L3) |
| indexer_pool_k_new | 3 | 4.79e-07 | 1.79e-07 (L3) |
| indexer_k (gathered pools) | 3 | 4.79e-07 | 1.79e-07 (L3) |
| indexer_q | 3 | 1.03e-06 | 3.30e-07 (L3) |
| indexer_score (visible) | 3 | 5.20e-07 | 8.01e-08 (L3) |

Not in the table because they are integer or set comparisons: MoE top-k ids per layer (equal as sets on all 8
layers, per component and end to end), QSA selected cells on layer 3 (equal for 64/64 tokens, 0 tie variants),
pool visibility pattern and masked padded pools, greedy argmax of the logits (equal for 64/64 tokens).

## Not covered (UNKNOWN / NOT RUN)

| what | status | why / what would close it |
|---|---|---|
| MTP block (`graph_mtp`, LLM_GRAPH_TYPE_DECODER_MTP, G:529-618) | NOT COVERED | the tiny model has no nextn layer; needs an MTP context in the dump tool and a model with `nextn_predict_layers` = 1 |
| decode and several ubatches: carry of the GDN state, the GDN and PLE conv histories (G:1296-1348), the k-pool pooled-key cache across batches, new-pool marking during decode, PLE predecessors read from KV cells of earlier ubatches, rollback snapshots (`n_rs_seq` > 0) | NOT RUN | only one 64-token prefill ubatch from an empty cache; needs a second dump run with `-ub` smaller than the prompt and a token-by-token decode |
| several sequences, cells shared between sequences (`cache_safe` = false, G:798-803), multimodal batches (image token, 2D M-RoPE positions) | NOT COVERED | single text sequence only |
| unfused graph branches: HC mixer and combine (G:345-358, 388-393), chunked / autoregressive delta net (src/models/delta-net-base.cpp) | NOT EXERCISED | the CPU always takes the fused ops (finding 1); same math by the op contracts (ggml/include/ggml.h:2706-2720), not measured; a backend without the fused ops would take these branches |
| quantized weights (real files are k-quant GGUFs), f16 KV cache, flash attention | NOT COVERED | the tiny model is all f32 with f32 caches and `-fa off`; a quantized check must compare against the dequantized weights |
| GPU backends (CUDA kernels of these ops) | NOT RUN | no GPU in the sandbox |
| real model hyper-parameters (hc count, low rank, head sizes, IMRoPE sections, PLE heads and table size, layer pattern) | UNKNOWN | the tiny config only respects the loader's constraints (ple_head_dim * ple_n_heads = n_embd, ssm.inner_size = state_size * time_step_rank, compress ratio divides top_k); read the real GGUF's metadata |
| uint64 wrap of the PLE hash inside llama.cpp | NOT RUN against llama.cpp | the tiny dump never wraps; the oracle's wrap is checked against numpy uint64 only (property test). With multipliers below 2^45 (conversion/qwen4exp.py:62 says 45-bit) and token ids below 2^18 no product reaches 2^63 (PROVISIONAL arithmetic), so real text should never wrap |
| dense full-attention path in real files | PROVISIONAL: unused | the converter writes the QSA ratio on every full-attention layer (conversion/qwen4exp.py:83-89), so real files take QSA everywhere; the tiny model's dense layer 7 covers the branch anyway |
| equality of the llama.cpp graph with the original model code | UNKNOWN | this is oracle-versus-llama.cpp parity only; the original implementation is not available offline |
| the new CI step in GitHub Actions | NOT RUN | branch not pushed; the same command passes locally (below) |

## Commands and literal output

Environment: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 CPUs; c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0;
Python 3.13.16; numpy 2.5.3; Linux 6.18.44-fc-v114; cmake 3.28.3. `<llama.cpp>` =
`/tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad/llama.cpp` (read only), `<scratch>` = `/tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad/n2`.

### Pin and source checks

```
$ cd <llama.cpp> && git rev-parse HEAD && git status --short | wc -l && grep -c "EXPERT_WEIGHTS" src/models/qwen4exp.cpp
ec7630a640789c393694fb194f1bbbf0369fc62d
0
0
$ python3 -c "from ref import qwen4exp as Q; print('n_rot=64 sections [11,11,10,0]:', sorted(set(Q.imrope_sector_map(64, [11, 11, 10, 0]).tolist()))); print('n_rot=8  sections [1,1,1,1]  :', Q.imrope_sector_map(8, [1, 1, 1, 1]).tolist())"
n_rot=64 sections [11,11,10,0]: [0, 1, 2]
n_rot=8  sections [1,1,1,1]  : [0, 1, 2, 3]
```

### From an empty build directory (what the CI step does), at 7e77af4 (ref and parity test as at HEAD)

```
$ TMPDIR=<scratch> bash -c 'time tools/tinygen/run_parity_qwen4exp.sh <llama.cpp> <scratch>/../build-n2-fresh' > fresh_run.log 2>&1; echo "exit=$?" >> fresh_run.log
$ tail -n 9 fresh_run.log
Ran 17 tests in 0.234s

OK
artifacts: /tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad/n2/tmp.qVGCIVvOb1

real	2m7.165s
user	6m48.222s
sys	0m21.147s
exit=0
```

### Final run at 67217b9 (full `-v` output with the per-component table)

```
$ git log --oneline -n 1
67217b9 [parity-qa] ref test: PLE hash wraps mod 2^64 exactly like uint64_t products (the tiny dump never wraps); 16 property tests
$ TMPDIR=<scratch> tools/tinygen/run_parity_qwen4exp.sh <llama.cpp> <scratch>/../build-n2-fresh > final_run.log 2>&1; echo "exit=$?"
exit=0
$ cat final_run.log
CMAKE_BUILD_TYPE=Release
wrote /tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad/n2/tmp.JpXoIQ0QTJ/qwen4exp.gguf
test_attention_core_and_gate (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attention_core_and_gate) ... ok
test_attn_projections_norms_rope (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attn_projections_norms_rope) ... ok
test_embedding_hc_init (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_embedding_hc_init) ... ok
test_end_to_end (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_end_to_end)
Run the oracle from the token ids alone. Discrete choices (MoE top-k, QSA pools) are the oracle's ... ok
test_gdn_heads_differ (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_gdn_heads_differ) ... ok
test_gdn_l2norm_scan_and_output (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_gdn_l2norm_scan_and_output) ... ok
test_gdn_projections_and_gates (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_gdn_projections_and_gates) ... ok
test_hc_combine (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_hc_combine) ... ok
test_hc_mixers (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_hc_mixers) ... ok
test_logits (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_logits) ... ok
test_moe_experts_and_shared (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_moe_experts_and_shared) ... ok
test_moe_router (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_moe_router) ... ok
test_ple_hash_rows (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_ple_hash_rows) ... ok
test_ple_module (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_ple_module) ... ok
test_qsa_indexer_and_selection (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_qsa_indexer_and_selection) ... ok
test_qsa_mask_is_sparse (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_qsa_mask_is_sparse) ... ok
test_rope_sections_matter (tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_rope_sections_matter) ... ok

per-component errors (rel = max|got-want| / max|want|):
component                                        layer    max abs        rel  argmax
attn_pregate = kqv_out (qsa)                         3  1.313e-07  1.223e-07  (35, 20)
gate_sigmoid                                         3  6.622e-08  7.808e-08  (63, 4)
attn_gated                                           3  4.876e-08  8.502e-08  (47, 40)
attn_output (wo)                                     3  4.687e-08  8.202e-08  (0, 37)
attn_pregate = kqv_out (dense causal)                7  1.106e-07  8.171e-08  (5, 41)
gate_sigmoid                                         7  6.578e-08  7.875e-08  (61, 47)
attn_gated                                           7  4.779e-08  5.915e-08  (0, 26)
attn_output (wo)                                     7  4.670e-08  8.008e-08  (1, 13)
Qcur_full                                            3  1.839e-07  1.038e-07  (9, 110)
Qcur_reshaped (q half)                               3  1.839e-07  1.160e-07  (9, 3, 14)
gate_reshaped (gate half)                            3  1.606e-07  9.069e-08  (22, 43)
Qcur_normed                                          3  4.557e-07  1.413e-07  (44, 0, 6)
Kcur (proj)                                          3  1.401e-07  8.437e-08  (22, 6)
Kcur_normed                                          3  3.834e-07  1.149e-07  (3, 0, 15)
Vcur                                                 3  1.621e-07  1.004e-07  (50, 26)
Qcur (IMRoPE)                                        3  1.226e-06  4.004e-07  (50, 3, 5)
Kcur (IMRoPE)                                        3  7.906e-07  2.368e-07  (48, 0, 5)
Qcur_full                                            7  2.153e-07  1.205e-07  (48, 0)
Qcur_reshaped (q half)                               7  2.153e-07  1.205e-07  (48, 0, 0)
gate_reshaped (gate half)                            7  1.793e-07  1.011e-07  (26, 40)
Qcur_normed                                          7  4.561e-07  1.413e-07  (51, 3, 4)
Kcur (proj)                                          7  1.564e-07  7.885e-08  (22, 23)
Kcur_normed                                          7  4.172e-07  1.445e-07  (54, 1, 11)
Vcur                                                 7  1.585e-07  8.654e-08  (10, 24)
Qcur (IMRoPE)                                        7  9.693e-07  3.080e-07  (58, 2, 5)
Kcur (IMRoPE)                                        7  1.019e-06  3.530e-07  (59, 0, 5)
hc_init (exact)                                     -1  0.000e+00  0.000e+00  (0, 0, 0)
e2e l_last                                           0  1.437e-06  3.136e-07  (50, 0, 54)
e2e l_last                                           1  1.851e-06  2.508e-07  (43, 1, 14)
e2e l_last                                           2  2.856e-06  3.540e-07  (41, 3, 43)
e2e l_last                                           3  3.043e-06  3.749e-07  (41, 3, 21)
e2e l_last                                           4  3.896e-06  5.078e-07  (5, 0, 6)
e2e l_last                                           5  4.728e-06  5.819e-07  (55, 2, 43)
e2e l_last                                           6  4.622e-06  5.396e-07  (5, 0, 6)
e2e l_last                                           7  4.673e-06  5.426e-07  (5, 0, 6)
e2e result_norm                                     -1  1.280e-06  6.950e-07  (5, 6)
e2e logits                                          -1  1.490e-06  7.439e-07  (55, 87)
e2e: QSA tie replays = 0 layer(s)                   -1  0.000e+00  0.000e+00  ()
q l2 norm                                            0  8.549e-08  8.927e-08  (61, 0, 6)
k l2 norm                                            0  9.171e-08  9.477e-08  (19, 0, 12)
v split                                              0  0.000e+00  0.000e+00  (0, 0, 0)
gdn scan (attn_output)                               0  1.285e-08  1.086e-07  (25, 0, 8)
final_output (gated norm)                            0  3.109e-07  1.079e-07  (11, 6)
linear_attn_out                                      0  1.724e-07  8.327e-08  (56, 22)
q l2 norm                                            1  8.665e-08  9.193e-08  (55, 0, 1)
k l2 norm                                            1  1.017e-07  1.031e-07  (44, 1, 6)
v split                                              1  0.000e+00  0.000e+00  (0, 0, 0)
gdn scan (attn_output)                               1  1.438e-08  1.627e-07  (23, 3, 9)
final_output (gated norm)                            1  3.258e-07  1.355e-07  (34, 38)
linear_attn_out                                      1  1.750e-07  9.944e-08  (32, 62)
q l2 norm                                            2  8.390e-08  8.808e-08  (61, 1, 2)
k l2 norm                                            2  8.914e-08  9.651e-08  (58, 1, 11)
v split                                              2  0.000e+00  0.000e+00  (0, 0, 0)
gdn scan (attn_output)                               2  1.043e-08  1.079e-07  (22, 1, 4)
final_output (gated norm)                            2  3.481e-07  1.480e-07  (24, 25)
linear_attn_out                                      2  1.749e-07  1.019e-07  (45, 2)
q l2 norm                                            4  7.940e-08  8.596e-08  (56, 1, 0)
k l2 norm                                            4  1.014e-07  1.086e-07  (18, 0, 1)
v split                                              4  0.000e+00  0.000e+00  (0, 0, 0)
gdn scan (attn_output)                               4  1.130e-08  9.938e-08  (43, 3, 0)
final_output (gated norm)                            4  3.519e-07  1.259e-07  (3, 23)
linear_attn_out                                      4  2.053e-07  1.023e-07  (19, 16)
q l2 norm                                            5  7.803e-08  8.147e-08  (49, 1, 1)
k l2 norm                                            5  9.482e-08  1.078e-07  (2, 0, 0)
v split                                              5  0.000e+00  0.000e+00  (0, 0, 0)
gdn scan (attn_output)                               5  1.256e-08  1.346e-07  (54, 2, 10)
final_output (gated norm)                            5  2.547e-07  1.072e-07  (29, 54)
linear_attn_out                                      5  1.689e-07  8.383e-08  (4, 3)
q l2 norm                                            6  8.396e-08  8.952e-08  (26, 1, 7)
k l2 norm                                            6  8.057e-08  8.793e-08  (29, 1, 11)
v split                                              6  0.000e+00  0.000e+00  (0, 0, 0)
gdn scan (attn_output)                               6  1.309e-08  7.335e-08  (30, 3, 4)
final_output (gated norm)                            6  4.042e-07  1.388e-07  (46, 22)
linear_attn_out                                      6  1.882e-07  9.946e-08  (28, 37)
linear_attn_qkv_mixed                                0  1.856e-07  9.349e-08  (46, 94)
z                                                    0  1.622e-07  8.505e-08  (43, 15)
beta_sigmoid                                         0  8.860e-08  1.140e-07  (12, 0)
alpha                                                0  1.534e-07  9.848e-08  (63, 2)
a_softplus                                           0  1.306e-07  6.272e-08  (55, 2)
gate (log decay)                                     0  2.649e-07  7.271e-08  (55, 2)
conv_output_raw                                      0  3.684e-07  1.199e-07  (24, 43)
conv_output_silu                                     0  2.966e-07  1.010e-07  (26, 43)
linear_attn_qkv_mixed                                1  1.796e-07  8.366e-08  (47, 0)
z                                                    1  1.767e-07  9.392e-08  (19, 53)
beta_sigmoid                                         1  6.405e-08  8.229e-08  (43, 1)
alpha                                                1  1.636e-07  1.167e-07  (55, 0)
a_softplus                                           1  1.453e-07  7.688e-08  (19, 2)
gate (log decay)                                     1  2.579e-07  1.009e-07  (17, 1)
conv_output_raw                                      1  3.139e-07  1.121e-07  (22, 121)
conv_output_silu                                     1  2.891e-07  1.095e-07  (44, 54)
linear_attn_qkv_mixed                                2  1.986e-07  1.120e-07  (12, 70)
z                                                    2  1.557e-07  8.501e-08  (31, 47)
beta_sigmoid                                         2  7.427e-08  9.149e-08  (32, 0)
alpha                                                2  1.546e-07  9.231e-08  (26, 3)
a_softplus                                           2  1.618e-07  1.016e-07  (58, 1)
gate (log decay)                                     2  2.029e-07  1.094e-07  (25, 0)
conv_output_raw                                      2  2.648e-07  1.173e-07  (28, 101)
conv_output_silu                                     2  2.480e-07  1.402e-07  (17, 3)
linear_attn_qkv_mixed                                4  2.009e-07  9.489e-08  (41, 99)
z                                                    4  2.394e-07  1.141e-07  (1, 11)
beta_sigmoid                                         4  6.513e-08  8.541e-08  (43, 0)
alpha                                                4  1.296e-07  1.025e-07  (46, 0)
a_softplus                                           4  1.171e-07  8.490e-08  (37, 1)
gate (log decay)                                     4  2.339e-07  9.998e-08  (29, 0)
conv_output_raw                                      4  2.555e-07  1.094e-07  (11, 78)
conv_output_silu                                     4  3.876e-07  2.088e-07  (11, 78)
linear_attn_qkv_mixed                                5  1.955e-07  9.514e-08  (22, 121)
z                                                    5  1.438e-07  8.229e-08  (10, 42)
beta_sigmoid                                         5  7.157e-08  8.813e-08  (52, 0)
alpha                                                5  9.613e-08  5.965e-08  (0, 1)
a_softplus                                           5  1.570e-07  9.002e-08  (50, 3)
gate (log decay)                                     5  3.783e-07  1.138e-07  (50, 3)
conv_output_raw                                      5  2.962e-07  1.132e-07  (15, 15)
conv_output_silu                                     5  3.123e-07  1.716e-07  (15, 15)
linear_attn_qkv_mixed                                6  2.246e-07  1.155e-07  (26, 59)
z                                                    6  1.591e-07  8.678e-08  (30, 55)
beta_sigmoid                                         6  6.517e-08  8.511e-08  (37, 0)
alpha                                                6  1.536e-07  8.728e-08  (10, 2)
a_softplus                                           6  1.334e-07  6.629e-08  (57, 0)
gate (log decay)                                     6  1.177e-07  8.463e-08  (57, 0)
conv_output_raw                                      6  2.332e-07  8.748e-08  (37, 86)
conv_output_silu                                     6  4.703e-07  1.887e-07  (37, 86)
hc_combine (attn)                                    0  2.950e-07  6.441e-08  (36, 3, 46)
hc_combine (ffn) = l_last                            0  3.945e-07  8.610e-08  (36, 3, 46)
hc_combine (attn)                                    1  6.124e-07  8.305e-08  (15, 2, 60)
hc_combine (ffn) = l_last                            1  5.077e-07  6.881e-08  (6, 0, 44)
hc_combine (attn)                                    2  3.088e-07  3.828e-08  (47, 0, 38)
hc_combine (ffn) = l_last                            2  4.960e-07  6.149e-08  (47, 0, 38)
hc_combine (attn)                                    3  2.493e-07  3.039e-08  (2, 3, 2)
hc_combine (ffn) = l_last                            3  4.483e-07  5.524e-08  (46, 2, 35)
hc_combine (attn)                                    4  3.651e-07  4.822e-08  (3, 3, 49)
hc_combine (ffn) = l_last                            4  5.174e-07  6.743e-08  (3, 1, 49)
hc_combine (attn)                                    5  3.627e-07  4.461e-08  (54, 0, 18)
hc_combine (ffn) = l_last                            5  4.382e-07  5.392e-08  (33, 0, 63)
hc_combine (attn)                                    6  3.290e-07  3.917e-08  (33, 0, 6)
hc_combine (ffn) = l_last                            6  4.691e-07  5.476e-08  (53, 0, 19)
hc_combine (attn)                                    7  2.445e-07  2.875e-08  (18, 3, 4)
hc_combine (ffn) = l_last                            7  4.580e-07  5.317e-08  (13, 0, 63)
hc_norm[0]                                           0  4.020e-07  1.099e-07  (28, 139)
hc_gate[0]                                           0  1.163e-07  1.374e-07  (23, 96)
hc_mixed[0]                                          0  2.799e-07  1.457e-07  (28, 11)
hc_inject[0]                                         0  4.572e-07  1.627e-07  (11, 3)
hc_norm[1]                                           0  3.911e-07  1.181e-07  (52, 163)
hc_gate[1]                                           0  1.645e-07  1.894e-07  (18, 199)
hc_mixed[1]                                          0  1.519e-07  9.655e-08  (32, 25)
hc_inject[1]                                         0  4.119e-07  1.705e-07  (11, 1)
hc_norm[0]                                           1  4.820e-07  1.123e-07  (57, 171)
hc_gate[0]                                           1  1.004e-07  1.367e-07  (35, 160)
hc_mixed[0]                                          1  1.678e-07  9.426e-08  (21, 47)
hc_inject[0]                                         1  5.122e-07  1.536e-07  (9, 0)
hc_norm[1]                                           1  5.310e-07  1.130e-07  (31, 236)
hc_gate[1]                                           1  1.457e-07  2.433e-07  (6, 120)
hc_mixed[1]                                          1  2.279e-07  1.254e-07  (59, 50)
hc_inject[1]                                         1  3.934e-07  1.603e-07  (37, 1)
hc_norm[0]                                           2  4.989e-07  1.030e-07  (36, 238)
hc_gate[0]                                           2  1.141e-07  1.836e-07  (9, 151)
hc_mixed[0]                                          2  1.596e-07  8.508e-08  (12, 6)
hc_inject[0]                                         2  4.057e-07  1.521e-07  (58, 3)
hc_norm[1]                                           2  5.014e-07  1.139e-07  (55, 98)
hc_gate[1]                                           2  1.221e-07  1.479e-07  (14, 113)
hc_mixed[1]                                          2  1.437e-07  8.189e-08  (36, 46)
hc_inject[1]                                         2  4.288e-07  1.737e-07  (15, 0)
hc_norm[0]                                           3  5.344e-07  1.177e-07  (1, 195)
hc_gate[0]                                           3  1.762e-07  1.924e-07  (31, 8)
hc_mixed[0]                                          3  1.934e-07  1.078e-07  (1, 3)
hc_inject[0]                                         3  3.105e-07  9.262e-08  (10, 3)
hc_norm[1]                                           3  4.549e-07  1.052e-07  (59, 178)
hc_gate[1]                                           3  9.643e-08  1.288e-07  (42, 17)
hc_mixed[1]                                          3  1.620e-07  8.779e-08  (45, 2)
hc_inject[1]                                         3  3.594e-07  1.107e-07  (14, 3)
hc_norm[0]                                           4  4.190e-07  1.006e-07  (1, 195)
hc_gate[0]                                           4  1.497e-07  1.505e-07  (7, 97)
hc_mixed[0]                                          4  1.540e-07  8.083e-08  (21, 58)
hc_inject[0]                                         4  3.937e-07  1.222e-07  (36, 1)
hc_norm[1]                                           4  4.316e-07  1.086e-07  (59, 50)
hc_gate[1]                                           4  1.509e-07  1.995e-07  (11, 21)
hc_mixed[1]                                          4  1.479e-07  8.056e-08  (44, 26)
hc_inject[1]                                         4  4.542e-07  1.479e-07  (52, 0)
hc_norm[0]                                           5  3.993e-07  9.915e-08  (20, 63)
hc_gate[0]                                           5  1.311e-07  1.885e-07  (10, 168)
hc_mixed[0]                                          5  1.653e-07  8.967e-08  (27, 6)
hc_inject[0]                                         5  4.978e-07  1.797e-07  (45, 0)
hc_norm[1]                                           5  4.004e-07  9.219e-08  (0, 62)
hc_gate[1]                                           5  1.375e-07  1.894e-07  (18, 36)
hc_mixed[1]                                          5  1.660e-07  7.739e-08  (45, 53)
hc_inject[1]                                         5  3.745e-07  1.281e-07  (22, 0)
hc_norm[0]                                           6  4.051e-07  9.157e-08  (24, 136)
hc_gate[0]                                           6  1.331e-07  1.812e-07  (38, 146)
hc_mixed[0]                                          6  2.047e-07  9.413e-08  (28, 48)
hc_inject[0]                                         6  4.517e-07  1.458e-07  (63, 1)
hc_norm[1]                                           6  3.974e-07  9.601e-08  (0, 62)
hc_gate[1]                                           6  1.291e-07  2.061e-07  (48, 166)
hc_mixed[1]                                          6  1.505e-07  8.410e-08  (40, 44)
hc_inject[1]                                         6  4.171e-07  1.471e-07  (60, 2)
hc_norm[0]                                           7  4.251e-07  9.939e-08  (62, 14)
hc_gate[0]                                           7  1.225e-07  1.672e-07  (58, 0)
hc_mixed[0]                                          7  2.580e-07  1.325e-07  (50, 6)
hc_inject[0]                                         7  5.256e-07  2.040e-07  (57, 1)
hc_norm[1]                                           7  3.864e-07  9.263e-08  (15, 27)
hc_gate[1]                                           7  1.716e-07  2.141e-07  (41, 1)
hc_mixed[1]                                          7  1.752e-07  9.129e-08  (60, 7)
hc_inject[1]                                         7  4.473e-07  1.450e-07  (30, 0)
hc_norm[0]                                          -1  4.320e-07  1.046e-07  (7, 13)
hc_gate[0]                                          -1  1.072e-07  1.640e-07  (20, 33)
hc_mixed[0]                                         -1  1.936e-07  1.051e-07  (50, 6)
result_output                                       -1  1.801e-07  8.992e-08  (46, 28)
logits.bin                                          -1  1.801e-07  8.992e-08  (46, 28)
ffn_moe_gate                                         0  1.510e-07  7.982e-08  (25, 1, 17)
ffn_moe_up                                           0  1.663e-07  9.658e-08  (41, 1, 1)
ffn_moe_swiglu                                       0  5.680e-08  3.910e-08  (45, 1, 0)
ffn_moe_down                                         0  1.149e-07  1.362e-07  (45, 1, 29)
ffn_moe_out (routed)                                 0  7.182e-08  1.732e-07  (52, 44)
ffn_shexp                                            0  8.197e-08  1.180e-07  (15, 6)
shared_expert_gate_sigmoid                           0  5.049e-08  6.691e-08  (46,)
ffn_out (routed + gated shared)                      0  9.197e-08  1.701e-07  (52, 44)
ffn_moe_gate                                         1  1.694e-07  9.210e-08  (6, 0, 0)
ffn_moe_up                                           1  1.522e-07  8.843e-08  (43, 1, 20)
ffn_moe_swiglu                                       1  6.804e-08  6.247e-08  (61, 0, 17)
ffn_moe_down                                         1  1.087e-07  1.758e-07  (28, 0, 16)
ffn_moe_out (routed)                                 1  9.881e-08  2.566e-07  (44, 31)
ffn_shexp                                            1  9.798e-08  1.543e-07  (16, 44)
shared_expert_gate_sigmoid                           1  6.248e-08  8.457e-08  (5,)
ffn_out (routed + gated shared)                      1  1.243e-07  2.600e-07  (44, 31)
ffn_moe_gate                                         2  1.704e-07  7.146e-08  (49, 1, 24)
ffn_moe_up                                           2  1.661e-07  8.738e-08  (34, 0, 27)
ffn_moe_swiglu                                       2  5.871e-08  6.192e-08  (31, 0, 3)
ffn_moe_down                                         2  9.601e-08  1.649e-07  (1, 0, 17)
ffn_moe_out (routed)                                 2  9.235e-08  2.114e-07  (20, 30)
ffn_shexp                                            2  9.380e-08  1.607e-07  (39, 44)
shared_expert_gate_sigmoid                           2  5.105e-08  7.214e-08  (58,)
ffn_out (routed + gated shared)                      2  1.128e-07  2.608e-07  (20, 30)
ffn_moe_gate                                         3  1.636e-07  9.519e-08  (24, 1, 31)
ffn_moe_up                                           3  1.436e-07  8.667e-08  (20, 0, 1)
ffn_moe_swiglu                                       3  9.148e-08  8.501e-08  (48, 1, 14)
ffn_moe_down                                         3  1.256e-07  1.807e-07  (48, 1, 6)
ffn_moe_out (routed)                                 3  7.584e-08  1.884e-07  (51, 61)
ffn_shexp                                            3  7.360e-08  1.631e-07  (11, 30)
shared_expert_gate_sigmoid                           3  5.232e-08  6.870e-08  (20,)
ffn_out (routed + gated shared)                      3  8.450e-08  1.856e-07  (22, 63)
ffn_moe_gate                                         4  1.646e-07  9.472e-08  (29, 1, 30)
ffn_moe_up                                           4  1.492e-07  8.219e-08  (23, 0, 31)
ffn_moe_swiglu                                       4  1.071e-07  9.106e-08  (24, 1, 22)
ffn_moe_down                                         4  1.239e-07  1.796e-07  (54, 1, 24)
ffn_moe_out (routed)                                 4  6.824e-08  1.739e-07  (42, 32)
ffn_shexp                                            4  1.167e-07  1.582e-07  (15, 18)
shared_expert_gate_sigmoid                           4  5.560e-08  8.032e-08  (62,)
ffn_out (routed + gated shared)                      4  8.176e-08  1.496e-07  (12, 15)
ffn_moe_gate                                         5  1.652e-07  8.108e-08  (62, 0, 20)
ffn_moe_up                                           5  1.517e-07  8.797e-08  (44, 1, 7)
ffn_moe_swiglu                                       5  7.734e-08  6.937e-08  (58, 0, 7)
ffn_moe_down                                         5  8.315e-08  1.037e-07  (35, 1, 17)
ffn_moe_out (routed)                                 5  7.185e-08  1.891e-07  (32, 38)
ffn_shexp                                            5  9.383e-08  1.677e-07  (21, 43)
shared_expert_gate_sigmoid                           5  5.987e-08  8.319e-08  (17,)
ffn_out (routed + gated shared)                      5  7.470e-08  1.669e-07  (32, 38)
ffn_moe_gate                                         6  1.428e-07  6.872e-08  (32, 1, 15)
ffn_moe_up                                           6  1.508e-07  8.616e-08  (12, 1, 13)
ffn_moe_swiglu                                       6  7.639e-08  7.275e-08  (12, 1, 13)
ffn_moe_down                                         6  1.131e-07  1.785e-07  (46, 0, 26)
ffn_moe_out (routed)                                 6  7.612e-08  2.052e-07  (23, 32)
ffn_shexp                                            6  1.420e-07  2.139e-07  (36, 49)
shared_expert_gate_sigmoid                           6  7.371e-08  9.839e-08  (0,)
ffn_out (routed + gated shared)                      6  9.381e-08  2.117e-07  (46, 26)
ffn_moe_gate                                         7  1.681e-07  8.473e-08  (18, 1, 10)
ffn_moe_up                                           7  1.633e-07  8.793e-08  (39, 0, 8)
ffn_moe_swiglu                                       7  7.628e-08  5.703e-08  (15, 1, 4)
ffn_moe_down                                         7  8.247e-08  1.079e-07  (28, 1, 34)
ffn_moe_out (routed)                                 7  7.789e-08  1.752e-07  (52, 15)
ffn_shexp                                            7  1.161e-07  2.185e-07  (50, 33)
shared_expert_gate_sigmoid                           7  4.700e-08  6.356e-08  (10,)
ffn_out (routed + gated shared)                      7  7.502e-08  1.450e-07  (11, 6)
ffn_moe_logits                                       0  1.449e-07  1.029e-07  (33, 7)
ffn_moe_probs (softmax)                              0  4.035e-08  9.129e-08  (28, 7)
ffn_moe_weights_norm                                 0  6.703e-08  8.361e-08  (60, 0)
ffn_moe_logits                                       1  1.150e-07  7.839e-08  (19, 0)
ffn_moe_probs (softmax)                              1  4.539e-08  1.136e-07  (9, 5)
ffn_moe_weights_norm                                 1  7.014e-08  9.303e-08  (60, 0)
ffn_moe_logits                                       2  1.772e-07  1.182e-07  (27, 5)
ffn_moe_probs (softmax)                              2  3.995e-08  1.036e-07  (7, 1)
ffn_moe_weights_norm                                 2  9.401e-08  1.294e-07  (31, 0)
ffn_moe_logits                                       3  1.161e-07  7.610e-08  (40, 0)
ffn_moe_probs (softmax)                              3  3.205e-08  9.109e-08  (3, 0)
ffn_moe_weights_norm                                 3  7.188e-08  9.839e-08  (43, 0)
ffn_moe_logits                                       4  1.525e-07  8.146e-08  (59, 7)
ffn_moe_probs (softmax)                              4  3.109e-08  8.990e-08  (28, 7)
ffn_moe_weights_norm                                 4  7.562e-08  1.062e-07  (5, 0)
ffn_moe_logits                                       5  1.345e-07  8.246e-08  (44, 4)
ffn_moe_probs (softmax)                              5  2.973e-08  7.392e-08  (45, 3)
ffn_moe_weights_norm                                 5  6.573e-08  9.358e-08  (45, 0)
ffn_moe_logits                                       6  1.841e-07  1.132e-07  (39, 4)
ffn_moe_probs (softmax)                              6  4.714e-08  1.340e-07  (53, 1)
ffn_moe_weights_norm                                 6  8.492e-08  1.205e-07  (40, 0)
ffn_moe_logits                                       7  1.541e-07  9.215e-08  (39, 5)
ffn_moe_probs (softmax)                              7  5.505e-08  1.265e-07  (39, 5)
ffn_moe_weights_norm                                 7  5.917e-08  7.686e-08  (36, 1)
ple_embd = table[hash rows] (exact)                 -1  0.000e+00  0.000e+00  (0, 0)
ple_gate                                             1  1.038e-07  1.251e-07  (51, 0)
ple_gated_value                                      1  4.552e-07  1.278e-07  (12, 3, 6)
ple_conv_out                                         1  7.745e-07  1.533e-07  (12, 1, 56)
ple out via hc_norm                                  1  7.477e-07  1.743e-07  (18, 6)
indexer_k_raw                                        3  1.660e-07  1.068e-07  (33, 9)
indexer_pool_k_new                                   3  4.789e-07  1.791e-07  (12, 5)
indexer_k (gathered pools)                           3  4.789e-07  1.791e-07  (12, 5)
indexer_q                                            3  1.027e-06  3.302e-07  (58, 1, 1)
indexer_score (visible)                              3  5.197e-07  8.012e-08  (180,)
qsa selection: tokens differing only by tie order = 0     3  0.000e+00  0.000e+00  ()
qsa selection: tokens attending to fewer than all causal cells = 53     3  0.000e+00  0.000e+00  ()

----------------------------------------------------------------------
Ran 17 tests in 0.247s

OK
artifacts: /tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad/n2/tmp.JpXoIQ0QTJ
```

The first line (`CMAKE_BUILD_TYPE=Release`) is printed by the llama.cpp CMake configure on stderr; the script
silences only stdout of cmake, like run_parity.sh and run_parity_mimo.sh.

Dump of that run, for findings 1 and 3 (fused paths; no routing-weight scale):

```
$ cd <scratch>/tmp.JpXoIQ0QTJ/dump && cat "q_conv_predelta-0__0.json" "v_conv_predelta-0__0.json" "hc_gate-0 (reshaped)__0.json"; ls | grep -c "ffn_moe_weights_scaled"; ls | wc -l
{"type":"f32","ne":[16,2,64,1]}
{"type":"f32","ne":[16,4,64,1]}
{"type":"f32","ne":[64,4,64,1]}
0
1023
```

### Oracle property tests, and what the CI `unit tests` step (discover) sees

```
$ python3 -m unittest tests.test_qwen4exp_ref -v
test_short_conv_last_tap_is_identity (tests.test_qwen4exp_ref.Gdn.test_short_conv_last_tap_is_identity) ... ok
test_value_head_reads_key_head_mod_hk (tests.test_qwen4exp_ref.Gdn.test_value_head_reads_key_head_mod_hk) ... ok
test_mixer_of_identical_streams_with_zero_gate (tests.test_qwen4exp_ref.HyperConnections.test_mixer_of_identical_streams_with_zero_gate) ... ok
test_zero_inject_is_plain_residual_add (tests.test_qwen4exp_ref.HyperConnections.test_zero_inject_is_plain_residual_add) ... ok
test_softmax_route (tests.test_qwen4exp_ref.Moe.test_softmax_route) ... ok
test_dilated_conv_taps (tests.test_qwen4exp_ref.Ple.test_dilated_conv_taps) ... ok
test_eos_and_sequence_start_reset_the_window (tests.test_qwen4exp_ref.Ple.test_eos_and_sequence_start_reset_the_window) ... ok
test_hash_matches_uint64_arithmetic (tests.test_qwen4exp_ref.Ple.test_hash_matches_uint64_arithmetic) ... ok
test_hash_wraps_like_uint64 (tests.test_qwen4exp_ref.Ple.test_hash_wraps_like_uint64) ... ok
test_budget_covering_every_pool_is_causal_attention (tests.test_qwen4exp_ref.Qsa.test_budget_covering_every_pool_is_causal_attention) ... ok
test_pool_is_mean_of_raw_keys (tests.test_qwen4exp_ref.Qsa.test_pool_is_mean_of_raw_keys) ... ok
test_selection_is_top_pools_plus_tail (tests.test_qwen4exp_ref.Qsa.test_selection_is_top_pools_plus_tail) ... ok
test_tie_variant_accepts_only_genuine_ties (tests.test_qwen4exp_ref.Qsa.test_tie_variant_accepts_only_genuine_ties) ... ok
test_fourth_section_uses_the_fourth_position (tests.test_qwen4exp_ref.Rope.test_fourth_section_uses_the_fourth_position) ... ok
test_imrope_equals_neox_when_sections_cover_all_pairs (tests.test_qwen4exp_ref.Rope.test_imrope_equals_neox_when_sections_cover_all_pairs) ... ok
test_rotation_preserves_pair_norms (tests.test_qwen4exp_ref.Rope.test_rotation_preserves_pair_norms) ... ok

----------------------------------------------------------------------
Ran 16 tests in 0.020s

OK
$ python3 -m unittest discover -s tests -p "test_qwen4exp*"
................sssssssssssssssss
----------------------------------------------------------------------
Ran 33 tests in 0.017s

OK (skipped=17)
```

## Mutation check

Each mutant is one single-point edit of the oracle (ref/qwen4exp.py, or ref/kda.py for M13), applied to a copy
of ref/ and the two test files (tools/tinygen/mutate_qwen4exp.py; the repository is never edited); both test
modules run against the same dump.

### Round 1 (HEAD 7f53a1d; ref and tests as committed in 400b25c): 13/13 caught, but with a hole

Run with the same mutant list from a scratchpad copy of the driver, against the first dump of this session
(`<scratch>/dump`, made by hand with the same generator, seed, token ids and run flags as the script, and a
superset of its DUMP_PREFIXES).

```
$ python3 <scratch>/mutate.py <repo> <scratch>/dump <scratch>/tiny.gguf <llama.cpp>/gguf-py <scratch>/mut
baseline (no mutation): rc=0 Ran 31 tests in 0.277s | OK
M1 hc_combine: drop the 1/hc inside the scatter sigmoid
    ref/qwen4exp.py: 'w = 2.0 * sigmoid(np.asarray(inject) / hc)' -> 'w = 2.0 * sigmoid(np.asarray(inject))'
    rc=1 Ran 31 tests in 0.300s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_hc_combine
M2 pooled indexer keys rotated with text positions (4th = 0, not p0)
    ref/qwen4exp.py: 'return np.stack([p0, p0, p0, p0], axis=1)' -> 'return np.stack([p0, p0, p0, np.zeros_like(p0)], axis=1)'
    rc=1 Ran 31 tests in 0.297s | FAILED (failures=1)
    failing: test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection
M3 PLE hash: an EOS predecessor no longer resets the window
    ref/qwen4exp.py: 'cut = cut or prev is None or prev == eos' -> 'cut = cut or prev is None'
    rc=1 Ran 31 tests in 0.360s | FAILED (failures=3)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_ref.test_eos_and_sequence_start_reset_the_window, test_qwen4exp_tiny_parity.test_ple_hash_rows
M4 GDN output gate silu(z) (Qwen3.5) instead of sigmoid(z)
    ref/qwen4exp.py: 'return rms_norm(scan, eps) * np.asarray(w_norm) * sigmoid(z)' -> 'return rms_norm(scan, eps) * np.asarray(w_norm) * silu(z)'
    rc=1 Ran 31 tests in 0.301s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_gdn_l2norm_scan_and_output
M5 MoE: routing weights not renormalised over the top-k
    ref/qwen4exp.py: 'wts = wts / np.maximum(wts.sum(-1, keepdims=True), 6.103515625e-5)' -> 'wts = wts / 1.0'
    rc=1 Ran 31 tests in 0.296s | FAILED (failures=4)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_moe_experts_and_shared, test_qwen4exp_tiny_parity.test_moe_router, test_qwen4exp_ref.test_softmax_route
M6 HC mixer: sum over streams instead of mean
    ref/qwen4exp.py: 'mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1) / hc' -> 'mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1)'
    rc=1 Ran 31 tests in 0.314s | FAILED (failures=3)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_hc_mixers, test_qwen4exp_ref.test_mixer_of_identical_streams_with_zero_gate
M7 QSA score: no relu on the head scores
    ref/qwen4exp.py: 's = np.maximum(np.einsum("thd,bd->thb", iq, pooled), 0.0).sum(axis=1) / np.sqrt(d)' -> 's = np.einsum("thd,bd->thb", iq, pooled).sum(axis=1) / np.sqrt(d)'
    rc=1 Ran 31 tests in 0.322s | FAILED (failures=1)
    failing: test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection
M8 IMRoPE: off-by-one in the t-section bound (pair 3 leaves the 4th section)
    ref/qwen4exp.py: 'elif sector % 3 == 0 and sector < 3 * s[0]:' -> 'elif sector % 3 == 0 and sector <= 3 * s[0]:'
    rc=1 Ran 31 tests in 0.345s | FAILED (failures=5)
    failing: test_qwen4exp_tiny_parity.test_attn_projections_norms_rope, test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_ref.test_fourth_section_uses_the_fourth_position, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection, test_qwen4exp_tiny_parity.test_rope_sections_matter
M9 PLE conv: dilation 1 instead of the n-gram size
    ref/qwen4exp.py: 'w_conv, dtype=np.float64), ngram)' -> 'w_conv, dtype=np.float64), 1)'
    rc=1 Ran 31 tests in 0.401s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_ple_module
M10 QSA tail: tail starts at kpool*floor(t/kpool) instead of kpool*floor((t+1)/kpool)
    ref/qwen4exp.py: 'tail0 = ((t + 1) // kpool) * kpool' -> 'tail0 = (t // kpool) * kpool'
    rc=1 Ran 31 tests in 0.316s | FAILED (failures=2)
    failing: test_qwen4exp_ref.test_budget_covering_every_pool_is_causal_attention, test_qwen4exp_ref.test_selection_is_top_pools_plus_tail
M11 shared expert: gate without sigmoid
    ref/qwen4exp.py: 'sh_gate = sigmoid(x @ w["ffn_gate_inp_shexp.weight"])' -> 'sh_gate = x @ w["ffn_gate_inp_shexp.weight"]'
    rc=1 Ran 31 tests in 0.295s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_moe_experts_and_shared
M12 text M-RoPE positions (p, p, p, p) instead of (p, p, p, 0)
    ref/qwen4exp.py: 'return np.stack([p, p, p, np.zeros_like(p)], axis=1)' -> 'return np.stack([p, p, p, p], axis=1)'
    rc=1 Ran 31 tests in 0.364s | FAILED (failures=4)
    failing: test_qwen4exp_tiny_parity.test_attn_projections_norms_rope, test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_ref.test_fourth_section_uses_the_fourth_position, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection
M13 GDN head map: value head h reads key head h // (Hv/Hk) instead of h % Hk
    ref/kda.py: 'hk = h % H_k' -> 'hk = h // (H_v // H_k)'
    rc=1 Ran 31 tests in 0.372s | FAILED (failures=3)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_gdn_l2norm_scan_and_output, test_qwen4exp_ref.test_value_head_reads_key_head_mod_hk
caught 13/13
```

Hole found: M10 (wrong QSA tail) failed only the property tests, not the dump parity test; M2 (pooled-key
position) and M7 (no relu) failed the per-component QSA test but not the end-to-end test. Cause: the selection
test validated a differing reference selection against a tail rule written inline in the test, and the
end-to-end test replayed any differing selection without checking it. Fix (7e77af4): both use the oracle's
Q.qsa_tail and Q.qsa_tie_variant, and the end-to-end test validates a tie before replaying it. Tolerances were
not changed.

### Final round: 13/13 caught, and every mutant now fails the dump parity module itself

Run at HEAD 67217b9 with tools/tinygen/mutate_qwen4exp.py uncommitted; the file was then committed unchanged as
dc05f27 (`git diff --stat dc05f27 -- tools/tinygen/mutate_qwen4exp.py ref tests` prints nothing).

```
$ python3 tools/tinygen/mutate_qwen4exp.py <repo> <scratch>/tmp.JpXoIQ0QTJ/dump <scratch>/tmp.JpXoIQ0QTJ/qwen4exp.gguf <llama.cpp>/gguf-py <scratch>/mut
baseline (no mutation): rc=0 Ran 33 tests in 0.269s | OK
M1 hc_combine: drop the 1/hc inside the scatter sigmoid
    ref/qwen4exp.py: 'w = 2.0 * sigmoid(np.asarray(inject) / hc)' -> 'w = 2.0 * sigmoid(np.asarray(inject))'
    rc=1 Ran 33 tests in 0.251s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_hc_combine
M2 pooled indexer keys rotated with text positions (4th = 0, not p0)
    ref/qwen4exp.py: 'return np.stack([p0, p0, p0, p0], axis=1)' -> 'return np.stack([p0, p0, p0, np.zeros_like(p0)], axis=1)'
    rc=1 Ran 33 tests in 0.264s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection
M3 PLE hash: an EOS predecessor no longer resets the window
    ref/qwen4exp.py: 'cut = cut or prev is None or prev == eos' -> 'cut = cut or prev is None'
    rc=1 Ran 33 tests in 0.282s | FAILED (failures=3)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_ref.test_eos_and_sequence_start_reset_the_window, test_qwen4exp_tiny_parity.test_ple_hash_rows
M4 GDN output gate silu(z) (Qwen3.5) instead of sigmoid(z)
    ref/qwen4exp.py: 'return rms_norm(scan, eps) * np.asarray(w_norm) * sigmoid(z)' -> 'return rms_norm(scan, eps) * np.asarray(w_norm) * silu(z)'
    rc=1 Ran 33 tests in 0.241s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_gdn_l2norm_scan_and_output
M5 MoE: routing weights not renormalised over the top-k
    ref/qwen4exp.py: 'wts = wts / np.maximum(wts.sum(-1, keepdims=True), 6.103515625e-5)' -> 'wts = wts / 1.0'
    rc=1 Ran 33 tests in 0.245s | FAILED (failures=4)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_moe_experts_and_shared, test_qwen4exp_tiny_parity.test_moe_router, test_qwen4exp_ref.test_softmax_route
M6 HC mixer: sum over streams instead of mean
    ref/qwen4exp.py: 'mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1) / hc' -> 'mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1)'
    rc=1 Ran 33 tests in 0.253s | FAILED (failures=3)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_hc_mixers, test_qwen4exp_ref.test_mixer_of_identical_streams_with_zero_gate
M7 QSA score: no relu on the head scores
    ref/qwen4exp.py: 's = np.maximum(np.einsum("thd,bd->thb", iq, pooled), 0.0).sum(axis=1) / np.sqrt(d)' -> 's = np.einsum("thd,bd->thb", iq, pooled).sum(axis=1) / np.sqrt(d)'
    rc=1 Ran 33 tests in 0.413s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection
M8 IMRoPE: off-by-one in the t-section bound (pair 3 leaves the 4th section)
    ref/qwen4exp.py: 'elif sector % 3 == 0 and sector < 3 * s[0]:' -> 'elif sector % 3 == 0 and sector <= 3 * s[0]:'
    rc=1 Ran 33 tests in 0.325s | FAILED (failures=5)
    failing: test_qwen4exp_tiny_parity.test_attn_projections_norms_rope, test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_ref.test_fourth_section_uses_the_fourth_position, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection, test_qwen4exp_tiny_parity.test_rope_sections_matter
M9 PLE conv: dilation 1 instead of the n-gram size
    ref/qwen4exp.py: 'w_conv, dtype=np.float64), ngram)' -> 'w_conv, dtype=np.float64), 1)'
    rc=1 Ran 33 tests in 0.394s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_ple_module
M10 QSA tail: tail starts at kpool*floor(t/kpool) instead of kpool*floor((t+1)/kpool)
    ref/qwen4exp.py: 'tail0 = ((t + 1) // kpool) * kpool' -> 'tail0 = (t // kpool) * kpool'
    rc=1 Ran 33 tests in 0.279s | FAILED (failures=4)
    failing: test_qwen4exp_ref.test_budget_covering_every_pool_is_causal_attention, test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection, test_qwen4exp_ref.test_selection_is_top_pools_plus_tail
M11 shared expert: gate without sigmoid
    ref/qwen4exp.py: 'sh_gate = sigmoid(x @ w["ffn_gate_inp_shexp.weight"])' -> 'sh_gate = x @ w["ffn_gate_inp_shexp.weight"]'
    rc=1 Ran 33 tests in 0.328s | FAILED (failures=2)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_moe_experts_and_shared
M12 text M-RoPE positions (p, p, p, p) instead of (p, p, p, 0)
    ref/qwen4exp.py: 'return np.stack([p, p, p, np.zeros_like(p)], axis=1)' -> 'return np.stack([p, p, p, p], axis=1)'
    rc=1 Ran 33 tests in 0.282s | FAILED (failures=4)
    failing: test_qwen4exp_tiny_parity.test_attn_projections_norms_rope, test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_ref.test_fourth_section_uses_the_fourth_position, test_qwen4exp_tiny_parity.test_qsa_indexer_and_selection
M13 GDN head map: value head h reads key head h // (Hv/Hk) instead of h % Hk
    ref/kda.py: 'hk = h % H_k' -> 'hk = h // (H_v // H_k)'
    rc=1 Ran 33 tests in 0.257s | FAILED (failures=3)
    failing: test_qwen4exp_tiny_parity.test_end_to_end, test_qwen4exp_tiny_parity.test_gdn_l2norm_scan_and_output, test_qwen4exp_ref.test_value_head_reads_key_head_mod_hk
caught 13/13
```

