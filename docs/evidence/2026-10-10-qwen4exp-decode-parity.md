# N6: qwen4exp several ubatches, token-by-token decode and rollback, CPU parity, 2026-10-10 (sandbox CPU, no GPU)

Task N6 of the hardware-free track (docs/reassessment/ROADMAP.md row N6): close the "decode and several ubatches" gap
of N2 (docs/evidence/2026-10-10-qwen4exp-tiny-parity.md, "Not covered" table). Branch `claude/n6-qwen4exp-decode`,
cut from `claude/relaxed-dijkstra-1d7i5o` at `c1a0336` (which contains N2).

Reference: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (pin check below), CPU build of `tools/g0` target
`dump-tensors`, the real `qwen4exp` graph. "G:<line>" = a line of src/models/qwen4exp.cpp at the pin; every cited
line was printed from the pinned checkout before citing.

| piece | file |
|---|---|
| oracle | ref/qwen4exp.py: N2's `forward()` unchanged, plus the incremental path `forward_chunk` / `cache_truncate` / `forward_chunks` (state carried between ubatches and calls, each item cited to where llama.cpp keeps it) |
| dump parity tests | tests/test_qwen4exp_decode_parity.py: 10 tests per run, 4 runs = 40; skip without `QWEN4EXP_DECODE_DUMP`, `QWEN4EXP_TINY_GGUF`, `LLAMA_CPP_GGUF_PY` |
| property tests | tests/test_qwen4exp_chunked_ref.py: 4 tests, numpy only, random 4-layer model (runs in the CI `unit tests` step) |
| driver | tools/tinygen/run_parity_qwen4exp_decode.sh (builds dump-tensors, writes the N2 tiny GGUF, runs the 4 dumps, runs both test modules) |
| dump tool | tools/dump/dump_tensors.cpp: opt-in `DUMP_PER_GRAPH`, `DUMP_CALLS`, `DUMP_N_RS_SEQ`; default output byte-identical (below) |
| mutation check | tools/tinygen/mutate_qwen4exp_decode.py (12 mutants on copies; the repository is never edited) |
| CI | step appended to `cpu-parity` in .github/workflows/ci.yml: NOT RUN in GitHub Actions (branch not pushed) |

## Runs

Model: the N2 tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0: 8 layers, gated delta net on 0-2 and 4-6, PLE on
layer 1, QSA on layer 3 with compress ratio 4 and budget 2 pools + tail, dense full attention on layer 7). Flags as N2:
`-c 512 -b 512 -t 1 -fa off -ctk f32 -ctv f32`. One 64-token sequence (`random.seed(6)`, ids in [3, 96)) with EOS (id 2)
at 15, 33, 47, 55, 56, 59, 61, so every run has a graph whose first token reads an EOS predecessor from an earlier graph.

| run | `-ub` | llama_process calls (`DUMP_CALLS`) | graphs | what it adds |
|---|---|---|---|---|
| ub16 | 16 | one call of 64 tokens | 4 x 16 tokens | carry across ubatches whose boundaries coincide with pool boundaries |
| ub10 | 10 | one call of 64 tokens | 6 x 10 + 1 x 4 | pools that straddle two ubatches (completed from raw keys of the previous ubatch), a short last ubatch |
| decode | 512 | `48,1,...,1`: prefill 48, then 16 calls of one token | 1 x 48 + 16 x 1 | decode with n_tokens = 1 per call: GDN state and conv history, PLE history and predecessors, KV cache, a pool completed by a one-token call (positions 51, 55, 59, 63), EOS decoded as a single token (55, 56, 59, 61) |
| rollback | 512, `n_rs_seq` 3 | `48,4,r2,1,...,1`: prefill 48; a 4-token draft 48..51 whose ids at 50-51 are wrong (44, 85 instead of 27, 68); `llama_memory_seq_rm(mem, 0, 50, -1)`; 14 calls of one token | 1 + 1 + 14 | rollback from the recurrent snapshots (src/llama-memory-recurrent.cpp:199-215), pool 12 first pooled from wrong keys and pooled again, KV cells re-used |

## Method

`DUMP_PER_GRAPH=1` writes the dumps of every computed graph (one per ubatch) into `g<NNN>/`, and `manifest.json` maps
the llama_process calls to graphs and positions. Every graph is compared row by row with

* (A) `[full]`: the full-sequence oracle `Q.forward(final 64 tokens)` at the graph's positions. Causality: an
  incremental result must equal the full recompute. Rows of the rejected draft tokens have no (A);
* (B) `[chunk]`: the incremental oracle `Q.forward_chunk`, replaying the same calls (rollback = `Q.cache_truncate`).
  The state each graph read is compared too: `conv_state_at-<il>` (GDN conv history; on the PLE layer occurrence 0
  is the PLE conv history, G:1423 runs before the mixer), `state_predelta-<il>` (delta-rule state, G:1073-1074),
  `new_state-<il>` (written back, `n_rs_seq` = 0 only, src/models/delta-net-base.cpp:546-560) and `indexer_k-3`
  (pooled keys gathered from the k-pool cache, G:797). For these state tensors (A) is derived independently of the
  incremental path: rows of the full-sequence qkv / PLE values before the graph, and the delta rule run over the
  first p0 tokens of the full-sequence q, k, v, g, beta.

(A) and (B) agree within 1e-10 relative (tests/test_qwen4exp_chunked_ref.py, six chunkings incl. one token per
call), so their error columns below are equal to the printed digits; (B) is what the mutation check of the state
carry bites on, (A) is the literal causality check.
Unlike N2's per-component tests (which recompute each op from the dumped inputs), every N6 comparison is against the
oracle run end to end from the token ids, so the errors include propagation through the earlier layers.

## Tolerances

As N2: normwise relative error max|got - want| / max|want| < 1e-4 for every f32 path; exact (0) for the embedding
gather, the PLE table gather (host-side hash, predecessors read from the KV cells of earlier ubatches/calls) and the
zero state at the sequence start; MoE top-k ids equal as sets; QSA selection equal, or another valid top-k by the
oracle's own scores (`Q.qsa_tie_variant`), counted and replayed (0 occurred). Since 2362cbe a difference is replayed
only if the oracle's own selection is also a valid top-k for the token (finding 9). No tolerance was loosened: the
largest error is 5.4e-6 (`final_output`, the gated RMS norm of the delta-rule output, layer 6), about 18x inside 1e-4.

## Findings (CONFIRMED by the runs below unless labelled)

1. **One graph per ubatch, detected from the callback.** `ggml_backend_sched_compute_splits` asks the eval callback
   about every node once, in node order (ggml/src/ggml-backend.cpp:1821-1848), so a graph starts where the first node
   asked (`model.input_embed`) comes round again. Every call computed ceil(n_tokens / n_ubatch) graphs (the tool fails
   otherwise) and each graph's `hc_init` rows are exactly its tokens (test_graphs_and_token_mapping). Every graph has
   1034 nodes; with `n_rs_seq` = 3 every graph has 1118 (PROVISIONAL attribution of the 84 extra nodes: the snapshot
   writes of G:1324-1345 and src/models/delta-net-base.cpp:563-603). In the default mode (no `DUMP_PER_GRAPH`,
   `DUMP_CALLS`, `DUMP_N_RS_SEQ`) the new tool writes the same 1022 files, byte for byte, as the binary built from
   the previous dump_tensors.cpp.
2. **Carried state = state of the prefix.** In every graph with p0 > 0 the GDN conv history (`conv_state_at`, the
   last 3 rows of the conv input), the delta-rule state (`state_predelta`) and the PLE conv history (the last 6 rows of
   the grouped-normed gated value) equal the full-sequence values before p0; at p0 = 0 they are exactly zero; the
   written-back `new_state` equals the state after the graph. Worst: 1.5e-6 (delta-rule state), 7.6e-7 (GDN conv
   history), 1.8e-7 (PLE conv history).
3. **Layouts.** `conv_state_at` is `[K-1, C]` with time fastest; the delta-rule state is `[S_v, S_v, H_v]` with S[i][j]
   at `[j*S_v + i]` (ggml/src/ggml-cpu/ops.cpp:11228), so a dump reads as the oracle's `state[h].T` (ref/kda.py keeps
   S[h, i, j]). On the PLE layer `conv_state_at-1` occurrence 0 is the PLE history ([6, 256]) and occurrence 1 the GDN
   history ([3, 128]).
4. **k-pool cache.** A pool is pooled once, in the graph that completes it (`indexer_pool_k_new`: rows
   [p0/4, (p0+n)/4) of the pool list), including pools completed by a one-token call and pools straddling two
   ubatches (ub10); every complete pool is read back from the cache in later graphs (`indexer_k`, worst 3.7e-7).
   The new-pool list is padded to n_new_g = max(n_new, 1, n_tokens/4 + 1) rows
   (src/llama-memory-hybrid-idx.cpp:769-770); a padded row re-pools the first member of pool 0 if some pool is new,
   else the ubatch's first token, rotated at position 0 (lines 947-967, 898), i.e. `rms_norm(k_raw[cell]) * k_norm`,
   checked for every padded row (worst 6.7e-7).
5. **PLE predecessors across graphs.** The gathered PLE rows are exactly the full-sequence rows in every graph,
   including first tokens whose predecessors (one an EOS) lie in an earlier ubatch or call.
6. **Rollback.** With `n_rs_seq` = 3 llama.cpp accepts the partial `llama_memory_seq_rm(0, 50, -1)` after the 4-token
   draft; the next graph (position 50) reads the delta-rule state, GDN and PLE conv histories of the kept 50-token
   prefix, writes no `new_state` (snapshot path), and all later graphs equal the full-sequence oracle of the final
   sequence. Pool 12 (48..51), first pooled from the wrong keys of the draft, is pooled again when position 51 is
   decoded and the cached value equals the correct one. The rows of the rejected tokens equal the incremental oracle
   that replays them.
7. **KV cell = position** for this single sequence, also after the rollback: the QSA selection read from
   `kq_mask_qsa` as cell indices equals the oracle's selection as positions in every graph.
8. **QSA in decode.** Selections equal the oracle's in all 258 rows of the four runs (64 + 64 + 64 + 66), 0 tie
   replays; the budget binds in the last graphs (a token attends to fewer cells than causal attention would).
9. **Hole found by the mutation check, fixed in 2362cbe.** In the first mutation round the dump tests did not catch
   S9 (QSA tail taken relative to the ubatch): the reference's selection differed from the mutant oracle's, was judged
   a valid top-k by `Q.qsa_tie_variant`, and was replayed over it. Now both selections must be valid top-k sets for
   the token's absolute position; the second round catches 12/12 in the dump tests alone. Tolerances unchanged.
10. **The same hole exists in N2's tests (not fixed here).** A mutant confined to `Q.qsa_select` (one pool fewer
    selected) passes all 17 N2 dump tests on the N2 dump; only the N2 property test
    `test_selection_is_top_pools_plus_tail` fails (output below). N2's M10 was caught because it mutates `qsa_tail`,
    which the validation shares. Suggested fix (for the lead, outside N6): in `_tie_variants` of
    tests/test_qwen4exp_tiny_parity.py also require `Q.qsa_tie_variant(scores[t], mine[t], t, ...)` to hold.

## Coverage: worst error per component (from the final run below)

Aggregated from the per-run table of the final run (`python3 aggregate.py head_run.log`, a scratchpad helper that takes
the maximum over the four runs; the per-run table is in the log). Every row passed TOL = 1e-4; exact rows passed with 0.
The aggregate of the fresh-build run at 675d429 is identical (`diff` below).

```
component                                                  run      layer graph    max abs        rel
final_output (gated norm) [full]                           rollback     6    12  6.621e-06  5.422e-06
final_output (gated norm) [chunk]                          rollback     6    12  6.621e-06  5.422e-06
linear_attn_out [full]                                     ub10         6     5  6.761e-06  4.284e-06
linear_attn_out [chunk]                                    ub10         6     5  6.761e-06  4.284e-06
gdn scan (attn_output) [full]                              rollback     6     9  8.872e-08  2.509e-06
gdn scan (attn_output) [chunk]                             rollback     6     9  8.872e-08  2.509e-06
attn_pregate (dense causal) [full]                         ub10         7     5  7.243e-07  2.449e-06
attn_pregate (dense causal) [chunk]                        ub10         7     5  7.243e-07  2.449e-06
Qcur (IMRoPE) [full]                                       ub10         7     5  5.441e-06  1.966e-06
Qcur (IMRoPE) [chunk]                                      ub10         7     5  5.441e-06  1.966e-06
ffn_out (routed + gated shared) [full]                     decode       7    15  3.985e-07  1.927e-06
ffn_out (routed + gated shared) [chunk]                    decode       7    15  3.985e-07  1.927e-06
Kcur (IMRoPE) [full]                                       ub10         7     5  5.480e-06  1.801e-06
Kcur (IMRoPE) [chunk]                                      ub10         7     5  5.480e-06  1.801e-06
attn_output (wo) [full]                                    decode       7    10  1.800e-07  1.775e-06
attn_output (wo) [chunk]                                   decode       7    10  1.800e-07  1.775e-06
state_predelta (delta-rule state) [full]                   decode       6    16  5.307e-07  1.454e-06
state_predelta (delta-rule state) [chunk]                  decode       6    16  5.307e-07  1.454e-06
new_state (written back) [chunk]                           decode       6    15  5.307e-07  1.454e-06
new_state (written back) [full]                            decode       6    15  5.307e-07  1.454e-06
result_norm [full]                                         ub10        -1     5  1.899e-06  1.340e-06
result_norm [chunk]                                        ub10        -1     5  1.899e-06  1.340e-06
Vcur [full]                                                decode       7    10  1.251e-06  1.240e-06
Vcur [chunk]                                               decode       7    10  1.251e-06  1.240e-06
l_last (layer output) [full]                               ub10         7     5  8.107e-06  1.223e-06
l_last (layer output) [chunk]                              ub10         7     5  8.107e-06  1.223e-06
result_output [full]                                       ub10        -1     5  1.921e-06  1.160e-06
result_output [chunk]                                      ub10        -1     5  1.921e-06  1.160e-06
logits.bin [full]                                          ub10        -1     5  1.921e-06  1.160e-06
logits.bin [chunk]                                         ub10        -1     5  1.921e-06  1.160e-06
hc_combine (attn) [full]                                   ub10         7     5  7.705e-06  1.155e-06
hc_combine (attn) [chunk]                                  ub10         7     5  7.705e-06  1.155e-06
conv_output_silu [full]                                    rollback     2     8  6.937e-07  1.118e-06
conv_output_silu [chunk]                                   rollback     2     8  6.937e-07  1.118e-06
indexer_q [full]                                           ub10         3     2  2.651e-06  9.491e-07
indexer_q [chunk]                                          ub10         3     2  2.651e-06  9.491e-07
conv_output_raw [full]                                     decode       6    15  1.165e-06  9.316e-07
conv_output_raw [chunk]                                    decode       6    15  1.165e-06  9.316e-07
attn_pregate (qsa) [full]                                  ub10         3     2  6.605e-07  9.090e-07
attn_pregate (qsa) [chunk]                                 ub10         3     2  6.605e-07  9.090e-07
conv_state_at (GDN conv history) [full]                    ub16         6     2  9.827e-07  7.644e-07
conv_state_at (GDN conv history) [chunk]                   ub16         6     2  9.827e-07  7.644e-07
indexer_score (visible) [chunk]                            ub10         3     4  3.591e-06  7.249e-07
indexer_pool_k_new (padding rows)                          rollback     3     2  1.414e-06  6.715e-07
indexer_k_raw [full]                                       rollback     3     2  4.705e-07  6.132e-07
indexer_k_raw [chunk]                                      rollback     3     2  4.705e-07  6.132e-07
ple_gated_value [full]                                     decode       1     8  6.223e-07  5.663e-07
ple_gated_value [chunk]                                    decode       1     8  6.223e-07  5.663e-07
ple_gate [full]                                            decode       1    14  2.417e-07  5.025e-07
ple_gate [chunk]                                           decode       1    14  2.417e-07  5.025e-07
indexer_pool_k_new (pools completed here) [chunk]          ub10         3     2  8.941e-07  4.110e-07
indexer_pool_k_new (pools completed here) [full]           ub10         3     2  8.941e-07  4.110e-07
indexer_k (pool cache, all complete pools) [chunk]         ub10         3     2  8.941e-07  3.743e-07
indexer_k (pool cache, all complete pools) [full]          ub10         3     2  8.941e-07  3.743e-07
ple_conv_out [full]                                        rollback     1     1  6.950e-07  2.496e-07
ple_conv_out [chunk]                                       rollback     1     1  6.950e-07  2.496e-07
conv_state_at (PLE conv history) [full]                    ub16         1     2  5.777e-07  1.835e-07
conv_state_at (PLE conv history) [chunk]                   ub16         1     2  5.777e-07  1.835e-07
hc_init (exact)                                            decode      -1     0  0.000e+00  0.000e+00
ple_embd = table[rows] (exact) [full]                      decode      -1     0  0.000e+00  0.000e+00
ple_embd = table[rows] (exact) [chunk]                     decode      -1     0  0.000e+00  0.000e+00
qsa selection: tie replays = 0                             decode       3    -1  0.000e+00  0.000e+00
components: 62; worst rel overall: 5.422e-06
```

Integer and set comparisons, not in the table: MoE top-k ids equal as sets in every graph and layer of the 4 runs (A
and B); QSA selections equal in every row (0 tie replays); pool visibility and masked padded pools; greedy argmax of
the logits equal in every row.

## Mutation check

Each mutant is one single-point edit of the incremental path of ref/qwen4exp.py, applied to a copy of ref/ and the
test files (tools/tinygen/mutate_qwen4exp_decode.py); both test modules run against the dumps of the final run.
"dump parity failing" lists the failing tests of tests/test_qwen4exp_decode_parity.py (`<run class>.<test>`, or
`setUpClass` when the mutant oracle raised), "property failing" those of tests/test_qwen4exp_chunked_ref.py.

| mutant | what is broken | caught by the dump tests | caught by the property tests |
|---|---|---|---|
| S1 | delta-rule state reset at every ubatch / call | yes (all 4 runs) | yes |
| S2 | GDN conv history dropped | yes (all 4 runs) | yes |
| S3 | PLE conv history dropped | yes (all 4 runs) | yes |
| S4 | PLE predecessors from earlier ubatches off by one | yes (all 4 runs, incl. the exact row test) | yes |
| S5 | pooled keys of earlier ubatches / calls forgotten | yes (all 4 runs) | yes |
| S6 | new pool rotated to its index within the ubatch | yes (all 4 runs) | yes |
| S7 | dense attention forgets cells of earlier ubatches / calls (KV cache) | yes (all 4 runs) | yes |
| S8 | RoPE positions restart in every ubatch / call | yes (all 4 runs) | yes |
| S9 | QSA tail relative to the ubatch | yes (all 4 runs; missed before 2362cbe, finding 9) | yes |
| S10 | QSA pool visibility relative to the ubatch | yes (ub16 tests; the mutant oracle raises in setUpClass of the other 3 runs: an empty cell set) | yes |
| S11 | rollback keeps the delta-rule state and conv history of the rejected tokens | yes (rollback run) | yes |
| S12 | rollback keeps the PLE conv history of the rejected tokens | yes (rollback run) | yes |

```
baseline (no mutation): rc=0 Ran 44 tests in 2.361s | OK
S1 GDN: delta-rule state reset at every ubatch / call
    ref/qwen4exp.py: 'out, state = gated_delta_rule(q, k, v, g, beta, state=state_in, normalize_qk=True, eps=eps)' -> 'out, state = gated_delta_rule(q, k, v, g, beta, state=None, normalize_qk=True, eps=eps)'
    rc=1 Ran 44 tests in 1.069s | FAILED (failures=31)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S2 GDN: conv history of the earlier ubatches dropped (zeros)
    ref/qwen4exp.py: 'conv_raw, conv_hist = short_conv_hist(qkv, w["ssm_conv1d.weight"], conv_in)' -> 'conv_raw, conv_hist = short_conv_hist(qkv, w["ssm_conv1d.weight"], np.zeros_like(conv_in))'
    rc=1 Ran 44 tests in 0.777s | FAILED (failures=31)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S3 PLE: conv history of the earlier ubatches dropped (zeros)
    ref/qwen4exp.py: 'full = np.vstack([hist_in, normed])' -> 'full = np.vstack([np.zeros_like(hist_in), normed])'
    rc=1 Ran 44 tests in 0.912s | FAILED (failures=31)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S4 PLE: predecessors from earlier ubatches off by one (newest cached token skipped)
    ref/qwen4exp.py: 'prefix = [int(t) for t in history[max(0, len(history) - n_prev):]]' -> 'prefix = [int(t) for t in history[max(0, len(history) - n_prev - 1):len(history) - 1]]'
    rc=1 Ran 44 tests in 0.808s | FAILED (failures=35)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_ple_rows_across_calls, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_ple_rows_across_calls, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_ple_rows_across_calls, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_ple_rows_across_calls, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S5 QSA: pooled keys of earlier ubatches / calls forgotten (zeros)
    ref/qwen4exp.py: 'idxc["pooled"] = np.concatenate([idxc["pooled"], new])' -> 'idxc["pooled"] = np.concatenate([np.zeros_like(idxc["pooled"]), new])'
    rc=1 Ran 44 tests in 2.112s | FAILED (failures=27)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S6 QSA: a new pool rotated to its index within the ubatch, not its sequence position
    ref/qwen4exp.py: 'new = imrope(new[:, None, :], pool_pos4(b1, kp)[b0:], **rope)[:, 0]' -> 'new = imrope(new[:, None, :], pool_pos4(b1 - b0, kp), **rope)[:, 0]'
    rc=1 Ran 44 tests in 1.624s | FAILED (failures=27)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S7 KV cache: dense attention forgets the cells of earlier ubatches / calls
    ref/qwen4exp.py: 'cells = [list(range(0, t + 1)) for t in np.asarray(pos).tolist()]' -> 'cells = [list(range(int(pos[0]), t + 1)) for t in np.asarray(pos).tolist()]'
    rc=1 Ran 44 tests in 2.099s | FAILED (failures=14)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Rollback.test_attention_over_the_kv_cache, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines
    property failing: test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S8 RoPE: positions restart at 0 in every ubatch / call
    ref/qwen4exp.py: 'p4 = text_pos4_at(pos)' -> 'p4 = text_pos4_at(pos - pos[0])'
    rc=1 Ran 44 tests in 0.937s | FAILED (failures=27)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S9 QSA: incomplete tail taken relative to the ubatch
    ref/qwen4exp.py: 'for r, t in enumerate(np.asarray(pos).tolist()):' -> 'for r, t in enumerate(range(len(pos))):'
    rc=1 Ran 44 tests in 1.043s | FAILED (failures=27)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S10 QSA: pool visibility relative to the ubatch
    ref/qwen4exp.py: 'visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.asarray(pos)[:, None] + 1)' -> 'visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.arange(len(pos))[:, None] + 1)'
    rc=1 Ran 14 tests in 0.711s | FAILED (failures=7, errors=5)
    dump parity failing: Decode.setUpClass, Rollback.setUpClass, Ubatch10.setUpClass, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S11 rollback: delta-rule state and conv history not rolled back (state after the rejected tokens)
    ref/qwen4exp.py: 'cache.clear()' -> 'fresh["gdn"] = cache["gdn"]\n    cache.clear()'
    rc=1 Ran 44 tests in 1.957s | FAILED (failures=8)
    dump parity failing: Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection
    property failing: test_rollback_then_continue_equals_the_kept_sequence
S12 rollback: PLE conv history not rolled back
    ref/qwen4exp.py: 'cache.clear()' -> 'fresh["ple"] = cache["ple"]\n    cache.clear()'
    rc=1 Ran 44 tests in 1.937s | FAILED (failures=8)
    dump parity failing: Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection
    property failing: test_rollback_then_continue_equals_the_kept_sequence
caught 12/12; caught by the dump parity tests alone 12/12
```

## Not covered (UNKNOWN / NOT RUN)

| what | status | why / what would close it |
|---|---|---|
| MTP block (`graph_mtp`, LLM_GRAPH_TYPE_DECODER_MTP, G:529-618) | NOT COVERED (MTP: NOT COVERED) | needs: a tiny model with `nextn_predict_layers` = 1 (block_count 9, the per-layer arrays with 9 entries, a full-attention QSA layer 8 with `blk.8.nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}`), the model loaded with `load_mtp`, a target context with `llama_set_embeddings_nextn(ctx, true, false)` (src/llama-ext.h:96) and a second context of type `LLAMA_CONTEXT_TYPE_MTP` fed like common/speculative.cpp:1511-1529 (token p with the target h_nextn of p-1, the first token of a call with the last row of the previous call), and an oracle of G:556-617 |
| several sequences, cells shared between sequences (`cache_safe` = false), a `seq_rm` that leaves a stale pool complete, multimodal batches | NOT COVERED | single text sequence. The stale marking of src/llama-memory-hybrid-idx.cpp:713-724 runs in the first graph after the rollback but marks nothing: the cut at 50 is inside pool 12, which is incomplete then (the padded `indexer_pool_k_new` row of that graph matches the n_new = 0 formula). Would need a cut that leaves a pool ending at or after it complete (e.g. `seq_add` / an edit, or several sequences with `seq_cp`) |
| rollback of more than `n_rs_seq` tokens, rollback inside a multi-ubatch call, `-ub` smaller than the draft | NOT RUN | one rollback of 2 <= 3 tokens after a 4-token single-ubatch draft; llama.cpp refuses larger partial rollbacks (src/llama-memory-recurrent.cpp:210-215) |
| unfused graph branches (chunked / autoregressive delta net, unfused HC) during decode | NOT EXERCISED | the CPU takes the fused ops (N2 finding 1) |
| quantized weights, f16 KV cache, flash attention, GPU backends | NOT COVERED / NOT RUN | as in N2: all f32, `-fa off`, no GPU in the sandbox |
| equality of the llama.cpp graph with the original model code | UNKNOWN | oracle-versus-llama.cpp parity only, as in N2 |
| the new CI step in GitHub Actions | NOT RUN | branch not pushed; the same command passes locally (below) |

## Commands and literal output

Environment: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 CPUs; c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0;
Python 3.13.16; numpy 2.5.3; Linux 6.18.44-fc-v114; cmake 3.28.3. `<S>` =
`/tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad`, `<llama.cpp>` = `<S>/llama.cpp`
(read only), `<repo>` = the worktree of this branch.

### Pin check

```
$ cd <llama.cpp> && git rev-parse HEAD && git status --short | wc -l
ec7630a640789c393694fb194f1bbbf0369fc62d
0
```

### From an empty build directory (what the CI step does), at 675d429

```
$ TMPDIR=<S>/n6 bash -c 'time tools/tinygen/run_parity_qwen4exp_decode.sh <llama.cpp> <S>/build-n6-fresh' > final_run.log 2>&1; echo "exit=$?" >> final_run.log
$ tail -n 12 final_run.log
test_rollback_then_continue_equals_the_kept_sequence (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_rollback_then_continue_equals_the_kept_sequence) ... ok

----------------------------------------------------------------------
Ran 44 tests in 2.203s

OK
artifacts: <S>/n6/tmp.6T5fLJ3cFu

real	2m10.672s
user	6m31.084s
sys	0m21.528s
exit=0
```

### Final run at 83634ed (same build directory; dump_tensors.cpp unchanged since d2f41c8), full `-v` output

83634ed changed only comments and line wrapping of the two test files.

```
$ git log --oneline -n 1
83634ed [parity-qa] qwen4exp decode tests: line numbers of two citations (src/llama-memory-hybrid-idx.cpp:769-770, 947-967, 898) printed from the pin; lines wrapped to 120 columns (no behaviour change)
$ TMPDIR=<S>/n6 tools/tinygen/run_parity_qwen4exp_decode.sh <llama.cpp> <S>/build-n6-fresh > head_run.log 2>&1; echo "exit=$?"
exit=0
$ cat head_run.log
CMAKE_BUILD_TYPE=Release
wrote <S>/n6/tmp.VmAUcfydpw/qwen4exp.gguf
ub16: 4 graphs, 1 calls, n_rs_seq 0
ub10: 7 graphs, 1 calls, n_rs_seq 0
decode: 17 graphs, 17 calls, n_rs_seq 0
rollback: 16 graphs, 17 calls, n_rs_seq 3
test_attention_over_the_kv_cache (tests.test_qwen4exp_decode_parity.Decode.test_attention_over_the_kv_cache)
Rotated q/k at absolute positions and the attention output over the cached cells (QSA layer and the dense ... ok
test_carried_state (tests.test_qwen4exp_decode_parity.Decode.test_carried_state)
The state each graph read: GDN conv history, delta-rule state, PLE conv history; and the state it wrote ... ok
test_gdn_layers (tests.test_qwen4exp_decode_parity.Decode.test_gdn_layers) ... ok
test_graphs_and_token_mapping (tests.test_qwen4exp_decode_parity.Decode.test_graphs_and_token_mapping)
One graph per ubatch, in order; each graph's embedding rows are exactly its tokens (proves the mapping ... ok
test_head_and_logits (tests.test_qwen4exp_decode_parity.Decode.test_head_and_logits) ... ok
test_moe_ids_and_combines (tests.test_qwen4exp_decode_parity.Decode.test_moe_ids_and_combines) ... ok
test_ple_module (tests.test_qwen4exp_decode_parity.Decode.test_ple_module) ... ok
test_ple_rows_across_calls (tests.test_qwen4exp_decode_parity.Decode.test_ple_rows_across_calls)
PLE rows (host-side hash) of every graph, exact: predecessors of the first tokens of a ubatch or a decode ... ok
test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Decode.test_qsa_pool_cache_and_selection)
k-pool cache across ubatches and calls: the pools a graph completes are pooled in that graph ... ok
test_run_exercises_its_mechanisms (tests.test_qwen4exp_decode_parity.Decode.test_run_exercises_its_mechanisms) ... ok
test_attention_over_the_kv_cache (tests.test_qwen4exp_decode_parity.Rollback.test_attention_over_the_kv_cache)
Rotated q/k at absolute positions and the attention output over the cached cells (QSA layer and the dense ... ok
test_carried_state (tests.test_qwen4exp_decode_parity.Rollback.test_carried_state)
The state each graph read: GDN conv history, delta-rule state, PLE conv history; and the state it wrote ... ok
test_gdn_layers (tests.test_qwen4exp_decode_parity.Rollback.test_gdn_layers) ... ok
test_graphs_and_token_mapping (tests.test_qwen4exp_decode_parity.Rollback.test_graphs_and_token_mapping)
One graph per ubatch, in order; each graph's embedding rows are exactly its tokens (proves the mapping ... ok
test_head_and_logits (tests.test_qwen4exp_decode_parity.Rollback.test_head_and_logits) ... ok
test_moe_ids_and_combines (tests.test_qwen4exp_decode_parity.Rollback.test_moe_ids_and_combines) ... ok
test_ple_module (tests.test_qwen4exp_decode_parity.Rollback.test_ple_module) ... ok
test_ple_rows_across_calls (tests.test_qwen4exp_decode_parity.Rollback.test_ple_rows_across_calls)
PLE rows (host-side hash) of every graph, exact: predecessors of the first tokens of a ubatch or a decode ... ok
test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Rollback.test_qsa_pool_cache_and_selection)
k-pool cache across ubatches and calls: the pools a graph completes are pooled in that graph ... ok
test_run_exercises_its_mechanisms (tests.test_qwen4exp_decode_parity.Rollback.test_run_exercises_its_mechanisms) ... ok
test_attention_over_the_kv_cache (tests.test_qwen4exp_decode_parity.Ubatch10.test_attention_over_the_kv_cache)
Rotated q/k at absolute positions and the attention output over the cached cells (QSA layer and the dense ... ok
test_carried_state (tests.test_qwen4exp_decode_parity.Ubatch10.test_carried_state)
The state each graph read: GDN conv history, delta-rule state, PLE conv history; and the state it wrote ... ok
test_gdn_layers (tests.test_qwen4exp_decode_parity.Ubatch10.test_gdn_layers) ... ok
test_graphs_and_token_mapping (tests.test_qwen4exp_decode_parity.Ubatch10.test_graphs_and_token_mapping)
One graph per ubatch, in order; each graph's embedding rows are exactly its tokens (proves the mapping ... ok
test_head_and_logits (tests.test_qwen4exp_decode_parity.Ubatch10.test_head_and_logits) ... ok
test_moe_ids_and_combines (tests.test_qwen4exp_decode_parity.Ubatch10.test_moe_ids_and_combines) ... ok
test_ple_module (tests.test_qwen4exp_decode_parity.Ubatch10.test_ple_module) ... ok
test_ple_rows_across_calls (tests.test_qwen4exp_decode_parity.Ubatch10.test_ple_rows_across_calls)
PLE rows (host-side hash) of every graph, exact: predecessors of the first tokens of a ubatch or a decode ... ok
test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Ubatch10.test_qsa_pool_cache_and_selection)
k-pool cache across ubatches and calls: the pools a graph completes are pooled in that graph ... ok
test_run_exercises_its_mechanisms (tests.test_qwen4exp_decode_parity.Ubatch10.test_run_exercises_its_mechanisms) ... ok
test_attention_over_the_kv_cache (tests.test_qwen4exp_decode_parity.Ubatch16.test_attention_over_the_kv_cache)
Rotated q/k at absolute positions and the attention output over the cached cells (QSA layer and the dense ... ok
test_carried_state (tests.test_qwen4exp_decode_parity.Ubatch16.test_carried_state)
The state each graph read: GDN conv history, delta-rule state, PLE conv history; and the state it wrote ... ok
test_gdn_layers (tests.test_qwen4exp_decode_parity.Ubatch16.test_gdn_layers) ... ok
test_graphs_and_token_mapping (tests.test_qwen4exp_decode_parity.Ubatch16.test_graphs_and_token_mapping)
One graph per ubatch, in order; each graph's embedding rows are exactly its tokens (proves the mapping ... ok
test_head_and_logits (tests.test_qwen4exp_decode_parity.Ubatch16.test_head_and_logits) ... ok
test_moe_ids_and_combines (tests.test_qwen4exp_decode_parity.Ubatch16.test_moe_ids_and_combines) ... ok
test_ple_module (tests.test_qwen4exp_decode_parity.Ubatch16.test_ple_module) ... ok
test_ple_rows_across_calls (tests.test_qwen4exp_decode_parity.Ubatch16.test_ple_rows_across_calls)
PLE rows (host-side hash) of every graph, exact: predecessors of the first tokens of a ubatch or a decode ... ok
test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Ubatch16.test_qsa_pool_cache_and_selection)
k-pool cache across ubatches and calls: the pools a graph completes are pooled in that graph ... ok
test_run_exercises_its_mechanisms (tests.test_qwen4exp_decode_parity.Ubatch16.test_run_exercises_its_mechanisms) ... ok

worst error per run and component over layers and graphs (rel = max|got-want| / max|want|):
run      component                                                  layer graph    max abs        rel
decode   Qcur (IMRoPE) [full]                                           7     0  3.863e-06  1.154e-06
decode   Qcur (IMRoPE) [chunk]                                          7     0  3.863e-06  1.154e-06
decode   Kcur (IMRoPE) [full]                                           7    10  4.549e-06  1.590e-06
decode   Kcur (IMRoPE) [chunk]                                          7    10  4.549e-06  1.590e-06
decode   Vcur [full]                                                    7    10  1.251e-06  1.240e-06
decode   Vcur [chunk]                                                   7    10  1.251e-06  1.240e-06
decode   attn_pregate (qsa) [full]                                      3     9  4.316e-07  6.678e-07
decode   attn_pregate (qsa) [chunk]                                     3     9  4.316e-07  6.678e-07
decode   attn_output (wo) [full]                                        7    10  1.800e-07  1.775e-06
decode   attn_output (wo) [chunk]                                       7    10  1.800e-07  1.775e-06
decode   attn_pregate (dense causal) [full]                             7    15  3.160e-07  1.394e-06
decode   attn_pregate (dense causal) [chunk]                            7    15  3.160e-07  1.394e-06
decode   conv_state_at (GDN conv history) [full]                        6    11  9.423e-07  6.555e-07
decode   conv_state_at (GDN conv history) [chunk]                       6    11  9.423e-07  6.555e-07
decode   state_predelta (delta-rule state) [full]                       6    16  5.307e-07  1.454e-06
decode   state_predelta (delta-rule state) [chunk]                      6    16  5.307e-07  1.454e-06
decode   new_state (written back) [chunk]                               6    15  5.307e-07  1.454e-06
decode   new_state (written back) [full]                                6    15  5.307e-07  1.454e-06
decode   conv_state_at (PLE conv history) [full]                        1    14  5.083e-07  1.735e-07
decode   conv_state_at (PLE conv history) [chunk]                       1    14  5.083e-07  1.735e-07
decode   conv_output_raw [full]                                         6    15  1.165e-06  9.316e-07
decode   conv_output_raw [chunk]                                        6    15  1.165e-06  9.316e-07
decode   conv_output_silu [full]                                        6    15  9.493e-07  1.014e-06
decode   conv_output_silu [chunk]                                       6    15  9.493e-07  1.014e-06
decode   gdn scan (attn_output) [full]                                  6    10  8.127e-08  2.298e-06
decode   gdn scan (attn_output) [chunk]                                 6    10  8.127e-08  2.298e-06
decode   final_output (gated norm) [full]                               5    15  5.527e-06  4.135e-06
decode   final_output (gated norm) [chunk]                              5    15  5.527e-06  4.135e-06
decode   linear_attn_out [full]                                         6    10  4.277e-06  3.686e-06
decode   linear_attn_out [chunk]                                        6    10  4.277e-06  3.686e-06
decode   hc_init (exact)                                               -1     0  0.000e+00  0.000e+00
decode   result_norm [full]                                            -1    10  1.275e-06  1.027e-06
decode   result_norm [chunk]                                           -1    10  1.275e-06  1.027e-06
decode   result_output [full]                                          -1    10  1.387e-06  1.125e-06
decode   result_output [chunk]                                         -1    10  1.387e-06  1.125e-06
decode   logits.bin [full]                                             -1    10  1.387e-06  1.125e-06
decode   logits.bin [chunk]                                            -1    10  1.387e-06  1.125e-06
decode   ffn_out (routed + gated shared) [full]                         7    15  3.985e-07  1.927e-06
decode   ffn_out (routed + gated shared) [chunk]                        7    15  3.985e-07  1.927e-06
decode   hc_combine (attn) [full]                                       7    10  5.196e-06  1.113e-06
decode   hc_combine (attn) [chunk]                                      7    10  5.196e-06  1.113e-06
decode   l_last (layer output) [full]                                   7    10  5.453e-06  1.132e-06
decode   l_last (layer output) [chunk]                                  7    10  5.453e-06  1.132e-06
decode   ple_gate [full]                                                1    14  2.417e-07  5.025e-07
decode   ple_gate [chunk]                                               1    14  2.417e-07  5.025e-07
decode   ple_gated_value [full]                                         1     8  6.223e-07  5.663e-07
decode   ple_gated_value [chunk]                                        1     8  6.223e-07  5.663e-07
decode   ple_conv_out [full]                                            1     3  6.184e-07  2.171e-07
decode   ple_conv_out [chunk]                                           1     3  6.184e-07  2.171e-07
decode   ple_embd = table[rows] (exact) [full]                         -1     0  0.000e+00  0.000e+00
decode   ple_embd = table[rows] (exact) [chunk]                        -1     0  0.000e+00  0.000e+00
decode   indexer_k_raw [full]                                           3     0  8.720e-07  5.455e-07
decode   indexer_k_raw [chunk]                                          3     0  8.720e-07  5.455e-07
decode   indexer_pool_k_new (pools completed here) [chunk]              3     8  6.327e-07  3.764e-07
decode   indexer_pool_k_new (pools completed here) [full]               3     8  6.327e-07  3.764e-07
decode   indexer_pool_k_new (padding rows)                              3     1  8.791e-07  3.980e-07
decode   indexer_k (pool cache, all complete pools) [chunk]             3     0  8.941e-07  3.289e-07
decode   indexer_k (pool cache, all complete pools) [full]              3     0  8.941e-07  3.289e-07
decode   indexer_q [full]                                               3     0  2.651e-06  8.488e-07
decode   indexer_q [chunk]                                              3     0  2.651e-06  8.488e-07
decode   indexer_score (visible) [chunk]                                3     0  3.591e-06  5.580e-07
decode   qsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
rollback Qcur (IMRoPE) [full]                                           7     0  3.863e-06  1.154e-06
rollback Qcur (IMRoPE) [chunk]                                          7     0  3.863e-06  1.154e-06
rollback Kcur (IMRoPE) [full]                                           7     7  2.745e-06  1.321e-06
rollback Kcur (IMRoPE) [chunk]                                          7     7  2.745e-06  1.321e-06
rollback Vcur [full]                                                    7     0  1.498e-06  9.038e-07
rollback Vcur [chunk]                                                   7     0  1.498e-06  9.038e-07
rollback attn_pregate (qsa) [full]                                      3     9  4.251e-07  8.882e-07
rollback attn_pregate (qsa) [chunk]                                     3     9  4.251e-07  8.882e-07
rollback attn_output (wo) [full]                                        7     9  1.267e-07  1.249e-06
rollback attn_output (wo) [chunk]                                       7     9  1.267e-07  1.249e-06
rollback attn_pregate (dense causal) [full]                             7     7  3.732e-07  1.477e-06
rollback attn_pregate (dense causal) [chunk]                            7     7  3.732e-07  1.477e-06
rollback conv_state_at (GDN conv history) [full]                        6    10  1.091e-06  7.591e-07
rollback conv_state_at (GDN conv history) [chunk]                       6    10  1.091e-06  7.591e-07
rollback state_predelta (delta-rule state) [full]                       2     9  1.890e-07  9.369e-07
rollback state_predelta (delta-rule state) [chunk]                      2     9  1.890e-07  9.369e-07
rollback conv_state_at (PLE conv history) [full]                        1    13  5.083e-07  1.735e-07
rollback conv_state_at (PLE conv history) [chunk]                       1    13  5.083e-07  1.735e-07
rollback conv_output_raw [full]                                         6     0  1.524e-06  6.494e-07
rollback conv_output_raw [chunk]                                        6     0  1.524e-06  6.494e-07
rollback conv_output_silu [full]                                        2     8  6.937e-07  1.118e-06
rollback conv_output_silu [chunk]                                       2     8  6.937e-07  1.118e-06
rollback gdn scan (attn_output) [full]                                  6     9  8.872e-08  2.509e-06
rollback gdn scan (attn_output) [chunk]                                 6     9  8.872e-08  2.509e-06
rollback final_output (gated norm) [full]                               6    12  6.621e-06  5.422e-06
rollback final_output (gated norm) [chunk]                              6    12  6.621e-06  5.422e-06
rollback linear_attn_out [full]                                         5     6  2.633e-06  3.011e-06
rollback linear_attn_out [chunk]                                        5     6  2.633e-06  3.011e-06
rollback hc_init (exact)                                               -1     0  0.000e+00  0.000e+00
rollback result_norm [full]                                            -1     0  1.622e-06  1.010e-06
rollback result_norm [chunk]                                           -1     0  1.622e-06  1.010e-06
rollback result_output [full]                                          -1     0  1.506e-06  8.570e-07
rollback result_output [chunk]                                         -1     0  1.506e-06  8.570e-07
rollback logits.bin [full]                                             -1     0  1.506e-06  8.570e-07
rollback logits.bin [chunk]                                            -1     0  1.506e-06  8.570e-07
rollback ffn_out (routed + gated shared) [full]                         7     7  4.030e-07  1.526e-06
rollback ffn_out (routed + gated shared) [chunk]                        7     7  4.030e-07  1.526e-06
rollback hc_combine (attn) [full]                                       7     5  3.506e-06  8.681e-07
rollback hc_combine (attn) [chunk]                                      7     5  3.506e-06  8.681e-07
rollback l_last (layer output) [full]                                   7     5  3.525e-06  8.814e-07
rollback l_last (layer output) [chunk]                                  7     5  3.525e-06  8.814e-07
rollback ple_gate [full]                                                1    13  2.417e-07  5.025e-07
rollback ple_gate [chunk]                                               1    13  2.417e-07  5.025e-07
rollback ple_gated_value [full]                                         1     7  6.223e-07  5.663e-07
rollback ple_gated_value [chunk]                                        1     7  6.223e-07  5.663e-07
rollback ple_conv_out [full]                                            1     1  6.950e-07  2.496e-07
rollback ple_conv_out [chunk]                                           1     1  6.950e-07  2.496e-07
rollback ple_embd = table[rows] (exact) [full]                         -1     0  0.000e+00  0.000e+00
rollback ple_embd = table[rows] (exact) [chunk]                        -1     0  0.000e+00  0.000e+00
rollback indexer_k_raw [full]                                           3     2  4.705e-07  6.132e-07
rollback indexer_k_raw [chunk]                                          3     2  4.705e-07  6.132e-07
rollback indexer_pool_k_new (pools completed here) [chunk]              3     3  1.000e-06  3.849e-07
rollback indexer_pool_k_new (pools completed here) [full]               3     3  1.000e-06  3.849e-07
rollback indexer_pool_k_new (padding rows)                              3     2  1.414e-06  6.715e-07
rollback indexer_k (pool cache, all complete pools) [chunk]             3     3  1.000e-06  3.679e-07
rollback indexer_k (pool cache, all complete pools) [full]              3     3  1.000e-06  3.679e-07
rollback indexer_q [full]                                               3     0  2.651e-06  8.488e-07
rollback indexer_q [chunk]                                              3     0  2.651e-06  8.488e-07
rollback indexer_score (visible) [chunk]                                3     0  3.591e-06  5.580e-07
rollback qsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
ub10     Qcur (IMRoPE) [full]                                           7     5  5.441e-06  1.966e-06
ub10     Qcur (IMRoPE) [chunk]                                          7     5  5.441e-06  1.966e-06
ub10     Kcur (IMRoPE) [full]                                           7     5  5.480e-06  1.801e-06
ub10     Kcur (IMRoPE) [chunk]                                          7     5  5.480e-06  1.801e-06
ub10     Vcur [full]                                                    7     5  1.757e-06  1.154e-06
ub10     Vcur [chunk]                                                   7     5  1.757e-06  1.154e-06
ub10     attn_pregate (qsa) [full]                                      3     2  6.605e-07  9.090e-07
ub10     attn_pregate (qsa) [chunk]                                     3     2  6.605e-07  9.090e-07
ub10     attn_output (wo) [full]                                        7     5  2.811e-07  1.762e-06
ub10     attn_output (wo) [chunk]                                       7     5  2.811e-07  1.762e-06
ub10     attn_pregate (dense causal) [full]                             7     5  7.243e-07  2.449e-06
ub10     attn_pregate (dense causal) [chunk]                            7     5  7.243e-07  2.449e-06
ub10     conv_state_at (GDN conv history) [full]                        5     6  8.033e-07  6.261e-07
ub10     conv_state_at (GDN conv history) [chunk]                       5     6  8.033e-07  6.261e-07
ub10     state_predelta (delta-rule state) [full]                       6     1  2.803e-07  1.374e-06
ub10     state_predelta (delta-rule state) [chunk]                      6     1  2.803e-07  1.374e-06
ub10     new_state (written back) [chunk]                               5     5  2.842e-07  1.083e-06
ub10     new_state (written back) [full]                                5     5  2.842e-07  1.083e-06
ub10     conv_state_at (PLE conv history) [full]                        1     4  5.629e-07  1.694e-07
ub10     conv_state_at (PLE conv history) [chunk]                       1     4  5.629e-07  1.694e-07
ub10     conv_output_raw [full]                                         6     3  1.461e-06  7.935e-07
ub10     conv_output_raw [chunk]                                        6     3  1.461e-06  7.935e-07
ub10     conv_output_silu [full]                                        6     1  1.191e-06  7.008e-07
ub10     conv_output_silu [chunk]                                       6     1  1.191e-06  7.008e-07
ub10     gdn scan (attn_output) [full]                                  6     0  8.739e-08  1.495e-06
ub10     gdn scan (attn_output) [chunk]                                 6     0  8.739e-08  1.495e-06
ub10     final_output (gated norm) [full]                               6     5  1.018e-05  3.867e-06
ub10     final_output (gated norm) [chunk]                              6     5  1.018e-05  3.867e-06
ub10     linear_attn_out [full]                                         6     5  6.761e-06  4.284e-06
ub10     linear_attn_out [chunk]                                        6     5  6.761e-06  4.284e-06
ub10     hc_init (exact)                                               -1     0  0.000e+00  0.000e+00
ub10     result_norm [full]                                            -1     5  1.899e-06  1.340e-06
ub10     result_norm [chunk]                                           -1     5  1.899e-06  1.340e-06
ub10     result_output [full]                                          -1     5  1.921e-06  1.160e-06
ub10     result_output [chunk]                                         -1     5  1.921e-06  1.160e-06
ub10     logits.bin [full]                                             -1     5  1.921e-06  1.160e-06
ub10     logits.bin [chunk]                                            -1     5  1.921e-06  1.160e-06
ub10     ffn_out (routed + gated shared) [full]                         7     5  6.738e-07  1.707e-06
ub10     ffn_out (routed + gated shared) [chunk]                        7     5  6.738e-07  1.707e-06
ub10     hc_combine (attn) [full]                                       7     5  7.705e-06  1.155e-06
ub10     hc_combine (attn) [chunk]                                      7     5  7.705e-06  1.155e-06
ub10     l_last (layer output) [full]                                   7     5  8.107e-06  1.223e-06
ub10     l_last (layer output) [chunk]                                  7     5  8.107e-06  1.223e-06
ub10     ple_gate [full]                                                1     2  2.295e-07  2.701e-07
ub10     ple_gate [chunk]                                               1     2  2.295e-07  2.701e-07
ub10     ple_gated_value [full]                                         1     2  6.379e-07  2.919e-07
ub10     ple_gated_value [chunk]                                        1     2  6.379e-07  2.919e-07
ub10     ple_conv_out [full]                                            1     3  6.249e-07  1.916e-07
ub10     ple_conv_out [chunk]                                           1     3  6.249e-07  1.916e-07
ub10     ple_embd = table[rows] (exact) [full]                         -1     0  0.000e+00  0.000e+00
ub10     ple_embd = table[rows] (exact) [chunk]                        -1     0  0.000e+00  0.000e+00
ub10     indexer_k_raw [full]                                           3     2  8.720e-07  5.455e-07
ub10     indexer_k_raw [chunk]                                          3     2  8.720e-07  5.455e-07
ub10     indexer_pool_k_new (pools completed here) [chunk]              3     2  8.941e-07  4.110e-07
ub10     indexer_pool_k_new (pools completed here) [full]               3     2  8.941e-07  4.110e-07
ub10     indexer_pool_k_new (padding rows)                              3     0  6.279e-07  3.404e-07
ub10     indexer_k (pool cache, all complete pools) [chunk]             3     2  8.941e-07  3.743e-07
ub10     indexer_k (pool cache, all complete pools) [full]              3     2  8.941e-07  3.743e-07
ub10     indexer_q [full]                                               3     2  2.651e-06  9.491e-07
ub10     indexer_q [chunk]                                              3     2  2.651e-06  9.491e-07
ub10     indexer_score (visible) [chunk]                                3     4  3.591e-06  7.249e-07
ub10     qsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
ub16     Qcur (IMRoPE) [full]                                           7     3  5.441e-06  1.749e-06
ub16     Qcur (IMRoPE) [chunk]                                          7     3  5.441e-06  1.749e-06
ub16     Kcur (IMRoPE) [full]                                           7     3  5.480e-06  1.801e-06
ub16     Kcur (IMRoPE) [chunk]                                          7     3  5.480e-06  1.801e-06
ub16     Vcur [full]                                                    7     3  1.757e-06  1.154e-06
ub16     Vcur [chunk]                                                   7     3  1.757e-06  1.154e-06
ub16     attn_pregate (qsa) [full]                                      3     1  6.605e-07  8.046e-07
ub16     attn_pregate (qsa) [chunk]                                     3     1  6.605e-07  8.046e-07
ub16     attn_output (wo) [full]                                        7     3  2.811e-07  1.620e-06
ub16     attn_output (wo) [chunk]                                       7     3  2.811e-07  1.620e-06
ub16     attn_pregate (dense causal) [full]                             7     3  7.243e-07  2.359e-06
ub16     attn_pregate (dense causal) [chunk]                            7     3  7.243e-07  2.359e-06
ub16     conv_state_at (GDN conv history) [full]                        6     2  9.827e-07  7.644e-07
ub16     conv_state_at (GDN conv history) [chunk]                       6     2  9.827e-07  7.644e-07
ub16     state_predelta (delta-rule state) [full]                       6     2  3.637e-07  9.750e-07
ub16     state_predelta (delta-rule state) [chunk]                      6     2  3.637e-07  9.750e-07
ub16     new_state (written back) [chunk]                               6     1  3.637e-07  9.750e-07
ub16     new_state (written back) [full]                                6     1  3.637e-07  9.750e-07
ub16     conv_state_at (PLE conv history) [full]                        1     2  5.777e-07  1.835e-07
ub16     conv_state_at (PLE conv history) [chunk]                       1     2  5.777e-07  1.835e-07
ub16     conv_output_raw [full]                                         6     2  1.461e-06  7.935e-07
ub16     conv_output_raw [chunk]                                        6     2  1.461e-06  7.935e-07
ub16     conv_output_silu [full]                                        6     0  1.191e-06  6.610e-07
ub16     conv_output_silu [chunk]                                       6     0  1.191e-06  6.610e-07
ub16     gdn scan (attn_output) [full]                                  6     0  8.739e-08  1.495e-06
ub16     gdn scan (attn_output) [chunk]                                 6     0  8.739e-08  1.495e-06
ub16     final_output (gated norm) [full]                               6     3  1.018e-05  3.867e-06
ub16     final_output (gated norm) [chunk]                              6     3  1.018e-05  3.867e-06
ub16     linear_attn_out [full]                                         6     3  6.761e-06  4.284e-06
ub16     linear_attn_out [chunk]                                        6     3  6.761e-06  4.284e-06
ub16     hc_init (exact)                                               -1     0  0.000e+00  0.000e+00
ub16     result_norm [full]                                            -1     3  1.899e-06  1.340e-06
ub16     result_norm [chunk]                                           -1     3  1.899e-06  1.340e-06
ub16     result_output [full]                                          -1     1  1.506e-06  8.570e-07
ub16     result_output [chunk]                                         -1     1  1.506e-06  8.570e-07
ub16     logits.bin [full]                                             -1     1  1.506e-06  8.570e-07
ub16     logits.bin [chunk]                                            -1     1  1.506e-06  8.570e-07
ub16     ffn_out (routed + gated shared) [full]                         7     3  6.738e-07  1.707e-06
ub16     ffn_out (routed + gated shared) [chunk]                        7     3  6.738e-07  1.707e-06
ub16     hc_combine (attn) [full]                                       7     3  7.705e-06  1.155e-06
ub16     hc_combine (attn) [chunk]                                      7     3  7.705e-06  1.155e-06
ub16     l_last (layer output) [full]                                   7     3  8.107e-06  1.223e-06
ub16     l_last (layer output) [chunk]                                  7     3  8.107e-06  1.223e-06
ub16     ple_gate [full]                                                1     1  2.295e-07  2.701e-07
ub16     ple_gate [chunk]                                               1     1  2.295e-07  2.701e-07
ub16     ple_gated_value [full]                                         1     1  6.379e-07  2.640e-07
ub16     ple_gated_value [chunk]                                        1     1  6.379e-07  2.640e-07
ub16     ple_conv_out [full]                                            1     3  6.950e-07  1.887e-07
ub16     ple_conv_out [chunk]                                           1     3  6.950e-07  1.887e-07
ub16     ple_embd = table[rows] (exact) [full]                         -1     0  0.000e+00  0.000e+00
ub16     ple_embd = table[rows] (exact) [chunk]                        -1     0  0.000e+00  0.000e+00
ub16     indexer_k_raw [full]                                           3     1  8.720e-07  5.455e-07
ub16     indexer_k_raw [chunk]                                          3     1  8.720e-07  5.455e-07
ub16     indexer_pool_k_new (pools completed here) [chunk]              3     1  8.941e-07  4.110e-07
ub16     indexer_pool_k_new (pools completed here) [full]               3     1  8.941e-07  4.110e-07
ub16     indexer_pool_k_new (padding rows)                              3     0  6.279e-07  3.404e-07
ub16     indexer_k (pool cache, all complete pools) [chunk]             3     1  8.941e-07  3.743e-07
ub16     indexer_k (pool cache, all complete pools) [full]              3     1  8.941e-07  3.743e-07
ub16     indexer_q [full]                                               3     1  2.651e-06  8.695e-07
ub16     indexer_q [chunk]                                              3     1  2.651e-06  8.695e-07
ub16     indexer_score (visible) [chunk]                                3     2  3.591e-06  7.249e-07
ub16     qsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
test_carried_state_is_the_state_of_the_prefix (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_carried_state_is_the_state_of_the_prefix) ... ok
test_every_chunking_reproduces_the_full_forward (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_every_chunking_reproduces_the_full_forward) ... ok
test_guards (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_guards)
The random model must exercise what the tests above claim: sparse QSA, EOS resets, pools that straddle a ... ok
test_rollback_then_continue_equals_the_kept_sequence (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_rollback_then_continue_equals_the_kept_sequence) ... ok

----------------------------------------------------------------------
Ran 44 tests in 2.227s

OK
artifacts: <S>/n6/tmp.VmAUcfydpw
```

The first line (`CMAKE_BUILD_TYPE=Release`) is printed by the llama.cpp CMake configure on stderr, as in N2.

### Rollback run manifest (call plan and graphs as llama.cpp computed them)

```
$ cat <artifacts>/n6/rollback/manifest.json; cat <artifacts>/n6/tokens.txt <artifacts>/n6/rollback/tokens.txt
{"n_ubatch":512,"n_rs_seq":3,"per_graph":true,"first_node":"model.input_embed","calls":[{"kind":"decode","n":48,"pos0":0,"tok0":0,"graphs":[0,1]},{"kind":"decode","n":4,"pos0":48,"tok0":48,"graphs":[1,2]},{"kind":"rollback","n":2,"pos0":50,"tok0":52,"graphs":[2,2]},{"kind":"decode","n":1,"pos0":50,"tok0":52,"graphs":[2,3]},{"kind":"decode","n":1,"pos0":51,"tok0":53,"graphs":[3,4]},{"kind":"decode","n":1,"pos0":52,"tok0":54,"graphs":[4,5]},{"kind":"decode","n":1,"pos0":53,"tok0":55,"graphs":[5,6]},{"kind":"decode","n":1,"pos0":54,"tok0":56,"graphs":[6,7]},{"kind":"decode","n":1,"pos0":55,"tok0":57,"graphs":[7,8]},{"kind":"decode","n":1,"pos0":56,"tok0":58,"graphs":[8,9]},{"kind":"decode","n":1,"pos0":57,"tok0":59,"graphs":[9,10]},{"kind":"decode","n":1,"pos0":58,"tok0":60,"graphs":[10,11]},{"kind":"decode","n":1,"pos0":59,"tok0":61,"graphs":[11,12]},{"kind":"decode","n":1,"pos0":60,"tok0":62,"graphs":[12,13]},{"kind":"decode","n":1,"pos0":61,"tok0":63,"graphs":[13,14]},{"kind":"decode","n":1,"pos0":62,"tok0":64,"graphs":[14,15]},{"kind":"decode","n":1,"pos0":63,"tok0":65,"graphs":[15,16]}],"graphs":[{"call":0,"n_asked":1118,"n_dumped":523},{"call":1,"n_asked":1118,"n_dumped":523},{"call":3,"n_asked":1118,"n_dumped":523},{"call":4,"n_asked":1118,"n_dumped":523},{"call":5,"n_asked":1118,"n_dumped":523},{"call":6,"n_asked":1118,"n_dumped":523},{"call":7,"n_asked":1118,"n_dumped":523},{"call":8,"n_asked":1118,"n_dumped":523},{"call":9,"n_asked":1118,"n_dumped":523},{"call":10,"n_asked":1118,"n_dumped":523},{"call":11,"n_asked":1118,"n_dumped":523},{"call":12,"n_asked":1118,"n_dumped":523},{"call":13,"n_asked":1118,"n_dumped":523},{"call":14,"n_asked":1118,"n_dumped":523},{"call":15,"n_asked":1118,"n_dumped":523},{"call":16,"n_asked":1118,"n_dumped":523}]}
76,13,65,36,7,3,21,87,78,63,50,43,5,37,65,2,55,71,72,90,15,27,75,73,92,36,87,81,90,14,57,45,14,2,55,35,59,92,15,28,92,84,40,15,8,78,28,2,49,65,27,68,76,85,92,2,2,84,49,2,80,2,41,48
76,13,65,36,7,3,21,87,78,63,50,43,5,37,65,2,55,71,72,90,15,27,75,73,92,36,87,81,90,14,57,45,14,2,55,35,59,92,15,28,92,84,40,15,8,78,28,2,49,65,44,85,27,68,76,85,92,2,2,84,49,2,80,2,41,48
```

### Mutation check at 83634ed, against the dumps of the final run

```
$ python3 tools/tinygen/mutate_qwen4exp_decode.py <repo> <S>/n6/tmp.VmAUcfydpw/n6 <S>/n6/tmp.VmAUcfydpw/qwen4exp.gguf <llama.cpp>/gguf-py <S>/n6/muthead
(output above, in "Mutation check")
$ python3 aggregate.py head_run.log > head_aggregate.txt; diff <(python3 aggregate.py final_run.log) head_aggregate.txt && echo "aggregate identical to the fresh-build run"
aggregate identical to the fresh-build run
```

### The default mode of the dump tool is unchanged

`<S>/build-n2-lead/dump-tensors` was built (2026-10-10 11:32) from the main checkout's tools/dump/dump_tensors.cpp,
which equals the file at c1a0336 (`cmp` below; the file is dated 2026-10-09 21:45). Same N2 model, tokens, prefixes and
flags (scratchpad script same_as_old.sh):

```
$ git show c1a0336:tools/dump/dump_tensors.cpp > dump_tensors.c1a0336.cpp && cmp /home/user/408032gb/tools/dump/dump_tensors.cpp dump_tensors.c1a0336.cpp && echo same-as-c1a0336
same-as-c1a0336
$ bash same_as_old.sh <S>/n6/tmp.nbblgtrcuq <S>/build-n6-fresh/dump-tensors
old exit=0 files=1022
new exit=0 files=1022
identical: 1022 files
```

The existing users of the tool pass with the new binary (built from the dump_tensors.cpp of d2f41c8 into
`<S>/build-n6`; `git diff --stat d2f41c8 -- tools/dump/dump_tensors.cpp tools/g0/CMakeLists.txt` prints nothing):

```
$ TMPDIR=<S>/n6 bash tools/tinygen/run_parity_qwen4exp.sh <llama.cpp> <S>/build-n6 > regress_qwen4exp.log 2>&1; echo "exit=$?"
exit=0
$ TMPDIR=<S>/n6 bash tools/tinygen/run_parity.sh <llama.cpp> <S>/build-n6 > regress_glm.log 2>&1; echo "exit=$?"
exit=0
$ TMPDIR=<S>/n6 bash tools/tinygen/run_parity_mimo.sh <llama.cpp> <S>/build-n6 > regress_mimo.log 2>&1; echo "exit=$?"
exit=0
$ grep -E "^Ran|^OK|^FAILED" regress_qwen4exp.log regress_glm.log regress_mimo.log
regress_qwen4exp.log:Ran 17 tests in 0.401s
regress_qwen4exp.log:OK
regress_glm.log:Ran 8 tests in 0.279s
regress_glm.log:OK
regress_mimo.log:Ran 5 tests in 0.084s
regress_mimo.log:OK
```

### N2's tie replay with a mutant confined to `Q.qsa_select` (finding 10; on a copy, scratchpad n2_select_mutant.py)

```
$ python3 n2_select_mutant.py <repo> <S>/n6/tmp.nbblgtrcuq/dump <S>/n6/tmp.nbblgtrcuq/qwen4exp.gguf <llama.cpp>/gguf-py <S>/n6
mutant: ref/qwen4exp.py qsa_select: 'top = vis[np.argsort(-scores[t, vis], kind="stable")[:n_top]]' -> 'top = vis[np.argsort(-scores[t, vis], kind="stable")[:n_top - 1]]'
tests.test_qwen4exp_tiny_parity: rc=0 Ran 17 tests in 0.296s | OK; failing: none
tests.test_qwen4exp_ref: rc=1 Ran 16 tests in 0.017s | FAILED (failures=1); failing: test_selection_is_top_pools_plus_tail
```

### Unit tests (what the CI `unit tests` step runs), at 83634ed

```
$ python3 -m unittest discover -s tests > discover.log 2>&1; echo "exit=$?"; tail -n 4 discover.log
exit=0
----------------------------------------------------------------------
Ran 412 tests in 92.454s

OK (skipped=80)
```
