# N6: qwen4exp several ubatches, token-by-token decode, rollback and MTP, CPU parity, 2026-10-10 (sandbox CPU, no GPU)

Task N6 of the hardware-free track (docs/reassessment/ROADMAP.md row N6): close the "decode and several ubatches" gap
of N2 (docs/evidence/2026-10-10-qwen4exp-tiny-parity.md, "Not covered" table), then the MTP block. Branch
`claude/n6-qwen4exp-decode`, cut from `claude/relaxed-dijkstra-1d7i5o` at `c1a0336` (which contains N2).

Reference: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (pin check below), CPU build of `tools/g0` target
`dump-tensors`, the real `qwen4exp` graph. "G:<line>" = a line of src/models/qwen4exp.cpp at the pin; every cited
line was printed from the pinned checkout before citing.

| piece | file |
|---|---|
| oracle | ref/qwen4exp.py: N2's `forward()` unchanged, plus the incremental path `forward_chunk` / `cache_truncate` / `forward_chunks` (state carried between ubatches and calls, each item cited to where llama.cpp keeps it) and the MTP block `mtp_forward_chunk` (G:529-618) |
| dump parity tests | tests/test_qwen4exp_decode_parity.py: 10 tests per run, 4 runs = 40; tests/test_qwen4exp_mtp_parity.py: 6 tests; both skip without `QWEN4EXP_DECODE_DUMP`, the GGUF variables and `LLAMA_CPP_GGUF_PY` |
| property tests | tests/test_qwen4exp_chunked_ref.py: 8 tests, numpy only, random 4-layer model (+ MTP layer) (run in the CI `unit tests` step) |
| driver | tools/tinygen/run_parity_qwen4exp_decode.sh (builds dump-tensors, writes the tiny GGUF with and without the MTP block, runs the 5 dumps, runs the three test modules) |
| dump tool | tools/dump/dump_tensors.cpp: opt-in `DUMP_PER_GRAPH`, `DUMP_CALLS`, `DUMP_N_RS_SEQ`, `DUMP_MTP`, `DUMP_MTP_DRAFT`, `DUMP_MTP_NO_RESERVE`; default output byte-identical (below) |
| generator | tools/tinygen/qwen4exp_tiny.py `--mtp` (default output byte-identical, below) |
| mutation check | tools/tinygen/mutate_qwen4exp_decode.py (17 mutants on copies; the repository is never edited) |
| CI | step appended to `cpu-parity` in .github/workflows/ci.yml: NOT RUN in GitHub Actions (branch not pushed) |

## Runs

Model: the N2 tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0: 8 layers, gated delta net on 0-2 and 4-6, PLE on
layer 1, QSA on layer 3 with compress ratio 4 and budget 2 pools + tail, dense full attention on layer 7); for the MTP
run the same model with `--mtp` (block_count 9, nextn_predict_layers 1, layer 8 a full-attention QSA layer with
`blk.8.nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}`; trunk weights identical). Flags as N2:
`-c 512 -b 512 -t 1 -fa off -ctk f32 -ctv f32`. One 64-token sequence (`random.seed(6)`, ids in [3, 96)) with EOS
(id 2) at 15, 33, 47, 55, 56, 59, 61, so every run has a graph whose first token reads an EOS predecessor from an
earlier graph.

| run | `-ub` | llama_process calls (`DUMP_CALLS`) | graphs | what it adds |
|---|---|---|---|---|
| ub16 | 16 | one call of 64 tokens | 4 x 16 tokens | carry across ubatches whose boundaries coincide with pool boundaries |
| ub10 | 10 | one call of 64 tokens | 6 x 10 + 1 x 4 | pools that straddle two ubatches (completed from raw keys of the previous ubatch), a short last ubatch |
| decode | 512 | `48,1,...,1`: prefill 48, then 16 calls of one token | 1 x 48 + 16 x 1 | decode with n_tokens = 1 per call: GDN state and conv history, PLE history and predecessors, KV cache, a pool completed by a one-token call (positions 51, 55, 59, 63), EOS decoded as a single token (55, 56, 59, 61) |
| rollback | 512, `n_rs_seq` 3 | `48,4,r2,1,...,1`: prefill 48; a 4-token draft 48..51 whose ids at 50-51 are wrong (44, 85 instead of 27, 68); `llama_memory_seq_rm(mem, 0, 50, -1)`; 14 calls of one token | 1 + 1 + 14 | rollback from the recurrent snapshots (src/llama-memory-recurrent.cpp:199-215), pool 12 first pooled from wrong keys and pooled again, KV cells re-used |
| mtp | 16 | target: `48,1,...,1` over the first 56 tokens (3 + 8 graphs); MTP context: the same 9 calls (3 + 8 graphs) + 4 greedy draft steps at 56..59 (4 graphs) | 11 + 15 | MTP block driven like common/speculative.cpp: token p paired with the target h_nextn of p-1 (lines 1511-1529), draft chain on the MTP's own h_nextn (lines 1700-1704), the MTP context's own KV and k-pool cache (pool 14 completed by the 4th draft step) |

## Method

`DUMP_PER_GRAPH=1` writes the dumps of every computed graph (one per ubatch) into `g<NNN>/`, and `manifest.json` maps
the llama_process calls to graphs and positions (the MTP context writes into `mtp/` with its own manifest, plus the
h row fed with every MTP token, `h_in.bin`). Every graph is compared row by row with

* (A) `[full]`: the full-sequence oracle `Q.forward(final 64 tokens)` at the graph's positions (for the MTP context:
  `Q.mtp_forward_chunk` over all processed positions at once, fed by the full-sequence target oracle). Causality: an
  incremental result must equal the full recompute. Rows of rejected draft tokens and MTP draft steps have no (A);
* (B) `[chunk]`: the incremental oracle `Q.forward_chunk` / `Q.mtp_forward_chunk`, replaying the same calls graph by
  graph (rollback = `Q.cache_truncate`; the MTP draft chain fed with the oracle's own h_nextn). The state each graph
  read is compared too: `conv_state_at-<il>` (GDN conv history; on the PLE layer occurrence 0 is the PLE conv
  history, G:1423 runs before the mixer), `state_predelta-<il>` (delta-rule state, G:1073-1074), `new_state-<il>`
  (written back, `n_rs_seq` = 0 only, src/models/delta-net-base.cpp:546-560) and `indexer_k-<il>` (pooled keys
  gathered from the k-pool cache, G:797). For these state tensors (A) is derived independently of the incremental
  path: rows of the full-sequence qkv / PLE values before the graph, and the delta rule run over the first p0 tokens
  of the full-sequence q, k, v, g, beta.

(A) and (B) agree within 1e-10 relative (tests/test_qwen4exp_chunked_ref.py, six chunkings incl. one token per
call; MTP: three chunkings), so their error columns below are equal to the printed digits; (B) is what the mutation
check of the state carry bites on, (A) is the literal causality check. Unlike N2's per-component tests (which
recompute each op from the dumped inputs), every N6 comparison is against the oracle run end to end from the token
ids, so the errors include propagation through the earlier layers.

## Tolerances

As N2: normwise relative error max|got - want| / max|want| < 1e-4 for every f32 path; exact (0) for the embedding
gathers (trunk and MTP), the PLE table gather (host-side hash, predecessors read from the KV cells of earlier
ubatches/calls) and the zero state at the sequence start; MoE top-k ids equal as sets; QSA selection equal, or
another valid top-k by the oracle's own scores (`Q.qsa_tie_variant`), counted and replayed (0 occurred). Since
2362cbe a difference is replayed only if the oracle's own selection is also a valid top-k for the token (finding 9).
Greedy argmax of every logits row equal; the MTP draft tokens equal the oracle's argmax chain. No tolerance was
loosened: the largest error is 5.4e-6 (`final_output`, the gated RMS norm of the delta-rule output, layer 6), about
18x inside 1e-4; the largest in the MTP context is 1.6e-6.

## Findings (CONFIRMED by the runs below unless labelled)

1. **One graph per ubatch, detected from the callback.** `ggml_backend_sched_compute_splits` asks the eval callback
   about every node once, in node order (ggml/src/ggml-backend.cpp:1821-1848), so a graph starts where the first node
   asked (`model.input_embed`; `mtp_tok_embd-8` in the MTP context) comes round again. Every call computed
   ceil(n_tokens / n_ubatch) graphs (the tool fails otherwise) and each graph's embedding rows are exactly its tokens
   (test_graphs_and_token_mapping, mtp_tok_embd). Every graph of ub16, ub10 and decode has 1034 nodes; with
   `n_rs_seq` = 3 every graph has 1118 (PROVISIONAL attribution of the 84 extra nodes: the snapshot writes of
   G:1324-1345 and src/models/delta-net-base.cpp:563-603); the target graphs of the MTP run (h_nextn exported) have
   1033 (the difference was not analysed); every graph of the MTP context has 182. In the default mode (no `DUMP_*`
   option beyond N2's) the new tool writes the same 1022 files, byte for byte, as the binary built from the previous
   dump_tensors.cpp; the generator's default GGUF is byte-identical to the one generated before `--mtp` existed.
2. **Carried state = state of the prefix.** In every graph with p0 > 0 the GDN conv history (`conv_state_at`, the
   last 3 rows of the conv input), the delta-rule state (`state_predelta`) and the PLE conv history (the last 6 rows of
   the grouped-normed gated value) equal the full-sequence values before p0; at p0 = 0 they are exactly zero; the
   written-back `new_state` equals the state after the graph. Worst: 1.5e-6 (delta-rule state), 7.6e-7 (GDN conv
   history), 1.8e-7 (PLE conv history).
3. **Layouts.** `conv_state_at` is `[K-1, C]` with time fastest; the delta-rule state is `[S_v, S_v, H_v]` with S[i][j]
   at `[j*S_v + i]` (ggml/src/ggml-cpu/ops.cpp:11228), so a dump reads as the oracle's `state[h].T` (ref/kda.py keeps
   S[h, i, j]). On the PLE layer `conv_state_at-1` occurrence 0 is the PLE history ([6, 256]) and occurrence 1 the GDN
   history ([3, 128]). In the MTP graph the three mixers of layer 8 are `hc_norm-8` occurrences 0 (attention), 1
   (MoE), 2 (head); the head's `hc_mixed` is renamed `result_norm`.
4. **k-pool cache.** A pool is pooled once, in the graph that completes it (`indexer_pool_k_new`: rows
   [p0/4, (p0+n)/4) of the pool list), including pools completed by a one-token call, by an MTP draft step and pools
   straddling two ubatches (ub10); every complete pool is read back from the cache in later graphs (`indexer_k`,
   worst 3.7e-7). The new-pool list is padded to n_new_g = max(n_new, 1, n_tokens/4 + 1) rows
   (src/llama-memory-hybrid-idx.cpp:769-770); a padded row re-pools the first member of pool 0 if some pool is new,
   else the ubatch's first token, rotated at position 0 (lines 947-967, 898), i.e. `rms_norm(k_raw[cell]) * k_norm`,
   checked for every padded row of the trunk runs (worst 6.7e-7).
5. **PLE predecessors across graphs.** The gathered PLE rows are exactly the full-sequence rows in every graph,
   including first tokens whose predecessors (one an EOS) lie in an earlier ubatch or call.
6. **Rollback.** With `n_rs_seq` = 3 llama.cpp accepts the partial `llama_memory_seq_rm(0, 50, -1)` after the 4-token
   draft; the next graph (position 50) reads the delta-rule state, GDN and PLE conv histories of the kept 50-token
   prefix, writes no `new_state` (snapshot path), and all later graphs equal the full-sequence oracle of the final
   sequence. Pool 12 (48..51), first pooled from the wrong keys of the draft, is pooled again when position 51 is
   decoded and the cached value equals the correct one. The rows of the rejected tokens equal the incremental oracle
   that replays them.
7. **KV cell = position** for this single sequence, also after the rollback and in the MTP context: the QSA
   selection read from `kq_mask_qsa` as cell indices equals the oracle's selection as positions in every graph.
8. **QSA in decode.** Selections equal the oracle's in all 258 rows of the four trunk runs (64 + 64 + 64 + 66) and all
   60 rows of the MTP context, 0 tie replays; the budget binds (a token attends to fewer cells than causal attention).
9. **Hole found by the mutation check, fixed in 2362cbe.** In the first mutation rounds the dump tests did not catch
   S9 (QSA tail taken relative to the ubatch): the reference's selection differed from the mutant oracle's, was judged
   a valid top-k by `Q.qsa_tie_variant`, and was replayed over it. Now both selections must be valid top-k sets for
   the token's absolute position; since then every mutant is caught by the dump tests alone. Tolerances unchanged.
10. **The same hole exists in N2's tests (not fixed here).** A mutant confined to `Q.qsa_select` (one pool fewer
    selected) passes all 17 N2 dump tests on the N2 dump; only the N2 property test
    `test_selection_is_top_pools_plus_tail` fails (output below). N2's M10 was caught because it mutates `qsa_tail`,
    which the validation shares. Suggested fix (for the lead, outside N6): in `_tie_variants` of
    tests/test_qwen4exp_tiny_parity.py also require `Q.qsa_tie_variant(scores[t], mine[t], t, ...)` to hold.
11. **MTP block.** `graph_mtp` (G:529-618) equals `Q.mtp_forward_chunk` in every graph of the MTP context, for the
    processed pairs (A and B) and the draft chain (B): e = rms_norm(tok_embd) * enorm repeated over the hc streams,
    h_norm = per-stream rms_norm(h) * hnorm, eh_proj applied to [e ; h_norm] per stream (feature-axis concat), one
    QSA attention layer with the MTP context's own KV and k-pool cache, the MoE, and a head mixer with
    `nextn.hc_head_*` (not the trunk's `output_hc_*`); h_nextn = the block's residual. The target's h_nextn is its
    residual before the final mixer (G:501-506; equal to the last `l_last`), and the tool feeds it like
    common/speculative.cpp (checked row by row: `h fed`, `mtp_h_input`). The greedy draft chain (4 steps) equals the
    oracle's argmax chain. Worst error in the MTP context 1.6e-6 (`hc_inject` of the attention mixer, a draft graph).
12. **llama.cpp crash in an MTP context without a scheduler re-reserve (workaround in the tool).**
    `llama_set_embeddings_nextn` does not ask for a scheduler reserve (src/llama-context.cpp:1230-1235;
    `set_embeddings` just above has its reserve commented out, 1226-1227). Without a reserve, the first MTP graph
    crashes in `ggml_gallocr_init_tensor` (ggml/src/ggml-alloc.c:987) on the input `out_ids`, whose allocation record
    has `buffer_id = -1` (backtrace below; reproduced with `DUMP_MTP_NO_RESERVE=1`, exit 139; with the workaround
    exit 0). Mechanism, PROVISIONAL (read from the code, not traced): `ggml_gallocr_needs_realloc` compares node
    count, leaf count and nodes only (ggml/src/ggml-alloc.c:1009-1050), so a graph whose leaves differ from the graph
    reserved at context creation (built with h_nextn off) keeps the old leaf assignments. The tool re-sets a default
    that does ask for the reserve, `llama_set_embeddings_layer_inp(ctx, 0, false)` (src/llama-context.cpp:1237-1245).
    Whether llama-server's MTP path hits the same crash: UNKNOWN (common/speculative.cpp sets
    `llama_set_embeddings_nextn` after creating the draft context too, lines 1417-1418, but other calls such as
    `llama_set_sampler` with backend sampling may request a reserve before the first decode; not run). Upstream
    llama.cpp is not modified.

## Coverage: worst error per component (from the final run below)

Trunk runs, aggregated over the four runs from the per-run table of the final run (`python3 aggregate.py
final3_run.log`, a scratchpad helper that takes the maximum over runs; the per-run table is in the log). Every row
passed TOL = 1e-4; exact rows passed with 0. The aggregate of the run at 83634ed (before the MTP work) is identical
(`diff` below).

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

MTP context (printed by tests/test_qwen4exp_mtp_parity.py; "layer" is always the MTP layer 8, graph -1 = the whole
target run):

```
MTP: worst error per component over graphs (rel = max|got-want| / max|want|):
component                                                  layer graph    max abs        rel
h_nextn (MTP residual) [full]                                  8     9  2.008e-06  7.865e-07
h_nextn (MTP residual) [chunk]                                 8     9  2.008e-06  7.865e-07
h_nextn rows (h_nextn.bin) [full]                              8     9  2.008e-06  7.865e-07
h_nextn rows (h_nextn.bin) [chunk]                             8     9  2.008e-06  7.865e-07
hc_norm (head mixer) [full]                                    8     9  2.051e-06  8.578e-07
hc_norm (head mixer) [chunk]                                   8     9  2.051e-06  8.578e-07
result_norm [full]                                             8     9  8.959e-07  8.491e-07
result_norm [chunk]                                            8     9  8.959e-07  8.491e-07
result_output [full]                                           8     2  1.173e-06  8.207e-07
result_output [chunk]                                          8     2  1.173e-06  8.207e-07
logits.bin [full]                                              8     2  1.173e-06  8.207e-07
logits.bin [chunk]                                             8     2  1.173e-06  8.207e-07
h fed (process) [chunk]                                        8     0  5.276e-06  8.548e-07
h fed (process) [full]                                         8     0  5.276e-06  8.548e-07
mtp_h_input in the graph [chunk]                               8     0  5.276e-06  8.548e-07
mtp_tok_embd (exact)                                           8     0  0.000e+00  0.000e+00
mtp_hnorm (per-stream norm of h) [full]                        8     2  3.445e-06  1.026e-06
mtp_hnorm (per-stream norm of h) [chunk]                       8     2  3.445e-06  1.026e-06
mtp_enorm (normed embedding, repeated) [full]                  8     4  3.877e-07  1.763e-07
mtp_enorm (normed embedding, repeated) [chunk]                 8     4  3.877e-07  1.763e-07
mtp_eh_proj [full]                                             8     0  2.105e-06  6.539e-07
mtp_eh_proj [chunk]                                            8     0  2.105e-06  6.539e-07
h fed (draft) [chunk]                                          8    11  2.748e-06  5.327e-07
hc_norm (attention mixer) [full]                               8     9  1.671e-06  6.318e-07
hc_norm (attention mixer) [chunk]                              8     9  1.671e-06  6.318e-07
hc_mixed (attention mixer) [full]                              8     9  7.692e-07  6.328e-07
hc_mixed (attention mixer) [chunk]                             8     9  7.692e-07  6.328e-07
hc_inject (attention mixer) [full]                             8     2  2.065e-06  6.397e-07
hc_inject (attention mixer) [chunk]                            8    11  1.786e-06  1.577e-06
Qcur (IMRoPE) [full]                                           8     1  2.382e-06  7.545e-07
Qcur (IMRoPE) [chunk]                                          8     1  2.382e-06  7.545e-07
Kcur (IMRoPE) [full]                                           8     2  1.912e-06  6.963e-07
Kcur (IMRoPE) [chunk]                                          8     2  1.912e-06  6.963e-07
Vcur [full]                                                    8     2  9.477e-07  6.442e-07
Vcur [chunk]                                                   8     2  9.477e-07  6.442e-07
attn_pregate (qsa) [full]                                      8    10  7.666e-07  1.132e-06
attn_pregate (qsa) [chunk]                                     8    10  7.666e-07  1.132e-06
attn_output (wo) [full]                                        8     5  2.083e-07  1.159e-06
attn_output (wo) [chunk]                                       8     5  2.083e-07  1.159e-06
hc_combine (attn) [full]                                       8     9  2.029e-06  7.997e-07
hc_combine (attn) [chunk]                                      8     9  2.029e-06  7.997e-07
hc_norm (MoE mixer) [full]                                     8     9  2.033e-06  8.611e-07
hc_norm (MoE mixer) [chunk]                                    8     9  2.033e-06  8.611e-07
hc_mixed (MoE mixer) [full]                                    8     9  9.146e-07  8.891e-07
hc_mixed (MoE mixer) [chunk]                                   8     9  9.146e-07  8.891e-07
ffn_out (routed + gated shared) [full]                         8     0  3.221e-07  7.313e-07
ffn_out (routed + gated shared) [chunk]                        8    11  3.078e-07  8.405e-07
hc_combine (ffn) [full]                                        8     9  2.008e-06  7.865e-07
hc_combine (ffn) [chunk]                                       8     9  2.008e-06  7.865e-07
indexer_k_raw [full]                                           8     9  5.823e-07  6.834e-07
indexer_k_raw [chunk]                                          8     9  5.823e-07  6.834e-07
indexer_pool_k_new (pools completed here) [chunk]              8     0  1.146e-06  5.304e-07
indexer_pool_k_new (pools completed here) [full]               8     0  1.146e-06  5.304e-07
indexer_k (pool cache) [chunk]                                 8     0  1.146e-06  5.304e-07
indexer_k (pool cache) [full]                                  8     0  1.146e-06  5.304e-07
indexer_score (visible) [chunk]                                8     1  3.797e-06  7.857e-07
qsa selection: tie replays = 0                                 8    -1  0.000e+00  0.000e+00
target h_nextn rows (h_nextn.bin) [full]                       8    -1  5.276e-06  7.253e-07
target logits [full]                                           8    -1  1.506e-06  8.570e-07
target h_nextn tensor [full]                                   8     0  5.276e-06  8.548e-07
```

Integer and set comparisons, not in the tables: MoE top-k ids equal as sets in every graph and layer of the 5 runs
(A and B); QSA selections equal in every row (0 tie replays); pool visibility and masked padded pools; greedy argmax
of the logits equal in every row; MTP draft tokens equal the oracle's argmax chain.

## Mutation check

Each mutant is one single-point edit of the incremental path (S1-S12) or of the MTP block (T1-T5) of
ref/qwen4exp.py, applied to a copy of ref/ and the test files (tools/tinygen/mutate_qwen4exp_decode.py); the three
test modules run against the dumps of the final run. "dump parity failing" lists the failing tests of
tests/test_qwen4exp_decode_parity.py and tests/test_qwen4exp_mtp_parity.py (`<run class>.<test>`, or `setUpClass`
when the mutant oracle raised), "property failing" those of tests/test_qwen4exp_chunked_ref.py.

| mutant | what is broken | caught by the dump tests | caught by the property tests |
|---|---|---|---|
| S1 | delta-rule state reset at every ubatch / call | yes (all 4 trunk runs) | yes |
| S2 | GDN conv history dropped | yes (all 4 trunk runs) | yes |
| S3 | PLE conv history dropped | yes (all 4 trunk runs) | yes |
| S4 | PLE predecessors from earlier ubatches off by one | yes (all 4 trunk runs, incl. the exact row test) | yes |
| S5 | pooled keys of earlier ubatches / calls forgotten | yes (4 trunk runs and the MTP context) | yes |
| S6 | new pool rotated to its index within the ubatch | yes (4 trunk runs and the MTP context) | yes |
| S7 | dense attention forgets cells of earlier ubatches / calls (KV cache) | yes (all 4 trunk runs) | yes |
| S8 | RoPE positions restart in every ubatch / call | yes (4 trunk runs and the MTP context) | yes |
| S9 | QSA tail relative to the ubatch | yes (4 trunk runs and the MTP context; missed before 2362cbe, finding 9) | yes |
| S10 | QSA pool visibility relative to the ubatch | yes (the mutant oracle raises in setUpClass of 3 trunk runs and the MTP class: an empty cell set; ub16 tests fail) | yes |
| S11 | rollback keeps the delta-rule state and conv history of the rejected tokens | yes (rollback run) | yes |
| S12 | rollback keeps the PLE conv history of the rejected tokens | yes (rollback run) | yes |
| T1 | MTP: eh_proj input [h ; e] instead of [e ; h] | yes (MTP) | no (property tests compare the oracle with itself) |
| T2 | MTP: h normed over the whole hc-wide row | yes (MTP) | no |
| T3 | MTP: embedding norm without its enorm gamma | yes (MTP) | no |
| T4 | MTP: head mixer with the trunk's output_hc_* | yes (MTP) | no |
| T5 | MTP: next draft step's h = the projected input | yes (MTP) | no |

```
baseline (no mutation): rc=0 Ran 54 tests in 2.548s | OK
S1 GDN: delta-rule state reset at every ubatch / call
    ref/qwen4exp.py: 'out, state = gated_delta_rule(q, k, v, g, beta, state=state_in, normalize_qk=True, eps=eps)' -> 'out, state = gated_delta_rule(q, k, v, g, beta, state=None, normalize_qk=True, eps=eps)'
    rc=1 Ran 54 tests in 1.015s | FAILED (failures=31)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S2 GDN: conv history of the earlier ubatches dropped (zeros)
    ref/qwen4exp.py: 'conv_raw, conv_hist = short_conv_hist(qkv, w["ssm_conv1d.weight"], conv_in)' -> 'conv_raw, conv_hist = short_conv_hist(qkv, w["ssm_conv1d.weight"], np.zeros_like(conv_in))'
    rc=1 Ran 54 tests in 1.009s | FAILED (failures=31)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S3 PLE: conv history of the earlier ubatches dropped (zeros)
    ref/qwen4exp.py: 'full = np.vstack([hist_in, normed])' -> 'full = np.vstack([np.zeros_like(hist_in), normed])'
    rc=1 Ran 54 tests in 1.026s | FAILED (failures=31)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S4 PLE: predecessors from earlier ubatches off by one (newest cached token skipped)
    ref/qwen4exp.py: 'prefix = [int(t) for t in history[max(0, len(history) - n_prev):]]' -> 'prefix = [int(t) for t in history[max(0, len(history) - n_prev - 1):len(history) - 1]]'
    rc=1 Ran 54 tests in 0.984s | FAILED (failures=35)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_ple_module, Decode.test_ple_rows_across_calls, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_ple_rows_across_calls, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_ple_module, Ubatch10.test_ple_rows_across_calls, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_ple_module, Ubatch16.test_ple_rows_across_calls, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S5 QSA: pooled keys of earlier ubatches / calls forgotten (zeros)
    ref/qwen4exp.py: 'idxc["pooled"] = np.concatenate([idxc["pooled"], new])' -> 'idxc["pooled"] = np.concatenate([np.zeros_like(idxc["pooled"]), new])'
    rc=1 Ran 54 tests in 2.203s | FAILED (failures=32)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_chunked_mtp_equals_one_shot, test_every_chunking_reproduces_the_full_forward, test_first_draft_step_is_processing_the_next_pair, test_rollback_then_continue_equals_the_kept_sequence
S6 QSA: a new pool rotated to its index within the ubatch, not its sequence position
    ref/qwen4exp.py: 'new = imrope(new[:, None, :], pool_pos4(b1, kp)[b0:], **rope)[:, 0]' -> 'new = imrope(new[:, None, :], pool_pos4(b1 - b0, kp), **rope)[:, 0]'
    rc=1 Ran 54 tests in 1.755s | FAILED (failures=33)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_chunked_mtp_equals_one_shot, test_every_chunking_reproduces_the_full_forward, test_first_draft_step_is_processing_the_next_pair, test_rollback_then_continue_equals_the_kept_sequence
S7 KV cache: dense attention forgets the cells of earlier ubatches / calls
    ref/qwen4exp.py: 'cells = [list(range(0, t + 1)) for t in np.asarray(pos).tolist()]' -> 'cells = [list(range(int(pos[0]), t + 1)) for t in np.asarray(pos).tolist()]'
    rc=1 Ran 54 tests in 2.162s | FAILED (failures=14)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Rollback.test_attention_over_the_kv_cache, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines
    property failing: test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
S8 RoPE: positions restart at 0 in every ubatch / call
    ref/qwen4exp.py: 'p4 = text_pos4_at(pos)' -> 'p4 = text_pos4_at(pos - pos[0])'
    rc=1 Ran 54 tests in 1.100s | FAILED (failures=33)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_chunked_mtp_equals_one_shot, test_every_chunking_reproduces_the_full_forward, test_first_draft_step_is_processing_the_next_pair, test_rollback_then_continue_equals_the_kept_sequence
S9 QSA: incomplete tail taken relative to the ubatch
    ref/qwen4exp.py: 'for r, t in enumerate(np.asarray(pos).tolist()):' -> 'for r, t in enumerate(range(len(pos))):'
    rc=1 Ran 54 tests in 1.256s | FAILED (failures=33)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_chunked_mtp_equals_one_shot, test_every_chunking_reproduces_the_full_forward, test_first_draft_step_is_processing_the_next_pair, test_rollback_then_continue_equals_the_kept_sequence
S10 QSA: pool visibility relative to the ubatch
    ref/qwen4exp.py: 'visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.asarray(pos)[:, None] + 1)' -> 'visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.arange(len(pos))[:, None] + 1)'
    rc=1 Ran 18 tests in 0.700s | FAILED (failures=7, errors=8)
    dump parity failing: Decode.setUpClass, Mtp.setUpClass, Rollback.setUpClass, Ubatch10.setUpClass, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_chunked_mtp_equals_one_shot, test_every_chunking_reproduces_the_full_forward, test_first_draft_step_is_processing_the_next_pair, test_rollback_then_continue_equals_the_kept_sequence
S11 rollback: delta-rule state and conv history not rolled back (state after the rejected tokens)
    ref/qwen4exp.py: 'cache.clear()' -> 'fresh["gdn"] = cache["gdn"]\n    cache.clear()'
    rc=1 Ran 54 tests in 2.074s | FAILED (failures=8)
    dump parity failing: Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection
    property failing: test_rollback_then_continue_equals_the_kept_sequence
S12 rollback: PLE conv history not rolled back
    ref/qwen4exp.py: 'cache.clear()' -> 'fresh["ple"] = cache["ple"]\n    cache.clear()'
    rc=1 Ran 54 tests in 2.123s | FAILED (failures=8)
    dump parity failing: Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_ple_module, Rollback.test_qsa_pool_cache_and_selection
    property failing: test_rollback_then_continue_equals_the_kept_sequence
T1 MTP: eh_proj input concatenated as [h ; e] instead of [e ; h]
    ref/qwen4exp.py: 'res = np.concatenate([e_rep, h_norm], axis=-1) @ np.asarray(w["nextn.eh_proj.weight"]).T' -> 'res = np.concatenate([h_norm, e_rep], axis=-1) @ np.asarray(w["nextn.eh_proj.weight"]).T'
    rc=1 Ran 54 tests in 2.745s | FAILED (failures=4)
    dump parity failing: Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache
    property failing: none
T2 MTP: h normed over the whole hc-wide row instead of per stream
    ref/qwen4exp.py: 'h_norm = grouped_rms_norm(np.asarray(h_prev, dtype=np.float64).reshape(T, hc, n), w["nextn.hnorm.weight"], eps)' -> 'h_norm = rms_norm(np.asarray(h_prev, dtype=np.float64), eps).reshape(T, hc, n) * w["nextn.hnorm.weight"]'
    rc=1 Ran 54 tests in 2.400s | FAILED (failures=4)
    dump parity failing: Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache
    property failing: none
T3 MTP: embedding norm without its enorm gamma
    ref/qwen4exp.py: 'e = rms_norm(W["token_embd.weight"][tokens], eps, w["nextn.enorm.weight"])' -> 'e = rms_norm(W["token_embd.weight"][tokens], eps)'
    rc=1 Ran 54 tests in 2.361s | FAILED (failures=4)
    dump parity failing: Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache
    property failing: none
T4 MTP: the head mixer uses the trunk's output_hc_* instead of nextn.hc_head_*
    ref/qwen4exp.py: 'head = hc_mix(res, w["nextn.hc_head_norm.weight"], w["nextn.hc_head_down.weight"], w["nextn.hc_head_up.weight"],' -> 'head = hc_mix(res, W["output_hc_norm.weight"], W["output_hc_down.weight"], W["output_hc_up.weight"],'
    rc=1 Ran 54 tests in 2.479s | FAILED (failures=1)
    dump parity failing: Mtp.test_mtp_head_and_draft_chain
    property failing: none
T5 MTP: the next draft step's h is the projected input, not the block output
    ref/qwen4exp.py: 'out["h_nextn"] = res.reshape(T, hc * n)' -> 'out["h_nextn"] = out["eh_proj"].reshape(T, hc * n)'
    rc=1 Ran 54 tests in 2.379s | FAILED (failures=4)
    dump parity failing: Mtp.test_mtp_head_and_draft_chain, Mtp.test_mtp_inputs_and_projection, Mtp.test_mtp_layer, Mtp.test_mtp_qsa_pool_cache
    property failing: none
caught 17/17; caught by the dump parity tests alone 17/17
```

The S9 hole in the earlier rounds (same mutant; both rounds against the dumps of the driver run made with the tests
of c766cb4, `<S>/n6/tmp.p2Xv5rdO5L`; they ran on the working tree before 2362cbe; a first round is not shown because
its runner did not yet count `setUpClass` errors; the `#` lines are labels added here, the rest is literal):

```
# round 2 (setUpClass restructured, tie fix not yet applied): mut2.log
baseline (no mutation): rc=0 Ran 44 tests in 2.929s | OK
caught 12/12; caught by the dump parity tests alone 11/12
S9 QSA: incomplete tail taken relative to the ubatch
    ref/qwen4exp.py: 'for r, t in enumerate(np.asarray(pos).tolist()):' -> 'for r, t in enumerate(range(len(pos))):'
    rc=1 Ran 44 tests in 4.288s | FAILED (failures=3)
    dump parity failing: none
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
# round 3 (tie fix applied; committed as 2362cbe): mut3.log
baseline (no mutation): rc=0 Ran 44 tests in 2.590s | OK
caught 12/12; caught by the dump parity tests alone 12/12
S9 QSA: incomplete tail taken relative to the ubatch
    ref/qwen4exp.py: 'for r, t in enumerate(np.asarray(pos).tolist()):' -> 'for r, t in enumerate(range(len(pos))):'
    rc=1 Ran 44 tests in 2.041s | FAILED (failures=27)
    dump parity failing: Decode.test_attention_over_the_kv_cache, Decode.test_carried_state, Decode.test_gdn_layers, Decode.test_head_and_logits, Decode.test_moe_ids_and_combines, Decode.test_qsa_pool_cache_and_selection, Rollback.test_attention_over_the_kv_cache, Rollback.test_carried_state, Rollback.test_gdn_layers, Rollback.test_head_and_logits, Rollback.test_moe_ids_and_combines, Rollback.test_qsa_pool_cache_and_selection, Ubatch10.test_attention_over_the_kv_cache, Ubatch10.test_carried_state, Ubatch10.test_gdn_layers, Ubatch10.test_head_and_logits, Ubatch10.test_moe_ids_and_combines, Ubatch10.test_qsa_pool_cache_and_selection, Ubatch16.test_attention_over_the_kv_cache, Ubatch16.test_carried_state, Ubatch16.test_gdn_layers, Ubatch16.test_head_and_logits, Ubatch16.test_moe_ids_and_combines, Ubatch16.test_qsa_pool_cache_and_selection
    property failing: test_carried_state_is_the_state_of_the_prefix, test_every_chunking_reproduces_the_full_forward, test_rollback_then_continue_equals_the_kept_sequence
```

## Not covered (UNKNOWN / NOT RUN)

| what | status | why / what would close it |
|---|---|---|
| MTP: the multi-head chain (`chain_heads`), MTP sharing the target memory (`is_mem_shared`), an MTP-only GGUF, MTP with `n_rs_seq` > 0 / rollback of drafts in the MTP context, sampling other than greedy top-1, backend sampling | NOT COVERED | one MTP head (nextn_predict_layers = 1), its own memory, greedy draft; qwen4exp MTP asserts a single block (G:531) |
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
(read only), `<repo>` = the worktree of this branch. Scratchpad helpers named below are not committed.

### Pin check

```
$ cd <llama.cpp> && git rev-parse HEAD && git status --short | wc -l
ec7630a640789c393694fb194f1bbbf0369fc62d
0
```

### Final run at 2c93501 from an empty build directory (what the CI step does), full `-v` output

```
$ git log --oneline -n 1
2c93501 [parity-qa] dump_tensors: the workaround comment cites set_embeddings_nextn itself (src/llama-context.cpp:1230-1235), not the commented-out reserve of set_embeddings above it; ci: N6 step name mentions MTP (no behaviour change)
$ TMPDIR=<S>/n6 bash -c 'time tools/tinygen/run_parity_qwen4exp_decode.sh <llama.cpp> <S>/build-n6-fresh3' > final3_run.log 2>&1; echo "exit=$?" >> final3_run.log
$ cat final3_run.log
CMAKE_BUILD_TYPE=Release
wrote <S>/n6/tmp.0pdwZ8LR8A/qwen4exp.gguf
wrote <S>/n6/tmp.0pdwZ8LR8A/qwen4exp_mtp.gguf
ub16: 4 graphs, 1 calls, n_rs_seq 0
ub10: 7 graphs, 1 calls, n_rs_seq 0
decode: 17 graphs, 17 calls, n_rs_seq 0
rollback: 16 graphs, 17 calls, n_rs_seq 3
mtp: 11 graphs, 9 calls, n_rs_seq 0
mtp (MTP context): 15 graphs, [('process', 48), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('draft', 1), ('draft', 1), ('draft', 1), ('draft', 1)]
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
test_mtp_head_and_draft_chain (tests.test_qwen4exp_mtp_parity.Mtp.test_mtp_head_and_draft_chain)
h_nextn (the next step's h), the head mixer, logits; the greedy draft chain. ... ok
test_mtp_inputs_and_projection (tests.test_qwen4exp_mtp_parity.Mtp.test_mtp_inputs_and_projection)
h fed with each MTP token (pairing of common/speculative.cpp), e/h norms and eh_proj (G:558-589). ... ok
test_mtp_layer (tests.test_qwen4exp_mtp_parity.Mtp.test_mtp_layer) ... ok
test_mtp_qsa_pool_cache (tests.test_qwen4exp_mtp_parity.Mtp.test_mtp_qsa_pool_cache)
The MTP context's own k-pool cache: pools completed by process and draft calls, read back later. ... ok
test_run_exercises_its_mechanisms (tests.test_qwen4exp_mtp_parity.Mtp.test_run_exercises_its_mechanisms) ... ok
test_target_h_nextn_and_logits (tests.test_qwen4exp_mtp_parity.Mtp.test_target_h_nextn_and_logits)
The rows the MTP context is fed: the target's h_nextn (before its final mixer) and its logits. ... ok

MTP: worst error per component over graphs (rel = max|got-want| / max|want|):
component                                                  layer graph    max abs        rel
h_nextn (MTP residual) [full]                                  8     9  2.008e-06  7.865e-07
h_nextn (MTP residual) [chunk]                                 8     9  2.008e-06  7.865e-07
h_nextn rows (h_nextn.bin) [full]                              8     9  2.008e-06  7.865e-07
h_nextn rows (h_nextn.bin) [chunk]                             8     9  2.008e-06  7.865e-07
hc_norm (head mixer) [full]                                    8     9  2.051e-06  8.578e-07
hc_norm (head mixer) [chunk]                                   8     9  2.051e-06  8.578e-07
result_norm [full]                                             8     9  8.959e-07  8.491e-07
result_norm [chunk]                                            8     9  8.959e-07  8.491e-07
result_output [full]                                           8     2  1.173e-06  8.207e-07
result_output [chunk]                                          8     2  1.173e-06  8.207e-07
logits.bin [full]                                              8     2  1.173e-06  8.207e-07
logits.bin [chunk]                                             8     2  1.173e-06  8.207e-07
h fed (process) [chunk]                                        8     0  5.276e-06  8.548e-07
h fed (process) [full]                                         8     0  5.276e-06  8.548e-07
mtp_h_input in the graph [chunk]                               8     0  5.276e-06  8.548e-07
mtp_tok_embd (exact)                                           8     0  0.000e+00  0.000e+00
mtp_hnorm (per-stream norm of h) [full]                        8     2  3.445e-06  1.026e-06
mtp_hnorm (per-stream norm of h) [chunk]                       8     2  3.445e-06  1.026e-06
mtp_enorm (normed embedding, repeated) [full]                  8     4  3.877e-07  1.763e-07
mtp_enorm (normed embedding, repeated) [chunk]                 8     4  3.877e-07  1.763e-07
mtp_eh_proj [full]                                             8     0  2.105e-06  6.539e-07
mtp_eh_proj [chunk]                                            8     0  2.105e-06  6.539e-07
h fed (draft) [chunk]                                          8    11  2.748e-06  5.327e-07
hc_norm (attention mixer) [full]                               8     9  1.671e-06  6.318e-07
hc_norm (attention mixer) [chunk]                              8     9  1.671e-06  6.318e-07
hc_mixed (attention mixer) [full]                              8     9  7.692e-07  6.328e-07
hc_mixed (attention mixer) [chunk]                             8     9  7.692e-07  6.328e-07
hc_inject (attention mixer) [full]                             8     2  2.065e-06  6.397e-07
hc_inject (attention mixer) [chunk]                            8    11  1.786e-06  1.577e-06
Qcur (IMRoPE) [full]                                           8     1  2.382e-06  7.545e-07
Qcur (IMRoPE) [chunk]                                          8     1  2.382e-06  7.545e-07
Kcur (IMRoPE) [full]                                           8     2  1.912e-06  6.963e-07
Kcur (IMRoPE) [chunk]                                          8     2  1.912e-06  6.963e-07
Vcur [full]                                                    8     2  9.477e-07  6.442e-07
Vcur [chunk]                                                   8     2  9.477e-07  6.442e-07
attn_pregate (qsa) [full]                                      8    10  7.666e-07  1.132e-06
attn_pregate (qsa) [chunk]                                     8    10  7.666e-07  1.132e-06
attn_output (wo) [full]                                        8     5  2.083e-07  1.159e-06
attn_output (wo) [chunk]                                       8     5  2.083e-07  1.159e-06
hc_combine (attn) [full]                                       8     9  2.029e-06  7.997e-07
hc_combine (attn) [chunk]                                      8     9  2.029e-06  7.997e-07
hc_norm (MoE mixer) [full]                                     8     9  2.033e-06  8.611e-07
hc_norm (MoE mixer) [chunk]                                    8     9  2.033e-06  8.611e-07
hc_mixed (MoE mixer) [full]                                    8     9  9.146e-07  8.891e-07
hc_mixed (MoE mixer) [chunk]                                   8     9  9.146e-07  8.891e-07
ffn_out (routed + gated shared) [full]                         8     0  3.221e-07  7.313e-07
ffn_out (routed + gated shared) [chunk]                        8    11  3.078e-07  8.405e-07
hc_combine (ffn) [full]                                        8     9  2.008e-06  7.865e-07
hc_combine (ffn) [chunk]                                       8     9  2.008e-06  7.865e-07
indexer_k_raw [full]                                           8     9  5.823e-07  6.834e-07
indexer_k_raw [chunk]                                          8     9  5.823e-07  6.834e-07
indexer_pool_k_new (pools completed here) [chunk]              8     0  1.146e-06  5.304e-07
indexer_pool_k_new (pools completed here) [full]               8     0  1.146e-06  5.304e-07
indexer_k (pool cache) [chunk]                                 8     0  1.146e-06  5.304e-07
indexer_k (pool cache) [full]                                  8     0  1.146e-06  5.304e-07
indexer_score (visible) [chunk]                                8     1  3.797e-06  7.857e-07
qsa selection: tie replays = 0                                 8    -1  0.000e+00  0.000e+00
target h_nextn rows (h_nextn.bin) [full]                       8    -1  5.276e-06  7.253e-07
target logits [full]                                           8    -1  1.506e-06  8.570e-07
target h_nextn tensor [full]                                   8     0  5.276e-06  8.548e-07
test_carried_state_is_the_state_of_the_prefix (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_carried_state_is_the_state_of_the_prefix) ... ok
test_every_chunking_reproduces_the_full_forward (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_every_chunking_reproduces_the_full_forward) ... ok
test_guards (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_guards)
The random model must exercise what the tests above claim: sparse QSA, EOS resets, pools that straddle a ... ok
test_rollback_then_continue_equals_the_kept_sequence (tests.test_qwen4exp_chunked_ref.ChunkedEqualsFull.test_rollback_then_continue_equals_the_kept_sequence) ... ok
test_chunked_mtp_equals_one_shot (tests.test_qwen4exp_chunked_ref.MtpChunkedEqualsOneShot.test_chunked_mtp_equals_one_shot) ... ok
test_first_draft_step_is_processing_the_next_pair (tests.test_qwen4exp_chunked_ref.MtpChunkedEqualsOneShot.test_first_draft_step_is_processing_the_next_pair) ... ok
test_guards (tests.test_qwen4exp_chunked_ref.MtpChunkedEqualsOneShot.test_guards) ... ok
test_trunk_unchanged_by_the_mtp_block (tests.test_qwen4exp_chunked_ref.MtpChunkedEqualsOneShot.test_trunk_unchanged_by_the_mtp_block) ... ok

----------------------------------------------------------------------
Ran 54 tests in 2.473s

OK
artifacts: <S>/n6/tmp.0pdwZ8LR8A

real	2m7.753s
user	6m41.618s
sys	0m22.030s
exit=0
```

The first line (`CMAKE_BUILD_TYPE=Release`) is printed by the llama.cpp CMake configure on stderr, as in N2.

### Manifests of the rollback run and of the MTP context

```
$ cat <artifacts>/n6/rollback/manifest.json; cat <artifacts>/n6/tokens.txt <artifacts>/n6/rollback/tokens.txt
{"n_ubatch":512,"n_rs_seq":3,"per_graph":true,"first_node":"model.input_embed","calls":[{"kind":"decode","n":48,"pos0":0,"tok0":0,"graphs":[0,1]},{"kind":"decode","n":4,"pos0":48,"tok0":48,"graphs":[1,2]},{"kind":"rollback","n":2,"pos0":50,"tok0":52,"graphs":[2,2]},{"kind":"decode","n":1,"pos0":50,"tok0":52,"graphs":[2,3]},{"kind":"decode","n":1,"pos0":51,"tok0":53,"graphs":[3,4]},{"kind":"decode","n":1,"pos0":52,"tok0":54,"graphs":[4,5]},{"kind":"decode","n":1,"pos0":53,"tok0":55,"graphs":[5,6]},{"kind":"decode","n":1,"pos0":54,"tok0":56,"graphs":[6,7]},{"kind":"decode","n":1,"pos0":55,"tok0":57,"graphs":[7,8]},{"kind":"decode","n":1,"pos0":56,"tok0":58,"graphs":[8,9]},{"kind":"decode","n":1,"pos0":57,"tok0":59,"graphs":[9,10]},{"kind":"decode","n":1,"pos0":58,"tok0":60,"graphs":[10,11]},{"kind":"decode","n":1,"pos0":59,"tok0":61,"graphs":[11,12]},{"kind":"decode","n":1,"pos0":60,"tok0":62,"graphs":[12,13]},{"kind":"decode","n":1,"pos0":61,"tok0":63,"graphs":[13,14]},{"kind":"decode","n":1,"pos0":62,"tok0":64,"graphs":[14,15]},{"kind":"decode","n":1,"pos0":63,"tok0":65,"graphs":[15,16]}],"graphs":[{"call":0,"n_asked":1118,"n_dumped":523},{"call":1,"n_asked":1118,"n_dumped":523},{"call":3,"n_asked":1118,"n_dumped":523},{"call":4,"n_asked":1118,"n_dumped":523},{"call":5,"n_asked":1118,"n_dumped":523},{"call":6,"n_asked":1118,"n_dumped":523},{"call":7,"n_asked":1118,"n_dumped":523},{"call":8,"n_asked":1118,"n_dumped":523},{"call":9,"n_asked":1118,"n_dumped":523},{"call":10,"n_asked":1118,"n_dumped":523},{"call":11,"n_asked":1118,"n_dumped":523},{"call":12,"n_asked":1118,"n_dumped":523},{"call":13,"n_asked":1118,"n_dumped":523},{"call":14,"n_asked":1118,"n_dumped":523},{"call":15,"n_asked":1118,"n_dumped":523},{"call":16,"n_asked":1118,"n_dumped":523}]}
76,13,65,36,7,3,21,87,78,63,50,43,5,37,65,2,55,71,72,90,15,27,75,73,92,36,87,81,90,14,57,45,14,2,55,35,59,92,15,28,92,84,40,15,8,78,28,2,49,65,27,68,76,85,92,2,2,84,49,2,80,2,41,48
76,13,65,36,7,3,21,87,78,63,50,43,5,37,65,2,55,71,72,90,15,27,75,73,92,36,87,81,90,14,57,45,14,2,55,35,59,92,15,28,92,84,40,15,8,78,28,2,49,65,44,85,27,68,76,85,92,2,2,84,49,2,80,2,41,48
$ cat <artifacts>/n6/mtp/mtp/manifest.json
{"n_ubatch":16,"n_rs_seq":0,"per_graph":true,"n_embd_out":256,"first_node":"mtp_tok_embd-8","calls":[{"kind":"process","n":48,"pos0":0,"tok0":0,"graphs":[0,3],"tokens":[76,13,65,36,7,3,21,87,78,63,50,43,5,37,65,2,55,71,72,90,15,27,75,73,92,36,87,81,90,14,57,45,14,2,55,35,59,92,15,28,92,84,40,15,8,78,28,2]},{"kind":"process","n":1,"pos0":48,"tok0":48,"graphs":[3,4],"tokens":[49]},{"kind":"process","n":1,"pos0":49,"tok0":49,"graphs":[4,5],"tokens":[65]},{"kind":"process","n":1,"pos0":50,"tok0":50,"graphs":[5,6],"tokens":[27]},{"kind":"process","n":1,"pos0":51,"tok0":51,"graphs":[6,7],"tokens":[68]},{"kind":"process","n":1,"pos0":52,"tok0":52,"graphs":[7,8],"tokens":[76]},{"kind":"process","n":1,"pos0":53,"tok0":53,"graphs":[8,9],"tokens":[85]},{"kind":"process","n":1,"pos0":54,"tok0":54,"graphs":[9,10],"tokens":[92]},{"kind":"process","n":1,"pos0":55,"tok0":55,"graphs":[10,11],"tokens":[2]},{"kind":"draft","n":1,"pos0":56,"tok0":56,"graphs":[11,12],"tokens":[0]},{"kind":"draft","n":1,"pos0":57,"tok0":57,"graphs":[12,13],"tokens":[91]},{"kind":"draft","n":1,"pos0":58,"tok0":58,"graphs":[13,14],"tokens":[40]},{"kind":"draft","n":1,"pos0":59,"tok0":59,"graphs":[14,15],"tokens":[82]}],"graphs":[{"call":0,"n_asked":182,"n_dumped":88},{"call":0,"n_asked":182,"n_dumped":88},{"call":0,"n_asked":182,"n_dumped":88},{"call":1,"n_asked":182,"n_dumped":88},{"call":2,"n_asked":182,"n_dumped":88},{"call":3,"n_asked":182,"n_dumped":88},{"call":4,"n_asked":182,"n_dumped":88},{"call":5,"n_asked":182,"n_dumped":88},{"call":6,"n_asked":182,"n_dumped":88},{"call":7,"n_asked":182,"n_dumped":88},{"call":8,"n_asked":182,"n_dumped":88},{"call":9,"n_asked":182,"n_dumped":88},{"call":10,"n_asked":182,"n_dumped":88},{"call":11,"n_asked":182,"n_dumped":88},{"call":12,"n_asked":182,"n_dumped":88}]}
```

### Mutation check at 2c93501, against the dumps of the final run

```
$ python3 tools/tinygen/mutate_qwen4exp_decode.py <repo> <S>/n6/tmp.0pdwZ8LR8A/n6 <S>/n6/tmp.0pdwZ8LR8A/qwen4exp.gguf <llama.cpp>/gguf-py <S>/n6/mutfinal3
(output above, in "Mutation check")
$ python3 aggregate.py final3_run.log > final3_aggregate.txt; diff head_aggregate.txt final3_aggregate.txt && echo "decode-run aggregate identical to 83634ed's"
decode-run aggregate identical to 83634ed's
```

`head_aggregate.txt` is the aggregate of the run at 83634ed (the version of this file in d07f6e9 has that run).

### The MTP crash and the workaround (finding 12), with the dump tool at 2c93501

`<S>/build-n6-dbg` is a RelWithDebInfo build of the same sources (tools/g0, `-DCMAKE_BUILD_TYPE=RelWithDebInfo`).
Scratchpad script crash_repro.sh: 48 tokens, `DUMP_CALLS=48`, `-ub 16`, `DUMP_MTP=1`.

```
$ bash crash_repro.sh <S>/build-n6-fresh3/dump-tensors <S>/build-n6-dbg/dump-tensors <S>/n6/tmp.0pdwZ8LR8A/qwen4exp_mtp.gguf
crash_repro.sh: line 10: 20578 Segmentation fault      DUMP_DIR=crash_rel DUMP_TOKENS="$TOK" DUMP_PREFIXES="result_" DUMP_CALLS="48" DUMP_MTP=1 DUMP_MTP_NO_RESERVE=$nr "$REL" -m "$GGUF" -c 512 -b 512 -ub 16 -t 1 -fa off -ctk f32 -ctv f32 > crash_rel.log 2>&1
release build, DUMP_MTP_NO_RESERVE=1: exit=139
release build, DUMP_MTP_NO_RESERVE=0: exit=0
Thread 1 "dump-tensors" received signal SIGSEGV, Segmentation fault.
ggml_gallocr_init_tensor (galloc=0x5555557163f0, tensor=0x7fffdcdec150, tensor_alloc=0x55555574ed30) at <S>/llama.cpp/ggml/src/ggml-alloc.c:987
987	            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
#0  ggml_gallocr_init_tensor (galloc=0x5555557163f0, tensor=0x7fffdcdec150, tensor_alloc=0x55555574ed30) at <S>/llama.cpp/ggml/src/ggml-alloc.c:987
#1  0x00007ffff7f47a78 in ggml_gallocr_alloc_graph (galloc=0x5555557163f0, graph=0x5555556ed288) at <S>/llama.cpp/ggml/src/ggml-alloc.c:1081
#2  0x00007ffff7f4de69 in ggml_backend_sched_alloc_splits (sched=0x5555556ed130) at <S>/llama.cpp/ggml/src/ggml-backend.cpp:1627
#3  ggml_backend_sched_alloc_graph (sched=0x5555556ed130, graph=<optimized out>) at <S>/llama.cpp/ggml/src/ggml-backend.cpp:2018
#4  0x00007ffff77eeba2 in llama_context::process_ubatch (this=this@entry=0x555555627e20, ubatch=..., gtype=LLM_GRAPH_TYPE_DECODER_MTP, mctx=mctx@entry=0x5555557ba300, ret=@0x7fffffff93ec: GGML_STATUS_SUCCESS) at /usr/include/c++/13/bits/unique_ptr.h:199
#5  0x00007ffff77f5acc in llama_context::decode (this=0x555555627e20, batch_inp=...) at <S>/llama.cpp/src/llama-context.cpp:1891
#6  0x000055555555cb58 in operator() (__closure=0x7fffffff9de0, kind="process", ids=std::vector of length 48, capacity 48 = {...}, pos0=pos0@entry=0, h_rows=std::vector of length 48, capacity 48 = {...}) at <repo>/tools/dump/dump_tensors.cpp:311
#7  0x000055555555d793 in operator() (__closure=__closure@entry=0x7fffffff9e80, tok0=<optimized out>, pos0=pos0@entry=0, n=n@entry=48) at /usr/include/c++/13/bits/basic_string.tcc:242
#8  0x000055555555a04a in main (argc=<optimized out>, argv=<optimized out>) at <repo>/tools/dump/dump_tensors.cpp:392
$1 = "out_ids", '\000' <repeats 56 times>
$2 = 1
$3 = {buffer_id = -1, addr = {chunk = -1, offset = 18446744073709551615}, size_max = 0}
```

### The default modes of the dump tool and of the generator are unchanged

`<S>/build-n2-lead/dump-tensors` was built (2026-10-10 11:32) from the main checkout's tools/dump/dump_tensors.cpp,
which equals the file at c1a0336 (`cmp` below; the file is dated 2026-10-09 21:45). Same N2 model, tokens, prefixes and
flags (scratchpad script same_as_old.sh):

```
$ git show c1a0336:tools/dump/dump_tensors.cpp > dump_tensors.c1a0336.cpp && cmp /home/user/408032gb/tools/dump/dump_tensors.cpp dump_tensors.c1a0336.cpp && echo same-as-c1a0336
same-as-c1a0336
$ bash same_as_old.sh <S>/n6/tmp.nbblgtrcuq <S>/build-n6-fresh3/dump-tensors
old exit=0 files=1022
new exit=0 files=1022
identical: 1022 files
$ cmp <S>/n6/tmp.0pdwZ8LR8A/qwen4exp.gguf <S>/n6/tmp.VmAUcfydpw/qwen4exp.gguf && echo "default tiny GGUF at 2c93501 byte-identical to the one generated at 83634ed"
default tiny GGUF at 2c93501 byte-identical to the one generated at 83634ed
```

The existing users of the tool pass with the final binary:

```
$ TMPDIR=<S>/n6 bash tools/tinygen/run_parity_qwen4exp.sh <llama.cpp> <S>/build-n6-fresh3 > regress3_qwen4exp.log 2>&1; echo "exit=$?"
exit=0
$ TMPDIR=<S>/n6 bash tools/tinygen/run_parity.sh <llama.cpp> <S>/build-n6-fresh3 > regress3_glm.log 2>&1; echo "exit=$?"
exit=0
$ TMPDIR=<S>/n6 bash tools/tinygen/run_parity_mimo.sh <llama.cpp> <S>/build-n6-fresh3 > regress3_mimo.log 2>&1; echo "exit=$?"
exit=0
$ grep -E "^Ran|^OK|^FAILED" regress3_qwen4exp.log regress3_glm.log regress3_mimo.log
regress3_qwen4exp.log:Ran 17 tests in 0.223s
regress3_qwen4exp.log:OK
regress3_glm.log:Ran 8 tests in 0.182s
regress3_glm.log:OK
regress3_mimo.log:Ran 5 tests in 0.056s
regress3_mimo.log:OK
```

### N2's tie replay with a mutant confined to `Q.qsa_select` (finding 10; on a copy, scratchpad n2_select_mutant.py)

```
$ python3 n2_select_mutant.py <repo> <S>/n6/tmp.nbblgtrcuq/dump <S>/n6/tmp.nbblgtrcuq/qwen4exp.gguf <llama.cpp>/gguf-py <S>/n6
mutant: ref/qwen4exp.py qsa_select: 'top = vis[np.argsort(-scores[t, vis], kind="stable")[:n_top]]' -> 'top = vis[np.argsort(-scores[t, vis], kind="stable")[:n_top - 1]]'
tests.test_qwen4exp_tiny_parity: rc=0 Ran 17 tests in 0.296s | OK; failing: none
tests.test_qwen4exp_ref: rc=1 Ran 16 tests in 0.017s | FAILED (failures=1); failing: test_selection_is_top_pools_plus_tail
```

### Unit tests (what the CI `unit tests` step runs), at 2c93501

```
$ python3 -m unittest discover -s tests > discover_final3.log 2>&1; echo "exit=$?"; tail -n 4 discover_final3.log
exit=0
----------------------------------------------------------------------
Ran 422 tests in 94.176s

OK (skipped=86)
```
