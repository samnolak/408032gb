# N9: glm5-next (GLM-5.3-Flash) several ubatches, token-by-token decode, rollback and MTP, CPU parity, 2026-10-10 (sandbox CPU, no GPU)

Task N9 of the hardware-free track: close gap 4 of docs/reassessment/CORRECTNESS.md (GLM had oracle parity for one
prefill only, op by op, tests/test_glm_tiny_parity.py). Same method as N6 for qwen4exp
(docs/evidence/2026-10-10-qwen4exp-decode-parity.md). Branch `claude/n9-glm-decode-parity`, cut from
`origin/claude/research-landscape` at `9ef5f68`. Role: parity-qa.

References: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) and
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin, tools/engines/pins.json), both CPU builds of `tools/g0`
target `dump-tensors`, the real `glm5-next` graph. "G" = src/models/glm5-next.cpp, "M" = src/llama-memory-hybrid-idx.cpp
at ec7630a unless "@79e2e74" is written; every cited line was printed from the pinned checkout before citing.

| piece | file |
|---|---|
| oracle | ref/glm5next.py: `forward()` (whole model from ref/mhc.py, ref/kda.py, ref/dsa.py, ref/router.py), the incremental path `forward_chunk` / `cache_truncate` / `forward_chunks`, the MTP block `mtp_forward_chunk`, `hparams()` from GGUF fields |
| prefill check of the whole-model oracle | tests/test_glm_tiny_parity.py `GlmTinyWholeModel` (new class; the 8 op tests unchanged) |
| dump parity tests | tests/test_glm5next_decode_parity.py: 9 tests per run, 5 runs = 45; tests/test_glm5next_mtp_parity.py: 6 tests (only where the pin has the MTP graph); both skip without `GLM5NEXT_DECODE_DUMP`, the GGUF variables and `LLAMA_CPP_GGUF_PY` |
| property tests | tests/test_glm5next_chunked_ref.py: 7 tests, numpy only, random 5-layer model + MTP layer (runs in the CI `unit tests` step) |
| driver | tools/tinygen/run_parity_glm5next_decode.sh (builds dump-tensors, writes the tiny GGUF with and without the MTP block, runs the dumps, runs the three test modules; `PARITY_LLAMA_COMMIT` from the checkout) |
| generator | tools/tinygen/glm5next_tiny.py `--mtp`, taken verbatim from `origin/claude/n7-server-hybrid-cpu` (default output byte-identical, below) |
| mutation check | docs/evidence/mutation-scripts/2026-10-10-glm5next-decode.py (15 mutants on copies; the repository is never edited) |
| CI | step after N6 in `.github/workflows/ci.yml` job `cpu-parity` (ec7630a) and in `.github/workflows/parity-r1-pin.yml` (R1 pin); NOT RUN in GitHub Actions from this branch (not pushed by this role) |

## Runs

Model: the G1 tiny model (tools/tinygen/glm5next_tiny.py --layers 12, seed 0): 12 layers, KDA (4 heads of 16, conv
kernel 4, gate lower bound -5) on 0-2, 4-6, 8-10; DSA (nope MLA, kv_lora 16, k-pool indexer with kpool 4, budget 2 pools +
tail, 2 indexer heads) on 3, 7, 11; dense SwiGLU on layer 0; MoE 8 experts top-2 + 1 shared expert elsewhere; mHC with
hc = 4 and 20 Sinkhorn iterations; SwiGLU clamp 0.5 on every FFN. For the MTP run the same model with `--mtp`
(block_count 13, nextn_predict_layers 1, layer 12 a DSA + MoE layer with `blk.12.nextn.{eh_proj,enorm,hnorm}`, no mHC;
trunk weights identical). Flags as G1: `-c 512 -b 512 -t 1 -fa off -ctk f32 -ctv f32`. One 64-token sequence
(`random.seed(9)`, ids in [3, 96)).

| run | `-ub` | llama_process calls (`DUMP_CALLS`) | graphs | what it adds |
|---|---|---|---|---|
| ub16 | 16 | one call of 64 tokens | 4 x 16 | carry across ubatches whose boundaries coincide with pool boundaries |
| ub10 | 10 | one call of 64 tokens | 6 x 10 + 1 x 4 | pools straddling two ubatches, a short last ubatch |
| ub18 | 18 | one call of 64 tokens | 3 x 18 + 1 x 10 | ubatches of more than 16 tokens continuing a cache (the mask path at ec7630a), straddling pools |
| decode | 512 | `48,1,...,1`: prefill 48, then 16 one-token calls | 1 x 48 + 16 x 1 | decode with n_tokens = 1: KDA state and conv history, MLA latent cache, pools completed by one-token calls |
| rollback | 512, `n_rs_seq` 3 | `48,4,r2,1,...,1`: prefill 48; a 4-token draft 48..51 whose ids at 50-51 are wrong; `llama_memory_seq_rm(mem, 0, 50, -1)`; 14 one-token calls | 1 + 1 + 14 | rollback from the KDA snapshots (G:217-227, src/models/delta-net-base.cpp:563-603), pool 12 first pooled from wrong rows and pooled again, cells re-used |
| mtp (79e2e74 only) | 16 | target: `48,1,...,1` over the first 56 tokens (3 + 8 graphs); MTP context: the same 9 calls + 4 greedy draft steps at 56..59 | 11 + 15 | MTP block driven like common/speculative.cpp (tools/dump/dump_tensors.cpp `DUMP_MTP`): token p paired with the target h_nextn of p-1, the draft chain on the MTP's own h_nextn, the MTP context's own MLA and k-pool cache |

## Method

As N6. `DUMP_PER_GRAPH=1` writes every computed graph (one per ubatch) into `g<NNN>/`; `manifest.json` maps calls to
graphs and positions. Every graph is compared row by row with (A) `[full]`: `G.forward(final 64 tokens)` at the graph's
positions (prefill semantics of ref/dsa.py, KDA from a zero state; for the MTP context `G.mtp_forward_chunk` over all
processed positions at once, fed by the full-sequence target oracle) and (B) `[chunk]`: `G.forward_chunk` /
`G.mtp_forward_chunk` replaying the same calls (rollback = `G.cache_truncate`). Rows of rejected draft tokens and MTP
draft steps have no (A). Components: mHC (`hc_post`, `hc_comb` both sides, `hc_attn_pre`, `hc_attn_post`,
`hc_ffn_pre`, `l_out`), KDA (`attn_norm`, `kda_{q,k,v}_conv`, `kda_g1`, `kda_beta`, `kda_scan_out`, `kda_normed`,
`kda_out`, `new_state`), DSA (`q_resid`, `kv_cmpr`, `q_absorbed`, `indexer_{k,gate,q,weights}`, `indexer_pool_k_new`
incl. padding rows, `indexer_pool_k`, `indexer_score`, the selection, `kv_gathered` / `kq_soft_max_gathered` on the
ec7630a gather path, `kqv_out`, `attn_out`), FFN (`ffn_norm`, `ffn_moe_topk`, `ffn_moe_weights_scaled`, `ffn_moe_out`,
`ffn_shexp`, `ffn_out`), head (`hc_head`, `result_norm`, `result_output`, logits.bin, greedy argmax); MTP
(`mtp_tok_embd`, `mtp_enorm`, `mtp_hnorm`, `mtp_eh_proj`, the DSA tensors of layer 12, `mtp_ffn_*`, `mtp_post_ffn`,
`mtp_shared_head_norm`, `result_output`, h_in.bin / h_nextn.bin rows, the draft chain).

The carried state: `new_state-<il>` (KDA state written back, `n_rs_seq` = 0 only) is compared with (B) and with the
delta rule run over the first p0 + n tokens of the full-sequence q, k, v, g, beta (A). The conv history and the state
a graph read have no `cb()` name in glm5-next.cpp (G:206-213, 733-735), so they are checked through the rows that
read them: separate components for the first d_conv - 1 rows of `kda_{q,k,v}_conv` and the first row of
`kda_scan_out` of every graph with p0 > 0.

Tolerances as N6: normwise relative error max|got - want| / max|want| < 1e-4 for f32 paths; exact for the embedding
gather; MoE ids equal as sets; DSA selection equal, or another valid top-k by the oracle's own scores (`G.tie_variant`,
replayed only if the oracle's own selection is valid too), counted. Pin-dependent details follow `PARITY_LLAMA_COMMIT`
(the gather path, the padding rows; finding 2, 3).

## Findings (CONFIRMED by the literal outputs below unless labelled)

1. **No llama.cpp-vs-oracle mismatch at either pin.** ec7630a: 45/45 dump tests OK, worst normwise error 4.431e-05
   (`kda_scan_out` first row of graph 6, layer 9, ub10); 79e2e74: 45/45 + 6/6 MTP OK, worst 3.605e-05 (`kda_out`
   layer 8, ub16), MTP worst 1.675e-05 (`mtp_ffn_out`). Every greedy argmax equal; DSA tie replays 0 in every run.
   No tolerance was loosened; the only check moved is finding 4.
2. **ec7630a has a second DSA path that only small ubatches take.** For n_tokens <= 16 and n_kv > n_sel the graph
   gathers the selected latents and runs a dense softmax over n_sel = 11 slots with a host-side slot mask (G:324-331,
   883-886, 971-996) instead of the additive mask over all 256 cells (G:889-918, 998-1005). So at ec7630a ub16, ub10,
   the last ubatch of ub18, every decode step and the draft call run the gather path; the 48-token prefills and the
   first three ubatches of ub18 run the mask path (asserted per graph). 79e2e74 removed the gather path (G:990-1078
   @79e2e74): all graphs take the mask path there (asserted). The slot mask is not dumped; the test derives it from
   M:978-1011, 1013-1037 (first min(nv, 2) pool slots, then tail slots k < (t+1) % 4) and checks it against the
   dumped softmax (exactly 0 on dead slots); `kv_gathered` live rows equal the cached latents of the selected cells.
3. **k-pool padding rows differ by pin** (as for qwen4exp in N6): ec7630a re-pools one shared cell (cell 0 if a pool
   is new, else the ubatch's first cell; M:947-967), 79e2e74 one non-rep cell per padded entry (M:888-905 @79e2e74).
   A padded entry has kpool equal members, so its pooled key is that cell's indexer key; both rules hold at their pin
   (mutant U2 below shows the check bites).
4. **`kda_normed` amplifies upstream f32 error; checked as an op.** End to end it reached 9.7e-05 at ec7630a (ub18
   L10 pos 15 head 3), next to the 1e-4 bound. Localised (scratch probe, literal output below): that head row of the
   scan has RMS 1.6e-3 against max |scan| 0.50; its absolute error 1.6e-06 is in line with the rest of the tensor
   (normwise 1.6e-05), but the per-head RMS norm rescales the row to O(1), so its relative error 3.0e-04 becomes
   9.7e-05 normwise. The op itself, rms_norm(dumped scan) * ssm_norm, matches the dumped `kda_normed` to 9.98e-08.
   The test now checks `kda_normed` as an op on the dumped scan and keeps `kda_out` (which does not rescale rows) end
   to end. Not a llama.cpp bug (PROVISIONAL: the mechanism is the arithmetic above; no other cause was looked for).
5. **MTP exists at 79e2e74 only.** ec7630a throws "GLM5-Next NextN graph not implemented yet" for
   LLM_GRAPH_TYPE_DECODER_MTP (G:189-191), so there the driver skips the run and the MTP test skips (output below).
   At 79e2e74 graph_mtp (G:543-667 @79e2e74) has no mHC: eh_proj [enorm e ; hnorm h] -> DSA layer over the MTP
   context's own cache -> MoE + shared expert with plain residuals -> head norm; h is the target's post-norm h_nextn
   (n_embd wide, G:780-787 @79e2e74). In the masked MTP context the tensor named `h_nextn` is renamed
   `mtp_shared_head_norm` by the next `cb()` (G:650-656 @79e2e74), so the test reads it under that name.
6. **`glm5next_tiny.py --mtp` already existed on `claude/n7-server-hybrid-cpu`** (CORRECTNESS.md section 0); this
   branch first wrote its own (different draw order of the nextn tensors), then took the n7 file verbatim
   (`git diff origin/claude/n7-server-hybrid-cpu -- tools/tinygen/glm5next_tiny.py` is empty) so both branches build the
   same MTP model. With the superseded draw the MTP context had 3 DSA selections among pools tied at exactly 0.0 score
   (all heads' relu zero), replayed as valid ties (scratch probe output below); with the final model 0 ties.
7. **Mutation check** (literal output below): ec7630a 12/15 caught, 79e2e74 14/15; every mutant is caught at one pin
   at least, and each miss is a mutant of a path that pin does not have (T1, T2, U3: MTP, no graph at ec7630a; U1: the
   gather path, gone at 79e2e74). All ten state-carry mutants of the incremental path (S1-S10) are caught by the dump
   tests at both pins, and also by the property tests.

## Not covered (gaps that remain)

* several sequences / shared cells (`cache_safe` = false at ec7630a), rollback longer than `n_rs_seq`, rollback inside
  a multi-ubatch call, rollback in the MTP context: NOT RUN.
* f16 KV cache, flash attention, quantised weights (gaps 2-3 of CORRECTNESS.md): NOT RUN, every run here is f32.
* whether the CPU took the fused or the unfused branches of the mHC / lightning-indexer / delta-net ops
  (`cparams.fused_*`): UNKNOWN, not recorded; only the branch the CPU took is checked.
* the optional nextn tensors (`embed_tokens`, `shared_head_head`, `shared_head_norm`): not in the tiny model, so the
  MTP graph's fall-back to `token_embd` / `output` / `output_norm` is what is checked.
* CI: the two new workflow steps NOT RUN in GitHub Actions (this branch is not pushed by this role).

## Literal outputs

Pin check:

```
$ git -C <scratch>/llama-ec7630a rev-parse HEAD; git -C <scratch>/llama.cpp rev-parse HEAD
ec7630a640789c393694fb194f1bbbf0369fc62d
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
```

Environment: sandbox, 4 CPU cores, Python 3.13.16, numpy 2.5.3, no GPU.

### Whole-model oracle on the G1 prefill dump (before the incremental path)

`tools/tinygen/run_parity.sh <llama.cpp> <build>` (now with `GlmTinyWholeModel`), both pins, on HEAD `7d99e0b`:

```
llama.cpp: ec7630a640789c393694fb194f1bbbf0369fc62d
test_dsa_indexer_and_selection (tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_indexer_and_selection) ... ok
test_dsa_mla_attention (tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_mla_attention) ... ok
test_kda_gate (tests.test_glm_tiny_parity.GlmTinyParity.test_kda_gate) ... ok
test_kda_recurrence (tests.test_glm_tiny_parity.GlmTinyParity.test_kda_recurrence) ... ok
test_kda_short_conv (tests.test_glm_tiny_parity.GlmTinyParity.test_kda_short_conv) ... ok
test_mhc_attn (tests.test_glm_tiny_parity.GlmTinyParity.test_mhc_attn) ... ok
test_router (tests.test_glm_tiny_parity.GlmTinyParity.test_router) ... ok
test_swiglu_clamp (tests.test_glm_tiny_parity.GlmTinyParity.test_swiglu_clamp) ... ok
test_forward_matches_dump (tests.test_glm_tiny_parity.GlmTinyWholeModel.test_forward_matches_dump) ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.421s

OK
```

```
llama.cpp: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
test_dsa_indexer_and_selection (tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_indexer_and_selection) ... ok
test_dsa_mla_attention (tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_mla_attention) ... ok
test_kda_gate (tests.test_glm_tiny_parity.GlmTinyParity.test_kda_gate) ... ok
test_kda_recurrence (tests.test_glm_tiny_parity.GlmTinyParity.test_kda_recurrence) ... ok
test_kda_short_conv (tests.test_glm_tiny_parity.GlmTinyParity.test_kda_short_conv) ... ok
test_mhc_attn (tests.test_glm_tiny_parity.GlmTinyParity.test_mhc_attn) ... ok
test_router (tests.test_glm_tiny_parity.GlmTinyParity.test_router) ... ok
test_swiglu_clamp (tests.test_glm_tiny_parity.GlmTinyParity.test_swiglu_clamp) ... ok
test_forward_matches_dump (tests.test_glm_tiny_parity.GlmTinyWholeModel.test_forward_matches_dump) ... ok
----------------------------------------------------------------------
Ran 9 tests in 0.401s

OK
```

Per-layer errors of `forward()` on the G1 dump at ec7630a (scratch script `check_forward.py`, same comparisons as
`GlmTinyWholeModel` plus the mixers; the three largest of 88 lines, sorted):

```
$ python3 check_forward.py <g1 dump dir> <ec7630a>/gguf-py | awk '{print $NF, $0}' | sort -g -r | sed -n 1,3p
1.56e-05 kda_out-9                                rel 1.56e-05
1.37e-05 kqv_out-11                               rel 1.37e-05
1.22e-05 kda_out-10                               rel 1.22e-05
```

### Generator: default output byte-identical

```
$ genhash.sh before/          # glm5next_tiny.py before --mtp: default, --layers 12, --layers 12 --seed 3, --layers 8 --arch glm5next
26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5  default.gguf
26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5  l12.gguf
bd479a47fb26f16cc17783a5ac6193396520ddbea06dabedfb9aa659d780c3a5  l12s3.gguf
52236e9485f3f1a9746ce012af719be8a0abb873da515628cb925b7a8073f499  arch.gguf
$ genhash.sh n7/; diff before.txt n7.txt && echo DEFAULT-OUTPUT-IDENTICAL
DEFAULT-OUTPUT-IDENTICAL
```

### ec7630a: `tools/tinygen/run_parity_glm5next_decode.sh <ec7630a> <build>` on HEAD `7d99e0b`

`...` below stands for the verbose per-test lines (`... ok`), omitted; nothing else is cut.

```
ub16: 4 graphs, 1 calls, n_rs_seq 0
ub10: 7 graphs, 1 calls, n_rs_seq 0
ub18: 4 graphs, 1 calls, n_rs_seq 0
decode: 17 graphs, 17 calls, n_rs_seq 0
rollback: 16 graphs, 17 calls, n_rs_seq 3
mtp: not run, this llama.cpp has no glm5-next MTP graph (src/models/glm5-next.cpp)
...
setUpClass (tests.test_glm5next_mtp_parity.Mtp) ... skipped 'no MTP run under /tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n9/work-ec/n9 (llama.cpp ec7630a: no glm5-next MTP graph at ec7630a, see the module docstring)'
...
Ran 52 tests in 6.024s

OK (skipped=1)
```

(52 = 45 dump tests + 7 property tests; the MTP class is skipped as one.) Worst error per layer, as printed:

```
worst error per layer over runs, components and graphs (layer -1 = embedding and head), llama.cpp ec7630a:
layer run      component                                                  graph    max abs        rel
   -1 ub10     result_output [full]                                           1  5.138e-05  1.670e-05
    0 decode   kda_out [full]                                                11  1.929e-06  1.590e-06
    1 rollback kda_scan_out [full]                                           10  1.571e-07  3.450e-06
    2 ub10     ffn_shexp (shared, clamped) [chunk]                            2  1.491e-06  4.676e-06
    3 rollback indexer_score (visible) [chunk]                                5  1.079e-06  9.368e-06
    4 ub10     ffn_moe_out (routed) [full]                                    6  4.177e-06  7.608e-06
    5 ub18     kda_out [full]                                                 0  1.646e-05  1.004e-05
    6 rollback hc_post (attn) [full]                                         13  1.721e-06  1.819e-05
    7 ub10     hc_post (attn) [full]                                          6  3.083e-05  1.542e-05
    8 ub16     kda_out [full]                                                 0  4.656e-05  3.281e-05
    9 ub10     kda_scan_out (first row, reads the carried state) [full]       6  9.673e-07  4.431e-05
   10 ub16     kda_out [full]                                                 0  4.347e-05  2.933e-05
   11 ub10     hc_comb (ffn, Sinkhorn) [full]                                 1  3.691e-05  3.692e-05
```

The largest entries of the per-run, per-component table (`awk '/^worst error per run/{f=1;next} /^$/{f=0} f && $0 !~ /^run /' runner-ec.log | awk '{print $NF"\t"$0}' | sort -g -r | cut -f2- | head -8`):

```
ub10     kda_scan_out (first row, reads the carried state) [full]       9     6  9.673e-07  4.431e-05
ub10     kda_scan_out (first row, reads the carried state) [chunk]      9     6  9.673e-07  4.431e-05
ub16     hc_comb (ffn, Sinkhorn) [full]                                11     0  3.691e-05  3.692e-05
ub16     hc_comb (ffn, Sinkhorn) [chunk]                               11     0  3.691e-05  3.692e-05
ub10     hc_comb (ffn, Sinkhorn) [full]                                11     1  3.691e-05  3.692e-05
ub10     hc_comb (ffn, Sinkhorn) [chunk]                               11     1  3.691e-05  3.692e-05
ub18     hc_comb (ffn, Sinkhorn) [full]                                11     0  3.369e-05  3.370e-05
ub18     hc_comb (ffn, Sinkhorn) [chunk]                               11     0  3.369e-05  3.370e-05
```

### 79e2e74 (R1 pin): the same driver on HEAD `7d99e0b`

```
ub16: 4 graphs, 1 calls, n_rs_seq 0
ub10: 7 graphs, 1 calls, n_rs_seq 0
ub18: 4 graphs, 1 calls, n_rs_seq 0
decode: 17 graphs, 17 calls, n_rs_seq 0
rollback: 16 graphs, 17 calls, n_rs_seq 3
mtp: 11 graphs, 9 calls, n_rs_seq 0
mtp (MTP context): 15 graphs, [('process', 48), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('process', 1), ('draft', 1), ('draft', 1), ('draft', 1), ('draft', 1)]
...
Ran 58 tests in 5.971s

OK
```

```
worst error per layer over runs, components and graphs (layer -1 = embedding and head), llama.cpp 79e2e74:
layer run      component                                                  graph    max abs        rel
   -1 ub10     result_output [full]                                           1  4.983e-05  1.619e-05
    0 decode   kda_out [full]                                                11  1.929e-06  1.590e-06
    1 rollback kda_scan_out [full]                                           10  1.571e-07  3.450e-06
    2 ub10     ffn_shexp (shared, clamped) [chunk]                            2  1.491e-06  4.676e-06
    3 rollback indexer_score (visible) [chunk]                                5  1.079e-06  9.368e-06
    4 ub10     ffn_moe_out (routed) [full]                                    6  4.013e-06  7.310e-06
    5 ub10     kda_out [full]                                                 1  1.646e-05  1.040e-05
    6 decode   hc_post (attn) [full]                                         14  1.892e-06  2.000e-05
    7 ub10     hc_post (attn) [full]                                          6  3.322e-05  1.661e-05
    8 ub16     kda_out [full]                                                 0  5.115e-05  3.605e-05
    9 ub10     kda_scan_out (first row, reads the carried state) [full]       6  7.810e-07  3.577e-05
   10 ub16     kda_out [full]                                                 0  4.639e-05  3.130e-05
   11 ub10     hc_comb (ffn, Sinkhorn) [full]                                 1  3.369e-05  3.370e-05
```

MTP (layer 12), the largest entries of its table (same awk on the `MTP: worst error` block), and the tie counters of
all runs (`grep "tie replays" runner-r1.log`):

```
mtp_ffn_out [full]                                            12     0  1.054e-05  1.675e-05
mtp_ffn_out [chunk]                                           12     0  1.054e-05  1.675e-05
mtp_ffn_shexp (shared, clamped) [full]                        12     0  5.082e-06  1.402e-05
mtp_ffn_shexp (shared, clamped) [chunk]                       12     0  5.082e-06  1.402e-05
mtp_ffn_moe_out (routed) [full]                               12     0  7.761e-06  1.383e-05
mtp_ffn_moe_out (routed) [chunk]                              12     0  7.761e-06  1.383e-05
decode   dsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
rollback dsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
ub10     dsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
ub16     dsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
ub18     dsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
dsa selection: tie replays = 0                                12    -1  0.000e+00  0.000e+00
```

### Finding 4: the `kda_normed` probe (scratch `normed_probe.py ub18 0 10`, ec7630a dumps)

```
worst normed elem (np.int64(15), np.int64(3), np.int64(1)) abs 0.00037940292689642163 max|normed| 3.9150449434357393
scan rms of that head row 0.0016223160222096778  median rms 0.010567406323598824  max|scan| 0.5044316949490562
scan abs err in that row 1.6149853978991713e-06  row rel 0.00030418844657143917
per-row rel err of the scan (max over rows) 0.00030418844657143917
op-level: rms_norm(dumped scan) vs dumped normed, rel 9.978474451465864e-08
```

### Finding 6: the tie replays with the superseded MTP draw (scratch `ties_probe.py`, 79e2e74 dumps of that model)

```
pos 19 graph 1 (process): reference [8, 9, 10, 11, 16, 17, 18, 19] oracle [4, 5, 6, 7, 16, 17, 18, 19]; visible pool scores {0: -0.044525746, 1: 0.0, 2: 0.0, 3: -0.089643179, 4: 0.001931746}
pos 43 graph 2 (process): reference [28, 29, 30, 31, 40, 41, 42, 43] oracle [16, 17, 18, 19, 40, 41, 42, 43]; visible pool scores {0: -0.593622139, 1: -0.031526585, 2: -0.60668572, 3: -0.910722758, 4: 0.0, 5: -0.332357556, 6: -0.219137386, 7: 0.0, 8: 0.0, 9: -0.271815524, 10: 0.017305586}
pos 46 graph 2 (process): reference [8, 9, 10, 11, 16, 17, 18, 19, 44, 45, 46] oracle [0, 1, 2, 3, 16, 17, 18, 19, 44, 45, 46]; visible pool scores {0: 0.0, 1: -0.161817489, 2: 0.0, 3: 0.0, 4: 0.035896604, 5: 0.0, 6: 0.0, 7: 0.0, 8: -0.623256373, 9: -0.12069579, 10: -0.865162887}
```

### Mutation check

`python3 docs/evidence/mutation-scripts/2026-10-10-glm5next-decode.py <repo> <artifacts> <llama.cpp> <scratch>` on HEAD
`7d99e0b`, once per pin against that pin's dumps; shown through
`grep "^llama\|^baseline\|^[STU][0-9]*\|^    rc=\|^caught" <log> | cut -c1-120` (the per-mutant lists of failing tests
are in the full log). The mutant edits are in the script.

```
llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d; dumps /tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86
baseline (no mutation): rc=0 Ran 52 tests in 6.422s | OK (skipped=1)
S1 KDA: state reset at every ubatch / call
    rc=1 Ran 52 tests in 4.528s | FAILED (failures=38, skipped=1)
S2 KDA: conv history of the earlier ubatches dropped (zeros)
    rc=1 Ran 52 tests in 5.124s | FAILED (failures=38, skipped=1)
S3 KDA: conv history keeps the oldest rows instead of the newest
    rc=1 Ran 52 tests in 5.013s | FAILED (failures=38, skipped=1)
S4 DSA: MLA latents of the earlier ubatches / calls forgotten (zeros)
    rc=1 Ran 52 tests in 5.599s | FAILED (failures=39, skipped=1)
S5 DSA: pooled keys of the earlier ubatches / calls forgotten (zeros)
    rc=1 Ran 52 tests in 6.823s | FAILED (failures=39, skipped=1)
S6 DSA: a pool straddling two ubatches loses the gate rows of its earlier members (zeros)
    rc=1 Ran 52 tests in 6.341s | FAILED (failures=32, skipped=1)
S7 DSA: incomplete tail taken relative to the ubatch
    rc=1 Ran 52 tests in 4.720s | FAILED (failures=39, skipped=1)
S8 DSA: pool visibility relative to the ubatch
    rc=1 Ran 16 tests in 1.548s | FAILED (failures=7, errors=9, skipped=1)
S9 rollback: KDA conv history and state not rolled back (state after the rejected tokens)
    rc=1 Ran 52 tests in 6.102s | FAILED (failures=7, skipped=1)
S10 rollback: MLA latent / indexer cache not rolled back (rejected cells kept)
    rc=1 Ran 52 tests in 6.602s | FAILED (failures=7, skipped=1)
T1 MTP: eh_proj input concatenated as [h ; e] instead of [e ; h]
    rc=0 Ran 52 tests in 5.746s | OK (skipped=1)
T2 MTP: FFN residual taken from the normed input instead of eh_proj
    rc=0 Ran 52 tests in 6.591s | OK (skipped=1)
U1 test: gather-path tail slot live one cell too far (k <= n_tail)
    rc=1 Ran 52 tests in 5.659s | FAILED (failures=5, skipped=1)
U2 test: k-pool padding rule of the other pin
    rc=1 Ran 52 tests in 6.227s | FAILED (failures=2, skipped=1)
U3 test: MTP token p paired with the target h of p instead of p-1
    rc=0 Ran 52 tests in 6.742s | OK (skipped=1)
caught 12/15; caught by the dump parity tests alone 12/15
```

```
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6; dumps /tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86
baseline (no mutation): rc=0 Ran 58 tests in 6.507s | OK
S1 KDA: state reset at every ubatch / call
    rc=1 Ran 58 tests in 4.605s | FAILED (failures=38)
S2 KDA: conv history of the earlier ubatches dropped (zeros)
    rc=1 Ran 58 tests in 5.080s | FAILED (failures=38)
S3 KDA: conv history keeps the oldest rows instead of the newest
    rc=1 Ran 58 tests in 5.116s | FAILED (failures=38)
S4 DSA: MLA latents of the earlier ubatches / calls forgotten (zeros)
    rc=1 Ran 58 tests in 5.457s | FAILED (failures=43)
S5 DSA: pooled keys of the earlier ubatches / calls forgotten (zeros)
    rc=1 Ran 58 tests in 6.895s | FAILED (failures=42)
S6 DSA: a pool straddling two ubatches loses the gate rows of its earlier members (zeros)
    rc=1 Ran 58 tests in 6.665s | FAILED (failures=36)
S7 DSA: incomplete tail taken relative to the ubatch
    rc=1 Ran 58 tests in 4.632s | FAILED (failures=43)
S8 DSA: pool visibility relative to the ubatch
    rc=1 Ran 16 tests in 1.861s | FAILED (failures=7, errors=10)
S9 rollback: KDA conv history and state not rolled back (state after the rejected tokens)
    rc=1 Ran 58 tests in 6.245s | FAILED (failures=7)
S10 rollback: MLA latent / indexer cache not rolled back (rejected cells kept)
    rc=1 Ran 58 tests in 6.714s | FAILED (failures=7)
T1 MTP: eh_proj input concatenated as [h ; e] instead of [e ; h]
    rc=1 Ran 58 tests in 6.397s | FAILED (failures=4)
T2 MTP: FFN residual taken from the normed input instead of eh_proj
    rc=1 Ran 58 tests in 6.564s | FAILED (failures=4)
U1 test: gather-path tail slot live one cell too far (k <= n_tail)
    rc=0 Ran 58 tests in 6.392s | OK
U2 test: k-pool padding rule of the other pin
    rc=1 Ran 58 tests in 6.108s | FAILED (failures=3)
U3 test: MTP token p paired with the target h of p instead of p-1
    rc=1 Ran 58 tests in 6.770s | FAILED (failures=4)
caught 14/15; caught by the dump parity tests alone 14/15
```

(S8 "Ran 16 tests": the mutant makes `setUpClass` of several test classes raise; unittest counts each as one
error and does not run that class's tests; the round counts the mutant as caught.)

The full unit-test suite and `git log` for this report are in the final report of the task (they move with every
commit).
