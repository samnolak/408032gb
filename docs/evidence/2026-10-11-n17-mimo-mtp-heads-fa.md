# N17: mimo2 (MiMo-V2.x-Flash) three MTP heads (nextn_layer_offset 0-2) and the R1 default precision (-ctk f16 -ctv f16 -fa on), CPU parity, 2026-10-11 (sandbox CPU, no GPU)

Task N17 of the hardware-free track: the two mimo2 gaps the real MiMo-V2.6-Flash config made relevant (N16 evidence
`docs/evidence/2026-10-11-n16-mimo-decode-parity.md`, "Real config" and "Not covered"): (1) the MTP heads with
`nextn_layer_offset` 1 and 2 (`num_nextn_predict_layers` 3, `third_party/hf/mimo-v2.6-flash-chat/config.json:162`@1ce4d32;
llama.cpp's converter writes all three, `conversion/mimo.py:22-24`@79e2e74), and (2) the R1 default precision of the KV
cache and the attention (`-ctk f16 -ctv f16 -fa on`, `scripts/r1/configs/llamacpp-server.json`) for mimo2, whose sinks
(SWA layers only, `config.json` `add_swa_attention_sink_bias` true) go through the CPU flash attention. Optional item 3
(rollback in the MTP context) is done as part of (1). Branch `claude/n17-mimo-mtp-heads-fa`, cut from
`claude/n16-mimo-decode-parity` at `0b0eace`. Role: parity-qa (CI wiring commit as infra-ops). `<S>` stands for the
session scratch directory `/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`;
outputs are pasted with only that substitution.

References: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) and `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`
(R1 engine pin, tools/engines/pins.json), CPU builds of the `tools/g0` target `dump-tensors` (`GGML_NATIVE=OFF`: AVX2,
FMA, F16C on, AVX-512 off), fresh build directories `<S>/n17/build-ec`, `<S>/n17/build-r1` configured from this worktree
(the N16 builds were configured from `/home/user/wt-n16`, so they were not reused). Every cited line was printed from
the pinned checkouts (section "Source lines"; "O" = `ggml/src/ggml-cpu/ops.cpp`, "M" = `src/models/mimo2.cpp`,
"SP" = `common/speculative.cpp`, "SC" = `tools/server/server-context.cpp`, "KV" = `src/llama-kv-cache.cpp`).

| piece | file |
|---|---|
| generator | `tools/tinygen/mimo2_tiny.py --mtp-layers N` (N MTP blocks; block 6 drawn as with `--mtp`) and `--head-dims K,V` (192,128 = MiMo-V2.6-Flash); default, `--seed 3` and N16's MTP model byte-identical, `--mtp-layers 1` = `--mtp` (below) |
| dump tool | `tools/dump/dump_tensors.cpp` DUMP_MTP: several heads chained like SP `chain_heads`, plan steps `d<n>` (draft, then the server's seq_rm) and `r<n>` (rejection: seq_rm of both contexts, `accept()`'s pending row); the MTP manifest records head, outputs, rollback and seq_rm steps; N16's one-head run is reproduced bit for bit (below) |
| oracles | `ref/mimo2.py`: `mtp_forward_chunk(offset=k)`, `mtp_process`, `mtp_draft`, the per-layer record of which position each cell's K/V was computed for (`attn["foreign"]`, `kv_pos`), `prec=` (f16 store, llama.cpp's f32 kq scale `kq_scale`, CPU flash attention with sinks); `ref/fattn.py`: sinks in every CPU attention mode, `kq_scale_f32`, the f32 argument of `expf` in one_chunk (a flagged change to N10's oracle), the AVX2 lanes of `ggml_vec_dot_f16` for heads >= 32 wide |
| driver | `tools/tinygen/run_parity_mimo2_n17.sh` (`MIMO2_N17_WORK` keeps the artifacts) |
| dump parity tests | `tests/test_mimo2_mtp_heads_parity.py` (6 f32 + 7 f16/FA), `tests/test_mimo2_f16fa_parity.py` (5 runs x 8); skip without `MIMO2_N17_DUMP` and `LLAMA_CPP_GGUF_PY` |
| property tests | `tests/test_mimo2_heads_ref.py` (8), `tests/test_fattn_sinks_ref.py` (9), numpy only; also in the CI `unit tests` step |
| CI | step `n17-mimo2-mtp-heads-fa` in `.github/workflows/ci.yml` job `cpu-parity` (ec7630a) and `.github/workflows/parity-r1-pin.yml` job `parity` (79e2e74), through `tools/ci/check_unittest.py`; `tools/ci/parity_expectations.json`: Ran 70, no skips, at both commits. N16's step and expectations unchanged. The driver took 52 to 67 s here (4 cores, dump tool already built; 7.2, 7.8); on a GitHub runner UNKNOWN |
| scripts | `docs/evidence/mutation-scripts/2026-10-11-n17-*`: mutations, foreign-probe, fa-op-probe, pindiff, n16compat, bites, genhash, source-lines, final-runs, unit |

## 1. How llama.cpp chains several MTP heads (CONFIRMED from the source at both pins)

The chaining code is the same at both pins: `diff` of SP 1340-1747 @ec7630a against 1401-1828 @79e2e74 differs only in
the sampler (79e2e74 adds probabilistic drafting: `result_q`, `spec_retune`, and resets the sampler in `begin()`); the
lines below are identical in content (section "Source lines").

* **Graph.** `graph_mtp` builds one MTP layer, `il = n_layer + cparams.nextn_layer_offset`, and asserts
  `0 <= offset < n_layer_nextn` (M:276-279 @ec7630a, M:267-270 @79e2e74). The offset is set by
  `llama_set_nextn_layer_offset` (`src/llama-ext.h:101` at both pins; `src/llama-context.cpp:1248-1250` @ec7630a,
  `:1264-1266` @79e2e74) and a graph built for another offset is not reused (`src/llama-graph.h:873-875` @ec7630a,
  `:884-886` @79e2e74). The MTP context's memory holds every layer `il >= n_layer` (`src/llama-model.cpp:2807-2812`
  @ec7630a, `:2846-2851` @79e2e74), here layers 6, 7, 8, all sliding-window layers on the cells of one SWA cache.
  **Both pins run offsets 1 and 2** (CONFIRMED by the runs below: graphs named `mtp_*-7` and `mtp_*-8`).
* **Who drives several heads.** `common_speculative_impl_draft_mtp`: `n_mtp_layers = max(1, n_layer_nextn)` (SP:1378
  @ec7630a, 1439 @79e2e74), `chain_heads = n_mtp_layers > 1 && !is_mem_shared`, and with chain_heads the draft length is
  capped at the number of heads (SP:1420-1425 @ec7630a, 1481-1486 @79e2e74). llama-server calls it (`process()` after
  every target decode, `draft()`, `accept()`; SC:3044, 3747, 3964 @ec7630a, 3404, 4232, 4471 @79e2e74).
* **process() (every target batch):** each token k of the batch is paired with the target's h_nextn row k-1, the first
  with the pending row (SP:1519-1529 @ec7630a, 1583-1593 @79e2e74); then **for head = 0 .. H-1:
  `llama_memory_seq_rm(mtp, seq, first position of the batch, -1)`, `llama_set_nextn_layer_offset(head)`, decode the
  same batch with the same rows**; the offset is set back to 0 (SP:1533-1557 @ec7630a, 1597-1621 @79e2e74). So in
  process **every head takes the target's h** (not another head's), and each head writes the K/V of its own layer for
  every processed position.
* **draft() (one sequence):** step 0 runs head 0 on `[id_last]` at pos0 with the pending row (SP:1604-1611 @ec7630a,
  1681-1688 @79e2e74). Step i runs `seq_rm(pos0)` and offset i (SP:1623-1631 / 1700-1708), on the batch
  `[id_last, d_1, .., d_i]` at pos0 .. pos0+i with the rows `chain_h = [pending, h_0, .., h_{i-1}]`, where h_j is the
  h_nextn that step j output at its last (only) output row, position pos0+j (SP:1686-1696 @ec7630a, 1767-1777 @79e2e74;
  only the last row is an output). **So head k takes, for its new token d_k at pos0+k, the output h of head k-1 at
  pos0+k-1 ("the previous head's output"), and for the earlier rows the outputs of heads 0 .. k-2 and the target's
  pending row at pos0.** Greedy with the default `p_min = 0` (`common/common.h:332` @ec7630a, `:333` @79e2e74) the draft
  has exactly H steps.
* **llama-server around it:** after drafting, the draft positions are removed from the MTP memory
  (`llama_memory_seq_rm(ctx_dft, slot, ckpt.pos_max + 1, -1)`, SC:3063 @ec7630a, 3423 @79e2e74); after verification,
  `accept()` moves the pending row to the target row of the last accepted position (SP:1743-1745 @ec7630a, 1824-1826
  @79e2e74) and `common_memory::seq_rm` cuts both contexts after it (SC:3998 @ec7630a, 4505 @79e2e74;
  `common/common.cpp:1558-1563` @ec7630a). The iSWA cache accepts partial seq_rm, so the server takes this path and not
  checkpoints (`common_context_can_seq_rm`, `common/common.cpp:1502-1527` @ec7630a, N16 finding 4: seq_rm returns true).

The dump tool replays exactly these calls (`DUMP_MTP=1`, plan steps `d<n>`, `<n>`, `r<n>`); the deviations from the
server are documented in its header: process calls mark every row as an output (the server marks none; the tensors
before the output crop, M:378 @79e2e74, do not depend on it) and the token after a rejection or a draft is taken from
`DUMP_TOKENS` (teacher-forced) rather than sampled.

## 2. Runs

Model `n17.gguf` = `tools/tinygen/mimo2_tiny.py --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 3` (seed 0): the
N16 trunk (6 layers, global 0 and 5, SWA 1-4 with window 8, 4 query heads, KV heads 1 / 2, K head 24, V head 16, RoPE on
8 dims, dense layer 0, MoE 8 experts top-2), MTP block 6 identical to N16's `n16.gguf`, and MTP blocks 7, 8 (same layout:
fused attn_qkv, sinks, dense FFN, eh_proj / enorm / hnorm, layer_output_norm). `n17hd.gguf` = the same options with
`--head-dims 192,128` and no MTP block: MiMo-V2.6-Flash's K 192 / V 128 heads (`config.json:43, 315`@1ce4d32), RoPE on
64 dims. Token sequence: N16's 300 ids (`random.seed(16)`). Flags `-c 512 -b 512 -t 1`.

| run | model | flags | `-ub` | calls (`DUMP_CALLS`) | graphs | what it adds |
|---|---|---|---|---|---|---|
| mtp3 | n17 | f32, FA off | 64 | `240, 1 x 8, d3, 4, r2, d3, 4, r1, d3, 4, 1, 1, d3, 4, r3, 1 x 6`, then a 3-step draft (`DUMP_MTP_DRAFT=3`) | target 24 (21 calls, the first in 4 ubatches; plus 3 rollback records), MTP 87 (63 process calls = 21 x 3 heads in 72 graphs; 15 draft steps = 5 drafts x 3) | the three heads in process and draft; verify batches of 4 (positions 248, 250, 253, 259) with 2, 1, 0 and 3 rejected tokens; both SWA caches (256 cells) wrap; the call 253..256 straddles the MTP ring's end (finding 2) |
| fa/mtp3 | n17 | f16, FA on | 64 | as mtp3 | as mtp3 | the same at the R1 default precision (the MTP context gets the same `-ctk/-ctv/-fa`) |
| fa/pf64 | n17 | f16, FA on | 512 | one call of 64 | 1 | the tiled CPU flash attention (64 query rows) |
| fa/ub64 | n17 | f16, FA on | 64 | one call of 300 | 5 (64 x 4 + 44) | tiled and one_chunk in one call; the SWA ring wraps; the global layers read 512 cells in the last ubatch |
| fa/decode | n17 | f16, FA on | 64 | `240, 1 x 60` | 64 | one_chunk per token; from position 256 the global layers read 512 cells with one query row: the split-KV path |
| fa/rollback | n17 | f16, FA on | 64 | N16's `240, 1 x 16, 4, r2, 1 x 6, 12, r10, 1 x 34` | 62 | rollbacks inside and across the window at f16 / FA |
| fa/hd192 | n17hd | f16, FA on | 512 | `64, 1 x 16` | 17 | K 192 / V 128: the AVX2 SIMD loops of the f16 dot, scale and mad (DK, DV >= 32) |

## 3. Method

As N16 (and N10 for the precision): every graph is compared row by row with (A) the full-sequence oracle (for the MTP
heads: `mtp_forward_chunk(final tokens, offset=k)` fed by the full target oracle's h, process rows whose position is
kept; at f16 / FA each position in the CPU mode of the ubatch that computed its kept row) and (B) the incremental
oracle over the cell model of the caches (`forward_chunk`, `mtp_process`, `mtp_draft`, `cache_truncate` for every
rollback and seq_rm record of the manifests; draft steps fed llama.cpp's draft tokens, which are asserted equal to the
oracle's own argmax, and the oracle's own h rows). MTP checks: the graph of head k names layer 6 + k and no other;
token rows exact; the h rows fed (`h_in.bin`) against (B) and (A) and, **bit for bit on llama.cpp's own numbers**, the
pairing of section 1 (process: every head's rows = the target's `h_nextn.bin` rows shifted by one, the first the pending
row, after a rejection the row of the last kept position; draft step k: `[pending, h_nextn.bin of steps 0 .. k-1 at
their output row]`, tokens `[id_last, argmax of llama.cpp's logits of steps 0 .. k-1]`, outputs only the last row);
every graph_mtp component; f32: `kq-<il>` on all n_kv cells (stale and empty ones included) and `kq_soft_max-<il>` 0 on
masked cells; f16 / FA: the store and every read cell through the cache views, and the flash-attention op on
llama.cpp's own inputs. f16 / FA trunk checks (tests/test_mimo2_f16fa_parity.py docstring): the store (exact), every cell
the attention read against the oracle's stored rows, the flash-attention op with the layer's sinks on llama.cpp's own
q / K / V views over the cells (B)'s cell model makes visible (one_chunk within one binary16 spacing per element and at
most 0.1 % past half a spacing, tiled 1e-5), and the same op with the window n_swa - 1, n_swa + 1 or without sinks must
break that bound on some graph; then every component against (A) and (B); MoE expert sets and greedy tokens equal; the
precision-aware oracle closer to kqv_out than the plain f32 oracle by N10's factor 2 (1.5 for the 192 / 128 heads,
module docstring: measured 1.79).

Tolerances: f32 runs 1e-4 normwise (N16); f16 / FA: 1e-4 for what layer 0 computes before its attention, TOL16 =
16 U16 = 7.8e-3 from there on (N10's tolerance, same justification), exact for gathers, the store and the pairing.

## 4. Findings (CONFIRMED by the literal outputs below unless labelled)

1. **llama.cpp equals the cell-accurate oracle everywhere, at both pins: 70/70 through the CI checker at ec7630a and
   at 79e2e74.** Worst f32 MTP error 1.32e-06 (`mtp_Qcur` of head 2); the f16 / FA runs within 2.10e-03 (max |got - want| / max |want|,
   kqv_out of layer 5) against TOL16 = 7.8e-03, the plain f32 oracle up to 9.67e-03 on the same tensor (it would not
   pass TOL16). The two pins write bit-identical dumps for every run, graph and tensor they both name (0 of 40 152 files
   differ; the only difference is the rename of the last `l_out` to `h_nextn` at 79e2e74, as N16 finding 6).
2. **Finding: after a process call whose ubatch wraps the MTP context's SWA ring, head 0 reads K/V of other positions
   (llama.cpp behaviour, both pins; not an oracle mismatch: the cell oracle (B) matches it, kqv_out within 7.2e-07).**
   The cells are shared by the three head layers, each head writes only its own layer, and process() re-places the
   same batch once per head after `seq_rm(first position)` (section 1). seq_rm moves the cache head back to the first
   freed cell (KV:410-419 @ec7630a, 428-437 @79e2e74). In mtp3 the verify batch at positions 253..256 is placed by
   head 0 in cells [253, 254, 255, 0] (it wraps the 256-cell ring), so the freed cell 0 is the lowest and heads 1 and 2
   place the batch in cells [0, 1, 2, 3]; the cell positions follow the last head. Head 0's layer 6 then holds, under
   cells 0..3 (positions 253..256), the K/V of positions 256, 1, 2, 3. Every later head-0 row whose window reaches them
   (positions 257..263 here, process and draft) attends to those: kqv_out differs from (A) by up to 2.05 (relative),
   head 0's logits by up to 0.40 (7.2, foreign probe); heads 1 and 2 and all other rows match (A). The same at f16 / FA
   (2.05, 0.40). Consequence for MiMo-V2.6-Flash (PROVISIONAL, derived from the same placement rule, not run): the MTP
   context's SWA cache has pad(min(n_ctx, 128 + n_ubatch), 256) = 768 cells with n_ubatch 512 (KVI:73, N16), so any
   process call whose ubatch crosses the end of that ring (a prompt ubatch across a multiple of 768 cells, or a verify
   batch of up to 4 tokens there) leaves head 0 (the first draft token) attending to wrong K/V for the next ~128
   positions (the window); the size of the effect on acceptance with real weights is UNKNOWN. An engine of ours that
   chains heads should place a call once for all heads (or give each head its own cells). Whether upstream changed this
   after 79e2e74: UNKNOWN. The tests exclude exactly the rows the oracle flags (`attn["foreign"]`) from the (A)
   comparison and assert the deviation (test_head0_reads_foreign_kv); the property test reproduces it on a 16-cell ring.
3. **Which h each head takes** is the one of section 1, checked bit for bit on 54 144 elements of llama.cpp's own
   `h_in.bin` / `h_nextn.bin` / `logits.bin` per run; every draft token of llama.cpp equals the oracle's own argmax (5
   drafts x 3 steps per run).
4. **Rollback in the MTP context (optional item 3): done.** `r<n>` cuts both contexts as llama-server does and moves the
   pending row (rejections of 2, 1, 3 positions); the oracle replays it (mutant T1: keeping the call's last row as the
   pending row is caught). With three heads the server's seq_rm after a draft or a rejection is followed by process's
   own seq_rm from the same position, so it changes nothing there (PROVISIONAL reading of section 1; the one-head
   path, where it matters, has no drafts or rejections in these runs: NOT RUN).
5. **CPU flash attention with sinks, emulated (ref/fattn.py) and reproduced on llama.cpp's own inputs.** Where the sink
   enters: one_chunk after the cell loop, on the f32 copy of the f16 accumulator (s > M: the accumulator is scaled by
   expf(M - s) in f32 and S = S·ms + 1; else S += expf(s - M); O:9040-9056), only for the chunk starting at cell 0
   (O:9041); tiled per valid row after all tiles (O:9332-9349); the f32 softmax (FA off) as max over row and sink,
   sum += expf(sink - max) (O:5896-5912). The split-KV decode path (one row, n_kv >= 512; O:9490-9520) runs one_chunk
   per chunk; with -t 1 one chunk [0, n_kv), and the merge multiplies by expf(0) and expf(-inf) (O:9420-9435), so it
   is one_chunk (fa/decode reaches it on the global layers from position 256; the op check passes there). On llama.cpp's
   own q / K / V the emulation has **no element more than half a binary16 spacing off** (0 of 41 472 in fa/decode, 46 080
   fa/rollback, 16 896 fa/ub64, 49 152 fa/hd192, 17 280 fa/mtp3), tiled within 5.5e-07 (relative). A window of n_swa - 1 or
   n_swa + 1, or dropped sinks, breaks the bound on every graph x SWA layer of every run (256 of 256 in fa/decode).
   **The SWA mask with flash attention is the f32 path's mask**: `set_input_kq_mask` fills the f16 mask (flash
   attention) with the same template as the f32 one (KV:1566-1716, 1789-1793 @ec7630a; KV:1569-1719, 1792-1796 @79e2e74,
   identical code), values 0 and -inf (exact in f16), and the op check with (B)'s cell-accurate visibility (which N16
   checked exactly through kq_soft_max) reproduces every element.
6. **Two emulation details were missing, found here (root causes of the first mismatches, both fixed in the oracle):**
   (a) the kq scale is llama.cpp's f32 `1.0f/sqrtf(24)` = 0.20412414, one f32 step below float32(1/sqrt(24))
   (M:179 @ec7630a, 171 @79e2e74); (b) one_chunk calls `expf` on the f32 difference `Mold - M` / `s - M` (O:8997, 9003),
   which N10's `_online_f16` took in float64. With the N10 emulation up to 26 elements per run were past half a
   spacing (max 17 spacings); with (a) alone the runs with 24-wide heads keep up to 4 (max 3.1 spacings), with (a)
   and (b) 0 (7.2, fa-op probe; fa/hd192 also needs finding 7). (b) is a change to N10's oracle, flagged: the N10 step still passes 170/170 at both pins, and N10's one stray element per qwen4exp run (1/8192
   past half a spacing, `docs/evidence/2026-10-10-quant-fa-parity.md:434`) is now 0/8192, glm5-next's 0.078 of the
   one-spacing bound (`:482`) now 0.000 (7.6). What is left is at the f32 level (93.1 % to 93.8 % of the one_chunk
   elements bit-equal per run): with an fma in the running sum S 98.8 % to 99.7 % are (PROVISIONAL reading: such an fma
   is the likely cause of most of the rest; ref/fattn.py keeps two roundings, which changes no f16 rounding).
7. **K 24 / V 16 and K 192 / V 128.** DK != DV runs through every path (store, op, model). For heads >= 32 wide the CPU
   takes the AVX2 loops: the dot sums in 4 x 8 f32 lanes (vec.cpp:345-377, simd-mappings.h:595-620), which the emulation
   now models (`_dot_f16`); without it fa/hd192 has 29 elements past half a spacing (max 52 spacings), with it 0 (7.2,
   fa-op probe). The mad and scale loops round as before (fma, then f16). The aware oracle is 1.79x closer than the plain f32
   oracle there (3.5x to 5.2x for the tiny heads).
8. **f16 / FA whole model**: every component within TOL16 at both pins, MoE expert sets and greedy tokens equal, cache
   store exact (120 000 elements in fa/decode, 256 000 in fa/hd192), 89 % to 99 % of the cached elements exactly the
   oracle's (>= 50 % required). The R1 default precision does not change any discrete choice in these runs.
9. **The dump tool keeps N16**: N16's mtp run with the tool built from `0b0eace` and with the N17 tool, 11 534 files, 0
   differ, manifests equal apart from the N17 fields; N16's step 92/92 and N10's 170/170 at both pins on `d5b3537`; the other steps whose drivers use the dump tool
   (G1, G1 MiMo, N2, N6, N9, N10 quantized) at both pins on `9a95535` (7.8).
10. **Generator**: default, `--seed 3` and N16's MTP model byte-identical before and after, `--mtp-layers 1` = `--mtp`,
    with the gguf-py of both pins.
11. **Mutation check: 19 of 19 caught at each pin, by the same tests at both (7.7).** The mutants (one edit each, on
    copies of `ref/` and `tests/`): the head offset off by one in process (H1) and in draft (H2); the wrong h source
    (P1: in process head k > 0 takes head k-1's output; P2: every draft row the pending row; P3: draft rows not chained
    one back); no seq_rm before each head (Q1) or each draft step (Q2); the per-layer K/V position record dropped (K1);
    the test's rejection replay without `accept()`'s pending row (T1); the sink ignored in one_chunk (F1) or applied
    before the cell loop (F2); f16 rounding skipped for the one_chunk accumulator (R1), the cache store (R2), Q (R3);
    `expf` of the float64 difference (E1); the float64 kq scale (E2); the AVX2 dot lanes not modelled (E3: caught only
    by fa/hd192 and the transcription test, as heads under 32 wide do not take that path); K/V head-size confusion
    (V1: the scale from the V head size; V2: V rows sized by the K head, which fails setUpClass on the shape). **CI**: the
    step fails through the checker with no environment, an empty dump directory, without the MTP context or without
    fa/hd192, at both pins (7.2, "bites").

## 5. Not covered (NOT RUN unless labelled)

* real MiMo-V2.6-Flash weights or GGUF (header, tensor types); the size of finding 2 on acceptance (UNKNOWN).
* the llama-server binary itself: its call sequence is replayed by the dump tool from the source (section 1), the
  server was not run; probabilistic drafting (79e2e74 `result_q`), `p_min` > 0, several sequences.
* one MTP head with drafts and rejections (N16's model): NOT RUN; the one-head draft is checked by N16 (end-of-plan
  draft) and by the property test `test_one_head_draft_is_n16`.
* `-t` > 1 (several FA chunks, the split-KV merge across threads), GPU flash attention, quantized KV caches, the other
  precision configurations for mimo2 (`-ctk f32 -fa on`, `-ctk f16 -fa off`): NOT RUN.
* the target (trunk) tensors of mtp3 and fa/mtp3 component by component: not compared here (the trunk and plans of
  N16 and of fa/rollback cover them); the target's `h_nextn` rows enter as the h fed and are compared there.
* the ik_llama.cpp step (`n12-ik-glm5next`): NOT RUN; its dump tool is `tools/dump/ik_dump_tensors.cpp`, which N17 does
  not change.
* MXFP4 (or any quantized) MTP block; `nextn.embed_tokens` / `shared_head_*` tensors (fallbacks checked, as N16).
* GitHub Actions: NOT RUN (the branch is not pushed in this task).

## 6. Where llama.cpp and an oracle disagree, and why

| where | against | size | root cause | status |
|---|---|---|---|---|
| head 0 (layer 6) of the MTP context, the 11 rows at positions 257..263 after the process call 253..256 (process and draft graphs), mtp3 and fa/mtp3, both pins | (A) the full-sequence oracle | kqv_out up to 2.053 (f16 / FA 2.052), logits up to 0.404 | llama.cpp: the per-head `seq_rm` + re-placement of one batch on cells the three head layers share; head 0's batch wrapped the ring, heads 1 and 2 did not, so head 0's layer keeps K/V of other positions under the cells (finding 2) | llama.cpp behaviour, reproduced by the cell oracle (B) within 7.2e-07; the rows are excluded from the (A) assertion by the oracle's own flag and asserted to differ |
| the CPU one_chunk flash attention, N10's emulation on mimo2's rows | llama.cpp's own q / K / V | up to 26 elements per run past half a binary16 spacing, max 17 spacings | oracle: float64 kq scale instead of llama.cpp's f32 `1.0f/sqrtf(24)`, and `expf` of a float64 difference instead of the f32 one (finding 6) | fixed in `ref/mimo2.py` (`kq_scale`) and `ref/fattn.py` (`kq_scale_f32`, `_online_f16`); 0 elements now |
| the same op with K 192 / V 128 heads | llama.cpp's own q / K / V | 29 elements past half a spacing, max 52 spacings | oracle: the AVX2 f32 lanes of `ggml_vec_dot_f16` for heads >= 32 wide were not modelled (finding 7) | fixed (`_dot_f16`); 0 elements now |
| one_chunk at the f32 level | llama.cpp's own q / K / V | 6.2 % to 6.9 % of the elements per run not bit-equal, all within half a spacing | PROVISIONAL: an fma in the running sum S (with it 98.8 % to 99.7 % bit-equal) | not modelled; changes no f16 rounding |

No other disagreement: every other compared tensor of every run is within its tolerance against both (A) and (B), at
both pins, and the two pins wrote the same bits.

## 7. Literal outputs

`<S>` replaces the scratch directory in every output; nothing else is edited. The runs of 7.2 to 7.7 are on worktree
HEAD `d5b3537` (first line of 7.2). The commits after it change files under `docs/` and one docstring of `ref/fattn.py`
(7.10); on that commit, `9a95535`, the N17 step was run again at both pins with the same output apart from timings and
the work directory, the other CI steps whose drivers use the dump tool ran at both pins, and the unit discovery,
checker and hook tests ran (7.8).

### 7.1 Environment, pins, builds

```
$ nproc; python3 --version; python3 -c 'import numpy; print("numpy", numpy.__version__)'; gcc --version | head -1; cmake --version | head -1
4
Python 3.13.16
numpy 2.5.3
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
cmake version 3.28.3
$ git -C <S>/llama-ec7630a rev-parse HEAD; git -C <S>/llama.cpp rev-parse HEAD
ec7630a640789c393694fb194f1bbbf0369fc62d
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
$ cat <S>/n17/build.sh   # run once, both builds from empty directories
#!/usr/bin/env bash
# N17 fresh builds of dump-tensors (and llama-quantize) from /home/user/wt-n17/tools/g0 at both llama.cpp pins
set -u
S=<S>
W=/home/user/wt-n17
for p in ec r1; do
  if [ $p = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  t0=$(date +%s)
  cmake -S $W/tools/g0 -B $S/n17/build-$p -DLLAMA_CPP_DIR=$L -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON > $S/n17/cmake-$p.log 2>&1 || { echo "$p cmake FAILED"; continue; }
  t1=$(date +%s)
  cmake --build $S/n17/build-$p -j4 --target dump-tensors > $S/n17/build-$p.log 2>&1 || { echo "$p build FAILED"; continue; }
  t2=$(date +%s)
  echo "$p cmake $((t1-t0)) s, build $((t2-t1)) s"
done
echo BUILDS-DONE
$ cat <S>/n17/builds.log
ec cmake 2 s, build 241 s
r1 cmake 2 s, build 211 s
BUILDS-DONE
$ for p in ec r1; do echo "== build-$p"; grep -E "^(GGML_NATIVE|GGML_AVX2|GGML_AVX512|GGML_FMA|GGML_F16C|CMAKE_BUILD_TYPE|LLAMA_CPP_DIR)[:=]" <S>/n17/build-$p/CMakeCache.txt; done
== build-ec
CMAKE_BUILD_TYPE:STRING=Release
GGML_AVX2:BOOL=ON
GGML_AVX512:BOOL=OFF
GGML_F16C:BOOL=ON
GGML_FMA:BOOL=ON
GGML_NATIVE:BOOL=OFF
LLAMA_CPP_DIR:PATH=<S>/llama-ec7630a
== build-r1
CMAKE_BUILD_TYPE:STRING=Release
GGML_AVX2:BOOL=ON
GGML_AVX512:BOOL=OFF
GGML_F16C:BOOL=ON
GGML_FMA:BOOL=ON
GGML_NATIVE:BOOL=OFF
LLAMA_CPP_DIR:PATH=<S>/llama.cpp
```

### 7.2 Final runs (N17 step at both pins, N16 and N10 steps, pin comparison, N16 compatibility, bites, probes)

The "pin dump comparison" compares every file both pins wrote under each run (`2026-10-11-n17-pindiff.py`); "only at
one pin" are `h_nextn` (79e2e74) and `l_out-5` (ec7630a), the rename of N16 finding 6. The "bites" run the checker
on a complete run, without the environment, on an empty dump directory, without the MTP context's dumps and without
fa/hd192 (`2026-10-11-n17-bites.sh`). The foreign probe (`2026-10-11-n17-foreign-probe.py`) lists, per head-0 row
that reads foreign K/V, which cells hold which position and for which position layer 6's K/V was computed. The
flash-attention op probe (`2026-10-11-n17-fa-op-probe.py`) runs the one_chunk emulation on llama.cpp's own inputs in
five variants: N10's (float64 scale and exp argument, double dot), with the f32 scale, final (both fixes and the AVX2
dot), final with a double dot, final with an fma in S. The mutation rounds' outputs are in 7.7.

```
$ cd <S>/n17/cwd && bash /home/user/wt-n17/docs/evidence/mutation-scripts/2026-10-11-n17-final-runs.sh > <S>/n17/final.log 2>&1
$ cat <S>/n17/final.log
worktree HEAD d5b3537dc2a4a7c57712376ddb9fc544aee9e2dd; status: 0 changed or untracked files
n17 ec wrapper_rc=0 wall_s=52
n17 r1 wrapper_rc=0 wall_s=54
n16 ec wrapper_rc=0
n10 ec wrapper_rc=0
n16 r1 wrapper_rc=0
n10 r1 wrapper_rc=0
== pin dump comparison
mtp3         files in both   4010, differing    0; only at one pin    0 []; h_nextn.bin, logits.bin differing: none
mtp3/mtp     files in both   2961, differing    0; only at one pin    0 []; h_in.bin, h_nextn.bin, logits.bin differing: none
fa/mtp3      files in both   4202, differing    0; only at one pin    0 []; h_nextn.bin, logits.bin differing: none
fa/mtp3/mtp  files in both   3048, differing    0; only at one pin    0 []; h_in.bin, h_nextn.bin, logits.bin differing: none
fa/pf64      files in both    175, differing    0; only at one pin    2 ['h_nextn', 'l_out-5']; logits.bin differing: none
fa/ub64      files in both    871, differing    0; only at one pin   10 ['h_nextn', 'l_out-5']; logits.bin differing: none
fa/decode    files in both  11137, differing    0; only at one pin  128 ['h_nextn', 'l_out-5']; logits.bin differing: none
fa/rollback  files in both  10789, differing    0; only at one pin  124 ['h_nextn', 'l_out-5']; logits.bin differing: none
fa/hd192     files in both   2959, differing    0; only at one pin   34 ['h_nextn', 'l_out-5']; logits.bin differing: none
== one-head compatibility of the dump tool (ec7630a)
files: old 11534, new 11534, same names True, differing 0 []
manifest.json: equal without the N17 fields []: True
mtp/manifest.json: equal without the N17 fields ['head', 'out0', 'outputs']: True
== bites ec (ec7630a)
complete: checker rc=0 | Ran 70 tests in 45.955s OK | 
no-env: checker rc=1 | Ran 70 tests in 0.580s OK (skipped=53) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: skipped=53, expected skipped=0 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: skip not allowed here: tests.test_mimo2_mtp_heads_parity.MtpHeads.test_draft_tokens: 'set MIMO2_N17_DUMP, L 
empty-dir: checker rc=1 | Ran 17 tests in 0.563s OK (skipped=7) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 17 tests, expected 70 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: skipped=7, expected skipped=0 
no-mtp-context: checker rc=1 | Ran 64 tests in 41.618s OK (skipped=1) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 64 tests, expected 70 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: skipped=1, expected skipped=0 
no-hd192-run: checker rc=1 | Ran 62 tests in 38.604s OK (skipped=1) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 62 tests, expected 70 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: skipped=1, expected skipped=0 
== foreign probe ec
mtp3: process calls whose heads wrote different SWA cells
  positions 253..256: head 0 cells [253, 254, 255, 0]; head 1 cells [0, 1, 2, 3]; head 2 cells [0, 1, 2, 3]
graph kind    head pos  foreign cells (cell: holds pos / layer-6 K,V of pos)       kqv_out vs A    vs B   logits vs A    vs B
   54 process    0  257  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       4.698e-01 2.6e-07   8.694e-02 1.9e-07
   57 process    0  258  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       2.053e+00 2.6e-07   1.649e-01 2.9e-07
   60 draft      0  259  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3             n/a 3.0e-07         n/a 2.7e-07
   63 process    0  259  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       7.858e-01 3.6e-07   1.684e-01 2.7e-07
   63 process    0  260  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3             n/a 5.7e-07         n/a 3.9e-07
   63 process    0  261  1: 254 / 1, 2: 255 / 2, 3: 256 / 3                           n/a 2.8e-07         n/a 2.2e-07
   63 process    0  262  2: 255 / 2, 3: 256 / 3                                       n/a 3.2e-07         n/a 3.0e-07
   66 process    0  260  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       1.955e+00 4.7e-07   2.757e-01 3.2e-07
   69 process    0  261  1: 254 / 1, 2: 255 / 2, 3: 256 / 3                     1.154e+00 7.2e-07   2.467e-01 2.9e-07
   72 process    0  262  2: 255 / 2, 3: 256 / 3                                 1.094e+00 3.2e-07   4.039e-01 3.1e-07
   75 process    0  263  3: 256 / 3                                             8.174e-01 3.5e-07   2.267e-01 2.1e-07
== flash-attention op probe ec
run          variant             > 0.5 sp  > 1 sp   max sp          bit-equal
fa/ub64      N10                        7       5   16.991     7091/16896    
fa/ub64      f32 scale                  0       0    0.000    13940/16896    
fa/ub64      final                      0       0    0.000    15804/16896    
fa/ub64      final, double dot          0       0    0.000    15804/16896    
fa/ub64      final + fma S              0       0    0.000    16851/16896    
fa/decode    N10                       25      14   10.837    17836/41472    
fa/decode    f32 scale                  2       1    3.135    34557/41472    
fa/decode    final                      0       0    0.000    38605/41472    
fa/decode    final, double dot          0       0    0.000    38605/41472    
fa/decode    final + fma S              0       0    0.000    41282/41472    
fa/rollback  N10                       26      14    6.872    19910/46080    
fa/rollback  f32 scale                  4       1    3.135    38215/46080    
fa/rollback  final                      0       0    0.000    42929/46080    
fa/rollback  final, double dot          0       0    0.000    42929/46080    
fa/rollback  final + fma S              0       0    0.000    45890/46080    
fa/hd192     N10                        4       2    5.113    39927/49152    
fa/hd192     f32 scale                  4       2    5.113    39927/49152    
fa/hd192     final                      0       0    0.000    46125/49152    
fa/hd192     final, double dot         29      19   51.847    26067/49152    
fa/hd192     final + fma S              0       0    0.000    48575/49152    
== bites r1 (79e2e74)
complete: checker rc=0 | Ran 70 tests in 45.406s OK | 
no-env: checker rc=1 | Ran 70 tests in 0.475s OK (skipped=53) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: skipped=53, expected skipped=0 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_mimo2_mtp_heads_parity.MtpHeads.test_draft_tokens: 'set MIMO2_N17_DUMP, L 
empty-dir: checker rc=1 | Ran 17 tests in 0.479s OK (skipped=7) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 17 tests, expected 70 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: skipped=7, expected skipped=0 
no-mtp-context: checker rc=1 | Ran 64 tests in 37.797s OK (skipped=1) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 64 tests, expected 70 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: skipped=1, expected skipped=0 
no-hd192-run: checker rc=1 | Ran 62 tests in 35.402s OK (skipped=1) | ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 62 tests, expected 70 ::error::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: skipped=1, expected skipped=0 
== foreign probe r1
mtp3: process calls whose heads wrote different SWA cells
  positions 253..256: head 0 cells [253, 254, 255, 0]; head 1 cells [0, 1, 2, 3]; head 2 cells [0, 1, 2, 3]
graph kind    head pos  foreign cells (cell: holds pos / layer-6 K,V of pos)       kqv_out vs A    vs B   logits vs A    vs B
   54 process    0  257  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       4.698e-01 2.6e-07   8.694e-02 1.9e-07
   57 process    0  258  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       2.053e+00 2.6e-07   1.649e-01 2.9e-07
   60 draft      0  259  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3             n/a 3.0e-07         n/a 2.7e-07
   63 process    0  259  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       7.858e-01 3.6e-07   1.684e-01 2.7e-07
   63 process    0  260  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3             n/a 5.7e-07         n/a 3.9e-07
   63 process    0  261  1: 254 / 1, 2: 255 / 2, 3: 256 / 3                           n/a 2.8e-07         n/a 2.2e-07
   63 process    0  262  2: 255 / 2, 3: 256 / 3                                       n/a 3.2e-07         n/a 3.0e-07
   66 process    0  260  0: 253 / 256, 1: 254 / 1, 2: 255 / 2, 3: 256 / 3       1.955e+00 4.7e-07   2.757e-01 3.2e-07
   69 process    0  261  1: 254 / 1, 2: 255 / 2, 3: 256 / 3                     1.154e+00 7.2e-07   2.467e-01 2.9e-07
   72 process    0  262  2: 255 / 2, 3: 256 / 3                                 1.094e+00 3.2e-07   4.039e-01 3.1e-07
   75 process    0  263  3: 256 / 3                                             8.174e-01 3.5e-07   2.267e-01 2.1e-07
== flash-attention op probe r1
run          variant             > 0.5 sp  > 1 sp   max sp          bit-equal
fa/ub64      N10                        7       5   16.991     7091/16896    
fa/ub64      f32 scale                  0       0    0.000    13940/16896    
fa/ub64      final                      0       0    0.000    15804/16896    
fa/ub64      final, double dot          0       0    0.000    15804/16896    
fa/ub64      final + fma S              0       0    0.000    16851/16896    
fa/decode    N10                       25      14   10.837    17836/41472    
fa/decode    f32 scale                  2       1    3.135    34557/41472    
fa/decode    final                      0       0    0.000    38605/41472    
fa/decode    final, double dot          0       0    0.000    38605/41472    
fa/decode    final + fma S              0       0    0.000    41282/41472    
fa/rollback  N10                       26      14    6.872    19910/46080    
fa/rollback  f32 scale                  4       1    3.135    38215/46080    
fa/rollback  final                      0       0    0.000    42929/46080    
fa/rollback  final, double dot          0       0    0.000    42929/46080    
fa/rollback  final + fma S              0       0    0.000    45890/46080    
fa/hd192     N10                        4       2    5.113    39927/49152    
fa/hd192     f32 scale                  4       2    5.113    39927/49152    
fa/hd192     final                      0       0    0.000    46125/49152    
fa/hd192     final, double dot         29      19   51.847    26067/49152    
fa/hd192     final + fma S              0       0    0.000    48575/49152    
mutation ec rc=0
mutation r1 rc=0
FINAL-DONE
```

### 7.3 The N17 step through the CI checker at both pins

The end of each wrapped driver log, the whole-log comparison of the two pins (equal apart from the pin name, the
timings, the checkout path and one docstring that names 79e2e74), and the finding's own test output.

```
$ tail -n 6 <S>/n17/final-wrapped-ec.log
----------------------------------------------------------------------
Ran 70 tests in 42.983s

OK
artifacts: <S>/n17/work-ec
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
$ tail -n 6 <S>/n17/final-wrapped-r1.log
----------------------------------------------------------------------
Ran 70 tests in 40.574s

OK
artifacts: <S>/n17/work-r1
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
$ diff <(sed -e 's/ec7630a/PIN/g; s/in [0-9.]*s/in Xs/; s/llama-PIN/L/' <S>/n17/final-wrapped-ec.log) <(sed -e 's/79e2e74/PIN/g; s/in [0-9.]*s/in Xs/; s/llama-r1-PIN/L/;s/work-r1/work-ec/' <S>/n17/final-wrapped-r1.log)
739c739
< graph_mtp asserts 0 <= nextn_layer_offset < n_layer_nextn (src/models/mimo2.cpp:267-270 @79e2e74). ... ok
---
> graph_mtp asserts 0 <= nextn_layer_offset < n_layer_nextn (src/models/mimo2.cpp:267-270 @PIN). ... ok
767c767
< ::notice::n17-mimo2-mtp-heads-fa @ llama.cpp PIN: Ran 70, skipped 0 as expected; the driver's checkout <S>/L is at PIN
---
> ::notice::n17-mimo2-mtp-heads-fa @ llama.cpp PIN: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at PIN
$ grep -nE '^(Finding|ok$)' <S>/n17/final-wrapped-ec.log | head -4; sed -n 15,19p <S>/n17/final-wrapped-ec.log
17:Finding (docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md): process() re-places every call once per head ... 
19:ok
33:Finding (docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md): process() re-places every call once per head ... 
35:ok
One graph per ubatch of every MTP call; the graph of head k names layer n_layer + k (the offset reached the ... ok
test_head0_reads_foreign_kv (tests.test_mimo2_mtp_heads_parity.MtpHeads.test_head0_reads_foreign_kv)
Finding (docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md): process() re-places every call once per head ... 
mtp3: head-0 rows reading foreign K/V: 8 graphs, positions [257, 258, 259, 260, 261, 262, 263]; worst kqv_out vs (A) 2.053e+00, logits 4.039e-01
ok
```

### 7.4 MTP heads: worst error per layer and per component (ec7630a; 79e2e74 identical, 7.3)

Per layer (head k = layer 6 + k; f32 rows against 1e-4, f16 / FA against TOL16 = 7.8e-03):

```
$ sed -n 376,383p <S>/n17/final-wrapped-ec.log
MTP heads: worst error per run and layer (head k = layer n_layer + k; the head-0 rows that read foreign K/V against (B) only), llama.cpp ec7630a:
run      layer component                                    graph    max abs        rel
fa/mtp3      6 kqv_out [chunk]                                 36  1.406e-03  1.217e-03
fa/mtp3      7 kqv_out [full]                                  82  2.186e-03  1.597e-03
fa/mtp3      8 kqv_out [full]                                  26  1.330e-03  1.289e-03
mtp3         6 mtp_Kcur (RoPE) [full]                          78  3.284e-06  1.104e-06
mtp3         7 mtp_Qcur (RoPE) [full]                          25  2.489e-06  1.090e-06
mtp3         8 mtp_Qcur (RoPE) [full]                          71  3.749e-06  1.321e-06
```

Per run, component and head (`[full]` = (A), `[chunk]` = (B)):

```
$ sed -n 43,374p <S>/n17/final-wrapped-ec.log
MTP heads: worst error per run, component and head over the MTP graphs, llama.cpp ec7630a:
run      component                                    head graph    max abs        rel
mtp3     mtp_tok_embd (exact)                            0     0  0.000e+00  0.000e+00
mtp3     h_in (row fed) [full]                           0    66  1.820e-06  5.300e-07
mtp3     h_in (row fed) [chunk]                          0    66  1.820e-06  5.300e-07
mtp3     mtp_tok_embd (exact)                            1     4  0.000e+00  0.000e+00
mtp3     h_in (row fed) [full]                           1    67  1.820e-06  5.300e-07
mtp3     h_in (row fed) [chunk]                          1    67  1.820e-06  5.300e-07
mtp3     mtp_tok_embd (exact)                            2     8  0.000e+00  0.000e+00
mtp3     h_in (row fed) [full]                           2    68  1.820e-06  5.300e-07
mtp3     h_in (row fed) [chunk]                          2    68  1.820e-06  5.300e-07
mtp3     h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
mtp3     head 0 kqv_out vs (A), 11 rows reading foreign K/V (expected to differ)    0    -1  0.000e+00  2.053e+00
mtp3     head 0 logits vs (A), the same rows (expected to differ)    0    -1  0.000e+00  4.039e-01
mtp3     kq (all n_kv cells) [chunk]                     0    15  1.067e-05  7.095e-07
mtp3     kq_soft_max (cells) [chunk]                     0    69  1.968e-07  7.990e-07
mtp3     kq (all n_kv cells) [chunk]                     1    25  1.113e-05  7.732e-07
mtp3     kq_soft_max (cells) [chunk]                     1    55  3.080e-07  6.502e-07
mtp3     kq (all n_kv cells) [chunk]                     2    71  1.430e-05  8.491e-07
mtp3     kq_soft_max (cells) [chunk]                     2    71  3.802e-07  9.173e-07
mtp3     mtp_enorm [full]                                0    57  3.020e-07  1.230e-07
mtp3     mtp_enorm [chunk]                               0    57  3.020e-07  1.230e-07
mtp3     mtp_hnorm [full]                                0    66  1.128e-06  4.858e-07
mtp3     mtp_hnorm [chunk]                               0    66  1.128e-06  4.858e-07
mtp3     mtp_concat [e ; h] [full]                       0    66  1.128e-06  4.858e-07
mtp3     mtp_concat [e ; h] [chunk]                      0    66  1.128e-06  4.858e-07
mtp3     mtp_eh_proj [full]                              0    18  8.648e-07  4.081e-07
mtp3     mtp_eh_proj [chunk]                             0    18  8.648e-07  4.081e-07
mtp3     mtp_attn_norm [full]                            0    18  1.162e-06  4.423e-07
mtp3     mtp_attn_norm [chunk]                           0    18  1.162e-06  4.423e-07
mtp3     mtp_wqkv (fused projection) [full]              0    63  1.172e-06  4.557e-07
mtp3     mtp_wqkv (fused projection) [chunk]             0    60  1.172e-06  4.557e-07
mtp3     mtp_Qcur (RoPE) [full]                          0    33  2.523e-06  1.098e-06
mtp3     mtp_Qcur (RoPE) [chunk]                         0    33  2.523e-06  1.098e-06
mtp3     mtp_Kcur (RoPE) [full]                          0    78  3.284e-06  1.104e-06
mtp3     mtp_Kcur (RoPE) [chunk]                         0    78  3.284e-06  1.104e-06
mtp3     mtp_Vcur [full]                                 0    63  7.747e-07  4.265e-07
mtp3     mtp_Vcur [chunk]                                0    36  8.855e-07  6.677e-07
mtp3     kqv_out [full]                                  0    15  6.022e-07  5.258e-07
mtp3     kqv_out [chunk]                                 0    69  6.769e-07  7.158e-07
mtp3     mtp_attn_out [full]                             0     1  9.215e-07  5.206e-07
mtp3     mtp_attn_out [chunk]                            0    84  4.533e-07  6.634e-07
mtp3     mtp_attn_out_scaled [full]                      0     0  5.949e-07  5.281e-07
mtp3     mtp_attn_out_scaled [chunk]                     0    84  3.400e-07  6.634e-07
mtp3     mtp_ffn_inp [full]                              0    18  9.779e-07  3.998e-07
mtp3     mtp_ffn_inp [chunk]                             0    18  9.779e-07  3.998e-07
mtp3     mtp_ffn_norm [full]                             0    18  1.178e-06  4.078e-07
mtp3     mtp_ffn_norm [chunk]                            0    66  9.729e-07  4.182e-07
mtp3     ffn_up (dense) [full]                           0    18  1.658e-06  6.217e-07
mtp3     ffn_up (dense) [chunk]                          0    18  1.658e-06  6.217e-07
mtp3     ffn_gate (dense) [full]                         0    18  1.154e-06  4.939e-07
mtp3     ffn_gate (dense) [chunk]                        0    18  1.154e-06  4.939e-07
mtp3     ffn_swiglu (dense) [full]                       0    18  1.473e-06  6.225e-07
mtp3     ffn_swiglu (dense) [chunk]                      0    18  1.473e-06  6.225e-07
mtp3     mtp_ffn_out [full]                              0    12  6.577e-07  7.608e-07
mtp3     mtp_ffn_out [chunk]                             0    63  1.382e-06  7.856e-07
mtp3     mtp_post_ffn [full]                             0    18  1.623e-06  7.266e-07
mtp3     mtp_post_ffn [chunk]                            0    18  1.623e-06  7.266e-07
mtp3     h_nextn (graph output) [full]                   0    18  1.623e-06  7.266e-07
mtp3     h_nextn (graph output) [chunk]                  0    18  1.623e-06  7.266e-07
mtp3     mtp_shared_head_norm [full]                     0    18  1.680e-06  7.651e-07
mtp3     mtp_shared_head_norm [chunk]                    0    18  1.680e-06  7.651e-07
mtp3     result_output [full]                            0    18  2.279e-06  8.085e-07
mtp3     result_output [chunk]                           0    18  2.279e-06  8.085e-07
mtp3     logits.bin [full]                               0    18  2.279e-06  8.085e-07
mtp3     logits.bin [chunk]                              0    18  2.279e-06  8.085e-07
mtp3     h_nextn.bin [full]                              0    18  1.623e-06  7.266e-07
mtp3     h_nextn.bin [chunk]                             0    18  1.623e-06  7.266e-07
mtp3     mtp_enorm [full]                                1    25  2.751e-07  1.160e-07
mtp3     mtp_enorm [chunk]                               1    25  2.751e-07  1.160e-07
mtp3     mtp_hnorm [full]                                1    67  1.128e-06  4.957e-07
mtp3     mtp_hnorm [chunk]                               1    67  1.128e-06  4.957e-07
mtp3     mtp_concat [e ; h] [full]                       1    67  1.128e-06  4.567e-07
mtp3     mtp_concat [e ; h] [chunk]                      1    67  1.128e-06  4.567e-07
mtp3     mtp_eh_proj [full]                              1    19  8.692e-07  3.975e-07
mtp3     mtp_eh_proj [chunk]                             1    19  8.692e-07  3.975e-07
mtp3     mtp_attn_norm [full]                            1    22  8.257e-07  3.894e-07
mtp3     mtp_attn_norm [chunk]                           1    22  8.257e-07  3.894e-07
mtp3     mtp_wqkv (fused projection) [full]              1    19  8.332e-07  3.368e-07
mtp3     mtp_wqkv (fused projection) [chunk]             1    19  8.332e-07  3.368e-07
mtp3     mtp_Qcur (RoPE) [full]                          1    25  2.489e-06  1.090e-06
mtp3     mtp_Qcur (RoPE) [chunk]                         1    25  2.489e-06  1.090e-06
mtp3     mtp_Kcur (RoPE) [full]                          1    25  2.043e-06  9.117e-07
mtp3     mtp_Kcur (RoPE) [chunk]                         1    25  2.043e-06  9.117e-07
mtp3     mtp_Vcur [full]                                 1    13  6.712e-07  4.342e-07
mtp3     mtp_Vcur [chunk]                                1    13  6.712e-07  4.342e-07
mtp3     kqv_out [full]                                  1    55  9.758e-07  7.028e-07
mtp3     kqv_out [chunk]                                 1    55  9.758e-07  7.028e-07
mtp3     mtp_attn_out [full]                             1     7  1.022e-06  5.911e-07
mtp3     mtp_attn_out [chunk]                            1     7  1.022e-06  5.911e-07
mtp3     mtp_attn_out_scaled [full]                      1     7  7.963e-07  6.141e-07
mtp3     mtp_attn_out_scaled [chunk]                     1     7  7.963e-07  6.141e-07
mtp3     mtp_ffn_inp [full]                              1    40  9.113e-07  3.721e-07
mtp3     mtp_ffn_inp [chunk]                             1    40  9.113e-07  3.686e-07
mtp3     mtp_ffn_norm [full]                             1     4  1.506e-06  4.315e-07
mtp3     mtp_ffn_norm [chunk]                            1     4  1.506e-06  4.315e-07
mtp3     ffn_up (dense) [full]                           1    67  1.098e-06  4.148e-07
mtp3     ffn_up (dense) [chunk]                          1    67  1.098e-06  4.148e-07
mtp3     ffn_gate (dense) [full]                         1     4  1.332e-06  3.574e-07
mtp3     ffn_gate (dense) [chunk]                        1     4  1.332e-06  3.574e-07
mtp3     ffn_swiglu (dense) [full]                       1    19  1.550e-06  7.247e-07
mtp3     ffn_swiglu (dense) [chunk]                      1    19  1.550e-06  7.247e-07
mtp3     mtp_ffn_out [full]                              1    67  7.166e-07  6.414e-07
mtp3     mtp_ffn_out [chunk]                             1    67  7.166e-07  6.414e-07
mtp3     mtp_post_ffn [full]                             1    19  1.100e-06  3.860e-07
mtp3     mtp_post_ffn [chunk]                            1    19  1.100e-06  3.860e-07
mtp3     h_nextn (graph output) [full]                   1    19  1.100e-06  3.860e-07
mtp3     h_nextn (graph output) [chunk]                  1    19  1.100e-06  3.860e-07
mtp3     mtp_shared_head_norm [full]                     1    82  9.226e-07  4.283e-07
mtp3     mtp_shared_head_norm [chunk]                    1    82  9.226e-07  4.283e-07
mtp3     result_output [full]                            1    19  9.973e-07  4.532e-07
mtp3     result_output [chunk]                           1    64  1.344e-06  5.310e-07
mtp3     logits.bin [full]                               1    19  9.973e-07  4.532e-07
mtp3     logits.bin [chunk]                              1    64  1.344e-06  5.310e-07
mtp3     h_nextn.bin [full]                              1    19  1.100e-06  3.860e-07
mtp3     h_nextn.bin [chunk]                             1    19  1.100e-06  3.860e-07
mtp3     mtp_enorm [full]                                2    11  4.230e-07  1.137e-07
mtp3     mtp_enorm [chunk]                               2    50  2.977e-07  1.230e-07
mtp3     mtp_hnorm [full]                                2    68  1.137e-06  4.890e-07
mtp3     mtp_hnorm [chunk]                               2    68  1.137e-06  4.890e-07
mtp3     mtp_concat [e ; h] [full]                       2    68  1.137e-06  4.879e-07
mtp3     mtp_concat [e ; h] [chunk]                      2    68  1.137e-06  4.879e-07
mtp3     mtp_eh_proj [full]                              2    26  9.351e-07  5.172e-07
mtp3     mtp_eh_proj [chunk]                             2    26  9.351e-07  5.172e-07
mtp3     mtp_attn_norm [full]                            2    26  1.115e-06  4.833e-07
mtp3     mtp_attn_norm [chunk]                           2    26  1.115e-06  4.833e-07
mtp3     mtp_wqkv (fused projection) [full]              2    26  9.591e-07  3.993e-07
mtp3     mtp_wqkv (fused projection) [chunk]             2    26  9.591e-07  3.993e-07
mtp3     mtp_Qcur (RoPE) [full]                          2    71  3.749e-06  1.321e-06
mtp3     mtp_Qcur (RoPE) [chunk]                         2    71  3.749e-06  1.321e-06
mtp3     mtp_Kcur (RoPE) [full]                          2    35  3.253e-06  1.226e-06
mtp3     mtp_Kcur (RoPE) [chunk]                         2    35  3.253e-06  1.226e-06
mtp3     mtp_Vcur [full]                                 2    65  8.225e-07  3.265e-07
mtp3     mtp_Vcur [chunk]                                2    65  8.225e-07  3.265e-07
mtp3     kqv_out [full]                                  2    71  1.101e-06  8.350e-07
mtp3     kqv_out [chunk]                                 2    71  1.101e-06  8.350e-07
mtp3     mtp_attn_out [full]                             2    47  6.541e-07  5.523e-07
mtp3     mtp_attn_out [chunk]                            2    35  6.920e-07  5.297e-07
mtp3     mtp_attn_out_scaled [full]                      2    47  5.055e-07  5.691e-07
mtp3     mtp_attn_out_scaled [chunk]                     2    35  5.190e-07  5.297e-07
mtp3     mtp_ffn_inp [full]                              2    71  1.025e-06  4.172e-07
mtp3     mtp_ffn_inp [chunk]                             2    71  1.025e-06  4.172e-07
mtp3     mtp_ffn_norm [full]                             2    71  9.723e-07  4.199e-07
mtp3     mtp_ffn_norm [chunk]                            2    71  9.723e-07  4.199e-07
mtp3     ffn_up (dense) [full]                           2    20  9.589e-07  4.529e-07
mtp3     ffn_up (dense) [chunk]                          2    20  9.589e-07  4.529e-07
mtp3     ffn_gate (dense) [full]                         2    26  1.025e-06  4.007e-07
mtp3     ffn_gate (dense) [chunk]                        2    38  1.124e-06  4.301e-07
mtp3     ffn_swiglu (dense) [full]                       2    17  8.330e-07  7.117e-07
mtp3     ffn_swiglu (dense) [chunk]                      2    17  8.330e-07  7.117e-07
mtp3     mtp_ffn_out [full]                              2    26  7.843e-07  6.983e-07
mtp3     mtp_ffn_out [chunk]                             2    26  7.843e-07  6.983e-07
mtp3     mtp_post_ffn [full]                             2    26  1.247e-06  6.799e-07
mtp3     mtp_post_ffn [chunk]                            2    26  1.247e-06  6.799e-07
mtp3     h_nextn (graph output) [full]                   2    26  1.247e-06  6.799e-07
mtp3     h_nextn (graph output) [chunk]                  2    26  1.247e-06  6.799e-07
mtp3     mtp_shared_head_norm [full]                     2    26  1.316e-06  6.467e-07
mtp3     mtp_shared_head_norm [chunk]                    2    26  1.316e-06  6.467e-07
mtp3     result_output [full]                            2    74  1.309e-06  5.555e-07
mtp3     result_output [chunk]                           2    74  1.309e-06  5.555e-07
mtp3     logits.bin [full]                               2    74  1.309e-06  5.555e-07
mtp3     logits.bin [chunk]                              2    74  1.309e-06  5.555e-07
mtp3     h_nextn.bin [full]                              2    26  1.247e-06  6.799e-07
mtp3     h_nextn.bin [chunk]                             2    26  1.247e-06  6.799e-07
fa/mtp3  flash attention op (fa_tiled, own inputs)       0     0  3.864e-07  2.182e-07
fa/mtp3  flash attention op (fa_tiled, own inputs)       1     6  5.426e-07  2.285e-07
fa/mtp3  flash attention op (fa_tiled, own inputs)       2     8  4.655e-07  1.984e-07
fa/mtp3  flash attention op (one_chunk): elements > 0.5 spacing 0/17280   -1    -1  0.000e+00  0.000e+00
fa/mtp3  mtp_tok_embd (exact)                            0     0  0.000e+00  0.000e+00
fa/mtp3  h_in (row fed) [full]                           0    66  1.645e-03  4.794e-04
fa/mtp3  h_in (row fed) [chunk]                          0    15  9.692e-04  2.873e-04
fa/mtp3  mtp_tok_embd (exact)                            1     4  0.000e+00  0.000e+00
fa/mtp3  h_in (row fed) [full]                           1    67  1.645e-03  4.794e-04
fa/mtp3  h_in (row fed) [chunk]                          1    16  9.692e-04  2.873e-04
fa/mtp3  mtp_tok_embd (exact)                            2     8  0.000e+00  0.000e+00
fa/mtp3  h_in (row fed) [full]                           2    68  1.645e-03  4.794e-04
fa/mtp3  h_in (row fed) [chunk]                          2    17  9.692e-04  2.873e-04
fa/mtp3  h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
fa/mtp3  head 0 kqv_out vs (A), 11 rows reading foreign K/V (expected to differ)    0    -1  0.000e+00  2.052e+00
fa/mtp3  head 0 logits vs (A), the same rows (expected to differ)    0    -1  0.000e+00  4.039e-01
fa/mtp3  K (cache rows vs oracle)                        0     0  1.953e-03  5.225e-04
fa/mtp3  V (cache rows vs oracle)                        0     1  1.953e-03  5.513e-04
fa/mtp3  K (cache rows vs oracle)                        1     7  1.953e-03  3.820e-04
fa/mtp3  V (cache rows vs oracle)                        1     6  1.953e-03  5.110e-04
fa/mtp3  K (cache rows vs oracle)                        2     8  1.953e-03  5.609e-04
fa/mtp3  V (cache rows vs oracle)                        2    10  1.953e-03  5.139e-04
fa/mtp3  cache store: 67680 elements equal to the rounded pre-store value   -1    -1  0.000e+00  0.000e+00
fa/mtp3  cache rows equal to the oracle's stored rows: 1590144/1781760; lowest K 953923/1069056   -1    -1  0.000e+00  1.075e-01
fa/mtp3  mtp_enorm [full]                                0    57  3.020e-07  1.230e-07
fa/mtp3  mtp_enorm [chunk]                               0    57  3.020e-07  1.230e-07
fa/mtp3  mtp_hnorm [full]                                0    66  1.014e-03  4.369e-04
fa/mtp3  mtp_hnorm [chunk]                               0    15  6.136e-04  2.800e-04
fa/mtp3  mtp_concat [e ; h] [full]                       0    66  1.014e-03  4.369e-04
fa/mtp3  mtp_concat [e ; h] [chunk]                      0    60  7.082e-04  2.298e-04
fa/mtp3  mtp_eh_proj [full]                              0    63  1.090e-03  3.836e-04
fa/mtp3  mtp_eh_proj [chunk]                             0    27  4.827e-04  2.103e-04
fa/mtp3  mtp_attn_norm [full]                            0    63  1.078e-03  3.540e-04
fa/mtp3  mtp_attn_norm [chunk]                           0    27  5.137e-04  2.172e-04
fa/mtp3  mtp_wqkv (fused projection) [full]              0    63  1.247e-03  4.846e-04
fa/mtp3  mtp_wqkv (fused projection) [chunk]             0    18  6.122e-04  2.010e-04
fa/mtp3  mtp_Qcur (RoPE) [full]                          0    63  1.215e-03  5.439e-04
fa/mtp3  mtp_Qcur (RoPE) [chunk]                         0    18  6.122e-04  2.515e-04
fa/mtp3  mtp_Kcur (RoPE) [full]                          0    63  1.073e-03  5.502e-04
fa/mtp3  mtp_Kcur (RoPE) [chunk]                         0    18  5.068e-04  2.413e-04
fa/mtp3  mtp_Vcur [full]                                 0    63  8.698e-04  4.788e-04
fa/mtp3  mtp_Vcur [chunk]                                0    36  3.813e-04  2.876e-04
fa/mtp3  kqv_out [full]                                  0    39  1.406e-03  1.137e-03
fa/mtp3  kqv_out [chunk]                                 0    36  1.406e-03  1.217e-03
fa/mtp3  mtp_attn_out [full]                             0    81  5.468e-04  8.952e-04
fa/mtp3  mtp_attn_out [chunk]                            0    81  6.002e-04  9.824e-04
fa/mtp3  mtp_attn_out_scaled [full]                      0    81  4.101e-04  8.952e-04
fa/mtp3  mtp_attn_out_scaled [chunk]                     0    81  4.501e-04  9.824e-04
fa/mtp3  mtp_ffn_inp [full]                              0    81  6.396e-04  2.722e-04
fa/mtp3  mtp_ffn_inp [chunk]                             0    42  6.210e-04  2.535e-04
fa/mtp3  mtp_ffn_norm [full]                             0    81  7.088e-04  2.820e-04
fa/mtp3  mtp_ffn_norm [chunk]                            0    30  6.640e-04  2.674e-04
fa/mtp3  ffn_up (dense) [full]                           0    18  9.204e-04  3.452e-04
fa/mtp3  ffn_up (dense) [chunk]                          0    18  9.204e-04  3.452e-04
fa/mtp3  ffn_gate (dense) [full]                         0    18  8.306e-04  3.554e-04
fa/mtp3  ffn_gate (dense) [chunk]                        0    18  8.306e-04  3.554e-04
fa/mtp3  ffn_swiglu (dense) [full]                       0    18  1.727e-03  7.311e-04
fa/mtp3  ffn_swiglu (dense) [chunk]                      0    18  1.727e-03  7.311e-04
fa/mtp3  mtp_ffn_out [full]                              0    18  6.089e-04  4.733e-04
fa/mtp3  mtp_ffn_out [chunk]                             0    36  6.456e-04  5.623e-04
fa/mtp3  mtp_post_ffn [full]                             0    18  8.529e-04  3.818e-04
fa/mtp3  mtp_post_ffn [chunk]                            0    18  8.529e-04  3.818e-04
fa/mtp3  h_nextn (graph output) [full]                   0    18  8.529e-04  3.818e-04
fa/mtp3  h_nextn (graph output) [chunk]                  0    18  8.529e-04  3.818e-04
fa/mtp3  mtp_shared_head_norm [full]                     0    81  8.364e-04  3.777e-04
fa/mtp3  mtp_shared_head_norm [chunk]                    0    18  8.099e-04  3.688e-04
fa/mtp3  result_output [full]                            0    81  8.757e-04  4.430e-04
fa/mtp3  result_output [chunk]                           0    27  8.692e-04  3.243e-04
fa/mtp3  logits.bin [full]                               0    81  8.757e-04  4.430e-04
fa/mtp3  logits.bin [chunk]                              0    27  8.692e-04  3.243e-04
fa/mtp3  h_nextn.bin [full]                              0    18  8.529e-04  3.818e-04
fa/mtp3  h_nextn.bin [chunk]                             0    18  8.529e-04  3.818e-04
fa/mtp3  mtp_enorm [full]                                1    25  2.751e-07  1.160e-07
fa/mtp3  mtp_enorm [chunk]                               1    25  2.751e-07  1.160e-07
fa/mtp3  mtp_hnorm [full]                                1    67  1.067e-03  4.693e-04
fa/mtp3  mtp_hnorm [chunk]                               1    37  8.118e-04  3.304e-04
fa/mtp3  mtp_concat [e ; h] [full]                       1    64  1.443e-03  4.680e-04
fa/mtp3  mtp_concat [e ; h] [chunk]                      1    37  8.118e-04  2.942e-04
fa/mtp3  mtp_eh_proj [full]                              1    64  9.143e-04  3.096e-04
fa/mtp3  mtp_eh_proj [chunk]                             1     7  8.447e-04  2.140e-04
fa/mtp3  mtp_attn_norm [full]                            1    64  8.511e-04  3.309e-04
fa/mtp3  mtp_attn_norm [chunk]                           1     7  9.013e-04  2.594e-04
fa/mtp3  mtp_wqkv (fused projection) [full]              1    64  9.240e-04  3.012e-04
fa/mtp3  mtp_wqkv (fused projection) [chunk]             1    43  1.035e-03  3.753e-04
fa/mtp3  mtp_Qcur (RoPE) [full]                          1    64  9.240e-04  3.458e-04
fa/mtp3  mtp_Qcur (RoPE) [chunk]                         1    25  5.365e-04  2.349e-04
fa/mtp3  mtp_Kcur (RoPE) [full]                          1    64  7.504e-04  3.457e-04
fa/mtp3  mtp_Kcur (RoPE) [chunk]                         1    43  1.035e-03  4.606e-04
fa/mtp3  mtp_Vcur [full]                                 1    31  5.052e-04  3.259e-04
fa/mtp3  mtp_Vcur [chunk]                                1    31  5.052e-04  3.259e-04
fa/mtp3  kqv_out [full]                                  1    82  2.186e-03  1.597e-03
fa/mtp3  kqv_out [chunk]                                 1    46  1.821e-03  1.453e-03
fa/mtp3  mtp_attn_out [full]                             1    70  1.007e-03  1.095e-03
fa/mtp3  mtp_attn_out [chunk]                            1    43  9.238e-04  9.327e-04
fa/mtp3  mtp_attn_out_scaled [full]                      1    70  7.550e-04  1.095e-03
fa/mtp3  mtp_attn_out_scaled [chunk]                     1    43  6.929e-04  9.327e-04
fa/mtp3  mtp_ffn_inp [full]                              1    82  9.901e-04  3.826e-04
fa/mtp3  mtp_ffn_inp [chunk]                             1    31  9.254e-04  3.269e-04
fa/mtp3  mtp_ffn_norm [full]                             1    76  8.384e-04  3.680e-04
fa/mtp3  mtp_ffn_norm [chunk]                            1     7  1.255e-03  3.389e-04
fa/mtp3  ffn_up (dense) [full]                           1    58  1.145e-03  4.671e-04
fa/mtp3  ffn_up (dense) [chunk]                          1    40  9.520e-04  3.733e-04
fa/mtp3  ffn_gate (dense) [full]                         1    82  1.111e-03  5.291e-04
fa/mtp3  ffn_gate (dense) [chunk]                        1    31  9.525e-04  4.172e-04
fa/mtp3  ffn_swiglu (dense) [full]                       1    76  1.881e-03  9.412e-04
fa/mtp3  ffn_swiglu (dense) [chunk]                      1    16  1.101e-03  6.863e-04
fa/mtp3  mtp_ffn_out [full]                              1    82  9.125e-04  7.909e-04
fa/mtp3  mtp_ffn_out [chunk]                             1    31  7.021e-04  4.742e-04
fa/mtp3  mtp_post_ffn [full]                             1    82  1.393e-03  5.764e-04
fa/mtp3  mtp_post_ffn [chunk]                            1     7  1.718e-03  3.971e-04
fa/mtp3  h_nextn (graph output) [full]                   1    82  1.393e-03  5.764e-04
fa/mtp3  h_nextn (graph output) [chunk]                  1     7  1.718e-03  3.971e-04
fa/mtp3  mtp_shared_head_norm [full]                     1    82  1.212e-03  5.625e-04
fa/mtp3  mtp_shared_head_norm [chunk]                    1    55  8.844e-04  3.445e-04
fa/mtp3  result_output [full]                            1    64  1.432e-03  6.461e-04
fa/mtp3  result_output [chunk]                           1    55  9.923e-04  4.116e-04
fa/mtp3  logits.bin [full]                               1    64  1.432e-03  6.461e-04
fa/mtp3  logits.bin [chunk]                              1    55  9.923e-04  4.116e-04
fa/mtp3  h_nextn.bin [full]                              1    82  1.393e-03  5.764e-04
fa/mtp3  h_nextn.bin [chunk]                             1     7  1.718e-03  3.971e-04
fa/mtp3  mtp_enorm [full]                                2    11  4.230e-07  1.137e-07
fa/mtp3  mtp_enorm [chunk]                               2    50  2.977e-07  1.230e-07
fa/mtp3  mtp_hnorm [full]                                2    65  1.398e-03  4.743e-04
fa/mtp3  mtp_hnorm [chunk]                               2    17  6.040e-04  2.845e-04
fa/mtp3  mtp_concat [e ; h] [full]                       2    65  1.398e-03  4.743e-04
fa/mtp3  mtp_concat [e ; h] [chunk]                      2    62  8.108e-04  2.750e-04
fa/mtp3  mtp_eh_proj [full]                              2    65  1.192e-03  4.275e-04
fa/mtp3  mtp_eh_proj [chunk]                             2    20  5.909e-04  2.789e-04
fa/mtp3  mtp_attn_norm [full]                            2    65  1.161e-03  4.566e-04
fa/mtp3  mtp_attn_norm [chunk]                           2    20  5.908e-04  2.741e-04
fa/mtp3  mtp_wqkv (fused projection) [full]              2    26  8.291e-04  3.454e-04
fa/mtp3  mtp_wqkv (fused projection) [chunk]             2    26  8.291e-04  3.454e-04
fa/mtp3  mtp_Qcur (RoPE) [full]                          2    26  8.291e-04  3.860e-04
fa/mtp3  mtp_Qcur (RoPE) [chunk]                         2    26  8.291e-04  3.860e-04
fa/mtp3  mtp_Kcur (RoPE) [full]                          2    65  8.607e-04  3.540e-04
fa/mtp3  mtp_Kcur (RoPE) [chunk]                         2    29  5.166e-04  2.231e-04
fa/mtp3  mtp_Vcur [full]                                 2    65  8.611e-04  3.417e-04
fa/mtp3  mtp_Vcur [chunk]                                2    38  6.061e-04  2.329e-04
fa/mtp3  kqv_out [full]                                  2    26  1.330e-03  1.289e-03
fa/mtp3  kqv_out [chunk]                                 2    26  1.330e-03  1.289e-03
fa/mtp3  mtp_attn_out [full]                             2    56  1.305e-03  1.189e-03
fa/mtp3  mtp_attn_out [chunk]                            2    29  9.347e-04  8.638e-04
fa/mtp3  mtp_attn_out_scaled [full]                      2    56  9.789e-04  1.189e-03
fa/mtp3  mtp_attn_out_scaled [chunk]                     2    29  7.010e-04  8.638e-04
fa/mtp3  mtp_ffn_inp [full]                              2    26  9.703e-04  4.647e-04
fa/mtp3  mtp_ffn_inp [chunk]                             2    26  9.703e-04  4.647e-04
fa/mtp3  mtp_ffn_norm [full]                             2    71  9.397e-04  4.057e-04
fa/mtp3  mtp_ffn_norm [chunk]                            2    26  9.956e-04  3.824e-04
fa/mtp3  ffn_up (dense) [full]                           2    26  1.236e-03  5.665e-04
fa/mtp3  ffn_up (dense) [chunk]                          2    26  1.236e-03  5.665e-04
fa/mtp3  ffn_gate (dense) [full]                         2    59  1.215e-03  4.910e-04
fa/mtp3  ffn_gate (dense) [chunk]                        2    38  1.048e-03  4.009e-04
fa/mtp3  ffn_swiglu (dense) [full]                       2    71  1.937e-03  9.213e-04
fa/mtp3  ffn_swiglu (dense) [chunk]                      2    38  2.232e-03  6.187e-04
fa/mtp3  mtp_ffn_out [full]                              2    83  8.047e-04  7.788e-04
fa/mtp3  mtp_ffn_out [chunk]                             2    32  6.738e-04  7.091e-04
fa/mtp3  mtp_post_ffn [full]                             2    26  1.084e-03  5.915e-04
fa/mtp3  mtp_post_ffn [chunk]                            2    26  1.084e-03  5.915e-04
fa/mtp3  h_nextn (graph output) [full]                   2    26  1.084e-03  5.915e-04
fa/mtp3  h_nextn (graph output) [chunk]                  2    26  1.084e-03  5.915e-04
fa/mtp3  mtp_shared_head_norm [full]                     2    26  1.123e-03  5.518e-04
fa/mtp3  mtp_shared_head_norm [chunk]                    2    26  1.123e-03  5.518e-04
fa/mtp3  result_output [full]                            2    26  1.823e-03  6.856e-04
fa/mtp3  result_output [chunk]                           2    26  1.823e-03  6.856e-04
fa/mtp3  logits.bin [full]                               2    26  1.823e-03  6.856e-04
fa/mtp3  logits.bin [chunk]                              2    26  1.823e-03  6.856e-04
fa/mtp3  h_nextn.bin [full]                              2    26  1.084e-03  5.915e-04
fa/mtp3  h_nextn.bin [chunk]                             2    26  1.084e-03  5.915e-04
```

### 7.5 f16 / FA trunk: worst error per layer and per component (ec7630a; 79e2e74 identical, 7.3)

```
$ sed -n 721,729p <S>/n17/final-wrapped-ec.log
mimo2 -ctk f16 -ctv f16 -fa on: worst error per layer (layer -1 = embedding and head), llama.cpp ec7630a:
layer run          component                                                    graph    max abs        rel
   -1 fa/hd192     result_output [chunk]                                            8  1.138e-03  4.763e-04
    0 fa/hd192     kqv_out [full]                                                   6  9.344e-04  1.162e-03
    1 fa/hd192     kqv_out [full]                                                  14  1.570e-03  8.937e-04
    2 fa/rollback  attn_out (wo) [full]                                            24  9.958e-04  1.329e-03
    3 fa/hd192     kqv_out [full]                                                  16  2.944e-03  1.748e-03
    4 fa/rollback  kqv_out [chunk]                                                 25  1.079e-03  1.578e-03
    5 fa/rollback  kqv_out [chunk]                                                 38  1.247e-03  2.099e-03
```

```
$ sed -n 455,719p <S>/n17/final-wrapped-ec.log
mimo2 -ctk f16 -ctv f16 -fa on, llama.cpp ec7630a: worst error per run and component (rel = max|got-want| / max|want|) against the precision-aware oracle (asserted) and the plain f32 oracle (recorded)
run          component                                                          layer graph  rel aware    rel f32
fa/decode    attn_norm [full]                                                       5    22  3.762e-04        nan
fa/decode    attn_norm [chunk]                                                      5    53  2.342e-04        nan
fa/decode    wqkv (fused projection) [full]                                         5    22  5.154e-04        nan
fa/decode    wqkv (fused projection) [chunk]                                        5     6  2.892e-04        nan
fa/decode    Qcur (RoPE) [full]                                                     5    23  4.659e-04        nan
fa/decode    Qcur (RoPE) [chunk]                                                    5     5  3.158e-04        nan
fa/decode    Kcur (RoPE) [full]                                                     4    23  4.451e-04        nan
fa/decode    Kcur (RoPE) [chunk]                                                    5     6  3.738e-04        nan
fa/decode    Vcur [full]                                                            5    22  6.473e-04        nan
fa/decode    Vcur [chunk]                                                           5    22  3.095e-04        nan
fa/decode    kqv_out [full]                                                         5    40  2.054e-03  9.673e-03
fa/decode    kqv_out [chunk]                                                        5    40  2.099e-03  9.673e-03
fa/decode    attn_out (wo) [full]                                                   2    25  1.329e-03        nan
fa/decode    attn_out (wo) [chunk]                                                  5    18  1.144e-03        nan
fa/decode    attn_out_scaled (value scale) [full]                                   2    25  1.329e-03        nan
fa/decode    attn_out_scaled (value scale) [chunk]                                  5    18  1.144e-03        nan
fa/decode    kqv_out 2-norm rel: aware 1.815e-04, plain f32 6.474e-04              -1    -1  3.567e+00        nan
fa/decode    K (cache rows vs oracle)                                               3     3  5.814e-04        nan
fa/decode    V (cache rows vs oracle)                                               4     2  5.092e-04        nan
fa/decode    cache store: 120000 elements equal to the rounded pre-store value     -1    -1  0.000e+00        nan
fa/decode    cache rows equal to the oracle's stored rows: 7182948/7454720; lowest K 4308788/4472832    -1    -1  3.646e-02        nan
fa/decode    ffn_inp [full]                                                         5    21  4.139e-04        nan
fa/decode    ffn_inp [chunk]                                                        5    40  3.596e-04        nan
fa/decode    ffn_norm [full]                                                        5    22  4.326e-04        nan
fa/decode    ffn_norm [chunk]                                                       5    40  3.980e-04        nan
fa/decode    ffn_up (dense) [full]                                                  0     3  2.192e-05        nan
fa/decode    ffn_up (dense) [chunk]                                                 0     3  2.192e-05        nan
fa/decode    ffn_gate (dense) [full]                                                0     3  2.970e-05        nan
fa/decode    ffn_gate (dense) [chunk]                                               0     3  2.970e-05        nan
fa/decode    ffn_swiglu (dense) [full]                                              0     3  2.326e-05        nan
fa/decode    ffn_swiglu (dense) [chunk]                                             0     3  2.326e-05        nan
fa/decode    ffn_out (dense) [full]                                                 0     3  3.484e-05        nan
fa/decode    ffn_out (dense) [chunk]                                                0     3  3.484e-05        nan
fa/decode    l_out [full]                                                           5    21  4.908e-04        nan
fa/decode    l_out [chunk]                                                          5    40  3.621e-04        nan
fa/decode    ffn_moe_logits [full]                                                  5    12  7.755e-04        nan
fa/decode    ffn_moe_logits [chunk]                                                 5    12  7.755e-04        nan
fa/decode    ffn_moe_weights_norm [chunk]                                           5    45  1.023e-04        nan
fa/decode    ffn_moe_out [full]                                                     5    23  8.792e-04        nan
fa/decode    ffn_moe_out [chunk]                                                    5    12  6.361e-04        nan
fa/decode    flash attention op (fa_tiled, own inputs)                              5     2  5.078e-07        nan
fa/decode    flash attention op (fa_chunk, own inputs; spacings)                    2     3  4.883e-04        nan
fa/decode    flash attention op (one_chunk): elements > 0.5 spacing 0/41472, bit-equal 38605/41472    -1    -1  0.000e+00        nan
fa/decode    flash attention op with window n_swa - 1: graphs x SWA layers out of bound    -1    -1  2.560e+02        nan
fa/decode    flash attention op with window n_swa + 1: graphs x SWA layers out of bound    -1    -1  2.560e+02        nan
fa/decode    flash attention op with sinks dropped: graphs x SWA layers out of bound    -1    -1  2.560e+02        nan
fa/decode    embd (token rows, exact)                                              -1     0  0.000e+00        nan
fa/decode    result_norm [full]                                                    -1    23  4.254e-04        nan
fa/decode    result_norm [chunk]                                                   -1    40  3.752e-04        nan
fa/decode    result_output [full]                                                  -1    25  4.598e-04  1.488e-03
fa/decode    result_output [chunk]                                                 -1     3  3.741e-04  1.488e-03
fa/decode    logits.bin [full]                                                     -1    25  4.598e-04        nan
fa/decode    logits.bin [chunk]                                                    -1     3  3.741e-04        nan
fa/hd192     attn_norm [full]                                                       5     8  3.811e-04        nan
fa/hd192     attn_norm [chunk]                                                      5     8  3.811e-04        nan
fa/hd192     wqkv (fused projection) [full]                                         5    14  3.970e-04        nan
fa/hd192     wqkv (fused projection) [chunk]                                        5    14  3.970e-04        nan
fa/hd192     Qcur (RoPE) [full]                                                     5    14  4.180e-04        nan
fa/hd192     Qcur (RoPE) [chunk]                                                    5    14  4.180e-04        nan
fa/hd192     Kcur (RoPE) [full]                                                     5     8  3.827e-04        nan
fa/hd192     Kcur (RoPE) [chunk]                                                    5     8  3.827e-04        nan
fa/hd192     Vcur [full]                                                            4     5  4.028e-04        nan
fa/hd192     Vcur [chunk]                                                           4     5  4.028e-04        nan
fa/hd192     kqv_out [full]                                                         3    16  1.748e-03  2.914e-03
fa/hd192     kqv_out [chunk]                                                        3    16  1.748e-03  2.914e-03
fa/hd192     attn_out (wo) [full]                                                   5    15  1.191e-03        nan
fa/hd192     attn_out (wo) [chunk]                                                  5    15  1.191e-03        nan
fa/hd192     attn_out_scaled (value scale) [full]                                   5    15  1.191e-03        nan
fa/hd192     attn_out_scaled (value scale) [chunk]                                  5    15  1.191e-03        nan
fa/hd192     kqv_out 2-norm rel: aware 2.295e-04, plain f32 4.102e-04              -1    -1  1.787e+00        nan
fa/hd192     K (cache rows vs oracle)                                               5     0  4.878e-04        nan
fa/hd192     V (cache rows vs oracle)                                               5     0  5.160e-04        nan
fa/hd192     cache store: 256000 elements equal to the rounded pre-store value     -1    -1  0.000e+00        nan
fa/hd192     cache rows equal to the oracle's stored rows: 13548794/13926400; lowest K 8128992/8355840    -1    -1  2.711e-02        nan
fa/hd192     ffn_inp [full]                                                         5     8  4.103e-04        nan
fa/hd192     ffn_inp [chunk]                                                        5     8  4.103e-04        nan
fa/hd192     ffn_norm [full]                                                        4     8  4.128e-04        nan
fa/hd192     ffn_norm [chunk]                                                       4     8  4.128e-04        nan
fa/hd192     ffn_up (dense) [full]                                                  0     6  6.639e-05        nan
fa/hd192     ffn_up (dense) [chunk]                                                 0     6  6.639e-05        nan
fa/hd192     ffn_gate (dense) [full]                                                0     5  3.947e-05        nan
fa/hd192     ffn_gate (dense) [chunk]                                               0     5  3.947e-05        nan
fa/hd192     ffn_swiglu (dense) [full]                                              0    14  7.004e-05        nan
fa/hd192     ffn_swiglu (dense) [chunk]                                             0    14  7.004e-05        nan
fa/hd192     ffn_out (dense) [full]                                                 0    14  1.138e-04        nan
fa/hd192     ffn_out (dense) [chunk]                                                0    14  1.138e-04        nan
fa/hd192     l_out [full]                                                           4     8  4.005e-04        nan
fa/hd192     l_out [chunk]                                                          4     8  4.005e-04        nan
fa/hd192     ffn_moe_logits [full]                                                  4     8  4.932e-04        nan
fa/hd192     ffn_moe_logits [chunk]                                                 4     8  4.932e-04        nan
fa/hd192     ffn_moe_weights_norm [chunk]                                           5     6  1.003e-04        nan
fa/hd192     ffn_moe_out [full]                                                     5     8  9.497e-04        nan
fa/hd192     ffn_moe_out [chunk]                                                    5     8  9.497e-04        nan
fa/hd192     flash attention op (fa_tiled, own inputs)                              0     0  5.055e-07        nan
fa/hd192     flash attention op (fa_chunk, own inputs; spacings)                    1     4  3.662e-04        nan
fa/hd192     flash attention op (one_chunk): elements > 0.5 spacing 0/49152, bit-equal 46125/49152    -1    -1  0.000e+00        nan
fa/hd192     flash attention op with window n_swa - 1: graphs x SWA layers out of bound    -1    -1  6.800e+01        nan
fa/hd192     flash attention op with window n_swa + 1: graphs x SWA layers out of bound    -1    -1  6.800e+01        nan
fa/hd192     flash attention op with sinks dropped: graphs x SWA layers out of bound    -1    -1  6.800e+01        nan
fa/hd192     embd (token rows, exact)                                              -1     0  0.000e+00        nan
fa/hd192     result_norm [full]                                                    -1     1  3.724e-04        nan
fa/hd192     result_norm [chunk]                                                   -1     1  3.724e-04        nan
fa/hd192     result_output [full]                                                  -1     8  4.763e-04  7.716e-04
fa/hd192     result_output [chunk]                                                 -1     8  4.763e-04  7.716e-04
fa/hd192     logits.bin [full]                                                     -1     8  4.763e-04        nan
fa/hd192     logits.bin [chunk]                                                    -1     8  4.763e-04        nan
fa/pf64      attn_norm [full]                                                       4     0  7.327e-05        nan
fa/pf64      attn_norm [chunk]                                                      4     0  7.327e-05        nan
fa/pf64      wqkv (fused projection) [full]                                         5     0  8.445e-05        nan
fa/pf64      wqkv (fused projection) [chunk]                                        5     0  8.445e-05        nan
fa/pf64      Qcur (RoPE) [full]                                                     5     0  1.035e-04        nan
fa/pf64      Qcur (RoPE) [chunk]                                                    5     0  1.035e-04        nan
fa/pf64      Kcur (RoPE) [full]                                                     5     0  9.439e-05        nan
fa/pf64      Kcur (RoPE) [chunk]                                                    5     0  9.439e-05        nan
fa/pf64      Vcur [full]                                                            3     0  7.185e-05        nan
fa/pf64      Vcur [chunk]                                                           3     0  7.185e-05        nan
fa/pf64      kqv_out [full]                                                         3     0  3.306e-04  4.770e-04
fa/pf64      kqv_out [chunk]                                                        3     0  3.306e-04  4.770e-04
fa/pf64      attn_out (wo) [full]                                                   1     0  1.876e-04        nan
fa/pf64      attn_out (wo) [chunk]                                                  1     0  1.876e-04        nan
fa/pf64      attn_out_scaled (value scale) [full]                                   1     0  1.876e-04        nan
fa/pf64      attn_out_scaled (value scale) [chunk]                                  1     0  1.876e-04        nan
fa/pf64      kqv_out 2-norm rel: aware 5.655e-05, plain f32 2.917e-04              -1    -1  5.158e+00        nan
fa/pf64      K (cache rows vs oracle)                                               5     0  5.663e-04        nan
fa/pf64      V (cache rows vs oracle)                                               3     0  2.936e-04        nan
fa/pf64      cache store: 25600 elements equal to the rounded pre-store value      -1    -1  0.000e+00        nan
fa/pf64      cache rows equal to the oracle's stored rows: 101764/102400; lowest K 61047/61440    -1    -1  6.211e-03        nan
fa/pf64      ffn_inp [full]                                                         2     0  6.910e-05        nan
fa/pf64      ffn_inp [chunk]                                                        2     0  6.910e-05        nan
fa/pf64      ffn_norm [full]                                                        5     0  8.188e-05        nan
fa/pf64      ffn_norm [chunk]                                                       5     0  8.188e-05        nan
fa/pf64      ffn_up (dense) [full]                                                  0     0  1.322e-07        nan
fa/pf64      ffn_up (dense) [chunk]                                                 0     0  1.322e-07        nan
fa/pf64      ffn_gate (dense) [full]                                                0     0  1.246e-07        nan
fa/pf64      ffn_gate (dense) [chunk]                                               0     0  1.246e-07        nan
fa/pf64      ffn_swiglu (dense) [full]                                              0     0  1.946e-07        nan
fa/pf64      ffn_swiglu (dense) [chunk]                                             0     0  1.946e-07        nan
fa/pf64      ffn_out (dense) [full]                                                 0     0  2.159e-07        nan
fa/pf64      ffn_out (dense) [chunk]                                                0     0  2.159e-07        nan
fa/pf64      l_out [full]                                                           5     0  7.249e-05        nan
fa/pf64      l_out [chunk]                                                          5     0  7.249e-05        nan
fa/pf64      ffn_moe_logits [full]                                                  5     0  7.809e-05        nan
fa/pf64      ffn_moe_logits [chunk]                                                 5     0  7.809e-05        nan
fa/pf64      ffn_moe_weights_norm [chunk]                                           5     0  3.377e-05        nan
fa/pf64      ffn_moe_out [full]                                                     5     0  7.477e-05        nan
fa/pf64      ffn_moe_out [chunk]                                                    5     0  7.477e-05        nan
fa/pf64      flash attention op (fa_tiled, own inputs)                              5     0  2.407e-07        nan
fa/pf64      flash attention op with window n_swa - 1: graphs x SWA layers out of bound    -1    -1  4.000e+00        nan
fa/pf64      flash attention op with window n_swa + 1: graphs x SWA layers out of bound    -1    -1  4.000e+00        nan
fa/pf64      flash attention op with sinks dropped: graphs x SWA layers out of bound    -1    -1  4.000e+00        nan
fa/pf64      embd (token rows, exact)                                              -1     0  0.000e+00        nan
fa/pf64      result_norm [full]                                                    -1     0  8.795e-05        nan
fa/pf64      result_norm [chunk]                                                   -1     0  8.795e-05        nan
fa/pf64      result_output [full]                                                  -1     0  9.406e-05  3.310e-04
fa/pf64      result_output [chunk]                                                 -1     0  9.406e-05  3.310e-04
fa/pf64      logits.bin [full]                                                     -1     0  9.406e-05        nan
fa/pf64      logits.bin [chunk]                                                    -1     0  9.406e-05        nan
fa/rollback  attn_norm [full]                                                       5    22  3.954e-04        nan
fa/rollback  attn_norm [chunk]                                                      5    51  2.342e-04        nan
fa/rollback  wqkv (fused projection) [full]                                         5    21  4.448e-04        nan
fa/rollback  wqkv (fused projection) [chunk]                                        5     6  2.892e-04        nan
fa/rollback  Qcur (RoPE) [full]                                                     5    22  4.786e-04        nan
fa/rollback  Qcur (RoPE) [chunk]                                                    5     5  3.158e-04        nan
fa/rollback  Kcur (RoPE) [full]                                                     5    24  4.543e-04        nan
fa/rollback  Kcur (RoPE) [chunk]                                                    5     6  3.738e-04        nan
fa/rollback  Vcur [full]                                                            5    21  7.361e-04        nan
fa/rollback  Vcur [chunk]                                                           5    21  3.125e-04        nan
fa/rollback  kqv_out [full]                                                         5    38  2.054e-03  9.673e-03
fa/rollback  kqv_out [chunk]                                                        5    38  2.099e-03  9.673e-03
fa/rollback  attn_out (wo) [full]                                                   2    24  1.329e-03        nan
fa/rollback  attn_out (wo) [chunk]                                                  5    18  1.144e-03        nan
fa/rollback  attn_out_scaled (value scale) [full]                                   2    24  1.329e-03        nan
fa/rollback  attn_out_scaled (value scale) [chunk]                                  5    18  1.144e-03        nan
fa/rollback  kqv_out 2-norm rel: aware 1.915e-04, plain f32 6.627e-04              -1    -1  3.460e+00        nan
fa/rollback  K (cache rows vs oracle)                                               3     3  5.814e-04        nan
fa/rollback  V (cache rows vs oracle)                                               4     2  5.092e-04        nan
fa/rollback  cache store: 124800 elements equal to the rounded pre-store value     -1    -1  0.000e+00        nan
fa/rollback  cache rows equal to the oracle's stored rows: 6941676/7208960; lowest K 4164363/4325376    -1    -1  3.708e-02        nan
fa/rollback  ffn_inp [full]                                                         5    20  4.272e-04        nan
fa/rollback  ffn_inp [chunk]                                                        5    38  3.629e-04        nan
fa/rollback  ffn_norm [full]                                                        5    20  3.910e-04        nan
fa/rollback  ffn_norm [chunk]                                                       5    38  4.015e-04        nan
fa/rollback  ffn_up (dense) [full]                                                  0     3  2.192e-05        nan
fa/rollback  ffn_up (dense) [chunk]                                                 0     3  2.192e-05        nan
fa/rollback  ffn_gate (dense) [full]                                                0     3  2.970e-05        nan
fa/rollback  ffn_gate (dense) [chunk]                                               0     3  2.970e-05        nan
fa/rollback  ffn_swiglu (dense) [full]                                              0     3  2.326e-05        nan
fa/rollback  ffn_swiglu (dense) [chunk]                                             0     3  2.326e-05        nan
fa/rollback  ffn_out (dense) [full]                                                 0     3  3.484e-05        nan
fa/rollback  ffn_out (dense) [chunk]                                                0     3  3.484e-05        nan
fa/rollback  l_out [full]                                                           5    20  4.177e-04        nan
fa/rollback  l_out [chunk]                                                          5    38  3.633e-04        nan
fa/rollback  ffn_moe_logits [full]                                                  5    12  7.755e-04        nan
fa/rollback  ffn_moe_logits [chunk]                                                 5    12  7.755e-04        nan
fa/rollback  ffn_moe_weights_norm [chunk]                                           5    43  9.525e-05        nan
fa/rollback  ffn_moe_out [full]                                                     5    22  9.253e-04        nan
fa/rollback  ffn_moe_out [chunk]                                                    5    12  6.361e-04        nan
fa/rollback  flash attention op (fa_tiled, own inputs)                              5     2  5.078e-07        nan
fa/rollback  flash attention op (fa_chunk, own inputs; spacings)                    2     3  4.883e-04        nan
fa/rollback  flash attention op (one_chunk): elements > 0.5 spacing 0/46080, bit-equal 42929/46080    -1    -1  0.000e+00        nan
fa/rollback  flash attention op with window n_swa - 1: graphs x SWA layers out of bound    -1    -1  2.480e+02        nan
fa/rollback  flash attention op with window n_swa + 1: graphs x SWA layers out of bound    -1    -1  2.480e+02        nan
fa/rollback  flash attention op with sinks dropped: graphs x SWA layers out of bound    -1    -1  2.480e+02        nan
fa/rollback  embd (token rows, exact)                                              -1     0  0.000e+00        nan
fa/rollback  result_norm [full]                                                    -1    20  4.192e-04        nan
fa/rollback  result_norm [chunk]                                                   -1    38  3.764e-04        nan
fa/rollback  result_output [full]                                                  -1    24  4.717e-04  1.489e-03
fa/rollback  result_output [chunk]                                                 -1     3  3.741e-04  1.489e-03
fa/rollback  logits.bin [full]                                                     -1    24  4.717e-04        nan
fa/rollback  logits.bin [chunk]                                                    -1     3  3.741e-04        nan
fa/ub64      attn_norm [full]                                                       4     4  2.937e-04        nan
fa/ub64      attn_norm [chunk]                                                      5     4  1.922e-04        nan
fa/ub64      wqkv (fused projection) [full]                                         5     4  3.222e-04        nan
fa/ub64      wqkv (fused projection) [chunk]                                        5     4  3.190e-04        nan
fa/ub64      Qcur (RoPE) [full]                                                     5     4  3.428e-04        nan
fa/ub64      Qcur (RoPE) [chunk]                                                    5     4  3.428e-04        nan
fa/ub64      Kcur (RoPE) [full]                                                     5     4  3.611e-04        nan
fa/ub64      Kcur (RoPE) [chunk]                                                    5     4  2.800e-04        nan
fa/ub64      Vcur [full]                                                            5     4  3.045e-04        nan
fa/ub64      Vcur [chunk]                                                           5     4  2.050e-04        nan
fa/ub64      kqv_out [full]                                                         5     4  1.505e-03  5.982e-03
fa/ub64      kqv_out [chunk]                                                        5     4  1.508e-03  5.982e-03
fa/ub64      attn_out (wo) [full]                                                   5     4  8.094e-04        nan
fa/ub64      attn_out (wo) [chunk]                                                  4     4  7.653e-04        nan
fa/ub64      attn_out_scaled (value scale) [full]                                   5     4  8.094e-04        nan
fa/ub64      attn_out_scaled (value scale) [chunk]                                  4     4  7.653e-04        nan
fa/ub64      kqv_out 2-norm rel: aware 1.284e-04, plain f32 4.769e-04              -1    -1  3.714e+00        nan
fa/ub64      K (cache rows vs oracle)                                               5     0  5.663e-04        nan
fa/ub64      V (cache rows vs oracle)                                               4     2  5.092e-04        nan
fa/ub64      cache store: 120000 elements equal to the rounded pre-store value     -1    -1  0.000e+00        nan
fa/ub64      cache rows equal to the oracle's stored rows: 524337/532480; lowest K 314555/319488    -1    -1  1.529e-02        nan
fa/ub64      ffn_inp [full]                                                         4     4  2.493e-04        nan
fa/ub64      ffn_inp [chunk]                                                        5     4  1.718e-04        nan
fa/ub64      ffn_norm [full]                                                        4     4  2.550e-04        nan
fa/ub64      ffn_norm [chunk]                                                       5     4  2.071e-04        nan
fa/ub64      ffn_up (dense) [full]                                                  0     4  2.644e-05        nan
fa/ub64      ffn_up (dense) [chunk]                                                 0     4  2.644e-05        nan
fa/ub64      ffn_gate (dense) [full]                                                0     4  3.375e-05        nan
fa/ub64      ffn_gate (dense) [chunk]                                               0     4  3.375e-05        nan
fa/ub64      ffn_swiglu (dense) [full]                                              0     4  2.255e-05        nan
fa/ub64      ffn_swiglu (dense) [chunk]                                             0     4  2.255e-05        nan
fa/ub64      ffn_out (dense) [full]                                                 0     4  2.708e-05        nan
fa/ub64      ffn_out (dense) [chunk]                                                0     4  2.708e-05        nan
fa/ub64      l_out [full]                                                           5     4  2.715e-04        nan
fa/ub64      l_out [chunk]                                                          5     4  2.219e-04        nan
fa/ub64      ffn_moe_logits [full]                                                  5     4  3.755e-04        nan
fa/ub64      ffn_moe_logits [chunk]                                                 5     4  2.872e-04        nan
fa/ub64      ffn_moe_weights_norm [chunk]                                           4     4  8.866e-05        nan
fa/ub64      ffn_moe_out [full]                                                     5     4  4.643e-04        nan
fa/ub64      ffn_moe_out [chunk]                                                    5     4  3.979e-04        nan
fa/ub64      flash attention op (fa_tiled, own inputs)                              5     2  5.078e-07        nan
fa/ub64      flash attention op (fa_chunk, own inputs; spacings)                    0     4  4.883e-04        nan
fa/ub64      flash attention op (one_chunk): elements > 0.5 spacing 0/16896, bit-equal 15804/16896    -1    -1  0.000e+00        nan
fa/ub64      flash attention op with window n_swa - 1: graphs x SWA layers out of bound    -1    -1  2.000e+01        nan
fa/ub64      flash attention op with window n_swa + 1: graphs x SWA layers out of bound    -1    -1  2.000e+01        nan
fa/ub64      flash attention op with sinks dropped: graphs x SWA layers out of bound    -1    -1  2.000e+01        nan
fa/ub64      embd (token rows, exact)                                              -1     0  0.000e+00        nan
fa/ub64      result_norm [full]                                                    -1     4  2.897e-04        nan
fa/ub64      result_norm [chunk]                                                   -1     4  2.700e-04        nan
fa/ub64      result_output [full]                                                  -1     4  3.775e-04  1.055e-03
fa/ub64      result_output [chunk]                                                 -1     4  2.616e-04  1.055e-03
fa/ub64      logits.bin [full]                                                     -1     4  3.775e-04        nan
fa/ub64      logits.bin [chunk]                                                    -1     4  2.616e-04        nan
```

### 7.6 N16 and N10 steps on `d5b3537`, N10's flash-attention op lines

```
$ tail -n 1 <S>/n17/final-n16-ec.log
::notice::n16-mimo2-decode @ llama.cpp ec7630a: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
$ tail -n 1 <S>/n17/final-n10-ec.log
::notice::n10-f16fa @ llama.cpp ec7630a: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
$ tail -n 1 <S>/n17/final-n16-r1.log
::notice::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
$ tail -n 1 <S>/n17/final-n10-r1.log
::notice::n10-f16fa @ llama.cpp 79e2e74: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
$ grep -E 'flash attention op' <S>/n17/final-n10-ec.log
decode   flash attention op (fa_chunk, own inputs)                      3     6  2.008e-07        nan
pf64     flash attention op (fa_tiled, own inputs)                      7     0  1.343e-07        nan
rollback flash attention op (fa_chunk, own inputs)                      3     5  2.008e-07        nan
ub10     flash attention op (fa_chunk, own inputs)                      3     5  1.834e-07        nan
ub16     flash attention op (fa_chunk, own inputs)                      3     3  1.525e-07        nan
decode   flash attention op: elements > 0.5 spacing 0/8192
rollback flash attention op: elements > 0.5 spacing 0/8448
ub10     flash attention op: elements > 0.5 spacing 0/8192
ub16     flash attention op: elements > 0.5 spacing 0/8192
decode   flash attention op (fa_chunk, own inputs)                     11     0  2.462e-07        nan
pf64     flash attention op (fa_tiled, own inputs)                      7     0  2.103e-07        nan
rollback flash attention op (fa_chunk, own inputs)                     11     0  2.462e-07        nan
ub18     flash attention op (fa_chunk, own inputs)                      7     2  2.683e-07        nan
decode   flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
rollback flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
ub18     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
$ grep -E 'flash attention op' <S>/n17/final-n10-r1.log
decode   flash attention op (fa_chunk, own inputs)                      3     6  2.008e-07        nan
pf64     flash attention op (fa_tiled, own inputs)                      7     0  1.343e-07        nan
rollback flash attention op (fa_chunk, own inputs)                      3     5  2.008e-07        nan
ub10     flash attention op (fa_chunk, own inputs)                      3     5  1.834e-07        nan
ub16     flash attention op (fa_chunk, own inputs)                      3     3  1.525e-07        nan
decode   flash attention op: elements > 0.5 spacing 0/8192
rollback flash attention op: elements > 0.5 spacing 0/8448
ub10     flash attention op: elements > 0.5 spacing 0/8192
ub16     flash attention op: elements > 0.5 spacing 0/8192
decode   flash attention op (fa_chunk, own inputs)                     11     0  2.462e-07        nan
pf64     flash attention op (fa_tiled, own inputs)                      7     0  2.103e-07        nan
rollback flash attention op (fa_chunk, own inputs)                     11     9  2.526e-07        nan
ub10     flash attention op (fa_chunk, own inputs)                      7     4  2.931e-07        nan
ub16     flash attention op (fa_chunk, own inputs)                     11     0  2.462e-07        nan
ub18     flash attention op (fa_chunk, own inputs)                      7     2  2.683e-07        nan
decode   flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
rollback flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
ub10     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
ub16     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
ub18     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.000
```

### 7.7 Mutation check (19 mutants, each pin)

```
$ cat <S>/n17/final-mut-ec.log
llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d; dumps <S>/n17/work-ec/n17
baseline (no mutation): rc=0 Ran 70 tests in 44.816s | OK
H1 off-by-one head offset: head k computed with layer n_layer + (k + 1) mod H
    ref/mimo2.py: 'out of range [0, {hp.get(\'n_nextn\')})")\n    il = hp["n_layer"] + offset' -> 'out of range [0, {hp.get(\'n_nextn\')})")\n    il = hp["n_layer"] + (offset + 1) % max(1, hp.'
    rc=1 Ran 70 tests in 43.481s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_head_k_is_layer_n_layer_plus_k
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
H2 off-by-one head offset in draft: step k runs head min(k + 1, H - 1)
    ref/mimo2.py: 'out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=i if chain else 0, prec' -> 'out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=min(i + 1, n_heads - 1)'
    rc=1 Ran 70 tests in 43.343s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
P1 wrong h source in process: head k > 0 takes head k-1's h_nextn (shifted) instead of the target's
    ref/mimo2.py: 'out[head] = mtp_ubatches(tokens, h_rows, W, hp, cache, n_ubatch, head, prec, snap)' -> 'rows_k = h_rows if head == 0 else np.vstack([np.asarray(h_rows)[:1]] + [np.vstack([o["h_ne'
    rc=1 Ran 70 tests in 44.225s | FAILED (failures=10)
    mimo2_heads_ref: 2 failing: MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
P2 wrong h source in draft: every row of step k paired with the pending target row
    ref/mimo2.py: 'ids, rows = [int(id_last)] + fed, list(chain_h)' -> 'ids, rows = [int(id_last)] + fed, [pending] * (len(fed) + 1)'
    rc=1 Ran 70 tests in 39.402s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
P3 wrong h source in draft: the new token paired with the h of step k-2 (rows not chained one back)
    ref/mimo2.py: 'chain_h.append(h)' -> 'chain_h.append(chain_h[-1])'
    rc=1 Ran 70 tests in 43.223s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
Q1 process without seq_rm before each head (heads 1.. append after head 0)
    ref/mimo2.py: 'if n_heads > 1:\n            cache_truncate(cache, p0)' -> 'if False:\n            cache_truncate(cache, p0)'
    rc=1 Ran 57 tests in 42.259s | FAILED (failures=1, errors=4)
    mimo2_heads_ref: 3 failing: MtpHeadsRef.test_chained_draft, MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call
    mimo2_mtp_heads_parity: 2 failing: MtpHeads.setUpClass, MtpHeadsF16FA.setUpClass
Q2 draft step without seq_rm(pos0) (the prefix appended after the previous step's)
    ref/mimo2.py: 'if chain:\n            cache_truncate(cache, pos0)' -> 'if False:\n            cache_truncate(cache, pos0)'
    rc=1 Ran 70 tests in 43.883s | FAILED (failures=12)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 11 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_head0_reads_foreign_kv, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens ...
K1 per-layer K/V positions not recorded (every visible cell counted as foreign)
    ref/mimo2.py: 'cache["P"][il][idxs] = pos' -> ''
    rc=1 Ran 70 tests in 46.984s | FAILED (failures=3, errors=1)
    mimo2_heads_ref: 2 failing: MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call
    mimo2_mtp_heads_parity: 2 failing: MtpHeads.test_head0_reads_foreign_kv, MtpHeadsF16FA.test_head0_reads_foreign_kv
T1 test: after a rejection the pending h stays the call's last row (accept() ignored)
    tests/test_mimo2_mtp_heads_parity.py: 'pending = cls.hB_at[p0 - 1]' -> 'pending = pending'
    rc=1 Ran 70 tests in 46.814s | FAILED (failures=8)
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
F1 sink ignored in one_chunk flash attention
    ref/fattn.py: 'if sink is not None:                                                  # O:9034-9056: on VK' -> 'if False:                                                             # O:9034-9056: on VK'
    rc=1 Ran 70 tests in 29.484s | FAILED (failures=34)
    fattn_sinks_ref: 4 failing: SinkModes.test_gqa_and_head_sizes, SinkModes.test_one_chunk_is_the_transcribed_kernel, SinkModes.test_one_chunk_with_sink_against_the_float64_softmax, SinkModes.test_sink_changes_every_mode
    mimo2_f16fa_parity: 24 failing: FaDecode.test_attention, FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_cache_store_and_rows, FaDecode.test_ffn, FaDecode.test_flash_attention_op, FaDecode.test_head_and_logits ...
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 5 failing: MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_flash_attention_op, MtpHeadsF16FA.test_graphs_heads_and_pairing, MtpHeadsF16FA.test_kv_cells, MtpHeadsF16FA.test_mtp_heads_block
F2 sink applied before the cell loop instead of after it (as a first cell with a zero value row)
    ref/fattn.py: 'M = np.full(G, -np.inf)\n    S = np.zeros(G, dtype=np.float32)' -> 'M = np.full(G, -np.inf) if sink is None else f32(sink)\n    S = np.zeros(G, dtype=np.float3'
    rc=1 Ran 70 tests in 33.572s | FAILED (failures=6)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 4 failing: FaDecode.test_flash_attention_op, FaHeadDims192.test_flash_attention_op, FaRollback.test_flash_attention_op, FaUbatch64.test_flash_attention_op
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
R1 f16 rounding of the one_chunk accumulator skipped (ggml_vec_mad_f16 in f32)
    ref/fattn.py: 'acc = f16(acc + vc[i][None, :] * vs[:, None])                     # ggml_vec_mad_f16' -> 'acc = acc + vc[i][None, :] * vs[:, None]                          # ggml_vec_mad_f16'
    rc=1 Ran 70 tests in 32.289s | FAILED (failures=12)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 10 failing: FaDecode.test_attention, FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_flash_attention_op, FaHeadDims192.test_attention_output_vs_plain_oracle, FaHeadDims192.test_flash_attention_op, FaRollback.test_attention ...
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
R2 f16 rounding of the cache store skipped (oracle keeps the f32 K / V)
    ref/mimo2.py: 'ks, vs = (F.store(k, prec), F.store(v, prec)) if prec else (k, v)' -> 'ks, vs = (k, v)'
    rc=1 Ran 70 tests in 40.337s | FAILED (failures=5)
    mimo2_f16fa_parity: 3 failing: FaDecode.test_cache_store_and_rows, FaRollback.test_cache_store_and_rows, FaUbatch64.test_cache_store_and_rows
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_kv_cells
R3 Q not rounded to f16 in one_chunk
    ref/fattn.py: 'qq = f16(qh)                                                          # Q_q = from_float(q' -> 'qq = np.asarray(qh, dtype=np.float64)'
    rc=1 Ran 70 tests in 29.563s | FAILED (failures=9)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 7 failing: FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_flash_attention_op, FaHeadDims192.test_flash_attention_op, FaRollback.test_attention_output_vs_plain_oracle, FaRollback.test_flash_attention_op, FaUbatch64.test_attention_output_vs_plain_oracle ...
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
E1 expf of the float64 difference (the f32 argument of O:8997, 9003 not rounded)
    ref/fattn.py: 'ms = np.where(new, f32(np.exp(f32(np.where(new, M - s, 0.0)))), 1.0)\n        vs = np.where' -> 'ms = np.where(new, f32(np.exp(np.where(new, M - s, 0.0))), 1.0)\n        vs = np.where(new,'
    rc=1 Ran 70 tests in 30.199s | FAILED (failures=5)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 3 failing: FaDecode.test_flash_attention_op, FaHeadDims192.test_flash_attention_op, FaRollback.test_flash_attention_op
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
E2 attention scale float64 1/sqrt(d) instead of llama.cpp's f32 1.0f/sqrtf(d)
    ref/mimo2.py: 'return F.kq_scale_f32(hp["k_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])' -> 'return 1.0 / np.sqrt(hp["k_dim"])'
    rc=1 Ran 70 tests in 30.505s | FAILED (failures=5)
    mimo2_f16fa_parity: 3 failing: FaDecode.test_flash_attention_op, FaRollback.test_flash_attention_op, FaUbatch64.test_flash_attention_op
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
E3 the AVX2 lanes of the f16 dot not modelled (double sum for every head size)
    ref/fattn.py: 'n_simd = qq.shape[1] & ~(F16_STEP - 1)' -> 'n_simd = 0'
    rc=1 Ran 70 tests in 41.528s | FAILED (failures=2)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 1 failing: FaHeadDims192.test_flash_attention_op
V1 K/V head-size confusion: the attention scale from the V head size (1/sqrt(16) for K 24)
    ref/mimo2.py: 'return F.kq_scale_f32(hp["k_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])' -> 'return F.kq_scale_f32(hp["v_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])'
    rc=1 Ran 70 tests in 25.557s | FAILED (failures=36)
    mimo2_f16fa_parity: 30 failing: FaDecode.test_attention, FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_cache_store_and_rows, FaDecode.test_ffn, FaDecode.test_flash_attention_op, FaDecode.test_head_and_logits ...
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 5 failing: MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_flash_attention_op, MtpHeadsF16FA.test_graphs_heads_and_pairing, MtpHeadsF16FA.test_kv_cells, MtpHeadsF16FA.test_mtp_heads_block
V2 K/V head-size confusion: the V rows of the cache sized by the K head
    ref/mimo2.py: 'c["V"][il] = np.zeros((size, hp["n_head_kv"][il], hp["v_dim"]))' -> 'c["V"][il] = np.zeros((size, hp["n_head_kv"][il], hp["k_dim"]))'
    rc=1 Ran 17 tests in 12.326s | FAILED (errors=13)
    mimo2_f16fa_parity: 5 failing: FaDecode.setUpClass, FaHeadDims192.setUpClass, FaPrefill64.setUpClass, FaRollback.setUpClass, FaUbatch64.setUpClass
    mimo2_heads_ref: 6 failing: MtpHeadsRef.test_chained_draft, MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call, MtpHeadsRef.test_one_head_draft_is_n16, MtpHeadsRef.test_precision_argument, MtpHeadsRef.test_teacher_forced_draft_tokens
    mimo2_mtp_heads_parity: 2 failing: MtpHeads.setUpClass, MtpHeadsF16FA.setUpClass
caught 19/19
$ cat <S>/n17/final-mut-r1.log
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6; dumps <S>/n17/work-r1/n17
baseline (no mutation): rc=0 Ran 70 tests in 42.389s | OK
H1 off-by-one head offset: head k computed with layer n_layer + (k + 1) mod H
    ref/mimo2.py: 'out of range [0, {hp.get(\'n_nextn\')})")\n    il = hp["n_layer"] + offset' -> 'out of range [0, {hp.get(\'n_nextn\')})")\n    il = hp["n_layer"] + (offset + 1) % max(1, hp.'
    rc=1 Ran 70 tests in 41.409s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_head_k_is_layer_n_layer_plus_k
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
H2 off-by-one head offset in draft: step k runs head min(k + 1, H - 1)
    ref/mimo2.py: 'out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=i if chain else 0, prec' -> 'out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=min(i + 1, n_heads - 1)'
    rc=1 Ran 70 tests in 42.320s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
P1 wrong h source in process: head k > 0 takes head k-1's h_nextn (shifted) instead of the target's
    ref/mimo2.py: 'out[head] = mtp_ubatches(tokens, h_rows, W, hp, cache, n_ubatch, head, prec, snap)' -> 'rows_k = h_rows if head == 0 else np.vstack([np.asarray(h_rows)[:1]] + [np.vstack([o["h_ne'
    rc=1 Ran 70 tests in 42.092s | FAILED (failures=10)
    mimo2_heads_ref: 2 failing: MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
P2 wrong h source in draft: every row of step k paired with the pending target row
    ref/mimo2.py: 'ids, rows = [int(id_last)] + fed, list(chain_h)' -> 'ids, rows = [int(id_last)] + fed, [pending] * (len(fed) + 1)'
    rc=1 Ran 70 tests in 42.396s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
P3 wrong h source in draft: the new token paired with the h of step k-2 (rows not chained one back)
    ref/mimo2.py: 'chain_h.append(h)' -> 'chain_h.append(chain_h[-1])'
    rc=1 Ran 70 tests in 41.948s | FAILED (failures=9)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
Q1 process without seq_rm before each head (heads 1.. append after head 0)
    ref/mimo2.py: 'if n_heads > 1:\n            cache_truncate(cache, p0)' -> 'if False:\n            cache_truncate(cache, p0)'
    rc=1 Ran 57 tests in 40.429s | FAILED (failures=1, errors=4)
    mimo2_heads_ref: 3 failing: MtpHeadsRef.test_chained_draft, MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call
    mimo2_mtp_heads_parity: 2 failing: MtpHeads.setUpClass, MtpHeadsF16FA.setUpClass
Q2 draft step without seq_rm(pos0) (the prefix appended after the previous step's)
    ref/mimo2.py: 'if chain:\n            cache_truncate(cache, pos0)' -> 'if False:\n            cache_truncate(cache, pos0)'
    rc=1 Ran 70 tests in 42.297s | FAILED (failures=12)
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_chained_draft
    mimo2_mtp_heads_parity: 11 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_head0_reads_foreign_kv, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens ...
K1 per-layer K/V positions not recorded (every visible cell counted as foreign)
    ref/mimo2.py: 'cache["P"][il][idxs] = pos' -> ''
    rc=1 Ran 70 tests in 42.984s | FAILED (failures=3, errors=1)
    mimo2_heads_ref: 2 failing: MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call
    mimo2_mtp_heads_parity: 2 failing: MtpHeads.test_head0_reads_foreign_kv, MtpHeadsF16FA.test_head0_reads_foreign_kv
T1 test: after a rejection the pending h stays the call's last row (accept() ignored)
    tests/test_mimo2_mtp_heads_parity.py: 'pending = cls.hB_at[p0 - 1]' -> 'pending = pending'
    rc=1 Ran 70 tests in 46.475s | FAILED (failures=8)
    mimo2_mtp_heads_parity: 8 failing: MtpHeads.test_draft_tokens, MtpHeads.test_graphs_heads_and_pairing, MtpHeads.test_kv_cells, MtpHeads.test_mtp_heads_block, MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_graphs_heads_and_pairing ...
F1 sink ignored in one_chunk flash attention
    ref/fattn.py: 'if sink is not None:                                                  # O:9034-9056: on VK' -> 'if False:                                                             # O:9034-9056: on VK'
    rc=1 Ran 70 tests in 26.438s | FAILED (failures=34)
    fattn_sinks_ref: 4 failing: SinkModes.test_gqa_and_head_sizes, SinkModes.test_one_chunk_is_the_transcribed_kernel, SinkModes.test_one_chunk_with_sink_against_the_float64_softmax, SinkModes.test_sink_changes_every_mode
    mimo2_f16fa_parity: 24 failing: FaDecode.test_attention, FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_cache_store_and_rows, FaDecode.test_ffn, FaDecode.test_flash_attention_op, FaDecode.test_head_and_logits ...
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 5 failing: MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_flash_attention_op, MtpHeadsF16FA.test_graphs_heads_and_pairing, MtpHeadsF16FA.test_kv_cells, MtpHeadsF16FA.test_mtp_heads_block
F2 sink applied before the cell loop instead of after it (as a first cell with a zero value row)
    ref/fattn.py: 'M = np.full(G, -np.inf)\n    S = np.zeros(G, dtype=np.float32)' -> 'M = np.full(G, -np.inf) if sink is None else f32(sink)\n    S = np.zeros(G, dtype=np.float3'
    rc=1 Ran 70 tests in 31.421s | FAILED (failures=6)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 4 failing: FaDecode.test_flash_attention_op, FaHeadDims192.test_flash_attention_op, FaRollback.test_flash_attention_op, FaUbatch64.test_flash_attention_op
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
R1 f16 rounding of the one_chunk accumulator skipped (ggml_vec_mad_f16 in f32)
    ref/fattn.py: 'acc = f16(acc + vc[i][None, :] * vs[:, None])                     # ggml_vec_mad_f16' -> 'acc = acc + vc[i][None, :] * vs[:, None]                          # ggml_vec_mad_f16'
    rc=1 Ran 70 tests in 27.445s | FAILED (failures=12)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 10 failing: FaDecode.test_attention, FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_flash_attention_op, FaHeadDims192.test_attention_output_vs_plain_oracle, FaHeadDims192.test_flash_attention_op, FaRollback.test_attention ...
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
R2 f16 rounding of the cache store skipped (oracle keeps the f32 K / V)
    ref/mimo2.py: 'ks, vs = (F.store(k, prec), F.store(v, prec)) if prec else (k, v)' -> 'ks, vs = (k, v)'
    rc=1 Ran 70 tests in 41.407s | FAILED (failures=5)
    mimo2_f16fa_parity: 3 failing: FaDecode.test_cache_store_and_rows, FaRollback.test_cache_store_and_rows, FaUbatch64.test_cache_store_and_rows
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_kv_cells
R3 Q not rounded to f16 in one_chunk
    ref/fattn.py: 'qq = f16(qh)                                                          # Q_q = from_float(q' -> 'qq = np.asarray(qh, dtype=np.float64)'
    rc=1 Ran 70 tests in 29.233s | FAILED (failures=9)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 7 failing: FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_flash_attention_op, FaHeadDims192.test_flash_attention_op, FaRollback.test_attention_output_vs_plain_oracle, FaRollback.test_flash_attention_op, FaUbatch64.test_attention_output_vs_plain_oracle ...
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
E1 expf of the float64 difference (the f32 argument of O:8997, 9003 not rounded)
    ref/fattn.py: 'ms = np.where(new, f32(np.exp(f32(np.where(new, M - s, 0.0)))), 1.0)\n        vs = np.where' -> 'ms = np.where(new, f32(np.exp(np.where(new, M - s, 0.0))), 1.0)\n        vs = np.where(new,'
    rc=1 Ran 70 tests in 29.954s | FAILED (failures=5)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 3 failing: FaDecode.test_flash_attention_op, FaHeadDims192.test_flash_attention_op, FaRollback.test_flash_attention_op
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
E2 attention scale float64 1/sqrt(d) instead of llama.cpp's f32 1.0f/sqrtf(d)
    ref/mimo2.py: 'return F.kq_scale_f32(hp["k_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])' -> 'return 1.0 / np.sqrt(hp["k_dim"])'
    rc=1 Ran 70 tests in 30.652s | FAILED (failures=5)
    mimo2_f16fa_parity: 3 failing: FaDecode.test_flash_attention_op, FaRollback.test_flash_attention_op, FaUbatch64.test_flash_attention_op
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 1 failing: MtpHeadsF16FA.test_flash_attention_op
E3 the AVX2 lanes of the f16 dot not modelled (double sum for every head size)
    ref/fattn.py: 'n_simd = qq.shape[1] & ~(F16_STEP - 1)' -> 'n_simd = 0'
    rc=1 Ran 70 tests in 37.898s | FAILED (failures=2)
    fattn_sinks_ref: 1 failing: SinkModes.test_one_chunk_is_the_transcribed_kernel
    mimo2_f16fa_parity: 1 failing: FaHeadDims192.test_flash_attention_op
V1 K/V head-size confusion: the attention scale from the V head size (1/sqrt(16) for K 24)
    ref/mimo2.py: 'return F.kq_scale_f32(hp["k_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])' -> 'return F.kq_scale_f32(hp["v_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])'
    rc=1 Ran 70 tests in 22.490s | FAILED (failures=36)
    mimo2_f16fa_parity: 30 failing: FaDecode.test_attention, FaDecode.test_attention_output_vs_plain_oracle, FaDecode.test_cache_store_and_rows, FaDecode.test_ffn, FaDecode.test_flash_attention_op, FaDecode.test_head_and_logits ...
    mimo2_heads_ref: 1 failing: MtpHeadsRef.test_precision_argument
    mimo2_mtp_heads_parity: 5 failing: MtpHeadsF16FA.test_draft_tokens, MtpHeadsF16FA.test_flash_attention_op, MtpHeadsF16FA.test_graphs_heads_and_pairing, MtpHeadsF16FA.test_kv_cells, MtpHeadsF16FA.test_mtp_heads_block
V2 K/V head-size confusion: the V rows of the cache sized by the K head
    ref/mimo2.py: 'c["V"][il] = np.zeros((size, hp["n_head_kv"][il], hp["v_dim"]))' -> 'c["V"][il] = np.zeros((size, hp["n_head_kv"][il], hp["k_dim"]))'
    rc=1 Ran 17 tests in 9.510s | FAILED (errors=13)
    mimo2_f16fa_parity: 5 failing: FaDecode.setUpClass, FaHeadDims192.setUpClass, FaPrefill64.setUpClass, FaRollback.setUpClass, FaUbatch64.setUpClass
    mimo2_heads_ref: 6 failing: MtpHeadsRef.test_chained_draft, MtpHeadsRef.test_chained_process_equals_every_head_over_the_sequence, MtpHeadsRef.test_head0_reads_foreign_kv_after_a_wrapping_call, MtpHeadsRef.test_one_head_draft_is_n16, MtpHeadsRef.test_precision_argument, MtpHeadsRef.test_teacher_forced_draft_tokens
    mimo2_mtp_heads_parity: 2 failing: MtpHeads.setUpClass, MtpHeadsF16FA.setUpClass
caught 19/19
```

### 7.8 On `9a95535`: the N17 step again, the other dump-based steps, unit discovery, checker and hook tests, YAML

```
$ cat <S>/n17/confirm.sh
#!/usr/bin/env bash
# N17: the N17 CI step at both pins once more on the final code commit (after the docstring-only change of ref/fattn.py)
set -uo pipefail
S=<S>
WT=/home/user/wt-n17
cd $S/n17/cwd
echo "worktree HEAD $(git -C $WT rev-parse HEAD); status: $(git -C $WT status --short | wc -l) changed or untracked files"
for p in ec r1; do
  if [ $p = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  rm -rf $S/n17/confirm-$p
  t0=$(date +%s)
  MIMO2_N17_WORK=$S/n17/confirm-$p python3 $WT/tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --checkout $L -- \
    $WT/tools/tinygen/run_parity_mimo2_n17.sh $L $S/n17/build-$p > $S/n17/confirm-$p.log 2>&1
  echo "n17 $p wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
  tail -n 4 $S/n17/confirm-$p.log
  rm -rf $S/n17/confirm-$p
done
$ bash <S>/n17/confirm.sh > <S>/n17/confirm.out 2>&1; cat <S>/n17/confirm.out
worktree HEAD 9a95535d3f498d5c22812993e677c883a40a745f; status: 0 changed or untracked files
n17 ec wrapper_rc=0 wall_s=53

OK
artifacts: <S>/n17/confirm-ec
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n17 r1 wrapper_rc=0 wall_s=67

OK
artifacts: <S>/n17/confirm-r1
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
$ cd <S>/n17; for p in ec r1; do echo "== $p"; diff <(sed -e 's/in [0-9.]*s/in Xs/; s/confirm-/work-/g' confirm-$p.log) <(sed -e 's/in [0-9.]*s/in Xs/' final-wrapped-$p.log) && echo "identical apart from timings and the work directory"; done
== ec
identical apart from timings and the work directory
== r1
identical apart from timings and the work directory
```

```
$ cat <S>/n17/others.sh
#!/usr/bin/env bash
# N17: the other CI parity steps whose drivers use tools/dump/dump_tensors.cpp (changed by N17), at both pins, on the
# final code commit, through tools/ci/check_unittest.py; temporary work directories under <S>/n17/tmp (TMPDIR).
set -uo pipefail
S=<S>
WT=/home/user/wt-n17
cd $S/n17/cwd
echo "worktree HEAD $(git -C $WT rev-parse HEAD); status: $(git -C $WT status --short | wc -l) changed or untracked files"
for p in ec r1; do
  if [ $p = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  for st in g1-glm:run_parity.sh g1-mimo:run_parity_mimo.sh n2-qwen4exp:run_parity_qwen4exp.sh \
            n6-qwen4exp-decode:run_parity_qwen4exp_decode.sh n9-glm5next-decode:run_parity_glm5next_decode.sh \
            n10-quant:run_parity_quant.sh; do
    step=${st%%:*}; drv=${st##*:}
    rm -rf $S/n17/tmp; mkdir -p $S/n17/tmp
    t0=$(date +%s)
    TMPDIR=$S/n17/tmp GLM5NEXT_DECODE_WORK=$S/n17/tmp/w QUANT_WORK=$S/n17/tmp/w \
      python3 $WT/tools/ci/check_unittest.py --step $step --checkout $L -- $WT/tools/tinygen/$drv $L $S/n17/build-$p \
      > $S/n17/other-$step-$p.log 2>&1
    echo "$step $p wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
    tail -n 1 $S/n17/other-$step-$p.log
  done
done
rm -rf $S/n17/tmp
echo OTHERS-DONE
$ cd <S>/n17/cwd && bash <S>/n17/others.sh > <S>/n17/others.out 2>&1; cat <S>/n17/others.out
worktree HEAD 9a95535d3f498d5c22812993e677c883a40a745f; status: 0 changed or untracked files
g1-glm ec wrapper_rc=1 wall_s=2
::error::g1-glm @ llama.cpp ec7630a: the command exited with status 1
g1-mimo ec wrapper_rc=0 wall_s=1
::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n2-qwen4exp ec wrapper_rc=0 wall_s=2
::notice::n2-qwen4exp @ llama.cpp ec7630a: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n6-qwen4exp-decode ec wrapper_rc=0 wall_s=9
::notice::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n9-glm5next-decode ec wrapper_rc=0 wall_s=21
::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass); the driver's checkout <S>/llama-ec7630a is at ec7630a
n10-quant ec wrapper_rc=0 wall_s=190
::notice::n10-quant @ llama.cpp ec7630a: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
g1-glm r1 wrapper_rc=1 wall_s=2
::error::g1-glm @ llama.cpp 79e2e74: the command exited with status 1
g1-mimo r1 wrapper_rc=0 wall_s=2
::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n2-qwen4exp r1 wrapper_rc=0 wall_s=3
::notice::n2-qwen4exp @ llama.cpp 79e2e74: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n6-qwen4exp-decode r1 wrapper_rc=0 wall_s=29
::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n9-glm5next-decode r1 wrapper_rc=0 wall_s=33
::notice::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 58, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n10-quant r1 wrapper_rc=0 wall_s=138
::notice::n10-quant @ llama.cpp 79e2e74: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
OTHERS-DONE
# g1-glm failed above before running any test: its driver runs `python3 -m unittest tests.test_glm_tiny_parity` from
# the current directory (CI: the repository root), here a scratch directory. Again with the worktree on PYTHONPATH:
$ cat <S>/n17/g1.sh
#!/usr/bin/env bash
# N17: the g1-glm step at both pins on the final code commit. tools/tinygen/run_parity.sh runs
# `python3 -m unittest tests.test_glm_tiny_parity` relative to the current directory (CI runs it from the repository
# root); here the engine runs from a scratch directory, so the worktree goes on PYTHONPATH.
set -uo pipefail
S=<S>
WT=/home/user/wt-n17
cd $S/n17/cwd
echo "worktree HEAD $(git -C $WT rev-parse HEAD); status: $(git -C $WT status --short | wc -l) changed or untracked files"
for p in ec r1; do
  if [ $p = ec ]; then L=$S/llama-ec7630a; else L=$S/llama.cpp; fi
  rm -rf $S/n17/tmp; mkdir -p $S/n17/tmp
  t0=$(date +%s)
  TMPDIR=$S/n17/tmp PYTHONPATH=$WT python3 $WT/tools/ci/check_unittest.py --step g1-glm --checkout $L -- \
    $WT/tools/tinygen/run_parity.sh $L $S/n17/build-$p > $S/n17/other-g1-glm-$p.log 2>&1
  echo "g1-glm $p wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
  tail -n 1 $S/n17/other-g1-glm-$p.log
done
rm -rf $S/n17/tmp
$ bash <S>/n17/g1.sh > <S>/n17/g1.out 2>&1; cat <S>/n17/g1.out
worktree HEAD 9a95535d3f498d5c22812993e677c883a40a745f; status: 0 changed or untracked files
g1-glm ec wrapper_rc=0 wall_s=2
::notice::g1-glm @ llama.cpp ec7630a: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
g1-glm r1 wrapper_rc=0 wall_s=2
::notice::g1-glm @ llama.cpp 79e2e74: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
```

```
$ bash docs/evidence/mutation-scripts/2026-10-11-n17-unit.sh
HEAD 9a95535d3f498d5c22812993e677c883a40a745f
$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests
rc=0
----------------------------------------------------------------------
Ran 1026 tests in 130.213s

OK (skipped=491)
$ python3 -m unittest tests.test_check_unittest tests.test_hooks
rc=0
----------------------------------------------------------------------
Ran 54 tests in 14.622s

OK
$ python3 -c "import yaml; ..." (YAML parse of the changed workflows)
.github/workflows/ci.yml parsed; jobs ['cpu-parity', 'ref-torch-cpu', 'cuda-compile', 'strata-glm-sm89', 'lt-strata-sm89'] ; N17 steps [('cpu-parity', 'N17 mimo2 three MTP heads (nextn_layer_offset 0-2, drafts, rejections) and the R1 default precision (-ctk f16 -ctv f16 -fa on, sinks, K/V heads 24/16 and 192/128) against the real mimo2 graph (CPU)')]
.github/workflows/parity-r1-pin.yml parsed; jobs ['parity', 'ik-glm5next'] ; N17 steps [('parity', 'N17 mimo2 three MTP heads and the R1 default precision (-ctk f16 -ctv f16 -fa on) at the R1 pin (CPU)')]
status: 0 changed or untracked files
```

### 7.9 Generator before and after (N17 options), with the gguf-py of both pins

```
$ bash docs/evidence/mutation-scripts/2026-10-11-n17-genhash.sh /home/user/wt-n17 0b0eace <S>/n17/genhash <S>/llama-ec7630a/gguf-py <S>/llama.cpp/gguf-py
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  before-79e2e74-default.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  before-79e2e74-defaultfusedqkvsinksswavaluescale075mtp.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  before-79e2e74-defaultseed3.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  before-ec7630a-default.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  before-ec7630a-defaultfusedqkvsinksswavaluescale075mtp.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  before-ec7630a-defaultseed3.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  after-79e2e74-default.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  after-79e2e74-defaultfusedqkvsinksswavaluescale075mtp.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  after-79e2e74-defaultseed3.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  after-79e2e74-mtplayers1.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  after-ec7630a-default.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  after-ec7630a-defaultfusedqkvsinksswavaluescale075mtp.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  after-ec7630a-defaultseed3.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  after-ec7630a-mtplayers1.gguf
rc=0
```

### 7.10 CI wiring and the commits after the runs

```
$ grep -n 'n17' .github/workflows/ci.yml .github/workflows/parity-r1-pin.yml
.github/workflows/ci.yml:52:        run: python3 tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n17.sh ../llama.cpp ../build-tools
.github/workflows/parity-r1-pin.yml:19:# the R1 default precision of mimo2 (f16 caches, CPU flash attention with sinks), tools/tinygen/run_parity_mimo2_n17.sh.
.github/workflows/parity-r1-pin.yml:71:        run: python3 tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n17.sh ../llama.cpp ../build-tools
$ git diff --stat d5b3537 9a95535
 .../mutation-scripts/2026-10-11-n17-source-lines.sh     | 17 +++++++++++++++++
 ref/fattn.py                                            |  4 ++--
 2 files changed, 19 insertions(+), 2 deletions(-)
$ git diff d5b3537 9a95535 -- ref/
diff --git a/ref/fattn.py b/ref/fattn.py
index d361ff6..40bd744 100644
--- a/ref/fattn.py
+++ b/ref/fattn.py
@@ -66,8 +66,8 @@ an f32 difference (O:8997, 9003). N17 changed the second for every caller: _onli
 difference before, which put up to 3 binary16 spacings between it and llama.cpp on mimo2's 240-cell rows (in N10's
 qwen4exp runs the one element per run that was past half a spacing is not any more; the N10 step passes as before;
 docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md). Left at the f32 level: the running sum S is computed with two
-roundings here; llama.cpp's matches an fma in 99.6 % of the mimo2 elements (PROVISIONAL reading of the same evidence),
-which changes no f16 rounding.
+roundings here; with an fma in S instead, 98.8 % to 99.7 % of the mimo2 one_chunk elements per run are bit-equal to
+llama.cpp's (93.1 % to 93.8 % as written; PROVISIONAL reading of the same evidence), which changes no f16 rounding.
 """
 import numpy as np
 
```

### 7.11 Source lines

Every range cited above, printed from the two checkouts, and the diffs behind "identical at both pins".

```
$ bash docs/evidence/mutation-scripts/2026-10-11-n17-source-lines.sh <S>/llama-ec7630a <S>/llama.cpp
--- ec7630a src/models/mimo2.cpp:276,279
const int il = hparams.n_layer() + cparams.nextn_layer_offset;
GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
"nextn_layer_offset out of range [0, n_layer_nextn)");
--- 79e2e74 src/models/mimo2.cpp:267,270
const int il = hparams.n_layer() + cparams.nextn_layer_offset;
GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
"nextn_layer_offset out of range [0, n_layer_nextn)");
--- ec7630a src/models/mimo2.cpp:179,179
Qcur, Kcur, Vcur, nullptr, sinks, nullptr, 1.0f/sqrtf(float(n_embd_head_k)), il);
--- 79e2e74 src/models/mimo2.cpp:171,171
Qcur, Kcur, Vcur, nullptr, sinks, nullptr, 1.0f/sqrtf(float(n_embd_head_k)), il);
--- ec7630a src/llama-ext.h:101,101
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);
--- 79e2e74 src/llama-ext.h:101,101
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);
--- ec7630a src/llama-context.cpp:1248,1250
void llama_context::set_nextn_layer_offset(int32_t offset) {
cparams.nextn_layer_offset = offset;
}
--- 79e2e74 src/llama-context.cpp:1264,1266
void llama_context::set_nextn_layer_offset(int32_t offset) {
cparams.nextn_layer_offset = offset;
}
--- ec7630a src/llama-graph.h:873,875
if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {
return false;
}
--- 79e2e74 src/llama-graph.h:884,886
if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {
return false;
}
--- ec7630a src/llama-model.cpp:2807,2812
if (hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0) {
if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
} else {
filter = [&](uint32_t il) { return il <  hparams.n_layer(); };
}
--- 79e2e74 src/llama-model.cpp:2846,2851
if (hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0) {
if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
} else {
filter = [&](uint32_t il) { return il <  hparams.n_layer(); };
}
--- ec7630a common/speculative.cpp:1378,1378
n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));
--- ec7630a common/speculative.cpp:1420,1425
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

if (chain_heads) {
this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

--- ec7630a common/speculative.cpp:1519,1529
for (int k = 0; k < n_tokens; ++k) {
const llama_seq_id seq_id = batch_in.tokens[k].seq_id;

const int32_t idx = batch.add(batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false);

const float * h_row = k == i_batch_beg[seq_id]
? pending_h[seq_id].data()
: h_tgt + (size_t) (k - 1) * n_embd;

batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
}
--- ec7630a common/speculative.cpp:1533,1557
bool ok = true;
for (int head = 0; head < n_mtp_layers; ++head) {
if (chain_heads) {
// ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
if (i_batch_beg[seq_id] < 0) {
continue;
}
llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
}
llama_set_nextn_layer_offset(ctx_dft, head);
}

const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
if (rc != 0) {
SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
head, (int) rc, (int) batch_in.tokens[0].pos[0]);
ok = false;
break;
}
}

if (chain_heads) {
llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
}
--- ec7630a common/speculative.cpp:1604,1611
const int32_t idx = batch.add(dp.id_last, dp.pos0, seq_id, true);
batch.set_embd(idx, { pending_h[seq_id].data(), 1, (size_t) n_embd });

i_last[seq_id] = idx;

if (chain_heads) {
chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
}
--- ec7630a common/speculative.cpp:1623,1631
if (chain_heads) {
auto * mem_dft = llama_get_memory(ctx_dft);
for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
if (drafting[seq_id]) {
llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
}
}
llama_set_nextn_layer_offset(ctx_dft, i);
}
--- ec7630a common/speculative.cpp:1686,1696
if (chain_heads) {
// ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
for (int t = 0; t < n_rows; ++t) {
const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
const int32_t idx = batch.add(tok, dp.pos0 + t, seq_id, t == n_rows - 1);
batch.set_embd(idx, { chain_h[seq_id].data() + (size_t) t * n_embd, 1, (size_t) n_embd });
i_last[seq_id] = idx;
}
--- ec7630a common/speculative.cpp:1743,1745
const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
const size_t row_bytes = (size_t) n_embd * sizeof(float);
std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
--- 79e2e74 common/speculative.cpp:1439,1439
n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));
--- 79e2e74 common/speculative.cpp:1481,1486
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

if (chain_heads) {
this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

--- 79e2e74 common/speculative.cpp:1583,1593
for (int k = 0; k < n_tokens; ++k) {
const llama_seq_id seq_id = batch_in.tokens[k].seq_id;

const int32_t idx = batch.add(batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false);

const float * h_row = k == i_batch_beg[seq_id]
? pending_h[seq_id].data()
: h_tgt + (size_t) (k - 1) * n_embd;

batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
}
--- 79e2e74 common/speculative.cpp:1597,1621
bool ok = true;
for (int head = 0; head < n_mtp_layers; ++head) {
if (chain_heads) {
// ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
if (i_batch_beg[seq_id] < 0) {
continue;
}
llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
}
llama_set_nextn_layer_offset(ctx_dft, head);
}

const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
if (rc != 0) {
SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
head, (int) rc, (int) batch_in.tokens[0].pos[0]);
ok = false;
break;
}
}

if (chain_heads) {
llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
}
--- 79e2e74 common/speculative.cpp:1681,1688
const int32_t idx = batch.add(dp.id_last, dp.pos0, seq_id, true);
batch.set_embd(idx, { pending_h[seq_id].data(), 1, (size_t) n_embd });

i_last[seq_id] = idx;

if (chain_heads) {
chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
}
--- 79e2e74 common/speculative.cpp:1700,1708
if (chain_heads) {
auto * mem_dft = llama_get_memory(ctx_dft);
for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
if (drafting[seq_id]) {
llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
}
}
llama_set_nextn_layer_offset(ctx_dft, i);
}
--- 79e2e74 common/speculative.cpp:1767,1777
if (chain_heads) {
// ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
for (int t = 0; t < n_rows; ++t) {
const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
const int32_t idx = batch.add(tok, dp.pos0 + t, seq_id, t == n_rows - 1);
batch.set_embd(idx, { chain_h[seq_id].data() + (size_t) t * n_embd, 1, (size_t) n_embd });
i_last[seq_id] = idx;
}
--- 79e2e74 common/speculative.cpp:1824,1826
const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
const size_t row_bytes = (size_t) n_embd * sizeof(float);
std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
--- ec7630a common/common.h:332,332
float p_min   = 0.0f; // minimum speculative decoding probability (greedy)
--- 79e2e74 common/common.h:333,333
float p_min   = 0.0f; // minimum speculative decoding probability (greedy)
--- ec7630a tools/server/server-context.cpp:3044,3044
common_speculative_draft(spec.get());
--- ec7630a tools/server/server-context.cpp:3747,3747
ok = common_speculative_process(spec.get(), batch.view);
--- 79e2e74 tools/server/server-context.cpp:3404,3404
common_speculative_draft(spec.get());
--- 79e2e74 tools/server/server-context.cpp:4232,4232
ok = common_speculative_process(spec.get(), batch.view);
--- ec7630a tools/server/server-context.cpp:3063,3063
if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
--- ec7630a tools/server/server-context.cpp:3964,3964
common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
--- ec7630a tools/server/server-context.cpp:3998,3998
slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
--- 79e2e74 tools/server/server-context.cpp:3423,3423
if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
--- 79e2e74 tools/server/server-context.cpp:4471,4471
common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
--- 79e2e74 tools/server/server-context.cpp:4505,4505
slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
--- ec7630a common/common.cpp:1558,1563
void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
if (ctx_dft) {
common_context_seq_rm(ctx_dft, seq_id, p0, p1);
}
}
--- ec7630a common/common.cpp:1502,1527
common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

llama_memory_clear(mem, true);

// eval 2 tokens to check if the context is compatible
std::vector<llama_token> tmp;
tmp.push_back(0);
tmp.push_back(0);

int ret;
{
common_batch batch = common_batch_get_one(ctx, tmp);
ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
}
if (ret != 0) {
COM_ERR("llama_process() failed: %d\n", ret);
res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
goto done;
}

// try to remove the last tokens
if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
COM_TRC("%s", "the context does not support partial sequence removal\n");
res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
goto done;
}
--- ec7630a src/llama-kv-cache.cpp:410,419
if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
if (new_head == cells.size()) {
new_head = i;
}
}
}

// If we freed up a slot, set head to it so searching can start there.
if (new_head != cells.size() && new_head < head) {
head = new_head;
--- 79e2e74 src/llama-kv-cache.cpp:428,437
if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
if (new_head == cells.size()) {
new_head = i;
}
}
}

// If we freed up a slot, set head to it so searching can start there.
if (new_head != cells.size() && new_head < head) {
head = new_head;
--- ec7630a src/llama-graph.cpp:38,43
// flash attention requires an f16 mask
const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
ggml_set_input(res);
ggml_set_name(res, "attn_inp_kq_mask");
--- 79e2e74 src/llama-graph.cpp:39,44
// flash attention requires an f16 mask
const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
ggml_set_input(res);
ggml_set_name(res, "attn_inp_kq_mask");
--- ec7630a src/llama-kv-cache.cpp:1581,1582
const T mask_keep = llama_cast<T>(0.0f);
const T mask_drop = llama_cast<T>(-INFINITY);
--- ec7630a src/llama-kv-cache.cpp:1694,1711
// apply SWA if any
if (swa) {
// see llama_non_causal_type
const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
goto skip;
}
}

if (alibi) {
data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
} else {
data[idst + j] = mask_keep;
}

continue;
skip:
data[idst + j] = mask_drop;
--- ec7630a src/llama-kv-cache.cpp:1789,1793
if (dst->type == GGML_TYPE_F16) {
set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
} else {
set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
}
--- 79e2e74 src/llama-kv-cache.cpp:1792,1796
if (dst->type == GGML_TYPE_F16) {
set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
} else {
set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
}
--- ec7630a src/llama-graph.cpp:2651,2675
const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
if (use_flash_attn) {
GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

if (v_trans) {
v = ggml_transpose(ctx0, v);
}

// this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
if (k->type == GGML_TYPE_F32) {
k = ggml_cast(ctx0, k, GGML_TYPE_F16);
}

if (v->type == GGML_TYPE_F32) {
v = ggml_cast(ctx0, v, GGML_TYPE_F16);
}

cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

ggml_flash_attn_ext_add_sinks(cur, sinks);
GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
ggml_prec_set_acc(cur, GGML_PREC_F32);
--- ec7630a ggml/src/ggml.c:5579,5593
void ggml_flash_attn_ext_add_sinks(
struct ggml_tensor * a,
struct ggml_tensor * sinks) {
if (!sinks) {
a->src[4] = NULL;
return;
}

GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
GGML_ASSERT(a->src[4] == NULL);
GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
GGML_ASSERT(sinks->type == GGML_TYPE_F32);

a->src[4] = sinks;
}
--- ec7630a ggml/src/ggml.c:4218,4232
void ggml_soft_max_add_sinks(
struct ggml_tensor * a,
struct ggml_tensor * sinks) {
if (!sinks) {
a->src[2] = NULL;
return;
}

GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
GGML_ASSERT(a->src[2] == NULL);
GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
GGML_ASSERT(sinks->type == GGML_TYPE_F32);

a->src[2] = sinks;
}
--- ec7630a ggml/src/ggml-cpu/ops.cpp:8917,8918
ggml_type         const k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
--- ec7630a ggml/src/ggml-cpu/ops.cpp:8961,8961
q_to_vec_dot(pq, Q_q, DK);
--- ec7630a ggml/src/ggml-cpu/ops.cpp:8976,8978
kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

s = s*scale; // scale KQ value
--- ec7630a ggml/src/ggml-cpu/ops.cpp:8993,9007
if (v->type == GGML_TYPE_F16) {
if (s > M) {
// s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
M = s;
ms = expf(Mold - M);

// V = V*expf(Mold - M)
ggml_vec_scale_f16(DV, VKQ16, ms);
} else {
// no new maximum, ms == 1.0f, vs != 1.0f
vs = expf(s - M);
}

// V += v*expf(s - M)
ggml_vec_mad_f16(DV, VKQ16, (const ggml_fp16_t *) v_data, vs);
--- ec7630a ggml/src/ggml-cpu/ops.cpp:9031,9056
S = S*ms + vs; // scale and increment sum with partial sum
}

if (v->type == GGML_TYPE_F16) {
for (int64_t d = 0; d < DV; ++d) {
VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
}
}

// sinks - apply only on the first kv-chunk
if (sinks && ic_start == 0) {
const float s = ((float *)((char *) sinks->data))[h];

float ms = 1.0f;
float vs = 1.0f;

if (s > M) {
ms = expf(M - s);
M = s;
ggml_vec_scale_f32(DV, VKQ32, ms);
} else {
vs = expf(s - M);
}

S = S*ms + vs;
}
--- ec7630a ggml/src/ggml-cpu/ops.cpp:9066,9068
// V /= S
const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
ggml_vec_scale_f32(DV, VKQ32, S_inv);
--- ec7630a ggml/src/ggml-cpu/ops.cpp:9332,9354
// sinks (apply only to valid rows in the tile)
if (sinks) {
const float s = ((float *)((char *) sinks->data))[h];

for (int tq = 0; tq < tile_rows; tq++) {
float ms = 1.0f;
float vs = 1.0f;

if (s > M[tq]) {
ms = expf(M[tq] - s);
ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
} else {
vs = expf(s - M[tq]);
}

S[tq] = S[tq] * ms + vs;
}
}

for (int tq = 0; tq < tile_rows; tq++) {
// V /= S
const float S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);
--- ec7630a ggml/src/ggml-cpu/ops.cpp:9490,9520
const bool use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >=

if (use_split_kv_path) {
const int64_t chunk_size = (nek1 + nth - 1) / nth;

// Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
const int64_t partial_size  = 2 + DV;
float *       partials_base = (float *) params->wdata + nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);

const int64_t ic_start = ith * chunk_size;
const int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

const int64_t partial_stride = nth * partial_size;
float *       chunk_partials = partials_base + ith * partial_size;

if (ic_start < nek1) {
for (int64_t q_head = 0; q_head < neq2; q_head++) {
ggml_compute_forward_flash_attn_ext_f16_one_chunk(
params, dst, q_head, q_head + 1, ic_start, ic_end,
chunk_partials, partial_stride);
}
} else {
for (int64_t q_head = 0; q_head < neq2; q_head++) {
float * q_partials = chunk_partials + q_head * partial_stride;
q_partials[0] = -INFINITY;  // M
q_partials[1] = 0.0f;       // S
}
}

ggml_barrier(params->threadpool);
ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
--- ec7630a ggml/src/ggml-cpu/ops.cpp:9420,9435
const float M_new     = fmaxf(M_final, M_chunk);
const float scale_old = expf(M_final - M_new);
const float scale_new = expf(M_chunk - M_new);

for (int64_t d = 0; d < DV; ++d) {
VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
}
S_final = S_final * scale_old + S_chunk * scale_new;
M_final = M_new;
}

// Normalize and write to output
if (S_final != 0.0f) {
const float S_inv = 1.0f / S_final;
ggml_vec_scale_f32(DV, VKQ_final, S_inv);
}
--- ec7630a ggml/src/ggml-cpu/ops.cpp:9546,9551
static constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
bool use_tiled = !use_ref &&
(q->type == GGML_TYPE_F32 &&
kv_is_f32_or_f16 &&
k->type == v->type &&
neq1 >= Q_TILE_SZ);
--- ec7630a ggml/src/ggml-cpu/ops.cpp:5896,5912
float max = -INFINITY;
ggml_vec_max_f32(ne00, &max, wp);

// if we have sinks, make a correction as if they were included in the softmax
if (sk) {
max = MAX(max, sk[i02]);
}

ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
assert(sum > 0.0);

if (sk) {
sum += (ggml_float) expf(sk[i02] - max);
}

sum = 1.0/sum;
ggml_vec_scale_f32(ne00, dp, sum);
--- 79e2e74 ggml/src/ggml-cpu/ops.cpp:9039,9055
// sinks - apply only on the first kv-chunk
if (sinks && ic_start == 0) {
const float s = ((float *)((char *) sinks->data))[h];

float ms = 1.0f;
float vs = 1.0f;

if (s > M) {
ms = expf(M - s);
M = s;
ggml_vec_scale_f32(DV, VKQ32, ms);
} else {
vs = expf(s - M);
}

S = S*ms + vs;
}
--- ec7630a ggml/src/ggml-cpu/vec.cpp:345,377
const int np = (n & ~(GGML_F16_DOT_STEP - 1));

GGML_F16_DOT_VEC sum[GGML_F16_DOT_ARR] = { GGML_F16_DOT_VEC_ZERO };

GGML_F16_DOT_VEC ax[GGML_F16_DOT_ARR];
GGML_F16_DOT_VEC ay[GGML_F16_DOT_ARR];

for (int i = 0; i < np; i += GGML_F16_DOT_STEP) {
for (int j = 0; j < GGML_F16_DOT_ARR; j++) {
ax[j] = GGML_F16_DOT_VEC_LOAD(x + i + j*GGML_F16_DOT_EPR, j);
ay[j] = GGML_F16_DOT_VEC_LOAD(y + i + j*GGML_F16_DOT_EPR, j);

sum[j] = GGML_F16_DOT_VEC_FMA(sum[j], ax[j], ay[j]);
}
}

// reduce sum0..sum3 to sum0
GGML_F16_DOT_VEC_REDUCE(sumf, sum);

// leftovers
for (int i = np; i < n; ++i) {
sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
}
// if you hit this, you are likely running outside the FP range
assert(!isnan(sumf) && !isinf(sumf));
#endif
#else
for (int i = 0; i < n; ++i) {
sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
}
#endif // GGML_SIMD

*s = sumf;
--- ec7630a ggml/src/ggml-cpu/vec.h:556,581
#elif defined(GGML_SIMD)
const int np = (n & ~(GGML_F16_STEP - 1));

GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

GGML_F16_VEC ax[GGML_F16_ARR];
GGML_F16_VEC ay[GGML_F16_ARR];

for (int i = 0; i < np; i += GGML_F16_STEP) {
for (int j = 0; j < GGML_F16_ARR; j++) {
ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
}
}
#else
// scalar path
const int np = 0;
#endif

// scalar and leftovers
for (int i = np; i < n; ++i) {
y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
}
--- ec7630a ggml/src/ggml-cpu/vec.h:833,855
#elif defined(GGML_SIMD)
const int np = (n & ~(GGML_F16_STEP - 1));

GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

GGML_F16_VEC ay[GGML_F16_ARR];

for (int i = 0; i < np; i += GGML_F16_STEP) {
for (int j = 0; j < GGML_F16_ARR; j++) {
ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
}
}
#else
// scalar path
const int np = 0;
#endif
// scalar and leftovers
for (int i = np; i < n; ++i) {
y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
}
--- ec7630a ggml/src/ggml-cpu/simd-mappings.h:595,620
#if defined(__FMA__)
#define GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
#define GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define GGML_F32x8_ADD     _mm256_add_ps
#define GGML_F32x8_MUL     _mm256_mul_ps
#define GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
int offset = GGML_F32_ARR >> 1;                               \
for (int i = 0; i < offset; ++i) {                            \
x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
}                                                             \
offset >>= 1;                                                 \
for (int i = 0; i < offset; ++i) {                            \
x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
}                                                             \
offset >>= 1;                                                 \
for (int i = 0; i < offset; ++i) {                            \
x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
}                                                             \
const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
_mm256_extractf128_ps(x[0], 1)); \
const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} while (0)
--- ec7630a ggml/src/ggml-cpu/simd-mappings.h:635,647
#define GGML_F16_STEP 32
#define GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define GGML_F32Cx8             __m256
#define GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#define GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
--- ec7630a ggml/src/ggml-cpu/simd-mappings.h:670,680
#define GGML_F32Cx8_FMA         GGML_F32x8_FMA
#define GGML_F32Cx8_ADD         _mm256_add_ps
#define GGML_F32Cx8_MUL         _mm256_mul_ps
#define GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#define GGML_F16_VEC                GGML_F32Cx8
#define GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#define GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
--- ec7630a ggml/src/ggml-cpu/simd-mappings.h:1333,1340
#define GGML_F16_DOT_STEP           GGML_F16_STEP
#define GGML_F16_DOT_EPR            GGML_F16_EPR
#define GGML_F16_DOT_ARR            GGML_F16_ARR
#define GGML_F16_DOT_VEC            GGML_F16_VEC
#define GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#define GGML_F16_DOT_VEC_LOAD       GGML_F16_VEC_LOAD
#define GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#define GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE
--- 79e2e74 conversion/mimo.py:22,24
# MiMo V2-Flash, V2.5 and V2.5-Pro all ship 3 trained MTP layers under model.mtp.layers.{0,1,2}.
# The HF config does not expose the count, so it's hardcoded to match the count found in the safetensors.
_n_nextn = 3
--- 79e2e74 conversion/mimo.py:290,298
m = re.match(r"^model\.mtp\.layers\.(\d+)\.(.*)$", name)
if m is not None:
mtp_idx = int(m.group(1))
assert mtp_idx < self._n_nextn, f"MTP layer index {mtp_idx} >= _n_nextn ({self._n_nextn})"
rest = m.group(2)
n_layer_text = self.hparams["num_hidden_layers"]
new_bid = n_layer_text + mtp_idx
name = f"model.layers.{new_bid}.{rest}"
bid = new_bid
--- 79e2e74 src/models/mimo2.cpp:378,378
cur = ggml_get_rows(ctx0, cur, inp_out_ids);
--- ec7630a src/llama-kv-cache-iswa.cpp:73,73
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
--- 79e2e74 src/llama-kv-cache-iswa.cpp:73,73
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
--- diff ggml/src/ggml-cpu/ops.cpp:8843,9592 @ec7630a ggml/src/ggml-cpu/ops.cpp:8842,9591 @79e2e74
identical (750 lines)
--- diff src/llama-kv-cache.cpp:1566,1716 @ec7630a src/llama-kv-cache.cpp:1569,1719 @79e2e74
identical (151 lines)
--- diff common/speculative.cpp:1340,1747 @ec7630a common/speculative.cpp:1401,1828 @79e2e74
118a119,121
>         // reset here rather than per round, or two identical requests differ
>         common_sampler_reset(smpls[seq_id].get());
> 
263c266,279
<             common_sampler_reset(smpls[seq_id].get());
---
>             // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
>             if (!params.probabilistic) {
>                 dp.result_q = nullptr;
>             }
> 
>             // result_q is only set when the caller wants rejection, so it also gates the retune
>             if (dp.result_q) {
>                 spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
>             }
> 
>             // a reset reseeds the chain, which breaks probabilistic drafting
>             if (!dp.result_q) {
>                 common_sampler_reset(smpls[seq_id].get());
>             }
312c328
<                 common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
---
>                 const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
324c340
<                 const llama_token id = cur_p->data[0].id;
---
>                 const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;
339a356,359
> 
>                 if (dp.result_q) {
>                     dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
>                 }
```
