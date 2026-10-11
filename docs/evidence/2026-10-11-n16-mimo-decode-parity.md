# N16: mimo2 (MiMo-V2.x-Flash) beyond the single G1 prefill: several ubatches, decode past the sliding-window cache, rollbacks, MTP and MXFP4 experts, CPU parity, 2026-10-11 (sandbox CPU, no GPU)

Task N16 of the hardware-free track: close gap 12 of docs/reassessment/CORRECTNESS.md (MiMo had oracle parity for the
attention and the router of one prefill only, `docs/evidence/2026-10-02-mimo-g1-parity.md`). Same method as N9 for
glm5-next (`docs/evidence/2026-10-10-glm5next-decode-parity.md`) and N6 for qwen4exp; the MXFP4 part follows N10 part B
(`docs/evidence/2026-10-10-quant-fa-parity.md`). Branch `claude/n16-mimo-decode-parity`, cut from
`claude/n14-chat-templates` at `ac419b0`. Role: parity-qa (CI wiring commit as infra-ops, the vendoring request as
model-analyst). `<S>` below stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution.

References: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) and
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin, tools/engines/pins.json), CPU builds of `tools/g0` targets
`dump-tensors` and `llama-quantize` (`-DG0_QUANTIZE=ON`, `GGML_NATIVE=OFF`: AVX2 on, AVX-512 off), fresh build
directories `<S>/n16/build-ec`, `<S>/n16/build-r1` configured from this worktree. "M" = src/models/mimo2.cpp, "KV" =
src/llama-kv-cache.cpp, "KVI" = src/llama-kv-cache-iswa.cpp at ec7630a unless "@79e2e74" is written; every cited line
was printed from the pinned checkout (section "Source lines").

| piece | file |
|---|---|
| whole-model oracle | ref/mimo2.py: `forward()` (from ref/mimo.py and ref/router.py, both unchanged), the incremental path `forward_chunk` / `cache_truncate` / `forward_calls` over a cell model of llama.cpp's iSWA KV caches (`Cells`: find_slot, apply_ubatch with its purge, seq_rm, n_kv), the MTP block `mtp_forward_chunk`, `hparams()` from GGUF fields |
| MXFP4 | ref/quant.py, appended (nothing existing changed): `e8m0_to_fp32_half`, `mxfp4_split` / `mxfp4_codes` / `mxfp4_dequant` (dequantize_row_mxfp4), `mxfp4_q8_0_dot` (the block form of ggml_vec_dot_mxfp4_q8_0); the products use the existing `QWeight` with vec_dot_type Q8_0 |
| generator | tools/tinygen/mimo2_tiny.py `--fused-qkv`, `--sinks swa`, `--value-scale`, `--mtp` (default output byte-identical, below) |
| driver | tools/tinygen/run_parity_mimo2_decode.sh (builds both targets, writes the models, runs 8 f32 runs + the MTP context + 4 MXFP4 runs, runs the five test modules; `PARITY_LLAMA_COMMIT` from the checkout; `MIMO2_DECODE_WORK` keeps the artifacts) |
| dump parity tests | tests/test_mimo2_decode_parity.py (6 tests per run, 8 runs, + 1 for the evict run = 49), tests/test_mimo2_mtp_parity.py (4), tests/test_mimo2_quant_parity.py (7 per run, 4 runs = 28); all skip without `MIMO2_DECODE_DUMP` and `LLAMA_CPP_GGUF_PY` |
| property tests | tests/test_mimo2_chunked_ref.py (7, numpy only), tests/test_mxfp4_ref.py (4; one compares with the pinned gguf-py and skips without it); both run in the CI `unit tests` step as well |
| CI | step `n16-mimo2-decode` in `.github/workflows/ci.yml` job `cpu-parity` (ec7630a) and in `.github/workflows/parity-r1-pin.yml` job `parity` (R1 pin), both through `tools/ci/check_unittest.py`; `tools/ci/parity_expectations.json`: Ran 92, no skips, at both commits |
| mutation check, probes, run scripts | docs/evidence/mutation-scripts/2026-10-11-n16-mimo2-mutations.py (19 mutants), -bites.sh, -genhash.sh, -evict-probe.py, -pindiff.py, -repack-diff.py, -build.sh, -final-runs.sh, -unit.sh |
| real config | third_party/hf/mimo-v2.6-flash-chat/REQUEST.json (vendored by `vendor-hf-branch` when the branch is pushed; not used here) |

## What the pins have (CONFIRMED from the source unless labelled)

* **The MTP graph exists at both pins.** `build_arch_graph` returns `graph_mtp` for `LLM_GRAPH_TYPE_DECODER_MTP`
  (M:86-91), `graph_mtp` is M:272-407; at 79e2e74 the file differs only in the loader's NextN flags (`nextn_flags`) and
  in naming the final hidden state (diff hunks below). The block: e = rms_norm(tok_embd[token]) * enorm, h =
  rms_norm(h_prev) * hnorm, eh_proj [e ; h], one attention layer (fused `attn_qkv` required, M:285), value scale, a dense
  FFN (required, M:375), the head norm from shared_head_norm, else layer_out_norm, else output_norm (M:392-394), the LM
  head. h_prev is the target's **pre-norm** last hidden state (M:244-246; GLM's h_nextn is post-norm, N9 finding 5).
  The MTP layer is a sliding-window layer: the converter appends `[1] * n_nextn` to the SWA pattern and gives it the
  SWA KV-head count (conversion/mimo.py:155-160, same file at both pins), and the MTP context's memory holds that layer
  alone (`src/llama-model.cpp:2808-2811`).
* **The trunk names its final hidden state h_nextn at 79e2e74 in every graph** (M:236-238 @79e2e74), only with
  embeddings_nextn at ec7630a (M:244-246): the last layer's `l_out` is renamed in the dumps. The tests follow it.
* **KV caches**: an iSWA pair, the SWA cache `GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256)` cells (KVI:73), the base
  cache `n_ctx` cells with no window (KVI:95-98). A ubatch takes empty cells and cells whose position is masked by the
  window relative to the cache's `seq_pos_max + 1` (KV:1057-1058), walking from the head; `head = last cell + 1`
  (KV:1184); n_kv = min(size, max(256, pad(used_max_p1, 256))) (KV:1265-1270); buffers start at zero (KV:293); the
  mask hides empty cells, future positions and, on SWA layers, `p1 - p0 >= n_swa` (KV:1656-1700).
  `llama_memory_seq_rm` frees every cell of a position >= p in both caches and returns true (KVI:116-117, KV:410-419).
* **mimo2 never reads `expert_weights_scale`** (M:3-23 reads no such key; the hparams default 0 makes build_moe_ffn skip
  the scale, src/llama-graph.cpp:2171): G1 finding 4's UNKNOWN about the real GGUF's key does not matter for llama.cpp.
  The value scale (`attention.value_scale`, read only if != 1, M:14-17) multiplies attn_out after W_o (M:182-185).
* **MXFP4 on the CPU** (both pins; every cited file identical or changed only outside the cited lines, diff below):
  block_mxfp4 = 1 E8M0 byte + 16 bytes of nibbles for 32 weights, weight j in the low nibble of qs[j], j + 16 in the
  high nibble (ggml/src/ggml-common.h:214-219, ggml/src/ggml-quants.c:577-584); values `kvalues_mxfp4` = e2m1 doubled
  (ggml-common.h:1126-1128) times `GGML_E8M0_TO_FP32_HALF(e)` = 2^(e-128) (ggml/src/ggml-impl.h:489-501; the CPU
  backend reads a table filled with it, simd-mappings.h:129-131). vec_dot_type Q8_0 (ggml-cpu.c:287-292): the
  activations go through the AVX2 quantize_row_q8_0, then integer block sums times f16(d_x) * E8M0_HALF(e)
  (arch/x86/quants.c:956-960 pairs of blocks, 991-1000 the scalar tail). On AVX2, MXFP4 weights with ne[1] % 8 == 0 are
  repacked to mxfp4_8x8 (repack.cpp:5094-5098); `mul_mat_id` converts every activation row with from_float
  (repack.cpp:4839-4841). `llama-quantize ... MXFP4_MOE`: 3-D tensors with ne[2] > 1 to MXFP4, every other matrix to
  Q8_0 (src/llama-quant.cpp:512-523). So MXFP4 is usable for mimo2 on the CPU at both pins.

## Runs

Models (tools/tinygen/mimo2_tiny.py, seed 0, the G1 hyper-parameters, CONFIRMED at tools/tinygen/mimo2_tiny.py:32-35):
6 layers, global layers 0 and 5 (1 KV head), sliding-window layers 1-4 (2 KV heads, window 8), 4 query heads, K dim 24,
V dim 16, NEOX RoPE on 8 dims (base 10000, 5000 on SWA layers), a dense layer 0, MoE 8 experts top-2 elsewhere.
`split.gguf` = the G1 model (split attn_q/k/v, sinks on all layers, no value scale); `n16.gguf` = `--fused-qkv --sinks
swa --value-scale 0.75 --mtp` (the trunk weights are the same draws; fused qkv rows = [q; k; v]; MTP block = layer 6).
Flags as G1: `-c 512 -b 512 -t 1 -fa off -ctk f32 -ctv f32`. One 300-token sequence (`random.seed(16)`, ids in
[3, 96)). With `-ub` <= 248 the SWA cache has 256 cells, so the runs past position 256 wrap it.

| run | model | `-ub` | calls (`DUMP_CALLS`) | graphs | what it adds |
|---|---|---|---|---|---|
| ub64 | n16 | 64 | one call of 300 | 64 x 4 + 44 | the 5th ubatch (256..299) goes to SWA cells 0..43 |
| ub13 | n16 | 13 | one call of 300 | 23 x 13 + 1 | ubatches longer than the window; the ubatch 247..259 straddles the end of the SWA cache (cells 247..255, 0..3, asserted) |
| ub5 | n16 | 5 | one call of 300 | 60 x 5 | the window (8) spans two or three ubatches |
| decode | n16 | 64 | `240,1 x 60` | 4 + 60 | one-token decode past position 256 (the ring wraps during the decode) |
| rollback | n16 | 64 | `240,1 x 16,4,r2,1 x 6,12,r10,1 x 34` | 62 | a draft 256..259 (2 wrong) rolled back by 2 < n_swa; a draft 264..275 (10 wrong) rolled back by 10 > n_swa; the re-decoded 258, 259 reuse cells 2, 3 (asserted) |
| evict | n16 | 64 | `280,r260,24,1 x 4` | 5 + 1 + 4 | a rollback deeper than the SWA cache retains (finding 4) |
| split | split | 64 | `240,1 x 20` | 4 + 20 | the split-qkv path, sinks on the global layers, no value scale |
| mtp | n16 | 64 | target `240,1 x 24`; MTP context: the same 25 calls + 4 draft steps | 28 + 32 | the MTP block driven like common/speculative.cpp (`DUMP_MTP=1 DUMP_MTP_DRAFT=4`), its own 256-cell SWA cache wrapping at 256 |
| mxfp4/pf64, decode, pf64-nr, decode-nr | n16 as MXFP4_MOE | 512 | `64`; `48,1 x 16` | 1; 17 | MXFP4 experts on the repacked AVX2 path and (`--no-repack`) on ggml_vec_dot_mxfp4_q8_0 |

## Method

As N6/N9. `DUMP_PER_GRAPH=1` writes every graph (one per ubatch) into `g<NNN>/`; `manifest.json` maps calls to graphs
and positions. Every graph is compared row by row with (A) `[full]`: `M.forward(final tokens)` at the graph's positions
(prefill semantics: all keys of the sequence available); for the MTP context `M.mtp_forward_chunk` over all processed
positions at once, fed by the full target oracle's h; rejected draft rows and MTP draft steps have no (A); the evict run
has none (finding 4); and (B) `[chunk]`: `M.forward_chunk` / `M.mtp_forward_chunk` replaying the same calls with the cell
model (rollback = `M.cache_truncate`, which frees cells exactly as seq_rm does and restores nothing).

The KV cache is checked through two tensors of every layer and graph: `kq` (the raw K . q products over all n_kv cells
the graph reads, [n_kv, n_tokens, n_head]) must equal (B) on **every** cell, the empty (zero K), stale (freed but not
overwritten) and masked ones too, so every cell's content and position is checked; `kq_soft_max` must be exactly 0 on
every cell the mask hides and > 0 on the visible ones, equal to (B), and to (A) placed at the cells (B) says hold
those positions. n_kv is asserted per graph. Components otherwise: `embd` (exact), `attn_norm`, `wqkv` (fused) or the
split projections, `Qcur` / `Kcur` after RoPE, `Vcur`, `kqv_out`, `attn_out`, `attn_out_scaled`, `ffn_inp`, `ffn_norm`,
the dense `ffn_up` / `ffn_gate` / `ffn_swiglu` / `ffn_out` or the MoE `ffn_moe_logits`, `ffn_moe_topk` (sets),
`ffn_moe_weights_norm`, `ffn_moe_out`, `l_out` (or `h_nextn`), `result_norm`, `result_output`, logits.bin, greedy argmax;
MTP: the components listed in tests/test_mimo2_mtp_parity.py plus h_in.bin / h_nextn.bin rows and the draft tokens.

Tolerances (unchanged from N2/N6/N9): normwise relative error max|got - want| / max|want| < 1e-4 for f32 paths; exact
for the gathers and the zero cells; MoE ids equal as sets (no tie replays were needed). MXFP4 products: N10's metric,
scaled error max|got - want| / max(|W| |x_q|) < TOL_OP = 1e-5 per product from llama.cpp's own input, the activation
quantization emulated; without it every product must be off by more than TOL_OP in every graph and 20 TOL_OP in some.

## Findings (CONFIRMED by the literal outputs below unless labelled)

1. **No llama.cpp-vs-oracle mismatch at either pin.** The driver through the CI checker: 92/92 at ec7630a and at
   79e2e74. Worst f32 error 2.440e-06 (`Qcur` after RoPE, layer 3, rollback graph 48; RoPE in f32 at positions near 300),
   MTP worst 1.128e-06 (`mtp_Qcur`), every greedy argmax and every draft token equal. MXFP4 runs: worst scaled error
   5.28e-08 (Q8_0, `wqkv`), 1.91e-08 (MXFP4 experts), router (F32) 5.07e-08, against TOL_OP 1e-5; without the
   activation quantization 1.15e-03 to 1.09e-02. The per-layer table is below. No tolerance was loosened, no test
   changed.
2. **The sliding-window cache is a 256-cell ring, and the cell model reproduces llama.cpp's every placement.** With
   n_swa 8 and -ub 64 (13, 5) the SWA cache has 256 cells; positions 256.. overwrite the oldest cells (one token per
   call: cell = position mod 256), a ubatch straddling the end continues at cell 0, the base cache never wraps (cell =
   position, asserted in every graph). `kq` matches (B) on every one of the n_kv cells of every graph, including the
   cells that hold out-of-window and stale K rows, and `kq_soft_max` is exactly 0 on every masked cell.
3. **Rollbacks inside and across the window behave like the oracle.** After `llama_memory_seq_rm(0, 258, -1)` (by 2,
   inside the window) the freed cells 2, 3 are reused for the re-decoded 258, 259; after the rollback by 10 (more than
   the window) every later row equals the full-sequence oracle of the final sequence; the rejected rows equal the
   incremental oracle that replays them. No snapshot is involved (mimo2 has no recurrent state).
4. **A rollback deeper than the SWA cache retains is accepted and silently changes the result** (not a bug in the
   oracle sense: the cell-accurate oracle (B) matches llama.cpp, kqv_out within 6.2e-07 and logits within 4.5e-07; the
   full-sequence oracle (A) cannot).
   After prefill 280 the SWA cache holds positions 24..279; `llama_memory_seq_rm(0, 20, -1)` frees all 256 SWA cells and
   returns true (KVI:116-117 combine the two caches' results, KV:382-420 never refuses). Position 20 then attends, in
   every SWA layer (they share the SWA cache's cells and mask), to itself alone instead of positions 13..20 (21 to two,
   ..., 26 to seven cells); kqv_out of layer 1 differs from (A) by 0.53 to 1.51 (relative) at 20..26, and through the
   global layers the logits stay off for every later position (0.10..1.03 at 20..31, still 2.4e-02..4.4e-02 at 44..47). llama-server avoids this for prompt reuse:
   it compares `llama_memory_seq_pos_min` with `pos_next - n_swa` and restores a checkpoint or reprocesses
   (tools/server/server-context.cpp:3674-3677, 3726 @79e2e74). An engine that rolls back with seq_rm itself (MTP or
   n-gram speculation) must keep the rollback depth within what the SWA cache retains: a rollback to p needs positions
   p - n_swa + 1 .. p - 1, and a cache of C cells that has wrapped holds only the last C positions, so the depth must
   be at most C - n_swa + 1 (here 249) (PROVISIONAL derivation from the placement rule, consistent with the
   property test `test_deep_rollback_loses_the_window`). For a real MiMo-V2.6-Flash file with a 128-token window and
   n_ubatch 512, C = 768 (PROVISIONAL: n_swa 128 is the value docs/reassessment/research/R1-models.md:41 reports for
   the V2.6 recipe; the GGUF value is UNKNOWN until a real header is read), far above any draft length.
5. **MTP.** The block matches (A) and (B) in all 32 graphs of the MTP context (28 process graphs and 4 draft steps),
   including its own SWA cache wrapping at 256 (asserted), the head norm `layer_output_norm` (the middle of the
   fallback chain, M:392-394), the fused qkv and the value scale. The pairing of common/speculative.cpp (token p with
   the target's h of p-1) is the one in the dumps (`h_in.bin`, tested). PROVISIONAL consequence: a GGUF whose MTP layer
   has split q/k/v or a MoE FFN would abort at graph build (M:285, 375 are GGML_ASSERTs); whether any MiMo-V2.6 GGUF has
   that is UNKNOWN.
6. **The two pins compute bit-identical dumps** for every run, graph and tensor they both name (13 run directories,
   53 218 tensor files compared, 0 differ; logits, h rows equal); the only difference is the rename of the last `l_out`
   to `h_nextn` at 79e2e74 (pin dump comparison below). The generated and the quantized GGUFs are byte-identical at both
   pins (sha256 below).
7. **MXFP4.** The quantizer writes e = floor(log2 amax) - 2 + 127 per block (ggml-quants.c:368; checked on every block
   of every expert tensor) and every dequantized weight lies within 4 d of its f32 weight (d = 2^(e-128)); ref/quant.py's
   own dequantization equals the pinned gguf-py's bit for bit (all 255 finite exponents, denormal scales included, and
   the model's tensors). Every expert product matches from llama.cpp's own input on both paths: the repacked AVX2
   mxfp4_8x8 path (a `CPU_REPACK` buffer of 0.12 MiB, all 15 expert tensors) and ggml_vec_dot_mxfp4_q8_0
   (`--no-repack`). The down projections (rows of one 32-block) are exact (scaled error 0): one integer sum times one
   f32 scale. The two paths differ in the last bits of a few gate / up products (6 of 166 tensors of pf64, 10 of 2822 of
   decode; summation order), which the next Q8_0 activation quantization absorbs: logits identical.
8. **Generator.** The default output of tools/tinygen/mimo2_tiny.py is byte-identical before and after the N16 options
   with the gguf-py of both pins, and G1 MiMo stays 5/5 at both pins (below).
9. **Mutation check** (below): 19/19 caught at both pins. W4 (no purge after an overwrite) is caught by the property
   test only: no run here reaches the purge (PROVISIONAL reason: with one sequence the head walk always reaches the
   oldest cell first, so an overwrite never leaves an older position behind; the dump tests pass with the mutant).
10. **CI**: the step fails through the checker with no environment, an empty dump directory, the mtp run missing or one
    MXFP4 run missing, at both pins (bites below); `tests.test_check_unittest` and `tests.test_hooks` pass.

## Not covered (NOT RUN unless labelled)

* `-ctk f16 -ctv f16 -fa on` (the R1 default precision) for mimo2: NOT RUN; flash attention with sinks on the CPU
  (`ggml_flash_attn_ext_add_sinks`) is not emulated anywhere yet.
* several sequences / shared cells, `--swa-full`, `-t` > 1, the purge path of apply_ubatch with real dumps (finding 9).
* rollback in the MTP context (DUMP_MTP has no rollback), MTP heads with `nextn_layer_offset` > 0 (the converter writes
  3 MTP layers for V2-Flash/V2.5, conversion/mimo.py:22-24; the dump tool drives offset 0 only), a quantized MTP block.
* the optional `nextn.embed_tokens`, `shared_head_head`, `shared_head_norm` tensors (not in the tiny model: the fallbacks
  `token_embd`, `output`, `layer_output_norm` are what is checked).
* MXFP4 for anything but the routed experts, NVFP4, Q4_0/IQ4_NL (they share the Q8_0 activation path), MXFP4 in the
  MTP block; MXFP4 GPU kernels.
* the real MiMo-V2.6-Flash hyper-parameters (window, sinks on global layers, value scale, fused qkv, MTP layer count):
  UNKNOWN until `third_party/hf/mimo-v2.6-flash-chat` is vendored (the request is committed; the repo id is the one
  docs/reassessment/research/R1-models.md:41 and :267 recommend, `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`;
  `models/mimo-v2.6-flash.json:3` names `...-Flash-RL`, marked outdated at R1-models.md:267); the remote-code file
  names in the request are a guess (PROVISIONAL), a wrong name is recorded as absent upstream.
* GitHub Actions: NOT RUN (this task does not push).

## Literal outputs

Pin check and environment (sandbox, no GPU):

```
$ git -C <S>/llama-ec7630a rev-parse HEAD; git -C <S>/llama.cpp rev-parse HEAD
ec7630a640789c393694fb194f1bbbf0369fc62d
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
$ nproc; python3 --version; python3 -c "import numpy; print(numpy.__version__)"; nvidia-smi
4
Python 3.13.16
2.5.3
/bin/bash: line 1: nvidia-smi: command not found
```

Fresh builds (`<S>/n16/build.sh`: `cmake -S tools/g0 -B <S>/n16/build-<pin> -DLLAMA_CPP_DIR=<checkout> -DCMAKE_BUILD_TYPE=Release
-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON`, then `cmake --build ... -j4 --target dump-tensors llama-quantize`):

```
$ cat <S>/n16/builds.log
ec cmake 2 s, build 160 s
r1 cmake 1 s, build 161 s
BUILDS-DONE
$ grep -E "^(CMAKE_HOME_DIRECTORY|LLAMA_CPP_DIR|G0_QUANTIZE|GGML_AVX2|GGML_AVX512|GGML_NATIVE)[:=]" <S>/n16/build-{ec,r1}/CMakeCache.txt
<S>/n16/build-ec/CMakeCache.txt:G0_QUANTIZE:BOOL=ON
<S>/n16/build-ec/CMakeCache.txt:GGML_AVX2:BOOL=ON
<S>/n16/build-ec/CMakeCache.txt:GGML_AVX512:BOOL=OFF
<S>/n16/build-ec/CMakeCache.txt:GGML_NATIVE:BOOL=OFF
<S>/n16/build-ec/CMakeCache.txt:LLAMA_CPP_DIR:PATH=<S>/llama-ec7630a
<S>/n16/build-ec/CMakeCache.txt:CMAKE_HOME_DIRECTORY:INTERNAL=/home/user/wt-n16/tools/g0
<S>/n16/build-r1/CMakeCache.txt:G0_QUANTIZE:BOOL=ON
<S>/n16/build-r1/CMakeCache.txt:GGML_AVX2:BOOL=ON
<S>/n16/build-r1/CMakeCache.txt:GGML_AVX512:BOOL=OFF
<S>/n16/build-r1/CMakeCache.txt:GGML_NATIVE:BOOL=OFF
<S>/n16/build-r1/CMakeCache.txt:LLAMA_CPP_DIR:PATH=<S>/llama.cpp
<S>/n16/build-r1/CMakeCache.txt:CMAKE_HOME_DIRECTORY:INTERNAL=/home/user/wt-n16/tools/g0
```

### The final runs (`<S>/n16/final.sh`, on the committed HEAD)

`final.sh` prints the worktree HEAD, runs the CI step command of both workflows at both pins from an empty work directory
(`MIMO2_DECODE_WORK=<S>/n16/work-<pin> python3 tools/ci/check_unittest.py --step n16-mimo2-decode --checkout <checkout> --
tools/tinygen/run_parity_mimo2_decode.sh <checkout> <S>/n16/build-<pin>`, cwd `<S>/n16/cwd`), then the pin dump
comparison, the bites, the evict probe and the mutation rounds (sections below). Its log, the step lines:

```
worktree HEAD a003c5e90feea9cbd6d187857cc3488f4b383c78; status: 0 changed files
ec wrapper_rc=0 wall_s=24
r1 wrapper_rc=0 wall_s=20
mutation ec rc=0
mutation r1 rc=0
FINAL-DONE
```

Driver through the checker at ec7630a (`grep -E` of the run summaries, the unittest summary and the checker's annotation; the verbose test lines and the tables are cut here, the tables follow):

```
CMAKE_BUILD_TYPE=Release
wrote <S>/n16/work-ec/n16/split.gguf
wrote <S>/n16/work-ec/n16/n16.gguf
ub64: 5 graphs, 1 calls
ub13: 24 graphs, 1 calls
ub5: 60 graphs, 1 calls
decode: 64 graphs, 61 calls
rollback: 62 graphs, 61 calls
evict: 10 graphs, 7 calls
split: 24 graphs, 21 calls
mtp: 28 graphs, 25 calls
mtp (MTP context): 32 graphs, [('process', 1), ('process', 1), ('draft', 1), ('draft', 1), ('draft', 1), ('draft', 1)]
mxfp4/pf64: 1 graphs, 1 calls
mxfp4/decode: 17 graphs, 17 calls
mxfp4/pf64-nr: 1 graphs, 1 calls
mxfp4/decode-nr: 17 graphs, 17 calls
Ran 92 tests in 13.637s
OK
artifacts: <S>/n16/work-ec
::notice::n16-mimo2-decode @ llama.cpp ec7630a: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
```

Driver through the checker at 79e2e74 (`grep -E` of the run summaries, the unittest summary and the checker's annotation; the verbose test lines and the tables are cut here, the tables follow):

```
CMAKE_BUILD_TYPE=Release
wrote <S>/n16/work-r1/n16/split.gguf
wrote <S>/n16/work-r1/n16/n16.gguf
ub64: 5 graphs, 1 calls
ub13: 24 graphs, 1 calls
ub5: 60 graphs, 1 calls
decode: 64 graphs, 61 calls
rollback: 62 graphs, 61 calls
evict: 10 graphs, 7 calls
split: 24 graphs, 21 calls
mtp: 28 graphs, 25 calls
mtp (MTP context): 32 graphs, [('process', 1), ('process', 1), ('draft', 1), ('draft', 1), ('draft', 1), ('draft', 1)]
mxfp4/pf64: 1 graphs, 1 calls
mxfp4/decode: 17 graphs, 17 calls
mxfp4/pf64-nr: 1 graphs, 1 calls
mxfp4/decode-nr: 17 graphs, 17 calls
Ran 92 tests in 12.963s
OK
artifacts: <S>/n16/work-r1
::notice::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
```

### Per-layer table (gap 14), both pins

As printed by tests/test_mimo2_decode_parity.py (`sed -n '/^worst error per layer/,/^ *5 /p'` of the logs):

```
worst error per layer over runs, components and graphs (layer -1 = embedding and head), llama.cpp ec7630a:
layer run       component                                            graph    max abs        rel
   -1 decode    result_output [full]                                    36  1.996e-06  7.561e-07
    0 decode    kq_soft_max (cells) [chunk]                             48  5.581e-08  1.735e-06
    1 rollback  kq_soft_max (cells) [full]                              53  6.730e-07  2.057e-06
    2 rollback  Kcur (RoPE) [chunk]                                     51  4.096e-06  2.112e-06
    3 rollback  Qcur (RoPE) [full]                                      48  5.020e-06  2.440e-06
    4 decode    Kcur (RoPE) [full]                                      53  3.250e-06  1.799e-06
    5 decode    kq_soft_max (cells) [full]                               4  7.748e-08  1.540e-06
```

```
worst error per layer over runs, components and graphs (layer -1 = embedding and head), llama.cpp 79e2e74:
layer run       component                                            graph    max abs        rel
   -1 decode    result_output [full]                                    36  1.996e-06  7.561e-07
    0 decode    kq_soft_max (cells) [chunk]                             48  5.581e-08  1.735e-06
    1 rollback  kq_soft_max (cells) [full]                              53  6.730e-07  2.057e-06
    2 rollback  Kcur (RoPE) [chunk]                                     51  4.096e-06  2.112e-06
    3 rollback  Qcur (RoPE) [full]                                      48  5.020e-06  2.440e-06
    4 decode    Kcur (RoPE) [full]                                      53  3.250e-06  1.799e-06
    5 decode    kq_soft_max (cells) [full]                               4  7.748e-08  1.540e-06
```

Worst error per run (the largest `rel` of each run in the per-run, per-component table; command:
`awk '/^worst error per run and component/{f=1;next} /^worst error per layer/{f=0} f && /^(ub64|ub13|ub5|decode|rollback|evict|split|mtp) /' <log> | grep -v "expected to differ" | awk '{rel=$NF; run=$1; if (!(run in m) || rel+0 > m[run]+0) {m[run]=rel; l[run]=$0}} END {for (r in l) print l[r]}' | sort`,
then the one entry recorded as expected to differ). Columns: run, component, layer, graph, max abs, rel. Identical at
both pins (`diff` of the two outputs is empty):

```
decode    Qcur (RoPE) [full]                                       3    50  4.961e-06  2.411e-06
evict     Kcur (RoPE) [chunk]                                      4     4  4.409e-06  1.455e-06
mtp       kq_soft_max (cells) [chunk]                              5     4  7.748e-08  1.540e-06
rollback  Qcur (RoPE) [full]                                       3    48  5.020e-06  2.440e-06
split     ffn_moe_out [full]                                       4     5  8.126e-07  1.344e-06
ub13      Qcur (RoPE) [full]                                       2    22  5.786e-06  1.700e-06
ub5       Qcur (RoPE) [full]                                       2    57  5.786e-06  1.804e-06
ub64      Qcur (RoPE) [full]                                       2     4  5.786e-06  1.610e-06
evict     kqv_out vs (A) at position 20 (expected to differ)       1     5  1.980e+00  1.129e+00
```

MTP table at ec7630a (identical at 79e2e74, `diff` without the header line is empty):

```
MTP (layer 6): worst error per component over the MTP graphs, llama.cpp ec7630a:
component                                    graph    max abs        rel
mtp_tok_embd (exact)                             0  0.000e+00  0.000e+00
h_in (row fed) [full]                            6  1.453e-06  4.405e-07
h_in (row fed) [chunk]                           6  1.453e-06  4.405e-07
mtp_enorm [full]                                22  3.020e-07  1.230e-07
mtp_enorm [chunk]                               22  3.020e-07  1.230e-07
mtp_hnorm [full]                                24  9.813e-07  4.228e-07
mtp_hnorm [chunk]                               24  9.813e-07  4.228e-07
mtp_concat [e ; h] [full]                       24  9.813e-07  4.228e-07
mtp_concat [e ; h] [chunk]                      24  9.813e-07  4.228e-07
mtp_eh_proj [full]                               6  8.648e-07  4.081e-07
mtp_eh_proj [chunk]                              6  8.648e-07  4.081e-07
mtp_attn_norm [full]                             6  1.162e-06  4.423e-07
mtp_attn_norm [chunk]                            6  1.162e-06  4.423e-07
mtp_wqkv (fused projection) [full]               6  1.139e-06  3.742e-07
mtp_wqkv (fused projection) [chunk]              6  1.139e-06  3.742e-07
mtp_Qcur (RoPE) [full]                          11  2.523e-06  1.098e-06
mtp_Qcur (RoPE) [chunk]                         31  2.406e-06  1.128e-06
mtp_Kcur (RoPE) [full]                           3  3.434e-06  1.082e-06
mtp_Kcur (RoPE) [chunk]                          3  3.434e-06  1.082e-06
mtp_Vcur [full]                                 12  8.855e-07  6.677e-07
mtp_Vcur [chunk]                                12  8.855e-07  6.677e-07
kq-6 (all n_kv cells) [chunk]                    5  1.067e-05  7.095e-07
kq_soft_max-6 (cells) [chunk]                   18  2.849e-07  5.312e-07
kq_soft_max-6 (cells) [full]                    18  2.849e-07  5.312e-07
kqv_out-6 [full]                                 5  6.022e-07  5.258e-07
kqv_out-6 [chunk]                                5  6.022e-07  5.258e-07
mtp_attn_out [full]                             20  6.364e-07  6.156e-07
mtp_attn_out [chunk]                            20  6.364e-07  6.156e-07
mtp_attn_out_scaled [full]                      20  4.773e-07  6.156e-07
mtp_attn_out_scaled [chunk]                     20  4.773e-07  6.156e-07
mtp_ffn_inp [full]                               6  9.779e-07  3.998e-07
mtp_ffn_inp [chunk]                              6  9.779e-07  3.998e-07
mtp_ffn_norm [full]                              6  1.178e-06  4.078e-07
mtp_ffn_norm [chunk]                             6  1.178e-06  4.078e-07
ffn_up-6 (dense) [full]                          6  1.658e-06  6.217e-07
ffn_up-6 (dense) [chunk]                         6  1.658e-06  6.217e-07
ffn_gate-6 (dense) [full]                        6  1.154e-06  4.939e-07
ffn_gate-6 (dense) [chunk]                       6  1.154e-06  4.939e-07
ffn_swiglu-6 (dense) [full]                     15  1.855e-06  8.391e-07
ffn_swiglu-6 (dense) [chunk]                    15  1.855e-06  8.391e-07
mtp_ffn_out [full]                              15  8.712e-07  7.776e-07
mtp_ffn_out [chunk]                             15  8.712e-07  7.776e-07
mtp_post_ffn [full]                              6  1.623e-06  7.266e-07
mtp_post_ffn [chunk]                             6  1.623e-06  7.266e-07
h_nextn (graph output) [full]                    6  1.623e-06  7.266e-07
h_nextn (graph output) [chunk]                   6  1.623e-06  7.266e-07
mtp_shared_head_norm [full]                      6  1.680e-06  7.651e-07
mtp_shared_head_norm [chunk]                     6  1.680e-06  7.651e-07
result_output [full]                             6  2.279e-06  8.085e-07
result_output [chunk]                            6  2.279e-06  8.085e-07
logits.bin [full]                                6  2.279e-06  8.085e-07
logits.bin [chunk]                               6  2.279e-06  8.085e-07
h_nextn.bin [full]                               6  1.623e-06  7.266e-07
h_nextn.bin [chunk]                              6  1.623e-06  7.266e-07
target h_nextn.bin [full]                       -1  1.955e-06  2.528e-07
target h_nextn.bin [chunk]                      -1  1.955e-06  2.528e-07
```

MXFP4 per-product table at ec7630a (identical at 79e2e74 except the commit in the header):

```
mimo2: quantized products from llama.cpp's own inputs (ec7630a640789c393694fb194f1bbbf0369fc62d), worst over layers and graphs. scaled = max|got-want| / max(|W| |x_q|) (asserted < TOL_OP = 1e-05; for a bounded product: the largest share of its bound, asserted <= 1); rel = max|got-want| / max|want|; no-act = the same without the activation quantization (min .. max over graphs):
type    run       product                                      weight       scaled       rel layer graph         no-act scaled            no-act rel
mxfp4   decode    embd (gather, exact)                         Q8_0       0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
mxfp4   decode    wqkv (fused attn_qkv)                        Q8_0       5.28e-08  1.18e-07     4     0  1.39e-03 .. 4.15e-03  2.53e-03 .. 9.89e-03
mxfp4   decode    attn_out (attn_output of kqv_out)            Q8_0       4.86e-08  1.17e-07     3    12  1.31e-03 .. 3.80e-03  3.09e-03 .. 1.04e-02
mxfp4   decode    ffn_up (dense)                               Q8_0       4.05e-08  1.07e-07     0     0  1.74e-03 .. 3.69e-03  3.98e-03 .. 8.94e-03
mxfp4   decode    ffn_gate (dense)                             Q8_0       4.41e-08  9.96e-08     0     0  1.66e-03 .. 3.49e-03  4.18e-03 .. 8.52e-03
mxfp4   decode    ffn_out (dense down)                         Q8_0       4.86e-08  1.14e-07     0    10  2.61e-03 .. 7.74e-03  6.30e-03 .. 1.39e-02
mxfp4   decode    ffn_moe_logits (router)                      F32        4.70e-08  2.46e-07     5     5  9.28e-09 .. 4.70e-08  1.71e-08 .. 2.46e-07
mxfp4   decode    ffn_moe_gate (experts)                       MXFP4      1.91e-08  4.97e-08     2     1  1.43e-03 .. 3.70e-03  2.92e-03 .. 1.01e-02
mxfp4   decode    ffn_moe_up (experts)                         MXFP4      1.87e-08  5.34e-08     5    11  1.15e-03 .. 3.55e-03  3.20e-03 .. 9.76e-03
mxfp4   decode    ffn_moe_down (experts)                       MXFP4      0.00e+00  0.00e+00     5    16  2.71e-03 .. 1.09e-02  4.00e-03 .. 1.59e-02
mxfp4   decode    result_output                                Q8_0       4.69e-08  1.00e-07    -1     0  1.66e-03 .. 2.82e-03  3.25e-03 .. 7.82e-03
mxfp4   decode-nr embd (gather, exact)                         Q8_0       0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
mxfp4   decode-nr wqkv (fused attn_qkv)                        Q8_0       5.28e-08  1.18e-07     4     0  1.39e-03 .. 4.15e-03  2.53e-03 .. 9.89e-03
mxfp4   decode-nr attn_out (attn_output of kqv_out)            Q8_0       4.86e-08  1.17e-07     3    12  1.31e-03 .. 3.80e-03  3.09e-03 .. 1.04e-02
mxfp4   decode-nr ffn_up (dense)                               Q8_0       4.05e-08  1.07e-07     0     0  1.74e-03 .. 3.69e-03  3.98e-03 .. 8.94e-03
mxfp4   decode-nr ffn_gate (dense)                             Q8_0       4.41e-08  9.96e-08     0     0  1.66e-03 .. 3.49e-03  4.18e-03 .. 8.52e-03
mxfp4   decode-nr ffn_out (dense down)                         Q8_0       4.86e-08  1.14e-07     0    10  2.61e-03 .. 7.74e-03  6.30e-03 .. 1.39e-02
mxfp4   decode-nr ffn_moe_logits (router)                      F32        4.70e-08  2.46e-07     5     5  9.28e-09 .. 4.70e-08  1.71e-08 .. 2.46e-07
mxfp4   decode-nr ffn_moe_gate (experts)                       MXFP4      1.91e-08  5.44e-08     2     1  1.43e-03 .. 3.70e-03  2.92e-03 .. 1.01e-02
mxfp4   decode-nr ffn_moe_up (experts)                         MXFP4      1.87e-08  5.34e-08     5    11  1.15e-03 .. 3.55e-03  3.20e-03 .. 9.76e-03
mxfp4   decode-nr ffn_moe_down (experts)                       MXFP4      0.00e+00  0.00e+00     5    16  2.71e-03 .. 1.09e-02  4.00e-03 .. 1.59e-02
mxfp4   decode-nr result_output                                Q8_0       4.69e-08  1.00e-07    -1     0  1.66e-03 .. 2.82e-03  3.25e-03 .. 7.82e-03
mxfp4   pf64      embd (gather, exact)                         Q8_0       0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
mxfp4   pf64      wqkv (fused attn_qkv)                        Q8_0       5.28e-08  1.09e-07     4     0  2.85e-03 .. 3.45e-03  5.31e-03 .. 6.82e-03
mxfp4   pf64      attn_out (attn_output of kqv_out)            Q8_0       3.71e-08  9.45e-08     3     0  1.85e-03 .. 3.10e-03  4.28e-03 .. 7.54e-03
mxfp4   pf64      ffn_up (dense)                               Q8_0       4.00e-08  8.36e-08     0     0  3.05e-03 .. 3.05e-03  6.37e-03 .. 6.37e-03
mxfp4   pf64      ffn_gate (dense)                             Q8_0       4.41e-08  8.16e-08     0     0  3.38e-03 .. 3.38e-03  6.24e-03 .. 6.24e-03
mxfp4   pf64      ffn_out (dense down)                         Q8_0       4.50e-08  9.15e-08     0     0  6.82e-03 .. 6.82e-03  1.39e-02 .. 1.39e-02
mxfp4   pf64      ffn_moe_logits (router)                      F32        5.07e-08  1.01e-07     3     0  3.55e-08 .. 5.07e-08  7.86e-08 .. 1.01e-07
mxfp4   pf64      ffn_moe_gate (experts)                       MXFP4      1.67e-08  3.32e-08     2     0  2.67e-03 .. 2.98e-03  5.24e-03 .. 6.29e-03
mxfp4   pf64      ffn_moe_up (experts)                         MXFP4      1.67e-08  3.48e-08     1     0  2.51e-03 .. 3.27e-03  4.65e-03 .. 6.44e-03
mxfp4   pf64      ffn_moe_down (experts)                       MXFP4      0.00e+00  0.00e+00     5     0  6.66e-03 .. 8.91e-03  8.75e-03 .. 1.25e-02
mxfp4   pf64      result_output                                Q8_0       4.69e-08  9.31e-08    -1     0  2.82e-03 .. 2.82e-03  5.60e-03 .. 5.60e-03
mxfp4   pf64-nr   embd (gather, exact)                         Q8_0       0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
mxfp4   pf64-nr   wqkv (fused attn_qkv)                        Q8_0       5.28e-08  1.09e-07     4     0  2.85e-03 .. 3.45e-03  5.31e-03 .. 6.82e-03
mxfp4   pf64-nr   attn_out (attn_output of kqv_out)            Q8_0       3.71e-08  9.45e-08     3     0  1.85e-03 .. 3.10e-03  4.28e-03 .. 7.54e-03
mxfp4   pf64-nr   ffn_up (dense)                               Q8_0       4.00e-08  8.36e-08     0     0  3.05e-03 .. 3.05e-03  6.37e-03 .. 6.37e-03
mxfp4   pf64-nr   ffn_gate (dense)                             Q8_0       4.41e-08  8.16e-08     0     0  3.38e-03 .. 3.38e-03  6.24e-03 .. 6.24e-03
mxfp4   pf64-nr   ffn_out (dense down)                         Q8_0       4.50e-08  9.15e-08     0     0  6.82e-03 .. 6.82e-03  1.39e-02 .. 1.39e-02
mxfp4   pf64-nr   ffn_moe_logits (router)                      F32        5.07e-08  1.01e-07     3     0  3.55e-08 .. 5.07e-08  7.86e-08 .. 1.01e-07
mxfp4   pf64-nr   ffn_moe_gate (experts)                       MXFP4      1.67e-08  3.32e-08     2     0  2.67e-03 .. 2.98e-03  5.24e-03 .. 6.29e-03
mxfp4   pf64-nr   ffn_moe_up (experts)                         MXFP4      1.67e-08  3.48e-08     1     0  2.51e-03 .. 3.27e-03  4.65e-03 .. 6.44e-03
mxfp4   pf64-nr   ffn_moe_down (experts)                       MXFP4      0.00e+00  0.00e+00     5     0  6.66e-03 .. 8.91e-03  8.75e-03 .. 1.25e-02
mxfp4   pf64-nr   result_output                                Q8_0       4.69e-08  9.31e-08    -1     0  2.82e-03 .. 2.82e-03  5.60e-03 .. 5.60e-03
```

### Pin dump comparison (`docs/evidence/mutation-scripts/2026-10-11-n16-pindiff.py`, from `final.sh`)

Two calls, `<work-ec>/n16 <work-r1>/n16` and `.../n16/mxfp4` (the `mxfp4` row of the first call has no graphs of its own;
its four runs are the last four rows):

```
== pin dump comparison
decode           files in both  10624, differing    0; only at one pin  128 ['h_nextn', 'l_out-5']; logits.bin differing: none
evict            files in both   1660, differing    0; only at one pin   20 ['h_nextn', 'l_out-5']; logits.bin differing: none
mtp              files in both   4676, differing    0; only at one pin    0 []; logits.bin, h_nextn.bin differing: none
mtp/mtp          files in both   1088, differing    0; only at one pin    0 []; logits.bin, h_nextn.bin, h_in.bin differing: none
mxfp4            files in both      0, differing    0; only at one pin    0 [];  differing: none
rollback         files in both  10292, differing    0; only at one pin  124 ['h_nextn', 'l_out-5']; logits.bin differing: none
split            files in both   4128, differing    0; only at one pin   48 ['h_nextn', 'l_out-5']; logits.bin differing: none
ub13             files in both   3984, differing    0; only at one pin   48 ['h_nextn', 'l_out-5']; logits.bin differing: none
ub5              files in both   9960, differing    0; only at one pin  120 ['h_nextn', 'l_out-5']; logits.bin differing: none
ub64             files in both    830, differing    0; only at one pin   10 ['h_nextn', 'l_out-5']; logits.bin differing: none
decode           files in both   2822, differing    0; only at one pin    0 []; logits.bin differing: none
decode-nr        files in both   2822, differing    0; only at one pin    0 []; logits.bin differing: none
pf64             files in both    166, differing    0; only at one pin    0 []; logits.bin differing: none
pf64-nr          files in both    166, differing    0; only at one pin    0 []; logits.bin differing: none
```

sha256 of the models at both pins:

```
$ for f in split n16 mxfp4/model; do sha256sum <S>/n16/work-ec/n16/$f.gguf <S>/n16/work-r1/n16/$f.gguf; done
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  <S>/n16/work-ec/n16/split.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  <S>/n16/work-r1/n16/split.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  <S>/n16/work-ec/n16/n16.gguf
4d544d7ef03e337fde81a8fb7bdc56bb77129eb7a86ee0beec3883b40347b945  <S>/n16/work-r1/n16/n16.gguf
6ba99037710d63dc7d5f121ed5746ed210c98b3a3d12e4e0064f3d7babf76680  <S>/n16/work-ec/n16/mxfp4/model.gguf
6ba99037710d63dc7d5f121ed5746ed210c98b3a3d12e4e0064f3d7babf76680  <S>/n16/work-r1/n16/mxfp4/model.gguf
```

### The step fails when its tests did not all run (`docs/evidence/mutation-scripts/2026-10-11-n16-bites.sh`, from `final.sh`)

The step's own test line through `tools/ci/check_unittest.py --step n16-mimo2-decode --commit <pin>`: complete dumps;
no environment; `MIMO2_DECODE_DUMP` an empty directory; a dump directory without the mtp run; one without
`mxfp4/decode-nr` (hard-link copies). Annotations cut at 160 characters by the script:

```
== bites ec (ec7630a)
complete: checker rc=0 | Ran 92 tests in 14.099s OK | 
no-env: checker rc=1 | Ran 92 tests in 0.622s OK (skipped=82) | ::error::n16-mimo2-decode @ llama.cpp ec7630a: skipped=82, expected skipped=0 ::error::n16-mimo2-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_mimo2_decode_parity.Run_decode.test_attention: 'set MIMO2_DECODE_DUMP, LLAMA_CP 
empty-dir: checker rc=1 | Ran 11 tests in 0.554s OK (skipped=13) | ::error::n16-mimo2-decode @ llama.cpp ec7630a: Ran 11 tests, expected 92 ::error::n16-mimo2-decode @ llama.cpp ec7630a: skipped=13, expected skipped=0 
no-mtp-run: checker rc=1 | Ran 82 tests in 10.061s OK (skipped=2) | ::error::n16-mimo2-decode @ llama.cpp ec7630a: Ran 82 tests, expected 92 ::error::n16-mimo2-decode @ llama.cpp ec7630a: skipped=2, expected skipped=0 
no-quant-run: checker rc=1 | Ran 85 tests in 12.119s OK (skipped=1) | ::error::n16-mimo2-decode @ llama.cpp ec7630a: Ran 85 tests, expected 92 ::error::n16-mimo2-decode @ llama.cpp ec7630a: skipped=1, expected skipped=0 
```

```
== bites r1 (79e2e74)
complete: checker rc=0 | Ran 92 tests in 12.212s OK | 
no-env: checker rc=1 | Ran 92 tests in 0.887s OK (skipped=82) | ::error::n16-mimo2-decode @ llama.cpp 79e2e74: skipped=82, expected skipped=0 ::error::n16-mimo2-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_mimo2_decode_parity.Run_decode.test_attention: 'set MIMO2_DECODE_DUMP, LLAMA_CP 
empty-dir: checker rc=1 | Ran 11 tests in 0.666s OK (skipped=13) | ::error::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 11 tests, expected 92 ::error::n16-mimo2-decode @ llama.cpp 79e2e74: skipped=13, expected skipped=0 
no-mtp-run: checker rc=1 | Ran 82 tests in 10.791s OK (skipped=2) | ::error::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 82 tests, expected 92 ::error::n16-mimo2-decode @ llama.cpp 79e2e74: skipped=2, expected skipped=0 
no-quant-run: checker rc=1 | Ran 85 tests in 11.291s OK (skipped=1) | ::error::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 85 tests, expected 92 ::error::n16-mimo2-decode @ llama.cpp 79e2e74: skipped=1, expected skipped=0 
```

### Finding 4: the deep rollback (`docs/evidence/mutation-scripts/2026-10-11-n16-evict-probe.py`, from `final.sh`)

Per re-decoded position: the cells llama.cpp's first SWA layer attended to (probabilities > 0 in the dumped
kq_soft_max, head 0), the positions (A) attends to, and the relative differences of kqv_out (layer 1) and of the logits
from (A) and from the cell-accurate (B). At ec7630a; the 79e2e74 output is the same, line for line (in `final.log`):

```
before the rollback: SWA cache holds positions 24..279 (256 of 256 cells); rollback to 20
after: SWA cache cells in use 0, base cache 20
pos  cells(L1, llama.cpp)  positions(L1, A)  kqv_out L1 vs A  vs B   logits vs A   vs B
 20                     1                 8        1.129e+00  1.6e-07     5.572e-01  2.1e-07
 21                     2                 8        1.509e+00  2.0e-07     8.927e-01  4.5e-07
 22                     3                 8        8.798e-01  6.2e-07     5.359e-01  2.9e-07
 23                     4                 8        7.864e-01  2.5e-07     1.034e+00  2.6e-07
 24                     5                 8        9.645e-01  3.0e-07     4.099e-01  3.7e-07
 25                     6                 8        5.313e-01  2.3e-07     4.903e-01  2.2e-07
 26                     7                 8        8.386e-01  1.8e-07     5.613e-01  2.6e-07
 27                     8                 8        2.136e-07  2.1e-07     2.883e-01  2.2e-07
 28                     8                 8        2.719e-07  2.7e-07     3.836e-01  2.2e-07
 29                     8                 8        1.923e-07  1.9e-07     2.081e-01  2.5e-07
 30                     8                 8        2.080e-07  2.1e-07     1.546e-01  3.7e-07
 31                     8                 8        2.439e-07  2.4e-07     1.037e-01  2.7e-07
 44                     8                 8        2.662e-07  2.7e-07     4.417e-02  2.3e-07
 45                     8                 8        2.357e-07  2.4e-07     3.403e-02  2.4e-07
 46                     8                 8        1.910e-07  1.9e-07     2.396e-02  1.8e-07
 47                     8                 8        2.219e-07  2.2e-07     3.131e-02  2.0e-07
```

### MXFP4: repacked against plain path, and the repack buffer

`docs/evidence/mutation-scripts/2026-10-11-n16-repack-diff.py` on the final dumps, both pins:

```
$ for p in ec r1; do echo "== $p"; python3 docs/evidence/mutation-scripts/2026-10-11-n16-repack-diff.py <S>/n16/work-$p/n16/mxfp4; done
== ec
pf64 vs pf64-nr: 166 tensor files, 6 differ; by name: {'ffn_moe_gate': 1, 'ffn_moe_swiglu': 3, 'ffn_moe_up': 2}; logits max |diff| 0.000e+00
decode vs decode-nr: 2822 tensor files, 10 differ; by name: {'ffn_moe_gate': 2, 'ffn_moe_swiglu': 5, 'ffn_moe_up': 3}; logits max |diff| 0.000e+00
== r1
pf64 vs pf64-nr: 166 tensor files, 6 differ; by name: {'ffn_moe_gate': 1, 'ffn_moe_swiglu': 3, 'ffn_moe_up': 2}; logits max |diff| 0.000e+00
decode vs decode-nr: 2822 tensor files, 10 differ; by name: {'ffn_moe_gate': 2, 'ffn_moe_swiglu': 5, 'ffn_moe_up': 3}; logits max |diff| 0.000e+00
```

The model buffers the runs allocated (the driver runs the MXFP4 dumps at `-lv 4`; `tests/test_mimo2_quant_parity.py`
`test_repack_path` asserts the same):

```
$ for p in ec r1; do for r in pf64 decode pf64-nr decode-nr; do echo "== $p $r"; grep "model buffer size" <S>/n16/work-$p/n16/mxfp4_$r.log; done; done
== ec pf64
0.00.013.179 I load_tensors:   CPU_Mapped model buffer size =     0.25 MiB
0.00.013.199 I load_tensors:   CPU_REPACK model buffer size =     0.12 MiB
== ec decode
0.00.012.252 I load_tensors:   CPU_Mapped model buffer size =     0.25 MiB
0.00.012.278 I load_tensors:   CPU_REPACK model buffer size =     0.12 MiB
== ec pf64-nr
0.00.015.377 I load_tensors:   CPU_Mapped model buffer size =     0.26 MiB
== ec decode-nr
0.00.015.392 I load_tensors:   CPU_Mapped model buffer size =     0.26 MiB
== r1 pf64
0.00.010.659 I load_tensors:   CPU_Mapped model buffer size =     0.25 MiB
0.00.010.687 I load_tensors:   CPU_REPACK model buffer size =     0.12 MiB
== r1 decode
0.00.007.791 I load_tensors:   CPU_Mapped model buffer size =     0.25 MiB
0.00.007.838 I load_tensors:   CPU_REPACK model buffer size =     0.12 MiB
== r1 pf64-nr
0.00.008.230 I load_tensors:   CPU_Mapped model buffer size =     0.26 MiB
== r1 decode-nr
0.00.009.300 I load_tensors:   CPU_Mapped model buffer size =     0.26 MiB
```

The quantizer's choice per tensor (head of the llama-quantize log of the ec7630a run; the two experts lines):

```
$ grep -E "blk.1.ffn_(down|gate)_exps|output.weight|blk.0.attn_qkv" <S>/n16/work-ec/n16/mxfp4/quantize.log
[   1/  71] output.weight                        - [    64,     96,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.01 MiB
[   5/  71] blk.0.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
[   6/  71] blk.0.attn_qkv.weight                - [    64,    136,      1,      1], type =    f32, converting to q8_0 .. size =     0.03 MiB ->     0.01 MiB
[  12/  71] blk.1.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
[  16/  71] blk.1.ffn_down_exps.weight           - [    32,     64,      8,      1], type =    f32, converting to mxfp4 .. size =     0.06 MiB ->     0.01 MiB
[  17/  71] blk.1.ffn_gate_exps.weight           - [    64,     32,      8,      1], type =    f32, converting to mxfp4 .. size =     0.06 MiB ->     0.01 MiB
[  22/  71] blk.2.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
[  32/  71] blk.3.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
[  42/  71] blk.4.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
[  52/  71] blk.5.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
[  61/  71] blk.6.attn_output.weight             - [    64,     64,      1,      1], type =    f32, converting to q8_0 .. size =     0.02 MiB ->     0.00 MiB
```

### Generator: default output byte-identical; G1 MiMo stays 5/5

`docs/evidence/mutation-scripts/2026-10-11-n16-genhash.sh` (the generator at `8389ae3^` and at the worktree, default and
`--seed 3`, with the gguf-py of each pin):

```
$ bash docs/evidence/mutation-scripts/2026-10-11-n16-genhash.sh /home/user/wt-n16 8389ae3^ <S>/n16/genhash <S>/llama-ec7630a/gguf-py <S>/llama.cpp/gguf-py
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  before-79e2e74-default.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  before-79e2e74-defaultseed3.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  before-ec7630a-default.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  before-ec7630a-defaultseed3.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  after-79e2e74-default.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  after-79e2e74-defaultseed3.gguf
aa19506307b27924c004bb0e6643abb478f5afe32f7f61c58045aeb12eaf7a9d  after-ec7630a-default.gguf
7c24bcac618eb29c13533a3fcf787e7cb823d7f823600f26245760ae053ae21d  after-ec7630a-defaultseed3.gguf
```

G1 MiMo through the checker at both pins (run on this branch after `37ab8f4`; tools/tinygen/run_parity_mimo.sh,
tests/test_mimo_tiny_parity.py and the default generator output have not changed since):

```
$ for p in ec r1; do python3 tools/ci/check_unittest.py --step g1-mimo --checkout <checkout> -- tools/tinygen/run_parity_mimo.sh <checkout> <S>/n16/build-$p; done   (filtered: test lines, summary, annotation)
== ec: llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d
test_attention_swa_sinks_gqa (tests.test_mimo_tiny_parity.MimoTinyParity.test_attention_swa_sinks_gqa) ... ok
test_plain_swiglu (tests.test_mimo_tiny_parity.MimoTinyParity.test_plain_swiglu) ... ok
test_rope (tests.test_mimo_tiny_parity.MimoTinyParity.test_rope) ... ok
test_router (tests.test_mimo_tiny_parity.MimoTinyParity.test_router) ... ok
test_sinks_matter (tests.test_mimo_tiny_parity.MimoTinyParity.test_sinks_matter) ... ok
Ran 5 tests in 0.081s
OK
::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
== r1: llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
test_attention_swa_sinks_gqa (tests.test_mimo_tiny_parity.MimoTinyParity.test_attention_swa_sinks_gqa) ... ok
test_plain_swiglu (tests.test_mimo_tiny_parity.MimoTinyParity.test_plain_swiglu) ... ok
test_rope (tests.test_mimo_tiny_parity.MimoTinyParity.test_rope) ... ok
test_router (tests.test_mimo_tiny_parity.MimoTinyParity.test_router) ... ok
test_sinks_matter (tests.test_mimo_tiny_parity.MimoTinyParity.test_sinks_matter) ... ok
Ran 5 tests in 0.071s
OK
::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
```

## Mutation check

`python3 docs/evidence/mutation-scripts/2026-10-11-n16-mimo2-mutations.py /home/user/wt-n16 <S>/n16/work-<pin> <checkout>
<S>/n16/mut/<pin>` on HEAD `a003c5e` (from `final.sh`), once per pin against that pin's dumps; shown through
`grep -E "^(llama|baseline|[A-Z][0-9] |    rc=|    mimo2_|    mxfp4|caught)" <log> | cut -c1-170` (the per-module lists of
failing tests are cut at six names by the script and at 170 columns here; the mutant edits are in the script):

```
llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d; dumps <S>/n16/work-ec/n16
baseline (no mutation): rc=0 Ran 92 tests in 11.990s | OK
W1 window: a key at distance n_swa still visible (<= instead of <)
    rc=1 Ran 92 tests in 5.723s | FAILED (failures=38)
    mimo2_chunked_ref: 2 failing: Mimo2ChunkedRef.test_attend_is_ref_mimo_attention, Mimo2ChunkedRef.test_deep_rollback_loses_the_window
    mimo2_decode_parity: 33 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
W2 SWA cache without the 256-cell padding (n_swa + n_ubatch cells)
    rc=1 Ran 92 tests in 8.935s | FAILED (failures=18)
    mimo2_decode_parity: 16 failing: Run_decode.test_kv_cells, Run_decode.test_run_exercises_its_mechanisms, Run_evict.test_kv_cells, Run_evict.test_run_exercises_its_mec
    mimo2_mtp_parity: 2 failing: Mtp.test_mtp_block, Mtp.test_run_exercises_its_mechanisms
W3 SWA cells never reused (find_slot takes empty cells only)
    rc=1 Ran 39 tests in 5.115s | FAILED (errors=14)
    mimo2_chunked_ref: 5 failing: Mimo2ChunkedRef.test_cell_model, Mimo2ChunkedRef.test_deep_rollback_loses_the_window, Mimo2ChunkedRef.test_mtp_chunks_equal_the_full_blo
    mimo2_decode_parity: 8 failing: Run_decode.setUpClass, Run_evict.setUpClass, Run_mtp.setUpClass, Run_rollback.setUpClass, Run_split.setUpClass, Run_ub13.setUpClass ..
    mimo2_mtp_parity: 1 failing: Mtp.setUpClass
W4 no purge after an overwrite (KV:1160-1177)
    rc=1 Ran 92 tests in 11.380s | FAILED (failures=1)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_cell_model
W5 RoPE base of the global layers on the sliding-window layers
    rc=1 Ran 92 tests in 6.390s | FAILED (failures=35)
    mimo2_decode_parity: 32 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
S1 sinks ignored
    rc=1 Ran 92 tests in 5.660s | FAILED (failures=36)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_fused_split_value_scale_and_sinks
    mimo2_decode_parity: 32 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
S2 a layer without attn_sinks gets sink logits 0 instead of none
    rc=1 Ran 92 tests in 6.549s | FAILED (failures=32)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_attend_is_ref_mimo_attention
    mimo2_decode_parity: 28 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
R1 rollback length: the first rolled-back position is kept (seq_rm from n_keep + 1)
    rc=1 Ran 92 tests in 9.909s | FAILED (failures=12)
    mimo2_chunked_ref: 2 failing: Mimo2ChunkedRef.test_deep_rollback_loses_the_window, Mimo2ChunkedRef.test_rollback_inside_and_across_the_window
    mimo2_decode_parity: 10 failing: Run_evict.test_attention, Run_evict.test_ffn, Run_evict.test_head_and_logits, Run_evict.test_kv_cells, Run_evict.test_run_exercises_i
R2 rollback frees the base cache only (the SWA cache keeps the rejected tokens)
    rc=1 Ran 92 tests in 11.449s | FAILED (failures=11)
    mimo2_chunked_ref: 2 failing: Mimo2ChunkedRef.test_deep_rollback_loses_the_window, Mimo2ChunkedRef.test_rollback_inside_and_across_the_window
    mimo2_decode_parity: 9 failing: Run_evict.test_attention, Run_evict.test_ffn, Run_evict.test_head_and_logits, Run_evict.test_kv_cells, Run_rollback.test_attention, Ru
R3 seq_rm does not move the head back
    rc=1 Ran 92 tests in 12.050s | FAILED (failures=4)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_cell_model
    mimo2_decode_parity: 3 failing: Run_evict.test_kv_cells, Run_rollback.test_kv_cells, Run_rollback.test_run_exercises_its_mechanisms
V1 value scale ignored
    rc=1 Ran 92 tests in 6.151s | FAILED (failures=32)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_fused_split_value_scale_and_sinks
    mimo2_decode_parity: 28 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
G1 GQA tiled (h % n_kv) instead of grouped (h // r)
    rc=1 Ran 92 tests in 5.921s | FAILED (failures=36)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_attend_is_ref_mimo_attention
    mimo2_decode_parity: 32 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
M1 MTP: eh_proj input [h ; e] instead of [e ; h]
    rc=1 Ran 92 tests in 10.753s | FAILED (failures=2)
    mimo2_mtp_parity: 2 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block
M2 MTP: head norm output_norm instead of layer_output_norm
    rc=1 Ran 92 tests in 10.495s | FAILED (failures=3)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_mtp_chunks_equal_the_full_block
    mimo2_mtp_parity: 2 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block
M3 test: MTP token p paired with the target h of p instead of p-1 (full oracle)
    rc=1 Ran 92 tests in 12.117s | FAILED (failures=1)
    mimo2_mtp_parity: 1 failing: Mtp.test_mtp_block
X1 MXFP4: E8M0 scale 2^(e-127) instead of 2^(e-128)
    rc=1 Ran 92 tests in 10.981s | FAILED (failures=7)
    mimo2_quant_parity: 4 failing: Mimo_MXFP4_MOE_decode.test_mxfp4_weights_are_the_f32_weights_quantized, Mimo_MXFP4_MOE_decode_nr.test_mxfp4_weights_are_the_f32_weights
    mxfp4_ref: 3 failing: Mxfp4Ref.test_block_layout_and_values, Mxfp4Ref.test_dequant_is_gguf_py, Mxfp4Ref.test_e8m0_half_is_two_to_the_e_minus_128
X2 MXFP4: high nibble first
    rc=1 Ran 92 tests in 10.901s | FAILED (failures=5)
    mimo2_quant_parity: 4 failing: Mimo_MXFP4_MOE_decode.test_mxfp4_weights_are_the_f32_weights_quantized, Mimo_MXFP4_MOE_decode_nr.test_mxfp4_weights_are_the_f32_weights
    mxfp4_ref: 1 failing: Mxfp4Ref.test_dequant_is_gguf_py
X3 MXFP4: activations not converted to Q8_0 (vec_dot_type F32)
    rc=1 Ran 92 tests in 10.684s | FAILED (failures=6)
    mimo2_quant_parity: 4 failing: Mimo_MXFP4_MOE_decode.test_matmuls, Mimo_MXFP4_MOE_decode_nr.test_matmuls, Mimo_MXFP4_MOE_pf64.test_matmuls, Mimo_MXFP4_MOE_pf64_nr.tes
    mxfp4_ref: 2 failing: Mxfp4Ref.test_block_layout_and_values, Mxfp4Ref.test_vec_dot_is_the_dot_of_the_dequantized_operands
P1 gguf-py copy: MXFP4 E8M0 scale 2^(e-127) in the dequantize the oracle's weights come from
    rc=1 Ran 92 tests in 11.276s | FAILED (failures=9)
    mimo2_quant_parity: 8 failing: Mimo_MXFP4_MOE_decode.test_matmuls, Mimo_MXFP4_MOE_decode.test_mxfp4_weights_are_the_f32_weights_quantized, Mimo_MXFP4_MOE_decode_nr.te
    mxfp4_ref: 1 failing: Mxfp4Ref.test_dequant_is_gguf_py
caught 19/19
```

```
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6; dumps <S>/n16/work-r1/n16
baseline (no mutation): rc=0 Ran 92 tests in 11.649s | OK
W1 window: a key at distance n_swa still visible (<= instead of <)
    rc=1 Ran 92 tests in 5.545s | FAILED (failures=38)
    mimo2_chunked_ref: 2 failing: Mimo2ChunkedRef.test_attend_is_ref_mimo_attention, Mimo2ChunkedRef.test_deep_rollback_loses_the_window
    mimo2_decode_parity: 33 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
W2 SWA cache without the 256-cell padding (n_swa + n_ubatch cells)
    rc=1 Ran 92 tests in 8.736s | FAILED (failures=18)
    mimo2_decode_parity: 16 failing: Run_decode.test_kv_cells, Run_decode.test_run_exercises_its_mechanisms, Run_evict.test_kv_cells, Run_evict.test_run_exercises_its_mec
    mimo2_mtp_parity: 2 failing: Mtp.test_mtp_block, Mtp.test_run_exercises_its_mechanisms
W3 SWA cells never reused (find_slot takes empty cells only)
    rc=1 Ran 39 tests in 4.836s | FAILED (errors=14)
    mimo2_chunked_ref: 5 failing: Mimo2ChunkedRef.test_cell_model, Mimo2ChunkedRef.test_deep_rollback_loses_the_window, Mimo2ChunkedRef.test_mtp_chunks_equal_the_full_blo
    mimo2_decode_parity: 8 failing: Run_decode.setUpClass, Run_evict.setUpClass, Run_mtp.setUpClass, Run_rollback.setUpClass, Run_split.setUpClass, Run_ub13.setUpClass ..
    mimo2_mtp_parity: 1 failing: Mtp.setUpClass
W4 no purge after an overwrite (KV:1160-1177)
    rc=1 Ran 92 tests in 11.114s | FAILED (failures=1)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_cell_model
W5 RoPE base of the global layers on the sliding-window layers
    rc=1 Ran 92 tests in 6.374s | FAILED (failures=35)
    mimo2_decode_parity: 32 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
S1 sinks ignored
    rc=1 Ran 92 tests in 5.593s | FAILED (failures=36)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_fused_split_value_scale_and_sinks
    mimo2_decode_parity: 32 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
S2 a layer without attn_sinks gets sink logits 0 instead of none
    rc=1 Ran 92 tests in 6.423s | FAILED (failures=32)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_attend_is_ref_mimo_attention
    mimo2_decode_parity: 28 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
R1 rollback length: the first rolled-back position is kept (seq_rm from n_keep + 1)
    rc=1 Ran 92 tests in 10.790s | FAILED (failures=12)
    mimo2_chunked_ref: 2 failing: Mimo2ChunkedRef.test_deep_rollback_loses_the_window, Mimo2ChunkedRef.test_rollback_inside_and_across_the_window
    mimo2_decode_parity: 10 failing: Run_evict.test_attention, Run_evict.test_ffn, Run_evict.test_head_and_logits, Run_evict.test_kv_cells, Run_evict.test_run_exercises_i
R2 rollback frees the base cache only (the SWA cache keeps the rejected tokens)
    rc=1 Ran 92 tests in 9.870s | FAILED (failures=11)
    mimo2_chunked_ref: 2 failing: Mimo2ChunkedRef.test_deep_rollback_loses_the_window, Mimo2ChunkedRef.test_rollback_inside_and_across_the_window
    mimo2_decode_parity: 9 failing: Run_evict.test_attention, Run_evict.test_ffn, Run_evict.test_head_and_logits, Run_evict.test_kv_cells, Run_rollback.test_attention, Ru
R3 seq_rm does not move the head back
    rc=1 Ran 92 tests in 11.139s | FAILED (failures=4)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_cell_model
    mimo2_decode_parity: 3 failing: Run_evict.test_kv_cells, Run_rollback.test_kv_cells, Run_rollback.test_run_exercises_its_mechanisms
V1 value scale ignored
    rc=1 Ran 92 tests in 6.139s | FAILED (failures=32)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_fused_split_value_scale_and_sinks
    mimo2_decode_parity: 28 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
G1 GQA tiled (h % n_kv) instead of grouped (h // r)
    rc=1 Ran 92 tests in 5.843s | FAILED (failures=36)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_attend_is_ref_mimo_attention
    mimo2_decode_parity: 32 failing: Run_decode.test_attention, Run_decode.test_ffn, Run_decode.test_head_and_logits, Run_decode.test_kv_cells, Run_evict.test_attention, 
    mimo2_mtp_parity: 3 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block, Mtp.test_target_h_nextn
M1 MTP: eh_proj input [h ; e] instead of [e ; h]
    rc=1 Ran 92 tests in 10.652s | FAILED (failures=2)
    mimo2_mtp_parity: 2 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block
M2 MTP: head norm output_norm instead of layer_output_norm
    rc=1 Ran 92 tests in 11.010s | FAILED (failures=3)
    mimo2_chunked_ref: 1 failing: Mimo2ChunkedRef.test_mtp_chunks_equal_the_full_block
    mimo2_mtp_parity: 2 failing: Mtp.test_graphs_tokens_and_pairing, Mtp.test_mtp_block
M3 test: MTP token p paired with the target h of p instead of p-1 (full oracle)
    rc=1 Ran 92 tests in 10.959s | FAILED (failures=1)
    mimo2_mtp_parity: 1 failing: Mtp.test_mtp_block
X1 MXFP4: E8M0 scale 2^(e-127) instead of 2^(e-128)
    rc=1 Ran 92 tests in 10.651s | FAILED (failures=7)
    mimo2_quant_parity: 4 failing: Mimo_MXFP4_MOE_decode.test_mxfp4_weights_are_the_f32_weights_quantized, Mimo_MXFP4_MOE_decode_nr.test_mxfp4_weights_are_the_f32_weights
    mxfp4_ref: 3 failing: Mxfp4Ref.test_block_layout_and_values, Mxfp4Ref.test_dequant_is_gguf_py, Mxfp4Ref.test_e8m0_half_is_two_to_the_e_minus_128
X2 MXFP4: high nibble first
    rc=1 Ran 92 tests in 10.984s | FAILED (failures=5)
    mimo2_quant_parity: 4 failing: Mimo_MXFP4_MOE_decode.test_mxfp4_weights_are_the_f32_weights_quantized, Mimo_MXFP4_MOE_decode_nr.test_mxfp4_weights_are_the_f32_weights
    mxfp4_ref: 1 failing: Mxfp4Ref.test_dequant_is_gguf_py
X3 MXFP4: activations not converted to Q8_0 (vec_dot_type F32)
    rc=1 Ran 92 tests in 10.763s | FAILED (failures=6)
    mimo2_quant_parity: 4 failing: Mimo_MXFP4_MOE_decode.test_matmuls, Mimo_MXFP4_MOE_decode_nr.test_matmuls, Mimo_MXFP4_MOE_pf64.test_matmuls, Mimo_MXFP4_MOE_pf64_nr.tes
    mxfp4_ref: 2 failing: Mxfp4Ref.test_block_layout_and_values, Mxfp4Ref.test_vec_dot_is_the_dot_of_the_dequantized_operands
P1 gguf-py copy: MXFP4 E8M0 scale 2^(e-127) in the dequantize the oracle's weights come from
    rc=1 Ran 92 tests in 11.885s | FAILED (failures=9)
    mimo2_quant_parity: 8 failing: Mimo_MXFP4_MOE_decode.test_matmuls, Mimo_MXFP4_MOE_decode.test_mxfp4_weights_are_the_f32_weights_quantized, Mimo_MXFP4_MOE_decode_nr.te
    mxfp4_ref: 1 failing: Mxfp4Ref.test_dequant_is_gguf_py
caught 19/19
```

Every mutant is caught at both pins, 18 of 19 by a dump parity module (decode, MTP or MXFP4); W4 (no purge) by the
property test alone (finding 9). X1 and X2 mutate ref/quant.py's own MXFP4 dequantization, which the products do not use
(they use the pinned gguf-py's, as N10): they are caught by tests/test_mxfp4_ref.py and by the MXFP4 weight check of
tests/test_mimo2_quant_parity.py; P1 mutates the gguf-py copy and test_matmuls catches it. M1 and M3 are caught by the
MTP dump test only, M2 also by the MTP property test.

## Source lines

The pins, file by file (in `<S>`; every line cited above lies outside the hunks of a differing file: mimo2.cpp's hunks
are the loader flags and the h_nextn naming; llama-kv-cache.cpp's are the attention rotation of quantized KV caches
(312-364), the cell token for PLE / 2-D positions (1141), the rotation tensors (1450-1483), a PLE predecessor resolver
(1864-1892) and the state I/O (2251-2581), none in find_slot (898-1095), apply_ubatch's placement, seq_rm (382-420),
get_n_kv (1260-1274) or the mask (1566-1716); ops.cpp's are after the soft_max lines 5896-5912; ggml-cpu.c, repack.cpp
and ggml-quants.c change only outside the cited lines):

```
$ cd <S>; for f in <the files below>; do ... diff -q llama-ec7630a/$f llama.cpp/$f ...; done
differs   src/models/mimo2.cpp (29 changed lines; hunks: 28,36c28,30 104d97 106d98 188c180 239c231 244,246c236,238 248,250c240,241 )
differs   src/llama-kv-cache.cpp (84 changed lines; hunks: 312,313c312,313 321c321 334c334 338a339,356 341,342c359,360 346c364 1141c1159,1161 1450,1460c1470,1471 1471,1480c1482,1483 1864a1868,1870 1868c1874 1886c1892 2251a2258,2259 2543a2552,2553 2546a2557,2558 2559a2572,2581 )
same      src/llama-kv-cache-iswa.cpp
same      src/llama-kv-cells.h
differs   src/llama-hparams.h (4 changed lines; hunks: 67a68 73a75,76 370a374 )
same      src/llama-hparams.cpp
same      conversion/mimo.py
differs   ggml/src/ggml-cpu/ops.cpp (37 changed lines; hunks: 6015,6019c6015,6018 6067,6071c6066 6077,6078c6072,6077 6103,6107c6102 6113,6114c6108,6113 )
same      ggml/src/ggml-cpu/arch/x86/quants.c
same      ggml/src/ggml-cpu/arch/x86/repack.cpp
differs   ggml/src/ggml-cpu/repack.cpp (14 changed lines; hunks: 5241,5246c5241,5248 )
differs   ggml/src/ggml-cpu/ggml-cpu.c (4 changed lines; hunks: 1308c1308 1387c1387 )
same      ggml/src/ggml-cpu/simd-mappings.h
differs   ggml/src/ggml-quants.c (5 changed lines; hunks: 1039,1040c1039,1041 )
same      ggml/src/ggml-common.h
same      ggml/src/ggml-impl.h
same      gguf-py/gguf/quants.py
same      src/llama-quant.cpp
```

src/llama-graph.cpp differs in 407 lines; the hunks inside what mimo2 uses are the MoE-cache plumbing (a `moe_cache`
member, the `slots` argument of build_lora_mm_id, build_moe_cache_slots, which returns nullptr without a MoE cache) and
a GEGLU_ERF case in build_ffn; build_attn_mha (2624-2760) and the iSWA build_attn (3124-3200) lie between the hunks
2461 and 3521 (and the dumps of the two pins are bit-identical, section above):

```
$ diff llama-ec7630a/src/llama-graph.cpp llama.cpp/src/llama-graph.cpp | grep -E '^[0-9]' | tr '\n' ' '
4a5 76c77 82a84,125 89a133,135 94a141,142 131,145c179 354,355c388 1135,1136c1168 1178,1179c1210 1266,1267c1297 
1496a1527 1563,1564c1594,1599 1930a1966,1970 2178a2219,2221 2193c2236 2212c2255 2225c2268 2326c2369 2382a2426,2464 
2384c2466 2392,2399c2474,2486 2401,2403c2488,2493 2405c2495,2508 2407c2510,2511 2409c2513 2410a2515,2524 2423c2537 
2432d2545 2434,2436c2547,2548 2438,2439c2550,2558 2444c2563,2582 2452d2589 2455c2592,2595 2461a2602,2614 3521a3675 
3523d3676 3542,3543c3695,3703 3547c3707 3570,3571c3730,3731 3595c3755,3759 3780,3784c3944 3846,3847c4006 3856c4015 
3857a4017,4018 3859a4021 3861,3864c4023,4026 3866,3867c4028 3882c4043 3890c4051 3898c4059 3906c4067 
$ sed -n 2433,2435p llama.cpp/src/llama-graph.cpp
    if (moe_cache == nullptr) {
        return nullptr;
    }
```

Graph, caches, converter and server, printed from the checkouts (`sed -n '<a>,<b>p' <checkout>/<file>`, leading
blanks stripped, cut at 140 columns; `llama-ec7630a` = ec7630a, `llama.cpp` = 79e2e74):

```
--- llama-ec7630a src/models/mimo2.cpp:14,17
float value_scale = 0.0f;
if (ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, value_scale, false) && value_scale != 1.0f) {
hparams.f_attn_value_scale = value_scale;
}
--- llama-ec7630a src/models/mimo2.cpp:182,185
if (v_scale) {
cur = ggml_scale(ctx0, cur, v_scale);
cb(cur, "attn_out_scaled", il);
}
--- llama-ec7630a src/models/mimo2.cpp:244,246
if (emit_h_nextn) {
cb(cur, "h_nextn", -1);
res->t_h_nextn = cur;
--- llama.cpp src/models/mimo2.cpp:236,238
// pre-norm state for MTP; set even when extraction is off
cb(cur, "h_nextn", -1);
res->t_h_nextn = cur;
--- llama-ec7630a src/models/mimo2.cpp:285,285
GGML_ASSERT(layer.wqkv          && "MIMO2 MTP requires fused attn_qkv");
--- llama-ec7630a src/models/mimo2.cpp:375,375
GGML_ASSERT(layer.ffn_gate && layer.ffn_down && layer.ffn_up && "MIMO2 MTP requires dense FFN tensors");
--- llama-ec7630a src/models/mimo2.cpp:392,394
ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
? layer.nextn.shared_head_norm
: (layer.layer_out_norm ? layer.layer_out_norm : model.output_norm);
--- llama-ec7630a src/llama-kv-cache-iswa.cpp:73,73
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
--- llama-ec7630a src/llama-kv-cache-iswa.cpp:116,117
res = res & kv_base->seq_rm(seq_id, p0, p1);
res = res & kv_swa ->seq_rm(seq_id, p0, p1);
--- llama-ec7630a src/llama-kv-cache.cpp:293,293
ggml_backend_buffer_clear(buf, 0);
--- llama-ec7630a src/llama-kv-cache.cpp:410,419
if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
if (new_head == cells.size()) {
new_head = i;
}
}
}

// If we freed up a slot, set head to it so searching can start there.
if (new_head != cells.size() && new_head < head) {
head = new_head;
--- llama-ec7630a src/llama-kv-cache.cpp:1005,1006
if (head_cur > cells.get_used() + 2*n_tokens) {
head_cur = 0;
--- llama-ec7630a src/llama-kv-cache.cpp:1057,1058
if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
can_use = true;
--- llama-ec7630a src/llama-kv-cache.cpp:1172,1176
if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
__func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
--- llama-ec7630a src/llama-kv-cache.cpp:1184,1184
head = sinfo.idxs[s].back() + 1;
--- llama-ec7630a src/llama-kv-cache.cpp:1265,1270
const uint32_t n_pad_cur = std::max(n_pad, 256u);

for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
const auto & cells = v_cells[sinfo.strm[s]];

result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
--- llama-ec7630a src/llama-kv-cache.cpp:1695,1699
if (swa) {
// see llama_non_causal_type
const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
goto skip;
--- llama-ec7630a src/llama-model.cpp:2808,2811
if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
} else {
filter = [&](uint32_t il) { return il <  hparams.n_layer(); };
--- llama-ec7630a conversion/mimo.py:155,160
hybrid = list(self.hparams["hybrid_layer_pattern"]) + [1] * self._n_nextn
n_head_kv_arr = [n_head_kv_swa if use_swa == 1 else n_head_kv for use_swa in hybrid]
self.gguf_writer.add_head_count_kv(n_head_kv_arr)

self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
self.gguf_writer.add_sliding_window_pattern(hybrid)
--- llama-ec7630a conversion/mimo.py:170,172
v_scale = self.hparams.get("attention_value_scale")
if v_scale is not None:
self.gguf_writer.add_attn_value_scale(float(v_scale))
--- llama.cpp tools/server/server-context.cpp:3674,3677
const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
--- llama.cpp tools/server/server-context.cpp:3726,3726
if (pos_min >= pos_min_thold) {
```

MXFP4 (ec7630a; the same files at 79e2e74, see the list above):

```
--- llama-ec7630a ggml/src/ggml-common.h:214,219
#define QK_MXFP4 32
typedef struct {
uint8_t e; // E8M0
uint8_t qs[QK_MXFP4/2];
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");
--- llama-ec7630a ggml/src/ggml-common.h:1126,1128
GGML_TABLE_BEGIN(int8_t, kvalues_fp4, 16)
0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
GGML_TABLE_END()
--- llama-ec7630a ggml/src/ggml-impl.h:489,501
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
uint32_t bits;

// For x < 2: use precomputed denormal patterns
if (x < 2) {
// 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
bits = 0x00200000 << x;
}
// For x >= 2: normalized exponent adjustment
else {
// 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
bits = (uint32_t)(x - 1) << 23;
}
--- llama-ec7630a ggml/src/ggml-quants.c:577,584
const float d = GGML_E8M0_TO_FP32_HALF(x[i].e);

for (int j = 0; j < qk/2; ++j) {
const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];

y[i*qk + j + 0   ] = x0*d;
y[i*qk + j + qk/2] = x1*d;
--- llama-ec7630a ggml/src/ggml-quants.c:368,368
const uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;
--- llama-ec7630a ggml/src/ggml-cpu/ggml-cpu.c:287,292
[GGML_TYPE_MXFP4] = {
.from_float               = quantize_row_mxfp4,
.vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
.vec_dot_type             = GGML_TYPE_Q8_0,
.nrows                    = 1,
},
--- llama-ec7630a ggml/src/ggml-cpu/simd-mappings.h:129,131
// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#define GGML_CPU_E8M0_TO_FP32_HALF(x) ggml_table_f32_e8m0_half[(uint8_t)(x)]
--- llama-ec7630a ggml/src/ggml-cpu/arch/x86/quants.c:956,960
const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
const __m256 scale0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 0].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 0].e));
const __m256 scale1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 1].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e));
accum1 = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1);
accum2 = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2);
--- llama-ec7630a ggml/src/ggml-cpu/arch/x86/quants.c:991,1000
for (; ib < nb; ++ib) {
const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e);
int sumi1 = 0;
int sumi2 = 0;
for (int j = 0; j < QK_MXFP4/2; ++j) {
sumi1 += y[ib].qs[j +          0] * kvalues_fp4[x[ib].qs[j] & 0xf];
sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_fp4[x[ib].qs[j] >>  4];
}
sumf += d * (sumi1 + sumi2);
}
--- llama-ec7630a ggml/src/ggml-cpu/repack.cpp:5094,5098
} else if (cur->type == GGML_TYPE_MXFP4) {
if (ggml_cpu_has_avx2()) {
if (cur->ne[1] % 8 == 0) {
return &mxfp4_8x8_q8_0;
}
--- llama-ec7630a ggml/src/ggml-cpu/repack.cpp:4839,4841
for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
(void *)               (wdata + i12 * nbw2 + i11 * nbw1),
--- llama-ec7630a src/llama-quant.cpp:512,523
} else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
// MoE   tensors -> MXFP4
// other tensors -> Q8_0
// MLA projection tensors are also 3D, so match expert tensor roles explicitly.
const bool is_bailingmoe3_expert = arch == LLM_ARCH_BAILINGMOE3 &&
(category == tensor_category::FFN_UP ||
category == tensor_category::FFN_GATE ||
category == tensor_category::FFN_DOWN);
if (tensor->ne[2] > 1 && (arch != LLM_ARCH_BAILINGMOE3 || is_bailingmoe3_expert)) {
new_type = GGML_TYPE_MXFP4;
} else {
new_type = GGML_TYPE_Q8_0;
```

## CI and unit tests

The new steps (`grep -n "N16" .github/workflows/*.yml` after `37ab8f4`; both run the driver through the checker on the
checkout `--checkout` names):

```
.github/workflows/ci.yml:49:      - name: N16 mimo2 (MiMo) ubatches, decode past the SWA cache, rollbacks, MTP and MXFP4 experts against the real mimo2 graph (CPU)
.github/workflows/ci.yml-50-        run: python3 tools/ci/check_unittest.py --step n16-mimo2-decode --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_decode.sh ../llama.cpp ../build-tools
--
.github/workflows/parity-r1-pin.yml:66:      - name: N16 mimo2 ubatches, decode past the SWA cache, rollbacks, MTP and MXFP4 experts at the R1 pin (CPU)
.github/workflows/parity-r1-pin.yml-67-        run: python3 tools/ci/check_unittest.py --step n16-mimo2-decode --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_decode.sh ../llama.cpp ../build-tools
```

`tools/ci/parity_expectations.json` step `n16-mimo2-decode`: modules tests.test_mimo2_decode_parity,
tests.test_mimo2_mtp_parity, tests.test_mimo2_chunked_ref, tests.test_mimo2_quant_parity, tests.test_mxfp4_ref; Ran 92,
no skips, at ec7630a and 79e2e74. Local time of the step with the targets built: 24 s at ec7630a, 20 s at 79e2e74
(`final.log` above, wall seconds of the whole wrapped driver from an empty work directory; PROVISIONAL for a GitHub
runner, whose time is UNKNOWN). GitHub Actions: NOT RUN.

`docs/evidence/mutation-scripts/2026-10-11-n16-unit.sh` on the committed HEAD (the one untracked file is this
document):

```
HEAD 8806db588bd050a7a7dd3176271f4f5c378ac8d8
$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests
rc=0
----------------------------------------------------------------------
Ran 952 tests in 135.726s

OK (skipped=438)
$ python3 -m unittest tests.test_check_unittest tests.test_hooks
rc=0
----------------------------------------------------------------------
Ran 54 tests in 13.480s

OK
$ python3 -c "import yaml; ..." (YAML parse of the changed workflows)
.github/workflows/ci.yml parsed; jobs ['cpu-parity', 'ref-torch-cpu', 'cuda-compile', 'strata-glm-sm89', 'lt-strata-sm89'] ; N16 steps [('cpu-parity', 'N16 mimo2 (MiMo) ubatches, decode past the SWA cache, rollbacks, MTP and MXFP4 experts against the real mimo2 graph (CPU)')]
.github/workflows/parity-r1-pin.yml parsed; jobs ['parity', 'ik-glm5next'] ; N16 steps [('parity', 'N16 mimo2 ubatches, decode past the SWA cache, rollbacks, MTP and MXFP4 experts at the R1 pin (CPU)')]
status: 1 changed or untracked files
```
