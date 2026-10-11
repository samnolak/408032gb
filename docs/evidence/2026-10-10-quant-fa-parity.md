# N10: the R1 default precision (f16 KV cache, flash attention) and quantized weights, CPU parity, 2026-10-10 (sandbox CPU, no GPU)

Task N10 of the hardware-free track: close gaps 3 and 2 of docs/reassessment/CORRECTNESS.md section 3. Branch
`claude/n10-quant-fa-parity`, cut from `claude/n9-glm-decode-parity` at `af4db13` (the N9 glm5-next oracle and the N6
qwen4exp decode parity are reused). Role: parity-qa. CPU parity on tiny random-weight models only: nothing here says
anything about speed, GPUs or real weights.

References: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) and
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin, tools/engines/pins.json); CPU builds of `tools/g0` target
`dump-tensors` with `GGML_NATIVE=OFF` (AVX2 and F16C on, AVX-512 off: CMakeCache below; the sandbox CPU has AVX-512,
the build does not use it). "O" = ggml/src/ggml-cpu/ops.cpp, "LG" = src/llama-graph.cpp, "M" =
src/llama-memory-hybrid-idx.cpp, at ec7630a unless "@79e2e74" is written; every cited line was printed from the pinned
checkout (section "Source lines").

Part A (gap 3) is below; part B (gap 2, quantized weights) follows it.

## Part A: `-ctk f16 -ctv f16 -fa on` (gap 3)

The R1 templates run llama.cpp with an f16 KV cache and flash attention on (CORRECTNESS.md gap 3); every oracle
comparison so far ran with `-ctk f32 -ctv f32 -fa off`.

### What llama.cpp rounds on the CPU (CONFIRMED from the source at both pins unless labelled)

| question | answer | where |
|---|---|---|
| which attention goes through flash attention | qwen4exp: both full-attention layers (QSA layer 3 through `build_attn_qsa` -> `build_attn_mha`, dense layer 7 through `build_attn`). glm5-next: the DSA layers on the mask path (`build_attn_mha` with the MLA latent as K and V, `v_mla` = W_v_b applied after the attention); the ec7630a gather path (ubatches of <= 16 tokens) never runs flash attention, it gathers the latents with `ggml_get_rows` and runs f32 `mul_mat` / `soft_max`. The k-pool / lightning indexer scores never go through flash attention (f32 `ggml_lightning_indexer` / `mul_mat`). KDA and GDN layers have no attention | qwen4exp.cpp:996-1002, 919 (`build_attn_mha` in `build_attn_qsa`); glm5-next.cpp:971-1005 (@79e2e74 1074-1078, mask path only); LG:2651 |
| what the KV cache store rounds | every cached row: `cpy_k` / `cpy_v` are `ggml_set_rows`, which calls the F16 `from_float` = `ggml_cpu_fp32_to_fp16` = `_mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT)`: round to nearest even | src/llama-kv-cache.cpp:1328-1361; O:5477, 5491; ggml-cpu.c:222-226, 3492 |
| the indexer / k-pool caches | **affected**: `llama_memory_hybrid_idx` creates the indexer cache with the attention cache types (`-ctk`), so the raw indexer keys (qwen4exp), the key and gate rows (glm5-next) and the pooled keys are stored and read back in f16; pooling reads the cached rows; the scores use the gathered (f16) pooled keys. No Hadamard rotation for f16 (only for quantized cache types) | M:69-72; src/llama-model.cpp:2474-2477 (glm5-next), 2745-2748 (qwen4exp); glm5-next.cpp:790-794, 799-800, 817-824; qwen4exp.cpp:765-768, 772-774, 793-797; src/llama-kv-cache.cpp:321-325 |
| recurrent state | f32 (GGML_TYPE_F32), not affected | src/llama-model.cpp:2483-2484, 2754-2755 |
| f32 cache with `-fa on` | K and V are cast to f16 before `ggml_flash_attn_ext` anyway | LG:2660-2666 |
| CPU flash-attention paths | `use_tiled` for >= GGML_FA_TILE_Q = 64 query rows: q stays f32, K and V widened to f32, f32 GEMM softmax. Otherwise `one_chunk`: **q rounded to f16** (`q_to_vec_dot` = from_float of the vec_dot_type F16), scores from f16 x f16 products, and **the V accumulator VKQ16 is f16**, rounded after every scale (`ggml_vec_scale_f16`) and every cell (`ggml_vec_mad_f16`); the sum S is f32. `ggml_prec_set_acc(cur, GGML_PREC_F32)` does not change it: the CPU dispatcher runs the same function for GGML_PREC_DEFAULT and GGML_PREC_F32 | O:9547-9551, common.h:9; O:9214, 9256, 9319; O:8917-8918, 8961, 8976-8978, 8994-9007, 9031, 9036, 9067-9068; LG:2675; O:9581-9585 |
| split-KV decode path | one query row and n_kv >= 512: the same one_chunk loop per KV chunk (not reached in these runs: n_kv = 256) | O:9490 |
| pins | the CPU flash-attention code (one_chunk .. dispatcher) and `build_attn_mha` are identical at ec7630a and 79e2e74 (diff output below) | |

So the 64-token prefill (pf64) runs the tiled path, every ubatch of fewer than 64 tokens and every decode step the
one_chunk path with an f16 accumulator.

### Pieces

| piece | file |
|---|---|
| emulation of the store and the attention modes | ref/fattn.py: `f16()` (float64 -> float32 -> float16, round to nearest even), `attend()` with modes `exact`, `mm16` (flash attention off, f16 cache: q and the probabilities in f16), `fa_chunk` (one_chunk, float32 steps mimicked, f16 accumulator), `fa_tiled`; `precision(kv, fa)`, `store()`, `row_modes()`, `mm16_bound()` |
| precision-aware oracles | ref/qwen4exp.py, ref/glm5next.py, ref/dsa.py: `prec=` on `forward`, `forward_chunk`, `cache_truncate`, `forward_chunks`, `mtp_forward_chunk` (None keeps the f32 contract unchanged); rows returned as stored (`stored`); MoE overrides for replays (`("moe", il)`); ref/router.py `moe_tie_variant` |
| tests | tests/test_qwen4exp_f16fa_parity.py, tests/test_glm5next_f16fa_parity.py (the N6 / N9 test classes, subclassed, plus the f16 checks), shared tests/f16fa.py; tests/test_fattn_ref.py (11 property tests, numpy only, run in the CI unit step) |
| N6 / N9 bases | tests/test_qwen4exp_decode_parity.py, tests/test_glm5next_decode_parity.py: dump root, model, precision, tolerance and tie tolerances as class attributes; tie replays in computation order (f32 behaviour unchanged: below) |
| driver | tools/tinygen/run_parity_f16fa.sh (`PARITY_KV`, `PARITY_FA` choose the other configurations) |
| dump tool | tools/dump/dump_tensors.cpp: the .json sidecar also records `ggml_type` (an f16 cache view is widened to f32 in the .bin) |
| mutation check | docs/evidence/mutation-scripts/2026-10-10-n10-f16fa.py |
| probes | docs/evidence/mutation-scripts/2026-10-10-n10-logprob-probe.py (llama.cpp f16/FA against llama.cpp f32), 2026-10-10-n10-margin-probe.py (how close the discrete choices are to a tie) |
| CI | step "N10 ..." after N9 in `.github/workflows/ci.yml` job `cpu-parity` (ec7630a) and in `.github/workflows/parity-r1-pin.yml` (R1 pin) |

### Runs

Models: the N6 tiny qwen4exp (tools/tinygen/qwen4exp_tiny.py, seed 0) and the G1 / N9 tiny glm5-next
(tools/tinygen/glm5next_tiny.py --layers 12, seed 0); the N6 and N9 token sequences (64 ids). Flags
`-c 512 -b 512 -t 1 -ctk f16 -ctv f16 -fa on`. Runs per model: `pf64` (one 64-token ubatch: the N2 / G1 prefill shape,
tiled flash attention), `ub16`, `ub10`, (glm5-next `ub18`), `decode` (48 + 16 x 1) and `rollback` (n_rs_seq 3), the call
plans of N6 / N9. Attribution runs with the same driver: `PARITY_KV=f32 PARITY_FA=on`, `PARITY_KV=f16 PARITY_FA=off`,
`PARITY_KV=f32 PARITY_FA=off`. MTP runs at f16 / FA: NOT RUN (the oracle takes `prec`, no driver or test for it).

### Method

Every N6 / N9 check (both oracles: (A) full sequence, (B) incremental replay) runs on the f16 dumps with the oracle at
the run's precision: the values rounded where llama.cpp stores them, the attention in the mode of the graph
(`graph_mode`: one_chunk below 64 rows, tiled at 64, the ec7630a gather path "exact" over f16 latents; a rollback
rebuilds the kept prefix in the modes it had). Added checks:

* cache store, exact: the rows a graph wrote into the K / V, MLA-latent and indexer caches (the `set_rows` outputs,
  dumped as `cache_*_l<il> (view)`, ggml_type f16) equal f16 of that graph's own dumped pre-store values (Kcur, Vcur,
  indexer_k_raw, kv_cmpr, indexer_k, indexer_gate, indexer_pool_k_new), every element;
* cache rows against the rows the oracle stored, per cached quantity at least half of the elements equal;
* flash-attention op on llama.cpp's own inputs (dumped q, the K/V views the attention read, the cells of the dumped
  mask) against the dumped output: one_chunk <= 1 binary16 spacing per element (glm5-next: after W_v_b, the bound
  sum_l |W_v_b| spacing(O_l)); tiled 1e-5 normwise; `mm16` (flash attention off) one spacing per f16 probability;
* path: flash attention (no `kq_soft_max` node; glm5-next `fattn_mla`) iff `-fa on`; the cache types;
* the plain f32 oracle on the same dumps: every N6 / N9 comparison recorded (never asserted), its discrete choices
  counted, and the precision-aware oracle at least twice as close to the attention output in the 2-norm.

Tolerances (U16 = 2^-11, the unit roundoff of binary16):
* TOL = 16 U16 = 7.8e-3 normwise downstream of the first attention; 1e-4 (N6 / N9) for every value of the layers
  before it and for what the first attention layer computes before it reads the cache. Justification: the oracle
  rounds where llama.cpp rounds, but its float64 values and llama.cpp's f32 values differ by f32 errors, and wherever
  the two fall on different sides of a binary16 rounding boundary they end one spacing (<= 2 U16 relative) apart. The
  one_chunk accumulator is rounded at every cell, so the attention output carries a rounding noise of a few spacings
  that no oracle short of bit-equal f32 arithmetic removes (measured below: 1.36e-3 = 2.8 U16 at the qwen4exp dense
  layer). TOL allows two spacings (4 U16) of it, grown by up to 4 through the rest of the model. Measured worst value
  anywhere: 2.947e-3 = 6.0 U16 (glm5-next, 79e2e74, kda_scan_out of layer 9), a margin of 2.6.
* QSA / DSA selections and MoE expert sets: equal, or a valid top-k within TOL of the oracle's own scores, replayed and
  counted (N6 / N9 keep 1e-6 and never replay experts).
* cache: exact store; >= 50 % equal per quantity (measured >= 0.83).

### Findings (CONFIRMED by the literal outputs below unless labelled)

1. **No llama.cpp-vs-oracle mismatch at the R1 default precision, at either pin.** With the precision-aware oracle,
   `run_parity_f16fa.sh` passes 170/170 at ec7630a and 170/170 at 79e2e74 (qwen4exp 75 tests: 5 runs x 15, glm5-next
   84: 6 runs x 14, ref/fattn.py 11). Worst normwise error against the precision-aware oracle: qwen4exp 1.355e-3
   (attention output of the dense layer 7, rollback; head 2.2e-5), glm5-next 2.60e-3 at ec7630a / 2.95e-3 at
   79e2e74 (head 8.1e-4 / 8.4e-4). The other three precision configurations pass as well (170/170 each, both pins).
2. **Flash attention on the CPU has two numerically different paths and rounds more than the cache.** Below 64 query
   rows (every decode step, every ubatch of these runs except pf64) the one_chunk path rounds q to f16 and keeps the
   V accumulator in f16; at 64 rows and more the tiled path is f32 over f16 K / V. On llama.cpp's own inputs
   ref/fattn.py reproduces both: one_chunk with at most one element per graph and layer more than half a binary16
   spacing off (`1/8192` elements, the rest within half a spacing; glm5-next within 0.078 of the one-spacing bound),
   tiled to 1.3e-7 normwise. The same inputs through a float64 attention (the f32 contract) differ by 5e-4 to 9e-4
   normwise (first probe, below). `-fa on` with an f32 cache rounds K / V as well (LG:2660-2666).
3. **`-ctk f16` also rounds the indexer caches** (k-pool raw keys and pooled keys, glm5-next key | gate | pooled), not
   only K / V or the MLA latent: the cache tensors are f16 (`ggml_type` in the dumps) and every stored element equals
   f16 (round to nearest even) of llama.cpp's own pre-store value (9472 qwen4exp / 9984 glm5-next elements per run, 0
   differences, both pins). The QSA / DSA scores therefore use f16 pooled keys.
4. **The f16 accumulator sets a noise floor at the attention output.** Since the rounding of each accumulator step
   follows the smallest change of its inputs, the oracle's attention output differs from llama.cpp's by a noise of
   about one spacing even with every rounding site modelled: qwen4exp 2-norm 1e-5 to 4e-5 (99 % of the elements
   within half a spacing), glm5-next 3.1e-4 to 3.6e-4 in the one_chunk runs (67 % to 72 %) where upstream f32 errors
   are larger (KDA, mHC). The plain f32 oracle is 3.2x to 25x further away (2-norm, both models, both pins; about
   1000x in the two runs where it flips an expert); the test requires 2x.
5. **N7 finding 7 explained: one MoE near-tie flip.** llama.cpp at f16 / FA against llama.cpp at f32 / no FA (same
   model, tokens, call plan; `2026-10-10-n10-logprob-probe.py`): qwen4exp max |d log-softmax| 2.8e-4, glm5-next
   2.7e-3 (pf64) to 1.15e-2 with no discrete choice changed, and **0.603** in ec7630a ub16 / ub10, where exactly one
   MoE decision flipped (layer 7, position 60: experts {4, 5} vs {5, 7}) and everything after it follows (MoE layers
   9 and 10, the DSA selection of layer 11 at position 60). That is the size of N7's 0.416 to 0.86. The flipped
   decision is a near-tie: margin 7.7e-5 between the 2nd and 3rd expert in the plain f32 oracle, 2.9e-6 in the
   precision-aware one, which llama.cpp then flips; the test replays it (`moe experts: tie replays = 1`) and the rest of
   the run matches within TOL. No DSA / QSA selection flipped on its own: the only selection difference is downstream
   of the MoE flip (13 glm5-next selections per run are exact ties at score 0.0: all heads' relu zero, their order is
   ggml_top_k's, not precision). Margin probe (`2026-10-10-n10-margin-probe.py`): f16 / FA moves glm5-next's MoE
   selection margins by a median 2.5e-5 to 4.9e-5 (max 1.7e-3), qwen4exp's by 6.4e-7 to 1.4e-6 (max 2.3e-5); 6 of 704
   glm5-next and 19 of 512 qwen4exp MoE decisions have a margin below 1e-3. PROVISIONAL bound from these (margins
   uniform near 0 at that density): about 2e-4 to 4e-4 flips per glm5-next MoE decision, so about one flip in a
   288-token run of 11 MoE layers (N7's 256-token prompt + 32 steps), and about 0.1 for qwen4exp (8 MoE layers).
   Whether real GLM-5.3-Flash weights have that margin density: UNKNOWN (R2.1 / K2 must measure it).
6. **Attribution** (the head, normwise, against the plain f32 oracle; first table of each run): glm5-next 3.3e-3 at
   f32 / FA on and 3.5e-3 at 79e2e74 f16 / FA on, but 0.19 wherever the MoE flip happens (ec7630a with an f16 cache:
   f16 / FA on, f16 / FA off); qwen4exp 1.4e-4 to 1.9e-4 in every configuration that rounds anything. At ec7630a, f32 / FA on leaves ub16 and
   ub10 of glm5-next untouched (the gather path has no flash attention and the cache is f32).
7. **Mutation check** (literal output below): ec7630a 14/14 caught, 79e2e74 13/14; the miss (T2, no tie replay) is a
   mutant of a mechanism the 79e2e74 runs do not exercise (no near-tie flips there). A first round with the earlier
   test version caught 12/14 and 11/14: removing the f16 store from the oracle (O1) was caught only by a property
   test, the qwen4exp indexer-key rounding (O6) and a 2^-9 error in one latent the tiled attention read (D4) not at
   all. Fixes: the cache test compares llama.cpp's cache with the rows the oracle stored (not with its own
   re-rounding) and asserts the share per quantity; `F16Mixin.close` had replaced explicit op-level tolerances
   (1e-5) by TOL. The table below is the second round on the committed code.

### Not covered (gap 3)

* split-KV flash attention (one query row, n_kv >= 512): NOT RUN (n_kv = 256 here); numerically one_chunk per chunk
  plus a reduction (O:9490), not checked.
* `-t` > 1 (one_chunk chunks across threads, tiled tiles): NOT RUN, every run is `-t 1`.
* MTP contexts at f16 / FA: NOT RUN. Quantized KV types (`-ctk q8_0`, Hadamard rotation): NOT RUN.
* f16 dot products with DK, DV >= 32 (SIMD f16 paths with f32 lane sums): NOT RUN, the tiny heads are 16 wide
  (qwen4exp head_dim 16, glm5-next kv_lora 16); the emulation follows the scalar leftovers.
* the GPU (CUDA) flash attention, which R1 actually uses for the attention layers: NOT RUN (no GPU).

### Literal outputs (part A)

Pin check, environment, builds:

```
$ git -C <scratch>/llama-ec7630a rev-parse HEAD; git -C <scratch>/llama.cpp rev-parse HEAD
ec7630a640789c393694fb194f1bbbf0369fc62d
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
$ nproc; python3 --version; python3 -c "import numpy; print(numpy.__version__)"; nvidia-smi
4
Python 3.13.16
2.5.3
/bin/bash: line 1: nvidia-smi: command not found
$ grep -E "GGML_AVX2:|GGML_AVX512:|GGML_NATIVE:|GGML_F16C:|GGML_FMA:|CMAKE_BUILD_TYPE:" <scratch>/n10/build-{ec,r1}/CMakeCache.txt
<scratch>/n10/build-ec/CMakeCache.txt:CMAKE_BUILD_TYPE:STRING=Release
<scratch>/n10/build-ec/CMakeCache.txt:GGML_AVX2:BOOL=ON
<scratch>/n10/build-ec/CMakeCache.txt:GGML_AVX512:BOOL=OFF
<scratch>/n10/build-ec/CMakeCache.txt:GGML_F16C:BOOL=ON
<scratch>/n10/build-ec/CMakeCache.txt:GGML_FMA:BOOL=ON
<scratch>/n10/build-ec/CMakeCache.txt:GGML_NATIVE:BOOL=OFF
<scratch>/n10/build-r1/CMakeCache.txt:CMAKE_BUILD_TYPE:STRING=Release
<scratch>/n10/build-r1/CMakeCache.txt:GGML_AVX2:BOOL=ON
<scratch>/n10/build-r1/CMakeCache.txt:GGML_AVX512:BOOL=OFF
<scratch>/n10/build-r1/CMakeCache.txt:GGML_F16C:BOOL=ON
<scratch>/n10/build-r1/CMakeCache.txt:GGML_FMA:BOOL=ON
<scratch>/n10/build-r1/CMakeCache.txt:GGML_NATIVE:BOOL=OFF
```

(`<scratch>` = `/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`.)

### Source lines (printed from the pinned checkouts)

At ec7630a (`cd <scratch>/llama-ec7630a`):

```
$ sed -n "8917,8918p;8961p;8976,8978p;8994,9007p;9031p;9036p;9067,9068p" ggml/src/ggml-cpu/ops.cpp   # @ec7630a, one_chunk
    ggml_type         const k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
        q_to_vec_dot(pq, Q_q, DK);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value
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
            S = S*ms + vs; // scale and increment sum with partial sum
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);
$ sed -n "9547,9551p;9581,9585p" ggml/src/ggml-cpu/ops.cpp; sed -n "9p" ggml/src/ggml-cpu/common.h
        bool use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
        case GGML_PREC_DEFAULT:
        case GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
#define GGML_FA_TILE_Q  64
$ sed -n "9214p;9256p;9319p" ggml/src/ggml-cpu/ops.cpp   # tiled: q f32, K and V widened
                memcpy(Q_f32 + tq * DK, pq, DK * sizeof(float));
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)v_data, V32 + tk * DV, DV);
$ sed -n "2651p;2660,2666p;2675p" src/llama-graph.cpp
    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }
        ggml_prec_set_acc(cur, GGML_PREC_F32);
$ sed -n "222,226p;3492p" ggml/src/ggml-cpu/ggml-cpu.c
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
$ sed -n "69,72p" src/llama-memory-hybrid-idx.cpp; sed -n "2474,2477p;2483,2484p;2745,2748p;2754,2755p" src/llama-model.cpp

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
                res = new llama_memory_hybrid_idx(
                    /* model             */ *this,
                    /* attn_type_k       */ params.type_k,
                    /* attn_type_v       */ params.type_v,
                    /* recurrent_type_r  */ GGML_TYPE_F32,
                    /* recurrent_type_s  */ GGML_TYPE_F32,
                        res = new llama_memory_hybrid_idx(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
$ sed -n "321,325p" src/llama-kv-cache.cpp   # rotation only for quantized cache types
        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;
$ sed -n "790,794p;799,800p;817,824p;966,968p;971,972p;998,1005p" src/models/glm5-next.cpp
    // Cache rows store key | gate | pooled
    ggml_tensor * pzero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_indexer, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_concat(ctx0, ggml_concat(ctx0, ik, ig, 0), pzero, 0);
    packed = ggml_reshape_3d(ctx0, packed, 3*n_embd_indexer, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));
    // Pool the entries completed by this ubatch. The last one is a dummy when the ubatch completes none.
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    if (inp_kpool->cache_safe) {
        // Write before the pool gather.
        ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
    }

    ggml_tensor * pooled = nullptr;
    if (inp_kpool->cache_safe) {
        pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    ggml_build_forward_expand(gf, q_absorbed);
    ggml_build_forward_expand(gf, kv_cmpr);
    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv_cmpr, inp_attn->get_k_idxs(), il));
    if (inp_kpool->gather) {
        // Attend over gathered latents with the token dimension in ne[3].
        // The scatter selection already includes the causal mask.
        ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
        cb(mask, "kq_mask_dsa", il);

        ggml_tensor * k = mctx_mla->get_k(ctx0, il);
        ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

        out = build_attn_mha(q_absorbed, k, v, nullptr, mask, nullptr, layer.wv_b, inp_kpool->n_sel, kq_scale, il);
$ sed -n "765,768p;772,774p;793,797p;906,907p" src/models/qwen4exp.cpp

    ggml_tensor * pzero  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k_raw, pzero, 0), 2*idx_dim, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    // pool only the blocks this ubatch completes or regroups
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    ggml_tensor * pooled = nullptr;
    if (inp_kpool->cache_safe) {
        // write before the pool gather
        ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
        pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
```

At 79e2e74 (`cd <scratch>/llama.cpp`; the two diffs compare the pins):

```
$ diff <(sed -n "/static void ggml_compute_forward_flash_attn_ext_f16_one_chunk/,/^void ggml_compute_forward_flash_attn_ext(/p" <ec7630a>/ggml/src/ggml-cpu/ops.cpp) <(same @79e2e74) && echo FA-IDENTICAL
FA-IDENTICAL
$ diff <(build_attn_mha @ec7630a) <(build_attn_mha @79e2e74) && echo MHA-IDENTICAL
MHA-IDENTICAL
$ sed -n "1060p;1074,1078p" src/models/glm5-next.cpp   # @79e2e74: mask path only
        sel = build_kpool_select(cur, qr, kq_mask, layer, mctx_hyb, inp_kpool, il);

    ggml_tensor * k = mctx_mla->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * out = build_attn_mha(q_absorbed, k, v, nullptr, mask, nullptr, layer.wv_b, inp_kpool->n_sel, kq_scale, il);
```

### Driver at the R1 default precision, both pins

Literal, on HEAD `27cbb25` (code as committed; the later commits change docstrings and docs only). Cut from each
log: the first three lines (`CMAKE_BUILD_TYPE=Release`, two `wrote .../*.gguf`), the per-test lines of
`unittest -v`, the first table of each model (one row per run and component: 320 rows qwen4exp, 550 / 542
glm5-next; its four largest rows are shown under "largest errors"), the `tie replays = 0` summary rows, the
discrete-choice rows that are 0 (counted in the last line of that block), the last line `artifacts: <dir>`, and
the `time` lines. `docs/evidence/mutation-scripts/2026-10-10-n10-extract-tables.py <log>` printed the tables.

```
$ F16FA_WORK=<scratch>/n10/work-ec tools/tinygen/run_parity_f16fa.sh <scratch>/llama-ec7630a <scratch>/n10/build-ec
q pf64: 1 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q ub16: 4 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q ub10: 7 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q decode: 17 graphs, 17 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q rollback: 16 graphs, 17 calls, n_rs_seq 3 (-ctk f16 -ctv f16 -fa on)
g pf64: 1 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g ub16: 4 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g ub10: 7 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g ub18: 4 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g decode: 17 graphs, 17 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g rollback: 16 graphs, 17 calls, n_rs_seq 3 (-ctk f16 -ctv f16 -fa on)
Ran 170 tests in 22.345s
OK
```

```
--- qwen4exp
qwen4exp: worst error per layer (layer -1 = embedding and head), precision-aware oracle | plain f32 oracle
layer    max abs        rel component (aware)                                  |    max abs        rel component (f32)
   -1  2.725e-05  2.210e-05 rollback result_output [full]                      |  2.661e-04  1.860e-04 ub10 result_norm [full]
    0  1.114e-06  1.035e-06 rollback linear_attn_out [full]                    |  1.114e-06  1.035e-06 rollback linear_attn_out [full]
    1  2.210e-06  1.677e-06 rollback final_output (gated norm) [chunk]         |  2.210e-06  1.677e-06 rollback final_output (gated norm) [chunk]
    2  3.549e-06  1.759e-06 ub16 final_output (gated norm) [chunk]             |  3.549e-06  1.759e-06 ub16 final_output (gated norm) [chunk]
    3  9.958e-05  2.385e-04 decode attn_pregate (qsa) [full]                   |  9.124e-04  1.256e-03 ub10 attn_pregate (qsa) [full]
    4  2.841e-07  4.931e-06 decode gdn scan (attn_output) [full]               |  8.049e-04  4.117e-04 ub16 final_output (gated norm) [full]
    5  1.962e-05  9.759e-06 ub10 final_output (gated norm) [chunk]             |  8.810e-04  4.382e-04 ub10 final_output (gated norm) [full]
    6  1.042e-05  8.981e-06 rollback linear_attn_out [chunk]                   |  1.406e-03  6.538e-04 ub16 final_output (gated norm) [full]
    7  3.381e-04  1.355e-03 rollback attn_pregate (dense causal) [full]        |  5.363e-04  2.551e-03 decode attn_pregate (dense causal) [chunk]
qwen4exp: summaries (cache store and cache rows, flash-attention op, attention output, tie replays)
decode   attention output: 2-norm rel 3.668e-05, within 0.5 spacing 0.9912 of 8192 (aware)
decode   attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
decode   cache rows equal to the oracle's stored rows: 140305/140800; lowest indexer pooled key 3695/3712
decode   cache store: 9472 elements equal to the rounded pre-store value
decode   flash attention op: elements > 0.5 spacing 1/8192
pf64     attention output: 2-norm rel 2.369e-04, within 0.5 spacing 0.6437 of 8192 (plain)
pf64     attention output: 2-norm rel 9.601e-06, within 0.5 spacing 0.9930 of 8192 (aware)
pf64     cache rows equal to the oracle's stored rows: 9446/9472; lowest K 4079/4096
pf64     cache store: 9472 elements equal to the rounded pre-store value
rollback attention output: 2-norm rel 4.289e-05, within 0.5 spacing 0.9893 of 8448 (aware)
rollback attention output: 2-norm rel 5.675e-04, within 0.5 spacing 0.3456 of 8448 (plain)
rollback cache rows equal to the oracle's stored rows: 133361/133856; lowest indexer pooled key 3520/3536
rollback cache store: 9776 elements equal to the rounded pre-store value
rollback flash attention op: elements > 0.5 spacing 1/8448
ub10     attention output: 2-norm rel 4.243e-05, within 0.5 spacing 0.9902 of 8192 (aware)
ub10     attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
ub10     cache rows equal to the oracle's stored rows: 40386/40528; lowest indexer pooled key 1066/1072
ub10     cache store: 9472 elements equal to the rounded pre-store value
ub10     flash attention op: elements > 0.5 spacing 1/8192
ub16     attention output: 2-norm rel 4.243e-05, within 0.5 spacing 0.9902 of 8192 (aware)
ub16     attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
ub16     cache rows equal to the oracle's stored rows: 23598/23680; lowest indexer pooled key 637/640
ub16     cache store: 9472 elements equal to the rounded pre-store value
ub16     flash attention op: elements > 0.5 spacing 1/8192
qwen4exp: rows where an oracle's discrete choice differs from llama.cpp (the precision-aware oracle after its tie replays, listed in the first table), and N6/N9 checks the plain f32 oracle fails (assertion: ...)
(and 30 rows with 0, not shown)
largest errors against the precision-aware oracle (first table, 320 rows):
rollback attn_pregate (dense causal) [full]                             7     9  1.355e-03  2.551e-03
rollback attn_pregate (dense causal) [chunk]                            7     9  1.355e-03  2.551e-03
ub10     attn_pregate (dense causal) [full]                             7     5  1.142e-03  1.922e-03
ub10     attn_pregate (dense causal) [chunk]                            7     5  1.142e-03  1.922e-03
--- glm5-next
glm5-next: worst error per layer (layer -1 = embedding and head), precision-aware oracle | plain f32 oracle
layer    max abs        rel component (aware)                                  |    max abs        rel component (f32)
   -1  2.793e-03  8.071e-04 decode result_norm [full]                          |  5.475e-01  1.931e-01 ub10 result_norm [full]
    0  1.929e-06  1.590e-06 decode kda_out [full]                              |  1.929e-06  1.590e-06 decode kda_out [full]
    1  1.571e-07  3.450e-06 rollback kda_scan_out [full]                       |  1.571e-07  3.450e-06 rollback kda_scan_out [full]
    2  1.491e-06  4.676e-06 ub10 ffn_shexp (shared, clamped) [chunk]           |  1.491e-06  4.676e-06 ub10 ffn_shexp (shared, clamped) [chunk]
    3  4.883e-04  2.621e-04 decode indexer pooled key (cache rows vs oracle)   |  7.601e-05  1.289e-03 decode indexer_score (visible) [chunk]
    4  1.855e-04  1.856e-04 ub18 hc_comb (attn, Sinkhorn) [chunk]              |  3.070e-03  1.535e-03 decode hc_post (ffn) [full]
    5  8.998e-05  2.780e-04 ub18 ffn_shexp (shared, clamped) [chunk]           |  7.591e-03  7.592e-03 decode hc_comb (ffn, Sinkhorn) [full]
    6  8.938e-04  1.135e-03 decode kda_out [chunk]                             |  1.729e-02  8.647e-03 ub18 hc_post (ffn) [full]
    7  1.953e-03  6.605e-04 decode indexer gate (cache rows vs oracle)         |  7.729e-01  1.431e-01 ub10 l_out (layer output streams) [chunk]
    8  4.002e-05  1.025e-03 decode kda_scan_out [full]                         |  2.968e-02  3.178e-01 ub10 kda_scan_out (first row, reads the carried state) [full]
    9  2.400e-03  1.200e-03 ub18 hc_post (ffn) [full]                          |  2.076e-02  2.313e-01 ub10 kda_scan_out [full]
   10  8.416e-04  2.599e-03 ub18 ffn_shexp (shared, clamped) [full]            |  7.162e-01  3.582e-01 ub10 hc_post (attn) [chunk]
   11  1.257e-03  1.751e-03 ub18 ffn_out (routed + shared) [full]              |  1.057e+00  6.868e-01 ub10 attn_out (wo) [chunk]
glm5-next: summaries (cache store and cache rows, flash-attention op, attention output, tie replays)
decode   attention output: 2-norm rel 1.073e-03, within 0.5 spacing 0.3162 of 12288 (plain)
decode   attention output: 2-norm rel 3.058e-04, within 0.5 spacing 0.7198 of 12288 (aware)
decode   cache rows equal to the oracle's stored rows: 128600/148224; lowest indexer pooled key 9269/11136
decode   cache store: 9984 elements equal to the rounded pre-store value
decode   flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
pf64     attention output: 2-norm rel 4.029e-04, within 0.5 spacing 0.5083 of 12288 (plain)
pf64     attention output: 2-norm rel 4.537e-05, within 0.5 spacing 0.9625 of 12288 (aware)
pf64     cache rows equal to the oracle's stored rows: 9711/9984; lowest indexer pooled key 742/768
pf64     cache store: 9984 elements equal to the rounded pre-store value
rollback attention output: 2-norm rel 1.073e-03, within 0.5 spacing 0.3161 of 12672 (plain)
rollback attention output: 2-norm rel 3.056e-04, within 0.5 spacing 0.7196 of 12672 (aware)
rollback cache rows equal to the oracle's stored rows: 122230/140928; lowest indexer pooled key 8833/10608
rollback cache store: 10320 elements equal to the rounded pre-store value
rollback flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
ub10     attention output: 2-norm rel 4.660e-05, within 0.5 spacing 0.9613 of 12288 (aware)
ub10     attention output: 2-norm rel 4.732e-02, within 0.5 spacing 0.5008 of 12288 (plain)
ub10     cache rows equal to the oracle's stored rows: 41740/42672; lowest indexer pooled key 3110/3216
ub10     cache store: 9984 elements equal to the rounded pre-store value
ub10     moe experts: tie replays = 1
ub16     attention output: 2-norm rel 4.660e-05, within 0.5 spacing 0.9613 of 12288 (aware)
ub16     attention output: 2-norm rel 4.732e-02, within 0.5 spacing 0.5008 of 12288 (plain)
ub16     cache rows equal to the oracle's stored rows: 24411/24960; lowest indexer pooled key 1855/1920
ub16     cache store: 9984 elements equal to the rounded pre-store value
ub16     moe experts: tie replays = 1
ub18     attention output: 2-norm rel 1.100e-03, within 0.5 spacing 0.2983 of 12288 (plain)
ub18     attention output: 2-norm rel 3.249e-04, within 0.5 spacing 0.7054 of 12288 (aware)
ub18     cache rows equal to the oracle's stored rows: 23347/26784; lowest indexer pooled key 1693/2016
ub18     cache store: 9984 elements equal to the rounded pre-store value
ub18     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
glm5-next: rows where an oracle's discrete choice differs from llama.cpp (the precision-aware oracle after its tie replays, listed in the first table), and N6/N9 checks the plain f32 oracle fails (assertion: ...)
ub10     DSA selection (plain f32 oracle)                                1
ub10     MoE experts (plain f32 oracle)                                  3
ub10     assertion: DSA cells                                            1
ub10     assertion: DSA cells (full)                                     1
ub10     assertion: test_moe_and_ffn raised KeyError                     1
ub10     assertion: top-k                                                1
ub10     assertion: top-k (full)                                         1
ub16     DSA selection (plain f32 oracle)                                1
ub16     MoE experts (plain f32 oracle)                                  3
ub16     assertion: DSA cells                                            1
ub16     assertion: DSA cells (full)                                     1
ub16     assertion: test_moe_and_ffn raised KeyError                     1
ub16     assertion: top-k                                                1
ub16     assertion: top-k (full)                                         1
(and 32 rows with 0, not shown)
largest errors against the precision-aware oracle (first table, 550 rows):
ub18     ffn_shexp (shared, clamped) [full]                            10     1  2.599e-03  7.340e-03
ub18     ffn_shexp (shared, clamped) [chunk]                           10     1  2.599e-03  7.340e-03
rollback ffn_shexp (shared, clamped) [full]                            10     0  2.328e-03  5.784e-03
rollback ffn_shexp (shared, clamped) [chunk]                           10     0  2.328e-03  5.784e-03
```

```
$ F16FA_WORK=<scratch>/n10/work-r1 tools/tinygen/run_parity_f16fa.sh <scratch>/llama.cpp <scratch>/n10/build-r1
q pf64: 1 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q ub16: 4 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q ub10: 7 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q decode: 17 graphs, 17 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
q rollback: 16 graphs, 17 calls, n_rs_seq 3 (-ctk f16 -ctv f16 -fa on)
g pf64: 1 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g ub16: 4 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g ub10: 7 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g ub18: 4 graphs, 1 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g decode: 17 graphs, 17 calls, n_rs_seq 0 (-ctk f16 -ctv f16 -fa on)
g rollback: 16 graphs, 17 calls, n_rs_seq 3 (-ctk f16 -ctv f16 -fa on)
Ran 170 tests in 21.347s
OK
```

```
--- qwen4exp
qwen4exp: worst error per layer (layer -1 = embedding and head), precision-aware oracle | plain f32 oracle
layer    max abs        rel component (aware)                                  |    max abs        rel component (f32)
   -1  2.725e-05  2.210e-05 rollback result_output [full]                      |  2.661e-04  1.860e-04 ub10 result_norm [full]
    0  1.114e-06  1.035e-06 rollback linear_attn_out [full]                    |  1.114e-06  1.035e-06 rollback linear_attn_out [full]
    1  2.210e-06  1.677e-06 rollback final_output (gated norm) [chunk]         |  2.210e-06  1.677e-06 rollback final_output (gated norm) [chunk]
    2  3.549e-06  1.759e-06 ub16 final_output (gated norm) [chunk]             |  3.549e-06  1.759e-06 ub16 final_output (gated norm) [chunk]
    3  9.958e-05  2.385e-04 decode attn_pregate (qsa) [full]                   |  9.124e-04  1.256e-03 ub10 attn_pregate (qsa) [full]
    4  2.841e-07  4.931e-06 decode gdn scan (attn_output) [full]               |  8.049e-04  4.117e-04 ub16 final_output (gated norm) [full]
    5  1.962e-05  9.759e-06 ub10 final_output (gated norm) [chunk]             |  8.810e-04  4.382e-04 ub10 final_output (gated norm) [full]
    6  1.042e-05  8.981e-06 rollback linear_attn_out [chunk]                   |  1.406e-03  6.538e-04 ub16 final_output (gated norm) [full]
    7  3.381e-04  1.355e-03 rollback attn_pregate (dense causal) [full]        |  5.363e-04  2.551e-03 decode attn_pregate (dense causal) [chunk]
qwen4exp: summaries (cache store and cache rows, flash-attention op, attention output, tie replays)
decode   attention output: 2-norm rel 3.668e-05, within 0.5 spacing 0.9912 of 8192 (aware)
decode   attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
decode   cache rows equal to the oracle's stored rows: 140305/140800; lowest indexer pooled key 3695/3712
decode   cache store: 9472 elements equal to the rounded pre-store value
decode   flash attention op: elements > 0.5 spacing 1/8192
pf64     attention output: 2-norm rel 2.369e-04, within 0.5 spacing 0.6437 of 8192 (plain)
pf64     attention output: 2-norm rel 9.601e-06, within 0.5 spacing 0.9930 of 8192 (aware)
pf64     cache rows equal to the oracle's stored rows: 9446/9472; lowest K 4079/4096
pf64     cache store: 9472 elements equal to the rounded pre-store value
rollback attention output: 2-norm rel 4.289e-05, within 0.5 spacing 0.9893 of 8448 (aware)
rollback attention output: 2-norm rel 5.675e-04, within 0.5 spacing 0.3456 of 8448 (plain)
rollback cache rows equal to the oracle's stored rows: 133361/133856; lowest indexer pooled key 3520/3536
rollback cache store: 9776 elements equal to the rounded pre-store value
rollback flash attention op: elements > 0.5 spacing 1/8448
ub10     attention output: 2-norm rel 4.243e-05, within 0.5 spacing 0.9902 of 8192 (aware)
ub10     attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
ub10     cache rows equal to the oracle's stored rows: 40386/40528; lowest indexer pooled key 1066/1072
ub10     cache store: 9472 elements equal to the rounded pre-store value
ub10     flash attention op: elements > 0.5 spacing 1/8192
ub16     attention output: 2-norm rel 4.243e-05, within 0.5 spacing 0.9902 of 8192 (aware)
ub16     attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
ub16     cache rows equal to the oracle's stored rows: 23598/23680; lowest indexer pooled key 637/640
ub16     cache store: 9472 elements equal to the rounded pre-store value
ub16     flash attention op: elements > 0.5 spacing 1/8192
qwen4exp: rows where an oracle's discrete choice differs from llama.cpp (the precision-aware oracle after its tie replays, listed in the first table), and N6/N9 checks the plain f32 oracle fails (assertion: ...)
(and 30 rows with 0, not shown)
largest errors against the precision-aware oracle (first table, 320 rows):
rollback attn_pregate (dense causal) [full]                             7     9  1.355e-03  2.551e-03
rollback attn_pregate (dense causal) [chunk]                            7     9  1.355e-03  2.551e-03
ub10     attn_pregate (dense causal) [full]                             7     5  1.142e-03  1.922e-03
ub10     attn_pregate (dense causal) [chunk]                            7     5  1.142e-03  1.922e-03
--- glm5-next
glm5-next: worst error per layer (layer -1 = embedding and head), precision-aware oracle | plain f32 oracle
layer    max abs        rel component (aware)                                  |    max abs        rel component (f32)
   -1  2.793e-03  8.403e-04 ub10 result_norm [full]                            |  1.203e-02  3.500e-03 ub16 result_output [full]
    0  1.929e-06  1.590e-06 decode kda_out [full]                              |  1.929e-06  1.590e-06 decode kda_out [full]
    1  1.571e-07  3.450e-06 rollback kda_scan_out [full]                       |  1.571e-07  3.450e-06 rollback kda_scan_out [full]
    2  1.491e-06  4.676e-06 ub10 ffn_shexp (shared, clamped) [chunk]           |  1.491e-06  4.676e-06 ub10 ffn_shexp (shared, clamped) [chunk]
    3  6.419e-04  5.557e-04 decode kqv_out (sparse MLA) [full]                 |  7.601e-05  1.289e-03 decode indexer_score (visible) [chunk]
    4  1.822e-04  2.490e-04 decode hc_post (ffn) [full]                        |  3.070e-03  1.535e-03 ub10 hc_post (ffn) [full]
    5  5.139e-04  5.749e-04 decode kda_out [full]                              |  7.591e-03  7.620e-03 ub10 hc_comb (ffn, Sinkhorn) [full]
    6  7.668e-04  9.701e-04 decode kda_out [chunk]                             |  8.611e-03  1.098e-02 decode kda_out [chunk]
    7  7.365e-04  7.673e-04 decode indexer_pool_k_new (pools completed here) [ |  2.184e-03  7.340e-03 ub10 ffn_shexp (shared, clamped) [full]
    8  3.817e-05  1.581e-03 decode kda_scan_out [chunk]                        |  1.363e-02  9.609e-03 ub16 kda_out [full]
    9  1.349e-04  2.947e-03 decode kda_scan_out [full]                         |  9.598e-03  9.762e-03 ub10 hc_comb (attn, Sinkhorn) [full]
   10  8.416e-04  2.599e-03 ub10 ffn_shexp (shared, clamped) [full]            |  1.188e-02  8.015e-03 ub16 kda_out [full]
   11  1.257e-03  1.981e-03 ub10 ffn_out (routed + shared) [full]              |  5.155e-03  1.644e-02 decode indexer_score (visible) [chunk]
glm5-next: summaries (cache store and cache rows, flash-attention op, attention output, tie replays)
decode   attention output: 2-norm rel 1.137e-03, within 0.5 spacing 0.2784 of 12288 (plain)
decode   attention output: 2-norm rel 3.558e-04, within 0.5 spacing 0.6742 of 12288 (aware)
decode   cache rows equal to the oracle's stored rows: 127216/148224; lowest indexer pooled key 9195/11136
decode   cache store: 9984 elements equal to the rounded pre-store value
decode   flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
pf64     attention output: 2-norm rel 4.029e-04, within 0.5 spacing 0.5083 of 12288 (plain)
pf64     attention output: 2-norm rel 4.537e-05, within 0.5 spacing 0.9625 of 12288 (aware)
pf64     cache rows equal to the oracle's stored rows: 9711/9984; lowest indexer pooled key 742/768
pf64     cache store: 9984 elements equal to the rounded pre-store value
rollback attention output: 2-norm rel 1.139e-03, within 0.5 spacing 0.2764 of 12672 (plain)
rollback attention output: 2-norm rel 3.619e-04, within 0.5 spacing 0.6725 of 12672 (aware)
rollback cache rows equal to the oracle's stored rows: 120719/140928; lowest indexer pooled key 8730/10608
rollback cache store: 10320 elements equal to the rounded pre-store value
rollback flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
ub10     attention output: 2-norm rel 1.136e-03, within 0.5 spacing 0.2773 of 12288 (plain)
ub10     attention output: 2-norm rel 3.537e-04, within 0.5 spacing 0.6791 of 12288 (aware)
ub10     cache rows equal to the oracle's stored rows: 37154/42672; lowest indexer pooled key 2694/3216
ub10     cache store: 9984 elements equal to the rounded pre-store value
ub10     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
ub16     attention output: 2-norm rel 1.136e-03, within 0.5 spacing 0.2773 of 12288 (plain)
ub16     attention output: 2-norm rel 3.537e-04, within 0.5 spacing 0.6791 of 12288 (aware)
ub16     cache rows equal to the oracle's stored rows: 21710/24960; lowest indexer pooled key 1600/1920
ub16     cache store: 9984 elements equal to the rounded pre-store value
ub16     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
ub18     attention output: 2-norm rel 1.136e-03, within 0.5 spacing 0.2773 of 12288 (plain)
ub18     attention output: 2-norm rel 3.537e-04, within 0.5 spacing 0.6791 of 12288 (aware)
ub18     cache rows equal to the oracle's stored rows: 23207/26784; lowest indexer pooled key 1680/2016
ub18     cache store: 9984 elements equal to the rounded pre-store value
ub18     flash attention op: max |err| / (W_v_b x one spacing of O) = 0.078
glm5-next: rows where an oracle's discrete choice differs from llama.cpp (the precision-aware oracle after its tie replays, listed in the first table), and N6/N9 checks the plain f32 oracle fails (assertion: ...)
(and 36 rows with 0, not shown)
largest errors against the precision-aware oracle (first table, 542 rows):
decode   kda_scan_out [full]                                            9    11  2.947e-03  8.034e-03
decode   kda_scan_out [chunk]                                           9    11  2.947e-03  8.034e-03
decode   kda_scan_out (first row, reads the carried state) [full]       9    11  2.947e-03  8.034e-03
decode   kda_scan_out (first row, reads the carried state) [chunk]      9    11  2.947e-03  8.034e-03
```

### Other precision configurations (tests only, on dumps of the same driver with `PARITY_KV` / `PARITY_FA`)

Tests only (`tests.test_qwen4exp_f16fa_parity tests.test_glm5next_f16fa_parity tests.test_fattn_ref`) on HEAD
`27cbb25`, on the dumps `run_parity_f16fa.sh` made with the given `PARITY_KV` / `PARITY_FA` (work dirs
`<scratch>/n10/work-<pin>[-<kv>-<fa>]`). Shown per configuration: the unittest summary, the layer -1 (head) row of
each model's per-layer table (precision-aware | plain f32 oracle), and the attention-output rows of `pf64` and
`decode`; everything else is cut. Command, per pin and configuration:
`PARITY_KV=<kv> PARITY_FA=<fa> PARITY_LLAMA_COMMIT=<sha> LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py QWEN4EXP_F16FA_DUMP=<w>/q
QWEN4EXP_F16FA_GGUF=<w>/qwen4exp.gguf GLM5NEXT_F16FA_DUMP=<w>/g GLM5NEXT_F16FA_GGUF=<w>/glm5next.gguf python3 -m
unittest tests.test_qwen4exp_f16fa_parity tests.test_glm5next_f16fa_parity tests.test_fattn_ref`.

```
== ec7630a PARITY_KV=f16 PARITY_FA=on: Ran 170 tests in 21.498s OK
qwen4exp:    -1  2.725e-05  2.210e-05 rollback result_output [full]                      |  2.661e-04  1.860e-04 ub10 result_norm [full]
glm5-next:    -1  2.793e-03  8.071e-04 decode result_norm [full]                          |  5.475e-01  1.931e-01 ub10 result_norm [full]
decode   attention output: 2-norm rel 1.073e-03, within 0.5 spacing 0.3162 of 12288 (plain)
decode   attention output: 2-norm rel 3.058e-04, within 0.5 spacing 0.7198 of 12288 (aware)
decode   attention output: 2-norm rel 3.668e-05, within 0.5 spacing 0.9912 of 8192 (aware)
decode   attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
pf64     attention output: 2-norm rel 2.369e-04, within 0.5 spacing 0.6437 of 8192 (plain)
pf64     attention output: 2-norm rel 4.029e-04, within 0.5 spacing 0.5083 of 12288 (plain)
pf64     attention output: 2-norm rel 4.537e-05, within 0.5 spacing 0.9625 of 12288 (aware)
pf64     attention output: 2-norm rel 9.601e-06, within 0.5 spacing 0.9930 of 8192 (aware)
== ec7630a PARITY_KV=f32 PARITY_FA=on: Ran 170 tests in 20.883s OK
qwen4exp:    -1  2.725e-05  2.210e-05 rollback result_output [full]                      |  2.661e-04  1.860e-04 ub10 result_norm [full]
glm5-next:    -1  2.793e-03  8.071e-04 decode result_norm [full]                          |  1.203e-02  3.327e-03 ub18 result_output [full]
decode   attention output: 2-norm rel 1.053e-03, within 0.5 spacing 0.3629 of 12288 (plain)
decode   attention output: 2-norm rel 2.937e-04, within 0.5 spacing 0.7437 of 12288 (aware)
decode   attention output: 2-norm rel 3.668e-05, within 0.5 spacing 0.9912 of 8192 (aware)
decode   attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
pf64     attention output: 2-norm rel 2.369e-04, within 0.5 spacing 0.6437 of 8192 (plain)
pf64     attention output: 2-norm rel 4.029e-04, within 0.5 spacing 0.5083 of 12288 (plain)
pf64     attention output: 2-norm rel 4.537e-05, within 0.5 spacing 0.9625 of 12288 (aware)
pf64     attention output: 2-norm rel 9.601e-06, within 0.5 spacing 0.9930 of 8192 (aware)
== ec7630a PARITY_KV=f16 PARITY_FA=off: Ran 170 tests in 19.367s OK
qwen4exp:    -1  1.362e-05  9.003e-06 ub10 result_norm [chunk]                           |  1.731e-04  1.388e-04 rollback result_norm [chunk]
glm5-next:    -1  1.314e-03  3.464e-04 decode result_output [chunk]                       |  5.475e-01  1.931e-01 ub10 result_norm [full]
decode   attention output: 2-norm rel 1.915e-04, within 0.5 spacing 0.8382 of 12288 (aware)
decode   attention output: 2-norm rel 3.079e-05, within 0.5 spacing 0.9747 of 8192 (aware)
decode   attention output: 2-norm rel 3.300e-04, within 0.5 spacing 0.5227 of 8192 (plain)
decode   attention output: 2-norm rel 6.477e-04, within 0.5 spacing 0.4152 of 12288 (plain)
pf64     attention output: 2-norm rel 2.011e-04, within 0.5 spacing 0.8292 of 12288 (aware)
pf64     attention output: 2-norm rel 3.106e-05, within 0.5 spacing 0.9740 of 8192 (aware)
pf64     attention output: 2-norm rel 3.301e-04, within 0.5 spacing 0.5222 of 8192 (plain)
pf64     attention output: 2-norm rel 6.652e-04, within 0.5 spacing 0.3968 of 12288 (plain)
== ec7630a PARITY_KV=f32 PARITY_FA=off: Ran 170 tests in 17.782s OK
qwen4exp:    -1  1.899e-06  1.340e-06 ub10 result_norm [chunk]                           |  1.899e-06  1.340e-06 ub16 result_norm [chunk]
glm5-next:    -1  5.138e-05  1.670e-05 ub10 result_output [full]                          |  5.138e-05  1.670e-05 ub10 result_output [full]
decode   attention output: 2-norm rel 3.036e-06, within 0.5 spacing 0.9966 of 12288 (aware)
decode   attention output: 2-norm rel 3.036e-06, within 0.5 spacing 0.9966 of 12288 (plain)
decode   attention output: 2-norm rel 5.648e-07, within 0.5 spacing 0.9994 of 8192 (aware)
decode   attention output: 2-norm rel 5.648e-07, within 0.5 spacing 0.9994 of 8192 (plain)
pf64     attention output: 2-norm rel 3.125e-06, within 0.5 spacing 0.9967 of 12288 (aware)
pf64     attention output: 2-norm rel 3.125e-06, within 0.5 spacing 0.9967 of 12288 (plain)
pf64     attention output: 2-norm rel 5.658e-07, within 0.5 spacing 0.9994 of 8192 (aware)
pf64     attention output: 2-norm rel 5.658e-07, within 0.5 spacing 0.9994 of 8192 (plain)
== 79e2e74 PARITY_KV=f16 PARITY_FA=on: Ran 170 tests in 21.090s OK
qwen4exp:    -1  2.725e-05  2.210e-05 rollback result_output [full]                      |  2.661e-04  1.860e-04 ub10 result_norm [full]
glm5-next:    -1  2.793e-03  8.403e-04 ub10 result_norm [full]                            |  1.203e-02  3.500e-03 ub16 result_output [full]
decode   attention output: 2-norm rel 1.137e-03, within 0.5 spacing 0.2784 of 12288 (plain)
decode   attention output: 2-norm rel 3.558e-04, within 0.5 spacing 0.6742 of 12288 (aware)
decode   attention output: 2-norm rel 3.668e-05, within 0.5 spacing 0.9912 of 8192 (aware)
decode   attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
pf64     attention output: 2-norm rel 2.369e-04, within 0.5 spacing 0.6437 of 8192 (plain)
pf64     attention output: 2-norm rel 4.029e-04, within 0.5 spacing 0.5083 of 12288 (plain)
pf64     attention output: 2-norm rel 4.537e-05, within 0.5 spacing 0.9625 of 12288 (aware)
pf64     attention output: 2-norm rel 9.601e-06, within 0.5 spacing 0.9930 of 8192 (aware)
== 79e2e74 PARITY_KV=f32 PARITY_FA=on: Ran 170 tests in 21.699s OK
qwen4exp:    -1  2.725e-05  2.210e-05 rollback result_output [full]                      |  2.661e-04  1.860e-04 ub10 result_norm [full]
glm5-next:    -1  2.793e-03  8.403e-04 ub10 result_norm [full]                            |  1.203e-02  3.500e-03 ub16 result_output [full]
decode   attention output: 2-norm rel 1.137e-03, within 0.5 spacing 0.2784 of 12288 (plain)
decode   attention output: 2-norm rel 3.558e-04, within 0.5 spacing 0.6742 of 12288 (aware)
decode   attention output: 2-norm rel 3.668e-05, within 0.5 spacing 0.9912 of 8192 (aware)
decode   attention output: 2-norm rel 5.653e-04, within 0.5 spacing 0.3478 of 8192 (plain)
pf64     attention output: 2-norm rel 2.369e-04, within 0.5 spacing 0.6437 of 8192 (plain)
pf64     attention output: 2-norm rel 4.029e-04, within 0.5 spacing 0.5083 of 12288 (plain)
pf64     attention output: 2-norm rel 4.537e-05, within 0.5 spacing 0.9625 of 12288 (aware)
pf64     attention output: 2-norm rel 9.601e-06, within 0.5 spacing 0.9930 of 8192 (aware)
== 79e2e74 PARITY_KV=f16 PARITY_FA=off: Ran 170 tests in 18.691s OK
qwen4exp:    -1  1.362e-05  9.003e-06 ub10 result_norm [chunk]                           |  1.731e-04  1.388e-04 rollback result_norm [chunk]
glm5-next:    -1  1.314e-03  4.021e-04 ub10 result_output [full]                          |  7.200e-03  2.340e-03 ub10 result_output [chunk]
decode   attention output: 2-norm rel 2.003e-04, within 0.5 spacing 0.8302 of 12288 (aware)
decode   attention output: 2-norm rel 3.079e-05, within 0.5 spacing 0.9747 of 8192 (aware)
decode   attention output: 2-norm rel 3.300e-04, within 0.5 spacing 0.5227 of 8192 (plain)
decode   attention output: 2-norm rel 6.648e-04, within 0.5 spacing 0.3973 of 12288 (plain)
pf64     attention output: 2-norm rel 2.011e-04, within 0.5 spacing 0.8292 of 12288 (aware)
pf64     attention output: 2-norm rel 3.106e-05, within 0.5 spacing 0.9740 of 8192 (aware)
pf64     attention output: 2-norm rel 3.301e-04, within 0.5 spacing 0.5222 of 8192 (plain)
pf64     attention output: 2-norm rel 6.652e-04, within 0.5 spacing 0.3968 of 12288 (plain)
== 79e2e74 PARITY_KV=f32 PARITY_FA=off: Ran 170 tests in 19.418s OK
qwen4exp:    -1  1.899e-06  1.340e-06 ub10 result_norm [chunk]                           |  1.899e-06  1.340e-06 ub16 result_norm [chunk]
glm5-next:    -1  4.983e-05  1.619e-05 ub10 result_output [full]                          |  4.983e-05  1.619e-05 ub10 result_output [full]
decode   attention output: 2-norm rel 3.051e-06, within 0.5 spacing 0.9967 of 12288 (aware)
decode   attention output: 2-norm rel 3.051e-06, within 0.5 spacing 0.9967 of 12288 (plain)
decode   attention output: 2-norm rel 5.648e-07, within 0.5 spacing 0.9994 of 8192 (aware)
decode   attention output: 2-norm rel 5.648e-07, within 0.5 spacing 0.9994 of 8192 (plain)
pf64     attention output: 2-norm rel 3.125e-06, within 0.5 spacing 0.9967 of 12288 (aware)
pf64     attention output: 2-norm rel 3.125e-06, within 0.5 spacing 0.9967 of 12288 (plain)
pf64     attention output: 2-norm rel 5.658e-07, within 0.5 spacing 0.9994 of 8192 (aware)
pf64     attention output: 2-norm rel 5.658e-07, within 0.5 spacing 0.9994 of 8192 (plain)
```

### First probe: the flash-attention emulation on llama.cpp's own inputs (qwen4exp, ec7630a)

`python3 docs/evidence/mutation-scripts/2026-10-10-n10-fa-op-probe.py <scratch>/n10/work-ec/q <run> 3` for the runs
pf64, ub16, decode (graphs 0-3; full output). `fa_tiled` and `exact` coincide on these inputs: the cache views
already hold binary16 values (last column 0.0).

```
g0 L3 T64 fa_chunk  rel 5.538e-04 max spacings 638.00 share > 0.5 spacing 0.5518; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T64 fa_tiled  rel 1.043e-07 max spacings 0.20 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T64 exact     rel 1.043e-07 max spacings 0.20 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T16 fa_chunk  rel 4.157e-05 max spacings 0.93 share > 0.5 spacing 0.0010; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T16 fa_tiled  rel 4.689e-04 max spacings 86.33 share > 0.5 spacing 0.4883; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T16 exact     rel 4.689e-04 max spacings 86.33 share > 0.5 spacing 0.4883; K, V views minus f16(K, V): 0.0, 0.0
g1 L3 T16 fa_chunk  rel 7.265e-08 max spacings 0.00 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g1 L3 T16 fa_tiled  rel 9.171e-04 max spacings 637.86 share > 0.5 spacing 0.5723; K, V views minus f16(K, V): 0.0, 0.0
g1 L3 T16 exact     rel 9.171e-04 max spacings 637.86 share > 0.5 spacing 0.5723; K, V views minus f16(K, V): 0.0, 0.0
g2 L3 T16 fa_chunk  rel 1.318e-07 max spacings 0.00 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g2 L3 T16 fa_tiled  rel 6.841e-04 max spacings 162.42 share > 0.5 spacing 0.5605; K, V views minus f16(K, V): 0.0, 0.0
g2 L3 T16 exact     rel 6.841e-04 max spacings 162.42 share > 0.5 spacing 0.5605; K, V views minus f16(K, V): 0.0, 0.0
g3 L3 T16 fa_chunk  rel 1.525e-07 max spacings 0.00 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g3 L3 T16 fa_tiled  rel 6.505e-04 max spacings 381.60 share > 0.5 spacing 0.5869; K, V views minus f16(K, V): 0.0, 0.0
g3 L3 T16 exact     rel 6.505e-04 max spacings 381.60 share > 0.5 spacing 0.5869; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T48 fa_chunk  rel 4.157e-05 max spacings 0.93 share > 0.5 spacing 0.0003; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T48 fa_tiled  rel 5.538e-04 max spacings 637.86 share > 0.5 spacing 0.5404; K, V views minus f16(K, V): 0.0, 0.0
g0 L3 T48 exact     rel 5.538e-04 max spacings 637.86 share > 0.5 spacing 0.5404; K, V views minus f16(K, V): 0.0, 0.0
g1 L3 T1 fa_chunk  rel 3.569e-08 max spacings 0.00 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g1 L3 T1 fa_tiled  rel 5.492e-04 max spacings 13.30 share > 0.5 spacing 0.5312; K, V views minus f16(K, V): 0.0, 0.0
g1 L3 T1 exact     rel 5.492e-04 max spacings 13.30 share > 0.5 spacing 0.5312; K, V views minus f16(K, V): 0.0, 0.0
g2 L3 T1 fa_chunk  rel 6.439e-08 max spacings 0.00 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g2 L3 T1 fa_tiled  rel 8.705e-04 max spacings 20.96 share > 0.5 spacing 0.5938; K, V views minus f16(K, V): 0.0, 0.0
g2 L3 T1 exact     rel 8.705e-04 max spacings 20.96 share > 0.5 spacing 0.5938; K, V views minus f16(K, V): 0.0, 0.0
g3 L3 T1 fa_chunk  rel 1.190e-07 max spacings 0.00 share > 0.5 spacing 0.0000; K, V views minus f16(K, V): 0.0, 0.0
g3 L3 T1 fa_tiled  rel 5.389e-04 max spacings 26.02 share > 0.5 spacing 0.6875; K, V views minus f16(K, V): 0.0, 0.0
g3 L3 T1 exact     rel 5.389e-04 max spacings 26.02 share > 0.5 spacing 0.6875; K, V views minus f16(K, V): 0.0, 0.0
```

### llama.cpp f16 / FA against llama.cpp f32 / no FA, and the decision margins

`python3 docs/evidence/mutation-scripts/2026-10-10-n10-logprob-probe.py <scratch>/n10/work-<pin> <scratch>/n10/work-<pin>-f32-off` (full output, both pins):

```
== ec
qwen4exp pf64     max |d log-softmax| 1.199e-04 (row 3), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp ub16     max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp ub10     max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp decode   max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp rollback max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 66, discrete choices differing 0
glm5-next pf64     max |d log-softmax| 2.729e-03 (row 28), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next ub16     max |d log-softmax| 6.031e-01 (row 63), greedy tokens differing 0 of 64, discrete choices differing 4
    MoE L7 pos 60 (graph 3): [4, 5] vs [5, 7]
    MoE L9 pos 63 (graph 3): [0, 4] vs [2, 4]
    MoE L10 pos 60 (graph 3): [1, 5] vs [0, 5]
    DSA L11 pos 60 (graph 3): cells differ
glm5-next ub10     max |d log-softmax| 6.031e-01 (row 63), greedy tokens differing 0 of 64, discrete choices differing 4
    MoE L7 pos 60 (graph 6): [4, 5] vs [5, 7]
    MoE L9 pos 63 (graph 6): [0, 4] vs [2, 4]
    MoE L10 pos 60 (graph 6): [1, 5] vs [0, 5]
    DSA L11 pos 60 (graph 6): cells differ
glm5-next ub18     max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next decode   max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next rollback max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 66, discrete choices differing 0
== r1
qwen4exp pf64     max |d log-softmax| 1.199e-04 (row 3), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp ub16     max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp ub10     max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp decode   max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 64, discrete choices differing 0
qwen4exp rollback max |d log-softmax| 2.789e-04 (row 19), greedy tokens differing 0 of 66, discrete choices differing 0
glm5-next pf64     max |d log-softmax| 2.729e-03 (row 28), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next ub16     max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next ub10     max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next ub18     max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next decode   max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 64, discrete choices differing 0
glm5-next rollback max |d log-softmax| 1.150e-02 (row 43), greedy tokens differing 0 of 66, discrete choices differing 0
```

`python3 docs/evidence/mutation-scripts/2026-10-10-n10-margin-probe.py test_<model>_f16fa_parity <run>` for pf64,
ub16 and decode, env as for the tests (full output, ec7630a then 79e2e74; the changes after the flip in ec7630a ub16
are downstream of it, see the module docstring):

```
glm5next pf64     MoE experts: 704 decisions, exact ties (margin 0) 0, median margin 6.921e-02; change of the margin by the precision: median 2.513e-05, max 5.580e-04; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 6
glm5next pf64     sparse selection: 159 decisions, exact ties (margin 0) 13, median margin 5.589e-02; change of the margin by the precision: median 1.323e-04, max 1.588e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 16
qwen4exp pf64     MoE experts: 512 decisions, exact ties (margin 0) 0, median margin 2.149e-02; change of the margin by the precision: median 6.440e-07, max 1.447e-05; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 19
qwen4exp pf64     sparse selection: 53 decisions, exact ties (margin 0) 0, median margin 5.349e-01; change of the margin by the precision: median 5.696e-04, max 1.475e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 0
glm5next ub16     MoE experts: 704 decisions, exact ties (margin 0) 0, median margin 6.921e-02; change of the margin by the precision: median 2.533e-05, max 5.656e-02; 0 < margin < median change in 0; margin < change in 4; margin < 1e-3 in 6
glm5next ub16     sparse selection: 159 decisions, exact ties (margin 0) 13, median margin 5.589e-02; change of the margin by the precision: median 1.413e-04, max 2.013e-01; 0 < margin < median change in 0; margin < change in 1; margin < 1e-3 in 16
qwen4exp ub16     MoE experts: 512 decisions, exact ties (margin 0) 0, median margin 2.149e-02; change of the margin by the precision: median 1.436e-06, max 2.250e-05; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 19
qwen4exp ub16     sparse selection: 53 decisions, exact ties (margin 0) 0, median margin 5.349e-01; change of the margin by the precision: median 5.696e-04, max 1.475e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 0
glm5next decode   MoE experts: 704 decisions, exact ties (margin 0) 0, median margin 6.921e-02; change of the margin by the precision: median 3.801e-05, max 1.680e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 6
glm5next decode   sparse selection: 159 decisions, exact ties (margin 0) 13, median margin 5.589e-02; change of the margin by the precision: median 1.827e-04, max 1.073e-02; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 16
qwen4exp decode   MoE experts: 512 decisions, exact ties (margin 0) 0, median margin 2.149e-02; change of the margin by the precision: median 1.436e-06, max 2.250e-05; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 19
qwen4exp decode   sparse selection: 53 decisions, exact ties (margin 0) 0, median margin 5.349e-01; change of the margin by the precision: median 5.696e-04, max 1.475e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 0
glm5next pf64     MoE experts: 704 decisions, exact ties (margin 0) 0, median margin 6.921e-02; change of the margin by the precision: median 2.513e-05, max 5.580e-04; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 6
glm5next pf64     sparse selection: 159 decisions, exact ties (margin 0) 13, median margin 5.589e-02; change of the margin by the precision: median 1.323e-04, max 1.588e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 16
qwen4exp pf64     MoE experts: 512 decisions, exact ties (margin 0) 0, median margin 2.149e-02; change of the margin by the precision: median 6.440e-07, max 1.447e-05; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 19
qwen4exp pf64     sparse selection: 53 decisions, exact ties (margin 0) 0, median margin 5.349e-01; change of the margin by the precision: median 5.696e-04, max 1.475e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 0
glm5next ub16     MoE experts: 704 decisions, exact ties (margin 0) 0, median margin 6.921e-02; change of the margin by the precision: median 4.909e-05, max 1.680e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 6
glm5next ub16     sparse selection: 159 decisions, exact ties (margin 0) 13, median margin 5.589e-02; change of the margin by the precision: median 1.801e-04, max 1.073e-02; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 16
qwen4exp ub16     MoE experts: 512 decisions, exact ties (margin 0) 0, median margin 2.149e-02; change of the margin by the precision: median 1.436e-06, max 2.250e-05; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 19
qwen4exp ub16     sparse selection: 53 decisions, exact ties (margin 0) 0, median margin 5.349e-01; change of the margin by the precision: median 5.696e-04, max 1.475e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 0
glm5next decode   MoE experts: 704 decisions, exact ties (margin 0) 0, median margin 6.921e-02; change of the margin by the precision: median 4.909e-05, max 1.680e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 6
glm5next decode   sparse selection: 159 decisions, exact ties (margin 0) 13, median margin 5.589e-02; change of the margin by the precision: median 1.801e-04, max 1.073e-02; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 16
qwen4exp decode   MoE experts: 512 decisions, exact ties (margin 0) 0, median margin 2.149e-02; change of the margin by the precision: median 1.436e-06, max 2.250e-05; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 19
qwen4exp decode   sparse selection: 53 decisions, exact ties (margin 0) 0, median margin 5.349e-01; change of the margin by the precision: median 5.696e-04, max 1.475e-03; 0 < margin < median change in 0; margin < change in 0; margin < 1e-3 in 0
```

### Mutation check (part A)

`python3 docs/evidence/mutation-scripts/2026-10-10-n10-f16fa.py <repo> <scratch>/n10/work-<pin> <llama.cpp> <scratch>/n10/mut-<pin>` on
the code of HEAD `27cbb25` (the working tree before that commit, unchanged during the run), once per pin;
shown through `grep -E "^(llama|baseline|[OTD][0-9]+ |    rc=|    (fattn|glm5next|qwen4exp)|caught)" <log> | cut -c1-150`
(the per-mutant lists of failing tests are cut at 150 columns; the mutant edits are in the script).

```
llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d; dumps <scratch>/n10/work-e
baseline (no mutation): rc=0 Ran 170 tests in 20.863s | OK
O1 store(): no f16 rounding of what the caches hold
    rc=1 Ran 170 tests in 22.325s | FAILED (failures=12)
    fattn (property): F16Rounding.test_store_and_modes
    glm5next: Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_cache_contents, Ubatch10.test_cache_contents, Ubatch16.test_cac
    qwen4exp: Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_cache_contents, Ubatch10.test_cache_contents, Ubatch16.test_cac
O2 one_chunk: Q kept in f32 (no Q_q = f16(q))
    rc=1 Ran 170 tests in 23.091s | FAILED (failures=14)
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
O3 one_chunk: accumulator kept in f32 in the mad step (VKQ16 not rounded)
    rc=1 Ran 170 tests in 20.475s | FAILED (failures=17)
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
O4 tiled path from 16 query rows instead of 64
    rc=1 Ran 170 tests in 18.502s | FAILED (failures=23)
    fattn (property): F16Rounding.test_store_and_modes
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Decode.test_kda_layers, Decode.test_mhc, Rollback.test_att
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Decode.test_run_exercises_its_mechanisms, Rollback.test_at
O5 f16 rounding toward zero instead of to nearest even
    rc=1 Ran 170 tests in 19.561s | FAILED (failures=70)
    fattn (property): F16Rounding.test_idempotent_and_within_half_a_spacing, F16Rounding.test_round_to_nearest_even
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_cache_contents, Decode.test_dsa_pool_cache_and_selection, Decode.test_dsa_proj
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_attention_over_the_kv_cache, Decode.test_cache_contents, Decode.test_flash_att
O6 qwen4exp: raw indexer keys cached unrounded (incremental path only)
    rc=1 Ran 170 tests in 20.740s | FAILED (failures=5)
    qwen4exp: Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_cache_contents, Ubatch10.test_cache_contents, Ubatch16.test_cac
O7 one_chunk: the score not rounded to f32 (float64 q.k * scale)
    rc=1 Ran 170 tests in 21.148s | FAILED (failures=4)
    qwen4exp: Decode.test_flash_attention_op, Rollback.test_flash_attention_op, Ubatch10.test_flash_attention_op, Ubatch16.test_flash_attention_op
T1 tests: the f32 tolerance 1e-4 downstream of the attention too
    rc=1 Ran 170 tests in 18.071s | FAILED (failures=50)
    glm5next: Decode.test_cache_contents, Decode.test_dsa_pool_cache_and_selection, Decode.test_dsa_projections, Decode.test_head_and_logits, Decode.t
    qwen4exp: Decode.test_attention_over_the_kv_cache, Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_attention_over_the_kv_
T2 tests: no near-tie replay in the f16 runs (the N6/N9 settings)
    rc=1 Ran 170 tests in 19.515s | FAILED (failures=20)
    glm5next: Ubatch10.test_attention_output_vs_plain_oracle, Ubatch10.test_cache_contents, Ubatch10.test_dsa_pool_cache_and_selection, Ubatch10.test_
T3 tests: the plain f32 oracle used as the precision-aware one
    rc=1 Ran 170 tests in 18.459s | FAILED (failures=24, errors=18)
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_cache_contents, Decode.test_flash_attention_op, Decode.test_mhc, Decode.test_r
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_cache_contents, Decode.test_flash_attention_op, Decode.test_run_exercises_its_
D1 llama side: one cached K element one binary16 spacing off (qwen4exp L3, ub16 graph 1)
    rc=1 Ran 170 tests in 22.502s | FAILED (failures=1)
    qwen4exp: Ubatch16.test_cache_contents
D2 llama side: flash-attention output 2^-8 too large (qwen4exp L3, ub16 graph 2)
    rc=1 Ran 170 tests in 22.834s | FAILED (failures=2)
    qwen4exp: Ubatch16.test_attention_output_vs_plain_oracle, Ubatch16.test_flash_attention_op
D3 llama side: two latent-cache cells swapped (glm5-next L7, decode graph 3)
    rc=1 Ran 170 tests in 21.632s | FAILED (failures=1)
    glm5next: Decode.test_cache_contents
D4 llama side: one latent the attention read 2^-9 too large (glm5-next L3, pf64)
    rc=1 Ran 170 tests in 22.066s | FAILED (failures=1)
    glm5next: Prefill64.test_flash_attention_op
caught 14/14
```

```
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6; dumps <scratch>/n10/work-r
baseline (no mutation): rc=0 Ran 170 tests in 21.223s | OK
O1 store(): no f16 rounding of what the caches hold
    rc=1 Ran 170 tests in 21.102s | FAILED (failures=12)
    fattn (property): F16Rounding.test_store_and_modes
    glm5next: Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_cache_contents, Ubatch10.test_cache_contents, Ubatch16.test_cac
    qwen4exp: Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_cache_contents, Ubatch10.test_cache_contents, Ubatch16.test_cac
O2 one_chunk: Q kept in f32 (no Q_q = f16(q))
    rc=1 Ran 170 tests in 23.691s | FAILED (failures=22)
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_dsa_pool_cache_and_selection, Decode.test_dsa_projections, Decode.test_flash_a
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
O3 one_chunk: accumulator kept in f32 in the mad step (VKQ16 not rounded)
    rc=1 Ran 170 tests in 19.883s | FAILED (failures=27)
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Rollback.test_attention_output_vs_plain_oracle, Rollback.t
O4 tiled path from 16 query rows instead of 64
    rc=1 Ran 170 tests in 18.049s | FAILED (failures=27)
    fattn (property): F16Rounding.test_store_and_modes
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Decode.test_kda_layers, Decode.test_mhc, Rollback.test_att
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_flash_attention_op, Decode.test_run_exercises_its_mechanisms, Rollback.test_at
O5 f16 rounding toward zero instead of to nearest even
    rc=1 Ran 170 tests in 20.412s | FAILED (failures=80)
    fattn (property): F16Rounding.test_idempotent_and_within_half_a_spacing, F16Rounding.test_round_to_nearest_even
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_cache_contents, Decode.test_dsa_pool_cache_and_selection, Decode.test_dsa_proj
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_attention_over_the_kv_cache, Decode.test_cache_contents, Decode.test_flash_att
O6 qwen4exp: raw indexer keys cached unrounded (incremental path only)
    rc=1 Ran 170 tests in 20.404s | FAILED (failures=5)
    qwen4exp: Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_cache_contents, Ubatch10.test_cache_contents, Ubatch16.test_cac
O7 one_chunk: the score not rounded to f32 (float64 q.k * scale)
    rc=1 Ran 170 tests in 22.686s | FAILED (failures=4)
    qwen4exp: Decode.test_flash_attention_op, Rollback.test_flash_attention_op, Ubatch10.test_flash_attention_op, Ubatch16.test_flash_attention_op
T1 tests: the f32 tolerance 1e-4 downstream of the attention too
    rc=1 Ran 170 tests in 20.062s | FAILED (failures=54)
    glm5next: Decode.test_cache_contents, Decode.test_dsa_pool_cache_and_selection, Decode.test_dsa_projections, Decode.test_head_and_logits, Decode.t
    qwen4exp: Decode.test_attention_over_the_kv_cache, Decode.test_cache_contents, Prefill64.test_cache_contents, Rollback.test_attention_over_the_kv_
T2 tests: no near-tie replay in the f16 runs (the N6/N9 settings)
    rc=0 Ran 170 tests in 22.942s | OK
T3 tests: the plain f32 oracle used as the precision-aware one
    rc=1 Ran 170 tests in 18.984s | FAILED (failures=35, errors=22)
    glm5next: Decode.test_attention_output_vs_plain_oracle, Decode.test_cache_contents, Decode.test_dsa_pool_cache_and_selection, Decode.test_dsa_proj
    qwen4exp: Decode.test_attention_output_vs_plain_oracle, Decode.test_cache_contents, Decode.test_flash_attention_op, Decode.test_run_exercises_its_
D1 llama side: one cached K element one binary16 spacing off (qwen4exp L3, ub16 graph 1)
    rc=1 Ran 170 tests in 22.646s | FAILED (failures=1)
    qwen4exp: Ubatch16.test_cache_contents
D2 llama side: flash-attention output 2^-8 too large (qwen4exp L3, ub16 graph 2)
    rc=1 Ran 170 tests in 21.712s | FAILED (failures=2)
    qwen4exp: Ubatch16.test_attention_output_vs_plain_oracle, Ubatch16.test_flash_attention_op
D3 llama side: two latent-cache cells swapped (glm5-next L7, decode graph 3)
    rc=1 Ran 170 tests in 24.582s | FAILED (failures=1)
    glm5next: Decode.test_cache_contents
D4 llama side: one latent the attention read 2^-9 too large (glm5-next L3, pf64)
    rc=1 Ran 170 tests in 23.603s | FAILED (failures=1)
    glm5next: Prefill64.test_flash_attention_op
caught 13/14
```
