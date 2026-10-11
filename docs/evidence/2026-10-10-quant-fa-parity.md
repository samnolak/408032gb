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

## Part B: quantized weights (gap 2)

Every oracle comparison so far ran on f32 tiny models; the real files are k-quant GGUFs (CORRECTNESS.md gap 2). Here the
N2 / N6 qwen4exp and the G1 / N9 glm5-next tiny architectures are quantized by each pin's own `llama-quantize` to Q8_0
and Q4_K_M and every matrix product of the real graph is compared, from llama.cpp's own dumped input, with the
dequantized weights (`gguf.quants.dequantize` of the pin's gguf-py) and the CPU activation quantization emulated.

### What llama.cpp does with quantized weights on the CPU (CONFIRMED from the source at both pins unless labelled)

| question | answer | where (@ec7630a; printed below) |
|---|---|---|
| which tensors llama-quantize quantizes | 2-D and 3-D tensors whose name ends in `weight`, except `*_norm.weight`, `ffn_gate_inp`, `ssm_conv1d*`, `indexer.k_proj` / `indexer.q_proj` and, for glm5-next only, `hc_*`, the indexer (`indexer.attn_q_b`, `indexer.attn_k`, `indexer.proj`, `indexer_compressor_*`), `ssm_f_a/f_b/g_a/g_b`, `ssm_beta`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`. The qwen4exp PLE gammas `ple_norm_{key,query,conv}.weight` match none of these (the name has `_norm_key.weight`, not `_norm.weight`): they are quantized | src/llama-quant.cpp:287-346 (`_norm.weight` :300, glm5-next :333-346) |
| the row-length rule | a row (ne0) not divisible by the target's block falls back: Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0 (block 32), then F16 if still not divisible; QK_K = 256 | src/llama-quant.cpp:390-439 (:396, 416-418, 429, 438); ggml/src/ggml-common.h:89 |
| what the CPU does with the activations of a quantized product | `mul_mat` / `mul_mat_id` convert src1 row by row with the `from_float` of the weight type's `vec_dot_type`: F16 -> F16 (round to nearest even), Q8_0 / Q4_0 / Q5_0 / IQ4_NL / MXFP4 -> Q8_0 (32-blocks, scale stored as f16), Q4_1 / Q5_1 -> Q8_1, k-quants and IQ* -> Q8_K (256-blocks, f32 scale); the integer block dots are then scaled in f32. So the product equals the dot of the dequantized weight with the **dequantized activation**, up to f32 rounding of the block scaling and sums | ggml/src/ggml-cpu/ggml-cpu.c:1278-1279, 1360-1371, 1624, 1637; type traits :215-410 (:225, 243, 263, 275, 314, 324, 330) |
| Q8_0 activations on this build | the AVX2 `quantize_row_q8_0`: d = amax/127 stored as f16, id = 127/amax, round to nearest even; not the scalar `quantize_row_q8_0_ref` (id = 1/d, `roundf`). The tests tell the two apart (mutant Q2 below) | ggml/src/ggml-cpu/arch/x86/quants.c:302-398 (:331-337, 343); ggml/src/ggml-quants.c:276-298 |
| Q8_K activations | `quantize_row_q8_K` = `quantize_row_q8_K_ref` on x86: iscale = -127/max (max: the first element of largest abs), `nearest_int` (round to nearest even), d = 1/iscale in f32. The repacked Q4_K path (AVX2, rows a multiple of 8) quantizes four rows at a time with `ggml_quantize_mat_q8_K_4x8` (iscale = +-127/amax), the remaining rows with `from_float`, and its `mul_mat_id` every row with `from_float`: the same values, since q and d change sign together | x86 quants.c:505-507; ggml-quants.c:2768-2805, 621-626; ggml/src/ggml-cpu/repack.cpp:5006-5010, 4697-4705, 4839-4841; arch/x86/repack.cpp:290-512 (:382-388) |
| repacking | on by default for Q4_K weights with ne1 % 8 == 0 on AVX2 (here 7.12 MiB of the Q4_K_M qwen4exp model in a `CPU_REPACK` buffer); off with `-nr` / `--no-repack`. Q8_0, Q5_K, Q6_K are not repacked on x86 (their branches need NEON, or RISC-V vectors for Q8_0) | repack.cpp:5006-5010 (Q4_K), 5050-5126 (Q5_K, Q6_K, Q8_0); common/arg.cpp:2421-2428; buffer sizes below |
| F16 weights through llamafile sgemm | on AVX2 + F16C only with an F16 B: the activations are converted to f16 first | ggml/src/ggml-cpu/llamafile/sgemm.cpp:3966-3973 (@79e2e74 4057-4064, same text) |
| gathers of quantized tables | `ggml_get_rows` of the token embedding / PLE table dequantizes the rows; equal to gguf-py's `dequantize` bit for bit (checked by the tests, tolerance 0) | |
| pins | x86 quants.c, x86 repack.cpp, binary-ops.cpp, unary-ops.cpp, llama-quant.cpp, tools/quantize/quantize.cpp, gguf-py quants.py, ggml-common.h: identical. ggml-cpu.c: two conditions gain `!params->use_ref &&` (the default path unchanged, line numbers unchanged); repack.cpp: only the buffer-type struct after line 5240; ggml-quants.c: a weight-quantizer helper (lines 1039-1041, not the activation path); sgemm.cpp: partial loads added, the F16 branch unchanged. The quantized files are byte-identical at both pins (sha256 below) | diff output below |

### Finding: llama-quantize makes a qwen4exp model the CPU cannot run (both pins)

`llama-quantize` quantizes the three 2-D PLE gammas `blk.<il>.ple_norm_{key,query,conv}.weight` ([n_embd, hc], see the
table above), and the graph multiplies them elementwise (`ggml_mul(rms_norm(x), w)`, src/models/qwen4exp.cpp:1384-1408),
for which the CPU backend has no quantized src1: llama.cpp aborts on the first graph, at both pins
(`binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0`, ggml/src/ggml-cpu/binary-ops.cpp:135; literal output
below; CONFIRMED). The tiny model (default and `--quant-dims`) writes them 2-D, the shape `create_tensor` names
(qwen4exp.cpp:270-272). The loader also accepts any shape with the same element count for these three tensors
(`TENSOR_ALLOW_RESHAPE`, src/llama-model-loader.cpp:896-905), so a converter could write them 1-D, which llama-quantize
never quantizes; the converter only adds 1 to the HF tensor (conversion/qwen4exp.py:164-166), so the GGUF shape is the HF
checkpoint's, which is UNKNOWN here. Whether a real quantized Qwen3.8-Flash-Next GGUF hits this: UNKNOWN (it does if the
gammas are 2-D with a width divisible by 32 and no `--tensor-type` override is used). The driver keeps them F32 with
`--tensor-type ple_norm=f32` (src/llama-quant.cpp manual override path). glm5-next needs no override.

### Pieces

| piece | file |
|---|---|
| oracle of the quantized products | ref/quant.py: `QWeight` (a non-F32 GGUF tensor, dequantized; `x @ w.T` / `w @ x` apply `act_quant`, gathers and `np.asarray` read the dequantized values), `act_quant` (F16, Q8_0 as the AVX2 build, Q8_K; Q8_1 raises), `q8_0`, `q8_K`, `load`, `flip_bound` (bound for an input the test recomputes); ref/qwen4exp.py and ref/glm5next.py write every product as `w.T` (numerically unchanged for F32 arrays) |
| generators | tools/tinygen/qwen4exp_tiny.py `--quant-dims` (n_embd 256, head_dim 64, ssm_state 64, FFN widths 256), tools/tinygen/glm5next_tiny.py `--quant-dims` (n_embd 256, KDA head_dim 64, q_lora 256, v_mla 64, FFN widths 256): every input width llama-quantize may quantize a multiple of 256; default outputs byte-identical (sha256 before / after, both generators, default, `--mtp`, `--seed 3`; glm5next also `--arch glm5next`) |
| build | tools/g0/CMakeLists.txt adds the pinned checkout's tools/quantize (`EXCLUDE_FROM_ALL`): `llama-quantize` of the same pin, linked to the same libraries as `dump-tensors` |
| driver | tools/tinygen/run_parity_quant.sh |
| tests | tests/test_qwen4exp_quant_parity.py (with the shared `_QuantOpsBase`), tests/test_glm5next_quant_parity.py, tests/test_quant_ref.py (11 property tests, numpy only, in the CI unit step) |
| mutation check, probes | docs/evidence/mutation-scripts/2026-10-10-n10-quant.py; 2026-10-10-n10-quant-e2e-probe.py, -quant-types.py, -quant-dumpdiff.py, -quant-abort-probe.sh, -quant-summary.py |
| CI | step "N10 quantized weights ..." after the f16 step in `.github/workflows/ci.yml` job `cpu-parity` (ec7630a) and in `.github/workflows/parity-r1-pin.yml` (R1 pin) |

### Runs

Per model and type (Q8_0, Q4_K_M), `-c 512 -b 512 -t 1 -ctk f32 -ctv f32 -fa off`: `pf64` (one 64-token ubatch: Q4_K
activation rows four at a time), `ub10` (10 x 6 + 4: fours plus single rows), `decode` (48 + 16 x 1: one row, the gemv
path); for Q4_K_M also `pf64-nr` and `decode-nr` (`--no-repack`). Tokens: the N6 sequence (qwen4exp), the N9 sequence
(glm5-next). Types the products used (test_every_type_checked): qwen4exp Q8_0 model {Q8_0, F16, F32}, Q4_K_M model
{Q4_K, Q5_K, Q6_K, Q8_0, F16, F32}; glm5-next Q8_0 {Q8_0, F32}, Q4_K_M {Q4_K, Q5_K, Q6_K, Q8_0, F32}; gathered only:
Q4_K / Q8_0 token embeddings, the Q5_0 / Q8_0 PLE table (per-tensor lists below).

### Method

Teacher forcing per product: every product of every graph is recomputed from llama.cpp's dumped input with the dumped
expert ids and compared with its dumped output (cb() names; the list is in the test docstrings). Where the input has no
name (glm5-next: the KDA output gate `kda_normed * sigmoid(kda_g2)`), the test recomputes it in f32 the way ggml does;
where the output has no name, the check runs through the next named tensor (qwen4exp `hc_gate` through the F16 up
projection, the PLE key / value through `ple_gate` / `ple_gated_value`; glm5-next q / k / v through the short conv and
silu, with the history of the earlier graphs, `attn_q_a` through its rms_norm, `attn_q_b` through the F32 `attn_k_b`
absorption). Each product is evaluated twice: with the activation quantization (asserted) and without it (recorded, and
required to be clearly worse). The gathers of the quantized tables must be exact.

Tolerances (written out in the test docstrings):
* scaled error = max|got - want| / max(|W| |x_q|) < TOL_OP = 1e-5 for a product from its dumped input. The integer block
  dots agree exactly (same integers, same scales); what remains is f32 rounding of the block scaling and the sum over
  the blocks, to first order (n_b + 3) 2^-24 times the summed magnitudes: Q8_0 x Q8_0 with n_b = K/32 <= 32: 2.1e-6;
  k-quant x Q8_K with n_b = K/256 <= 4 but separately rounded scale and minimum terms (each up to about 3 times the block
  maximum, itself about 3 times the mean |w| of Gaussian weights): (4 + 3) 2^-24 x 9 = 3.8e-6 (PROVISIONAL derivation).
  The scaled metric divides by the size of the summed terms, not the cancelled result: a 4-output product (`beta`, the
  `hc_inject` heads) of a single decode row has normwise errors up to 3.8e-6 for the same f32 rounding.
* products whose input the test recomputes: `flip_bound` adds one quantization step times |W| for elements within 8 f32
  spacings of a rounding boundary (and, for Q8_0, a block scale near a binary16 boundary), plus at least half of the rows
  within TOL_OP. qwen4exp `hc_gate`: one binary16 spacing per f16 input of the up projection, plus the same row share.
* chains (glm5-next conv / rms_norm / absorption, qwen4exp PLE module): normwise < TOL_OP.
* gathers: exact.
* test_activation_quantization_visible: without the activation quantization the scaled error exceeds TOL_OP in every
  graph and 20 TOL_OP in some graph of the run, for every product and weight type.


### Findings (CONFIRMED by the literal outputs below unless labelled)

1. **No llama.cpp-vs-oracle mismatch on quantized weights, at either pin.** `run_parity_quant.sh` passes 99/99 at
   ec7630a and 99/99 at 79e2e74 (qwen4exp 48: 8 runs x 6 tests; glm5-next 40: 8 runs x 5; ref/quant.py 11). Every
   product of every graph, from llama.cpp's own input: worst scaled error qwen4exp 2.39e-7 (normwise 3.83e-6),
   glm5-next 1.32e-7 (normwise 5.46e-7), against TOL_OP = 1e-5; F32-weight products 4.9e-8 / 5.1e-8 scaled. Bounded
   products: qwen4exp `hc_gate` (F16 up projection) at most 0.70 of its one-spacing bound, 8684 of 8704 rows of all runs
   within TOL_OP (the other 20 have an f16 input on a rounding boundary); glm5-next KDA conv chains, KDA output projection,
   q_resid, q_absorbed: every row within TOL_OP (4608 / 4608, 1536 / 1536). PLE module 6.1e-7. Gathers of the quantized
   tables (Q4_K and Q8_0 token embeddings, Q5_0 and Q8_0 PLE table) exact. The qwen4exp table is identical at both
   pins; the glm5-next Q4_K_M rows differ in the third digit (the dumps differ, finding 5).
2. **The activation quantization is the effect to model, not the weights alone.** Without it (dequantized weights only,
   the reference CORRECTNESS.md proposed) the same products are off by 2.6e-4 to 5.8e-2 normwise (qwen4exp) and 1.8e-3
   to 1.5e-2 (glm5-next): scaled >= 5.5e-5 in every graph and >= 8.6e-4 in some graph of every run, at least 230 times
   the worst emulated scaled error (2.4e-7). llama.cpp quantizes every activation row to 8 bits (Q8_0 or Q8_K) or to f16 before a quantized or
   F16 product; F32 weights (routers, indexer projections, shared-expert gate) leave it f32 (there the two oracles
   coincide exactly).
3. **The exact quantizer of the build matters, and the tests resolve it.** The AVX2 Q8_0 activation quantizer (id =
   127/amax, round half to even) and the scalar reference (id = 1/d, `roundf`) differ on a few elements per run;
   mutant Q2 (the reference rule) fails test_matmuls in 4 of the 8 qwen4exp runs and 4 of the 8 glm5-next runs at both
   pins (mutation output below): the per-product tolerance resolves one activation integer off by one. Q8_K: the
   repacked four-row quantizer and the row quantizer agree (both Q4_K_M paths pass).
4. **Repacked and plain Q4_K paths give the same values within TOL_OP.** The default runs use the AVX2 repacked Q4_K
   kernels (7.12 MiB `CPU_REPACK` buffer), the `-nr` runs the plain ones (no such buffer); both pass every check. The
   repacked kernel sums in another order: 4 hc_gate rows of pf64 (3 of the attention mixers, 1 of the FFN mixers) have
   an f16 input on the other side of a boundary, 0 in pf64-nr.
5. **Same files and (qwen4exp) same numbers at both pins.** The quantized GGUFs are byte-identical at ec7630a and
   79e2e74 (sha256 below). qwen4exp: of the dumped tensors both pins have, only `indexer_*` of the QSA layer differ (the
   k-pool padding change, N6); every product input and output is bit-identical. glm5-next: the DSA attention output
   `kqv_out` differs (the pins attend differently: ec7630a gathers latents for ubatches <= 16 tokens, N9); in the Q8_0
   model that is all (the Q8_0 activation quantization of `attn_out` absorbs the f32 differences, every later tensor is
   bit-identical), in Q4_K_M 9555 of 19336 tensors differ (a Q8_K block rounds differently somewhere and the difference
   travels on); the per-product checks pass on both.
6. **A whole-model 1e-4 comparison is not possible on quantized weights; per product (or statistics) is.** The
   whole oracle from the token ids on the pf64 run (probe below): with the activation quantization emulated, layer
   outputs drift from 8e-8 (layer 0) to 3.8e-2 (layer 7) in Q8_0 and from 8e-6 to 3.8e-2 in Q4_K_M; logits 4.4e-2 and
   3.2e-2; greedy tokens 64/64 and 63/64; 1 and 2 MoE expert sets differ (oracle margins 1.3e-3, 4.0e-3, 9.4e-4: not
   ties). Without the activation quantization: 1.1e-2 to 9.6e-2 per layer (7.6e-2 and 8.0e-2 at the last layer); with
   the unquantized f32 weights 6.1e-2 to 2.1e-1, 25 and 92 differing expert sets, greedy 60/64 and 47/64. Even one layer from llama.cpp's own layer input
   reaches 2.6e-4 (Q8_0 layer 5) and 1.8e-3 (Q4_K_M layer 4). Mechanism, Q4_K_M layer 4: 3 of the 512 f16 inputs of
   the first hc_gate up projection round the other way (oracle float64 vs llama.cpp f32), hc_gate 2.9e-4, hc_mixed
   1.7e-5; the next Q8_K activation quantization turns that into whole steps (one step = 1/127 of the block maximum):
   qkv 8.1e-4, linear_attn_out 5.9e-3, ffn_out 1.4e-2. PROVISIONAL consequence for the engine work: parity of our
   engine with llama.cpp on quantized weights has to be per product from dumped inputs (this method) or statistical
   (KLD / top-1, B24); a whole-model elementwise tolerance would have to be about 5e-2 on this random model and would
   then tell a missing activation quantization apart by a factor of 2 at most (last layer 3.8e-2 against 7.6e-2 /
   8.0e-2).
7. **llama-quantize vs the qwen4exp PLE gammas** (section above): the CPU graph aborts on a model quantized without an
   override, at both pins.
8. **glm5-next: an unnamed input recomputed in f32 matches.** The KDA output gate recomputed in f32 the way ggml does
   gives every row of kda_out within TOL_OP (max rel 8.6e-8 in the probe); the same gate in float64 rounded once to f32
   misses one row of Q8_0 pf64 layer 4 by 2.3e-4: the least-squares input change of that row is spread over one 32-block
   in proportion to |q| (+0.096, +0.041, ... steps), the signature of the block's f16 scale rounding the other way, which
   `flip_bound`'s binary16 term covers (probe below).
9. **Mutation check** (literal output below): ec7630a caught 18/18, 79e2e74 caught 18/18; every mutant fails at least one test of a product
   check or a property test, every llama-side mutant exactly the check of the product it touches. Q1 (no activation
   quantization) makes the glm5-next classes error in setUpClass (`flip_bound` has no bound for unquantized
   activations) besides failing the qwen4exp and property tests. A first round on the qwen4exp-only code (`156d93f`,
   14 mutants, the same Q1-Q6, G1-G3, T1, D1-D4) caught 14/14 at both pins as well (not pasted; the code changed after
   it).
10. **CI time** (PROVISIONAL, local wall times on this 4-core sandbox, other agents' processes running): the driver takes
   58 s to 1 min 21 s with `dump-tensors` already built (it is, by the earlier steps of both jobs); building the
   `llama-quantize` target added 28 s here. About 1.5 to 2 min more for `cpu-parity` and for `parity-r1-pin`.

### Not covered (gap 2)

* MXFP4 (mimo2, the optional item), IQ4_XS and the other IQ types, Q2_K, Q3_K, Q4_0: NOT RUN (Q4_0 / IQ4_NL / MXFP4 take
  the emulated Q8_0 path, the others Q8_K, but no model here has them). Q4_1 / Q5_1: NOT RUN; their Q8_1 activations
  carry an f16 sum that enters the minimum term, not emulated (`act_quant` raises).
* The real UD-Q4_K_XL per-tensor mix: NOT RUN; which types it holds beyond Q4_K / Q5_K / Q6_K / Q8_0 is UNKNOWN here.
  imatrix quantization (changes the weights, not the CPU product): NOT RUN.
* `-t` > 1: NOT RUN; the conversion splits each row into per-thread block ranges (ggml-cpu.c:1359-1360, 1635-1636), blocks are independent,
  so the values cannot change (PROVISIONAL, from the source).
* Quantized MTP blocks, quantized KV cache: NOT RUN. Whole-model statistical parity (KLD / top-1) on quantized tiny
  models: NOT RUN.
* GPU: NOT RUN. CUDA quantizes activations for its quantized products differently (q8_1 tiles for MMQ / MMVQ): UNKNOWN
  here, not read.
* Real weights: NOT RUN.


### Literal outputs (part B)

Generators: the default outputs before and after `--quant-dims` (sha256, first 16 hex digits; "before" = the file at the parent commit of the change, gguf-py of ec7630a):

```
$ for a in "" --mtp "--seed 3" ["--arch glm5next"]; do <before>.py b.gguf $a; <after>.py a.gguf $a; ...; done
qwen4exp_tiny.py : before 0fe7442121108a67 after 0fe7442121108a67
qwen4exp_tiny.py --mtp: before 23bf70ffd45b0a45 after 23bf70ffd45b0a45
qwen4exp_tiny.py --seed 3: before eb04bf4fd733eaa0 after eb04bf4fd733eaa0
glm5next_tiny.py : before 26c83267fe1a7588 after 26c83267fe1a7588
glm5next_tiny.py --mtp: before f7833b6ce213f0fe after f7833b6ce213f0fe
glm5next_tiny.py --seed 3: before bd479a47fb26f16c after bd479a47fb26f16c
glm5next_tiny.py --arch glm5next: before 8a63da35ebde0f64 after 8a63da35ebde0f64
```

The PLE gammas without an override (`docs/evidence/mutation-scripts/2026-10-10-n10-quant-abort-probe.sh`, both pins; the bash job-control line is the shell's own report of the abort):

```
$ for pin: docs/evidence/mutation-scripts/2026-10-10-n10-quant-abort-probe.sh <llama.cpp> <g0 build> <scratch>/n10/abort-<pin>
== ec (ec7630a640789c393694fb194f1bbbf0369fc62d)
llama-quantize rc=0; types of the PLE gammas:
  blk.1.ple_norm_conv.weight Q8_0 [256, 4]
  blk.1.ple_norm_key.weight Q8_0 [256, 4]
  blk.1.ple_norm_query.weight Q8_0 [256, 4]
docs/evidence/mutation-scripts/2026-10-10-n10-quant-abort-probe.sh: line 17: 13012 Aborted                 env DUMP_DIR="$OUT/dump" DUMP_TOKENS="5,6,7,8,9,10,11,12" DUMP_PER_GRAPH=1 DUMP_PREFIXES="result_" "$BUILD/dump-tensors" -m "$OUT/q8_0.gguf" -c 512 -b 512 -ub 512 -t 1 -ctk f32 -ctv f32 -fa off > "$OUT/run.log" 2>&1
dump-tensors rc=134
<scratch>/llama-ec7630a/ggml/src/ggml-cpu/binary-ops.cpp:135: binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0
== r1 (79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6)
llama-quantize rc=0; types of the PLE gammas:
  blk.1.ple_norm_conv.weight Q8_0 [256, 4]
  blk.1.ple_norm_key.weight Q8_0 [256, 4]
  blk.1.ple_norm_query.weight Q8_0 [256, 4]
docs/evidence/mutation-scripts/2026-10-10-n10-quant-abort-probe.sh: line 17: 13043 Aborted                 env DUMP_DIR="$OUT/dump" DUMP_TOKENS="5,6,7,8,9,10,11,12" DUMP_PER_GRAPH=1 DUMP_PREFIXES="result_" "$BUILD/dump-tensors" -m "$OUT/q8_0.gguf" -c 512 -b 512 -ub 512 -t 1 -ctk f32 -ctv f32 -fa off > "$OUT/run.log" 2>&1
dump-tensors rc=134
<scratch>/llama.cpp/ggml/src/ggml-cpu/binary-ops.cpp:135: binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0
```

Per-tensor types and sha256 of the quantized models of the final driver runs (`2026-10-10-n10-quant-types.py`, gguf-py of each pin; the 79e2e74 output is identical: `diff` printed nothing):

```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-types.py <gguf-py> <work>/{Q8_0,Q4_K_M,glm5next/Q8_0,glm5next/Q4_K_M}/model.gguf > types-<pin>.txt; diff types-ec.txt types-r1.txt && echo "types and sha256 identical at both pins"
Q8_0/model.gguf: 214 tensors, sha256 28488db5ab74d34b701b4aea05bc6b41ada13690d799e820be8db3ffa28c2a5a
  attn_gate.weight                 Q8_0   (256, 256)         x6
  attn_k.weight                    Q8_0   (256, 128)         x2
  attn_k_norm.weight               F32    (64,)              x2
  attn_output.weight               Q8_0   (256, 256)         x2
  attn_q.weight                    Q8_0   (256, 512)         x2
  attn_q_norm.weight               F32    (64,)              x2
  attn_qkv.weight                  Q8_0   (256, 512)         x6
  attn_v.weight                    Q8_0   (256, 128)         x2
  ffn_down_exps.weight             Q8_0   (256, 256, 8)      x8
  ffn_down_shexp.weight            Q8_0   (256, 256)         x8
  ffn_gate_exps.weight             Q8_0   (256, 256, 8)      x8
  ffn_gate_inp.weight              F32    (256, 8)           x8
  ffn_gate_inp_shexp.weight        F32    (256,)             x8
  ffn_gate_shexp.weight            Q8_0   (256, 256)         x8
  ffn_up_exps.weight               Q8_0   (256, 256, 8)      x8
  ffn_up_shexp.weight              Q8_0   (256, 256)         x8
  hc_attn_down.weight              Q8_0   (1024, 8)          x8
  hc_attn_inject.weight            Q8_0   (1024, 4)          x8
  hc_attn_norm.weight              F32    (256, 4)           x8
  hc_attn_up.weight                F16    (8, 1024)          x8
  hc_ffn_down.weight               Q8_0   (1024, 8)          x8
  hc_ffn_inject.weight             Q8_0   (1024, 4)          x8
  hc_ffn_norm.weight               F32    (256, 4)           x8
  hc_ffn_up.weight                 F16    (8, 1024)          x8
  indexer.k_norm.weight            F32    (16,)              x2
  indexer.k_proj.weight            F32    (256, 16)          x2
  indexer.q_norm.weight            F32    (16,)              x2
  indexer.q_proj.weight            F32    (256, 64)          x2
  output.weight                    Q8_0   (256, 96)          x1
  output_hc_down.weight            Q8_0   (1024, 8)          x1
  output_hc_norm.weight            F32    (256, 4)           x1
  output_hc_up.weight              F16    (8, 1024)          x1
  per_layer_token_embd.weight      Q8_0   (64, 168)          x1
  ple_conv1d.weight                F16    (3, 1024)          x1
  ple_key.weight                   Q8_0   (256, 1024)        x1
  ple_norm_conv.weight             F32    (256, 4)           x1
  ple_norm_key.weight              F32    (256, 4)           x1
  ple_norm_query.weight            F32    (256, 4)           x1
  ple_value.weight                 Q8_0   (256, 256)         x1
  ssm_a                            F32    (4,)               x6
  ssm_alpha.weight                 Q8_0   (256, 4)           x6
  ssm_beta.weight                  Q8_0   (256, 4)           x6
  ssm_conv1d.weight                F32    (4, 512)           x6
  ssm_dt.bias                      F32    (4,)               x6
  ssm_norm.weight                  F32    (64,)              x6
  ssm_out.weight                   Q8_0   (256, 256)         x6
  token_embd.weight                Q8_0   (256, 96)          x1
Q4_K_M/model.gguf: 214 tensors, sha256 6c52cf0058de57b35e29172a28e57c480f35242104f9943ce7d563351a474de5
  attn_gate.weight                 Q4_K   (256, 256)         x6
  attn_k.weight                    Q8_0   (256, 128)         x2
  attn_k_norm.weight               F32    (64,)              x2
  attn_output.weight               Q5_K   (256, 256)         x2
  attn_q.weight                    Q4_K   (256, 512)         x2
  attn_q_norm.weight               F32    (64,)              x2
  attn_qkv.weight                  Q8_0   (256, 512)         x6
  attn_v.weight                    Q8_0   (256, 128)         x2
  ffn_down_exps.weight             Q4_K   (256, 256, 8)      x4
  ffn_down_exps.weight             Q6_K   (256, 256, 8)      x4
  ffn_down_shexp.weight            Q4_K   (256, 256)         x4
  ffn_down_shexp.weight            Q6_K   (256, 256)         x4
  ffn_gate_exps.weight             Q4_K   (256, 256, 8)      x8
  ffn_gate_inp.weight              F32    (256, 8)           x8
  ffn_gate_inp_shexp.weight        F32    (256,)             x8
  ffn_gate_shexp.weight            Q4_K   (256, 256)         x8
  ffn_up_exps.weight               Q4_K   (256, 256, 8)      x8
  ffn_up_shexp.weight              Q4_K   (256, 256)         x8
  hc_attn_down.weight              Q4_K   (1024, 8)          x8
  hc_attn_inject.weight            Q4_K   (1024, 4)          x8
  hc_attn_norm.weight              F32    (256, 4)           x8
  hc_attn_up.weight                F16    (8, 1024)          x8
  hc_ffn_down.weight               Q4_K   (1024, 8)          x4
  hc_ffn_down.weight               Q6_K   (1024, 8)          x4
  hc_ffn_inject.weight             Q4_K   (1024, 4)          x8
  hc_ffn_norm.weight               F32    (256, 4)           x8
  hc_ffn_up.weight                 F16    (8, 1024)          x8
  indexer.k_norm.weight            F32    (16,)              x2
  indexer.k_proj.weight            F32    (256, 16)          x2
  indexer.q_norm.weight            F32    (16,)              x2
  indexer.q_proj.weight            F32    (256, 64)          x2
  output.weight                    Q6_K   (256, 96)          x1
  output_hc_down.weight            Q4_K   (1024, 8)          x1
  output_hc_norm.weight            F32    (256, 4)           x1
  output_hc_up.weight              F16    (8, 1024)          x1
  per_layer_token_embd.weight      Q5_0   (64, 168)          x1
  ple_conv1d.weight                F16    (3, 1024)          x1
  ple_key.weight                   Q4_K   (256, 1024)        x1
  ple_norm_conv.weight             F32    (256, 4)           x1
  ple_norm_key.weight              F32    (256, 4)           x1
  ple_norm_query.weight            F32    (256, 4)           x1
  ple_value.weight                 Q4_K   (256, 256)         x1
  ssm_a                            F32    (4,)               x6
  ssm_alpha.weight                 Q4_K   (256, 4)           x6
  ssm_beta.weight                  Q4_K   (256, 4)           x6
  ssm_conv1d.weight                F32    (4, 512)           x6
  ssm_dt.bias                      F32    (4,)               x6
  ssm_norm.weight                  F32    (64,)              x6
  ssm_out.weight                   Q4_K   (256, 256)         x6
  token_embd.weight                Q4_K   (256, 96)          x1
Q8_0/model.gguf: 370 tensors, sha256 523c5f96ddd49924160e93d571aaf53b838e88a1629588b26076c413547a4405
  attn_k.weight                    Q8_0   (256, 256)         x9
  attn_k_b.weight                  F32    (16, 16, 4)        x3
  attn_kv_a_mqa.weight             F32    (256, 16)          x3
  attn_kv_a_norm.weight            F32    (16,)              x3
  attn_norm.weight                 F32    (256,)             x12
  attn_output.weight               Q8_0   (256, 256)         x12
  attn_q.weight                    Q8_0   (256, 256)         x9
  attn_q_a.weight                  Q8_0   (256, 256)         x3
  attn_q_a_norm.weight             F32    (256,)             x3
  attn_q_b.weight                  Q8_0   (256, 64)          x3
  attn_v.weight                    Q8_0   (256, 256)         x9
  attn_v_b.weight                  F32    (16, 64, 4)        x3
  exp_probs_b.bias                 F32    (8,)               x11
  ffn_down.weight                  Q8_0   (256, 256)         x1
  ffn_down_exps.weight             Q8_0   (256, 256, 8)      x11
  ffn_down_shexp.weight            Q8_0   (256, 256)         x11
  ffn_gate.weight                  Q8_0   (256, 256)         x1
  ffn_gate_exps.weight             Q8_0   (256, 256, 8)      x11
  ffn_gate_inp.weight              F32    (256, 8)           x11
  ffn_gate_shexp.weight            Q8_0   (256, 256)         x11
  ffn_norm.weight                  F32    (256,)             x12
  ffn_up.weight                    Q8_0   (256, 256)         x1
  ffn_up_exps.weight               Q8_0   (256, 256, 8)      x11
  ffn_up_shexp.weight              Q8_0   (256, 256)         x11
  hc_attn_base.weight              F32    (24,)              x12
  hc_attn_fn.weight                F32    (1024, 24)         x12
  hc_attn_scale.weight             F32    (3,)               x12
  hc_ffn_base.weight               F32    (24,)              x12
  hc_ffn_fn.weight                 F32    (1024, 24)         x12
  hc_ffn_scale.weight              F32    (3,)               x12
  indexer.attn_k.weight            F32    (256, 16)          x3
  indexer.attn_q_b.weight          F32    (256, 32)          x3
  indexer.k_norm.bias              F32    (16,)              x3
  indexer.k_norm.weight            F32    (16,)              x3
  indexer.proj.weight              F32    (256, 2)           x3
  indexer_compressor_ape.weight    F32    (16, 4)            x3
  indexer_compressor_gate.weight   F32    (256, 16)          x3
  output.weight                    Q8_0   (256, 96)          x1
  output_norm.weight               F32    (256,)             x1
  ssm_a                            F32    (4,)               x9
  ssm_beta.weight                  F32    (256, 4)           x9
  ssm_conv1d_k.weight              F32    (4, 1, 256)        x9
  ssm_conv1d_q.weight              F32    (4, 1, 256)        x9
  ssm_conv1d_v.weight              F32    (4, 1, 256)        x9
  ssm_dt.bias                      F32    (256,)             x9
  ssm_f_a.weight                   F32    (256, 64)          x9
  ssm_f_b.weight                   F32    (64, 256)          x9
  ssm_g_a.weight                   F32    (256, 64)          x9
  ssm_g_b.weight                   F32    (64, 256)          x9
  ssm_norm.weight                  F32    (64,)              x9
  token_embd.weight                Q8_0   (256, 96)          x1
Q4_K_M/model.gguf: 370 tensors, sha256 1935ca7939522d61c4a4765fbea79ba303f56db445570b8a2dcbe3edc464e947
  attn_k.weight                    Q8_0   (256, 256)         x9
  attn_k_b.weight                  F32    (16, 16, 4)        x3
  attn_kv_a_mqa.weight             F32    (256, 16)          x3
  attn_kv_a_norm.weight            F32    (16,)              x3
  attn_norm.weight                 F32    (256,)             x12
  attn_output.weight               Q5_K   (256, 256)         x12
  attn_q.weight                    Q4_K   (256, 256)         x9
  attn_q_a.weight                  Q8_0   (256, 256)         x3
  attn_q_a_norm.weight             F32    (256,)             x3
  attn_q_b.weight                  Q8_0   (256, 64)          x3
  attn_v.weight                    Q8_0   (256, 256)         x9
  attn_v_b.weight                  F32    (16, 64, 4)        x3
  exp_probs_b.bias                 F32    (8,)               x11
  ffn_down.weight                  Q6_K   (256, 256)         x1
  ffn_down_exps.weight             Q4_K   (256, 256, 8)      x6
  ffn_down_exps.weight             Q6_K   (256, 256, 8)      x5
  ffn_down_shexp.weight            Q4_K   (256, 256)         x6
  ffn_down_shexp.weight            Q6_K   (256, 256)         x5
  ffn_gate.weight                  Q4_K   (256, 256)         x1
  ffn_gate_exps.weight             Q4_K   (256, 256, 8)      x11
  ffn_gate_inp.weight              F32    (256, 8)           x11
  ffn_gate_shexp.weight            Q4_K   (256, 256)         x11
  ffn_norm.weight                  F32    (256,)             x12
  ffn_up.weight                    Q4_K   (256, 256)         x1
  ffn_up_exps.weight               Q4_K   (256, 256, 8)      x11
  ffn_up_shexp.weight              Q4_K   (256, 256)         x11
  hc_attn_base.weight              F32    (24,)              x12
  hc_attn_fn.weight                F32    (1024, 24)         x12
  hc_attn_scale.weight             F32    (3,)               x12
  hc_ffn_base.weight               F32    (24,)              x12
  hc_ffn_fn.weight                 F32    (1024, 24)         x12
  hc_ffn_scale.weight              F32    (3,)               x12
  indexer.attn_k.weight            F32    (256, 16)          x3
  indexer.attn_q_b.weight          F32    (256, 32)          x3
  indexer.k_norm.bias              F32    (16,)              x3
  indexer.k_norm.weight            F32    (16,)              x3
  indexer.proj.weight              F32    (256, 2)           x3
  indexer_compressor_ape.weight    F32    (16, 4)            x3
  indexer_compressor_gate.weight   F32    (256, 16)          x3
  output.weight                    Q6_K   (256, 96)          x1
  output_norm.weight               F32    (256,)             x1
  ssm_a                            F32    (4,)               x9
  ssm_beta.weight                  F32    (256, 4)           x9
  ssm_conv1d_k.weight              F32    (4, 1, 256)        x9
  ssm_conv1d_q.weight              F32    (4, 1, 256)        x9
  ssm_conv1d_v.weight              F32    (4, 1, 256)        x9
  ssm_dt.bias                      F32    (256,)             x9
  ssm_f_a.weight                   F32    (256, 64)          x9
  ssm_f_b.weight                   F32    (64, 256)          x9
  ssm_g_a.weight                   F32    (256, 64)          x9
  ssm_g_b.weight                   F32    (64, 256)          x9
  ssm_norm.weight                  F32    (64,)              x9
  token_embd.weight                Q4_K   (256, 96)          x1
types and sha256 identical at both pins
```

The f32 sources (identical at both pins):

```
$ sha256sum <work-ec>/qwen4exp-f32.gguf <work-r1>/qwen4exp-f32.gguf <work-ec>/glm5next/glm5next-f32.gguf <work-r1>/glm5next/glm5next-f32.gguf
5a3e071e0cc9c6b30d46e0c47595294325c734812747fa3be25a8c68415d21ee  <scratch>/n10/qw-ec/qwen4exp-f32.gguf
5a3e071e0cc9c6b30d46e0c47595294325c734812747fa3be25a8c68415d21ee  <scratch>/n10/qw-r1/qwen4exp-f32.gguf
bc3f35128ba18221e84131ce8044f5448e09b7b58bdbcf44651fe5a4919deb73  <scratch>/n10/qw-ec/glm5next/glm5next-f32.gguf
bc3f35128ba18221e84131ce8044f5448e09b7b58bdbcf44651fe5a4919deb73  <scratch>/n10/qw-r1/glm5next/glm5next-f32.gguf
```

Repacking (`-lv 4` prints the buffer sizes; same model and flags as the runs):

```
$ dump-tensors -m <work-ec>/Q4_K_M/model.gguf ... -lv 4  2>&1 | grep 'model buffer size'
I load_tensors:   CPU_Mapped model buffer size =    10.72 MiB
I load_tensors:   CPU_REPACK model buffer size =     7.12 MiB
$ dump-tensors -m <work-ec>/Q4_K_M/model.gguf ... -lv 4 --no-repack 2>&1 | grep 'model buffer size'
I load_tensors:   CPU_Mapped model buffer size =    10.72 MiB
$ dump-tensors -m <work-r1>/Q4_K_M/model.gguf ... -lv 4  2>&1 | grep 'model buffer size'
I load_tensors:   CPU_Mapped model buffer size =    10.72 MiB
I load_tensors:   CPU_REPACK model buffer size =     7.12 MiB
$ dump-tensors -m <work-r1>/Q4_K_M/model.gguf ... -lv 4 --no-repack 2>&1 | grep 'model buffer size'
I load_tensors:   CPU_Mapped model buffer size =    10.72 MiB
```

Pins: the source files of part B at ec7630a and 79e2e74:

```
$ for f in ...; do diff -q llama-ec7630a/$f llama.cpp/$f ...; done
differs   ggml/src/ggml-cpu/ggml-cpu.c (4 changed lines; hunks: 1308c1308 1387c1387 )
same      ggml/src/ggml-cpu/arch/x86/quants.c
same      ggml/src/ggml-cpu/arch/x86/repack.cpp
differs   ggml/src/ggml-cpu/repack.cpp (14 changed lines; hunks: 5241,5246c5241,5248 )
differs   ggml/src/ggml-quants.c (5 changed lines; hunks: 1039,1040c1039,1041 )
same      ggml/src/ggml-cpu/binary-ops.cpp
same      ggml/src/ggml-cpu/unary-ops.cpp
differs   ggml/src/ggml-cpu/llamafile/sgemm.cpp (93 changed lines; hunks: 386a387,460 494a569 496a572 551c627 576a653,667 )
same      src/llama-quant.cpp
same      tools/quantize/quantize.cpp
same      gguf-py/gguf/quants.py
same      ggml/src/ggml-common.h
$ diff llama-ec7630a/ggml/src/ggml-cpu/ggml-cpu.c llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
1308c1308
<     if (src1_cont) {
---
>     if (!params->use_ref && src1_cont) {
1387c1387
<     if (src1->type != vec_dot_type) {
---
>     if (!params->use_ref && src1->type != vec_dot_type) {
```

Source lines (printed from the ec7630a checkout):

```
$ sed -n "300p;333,346p" src/llama-quant.cpp   # tensor_allows_quantization: "_norm.weight" rule, the glm5-next exclusions
    quantize &= name.find("_norm.weight") == std::string::npos;
        quantize &= name.find("hc_")                     == std::string::npos;
        quantize &= name.find("indexer.attn_q_b")        == std::string::npos;
        quantize &= name.find("indexer.attn_k")          == std::string::npos;
        quantize &= name.find("indexer.proj")            == std::string::npos;
        quantize &= name.find("indexer_compressor_gate") == std::string::npos;
        quantize &= name.find("indexer_compressor_ape")  == std::string::npos;
        quantize &= name.find("ssm_f_a.weight")          == std::string::npos;
        quantize &= name.find("ssm_f_b.weight")          == std::string::npos;
        quantize &= name.find("ssm_g_a.weight")          == std::string::npos;
        quantize &= name.find("ssm_g_b.weight")          == std::string::npos;
        quantize &= name.find("ssm_beta.weight")         == std::string::npos;
        quantize &= name.find("attn_kv_a_mqa.weight")    == std::string::npos;
        quantize &= name.find("attn_k_b.weight")         == std::string::npos;
        quantize &= name.find("attn_v_b.weight")         == std::string::npos;
$ sed -n "390p;396p;416,418p;429p;438p" src/llama-quant.cpp   # tensor_type_fallback
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
        if (ncols % ggml_blck_size(return_type) != 0) {
            return_type = GGML_TYPE_F16;
$ grep -n "define QK_K" ggml/src/ggml-common.h
89:#define QK_K 256
$ sed -n "1278,1279p;1371p;1624p;1637p" ggml/src/ggml-cpu/ggml-cpu.c   # vec_dot_type of src0, its from_float, the src1 conversion (mul_mat; mul_mat_id)
    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
                        from_float((const float *) src1_row, wdata_row, ne10_block_size);
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
$ sed -n "225p;243p;263p;275p;314p;324p;330p" ggml/src/ggml-cpu/ggml-cpu.c   # vec_dot_type of F16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K
        .vec_dot_type             = GGML_TYPE_F16,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
$ sed -n "331,337p;343p" ggml/src/ggml-cpu/arch/x86/quants.c   # AVX2 quantize_row_q8_0
        const float d = maxScalar / 127.f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
$ sed -n "276p;288,289p;296p" ggml/src/ggml-quants.c   # scalar quantize_row_q8_0_ref (not used by this build)
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
            y[i].qs[j] = roundf(x0);
$ sed -n "505,507p" ggml/src/ggml-cpu/arch/x86/quants.c; sed -n "2789,2793p;2803p" ggml/src/ggml-quants.c
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        x += QK_K;
$ sed -n "5006,5010p;4697,4705p;4839,4841p" ggml/src/ggml-cpu/repack.cpp   # Q4_K repack on AVX2; activation rows by fours, the rest by from_float; mul_mat_id
            for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>((float *) (data_ptr + i11 * nb11),
                                                            (void *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            const int64_t i11_processed = ne11 - ne11 % 4;
            for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((float *) (data_ptr + i11 * nb11), (void *) (wdata_ptr + i11 * nbw1), ne10);
            }
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
                           (void *)               (wdata + i12 * nbw2 + i11 * nbw1),
    } else if (cur->type == GGML_TYPE_Q4_K) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
$ sed -n "382,388p" ggml/src/ggml-cpu/arch/x86/repack.cpp   # ggml_quantize_mat_q8_K_4x8 scale sign
            iscale[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;

            if(mask) {
                iscale[row_iter] = ( maxScalar != 0.0f ) ? -127.f / maxScalar: 0.0f;
            }

            y[i].d[row_iter] = maxScalar ? 1/iscale[row_iter] : 0;
$ sed -n "3966,3973p" ggml/src/ggml-cpu/llamafile/sgemm.cpp   # F16 tinyBLAS on AVX2 only with an F16 B
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, float> tb{ params, k,
                (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
$ sed -n "2421,2428p" common/arg.cpp
    add_opt(common_arg(
        {"--repack"},
        {"-nr", "--no-repack"},
        string_format("whether to enable weight repacking (default: %s)", params.no_extra_bufts ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_extra_bufts = !value;
        }
    ).set_env("LLAMA_ARG_REPACK"));
$ sed -n "1384,1388p;1405,1408p" src/models/qwen4exp.cpp   # the PLE gammas applied by ggml_mul
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);
    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);
$ sed -n "270,272p" src/models/qwen4exp.cpp; sed -n "164,166p" conversion/qwen4exp.py
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]
$ sed -n "896,905p" src/llama-model-loader.cpp   # TENSOR_ALLOW_RESHAPE checks the element count only
    if (allow_reshape) {
        // check total number of elements only
        const int64_t ncur = ggml_nelements(cur);
        int64_t nexp = 1;
        for (size_t i = 0; i < ne.size(); ++i) {
            nexp *= ne[i];
        }
        if (ncur != nexp) {
            is_ok = false;
        }
$ sed -n "135p" ggml/src/ggml-cpu/binary-ops.cpp
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
$ sed -n "31,33p" ggml/src/ggml-cpu/unary-ops.cpp
static inline float op_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}
```

Driver, ec7630a (`time QUANT_WORK=<scratch>/n10/qw-ec tools/tinygen/run_parity_quant.sh <llama.cpp> <scratch>/n10/build-ec`; cut: the 99 verbose `... ok` lines of unittest -v, and the tables, which follow):

```
Q8_0 pf64: 1 graphs, 1 calls
Q8_0 ub10: 7 graphs, 1 calls
Q8_0 decode: 17 graphs, 17 calls
glm5next/Q8_0 pf64: 1 graphs, 1 calls
glm5next/Q8_0 ub10: 7 graphs, 1 calls
glm5next/Q8_0 decode: 17 graphs, 17 calls
Q4_K_M pf64: 1 graphs, 1 calls
Q4_K_M ub10: 7 graphs, 1 calls
Q4_K_M decode: 17 graphs, 17 calls
Q4_K_M pf64-nr: 1 graphs, 1 calls
Q4_K_M decode-nr: 17 graphs, 17 calls
glm5next/Q4_K_M pf64: 1 graphs, 1 calls
glm5next/Q4_K_M ub10: 7 graphs, 1 calls
glm5next/Q4_K_M decode: 17 graphs, 17 calls
glm5next/Q4_K_M pf64-nr: 1 graphs, 1 calls
glm5next/Q4_K_M decode-nr: 17 graphs, 17 calls
[99 lines '... ok' cut]
Ran 99 tests in 41.134s
OK
artifacts: <scratch>/n10/qw-ec
real	1m21.380s
user	1m49.223s
sys	0m42.115s
```

Driver, 79e2e74 (`time QUANT_WORK=<scratch>/n10/qw-r1 tools/tinygen/run_parity_quant.sh <llama.cpp> <scratch>/n10/build-r1`; cut: the 99 verbose `... ok` lines of unittest -v, and the tables, which follow):

```
Q8_0 pf64: 1 graphs, 1 calls
Q8_0 ub10: 7 graphs, 1 calls
Q8_0 decode: 17 graphs, 17 calls
glm5next/Q8_0 pf64: 1 graphs, 1 calls
glm5next/Q8_0 ub10: 7 graphs, 1 calls
glm5next/Q8_0 decode: 17 graphs, 17 calls
Q4_K_M pf64: 1 graphs, 1 calls
Q4_K_M ub10: 7 graphs, 1 calls
Q4_K_M decode: 17 graphs, 17 calls
Q4_K_M pf64-nr: 1 graphs, 1 calls
Q4_K_M decode-nr: 17 graphs, 17 calls
glm5next/Q4_K_M pf64: 1 graphs, 1 calls
glm5next/Q4_K_M ub10: 7 graphs, 1 calls
glm5next/Q4_K_M decode: 17 graphs, 17 calls
glm5next/Q4_K_M pf64-nr: 1 graphs, 1 calls
glm5next/Q4_K_M decode-nr: 17 graphs, 17 calls
[99 lines '... ok' cut]
Ran 99 tests in 41.583s
OK
artifacts: <scratch>/n10/qw-r1
real	0m57.536s
user	1m47.627s
sys	0m19.053s
```

The tables of the ec7630a run (stderr of the tests, complete):

```
qwen4exp: quantized products from llama.cpp's own inputs (ec7630a640789c393694fb194f1bbbf0369fc62d), worst over layers and graphs. scaled = max|got-want| / max(|W| |x_q|) (asserted < TOL_OP = 1e-05; for a bounded product: the largest share of its bound, asserted <= 1); rel = max|got-want| / max|want|; no-act = the same without the activation quantization (min .. max over graphs):
type    run       product                                      weight       scaled       rel layer graph         no-act scaled            no-act rel
Q4_K_M  decode    hc_init, ple_embd (gathers, exact)           Q4_K,Q5_0  0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  decode    hc_inject (attn mixer)                       Q4_K       6.41e-08  1.88e-06     7     6  1.59e-04 .. 1.21e-03  1.89e-03 .. 3.23e-02
Q4_K_M  decode    hc_gate (attn mixer: down, f16 up)           Q4_K+F16   4.74e-01  1.27e-04     4    13                     -  2.26e-03 .. 3.23e-02
Q4_K_M  decode    hc_inject (ffn mixer)                        Q4_K       5.95e-08  2.35e-06     1    10  1.71e-04 .. 1.29e-03  2.06e-03 .. 4.70e-02
Q4_K_M  decode    hc_gate (ffn mixer: down, f16 up)            Q6_K+F16   2.01e-04  8.84e-08     6     0                     -  2.57e-03 .. 1.90e-02
Q4_K_M  decode    linear_attn_qkv_mixed                        Q8_0       4.06e-08  1.63e-07     5     3  1.09e-03 .. 2.03e-03  3.19e-03 .. 7.92e-03
Q4_K_M  decode    z                                            Q4_K       1.57e-07  5.15e-07     2     0  1.24e-03 .. 3.14e-03  4.02e-03 .. 1.10e-02
Q4_K_M  decode    beta                                         Q4_K       7.01e-08  2.27e-06     5     6  2.98e-04 .. 2.50e-03  1.84e-03 .. 5.84e-02
Q4_K_M  decode    alpha                                        Q4_K       8.57e-08  8.08e-07     1     3  2.87e-04 .. 2.68e-03  1.11e-03 .. 2.18e-02
Q4_K_M  decode    linear_attn_out                              Q4_K       2.20e-07  6.53e-07     5     0  1.72e-03 .. 6.11e-03  6.62e-03 .. 2.39e-02
Q4_K_M  decode    ffn_moe_logits                               F32        3.77e-08  4.61e-07     7     0  4.26e-09 .. 3.77e-08  1.70e-08 .. 4.61e-07
Q4_K_M  decode    ffn_moe_gate                                 Q4_K       1.42e-07  4.21e-07     1     0  1.22e-03 .. 3.01e-03  4.12e-03 .. 1.18e-02
Q4_K_M  decode    ffn_moe_up                                   Q4_K       1.37e-07  4.43e-07     3     0  1.19e-03 .. 3.14e-03  3.96e-03 .. 1.04e-02
Q4_K_M  decode    ffn_moe_down                                 Q6_K       8.23e-08  2.83e-07     6     9  2.98e-03 .. 1.08e-02  9.77e-03 .. 3.31e-02
Q4_K_M  decode    ffn_up (shared expert)                       Q4_K       1.58e-07  5.58e-07     2     0  1.19e-03 .. 3.04e-03  3.73e-03 .. 1.28e-02
Q4_K_M  decode    ffn_gate (shared expert)                     Q4_K       1.34e-07  4.52e-07     2     0  1.19e-03 .. 3.21e-03  3.79e-03 .. 1.09e-02
Q4_K_M  decode    ffn_shexp (shared expert down)               Q6_K       7.92e-08  2.70e-07     0     9  2.49e-03 .. 1.02e-02  7.60e-03 .. 2.96e-02
Q4_K_M  decode    shared_expert_gate                           F32        2.97e-08  8.86e-06     5     3  6.61e-12 .. 2.97e-08  6.02e-11 .. 8.86e-06
Q4_K_M  decode    hc_gate (ffn mixer: down, f16 up)            Q4_K+F16   4.70e-01  1.03e-04     5     0                     -  2.93e-03 .. 2.94e-02
Q4_K_M  decode    ffn_moe_down                                 Q4_K       1.13e-07  3.10e-07     2     3  2.90e-03 .. 1.04e-02  7.88e-03 .. 3.10e-02
Q4_K_M  decode    ffn_shexp (shared expert down)               Q4_K       1.12e-07  3.23e-07     4     0  2.43e-03 .. 9.29e-03  8.85e-03 .. 2.82e-02
Q4_K_M  decode    Qcur_full                                    Q4_K       1.33e-07  3.65e-07     3     0  1.40e-03 .. 2.82e-03  4.91e-03 .. 1.05e-02
Q4_K_M  decode    Kcur (projection)                            Q8_0       3.19e-08  1.34e-07     7    11  1.02e-03 .. 1.71e-03  3.92e-03 .. 8.24e-03
Q4_K_M  decode    Vcur (projection)                            Q8_0       3.74e-08  1.19e-07     7     0  9.87e-04 .. 2.02e-03  3.05e-03 .. 8.82e-03
Q4_K_M  decode    indexer_k_raw                                F32        3.23e-08  1.78e-07     3     0  1.27e-08 .. 3.23e-08  6.44e-08 .. 1.78e-07
Q4_K_M  decode    attn_output                                  Q5_K       1.59e-07  6.68e-07     3    13  1.24e-03 .. 2.89e-03  5.66e-03 .. 1.10e-02
Q4_K_M  decode    hc_gate (head mixer: down, f16 up)           Q4_K+F16   1.74e-04  9.17e-08    -1     0                     -  4.10e-03 .. 1.39e-02
Q4_K_M  decode    result_output                                Q6_K       4.93e-08  2.27e-07    -1     1  1.25e-03 .. 2.33e-03  4.90e-03 .. 9.48e-03
Q4_K_M  decode    hc_gate (attn mixer: down, f16 up): rows within TOL_OP 507/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 511/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    hc_gate (head mixer: down, f16 up): rows within TOL_OP 64/64 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    ple_gate (ple_key, ple_value inside)         Q4_K              -  6.10e-07     1     0                     -  5.75e-03 .. 5.75e-03
Q4_K_M  decode    ple_gated_value (ple_key, ple_value inside)  Q4_K              -  3.79e-07     1     0                     -  7.20e-03 .. 7.20e-03
Q4_K_M  decode    ple_conv_out (ple_key, ple_value inside)     Q4_K              -  3.22e-07     1     0                     -  8.69e-03 .. 8.69e-03
Q4_K_M  decode-nr hc_init, ple_embd (gathers, exact)           Q4_K,Q5_0  0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  decode-nr hc_inject (attn mixer)                       Q4_K       6.33e-08  1.85e-06     5    16  7.28e-05 .. 1.20e-03  9.40e-04 .. 2.34e-02
Q4_K_M  decode-nr hc_gate (attn mixer: down, f16 up)           Q4_K+F16   4.04e-01  9.61e-05     1     6                     -  3.33e-03 .. 2.36e-02
Q4_K_M  decode-nr hc_inject (ffn mixer)                        Q4_K       6.94e-08  3.83e-06     3    16  1.35e-04 .. 1.09e-03  1.61e-03 .. 4.60e-02
Q4_K_M  decode-nr hc_gate (ffn mixer: down, f16 up)            Q6_K+F16   1.99e-03  1.08e-07     0     0                     -  2.57e-03 .. 2.03e-02
Q4_K_M  decode-nr linear_attn_qkv_mixed                        Q8_0       4.18e-08  1.35e-07     4     0  9.95e-04 .. 2.09e-03  2.97e-03 .. 8.09e-03
Q4_K_M  decode-nr z                                            Q4_K       1.13e-07  4.18e-07     0    11  1.29e-03 .. 3.14e-03  4.48e-03 .. 1.10e-02
Q4_K_M  decode-nr beta                                         Q4_K       8.87e-08  2.27e-06     4    12  2.50e-04 .. 2.33e-03  1.93e-03 .. 5.84e-02
Q4_K_M  decode-nr alpha                                        Q4_K       8.57e-08  9.62e-07     1     3  2.55e-04 .. 2.31e-03  1.11e-03 .. 2.28e-02
Q4_K_M  decode-nr linear_attn_out                              Q4_K       2.39e-07  8.70e-07     5     6  1.72e-03 .. 6.51e-03  6.60e-03 .. 2.55e-02
Q4_K_M  decode-nr ffn_moe_logits                               F32        3.84e-08  2.94e-07     1     0  5.28e-09 .. 3.84e-08  1.82e-08 .. 2.94e-07
Q4_K_M  decode-nr ffn_moe_gate                                 Q4_K       1.16e-07  4.15e-07     1     0  1.27e-03 .. 3.00e-03  4.17e-03 .. 1.22e-02
Q4_K_M  decode-nr ffn_moe_up                                   Q4_K       1.17e-07  4.28e-07     1     0  1.19e-03 .. 3.14e-03  3.96e-03 .. 1.18e-02
Q4_K_M  decode-nr ffn_moe_down                                 Q6_K       8.81e-08  2.39e-07     0     6  2.98e-03 .. 1.08e-02  8.02e-03 .. 3.31e-02
Q4_K_M  decode-nr ffn_up (shared expert)                       Q4_K       1.02e-07  4.90e-07     1     6  1.24e-03 .. 3.04e-03  3.17e-03 .. 1.28e-02
Q4_K_M  decode-nr ffn_gate (shared expert)                     Q4_K       1.08e-07  4.54e-07     1    16  1.29e-03 .. 3.21e-03  3.48e-03 .. 1.14e-02
Q4_K_M  decode-nr ffn_shexp (shared expert down)               Q6_K       1.00e-07  3.41e-07     0     9  2.51e-03 .. 1.02e-02  6.40e-03 .. 2.96e-02
Q4_K_M  decode-nr shared_expert_gate                           F32        3.19e-08  5.40e-06     6     0  9.60e-11 .. 3.19e-08  9.20e-10 .. 5.40e-06
Q4_K_M  decode-nr hc_gate (ffn mixer: down, f16 up)            Q4_K+F16   6.74e-02  5.78e-06     5     0                     -  2.36e-03 .. 2.94e-02
Q4_K_M  decode-nr ffn_moe_down                                 Q4_K       1.27e-07  3.65e-07     2     3  2.58e-03 .. 1.04e-02  7.41e-03 .. 3.10e-02
Q4_K_M  decode-nr ffn_shexp (shared expert down)               Q4_K       1.23e-07  3.64e-07     5     4  2.13e-03 .. 9.29e-03  7.80e-03 .. 2.82e-02
Q4_K_M  decode-nr Qcur_full                                    Q4_K       8.85e-08  3.33e-07     3    14  1.34e-03 .. 3.17e-03  4.80e-03 .. 1.04e-02
Q4_K_M  decode-nr Kcur (projection)                            Q8_0       2.97e-08  1.37e-07     7    12  9.68e-04 .. 1.74e-03  3.73e-03 .. 8.61e-03
Q4_K_M  decode-nr Vcur (projection)                            Q8_0       3.23e-08  1.23e-07     7     1  1.02e-03 .. 1.93e-03  3.66e-03 .. 6.91e-03
Q4_K_M  decode-nr indexer_k_raw                                F32        3.02e-08  2.23e-07     3     0  1.10e-08 .. 3.02e-08  4.97e-08 .. 2.23e-07
Q4_K_M  decode-nr attn_output                                  Q5_K       1.18e-07  4.96e-07     3    13  1.38e-03 .. 2.86e-03  5.24e-03 .. 1.20e-02
Q4_K_M  decode-nr hc_gate (head mixer: down, f16 up)           Q4_K+F16   5.15e-01  2.37e-04    -1    16                     -  3.12e-03 .. 1.32e-02
Q4_K_M  decode-nr result_output                                Q6_K       5.69e-08  2.38e-07    -1     1  1.24e-03 .. 2.27e-03  4.94e-03 .. 1.06e-02
Q4_K_M  decode-nr hc_gate (attn mixer: down, f16 up): rows within TOL_OP 510/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr hc_gate (head mixer: down, f16 up): rows within TOL_OP 62/64 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr ple_gate (ple_key, ple_value inside)         Q4_K              -  3.29e-07     1     0                     -  5.75e-03 .. 5.75e-03
Q4_K_M  decode-nr ple_gated_value (ple_key, ple_value inside)  Q4_K              -  2.10e-07     1     0                     -  7.20e-03 .. 7.20e-03
Q4_K_M  decode-nr ple_conv_out (ple_key, ple_value inside)     Q4_K              -  1.81e-07     1     0                     -  8.69e-03 .. 8.69e-03
Q4_K_M  pf64      hc_init, ple_embd (gathers, exact)           Q4_K,Q5_0  0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  pf64      hc_inject (attn mixer)                       Q4_K       2.61e-08  2.22e-07     5     0  7.62e-04 .. 1.21e-03  5.16e-03 .. 1.02e-02
Q4_K_M  pf64      hc_gate (attn mixer: down, f16 up)           Q4_K+F16   5.40e-01  1.00e-04     5     0                     -  6.24e-03 .. 9.41e-03
Q4_K_M  pf64      hc_inject (ffn mixer)                        Q4_K       2.22e-08  1.66e-07     2     0  8.42e-04 .. 1.29e-03  6.34e-03 .. 8.55e-03
Q4_K_M  pf64      hc_gate (ffn mixer: down, f16 up)            Q6_K+F16   2.01e-04  8.84e-08     6     0                     -  7.14e-03 .. 9.52e-03
Q4_K_M  pf64      linear_attn_qkv_mixed                        Q8_0       3.52e-08  1.12e-07     5     0  1.61e-03 .. 2.03e-03  4.68e-03 .. 5.65e-03
Q4_K_M  pf64      z                                            Q4_K       1.57e-07  5.15e-07     2     0  2.22e-03 .. 2.86e-03  6.97e-03 .. 8.91e-03
Q4_K_M  pf64      beta                                         Q4_K       3.84e-08  1.61e-07     6     0  1.65e-03 .. 2.50e-03  7.05e-03 .. 1.06e-02
Q4_K_M  pf64      alpha                                        Q4_K       3.96e-08  1.53e-07     1     0  1.66e-03 .. 2.68e-03  5.90e-03 .. 9.12e-03
Q4_K_M  pf64      linear_attn_out                              Q4_K       2.28e-07  6.31e-07     1     0  4.48e-03 .. 6.11e-03  1.17e-02 .. 1.64e-02
Q4_K_M  pf64      ffn_moe_logits                               F32        3.77e-08  1.61e-07     7     0  2.87e-08 .. 3.77e-08  1.05e-07 .. 1.61e-07
Q4_K_M  pf64      ffn_moe_gate                                 Q4_K       1.42e-07  4.21e-07     1     0  2.56e-03 .. 3.01e-03  8.18e-03 .. 9.02e-03
Q4_K_M  pf64      ffn_moe_up                                   Q4_K       1.37e-07  4.27e-07     3     0  2.45e-03 .. 3.14e-03  7.40e-03 .. 9.99e-03
Q4_K_M  pf64      ffn_moe_down                                 Q6_K       4.77e-08  1.23e-07     0     0  6.50e-03 .. 8.67e-03  1.74e-02 .. 2.23e-02
Q4_K_M  pf64      ffn_up (shared expert)                       Q4_K       1.58e-07  5.58e-07     2     0  2.18e-03 .. 2.86e-03  6.03e-03 .. 9.02e-03
Q4_K_M  pf64      ffn_gate (shared expert)                     Q4_K       1.34e-07  4.18e-07     2     0  2.13e-03 .. 3.21e-03  5.62e-03 .. 1.01e-02
Q4_K_M  pf64      ffn_shexp (shared expert down)               Q6_K       2.38e-08  6.70e-08     6     0  7.12e-03 .. 9.52e-03  1.93e-02 .. 2.74e-02
Q4_K_M  pf64      shared_expert_gate                           F32        2.43e-08  1.03e-07     0     0  1.68e-08 .. 2.43e-08  6.30e-08 .. 1.03e-07
Q4_K_M  pf64      hc_gate (ffn mixer: down, f16 up)            Q4_K+F16   2.97e-01  4.64e-05     5     0                     -  5.46e-03 .. 8.68e-03
Q4_K_M  pf64      ffn_moe_down                                 Q4_K       9.64e-08  2.54e-07     2     0  7.45e-03 .. 1.04e-02  1.97e-02 .. 3.10e-02
Q4_K_M  pf64      ffn_shexp (shared expert down)               Q4_K       1.11e-07  2.63e-07     4     0  6.28e-03 .. 7.72e-03  1.71e-02 .. 2.02e-02
Q4_K_M  pf64      Qcur_full                                    Q4_K       1.33e-07  3.65e-07     3     0  2.40e-03 .. 2.82e-03  7.64e-03 .. 7.78e-03
Q4_K_M  pf64      Kcur (projection)                            Q8_0       2.95e-08  1.03e-07     7     0  1.65e-03 .. 1.66e-03  5.54e-03 .. 5.77e-03
Q4_K_M  pf64      Vcur (projection)                            Q8_0       3.74e-08  1.08e-07     7     0  1.69e-03 .. 2.02e-03  5.71e-03 .. 5.87e-03
Q4_K_M  pf64      indexer_k_raw                                F32        3.56e-08  1.46e-07     3     0  3.56e-08 .. 3.56e-08  1.46e-07 .. 1.46e-07
Q4_K_M  pf64      attn_output                                  Q5_K       2.31e-08  8.12e-08     7     0  1.76e-03 .. 2.21e-03  7.50e-03 .. 7.68e-03
Q4_K_M  pf64      hc_gate (head mixer: down, f16 up)           Q4_K+F16   1.79e-04  9.17e-08    -1     0                     -  9.87e-03 .. 9.87e-03
Q4_K_M  pf64      result_output                                Q6_K       1.99e-08  5.94e-08    -1     0  2.31e-03 .. 2.31e-03  6.90e-03 .. 6.90e-03
Q4_K_M  pf64      hc_gate (attn mixer: down, f16 up): rows within TOL_OP 509/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 511/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      hc_gate (head mixer: down, f16 up): rows within TOL_OP 64/64 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      ple_gate (ple_key, ple_value inside)         Q4_K              -  6.10e-07     1     0                     -  5.75e-03 .. 5.75e-03
Q4_K_M  pf64      ple_gated_value (ple_key, ple_value inside)  Q4_K              -  3.79e-07     1     0                     -  7.20e-03 .. 7.20e-03
Q4_K_M  pf64      ple_conv_out (ple_key, ple_value inside)     Q4_K              -  3.22e-07     1     0                     -  8.69e-03 .. 8.69e-03
Q4_K_M  pf64-nr   hc_init, ple_embd (gathers, exact)           Q4_K,Q5_0  0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  pf64-nr   hc_inject (attn mixer)                       Q4_K       2.42e-08  1.55e-07     2     0  7.62e-04 .. 1.20e-03  5.16e-03 .. 8.06e-03
Q4_K_M  pf64-nr   hc_gate (attn mixer: down, f16 up)           Q4_K+F16   4.34e-02  3.40e-06     5     0                     -  6.24e-03 .. 9.26e-03
Q4_K_M  pf64-nr   hc_inject (ffn mixer)                        Q4_K       2.17e-08  1.42e-07     2     0  8.86e-04 .. 1.08e-03  6.34e-03 .. 8.24e-03
Q4_K_M  pf64-nr   hc_gate (ffn mixer: down, f16 up)            Q6_K+F16   1.99e-03  1.08e-07     0     0                     -  7.32e-03 .. 8.35e-03
Q4_K_M  pf64-nr   linear_attn_qkv_mixed                        Q8_0       4.18e-08  1.22e-07     4     0  1.61e-03 .. 2.03e-03  4.88e-03 .. 5.81e-03
Q4_K_M  pf64-nr   z                                            Q4_K       5.46e-08  1.73e-07     5     0  2.10e-03 .. 2.86e-03  6.62e-03 .. 8.98e-03
Q4_K_M  pf64-nr   beta                                         Q4_K       3.85e-08  1.37e-07     1     0  1.51e-03 .. 2.14e-03  6.33e-03 .. 7.96e-03
Q4_K_M  pf64-nr   alpha                                        Q4_K       4.17e-08  1.61e-07     1     0  1.65e-03 .. 2.31e-03  5.90e-03 .. 7.77e-03
Q4_K_M  pf64-nr   linear_attn_out                              Q4_K       7.14e-08  2.21e-07     2     0  4.21e-03 .. 6.46e-03  1.17e-02 .. 1.73e-02
Q4_K_M  pf64-nr   ffn_moe_logits                               F32        4.58e-08  1.74e-07     6     0  3.07e-08 .. 4.58e-08  1.17e-07 .. 1.74e-07
Q4_K_M  pf64-nr   ffn_moe_gate                                 Q4_K       8.24e-08  2.44e-07     5     0  2.38e-03 .. 3.00e-03  7.04e-03 .. 8.60e-03
Q4_K_M  pf64-nr   ffn_moe_up                                   Q4_K       9.79e-08  2.97e-07     5     0  2.44e-03 .. 3.14e-03  7.37e-03 .. 8.63e-03
Q4_K_M  pf64-nr   ffn_moe_down                                 Q6_K       4.50e-08  1.16e-07     0     0  7.63e-03 .. 8.67e-03  1.81e-02 .. 2.23e-02
Q4_K_M  pf64-nr   ffn_up (shared expert)                       Q4_K       4.79e-08  1.46e-07     7     0  2.20e-03 .. 2.86e-03  6.65e-03 .. 9.02e-03
Q4_K_M  pf64-nr   ffn_gate (shared expert)                     Q4_K       4.41e-08  1.27e-07     6     0  2.35e-03 .. 3.21e-03  6.36e-03 .. 1.01e-02
Q4_K_M  pf64-nr   ffn_shexp (shared expert down)               Q6_K       2.44e-08  6.94e-08     6     0  5.87e-03 .. 9.96e-03  1.67e-02 .. 2.86e-02
Q4_K_M  pf64-nr   shared_expert_gate                           F32        2.96e-08  1.49e-07     5     0  1.53e-08 .. 2.96e-08  6.25e-08 .. 1.49e-07
Q4_K_M  pf64-nr   hc_gate (ffn mixer: down, f16 up)            Q4_K+F16   6.74e-02  5.78e-06     5     0                     -  6.80e-03 .. 8.67e-03
Q4_K_M  pf64-nr   ffn_moe_down                                 Q4_K       7.48e-08  2.09e-07     5     0  7.17e-03 .. 1.04e-02  1.97e-02 .. 3.10e-02
Q4_K_M  pf64-nr   ffn_shexp (shared expert down)               Q4_K       4.34e-08  1.10e-07     5     0  6.27e-03 .. 8.05e-03  1.53e-02 .. 2.02e-02
Q4_K_M  pf64-nr   Qcur_full                                    Q4_K       4.07e-08  1.29e-07     7     0  2.65e-03 .. 3.17e-03  8.43e-03 .. 8.73e-03
Q4_K_M  pf64-nr   Kcur (projection)                            Q8_0       2.84e-08  9.58e-08     3     0  1.47e-03 .. 1.74e-03  4.94e-03 .. 6.01e-03
Q4_K_M  pf64-nr   Vcur (projection)                            Q8_0       2.93e-08  8.81e-08     7     0  1.69e-03 .. 1.93e-03  5.60e-03 .. 5.71e-03
Q4_K_M  pf64-nr   indexer_k_raw                                F32        3.02e-08  1.24e-07     3     0  3.02e-08 .. 3.02e-08  1.24e-07 .. 1.24e-07
Q4_K_M  pf64-nr   attn_output                                  Q5_K       2.31e-08  8.12e-08     7     0  1.76e-03 .. 2.21e-03  7.50e-03 .. 7.68e-03
Q4_K_M  pf64-nr   hc_gate (head mixer: down, f16 up)           Q4_K+F16   1.71e-04  7.89e-08    -1     0                     -  9.43e-03 .. 9.43e-03
Q4_K_M  pf64-nr   result_output                                Q6_K       2.03e-08  6.09e-08    -1     0  2.14e-03 .. 2.14e-03  6.43e-03 .. 6.43e-03
Q4_K_M  pf64-nr   hc_gate (attn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   hc_gate (head mixer: down, f16 up): rows within TOL_OP 64/64 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   ple_gate (ple_key, ple_value inside)         Q4_K              -  3.38e-07     1     0                     -  5.75e-03 .. 5.75e-03
Q4_K_M  pf64-nr   ple_gated_value (ple_key, ple_value inside)  Q4_K              -  2.10e-07     1     0                     -  7.20e-03 .. 7.20e-03
Q4_K_M  pf64-nr   ple_conv_out (ple_key, ple_value inside)     Q4_K              -  1.81e-07     1     0                     -  8.69e-03 .. 8.69e-03
Q4_K_M  ub10      hc_init, ple_embd (gathers, exact)           Q4_K,Q5_0  0.00e+00  0.00e+00    -1     6  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  ub10      hc_inject (attn mixer)                       Q4_K       8.91e-08  1.23e-06     3     6  4.55e-04 .. 1.21e-03  3.57e-03 .. 1.21e-02
Q4_K_M  ub10      hc_gate (attn mixer: down, f16 up)           Q4_K+F16   3.99e-01  7.60e-05     1     5                     -  4.99e-03 .. 1.17e-02
Q4_K_M  ub10      hc_inject (ffn mixer)                        Q4_K       5.99e-08  7.60e-07     1     6  5.09e-04 .. 1.29e-03  3.68e-03 .. 1.52e-02
Q4_K_M  ub10      hc_gate (ffn mixer: down, f16 up)            Q6_K+F16   1.64e-03  1.23e-07     3     3                     -  4.76e-03 .. 1.31e-02
Q4_K_M  ub10      linear_attn_qkv_mixed                        Q8_0       4.18e-08  1.35e-07     4     1  1.35e-03 .. 2.13e-03  4.02e-03 .. 7.64e-03
Q4_K_M  ub10      z                                            Q4_K       1.61e-07  5.96e-07     2     1  1.79e-03 .. 2.99e-03  5.51e-03 .. 1.02e-02
Q4_K_M  ub10      beta                                         Q4_K       7.58e-08  8.23e-07     4     6  8.98e-04 .. 2.14e-03  3.68e-03 .. 1.59e-02
Q4_K_M  ub10      alpha                                        Q4_K       6.55e-08  3.27e-07     1     6  1.02e-03 .. 2.32e-03  4.20e-03 .. 1.22e-02
Q4_K_M  ub10      linear_attn_out                              Q4_K       2.10e-07  6.73e-07     2     4  2.95e-03 .. 6.80e-03  8.35e-03 .. 2.07e-02
Q4_K_M  ub10      ffn_moe_logits                               F32        4.87e-08  1.86e-07     6     2  1.97e-08 .. 4.87e-08  6.76e-08 .. 1.86e-07
Q4_K_M  ub10      ffn_moe_gate                                 Q4_K       1.40e-07  4.58e-07     3     3  1.75e-03 .. 3.17e-03  5.69e-03 .. 1.01e-02
Q4_K_M  ub10      ffn_moe_up                                   Q4_K       1.40e-07  4.55e-07     3     3  1.62e-03 .. 3.15e-03  5.29e-03 .. 9.59e-03
Q4_K_M  ub10      ffn_moe_down                                 Q6_K       8.18e-08  2.33e-07     7     0  4.42e-03 .. 9.47e-03  1.03e-02 .. 2.59e-02
Q4_K_M  ub10      ffn_up (shared expert)                       Q4_K       1.40e-07  5.03e-07     2     1  1.65e-03 .. 2.98e-03  4.61e-03 .. 9.41e-03
Q4_K_M  ub10      ffn_gate (shared expert)                     Q4_K       1.33e-07  4.72e-07     3     3  1.70e-03 .. 3.21e-03  5.21e-03 .. 1.01e-02
Q4_K_M  ub10      ffn_shexp (shared expert down)               Q6_K       5.94e-08  1.79e-07     7     6  4.00e-03 .. 1.02e-02  1.14e-02 .. 2.96e-02
Q4_K_M  ub10      shared_expert_gate                           F32        2.72e-08  2.44e-07     1     2  4.75e-09 .. 2.72e-08  2.36e-08 .. 2.44e-07
Q4_K_M  ub10      hc_gate (ffn mixer: down, f16 up)            Q4_K+F16   6.48e-01  9.44e-05     4     5                     -  4.41e-03 .. 1.23e-02
Q4_K_M  ub10      ffn_moe_down                                 Q4_K       9.92e-08  2.54e-07     2     5  4.26e-03 .. 1.05e-02  1.34e-02 .. 3.21e-02
Q4_K_M  ub10      ffn_shexp (shared expert down)               Q4_K       1.19e-07  3.70e-07     5     1  3.68e-03 .. 9.29e-03  1.14e-02 .. 2.86e-02
Q4_K_M  ub10      Qcur_full                                    Q4_K       1.37e-07  5.08e-07     3     1  1.75e-03 .. 3.01e-03  6.60e-03 .. 9.09e-03
Q4_K_M  ub10      Kcur (projection)                            Q8_0       2.96e-08  1.24e-07     3     2  1.31e-03 .. 1.74e-03  4.67e-03 .. 6.40e-03
Q4_K_M  ub10      Vcur (projection)                            Q8_0       2.91e-08  1.04e-07     7     6  1.34e-03 .. 2.09e-03  3.90e-03 .. 7.50e-03
Q4_K_M  ub10      indexer_k_raw                                F32        4.01e-08  1.74e-07     3     1  2.76e-08 .. 4.01e-08  1.22e-07 .. 1.74e-07
Q4_K_M  ub10      attn_output                                  Q5_K       1.05e-07  3.89e-07     3     6  1.76e-03 .. 2.82e-03  6.35e-03 .. 1.04e-02
Q4_K_M  ub10      hc_gate (head mixer: down, f16 up)           Q4_K+F16   7.01e-01  2.14e-04    -1     5                     -  5.78e-03 .. 9.18e-03
Q4_K_M  ub10      result_output                                Q6_K       3.88e-08  1.59e-07    -1     6  1.46e-03 .. 2.39e-03  5.99e-03 .. 8.76e-03
Q4_K_M  ub10      hc_gate (attn mixer: down, f16 up): rows within TOL_OP 510/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 511/512 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      hc_gate (head mixer: down, f16 up): rows within TOL_OP 61/64 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      ple_gate (ple_key, ple_value inside)         Q4_K              -  1.19e-07     1     0                     -  2.78e-03 .. 2.78e-03
Q4_K_M  ub10      ple_gated_value (ple_key, ple_value inside)  Q4_K              -  2.25e-07     1     0                     -  5.78e-03 .. 5.78e-03
Q4_K_M  ub10      ple_conv_out (ple_key, ple_value inside)     Q4_K              -  3.80e-07     1     0                     -  7.80e-03 .. 7.80e-03
Q8_0    decode    hc_init, ple_embd (gathers, exact)           Q8_0,Q8_0  0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q8_0    decode    hc_inject (attn mixer)                       Q8_0       2.11e-08  7.45e-07     2    14  1.20e-04 .. 1.03e-03  1.67e-03 .. 2.33e-02
Q8_0    decode    hc_gate (attn mixer: down, f16 up)           Q8_0+F16   1.23e-02  1.40e-06     0     0                     -  1.38e-03 .. 1.53e-02
Q8_0    decode    hc_inject (ffn mixer)                        Q8_0       2.95e-08  5.44e-07     2     1  1.36e-04 .. 8.57e-04  1.72e-03 .. 5.05e-02
Q8_0    decode    hc_gate (ffn mixer: down, f16 up)            Q8_0+F16   4.67e-03  3.53e-07     7     0                     -  2.06e-03 .. 2.41e-02
Q8_0    decode    linear_attn_qkv_mixed                        Q8_0       4.39e-08  1.51e-07     1     0  1.02e-03 .. 2.09e-03  3.11e-03 .. 8.35e-03
Q8_0    decode    z                                            Q8_0       3.39e-08  1.25e-07     1     0  9.54e-04 .. 2.07e-03  3.07e-03 .. 9.07e-03
Q8_0    decode    beta                                         Q8_0       3.23e-08  6.22e-07     2    15  5.51e-05 .. 2.11e-03  2.64e-04 .. 4.00e-02
Q8_0    decode    alpha                                        Q8_0       2.06e-08  2.12e-07     2     0  1.02e-04 .. 1.47e-03  9.62e-04 .. 2.67e-02
Q8_0    decode    linear_attn_out                              Q8_0       4.22e-08  1.36e-07     1     0  1.54e-03 .. 3.44e-03  5.06e-03 .. 1.30e-02
Q8_0    decode    ffn_moe_logits                               F32        3.83e-08  2.25e-07     4     0  4.23e-09 .. 3.83e-08  2.96e-08 .. 2.25e-07
Q8_0    decode    ffn_moe_gate                                 Q8_0       3.85e-08  1.34e-07     6     0  1.11e-03 .. 1.98e-03  3.38e-03 .. 7.55e-03
Q8_0    decode    ffn_moe_up                                   Q8_0       4.27e-08  1.53e-07     5    13  1.01e-03 .. 2.04e-03  2.74e-03 .. 8.39e-03
Q8_0    decode    ffn_moe_down                                 Q8_0       5.61e-08  1.60e-07     4     9  1.82e-03 .. 4.50e-03  5.38e-03 .. 1.28e-02
Q8_0    decode    ffn_up (shared expert)                       Q8_0       3.39e-08  1.43e-07     0     2  9.21e-04 .. 2.22e-03  3.14e-03 .. 8.31e-03
Q8_0    decode    ffn_gate (shared expert)                     Q8_0       3.97e-08  1.53e-07     2     5  9.77e-04 .. 1.90e-03  2.86e-03 .. 9.01e-03
Q8_0    decode    ffn_shexp (shared expert down)               Q8_0       5.47e-08  1.54e-07     0    10  1.66e-03 .. 4.70e-03  4.65e-03 .. 1.40e-02
Q8_0    decode    shared_expert_gate                           F32        2.75e-08  3.11e-05     0     0  5.48e-11 .. 2.75e-08  2.02e-10 .. 3.11e-05
Q8_0    decode    Qcur_full                                    Q8_0       3.91e-08  1.13e-07     3     0  1.10e-03 .. 1.88e-03  3.71e-03 .. 7.63e-03
Q8_0    decode    Kcur (projection)                            Q8_0       3.08e-08  1.44e-07     7     6  9.23e-04 .. 1.75e-03  3.56e-03 .. 8.06e-03
Q8_0    decode    Vcur (projection)                            Q8_0       2.78e-08  1.19e-07     7     0  1.02e-03 .. 1.98e-03  3.65e-03 .. 7.57e-03
Q8_0    decode    indexer_k_raw                                F32        3.43e-08  1.49e-07     3     0  9.07e-09 .. 3.43e-08  3.47e-08 .. 1.49e-07
Q8_0    decode    attn_output                                  Q8_0       2.99e-08  1.32e-07     7     2  1.20e-03 .. 1.74e-03  3.87e-03 .. 8.35e-03
Q8_0    decode    hc_gate (head mixer: down, f16 up)           Q8_0+F16   1.85e-02  3.20e-06    -1     0                     -  2.46e-03 .. 9.92e-03
Q8_0    decode    result_output                                Q8_0       3.69e-08  1.27e-07    -1     0  9.30e-04 .. 1.72e-03  3.81e-03 .. 7.42e-03
Q8_0    decode    hc_gate (attn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    hc_gate (head mixer: down, f16 up): rows within TOL_OP 64/64 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    ple_gate (ple_key, ple_value inside)         Q8_0              -  3.20e-07     1     0                     -  3.20e-07 .. 3.20e-07
Q8_0    decode    ple_gated_value (ple_key, ple_value inside)  Q8_0              -  2.18e-07     1     0                     -  2.18e-07 .. 2.18e-07
Q8_0    decode    ple_conv_out (ple_key, ple_value inside)     Q8_0              -  1.64e-07     1     0                     -  1.64e-07 .. 1.64e-07
Q8_0    pf64      hc_init, ple_embd (gathers, exact)           Q8_0,Q8_0  0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q8_0    pf64      hc_inject (attn mixer)                       Q8_0       2.04e-08  1.46e-07     6     0  6.49e-04 .. 1.03e-03  3.92e-03 .. 7.05e-03
Q8_0    pf64      hc_gate (attn mixer: down, f16 up)           Q8_0+F16   1.23e-02  1.40e-06     0     0                     -  4.65e-03 .. 6.73e-03
Q8_0    pf64      hc_inject (ffn mixer)                        Q8_0       2.88e-08  1.83e-07     2     0  6.19e-04 .. 8.57e-04  4.43e-03 .. 6.60e-03
Q8_0    pf64      hc_gate (ffn mixer: down, f16 up)            Q8_0+F16   4.67e-03  3.53e-07     7     0                     -  4.49e-03 .. 6.87e-03
Q8_0    pf64      linear_attn_qkv_mixed                        Q8_0       4.39e-08  1.25e-07     1     0  1.69e-03 .. 1.90e-03  4.73e-03 .. 5.93e-03
Q8_0    pf64      z                                            Q8_0       3.39e-08  1.09e-07     1     0  1.52e-03 .. 2.07e-03  4.88e-03 .. 6.68e-03
Q8_0    pf64      beta                                         Q8_0       2.94e-08  1.16e-07     2     0  1.23e-03 .. 2.11e-03  5.44e-03 .. 8.95e-03
Q8_0    pf64      alpha                                        Q8_0       2.06e-08  8.07e-08     2     0  1.15e-03 .. 1.47e-03  3.94e-03 .. 5.90e-03
Q8_0    pf64      linear_attn_out                              Q8_0       4.22e-08  1.14e-07     1     0  2.48e-03 .. 3.09e-03  6.95e-03 .. 8.95e-03
Q8_0    pf64      ffn_moe_logits                               F32        3.99e-08  1.56e-07     5     0  3.17e-08 .. 3.99e-08  1.17e-07 .. 1.56e-07
Q8_0    pf64      ffn_moe_gate                                 Q8_0       3.85e-08  1.25e-07     6     0  1.64e-03 .. 1.98e-03  4.53e-03 .. 6.33e-03
Q8_0    pf64      ffn_moe_up                                   Q8_0       4.15e-08  1.36e-07     2     0  1.70e-03 .. 2.02e-03  4.90e-03 .. 6.15e-03
Q8_0    pf64      ffn_moe_down                                 Q8_0       5.57e-08  1.56e-07     5     0  3.41e-03 .. 4.06e-03  8.05e-03 .. 1.04e-02
Q8_0    pf64      ffn_up (shared expert)                       Q8_0       3.25e-08  1.08e-07     0     0  1.64e-03 .. 2.09e-03  4.80e-03 .. 7.22e-03
Q8_0    pf64      ffn_gate (shared expert)                     Q8_0       3.62e-08  1.17e-07     2     0  1.61e-03 .. 1.90e-03  4.40e-03 .. 5.94e-03
Q8_0    pf64      ffn_shexp (shared expert down)               Q8_0       4.80e-08  1.35e-07     7     0  3.26e-03 .. 4.70e-03  7.41e-03 .. 1.20e-02
Q8_0    pf64      shared_expert_gate                           F32        2.75e-08  1.23e-07     0     0  1.57e-08 .. 2.75e-08  5.86e-08 .. 1.23e-07
Q8_0    pf64      Qcur_full                                    Q8_0       3.91e-08  1.13e-07     3     0  1.76e-03 .. 1.81e-03  5.19e-03 .. 6.22e-03
Q8_0    pf64      Kcur (projection)                            Q8_0       2.86e-08  9.99e-08     3     0  1.68e-03 .. 1.72e-03  5.28e-03 .. 5.85e-03
Q8_0    pf64      Vcur (projection)                            Q8_0       2.78e-08  9.79e-08     7     0  1.82e-03 .. 1.98e-03  5.53e-03 .. 6.50e-03
Q8_0    pf64      indexer_k_raw                                F32        4.05e-08  1.61e-07     3     0  4.05e-08 .. 4.05e-08  1.61e-07 .. 1.61e-07
Q8_0    pf64      attn_output                                  Q8_0       2.71e-08  1.10e-07     7     0  1.66e-03 .. 1.74e-03  5.67e-03 .. 7.66e-03
Q8_0    pf64      hc_gate (head mixer: down, f16 up)           Q8_0+F16   1.84e-02  3.16e-06    -1     0                     -  5.08e-03 .. 5.08e-03
Q8_0    pf64      result_output                                Q8_0       3.69e-08  1.08e-07    -1     0  1.61e-03 .. 1.61e-03  4.75e-03 .. 4.75e-03
Q8_0    pf64      hc_gate (attn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      hc_gate (head mixer: down, f16 up): rows within TOL_OP 64/64 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      ple_gate (ple_key, ple_value inside)         Q8_0              -  3.20e-07     1     0                     -  3.20e-07 .. 3.20e-07
Q8_0    pf64      ple_gated_value (ple_key, ple_value inside)  Q8_0              -  2.18e-07     1     0                     -  2.18e-07 .. 2.18e-07
Q8_0    pf64      ple_conv_out (ple_key, ple_value inside)     Q8_0              -  1.74e-07     1     0                     -  1.74e-07 .. 1.74e-07
Q8_0    ub10      hc_init, ple_embd (gathers, exact)           Q8_0,Q8_0  0.00e+00  0.00e+00    -1     6  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q8_0    ub10      hc_inject (attn mixer)                       Q8_0       2.09e-08  2.15e-07     2     6  3.09e-04 .. 1.06e-03  2.71e-03 .. 1.05e-02
Q8_0    ub10      hc_gate (attn mixer: down, f16 up)           Q8_0+F16   1.26e-02  1.74e-06     0     2                     -  3.68e-03 .. 9.72e-03
Q8_0    ub10      hc_inject (ffn mixer)                        Q8_0       2.90e-08  3.81e-07     2     4  3.57e-04 .. 8.59e-04  3.29e-03 .. 1.09e-02
Q8_0    ub10      hc_gate (ffn mixer: down, f16 up)            Q8_0+F16   4.95e-03  4.52e-07     7     4                     -  3.68e-03 .. 8.91e-03
Q8_0    ub10      linear_attn_qkv_mixed                        Q8_0       4.39e-08  1.51e-07     1     0  1.40e-03 .. 1.98e-03  4.25e-03 .. 7.35e-03
Q8_0    ub10      z                                            Q8_0       3.51e-08  1.21e-07     1     4  1.33e-03 .. 2.11e-03  4.18e-03 .. 7.06e-03
Q8_0    ub10      beta                                         Q8_0       3.05e-08  3.34e-07     2     6  7.22e-04 .. 2.24e-03  3.80e-03 .. 1.97e-02
Q8_0    ub10      alpha                                        Q8_0       2.08e-08  1.36e-07     1     2  8.26e-04 .. 1.49e-03  3.19e-03 .. 8.58e-03
Q8_0    ub10      linear_attn_out                              Q8_0       4.51e-08  1.16e-07     1     4  1.78e-03 .. 3.44e-03  5.43e-03 .. 1.16e-02
Q8_0    ub10      ffn_moe_logits                               F32        4.30e-08  2.13e-07     5     6  1.66e-08 .. 4.30e-08  7.01e-08 .. 2.13e-07
Q8_0    ub10      ffn_moe_gate                                 Q8_0       4.07e-08  1.42e-07     6     3  1.37e-03 .. 2.05e-03  4.19e-03 .. 6.83e-03
Q8_0    ub10      ffn_moe_up                                   Q8_0       4.37e-08  1.36e-07     2     2  1.31e-03 .. 2.04e-03  4.05e-03 .. 7.33e-03
Q8_0    ub10      ffn_moe_down                                 Q8_0       5.57e-08  1.64e-07     5     4  2.62e-03 .. 4.42e-03  6.36e-03 .. 1.17e-02
Q8_0    ub10      ffn_up (shared expert)                       Q8_0       3.38e-08  1.19e-07     0     4  1.33e-03 .. 2.20e-03  3.54e-03 .. 7.98e-03
Q8_0    ub10      ffn_gate (shared expert)                     Q8_0       3.69e-08  1.21e-07     2     5  1.25e-03 .. 1.92e-03  3.70e-03 .. 7.00e-03
Q8_0    ub10      ffn_shexp (shared expert down)               Q8_0       5.10e-08  1.64e-07     5     3  2.10e-03 .. 4.97e-03  5.99e-03 .. 1.51e-02
Q8_0    ub10      shared_expert_gate                           F32        2.81e-08  1.65e-07     0     1  3.73e-09 .. 2.81e-08  3.57e-08 .. 1.65e-07
Q8_0    ub10      Qcur_full                                    Q8_0       4.13e-08  1.18e-07     3     2  1.44e-03 .. 1.85e-03  4.26e-03 .. 6.75e-03
Q8_0    ub10      Kcur (projection)                            Q8_0       3.01e-08  1.12e-07     7     5  1.32e-03 .. 1.81e-03  4.53e-03 .. 6.94e-03
Q8_0    ub10      Vcur (projection)                            Q8_0       2.87e-08  1.08e-07     3     1  1.32e-03 .. 1.98e-03  3.61e-03 .. 7.32e-03
Q8_0    ub10      indexer_k_raw                                F32        4.35e-08  1.67e-07     3     6  2.47e-08 .. 4.35e-08  1.11e-07 .. 1.67e-07
Q8_0    ub10      attn_output                                  Q8_0       3.14e-08  1.10e-07     7     2  1.24e-03 .. 2.25e-03  4.40e-03 .. 7.66e-03
Q8_0    ub10      hc_gate (head mixer: down, f16 up)           Q8_0+F16   1.93e-02  5.02e-06    -1     0                     -  4.15e-03 .. 7.18e-03
Q8_0    ub10      result_output                                Q8_0       3.85e-08  1.23e-07    -1     2  1.21e-03 .. 1.69e-03  4.62e-03 .. 6.03e-03
Q8_0    ub10      hc_gate (attn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      hc_gate (ffn mixer: down, f16 up): rows within TOL_OP 512/512 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      hc_gate (head mixer: down, f16 up): rows within TOL_OP 64/64 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      ple_gate (ple_key, ple_value inside)         Q8_0              -  1.59e-07     1     0                     -  1.59e-07 .. 1.59e-07
Q8_0    ub10      ple_gated_value (ple_key, ple_value inside)  Q8_0              -  1.38e-07     1     0                     -  1.38e-07 .. 1.38e-07
Q8_0    ub10      ple_conv_out (ple_key, ple_value inside)     Q8_0              -  1.64e-07     1     0                     -  1.64e-07 .. 1.64e-07
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode_nr.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode_nr.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode_nr.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode_nr.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode_nr.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64_nr.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64_nr.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64_nr.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64_nr.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q4_K_M_pf64_nr.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q4_K_M_ub10.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q4_K_M_ub10.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q4_K_M_ub10.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q4_K_M_ub10.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q4_K_M_ub10.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q8_0_decode.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q8_0_decode.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q8_0_decode.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q8_0_decode.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q8_0_decode.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q8_0_pf64.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q8_0_pf64.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q8_0_pf64.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q8_0_pf64.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q8_0_pf64.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok
test_activation_quantization_visible (tests.test_glm5next_quant_parity.Glm_Q8_0_ub10.test_activation_quantization_visible)
Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every ... ok
test_every_type_checked (tests.test_glm5next_quant_parity.Glm_Q8_0_ub10.test_every_type_checked) ... ok
test_gathers_exact (tests.test_glm5next_quant_parity.Glm_Q8_0_ub10.test_gathers_exact) ... ok
test_matmuls (tests.test_glm5next_quant_parity.Glm_Q8_0_ub10.test_matmuls) ... ok
test_weight_types (tests.test_glm5next_quant_parity.Glm_Q8_0_ub10.test_weight_types)
llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-..., ... ok

glm5-next: quantized products from llama.cpp's own inputs (ec7630a640789c393694fb194f1bbbf0369fc62d), worst over layers and graphs. scaled = max|got-want| / max(|W| |x_q|) (asserted < TOL_OP = 1e-05; for a bounded product: the largest share of its bound, asserted <= 1); rel = max|got-want| / max|want|; no-act = the same without the activation quantization (min .. max over graphs):
type    run       product                                      weight       scaled       rel layer graph         no-act scaled            no-act rel
Q4_K_M  decode    inp_embd (gather, exact)                     Q4_K       0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  decode    kda_q_conv (attn_q, conv, silu)              Q4_K       3.54e-02  3.54e-07     9    12                     -  3.30e-03 .. 1.82e-02
Q4_K_M  decode    kda_k_conv (attn_k, conv, silu)              Q8_0       2.05e-02  2.05e-07    10    14                     -  2.26e-03 .. 1.23e-02
Q4_K_M  decode    kda_v_conv (attn_v, conv, silu)              Q8_0       2.26e-02  2.26e-07    10     3                     -  2.31e-03 .. 1.18e-02
Q4_K_M  decode    kda_out (attn_output of the recomputed gate) Q5_K       2.07e-02  8.04e-07     6    15                     -  8.52e-03 .. 3.51e-02
Q4_K_M  decode    ffn_up (dense)                               Q4_K       9.44e-08  2.97e-07     0     0  1.21e-03 .. 2.90e-03  4.31e-03 .. 9.14e-03
Q4_K_M  decode    ffn_gate (dense)                             Q4_K       9.58e-08  3.06e-07     0     0  1.44e-03 .. 2.33e-03  4.50e-03 .. 9.96e-03
Q4_K_M  decode    ffn_out (dense down)                         Q6_K       6.51e-08  3.17e-07     0     8  5.19e-04 .. 9.47e-04  2.15e-03 .. 4.21e-03
Q4_K_M  decode    ffn_moe_logits                               F32        4.31e-08  2.21e-07     7     0  3.96e-09 .. 4.31e-08  1.49e-08 .. 2.21e-07
Q4_K_M  decode    ffn_moe_gate                                 Q4_K       9.91e-08  3.42e-07     5     0  1.34e-03 .. 3.13e-03  4.65e-03 .. 1.23e-02
Q4_K_M  decode    ffn_moe_up                                   Q4_K       1.00e-07  3.16e-07     1     0  1.25e-03 .. 3.00e-03  4.43e-03 .. 1.33e-02
Q4_K_M  decode    ffn_moe_down                                 Q4_K       7.21e-08  3.27e-07     1     0  6.09e-04 .. 1.03e-03  2.28e-03 .. 4.46e-03
Q4_K_M  decode    ffn_up (shared expert)                       Q4_K       1.20e-07  3.91e-07     5     0  1.22e-03 .. 3.29e-03  4.43e-03 .. 1.23e-02
Q4_K_M  decode    ffn_gate (shared expert)                     Q4_K       1.14e-07  3.62e-07     5     0  1.18e-03 .. 3.13e-03  3.94e-03 .. 1.22e-02
Q4_K_M  decode    ffn_shexp (shared expert down)               Q4_K       9.65e-08  3.58e-07     7     0  5.58e-04 .. 1.04e-03  1.96e-03 .. 4.26e-03
Q4_K_M  decode    q_resid (attn_q_a, rms_norm)                 Q8_0       1.87e-02  1.87e-07     3     9                     -  3.79e-03 .. 8.58e-03
Q4_K_M  decode    q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.77e-02  2.77e-07     7    11                     -  2.72e-03 .. 9.17e-03
Q4_K_M  decode    attn_out (attn_output of kqv_out)            Q5_K       8.53e-08  3.32e-07     3     8  1.32e-03 .. 3.51e-03  5.19e-03 .. 1.44e-02
Q4_K_M  decode    ffn_moe_down                                 Q6_K       6.32e-08  2.58e-07    11    14  6.30e-04 .. 1.07e-03  2.27e-03 .. 4.15e-03
Q4_K_M  decode    ffn_shexp (shared expert down)               Q6_K       4.88e-08  2.30e-07     9    16  5.08e-04 .. 1.14e-03  1.83e-03 .. 4.22e-03
Q4_K_M  decode    result_output                                Q6_K       4.15e-08  2.00e-07    -1     5  1.03e-03 .. 2.51e-03  3.85e-03 .. 1.13e-02
Q4_K_M  decode    kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode    q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr inp_embd (gather, exact)                     Q4_K       0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  decode-nr kda_q_conv (attn_q, conv, silu)              Q4_K       5.21e-02  5.21e-07     0     6                     -  3.12e-03 .. 1.72e-02
Q4_K_M  decode-nr kda_k_conv (attn_k, conv, silu)              Q8_0       2.19e-02  2.19e-07     4     1                     -  2.08e-03 .. 1.22e-02
Q4_K_M  decode-nr kda_v_conv (attn_v, conv, silu)              Q8_0       3.17e-02  3.17e-07     6     1                     -  2.11e-03 .. 1.42e-02
Q4_K_M  decode-nr kda_out (attn_output of the recomputed gate) Q5_K       1.94e-02  8.54e-07     5     7                     -  8.67e-03 .. 3.39e-02
Q4_K_M  decode-nr ffn_up (dense)                               Q4_K       6.71e-08  2.74e-07     0     9  1.21e-03 .. 2.90e-03  4.31e-03 .. 9.14e-03
Q4_K_M  decode-nr ffn_gate (dense)                             Q4_K       7.44e-08  3.19e-07     0    15  1.44e-03 .. 2.33e-03  4.50e-03 .. 9.96e-03
Q4_K_M  decode-nr ffn_out (dense down)                         Q6_K       6.51e-08  3.17e-07     0     8  5.19e-04 .. 9.47e-04  2.15e-03 .. 4.21e-03
Q4_K_M  decode-nr ffn_moe_logits                               F32        4.84e-08  2.59e-07     1     0  4.27e-09 .. 4.84e-08  2.64e-08 .. 2.59e-07
Q4_K_M  decode-nr ffn_moe_gate                                 Q4_K       1.06e-07  4.07e-07    11    12  1.25e-03 .. 3.20e-03  4.40e-03 .. 1.23e-02
Q4_K_M  decode-nr ffn_moe_up                                   Q4_K       1.17e-07  3.93e-07    10     0  1.31e-03 .. 3.06e-03  4.46e-03 .. 1.15e-02
Q4_K_M  decode-nr ffn_moe_down                                 Q4_K       1.15e-07  5.46e-07     5     9  6.09e-04 .. 1.14e-03  2.12e-03 .. 5.51e-03
Q4_K_M  decode-nr ffn_up (shared expert)                       Q4_K       1.09e-07  3.65e-07     9     1  1.22e-03 .. 3.02e-03  4.67e-03 .. 1.19e-02
Q4_K_M  decode-nr ffn_gate (shared expert)                     Q4_K       9.63e-08  4.20e-07    10     1  1.21e-03 .. 3.02e-03  4.05e-03 .. 1.30e-02
Q4_K_M  decode-nr ffn_shexp (shared expert down)               Q4_K       1.01e-07  4.53e-07     5     3  5.24e-04 .. 1.20e-03  1.96e-03 .. 4.26e-03
Q4_K_M  decode-nr q_resid (attn_q_a, rms_norm)                 Q8_0       1.87e-02  1.87e-07     3     9                     -  3.79e-03 .. 7.91e-03
Q4_K_M  decode-nr q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.62e-02  2.62e-07     7     2                     -  2.89e-03 .. 1.20e-02
Q4_K_M  decode-nr attn_out (attn_output of kqv_out)            Q5_K       8.08e-08  3.22e-07     3    11  1.16e-03 .. 3.90e-03  4.58e-03 .. 1.34e-02
Q4_K_M  decode-nr ffn_moe_down                                 Q6_K       7.74e-08  3.26e-07     6    14  6.29e-04 .. 1.07e-03  1.76e-03 .. 4.15e-03
Q4_K_M  decode-nr ffn_shexp (shared expert down)               Q6_K       5.63e-08  2.40e-07    10    12  5.64e-04 .. 1.08e-03  1.90e-03 .. 5.21e-03
Q4_K_M  decode-nr result_output                                Q6_K       4.38e-08  1.93e-07    -1     9  1.21e-03 .. 2.71e-03  4.11e-03 .. 1.04e-02
Q4_K_M  decode-nr kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  decode-nr q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      inp_embd (gather, exact)                     Q4_K       0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  pf64      kda_q_conv (attn_q, conv, silu)              Q4_K       3.01e-02  3.01e-07    10     0                     -  5.54e-03 .. 1.11e-02
Q4_K_M  pf64      kda_k_conv (attn_k, conv, silu)              Q8_0       1.59e-02  1.59e-07     8     0                     -  3.83e-03 .. 7.20e-03
Q4_K_M  pf64      kda_v_conv (attn_v, conv, silu)              Q8_0       1.73e-02  1.73e-07     0     0                     -  3.65e-03 .. 6.59e-03
Q4_K_M  pf64      kda_out (attn_output of the recomputed gate) Q5_K       7.15e-03  2.70e-07     6     0                     -  1.47e-02 .. 2.52e-02
Q4_K_M  pf64      ffn_up (dense)                               Q4_K       9.44e-08  2.97e-07     0     0  2.90e-03 .. 2.90e-03  9.12e-03 .. 9.12e-03
Q4_K_M  pf64      ffn_gate (dense)                             Q4_K       9.44e-08  3.03e-07     0     0  2.30e-03 .. 2.30e-03  7.35e-03 .. 7.35e-03
Q4_K_M  pf64      ffn_out (dense down)                         Q6_K       2.13e-08  6.85e-08     0     0  9.09e-04 .. 9.09e-04  2.92e-03 .. 2.92e-03
Q4_K_M  pf64      ffn_moe_logits                               F32        4.28e-08  1.67e-07     8     0  2.81e-08 .. 4.28e-08  9.74e-08 .. 1.67e-07
Q4_K_M  pf64      ffn_moe_gate                                 Q4_K       9.75e-08  3.16e-07     5     0  2.20e-03 .. 2.95e-03  6.74e-03 .. 8.93e-03
Q4_K_M  pf64      ffn_moe_up                                   Q4_K       9.89e-08  3.14e-07     9     0  2.27e-03 .. 2.98e-03  6.78e-03 .. 9.22e-03
Q4_K_M  pf64      ffn_moe_down                                 Q4_K       7.19e-08  2.44e-07     1     0  8.33e-04 .. 1.01e-03  2.44e-03 .. 3.53e-03
Q4_K_M  pf64      ffn_up (shared expert)                       Q4_K       1.19e-07  3.91e-07     5     0  2.21e-03 .. 2.82e-03  5.19e-03 .. 8.70e-03
Q4_K_M  pf64      ffn_gate (shared expert)                     Q4_K       1.14e-07  3.62e-07     5     0  2.30e-03 .. 3.02e-03  6.63e-03 .. 9.75e-03
Q4_K_M  pf64      ffn_shexp (shared expert down)               Q4_K       9.33e-08  3.34e-07     7     0  8.51e-04 .. 9.57e-04  2.74e-03 .. 3.11e-03
Q4_K_M  pf64      q_resid (attn_q_a, rms_norm)                 Q8_0       1.49e-02  1.49e-07    11     0                     -  5.69e-03 .. 5.91e-03
Q4_K_M  pf64      q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.02e-02  2.02e-07     3     0                     -  4.74e-03 .. 5.84e-03
Q4_K_M  pf64      attn_out (attn_output of kqv_out)            Q5_K       2.60e-08  1.07e-07     3     0  1.66e-03 .. 1.87e-03  5.98e-03 .. 7.26e-03
Q4_K_M  pf64      ffn_moe_down                                 Q6_K       4.44e-08  1.53e-07     6     0  8.36e-04 .. 1.07e-03  2.84e-03 .. 3.67e-03
Q4_K_M  pf64      ffn_shexp (shared expert down)               Q6_K       2.27e-08  7.88e-08    11     0  8.25e-04 .. 1.14e-03  2.50e-03 .. 3.97e-03
Q4_K_M  pf64      result_output                                Q6_K       1.89e-08  6.41e-08    -1     0  2.51e-03 .. 2.51e-03  8.50e-03 .. 8.50e-03
Q4_K_M  pf64      kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64      q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   inp_embd (gather, exact)                     Q4_K       0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  pf64-nr   kda_q_conv (attn_q, conv, silu)              Q4_K       1.82e-02  1.82e-07     9     0                     -  5.54e-03 .. 9.21e-03
Q4_K_M  pf64-nr   kda_k_conv (attn_k, conv, silu)              Q8_0       1.63e-02  1.63e-07    10     0                     -  3.79e-03 .. 6.01e-03
Q4_K_M  pf64-nr   kda_v_conv (attn_v, conv, silu)              Q8_0       2.00e-02  2.00e-07     5     0                     -  3.65e-03 .. 6.44e-03
Q4_K_M  pf64-nr   kda_out (attn_output of the recomputed gate) Q5_K       8.58e-03  3.01e-07     2     0                     -  1.53e-02 .. 2.52e-02
Q4_K_M  pf64-nr   ffn_up (dense)                               Q4_K       3.49e-08  1.10e-07     0     0  2.90e-03 .. 2.90e-03  9.12e-03 .. 9.12e-03
Q4_K_M  pf64-nr   ffn_gate (dense)                             Q4_K       4.35e-08  1.40e-07     0     0  2.30e-03 .. 2.30e-03  7.35e-03 .. 7.35e-03
Q4_K_M  pf64-nr   ffn_out (dense down)                         Q6_K       2.13e-08  6.85e-08     0     0  9.09e-04 .. 9.09e-04  2.92e-03 .. 2.92e-03
Q4_K_M  pf64-nr   ffn_moe_logits                               F32        5.12e-08  2.07e-07     7     0  2.82e-08 .. 5.12e-08  1.07e-07 .. 2.07e-07
Q4_K_M  pf64-nr   ffn_moe_gate                                 Q4_K       7.21e-08  2.29e-07     3     0  2.22e-03 .. 2.82e-03  6.50e-03 .. 8.90e-03
Q4_K_M  pf64-nr   ffn_moe_up                                   Q4_K       9.27e-08  2.68e-07     3     0  2.24e-03 .. 2.92e-03  6.29e-03 .. 9.06e-03
Q4_K_M  pf64-nr   ffn_moe_down                                 Q4_K       6.34e-08  2.15e-07     2     0  8.33e-04 .. 1.10e-03  2.83e-03 .. 3.37e-03
Q4_K_M  pf64-nr   ffn_up (shared expert)                       Q4_K       4.54e-08  1.47e-07     2     0  2.11e-03 .. 4.24e-03  5.77e-03 .. 1.36e-02
Q4_K_M  pf64-nr   ffn_gate (shared expert)                     Q4_K       3.94e-08  1.25e-07     9     0  2.36e-03 .. 3.02e-03  6.99e-03 .. 9.75e-03
Q4_K_M  pf64-nr   ffn_shexp (shared expert down)               Q4_K       4.91e-08  1.53e-07     5     0  8.23e-04 .. 9.87e-04  2.74e-03 .. 3.11e-03
Q4_K_M  pf64-nr   q_resid (attn_q_a, rms_norm)                 Q8_0       1.68e-02  1.68e-07     7     0                     -  5.85e-03 .. 5.93e-03
Q4_K_M  pf64-nr   q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.02e-02  2.02e-07     3     0                     -  4.73e-03 .. 5.30e-03
Q4_K_M  pf64-nr   attn_out (attn_output of kqv_out)            Q5_K       3.62e-08  1.49e-07    11     0  1.66e-03 .. 1.93e-03  5.98e-03 .. 7.93e-03
Q4_K_M  pf64-nr   ffn_moe_down                                 Q6_K       4.76e-08  1.64e-07     6     0  9.02e-04 .. 1.07e-03  2.89e-03 .. 3.67e-03
Q4_K_M  pf64-nr   ffn_shexp (shared expert down)               Q6_K       2.00e-08  6.77e-08     3     0  8.32e-04 .. 9.44e-04  2.65e-03 .. 3.22e-03
Q4_K_M  pf64-nr   result_output                                Q6_K       2.10e-08  6.99e-08    -1     0  2.71e-03 .. 2.71e-03  9.00e-03 .. 9.00e-03
Q4_K_M  pf64-nr   kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  pf64-nr   q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      inp_embd (gather, exact)                     Q4_K       0.00e+00  0.00e+00    -1     6  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q4_K_M  ub10      kda_q_conv (attn_q, conv, silu)              Q4_K       3.55e-02  3.55e-07     2     2                     -  3.67e-03 .. 1.21e-02
Q4_K_M  ub10      kda_k_conv (attn_k, conv, silu)              Q8_0       1.85e-02  1.85e-07     4     5                     -  3.14e-03 .. 8.12e-03
Q4_K_M  ub10      kda_v_conv (attn_v, conv, silu)              Q8_0       2.22e-02  2.22e-07     6     3                     -  3.40e-03 .. 9.34e-03
Q4_K_M  ub10      kda_out (attn_output of the recomputed gate) Q5_K       1.73e-02  6.70e-07     9     6                     -  1.07e-02 .. 2.61e-02
Q4_K_M  ub10      ffn_up (dense)                               Q4_K       9.46e-08  3.50e-07     0     0  1.81e-03 .. 2.90e-03  6.47e-03 .. 1.08e-02
Q4_K_M  ub10      ffn_gate (dense)                             Q4_K       9.58e-08  3.48e-07     0     2  1.98e-03 .. 2.33e-03  6.74e-03 .. 8.49e-03
Q4_K_M  ub10      ffn_out (dense down)                         Q6_K       4.15e-08  1.65e-07     0     6  7.60e-04 .. 9.24e-04  2.39e-03 .. 3.39e-03
Q4_K_M  ub10      ffn_moe_logits                               F32        4.11e-08  1.89e-07    11     2  1.45e-08 .. 4.11e-08  6.87e-08 .. 1.89e-07
Q4_K_M  ub10      ffn_moe_gate                                 Q4_K       1.14e-07  4.16e-07     5     2  1.82e-03 .. 2.95e-03  5.46e-03 .. 1.11e-02
Q4_K_M  ub10      ffn_moe_up                                   Q4_K       1.01e-07  3.35e-07     1     0  1.78e-03 .. 2.99e-03  5.50e-03 .. 1.02e-02
Q4_K_M  ub10      ffn_moe_down                                 Q4_K       7.34e-08  2.81e-07     1     0  7.56e-04 .. 1.08e-03  2.34e-03 .. 3.63e-03
Q4_K_M  ub10      ffn_up (shared expert)                       Q4_K       1.16e-07  4.11e-07    10     1  1.80e-03 .. 2.94e-03  4.22e-03 .. 1.06e-02
Q4_K_M  ub10      ffn_gate (shared expert)                     Q4_K       1.32e-07  4.77e-07     5     2  1.70e-03 .. 3.06e-03  5.60e-03 .. 1.06e-02
Q4_K_M  ub10      ffn_shexp (shared expert down)               Q4_K       9.81e-08  3.92e-07     7     0  7.07e-04 .. 9.61e-04  2.30e-03 .. 4.13e-03
Q4_K_M  ub10      q_resid (attn_q_a, rms_norm)                 Q8_0       1.71e-02  1.71e-07    11     2                     -  4.92e-03 .. 6.92e-03
Q4_K_M  ub10      q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.19e-02  2.19e-07    11     4                     -  3.49e-03 .. 8.31e-03
Q4_K_M  ub10      attn_out (attn_output of kqv_out)            Q5_K       5.57e-08  2.22e-07    11     6  1.64e-03 .. 3.01e-03  5.98e-03 .. 9.49e-03
Q4_K_M  ub10      ffn_moe_down                                 Q6_K       6.70e-08  2.80e-07     6     1  7.74e-04 .. 1.12e-03  2.49e-03 .. 4.29e-03
Q4_K_M  ub10      ffn_shexp (shared expert down)               Q6_K       4.59e-08  1.95e-07     3     6  7.31e-04 .. 1.22e-03  2.58e-03 .. 4.36e-03
Q4_K_M  ub10      result_output                                Q6_K       3.60e-08  1.32e-07    -1     6  1.77e-03 .. 2.58e-03  6.49e-03 .. 9.00e-03
Q4_K_M  ub10      kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q4_K_M  ub10      q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    inp_embd (gather, exact)                     Q8_0       0.00e+00  0.00e+00    -1    16  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q8_0    decode    kda_q_conv (attn_q, conv, silu)              Q8_0       2.48e-02  2.48e-07     6    11                     -  2.63e-03 .. 1.22e-02
Q8_0    decode    kda_k_conv (attn_k, conv, silu)              Q8_0       2.31e-02  2.31e-07    10    15                     -  2.28e-03 .. 1.10e-02
Q8_0    decode    kda_v_conv (attn_v, conv, silu)              Q8_0       3.26e-02  3.26e-07     8    14                     -  2.22e-03 .. 1.01e-02
Q8_0    decode    kda_out (attn_output of the recomputed gate) Q8_0       3.58e-03  1.48e-07     9    16                     -  5.15e-03 .. 1.41e-02
Q8_0    decode    ffn_up (dense)                               Q8_0       2.99e-08  1.14e-07     0     0  1.04e-03 .. 1.97e-03  4.29e-03 .. 7.16e-03
Q8_0    decode    ffn_gate (dense)                             Q8_0       2.87e-08  9.86e-08     0     0  1.06e-03 .. 1.89e-03  3.42e-03 .. 7.18e-03
Q8_0    decode    ffn_out (dense down)                         Q8_0       3.64e-08  1.38e-07     0     9  5.84e-04 .. 8.88e-04  1.88e-03 .. 4.00e-03
Q8_0    decode    ffn_moe_logits                               F32        4.25e-08  2.06e-07    10     0  4.89e-09 .. 4.25e-08  2.86e-08 .. 2.06e-07
Q8_0    decode    ffn_moe_gate                                 Q8_0       5.20e-08  1.59e-07     9     0  1.08e-03 .. 2.12e-03  3.59e-03 .. 8.42e-03
Q8_0    decode    ffn_moe_up                                   Q8_0       4.12e-08  1.36e-07     9     3  9.77e-04 .. 2.13e-03  3.39e-03 .. 7.86e-03
Q8_0    decode    ffn_moe_down                                 Q8_0       3.73e-08  1.58e-07     5     8  6.20e-04 .. 1.10e-03  2.15e-03 .. 4.85e-03
Q8_0    decode    ffn_up (shared expert)                       Q8_0       4.07e-08  1.37e-07    11     5  9.38e-04 .. 1.83e-03  3.39e-03 .. 8.45e-03
Q8_0    decode    ffn_gate (shared expert)                     Q8_0       3.77e-08  1.47e-07     5     0  9.78e-04 .. 2.45e-03  3.67e-03 .. 1.14e-02
Q8_0    decode    ffn_shexp (shared expert down)               Q8_0       3.43e-08  1.41e-07     6    16  5.70e-04 .. 1.05e-03  1.97e-03 .. 4.82e-03
Q8_0    decode    q_resid (attn_q_a, rms_norm)                 Q8_0       2.07e-02  2.07e-07     3     9                     -  3.09e-03 .. 8.71e-03
Q8_0    decode    q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       3.00e-02  3.00e-07    11    13                     -  2.84e-03 .. 1.04e-02
Q8_0    decode    attn_out (attn_output of kqv_out)            Q8_0       3.58e-08  1.66e-07     3    14  9.74e-04 .. 2.30e-03  3.29e-03 .. 8.01e-03
Q8_0    decode    result_output                                Q8_0       3.70e-08  1.36e-07    -1    13  8.64e-04 .. 1.56e-03  3.70e-03 .. 7.18e-03
Q8_0    decode    kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    decode    q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      inp_embd (gather, exact)                     Q8_0       0.00e+00  0.00e+00    -1     0  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q8_0    pf64      kda_q_conv (attn_q, conv, silu)              Q8_0       1.95e-02  1.95e-07     4     0                     -  4.00e-03 .. 8.03e-03
Q8_0    pf64      kda_k_conv (attn_k, conv, silu)              Q8_0       1.80e-02  1.80e-07     2     0                     -  4.01e-03 .. 5.92e-03
Q8_0    pf64      kda_v_conv (attn_v, conv, silu)              Q8_0       1.52e-02  1.52e-07     1     0                     -  3.91e-03 .. 6.36e-03
Q8_0    pf64      kda_out (attn_output of the recomputed gate) Q8_0       3.71e-03  1.36e-07     9     0                     -  6.83e-03 .. 1.17e-02
Q8_0    pf64      ffn_up (dense)                               Q8_0       2.99e-08  9.45e-08     0     0  1.97e-03 .. 1.97e-03  6.22e-03 .. 6.22e-03
Q8_0    pf64      ffn_gate (dense)                             Q8_0       2.84e-08  9.06e-08     0     0  1.77e-03 .. 1.77e-03  5.65e-03 .. 5.65e-03
Q8_0    pf64      ffn_out (dense down)                         Q8_0       3.64e-08  1.06e-07     0     0  8.72e-04 .. 8.72e-04  2.55e-03 .. 2.55e-03
Q8_0    pf64      ffn_moe_logits                               F32        4.16e-08  1.47e-07    10     0  2.86e-08 .. 4.16e-08  9.65e-08 .. 1.47e-07
Q8_0    pf64      ffn_moe_gate                                 Q8_0       5.20e-08  1.59e-07     9     0  1.66e-03 .. 2.12e-03  4.70e-03 .. 6.73e-03
Q8_0    pf64      ffn_moe_up                                   Q8_0       3.41e-08  1.07e-07     5     0  1.64e-03 .. 2.05e-03  4.61e-03 .. 6.39e-03
Q8_0    pf64      ffn_moe_down                                 Q8_0       3.32e-08  1.12e-07     5     0  8.83e-04 .. 1.08e-03  2.86e-03 .. 3.73e-03
Q8_0    pf64      ffn_up (shared expert)                       Q8_0       3.83e-08  1.21e-07    11     0  1.53e-03 .. 1.97e-03  4.91e-03 .. 6.46e-03
Q8_0    pf64      ffn_gate (shared expert)                     Q8_0       3.77e-08  1.26e-07     5     0  1.54e-03 .. 2.01e-03  4.75e-03 .. 6.65e-03
Q8_0    pf64      ffn_shexp (shared expert down)               Q8_0       3.23e-08  1.05e-07     8     0  8.31e-04 .. 1.03e-03  2.46e-03 .. 3.47e-03
Q8_0    pf64      q_resid (attn_q_a, rms_norm)                 Q8_0       1.89e-02  1.89e-07     7     0                     -  5.75e-03 .. 6.56e-03
Q8_0    pf64      q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.00e-02  2.00e-07    11     0                     -  4.98e-03 .. 6.11e-03
Q8_0    pf64      attn_out (attn_output of kqv_out)            Q8_0       2.96e-08  1.13e-07     3     0  1.33e-03 .. 1.41e-03  4.22e-03 .. 5.40e-03
Q8_0    pf64      result_output                                Q8_0       2.72e-08  8.66e-08    -1     0  1.52e-03 .. 1.52e-03  4.84e-03 .. 4.84e-03
Q8_0    pf64      kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    pf64      q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      inp_embd (gather, exact)                     Q8_0       0.00e+00  0.00e+00    -1     6  0.00e+00 .. 0.00e+00  0.00e+00 .. 0.00e+00
Q8_0    ub10      kda_q_conv (attn_q, conv, silu)              Q8_0       2.17e-02  2.17e-07     4     2                     -  2.93e-03 .. 1.01e-02
Q8_0    ub10      kda_k_conv (attn_k, conv, silu)              Q8_0       2.00e-02  2.00e-07     2     1                     -  2.76e-03 .. 8.16e-03
Q8_0    ub10      kda_v_conv (attn_v, conv, silu)              Q8_0       1.93e-02  1.93e-07     5     4                     -  3.56e-03 .. 9.55e-03
Q8_0    ub10      kda_out (attn_output of the recomputed gate) Q8_0       4.70e-03  1.60e-07     9     6                     -  6.36e-03 .. 1.22e-02
Q8_0    ub10      ffn_up (dense)                               Q8_0       3.07e-08  1.06e-07     0     4  1.50e-03 .. 1.97e-03  4.73e-03 .. 7.27e-03
Q8_0    ub10      ffn_gate (dense)                             Q8_0       2.89e-08  1.06e-07     0     2  1.50e-03 .. 1.87e-03  4.99e-03 .. 6.42e-03
Q8_0    ub10      ffn_out (dense down)                         Q8_0       3.64e-08  1.32e-07     0     5  7.57e-04 .. 9.28e-04  2.49e-03 .. 3.30e-03
Q8_0    ub10      ffn_moe_logits                               F32        4.25e-08  2.23e-07    10     3  1.13e-08 .. 4.25e-08  5.56e-08 .. 2.23e-07
Q8_0    ub10      ffn_moe_gate                                 Q8_0       5.34e-08  1.59e-07     9     4  1.40e-03 .. 2.15e-03  4.48e-03 .. 7.18e-03
Q8_0    ub10      ffn_moe_up                                   Q8_0       3.55e-08  1.26e-07     5     1  1.38e-03 .. 2.11e-03  3.82e-03 .. 7.25e-03
Q8_0    ub10      ffn_moe_down                                 Q8_0       3.63e-08  1.27e-07     5     5  7.53e-04 .. 1.08e-03  2.45e-03 .. 4.00e-03
Q8_0    ub10      ffn_up (shared expert)                       Q8_0       3.90e-08  1.43e-07    11     2  1.18e-03 .. 2.05e-03  4.22e-03 .. 7.53e-03
Q8_0    ub10      ffn_gate (shared expert)                     Q8_0       3.89e-08  1.53e-07     5     2  1.38e-03 .. 2.07e-03  4.05e-03 .. 7.08e-03
Q8_0    ub10      ffn_shexp (shared expert down)               Q8_0       3.25e-08  1.29e-07     8     0  6.42e-04 .. 1.05e-03  2.25e-03 .. 3.75e-03
Q8_0    ub10      q_resid (attn_q_a, rms_norm)                 Q8_0       1.93e-02  1.93e-07     7     0                     -  4.50e-03 .. 7.03e-03
Q8_0    ub10      q_absorbed (attn_q_b, F32 attn_k_b)          Q8_0       2.19e-02  2.19e-07    11     4                     -  3.23e-03 .. 9.88e-03
Q8_0    ub10      attn_out (attn_output of kqv_out)            Q8_0       3.20e-08  1.13e-07     7     2  1.16e-03 .. 1.73e-03  3.91e-03 .. 6.46e-03
Q8_0    ub10      result_output                                Q8_0       2.76e-08  1.07e-07    -1     4  1.28e-03 .. 1.54e-03  4.63e-03 .. 5.68e-03
Q8_0    ub10      kda_q_conv (attn_q, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      kda_k_conv (attn_k, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      kda_v_conv (attn_v, conv, silu): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      kda_out (attn_output of the recomputed gate): rows within TOL_OP 576/576 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      q_resid (attn_q_a, rms_norm): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
Q8_0    ub10      q_absorbed (attn_q_b, F32 attn_k_b): rows within TOL_OP 192/192 -                 -  0.00e+00    -1    -1                     -                     -
test_block_size_rule (tests.test_quant_ref.ActivationQuantization.test_block_size_rule) ... ok
test_f16_and_f32 (tests.test_quant_ref.ActivationQuantization.test_f16_and_f32) ... ok
test_integer_dot_identity (tests.test_quant_ref.ActivationQuantization.test_integer_dot_identity)
llama.cpp's Q8_0 x Q8_0 dot is sum_b d_w d_a sum(q_w q_a); the oracle's dequantized-row product is the same ... ok
test_q8_0_block_rule (tests.test_quant_ref.ActivationQuantization.test_q8_0_block_rule)
AVX2 quantize_row_q8_0: per 32-block d = f16(amax / 127), q = rint(x * 127 / amax); the element of largest ... ok
test_q8_0_rounds_half_to_even (tests.test_quant_ref.ActivationQuantization.test_q8_0_rounds_half_to_even)
x * id exactly on a half step: the x86 path rounds to even (_MM_ROUND_NEAREST), the scalar reference ... ok
test_q8_K_block_rule (tests.test_quant_ref.ActivationQuantization.test_q8_K_block_rule)
quantize_row_q8_K_ref: per 256-block iscale = -127 / max, q = nearest_int(iscale * x), d = 1 / iscale; in the ... ok
test_zero_block (tests.test_quant_ref.ActivationQuantization.test_zero_block) ... ok
test_f16_weight (tests.test_quant_ref.Wrapper.test_f16_weight) ... ok
test_matmul_one_row_and_rows_and_experts (tests.test_quant_ref.Wrapper.test_matmul_one_row_and_rows_and_experts) ... ok
test_ndarray_never_silently_multiplies_a_qweight (tests.test_quant_ref.Wrapper.test_ndarray_never_silently_multiplies_a_qweight) ... ok
test_rmatmul_quantizes_the_activations (tests.test_quant_ref.Wrapper.test_rmatmul_quantizes_the_activations) ... ok

----------------------------------------------------------------------
```

The tables of the 79e2e74 run against those of ec7630a (header line of each model removed: it names the commit); every differing row is a glm5-next Q4_K_M row (the qwen4exp part, the first 322 lines, is identical):

```
$ diff <(sed 1d tables-ec.txt | grep -v '^glm5-next: quantized') <(sed 1d tables-r1.txt | grep -v '^glm5-next: quantized') | grep -E '^[0-9]'; ... | awk '{print $1, $2}' | sort | uniq -c
324,325c324,325 327c327 331,333c331,333 335,336c335,336 339,343c339,343 351,354c351,354 358,363c358,363 365,370c365,370 432,435c432,435 439c439 441,443c441,443 445,451c445,451 
     44 < Q4_K_M
     44 > Q4_K_M
```

Summary of both tables (`2026-10-10-n10-quant-summary.py`):

```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-summary.py <scratch>/n10/qw-ec.log <scratch>/n10/qw-r1.log
qw-ec.log
 qwen4exp: 239 table rows
  quantized products (scaled check): worst scaled error: 2.39e-07 (Q4_K_M decode-nr linear_attn_out Q4_K L5 g6)
  quantized products (scaled check): worst normwise error: 3.83e-06 (Q4_K_M decode-nr hc_inject (ffn mixer) Q4_K L3 g16)
    without activation quantization: smallest scaled error of a graph: 5.51e-05 (Q8_0 decode beta Q8_0 L2 g15)
    without activation quantization: smallest per-run worst scaled error: 8.57e-04 (Q8_0 decode hc_inject (ffn mixer) Q8_0 L2 g1)
    without activation quantization: smallest normwise error of a graph: 2.64e-04 (Q8_0 decode beta Q8_0 L2 g15)
    without activation quantization: largest normwise error: 5.84e-02 (Q4_K_M decode beta Q4_K L5 g6)
  F32 products: worst scaled error: 4.87e-08 (Q4_K_M ub10 ffn_moe_logits F32 L6 g2)
  F32 products: worst normwise error: 3.11e-05 (Q8_0 decode shared_expert_gate F32 L0 g0)
  bounded products: largest share of the bound: 7.01e-01 (Q4_K_M ub10 hc_gate (head mixer: down, f16 up) Q4_K+F16 L-1 g5)
  bounded products: worst normwise error: 2.37e-04 (Q4_K_M decode-nr hc_gate (head mixer: down, f16 up) Q4_K+F16 L-1 g16)
  PLE module: worst normwise error: 6.10e-07 (Q4_K_M decode ple_gate (ple_key, ple_value inside) Q4_K L1 g0)
  rows within TOL_OP, all runs: hc_gate (attn mixer: down, f16 up): 4084/4096
  rows within TOL_OP, all runs: hc_gate (ffn mixer: down, f16 up): 4093/4096
  rows within TOL_OP, all runs: hc_gate (head mixer: down, f16 up): 507/512
 glm5-next: 162 table rows
  quantized products (scaled check): worst scaled error: 1.32e-07 (Q4_K_M ub10 ffn_gate (shared expert) Q4_K L5 g2)
  quantized products (scaled check): worst normwise error: 5.46e-07 (Q4_K_M decode-nr ffn_moe_down Q4_K L5 g9)
    without activation quantization: smallest scaled error of a graph: 5.08e-04 (Q4_K_M decode ffn_shexp (shared expert down) Q6_K L9 g16)
    without activation quantization: smallest per-run worst scaled error: 8.72e-04 (Q8_0 pf64 ffn_out (dense down) Q8_0 L0 g0)
    without activation quantization: smallest normwise error of a graph: 1.76e-03 (Q4_K_M decode-nr ffn_moe_down Q6_K L6 g14)
    without activation quantization: largest normwise error: 1.44e-02 (Q4_K_M decode attn_out (attn_output of kqv_out) Q5_K L3 g8)
  F32 products: worst scaled error: 5.12e-08 (Q4_K_M pf64-nr ffn_moe_logits F32 L7 g0)
  F32 products: worst normwise error: 2.59e-07 (Q4_K_M decode-nr ffn_moe_logits F32 L1 g0)
  bounded products: largest share of the bound: 5.21e-02 (Q4_K_M decode-nr kda_q_conv (attn_q, conv, silu) Q4_K L0 g6)
  bounded products: worst normwise error: 8.54e-07 (Q4_K_M decode-nr kda_out (attn_output of the recomputed gate) Q5_K L5 g7)
  PLE module: worst normwise error: none
  rows within TOL_OP, all runs: kda_q_conv (attn_q, conv, silu): 4608/4608
  rows within TOL_OP, all runs: kda_k_conv (attn_k, conv, silu): 4608/4608
  rows within TOL_OP, all runs: kda_v_conv (attn_v, conv, silu): 4608/4608
  rows within TOL_OP, all runs: kda_out (attn_output of the recomputed gate): 4608/4608
  rows within TOL_OP, all runs: q_resid (attn_q_a, rms_norm): 1536/1536
  rows within TOL_OP, all runs: q_absorbed (attn_q_b, F32 attn_k_b): 1536/1536
qw-r1.log
 qwen4exp: 239 table rows
  quantized products (scaled check): worst scaled error: 2.39e-07 (Q4_K_M decode-nr linear_attn_out Q4_K L5 g6)
  quantized products (scaled check): worst normwise error: 3.83e-06 (Q4_K_M decode-nr hc_inject (ffn mixer) Q4_K L3 g16)
    without activation quantization: smallest scaled error of a graph: 5.51e-05 (Q8_0 decode beta Q8_0 L2 g15)
    without activation quantization: smallest per-run worst scaled error: 8.57e-04 (Q8_0 decode hc_inject (ffn mixer) Q8_0 L2 g1)
    without activation quantization: smallest normwise error of a graph: 2.64e-04 (Q8_0 decode beta Q8_0 L2 g15)
    without activation quantization: largest normwise error: 5.84e-02 (Q4_K_M decode beta Q4_K L5 g6)
  F32 products: worst scaled error: 4.87e-08 (Q4_K_M ub10 ffn_moe_logits F32 L6 g2)
  F32 products: worst normwise error: 3.11e-05 (Q8_0 decode shared_expert_gate F32 L0 g0)
  bounded products: largest share of the bound: 7.01e-01 (Q4_K_M ub10 hc_gate (head mixer: down, f16 up) Q4_K+F16 L-1 g5)
  bounded products: worst normwise error: 2.37e-04 (Q4_K_M decode-nr hc_gate (head mixer: down, f16 up) Q4_K+F16 L-1 g16)
  PLE module: worst normwise error: 6.10e-07 (Q4_K_M decode ple_gate (ple_key, ple_value inside) Q4_K L1 g0)
  rows within TOL_OP, all runs: hc_gate (attn mixer: down, f16 up): 4084/4096
  rows within TOL_OP, all runs: hc_gate (ffn mixer: down, f16 up): 4093/4096
  rows within TOL_OP, all runs: hc_gate (head mixer: down, f16 up): 507/512
 glm5-next: 162 table rows
  quantized products (scaled check): worst scaled error: 1.32e-07 (Q4_K_M ub10 ffn_gate (shared expert) Q4_K L5 g2)
  quantized products (scaled check): worst normwise error: 5.46e-07 (Q4_K_M decode-nr ffn_moe_down Q4_K L5 g9)
    without activation quantization: smallest scaled error of a graph: 5.08e-04 (Q4_K_M decode ffn_shexp (shared expert down) Q6_K L10 g8)
    without activation quantization: smallest per-run worst scaled error: 8.72e-04 (Q8_0 pf64 ffn_out (dense down) Q8_0 L0 g0)
    without activation quantization: smallest normwise error of a graph: 1.83e-03 (Q4_K_M decode ffn_shexp (shared expert down) Q6_K L10 g8)
    without activation quantization: largest normwise error: 1.45e-02 (Q4_K_M decode ffn_moe_up Q4_K L1 g0)
  F32 products: worst scaled error: 5.12e-08 (Q4_K_M pf64-nr ffn_moe_logits F32 L7 g0)
  F32 products: worst normwise error: 2.72e-07 (Q4_K_M ub10 ffn_moe_logits F32 L8 g3)
  bounded products: largest share of the bound: 5.21e-02 (Q4_K_M decode-nr kda_q_conv (attn_q, conv, silu) Q4_K L0 g6)
  bounded products: worst normwise error: 8.87e-07 (Q4_K_M decode-nr kda_out (attn_output of the recomputed gate) Q5_K L10 g5)
  PLE module: worst normwise error: none
  rows within TOL_OP, all runs: kda_q_conv (attn_q, conv, silu): 4608/4608
  rows within TOL_OP, all runs: kda_k_conv (attn_k, conv, silu): 4608/4608
  rows within TOL_OP, all runs: kda_v_conv (attn_v, conv, silu): 4608/4608
  rows within TOL_OP, all runs: kda_out (attn_output of the recomputed gate): 4608/4608
  rows within TOL_OP, all runs: q_resid (attn_q_a, rms_norm): 1536/1536
  rows within TOL_OP, all runs: q_absorbed (attn_q_b, F32 attn_k_b): 1536/1536
```

Dumps of the two pins compared (`2026-10-10-n10-quant-dumpdiff.py`; for glm5-next Q4_K_M the name list is cut at 400 characters):

```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-dumpdiff.py <scratch>/n10/qw-ec/<dir> <scratch>/n10/qw-r1/<dir>
== qwen4exp Q8_0
.bin files: 13378 and 13353, 13328 in both, 49 of them differ
differing, by tensor name: {'indexer_score-3': 25, 'indexer_k-3': 12, 'indexer_pool_k_new-3': 12}
only in the first: ['indexer_k-3 (reshaped)', 'indexer_q-3 (reshaped)']
only in the second: ['indexer_score-3 (reshaped)']
== qwen4exp Q4_K_M
.bin files: 23010 and 22967, 22924 in both, 91 of them differ
differing, by tensor name: {'indexer_score-3': 43, 'indexer_k-3': 24, 'indexer_pool_k_new-3': 24}
only in the first: ['indexer_k-3 (reshaped)', 'indexer_q-3 (reshaped)']
only in the second: ['indexer_score-3 (reshaped)']
== glm5next Q8_0
.bin files: 11309 and 11378, 11240 in both, 69 of them differ
differing, by tensor name: {'kqv_out-11': 23, 'kqv_out-3': 23, 'kqv_out-7': 23}
only in the first: ['q_absorbed-11 (permuted)', 'q_absorbed-3 (permuted)', 'q_absorbed-7 (permuted)']
only in the second: ['q_absorbed-11 (view)', 'q_absorbed-11 (view) (permuted)', 'q_absorbed-3 (view)', 'q_absorbed-3 (view) (permuted)', 'q_absorbed-7 (view)', 'q_absorbed-7 (view) (permuted)']
== glm5next Q4_K_M
.bin files: 19453 and 19570, 19336 in both, 9555 of them differ
differing, by tensor name: {'attn_norm-10': 38, 'attn_norm-11': 38, 'attn_norm-4': 25, 'attn_norm-5': 27, 'attn_norm-6': 31, 'attn_norm-7': 37, 'attn_norm-8': 38, 'attn_norm-9': 38, 'attn_out-11': 36, 'attn_out-3': 25, 'attn_out-7': 26, 'attn_output-10': 38, 'attn_output-4': 38, 'attn_output-5': 38, 'attn_output-6': 38, 'attn_output-8': 38, 'attn_output-9': 38, 'ffn_gate-10': 38, 'ffn_gate-11': 38
only in the first: ['q_absorbed-11 (permuted)', 'q_absorbed-3 (permuted)', 'q_absorbed-7 (permuted)']
only in the second: ['q_absorbed-11 (view)', 'q_absorbed-11 (view) (permuted)', 'q_absorbed-3 (view)', 'q_absorbed-3 (view) (permuted)', 'q_absorbed-7 (view)', 'q_absorbed-7 (view) (permuted)']
```

Whole model and single layers from the token ids / dumped layer inputs (`2026-10-10-n10-quant-e2e-probe.py`, ec7630a, the pf64 runs of the final driver):

```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-e2e-probe.py . <scratch>/n10/qw-ec <llama-ec7630a>/gguf-py
== Q8_0 pf64 (1) whole oracle from the token ids: l_last rel error per layer 0..7, logits, greedy tokens equal, MoE sets differing (layer, position, llama.cpp, oracle, oracle margin)
  emulated      8.0e-08 4.5e-05 2.3e-04 5.4e-04 3.3e-03 6.9e-03 3.9e-02 3.8e-02 | logits 4.44e-02 | greedy 64/64 | MoE sets differing 1: [(6, 52, [0, 1], [0, 2], '1.3e-03')]
  no act quant  9.6e-02 6.7e-02 5.5e-02 5.7e-02 6.0e-02 5.9e-02 7.1e-02 7.6e-02 | logits 6.54e-02 | greedy 59/64 | MoE sets differing 14: [(0, 47, [1, 2], [2, 4], '4.4e-04'), (1, 9, [1, 7], [4, 7], '1.6e-03'), (3, 9, [1, 3], [1, 7], '1.7e-02')] ...
  f32 weights   9.8e-02 6.9e-02 6.1e-02 6.3e-02 7.0e-02 7.3e-02 8.1e-02 8.2e-02 | logits 8.54e-02 | greedy 60/64 | MoE sets differing 25: [(0, 35, [0, 4], [4, 5], '3.9e-03'), (0, 47, [1, 2], [2, 4], '4.8e-03'), (1, 9, [1, 7], [4, 7], '6.3e-03')] ...
== Q8_0 pf64 (2) one layer at a time from llama.cpp's dumped layer input: l_last rel error per layer
  emulated      L0 8.0e-08 L1 4.5e-05 L2 6.3e-08 L3 3.7e-05 L4 7.5e-08 L5 2.6e-04 L6 2.7e-06 L7 4.1e-05
  no act quant  L0 8.8e-03(MoE replayed at [47]) L1 4.2e-03 L2 5.0e-03 L3 1.0e-03 L4 5.4e-03(MoE replayed at [59]) L5 4.5e-03 L6 4.4e-03 L7 1.0e-03
== Q4_K_M pf64 (1) whole oracle from the token ids: l_last rel error per layer 0..7, logits, greedy tokens equal, MoE sets differing (layer, position, llama.cpp, oracle, oracle margin)
  emulated      8.0e-06 6.5e-06 1.4e-04 7.5e-04 3.5e-03 5.1e-03 3.7e-02 3.8e-02 | logits 3.19e-02 | greedy 63/64 | MoE sets differing 2: [(6, 59, [5, 7], [2, 7], '4.0e-03'), (7, 33, [6, 7], [1, 7], '9.4e-04')]
  no act quant  1.1e-02 3.9e-02 3.8e-02 4.0e-02 5.6e-02 6.2e-02 7.2e-02 8.0e-02 | logits 7.58e-02 | greedy 61/64 | MoE sets differing 14: [(1, 0, [3, 7], [3, 4], '8.8e-03'), (1, 9, [4, 7], [1, 7], '2.1e-03'), (1, 42, [4, 6], [3, 6], '3.9e-03')] ...
  f32 weights   1.2e-01 1.1e-01 1.2e-01 1.3e-01 1.6e-01 1.8e-01 2.0e-01 2.1e-01 | logits 2.22e-01 | greedy 47/64 | MoE sets differing 92: [(0, 25, [6, 7], [5, 7], '1.4e-02'), (0, 38, [3, 4], [4, 7], '4.4e-02'), (0, 41, [3, 6], [1, 3], '5.8e-02')] ...
== Q4_K_M pf64 (2) one layer at a time from llama.cpp's dumped layer input: l_last rel error per layer
  emulated      L0 8.0e-06 L1 6.5e-06 L2 7.5e-06 L3 6.1e-08 L4 1.8e-03 L5 1.3e-03 L6 6.2e-04 L7 7.0e-08
  no act quant  L0 1.1e-02 L1 7.2e-03(MoE replayed at [9, 42]) L2 6.6e-03 L3 1.5e-03 L4 5.4e-03 L5 4.6e-03 L6 6.1e-03(MoE replayed at [59]) L7 1.5e-03
== Q4_K_M pf64 (3) layer 4 from its dumped input, component errors in computation order (rel)
  hc_norm[0]             1.1e-07
  hc_gate[0]             2.9e-04
  hc_mixed[0]            1.7e-05
  linear_attn_qkv_mixed  8.1e-04
  z                      1.3e-03
  final_output           7.7e-04
  linear_attn_out        5.9e-03
  hc_combine             1.6e-03
  hc_mixed[1]            1.9e-03
  ffn_out                1.4e-02
  hc_gate[0] up-projection inputs (f16) that differ: 3 of 512: (token 23, input 5: oracle 0.000843524933, llama.cpp 0.00084400177), (token 47, input 5: oracle -0.000453948975, llama.cpp -0.000454187393), (token 60, input 2: oracle 0.0885009766, llama.cpp 0.0885620117)
```

The glm5-next KDA output gate recomputed two ways (`2026-10-10-n10-quant-gate-probe.py`, ec7630a):

```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-gate-probe.py . <scratch>/n10/qw-ec/glm5next <llama-ec7630a>/gguf-py Q8_0 pf64 4 0
Q8_0 pf64 L4 g0 float64 gate rounded to f32: elements beyond flip_bound(8 spacings) + TOL_OP: 0
Q8_0 pf64 L4 g0 float64 gate rounded to f32: elements beyond flip_bound(512 spacings) + TOL_OP: 0
Q8_0 pf64 L4 g0 float64 gate rounded to f32: elements beyond flip_bound(4096 spacings) + TOL_OP: 0
  rows beyond TOL_OP: [28], max rel 2.32e-04
Q8_0 pf64 L4 g0 f32 gate as ggml: elements beyond flip_bound(8 spacings) + TOL_OP: 0
Q8_0 pf64 L4 g0 f32 gate as ggml: elements beyond flip_bound(512 spacings) + TOL_OP: 0
Q8_0 pf64 L4 g0 f32 gate as ggml: elements beyond flip_bound(4096 spacings) + TOL_OP: 0
  rows beyond TOL_OP: [], max rel 8.63e-08
  float64 gate, row 28: largest input changes (element: change in steps of its block) 128: +0.096, 140: +0.041, 130: +0.025, 141: +0.025, 150: +0.023, 145: +0.016
```

### Mutation check (part B)

`python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant.py <repo> <scratch>/n10/qw-<pin> <llama.cpp> <scratch>/n10/qmut2-<pin>` on the code of `85d3095` (ref/ and tests/ of the working tree, unchanged during the run), once per pin; shown through `grep -E "^(llama|baseline|[QGTD][0-9]+ |    rc=|    (quant_ref|glm5next|qwen4exp)|caught|real)" <log> | cut -c1-150` (the lists of failing tests are cut at 150 columns; the mutant edits are in the script).

```
llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d; dumps <scratch>/n10/qw-ec
baseline (no mutation): rc=0 Ran 99 tests in 51.570s | OK
Q1 no activation quantization at all (dequantized weights only)
    rc=1 Ran 59 tests in 29.890s | FAILED (failures=16, errors=8)
    glm5next: Glm_Q4_K_M_decode.setUpClass, Glm_Q4_K_M_decode_nr.setUpClass, Glm_Q4_K_M_pf64.setUpClass, Glm_Q4_K_M_pf64_nr.setUpClass, Glm_Q4_K_M_ub1
    quant_ref (property): Wrapper.test_f16_weight, Wrapper.test_matmul_one_row_and_rows_and_experts, Wrapper.test_rmatmul_quantizes_the_activations
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode.test_ple_module, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_decode_nr.test_ple_module, Q4_K_M_pf64.
Q2 Q8_0 activations by the scalar reference rule (id = 1/d, round half away from zero)
    rc=1 Ran 99 tests in 53.161s | FAILED (failures=9)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q8_0_decode.test_matmuls, Glm_Q8_0_pf64.test_matmuls, Glm_Q8_0_ub10.test_matmuls
    quant_ref (property): ActivationQuantization.test_q8_0_rounds_half_to_even
    qwen4exp: Q4_K_M_ub10.test_matmuls, Q8_0_decode.test_matmuls, Q8_0_pf64.test_matmuls, Q8_0_ub10.test_matmuls
Q3 Q8_0 block scale kept in f32 (not rounded to f16)
    rc=1 Ran 99 tests in 42.041s | FAILED (failures=17)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    quant_ref (property): ActivationQuantization.test_q8_0_block_rule
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
Q4 k-quant activations quantized like Q8_0 (32-blocks, f16 scale) instead of Q8_K
    rc=1 Ran 99 tests in 42.064s | FAILED (failures=15)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode.test_ple_module, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_decode_nr.test_ple_module, Q4_K_M_pf64.
Q5 F16 weights: activations not rounded to f16
    rc=1 Ran 99 tests in 43.432s | FAILED (failures=9)
    quant_ref (property): ActivationQuantization.test_f16_and_f32
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
Q6 mul_mat_id: expert e reads the next expert's matrix
    rc=1 Ran 99 tests in 50.649s | FAILED (failures=17)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    quant_ref (property): Wrapper.test_matmul_one_row_and_rows_and_experts
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
G1 gguf-py Q4_K dequantize: the minimum added instead of subtracted
    rc=1 Ran 99 tests in 42.932s | FAILED (failures=25)
    glm5next: Glm_Q4_K_M_decode.test_gathers_exact, Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_gathers_exact, Glm_Q4_K_M_decode_nr.test
    qwen4exp: Q4_K_M_decode.test_gathers_exact, Q4_K_M_decode.test_matmuls, Q4_K_M_decode.test_ple_module, Q4_K_M_decode_nr.test_gathers_exact, Q4_K_M
G2 gguf-py Q6_K dequantize: offset 31 instead of 32
    rc=1 Ran 99 tests in 43.330s | FAILED (failures=10)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
G3 gguf-py Q5_0 dequantize: the fifth bit dropped
    rc=1 Ran 99 tests in 44.029s | FAILED (failures=5)
    qwen4exp: Q4_K_M_decode.test_gathers_exact, Q4_K_M_decode_nr.test_gathers_exact, Q4_K_M_pf64.test_gathers_exact, Q4_K_M_pf64_nr.test_gathers_exact
T1 tests: TOL_OP 1e-2 (a tolerance that would hide the activation quantization)
    rc=1 Ran 99 tests in 43.739s | FAILED (failures=16)
    glm5next: Glm_Q4_K_M_decode.test_activation_quantization_visible, Glm_Q4_K_M_decode_nr.test_activation_quantization_visible, Glm_Q4_K_M_pf64.test_
    qwen4exp: Q4_K_M_decode.test_activation_quantization_visible, Q4_K_M_decode_nr.test_activation_quantization_visible, Q4_K_M_pf64.test_activation_q
D1 llama side: one expert output element 2^-8 too large (Q4_K_M pf64, ffn_moe_down L2)
    rc=1 Ran 99 tests in 45.214s | FAILED (failures=1)
    qwen4exp: Q4_K_M_pf64.test_matmuls
D2 llama side: one token routed to other experts than the products used (Q8_0 ub10 g3, L5)
    rc=1 Ran 99 tests in 53.654s | FAILED (failures=1)
    qwen4exp: Q8_0_ub10.test_matmuls
D3 llama side: one embedding element one f32 spacing off (Q4_K_M decode g5, hc_init)
    rc=1 Ran 99 tests in 45.452s | FAILED (failures=1)
    qwen4exp: Q4_K_M_decode.test_gathers_exact
D4 llama side: result_output computed without activation quantization (Q8_0 decode g7)
    rc=1 Ran 99 tests in 40.817s | FAILED (failures=2)
    qwen4exp: Q8_0_decode.test_activation_quantization_visible, Q8_0_decode.test_matmuls
D5 llama side, glm5-next: one DSA attention output element 2^-6 too large (Q4_K_M decode g10, kqv_out L7)
    rc=1 Ran 99 tests in 98.157s | FAILED (failures=1)
    glm5next: Glm_Q4_K_M_decode.test_matmuls
D6 llama side, glm5-next: one KDA output-gate element off by 0.5 (Q8_0 pf64, kda_g2 L4)
    rc=1 Ran 99 tests in 43.423s | FAILED (failures=1)
    glm5next: Glm_Q8_0_pf64.test_matmuls
D7 llama side, glm5-next: one KDA input element 2^-6 too large (Q4_K_M ub10 g2, attn_norm L1)
    rc=1 Ran 99 tests in 40.004s | FAILED (failures=1)
    glm5next: Glm_Q4_K_M_ub10.test_matmuls
D8 llama side, glm5-next: one embedding element one f32 spacing off (Q8_0 decode g3, inp_embd)
    rc=1 Ran 99 tests in 39.585s | FAILED (failures=1)
    glm5next: Glm_Q8_0_decode.test_gathers_exact
caught 18/18
real	15m47.062s
```

```
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6; dumps <scratch>/n10/qw-r1
baseline (no mutation): rc=0 Ran 99 tests in 44.810s | OK
Q1 no activation quantization at all (dequantized weights only)
    rc=1 Ran 59 tests in 18.609s | FAILED (failures=16, errors=8)
    glm5next: Glm_Q4_K_M_decode.setUpClass, Glm_Q4_K_M_decode_nr.setUpClass, Glm_Q4_K_M_pf64.setUpClass, Glm_Q4_K_M_pf64_nr.setUpClass, Glm_Q4_K_M_ub1
    quant_ref (property): Wrapper.test_f16_weight, Wrapper.test_matmul_one_row_and_rows_and_experts, Wrapper.test_rmatmul_quantizes_the_activations
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode.test_ple_module, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_decode_nr.test_ple_module, Q4_K_M_pf64.
Q2 Q8_0 activations by the scalar reference rule (id = 1/d, round half away from zero)
    rc=1 Ran 99 tests in 38.597s | FAILED (failures=9)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q8_0_decode.test_matmuls, Glm_Q8_0_pf64.test_matmuls, Glm_Q8_0_ub10.test_matmuls
    quant_ref (property): ActivationQuantization.test_q8_0_rounds_half_to_even
    qwen4exp: Q4_K_M_ub10.test_matmuls, Q8_0_decode.test_matmuls, Q8_0_pf64.test_matmuls, Q8_0_ub10.test_matmuls
Q3 Q8_0 block scale kept in f32 (not rounded to f16)
    rc=1 Ran 99 tests in 37.488s | FAILED (failures=17)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    quant_ref (property): ActivationQuantization.test_q8_0_block_rule
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
Q4 k-quant activations quantized like Q8_0 (32-blocks, f16 scale) instead of Q8_K
    rc=1 Ran 99 tests in 36.907s | FAILED (failures=15)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode.test_ple_module, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_decode_nr.test_ple_module, Q4_K_M_pf64.
Q5 F16 weights: activations not rounded to f16
    rc=1 Ran 99 tests in 38.311s | FAILED (failures=9)
    quant_ref (property): ActivationQuantization.test_f16_and_f32
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
Q6 mul_mat_id: expert e reads the next expert's matrix
    rc=1 Ran 99 tests in 37.238s | FAILED (failures=17)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    quant_ref (property): Wrapper.test_matmul_one_row_and_rows_and_experts
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
G1 gguf-py Q4_K dequantize: the minimum added instead of subtracted
    rc=1 Ran 99 tests in 36.727s | FAILED (failures=25)
    glm5next: Glm_Q4_K_M_decode.test_gathers_exact, Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_gathers_exact, Glm_Q4_K_M_decode_nr.test
    qwen4exp: Q4_K_M_decode.test_gathers_exact, Q4_K_M_decode.test_matmuls, Q4_K_M_decode.test_ple_module, Q4_K_M_decode_nr.test_gathers_exact, Q4_K_M
G2 gguf-py Q6_K dequantize: offset 31 instead of 32
    rc=1 Ran 99 tests in 37.585s | FAILED (failures=10)
    glm5next: Glm_Q4_K_M_decode.test_matmuls, Glm_Q4_K_M_decode_nr.test_matmuls, Glm_Q4_K_M_pf64.test_matmuls, Glm_Q4_K_M_pf64_nr.test_matmuls, Glm_Q4
    qwen4exp: Q4_K_M_decode.test_matmuls, Q4_K_M_decode_nr.test_matmuls, Q4_K_M_pf64.test_matmuls, Q4_K_M_pf64_nr.test_matmuls, Q4_K_M_ub10.test_matmu
G3 gguf-py Q5_0 dequantize: the fifth bit dropped
    rc=1 Ran 99 tests in 38.441s | FAILED (failures=5)
    qwen4exp: Q4_K_M_decode.test_gathers_exact, Q4_K_M_decode_nr.test_gathers_exact, Q4_K_M_pf64.test_gathers_exact, Q4_K_M_pf64_nr.test_gathers_exact
T1 tests: TOL_OP 1e-2 (a tolerance that would hide the activation quantization)
    rc=1 Ran 99 tests in 39.071s | FAILED (failures=16)
    glm5next: Glm_Q4_K_M_decode.test_activation_quantization_visible, Glm_Q4_K_M_decode_nr.test_activation_quantization_visible, Glm_Q4_K_M_pf64.test_
    qwen4exp: Q4_K_M_decode.test_activation_quantization_visible, Q4_K_M_decode_nr.test_activation_quantization_visible, Q4_K_M_pf64.test_activation_q
D1 llama side: one expert output element 2^-8 too large (Q4_K_M pf64, ffn_moe_down L2)
    rc=1 Ran 99 tests in 39.532s | FAILED (failures=1)
    qwen4exp: Q4_K_M_pf64.test_matmuls
D2 llama side: one token routed to other experts than the products used (Q8_0 ub10 g3, L5)
    rc=1 Ran 99 tests in 40.756s | FAILED (failures=1)
    qwen4exp: Q8_0_ub10.test_matmuls
D3 llama side: one embedding element one f32 spacing off (Q4_K_M decode g5, hc_init)
    rc=1 Ran 99 tests in 40.214s | FAILED (failures=1)
    qwen4exp: Q4_K_M_decode.test_gathers_exact
D4 llama side: result_output computed without activation quantization (Q8_0 decode g7)
    rc=1 Ran 99 tests in 41.906s | FAILED (failures=2)
    qwen4exp: Q8_0_decode.test_activation_quantization_visible, Q8_0_decode.test_matmuls
D5 llama side, glm5-next: one DSA attention output element 2^-6 too large (Q4_K_M decode g10, kqv_out L7)
    rc=1 Ran 99 tests in 42.470s | FAILED (failures=1)
    glm5next: Glm_Q4_K_M_decode.test_matmuls
D6 llama side, glm5-next: one KDA output-gate element off by 0.5 (Q8_0 pf64, kda_g2 L4)
    rc=1 Ran 99 tests in 40.925s | FAILED (failures=1)
    glm5next: Glm_Q8_0_pf64.test_matmuls
D7 llama side, glm5-next: one KDA input element 2^-6 too large (Q4_K_M ub10 g2, attn_norm L1)
    rc=1 Ran 99 tests in 40.588s | FAILED (failures=1)
    glm5next: Glm_Q4_K_M_ub10.test_matmuls
D8 llama side, glm5-next: one embedding element one f32 spacing off (Q8_0 decode g3, inp_embd)
    rc=1 Ran 99 tests in 40.560s | FAILED (failures=1)
    glm5next: Glm_Q8_0_decode.test_gathers_exact
caught 18/18
real	13m5.527s
```

### Final runs on the final code (ref/, tests/, tools/ of `8ca21dc`; every driver, both pins; the unit discovery)

Each driver as `time <driver> <llama.cpp> <scratch>/n10/build-<pin>`, shown through `grep -E "^(Ran|OK|FAILED)|^real" <log>`; the unit
discovery as `time python3 -m unittest discover -s tests` without dump variables (the parity classes skip):

```
ec run_parity.sh: Ran 9 tests in 0.374s OK real 0m3.286s 
ec run_parity_mimo.sh: Ran 5 tests in 0.148s OK real 0m1.321s 
ec run_parity_qwen4exp.sh: Ran 17 tests in 0.342s OK real 0m1.802s 
ec run_parity_qwen4exp_decode.sh: Ran 54 tests in 3.553s OK real 0m7.451s 
ec run_parity_glm5next_decode.sh: Ran 52 tests in 5.545s OK (skipped=1) real 0m9.180s 
ec run_parity_f16fa.sh: Ran 170 tests in 23.679s OK real 0m28.997s 
ec run_parity_quant.sh: Ran 99 tests in 41.134s OK real 1m21.380s 
r1 run_parity.sh: Ran 9 tests in 0.420s OK real 0m2.259s 
r1 run_parity_mimo.sh: Ran 5 tests in 0.071s OK real 0m1.358s 
r1 run_parity_qwen4exp.sh: Ran 17 tests in 0.438s OK real 0m1.887s 
r1 run_parity_qwen4exp_decode.sh: Ran 54 tests in 3.671s OK real 0m7.671s 
r1 run_parity_glm5next_decode.sh: Ran 58 tests in 5.610s OK real 0m10.208s 
r1 run_parity_f16fa.sh: Ran 170 tests in 28.469s OK real 0m35.315s 
r1 run_parity_quant.sh: Ran 99 tests in 41.583s OK real 0m57.536s 
unit discovery: Ran 767 tests in 128.663s OK (skipped=313) real 2m9.605s 
```

After these runs only the module docstring of ref/glm5next.py gained a paragraph on quantized weights (no code); the
glm5-next tests were rerun on it:

```
$ GLM5NEXT_QUANT_DUMP=<scratch>/n10/qw-ec/glm5next ... python3 -m unittest tests.test_glm5next_quant_parity 2>&1 | grep -E "^(Ran|OK|FAILED)"
Ran 40 tests in 24.418s
OK
$ tools/tinygen/run_parity_glm5next_decode.sh <llama-ec7630a> <scratch>/n10/build-ec 2>&1 | grep -E "^(Ran|OK|FAILED)"
Ran 52 tests in 5.630s
OK (skipped=1)
```
