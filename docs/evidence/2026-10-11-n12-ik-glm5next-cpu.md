# N12: the `glm5next` dialect and ik_llama.cpp on CPU (gap 5 of CORRECTNESS.md)

Date 2026-10-11. Branch `claude/n12-ik-glm5next-cpu`, cut from `claude/n10-quant-fa-parity` at `0d4d582`. Roles:
engine-integrator (dump tool, build), parity-qa (oracle, tests, evidence), infra-ops (CI). Machine: the session
sandbox, 4 cores, no GPU. Tiny random-weight models only; nothing below is a performance claim or a claim about the
real UD-Q4_K_XL file beyond its documented header facts.

Pinned sources read: ik_llama.cpp `5194a9e1451b885e24dd5a065f650864e8807c59` (tools/engines/pins.json; "IK" below,
with "H" = src/llama-hparams.cpp, "L" = src/llama.cpp, "B" = src/graphs/build_glm5next.cpp, "C" = common/common.h,
"CC" = common/common.cpp); llama.cpp `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (the R1 pin, "ML"). Every cited line
was printed from those checkouts in this session.

## Findings

1. **ik_llama.cpp builds on CPU at the pin** with the pins.json flags minus the four CUDA ones: AVX2 on, AVX-512 off
   (CONFIRMED, section 1). Standalone build of `llama-server llama-eval-callback llama-cli llama-tokenize`: configure
   2 s, build 422 s on 4 cores; the out-of-tree build of `tools/ik` (ggml, llama, common and our dump tool): 1 s +
   325 s (CONFIRMED, wall times of this sandbox, not benchmarks).
2. **ik loads our tiny GLM under both names.** `glm5next` (`--dialect ik`, and `--arch glm5next`) and `glm5-next`
   (the alias, `src/llama-model-loader.h:149-151`) load and give bit-identical logits (CONFIRMED, test
   `IkAlias`). The loader rejects any unclaimed tensor or wrong shape (`src/llama-model-loader.cpp:1030-1043`,
   `1107-1114`), so a clean load means every tensor name and shape matched. No key ik requires is missing from our
   files; the only key nobody reads is `expert_group_count` (CONFIRMED, section 2). Neither ik nor llama.cpp reads
   `expert_group_count` at their pins; ik ignores `attention.indexer.kpool_select_tail` and `attention.indexer.types`
   (the tail is always appended, B:210-214; every MLA layer has its own indexer, H:2563-2566). What the real UD
   file's header holds beyond `glm5next.expert_group_count` is UNKNOWN here (its value too); `--dialect ik` writes
   that key as 1
   (PROVISIONAL choice: one group, no restriction).
3. **ik computes a different function from llama.cpp's glm5-next on the same weights**, by six documented
   deviations (ref/glm5next_ik.py, each cited there), all reproduced by the ik-semantics oracle on ik's own inputs
   to 3.3e-7 (1.0e-5 for flash attention in single-token graphs) (CONFIRMED, section 3):
   1. SwiGLU clamp **after** the SiLU: `min(silu(g), L) * clamp(u, -L, L)` (ggml/src/ggml.c:17374-17377,
      ggml/src/iqk/iqk_mul_mat.cpp:155-157, 169-170). llama.cpp clamps the gate before it,
      `silu(min(g, L)) * clamp(u, -L, L)` (ML ggml/src/ggml-cpu/ops.cpp:3614-3615), as the DeepSeek-V4.1 reference
      does (third_party/deepseek-v41-flash/inference/model.py:845-848). Which one GLM-5.3-Flash was trained with:
      UNKNOWN (no GLM modeling code is vendored). The two differ only where the gate pre-activation exceeds L, by at
      most `L * sigmoid(-L)`: 4.54e-4 per activation for GLM-5.3-Flash's L = 10 (docs/models/glm-5.3-flash.md:16;
      CONFIRMED arithmetic, section 3), 0.189 for the tiny model's 0.5. On the tiny model this is the dominant
      difference: ik's FFN output is 0.48-0.71 away from llama.cpp's semantics on ik's own inputs.
   2. Indexer key LayerNorm eps = the RMS eps when `attention.layer_norm_epsilon` is absent (H:2461-2468); llama.cpp
      uses 0 then. Measured 1.0e-5 on the indexer key; no effect on the selection seen here.
   3. KDA q/k L2 norm `x / max(|x|, rms_eps)` (ggml.c:19218, src/llama-kda.cpp:225) instead of
      `x / sqrt(|x|^2 + 1e-6)`: 1.25e-7 on random vectors, below every tolerance (the one mutant that survives).
   4. DSA k-pool selection only with `--dsa` (C:454, B:268; default dense MLA). With it, the unused tail slots hold
      cell 0 (L:5934) and the sparse mask unmasks every listed cell (ggml/src/iqk/iqk_cpu_ops.cpp:1106-1108): **position
      0 is added to the cells of every token whose tail has fewer than kpool-1 cells, unless pool 0 is selected or
      the tail itself contains position 0** (90 row-layers of 192 on the 64-token prefill). llama.cpp fills those slots with the sentinel `n_kv`, which
      selects nothing (ML src/llama-memory-hybrid-idx.cpp:814, 949-975). Removing cell 0 from those rows moves ik's
      attention output by 0.38-0.54 (normwise) on the tiny model. On GLM-5.3-Flash (top-k 2048, kpool 4) it matters
      only once a token has more than 512 complete pools visible, i.e. beyond about 2048 tokens (PROVISIONAL, from the
      selection rule; not run).
   5. Cache precision: with `-fa off` (and `-mla` ≠ 1; default 3, C:446) the MLA latent cache takes the **`-ctv`** type
      (L:1496-1497), and `-ctv f32` is refused without flash attention (L:9257-9259); with `-fa on` an f32 K/V cache
      aborts in the iqk kernel (ggml/src/iqk/iqk_flash_attn.cpp:280-298). So **no CPU configuration of ik keeps the MLA
      latent in f32**. The log still prints the `-ctk` type as "c^KV" (L:9808-9811), e.g. `c^KV (f32)` for a cache
      that holds f16. The indexer key | gate cache has its own type `-ictk`, default f16 (C:495); llama.cpp uses the
      `-ctk` type for it. With `-fa off` the attention itself is f32 over the f16 latent.
   6. CPU flash attention (`-fa on`, ik's default, C:445) runs the iqk kernels; over an f16 latent it equals our
      `ref/fattn.py` "fa_chunk" emulation (q in f16, online softmax, f16 V accumulator) to 1.6e-7 at 64 query rows
      and 1.0e-5 on single-token graphs (CONFIRMED on these runs; the kernel itself is not transcribed, so the
      mechanism is PROVISIONAL). llama.cpp switches to its f32 tiled kernel at 64 rows.
   Whole model on the 64-token prefill: ik's logits are 1.26e-4 from the ik-semantics oracle (argmax 64/64) and 0.905
   from the llama.cpp-semantics oracle (argmax 19/64) (CONFIRMED, section 3). Everything else (mHC/Sinkhorn, KDA,
   MLA, indexer pooling and scores, the sigmoid router with selection bias, shared expert, head) agrees within the
   N9/N10 tolerances; the MTP graph was not compared (NOT RUN, see below).
4. **Precision contract.** f32 caches with FA off cannot be run on ik (finding 3.5). The closest, `-fa off -ctk f32
   -ctv f16 -ictk f32`, passes every check; so do `-ictk f16` and the R1 template's `-fa on -ctk f16 -ctv f16`
   (prefill and decode). The chained tolerance after the first DSA layer is N10's 16 * 2^-11; the worst chained
   error is 2.7e-3 in the driver's runs and 3.2e-3 on the server's token sequence (CONFIRMED, sections 3, 4).
5. **ik's llama-server with the template's flags** (section 4):
   * the template argv as is does not start on a CPU build: `error: invalid parameter for argument: -ts`
     (CC:2285-2287: a CPU build has `llama_max_devices()` = 1, L:8924-8925). `-ngl`, `-sm`, `-ts` are parsed but
     hidden from `--help` on a build without GPU offload (CC:2222, 2253, 2277, 3585). Without those three flags the server starts
     (CONFIRMED). Every other template flag is in `--help`: `--ctx-size --parallel --jinja -ncmoe -fa --dsa -ctk -ctv
     -t --threads-batch -b -ub -muge`; `-ictk` and `--slot-save-path` exist too (CONFIRMED, section 4).
   * `llama-server --help` exits with **1** (examples/server/server.cpp:480-482; llama.cpp's exits 0): in
     `scripts/r1/run_r1.py` the `engine_version`/`help` item accepts only exit code 0 (run_r1.py:232, 1082), so it
     will be recorded as failed; the flags check still reads the text (stdout) (PROVISIONAL: code reading, run_r1 not
     run against ik).
   * `/health` 200 `{"status":"ok","slots_idle":1,"slots_processing":0}`; `n_predict` and `ignore_eos` are honoured
     (EOS forced by logit bias: 1 token without `ignore_eos`, 16 with it).
   * `/completion` with `n_probs` answers in the **old** layout: `completion_probabilities[i] = {"content", "probs":
     [{"tok_str", "prob"}]}` (probabilities, no ids; examples/server/server-common.cpp:489-499), no `stop_type`
     (`stopped_eos/_word/_limit` instead, server-context.cpp:2598) and no `tokens` even with `return_tokens`.
     llama.cpp at the R1 pin returns `{"id","token","bytes","logprob","top_logprobs"}` and `stop_type`
     (ML tools/server/server-task.cpp:274-289, 346, 352). `/v1/completions` `logprobs` uses llama.cpp's layout
     (`{"content":[{"id","token","bytes","logprob","top_logprobs"}]}`), which N11's logprob_compare reads.
   * `/tokenize` returns the same ids as llama.cpp on a byte-BPE tiny vocabulary; `with_pieces` is ignored (ids only;
     llama.cpp returns `{"id","piece"}`). On the id-only test vocabulary `/tokenize` answers HTTP 500
     `unordered_map::at` (a property of that synthetic vocabulary with this engine; llama.cpp not tried on it).
   * `/slots` save / erase / restore works with `--slot-save-path` (404 without; the template has none and says
     `supports_slots: false`): identical continuation after restore; the prompt re-processed after a restore was 48 of
     48 tokens for a 48-token prompt and 5 of 2547 for a 2547-token one (which prefix ik can reuse depends on its
     checkpoints: mechanism UNKNOWN, not traced). **Restoring the save of an empty slot aborts the server** (`GGML_ASSERT(kv_self.cells[kv_self.head].pos == batch.pos[0])`, L:11739; process exit -6).
   * The server's greedy continuation of the N9 prompt with the template's flags equals the dump tool's argmax at the
     same flags (16/16), and that run passes the parity test (CONFIRMED, section 4). Without the tie replays of the
     test the oracle's own continuation differs from step 3 on (7 of 16 equal in a superseded version of the probe,
     whose output was not kept: PROVISIONAL): exact zero DSA scores (two indexer heads, ReLU) are frequent on the tiny
     model, and the choice among them is ik's (20 replays per 64-token run, section 3).
6. **R1 tools against ik** (N11's tools from `git archive claude/n11-r1-tools-cpu-server`, not merged here):
   `prompts.py` fits with ik's `/tokenize` (dialect detected, same counts as llama.cpp); `logprob_compare.py` parses
   ik's `/v1/completions` (29 of 40 prefixes, top-1 agreement 1.0 between `-fa off` and the template flags; the 11
   errors are all the known llama-server case of a token ending in an incomplete UTF-8 sequence);
   `slot_persist.py` works but compares by text, because ik returns no `tokens` (CONFIRMED, section 4). N7's
   `tools/srvcheck/hybrid_server_checks.py` reads `completion_probabilities[].top_logprobs` and `tokens`, which ik
   does not return: against ik its `same_tokens` would compare `None == None` (PROVISIONAL: code reading, not run).
7. **CI**: the job `ik-glm5next` of `.github/workflows/parity-r1-pin.yml` runs the driver; locally the whole driver
   from a clean build directory took 353 s (section 6), under the 8-minute budget. Not run on GitHub here (NOT RUN:
   no push from this session).

## Problems for the owners of other files (report only, nothing edited)

* `scripts/r1/configs/ik-llamacpp-server.json`: (a) the `_label`'s UNKNOWNs are now answered on CPU (finding 5);
  (b) `supports_slots: false` is accurate for its argv (no `--slot-save-path`), but the API works when the path is
  given; restoring a save of an empty slot crashes ik; (c) no `-ictk`: the indexer cache is f16 whatever `KV_TYPE` is
  (q8_0 KV would leave it f16, unlike llama.cpp); (d) `help_argv` exits 1 on ik (finding 5); (e) on a CPU build the argv
  does not start (`-ts`), which only matters for CPU runs of this template.
* `scripts/engines/build_engine.py` was not run here (CUDA build); nothing to report on it from this CPU work.
* N11's `tools/bench/prompts.py --verify` exits 1 on target 1024 (1022 tokens marked exact) against ik **and** against
  llama.cpp at the R1 pin on the same tiny text vocabulary: a fitting quirk of the tool, not of ik (section 4).
* N7's `hybrid_server_checks.py` would pass vacuously against ik (finding 6).

## 1. Fetch and build

Shallow fetch by URL at the pin, no remote added (scratch `ik-src`), CPU build with the pins.json flags without
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES -DCMAKE_CUDA_COMPILER -DCUDAToolkit_ROOT`:
```
$ git init -q . ; git fetch -q --depth 1 https://github.com/ikawrakow/ik_llama.cpp 5194a9e1451b885e24dd5a065f650864e8807c59 && git checkout -q FETCH_HEAD
$ cmake -S ik-src -B ik-build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
$ cmake --build ik-build -j4 --target llama-server llama-eval-callback llama-cli llama-tokenize
$ git -C ik-src rev-parse HEAD
5194a9e1451b885e24dd5a065f650864e8807c59
$ git -C ik-src log -1 --format="%H %ci %s"
5194a9e1451b885e24dd5a065f650864e8807c59 2026-10-09 15:55:35 +0200 Fix convert_imatrix_gguf_to_dat for multidimensional imatrix entries (#2626)
$ grep -E "^(CMAKE_BUILD_TYPE|GGML_NATIVE|GGML_AVX|GGML_AVX2|GGML_AVX512|GGML_AVX512_VBMI|GGML_AVX512_VNNI|GGML_AVX512_BF16|GGML_FMA|GGML_F16C|GGML_CUDA|LLAMA_CURL):" ik-build/CMakeCache.txt
CMAKE_BUILD_TYPE:STRING=Release
GGML_AVX:BOOL=ON
GGML_AVX2:BOOL=ON
GGML_AVX512:BOOL=OFF
GGML_AVX512_BF16:BOOL=OFF
GGML_AVX512_VBMI:BOOL=OFF
GGML_AVX512_VNNI:BOOL=OFF
GGML_CUDA:BOOL=OFF
GGML_F16C:BOOL=ON
GGML_FMA:BOOL=ON
GGML_NATIVE:BOOL=OFF
LLAMA_CURL:BOOL=OFF
$ grep -h "C_FLAGS\|CXX_FLAGS" ik-build/ggml/src/CMakeFiles/ggml.dir/flags.make | tr " " "\n" | grep -E "^-m|march" | sort | uniq -c
      2 -mavx
      2 -mavx2
      2 -mf16c
      2 -mfma
$ cat build.times; tail -n 1 build.log   # configure and build wall seconds of the commands above
configure_s=2 build_s=422
build_rc=0
$ ik-build/bin/llama-server --version
version: 1 (5194a9e)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 for x86_64-linux-gnu
$ ls ik-build/bin
ik-dump-tensors
llama-cli
llama-eval-callback
llama-server
llama-tokenize
```

## 2. Loader contract

`docs/evidence/mutation-scripts/2026-10-11-n12-loader-contract.py <ik> <llama.cpp @79e2e74> <gguf-py> ik.gguf
llama.gguf ikmtp.gguf` (ik.gguf: `--layers 12 --dialect ik`; llama.gguf: `--layers 12`; ikmtp.gguf: `--dialect ik
--mtp`):
```
key (%s = arch prefix)          ik @5194a9e (line)          llama.cpp glm5-next.cpp @79e2e74 (line; arch-specific reads only: the generic ones of src/llama-model.cpp are not tabulated, '-')
  %s.block_count                               required  H:150      -
  general.name                                 optional  H:153      -
  %s.vocab_size                                optional  H:156      -
  %s.context_length                            required  H:163      -
  %s.embedding_length                          required  H:164      -
  %s.embedding_length_out                      optional  H:165      -
  %s.expert_count                              optional  H:169      -
  %s.expert_used_count                         optional  H:170      -
  %s.feed_forward_length                       optional  H:187      -
  %s.attention.head_count                      optional  H:188      -
  %s.attention.head_count_kv                   optional  H:193      -
  %s.rope.scaling.finetuned                    optional  H:196      -
  %s.rope.scaling.original_context_length      optional  H:200      -
  %s.rope.freq_base                            optional  H:204      -
  %s.rope.scaling.type                         optional  H:207      -
  %s.rope.scaling.factor                       optional  H:213      -
  %s.rope.scale_linear                         optional  H:215      -
  %s.rope.scaling.attn_factor                  optional  H:223      -
  %s.attention.key_length                      optional  H:231      -
  %s.attention.value_length                    optional  H:234      -
  %s.rope.dimension_count                      optional  H:239      -
  %s.attention.layer_norm_rms_epsilon          required  H:2460     required :8
  %s.attention.layer_norm_epsilon              optional  H:2465     optional :9
  %s.nextn_predict_layers                      optional  H:2472     -
  %s.attention.q_lora_rank                     required  H:2484     required :12
  %s.attention.kv_lora_rank                    required  H:2485     required :13
  %s.attention.key_length_mla                  required  H:2486     required :10
  %s.attention.value_length_mla                required  H:2487     required :11
  %s.expert_shared_count                       optional  H:2492     required :26
  %s.expert_feed_forward_length                required  H:2493     required :25
  %s.expert_shared_feed_forward_length         optional  H:2494     -
  %s.leading_dense_block_count                 optional  H:2498     optional :27
  %s.expert_weights_scale                      optional  H:2499     optional :28
  %s.expert_weights_norm                       optional  H:2500     optional :29
  %s.expert_gating_func                        optional  H:2501     optional :30
  %s.swiglu_clamp_exp                          optional  H:2506     optional :34
  %s.swiglu_clamp_shexp                        optional  H:2507     optional :35
  %s.attention.indexer.head_count              optional  H:2513     required :40
  %s.attention.indexer.key_length              optional  H:2514     required :41
  %s.attention.indexer.top_k                   optional  H:2515     required :42
  %s.attention.indexer.kpool                   optional  H:2516     required :43
  %s.hyper_connection.count                    required  H:2527     required :50
  %s.hyper_connection.sinkhorn_iterations      required  H:2528     required :51
  %s.hyper_connection.epsilon                  optional  H:2529     required :52
  %s.ssm.conv_kernel                           required  H:2540     required :14
  %s.kda.head_dim                              optional  H:2541     required :15
  %s.ssm.state_size                            fallback  H:2544     -
  %s.kda.gate_lower_bound                      optional  H:2546     optional :16
  %s.ssm.group_count                           optional  H:2555     -
  %s.attention.indexer.kpool_select_tail       -                    optional :44   (llama.cpp only)
  %s.attention.indexer.types                   -                    optional :47   (llama.cpp only)

ik.gguf: general.architecture = glm5next, 35 glm5next.* keys, 370 tensors
  keys the file has that neither table reads: ['glm5next.expert_group_count']
  ik-required keys missing: []

llama.gguf: general.architecture = glm5-next, 34 glm5-next.* keys, 370 tensors
  keys the file has that neither table reads: []
  ik-required keys missing: []

ikmtp.gguf: general.architecture = glm5next, 36 glm5next.* keys, 398 tensors
  keys the file has that neither table reads: ['glm5next.expert_group_count']
  ik-required keys missing: []
```
Layer pattern, as ik derives it: `attention.head_count_kv` per layer, 0 = KDA, otherwise MLA with its own DSA indexer
(H:2559-2566 via `recurrent_layer_arr`, `indexer_is_full`); MoE from `leading_dense_block_count`; mHC on every trunk
block, not on the MTP block (B:570-651); the MTP block is layer `n_layer - nextn_predict_layers`, loaded only when MTP
is requested, otherwise its 28 tensors are skipped and the trunk logits equal the model without MTP bit for bit
(CONFIRMED below). Tensor shapes are those of `create_glm5next_tensors` (src/llama-load-tensors.cpp:4702-4835); ik's
DSA uses wk_b / wv_b (absorbed MLA), and `llm_prepare_mla` additionally computes a Q8_0 `attn_kv_b` that the
glm5next graph does not use (B has no reference to it). `docs/evidence/mutation-scripts/2026-10-11-n12-ik-mtp-load.sh`:
```
wrote <scratch>/n12/mtpw/ikmtp.gguf
3419aea3efffe07e38f9bb56808d0903eb3ca79eae5f435df6738797a59b7902
flags '-fa off -ctk f32 -ctv f16 -ictk f32 --dsa ': exit 0
llm_load_print_meta: nextn_predict_layers   = 1
done_getting_tensors: skipped 28 unused tensors (0.3 MiB) -- e.g. a NextN/MTP head when MTP is not requested; per-tensor detail at debug log level
============ llm_prepare_mla: need to compute 3 wkv_b tensors
Computed blk.3.attn_kv_b.weight as 16 x 128 of type q8_0 and stored in buffer CPU
Computed blk.7.attn_kv_b.weight as 16 x 128 of type q8_0 and stored in buffer CPU
Computed blk.11.attn_kv_b.weight as 16 x 128 of type q8_0 and stored in buffer CPU
ik-dump: 1 calls, 1 graphs, 64 tokens
  logits vs the model without MTP (p64 run): max |diff| = 0.0
flags '-fa off -ctk f32 -ctv f16 -ictk f32 --dsa -mtp': exit 1
legacy speculative option '-mtp' is disabled; use --spec-type mtp:n_max=1,p_min=0.0
  no logits (the run did not get that far)
```

The two rejected f32 configurations (finding 3.5), literal (the iqk type list is cut to the lines the grep kept;
the full list also has `q8_KV`):
```
$ DUMP_DIR=x DUMP_TOKENS=5,9,11,20 ik-dump-tensors -m ik.gguf -c 512 -b 512 -ub 512 -t 1 -fa off -ctk f32 -ctv f32 --dsa
rc=1
llama_init_from_model: V cache quantization requires flash_attn
llama_init_from_gpt_params: error: failed to create context with model '/tmp/claude-0/-home-user-408032gb/83634621-ad4f-
ik-dump: failed to load the model
$ DUMP_DIR=y DUMP_TOKENS=5,9,11,20 ik-dump-tensors -m ik.gguf -c 512 -b 512 -ub 512 -t 1 -fa on -ctk f32 -ctv f32 --dsa
/bin/bash: line 21: 12016 Aborted                 DUMP_DIR=$N/rej/y DUMP_TOKENS=5,9,11,20 $D -m $M -c 512 -b 512 -ub 512 -t 1 -fa on -ctk f32 -ctv f32 --dsa > rej2.log 2>&1
rc=134
============ llm_prepare_mla: need to compute 3 wkv_b tensors
==================== K cache f32 coupled with V cache f32 is not a supported combination on the CPU backend.
Supported types are:
    iq4_nl
    q4_0
    q4_1
    q6_0
    q8_KV
    q8_0
    f16
<scratch>/n12/ik-src/ggml/src/iqk/iqk_flash_attn.cpp:298: Fatal error
$ ik-dump-tensors ... -fa off -ctk f32 -ctv f16 -ictk f32 --dsa   # the configuration the tests use: KV log line
rc=0
llama_init_from_model: KV self size  =    0.05 MiB, c^KV (f32):    0.05 MiB, kv^T (f16):    0.00 MiB
$ cat <work>/n12/p64/g000/kv_cache-3__0.json   # the latent cache of that configuration, as dumped
{"type": "f32", "ggml_type": "f16", "ne": [16, 64, 1, 1]}
$ cat <work>/n12/p64/g000/dsa_cached_all-3__0.json <work>/n12/ictk16/g000/dsa_cached_all-3__0.json
{"type": "f32", "ggml_type": "f32", "ne": [32, 64, 1, 1]}
{"type": "f32", "ggml_type": "f16", "ne": [32, 64, 1, 1]}
```

## 3. Does ik compute the same model?

Driver: `LLAMA_CPP_GGUF_PY=<llama.cpp @79e2e74>/gguf-py tools/tinygen/run_parity_ik_glm5next.sh <ik @5194a9e>
<build dir>`. Model sha256 (`--layers 12 --dialect ik`): `6050ee362d635168a153380bd9b3b703223e444f19e3c7e12e32b1ecef64a6f0` (the default `--layers 12` file, the N9/G1 model under `glm5-next`: `26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5`, unchanged). Tokens: the 64 ids of N9 (seed 9). Literal output
(the per-test `ok` lines are cut to the summary; the table is the test's own):
```
$ LLAMA_CPP_GGUF_PY=<llama.cpp @79e2e74>/gguf-py IK_GLM5NEXT_WORK=<scratch>/n12/work-ci tools/tinygen/run_parity_ik_glm5next.sh <scratch>/n12/ik-src <scratch>/n12/build-ci
GGML_AVX2:BOOL=ON GGML_AVX512:BOOL=OFF GGML_CUDA:BOOL=OFF GGML_NATIVE:BOOL=OFF 
wrote <scratch>/n12/work-ci/ik.gguf
wrote <scratch>/n12/work-ci/llama.gguf
p64: ik-dump: 1 calls, 1 graphs, 64 tokens [-fa off -ctk f32 -ctv f16 -ictk f32 --dsa]
ub16: ik-dump: 1 calls, 4 graphs, 64 tokens [-fa off -ctk f32 -ctv f16 -ictk f32 --dsa]
decode: ik-dump: 17 calls, 17 graphs, 64 tokens [-fa off -ctk f32 -ctv f16 -ictk f32 --dsa]
dense: ik-dump: 1 calls, 1 graphs, 64 tokens [-fa off -ctk f32 -ctv f16 -ictk f32]
ictk16: ik-dump: 1 calls, 1 graphs, 64 tokens [-fa off -ctk f32 -ctv f16 -ictk f16 --dsa]
fa16: ik-dump: 1 calls, 1 graphs, 64 tokens [-fa on -ctk f16 -ctv f16 --dsa]
fa16dec: ik-dump: 17 calls, 17 graphs, 64 tokens [-fa on -ctk f16 -ctv f16 --dsa]
alias: ik-dump: 1 calls, 1 graphs, 64 tokens [-fa off -ctk f32 -ctv f16 -ictk f32 --dsa]

worst normwise relative error per run and check (layer, graph):
  decode   hc_attn_pre                                  2.678e-04  (L11, g11)
  decode   KDA out                                      6.730e-04  (L10, g11)
  decode   mHC attn post                                3.117e-04  (L11, g11)
  decode   ffn_out                                      4.644e-04  (L11, g11)
  decode   l_out                                        2.975e-04  (L11, g11)
  decode   DSA kqv_out                                  2.748e-04  (L11, g12)
  decode   logits                                       1.279e-04
  decode   greedy argmax equal on 64/64 rows
  decode   Qcur (local)                                 2.267e-07  (L11, g1)
  decode   kqv_out (local)                              2.301e-07  (L3, g7)
  decode   indexer key (local, ik eps)                  1.763e-07  (L7, g3)
  decode   pooled keys (local)                          6.836e-08  (L3, g0)
  decode   indexer scores (local)                       2.280e-07  (L3, g15)
  decode   ffn_out (local)                              3.106e-07  (L2, g1)
  decode   DSA rows attending cell 0 by tail padding / tie replays: 83 / 20
  dense    hc_attn_pre                                  9.362e-05  (L11, g0)
  dense    KDA out                                      3.720e-04  (L8, g0)
  dense    mHC attn post                                1.650e-04  (L10, g0)
  dense    ffn_out                                      1.991e-04  (L10, g0)
  dense    l_out                                        1.615e-04  (L9, g0)
  dense    DSA kqv_out                                  1.892e-04  (L11, g0)
  dense    logits                                       1.076e-04
  dense    greedy argmax equal on 64/64 rows
  dense    Qcur (local)                                 1.853e-07  (L3, g0)
  dense    kqv_out (local)                              1.697e-07  (L7, g0)
  dense    ffn_out (local)                              2.153e-07  (L9, g0)
  fa16     hc_attn_pre                                  5.832e-04  (L10, g0)
  fa16     KDA out                                      2.453e-03  (L10, g0)
  fa16     mHC attn post                                8.782e-04  (L9, g0)
  fa16     ffn_out                                      1.346e-03  (L11, g0)
  fa16     l_out                                        9.028e-04  (L9, g0)
  fa16     DSA kqv_out                                  8.127e-04  (L11, g0)
  fa16     logits                                       5.295e-04
  fa16     greedy argmax equal on 64/64 rows
  fa16     Qcur (local)                                 1.853e-07  (L3, g0)
  fa16     kqv_out (local)                              1.603e-07  (L7, g0)
  fa16     indexer key (local, ik eps)                  1.657e-07  (L11, g0)
  fa16     pooled keys (local)                          7.410e-08  (L7, g0)
  fa16     indexer scores (local)                       8.753e-08  (L7, g0)
  fa16     ffn_out (local)                              1.887e-07  (L7, g0)
  fa16     DSA rows attending cell 0 by tail padding / tie replays: 83 / 20
  fa16dec  hc_attn_pre                                  1.131e-03  (L8, g13)
  fa16dec  KDA out                                      2.657e-03  (L9, g5)
  fa16dec  mHC attn post                                8.782e-04  (L9, g0)
  fa16dec  ffn_out                                      1.346e-03  (L11, g0)
  fa16dec  l_out                                        9.104e-04  (L9, g0)
  fa16dec  DSA kqv_out                                  1.154e-03  (L11, g8)
  fa16dec  logits                                       5.295e-04
  fa16dec  greedy argmax equal on 64/64 rows
  fa16dec  Qcur (local)                                 3.246e-07  (L11, g13)
  fa16dec  kqv_out (local)                              1.001e-05  (L11, g12)
  fa16dec  indexer key (local, ik eps)                  1.674e-07  (L7, g2)
  fa16dec  pooled keys (local)                          7.410e-08  (L7, g0)
  fa16dec  indexer scores (local)                       2.244e-07  (L3, g10)
  fa16dec  ffn_out (local)                              3.106e-07  (L2, g1)
  fa16dec  DSA rows attending cell 0 by tail padding / tie replays: 83 / 20
  ictk16   hc_attn_pre                                  9.834e-05  (L10, g0)
  ictk16   KDA out                                      6.097e-04  (L8, g0)
  ictk16   mHC attn post                                1.705e-04  (L11, g0)
  ictk16   ffn_out                                      2.779e-04  (L11, g0)
  ictk16   l_out                                        1.606e-04  (L11, g0)
  ictk16   DSA kqv_out                                  1.892e-04  (L11, g0)
  ictk16   logits                                       1.261e-04
  ictk16   greedy argmax equal on 64/64 rows
  ictk16   Qcur (local)                                 1.853e-07  (L3, g0)
  ictk16   kqv_out (local)                              1.604e-07  (L7, g0)
  ictk16   indexer key (local, ik eps)                  1.651e-07  (L7, g0)
  ictk16   pooled keys (local)                          6.700e-08  (L7, g0)
  ictk16   indexer scores (local)                       8.459e-08  (L7, g0)
  ictk16   ffn_out (local)                              2.087e-07  (L7, g0)
  ictk16   DSA rows attending cell 0 by tail padding / tie replays: 83 / 20
  p64      hc_attn_pre                                  9.834e-05  (L10, g0)
  p64      KDA out                                      6.097e-04  (L8, g0)
  p64      mHC attn post                                1.705e-04  (L11, g0)
  p64      ffn_out                                      2.779e-04  (L11, g0)
  p64      l_out                                        1.606e-04  (L11, g0)
  p64      DSA kqv_out                                  1.892e-04  (L11, g0)
  p64      logits                                       1.261e-04
  p64      greedy argmax equal on 64/64 rows
  p64      Qcur (local)                                 1.853e-07  (L3, g0)
  p64      kqv_out (local)                              1.604e-07  (L7, g0)
  p64      indexer key (local, ik eps)                  1.651e-07  (L7, g0)
  p64      pooled keys (local)                          6.836e-08  (L3, g0)
  p64      indexer scores (local)                       1.092e-07  (L3, g0)
  p64      ffn_out (local)                              2.087e-07  (L7, g0)
  p64      DSA rows attending cell 0 by tail padding / tie replays: 83 / 20
  ub16     hc_attn_pre                                  1.117e-04  (L10, g1)
  ub16     KDA out                                      6.993e-04  (L8, g1)
  ub16     mHC attn post                                2.113e-04  (L11, g3)
  ub16     ffn_out                                      3.197e-04  (L11, g3)
  ub16     l_out                                        1.989e-04  (L11, g3)
  ub16     DSA kqv_out                                  2.911e-04  (L11, g1)
  ub16     logits                                       1.315e-04
  ub16     greedy argmax equal on 64/64 rows
  ub16     Qcur (local)                                 1.829e-07  (L3, g1)
  ub16     kqv_out (local)                              1.697e-07  (L7, g0)
  ub16     indexer key (local, ik eps)                  1.993e-07  (L7, g0)
  ub16     pooled keys (local)                          8.505e-08  (L3, g2)
  ub16     indexer scores (local)                       1.401e-07  (L3, g3)
  ub16     ffn_out (local)                              2.480e-07  (L3, g3)
  ub16     DSA rows attending cell 0 by tail padding / tie replays: 83 / 20

----------------------------------------------------------------------
Ran 43 tests in 11.849s

OK
artifacts: <scratch>/n12/work-ci
driver rc=0 wall_s=353
```
Deviation sizes on ik's own dumps,
`docs/evidence/mutation-scripts/2026-10-11-n12-deviations.py` (environment as the test's):
```
p64: item 1 SwiGLU clamp, ffn_out from ik's ffn_norm: ik semantics 2.09e-07, llama.cpp 4.84e-01 (expert gate values above the limit in llama.cpp's own route: 14007)
p64: item 2 indexer LayerNorm eps, from ik's attn_norm: rms eps 1.65e-07, eps 0 1.01e-05
p64: item 4 attention over ik's cells 1.60e-07; without the padded cell 0 3.76e-01 (90 row-layers padded)
p64: logits ik vs ik-semantics oracle 1.26e-04, argmax equal 64/64; vs llama.cpp-semantics oracle 9.05e-01, argmax equal 19/64
decode: item 1 SwiGLU clamp, ffn_out from ik's ffn_norm: ik semantics 3.11e-07, llama.cpp 7.10e-01 (expert gate values above the limit in llama.cpp's own route: 14007)
decode: item 2 indexer LayerNorm eps, from ik's attn_norm: rms eps 1.76e-07, eps 0 1.01e-05
decode: item 4 attention over ik's cells 2.30e-07; without the padded cell 0 5.38e-01 (90 row-layers padded)
decode: logits ik vs ik-semantics oracle 1.28e-04, argmax equal 64/64; vs llama.cpp-semantics oracle 9.05e-01, argmax equal 19/64
fa16: item 1 SwiGLU clamp, ffn_out from ik's ffn_norm: ik semantics 1.89e-07, llama.cpp 4.84e-01 (expert gate values above the limit in llama.cpp's own route: 14006)
fa16: item 2 indexer LayerNorm eps, from ik's attn_norm: rms eps 1.66e-07, eps 0 1.01e-05
fa16: item 4 attention over ik's cells 1.60e-07; without the padded cell 0 3.75e-01 (90 row-layers padded)
fa16: logits ik vs ik-semantics oracle 5.29e-04, argmax equal 64/64; vs llama.cpp-semantics oracle 9.06e-01, argmax equal 19/64
item 1 bound for GLM-5.3-Flash (swiglu_limit 10.0): max |min(silu(g), L) - silu(min(g, L))| = L * sigmoid(-L) = 4.540e-04; for the tiny model's 0.5: 1.888e-01
item 3 L2 norm, 1e5 random 16-vectors: max |ik - llama.cpp| = 1.25e-07
```

## 4. ik's llama-server

Flags of the template against ik's `--help` (CPU build), `docs/evidence/mutation-scripts/2026-10-11-n12-ik-help-flags.sh` (each line: the flag, then the first `--help` line that mentions it, cut at 110 characters):
```
ik llama-server --help: exit 1, stdout 473 lines, stderr 0 lines
llama.cpp (R1 pin) llama-server --help: exit 0
-m               -m,    --model FNAME            model path (default: models/$filename with filename from --hf-file
--host           --host HOST              ip address to listen (default: 127.0.0.1)
--port           --port PORT              port to listen (default: 8080)
--ctx-size       -c,    --ctx-size N             size of the prompt context (default: 0, 0 = loaded from model)
--parallel       -np,   --parallel N             number of parallel sequences to decode (default: 1)
--jinja          --jinja                  set custom jinja chat template (default: template taken from model's metadata)
-ngl             NOT IN --help
-sm              -gap, --graph-attn-precision    Flash-attn precision under -sm graph (default: f16)
-ts              NOT IN --help
-ncmoe           -ncmoe, --n-cpu-moe N           keep MoE weights of the first N layers in CPU memory
-fa              -fa, --flash-attn (auto|on|off|0|1)
--dsa            -dsa,  --dsa                    enable GLM DSA sparse attention (GLM-DSA arch only; default: disabled)
-ctk             -ctk,  --cache-type-k TYPE      KV cache data type for K (default: f16)
-ctv             -ctv,  --cache-type-v TYPE      KV cache data type for V (default: f16)
-t               -t,    --threads N              number of threads to use during generation (default: 4)
--threads-batch  -tb,   --threads-batch N        number of threads to use during batch and prompt processing (default: same as 
-b               -b,    --batch-size N           logical maximum batch size (default: 2048)
-ub              -ub,   --ubatch-size N          physical maximum batch size (default: 512)
-muge            -muge,  --merge-up-gate-experts 
-ictk            -ictk, --indexer-cache-type-k TYPE
--slot-save-path --slot-save-path PATH    path to save slot kv cache (default: disabled)
```
`-sm` is found only inside the description of `-gap`; its own entry is hidden like those of `-ngl` and `-ts` (CC:3585 prints them only when `llama_supports_gpu_offload()`). A text search such as run_r1's flags check (scripts/r1/run_r1.py:1156-1177) would therefore report `-ngl` and `-ts` missing and `-sm` present on a CPU build; on a CUDA build all three should be listed (PROVISIONAL: not built here).

`docs/evidence/mutation-scripts/2026-10-11-n12-ik-server-probe.py` (template argv substituted for 4 CPU cores; JSON
key layouts of /props, /v1/models, /completion and /slots cut: they are printed in full by the script):
```
argv (substituted): <scratch>/n12/ik-build/bin/llama-server -m <scratch>/n12/work-ci/ik.gguf --host 127.0.0.1 --port 18471 --ctx-size 262144 --parallel 1 --jinja -ngl 999 -sm layer -ts 1,1,1,1 -ncmoe 17 -fa on --dsa -ctk f16 -ctv f16 -t 4 --threads-batch 4 -b 2048 -ub 1024 -muge
--version rc 0 | ['version: 1 (5194a9e)', 'built with cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 for x86_64-linux-gnu']
template argv: did not become healthy (rc 1) None
log lines with 'warning: not compiled with GPU offload': ['warning: not compiled with GPU offload support, --gpu-layers option will be ignored']
log lines with 'error:': ['error: invalid parameter for argument: -ts']
log lines with 'warning: llama.cpp was compiled without': ['warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting the split mode has no effect.']
CPU argv (the template's without -ngl/-sm/-ts): -m <scratch>/n12/work-ci/ik.gguf --host 127.0.0.1 --port 18471 --ctx-size 262144 --parallel 1 --jinja -ncmoe 17 -fa on --dsa -ctk f16 -ctv f16 -t 4 --threads-batch 4 -b 2048 -ub 1024 -muge
CPU argv: started (200, '{"status":"ok","slots_idle":1,"slots_processing":0}')
log lines with 'n_ctx': ['llm_load_print_meta: n_ctx_train      = 512', 'llm_load_print_meta: n_ctx_orig_yarn  = 512', 'llama_init_from_model: n_ctx         = 262144']
log lines with 'merge': ['print_info: n_merges              = 0', 'merge_up_gate_exps: merging up/gate in layer 1', 'merge_up_gate_exps: merging up/gate in layer 2']
--- GET /props: HTTP 200
--- GET /v1/models: HTTP 200
--- POST /tokenize {content: 't5t9'}: HTTP 500
--- POST /tokenize with_pieces: HTTP 500
--- POST /detokenize [5, 9]: HTTP 200
--- POST /completion n_predict 16 ignore_eos n_probs 3 (layout): HTTP 200
tokens_predicted 16 stop_type None tokens None
completion_probabilities[0]: {"content": "t62", "probs": [{"tok_str": "t62", "prob": 0.060870010405778885}, {"tok_str": "t47", "prob": 0.029555555433034897}, {"tok_str": "t87", "prob": 0.028698427602648735}]}
server greedy tokens: 62,84,5,52,66,39,9,27,6,54,44,17,36,50,88,22
--- POST /completion logit_bias EOS +100, ignore_eos false: HTTP 200
tokens_predicted 1 stop_type None
--- POST /completion logit_bias EOS +100, ignore_eos true: HTTP 200
tokens_predicted 16 stop_type None
--- POST /v1/completions logprobs 3 max_tokens 2 (layout): HTTP 200
choices[0].logprobs: {"content": [{"id": 62, "token": "t62", "bytes": [116, 54, 50], "logprob": -2.7990145683288574, "top_logprobs": [{"id": 62, "token": "t62", "bytes": [116, 54, 50], "logprob": -2.7990145683288574}, {"id": 47, "token": "t47", "bytes": [116, 52, 55], "logprob": -3.5214836597442627}, {"id": 87, "token": "t87", "bytes": [116, 56, 55], "logprob": -3.550912857055664}]}, {"id": 84, "t
--- GET /slots (no --slot-save-path): HTTP 200
--- POST /slots/0?action=save (no --slot-save-path): HTTP 404
CPU argv + --slot-save-path: started
completion before save: HTTP 200 t62t84t5t52t66t39t9t27 tokens_cached 55
--- POST /slots/0?action=save: HTTP 200
files in slot dir: {'n12.bin.tokens.json': 242, 'n12.bin': 74504}
--- POST /slots/0?action=erase: HTTP 200
--- POST /slots/0?action=restore: HTTP 200
before save: t62t84t5t52t66t39t9t27 | after restore: t62t84t5t52t66t39t9t27 | identical: True | prompt_n after restore: 48 of 48 | tokens_cached 55
--- POST /slots/0?action=erase: HTTP 200
--- POST /slots/0?action=save (empty slot): HTTP 200
restore of the empty save: no HTTP answer: RemoteDisconnected Remote end closed connection without response
server process after it: returncode -6
server log: ['<scratch>/n12/ik-src/src/llama.cpp:11739: GGML_ASSERT(kv_self.cells[kv_self.head].pos == batch.pos[0]) failed']
```
Server tokens against the engine at the same flags,
`docs/evidence/mutation-scripts/2026-10-11-n12-server-greedy.sh` (table and summary of the test; `ok` lines cut):
```
ik-dump: 17 calls, 17 graphs, 64 tokens
engine argmax at positions 47..62: 62,84,5,52,66,39,9,27,6,54,44,17,36,50,88,22
server tokens 48..63:             62,84,5,52,66,39,9,27,6,54,44,17,36,50,88,22
equal: 16 / 16
......
worst normwise relative error per run and check (layer, graph):
  fa16dec  hc_attn_pre                                  1.142e-03  (L10, g14)
  fa16dec  KDA out                                      3.167e-03  (L9, g14)
  fa16dec  mHC attn post                                1.284e-03  (L9, g14)
  fa16dec  ffn_out                                      1.917e-03  (L11, g14)
  fa16dec  l_out                                        1.427e-03  (L9, g14)
  fa16dec  DSA kqv_out                                  1.382e-03  (L11, g7)
  fa16dec  logits                                       5.295e-04
  fa16dec  greedy argmax equal on 64/64 rows
  fa16dec  Qcur (local)                                 3.247e-07  (L3, g5)
  fa16dec  kqv_out (local)                              1.906e-07  (L7, g3)
  fa16dec  indexer key (local, ik eps)                  1.797e-07  (L11, g11)
  fa16dec  pooled keys (local)                          7.410e-08  (L7, g0)
  fa16dec  indexer scores (local)                       1.680e-07  (L7, g4)
  fa16dec  ffn_out (local)                              2.855e-07  (L8, g7)
  fa16dec  DSA rows attending cell 0 by tail padding / tie replays: 82 / 19

----------------------------------------------------------------------
Ran 6 tests in 4.110s

OK
```
N11's tools against two ik servers on a byte-BPE tiny model (N11's `glm5next_tiny.py --arch glm5next --text-vocab`,
sha256 `4210b67775a0dddc87f8453a01f8b7e7aa83447098beaa4a52ca85539ce72d26`),
`docs/evidence/mutation-scripts/2026-10-11-n12-ik-logprob.sh`:
```
both servers healthy
--- POST /tokenize {content: 'Hello world'}
{"tokens":[75,104,111,111,114,281,114,117,111,103]}
--- POST /tokenize with_pieces
{"tokens":[75,104,111,111,114,281,114,117,111,103]}
--- prompts.py --verify
{"kind": "prose", "target": 256, "tokens": 256, "exact": true, "method": "server-tokenize", "chars": 290, "n_units": 37, "pad": 0, "calls": 4, "verify_usage_prompt_tokens": 306}
{"kind": "prose", "target": 1024, "tokens": 1022, "exact": true, "method": "server-tokenize", "chars": 1168, "n_units": 157, "pad": 1, "calls": 8, "verify_usage_prompt_tokens": 1072}
prompts.py rc=1
--- logprob_compare ref (-fa off) vs cand (template CPU flags)
29 prefixes (11 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0014 median 0.0000 p95 0.0091 max 0.0239 -> <scratch>/n12/lp/work/lp.json
summary {"n": 29, "top1_agree": 1.0, "ref_in_cand_topk": 1.0, "kl_topk_mean": 0.001409158882013493, "kl_topk_median": 7.833439596433e-07, "kl_topk_p95": 0.009147078063053317, "kl_topk_max": 0.023908550855498735}
errors 11 {'incomplete UTF-8 (known llama-server limitation, N11)': 11}
```
The same `/tokenize` and `prompts.py` on llama.cpp's llama-server at the R1 pin (scratch build of N7), same weights
under the name `glm5-next`, `docs/evidence/mutation-scripts/2026-10-11-n12-mainline-tokenize.sh`:
```
version: 0.6.0-dev (build 1, commit 79e2e74)
wrote <scratch>/n12/mltok/glm5-next_text.gguf
--- POST /tokenize {content: 'Hello world'}
{"tokens":[75,104,111,111,114,281,114,117,111,103]}
--- POST /tokenize with_pieces
{"tokens":[{"id":75,"piece":"H"},{"id":104,"piece":"e"},{"id":111,"piece":"l"},{"id":111,"piece":"l"},{"id":114,"piece":"o"},{"id":281,"piece":" w"},{"id":114,"piece":"o"},{"id":117,"piece":"r"},{"id":111,"piece":"l"},{"id":103,"piece":"d"}]}
--- prompts.py --verify
{"kind": "prose", "target": 256, "tokens": 256, "exact": true, "method": "server-tokenize", "chars": 290, "n_units": 37, "pad": 0, "calls": 4, "verify_usage_prompt_tokens": 306}
{"kind": "prose", "target": 1024, "tokens": 1022, "exact": true, "method": "server-tokenize", "chars": 1168, "n_units": 157, "pad": 1, "calls": 8, "verify_usage_prompt_tokens": 1072}
prompts.py rc=1
```
N11's `slot_persist.py` against ik (template CPU flags + `--slot-save-path`, 3000-character prompt of the corpus),
`docs/evidence/mutation-scripts/2026-10-11-n12-ik-slot-persist.sh` (the first lines of the tool's JSON summary are cut
by its `tail -n 5`):
```
 "warm_prompt_n_p50": 5.0,
 "saved_bytes": 1136236.0,
 "server_restore_ms_p50": 1.6564999999999999,
 "all_continuations_identical": true
}
{'same_continuation': True, 'compared': 'content'} prompt_n cold 2547 warm 5 | restore {"id_slot": 0, "filename": "k3-slot0.bin", "n_restored": 2562, "n_read": 826604, "timings": {"restore_ms": 1.8
{'same_continuation': True, 'compared': 'content'} prompt_n cold 2547 warm 5 | restore {"id_slot": 0, "filename": "k3-slot0.bin", "n_restored": 2562, "n_read": 826604, "timings": {"restore_ms": 1.4
errors []
```

## 5. Mutation check

`docs/evidence/mutation-scripts/2026-10-11-n12-ik-mutation.py <repo> <artifacts> <gguf-py> <scratch>` on copies of
`ref/` and the test, and on copies of single dumped runs:
```
artifacts <scratch>/n12/work-ci
baseline (no mutation): rc=0 Ran 43 tests in 11.528s | OK
O1 SwiGLU clamp before the SiLU (llama.cpp's ggml_swiglu_clamp)
    rc=1 Ran 43 tests in 2.167s | FAILED (failures=20)
    failed: IkDecode.test_chain, IkDense.test_chain, IkFa16.test_chain, IkFa16Decode.test_chain, IkIctk16.test_chain, IkP64.test_chain, IkUb16.test_chain, IkDecode.test_ffn_local, IkDense.test_ffn_local, IkFa16.test_ffn_local, IkFa16Decode.test_ffn_local, IkIctk16.test_ffn_local, IkP64.test_ffn_local, IkUb16.test_ffn_local, IkDecode.test_selection, IkFa16.test_selection, IkFa16Decode.test_selection, IkIctk16.test_selection, IkP64.test_selection, IkUb16.test_selection
O2 no cell 0 from ik's tail padding (llama.cpp's sentinel)
    rc=1 Ran 43 tests in 2.258s | FAILED (failures=12)
    failed: IkDecode.test_chain, IkFa16.test_chain, IkFa16Decode.test_chain, IkIctk16.test_chain, IkP64.test_chain, IkUb16.test_chain, IkDecode.test_selection, IkFa16.test_selection, IkFa16Decode.test_selection, IkIctk16.test_selection, IkP64.test_selection, IkUb16.test_selection
O3 indexer LayerNorm eps 0 when the key is absent (llama.cpp's rule)
    rc=1 Ran 43 tests in 11.699s | FAILED (failures=6)
    failed: IkDecode.test_dsa_local, IkFa16.test_dsa_local, IkFa16Decode.test_dsa_local, IkIctk16.test_dsa_local, IkP64.test_dsa_local, IkUb16.test_dsa_local
O4 KDA q/k L2 norm as llama.cpp: x / sqrt(|x|^2 + 1e-6) (expected to survive)
    rc=0 Ran 43 tests in 11.495s | OK
O5 MLA latent kept in f32 by the oracle (no f16 store)
    rc=1 Ran 43 tests in 11.629s | FAILED (failures=1)
    failed: IkDense.test_chain
O6 a pool visible one token before it is complete
    rc=1 Ran 43 tests in 2.404s | FAILED (failures=12)
    failed: IkDecode.test_chain, IkFa16.test_chain, IkFa16Decode.test_chain, IkIctk16.test_chain, IkP64.test_chain, IkUb16.test_chain, IkDecode.test_selection, IkFa16.test_selection, IkFa16Decode.test_selection, IkIctk16.test_selection, IkP64.test_selection, IkUb16.test_selection
O7 flash attention emulated as exact f32 attention (no fa_chunk)
    rc=1 Ran 43 tests in 10.692s | FAILED (failures=3)
    failed: IkFa16Decode.test_chain, IkFa16.test_dsa_local, IkFa16Decode.test_dsa_local
T1 test: the latent cache type taken from -ctk also with -fa off (llama.cpp's rule)
    rc=1 Ran 43 tests in 11.562s | FAILED (failures=6)
    failed: IkDecode.test_cache_stores, IkDense.test_cache_stores, IkIctk16.test_cache_stores, IkP64.test_cache_stores, IkUb16.test_cache_stores, IkDense.test_chain
T2 test: -ictk default f32 instead of ik's f16
    rc=1 Ran 43 tests in 12.095s | FAILED (failures=2)
    failed: IkFa16.test_cache_stores, IkFa16Decode.test_cache_stores
T3 test: no tie replay
    rc=1 Ran 43 tests in 2.203s | FAILED (failures=12)
    failed: IkDecode.test_chain, IkFa16.test_chain, IkFa16Decode.test_chain, IkIctk16.test_chain, IkP64.test_chain, IkUb16.test_chain, IkDecode.test_selection, IkFa16.test_selection, IkFa16Decode.test_selection, IkIctk16.test_selection, IkP64.test_selection, IkUb16.test_selection
D1 ik side: one cached latent element one binary16 spacing off (L7, decode graph 5)
    rc=1 Ran 43 tests in 11.399s | FAILED (failures=1)
    failed: IkDecode.test_cache_stores
D2 ik side: one attended cell changed (L11, p64, token 40)
    rc=1 Ran 43 tests in 12.287s | FAILED (failures=3)
    failed: IkP64.test_chain, IkP64.test_dsa_local, IkP64.test_selection
D3 ik side: FFN output 2^-12 too large (L5, p64)
    rc=1 Ran 43 tests in 11.777s | FAILED (failures=1)
    failed: IkP64.test_ffn_local
D4 ik side: one indexer cache row rounded to f16 under -ictk f32 (L3, ub16 graph 2)
    rc=1 Ran 43 tests in 11.322s | FAILED (failures=2)
    failed: IkUb16.test_cache_stores, IkUb16.test_dsa_local
caught 13/14
```
O4 (llama.cpp's L2 eps) survives as expected: the two L2 forms differ by 1.25e-7 (section 3), below every tolerance;
the test cannot tell them apart and does not claim to.

## 6. CI

`.github/workflows/parity-r1-pin.yml` gets a second job, `ik-glm5next`, in parallel with `parity` (the existing job
does not get longer): checkout, `pip install numpy pyyaml`, a shallow fetch of llama.cpp at the R1 pin (for its
gguf-py), a shallow fetch of ik_llama.cpp at its pin, then `tools/tinygen/run_parity_ik_glm5next.sh`. The path filters
gain `tests/test_ik_glm5next_parity.py` and `tools/ik/**` (`tools/tinygen/**`, `tools/dump/**`, `ref/**` and
`tools/engines/pins.json` were already in them).

Time: the same driver from an empty build directory on the 4 sandbox cores, `driver rc=0 wall_s=353` (last line of the
output in section 3; most of it is the build of ggml, llama, common and the dump tool: the earlier timed `tools/ik`
build alone took configure 1 s + build 325 s). 353 s is inside the task's 8-minute budget, so the step was added
(CONFIRMED locally). Its time on a GitHub runner: UNKNOWN until the job runs (NOT RUN: no push from this session).

## NOT RUN

* MTP: ik drives its MTP block only through `--spec-type mtp:...` in the speculative loop (`-mtp` prints "legacy
  speculative option '-mtp' is disabled"); the dump tool does not run it. Numerical MTP parity of ik: NOT RUN. (Known
  difference by reading: ik zeroes the token embedding at position 0 in the MTP graph, B:396-404; llama.cpp's
  graph_mtp normalises the embedding as it is, ML src/models/glm5-next.cpp:583-592.)
* Multiple sequences, `-np` > 1, rollback (`llama_kv_cache_seq_rm` on ik's recurrent checkpoints): NOT RUN.
* Quantized weights (Q4_K etc.) on ik, `-muge` other than through the server-flag run, `-ctk q8_0`: NOT RUN.
* CUDA, `-sm graph/layer`, `-ncmoe` with a GPU, 262K contexts with real selection pressure: NOT RUN (no GPU).
* The real UD-Q4_K_XL file: NOT RUN; nothing here is about its contents beyond its documented header facts.
* `scripts/r1/run_r1.py` with ik-llamacpp-server.json: NOT RUN (its argv cannot start on a CPU build, finding 5).
* The ik job on GitHub Actions: NOT RUN (no push from this session).
