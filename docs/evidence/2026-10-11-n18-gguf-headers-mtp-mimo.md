# N18: more real GGUF headers (separate MTP files, the ggml-org GLM files, the MiMo listing) with the N15 machinery

Date 2026-10-11. Role: model-analyst. Branch `claude/n18-gguf-headers-mtp-mimo`, cut from `claude/n15-gguf-headers`
@`67bd5ba` (Draft PR #12). No hardware used. `<S>` stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; the outputs are pasted with only that
substitution.

How the work went:
- Round 1 extended the tooling and committed the requests (section 1). The lead pushed `8c0b424`.
- Workflow `gguf-headers` run 38145613553 read the headers and committed them as bot commit `df9bea1`, which was
  fast-forwarded here.
- Round 2 analyses them offline (sections 2–8) and adds the MiMo header requests (section 9).
- Round 3: the lead pushed `65c3805`; run 38147482333 committed the MiMo headers as bot commit `6161763`, which was
  fast-forwarded here. They are analysed in sections 13–15 and appendix B. The CORRECTNESS note is at the end of the
  gap 9 status paragraph.
- docs/reassessment/CORRECTNESS.md gap 9.

Labels: CONFIRMED means read from the committed header bytes, the hub listing, or the source at a pin (file:line,
printed back with `sed -n` / located by the analyzer at run time). PROVISIONAL means by reading: nothing was loaded,
there are no weights here.

## Result (round 3: MiMo-V2.6-Flash-MOPD)

Run 38147482333 (bot commit `6161763`) read the headers of the four MiMo requests: 31 376 608 B fetched, integrity OK
for all 5 files (section 13).

1. **Hyperparameters against the vendored config** (`data/gguf-headers/mimo-v2.6-flash-config/config.json`, copy of
   bot `1ce4d32`), CONFIRMED for all four files (analyzer cross-check):
   - window 128;
   - full-attention layers 0, 5, 11, 17, 23, 29, 35, 41, 47; the other 39 are sliding-window;
   - KV heads 4 (full) / 8 (SWA); 64 query heads;
   - K 192 / V 128; RoPE on 64 dims (`int(192 × 0.334)`);
   - freq bases 1e7 (full) / 1e4 (SWA); value scale 0.707;
   - **sinks on the 39 SWA trunk layers and on the 3 NextN layers, on no full-attention layer**;
   - `nextn_predict_layers` 3 with the 3 NextN layers as SWA layers with 8 KV heads (converter: + 1 / + SWA KV per MTP
     layer).

   | file | config.json cross-check | not in the file |
   |---|---|---|
   | ggml-org trunk | 21 of 22 OK | `nextn_predict_layers`: trunk only, block_count 48 |
   | ggml-org `mtp-` | 22 of 22 OK | — |
   | QuantaPlanta | 22 of 22 OK | — |
   | kernelpool | 20 of 22 OK | the two expert-group keys (n_group = topk_group = 1: no group step either way) |

2. **Experts and D5**, CONFIRMED:
   - In all three MXFP4 files the routed experts are **MXFP4** in all 47 MoE layers (`ffn_{gate,up,down}_exps`).
     - llama.cpp's converter writes exactly that from the checkpoint's MXFP4 store (`conversion/mimo.py:243`,
       `raw_dtype=MXFP4`) and sets `MOSTLY_MXFP4_MOE` (`:338-340`; `= 38`, `include/llama.h:156@ec7630a` / `:157@79e2e74`).
     - `llama-quantize … MXFP4_MOE` gives the same expert type (`src/llama-quant.cpp:512-523`).
   - **D5 PASS** for all four, routed experts at 4.25 bpw.

   | file | non-expert matrices | total size | bpw |
   |---|---|---|---|
   | ggml-org trunk | Q8_0 | 155.87 GiB | 4.336 |
   | QuantaPlanta | BF16 | 162.89 GiB | 4.517 |
   | kernelpool | Q8_0, but `token_embd` BF16 | 157.39 GiB | 4.365 |
   | ggml-org `mtp-` | Q8_0 | 2.214 GiB | 8.501 |

3. **The three MXFP4 files compared**, CONFIRMED:
   - Same trunk tensor names and shapes (layers 0–47 and the non-layer tensors, 472). Same tokens and merges; the
     template is the vendored MOPD one (`11ea52e1…`) in all four files.
   - QuantaPlanta and kernelpool also hold the 3 NextN layers (`blk.48`–`50`, 12 tensors each). Their 987 820 224
     params more than ggml-org's (round 2) are these layers.
   - The ggml-org `mtp-…-Q8_0` holds the same 3 NextN layers (names and shapes = QuantaPlanta's) plus `token_embd`,
     `output_norm`, `output`.
   - **Which is llama.cpp's converter output** (PROVISIONAL, from the metadata; no converter run):
     - **QuantaPlanta** = `convert_hf_to_gguf.py` with BF16 for the non-MXFP4 tensors: `general.file_type` 38, the
       gguf-py key set, MTP head norm as `layer_output_norm`. llama.cpp's tensor map sends
       `model.layers.{bid}.final_layernorm` there (`gguf-py/gguf/tensor_mapping.py:743@ec7630a`, `:755@79e2e74`).
     - **ggml-org** = the same layout with Q8_0 non-expert matrices: the converter with `--outtype q8_0`, or
       `llama-quantize MXFP4_MOE`, which gives the same types. Its trunk ships without the NextN layers, which come as
       separate `mtp-` files.
     - **kernelpool** is not from llama.cpp's converter. It has no `general.file_type` or `quantization_version`. It
       has its own keys: `mimo2.qkv_chunks` = 4, `mimo2.quantization`, `vocab_size`, `leading_dense_block_count`,
       `expert_weights_scale` / `_norm`, `general.source.revision` = `2479e2d…` (not the vendored `b12f557`).
       - Its MTP head norm is in `nextn.shared_head_norm` (llama.cpp maps `…shared_head.norm` there,
         `tensor_mapping.py:2772` / `:2874`).
       - 6 FIM tokens (`<|fim_prefix|>` … `<|file_sep|>`) are USER_DEFINED instead of CONTROL.
     - None of kernelpool's extra keys is read by `mimo2` at either pin: grep of `src/models/mimo2.cpp` and the
       `load_hparams` body. `vocab_size` is read only for a file without a vocabulary (`src/llama-vocab.cpp:1963` /
       `:2076`).
     - **Risk, UNKNOWN:** `qkv_chunks = 4` suggests the fused `attn_qkv` rows are stored in 4 TP chunks.
       llama.cpp splits it as one `[Q | K | V]` (`src/models/mimo2.cpp:134-142@ec7630a`), and llama.cpp's converter
       regroups TP chunks (`conversion/mimo.py:35-98`). If kernelpool did not regroup, llama.cpp would compute wrong
       Q/K/V with this file. To confirm: compare the Q8_0 bytes of a few `blk.0.attn_qkv` rows of ggml-org and
       kernelpool (a data read, not a header).
4. **Loader verdict per pin** (PROVISIONAL, by reading; nothing loaded):
   - Every tensor of the four files is one `llama_model_mimo2::load_arch_tensors` creates, and none of the required ones
     is missing, at both pins (`committed-facts.py mimoloader`, parsed from `src/models/mimo2.cpp` + `src/llama-arch.cpp`).
     The 8 keys read without a default are present in every file.
   - **ggml-org trunk:** a plain trunk-only model.
   - **QuantaPlanta, kernelpool:** trunk + NextN in one file. MTP runs on the target's own NextN layers:
     `--spec-type draft-mtp`, no `-md`. The NextN graph exists at both pins (`mimo2.cpp:88@ec7630a`, `:82@79e2e74`).
   - **ggml-org `mtp-` file:** a separate MTP model at both pins. MTP-only detection is at `mimo2.cpp:28` at both
     pins, with different code: `mtp_only` at ec7630a, `nextn_flags(ml)` at 79e2e74 (`src/llama-model.cpp:3485`).
     The required `token_embd`, `output_norm` and `output` are present.
   - `-hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4 --mtp` would pick `mtp-…-Q4_0` (4 bits, the tag's bits;
     `find_best_sibling`, section 5).
   - The trunk emits `h_nextn` for the drafter: at ec7630a only when `cparams.embeddings_nextn` is set
     (`mimo2.cpp:104`, `:244-246`), at 79e2e74 always (`:236-238`).
   - The source lines cited in items 2–4 are printed back in section 14.
5. **Branches only the real geometry takes vs N16/N17's tiny models.** CONFIRMED by the probes on tiny models built
   from those branches' generator:
   - N16 `0b0eace`: `n16.gguf`;
   - N17 `0352df4`, a local branch not on origin: `n17.gguf` with 3 MTP heads, and `n17hd.gguf` with K 192 / V 128,
     RoPE 64.

   | file | trunk branches | branches no N16/N17 tiny model takes |
   |---|---|---|
   | ggml-org trunk | all exercised: fused qkv, sinks on SWA only, value scale, K ≠ V, partial RoPE with two bases, dense layer 0 + sigmoid MoE | — |
   | QuantaPlanta | all exercised | — |
   | kernelpool | all exercised | MTP head norm `nextn.shared_head_norm` (N17 section 5: `shared_head_*` "fallbacks checked" only) |
   | ggml-org `mtp-` | — | the MTP-only separate file (`-md`) |

   (On this branch's older generator, the analyzer marks value scale, sinks, qkv and the MTP branches NOT IN TINY,
   appendix B.)
   - What N16 / N17 cover of these files, PROVISIONAL (their evidence, not re-run here):
     - N16: MXFP4 experts with Q8_0 matrices on the CPU, i.e. `llama-quantize MXFP4_MOE`, the ggml-org layout; the
       fused qkv, SWA-only sinks and value-scale path; one MTP head.
     - N17: the 3 MTP heads (offsets 0–2), f16 KV + flash attention with sinks, and K 192 / V 128 heads.
   - **Not covered:**
     - the sizes: window 128 (tiny 8), 64 / 4 / 8 heads, 256 experts top-8, 48 layers;
     - BF16 matrices (QuantaPlanta) and BF16 `token_embd` (kernelpool);
     - a quantized (Q8_0) MTP block (N17 section 5);
     - the separate MTP file path.
6. **Templates**, CONFIRMED: all four embed the vendored MOPD template (`11ea52e156de…`, sha256 of
   `git show origin/claude/n16-mimo-decode-parity:third_party/hf/mimo-v2.6-flash-chat/chat_template.jinja`).

## Result (round 2)

1. **The runner read only headers.** CONFIRMED: run 38145613553, conclusion success, steps 1–7 success; the GitHub API
   output is in section 2.
   - Bytes fetched, from PIN.json:
     - Qwen MTP: 100 663 296 B (6 × 16 777 216);
     - ggml-org GLM Q4_K: 9 692 096 B;
     - ggml-org mtp-GLM: 33 554 432 B (2 × 16 777 216);
     - together 143 909 824 B, against 220 220 026 176 B of files.
   - Every header is ≤ 10 946 688 B. The Range doubling stopped at 16 MiB, below the 32 MiB cap.
   - Integrity is CONFIRMED for all 10 files: the committed headers re-derive their summaries, the hub etag equals the
     LFS sha256, the repo commit equals the pin, and every size check is exact.
   - The metadata-only first shard of the ggml-org Q4_K is the whole file. It is the header plus 1 padding byte, and
     its sha256 is the LFS sha256 (`7d658490…`).
2. **The 6 Qwen MTP files** (`unsloth/Qwen3.8-Flash-Next-GGUF` @`38bb39e`, `MTP/`), CONFIRMED from the headers:
   - Each is `qwen4exp`, with `block_count` 49 and `nextn_predict_layers` 1.
   - Each holds only layer 48. That layer has 32 tensors, the same names as the NextN layer of the tiny
     `qwen4exp --mtp` model, which loads at both pins.
   - Each carries the full tokenizer (248 320 tokens) and no `ple.*` keys.
   - Each embeds the **vendored HF chat template** (sha256 `c3cf9e34…`), not the fixed Unsloth template of the
     UD-Q4_K_XL (`12827f24…`).
   - The plain files add `token_embd` and `output`.
   - **The `-shared-` files lack exactly `token_embd.weight` and `output.weight`.** File size difference = their bytes +
     64 B of header. They add the KV `qwen4exp.nextn_shared_target_tensors = true`, which none of the three checkouts
     reads (ec7630a, 79e2e74, ik 5194a9e: `grep` finds nothing).
   - **Loader verdict** (PROVISIONAL, by reading; nothing loaded):
     - plain BF16 / Q4_K_M / Q8_0: loadable as a separate MTP model at **both** pins (`-md FILE --spec-type draft-mtp`);
     - all three `-shared-`: **NOT loadable at either pin**, because `token_embd` is created with flags 0
       (`src/models/qwen4exp.cpp:183@ec7630a`, `:182@79e2e74`).
   - **New: in all 6 files the NextN layer's compress ratio is 0** (`attention.compress_ratios[48] = 0`, CONFIRMED).
     llama.cpp's converter writes the QSA ratio for every MTP layer: `conversion/qwen4exp.py:85-89` at both pins,
     comment "the MTP block is a full-attention QSA layer too". So at the pins the MTP layer of these files runs dense
     full attention instead of QSA (PROVISIONAL, by reading): `qsa = inp_kpool != nullptr &&
     hparams.dsv4_compress_ratios[il] > 0`, `src/models/qwen4exp.cpp:942@ec7630a` / `:939@79e2e74`.
     - The effect on the drafts is UNKNOWN. It can only appear when the context is longer than the QSA budget (2048).
     - Which attention the HF model uses in its MTP layer is PROVISIONAL (from the converter's comment).
   - D5: PASS for all 6.
3. **Separate MTP files at the pins** (CONFIRMED by reading; section 5).
   - **How a file is attached:** as a draft model, `-md FILE --spec-type draft-mtp`. It is loaded as its own
     `llama_model` with an MTP context; its output width must equal the target's. The qwen4exp trunk emits the hidden
     state the drafter needs, also when it has no NextN layer.
   - **Which pins accept an MTP-only file:**
     - qwen4exp and mimo2: both pins;
     - glm5-next: only 79e2e74. ec7630a has no MTP-only handling and no GLM MTP graph at all.
   - **What a separate file must contain:**
     - all required KV, with `block_count` = trunk + NextN layers and `nextn_predict_layers` ≥ 1;
     - the tokenizer;
     - `token_embd` (glm5-next and mimo2 also `output_norm` and `output`; qwen4exp falls back to `token_embd` for the
       head);
     - the NextN layers;
     - no tensor the loader does not create;
     - for qwen4exp, no `ple.*` keys.
   - **`-hf … --mtp`** (`find_best_sibling`, identical at both pins) only takes an `mtp-` file from the model's own
     directory or a parent directory. So it does **not** find Unsloth's `MTP/` folder for a model in `UD-Q4_K_XL/`
     (PROVISIONAL, by reading). For ggml-org's root-level files with tag Q4_K it would pick `mtp-…-Q4_0`, the closest
     bits.
4. **ggml-org GLM-5.3-Flash Q4_K trunk** (`ggml-org/GLM-5.3-Flash-GGUF` @`c259a9f`), CONFIRMED from the headers:
   - `glm5-next`, 45 layers, **no NextN layer**: no `nextn_predict_layers`, no `nextn` tensors, 1383 tensors.
     - Its parameter count (313 326 811 966, from the tensor shapes) is exactly the unsloth UD-Q4_K_XL's minus the
       unsloth NextN layer `blk.45` (7 432 592 416).
   - **Its embedded chat template is byte-equal to the vendored HF template** (10 950 B, `0c4099f3…`). It therefore
     carries the same `{% set reasoning_content %}` block as the vendored template (N14 1(a)).
   - Config cross-check: 33 of 34 OK. `num_nextn_predict_layers` is ABSENT, as a trunk-only file should be.
   - Unlike unsloth's file, it writes `attention.indexer.types` (all true) and `kpool_select_tail` (true): the same
     graph as the defaults.
   - **D5 audit: PASS.** Routed experts are 100 % Q4_K (4.500 bpw); 168.91 GiB at 4.631 bpw.
   - Not Q8_0, all above the floor:
     - BF16: the MLA `attn_k_b`, `attn_v_b`, `attn_kv_a_mqa`, the indexer `attn_k` / `attn_q_b`, the KDA
       `ssm_f_a` / `ssm_f_b` / `ssm_g_a` / `ssm_g_b` / `ssm_beta`;
     - F32: the router, `hc_*_fn`, `indexer.proj`, `indexer_compressor_ape` / `_gate`, the conv weights.
5. **The ggml-org mtp-GLM files** (`mtp-GLM-5.3-Flash-{Q4_0,Q8_0}.gguf`), CONFIRMED from the headers:
   - `glm5-next`, `block_count` 46, `nextn_predict_layers` 1.
   - Only `blk.45`: 29 tensors, **names and shapes equal to unsloth's `blk.45`** (`nextn.shared_head_norm` included),
     7 432 592 416 params.
   - Plus `token_embd`, `output_norm`, `output`, the tokenizer and the HF template. Config 34 of 34 OK.
   - Loader (PROVISIONAL, by reading):
     - **NOT loadable at ec7630a** (`glm5-next.cpp:68`: no MTP-only handling; `:190`: the MTP graph throws);
     - **loadable at 79e2e74** (`glm5-next.cpp:67`, `src/llama-model.cpp:3491`, graph `glm5-next.cpp:188`).
   - D5: PASS (Q4_0 experts 4.5 bpw).
6. **MiMo listing** (bot `df9bea1`, CONFIRMED):

| repo | revision | files |
|---|---|---|
| ggml-org MOPD | `e6d4bce` | MXFP4 (2 shards, 155.87 GiB tensor shard), Q2_K, Q2_K_S, `mtp-` BF16 / Q4_0 / Q8_0, `dflash-` BF16 / Q8_0, mmproj |
| ggml-org RL | `b0b2d0f` | the same layout; its MXFP4 and Q2_K first shards have one LFS sha256 |
| QuantaPlanta MOPD MXFP4 | `ac33b37` | one MXFP4 file, 162.90 GiB |
| kernelpool MOPD MXFP4 | `44d9191` | one MXFP4 file, 157.40 GiB, a DFlash Q8_0, vision files |

   - The hub-parsed chat template of all four repos equals the vendored MOPD template: sha256 `11ea52e1…` of
     `git show origin/claude/n16-mimo-decode-parity:third_party/hf/mimo-v2.6-flash-chat/chat_template.jinja`.
   - Both unsloth names answer HTTP 401: absent or private, which one is UNKNOWN. The search (30 repos) lists no
     unsloth MiMo repo.
   - The single-file repos report 987 820 224 params more than ggml-org's. PROVISIONAL: the 3 NextN layers, which
     ggml-org ships as `mtp-` files. The headers will tell.
7. **MiMo header requests (round 3):**
   - the ggml-org MOPD MXFP4 trunk (2 shards) and its `mtp-…-Q8_0`;
   - QuantaPlanta's and kernelpool's single MXFP4 files.
   - Exact names and revisions from the listing. `--plan`: 4 pending, worst case 167 772 160 B (section 9).
   - The cross-check config is a byte-exact copy of the vendored MOPD config in `data/gguf-headers/mimo-v2.6-flash-config/`
     (bot commit `1ce4d32`, `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @`b12f5573350bb1704513fcf119fca48e498b0423`; N16 not merged).

## Result (round 1: tooling)

- **Fetcher:** a discover request can name repos (`"repos"`), listed at a pinned revision with size and LFS sha256. It
  reads no file.
- **Analyzer: mimo2.**
  - Config map after `conversion/mimo.py`. The KV-head array and the SWA pattern get + 1 SWA entry per MTP layer of
    the file; rope dims = `int(head_dim × factor)`; both bases; the value scale.
  - 12 graph-branch probes with file:line at both pins.
  - The tiny model `tools/tinygen/mimo2_tiny.py`. N16's richer variant is skipped and noted, because its generator is
    not on this branch.
- **Analyzer: all architectures.** Required keys per `--src` checkout; expert types per projection.
- **Analyzer: mode `mtp_files`.** Each file is its own model. It is compared with a reference NextN layer and the trunk
  KV, and gets a loader verdict per pin from `gguf_branches.MTP_LOAD` / `MTP_ATTACH`, whose anchors are checked at the
  pins.
- **Tests and mutation checks:** section 10. Commits: `git log` at the end of the report.

## 1. Inputs

| input | value | label |
|---|---|---|
| llama.cpp CI pin | `<S>/llama-ec7630a`, HEAD `ec7630a640789c393694fb194f1bbbf0369fc62d` | CONFIRMED (`git rev-parse HEAD`, analyzer `sources:` line) |
| llama.cpp R1 pin | `<S>/llama.cpp`, HEAD `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` | CONFIRMED (same) |
| ik_llama.cpp | `<S>/n12/ik-src`, HEAD `5194a9e` (grep for the shared-target key only) | CONFIRMED |
| requests, round 1 | `data/gguf-headers/{qwen3.8-flash-next-mtp, glm-5.3-flash-ggml-org-q4_k, glm-5.3-flash-ggml-org-mtp, discover-mimo}/REQUEST.json` (commit `fd532fb`) | CONFIRMED |
| Qwen MTP names | `third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl/PIN.json:77-102` (N15 listing) | CONFIRMED |
| ggml-org GLM revision | `third_party/gguf-headers/discover-glm-mimo/LISTING.json:861` `c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa` | CONFIRMED |
| MiMo config | `data/gguf-headers/mimo-v2.6-flash-config/config.json`, sha256 `61bea4a0…e621`, git blob `9e0d9673…02b4` = the vendored file of bot commit `1ce4d32` | CONFIRMED (`sha256sum`, `git hash-object`, `SOURCE.md`) |

## 2. The runner run (GitHub API, read-only)

```
$ gh api repos/samnolak/408032gb/actions/runs/38145613553 --jq '{status, conclusion, head_sha, event, created_at, updated_at, head_branch}'
{"conclusion":"success","created_at":"2026-10-11T14:03:06Z","event":"push","head_branch":"claude/n18-gguf-headers-mtp-mimo","head_sha":"8c0b42453ec58d19973a305e6294384f28a0c8bd","status":"completed","updated_at":"2026-10-11T14:05:04Z"}
$ gh api repos/samnolak/408032gb/actions/jobs/114488191911 --jq '.steps[] | "\(.number) \(.name) \(.conclusion) \(.started_at) \(.completed_at)"'
1 Set up job success 2026-10-11T14:03:10Z 2026-10-11T14:03:11Z
2 Run actions/checkout@v4 success 2026-10-11T14:03:11Z 2026-10-11T14:03:12Z
3 gguf-py of llama.cpp at the CI pin and numpy, PyYAML (for the self-test) success 2026-10-11T14:03:12Z 2026-10-11T14:03:21Z
4 self-test of the fetcher against a local Range server (no network; fails on any skip but llama-tokenize's) success 2026-10-11T14:03:21Z 2026-10-11T14:03:30Z
5 check the pending requests against the byte caps (fetches nothing; fails on a cap violation) success 2026-10-11T14:03:30Z 2026-10-11T14:03:30Z
6 read the GGUF headers with Range requests success 2026-10-11T14:03:30Z 2026-10-11T14:04:57Z
7 commit and push to this claude/* branch success 2026-10-11T14:04:57Z 2026-10-11T14:05:01Z
14 Post Run actions/checkout@v4 success 2026-10-11T14:05:01Z 2026-10-11T14:05:01Z
15 Complete job success 2026-10-11T14:05:01Z 2026-10-11T14:05:01Z
$ gh api repos/samnolak/408032gb/check-runs/114488191911/annotations --jq '.[] | "\(.annotation_level) | \(.message[0:330])"'
warning | Node.js 20 is deprecated. The following actions target Node.js 20 but are being forced to run on Node.js 24: actions/checkout@v4. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/
notice | headers commit df9bea130d509ed942e349f94d40785342774271
notice | unsloth/Qwen3.8-Flash-Next-GGUF:MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf: GGUF v3, arch 'qwen4exp', 51 KV, 34 tensors, header 10946688 B (sha256 4c7d482b072ecb8e), 16777216 B fetched in 7 Range requests; size check exact; chat_template c3cf9e34abf4f9e3
notice | unsloth/Qwen3.8-Flash-Next-GGUF:MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf: GGUF v3, arch 'qwen4exp', 51 KV, 34 tensors, header 10946688 B (sha256 77b6c9be7f4c1d6b), 16777216 B fetched in 7 Range requests; size check exact; chat_template c3cf9e34abf4f9e3
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/glm-5.3-flash-ggml-org-q4_k/REQUEST.json: status ok, revision c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa, 2 headers, 9692096 B fetched, 2498478 B committed (gzip)
notice | ggml-org/GLM-5.3-Flash-GGUF:GLM-5.3-Flash-Q4_K-00002-of-00002.gguf: GGUF v3, arch None, 3 KV, 1383 tensors, header 86720 B (sha256 afcd96c0576ab011), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | ggml-org/GLM-5.3-Flash-GGUF:GLM-5.3-Flash-Q4_K-00001-of-00002.gguf: GGUF v3, arch 'glm5-next', 68 KV, 0 tensors, header 9429951 B (sha256 c845390f795b286f), 9429952 B fetched in 7 Range requests; size check exact (no tensors: the writer's padding after the KV block); chat_template 0c4099f3382d6c92
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/glm-5.3-flash-ggml-org-mtp/REQUEST.json: status ok, revision c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa, 2 headers, 33554432 B fetched, 4975048 B committed (gzip)
notice | ggml-org/GLM-5.3-Flash-GGUF:mtp-GLM-5.3-Flash-Q8_0.gguf: GGUF v3, arch 'glm5-next', 66 KV, 32 tensors, header 9432000 B (sha256 5518d3c4d9de9c29), 16777216 B fetched in 7 Range requests; size check exact; chat_template 0c4099f3382d6c92
notice | ggml-org/GLM-5.3-Flash-GGUF:mtp-GLM-5.3-Flash-Q4_0.gguf: GGUF v3, arch 'glm5-next', 66 KV, 32 tensors, header 9432000 B (sha256 54c831d97a0dffcc), 16777216 B fetched in 7 Range requests; size check exact; chat_template 0c4099f3382d6c92
notice | discover 'MiMo-V2.6-Flash': 30 GGUF repos; architecture in []: 
notice | discover named repos: ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @ e6d4bced7393 (13 .gguf, mimo2), ggml-org/MiMo-V2.6-Flash-RL-GGUF @ b0b2d0fb0b56 (11 .gguf, mimo2), QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ ac33b3743d70 (4 .gguf, mimo2), kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ 44d9191bb7a9 (4 .gguf, mimo2), unsloth/MiMo-V2.6-
warning | discover unsloth/MiMo-V2.6-Flash-GGUF: HTTP 401 for the revision API; recorded
warning | discover unsloth/MiMo-V2.6-Flash-MOPD-GGUF: HTTP 401 for the revision API; recorded
```

As in N15, GitHub keeps at most 10 notice annotations per step (PROVISIONAL, its documented limit). Four of the Qwen
per-file notices and the Qwen closing notice are therefore missing; their numbers are in PIN.json (section 3).
`git fetch origin claude/n18-gguf-headers-mtp-mimo && git merge --ff-only origin/claude/n18-gguf-headers-mtp-mimo`
brought `df9bea1` (author github-actions[bot], 24 files).

## 3. Bytes fetched and committed, integrity

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py pins
== qwen3.8-flash-next-mtp ok unsloth/Qwen3.8-Flash-Next-GGUF 38bb39ee97821de2c9009abb7e93950eec396e66 -> 38bb39ee97821de2c9009abb7e93950eec396e66 gated False ['license:other'] fetched_utc 2026-10-11T14:04:01Z workflow_sha 8c0b42453ec58d19973a305e6294384f28a0c8bd
   bytes_fetched 100663296 committed_bytes (gz) 18509753 files in request 6 max_header_bytes 33554432
   mtp-Qwen3.8-Flash-Next-BF16.gguf: size 7770760320, lfs 8ac04a65440a, header 10946688, fetched 16777216 in 7, gz 3084942, size_check(PIN) 'exact', etag_ok True, commit_ok True
   mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf: size 2786204800, lfs b646ef60eaae, header 10946688, fetched 16777216 in 7, gz 3084972, size_check(PIN) 'exact', etag_ok True, commit_ok True
   mtp-Qwen3.8-Flash-Next-Q8_0.gguf: size 4137429120, lfs cd87e5d1a4da, header 10946688, fetched 16777216 in 7, gz 3084980, size_check(PIN) 'exact', etag_ok True, commit_ok True
   mtp-Qwen3.8-Flash-Next-shared-BF16.gguf: size 5227963456, lfs 32473fa74e8e, header 10946624, fetched 16777216 in 7, gz 3084937, size_check(PIN) 'exact', etag_ok True, commit_ok True
   mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf: size 1907151936, lfs f521868a9e14, header 10946624, fetched 16777216 in 7, gz 3084963, size_check(PIN) 'exact', etag_ok True, commit_ok True
   mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf: size 2786568256, lfs 5ff54097406a, header 10946624, fetched 16777216 in 7, gz 3084959, size_check(PIN) 'exact', etag_ok True, commit_ok True
== glm-5.3-flash-ggml-org-q4_k ok ggml-org/GLM-5.3-Flash-GGUF c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa -> c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa gated False ['license:other'] fetched_utc 2026-10-11T14:03:51Z workflow_sha 8c0b42453ec58d19973a305e6294384f28a0c8bd
   bytes_fetched 9692096 committed_bytes (gz) 2498478 files in request 2 max_header_bytes 33554432
   GLM-5.3-Flash-Q4_K-00001-of-00002.gguf: size 9429952, lfs 7d658490f34f, header 9429951, fetched 9429952 in 7, gz 2486966, size_check(PIN) "exact (no tensors: the writer's padding after the KV block)", etag_ok True, commit_ok True
   GLM-5.3-Flash-Q4_K-00002-of-00002.gguf: size 181363033792, lfs 27636dd16f76, header 86720, fetched 262144 in 1, gz 11512, size_check(PIN) 'exact', etag_ok True, commit_ok True
== glm-5.3-flash-ggml-org-mtp ok ggml-org/GLM-5.3-Flash-GGUF c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa -> c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa gated False ['license:other'] fetched_utc 2026-10-11T14:03:32Z workflow_sha 8c0b42453ec58d19973a305e6294384f28a0c8bd
   bytes_fetched 33554432 committed_bytes (gz) 4975048 files in request 2 max_header_bytes 33554432
   mtp-GLM-5.3-Flash-Q4_0.gguf: size 4947287104, lfs 7a36bad56b6b, header 9432000, fetched 16777216 in 7, gz 2487517, size_check(PIN) 'exact', etag_ok True, commit_ok True
   mtp-GLM-5.3-Flash-Q8_0.gguf: size 9284197440, lfs 17a89535ce31, header 9432000, fetched 16777216 in 7, gz 2487531, size_check(PIN) 'exact', etag_ok True, commit_ok True
== discover-mimo fetched_utc 2026-10-11T14:03:30Z request data/gguf-headers/discover-mimo/REQUEST.json sha256 f0bf3de0d878
exit 0
```

The analyzer re-derives each summary from its committed header (appendix A, the `integrity:` lines): OK for all 10.
Size checks: the 7 files with tensors are exact; the metadata-only first shard is data offset + 1 padding byte, and
its whole-file sha256 matches:

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py metashard
GLM-5.3-Flash-Q4_K-00001-of-00002.gguf: header 9429951 + 1 zero bytes = 9429952 (file 9429952)
   sha256(header + zeros) 7d658490f34f56ac9eb3b46ceb4778599ec39f4e1a77c63e456a1240efc64ed3
   PIN.json lfs_sha256    7d658490f34f56ac9eb3b46ceb4778599ec39f4e1a77c63e456a1240efc64ed3
   equal: True
exit 0
```

- The fetcher's doubling read 16 777 216 B for each header of 9.4–10.9 MB. The surplus past the headers is tensor data,
  discarded and never committed:
  - Qwen MTP: 6 × (16 777 216 − 10 946 688 or 10 946 624);
  - mtp-GLM: 2 × (16 777 216 − 9 432 000).
- Committed (gzip): Qwen MTP 18 509 753 B, ggml-org Q4_K 2 498 478 B, mtp-GLM 4 975 048 B, plus summaries, PIN.json and
  the MiMo LISTING.json.
- Round 1 expected "about 100–130 MB" fetched. Measured: 143 909 824 B; correction in section 12.

## 4. The Qwen MTP files

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py qwenmtp
vendored HF template sha256 c3cf9e34abf4f9e3, UD-Q4_K_XL embedded 12827f24b742ea4e
mtp-Qwen3.8-Flash-Next-BF16.gguf: 34 tensors, 51 KV, general.name 'Ckpt_Q38', size_label '512x1.4B', file_type 32, nextn_shared_target_tensors None, compress_ratios len 49 [48] = 0, non-layer {'token_embd.weight': ('BF16', [2560, 248320], 1271398400), 'output.weight': ('BF16', [2560, 248320], 1271398400)}, template c3cf9e34abf4f9e3 (= vendored: True)
mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf: 34 tensors, 51 KV, general.name 'Ckpt_Q38', size_label '512x1.4B', file_type 15, nextn_shared_target_tensors None, compress_ratios len 49 [48] = 0, non-layer {'output.weight': ('Q6_K', [2560, 248320], 521472000), 'token_embd.weight': ('Q4_K', [2560, 248320], 357580800)}, template c3cf9e34abf4f9e3 (= vendored: True)
mtp-Qwen3.8-Flash-Next-Q8_0.gguf: 34 tensors, 51 KV, general.name 'Ckpt_Q38', size_label '512x1.4B', file_type 7, nextn_shared_target_tensors None, compress_ratios len 49 [48] = 0, non-layer {'output.weight': ('Q8_0', [2560, 248320], 675430400), 'token_embd.weight': ('Q8_0', [2560, 248320], 675430400)}, template c3cf9e34abf4f9e3 (= vendored: True)
mtp-Qwen3.8-Flash-Next-shared-BF16.gguf: 32 tensors, 52 KV, general.name 'Ckpt_Q38', size_label '512x95M', file_type 32, nextn_shared_target_tensors True, compress_ratios len 49 [48] = 0, non-layer none, template c3cf9e34abf4f9e3 (= vendored: True)
mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf: 32 tensors, 52 KV, general.name 'Ckpt_Q38', size_label '512x95M', file_type 15, nextn_shared_target_tensors True, compress_ratios len 49 [48] = 0, non-layer none, template c3cf9e34abf4f9e3 (= vendored: True)
mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf: 32 tensors, 52 KV, general.name 'Ckpt_Q38', size_label '512x95M', file_type 7, nextn_shared_target_tensors True, compress_ratios len 49 [48] = 0, non-layer none, template c3cf9e34abf4f9e3 (= vendored: True)
BF16: plain-only tensors ['output.weight', 'token_embd.weight'], shared-only [], common shapes equal True, common types differ []; KV plain-only [], shared-only ['qwen4exp.nextn_shared_target_tensors']; file sizes 7770760320 - 5227963456 = 2542796864 B, plain-only tensor bytes 2542796800 (+ header difference 64 B)
Q4_K_M: plain-only tensors ['output.weight', 'token_embd.weight'], shared-only [], common shapes equal True, common types differ []; KV plain-only [], shared-only ['qwen4exp.nextn_shared_target_tensors']; file sizes 2786204800 - 1907151936 = 879052864 B, plain-only tensor bytes 879052800 (+ header difference 64 B)
Q8_0: plain-only tensors ['output.weight', 'token_embd.weight'], shared-only [], common shapes equal True, common types differ []; KV plain-only [], shared-only ['qwen4exp.nextn_shared_target_tensors']; file sizes 4137429120 - 2786568256 = 1350860864 B, plain-only tensor bytes 1350860800 (+ header difference 64 B)
exit 0
```

From these lines and the analyzer (appendix A, `qwen3.8-flash-next-mtp`):

- **Layout**, CONFIRMED:
  - `MTP-only file`: `blk.48` only; `blk.0.hc_attn_norm` (the qwen4exp trunk probe) absent.
  - `block_count` 49 = the 48 trunk layers + 1, `nextn_predict_layers` 1.
  - The NextN layer's 32 tensor names equal the tiny `qwen4exp --mtp` NextN layer: "missing [], missing optional [],
    extra []". That is the tiny model of `tests/test_qwen4exp_mtp_parity.py`, which N15 used as "loads at both pins".
  - Split `attn_q/k/v`, with the QSA indexer tensors (`indexer.{q,k}_proj`, `{q,k}_norm`), hc mixers, the shared
    expert, and `nextn.{eh_proj, enorm, hnorm, hc_head_norm, hc_head_down, hc_head_up}`.
- **KV**, CONFIRMED:
  - Equal to the UD-Q4_K_XL trunk except `block_count`, `nextn_predict_layers`, the 49-entry `compress_ratios`, and the
    PLE keys. All 10 PLE keys of the trunk are absent (9 `ple.*` + `embedding_length_per_layer_input`), as
    `conversion/qwen4exp.py:95` writes an MTP-only file at both pins. The `-shared-` files add
    `nextn_shared_target_tensors`.
  - Required keys: 16 of 16 present at both pins. The 9 PLE keys are only read inside `if (n_ple > 0)`.
  - The token list is equal to the trunk's (`token list equal: True`). `general.name` is `'Ckpt_Q38'`; the trunk's is
    `Qwen3.8 Flash Next`.
- **`-shared-`**, CONFIRMED:
  - Exactly `token_embd.weight` and `output.weight` are missing; the file sizes differ by their bytes + 64 B of header.
  - Plus `qwen4exp.nextn_shared_target_tensors = true`, a key no checkout reads:

```
$ for r in llama-ec7630a llama.cpp n12/ik-src; do echo "== $r $(git -C <S>/$r rev-parse --short HEAD)"; grep -rn "shared_target\|SHARED_TARGET" <S>/$r/src <S>/$r/common <S>/$r/gguf-py <S>/$r/conversion; done
== llama-ec7630a ec7630a
== llama.cpp 79e2e74
== n12/ik-src 5194a9e
```

- **Loader** (PROVISIONAL, by reading; appendix A `loader [...]`):

| file | ec7630a | 79e2e74 |
|---|---|---|
| plain BF16 / Q4_K_M / Q8_0 | loadable as a separate MTP model (detect `qwen4exp.cpp:179`, graph `:307`) | loadable (detect `:178`, graph `:306`) |
| `-shared-` BF16 / Q4_K_M / Q8_0 | NOT: `token_embd.weight` absent, created with flags 0 (`qwen4exp.cpp:183`) | NOT (`:182`) |

  - The `-shared-` files presumably target a runtime that takes the embedding and head from the trunk. For
    qwen4exp neither pin does that: only EAGLE3 and DFlash borrow them through `ctx_other`
    (`src/llama-context.cpp:156@ec7630a`, `:159@79e2e74`).
- **NextN compress ratio 0**, CONFIRMED in all six files: `compress_ratios len 49 [48] = 0`.
  - The converter at both pins appends `[ratio] * (block_count - n_layer)` (`conversion/qwen4exp.py:85-89`, comment
    "the MTP block is a full-attention QSA layer too"). It would write 4 here, as for the 12 QSA trunk layers.
  - The MTP graph calls the common attention builder (`graph_mtp` `src/models/qwen4exp.cpp:593@ec7630a` /
    `:590@79e2e74`). That builder selects QSA only when the layer's ratio is > 0 (`:942` / `:939`, `const bool qsa =
    inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;`).
  - So with these files the NextN layer runs dense full attention (PROVISIONAL, by reading). The two differ only when
    more cells are visible than the QSA selection (budget 2048).
  - Whether the HF model's MTP layer uses QSA is PROVISIONAL: llama.cpp's converter comment says so; the HF modelling
    code is not vendored. The effect on draft acceptance is UNKNOWN.
- **Template**, CONFIRMED: all six embed the vendored HF template `c3cf9e34…`, not the fixed one of the UD-Q4_K_XL
  trunk (`12827f24…`, N15). A draft model's template is not used for rendering; the server renders with the target's
  (PROVISIONAL, not traced).
- **D5**, CONFIRMED: PASS for all six.
  - Q4_K_M: routed experts Q4_K + Q8_0, 5.833 bpw; `token_embd` Q4_K 4.500; `output` Q6_K; `indexer.{q,k}_proj` BF16.
  - BF16 / Q8_0: 16 / 8.5 bpw.

## 5. Separate MTP files at the two pins: how they are attached, what they must contain

The analyzer prints the attach path per pin (appendix A, "how llama.cpp attaches a separate MTP file"). Every line is
located at run time; BranchAnchors checks each anchor at both pins. In short, the same at both pins:

- **Local file.** `-md FILE` (`common/arg.cpp:4241` / `:4276`) together with `--spec-type draft-mtp` (`:4249` / `:4284`;
  name `common/speculative.cpp:37` / `:76`).
  - With a draft path the file is loaded as its own `llama_model` (`common/speculative.cpp:2539` / `:2623`) with an MTP
    context (`:2526` / `:2610`).
  - Without one, the MTP context runs on the target's own NextN layers (`:2560` / `:2644`).
  - The MTP model's output width must equal the target's (`:1376` / `:1437`). The target is switched to emit the NextN
    input (`llama_set_embeddings_nextn(ctx_tgt, true, …)`, `common/speculative.cpp:1417@ec7630a`, `:1478@79e2e74`).
  - The qwen4exp trunk builds `h_nextn` from its last hc state whether or not it has a NextN layer
    (`src/models/qwen4exp.cpp:502-505@ec7630a` under `embeddings_nextn`, `:501-503@79e2e74` always).
- **`-hf REPO:TAG --mtp`** (download example, `common/arg.cpp:3084` / `:3097`).
  - It picks the sibling whose name contains `mtp-` (`common/download.cpp:616` / `:605`) through `find_best_sibling`
    (`:545` / `:534`, identical text at both pins).
  - That function only accepts a sibling whose directory is the model's directory or one of its parents (`std::mismatch`
    of the path parts, `:579` / `:568`). Then it prefers the exact tag, then the closest quantization bits.
  - So for `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` (model in `UD-Q4_K_XL/`) the `MTP/mtp-*.gguf` files are **not**
    found (PROVISIONAL, by reading): a local `-md` is the way.
  - For `ggml-org/GLM-5.3-Flash-GGUF:Q4_K` (all files at the root) it would pick `mtp-GLM-5.3-Flash-Q4_0.gguf` (4 bits,
    tag 4) (PROVISIONAL).
- **What the file must contain**, by arch loader (`gguf_branches.MTP_LOAD`; the verdicts above and in section 7 apply
  it):

| arch | MTP-only file recognised | required non-layer tensors | NextN graph |
|---|---|---|---|
| glm5-next | 79e2e74 only (`glm5-next.cpp:67`, `src/llama-model.cpp:3491`); ec7630a has no handling (`glm5-next.cpp:68`) | `token_embd`, `output_norm`, `output` (`:74,76,77@ec7630a`, `:71,73,74@79e2e74`) | 79e2e74 (`:188`); ec7630a throws (`:190`) |
| qwen4exp | both (`qwen4exp.cpp:179` / `:178`) | `token_embd` (`:183` / `:182`); `output` falls back to it; `per_layer_token_embd` if `ple.layers` is set (`:214` / `:213`) | both (`:307` / `:306`) |
| mimo2 | both (`mimo2.cpp:28` at both, different code) | `token_embd`, `output_norm`, `output` (`:38,41,42` / `:32,35,36`) | both |

  Common to all three, at both pins:
  - the tokenizer (`src/llama-vocab.cpp:1944` / `:2057`: `tokenizer.ggml.model` is required);
  - every key `load_hparams` / `load_arch_hparams` read without a default;
  - no tensor the loader does not create (`src/llama-model-loader.cpp:1398`: "wrong number of tensors").

  The mainline converter's `--mtp` export writes exactly the embeddings, the final norm, the head and the NextN layers
  into `mtp-<name>.gguf` (`conversion/glm.py:450-451, 591`; `conversion/mimo.py:264-265, 285`). That is the layout of
  the ggml-org mtp-GLM files and of the plain Unsloth Qwen ones.

## 6. The ggml-org GLM-5.3-Flash Q4_K trunk

Analyzer, appendix A (`glm-5.3-flash-ggml-org-q4_k`):

- **(e) identity:** `glm5-next`, `gpt2` / `glm4`, 154 880 tokens, the same special token ids as unsloth's.
- **(d) groups:** 1 / 1, no group step.
- **(c) template:** BYTE-EQUAL to `third_party/hf/glm-5.3-flash-chat/chat_template.jinja` (10 950 B, `0c4099f3…`). So
  N14's finding 1(a) applies to this file as to the vendored template: llama-server @79e2e74 carries `reasoning_content`
  into the next assistant turn (N14, CONFIRMED there for the vendored template). The unsloth UD-Q4_K_XL embeds a
  rewritten template (N15 7.5).
- **(b) config:** 33 of 34 OK; `num_nextn_predict_layers` ABSENT (trunk only).
  - New against unsloth's file: `attention.indexer.types` is written (45 × true) and `kpool_select_tail` = true. Both
    equal the loader's defaults (`glm5-next.cpp:46-47`, fill 1 then the optional array; `src/llama-hparams.h:286@ec7630a` /
    `:289@79e2e74`, tail true), so the graph is the same (PROVISIONAL, by reading).
  - Probes not exercised by the tiny models:
    - `glm.indexer_types`: the array is read instead of the default fill;
    - `glm.shexp_clamp`, `glm.layer_norm_eps`: as on unsloth's file.
  - `glm.mtp_heads`: "trunk embedding, head and norm", since there is no NextN layer.
- **Keys:** 20 of 20 required present at both pins.
- **Experts:** `ffn_{gate,up,down}_exps` Q4_K in all 42 MoE layers.
- **(a)+(f) D5:** PASS, 313.33 B params, 168.91 GiB, 4.631 bpw; routed experts 100 % Q4_K at 4.500 bpw.
  - Non-Q8_0 non-expert matrices (all ≥ 16 bpw):

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py matrices
    11 x attn_k_b.weight BF16 [256, 512, 64]
    11 x attn_kv_a_mqa.weight BF16 [4096, 512]
    11 x attn_v_b.weight BF16 [512, 256, 64]
    42 x ffn_gate_inp.weight F32 [4096, 288]
    45 x hc_attn_fn.weight F32 [16384, 24]
    45 x hc_ffn_fn.weight F32 [16384, 24]
    11 x indexer.attn_k.weight BF16 [4096, 128]
    11 x indexer.attn_q_b.weight BF16 [1536, 4096]
    11 x indexer.proj.weight F32 [4096, 32]
    11 x indexer_compressor_ape.weight F32 [128, 4]
    11 x indexer_compressor_gate.weight F32 [4096, 128]
    34 x ssm_beta.weight BF16 [4096, 64]
    34 x ssm_conv1d_k.weight F32 [4, 1, 8192]
    34 x ssm_conv1d_q.weight F32 [4, 1, 8192]
    34 x ssm_conv1d_v.weight F32 [4, 1, 8192]
    34 x ssm_f_a.weight BF16 [4096, 128]
    34 x ssm_f_b.weight BF16 [128, 8192]
    34 x ssm_g_a.weight BF16 [4096, 128]
    34 x ssm_g_b.weight BF16 [128, 8192]
exit 0
```

## 7. The ggml-org mtp-GLM files against unsloth's NextN layer

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py glmmtp
ggml-org Q4_K trunk: block_count 45, nextn_predict_layers None, layers with tensors [42, 43, 44] (last 3), nextn tensors [], tensors 1383
unsloth UD-Q4_K_XL blk.45: 29 tensors
mtp-GLM-5.3-Flash-Q4_0.gguf: blk.45 29 tensors, names equal to unsloth blk.45: True (only here [], only unsloth []), shapes equal True, types {'BF16': 5, 'F32': 14, 'Q4_0': 10}, params blk.45 7432592416, all tensors 8701373472
mtp-GLM-5.3-Flash-Q8_0.gguf: blk.45 29 tensors, names equal to unsloth blk.45: True (only here [], only unsloth []), shapes equal True, types {'BF16': 5, 'F32': 14, 'Q8_0': 10}, params blk.45 7432592416, all tensors 8701373472
unsloth blk.45 params 7432592416; unsloth all 320759404382 - ggml-org trunk 313326811966 = 7432592416
exit 0
```

- **Content**, CONFIRMED (appendix A, `glm-5.3-flash-ggml-org-mtp`):
  - Both files hold `blk.45` only, with the same 29 tensor names and shapes as `blk.45` of the unsloth UD-Q4_K_XL,
    `nextn.shared_head_norm` included.
  - Types: 5 BF16 (MLA `k_b`, `v_b`, `kv_a_mqa`, indexer `attn_k`, `attn_q_b`), 14 F32, 10 Q4_0 or Q8_0.
  - Plus `token_embd`, `output_norm`, `output` (Q4_0 or Q8_0), the full tokenizer and the HF template.
  - KV: equal to the trunk's except `block_count` 46, `nextn_predict_layers` 1, and the per-layer arrays one entry
    longer. Config 34 of 34 OK.
  - The NextN layer's 7 432 592 416 params are exactly the difference between the unsloth file and the ggml-org trunk.
    So ggml-org's Q4_K trunk + one of these files = the unsloth model's tensor set, CONFIRMED by names and shapes; the
    quantization differs.
- **Loader** (PROVISIONAL, by reading):
  - ec7630a: NOT loadable (no MTP-only handling, `glm5-next.cpp:68`; the NextN graph throws, `:190`). With ec7630a,
    GLM MTP is not possible with any file.
  - 79e2e74: loadable (detect `glm5-next.cpp:67` + `src/llama-model.cpp:3491`; graph `:188`). The head norm is the
    file's `nextn.shared_head_norm` (N15 7.6, `glm5-next.cpp:647@79e2e74`).
- **D5:** PASS (Q4_0: experts 4.5 bpw, embedding and head 4.5 bpw).

## 8. The MiMo listing

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py listing
ggml-org/MiMo-V2.6-Flash-MOPD-GGUF: ok, revision e6d4bced739310a721eefad4c247ec7e0247c89d, http None, arch mimo2, total params 308778780864, template sha256 11ea52e156de38a4, lastModified 2026-10-11T07:58:04.000Z
    MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf  5948640  a353e75ccd30f347dfd68a8cf1be8be0fb989fcca94b8e72fd7d1e35241b5f51
    MiMo-V2.6-Flash-MOPD-MXFP4-00002-of-00002.gguf  167363516512  3adf8ea80157dd066945a954428ba0e507c15c0e5a8b6b276ac92df0948698bc
    MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf  5948928  612ae7fd2064a2ba296b72cc9a0765e5ebd91541996e54718b6f06b883140cfe
    MiMo-V2.6-Flash-MOPD-Q2_K-00002-of-00002.gguf  126360000608  637921c1b5b556f422c548c2b2ff96384baaeb86a86c2081a5c925bfaedf5f70
    MiMo-V2.6-Flash-MOPD-Q2_K_S-00001-of-00002.gguf  5948928  ad88631792326298ea2b4d828eafa11daa21d0c754e73f7cb1f854413371cf3b
    MiMo-V2.6-Flash-MOPD-Q2_K_S-00002-of-00002.gguf  105858242656  4f21f6083146630b2b633a8a68193c97817e2d102c30e7d92c8a7c8a6fd6949a
    dflash-MiMo-V2.6-Flash-MOPD-BF16.gguf  5441972448  c909b7c0d0f10061a212fba30f9c0d7fb6261ea293a5ebd3558f4badacb56b5d
    dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  2893932768  cfd70fef40bc4ef2f6c186516bdfa8c861ebd7b8e5f44ed4703a41dde1492a67
    mmproj-MiMo-V2.6-Flash-MOPD-BF16.gguf  2748509792  9b7d98d8afe2a6ae25635ecc490fb37883f912bf1d5648d1334fdc703e4a565e
    mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  1564438112  7fa58c0f20342fad37ef1962e8a9fd37ef834038b377e4f7e93f905e013e55db
    mtp-MiMo-V2.6-Flash-MOPD-BF16.gguf  4481536224  d4292135824f47b8f8e1219780ed53a50ad9c3fb520d5670db7a9c6760c32e20
    mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf  1264898272  bc9914785441a04596e84d919b59aed352336eda7fe323f1ce1eecf15422ba71
    mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  2383728864  4589839383ca7524fa1bdf635b8cef0da59b7e909d84a433f6006f7b64fc17ec
ggml-org/MiMo-V2.6-Flash-RL-GGUF: ok, revision b0b2d0fb0b56611c761e075425c88989faf1ed1f, http None, arch mimo2, total params 308778780864, template sha256 11ea52e156de38a4, lastModified 2026-10-11T08:23:14.000Z
    MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf  5948640  a34f63df32086bba7a14aa8a2f6c2f111f6e626ad7b10be18957e42d14d84738
    MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf  167363516512  10b74ff85577de18b336718c19905a50bcbccfee9cae7b9edd446667084f623c
    MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf  5948640  a34f63df32086bba7a14aa8a2f6c2f111f6e626ad7b10be18957e42d14d84738
    MiMo-V2.6-Flash-RL-Q2_K-00002-of-00002.gguf  126360000608  d037fe34f375e55945dbb981ae6bf4bd357cb19ee8c04f56fbd7c4dde4cdbd63
    dflash-MiMo-V2.6-Flash-RL-BF16.gguf  5441972448  02bac7c24be20ee5ca82a3cd368a1d10aa23706ebf6374e572725e79191ffa2e
    dflash-MiMo-V2.6-Flash-RL-Q8_0.gguf  2893932768  54e787fb5c89be3d94c7e7a838e40bf46400db42b99c977f8321e5df954115de
    mmproj-MiMo-V2.6-Flash-RL-BF16.gguf  2748509792  1561e6127de6c1aa0d11b50c95205b65dbc6e4a4b8a6bf899cac1f7471fa0414
    mmproj-MiMo-V2.6-Flash-RL-Q8_0.gguf  1564438112  9f3aeb85745e89d3f57af792f3d90704bdf8b4a2340ac5c2abb74ba6498490f4
    mtp-MiMo-V2.6-Flash-RL-BF16.gguf  4481536224  4719f59c59079fe458da67640719b6a463731c6facb620d7f06e7feb9a7f1441
    mtp-MiMo-V2.6-Flash-RL-Q4_0.gguf  1264898272  5318c079bf646ed702271281609abc5b7cfc7fecdd354f42b4296b9aad4a83d7
    mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf  2383728864  cbcd8a22e62754b4133dc6f9ba87356268aa2d6377ee3e78be5d61172643969e
QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF: ok, revision ac33b3743d70ed57a8cc6b847698d2ebbddb7784, http None, arch mimo2, total params 309766601088, template sha256 11ea52e156de38a4, lastModified 2026-09-27T20:57:35.000Z
    MiMo-V2.6-Flash-MOPD-MXFP4.gguf  174908254272  fc1b8122d1a31db98b20ed5c9ddb3d4d7e28d37b47cb2f1e843354788fbfeaa6
    mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  1564438112  156c3891b52d831ca9db2329cf5b2c8dd8ecdab8758354d42a912df79bb247ac
    mmproj-MiMo-V2.6-Flash-MOPD-bf16.gguf  2748509792  8524ec7a6ce44be56ee01921749161e5dbf1c5dd464e2d5c95da505e4b7287d4
    mmproj-MiMo-V2.6-Flash-MOPD-f16.gguf  2744577632  973931512a704ff99c4e7e7bdd2bf04f4540dce699cb8a51feb4c3683d8700dd
kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF: ok, revision 44d9191bb7a95125c8b971ce0593bd802ae6dd48, http None, arch mimo2, total params 309766601088, template sha256 11ea52e156de38a4, lastModified 2026-10-04T03:08:13.000Z
    MiMo-V2.6-Flash-MOPD-DFlash-Q8_0.gguf  1559980992  336b16e8f7ea6f6540fe9993d6ed053f47cca6220285f79f9a320a6a95dadd75
    MiMo-V2.6-Flash-MOPD-MXFP4.gguf  169005098944  5e5dfb6d96466f40e73a235467949aa836e14d4de2310cff220409923937c99b
    MiMo-V2.6-Flash-MOPD-Vision-F32.gguf  2914399616  5eaa8193dcfc090b8aa4571a7f4bb5b5a84b163c1380a78793461e84044804c2
    MiMo-V2.6-Flash-MOPD-Vision-Q8_0.gguf  775591296  4499a8191bf53d04abb30b13ee1301945863a6faee81a8cd45b687b54a255691
unsloth/MiMo-V2.6-Flash-MOPD-GGUF: repo_unavailable, revision None, http 401, arch None, total params None, template sha256 -, lastModified None
unsloth/MiMo-V2.6-Flash-GGUF: repo_unavailable, revision None, http 401, arch None, total params None, template sha256 -, lastModified None
same LFS sha256 a34f63df32086bba: ['ggml-org/MiMo-V2.6-Flash-RL-GGUF:MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf', 'ggml-org/MiMo-V2.6-Flash-RL-GGUF:MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf']
search 'MiMo-V2.6-Flash': 30 repos, unsloth among them: []
exit 0
```

- The four named repos answer with a pinned revision; their files are listed in appendix A (`discover-mimo`).
- The hub-parsed chat template of all four is the vendored MOPD template, CONFIRMED:

```
$ git show origin/claude/n16-mimo-decode-parity:third_party/hf/mimo-v2.6-flash-chat/chat_template.jinja | sha256sum
11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059  -
```

- The ggml-org repos ship the trunk without the NextN layers plus `mtp-` sidecars (BF16 / Q4_0 / Q8_0), and `dflash-`
  sidecars. The single-file MXFP4 repos report 987 820 224 params more. PROVISIONAL: the three NextN layers; the
  headers of round 3 will show.
- In the RL repo, `MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf` and `…-Q2_K-00001-of-00002.gguf` have the same LFS
  sha256 (`a34f63df…`), so the same metadata shard, including `general.file_type`. CONFIRMED that the bytes are equal;
  why is UNKNOWN. The MOPD repo's two first shards differ.
- unsloth: HTTP 401 for both names (absent or private: UNKNOWN which), and none in the 30 search results.

## 9. MiMo header requests (round 3) and the plan

Chosen: at most 3 repos, all the MOPD variant (the vendored config's), the MXFP4 quant each, and the `mtp-` sidecar
where the repo has one (only ggml-org). Exact names and revisions from `LISTING.json` (bot `df9bea1`):

| request | repo @ revision | files | analysis |
|---|---|---|---|
| `mimo-v2.6-flash-mopd-ggml-org-mxfp4` | `ggml-org/MiMo-V2.6-Flash-MOPD-GGUF` @`e6d4bced739310a721eefad4c247ec7e0247c89d` | `MiMo-V2.6-Flash-MOPD-MXFP4-0000?-of-00002.gguf` (LISTING.json:428, 433) | mimo2, tiny, config |
| `mimo-v2.6-flash-mopd-ggml-org-mtp` | same | `mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf` (:488) | mtp_files; reference = QuantaPlanta's first NextN layer; trunk = the request above |
| `mimo-v2.6-flash-mopd-quantaplanta-mxfp4` | `QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF` @`ac33b3743d70ed57a8cc6b847698d2ebbddb7784` | `MiMo-V2.6-Flash-MOPD-MXFP4.gguf` (:574) | mimo2, tiny, config |
| `mimo-v2.6-flash-mopd-kernelpool-mxfp4` | `kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF` @`44d9191bb7a95125c8b971ce0593bd802ae6dd48` | `MiMo-V2.6-Flash-MOPD-MXFP4.gguf` (:612) | mimo2, tiny, config |

```
$ python3 scripts/gguf_headers_fetch.py --plan
/home/user/wt-n18/data/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mtp/REQUEST.json: ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @ e6d4bced739310a721eefad4c247ec7e0247c89d, globs ['mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf'], 1 files x cap 33554432 B = at most 33554432 B
/home/user/wt-n18/data/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mxfp4/REQUEST.json: ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @ e6d4bced739310a721eefad4c247ec7e0247c89d, globs ['MiMo-V2.6-Flash-MOPD-MXFP4-0000?-of-00002.gguf'], 2 files x cap 33554432 B = at most 67108864 B
/home/user/wt-n18/data/gguf-headers/mimo-v2.6-flash-mopd-kernelpool-mxfp4/REQUEST.json: kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ 44d9191bb7a95125c8b971ce0593bd802ae6dd48, globs ['MiMo-V2.6-Flash-MOPD-MXFP4.gguf'], 1 files x cap 33554432 B = at most 33554432 B
/home/user/wt-n18/data/gguf-headers/mimo-v2.6-flash-mopd-quantaplanta-mxfp4/REQUEST.json: QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ ac33b3743d70ed57a8cc6b847698d2ebbddb7784, globs ['MiMo-V2.6-Flash-MOPD-MXFP4.gguf'], 1 files x cap 33554432 B = at most 33554432 B
pending requests: 4; worst-case download 167772160 B (per-file hard cap 67108864 B, per-run cap 1073741824 B)
exit 0
```

A dry run of these four requests against the offline fake hub (synthetic mimo2 contents built with the test helpers,
exact names and revisions) fetched and analysed all of them: the config path, the tiny mimo2 build, the reference
request and the trunk request resolve. `tests/test_gguf_headers_committed.py`
`CommittedN18.test_mimo_listing` checks that every MiMo request names a revision and exactly `expect_count` files of
this listing.

## 10. Tests and mutation checks

Round 1, at `8c0b424` (reported to the lead then):
- full suite with gguf-py: `Ran 907 tests … OK (skipped=358)`;
- hooks + check_unittest: `Ran 54 tests … OK`;
- both gguf-headers modules with every optional environment: `Ran 47 … OK`, 47 ok, 0 skipped;
- the workflow's self-test filter passes;
- N18 mutations: 20 / 20 killed;
- N15 mutations on this branch: 19 / 19 killed.

Round 2, at `ecd05be` (code HEAD; later commits only add this document):

```
$ export TMPDIR=<S>/n18/tmp
$ cd /home/user/wt-n18 && git log --oneline -n 1 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > <S>/n18/full-suite-ecd05be.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n18/full-suite-ecd05be.log
ecd05be [model-analyst] tests: facts of the N18 headers (bot commit df9bea1): Qwen MTP files, ggml-org GLM trunk + mtp files, MiMo listing
exit 0
Ran 912 tests in 184.326s
OK (skipped=358)
$ python3 -m unittest tests.test_hooks tests.test_check_unittest 2>&1 | tail -3
Ran 54 tests in 18.652s

OK
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py LLAMA_CPP_SRC_79E2E74=<S>/llama.cpp LLAMA_TOKENIZE_BIN=<S>/n15/build-tok/bin/llama-tokenize python3 -m unittest tests.test_gguf_headers tests.test_gguf_headers_committed -v > <S>/n18/modules-ecd05be.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n18/modules-ecd05be.log; grep -c " \.\.\. ok$" <S>/n18/modules-ecd05be.log; grep -c "skipped" <S>/n18/modules-ecd05be.log
exit 0
Ran 52 tests in 35.405s
OK
52
0
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest tests.test_gguf_headers -v > <S>/n18/selftest-ecd05be.log 2>&1   (the workflow's self-test filter: no skip but llama-tokenize's)
self-test unittest rc=0
Ran 39 tests in 19.290s
OK (skipped=1)
step passes
```

912 = 907 + the 5 tests of `CommittedN18`. Skips are unchanged (358 with gguf-py): the new pin-dependent test runs with
the CI pin's checkout. Round-2 code changes:
- `239e8c0`: section (c) without a vendored template;
- `ecd05be`: committed-header tests.

N18 mutation check at `ecd05be`:

```
$ cd <S>/n18 && git -C /home/user/wt-n18 log --oneline -n 1 && bash /home/user/wt-n18/docs/evidence/mutation-scripts/2026-10-11-n18-mutations.sh > mutations-ecd05be.log 2>&1; echo "exit $?"; cat mutations-ecd05be.log
ecd05be [model-analyst] tests: facts of the N18 headers (bot commit df9bea1): Qwen MTP files, ggml-org GLM trunk + mtp files, MiMo listing
exit 0
HEAD ecd05be
base exit 0 (Ran 39 tests in 18.502s, OK (skipped=1))
M20-named-revision-unchecked KILLED  (Fetch.test_discover_named_repo_revision_is_pinned_and_requests_are_checked )
M21-named-listed-again-by-search KILLED  (Fetch.test_discover_lists_named_repos_and_reads_no_file )
M22-named-unavailable-raises KILLED  (Fetch.test_discover_lists_named_repos_and_reads_no_file N18Analyzer.test_listing_renders_named_repos )
M23-duplicate-repos-accepted KILLED  (Fetch.test_discover_named_repo_revision_is_pinned_and_requests_are_checked )
M24-optional-counted-missing KILLED  (N18Analyzer.test_reference_diff_takes_fused_and_split_forms_as_equal N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M25-no-fused-split-alternatives KILLED  (N18Analyzer.test_reference_diff_takes_fused_and_split_forms_as_equal )
M26-globals-unchecked KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M27-no-mtp-only-handling-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M28-missing-graph-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M29-tokenizer-unchecked KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M30-extra-tensors-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M31-lax-key-regex KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M32-conditional-generic-keys-required KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M33-pattern-without-mtp-layers KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M34-kv-array-without-mtp-layers KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M35-sinks-label-inverted KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit N18Analyzer.test_mimo2_tiny_exercises_its_own_branches )
M36-mtp-qkv-unchecked KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M37-value-scale-inverted KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M38-trunk-probe-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M39-tiny-workdir-not-created KILLED  (N18Analyzer.test_mimo2_tiny_exercises_its_own_branches )
```

## 11. NOT RUN

- Loading any file in llama.cpp. Every loader verdict is PROVISIONAL, by reading the loaders at both pins.
- Draft acceptance with the Unsloth MTP files, given the NextN compress ratio 0 (section 4): UNKNOWN until run.
- Round 3 has since read the MiMo headers (sections 13–15); its own NOT RUN list is at the end of section 15.
- The tiny mimo2 NextN variant: it needs the N16 generator (`claude/n16-mimo-decode-parity`, not merged); the analyzer
  notes the skip.
- `-hf … --mtp` downloads: the sibling choice of section 5 is by reading `find_best_sibling`, not run.
- Which revision the 3090 host's GLM files are: still the N15 commands (N15 7.3).

## 12. Corrections and deviations

- **Round 1's expected fetch, "about 100–130 MB", was low.** Measured 143 909 824 B: 8 headers of 9.4–10.9 MB each
  pulled a 16 MiB Range by doubling.
- **Round 1 named the MiMo split** (listing first, headers next) instead of guessing globs, as the task allowed. The
  MiMo headers therefore need a third runner pass.
- **Round 2's PROVISIONAL reading of the extra 987 820 224 params** of the single-file MiMo repos (the 3 NextN
  layers) is CONFIRMED by the headers: same trunk names and shapes, plus `blk.48`–`50` (Result (round 3) item 3).
- **The mimo2 sinks / qkv probes counted trunk layers an MTP-only file does not hold.** On the ggml-org `mtp-` file
  they reported "SWA layers without sinks". Fixed in `65ea5b5`, guarded by mutation M40.
- **The MiMo config** is a byte-exact copy under `data/gguf-headers/mimo-v2.6-flash-config/` (lead's instruction; N16
  not merged). The vendored chat template is not copied: its sha256 is compared through `git show` (section 8).

## 13. Round 3: the runner run, bytes, integrity

```
$ gh api repos/samnolak/408032gb/actions/runs/38147482333 --jq '{status, conclusion, head_sha, event, created_at, updated_at, head_branch}'
{"conclusion":"success","created_at":"2026-10-11T14:30:48Z","event":"push","head_branch":"claude/n18-gguf-headers-mtp-mimo","head_sha":"65c380517e5fe53f21c2361490854cde46265e22","status":"completed","updated_at":"2026-10-11T14:31:32Z"}
$ JOB=$(gh api repos/samnolak/408032gb/actions/runs/38147482333/jobs --jq '.jobs[0].id'); echo "job $JOB"
job 114493770093
$ gh api repos/samnolak/408032gb/actions/jobs/$JOB --jq '.steps[] | "\(.number) \(.name) \(.conclusion) \(.started_at) \(.completed_at)"'
1 Set up job success 2026-10-11T14:30:51Z 2026-10-11T14:30:52Z
2 Run actions/checkout@v4 success 2026-10-11T14:30:52Z 2026-10-11T14:30:54Z
3 gguf-py of llama.cpp at the CI pin and numpy, PyYAML (for the self-test) success 2026-10-11T14:30:54Z 2026-10-11T14:31:00Z
4 self-test of the fetcher against a local Range server (no network; fails on any skip but llama-tokenize's) success 2026-10-11T14:31:00Z 2026-10-11T14:31:07Z
5 check the pending requests against the byte caps (fetches nothing; fails on a cap violation) success 2026-10-11T14:31:07Z 2026-10-11T14:31:07Z
6 read the GGUF headers with Range requests success 2026-10-11T14:31:07Z 2026-10-11T14:31:27Z
7 commit and push to this claude/* branch success 2026-10-11T14:31:27Z 2026-10-11T14:31:29Z
14 Post Run actions/checkout@v4 success 2026-10-11T14:31:29Z 2026-10-11T14:31:30Z
15 Complete job success 2026-10-11T14:31:30Z 2026-10-11T14:31:30Z
$ gh api repos/samnolak/408032gb/check-runs/$JOB/annotations --jq '.[] | "\(.annotation_level) | \(.message[0:330])"'
warning | Node.js 20 is deprecated. The following actions target Node.js 20 but are being forced to run on Node.js 24: actions/checkout@v4. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/
notice | headers commit 6161763cce63081500ad121104f1eae9ff144c60
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/mimo-v2.6-flash-mopd-quantaplanta-mxfp4/REQUEST.json: status ok, revision ac33b3743d70ed57a8cc6b847698d2ebbddb7784, 1 headers, 8388608 B fetched, 1665898 B committed (gzip)
notice | QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF:MiMo-V2.6-Flash-MOPD-MXFP4.gguf: GGUF v3, arch 'mimo2', 43 KV, 508 tensors, header 5980736 B (sha256 2e3a7ba2f0b09c15), 8388608 B fetched in 6 Range requests; size check exact; chat_template 11ea52e156de38a4
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/mimo-v2.6-flash-mopd-kernelpool-mxfp4/REQUEST.json: status ok, revision 44d9191bb7a95125c8b971ce0593bd802ae6dd48, 1 headers, 8388608 B fetched, 1666380 B committed (gzip)
notice | kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF:MiMo-V2.6-Flash-MOPD-MXFP4.gguf: GGUF v3, arch 'mimo2', 41 KV, 508 tensors, header 5980608 B (sha256 98002a80c3b0951e), 8388608 B fetched in 6 Range requests; size check exact; chat_template 11ea52e156de38a4
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mxfp4/REQUEST.json: status ok, revision e6d4bced739310a721eefad4c247ec7e0247c89d, 2 headers, 6210784 B fetched, 1664824 B committed (gzip)
notice | ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MiMo-V2.6-Flash-MOPD-MXFP4-00002-of-00002.gguf: GGUF v3, arch None, 3 KV, 472 tensors, header 30048 B (sha256 b11fbbabdcab1b51), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf: GGUF v3, arch 'mimo2', 45 KV, 0 tensors, header 5948636 B (sha256 f2e9498e281706e3), 5948640 B fetched in 6 Range requests; size check exact (no tensors: the writer's padding after the KV block); chat_template 11ea52e156de38a4
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mtp/REQUEST.json: status ok, revision e6d4bced739310a721eefad4c247ec7e0247c89d, 1 headers, 8388608 B fetched, 1661660 B committed (gzip)
notice | ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf: GGUF v3, arch 'mimo2', 43 KV, 39 tensors, header 5950944 B (sha256 8805a08513f9777f), 8388608 B fetched in 6 Range requests; size check exact; chat_template 11ea52e156de38a4
```

`git fetch origin claude/n18-gguf-headers-mtp-mimo && git merge --ff-only origin/claude/n18-gguf-headers-mtp-mimo`
brought `6161763` (author github-actions[bot], 14 files).

- Bytes fetched, CONFIRMED (PIN.json, below):
  - ggml-org MXFP4: 6 210 784 B, i.e. the metadata-only first shard whole (5 948 640 B) plus one 256 KiB Range of
    shard 2;
  - the `mtp-` file, QuantaPlanta and kernelpool: 8 388 608 B each;
  - together 31 376 608 B.
- Headers: 5 948 636, 30 048, 5 950 944, 5 980 736 and 5 980 608 B.
- Integrity, CONFIRMED for all 5 files: the `integrity:` lines of appendix B, and etag = LFS sha256 and repo commit =
  pin in PIN.json. The ggml-org first shard is the header plus 4 padding bytes, and it hashes to its LFS sha256.

The facts printer was run in `/home/user/wt-n18` at `bc26480`, right after the analyzer of appendix B. Each output file
starts with the command it ran:

```
$ cd /home/user/wt-n18; for c in mimopins mimo mimotiny; do echo "\$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py $c" > <S>/n18/r3/facts-$c.txt; LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py $c >> <S>/n18/r3/facts-$c.txt 2>&1; echo "exit $?" >> <S>/n18/r3/facts-$c.txt; done; echo "\$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py mimoloader ec7630a=<S>/llama-ec7630a 79e2e74=<S>/llama.cpp" > <S>/n18/r3/facts-mimoloader.txt; python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py mimoloader ec7630a=<S>/llama-ec7630a 79e2e74=<S>/llama.cpp >> <S>/n18/r3/facts-mimoloader.txt 2>&1; echo "exit $?" >> <S>/n18/r3/facts-mimoloader.txt; grep -h "^exit" <S>/n18/r3/facts-*.txt; grep -c "Warning" <S>/n18/r3/facts-*.txt
exit 0
exit 0
exit 0
exit 0
<S>/n18/r3/facts-mimo.txt:0
<S>/n18/r3/facts-mimoloader.txt:0
<S>/n18/r3/facts-mimopins.txt:0
<S>/n18/r3/facts-mimotiny.txt:0
```

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py mimopins
== mimo-v2.6-flash-mopd-ggml-org-mxfp4 ok ggml-org/MiMo-V2.6-Flash-MOPD-GGUF e6d4bced739310a721eefad4c247ec7e0247c89d -> e6d4bced739310a721eefad4c247ec7e0247c89d gated False ['license:mit'] fetched_utc 2026-10-11T14:31:13Z workflow_sha 65c380517e5fe53f21c2361490854cde46265e22
   bytes_fetched 6210784 committed_bytes (gz) 1664824
   MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf: size 5948640, lfs a353e75ccd30, header 5948636, fetched 5948640 in 6, gz 1661133, size_check(PIN) "exact (no tensors: the writer's padding after the KV block)", etag_ok True, commit_ok True
   MiMo-V2.6-Flash-MOPD-MXFP4-00002-of-00002.gguf: size 167363516512, lfs 3adf8ea80157, header 30048, fetched 262144 in 1, gz 3691, size_check(PIN) 'exact', etag_ok True, commit_ok True
== mimo-v2.6-flash-mopd-ggml-org-mtp ok ggml-org/MiMo-V2.6-Flash-MOPD-GGUF e6d4bced739310a721eefad4c247ec7e0247c89d -> e6d4bced739310a721eefad4c247ec7e0247c89d gated False ['license:mit'] fetched_utc 2026-10-11T14:31:07Z workflow_sha 65c380517e5fe53f21c2361490854cde46265e22
   bytes_fetched 8388608 committed_bytes (gz) 1661660
   mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf: size 2383728864, lfs 4589839383ca, header 5950944, fetched 8388608 in 6, gz 1661660, size_check(PIN) 'exact', etag_ok True, commit_ok True
== mimo-v2.6-flash-mopd-quantaplanta-mxfp4 ok QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF ac33b3743d70ed57a8cc6b847698d2ebbddb7784 -> ac33b3743d70ed57a8cc6b847698d2ebbddb7784 gated False ['license:mit'] fetched_utc 2026-10-11T14:31:24Z workflow_sha 65c380517e5fe53f21c2361490854cde46265e22
   bytes_fetched 8388608 committed_bytes (gz) 1665898
   MiMo-V2.6-Flash-MOPD-MXFP4.gguf: size 174908254272, lfs fc1b8122d1a3, header 5980736, fetched 8388608 in 6, gz 1665898, size_check(PIN) 'exact', etag_ok True, commit_ok True
== mimo-v2.6-flash-mopd-kernelpool-mxfp4 ok kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF 44d9191bb7a95125c8b971ce0593bd802ae6dd48 -> 44d9191bb7a95125c8b971ce0593bd802ae6dd48 gated False ['license:mit'] fetched_utc 2026-10-11T14:31:19Z workflow_sha 65c380517e5fe53f21c2361490854cde46265e22
   bytes_fetched 8388608 committed_bytes (gz) 1666380
   MiMo-V2.6-Flash-MOPD-MXFP4.gguf: size 169005098944, lfs 5e5dfb6d9646, header 5980608, fetched 8388608 in 6, gz 1666380, size_check(PIN) 'exact', etag_ok True, commit_ok True
MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf: header 5948636 + 4 zero bytes = file 5948640; sha256 equal to LFS: True
exit 0
```

## 14. Round 3: the MiMo facts

The three MXFP4 files and the `mtp-` file compared (KV, general keys, tensor names / shapes / types, templates):

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py mimo
mimo-v2.6-flash-mopd-ggml-org-mxfp4: 472 tensors, 45 KV, layers 0-47, nextn_predict_layers None, general {"general.architecture": "mimo2", "general.file_type": 38, "general.languages": ["en", "zh"], "general.license": "mit", "general.name": "MiMo-V2.6-Flash-MOPD", "general.quantization_version": 2, "general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, "general.size_label": "256x7.2B", "general.tags": ["text-generation", "multimodal", "vision-language", "audio", "agent", "video-understanding", "long-context", "mimo_v2", "transformers"], "general.type": "model"}
mimo-v2.6-flash-mopd-ggml-org-mtp: 39 tensors, 43 KV, layers 48-50, nextn_predict_layers 3, general {"general.architecture": "mimo2", "general.file_type": 7, "general.languages": ["en", "zh"], "general.license": "mit", "general.name": "MiMo-V2.6-Flash-MOPD", "general.quantization_version": 2, "general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, "general.size_label": "2.2B", "general.tags": ["text-generation", "multimodal", "vision-language", "audio", "agent", "video-understanding", "long-context", "mimo_v2", "transformers"], "general.type": "model"}
mimo-v2.6-flash-mopd-quantaplanta-mxfp4: 508 tensors, 43 KV, layers 0-50, nextn_predict_layers 3, general {"general.architecture": "mimo2", "general.file_type": 38, "general.languages": ["en", "zh"], "general.license": "mit", "general.name": "MiMo V2.6 Flash MOPD", "general.quantization_version": 2, "general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, "general.size_label": "256x8.2B", "general.tags": ["text-generation", "multimodal", "vision-language", "audio", "agent", "video-understanding", "long-context", "mimo_v2", "transformers"], "general.type": "model"}
mimo-v2.6-flash-mopd-kernelpool-mxfp4: 508 tensors, 41 KV, layers 0-50, nextn_predict_layers 3, general {"general.alignment": 32, "general.architecture": "mimo2", "general.name": "MiMo V2.6 Flash MOPD", "general.source.revision": "2479e2d0029eca9a34cc7e7f55a121925f81908e", "general.source.url": "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD"}
arch KV that differ between the three MXFP4 files (value or presence):
   attention.head_count_kv: ggml-org-mxfp4 [48] first 48 equal to ggml-org's: True, after: [] | quantaplanta-mxfp4 [51] first 48 equal to ggml-org's: True, after: [8, 8, 8] | kernelpool-mxfp4 [51] first 48 equal to ggml-org's: True, after: [8, 8, 8]
   attention.sliding_window_pattern: ggml-org-mxfp4 [48] first 48 equal to ggml-org's: True, after: [] | quantaplanta-mxfp4 [51] first 48 equal to ggml-org's: True, after: [1, 1, 1] | kernelpool-mxfp4 [51] first 48 equal to ggml-org's: True, after: [1, 1, 1]
   block_count: ggml-org-mxfp4 48 | quantaplanta-mxfp4 51 | kernelpool-mxfp4 51
   expert_group_count: ggml-org-mxfp4 1 | quantaplanta-mxfp4 1 | kernelpool-mxfp4 "(absent)"
   expert_group_used_count: ggml-org-mxfp4 1 | quantaplanta-mxfp4 1 | kernelpool-mxfp4 "(absent)"
   expert_weights_norm: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 "(absent)" | kernelpool-mxfp4 true
   expert_weights_scale: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 "(absent)" | kernelpool-mxfp4 1.0
   leading_dense_block_count: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 "(absent)" | kernelpool-mxfp4 1
   nextn_predict_layers: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 3 | kernelpool-mxfp4 3
   qkv_chunks: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 "(absent)" | kernelpool-mxfp4 4
   quantization: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 "(absent)" | kernelpool-mxfp4 "released MXFP4 gate/up/down; Q8_0 attention/dense/head"
   vocab_size: ggml-org-mxfp4 "(absent)" | quantaplanta-mxfp4 "(absent)" | kernelpool-mxfp4 152576
   split.count: ggml-org-mxfp4 2 | quantaplanta-mxfp4 (absent) | kernelpool-mxfp4 (absent)
   split.no: ggml-org-mxfp4 0 | quantaplanta-mxfp4 (absent) | kernelpool-mxfp4 (absent)
   split.tensors.count: ggml-org-mxfp4 472 | quantaplanta-mxfp4 (absent) | kernelpool-mxfp4 (absent)
   tokenizer.ggml.token_type: ggml-org-mxfp4 "6a2b48747cdecdc8ab740d9ca29fb07e668a465 | quantaplanta-mxfp4 "6a2b48747cdecdc8ab740d9ca29fb07e668a465 | kernelpool-mxfp4 "9aab44baf2f6ab1746ea3d79da308225734b1f0
trunk tensors (layers 0-47 + non-layer) mimo-v2.6-flash-mopd-quantaplanta-mxfp4 vs ggml-org: names equal True (only here [], only ggml-org []), shapes equal True, type pairs (ggml-org, this) that differ {('Q8_0', 'BF16'): 101}
trunk tensors (layers 0-47 + non-layer) mimo-v2.6-flash-mopd-kernelpool-mxfp4 vs ggml-org: names equal True (only here [], only ggml-org []), shapes equal True, type pairs (ggml-org, this) that differ {('Q8_0', 'BF16'): 1}
mimo-v2.6-flash-mopd-ggml-org-mxfp4: non-expert matrices by type {'Q8_0': 101, 'F32': 47}; non-layer [('output.weight', 'Q8_0'), ('output_norm.weight', 'F32'), ('token_embd.weight', 'Q8_0')]
mimo-v2.6-flash-mopd-quantaplanta-mxfp4: non-expert matrices by type {'BF16': 101, 'F32': 47}; non-layer [('output.weight', 'BF16'), ('output_norm.weight', 'F32'), ('token_embd.weight', 'BF16')]
mimo-v2.6-flash-mopd-kernelpool-mxfp4: non-expert matrices by type {'BF16': 1, 'Q8_0': 100, 'F32': 47}; non-layer [('output.weight', 'Q8_0'), ('output_norm.weight', 'F32'), ('token_embd.weight', 'BF16')]
mimo-v2.6-flash-mopd-ggml-org-mtp: NextN layers blk.48 12 tensors, names = QuantaPlanta's True, shapes = True, types {'F32': 6, 'Q8_0': 6}; blk.49 12 tensors, names = QuantaPlanta's True, shapes = True, types {'F32': 6, 'Q8_0': 6}; blk.50 12 tensors, names = QuantaPlanta's True, shapes = True, types {'F32': 6, 'Q8_0': 6}
   only here [], only QuantaPlanta []
mimo-v2.6-flash-mopd-quantaplanta-mxfp4: NextN layers blk.48 12 tensors, names = QuantaPlanta's True, shapes = True, types {'BF16': 6, 'F32': 6}; blk.49 12 tensors, names = QuantaPlanta's True, shapes = True, types {'BF16': 6, 'F32': 6}; blk.50 12 tensors, names = QuantaPlanta's True, shapes = True, types {'BF16': 6, 'F32': 6}
   only here [], only QuantaPlanta []
mimo-v2.6-flash-mopd-kernelpool-mxfp4: NextN layers blk.48 12 tensors, names = QuantaPlanta's False, shapes = True, types {'Q8_0': 6, 'F32': 6}; blk.49 12 tensors, names = QuantaPlanta's False, shapes = True, types {'Q8_0': 6, 'F32': 6}; blk.50 12 tensors, names = QuantaPlanta's False, shapes = True, types {'Q8_0': 6, 'F32': 6}
   only here ['nextn.shared_head_norm.weight'], only QuantaPlanta ['layer_output_norm.weight']
mimo-v2.6-flash-mopd-ggml-org-mxfp4: embedded chat template sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059
mimo-v2.6-flash-mopd-ggml-org-mtp: embedded chat template sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059
mimo-v2.6-flash-mopd-quantaplanta-mxfp4: embedded chat template sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059
mimo-v2.6-flash-mopd-kernelpool-mxfp4: embedded chat template sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059
exit 0
```

`attention.head_count_kv` and `attention.sliding_window_pattern` differ only by the 3 NextN entries; the first 48 are
equal. Tokens, merges and token types, decoded from the committed header bytes, and a grep for kernelpool's own keys
(run in `/home/user/wt-n18` at `0f61e10`; the headers are unchanged since `6161763`):

```
$ S=<S>; for r in llama-ec7630a llama.cpp n12/ik-src; do echo "== $r $(git -C $S/$r rev-parse --short HEAD)"; grep -rn "qkv_chunks\|QKV_CHUNKS\|\.quantization\"\|LLM_KV_QUANTIZATION" $S/$r/src $S/$r/common $S/$r/gguf-py $S/$r/conversion 2>/dev/null | head -5; done; cd /home/user/wt-n18 && python3 - <<'EOF'
import gzip, json, sys, collections
sys.path.insert(0, "tools/gguf_headers"); sys.path.insert(0, "tools")
import gguf_header as G
def kv(path):
    h = G.parse(gzip.decompress(open(path, "rb").read()))
    return {e["key"]: e["value"] for e in h.kv}
a = kv("third_party/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mxfp4/MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf.header.gz")
b = kv("third_party/gguf-headers/mimo-v2.6-flash-mopd-kernelpool-mxfp4/MiMo-V2.6-Flash-MOPD-MXFP4.gguf.header.gz")
q = kv("third_party/gguf-headers/mimo-v2.6-flash-mopd-quantaplanta-mxfp4/MiMo-V2.6-Flash-MOPD-MXFP4.gguf.header.gz")
ta, tb, tq = a["tokenizer.ggml.token_type"], b["tokenizer.ggml.token_type"], q["tokenizer.ggml.token_type"]
toks = a["tokenizer.ggml.tokens"]
print("tokens equal a/b/q:", toks == b["tokenizer.ggml.tokens"] == q["tokenizer.ggml.tokens"], "merges equal:", a["tokenizer.ggml.merges"] == b["tokenizer.ggml.merges"] == q["tokenizer.ggml.merges"])
print("token_type ggml-org == quantaplanta:", ta == tq)
d = [i for i in range(len(ta)) if ta[i] != tb[i]]
print("token_type differs at", len(d), "ids; (ggml-org, kernelpool) type pairs", dict(collections.Counter((ta[i], tb[i]) for i in d)))
print("first ids:", [(i, toks[i], ta[i], tb[i]) for i in d[:12]])
print("last ids:", [(i, toks[i], ta[i], tb[i]) for i in d[-6:]])
print("counts by type ggml-org", dict(collections.Counter(ta)), "kernelpool", dict(collections.Counter(tb)))
for k in sorted(set(a) | set(b)):
    if k.startswith("tokenizer.") and k not in ("tokenizer.ggml.tokens","tokenizer.ggml.merges","tokenizer.ggml.token_type","tokenizer.chat_template"):
        if a.get(k) != b.get(k): print("tokenizer key differs", k, a.get(k), b.get(k))
EOF
== llama-ec7630a ec7630a
== llama.cpp 79e2e74
== n12/ik-src 5194a9e
tokens equal a/b/q: True merges equal: True
token_type ggml-org == quantaplanta: True
token_type differs at 6 ids; (ggml-org, kernelpool) type pairs {(3, 4): 6}
first ids: [(151659, '<|fim_prefix|>', 3, 4), (151660, '<|fim_middle|>', 3, 4), (151661, '<|fim_suffix|>', 3, 4), (151662, '<|fim_pad|>', 3, 4), (151663, '<|repo_name|>', 3, 4), (151664, '<|file_sep|>', 3, 4)]
last ids: [(151659, '<|fim_prefix|>', 3, 4), (151660, '<|fim_middle|>', 3, 4), (151661, '<|fim_suffix|>', 3, 4), (151662, '<|fim_pad|>', 3, 4), (151663, '<|repo_name|>', 3, 4), (151664, '<|file_sep|>', 3, 4)]
counts by type ggml-org {1: 151643, 3: 26, 4: 6, 5: 901} kernelpool {1: 151643, 3: 20, 4: 12, 5: 901}
```

The grep prints nothing at any of the three checkouts. Token type 3 is CONTROL and 4 is USER_DEFINED
(`gguf-py/gguf/constants.py:5752-5758@ec7630a`, `class TokenType`).

The probes on the real files against N16's and N17's tiny models. The models were built from each branch's
generator: the `mimo2_tiny.py` text of each branch via `git show`, into a temporary directory; nothing merged.

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py mimotiny
tiny N16 n16.gguf (--fused-qkv --sinks swa --value-scale 0.75 --mtp) [0] @0b0eace: block_count 7, nextn 1, K/V 24/16, n_rot 8
tiny N17 n17.gguf + n17hd.gguf (--mtp-layers 3; --head-dims 192,128) [0] @0352df4: block_count 9, nextn 3, K/V 24/16, n_rot 8
tiny N17 n17.gguf + n17hd.gguf (--mtp-layers 3; --head-dims 192,128) [1] @0352df4: block_count 6, nextn None, K/V 192/128, n_rot 64
== mimo-v2.6-flash-mopd-ggml-org-mxfp4
   [tiny OK] moe.groups: real ['no group step']; not exercised []
   [tiny OK] moe.used_array: real ['scalar']; not exercised []
   [tiny OK] mimo.swa_pattern: real ['full-attention layers', 'sliding-window layers']; not exercised []
   [tiny OK] mimo.kv_heads: real ['per-layer KV heads, full != SWA']; not exercised []
   [tiny OK] mimo.head_dims: real ['K head dim != V head dim']; not exercised []
   [tiny OK] mimo.rope: real ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']; not exercised []
   [tiny OK] mimo.value_scale: real ['attn_out x value_scale']; not exercised []
   [tiny OK] mimo.sinks: real ['SWA layers with sinks', 'full layers without sinks']; not exercised []
   [tiny OK] mimo.qkv: real ['fused attn_qkv']; not exercised []
   [tiny OK] mimo.ffn: real ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']; not exercised []
   [tiny OK] mimo.mtp: real ['nextn_predict_layers 0']; not exercised []
   [tiny OK] mimo.mtp_block: real ['no MTP layers']; not exercised []
   [tiny OK] mimo.mtp_head: real ['no MTP tensors']; not exercised []
   [tiny OK] mimo.file_layout: real ['trunk only (no nextn_predict_layers)']; not exercised []
== mimo-v2.6-flash-mopd-ggml-org-mtp
   [tiny OK] moe.groups: real ['no group step']; not exercised []
   [tiny OK] moe.used_array: real ['scalar']; not exercised []
   [tiny OK] mimo.swa_pattern: real ['full-attention layers', 'sliding-window layers']; not exercised []
   [tiny OK] mimo.kv_heads: real ['per-layer KV heads, full != SWA']; not exercised []
   [tiny OK] mimo.head_dims: real ['K head dim != V head dim']; not exercised []
   [tiny OK] mimo.rope: real ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']; not exercised []
   [tiny OK] mimo.value_scale: real ['attn_out x value_scale']; not exercised []
   [tiny OK] mimo.sinks: real []; not exercised []
   [tiny OK] mimo.qkv: real []; not exercised []
   [tiny OK] mimo.ffn: real []; not exercised []
   [tiny OK] mimo.mtp: real ['nextn_predict_layers 3']; not exercised []
   [tiny OK] mimo.mtp_block: real ['MTP layer with a dense FFN', 'MTP layer with fused attn_qkv']; not exercised []
   [tiny OK] mimo.mtp_head: real ['head norm layer_output_norm', 'trunk output head', 'trunk token_embd']; not exercised []
   [NOT IN N16/N17 TINY] mimo.file_layout: real ['MTP-only file']; not exercised ['MTP-only file']
== mimo-v2.6-flash-mopd-quantaplanta-mxfp4
   [tiny OK] moe.groups: real ['no group step']; not exercised []
   [tiny OK] moe.used_array: real ['scalar']; not exercised []
   [tiny OK] mimo.swa_pattern: real ['full-attention layers', 'sliding-window layers']; not exercised []
   [tiny OK] mimo.kv_heads: real ['per-layer KV heads, full != SWA']; not exercised []
   [tiny OK] mimo.head_dims: real ['K head dim != V head dim']; not exercised []
   [tiny OK] mimo.rope: real ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']; not exercised []
   [tiny OK] mimo.value_scale: real ['attn_out x value_scale']; not exercised []
   [tiny OK] mimo.sinks: real ['SWA layers with sinks', 'full layers without sinks']; not exercised []
   [tiny OK] mimo.qkv: real ['fused attn_qkv']; not exercised []
   [tiny OK] mimo.ffn: real ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']; not exercised []
   [tiny OK] mimo.mtp: real ['nextn_predict_layers 3']; not exercised []
   [tiny OK] mimo.mtp_block: real ['MTP layer with a dense FFN', 'MTP layer with fused attn_qkv']; not exercised []
   [tiny OK] mimo.mtp_head: real ['head norm layer_output_norm', 'trunk output head', 'trunk token_embd']; not exercised []
   [tiny OK] mimo.file_layout: real ['trunk + MTP in one file']; not exercised []
== mimo-v2.6-flash-mopd-kernelpool-mxfp4
   [tiny OK] moe.groups: real ['no group step']; not exercised []
   [tiny OK] moe.used_array: real ['scalar']; not exercised []
   [tiny OK] mimo.swa_pattern: real ['full-attention layers', 'sliding-window layers']; not exercised []
   [tiny OK] mimo.kv_heads: real ['per-layer KV heads, full != SWA']; not exercised []
   [tiny OK] mimo.head_dims: real ['K head dim != V head dim']; not exercised []
   [tiny OK] mimo.rope: real ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']; not exercised []
   [tiny OK] mimo.value_scale: real ['attn_out x value_scale']; not exercised []
   [tiny OK] mimo.sinks: real ['SWA layers with sinks', 'full layers without sinks']; not exercised []
   [tiny OK] mimo.qkv: real ['fused attn_qkv']; not exercised []
   [tiny OK] mimo.ffn: real ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']; not exercised []
   [tiny OK] mimo.mtp: real ['nextn_predict_layers 3']; not exercised []
   [tiny OK] mimo.mtp_block: real ['MTP layer with a dense FFN', 'MTP layer with fused attn_qkv']; not exercised []
   [NOT IN N16/N17 TINY] mimo.mtp_head: real ['head norm nextn.shared_head_norm', 'trunk output head', 'trunk token_embd']; not exercised ['head norm nextn.shared_head_norm']
   [tiny OK] mimo.file_layout: real ['trunk + MTP in one file']; not exercised []
exit 0
```

The loader of each pin against each file's tensor names:

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n18-committed-facts.py mimoloader ec7630a=<S>/llama-ec7630a 79e2e74=<S>/llama.cpp
[ec7630a] load_arch_tensors creates: global:output.weight, global:output_norm.weight, global:token_embd.weight, nextn:layer_output_norm.weight (opt), nextn:nextn.eh_proj.weight, nextn:nextn.embed_tokens.weight (opt), nextn:nextn.enorm.weight, nextn:nextn.hnorm.weight, nextn:nextn.shared_head_head.weight (opt), nextn:nextn.shared_head_norm.weight (opt), trunk:attn_norm.weight, trunk:attn_output.weight, trunk:attn_qkv|attn_q+attn_k+attn_v, trunk:attn_sinks.weight (opt), trunk:exp_probs_b.bias (opt), trunk:ffn_down.weight (opt), trunk:ffn_down_exps.weight (opt), trunk:ffn_gate.weight (opt), trunk:ffn_gate_exps.weight (opt), trunk:ffn_gate_inp.weight (opt), trunk:ffn_norm.weight, trunk:ffn_up.weight (opt), trunk:ffn_up_exps.weight (opt)
   mimo-v2.6-flash-mopd-ggml-org-mxfp4: layers 0-47 (n_layer 48), tensors the loader does not create [], required ones absent []
   mimo-v2.6-flash-mopd-ggml-org-mtp: layers 48-50 (n_layer 48), tensors the loader does not create [], required ones absent []
   mimo-v2.6-flash-mopd-quantaplanta-mxfp4: layers 0-50 (n_layer 48), tensors the loader does not create [], required ones absent []
   mimo-v2.6-flash-mopd-kernelpool-mxfp4: layers 0-50 (n_layer 48), tensors the loader does not create [], required ones absent []
[79e2e74] load_arch_tensors creates: global:output.weight, global:output_norm.weight, global:token_embd.weight, nextn:layer_output_norm.weight (opt), nextn:nextn.eh_proj.weight, nextn:nextn.embed_tokens.weight (opt), nextn:nextn.enorm.weight, nextn:nextn.hnorm.weight, nextn:nextn.shared_head_head.weight (opt), nextn:nextn.shared_head_norm.weight (opt), trunk:attn_norm.weight, trunk:attn_output.weight, trunk:attn_qkv|attn_q+attn_k+attn_v, trunk:attn_sinks.weight (opt), trunk:exp_probs_b.bias (opt), trunk:ffn_down.weight (opt), trunk:ffn_down_exps.weight (opt), trunk:ffn_gate.weight (opt), trunk:ffn_gate_exps.weight (opt), trunk:ffn_gate_inp.weight (opt), trunk:ffn_norm.weight, trunk:ffn_up.weight (opt), trunk:ffn_up_exps.weight (opt)
   mimo-v2.6-flash-mopd-ggml-org-mxfp4: layers 0-47 (n_layer 48), tensors the loader does not create [], required ones absent []
   mimo-v2.6-flash-mopd-ggml-org-mtp: layers 48-50 (n_layer 48), tensors the loader does not create [], required ones absent []
   mimo-v2.6-flash-mopd-quantaplanta-mxfp4: layers 0-50 (n_layer 48), tensors the loader does not create [], required ones absent []
   mimo-v2.6-flash-mopd-kernelpool-mxfp4: layers 0-50 (n_layer 48), tensors the loader does not create [], required ones absent []
exit 0
```

The source lines cited in Result (round 3), printed back:

```
$ bash <S>/n18/r3/anchors-cmd.sh > <S>/n18/r3/anchors.txt 2>&1; echo "rc=$?"; cat <S>/n18/r3/anchors.txt
rc=0
== ec7630a ec7630a, 79e2e74 79e2e74
-- conversion/mimo.py:243 (ec7630a)
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
-- conversion/mimo.py:338-340 (ec7630a)
        if self._is_mxfp4_packed():
            self._is_mxfp4 = True
            self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE
-- include/llama.h:156 (ec7630a) / :157 (79e2e74)
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
-- src/llama-quant.cpp:512-523 (ec7630a)
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
-- gguf-py/gguf/tensor_mapping.py:743 (ec7630a) / :755 (79e2e74)
            "model.layers.{bid}.final_layernorm",           # bailingmoe2
            "model.layers.{bid}.final_layernorm",           # bailingmoe2
-- gguf-py/gguf/tensor_mapping.py:2772 (ec7630a) / :2874 (79e2e74)
            "model.layers.{bid}.shared_head.norm",
            "model.layers.{bid}.shared_head.norm",
-- src/llama-vocab.cpp:1963 (ec7630a) / :2076 (79e2e74)
            if (ml.get_key(LLM_KV_VOCAB_SIZE, n_tokens, false)) {
            if (ml.get_key(LLM_KV_VOCAB_SIZE, n_tokens, false)) {
-- src/models/mimo2.cpp:134-142 (ec7630a)
                const size_t row_k    = ggml_row_size(qkv->type, n_embd_head_k);
                const size_t row_v    = ggml_row_size(qkv->type, n_embd_head_v);
                const size_t row_full = qkv->nb[1];
                const size_t k_off    = row_k * n_head_l;
                const size_t v_off    = k_off + row_k * n_head_kv_l;

                Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_l,    n_tokens, row_k, row_full, 0);
                Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv_l, n_tokens, row_k, row_full, k_off);
                Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_kv_l, n_tokens, row_v, row_full, v_off);
-- conversion/mimo.py:35-38 (ec7630a, start of the TP regrouping)
    def _tp_aware_qkv_dequant(weight: Tensor, scale_inv: Tensor,
                              n_q: int, n_kv: int, hd: int, vhd: int,
                              bs: int = 128) -> Tensor:
        # MiMo-V2.5 (TP=4) and V2.5-Pro (TP=8) ship qkv_proj sharded across TP
-- src/models/mimo2.cpp:28 (ec7630a / 79e2e74)
    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const auto nf = nextn_flags(ml);
-- src/models/mimo2.cpp:88 (ec7630a) / :82 (79e2e74)
        return std::make_unique<graph_mtp>(*this, params);
        return std::make_unique<graph_mtp>(*this, params);
-- src/models/mimo2.cpp:104 (ec7630a)
    const bool emit_h_nextn = cparams.embeddings_nextn;
-- src/models/mimo2.cpp:244-246 (ec7630a) / :236-238 (79e2e74)
    if (emit_h_nextn) {
        cb(cur, "h_nextn", -1);
        res->t_h_nextn = cur;
    // pre-norm state for MTP; set even when extraction is off
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;
-- grep of kernelpool's extra keys in src/models/mimo2.cpp at both pins
ec7630a: 0 matching lines
79e2e74: 0 matching lines
-- src/llama-model.cpp:3485 (79e2e74)
llama_model_base::nextn_flags_t llama_model_base::nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe) const {
$ cat <S>/n18/r3/anchors-cmd.sh
S=<S>
E=$S/llama-ec7630a; R=$S/llama.cpp
echo "== ec7630a $(git -C $E rev-parse --short HEAD), 79e2e74 $(git -C $R rev-parse --short HEAD)"
echo "-- conversion/mimo.py:243 (ec7630a)"; sed -n 243p $E/conversion/mimo.py
echo "-- conversion/mimo.py:338-340 (ec7630a)"; sed -n 338,340p $E/conversion/mimo.py
echo "-- include/llama.h:156 (ec7630a) / :157 (79e2e74)"; sed -n 156p $E/include/llama.h; sed -n 157p $R/include/llama.h
echo "-- src/llama-quant.cpp:512-523 (ec7630a)"; sed -n 512,523p $E/src/llama-quant.cpp
echo "-- gguf-py/gguf/tensor_mapping.py:743 (ec7630a) / :755 (79e2e74)"; sed -n 743p $E/gguf-py/gguf/tensor_mapping.py; sed -n 755p $R/gguf-py/gguf/tensor_mapping.py
echo "-- gguf-py/gguf/tensor_mapping.py:2772 (ec7630a) / :2874 (79e2e74)"; sed -n 2772p $E/gguf-py/gguf/tensor_mapping.py; sed -n 2874p $R/gguf-py/gguf/tensor_mapping.py
echo "-- src/llama-vocab.cpp:1963 (ec7630a) / :2076 (79e2e74)"; sed -n 1963p $E/src/llama-vocab.cpp; sed -n 2076p $R/src/llama-vocab.cpp
echo "-- src/models/mimo2.cpp:134-142 (ec7630a)"; sed -n 134,142p $E/src/models/mimo2.cpp
echo "-- conversion/mimo.py:35-38 (ec7630a, start of the TP regrouping)"; sed -n 35,38p $E/conversion/mimo.py
echo "-- src/models/mimo2.cpp:28 (ec7630a / 79e2e74)"; sed -n 28p $E/src/models/mimo2.cpp; sed -n 28p $R/src/models/mimo2.cpp
echo "-- src/models/mimo2.cpp:88 (ec7630a) / :82 (79e2e74)"; sed -n 88p $E/src/models/mimo2.cpp; sed -n 82p $R/src/models/mimo2.cpp
echo "-- src/models/mimo2.cpp:104 (ec7630a)"; sed -n 104p $E/src/models/mimo2.cpp
echo "-- src/models/mimo2.cpp:244-246 (ec7630a) / :236-238 (79e2e74)"; sed -n 244,246p $E/src/models/mimo2.cpp; sed -n 236,238p $R/src/models/mimo2.cpp
echo "-- grep of kernelpool's extra keys in src/models/mimo2.cpp at both pins"; for d in $E $R; do echo "$(git -C $d rev-parse --short HEAD): $(grep -c "qkv_chunks\|QKV_CHUNKS\|LEADING_DENSE\|EXPERT_WEIGHTS_SCALE\|EXPERT_WEIGHTS_NORM\|VOCAB_SIZE" $d/src/models/mimo2.cpp) matching lines"; done
echo "-- src/llama-model.cpp:3485 (79e2e74)"; sed -n 3485p $R/src/llama-model.cpp
```

## 15. Round 3: tests and mutation check

- Round-3 code changes:
  - `65ea5b5`: the mimo2 sinks / qkv probes skip trunk layers that an MTP-only file does not hold;
  - `7edc27f`: the facts printer gains `mimopins`, `mimo`, `mimotiny`, `mimoloader`;
  - `bc26480`: `CommittedN18Mimo`, 5 tests;
  - `0f61e10`: mutation M40.
- `0f61e10` changes only the mutation script, so the suite at `bc26480` is the suite of the code HEAD.

```
$ export TMPDIR=<S>/n18/tmp
$ cd /home/user/wt-n18 && git log --oneline -n 1 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > <S>/n18/full-suite-bc26480.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n18/full-suite-bc26480.log
bc26480 [model-analyst] tests: facts of the N18 MiMo headers (bot 6161763): config cross-check, MXFP4 experts and D5, the three MXFP4 files and the mtp file compared
exit 0
Ran 917 tests in 218.227s
OK (skipped=358)
$ python3 -m unittest tests.test_hooks tests.test_check_unittest 2>&1 | tail -3
Ran 54 tests in 19.155s

OK
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py LLAMA_CPP_SRC_79E2E74=<S>/llama.cpp LLAMA_TOKENIZE_BIN=<S>/n15/build-tok/bin/llama-tokenize python3 -m unittest tests.test_gguf_headers tests.test_gguf_headers_committed -v > <S>/n18/modules-bc26480.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n18/modules-bc26480.log; grep -c " \.\.\. ok$" <S>/n18/modules-bc26480.log; grep -c "skipped" <S>/n18/modules-bc26480.log
exit 0
Ran 57 tests in 35.201s
OK
57
0
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest tests.test_gguf_headers -v > <S>/n18/selftest-bc26480.log 2>&1; echo "self-test unittest rc=$?"; grep -E "^(Ran|OK|FAILED)" <S>/n18/selftest-bc26480.log; skips=$(grep -E ' \.\.\. skipped ' <S>/n18/selftest-bc26480.log | grep -v '^test_llama_tokenize_runs_on_the_vocab_only_file ' || true); [ -n "$skips" ] && echo "step FAILS: $skips" || echo "step passes"
self-test unittest rc=0
Ran 39 tests in 18.612s
OK (skipped=1)
step passes
```

917 = 912 + the 5 `CommittedN18Mimo` tests. Skips are unchanged; the pin-dependent MiMo test runs with the CI pin's
checkout.

```
$ export TMPDIR=<S>/n18/tmp; cd <S>/n18 && git -C /home/user/wt-n18 log --oneline -n 1 && bash /home/user/wt-n18/docs/evidence/mutation-scripts/2026-10-11-n18-mutations.sh > mutations-r3.log 2>&1; echo "exit $?"; tail -2 mutations-r3.log | cut -c1-200; head -2 mutations-r3.log; grep -c KILLED mutations-r3.log; grep -c "SURVIVED\|NOT APPLIED" mutations-r3.log
0f61e10 [model-analyst] evidence scripts: N18 mutation M40 (mimo2 probes count trunk layers an MTP-only file does not hold)
exit 0
M39-tiny-workdir-not-created KILLED  (N18Analyzer.test_mimo2_tiny_exercises_its_own_branches )
M40-mimo-probes-count-absent-layers KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
HEAD 0f61e10
base exit 0 (Ran 39 tests in 17.940s, OK (skipped=1))
21
0
$ cat <S>/n18/mutations-r3.log
HEAD 0f61e10
base exit 0 (Ran 39 tests in 17.940s, OK (skipped=1))
M20-named-revision-unchecked KILLED  (Fetch.test_discover_named_repo_revision_is_pinned_and_requests_are_checked )
M21-named-listed-again-by-search KILLED  (Fetch.test_discover_lists_named_repos_and_reads_no_file )
M22-named-unavailable-raises KILLED  (Fetch.test_discover_lists_named_repos_and_reads_no_file N18Analyzer.test_listing_renders_named_repos )
M23-duplicate-repos-accepted KILLED  (Fetch.test_discover_named_repo_revision_is_pinned_and_requests_are_checked )
M24-optional-counted-missing KILLED  (N18Analyzer.test_reference_diff_takes_fused_and_split_forms_as_equal N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M25-no-fused-split-alternatives KILLED  (N18Analyzer.test_reference_diff_takes_fused_and_split_forms_as_equal )
M26-globals-unchecked KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M27-no-mtp-only-handling-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M28-missing-graph-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M29-tokenizer-unchecked KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M30-extra-tensors-ignored KILLED  (N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M31-lax-key-regex KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M32-conditional-generic-keys-required KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M33-pattern-without-mtp-layers KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M34-kv-array-without-mtp-layers KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M35-sinks-label-inverted KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit N18Analyzer.test_mimo2_tiny_exercises_its_own_branches )
M36-mtp-qkv-unchecked KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M37-value-scale-inverted KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
M38-trunk-probe-ignored KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit N18Analyzer.test_separate_mtp_files_loader_verdicts_per_pin )
M39-tiny-workdir-not-created KILLED  (N18Analyzer.test_mimo2_tiny_exercises_its_own_branches )
M40-mimo-probes-count-absent-layers KILLED  (N18Analyzer.test_mimo2_config_branches_keys_experts_audit )
```

21 of 21 killed.

NOT RUN in round 3:
- loading any MiMo file;
- reading `attn_qkv` rows to settle kernelpool's `qkv_chunks` question;
- N16 / N17 parity runs: not re-run here; their results are their branches' evidence;
- the BF16 and Q8_0-MTP numerics.

## Appendix A. `tools/gguf_headers/analyze.py` on the N18 headers, full output

At `ecd05be`:

```
$ cd <S>/n18/r2 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 /home/user/wt-n18/tools/gguf_headers/analyze.py --src ec7630a=<S>/llama-ec7630a --src 79e2e74=<S>/llama.cpp --name qwen3.8-flash-next-mtp --name glm-5.3-flash-ggml-org-q4_k --name glm-5.3-flash-ggml-org-mtp --name discover-mimo --json analyze.json > analyze.txt; echo "exit $?"
exit 0
sources: ec7630a = ec7630a640789c393694fb194f1bbbf0369fc62d (<S>/llama-ec7630a), 79e2e74 = 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (<S>/llama.cpp)

=== qwen3.8-flash-next-mtp (third_party/gguf-headers/qwen3.8-flash-next-mtp)
status ok, unsloth/Qwen3.8-Flash-Next-GGUF @ 38bb39ee97821de2c9009abb7e93950eec396e66
integrity: mtp-Qwen3.8-Flash-Next-BF16.gguf OK, mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf OK, mtp-Qwen3.8-Flash-Next-Q8_0.gguf OK, mtp-Qwen3.8-Flash-Next-shared-BF16.gguf OK, mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf OK, mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf OK
mode mtp_files; NextN reference: tiny qwen4exp --mtp, blk.8; trunk request: qwen3.8-flash-next-ud-q4_k_xl
  MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf: 7770760320 B, LFS sha256 8ac04a65440ade36c4001c432f5dba256028e047bbaa71ff6161f91fe149a99c, header 10946688 B (sha256 77b6c9be7f4c1d6bd3ff900095592d0e15e2be89d89f19da78ddb257059e536f), fetched 16777216 B, committed 3084942 B gz, 34 tensors, size check exact, offsets contiguous
  MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf: 2786204800 B, LFS sha256 b646ef60eaae2a9ed849e75f15f399629ca22633555e99e809959e95f22a1575, header 10946688 B (sha256 4c7d482b072ecb8e1a5b25f00e0559208d006e55078e6247f8503c900a3de7fa), fetched 16777216 B, committed 3084972 B gz, 34 tensors, size check exact, offsets contiguous
  MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf: 4137429120 B, LFS sha256 cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e, header 10946688 B (sha256 9e613d2259aa555abed34853e34efdddef0596ae2ae29225b4c2ef833f865ec0), fetched 16777216 B, committed 3084980 B gz, 34 tensors, size check exact, offsets contiguous
  MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf: 5227963456 B, LFS sha256 32473fa74e8ed0f0026219b09b43d303f52f328c0f8b0caf4b24bf8834cc596b, header 10946624 B (sha256 ab42ddf3f18ca34d106e63ab6e8b388784db657f4798544d5ce21f87de48e354), fetched 16777216 B, committed 3084937 B gz, 32 tensors, size check exact, offsets contiguous
  MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf: 1907151936 B, LFS sha256 f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc, header 10946624 B (sha256 e0eab755571a607d96e1bfad1732accdfa775b34e4d2d9dd89255189ec344a6f), fetched 16777216 B, committed 3084963 B gz, 32 tensors, size check exact, offsets contiguous
  MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf: 2786568256 B, LFS sha256 5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6, header 10946624 B (sha256 85b40be410e28e9e07bad48062b9fb8db0e3d837658a4bfd999f7f20e58836a6), fetched 16777216 B, committed 3084959 B gz, 32 tensors, size check exact, offsets contiguous

how llama.cpp attaches a separate MTP file:
  [ec7630a] common/arg.cpp:4241: a local MTP file is given as the draft model: -md FILE (sets speculative.draft.mparams.path)
  [ec7630a] common/arg.cpp:4249: --spec-type draft-mtp selects the MTP drafter
  [ec7630a] common/speculative.cpp:37: the type name draft-mtp
  [ec7630a] common/arg.cpp:3084: --mtp (download example): also fetch the repo's MTP sidecar
  [ec7630a] common/download.cpp:616: -hf: the sidecar is the sibling file whose name starts with mtp-
  [ec7630a] common/download.cpp:649: an mtp- file is never taken as the model
  [ec7630a] common/arg.cpp:644: the downloaded sidecar becomes the draft model path
  [ec7630a] common/speculative.cpp:2539: with a draft path the MTP file is loaded as its own llama_model (llama_model_load_from_file)
  [ec7630a] common/speculative.cpp:2526: its context is an MTP context
  [ec7630a] common/speculative.cpp:2560: without a draft path the MTP context runs on the target model's own NextN layers
  [ec7630a] common/speculative.cpp:1376: the MTP model's output width must equal the target's (h_nextn rows)
  [ec7630a] common/common.cpp:1625: draft-mtp sets load_mtp (otherwise NextN tensors are skipped)
  [ec7630a] src/llama-vocab.cpp:1944: a separate model file needs its own tokenizer metadata (tokenizer.ggml.model is required)
  [ec7630a] src/llama-model-loader.cpp:1398: every tensor in the file must be created by the arch loader
  [ec7630a] src/llama-context.cpp:156: only EAGLE3 / DFlash drafts borrow the target's embeddings and head through ctx_other
  [ec7630a] src/llama-model.cpp:2802,2809: an MTP context's KV cache holds only the NextN layers
  [79e2e74] common/arg.cpp:4276: a local MTP file is given as the draft model: -md FILE (sets speculative.draft.mparams.path)
  [79e2e74] common/arg.cpp:4284: --spec-type draft-mtp selects the MTP drafter
  [79e2e74] common/speculative.cpp:76: the type name draft-mtp
  [79e2e74] common/arg.cpp:3097: --mtp (download example): also fetch the repo's MTP sidecar
  [79e2e74] common/download.cpp:605: -hf: the sidecar is the sibling file whose name starts with mtp-
  [79e2e74] common/download.cpp:638: an mtp- file is never taken as the model
  [79e2e74] common/arg.cpp:644: the downloaded sidecar becomes the draft model path
  [79e2e74] common/speculative.cpp:2623: with a draft path the MTP file is loaded as its own llama_model (llama_model_load_from_file)
  [79e2e74] common/speculative.cpp:2610: its context is an MTP context
  [79e2e74] common/speculative.cpp:2644: without a draft path the MTP context runs on the target model's own NextN layers
  [79e2e74] common/speculative.cpp:1437: the MTP model's output width must equal the target's (h_nextn rows)
  [79e2e74] common/common.cpp:1690: draft-mtp sets load_mtp (otherwise NextN tensors are skipped)
  [79e2e74] src/llama-vocab.cpp:2057: a separate model file needs its own tokenizer metadata (tokenizer.ggml.model is required)
  [79e2e74] src/llama-model-loader.cpp:1398: every tensor in the file must be created by the arch loader
  [79e2e74] src/llama-context.cpp:159: only EAGLE3 / DFlash drafts borrow the target's embeddings and head through ctx_other
  [79e2e74] src/llama-model.cpp:2841,2848: an MTP context's KV cache holds only the NextN layers

--- MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf: arch qwen4exp, block_count 49, nextn_predict_layers 1, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen35, 248320 tokens; embedded chat template: True
  non-layer tensors: output.weight BF16 [2560, 248320], token_embd.weight BF16 [2560, 248320]
  layers present: [48]
  blk.48: 32 tensors: ['attn_k.weight', 'attn_k_norm.weight', 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'hc_attn_down.weight', 'hc_attn_inject.weight', 'hc_attn_norm.weight', 'hc_attn_up.weight', 'hc_ffn_down.weight', 'hc_ffn_inject.weight', 'hc_ffn_norm.weight', 'hc_ffn_up.weight', 'indexer.k_norm.weight', 'indexer.k_proj.weight', 'indexer.q_norm.weight', 'indexer.q_proj.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hc_head_down.weight', 'nextn.hc_head_norm.weight', 'nextn.hc_head_up.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the trunk (qwen4exp): arch keys that differ: attention.compress_ratios: trunk [48] 0 x36, 4 x12 / this [49] 0 x37, 4 x12; block_count: trunk 48 / this 49; embedding_length_per_layer_input: trunk 160 / this (absent); nextn_predict_layers: trunk (absent) / this 1; ple.conv_kernel: trunk 4 / this (absent); ple.eos_token_id: trunk 248044 / this (absent); ple.head_offsets: trunk [16] 0 x1, 20000003 x1, 40000026 x1, ... / this (absent); ple.head_vocab_sizes: trunk [16] 20000003 x1, 20000023 x1, 200000... / this (absent); ple.heads_per_ngram: trunk 8 / this (absent); ple.image_token_id: trunk 248056 / this (absent); ple.layer_multipliers: trunk [23703573157769, 20109073645365, 8052... / this (absent); ple.layers: trunk [1] / this (absent); ple.ngram_size: trunk 3 / this (absent); token list equal: True; token_embd trunk [2560, 248320] / this [2560, 248320]
  config.json cross-check: 24 of 28 OK; ABSENT ple_layer_ids = [2] -> ple.layers = null; ABSENT ngram_size = 3 -> ple.ngram_size = null; ABSENT heads_per_ngram = 8 -> ple.heads_per_ngram = null; ABSENT ple_conv_kernel_size = 4 -> ple.conv_kernel = null
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
    [79e2e74] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
  routed-expert types per projection (layers): {"ffn_down_exps": {"BF16": 1}, "ffn_gate_exps": {"BF16": 1}, "ffn_up_exps": {"BF16": 1}}
  audit: PASS, 7.227 GiB, 16.006 bpw; other 16.634 bpw {'F32': 1355008, 'BF16': 32849920}, routed_experts 16.000 bpw {'BF16': 2516582400}, shared_experts 16.008 bpw {'F32': 2560, 'BF16': 4915200}, attention 16.000 bpw {'BF16': 51445760}, embedding 16.000 bpw {'BF16': 635699200}, output_head 16.000 bpw {'BF16': 635699200}
  loader [ec7630a]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/qwen4exp.cpp:179
      graph: src/models/qwen4exp.cpp:307
  loader [79e2e74]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/qwen4exp.cpp:178
      graph: src/models/qwen4exp.cpp:306

--- MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf: arch qwen4exp, block_count 49, nextn_predict_layers 1, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen35, 248320 tokens; embedded chat template: True
  non-layer tensors: output.weight Q6_K [2560, 248320], token_embd.weight Q4_K [2560, 248320]
  layers present: [48]
  blk.48: 32 tensors: ['attn_k.weight', 'attn_k_norm.weight', 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'hc_attn_down.weight', 'hc_attn_inject.weight', 'hc_attn_norm.weight', 'hc_attn_up.weight', 'hc_ffn_down.weight', 'hc_ffn_inject.weight', 'hc_ffn_norm.weight', 'hc_ffn_up.weight', 'indexer.k_norm.weight', 'indexer.k_proj.weight', 'indexer.q_norm.weight', 'indexer.q_proj.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hc_head_down.weight', 'nextn.hc_head_norm.weight', 'nextn.hc_head_up.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the trunk (qwen4exp): arch keys that differ: attention.compress_ratios: trunk [48] 0 x36, 4 x12 / this [49] 0 x37, 4 x12; block_count: trunk 48 / this 49; embedding_length_per_layer_input: trunk 160 / this (absent); nextn_predict_layers: trunk (absent) / this 1; ple.conv_kernel: trunk 4 / this (absent); ple.eos_token_id: trunk 248044 / this (absent); ple.head_offsets: trunk [16] 0 x1, 20000003 x1, 40000026 x1, ... / this (absent); ple.head_vocab_sizes: trunk [16] 20000003 x1, 20000023 x1, 200000... / this (absent); ple.heads_per_ngram: trunk 8 / this (absent); ple.image_token_id: trunk 248056 / this (absent); ple.layer_multipliers: trunk [23703573157769, 20109073645365, 8052... / this (absent); ple.layers: trunk [1] / this (absent); ple.ngram_size: trunk 3 / this (absent); token list equal: True; token_embd trunk [2560, 248320] / this [2560, 248320]
  config.json cross-check: 24 of 28 OK; ABSENT ple_layer_ids = [2] -> ple.layers = null; ABSENT ngram_size = 3 -> ple.ngram_size = null; ABSENT heads_per_ngram = 8 -> ple.heads_per_ngram = null; ABSENT ple_conv_kernel_size = 4 -> ple.conv_kernel = null
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
    [79e2e74] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q8_0": 1}, "ffn_gate_exps": {"Q4_K": 1}, "ffn_up_exps": {"Q4_K": 1}}
  audit: PASS, 2.585 GiB, 5.724 bpw; output_head 6.562 bpw {'Q6_K': 635699200}, embedding 4.500 bpw {'Q4_K': 635699200}, attention 4.919 bpw {'Q4_K': 48496640, 'Q6_K': 1310720, 'BF16': 1638400}, other 6.074 bpw {'F32': 1355008, 'Q4_K': 19742720, 'Q5_0': 9830400, 'Q6_K': 3276800}, routed_experts 5.833 bpw {'Q8_0': 838860800, 'Q4_K': 1677721600}, shared_experts 5.847 bpw {'Q8_0': 1638400, 'F32': 2560, 'Q4_K': 3276800}
  loader [ec7630a]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/qwen4exp.cpp:179
      graph: src/models/qwen4exp.cpp:307
  loader [79e2e74]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/qwen4exp.cpp:178
      graph: src/models/qwen4exp.cpp:306

--- MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf: arch qwen4exp, block_count 49, nextn_predict_layers 1, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen35, 248320 tokens; embedded chat template: True
  non-layer tensors: output.weight Q8_0 [2560, 248320], token_embd.weight Q8_0 [2560, 248320]
  layers present: [48]
  blk.48: 32 tensors: ['attn_k.weight', 'attn_k_norm.weight', 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'hc_attn_down.weight', 'hc_attn_inject.weight', 'hc_attn_norm.weight', 'hc_attn_up.weight', 'hc_ffn_down.weight', 'hc_ffn_inject.weight', 'hc_ffn_norm.weight', 'hc_ffn_up.weight', 'indexer.k_norm.weight', 'indexer.k_proj.weight', 'indexer.q_norm.weight', 'indexer.q_proj.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hc_head_down.weight', 'nextn.hc_head_norm.weight', 'nextn.hc_head_up.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the trunk (qwen4exp): arch keys that differ: attention.compress_ratios: trunk [48] 0 x36, 4 x12 / this [49] 0 x37, 4 x12; block_count: trunk 48 / this 49; embedding_length_per_layer_input: trunk 160 / this (absent); nextn_predict_layers: trunk (absent) / this 1; ple.conv_kernel: trunk 4 / this (absent); ple.eos_token_id: trunk 248044 / this (absent); ple.head_offsets: trunk [16] 0 x1, 20000003 x1, 40000026 x1, ... / this (absent); ple.head_vocab_sizes: trunk [16] 20000003 x1, 20000023 x1, 200000... / this (absent); ple.heads_per_ngram: trunk 8 / this (absent); ple.image_token_id: trunk 248056 / this (absent); ple.layer_multipliers: trunk [23703573157769, 20109073645365, 8052... / this (absent); ple.layers: trunk [1] / this (absent); ple.ngram_size: trunk 3 / this (absent); token list equal: True; token_embd trunk [2560, 248320] / this [2560, 248320]
  config.json cross-check: 24 of 28 OK; ABSENT ple_layer_ids = [2] -> ple.layers = null; ABSENT ngram_size = 3 -> ple.ngram_size = null; ABSENT heads_per_ngram = 8 -> ple.heads_per_ngram = null; ABSENT ple_conv_kernel_size = 4 -> ple.conv_kernel = null
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
    [79e2e74] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q8_0": 1}, "ffn_gate_exps": {"Q8_0": 1}, "ffn_up_exps": {"Q8_0": 1}}
  audit: PASS, 3.843 GiB, 8.511 bpw; output_head 8.500 bpw {'Q8_0': 635699200}, embedding 8.500 bpw {'Q8_0': 635699200}, attention 8.739 bpw {'Q8_0': 49807360, 'BF16': 1638400}, other 9.431 bpw {'F32': 1355008, 'Q8_0': 32849920}, routed_experts 8.500 bpw {'Q8_0': 2516582400}, shared_experts 8.512 bpw {'Q8_0': 4915200, 'F32': 2560}
  loader [ec7630a]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/qwen4exp.cpp:179
      graph: src/models/qwen4exp.cpp:307
  loader [79e2e74]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/qwen4exp.cpp:178
      graph: src/models/qwen4exp.cpp:306

--- MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf: arch qwen4exp, block_count 49, nextn_predict_layers 1, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen35, 248320 tokens; embedded chat template: True
  non-layer tensors: none
  layers present: [48]
  blk.48: 32 tensors: ['attn_k.weight', 'attn_k_norm.weight', 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'hc_attn_down.weight', 'hc_attn_inject.weight', 'hc_attn_norm.weight', 'hc_attn_up.weight', 'hc_ffn_down.weight', 'hc_ffn_inject.weight', 'hc_ffn_norm.weight', 'hc_ffn_up.weight', 'indexer.k_norm.weight', 'indexer.k_proj.weight', 'indexer.q_norm.weight', 'indexer.q_proj.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hc_head_down.weight', 'nextn.hc_head_norm.weight', 'nextn.hc_head_up.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the trunk (qwen4exp): arch keys that differ: attention.compress_ratios: trunk [48] 0 x36, 4 x12 / this [49] 0 x37, 4 x12; block_count: trunk 48 / this 49; embedding_length_per_layer_input: trunk 160 / this (absent); nextn_predict_layers: trunk (absent) / this 1; nextn_shared_target_tensors: trunk (absent) / this true; ple.conv_kernel: trunk 4 / this (absent); ple.eos_token_id: trunk 248044 / this (absent); ple.head_offsets: trunk [16] 0 x1, 20000003 x1, 40000026 x1, ... / this (absent); ple.head_vocab_sizes: trunk [16] 20000003 x1, 20000023 x1, 200000... / this (absent); ple.heads_per_ngram: trunk 8 / this (absent); ple.image_token_id: trunk 248056 / this (absent); ple.layer_multipliers: trunk [23703573157769, 20109073645365, 8052... / this (absent); ple.layers: trunk [1] / this (absent); ple.ngram_size: trunk 3 / this (absent); token list equal: True; token_embd trunk [2560, 248320] / this None
  config.json cross-check: 24 of 28 OK; ABSENT ple_layer_ids = [2] -> ple.layers = null; ABSENT ngram_size = 3 -> ple.ngram_size = null; ABSENT heads_per_ngram = 8 -> ple.heads_per_ngram = null; ABSENT ple_conv_kernel_size = 4 -> ple.conv_kernel = null
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
    [79e2e74] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
  routed-expert types per projection (layers): {"ffn_down_exps": {"BF16": 1}, "ffn_gate_exps": {"BF16": 1}, "ffn_up_exps": {"BF16": 1}}
  audit: PASS, 4.859 GiB, 16.008 bpw; other 16.634 bpw {'F32': 1355008, 'BF16': 32849920}, routed_experts 16.000 bpw {'BF16': 2516582400}, shared_experts 16.008 bpw {'F32': 2560, 'BF16': 4915200}, attention 16.000 bpw {'BF16': 51445760}
  loader [ec7630a]: NOT loadable as a separate MTP model
      - token_embd.weight absent, created with flags 0 (src/models/qwen4exp.cpp:183)
      detect: src/models/qwen4exp.cpp:179
      graph: src/models/qwen4exp.cpp:307
  loader [79e2e74]: NOT loadable as a separate MTP model
      - token_embd.weight absent, created with flags 0 (src/models/qwen4exp.cpp:182)
      detect: src/models/qwen4exp.cpp:178
      graph: src/models/qwen4exp.cpp:306

--- MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf: arch qwen4exp, block_count 49, nextn_predict_layers 1, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen35, 248320 tokens; embedded chat template: True
  non-layer tensors: none
  layers present: [48]
  blk.48: 32 tensors: ['attn_k.weight', 'attn_k_norm.weight', 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'hc_attn_down.weight', 'hc_attn_inject.weight', 'hc_attn_norm.weight', 'hc_attn_up.weight', 'hc_ffn_down.weight', 'hc_ffn_inject.weight', 'hc_ffn_norm.weight', 'hc_ffn_up.weight', 'indexer.k_norm.weight', 'indexer.k_proj.weight', 'indexer.q_norm.weight', 'indexer.q_proj.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hc_head_down.weight', 'nextn.hc_head_norm.weight', 'nextn.hc_head_up.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the trunk (qwen4exp): arch keys that differ: attention.compress_ratios: trunk [48] 0 x36, 4 x12 / this [49] 0 x37, 4 x12; block_count: trunk 48 / this 49; embedding_length_per_layer_input: trunk 160 / this (absent); nextn_predict_layers: trunk (absent) / this 1; nextn_shared_target_tensors: trunk (absent) / this true; ple.conv_kernel: trunk 4 / this (absent); ple.eos_token_id: trunk 248044 / this (absent); ple.head_offsets: trunk [16] 0 x1, 20000003 x1, 40000026 x1, ... / this (absent); ple.head_vocab_sizes: trunk [16] 20000003 x1, 20000023 x1, 200000... / this (absent); ple.heads_per_ngram: trunk 8 / this (absent); ple.image_token_id: trunk 248056 / this (absent); ple.layer_multipliers: trunk [23703573157769, 20109073645365, 8052... / this (absent); ple.layers: trunk [1] / this (absent); ple.ngram_size: trunk 3 / this (absent); token list equal: True; token_embd trunk [2560, 248320] / this None
  config.json cross-check: 24 of 28 OK; ABSENT ple_layer_ids = [2] -> ple.layers = null; ABSENT ngram_size = 3 -> ple.ngram_size = null; ABSENT heads_per_ngram = 8 -> ple.heads_per_ngram = null; ABSENT ple_conv_kernel_size = 4 -> ple.conv_kernel = null
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
    [79e2e74] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q8_0": 1}, "ffn_gate_exps": {"Q4_K": 1}, "ffn_up_exps": {"Q4_K": 1}}
  audit: PASS, 1.766 GiB, 5.818 bpw; attention 4.919 bpw {'Q4_K': 48496640, 'Q6_K': 1310720, 'BF16': 1638400}, other 6.074 bpw {'F32': 1355008, 'Q4_K': 19742720, 'Q5_0': 9830400, 'Q6_K': 3276800}, routed_experts 5.833 bpw {'Q8_0': 838860800, 'Q4_K': 1677721600}, shared_experts 5.847 bpw {'Q8_0': 1638400, 'F32': 2560, 'Q4_K': 3276800}
  loader [ec7630a]: NOT loadable as a separate MTP model
      - token_embd.weight absent, created with flags 0 (src/models/qwen4exp.cpp:183)
      detect: src/models/qwen4exp.cpp:179
      graph: src/models/qwen4exp.cpp:307
  loader [79e2e74]: NOT loadable as a separate MTP model
      - token_embd.weight absent, created with flags 0 (src/models/qwen4exp.cpp:182)
      detect: src/models/qwen4exp.cpp:178
      graph: src/models/qwen4exp.cpp:306

--- MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf: arch qwen4exp, block_count 49, nextn_predict_layers 1, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen35, 248320 tokens; embedded chat template: True
  non-layer tensors: none
  layers present: [48]
  blk.48: 32 tensors: ['attn_k.weight', 'attn_k_norm.weight', 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'hc_attn_down.weight', 'hc_attn_inject.weight', 'hc_attn_norm.weight', 'hc_attn_up.weight', 'hc_ffn_down.weight', 'hc_ffn_inject.weight', 'hc_ffn_norm.weight', 'hc_ffn_up.weight', 'indexer.k_norm.weight', 'indexer.k_proj.weight', 'indexer.q_norm.weight', 'indexer.q_proj.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hc_head_down.weight', 'nextn.hc_head_norm.weight', 'nextn.hc_head_up.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the trunk (qwen4exp): arch keys that differ: attention.compress_ratios: trunk [48] 0 x36, 4 x12 / this [49] 0 x37, 4 x12; block_count: trunk 48 / this 49; embedding_length_per_layer_input: trunk 160 / this (absent); nextn_predict_layers: trunk (absent) / this 1; nextn_shared_target_tensors: trunk (absent) / this true; ple.conv_kernel: trunk 4 / this (absent); ple.eos_token_id: trunk 248044 / this (absent); ple.head_offsets: trunk [16] 0 x1, 20000003 x1, 40000026 x1, ... / this (absent); ple.head_vocab_sizes: trunk [16] 20000003 x1, 20000023 x1, 200000... / this (absent); ple.heads_per_ngram: trunk 8 / this (absent); ple.image_token_id: trunk 248056 / this (absent); ple.layer_multipliers: trunk [23703573157769, 20109073645365, 8052... / this (absent); ple.layers: trunk [1] / this (absent); ple.ngram_size: trunk 3 / this (absent); token list equal: True; token_embd trunk [2560, 248320] / this None
  config.json cross-check: 24 of 28 OK; ABSENT ple_layer_ids = [2] -> ple.layers = null; ABSENT ngram_size = 3 -> ple.ngram_size = null; ABSENT heads_per_ngram = 8 -> ple.heads_per_ngram = null; ABSENT ple_conv_kernel_size = 4 -> ple.conv_kernel = null
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
    [79e2e74] 16 required, missing in the file: []; 9 only inside a nested block (conditional), missing: ['qwen4exp.ple.layers', 'qwen4exp.ple.ngram_size', 'qwen4exp.ple.heads_per_ngram', 'qwen4exp.ple.conv_kernel', 'qwen4exp.ple.eos_token_id', 'qwen4exp.embedding_length_per_layer_input', 'qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes'] (qwen4exp.ple.layers src/models/qwen4exp.cpp:93, qwen4exp.ple.ngram_size src/models/qwen4exp.cpp:106, qwen4exp.ple.heads_per_ngram src/models/qwen4exp.cpp:107, qwen4exp.ple.conv_kernel src/models/qwen4exp.cpp:108, qwen4exp.ple.eos_token_id src/models/qwen4exp.cpp:109, qwen4exp.embedding_length_per_layer_input src/models/qwen4exp.cpp:112, qwen4exp.ple.layer_multipliers src/models/qwen4exp.cpp:129, qwen4exp.ple.head_offsets src/models/qwen4exp.cpp:134, qwen4exp.ple.head_vocab_sizes src/models/qwen4exp.cpp:135)
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q8_0": 1}, "ffn_gate_exps": {"Q8_0": 1}, "ffn_up_exps": {"Q8_0": 1}}
  audit: PASS, 2.585 GiB, 8.517 bpw; attention 8.739 bpw {'Q8_0': 49807360, 'BF16': 1638400}, other 9.431 bpw {'F32': 1355008, 'Q8_0': 32849920}, routed_experts 8.500 bpw {'Q8_0': 2516582400}, shared_experts 8.512 bpw {'Q8_0': 4915200, 'F32': 2560}
  loader [ec7630a]: NOT loadable as a separate MTP model
      - token_embd.weight absent, created with flags 0 (src/models/qwen4exp.cpp:183)
      detect: src/models/qwen4exp.cpp:179
      graph: src/models/qwen4exp.cpp:307
  loader [79e2e74]: NOT loadable as a separate MTP model
      - token_embd.weight absent, created with flags 0 (src/models/qwen4exp.cpp:182)
      detect: src/models/qwen4exp.cpp:178
      graph: src/models/qwen4exp.cpp:306

=== glm-5.3-flash-ggml-org-q4_k (third_party/gguf-headers/glm-5.3-flash-ggml-org-q4_k)
status ok, ggml-org/GLM-5.3-Flash-GGUF @ c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa
integrity: GLM-5.3-Flash-Q4_K-00001-of-00002.gguf OK, GLM-5.3-Flash-Q4_K-00002-of-00002.gguf OK
  GLM-5.3-Flash-Q4_K-00001-of-00002.gguf: 9429952 B, LFS sha256 7d658490f34f56ac9eb3b46ceb4778599ec39f4e1a77c63e456a1240efc64ed3, header 9429951 B (sha256 c845390f795b286f49f490291cc5ae98ad0963e5f59bf8fd819ed1eb3fbccc64), fetched 9429952 B, committed 2486966 B gz, 0 tensors, size check exact (no tensors: the writer's padding after the KV block), offsets contiguous
  GLM-5.3-Flash-Q4_K-00002-of-00002.gguf: 181363033792 B, LFS sha256 27636dd16f76d39aa95db1de081cf458f705f1c78f8f1010b2fac3edc416c116, header 86720 B (sha256 afcd96c0576ab0116fc05c35563344dd7ebd6f65063ef20e5375942235c4de3f), fetched 262144 B, committed 11512 B gz, 1383 tensors, size check exact, offsets contiguous

(e) identity
  architecture: glm5-next
  general.name: GLM-5.3-Flash
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: glm4
  vocab (tokenizer.ggml.tokens): 154880
  merges: 321649
  token_embd shape: [4096, 154880]
  output shape: [4096, 154880]
  tokenizer.ggml.add_bos_token: false
  tokenizer.ggml.add_eos_token: false
  tokenizer.ggml.bos_token_id: 154822
  tokenizer.ggml.eom_token_id: 154829
  tokenizer.ggml.eos_token_id: 154820
  tokenizer.ggml.eot_token_id: 154827
  tokenizer.ggml.padding_token_id: 154820
  tokenizer.ggml.unknown_token_id: 154820
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.file_type": 15, "general.languages": ["en", "zh"], "general.license": "mit", "general.quantization_version": 2, "general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, ...
  [ec7630a] arch glm5-next: src/llama-arch.cpp:156
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer glm4: src/llama-vocab.cpp:2290
  [79e2e74] arch glm5-next: src/llama-arch.cpp:158
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer glm4: src/llama-vocab.cpp:2407

(d) expert_group_count = 1, expert_group_used_count = 1 (keys present: [True, True]): no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)
  [ec7630a] src/llama-graph.cpp:2105
  [79e2e74] src/llama-graph.cpp:2145

(c) chat template vs third_party/hf/glm-5.3-flash-chat/chat_template.jinja: BYTE-EQUAL
  10950 B, sha256 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5

(b) hyperparameters, real vs tiny
  attention.head_count                       | real: 64 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  attention.head_count_kv                    | real: [45] 0 x34, 1 x11 | tiny default: [12] 0 x9, 1 x3 | tiny --mtp: [13] 0 x9, 1 x4 | tiny --quant-dims: [12] 0 x9, 1 x3 | tiny --dialect ik: [12] 0 x9, 1 x3
  attention.indexer.head_count               | real: 32 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2 | tiny --dialect ik: 2
  attention.indexer.key_length               | real: 128 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.indexer.kpool                    | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  attention.indexer.kpool_select_tail        | real: true | tiny default: true | tiny --mtp: true | tiny --quant-dims: true | tiny --dialect ik: true
  attention.indexer.top_k                    | real: 2048 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8 | tiny --dialect ik: 8
  attention.indexer.types                    | real: [45] true x45 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  attention.key_length                       | real: 512 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.key_length_mla                   | real: 256 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.kv_lora_rank                     | real: 512 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.layer_norm_epsilon               | real: 9.999999974752427e-07 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  attention.layer_norm_rms_epsilon           | real: 9.999999747378752e-06 | tiny default: 9.999999747378752e-06 | tiny --mtp: 9.999999747378752e-06 | tiny --quant-dims: 9.999999747378752e-06 | tiny --dialect ik: 9.999999747378752e-06
  attention.q_lora_rank                      | real: 1536 | tiny default: 32 | tiny --mtp: 32 | tiny --quant-dims: 256 | tiny --dialect ik: 32
  attention.value_length                     | real: 512 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.value_length_mla                 | real: 256 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64 | tiny --dialect ik: 16
  block_count                                | real: 45 | tiny default: 12 | tiny --mtp: 13 | tiny --quant-dims: 12 | tiny --dialect ik: 12
  context_length                             | real: 1048576 | tiny default: 512 | tiny --mtp: 512 | tiny --quant-dims: 512 | tiny --dialect ik: 512
  embedding_length                           | real: 4096 | tiny default: 64 | tiny --mtp: 64 | tiny --quant-dims: 256 | tiny --dialect ik: 64
  expert_count                               | real: 288 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8 | tiny --dialect ik: 8
  expert_feed_forward_length                 | real: 2048 | tiny default: 32 | tiny --mtp: 32 | tiny --quant-dims: 256 | tiny --dialect ik: 32
  expert_gating_func                         | real: 2 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2 | tiny --dialect ik: 2
  expert_group_count                         | real: 1 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: 1
  expert_group_used_count                    | real: 1 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  expert_shared_count                        | real: 1 | tiny default: 1 | tiny --mtp: 1 | tiny --quant-dims: 1 | tiny --dialect ik: 1
  expert_used_count                          | real: 8 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2 | tiny --dialect ik: 2
  expert_weights_norm                        | real: true | tiny default: true | tiny --mtp: true | tiny --quant-dims: true | tiny --dialect ik: true
  expert_weights_scale                       | real: 2.5 | tiny default: 2.5 | tiny --mtp: 2.5 | tiny --quant-dims: 2.5 | tiny --dialect ik: 2.5
  feed_forward_length                        | real: 12288 | tiny default: 96 | tiny --mtp: 96 | tiny --quant-dims: 256 | tiny --dialect ik: 96
  hyper_connection.count                     | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  hyper_connection.epsilon                   | real: 9.999999974752427e-07 | tiny default: 9.999999974752427e-07 | tiny --mtp: 9.999999974752427e-07 | tiny --quant-dims: 9.999999974752427e-07 | tiny --dialect ik: 9.999999974752427e-07
  hyper_connection.sinkhorn_iterations       | real: 20 | tiny default: 20 | tiny --mtp: 20 | tiny --quant-dims: 20 | tiny --dialect ik: 20
  kda.gate_lower_bound                       | real: -5.0 | tiny default: -5.0 | tiny --mtp: -5.0 | tiny --quant-dims: -5.0 | tiny --dialect ik: -5.0
  kda.head_dim                               | real: 128 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64 | tiny --dialect ik: 16
  leading_dense_block_count                  | real: 3 | tiny default: 1 | tiny --mtp: 1 | tiny --quant-dims: 1 | tiny --dialect ik: 1
  nextn_predict_layers                       | real: (absent) | tiny default: (absent) | tiny --mtp: 1 | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  rope.dimension_count                       | real: 0 | tiny default: 0 | tiny --mtp: 0 | tiny --quant-dims: 0 | tiny --dialect ik: 0
  ssm.conv_kernel                            | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  swiglu_clamp_exp                           | real: [45] 10.0 x45 | tiny default: [12] 0.5 x12 | tiny --mtp: [13] 0.5 x13 | tiny --quant-dims: [12] 0.5 x12 | tiny --dialect ik: [12] 0.5 x12
  swiglu_clamp_shexp                         | real: [45] 10.0 x45 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  vocab_size                                 | real: 154880 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  config.json cross-check:
    OK     num_hidden_layers = 45  ->  layers = 45
    ABSENT num_nextn_predict_layers = 1  ->  nextn_predict_layers = null
    OK     hidden_size = 4096  ->  embedding_length = 4096
    OK     intermediate_size = 12288  ->  feed_forward_length = 12288
    OK     moe_intermediate_size = 2048  ->  expert_feed_forward_length = 2048
    OK     n_routed_experts = 288  ->  expert_count = 288
    OK     num_experts_per_tok = 8  ->  expert_used_count = 8
    OK     n_shared_experts = 1  ->  expert_shared_count = 1
    OK     first_k_dense_replace = 3  ->  leading_dense_block_count = 3
    OK     routed_scaling_factor = 2.5  ->  expert_weights_scale = 2.5
    OK     norm_topk_prob = true  ->  expert_weights_norm = true
    OK     n_group = 1  ->  expert_group_count = 1
    OK     topk_group = 1  ->  expert_group_used_count = 1
    OK     swiglu_limit = 10.0  ->  swiglu_clamp_exp = [45] 10.0 x45
    OK     num_attention_heads = 64  ->  attention.head_count = 64
    OK     q_lora_rank = 1536  ->  attention.q_lora_rank = 1536
    OK     kv_lora_rank = 512  ->  attention.kv_lora_rank = 512
    OK     qk_head_dim = 256  ->  attention.key_length_mla = 256
    OK     v_head_dim = 256  ->  attention.value_length_mla = 256
    OK     qk_rope_head_dim = 0  ->  rope.dimension_count = 0
    OK     rms_norm_eps = 1e-05  ->  attention.layer_norm_rms_epsilon = 9.999999747378752e-06
    OK     index_n_heads = 32  ->  attention.indexer.head_count = 32
    OK     index_head_dim = 128  ->  attention.indexer.key_length = 128
    OK     index_topk = 2048  ->  attention.indexer.top_k = 2048
    OK     index_kpool = 4  ->  attention.indexer.kpool = 4
    OK     index_kpool_always_select_tail = true  ->  attention.indexer.kpool_select_tail = true
    OK     hc_mult = 4  ->  hyper_connection.count = 4
    OK     hc_sinkhorn_iters = 20  ->  hyper_connection.sinkhorn_iterations = 20
    OK     hc_eps = 1e-06  ->  hyper_connection.epsilon = 9.999999974752427e-07
    OK     max_position_embeddings = 1048576  ->  context_length = 1048576
    OK     linear_attn_config.head_dim = 128  ->  kda.head_dim = 128
    OK     linear_attn_config.gate_lower_bound = -5.0  ->  kda.gate_lower_bound = -5.0
    OK     linear_attn_config.short_conv_kernel_size = 4  ->  ssm.conv_kernel = 4
    OK     vocab_size = 154880  ->  @vocab = {"tokens": 154880, "token_embd rows": 154880}
  graph branches:
    [tiny OK] glm.kda_gate_bound: KDA decay gate: bounded sigmoid vs softplus
       real: ['bounded sigmoid decay']  kda.gate_lower_bound = -5.0
       tiny: default: ['bounded sigmoid decay']; --mtp: ['bounded sigmoid decay']; --quant-dims: ['bounded sigmoid decay']; --dialect ik: ['bounded sigmoid decay']
       [ec7630a] src/models/glm5-next.cpp:16; src/models/glm5-next.cpp:714
       [79e2e74] src/models/glm5-next.cpp:16; src/models/glm5-next.cpp:832
    [tiny OK] glm.layer_pattern: KDA (head_count_kv 0) vs DSA layers
       real: ['KDA + DSA every 4th layer (3, 7, ...)']  DSA layers [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43] of 45
       tiny: default: ['KDA + DSA every 4th layer (3, 7, ...)']; --mtp: ['KDA + DSA every 4th layer (3, 7, ...)']; --quant-dims: ['KDA + DSA every 4th layer (3, 7, ...)']; --dialect ik: ['KDA + DSA every 4th layer (3, 7, ...)']
       [ec7630a] src/models/glm5-next.cpp:22
       [79e2e74] src/models/glm5-next.cpp:22
    [NOT IN TINY] glm.indexer_types: DSA indexer full per layer vs shared with the previous DSA layer
       real: ['full indexer']  attention.indexer.types = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
       tiny: default: ['full indexer in every DSA layer']; --mtp: ['full indexer in every DSA layer']; --quant-dims: ['full indexer in every DSA layer']; --dialect ik: ['full indexer in every DSA layer']
       [ec7630a] src/models/glm5-next.cpp:47; src/models/glm5-next.cpp:958
       [79e2e74] src/models/glm5-next.cpp:47; src/models/glm5-next.cpp:1059
    [tiny OK] glm.kpool_select_tail: DSA selection appends the incomplete k-pool tail
       real: ['tail appended']  kpool_select_tail = True
       tiny: default: ['tail appended']; --mtp: ['tail appended']; --quant-dims: ['tail appended']; --dialect ik: ['tail appended']
       [ec7630a] src/models/glm5-next.cpp:875
       [79e2e74] src/models/glm5-next.cpp:983
    [tiny OK] glm.dsa_threshold: sparse DSA selection only above top_k context tokens; at ec7630a ubatches <= 16 gather the selection
       real: ['selection active above top_k']  top_k 2048, kpool 4: at most 2051 selected cells per token (ec7630a gathers them)
       tiny: default: ['selection active above top_k']; --mtp: ['selection active above top_k']; --quant-dims: ['selection active above top_k']; --dialect ik: ['selection active above top_k']
       [ec7630a] src/models/glm5-next.cpp:328,858; src/models/glm5-next.cpp:331
       [79e2e74] src/models/glm5-next.cpp:320,966; not found
    [tiny OK] glm.rope: MLA must be nope-only (rope.dimension_count 0)
       real: ['n_rot 0 (nope MLA)']  rope.dimension_count = 0, key_length = 512
       tiny: default: ['n_rot 0 (nope MLA)']; --mtp: ['n_rot 0 (nope MLA)']; --quant-dims: ['n_rot 0 (nope MLA)']; --dialect ik: ['n_rot 0 (nope MLA)']
       [ec7630a] src/models/glm5-next.cpp:935; src/llama-model.cpp:1448
       [79e2e74] src/models/glm5-next.cpp:1036; src/llama-model.cpp:1476
    [tiny OK] glm.hc: mHC with 4 streams only
       real: ['hc 4']  sinkhorn 20, eps 9.999999974752427e-07
       tiny: default: ['hc 4']; --mtp: ['hc 4']; --quant-dims: ['hc 4']; --dialect ik: ['hc 4']
       [ec7630a] src/models/glm5-next.cpp:53
       [79e2e74] src/models/glm5-next.cpp:53
    [tiny OK] glm.dense_lead: leading dense FFN layers, then MoE + shared expert
       real: ['MoE + shared expert', 'dense FFN']  leading_dense_block_count = 3
       tiny: default: ['MoE + shared expert', 'dense FFN']; --mtp: ['MoE + shared expert', 'dense FFN']; --quant-dims: ['MoE + shared expert', 'dense FFN']; --dialect ik: ['MoE + shared expert', 'dense FFN']
       [ec7630a] src/models/glm5-next.cpp:616
       [79e2e74] src/models/glm5-next.cpp:735
    [tiny OK] glm.gating: router gating function (absent or none = sigmoid)
       real: ['sigmoid']  expert_gating_func = 2
       tiny: default: ['sigmoid']; --mtp: ['sigmoid']; --quant-dims: ['sigmoid']; --dialect ik: ['sigmoid']
       [ec7630a] src/models/glm5-next.cpp:31
       [79e2e74] src/models/glm5-next.cpp:31
    [NOT IN TINY] glm.shexp_clamp: shared-expert SwiGLU clamp: own key or the routed one
       real: ['own shexp clamp key']  swiglu_clamp_shexp = [10.0]
       tiny: default: ['shexp clamp = routed clamp']; --mtp: ['shexp clamp = routed clamp']; --quant-dims: ['shexp clamp = routed clamp']; --dialect ik: ['shexp clamp = routed clamp']
       [ec7630a] src/models/glm5-next.cpp:35; src/llama-graph.cpp:1853
       [79e2e74] src/models/glm5-next.cpp:35; src/llama-graph.cpp:1888
    [NOT IN TINY] glm.layer_norm_eps: LayerNorm eps of the indexer key norm (key absent: 0)
       real: ['eps 9.999999974752427e-07']  
       tiny: default: ['eps 0 (key absent)']; --mtp: ['eps 0 (key absent)']; --quant-dims: ['eps 0 (key absent)']; --dialect ik: ['eps 0 (key absent)']
       [ec7630a] src/models/glm5-next.cpp:9; src/llama-graph.cpp:1612
       [79e2e74] src/models/glm5-next.cpp:9; src/llama-graph.cpp:1647
    [tiny OK] glm.ff_exp_array: expert FFN width: one value or per layer
       real: ['scalar']  expert_feed_forward_length = 2048
       tiny: default: ['scalar']; --mtp: ['scalar']; --quant-dims: ['scalar']; --dialect ik: ['scalar']
       [ec7630a] src/models/glm5-next.cpp:25
       [79e2e74] src/models/glm5-next.cpp:25
    [tiny OK] glm.kda_qkv: KDA q/k/v projections: separate (the graph reads wq/wk/wv) or fused attn_qkv
       real: ['separate attn_q/k/v']  
       tiny: default: ['separate attn_q/k/v']; --mtp: ['separate attn_q/k/v']; --quant-dims: ['separate attn_q/k/v']; --dialect ik: ['separate attn_q/k/v']
       [ec7630a] src/models/glm5-next.cpp:109; src/llama-model.cpp:3405
       [79e2e74] src/models/glm5-next.cpp:107; src/llama-model.cpp:3447
    [tiny OK] glm.conv_rank: KDA conv1d weights 4-D {d_conv,1,d_inner,1} or 3-D
       real: ['3-D conv1d']  
       tiny: default: ['3-D conv1d']; --mtp: ['3-D conv1d']; --quant-dims: ['3-D conv1d']; --dialect ik: ['3-D conv1d']
       [ec7630a] src/models/glm5-next.cpp:102
       [79e2e74] src/models/glm5-next.cpp:100
    [tiny OK] glm.mtp: MTP (NextN) block: count; the MTP graph exists at 79e2e74 only
       real: ['nextn_predict_layers 0']  
       tiny: default: ['nextn_predict_layers 0']; --mtp: ['nextn_predict_layers 1']; --quant-dims: ['nextn_predict_layers 0']; --dialect ik: ['nextn_predict_layers 0']
       [ec7630a] not found
       [79e2e74] src/models/glm5-next.cpp:547
    [tiny OK] glm.mtp_heads: MTP block's own embedding / head / head norm instead of the trunk's
       real: ['trunk embedding, head and norm']  
       tiny: default: ['trunk embedding, head and norm']; --mtp: ['trunk embedding, head and norm']; --quant-dims: ['trunk embedding, head and norm']; --dialect ik: ['trunk embedding, head and norm']
       [ec7630a] not found; not found; not found
       [79e2e74] src/models/glm5-next.cpp:584; src/models/glm5-next.cpp:647; src/models/glm5-next.cpp:658
    [tiny OK] moe.groups: group-limited routing (expert_group_count > 1)
       real: ['no group step']  expert_group_count = 1, used = 1
       tiny: default: ['no group step']; --mtp: ['no group step']; --quant-dims: ['no group step']; --dialect ik: ['no group step']
       [ec7630a] src/llama-model.cpp:1334; src/llama-graph.cpp:2105
       [79e2e74] src/llama-model.cpp:1362; src/llama-graph.cpp:2145
    [tiny OK] moe.norm_w: renormalise the selected expert weights
       real: ['renormalised']  expert_weights_norm = True
       tiny: default: ['renormalised']; --mtp: ['renormalised']; --quant-dims: ['renormalised']; --dialect ik: ['renormalised']
       [ec7630a] src/llama-graph.cpp:2156
       [79e2e74] src/llama-graph.cpp:2196
    [tiny OK] moe.scale: routed weights scaled (scale not 0 or 1)
       real: ['scaled']  expert_weights_scale = 2.5
       tiny: default: ['scaled']; --mtp: ['scaled']; --quant-dims: ['scaled']; --dialect ik: ['scaled']
       [ec7630a] src/llama-graph.cpp:2171
       [79e2e74] src/llama-graph.cpp:2211
    [tiny OK] moe.clamp_glm: routed SwiGLU clamp (glm5-next: fused swiglu_clamp)
       real: ['fused swiglu_clamp']  swiglu_clamp_exp = [10.0]
       tiny: default: ['fused swiglu_clamp']; --mtp: ['fused swiglu_clamp']; --quant-dims: ['fused swiglu_clamp']; --dialect ik: ['fused swiglu_clamp']
       [ec7630a] src/llama-graph.cpp:2247
       [79e2e74] src/llama-graph.cpp:2290
    [tiny OK] moe.used_array: experts per token: one value or per layer
       real: ['scalar']  expert_used_count = 8, expert_count = 288
       tiny: default: ['scalar']; --mtp: ['scalar']; --quant-dims: ['scalar']; --dialect ik: ['scalar']
       [ec7630a] src/llama-model.cpp:1333
       [79e2e74] src/llama-model.cpp:1361
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 20 required, missing in the file: []; none conditional
    [79e2e74] 20 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q4_K": 42}, "ffn_gate_exps": {"Q4_K": 42}, "ffn_up_exps": {"Q4_K": 42}}

(a) tools/gguf_quant_audit.py on the header-only shards (files total 181372463744 B = 168.916 GiB)
  # GLM-5.3-Flash-Q4_K-00001-of-00002.gguf (2 file(s)), architecture 'glm5-next'
  total 313.33B params, 168.91 GiB, 4.631 bpw
    routed_experts   304.406B    159.47 GiB   4.500 bpw  [Q4_K 100.0%]
    shared_experts     1.057B      1.05 GiB   8.500 bpw  [Q8_0 100.0%]
    attention          6.057B      6.37 GiB   9.035 bpw  [Q8_0 93.2%, BF16 6.6%, F32 0.2%]
    dense_ffn          0.453B      0.45 GiB   8.500 bpw  [Q8_0 100.0%]
    embedding          0.634B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    output_head        0.634B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    other              0.085B      0.32 GiB  32.000 bpw  [F32 100.0%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: blk.10.ffn_down_exps.weight Q4_K 4.500, blk.10.ffn_gate_exps.weight Q4_K 4.500, blk.10.ffn_up_exps.weight Q4_K 4.500, blk.11.ffn_down_exps.weight Q4_K 4.500, blk.11.ffn_gate_exps.weight Q4_K 4.500, blk.11.ffn_up_exps.weight Q4_K 4.500

=== glm-5.3-flash-ggml-org-mtp (third_party/gguf-headers/glm-5.3-flash-ggml-org-mtp)
status ok, ggml-org/GLM-5.3-Flash-GGUF @ c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa
integrity: mtp-GLM-5.3-Flash-Q4_0.gguf OK, mtp-GLM-5.3-Flash-Q8_0.gguf OK
mode mtp_files; NextN reference: request glm-5.3-flash-ud-q4_k_xl, blk.45; trunk request: glm-5.3-flash-ggml-org-q4_k
  mtp-GLM-5.3-Flash-Q4_0.gguf: 4947287104 B, LFS sha256 7a36bad56b6b11542f898d660821a82a580eb6d319c0380566fb737e3bb2ad78, header 9432000 B (sha256 54c831d97a0dffcc1ec4eae4f697a51754026dcdc611151a28699d13a3887d10), fetched 16777216 B, committed 2487517 B gz, 32 tensors, size check exact, offsets contiguous
  mtp-GLM-5.3-Flash-Q8_0.gguf: 9284197440 B, LFS sha256 17a89535ce31a2e6aebbf50eb44ba850ec0e8263805d97004c85ac3ab06c636f, header 9432000 B (sha256 5518d3c4d9de9c29cc28f6972789e10719110b95fd6af4dd8e46130284df0c74), fetched 16777216 B, committed 2487531 B gz, 32 tensors, size check exact, offsets contiguous

how llama.cpp attaches a separate MTP file:
  [ec7630a] common/arg.cpp:4241: a local MTP file is given as the draft model: -md FILE (sets speculative.draft.mparams.path)
  [ec7630a] common/arg.cpp:4249: --spec-type draft-mtp selects the MTP drafter
  [ec7630a] common/speculative.cpp:37: the type name draft-mtp
  [ec7630a] common/arg.cpp:3084: --mtp (download example): also fetch the repo's MTP sidecar
  [ec7630a] common/download.cpp:616: -hf: the sidecar is the sibling file whose name starts with mtp-
  [ec7630a] common/download.cpp:649: an mtp- file is never taken as the model
  [ec7630a] common/arg.cpp:644: the downloaded sidecar becomes the draft model path
  [ec7630a] common/speculative.cpp:2539: with a draft path the MTP file is loaded as its own llama_model (llama_model_load_from_file)
  [ec7630a] common/speculative.cpp:2526: its context is an MTP context
  [ec7630a] common/speculative.cpp:2560: without a draft path the MTP context runs on the target model's own NextN layers
  [ec7630a] common/speculative.cpp:1376: the MTP model's output width must equal the target's (h_nextn rows)
  [ec7630a] common/common.cpp:1625: draft-mtp sets load_mtp (otherwise NextN tensors are skipped)
  [ec7630a] src/llama-vocab.cpp:1944: a separate model file needs its own tokenizer metadata (tokenizer.ggml.model is required)
  [ec7630a] src/llama-model-loader.cpp:1398: every tensor in the file must be created by the arch loader
  [ec7630a] src/llama-context.cpp:156: only EAGLE3 / DFlash drafts borrow the target's embeddings and head through ctx_other
  [ec7630a] src/llama-model.cpp:2802,2809: an MTP context's KV cache holds only the NextN layers
  [79e2e74] common/arg.cpp:4276: a local MTP file is given as the draft model: -md FILE (sets speculative.draft.mparams.path)
  [79e2e74] common/arg.cpp:4284: --spec-type draft-mtp selects the MTP drafter
  [79e2e74] common/speculative.cpp:76: the type name draft-mtp
  [79e2e74] common/arg.cpp:3097: --mtp (download example): also fetch the repo's MTP sidecar
  [79e2e74] common/download.cpp:605: -hf: the sidecar is the sibling file whose name starts with mtp-
  [79e2e74] common/download.cpp:638: an mtp- file is never taken as the model
  [79e2e74] common/arg.cpp:644: the downloaded sidecar becomes the draft model path
  [79e2e74] common/speculative.cpp:2623: with a draft path the MTP file is loaded as its own llama_model (llama_model_load_from_file)
  [79e2e74] common/speculative.cpp:2610: its context is an MTP context
  [79e2e74] common/speculative.cpp:2644: without a draft path the MTP context runs on the target model's own NextN layers
  [79e2e74] common/speculative.cpp:1437: the MTP model's output width must equal the target's (h_nextn rows)
  [79e2e74] common/common.cpp:1690: draft-mtp sets load_mtp (otherwise NextN tensors are skipped)
  [79e2e74] src/llama-vocab.cpp:2057: a separate model file needs its own tokenizer metadata (tokenizer.ggml.model is required)
  [79e2e74] src/llama-model-loader.cpp:1398: every tensor in the file must be created by the arch loader
  [79e2e74] src/llama-context.cpp:159: only EAGLE3 / DFlash drafts borrow the target's embeddings and head through ctx_other
  [79e2e74] src/llama-model.cpp:2841,2848: an MTP context's KV cache holds only the NextN layers

--- mtp-GLM-5.3-Flash-Q4_0.gguf: arch glm5-next, block_count 46, nextn_predict_layers 1, n_layer 45; layout: MTP-only file
  tokenizer: model gpt2, pre glm4, 154880 tokens; embedded chat template: True
  non-layer tensors: output.weight Q4_0 [4096, 154880], output_norm.weight F32 [4096], token_embd.weight Q4_0 [4096, 154880]
  layers present: [45]
  blk.45: 29 tensors: ['attn_k_b.weight', 'attn_kv_a_mqa.weight', 'attn_kv_a_norm.weight', 'attn_norm.weight', 'attn_output.weight', 'attn_q_a.weight', 'attn_q_a_norm.weight', 'attn_q_b.weight', 'attn_v_b.weight', 'exp_probs_b.bias', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_shexp.weight', 'ffn_norm.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'indexer.attn_k.weight', 'indexer.attn_q_b.weight', 'indexer.k_norm.bias', 'indexer.k_norm.weight', 'indexer.proj.weight', 'indexer_compressor_ape.weight', 'indexer_compressor_gate.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight']
  vs the reference NextN layer: blk.45 missing [], missing optional [], extra []
  vs the trunk (glm5-next): arch keys that differ: attention.head_count_kv: trunk [45] 0 x34, 1 x11 / this [46] 0 x34, 1 x12; block_count: trunk 45 / this 46; nextn_predict_layers: trunk (absent) / this 1; swiglu_clamp_exp: trunk [45] 10.0 x45 / this [46] 10.0 x46; swiglu_clamp_shexp: trunk [45] 10.0 x45 / this [46] 10.0 x46; token list equal: True; token_embd trunk [4096, 154880] / this [4096, 154880]
  config.json cross-check: 34 of 34 OK
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 20 required, missing in the file: []; none conditional
    [79e2e74] 20 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q4_0": 1}, "ffn_gate_exps": {"Q4_0": 1}, "ffn_up_exps": {"Q4_0": 1}}
  audit: PASS, 4.599 GiB, 4.540 bpw; output_head 4.500 bpw {'Q4_0': 634388480}, other 5.455 bpw {'F32': 1206816, 'Q4_0': 33554432}, embedding 4.500 bpw {'Q4_0': 634388480}, attention 7.010 bpw {'BF16': 25690112, 'Q4_0': 98566144, 'F32': 655872}, routed_experts 4.500 bpw {'Q4_0': 7247757312}, shared_experts 4.500 bpw {'Q4_0': 25165824}
  loader [ec7630a]: NOT loadable as a separate MTP model
      - no MTP-only handling (src/models/glm5-next.cpp:68): every trunk tensor is required, the file has none (blk.0.attn_norm.weight absent)
      - the MTP graph is not implemented (src/models/glm5-next.cpp:190)
      no_detect: src/models/glm5-next.cpp:68
      no_graph: src/models/glm5-next.cpp:190
  loader [79e2e74]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/glm5-next.cpp:67; src/llama-model.cpp:3491
      graph: src/models/glm5-next.cpp:188

--- mtp-GLM-5.3-Flash-Q8_0.gguf: arch glm5-next, block_count 46, nextn_predict_layers 1, n_layer 45; layout: MTP-only file
  tokenizer: model gpt2, pre glm4, 154880 tokens; embedded chat template: True
  non-layer tensors: output.weight Q8_0 [4096, 154880], output_norm.weight F32 [4096], token_embd.weight Q8_0 [4096, 154880]
  layers present: [45]
  blk.45: 29 tensors: ['attn_k_b.weight', 'attn_kv_a_mqa.weight', 'attn_kv_a_norm.weight', 'attn_norm.weight', 'attn_output.weight', 'attn_q_a.weight', 'attn_q_a_norm.weight', 'attn_q_b.weight', 'attn_v_b.weight', 'exp_probs_b.bias', 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight', 'ffn_gate_inp.weight', 'ffn_gate_shexp.weight', 'ffn_norm.weight', 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'indexer.attn_k.weight', 'indexer.attn_q_b.weight', 'indexer.k_norm.bias', 'indexer.k_norm.weight', 'indexer.proj.weight', 'indexer_compressor_ape.weight', 'indexer_compressor_gate.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight']
  vs the reference NextN layer: blk.45 missing [], missing optional [], extra []
  vs the trunk (glm5-next): arch keys that differ: attention.head_count_kv: trunk [45] 0 x34, 1 x11 / this [46] 0 x34, 1 x12; block_count: trunk 45 / this 46; nextn_predict_layers: trunk (absent) / this 1; swiglu_clamp_exp: trunk [45] 10.0 x45 / this [46] 10.0 x46; swiglu_clamp_shexp: trunk [45] 10.0 x45 / this [46] 10.0 x46; token list equal: True; token_embd trunk [4096, 154880] / this [4096, 154880]
  config.json cross-check: 34 of 34 OK
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 20 required, missing in the file: []; none conditional
    [79e2e74] 20 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {"ffn_down_exps": {"Q8_0": 1}, "ffn_gate_exps": {"Q8_0": 1}, "ffn_up_exps": {"Q8_0": 1}}
  audit: PASS, 8.638 GiB, 8.527 bpw; output_head 8.500 bpw {'Q8_0': 634388480}, other 9.316 bpw {'F32': 1206816, 'Q8_0': 33554432}, embedding 8.500 bpw {'Q8_0': 634388480}, attention 10.166 bpw {'BF16': 25690112, 'Q8_0': 98566144, 'F32': 655872}, routed_experts 8.500 bpw {'Q8_0': 7247757312}, shared_experts 8.500 bpw {'Q8_0': 25165824}
  loader [ec7630a]: NOT loadable as a separate MTP model
      - no MTP-only handling (src/models/glm5-next.cpp:68): every trunk tensor is required, the file has none (blk.0.attn_norm.weight absent)
      - the MTP graph is not implemented (src/models/glm5-next.cpp:190)
      no_detect: src/models/glm5-next.cpp:68
      no_graph: src/models/glm5-next.cpp:190
  loader [79e2e74]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/glm5-next.cpp:67; src/llama-model.cpp:3491
      graph: src/models/glm5-next.cpp:188

=== discover-mimo (third_party/gguf-headers/discover-mimo)
named repo ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @ e6d4bced739310a721eefad4c247ec7e0247c89d (requested main): ok, arch=mimo2, lastModified=2026-10-11T07:58:04.000Z, 13 .gguf, 391.5 GiB
    MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf  5948640 B = 0.01 GiB  lfs a353e75ccd30
    MiMo-V2.6-Flash-MOPD-MXFP4-00002-of-00002.gguf  167363516512 B = 155.87 GiB  lfs 3adf8ea80157
    MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf  5948928 B = 0.01 GiB  lfs 612ae7fd2064
    MiMo-V2.6-Flash-MOPD-Q2_K-00002-of-00002.gguf  126360000608 B = 117.68 GiB  lfs 637921c1b5b5
    MiMo-V2.6-Flash-MOPD-Q2_K_S-00001-of-00002.gguf  5948928 B = 0.01 GiB  lfs ad8863179232
    MiMo-V2.6-Flash-MOPD-Q2_K_S-00002-of-00002.gguf  105858242656 B = 98.59 GiB  lfs 4f21f6083146
    dflash-MiMo-V2.6-Flash-MOPD-BF16.gguf  5441972448 B = 5.07 GiB  lfs c909b7c0d0f1
    dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  2893932768 B = 2.70 GiB  lfs cfd70fef40bc
    mmproj-MiMo-V2.6-Flash-MOPD-BF16.gguf  2748509792 B = 2.56 GiB  lfs 9b7d98d8afe2
    mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  1564438112 B = 1.46 GiB  lfs 7fa58c0f2034
    mtp-MiMo-V2.6-Flash-MOPD-BF16.gguf  4481536224 B = 4.17 GiB  lfs d4292135824f
    mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf  1264898272 B = 1.18 GiB  lfs bc9914785441
    mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  2383728864 B = 2.22 GiB  lfs 4589839383ca
named repo ggml-org/MiMo-V2.6-Flash-RL-GGUF @ b0b2d0fb0b56611c761e075425c88989faf1ed1f (requested main): ok, arch=mimo2, lastModified=2026-10-11T08:23:14.000Z, 11 .gguf, 292.9 GiB
    MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf  5948640 B = 0.01 GiB  lfs a34f63df3208
    MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf  167363516512 B = 155.87 GiB  lfs 10b74ff85577
    MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf  5948640 B = 0.01 GiB  lfs a34f63df3208
    MiMo-V2.6-Flash-RL-Q2_K-00002-of-00002.gguf  126360000608 B = 117.68 GiB  lfs d037fe34f375
    dflash-MiMo-V2.6-Flash-RL-BF16.gguf  5441972448 B = 5.07 GiB  lfs 02bac7c24be2
    dflash-MiMo-V2.6-Flash-RL-Q8_0.gguf  2893932768 B = 2.70 GiB  lfs 54e787fb5c89
    mmproj-MiMo-V2.6-Flash-RL-BF16.gguf  2748509792 B = 2.56 GiB  lfs 1561e6127de6
    mmproj-MiMo-V2.6-Flash-RL-Q8_0.gguf  1564438112 B = 1.46 GiB  lfs 9f3aeb85745e
    mtp-MiMo-V2.6-Flash-RL-BF16.gguf  4481536224 B = 4.17 GiB  lfs 4719f59c5907
    mtp-MiMo-V2.6-Flash-RL-Q4_0.gguf  1264898272 B = 1.18 GiB  lfs 5318c079bf64
    mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf  2383728864 B = 2.22 GiB  lfs cbcd8a22e627
named repo QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ ac33b3743d70ed57a8cc6b847698d2ebbddb7784 (requested main): ok, arch=mimo2, lastModified=2026-09-27T20:57:35.000Z, 4 .gguf, 169.5 GiB
    MiMo-V2.6-Flash-MOPD-MXFP4.gguf  174908254272 B = 162.90 GiB  lfs fc1b8122d1a3
    mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf  1564438112 B = 1.46 GiB  lfs 156c3891b52d
    mmproj-MiMo-V2.6-Flash-MOPD-bf16.gguf  2748509792 B = 2.56 GiB  lfs 8524ec7a6ce4
    mmproj-MiMo-V2.6-Flash-MOPD-f16.gguf  2744577632 B = 2.56 GiB  lfs 973931512a70
named repo kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ 44d9191bb7a95125c8b971ce0593bd802ae6dd48 (requested main): ok, arch=mimo2, lastModified=2026-10-04T03:08:13.000Z, 4 .gguf, 162.3 GiB
    MiMo-V2.6-Flash-MOPD-DFlash-Q8_0.gguf  1559980992 B = 1.45 GiB  lfs 336b16e8f7ea
    MiMo-V2.6-Flash-MOPD-MXFP4.gguf  169005098944 B = 157.40 GiB  lfs 5e5dfb6d9646
    MiMo-V2.6-Flash-MOPD-Vision-F32.gguf  2914399616 B = 2.71 GiB  lfs 5eaa8193dcfc
    MiMo-V2.6-Flash-MOPD-Vision-Q8_0.gguf  775591296 B = 0.72 GiB  lfs 4499a8191bf5
named repo unsloth/MiMo-V2.6-Flash-MOPD-GGUF @ None (requested main): repo_unavailable, HTTP 401
named repo unsloth/MiMo-V2.6-Flash-GGUF @ None (requested main): repo_unavailable, HTTP 401
search 'MiMo-V2.6-Flash' (architectures []): 30 repos
    ProCreations/MiMo-V2.6-Flash-RL-NVFP4  arch=mimo2  downloads=11177  sha=34cebc5d7574
    ggml-org/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=15855  sha=b0b2d0fb0b56
    MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-GGUF  arch=mimo2  downloads=344  sha=fbbb29584eb9
    MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF  arch=mimo2  downloads=587  sha=0ecc6864d187
    pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF  arch=mimo2  downloads=2853  sha=a37b69042dbe
    patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF  arch=mimo2  downloads=6615  sha=8d0abe0abfbe
    AesSedai/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=1669  sha=b6e95d58c343
    ProCreations/MiMo-V2.6-Flash-RL-IQ3_XXS-GGUF  arch=mimo2  downloads=712  sha=9b2bf573c44e
    TrevorJS/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=1442  sha=dce9d8c8e576
    Baekpica/MiMo-V2.6-Flash-RL-Mixed-Quant-GGUF  arch=mimo2  downloads=3673  sha=7cbefe60ca47
    Baekpica/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=840  sha=a485ac9ebf51
    kernelpool/MiMo-V2.6-Flash-MXFP4-GGUF  arch=mimo2  downloads=571  sha=f4a1c0708d4a
    jedisct1/MiMo-V2.6-Flash-RL-GGUF-100K  arch=mimo2  downloads=240  sha=a207209263a0
    YanissAmz/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=1277  sha=e7a3bd785397
    AliceThirty/MiMo-V2.6-Flash-RL-UNCENSORED-gguf  arch=mimo2  downloads=2028  sha=5cc3ec2959c0
    0xSojalSec/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF  arch=mimo2  downloads=153  sha=c47289f56ace
    QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF  arch=mimo2  downloads=48815  sha=ac33b3743d70
    ProCreations/MiMo-V2.6-Flash-MOPD-NVFP4  arch=mimo2  downloads=1204  sha=da654d8ed8d0
    ddh0/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=1200  sha=b9e3e8a3370a
    kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF  arch=mimo2  downloads=602  sha=44d9191bb7a9
    ggml-org/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=5621  sha=e6d4bced7393
    webmp3/Sakura-MiMo-V2.6-Flash-MOPD-P160-GGUF  arch=mimo2  downloads=691  sha=a0d8ac025575
    fresherbz/MiMo-V2.6-Flash-MOPD-IQ2_XXS-GGUF  arch=mimo2  downloads=204  sha=b393fdeb4476
    Baekpica/MiMo-V2.6-Flash-MOPD-Mixed-Quant-GGUF  arch=mimo2  downloads=554  sha=c2de39b1ac00
    AMAImedia/MiMo-V2.6-Flash-MOPD-BPW2.5-GGUF  arch=mimo2  downloads=335  sha=a98d84c2f711
    CarnaK/MiMo-V2.6-Flash-RL-Heretic-MTP-Q3_K-GGUF  arch=None  downloads=0  sha=4d21819d3ede
    cafonez/MiMo-V2.6-Flash-RL-UNCENSORED-Gorgon-GGUF  arch=mimo2  downloads=224  sha=998bdd418f7a
    kernelpool/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=286  sha=3ce46a094a33
    spiritfather/MiMo-V2.6-Flash-MOPD-Yamz-Uncensored-GGUF  arch=mimo2  downloads=1180  sha=f1adc2e15d07
    petr567/MiMo-V2.6-Flash-MOPD-Strix-Halo-Recipe  arch=None  downloads=0  sha=29500f0a8910
```

## Appendix B. `tools/gguf_headers/analyze.py` on the MiMo headers (round 3), full output

At `bc26480`:

```
$ export TMPDIR=<S>/n18/tmp; cd <S>/n18/r3 && git -C /home/user/wt-n18 log --oneline -n 1 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 /home/user/wt-n18/tools/gguf_headers/analyze.py --src ec7630a=<S>/llama-ec7630a --src 79e2e74=<S>/llama.cpp --name mimo-v2.6-flash-mopd-ggml-org-mxfp4 --name mimo-v2.6-flash-mopd-ggml-org-mtp --name mimo-v2.6-flash-mopd-quantaplanta-mxfp4 --name mimo-v2.6-flash-mopd-kernelpool-mxfp4 --json analyze.json > analyze.txt; echo "exit $?"; wc -l analyze.txt
bc26480 [model-analyst] tests: facts of the N18 MiMo headers (bot 6161763): config cross-check, MXFP4 experts and D5, the three MXFP4 files and the mtp file compared
exit 0
570 analyze.txt
$ cat <S>/n18/r3/analyze.txt
sources: ec7630a = ec7630a640789c393694fb194f1bbbf0369fc62d (<S>/llama-ec7630a), 79e2e74 = 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (<S>/llama.cpp)

=== mimo-v2.6-flash-mopd-ggml-org-mxfp4 (third_party/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mxfp4)
status ok, ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @ e6d4bced739310a721eefad4c247ec7e0247c89d
integrity: MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf OK, MiMo-V2.6-Flash-MOPD-MXFP4-00002-of-00002.gguf OK
  MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf: 5948640 B, LFS sha256 a353e75ccd30f347dfd68a8cf1be8be0fb989fcca94b8e72fd7d1e35241b5f51, header 5948636 B (sha256 f2e9498e281706e33120fa0fef43caaff9e84cf133ee2839d8830b6e8c72b213), fetched 5948640 B, committed 1661133 B gz, 0 tensors, size check exact (no tensors: the writer's padding after the KV block), offsets contiguous
  MiMo-V2.6-Flash-MOPD-MXFP4-00002-of-00002.gguf: 167363516512 B, LFS sha256 3adf8ea80157dd066945a954428ba0e507c15c0e5a8b6b276ac92df0948698bc, header 30048 B (sha256 b11fbbabdcab1b5121cdfb1c6ad0a684cb53b22ee6a44e047d124f09bff5e1f8), fetched 262144 B, committed 3691 B gz, 472 tensors, size check exact, offsets contiguous

(e) identity
  architecture: mimo2
  general.name: MiMo-V2.6-Flash-MOPD
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: qwen2
  vocab (tokenizer.ggml.tokens): 152576
  merges: 151387
  token_embd shape: [4096, 152576]
  output shape: [4096, 152576]
  tokenizer.ggml.add_bos_token: false
  tokenizer.ggml.add_eos_token: false
  tokenizer.ggml.eos_token_id: 151645
  tokenizer.ggml.padding_token_id: 151643
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.file_type": 38, "general.languages": ["en", "zh"], "general.license": "mit", "general.quantization_version": 2, "general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, ...
  [ec7630a] arch mimo2: src/llama-arch.cpp:149
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer qwen2: src/llama-vocab.cpp:2259
  [79e2e74] arch mimo2: src/llama-arch.cpp:151
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer qwen2: src/llama-vocab.cpp:2376

(d) expert_group_count = 1, expert_group_used_count = 1 (keys present: [True, True]): no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)
  [ec7630a] src/llama-graph.cpp:2105
  [79e2e74] src/llama-graph.cpp:2145

(c) chat template vs None: no vendored template named in the request; embedded 3867 B sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059

(b) hyperparameters, real vs tiny (tiny variants skipped: --fused-qkv --sinks swa --value-scale 0.75 --mtp (tools/tinygen/mimo2_tiny.py build() takes no ['fused_qkv', 'mtp', 'sinks', 'value_scale']))
  attention.head_count                       | real: 64 | tiny default: 4
  attention.head_count_kv                    | real: [48] 8 x39, 4 x9 | tiny default: [1, 2, 2, 2, 2, 1]
  attention.key_length                       | real: 192 | tiny default: 24
  attention.layer_norm_rms_epsilon           | real: 9.999999974752427e-07 | tiny default: 9.999999747378752e-06
  attention.sliding_window                   | real: 128 | tiny default: 8
  attention.sliding_window_pattern           | real: [48] 1 x39, 0 x9 | tiny default: [false, true, true, true, true, false]
  attention.value_length                     | real: 128 | tiny default: 16
  attention.value_scale                      | real: 0.7070000171661377 | tiny default: (absent)
  block_count                                | real: 48 | tiny default: 6
  context_length                             | real: 1048576 | tiny default: 512
  embedding_length                           | real: 4096 | tiny default: 64
  expert_count                               | real: 256 | tiny default: 8
  expert_feed_forward_length                 | real: 2048 | tiny default: 32
  expert_gating_func                         | real: 2 | tiny default: (absent)
  expert_group_count                         | real: 1 | tiny default: (absent)
  expert_group_used_count                    | real: 1 | tiny default: (absent)
  expert_used_count                          | real: 8 | tiny default: 2
  feed_forward_length                        | real: 16384 | tiny default: 96
  rope.dimension_count                       | real: 64 | tiny default: 8
  rope.freq_base                             | real: 10000000.0 | tiny default: 10000.0
  rope.freq_base_swa                         | real: 10000.0 | tiny default: 5000.0
  config.json cross-check:
    OK     num_hidden_layers = 48  ->  layers = 48
    ABSENT num_nextn_predict_layers = 3  ->  nextn_predict_layers = null
    OK     hidden_size = 4096  ->  embedding_length = 4096
    OK     intermediate_size = 16384  ->  feed_forward_length = 16384
    OK     moe_intermediate_size = 2048  ->  expert_feed_forward_length = 2048
    OK     n_routed_experts = 256  ->  expert_count = 256
    OK     num_experts_per_tok = 8  ->  expert_used_count = 8
    OK     num_attention_heads = 64  ->  attention.head_count = 64
    OK     hybrid_layer_pattern = [48] 1 x39, 0 x9 (swa_num_key_value_heads where 1 else num_key_value_heads, + SWA per MTP layer -> [48] 8 x39, 4 x9)  ->  attention.head_count_kv = [48] 8 x39, 4 x9
    OK     head_dim = 192  ->  attention.key_length = 192
    OK     v_head_dim = 128  ->  attention.value_length = 128
    OK     sliding_window = 128  ->  attention.sliding_window = 128
    OK     hybrid_layer_pattern = [48] 1 x39, 0 x9 (+ 1 per MTP layer -> [48] 1 x39, 0 x9)  ->  attention.sliding_window_pattern = [48] 1 x39, 0 x9
    OK     partial_rotary_factor = 0.334 (int(head_dim x factor) -> 64)  ->  rope.dimension_count = 64
    OK     rope_theta = 10000000.0  ->  rope.freq_base = 10000000.0
    OK     swa_rope_theta = 10000.0  ->  rope.freq_base_swa = 10000.0
    OK     attention_value_scale = 0.707  ->  attention.value_scale = 0.7070000171661377
    OK     layernorm_epsilon = 1e-06  ->  attention.layer_norm_rms_epsilon = 9.999999974752427e-07
    OK     max_position_embeddings = 1048576  ->  context_length = 1048576
    OK     n_group = 1  ->  expert_group_count = 1
    OK     topk_group = 1  ->  expert_group_used_count = 1
    OK     vocab_size = 152576  ->  @vocab = {"tokens": 152576, "token_embd rows": 152576}
  graph branches:
    [tiny OK] moe.groups: group-limited routing (expert_group_count > 1)
       real: ['no group step']  expert_group_count = 1, used = 1
       tiny: default: ['no group step']
       [ec7630a] src/llama-model.cpp:1334; src/llama-graph.cpp:2105
       [79e2e74] src/llama-model.cpp:1362; src/llama-graph.cpp:2145
    [tiny OK] moe.used_array: experts per token: one value or per layer
       real: ['scalar']  expert_used_count = 8, expert_count = 256
       tiny: default: ['scalar']
       [ec7630a] src/llama-model.cpp:1333
       [79e2e74] src/llama-model.cpp:1361
    [tiny OK] mimo.swa_pattern: sliding-window / full-attention layers from the required pattern array
       real: ['full-attention layers', 'sliding-window layers']  window 128; full-attention trunk layers [0, 5, 11, 17, 23, 29, 35, 41, 47]; sliding-window 39 of 48; pattern entries of the MTP layers []; pattern length 48, block_count 48
       tiny: default: ['full-attention layers', 'sliding-window layers']
       [ec7630a] src/models/mimo2.cpp:12; src/models/mimo2.cpp:9
       [79e2e74] src/models/mimo2.cpp:12; src/models/mimo2.cpp:9
    [tiny OK] mimo.kv_heads: KV heads per layer (full and sliding-window layers differ)
       real: ['per-layer KV heads, full != SWA']  head_count 64; KV heads by trunk layer kind {'full': [4], 'SWA': [8]}; MTP layers []
       tiny: default: ['per-layer KV heads, full != SWA']
       [ec7630a] src/llama-model.cpp:1404; src/models/mimo2.cpp:114,288
       [79e2e74] src/llama-model.cpp:1432; src/models/mimo2.cpp:106,279
    [tiny OK] mimo.head_dims: K and V head dims (key_length / value_length)
       real: ['K head dim != V head dim']  key_length 192, value_length 128, key_length_swa None, value_length_swa None
       tiny: default: ['K head dim != V head dim']
       [ec7630a] src/llama-model.cpp:1443; src/models/mimo2.cpp:135,335
       [79e2e74] src/llama-model.cpp:1471; src/models/mimo2.cpp:127,326
    [tiny OK] mimo.rope: partial rotary and the two freq bases (full layers / SWA layers)
       real: ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']  rope.dimension_count 64, key_length 192, freq_base 10000000.0, freq_base_swa 10000.0
       tiny: default: ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']
       [ec7630a] src/models/mimo2.cpp:10; src/llama-model.cpp:2330; src/llama-hparams.h:151
       [79e2e74] src/models/mimo2.cpp:10; src/llama-model.cpp:2362; src/llama-hparams.h:154
    [NOT IN TINY] mimo.value_scale: attn_out scaled by attention.value_scale (read only if != 1)
       real: ['attn_out x value_scale']  attention.value_scale = 0.7070000171661377
       tiny: default: ['no value scale']
       [ec7630a] src/models/mimo2.cpp:15; src/models/mimo2.cpp:183,365
       [79e2e74] src/models/mimo2.cpp:15; src/models/mimo2.cpp:175,356
    [NOT IN TINY] mimo.sinks: attention sinks per layer kind (attn_sinks is TENSOR_NOT_REQUIRED)
       real: ['SWA layers with sinks', 'full layers without sinks']  trunk layers with attn_sinks: 1-4, 6-10, 12-16, 18-22, 24-28, 30-34, 36-40, 42-46 (39 of 48; full-attention layers among them []); MTP layers with attn_sinks: []
       tiny: default: ['SWA layers with sinks', 'full layers with sinks']
       [ec7630a] src/models/mimo2.cpp:57; src/models/mimo2.cpp:175
       [79e2e74] src/models/mimo2.cpp:51; src/models/mimo2.cpp:167
    [NOT IN TINY] mimo.qkv: attention projections: fused attn_qkv or split attn_q/k/v
       real: ['fused attn_qkv']  
       tiny: default: ['split attn_q/k/v']
       [ec7630a] src/models/mimo2.cpp:129
       [79e2e74] src/models/mimo2.cpp:121
    [tiny OK] mimo.ffn: dense FFN (no ffn_gate_inp) or MoE with sigmoid + bias routing
       real: ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']  dense trunk layers [0]; expert_count 256, used 8, expert width 2048, dense width 16384
       tiny: default: ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']
       [ec7630a] src/models/mimo2.cpp:202; src/models/mimo2.cpp:222
       [79e2e74] src/models/mimo2.cpp:194; src/models/mimo2.cpp:214
    [tiny OK] mimo.mtp: MTP (NextN) layers: count; graph_mtp runs layer n_layer + nextn_layer_offset
       real: ['nextn_predict_layers 0']  no MTP layers
       tiny: default: ['nextn_predict_layers 0']
       [ec7630a] src/models/mimo2.cpp:276; src/models/mimo2.cpp:278
       [79e2e74] src/models/mimo2.cpp:267; src/models/mimo2.cpp:269
    [tiny OK] mimo.mtp_block: what graph_mtp asserts of an MTP layer: fused attn_qkv and a dense FFN
       real: ['no MTP layers']  
       tiny: default: ['no MTP layers']
       [ec7630a] src/models/mimo2.cpp:285; src/models/mimo2.cpp:375
       [79e2e74] src/models/mimo2.cpp:276; src/models/mimo2.cpp:366
    [tiny OK] mimo.mtp_head: MTP embedding / head / head norm: own NextN tensors or the fallbacks
       real: ['no MTP tensors']  
       tiny: default: ['no MTP tensors']
       [ec7630a] src/models/mimo2.cpp:303; src/models/mimo2.cpp:394; src/models/mimo2.cpp:399
       [79e2e74] src/models/mimo2.cpp:294; src/models/mimo2.cpp:385; src/models/mimo2.cpp:390
    [tiny OK] mimo.file_layout: trunk + MTP, MTP-only or trunk-only file (loader flags)
       real: ['trunk only (no nextn_predict_layers)']  
       tiny: default: ['trunk only (no nextn_predict_layers)']
       [ec7630a] src/models/mimo2.cpp:28; not found
       [79e2e74] not found; src/models/mimo2.cpp:28
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 8 required, missing in the file: []; none conditional
    [79e2e74] 8 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {"ffn_down_exps": {"MXFP4": 47}, "ffn_gate_exps": {"MXFP4": 47}, "ffn_up_exps": {"MXFP4": 47}}

(a) tools/gguf_quant_audit.py on the header-only shards (files total 167369465152 B = 155.875 GiB)
  # MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf (2 file(s)), architecture 'mimo2'
  total 308.78B params, 155.87 GiB, 4.336 bpw
    routed_experts   302.795B    149.81 GiB   4.250 bpw  [MXFP4 100.0%]
    attention          4.483B      4.44 GiB   8.500 bpw  [Q8_0 100.0%, F32 0.0%]
    dense_ffn          0.201B      0.20 GiB   8.500 bpw  [Q8_0 100.0%]
    embedding          0.625B      0.62 GiB   8.500 bpw  [Q8_0 100.0%]
    output_head        0.625B      0.62 GiB   8.500 bpw  [Q8_0 100.0%]
    other              0.050B      0.19 GiB  32.000 bpw  [F32 100.0%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: blk.1.ffn_down_exps.weight MXFP4 4.250, blk.1.ffn_gate_exps.weight MXFP4 4.250, blk.1.ffn_up_exps.weight MXFP4 4.250, blk.2.ffn_down_exps.weight MXFP4 4.250, blk.2.ffn_gate_exps.weight MXFP4 4.250, blk.2.ffn_up_exps.weight MXFP4 4.250

=== mimo-v2.6-flash-mopd-ggml-org-mtp (third_party/gguf-headers/mimo-v2.6-flash-mopd-ggml-org-mtp)
status ok, ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @ e6d4bced739310a721eefad4c247ec7e0247c89d
integrity: mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf OK
mode mtp_files; NextN reference: request mimo-v2.6-flash-mopd-quantaplanta-mxfp4, blk.48; trunk request: mimo-v2.6-flash-mopd-ggml-org-mxfp4
  mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf: 2383728864 B, LFS sha256 4589839383ca7524fa1bdf635b8cef0da59b7e909d84a433f6006f7b64fc17ec, header 5950944 B (sha256 8805a08513f9777f59845d1bfa486ef52eb3a77c63952cf6031eee55f28c413b), fetched 8388608 B, committed 1661660 B gz, 39 tensors, size check exact, offsets contiguous

how llama.cpp attaches a separate MTP file:
  [ec7630a] common/arg.cpp:4241: a local MTP file is given as the draft model: -md FILE (sets speculative.draft.mparams.path)
  [ec7630a] common/arg.cpp:4249: --spec-type draft-mtp selects the MTP drafter
  [ec7630a] common/speculative.cpp:37: the type name draft-mtp
  [ec7630a] common/arg.cpp:3084: --mtp (download example): also fetch the repo's MTP sidecar
  [ec7630a] common/download.cpp:616: -hf: the sidecar is the sibling file whose name starts with mtp-
  [ec7630a] common/download.cpp:649: an mtp- file is never taken as the model
  [ec7630a] common/arg.cpp:644: the downloaded sidecar becomes the draft model path
  [ec7630a] common/speculative.cpp:2539: with a draft path the MTP file is loaded as its own llama_model (llama_model_load_from_file)
  [ec7630a] common/speculative.cpp:2526: its context is an MTP context
  [ec7630a] common/speculative.cpp:2560: without a draft path the MTP context runs on the target model's own NextN layers
  [ec7630a] common/speculative.cpp:1376: the MTP model's output width must equal the target's (h_nextn rows)
  [ec7630a] common/common.cpp:1625: draft-mtp sets load_mtp (otherwise NextN tensors are skipped)
  [ec7630a] src/llama-vocab.cpp:1944: a separate model file needs its own tokenizer metadata (tokenizer.ggml.model is required)
  [ec7630a] src/llama-model-loader.cpp:1398: every tensor in the file must be created by the arch loader
  [ec7630a] src/llama-context.cpp:156: only EAGLE3 / DFlash drafts borrow the target's embeddings and head through ctx_other
  [ec7630a] src/llama-model.cpp:2802,2809: an MTP context's KV cache holds only the NextN layers
  [79e2e74] common/arg.cpp:4276: a local MTP file is given as the draft model: -md FILE (sets speculative.draft.mparams.path)
  [79e2e74] common/arg.cpp:4284: --spec-type draft-mtp selects the MTP drafter
  [79e2e74] common/speculative.cpp:76: the type name draft-mtp
  [79e2e74] common/arg.cpp:3097: --mtp (download example): also fetch the repo's MTP sidecar
  [79e2e74] common/download.cpp:605: -hf: the sidecar is the sibling file whose name starts with mtp-
  [79e2e74] common/download.cpp:638: an mtp- file is never taken as the model
  [79e2e74] common/arg.cpp:644: the downloaded sidecar becomes the draft model path
  [79e2e74] common/speculative.cpp:2623: with a draft path the MTP file is loaded as its own llama_model (llama_model_load_from_file)
  [79e2e74] common/speculative.cpp:2610: its context is an MTP context
  [79e2e74] common/speculative.cpp:2644: without a draft path the MTP context runs on the target model's own NextN layers
  [79e2e74] common/speculative.cpp:1437: the MTP model's output width must equal the target's (h_nextn rows)
  [79e2e74] common/common.cpp:1690: draft-mtp sets load_mtp (otherwise NextN tensors are skipped)
  [79e2e74] src/llama-vocab.cpp:2057: a separate model file needs its own tokenizer metadata (tokenizer.ggml.model is required)
  [79e2e74] src/llama-model-loader.cpp:1398: every tensor in the file must be created by the arch loader
  [79e2e74] src/llama-context.cpp:159: only EAGLE3 / DFlash drafts borrow the target's embeddings and head through ctx_other
  [79e2e74] src/llama-model.cpp:2841,2848: an MTP context's KV cache holds only the NextN layers

--- mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf: arch mimo2, block_count 51, nextn_predict_layers 3, n_layer 48; layout: MTP-only file
  tokenizer: model gpt2, pre qwen2, 152576 tokens; embedded chat template: True
  non-layer tensors: output.weight Q8_0 [4096, 152576], output_norm.weight F32 [4096], token_embd.weight Q8_0 [4096, 152576]
  layers present: [48, 49, 50]
  blk.48: 12 tensors: ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight', 'layer_output_norm.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight']
  blk.49: 12 tensors: ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight', 'layer_output_norm.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight']
  blk.50: 12 tensors: ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight', 'layer_output_norm.weight', 'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight']
  vs the reference NextN layer: blk.48 missing [], missing optional [], extra []
  vs the reference NextN layer: blk.49 missing [], missing optional [], extra []
  vs the reference NextN layer: blk.50 missing [], missing optional [], extra []
  vs the trunk (mimo2): arch keys that differ: attention.head_count_kv: trunk [48] 8 x39, 4 x9 / this [51] 8 x42, 4 x9; attention.sliding_window_pattern: trunk [48] 1 x39, 0 x9 / this [51] 1 x42, 0 x9; block_count: trunk 48 / this 51; nextn_predict_layers: trunk (absent) / this 3; token list equal: True; token_embd trunk [4096, 152576] / this [4096, 152576]
  config.json cross-check: 22 of 22 OK
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 8 required, missing in the file: []; none conditional
    [79e2e74] 8 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {}
  audit: PASS, 2.214 GiB, 8.501 bpw; output_head 8.500 bpw {'Q8_0': 624951296}, other 8.515 bpw {'F32': 65536, 'Q8_0': 100663296}, embedding 8.500 bpw {'Q8_0': 624951296}, attention 8.500 bpw {'Q8_0': 283115520, 'F32': 192}, dense_ffn 8.500 bpw {'Q8_0': 603979776}
  loader [ec7630a]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/mimo2.cpp:28
      graph: src/models/mimo2.cpp:88
  loader [79e2e74]: loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)
      detect: src/models/mimo2.cpp:28
      graph: src/models/mimo2.cpp:82

=== mimo-v2.6-flash-mopd-quantaplanta-mxfp4 (third_party/gguf-headers/mimo-v2.6-flash-mopd-quantaplanta-mxfp4)
status ok, QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ ac33b3743d70ed57a8cc6b847698d2ebbddb7784
integrity: MiMo-V2.6-Flash-MOPD-MXFP4.gguf OK
  MiMo-V2.6-Flash-MOPD-MXFP4.gguf: 174908254272 B, LFS sha256 fc1b8122d1a31db98b20ed5c9ddb3d4d7e28d37b47cb2f1e843354788fbfeaa6, header 5980736 B (sha256 2e3a7ba2f0b09c1582dddb30f868edf0547569102908d6114aaec37fcc7d1980), fetched 8388608 B, committed 1665898 B gz, 508 tensors, size check exact, offsets contiguous

(e) identity
  architecture: mimo2
  general.name: MiMo V2.6 Flash MOPD
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: qwen2
  vocab (tokenizer.ggml.tokens): 152576
  merges: 151387
  token_embd shape: [4096, 152576]
  output shape: [4096, 152576]
  tokenizer.ggml.add_bos_token: false
  tokenizer.ggml.add_eos_token: false
  tokenizer.ggml.eos_token_id: 151645
  tokenizer.ggml.padding_token_id: 151643
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.file_type": 38, "general.languages": ["en", "zh"], "general.license": "mit", "general.quantization_version": 2, "general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, ...
  [ec7630a] arch mimo2: src/llama-arch.cpp:149
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer qwen2: src/llama-vocab.cpp:2259
  [79e2e74] arch mimo2: src/llama-arch.cpp:151
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer qwen2: src/llama-vocab.cpp:2376

(d) expert_group_count = 1, expert_group_used_count = 1 (keys present: [True, True]): no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)
  [ec7630a] src/llama-graph.cpp:2105
  [79e2e74] src/llama-graph.cpp:2145

(c) chat template vs None: no vendored template named in the request; embedded 3867 B sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059

(b) hyperparameters, real vs tiny (tiny variants skipped: --fused-qkv --sinks swa --value-scale 0.75 --mtp (tools/tinygen/mimo2_tiny.py build() takes no ['fused_qkv', 'mtp', 'sinks', 'value_scale']))
  attention.head_count                       | real: 64 | tiny default: 4
  attention.head_count_kv                    | real: [51] 8 x42, 4 x9 | tiny default: [1, 2, 2, 2, 2, 1]
  attention.key_length                       | real: 192 | tiny default: 24
  attention.layer_norm_rms_epsilon           | real: 9.999999974752427e-07 | tiny default: 9.999999747378752e-06
  attention.sliding_window                   | real: 128 | tiny default: 8
  attention.sliding_window_pattern           | real: [51] 1 x42, 0 x9 | tiny default: [false, true, true, true, true, false]
  attention.value_length                     | real: 128 | tiny default: 16
  attention.value_scale                      | real: 0.7070000171661377 | tiny default: (absent)
  block_count                                | real: 51 | tiny default: 6
  context_length                             | real: 1048576 | tiny default: 512
  embedding_length                           | real: 4096 | tiny default: 64
  expert_count                               | real: 256 | tiny default: 8
  expert_feed_forward_length                 | real: 2048 | tiny default: 32
  expert_gating_func                         | real: 2 | tiny default: (absent)
  expert_group_count                         | real: 1 | tiny default: (absent)
  expert_group_used_count                    | real: 1 | tiny default: (absent)
  expert_used_count                          | real: 8 | tiny default: 2
  feed_forward_length                        | real: 16384 | tiny default: 96
  nextn_predict_layers                       | real: 3 | tiny default: (absent)
  rope.dimension_count                       | real: 64 | tiny default: 8
  rope.freq_base                             | real: 10000000.0 | tiny default: 10000.0
  rope.freq_base_swa                         | real: 10000.0 | tiny default: 5000.0
  config.json cross-check:
    OK     num_hidden_layers = 48  ->  layers = 48
    OK     num_nextn_predict_layers = 3  ->  nextn_predict_layers = 3
    OK     hidden_size = 4096  ->  embedding_length = 4096
    OK     intermediate_size = 16384  ->  feed_forward_length = 16384
    OK     moe_intermediate_size = 2048  ->  expert_feed_forward_length = 2048
    OK     n_routed_experts = 256  ->  expert_count = 256
    OK     num_experts_per_tok = 8  ->  expert_used_count = 8
    OK     num_attention_heads = 64  ->  attention.head_count = 64
    OK     hybrid_layer_pattern = [48] 1 x39, 0 x9 (swa_num_key_value_heads where 1 else num_key_value_heads, + SWA per MTP layer -> [51] 8 x42, 4 x9)  ->  attention.head_count_kv = [51] 8 x42, 4 x9
    OK     head_dim = 192  ->  attention.key_length = 192
    OK     v_head_dim = 128  ->  attention.value_length = 128
    OK     sliding_window = 128  ->  attention.sliding_window = 128
    OK     hybrid_layer_pattern = [48] 1 x39, 0 x9 (+ 1 per MTP layer -> [51] 1 x42, 0 x9)  ->  attention.sliding_window_pattern = [51] 1 x42, 0 x9
    OK     partial_rotary_factor = 0.334 (int(head_dim x factor) -> 64)  ->  rope.dimension_count = 64
    OK     rope_theta = 10000000.0  ->  rope.freq_base = 10000000.0
    OK     swa_rope_theta = 10000.0  ->  rope.freq_base_swa = 10000.0
    OK     attention_value_scale = 0.707  ->  attention.value_scale = 0.7070000171661377
    OK     layernorm_epsilon = 1e-06  ->  attention.layer_norm_rms_epsilon = 9.999999974752427e-07
    OK     max_position_embeddings = 1048576  ->  context_length = 1048576
    OK     n_group = 1  ->  expert_group_count = 1
    OK     topk_group = 1  ->  expert_group_used_count = 1
    OK     vocab_size = 152576  ->  @vocab = {"tokens": 152576, "token_embd rows": 152576}
  graph branches:
    [tiny OK] moe.groups: group-limited routing (expert_group_count > 1)
       real: ['no group step']  expert_group_count = 1, used = 1
       tiny: default: ['no group step']
       [ec7630a] src/llama-model.cpp:1334; src/llama-graph.cpp:2105
       [79e2e74] src/llama-model.cpp:1362; src/llama-graph.cpp:2145
    [tiny OK] moe.used_array: experts per token: one value or per layer
       real: ['scalar']  expert_used_count = 8, expert_count = 256
       tiny: default: ['scalar']
       [ec7630a] src/llama-model.cpp:1333
       [79e2e74] src/llama-model.cpp:1361
    [tiny OK] mimo.swa_pattern: sliding-window / full-attention layers from the required pattern array
       real: ['full-attention layers', 'sliding-window layers']  window 128; full-attention trunk layers [0, 5, 11, 17, 23, 29, 35, 41, 47]; sliding-window 39 of 48; pattern entries of the MTP layers [1, 1, 1]; pattern length 51, block_count 51
       tiny: default: ['full-attention layers', 'sliding-window layers']
       [ec7630a] src/models/mimo2.cpp:12; src/models/mimo2.cpp:9
       [79e2e74] src/models/mimo2.cpp:12; src/models/mimo2.cpp:9
    [tiny OK] mimo.kv_heads: KV heads per layer (full and sliding-window layers differ)
       real: ['per-layer KV heads, full != SWA']  head_count 64; KV heads by trunk layer kind {'full': [4], 'SWA': [8]}; MTP layers [8, 8, 8]
       tiny: default: ['per-layer KV heads, full != SWA']
       [ec7630a] src/llama-model.cpp:1404; src/models/mimo2.cpp:114,288
       [79e2e74] src/llama-model.cpp:1432; src/models/mimo2.cpp:106,279
    [tiny OK] mimo.head_dims: K and V head dims (key_length / value_length)
       real: ['K head dim != V head dim']  key_length 192, value_length 128, key_length_swa None, value_length_swa None
       tiny: default: ['K head dim != V head dim']
       [ec7630a] src/llama-model.cpp:1443; src/models/mimo2.cpp:135,335
       [79e2e74] src/llama-model.cpp:1471; src/models/mimo2.cpp:127,326
    [tiny OK] mimo.rope: partial rotary and the two freq bases (full layers / SWA layers)
       real: ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']  rope.dimension_count 64, key_length 192, freq_base 10000000.0, freq_base_swa 10000.0
       tiny: default: ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']
       [ec7630a] src/models/mimo2.cpp:10; src/llama-model.cpp:2330; src/llama-hparams.h:151
       [79e2e74] src/models/mimo2.cpp:10; src/llama-model.cpp:2362; src/llama-hparams.h:154
    [NOT IN TINY] mimo.value_scale: attn_out scaled by attention.value_scale (read only if != 1)
       real: ['attn_out x value_scale']  attention.value_scale = 0.7070000171661377
       tiny: default: ['no value scale']
       [ec7630a] src/models/mimo2.cpp:15; src/models/mimo2.cpp:183,365
       [79e2e74] src/models/mimo2.cpp:15; src/models/mimo2.cpp:175,356
    [NOT IN TINY] mimo.sinks: attention sinks per layer kind (attn_sinks is TENSOR_NOT_REQUIRED)
       real: ['SWA layers with sinks', 'full layers without sinks']  trunk layers with attn_sinks: 1-4, 6-10, 12-16, 18-22, 24-28, 30-34, 36-40, 42-46 (39 of 48; full-attention layers among them []); MTP layers with attn_sinks: [48, 49, 50]
       tiny: default: ['SWA layers with sinks', 'full layers with sinks']
       [ec7630a] src/models/mimo2.cpp:57; src/models/mimo2.cpp:175
       [79e2e74] src/models/mimo2.cpp:51; src/models/mimo2.cpp:167
    [NOT IN TINY] mimo.qkv: attention projections: fused attn_qkv or split attn_q/k/v
       real: ['fused attn_qkv']  
       tiny: default: ['split attn_q/k/v']
       [ec7630a] src/models/mimo2.cpp:129
       [79e2e74] src/models/mimo2.cpp:121
    [tiny OK] mimo.ffn: dense FFN (no ffn_gate_inp) or MoE with sigmoid + bias routing
       real: ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']  dense trunk layers [0]; expert_count 256, used 8, expert width 2048, dense width 16384
       tiny: default: ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']
       [ec7630a] src/models/mimo2.cpp:202; src/models/mimo2.cpp:222
       [79e2e74] src/models/mimo2.cpp:194; src/models/mimo2.cpp:214
    [NOT IN TINY] mimo.mtp: MTP (NextN) layers: count; graph_mtp runs layer n_layer + nextn_layer_offset
       real: ['nextn_predict_layers 3']  blk.48: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight'], other ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight', 'layer_output_norm.weight']; blk.49: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight'], other ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight', 'layer_output_norm.weight']; blk.50: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight'], other ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight', 'layer_output_norm.weight']
       tiny: default: ['nextn_predict_layers 0']
       [ec7630a] src/models/mimo2.cpp:276; src/models/mimo2.cpp:278
       [79e2e74] src/models/mimo2.cpp:267; src/models/mimo2.cpp:269
    [NOT IN TINY] mimo.mtp_block: what graph_mtp asserts of an MTP layer: fused attn_qkv and a dense FFN
       real: ['MTP layer with a dense FFN', 'MTP layer with fused attn_qkv']  
       tiny: default: ['no MTP layers']
       [ec7630a] src/models/mimo2.cpp:285; src/models/mimo2.cpp:375
       [79e2e74] src/models/mimo2.cpp:276; src/models/mimo2.cpp:366
    [NOT IN TINY] mimo.mtp_head: MTP embedding / head / head norm: own NextN tensors or the fallbacks
       real: ['head norm layer_output_norm', 'trunk output head', 'trunk token_embd']  
       tiny: default: ['no MTP tensors']
       [ec7630a] src/models/mimo2.cpp:303; src/models/mimo2.cpp:394; src/models/mimo2.cpp:399
       [79e2e74] src/models/mimo2.cpp:294; src/models/mimo2.cpp:385; src/models/mimo2.cpp:390
    [NOT IN TINY] mimo.file_layout: trunk + MTP, MTP-only or trunk-only file (loader flags)
       real: ['trunk + MTP in one file']  
       tiny: default: ['trunk only (no nextn_predict_layers)']
       [ec7630a] src/models/mimo2.cpp:28; not found
       [79e2e74] not found; src/models/mimo2.cpp:28
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 8 required, missing in the file: []; none conditional
    [79e2e74] 8 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {"ffn_down_exps": {"MXFP4": 47}, "ffn_gate_exps": {"MXFP4": 47}, "ffn_up_exps": {"MXFP4": 47}}

(a) tools/gguf_quant_audit.py on the header-only shards (files total 174908254272 B = 162.896 GiB)
  # MiMo-V2.6-Flash-MOPD-MXFP4.gguf (1 file(s)), architecture 'mimo2'
  total 309.77B params, 162.89 GiB, 4.517 bpw
    routed_experts   302.795B    149.81 GiB   4.250 bpw  [MXFP4 100.0%]
    attention          4.766B      8.88 GiB  16.000 bpw  [BF16 100.0%, F32 0.0%]
    dense_ffn          0.805B      1.50 GiB  16.000 bpw  [BF16 100.0%]
    embedding          0.625B      1.16 GiB  16.000 bpw  [BF16 100.0%]
    output_head        0.625B      1.16 GiB  16.000 bpw  [BF16 100.0%]
    other              0.150B      0.37 GiB  21.292 bpw  [BF16 66.9%, F32 33.1%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: blk.1.ffn_down_exps.weight MXFP4 4.250, blk.1.ffn_gate_exps.weight MXFP4 4.250, blk.1.ffn_up_exps.weight MXFP4 4.250, blk.2.ffn_down_exps.weight MXFP4 4.250, blk.2.ffn_gate_exps.weight MXFP4 4.250, blk.2.ffn_up_exps.weight MXFP4 4.250

=== mimo-v2.6-flash-mopd-kernelpool-mxfp4 (third_party/gguf-headers/mimo-v2.6-flash-mopd-kernelpool-mxfp4)
status ok, kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF @ 44d9191bb7a95125c8b971ce0593bd802ae6dd48
integrity: MiMo-V2.6-Flash-MOPD-MXFP4.gguf OK
  MiMo-V2.6-Flash-MOPD-MXFP4.gguf: 169005098944 B, LFS sha256 5e5dfb6d96466f40e73a235467949aa836e14d4de2310cff220409923937c99b, header 5980608 B (sha256 98002a80c3b0951ef5e9c1963ed47cbe3a57a9ff62fd7200da7c6c5351f36797), fetched 8388608 B, committed 1666380 B gz, 508 tensors, size check exact, offsets contiguous

(e) identity
  architecture: mimo2
  general.name: MiMo V2.6 Flash MOPD
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: qwen2
  vocab (tokenizer.ggml.tokens): 152576
  merges: 151387
  token_embd shape: [4096, 152576]
  output shape: [4096, 152576]
  tokenizer.ggml.add_bos_token: false
  tokenizer.ggml.add_eos_token: false
  tokenizer.ggml.eos_token_id: 151645
  tokenizer.ggml.padding_token_id: 151643
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.alignment": 32, "general.source.revision": "2479e2d0029eca9a34cc7e7f55a121925f81908e", "general.source.url": "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD"}
  [ec7630a] arch mimo2: src/llama-arch.cpp:149
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer qwen2: src/llama-vocab.cpp:2259
  [79e2e74] arch mimo2: src/llama-arch.cpp:151
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer qwen2: src/llama-vocab.cpp:2376

(d) expert_group_count = None, expert_group_used_count = None (keys present: [False, False]): no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)
  [ec7630a] src/llama-graph.cpp:2105
  [79e2e74] src/llama-graph.cpp:2145

(c) chat template vs None: no vendored template named in the request; embedded 3867 B sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059

(b) hyperparameters, real vs tiny (tiny variants skipped: --fused-qkv --sinks swa --value-scale 0.75 --mtp (tools/tinygen/mimo2_tiny.py build() takes no ['fused_qkv', 'mtp', 'sinks', 'value_scale']))
  attention.head_count                       | real: 64 | tiny default: 4
  attention.head_count_kv                    | real: [51] 8 x42, 4 x9 | tiny default: [1, 2, 2, 2, 2, 1]
  attention.key_length                       | real: 192 | tiny default: 24
  attention.layer_norm_rms_epsilon           | real: 9.999999974752427e-07 | tiny default: 9.999999747378752e-06
  attention.sliding_window                   | real: 128 | tiny default: 8
  attention.sliding_window_pattern           | real: [51] 1 x42, 0 x9 | tiny default: [false, true, true, true, true, false]
  attention.value_length                     | real: 128 | tiny default: 16
  attention.value_scale                      | real: 0.7070000171661377 | tiny default: (absent)
  block_count                                | real: 51 | tiny default: 6
  context_length                             | real: 1048576 | tiny default: 512
  embedding_length                           | real: 4096 | tiny default: 64
  expert_count                               | real: 256 | tiny default: 8
  expert_feed_forward_length                 | real: 2048 | tiny default: 32
  expert_gating_func                         | real: 2 | tiny default: (absent)
  expert_used_count                          | real: 8 | tiny default: 2
  expert_weights_norm                        | real: true | tiny default: (absent)
  expert_weights_scale                       | real: 1.0 | tiny default: (absent)
  feed_forward_length                        | real: 16384 | tiny default: 96
  leading_dense_block_count                  | real: 1 | tiny default: (absent)
  nextn_predict_layers                       | real: 3 | tiny default: (absent)
  qkv_chunks                                 | real: 4 | tiny default: (absent)
  quantization                               | real: released MXFP4 gate/up/down; Q8_0 attention/d... | tiny default: (absent)
  rope.dimension_count                       | real: 64 | tiny default: 8
  rope.freq_base                             | real: 10000000.0 | tiny default: 10000.0
  rope.freq_base_swa                         | real: 10000.0 | tiny default: 5000.0
  vocab_size                                 | real: 152576 | tiny default: (absent)
  config.json cross-check:
    OK     num_hidden_layers = 48  ->  layers = 48
    OK     num_nextn_predict_layers = 3  ->  nextn_predict_layers = 3
    OK     hidden_size = 4096  ->  embedding_length = 4096
    OK     intermediate_size = 16384  ->  feed_forward_length = 16384
    OK     moe_intermediate_size = 2048  ->  expert_feed_forward_length = 2048
    OK     n_routed_experts = 256  ->  expert_count = 256
    OK     num_experts_per_tok = 8  ->  expert_used_count = 8
    OK     num_attention_heads = 64  ->  attention.head_count = 64
    OK     hybrid_layer_pattern = [48] 1 x39, 0 x9 (swa_num_key_value_heads where 1 else num_key_value_heads, + SWA per MTP layer -> [51] 8 x42, 4 x9)  ->  attention.head_count_kv = [51] 8 x42, 4 x9
    OK     head_dim = 192  ->  attention.key_length = 192
    OK     v_head_dim = 128  ->  attention.value_length = 128
    OK     sliding_window = 128  ->  attention.sliding_window = 128
    OK     hybrid_layer_pattern = [48] 1 x39, 0 x9 (+ 1 per MTP layer -> [51] 1 x42, 0 x9)  ->  attention.sliding_window_pattern = [51] 1 x42, 0 x9
    OK     partial_rotary_factor = 0.334 (int(head_dim x factor) -> 64)  ->  rope.dimension_count = 64
    OK     rope_theta = 10000000.0  ->  rope.freq_base = 10000000.0
    OK     swa_rope_theta = 10000.0  ->  rope.freq_base_swa = 10000.0
    OK     attention_value_scale = 0.707  ->  attention.value_scale = 0.7070000171661377
    OK     layernorm_epsilon = 1e-06  ->  attention.layer_norm_rms_epsilon = 9.999999974752427e-07
    OK     max_position_embeddings = 1048576  ->  context_length = 1048576
    ABSENT n_group = 1  ->  expert_group_count = null
    ABSENT topk_group = 1  ->  expert_group_used_count = null
    OK     vocab_size = 152576  ->  @vocab = {"tokens": 152576, "token_embd rows": 152576}
  graph branches:
    [tiny OK] moe.groups: group-limited routing (expert_group_count > 1)
       real: ['no group step']  expert_group_count = None, used = None
       tiny: default: ['no group step']
       [ec7630a] src/llama-model.cpp:1334; src/llama-graph.cpp:2105
       [79e2e74] src/llama-model.cpp:1362; src/llama-graph.cpp:2145
    [tiny OK] moe.used_array: experts per token: one value or per layer
       real: ['scalar']  expert_used_count = 8, expert_count = 256
       tiny: default: ['scalar']
       [ec7630a] src/llama-model.cpp:1333
       [79e2e74] src/llama-model.cpp:1361
    [tiny OK] mimo.swa_pattern: sliding-window / full-attention layers from the required pattern array
       real: ['full-attention layers', 'sliding-window layers']  window 128; full-attention trunk layers [0, 5, 11, 17, 23, 29, 35, 41, 47]; sliding-window 39 of 48; pattern entries of the MTP layers [1, 1, 1]; pattern length 51, block_count 51
       tiny: default: ['full-attention layers', 'sliding-window layers']
       [ec7630a] src/models/mimo2.cpp:12; src/models/mimo2.cpp:9
       [79e2e74] src/models/mimo2.cpp:12; src/models/mimo2.cpp:9
    [tiny OK] mimo.kv_heads: KV heads per layer (full and sliding-window layers differ)
       real: ['per-layer KV heads, full != SWA']  head_count 64; KV heads by trunk layer kind {'full': [4], 'SWA': [8]}; MTP layers [8, 8, 8]
       tiny: default: ['per-layer KV heads, full != SWA']
       [ec7630a] src/llama-model.cpp:1404; src/models/mimo2.cpp:114,288
       [79e2e74] src/llama-model.cpp:1432; src/models/mimo2.cpp:106,279
    [tiny OK] mimo.head_dims: K and V head dims (key_length / value_length)
       real: ['K head dim != V head dim']  key_length 192, value_length 128, key_length_swa None, value_length_swa None
       tiny: default: ['K head dim != V head dim']
       [ec7630a] src/llama-model.cpp:1443; src/models/mimo2.cpp:135,335
       [79e2e74] src/llama-model.cpp:1471; src/models/mimo2.cpp:127,326
    [tiny OK] mimo.rope: partial rotary and the two freq bases (full layers / SWA layers)
       real: ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']  rope.dimension_count 64, key_length 192, freq_base 10000000.0, freq_base_swa 10000.0
       tiny: default: ['own SWA freq base (rope.freq_base_swa)', 'partial rotary (n_rot < K head dim)']
       [ec7630a] src/models/mimo2.cpp:10; src/llama-model.cpp:2330; src/llama-hparams.h:151
       [79e2e74] src/models/mimo2.cpp:10; src/llama-model.cpp:2362; src/llama-hparams.h:154
    [NOT IN TINY] mimo.value_scale: attn_out scaled by attention.value_scale (read only if != 1)
       real: ['attn_out x value_scale']  attention.value_scale = 0.7070000171661377
       tiny: default: ['no value scale']
       [ec7630a] src/models/mimo2.cpp:15; src/models/mimo2.cpp:183,365
       [79e2e74] src/models/mimo2.cpp:15; src/models/mimo2.cpp:175,356
    [NOT IN TINY] mimo.sinks: attention sinks per layer kind (attn_sinks is TENSOR_NOT_REQUIRED)
       real: ['SWA layers with sinks', 'full layers without sinks']  trunk layers with attn_sinks: 1-4, 6-10, 12-16, 18-22, 24-28, 30-34, 36-40, 42-46 (39 of 48; full-attention layers among them []); MTP layers with attn_sinks: [48, 49, 50]
       tiny: default: ['SWA layers with sinks', 'full layers with sinks']
       [ec7630a] src/models/mimo2.cpp:57; src/models/mimo2.cpp:175
       [79e2e74] src/models/mimo2.cpp:51; src/models/mimo2.cpp:167
    [NOT IN TINY] mimo.qkv: attention projections: fused attn_qkv or split attn_q/k/v
       real: ['fused attn_qkv']  
       tiny: default: ['split attn_q/k/v']
       [ec7630a] src/models/mimo2.cpp:129
       [79e2e74] src/models/mimo2.cpp:121
    [tiny OK] mimo.ffn: dense FFN (no ffn_gate_inp) or MoE with sigmoid + bias routing
       real: ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']  dense trunk layers [0]; expert_count 256, used 8, expert width 2048, dense width 16384
       tiny: default: ['MoE (sigmoid + exp_probs_b, renormalised top-k)', 'dense FFN']
       [ec7630a] src/models/mimo2.cpp:202; src/models/mimo2.cpp:222
       [79e2e74] src/models/mimo2.cpp:194; src/models/mimo2.cpp:214
    [NOT IN TINY] mimo.mtp: MTP (NextN) layers: count; graph_mtp runs layer n_layer + nextn_layer_offset
       real: ['nextn_predict_layers 3']  blk.48: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight'], other ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight']; blk.49: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight'], other ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight']; blk.50: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight'], other ['attn_norm.weight', 'attn_output.weight', 'attn_qkv.weight', 'attn_sinks.weight', 'ffn_down.weight', 'ffn_gate.weight', 'ffn_norm.weight', 'ffn_up.weight']
       tiny: default: ['nextn_predict_layers 0']
       [ec7630a] src/models/mimo2.cpp:276; src/models/mimo2.cpp:278
       [79e2e74] src/models/mimo2.cpp:267; src/models/mimo2.cpp:269
    [NOT IN TINY] mimo.mtp_block: what graph_mtp asserts of an MTP layer: fused attn_qkv and a dense FFN
       real: ['MTP layer with a dense FFN', 'MTP layer with fused attn_qkv']  
       tiny: default: ['no MTP layers']
       [ec7630a] src/models/mimo2.cpp:285; src/models/mimo2.cpp:375
       [79e2e74] src/models/mimo2.cpp:276; src/models/mimo2.cpp:366
    [NOT IN TINY] mimo.mtp_head: MTP embedding / head / head norm: own NextN tensors or the fallbacks
       real: ['head norm nextn.shared_head_norm', 'trunk output head', 'trunk token_embd']  
       tiny: default: ['no MTP tensors']
       [ec7630a] src/models/mimo2.cpp:303; src/models/mimo2.cpp:394; src/models/mimo2.cpp:399
       [79e2e74] src/models/mimo2.cpp:294; src/models/mimo2.cpp:385; src/models/mimo2.cpp:390
    [NOT IN TINY] mimo.file_layout: trunk + MTP, MTP-only or trunk-only file (loader flags)
       real: ['trunk + MTP in one file']  
       tiny: default: ['trunk only (no nextn_predict_layers)']
       [ec7630a] src/models/mimo2.cpp:28; not found
       [79e2e74] not found; src/models/mimo2.cpp:28
  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):
    [ec7630a] 8 required, missing in the file: []; none conditional
    [79e2e74] 8 required, missing in the file: []; none conditional
  routed-expert types per projection (layers): {"ffn_down_exps": {"MXFP4": 47}, "ffn_gate_exps": {"MXFP4": 47}, "ffn_up_exps": {"MXFP4": 47}}

(a) tools/gguf_quant_audit.py on the header-only shards (files total 169005098944 B = 157.398 GiB)
  # MiMo-V2.6-Flash-MOPD-MXFP4.gguf (1 file(s)), architecture 'mimo2'
  total 309.77B params, 157.39 GiB, 4.365 bpw
    routed_experts   302.795B    149.81 GiB   4.250 bpw  [MXFP4 100.0%]
    attention          4.766B      4.72 GiB   8.500 bpw  [Q8_0 100.0%, F32 0.0%]
    dense_ffn          0.805B      0.80 GiB   8.500 bpw  [Q8_0 100.0%]
    embedding          0.625B      1.16 GiB  16.000 bpw  [BF16 100.0%]
    output_head        0.625B      0.62 GiB   8.500 bpw  [Q8_0 100.0%]
    other              0.150B      0.28 GiB  16.273 bpw  [Q8_0 66.9%, F32 33.1%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: blk.1.ffn_gate_exps.weight MXFP4 4.250, blk.1.ffn_up_exps.weight MXFP4 4.250, blk.1.ffn_down_exps.weight MXFP4 4.250, blk.2.ffn_gate_exps.weight MXFP4 4.250, blk.2.ffn_up_exps.weight MXFP4 4.250, blk.2.ffn_down_exps.weight MXFP4 4.250
```
