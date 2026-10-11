# N15: real GGUF headers by HTTP Range reads, without downloading weights (CORRECTNESS gap 9) — round 1

Date 2026-10-11. Role: model-analyst. Branch `claude/n15-gguf-headers`, cut from `claude/n14-chat-templates` @`ac419b0`.
Hardware-free. `<S>` stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution. **Round 1 builds and tests the tooling and commits the requests; every fact about the real files is
PENDING runner** (sections 7–8). The runner job starts when the lead pushes this branch; round 2 (analysis of the bot's
commit, the "Статус пробела 9" paragraph in docs/reassessment/CORRECTNESS.md) follows.

## Result (round 1)

1. **Fetcher, CONFIRMED by tests (section 5):** `scripts/gguf_headers_fetch.py` (stdlib) pins the revision through the
   hub revision API, lists the tree (paginated by `Link`), matches globs, and reads each file's header with Range
   requests that start at `initial_bytes` and at least double until `tools/gguf_headers/gguf_header.py` parses magic,
   version, all KV pairs and all tensor infos up to the aligned data offset. It never asks past `max_header_bytes`
   (hard cap 64 MiB per file) or the file size; a header that needs more is an error, never a full download; a `200`
   answer to a Range request is rejected before its body is read; every `206` must carry exactly the asked range and
   the tree listing's file size. Outputs per file: `<file>.header.gz` (bytes `[0, data_offset)`, gzip -9, mtime 0),
   `<file>.summary.json` (all KV, arrays over 4096 elements as count + sha256; tensor list), and per request `PIN.json`
   (LFS sha256 and size of the whole file from the tree API, header bytes and sha256, bytes fetched, the Range
   requests, the resolve redirect's `X-Linked-Etag` / `X-Repo-Commit` / `X-Linked-Size`) or `ERROR.json`. A 401/403/404
   from the revision API is recorded (`status: repo_unavailable`), as is a glob without matches (`status: no_match`,
   with the repo's `.gguf` listing); any other count than `expect_count`, a revision mismatch or the cap are errors.
2. **Workflow `.github/workflows/gguf-headers.yml`, CONFIRMED by its test (section 5) and a YAML parse:** push on
   `claude/**` with paths `data/gguf-headers/*/REQUEST.json`, plus `workflow_dispatch`; job `if:
   startsWith(github.ref_name, 'claude/')`; branch through `env: REF`; commit loop `git pull -q --rebase origin "$REF"`
   and a push to `HEAD:"$REF"` (the only push line); never `main`; timeout 20 min; steps: gguf-py at `ec7630a` + numpy +
   PyYAML, self-test of `tests.test_gguf_headers` (fails on any skip except the one that needs a llama.cpp build),
   `--plan` (fails on a cap violation before any download), fetch, commit (also after a failed fetch, so `ERROR.json`
   reaches the branch; the job stays failed).
3. **Deviation from the task text, by a hook:** the requests are `data/gguf-headers/<name>/REQUEST.json`, not
   `third_party/gguf-headers/<name>/REQUEST.json`. The project hook `scripts/hooks/guard-upstream.sh` blocks agents'
   Edit/Write anywhere under `third_party/` (CONFIRMED: the Write calls were refused with "Blocked: …
   third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/REQUEST.json is a pinned upstream tree"); writing the file through
   Bash instead would have worked around the guard, so it was not done. The CI-fetched bytes still go to
   `third_party/gguf-headers/<name>/` (the bot is not subject to the hook; `third_party/README.md`: "nothing here is
   typed or edited by an agent"). The README row for `third_party/gguf-headers/` is not added for the same reason
   (`data/gguf-headers/README.md` documents the mechanism).
4. **Analyzer `tools/gguf_headers/analyze.py`, CONFIRMED on generated inputs (section 5); real inputs PENDING runner:**
   integrity (header bytes vs PIN sha256, summary re-derived byte for byte), (e) identity and whether each `--src`
   checkout knows the architecture / tokenizer model / pre-tokenizer, (d) expert groups, (c) embedded chat template vs
   the vendored HF template, (b) real KV next to the tinygen variants + a `config.json` cross-check + 35 graph-branch
   probes (`tools/gguf_headers/gguf_branches.py`: 21 for glm5-next, 17 for qwen4exp, 3 shared), (a)+(f)
   `tools/gguf_quant_audit.py` on header-only shards (unchanged: it already reads headers only; the test shows its
   result on the rebuilt header-only shards equals its result on the full files).
5. **Stretch prepared, not run on real data:** `tools/gguf_headers/vocab_gguf.py` writes a vocab-only GGUF from a
   committed header; the `llama-tokenize` of `79e2e74` reads it (section 6). The token-level check of N14's prefix
   claims needs the real vocabularies: round 2.
6. **Tests:** `tests/test_gguf_headers.py`, 31 tests, offline (local Range-capable fake hub); all 31 run and pass with
   every optional environment; mutation check 18/18 killed (section 9).

## 1. Inputs

| input | value | label |
|---|---|---|
| llama.cpp CI pin | `<S>/llama-ec7630a`, HEAD `ec7630a640789c393694fb194f1bbbf0369fc62d` (`git rev-parse HEAD`) | CONFIRMED |
| llama.cpp R1 pin | `<S>/llama.cpp`, HEAD `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` | CONFIRMED |
| ik_llama.cpp | `<S>/n12/ik-src`, HEAD `5194a9e1451b885e24dd5a065f650864e8807c59` (read only, for `--src ik=`) | CONFIRMED |
| GGUF format | `ggml/src/gguf.cpp` @ec7630a: alignment key `general.alignment` must be uint32 and a power of 2 (:622-636), data section padded only when `n_tensors > 0` (:781-788), tensor offsets must equal the running padded sizes (:790-820) | CONFIRMED (read) |
| gguf-py reader/writer | `<S>/llama-ec7630a/gguf-py`: `GGUFReader.data_offset` always pads (gguf_reader.py:190-193), `GGUFWriter` pads after the infos (gguf_writer.py:447) | CONFIRMED (read); the parser follows gguf.cpp |
| repo ids | Qwen: `unsloth/Qwen3.8-Flash-Next-GGUF` @`38bb39ee97821de2c9009abb7e93950eec396e66` (`scripts/r1/configs/llamacpp-server.json:51`); GLM: no repo id anywhere in the repository (grep of scripts/r1/configs, tools/engines, docs/DECISIONS.md, docs/evidence/2026-10-03-*, whole repo); the host path `GLM-5.3-Flash-GGUF/UD-Q4_K_XL/` (AGENTS.md) suggests `unsloth/GLM-5.3-Flash-GGUF` | Qwen CONFIRMED (from the config); GLM repo id UNKNOWN until the runner answers |

## 2. What the runner will fetch (`--plan`, fetches nothing)

```
$ cd <S>/n15 && python3 /home/user/wt-n15/scripts/gguf_headers_fetch.py --plan --requests /home/user/wt-n15/data/gguf-headers --out /home/user/wt-n15/third_party/gguf-headers
/home/user/wt-n15/data/gguf-headers/discover-glm-mimo/REQUEST.json: discover, 2 searches, reads no file
/home/user/wt-n15/data/gguf-headers/glm-5.3-flash-ud-q4_k_xl/REQUEST.json: unsloth/GLM-5.3-Flash-GGUF @ main, globs ['*GLM-5.3-Flash-UD-Q4_K_XL-0000?-of-00006.gguf'], 6 files x cap 67108864 B = at most 402653184 B
/home/user/wt-n15/data/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl/REQUEST.json: unsloth/Qwen3.8-Flash-Next-GGUF @ 38bb39ee97821de2c9009abb7e93950eec396e66, globs ['*Qwen3.8-Flash-Next-UD-Q4_K_XL-0000?-of-00004.gguf'], 4 files x cap 67108864 B = at most 268435456 B
pending requests: 3; worst-case download 671088640 B (per-file hard cap 67108864 B, per-run cap 1073741824 B)
exit 0
```

- **Maximum file bytes the run can download: 671 088 640 B (640 MiB)**, CONFIRMED as the enforced bound: per file at
  most `max_header_bytes` = 64 MiB (the fetcher never requests past it; tests `test_cap_exceeded_…`, mutation M8),
  10 files; the plan refuses more than 1 GiB per run. Hub API answers are separate and capped at 32 MiB each
  (`API_MAX_BYTES`): per headers request one revision call and the tree pages (at most 50); the discover request makes
  2 search calls and at most 25 tree listings. The self-test step also fetches llama.cpp `ec7630a` (shallow) from
  GitHub and numpy / PyYAML from PyPI — not Hugging Face.
- **Expected bytes, PROVISIONAL** (estimate, not measured): first shards carry the vocabularies (GLM `vocab_size`
  154880, Qwen 248320 in the vendored `config.json`) plus merges, roughly 5–10 MB of header each (assumed: about
  7 B per token string + its 8 B length, about as many merges as tokens at about 18 B each, 4 B per token type); with
  `initial_bytes` 256 KiB and doubling, fetched ≤ 2 × header per first shard and 256 KiB per other shard (their headers
  hold only split keys and tensor infos): about 15–25 MB per model. Committed (gzip): about 2–4 MB per model plus the
  summaries. The actual numbers will be `bytes_fetched` / `committed_bytes` in each PIN.json (round 2).
- GLM: if `unsloth/GLM-5.3-Flash-GGUF` does not exist (or the glob matches nothing), PIN.json records
  `repo_unavailable` / `no_match` and the job stays green; `discover-glm-mimo/LISTING.json` lists the GLM-5.3-Flash
  GGUF repos with the hub's parsed `gguf.architecture` (`glm5-next` / `glm5next` marked) and their `.gguf` paths, and
  MiMo-V2.6-Flash (`mimo2`). A corrected request is then one new commit.

## 3. Safety properties against `vendor-hf-branch.yml`

| property | vendor-hf-branch.yml | gguf-headers.yml |
|---|---|---|
| trigger | push `claude/**`, paths `third_party/hf/*/REQUEST.json`; workflow_dispatch | push `claude/**`, paths `data/gguf-headers/*/REQUEST.json`; workflow_dispatch |
| job guard | `if: startsWith(github.ref_name, 'claude/')` | same |
| branch name in scripts | `env: REF` | same (test: no `${{` in any `run`) |
| commit loop | `git pull -q --rebase origin "$REF"` + push `HEAD:"$REF"`, 3 tries | same (test: it is the only line with `push`) |
| main | never | never (test: no `main` in any `run`) |
| timeout | 15 min | 20 min |
| download bound | small text files (< 1 MiB, non-LFS) | 64 MiB per file, 1 GiB per run, `--plan` step before the fetch |

## 4. Analyzer and branch probes

`python3 tools/gguf_headers/analyze.py [--src ec7630a=<S>/llama-ec7630a] [--src 79e2e74=<S>/llama.cpp] [--src
ik=<S>/n12/ik-src] [--json OUT]` (with `LLAMA_CPP_GGUF_PY` and numpy for the tiny models). Every file:line it prints is
located at run time in the checkout given (literal anchor text), never typed. The probes (`tools/gguf_headers/
gguf_branches.py`; each anchor's presence at `ec7630a` / `79e2e74` is checked by `BranchAnchors`):

- glm5-next (21): `glm.kda_gate_bound` (bounded sigmoid vs softplus decay), `glm.layer_pattern` (DSA every 4th
  layer), `glm.indexer_types` (full vs shared indexer: `attention.indexer.types`), `glm.kpool_select_tail`,
  `glm.dsa_threshold` (selection only above top_k context; ec7630a's ubatch ≤ 16 gather path), `glm.rope` (nope-only
  assert; an absent `rope.dimension_count` would default to the key length and fail it), `glm.hc`, `glm.dense_lead`,
  `glm.gating`, `glm.shexp_clamp`, `glm.layer_norm_eps` (indexer key LayerNorm eps; 0 when the key is absent),
  `glm.ff_exp_array`, `glm.kda_qkv` (fused `attn_qkv` is loadable but the graph reads wq/wk/wv), `glm.conv_rank`,
  `glm.mtp`, `glm.mtp_heads` (own nextn embedding / head / norm), `moe.groups`, `moe.norm_w`, `moe.scale`,
  `moe.clamp_glm`, `moe.used_array`.
- qwen4exp (17): `moe.groups`, `moe.scale`, `moe.clamp_qwen` (qwen4exp is not in the fused clamp list),
  `moe.used_array`, `rope.scaling` (YaRN etc.), `qwen.hc`, `qwen.recurrent_source`, `qwen.full_attention` (QSA vs
  dense full attention), `qwen.ple`, `qwen.ple_image_token`, `qwen.output` (tied or own head), `qwen.attn_scale`,
  `qwen.ff_defaults`, `qwen.rope_sections`, `qwen.partial_rope`, `qwen.gate_up_fused`, `qwen.mtp`.

A real-file branch counts as "exercised by tiny" when its labels are within the union over the tinygen variants
(glm: default 12 layers, `--mtp`, `--quant-dims`, `--dialect ik`; qwen: default, `--mtp`, `--quant-dims`). On the tiny
models themselves every probe reports "tiny OK" (test `test_tiny_models_exercise_their_own_branches`). The
`config.json` cross-check uses llama.cpp's key names; keys a different converter (Unsloth `glm5next`) names
differently show as absent, which is itself the schema answer gap 5 asked for.

## 5. Tests

At `297e319`:

```
$ cd /home/user/wt-n15 && git log --oneline -n 1 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > <S>/n15/full-suite-297e319.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n15/full-suite-297e319.log
297e319 [model-analyst] tools/gguf_headers: gguf_header.py docstring cites gguf.cpp:790-820 (the offset check block), not :790-816
exit 0
Ran 891 tests in 146.143s
OK (skipped=358)
```

The one new skip in `discover` is `VocabOnly.test_llama_tokenize_runs_on_the_vocab_only_file` (no `LLAMA_TOKENIZE_BIN`
in that run); the run that failed to import the module before `098217c` had `Ran 861 tests … FAILED (errors=1,
skipped=357)` (`<S>/n15/full-suite.log`), i.e. 860 tests + the import error, 357 skips.

```
$ python3 -m unittest tests.test_hooks tests.test_check_unittest 2>&1 | tail -3
Ran 54 tests in 13.359s

OK
```

```
$ python3 -c "import yaml; yaml.safe_load(open('.github/workflows/gguf-headers.yml')); print('YAML OK .github/workflows/gguf-headers.yml')"
YAML OK .github/workflows/gguf-headers.yml
```

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py LLAMA_CPP_SRC_79E2E74=<S>/llama.cpp LLAMA_TOKENIZE_BIN=<S>/n15/build-tok/bin/llama-tokenize python3 -m unittest tests.test_gguf_headers -v > <S>/n15/module-all-env-297e319.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n15/module-all-env-297e319.log; grep -c " \.\.\. ok$" <S>/n15/module-all-env-297e319.log; grep -c "skipped" <S>/n15/module-all-env-297e319.log
exit 0
Ran 31 tests in 15.392s
OK
31
0
```

(at `297e319`: 31 lines end in "... ok", none contains "skipped". `gguf-headers.yml` is the only workflow changed
since `ac419b0`: `git diff --name-only ac419b0..HEAD -- .github/workflows`.)

The workflow's self-test filter, simulated locally at `762585f` (its commit message):
`with-gguf-py: unittest rc=0, Ran 31 tests in 16.336s OK (skipped=1), step passes` /
`without: unittest rc=0, Ran 31 tests in 12.607s OK (skipped=4), step FAILS (3 other skips)`.

What the classes check: `Parser` (exact `NeedMore` needs; alignment rules of gguf.cpp incl. 256-byte alignment and
no padding without tensors; rejections; big-array digests; contiguous offsets; byte-for-byte rebuild); `Fetch` against
the fake hub (stops at the header with contiguous ranges and no byte fetched twice, bytes served == bytes recorded;
revision pinned from `main`, PIN bound to the request's sha256; summary re-derived from the committed header,
deterministic gzip; audit on header-only shards == audit on the files; cap exceeded → error with ≤ cap bytes served
and no outputs; `--plan` failures with zero hub requests; revision mismatch; 404 and 401 recorded; no match recorded
with the listing; count mismatch; a 200 answer rejected; a Range answer for another file size rejected; wrong etag
warned; done requests not fetched again, failed ones retried; tree pagination; discovery lists and reads no file);
`Tinygen` (data and tensor offsets, sizes and shapes equal gguf-py's `GGUFReader` on glm5next `--mtp --dialect ik`
and qwen4exp `--mtp`); `Workflow` (section 3); `Analyzer` (template equal / different, groups 1/1 vs 8/4, identity,
audit, integrity failure, pending / unavailable, tiny models exercise their own branches); `VocabOnly`; `BranchAnchors`.

## 6. Stretch: vocab-only GGUF and llama.cpp's tokenizer (prepared; the real check is round 2)

`llama-tokenize` of the R1 pin built in scratch (`cmake -S <S>/llama.cpp -B <S>/n15/build-tok -DGGML_NATIVE=OFF
-DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF`, target `llama-tokenize`; `version: 0.6.0-dev (build 1,
commit 79e2e74)`). On a GLM-like split header with a byte-level BPE vocabulary (256 bytes + one merge `h e`):

```
$ cd <S>/n15/vocab
$ python3 /home/user/wt-n15/tools/gguf_headers/vocab_gguf.py hdr.bin v-glm5-next.gguf --arch glm5-next
wrote v-glm5-next.gguf: 4069 B, 11 KV, architecture 'glm5-next', tokenizer 'gpt2' / pre 'glm4'
$ python3 /home/user/wt-n15/tools/gguf_headers/vocab_gguf.py hdr.bin v-glm5next.gguf
wrote v-glm5next.gguf: 4066 B, 11 KV, architecture 'glm5next', tokenizer 'gpt2' / pre 'glm4'
$ <S>/n15/build-tok/bin/llama-tokenize -m v-glm5-next.gguf -p hehe --ids --log-disable
[259, 259]
exit 0
$ <S>/n15/build-tok/bin/llama-tokenize -m v-glm5next.gguf -p hehe --ids 2>&1 | grep -i "unknown\|error\|failed"
0.00.003.555 E llama_model_load: error loading model: unknown model architecture: 'glm5next'
0.00.003.589 E llama_model_load_from_file_impl: failed to load model
0.00.003.589 E error: could not load model from file 'v-glm5next.gguf'.
exit 1
```

What this shows (CONFIRMED): the pipeline header → vocab-only GGUF → llama.cpp tokenizer works, and the R1 pin needs
the architecture renamed for a `glm5next` file. What it does not show: anything about the real vocabularies. Round 2:
vocab-only files from the committed first shards, then `tokens(prompt_k) ++ tokens(turn_k)` against
`tokens(prompt_{k+1})` over N14's sessions (`tools/chattmpl/analysis.py` `loop_checks`), with the limit that a model's
sampled tokens are approximated by the canonical tokenization of its turn.

## 7. Results on the real files — PENDING runner

- (a) `tools/gguf_quant_audit.py` on the header-only shards: PENDING runner.
- (b) real hyperparameters vs the tiny configs, `config.json` cross-check, probes not exercised by the tiny models with
  file:line at both pins: PENDING runner.
- (c) embedded `tokenizer.chat_template` vs `third_party/hf/glm-5.3-flash-chat/chat_template.jinja` (`eb9eb20`) and
  `third_party/hf/qwen3.8-flash-next-chat/chat_template.jinja` (`de4b8e4`), byte for byte: PENDING runner (closes N14's
  UNKNOWN on the Unsloth-embedded template).
- (d) `glm5next.expert_group_count` / `expert_group_used_count` of the Unsloth file: PENDING runner (closes the N12 /
  N14 UNKNOWN).
- (e) architecture string, tokenizer model / pre-tokenizer, vocab size, special ids; known at `ec7630a` / `79e2e74` /
  ik `5194a9e`: PENDING runner.
- (f) ADR-012 D5 (UD-Q4_K_XL ≥ 4 bpw for routed experts and non-expert matrices): PENDING runner.
- GLM repo id and file listing; discovery listing (glm5-next GGUFs for llama.cpp mainline, MiMo-V2.6-Flash): PENDING
  runner.

To confirm the committed GLM header against the real files on `epyc-ai` without moving them (UNKNOWN until run there):
`head -c <header_bytes> GLM-5.3-Flash-UD-Q4_K_XL-0000N-of-00006.gguf | sha256sum` must print the file's
`header_sha256` from PIN.json (both values PENDING runner). The header's integrity relative to the whole-file LFS
sha256 cannot be verified without reading the whole file; what binds it is the pinned commit in the resolve URL, the
`Content-Range` total equal to the LFS size, the hub's `X-Linked-Etag` (recorded, compared with the LFS sha256), and
the size check (header data offset + padded tensor sizes = file size, recorded per file).

## 8. NOT RUN

- The runner job itself (the sandbox cannot push or reach huggingface.co): NOT RUN until the lead pushes.
- Any check on real headers (section 7).
- Token-level check of the N14 claims on the real vocabularies (section 6): needs the committed headers.
- `analyze.py --src ik=`: run only on generated inputs here; ik's source layout differs (`src/models/glm5-next.cpp`,
  `src/llama-graph.cpp` print "file absent" at ik), so for ik only the architecture / tokenizer support lines are
  meaningful.

## 9. Mutation check (`docs/evidence/mutation-scripts/2026-10-11-n15-mutations.sh`)

Each mutation runs the whole `tests.test_gguf_headers` module on a fresh `git archive HEAD` copy, with gguf-py of
`ec7630a`, `LLAMA_CPP_SRC_79E2E74` set (no `LLAMA_TOKENIZE_BIN`: the one skip in `base`).

```
$ cd <S>/n15 && git -C /home/user/wt-n15 log --oneline -n 1 && bash /home/user/wt-n15/docs/evidence/mutation-scripts/2026-10-11-n15-mutations.sh > mutations-297e319.log 2>&1; echo "exit $?"; cut -c1-400 mutations-297e319.log
297e319 [model-analyst] tools/gguf_headers: gguf_header.py docstring cites gguf.cpp:790-820 (the offset check block), not :790-816
exit 0
HEAD 297e319
base exit 0 (Ran 31 tests in 14.331s, OK (skipped=1))
M1-no-data-padding KILLED  (Fetch.test_fetch_stops_at_the_header_and_pins_the_revision Parser.test_alignment_rules_of_gguf_cpp Parser.test_needmore_reports_the_exact_next_need Parser.test_summary_digests_big_arrays_and_keeps_the_rest Tinygen.test_offsets_agree_with_gguf_py )
M2-pad-without-tensors KILLED  (Parser.test_alignment_rules_of_gguf_cpp Parser.test_rebuild_copies_values_byte_for_byte VocabOnly.test_vocab_only_file_keeps_the_tokenizer_bytes_and_renames_the_arch )
M3-default-alignment-16 KILLED  (Fetch.test_fetch_stops_at_the_header_and_pins_the_revision Parser.test_summary_digests_big_arrays_and_keeps_the_rest Tinygen.test_offsets_agree_with_gguf_py )
M4-ignore-alignment-key KILLED  (Parser.test_alignment_rules_of_gguf_cpp )
M5-no-contiguity-check KILLED  (Parser.test_tensor_offsets_must_be_contiguous_like_gguf_cpp )
M6-inline-big-arrays KILLED  (Parser.test_summary_digests_big_arrays_and_keeps_the_rest )
M7-keep-infos-end-only KILLED  (Analyzer.test_integrity_failure_is_reported Analyzer.test_template_groups_identity_and_audit Analyzer.test_tiny_models_exercise_their_own_branches Fetch.test_fetch_stops_at_the_header_and_pins_the_revision Fetch.test_summary_round_trips_from_the_committed_header Tinygen.test_offsets_agree_with_gguf_py )
M8-no-cap KILLED  (Fetch.test_cap_exceeded_is_an_error_and_never_a_full_download )
M9-accept-200 KILLED  (Fetch.test_a_server_that_ignores_range_is_rejected_unread )
M10-no-total-check KILLED  (Fetch.test_a_range_of_another_file_size_is_rejected )
M11-no-revision-pin-check KILLED  (Fetch.test_revision_mismatch_is_an_error )
M12-refetch-done-requests KILLED  (Fetch.test_done_requests_are_not_fetched_again_and_errors_retry )
M13-no-run-cap KILLED  (Fetch.test_plan_fails_before_any_request_on_caps )
M14-no-hard-cap KILLED  (Fetch.test_plan_fails_before_any_request_on_caps )
M15-etag-inverted KILLED  (Fetch.test_fetch_stops_at_the_header_and_pins_the_revision Fetch.test_wrong_etag_is_recorded_and_warned )
M16-template-always-equal KILLED  (Analyzer.test_template_groups_identity_and_audit )
M17-groups-threshold KILLED  (Analyzer.test_template_groups_identity_and_audit )
M18-no-summary-rederivation KILLED  (Analyzer.test_integrity_failure_is_reported )
```

**18 of 18 killed.** The offset / alignment logic of the parser and fetcher (M1–M5, M7) is killed by the parser tests
and by the comparison with gguf-py on the tinygen models. M9 (status check removed) is killed by the error-text
assertion only: the request still fails, on the second guard, before any body is read (`FetchError: Content-Range ''
does not match bytes 0-4095/6701696`, `<S>/n15/mut/M9-accept-200/test.log`), i.e. M9 removes one of two guards. An
earlier run of the script at `35db1e3` (before the vocab-only tests) also gave 18/18; a run started at `762585f`
straddled the docstring-only commit `297e319` and is not cited.

## 10. Corrections

- Commit `5392c17`'s message says "36 probes"; the table has 35 (`python3 -c "…print(len(B.PROBES))"` → `total 35`).
  It also names `branches.py`, renamed to `gguf_branches.py` in `098217c` (the generic module name `analyze` had
  collided with `tools/g0/analyze.py` under `unittest discover`: `ImportError … AttributeError: module 'analyze' has
  no attribute 'git_head'`, log `<S>/n15/full-suite.log`).
- Commit `88e71f8` describes requests under `third_party/gguf-headers/`; `676ed79` moved them to `data/gguf-headers/`
  (Result 3).
