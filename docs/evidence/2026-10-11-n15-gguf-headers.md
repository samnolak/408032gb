# N15: real GGUF headers by HTTP Range reads, without downloading weights (CORRECTNESS gap 9)

Date 2026-10-11. Role: model-analyst. Branch `claude/n15-gguf-headers`, cut from `claude/n14-chat-templates` @`ac419b0`.
Hardware-free. `<S>` stands for the session scratch directory
`<S>`; outputs are pasted with only that
substitution. Round 1 built and tested the tooling and committed the requests (sections 1–6, 9). The lead pushed
`4eae257`; workflow `gguf-headers` run 38136140582 read the headers and committed them as bot commit `6cce416`
(fast-forwarded here). Round 2 analyses them offline (section 7, Result (round 2), appendices A–C).

## Result (round 2: the real files)

1. **The runner read only headers, CONFIRMED** (run 38136140582: conclusion success, job 48 s, steps 1–7 success, read
   through the GitHub API, section 7.1; the step logs themselves are not readable from the sandbox: the proxy refuses
   the log blob host, so the self-test's literal output is NOT available — only its step conclusion). File bytes
   fetched (PIN.json): GLM 10 740 704 B, Qwen 11 733 056 B, together 22 473 760 B of the files' 311 041 976 256 B
   (289.7 GiB). Both first
   shards are **metadata-only files** (0 tensors: llama-gguf-split `--no-tensor-first-split`), so for them the whole
   file was read (9 429 984 B and 10 946 624 B: header + 28 / 6 bytes of writer padding); every tensor shard was read
   with one 262 144-byte Range, of which 1 929 952 B in total are tensor data past the headers (discarded, not
   committed). Committed: GLM 2 778 264 B, Qwen 3 338 303 B, discovery 128 062 B.
2. **Integrity, CONFIRMED:** all 10 committed headers decompress to the sha256 PIN.json records and re-derive their
   summaries byte for byte; hub `X-Linked-Etag` = LFS sha256 and `X-Repo-Commit` = pinned revision for all 10; header
   data offset + padded tensor sizes = file size for all 10 (the fetcher's own size check misread the metadata-only
   shards as "mismatch"; fixed in `9c5384c`, section 7.2).
3. **GLM: `unsloth/GLM-5.3-Flash-GGUF` @`a38483c8cd5df544f53d70fb281afe97369d5ab6`, architecture string `glm5-next`
   — not `glm5next`** (CONFIRMED from the header). Its tensors total 185.983 GiB at 4.981 bpw, the numbers AGENTS.md and
   PROBLEMS.md give for the host files they call `glm5next`; the repo was last modified 2026-10-05 (listing), after the
   host note of 2026-10-03. That the host holds an earlier upload of the same quantization with other metadata is
   PROVISIONAL (equal size and bpw, different architecture string); UNKNOWN until checked on epyc-ai (section 7.3).
4. **(e) Tokenizers and support, CONFIRMED:** GLM `gpt2` / pre `glm4`, 154 880 tokens; Qwen `gpt2` / pre `qwen35`,
   248 320 tokens. Both llama.cpp pins know `glm5-next`, `qwen4exp`, `gpt2`, `glm4`, `qwen35` (file:line in section 7.4);
   neither knows `glm5next`; ik `5194a9e` knows all of them. That both files load at both pins is PROVISIONAL (every
   required key is present and every layer has exactly the tensor names of the tiny models that load there, plus one
   optional tensor; not loaded: no weights).
5. **(d) `glm5-next.expert_group_count` = 1, `expert_group_used_count` = 1, CONFIRMED** — no group step at either pin
   (`src/llama-graph.cpp:2105@ec7630a`, `:2145@79e2e74` need > 1): N12's "possible 7th routing difference" does not
   arise on this file. Closes the N12 / N14 UNKNOWN. (Qwen: keys absent, no group step.)
6. **(c) The embedded chat templates are NOT byte-equal to the vendored HF ones, CONFIRMED** (closes N14's UNKNOWN; full
   diffs in appendix A). GLM (10 648 B vs 10 950 B): syntax rewrites only on N14's inputs — `x.0` → `x[0]`, `~` → `+`,
   4 `{%- break -%}` removed, `elif content is not none` → `else`; jinja2 renders all 68 of N14's inputs byte-identically
   with both. **It keeps the `{% set reasoning_content %}`-inside-`{% for m in messages %}` block** (vendored lines
   143–150, unchanged), so N14's llama.cpp Jinja leak applies to what R1 actually runs: llama-server @79e2e74 renders
   `<think>R1</think>a2` with the embedded template, the same 21 differences from jinja2 as with the vendored one
   (CONFIRMED live, section 7.5). Qwen (9 993 B vs 8 952 B, "Unsloth fixes - developer role, merged system messages,
   tool calling"): `reasoning_effort: "high"` is mapped to `xhigh` — N14's HTTP 500 is gone (server vs jinja2 70 of 72
   equal, the 2 others the by-design `--prefill-assistant` case); a system-only conversation is no longer refused;
   leading system/developer messages are merged; tool-call arguments given as a JSON string raise. N14's B3 prefix
   results are unchanged for both embedded templates.
7. **(b) Real hyperparameters vs the tiny models:** every key present matches the vendored `config.json` (GLM 33 of
   34 — the 34th, `index_kpool_always_select_tail`, is absent and llama.cpp's default `true` equals it; Qwen 27 of 28 —
   `mtp_num_hidden_layers` 1 is absent: no MTP block; CONFIRMED). Graph branches only the real geometry takes (CONFIRMED by the probes, file:line at both pins in section
   7.6): GLM — indexer key LayerNorm eps 1e-6 (tiny: key absent, eps 0), the MTP block's own
   `nextn.shared_head_norm` (79e2e74 `glm5-next.cpp:647`; ec7630a has no MTP graph), the shared-expert clamp from its
   own key (same value 10 as the routed clamp); Qwen — layer pattern from `full_attention_interval` (no
   `recurrent_layers` array: `qwen4exp.cpp:149` else-branch), the PLE image-token key (used only for image batches).
   **The Qwen UD-Q4_K_XL file has no MTP block** (no `nextn_predict_layers`, no `nextn` tensors; config has 1).
8. **(f) ADR-012 D5, CONFIRMED PASS for both** (`tools/gguf_quant_audit.py` on the header-only shards): GLM routed
   experts 4.874 bpw [Q4_K 65.1 %, Q5_K 32.6 %, Q6_K 2.3 %], every other matrix Q8_0; Qwen routed experts 5.101 bpw
   [Q4_K 65.3 %, Q5_1 29.9 %, Q8_0 3.5 %, Q5_K 1.4 %] (its `ffn_down_exps` are Q5_1 / Q8_0: width 640 is not a multiple
   of 256), the PLE table `per_layer_token_embd.weight` IQ4_NL [160, 320 001 536] = 26.822 GiB of the 103.678 GiB.
   Lowest matrix: 4.5 bpw (Q4_K, IQ4_NL) in both.
9. **Discovery (listing only), CONFIRMED from LISTING.json:** 95 GGUF repos for "GLM-5.3-Flash" (34 `glm5-next`, 46
   `glm5next`), 30 for "MiMo-V2.6-Flash" (28 `mimo2`). glm5-next GGUFs for llama.cpp mainline exist: the current
   `unsloth/GLM-5.3-Flash-GGUF` itself, and `ggml-org/GLM-5.3-Flash-GGUF` @`c259a9f` (Q2_K, Q2_K_S, Q4_K = 168.91 GiB,
   separate `mtp-GLM-5.3-Flash-Q4_0/Q8_0.gguf`; its embedded template sha256, as parsed by the hub, equals the vendored
   HF template's). MiMo: `ggml-org/MiMo-V2.6-Flash-MOPD-GGUF` and `-RL-GGUF` (mimo2) among 28; their file lists were not
   fetched (the 25-repo listing cap was spent on GLM).
10. **Stretch, token level, CONFIRMED for the tokenizer only** (llama-tokenize @79e2e74 on vocab-only GGUFs built from
    the committed headers; the server's own tokenization path and cache are not exercised; sampled tokens approximated
    by the canonical tokenization of each turn; section 7.9): GLM — token-level prefix stability equals byte-level on
    all 24 request pairs of N14's 4 sessions; all 16 template tags are single tokens. Qwen — equal on 23 of 24; **one
    break that bytes do not show**: in the session with an empty reasoning turn the prompt ends `<think>\n` and the turn
    starts `\n</think>`; tokenized whole, the next prompt merges them into one `ĊĊ` token, so reuse stops at token
    1058 of 1117. `<think>`, `</think>`, `<tool_call>`, `<tool_response>` are single (user-defined) tokens;
    `<function=`, `<parameter=` are 3 ordinary tokens each.

## Result (round 1: tooling)

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
4. **Analyzer `tools/gguf_headers/analyze.py`, CONFIRMED on generated inputs (section 5); on the real inputs: section 7:**
   integrity (header bytes vs PIN sha256, summary re-derived byte for byte), (e) identity and whether each `--src`
   checkout knows the architecture / tokenizer model / pre-tokenizer, (d) expert groups, (c) embedded chat template vs
   the vendored HF template, (b) real KV next to the tinygen variants + a `config.json` cross-check + 35 graph-branch
   probes (`tools/gguf_headers/gguf_branches.py`: 21 for glm5-next, 17 for qwen4exp, 3 shared), (a)+(f)
   `tools/gguf_quant_audit.py` on header-only shards (unchanged: it already reads headers only; the test shows its
   result on the rebuilt header-only shards equals its result on the full files).
5. **Stretch:** `tools/gguf_headers/vocab_gguf.py` writes a vocab-only GGUF from a committed header; the
   `llama-tokenize` of `79e2e74` reads it (section 6). Run on the real vocabularies in round 2 (section 7.9).
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

## 7. Results on the real files (round 2)

`git fetch origin claude/n15-gguf-headers && git merge --ff-only origin/claude/n15-gguf-headers` brought bot commit
`6cce416` ("[model-analyst] gguf-headers: GGUF headers by Range reads under third_party/gguf-headers for
claude/n15-gguf-headers (workflow gguf-headers, run 38136140582)", author github-actions[bot], 23 files). Everything
below is computed offline from those committed bytes; the analyzer's full output is appendix A.

### 7.1 The runner run (GitHub API, read-only)

```
$ gh run view 38136140582 -R samnolak/408032gb      (first lines)
✓ claude/n15-gguf-headers gguf-headers · 38136140582
Triggered via push about 19 minutes ago

JOBS
✓ headers in 48s (ID 114460278964)
$ gh api repos/samnolak/408032gb/actions/runs/38136140582 --jq '{status, conclusion, head_sha, event, created_at, updated_at, head_branch}'
{"conclusion":"success","created_at":"2026-10-11T11:35:46Z","event":"push","head_branch":"claude/n15-gguf-headers","head_sha":"4eae257eb4f68bf6a75fb32c98fec730ddb57497","status":"completed","updated_at":"2026-10-11T11:36:38Z"}
$ gh api repos/samnolak/408032gb/actions/jobs/114460278964 --jq '.steps[] | "\(.number) \(.name) \(.conclusion) \(.started_at) \(.completed_at)"'
1 Set up job success 2026-10-11T11:35:50Z 2026-10-11T11:35:51Z
2 Run actions/checkout@v4 success 2026-10-11T11:35:51Z 2026-10-11T11:35:52Z
3 gguf-py of llama.cpp at the CI pin and numpy, PyYAML (for the self-test) success 2026-10-11T11:35:52Z 2026-10-11T11:36:00Z
4 self-test of the fetcher against a local Range server (no network; fails on any skip but llama-tokenize's) success 2026-10-11T11:36:00Z 2026-10-11T11:36:06Z
5 check the pending requests against the byte caps (fetches nothing; fails on a cap violation) success 2026-10-11T11:36:06Z 2026-10-11T11:36:06Z
6 read the GGUF headers with Range requests success 2026-10-11T11:36:06Z 2026-10-11T11:36:32Z
7 commit and push to this claude/* branch success 2026-10-11T11:36:32Z 2026-10-11T11:36:35Z
14 Post Run actions/checkout@v4 success 2026-10-11T11:36:35Z 2026-10-11T11:36:35Z
15 Complete job success 2026-10-11T11:36:35Z 2026-10-11T11:36:35Z
```

The 14 annotations of the job (`gh api repos/samnolak/408032gb/check-runs/114460278964/annotations`, level | message,
messages cut at 330 characters) — the Qwen request's closing notice is not among them (PROVISIONAL: GitHub's documented
limit of 10 notice annotations per step; the fetch step emitted 11 notices); its numbers are in PIN.json:

```
14 annotations
warning | Node.js 20 is deprecated. The following actions target Node.js 20 but are being forced to run on Node.js 24: actions/checkout@v4. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/
notice | headers commit 6cce416c4d46889a69cf44550757a8ae88a5da01
notice | unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf: GGUF v3, arch 'qwen4exp', 67 KV, 0 tensors, header 10946618 B (sha256 acc44040f74d0e55), 10946624 B fetched in 7 Range requests; size check mismatch: header implies 10946618 B; chat_template 12827f24b742ea4e
warning | unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf: mismatch: header implies 10946618 B (tensor offsets contiguous)
notice | /home/runner/work/408032gb/408032gb/data/gguf-headers/glm-5.3-flash-ud-q4_k_xl/REQUEST.json: status ok, revision a38483c8cd5df544f53d70fb281afe97369d5ab6, 6 headers, 10740704 B fetched, 2500627 B committed (gzip)
notice | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf: GGUF v3, arch None, 3 KV, 17 tensors, header 1248 B (sha256 dd1a523a0a9dcbad), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf: GGUF v3, arch None, 3 KV, 341 tensors, header 21632 B (sha256 1b4a3a705d62bdf4), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf: GGUF v3, arch None, 3 KV, 320 tensors, header 20256 B (sha256 7176860b6d2fdc38), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf: GGUF v3, arch None, 3 KV, 337 tensors, header 21376 B (sha256 34fc5c898088996d), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf: GGUF v3, arch None, 3 KV, 397 tensors, header 24608 B (sha256 a1e0c64d7bf1be65), 262144 B fetched in 1 Range requests; size check exact; chat_template absent
notice | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf: GGUF v3, arch 'glm5-next', 73 KV, 0 tensors, header 9429956 B (sha256 0c90bc17e2486d3e), 9429984 B fetched in 7 Range requests; size check mismatch: header implies 9429956 B; chat_template a4fddbbf0b432101
warning | unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf: mismatch: header implies 9429956 B (tensor offsets contiguous)
notice | discover 'MiMo-V2.6-Flash': 30 GGUF repos; architecture in ['mimo2']: ProCreations/MiMo-V2.6-Flash-RL-NVFP4 (mimo2), ggml-org/MiMo-V2.6-Flash-RL-GGUF (mimo2), MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-GGUF (mimo2), MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF (mimo2), pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-G
notice | discover 'GLM-5.3-Flash': 95 GGUF repos; architecture in ['glm5-next', 'glm5next']: huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF (glm5-next), unsloth/GLM-5.3-Flash-GGUF (glm5-next), autotrust/GLM-5.3-Flash-GGUF-DGX-Spark (glm5-next), orcarouter/GLM-5.3-Flash-Uncensored-GGUF (glm5next), BoldingBuilds/orcarouter_GLM-5.3-Flash-U
```

`gh run view 38136140582 --log` and the job-logs API fail from the sandbox (`failed to get run log: Get
"https://results-receiver.actions.githubusercontent.com/…": Forbidden`; the job log redirects to a blob host the proxy
refuses), so the self-test step's `Ran N tests` line is NOT available here; its step conclusion is `success`, which with
the step's own check means unittest OK with no skip other than the llama-tokenize one.

### 7.2 Bytes fetched and committed, integrity

From the committed PIN.json files (literal output of a short script over them):

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n15-committed-facts.py pins
== glm-5.3-flash-ud-q4_k_xl ok unsloth/GLM-5.3-Flash-GGUF main -> a38483c8cd5df544f53d70fb281afe97369d5ab6 gated False ['license:mit'] fetched_utc 2026-10-11T11:36:09Z
   bytes_fetched 10740704 committed_bytes (gz) 2500627 gguf files in repo 70
   GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf: size 9429984, lfs b59c030090f0, header 9429956, fetched 9429984 in 7, gz 2487157, size_check(PIN) 'mismatch: header implies 9429956 B', etag_ok True, commit_ok True
   GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf: size 49198039200, lfs d3ecb6ff3957, header 24608, fetched 262144 in 1, gz 3679, size_check(PIN) 'exact', etag_ok True, commit_ok True
   GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf: size 49988626304, lfs 328073c004e7, header 21376, fetched 262144 in 1, gz 3167, size_check(PIN) 'exact', etag_ok True, commit_ok True
   GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf: size 48700701344, lfs 6803ab7effa6, header 20256, fetched 262144 in 1, gz 3031, size_check(PIN) 'exact', etag_ok True, commit_ok True
   GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf: size 49026026752, lfs 852f3df9dacd, header 21632, fetched 262144 in 1, gz 3209, size_check(PIN) 'exact', etag_ok True, commit_ok True
   GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf: size 2784497888, lfs c6ba510fafc1, header 1248, fetched 262144 in 1, gz 384, size_check(PIN) 'exact', etag_ok True, commit_ok True
== qwen3.8-flash-next-ud-q4_k_xl ok unsloth/Qwen3.8-Flash-Next-GGUF 38bb39ee97821de2c9009abb7e93950eec396e66 -> 38bb39ee97821de2c9009abb7e93950eec396e66 gated False ['license:other'] fetched_utc 2026-10-11T11:36:21Z
   bytes_fetched 11733056 committed_bytes (gz) 3096890 gguf files in repo 56
   Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf: size 10946624, lfs 4448186216b3, header 10946618, fetched 10946624 in 7, gz 3085545, size_check(PIN) 'mismatch: header implies 10946618 B', etag_ok True, commit_ok True
   Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf: size 49859583136, lfs 3f342f1c1580, header 18720, fetched 262144 in 1, gz 2945, size_check(PIN) 'exact', etag_ok True, commit_ok True
   Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf: size 49376141504, lfs 56758f40269c, header 48064, fetched 262144 in 1, gz 6577, size_check(PIN) 'exact', etag_ok True, commit_ok True
   Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf: size 12087983520, lfs 753bda48b98b, header 11296, fetched 262144 in 1, gz 1823, size_check(PIN) 'exact', etag_ok True, commit_ok True
```

The two first shards hold 0 tensors (metadata only); the fetcher's doubling Range reads ended at the file size, so the
whole file was read: 9 429 984 = 9 429 956 (header) + 28 bytes of zero padding, 10 946 624 = 10 946 618 + 6. gguf.cpp's
writer pads after the tensor infos unconditionally (`gguf_write_out`, `ggml/src/gguf.cpp:1671` @ec7630a and @79e2e74)
while its reader starts the data section without padding when there are no tensors (`gguf.cpp:781`); the fetcher
@`4eae257` compared the file size with the unpadded offset and recorded "mismatch" (and a warning annotation). Fixed in
`9c5384c` (`gguf_header.size_check`, test `test_size_check_of_tensor_and_metadata_only_shards`); the analyzer recomputes
it: "exact" for all 10 files. Bytes of tensor data read past the headers (first Range of each tensor shard minus its
header): GLM 1 221 600 B, Qwen 708 352 B — discarded, never committed. Committed per directory: GLM 2 778 264 B (header.gz
2 500 627, summaries 255 875, PIN.json 21 762), Qwen 3 338 303 B (3 096 890 / 224 389 / 17 024), discovery LISTING.json
128 062 B. Integrity (analyzer, appendix A): every header.gz decompresses to PIN.json's `header_sha256`, and
`gguf_header.summarize()` of it equals the committed summary byte for byte — OK for all 10.

### 7.3 GLM: repo, revision, architecture string, and the host files

`unsloth/GLM-5.3-Flash-GGUF` @`a38483c8cd5df544f53d70fb281afe97369d5ab6` (requested `main`), MIT, not gated; 70 `.gguf`
files in 12 quant folders + 2 mmproj (LISTING.json). The UD-Q4_K_XL header says `general.architecture = "glm5-next"`
with `glm5-next.*` keys (73 KV), `general.quantized_by = "Unsloth"`, `general.file_type = 15`. Tensor payload: 1412
tensors, 185.983 GiB, 4.981 bpw; files 199 707 321 472 B (CONFIRMED, appendix A). AGENTS.md ("Weights on host
epyc-ai") and PROBLEMS.md (`GLM-5.3-Flash-GGUF/UD-Q4_K_XL/*-00001-of-00006.gguf | glm5next | 185.983 GiB (4.981 BPW)`)
describe the host files with the same size and bpw but architecture `glm5next`; the repo's `lastModified` is
2026-10-05T22:23:39Z, after that host note (2026-10-03). PROVISIONAL: the host holds an earlier upload of the same
quantization with other metadata. UNKNOWN until run on epyc-ai (no download needed there):

- `head -c 9429956 <host>/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf | sha256sum` → `0c90bc17e2486d3ed599dfda7e2b244bdded27c48d0680569077d741c11b1016` if the host's metadata shard is this revision;
- `sha256sum <host>/GLM-5.3-Flash-UD-Q4_K_XL-0000N-of-00006.gguf` for N = 2…6 against PIN.json's `lfs_sha256`
  (`d3ecb6ff…`, `328073c0…`, `6803ab7e…`, `852f3df9…`, `c6ba510f…`): the tensor shards carry no architecture string
  (3 KV: `split.no`, `split.count`, `split.tensors.count`), so equal hashes would mean byte-identical weights with only
  the metadata shard relabelled.

### 7.4 (e) Identity and support at the pins

Analyzer output (appendix A), both models:

```
=== glm-5.3-flash-ud-q4_k_xl (third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl)
(e) identity
  architecture: glm5-next
  general.name: GLM 5.3 Flash
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: glm4
  vocab (tokenizer.ggml.tokens): 154880
  merges: 321649
  token_embd shape: [4096, 154880]
  output shape: [4096, 154880]
  tokenizer.ggml.bos_token_id: 154822
  tokenizer.ggml.eom_token_id: 154829
  tokenizer.ggml.eos_token_id: 154820
  tokenizer.ggml.eot_token_id: 154827
  tokenizer.ggml.padding_token_id: 154820
  tokenizer.ggml.unknown_token_id: 154820
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.file_type": 15, "general.languages": ["en", "zh"], "general.license": "mit", "general.quantization_version": 2, "general.quantized_by": "Unsloth", "general.repo_url": "https://huggingface...
  [ec7630a] arch glm5-next: src/llama-arch.cpp:156
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer glm4: src/llama-vocab.cpp:2290
  [79e2e74] arch glm5-next: src/llama-arch.cpp:158
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer glm4: src/llama-vocab.cpp:2407
  [ik] arch glm5-next: src/llama-arch.cpp:91
  [ik] tokenizer model gpt2: src/llama-vocab.cpp:1816
  [ik] pre-tokenizer glm4: src/llama-vocab.cpp:2074

=== qwen3.8-flash-next-ud-q4_k_xl (third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl)
(e) identity
  architecture: qwen4exp
  general.name: Qwen3.8 Flash Next
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: qwen35
  vocab (tokenizer.ggml.tokens): 248320
  merges: 247587
  token_embd shape: [2560, 248320]
  output shape: [2560, 248320]
  tokenizer.ggml.add_bos_token: false
  tokenizer.ggml.bos_token_id: 248044
  tokenizer.ggml.eos_token_id: 248046
  tokenizer.ggml.padding_token_id: 248044
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.description": "A Preview of the Qwen4 Architecture", "general.file_type": 15, "general.quantization_version": 2, "general.quantized_by": "Unsloth", "general.repo_url": "https://huggingfac...
  [ec7630a] arch qwen4exp: src/llama-arch.cpp:43
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer qwen35: src/llama-vocab.cpp:2266
  [79e2e74] arch qwen4exp: src/llama-arch.cpp:44
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer qwen35: src/llama-vocab.cpp:2383
  [ik] arch qwen4exp: src/llama-arch.cpp:38
  [ik] tokenizer model gpt2: src/llama-vocab.cpp:1816
  [ik] pre-tokenizer qwen35: src/llama-vocab.cpp:2054

```

Every file:line above was printed back and checked (`sed -n`): `src/llama-arch.cpp:156@ec7630a` / `:158@79e2e74`
`{ LLM_ARCH_GLM5_NEXT, "glm5-next" }`, `:43` / `:44` `qwen4exp`, `src/llama-vocab.cpp:2290` / `:2407` `tokenizer_pre ==
"glm4"`, `:2266` / `:2383` `"qwen35"`, ik `src/llama-arch.cpp:91` `{ LLM_ARCH_GLM5NEXT_DASHED, "glm5-next" }`. Neither
llama.cpp pin knows `glm5next` (test `test_architecture_and_pre_tokenizer_known_at_the_pins`). Loadability (PROVISIONAL,
by reading; nothing loaded): every key `load_arch_hparams` of glm5-next / qwen4exp reads without a default is
present, and per layer the real files have exactly the tensor names of the tinygen models that load at both pins, except
`blk.45.nextn.shared_head_norm.weight`, which both pins create as optional (`glm5-next.cpp:183@ec7630a`,
`:181@79e2e74`, `TENSOR_NOT_REQUIRED`). The generic `llama_model_base::load_hparams` keys (`context_length`,
`embedding_length`, `block_count`, `attention.head_count`, …) are in the KV tables of appendix A.

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n15-committed-facts.py keys ec7630a=<S>/llama-ec7630a 79e2e74=<S>/llama.cpp
ec7630a glm5-next: 16 keys read without a default, missing in the real file: []
ec7630a qwen4exp: 21 keys read without a default, missing in the real file: []
79e2e74 glm5-next: 16 keys read without a default, missing in the real file: []
79e2e74 qwen4exp: 21 keys read without a default, missing in the real file: []
```

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 docs/evidence/mutation-scripts/2026-10-11-n15-committed-facts.py tensors
== glm-5.3-flash-ud-q4_k_xl vs tiny glm5next --mtp: non-layer tensors real-only [], tiny-only []
   real layers [0, 1, 2] (3): same tensor names as a tiny layer
   real layers [4, 5, 6, 8, 9, 10]... (31): same tensor names as a tiny layer
   real layers [3, 7, 11, 15, 19, 23]... (11): same tensor names as a tiny layer
   real layers ['mtp'] (1): no identical tiny layer; real-only ['nextn.shared_head_norm.weight'], tiny-only []
== qwen3.8-flash-next-ud-q4_k_xl vs tiny qwen4exp --mtp: non-layer tensors real-only [], tiny-only []
   real layers [0, 2, 4, 5, 6, 8]... (35): same tensor names as a tiny layer
   real layers [1] (1): same tensor names as a tiny layer
   real layers [3, 7, 11, 15, 19, 23]... (12): same tensor names as a tiny layer
```

### 7.5 (c) The embedded chat templates

Byte comparison and diffs: appendix A, "(c)" of each model (GLM: first difference at byte 3039, Qwen: at byte 1971; neither is equal up to trailing newlines). What the differences do, by running them (`docs/evidence/mutation-scripts/2026-10-11-n15-embedded-
templates.py`: N14's `section_render` / `section_prefix` of `tools/chattmpl/run_n14.py`, unchanged, on the vendored and
on the embedded templates with llama-server @79e2e74 and N14's tiny byte-vocabulary models, sha256 `3fb41bfc…64f3d` /
`135bf3f8…3946c` as in N14; plus jinja2(vendored) vs jinja2(embedded)); summary lines (appendix B has all):

```
$ cd <S>/n15/emb && python3 /home/user/wt-n15/docs/evidence/mutation-scripts/2026-10-11-n15-embedded-templates.py --server-bin <S>/llama.cpp/build-cpu/bin/llama-server --glm-gguf <S>/n14/final/models/glm5next_bytes.gguf --qwen-gguf <S>/n14/final/models/qwen4exp_bytes.gguf --out <S>/n15/emb/out > summary.txt
(selected lines of summary.txt, appendix B; indented lines prefixed here with the section they belong to)
glm/vendored render (server vs jinja2): 51 of 72 equal
glm/embedded render (server vs jinja2): 51 of 72 equal
[glm/embedded] default/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
glm/jinja2_vendored_vs_embedded: 68 of 68 renders byte-equal
qwen/vendored render (server vs jinja2): 62 of 72 equal
[qwen/vendored] prefix default/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
[qwen/vendored] switch effort_high: {"error": "RuntimeError: HTTP 500 {\"error\": {\"code\": 500, \"message\": \"\\n------------\\nWhile executing CallExpression at line 49, column 28 in source:\\n...', 'low') %}↵ {{- raise_exception('U
qwen/embedded render (server vs jinja2): 70 of 72 equal
[qwen/embedded] prefix default/server: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
[qwen/embedded] switch effort_high: {"shared_bytes": 5093, "prompt_bytes": 5093, "at": "<|im_start|>assistant\n<think>\n‖"}
qwen/jinja2_vendored_vs_embedded: 60 of 68 renders byte-equal
```

- GLM: the embedded template is a syntax-level rewrite on N14's inputs (68 of 68 jinja2 renders identical); its
  message loop still sets `reasoning_content` conditionally inside `{% for m in messages %}` (lines 143–150 in both), and
  llama-server renders the previous turn's reasoning into a turn without one (`<|assistant|><think>R1</think>a2`, the
  same 21 rows as with the vendored template) — CONFIRMED live. What the 4 removed `{%- break -%}` and the `else`
  branch (`{{- content }}` also for a non-string content) change outside N14's inputs: UNKNOWN.
- Qwen: `reasoning_effort: "high"` renders as `xhigh` (the switch to `high` now shares all 5093 bytes with the default
  prompt) instead of an exception / HTTP 500; the system-only conversation (`conv[:1]`) is no longer refused (N14:
  "No user query found in messages."); otherwise the same prefix behaviour.
- R1's llama-server runs with `--jinja` and no `--chat-template-file` (`scripts/r1/configs/llamacpp-server.json`), i.e.
  with these embedded templates.

### 7.6 (b) Hyperparameters and the graph branches only the real geometry takes

All arch-prefixed KV next to the four / three tinygen variants and the `config.json` cross-check: appendix A, "(b)".
The probes that the tiny models do not exercise (analyzer, literal; anchors located in each checkout at run time):

```
    [NOT IN TINY] glm.shexp_clamp: shared-expert SwiGLU clamp: own key or the routed one
       real: ['own shexp clamp key']  swiglu_clamp_shexp = [10.0]
       tiny: default: ['shexp clamp = routed clamp']; --mtp: ['shexp clamp = routed clamp']; --quant-dims: ['shexp clamp = routed clamp']; --dialect ik: ['shexp clamp = routed clamp']
       [ec7630a] src/models/glm5-next.cpp:35; src/llama-graph.cpp:1853
       [79e2e74] src/models/glm5-next.cpp:35; src/llama-graph.cpp:1888
       [ik] src/models/glm5-next.cpp: file absent; src/llama-graph.cpp: file absent
    [NOT IN TINY] glm.layer_norm_eps: LayerNorm eps of the indexer key norm (key absent: 0)
       real: ['eps 9.999999974752427e-07']  
       tiny: default: ['eps 0 (key absent)']; --mtp: ['eps 0 (key absent)']; --quant-dims: ['eps 0 (key absent)']; --dialect ik: ['eps 0 (key absent)']
       [ec7630a] src/models/glm5-next.cpp:9; src/llama-graph.cpp:1612
       [79e2e74] src/models/glm5-next.cpp:9; src/llama-graph.cpp:1647
       [ik] src/models/glm5-next.cpp: file absent; src/llama-graph.cpp: file absent
    [NOT IN TINY] glm.mtp_heads: MTP block's own embedding / head / head norm instead of the trunk's
       real: ['own nextn.shared_head_norm']  
       tiny: default: ['trunk embedding, head and norm']; --mtp: ['trunk embedding, head and norm']; --quant-dims: ['trunk embedding, head and norm']; --dialect ik: ['trunk embedding, head and norm']
       [ec7630a] not found; not found; not found
       [79e2e74] src/models/glm5-next.cpp:584; src/models/glm5-next.cpp:647; src/models/glm5-next.cpp:658
       [ik] src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent
    [NOT IN TINY] qwen.recurrent_source: linear/full layer pattern from the array or the interval rule
       real: ['full_attention_interval rule']  full_attention_interval = 4
       tiny: default: ['recurrent_layers array']; --mtp: ['recurrent_layers array']; --quant-dims: ['recurrent_layers array']
       [ec7630a] src/models/qwen4exp.cpp:149
       [79e2e74] src/models/qwen4exp.cpp:149
       [ik] src/models/qwen4exp.cpp: file absent
    [NOT IN TINY] qwen.ple_image_token: PLE image token id: own key or the EOS fallback
       real: ['image token id key']  
       tiny: default: ['EOS fallback']; --mtp: ['EOS fallback']; --quant-dims: ['EOS fallback']
       [ec7630a] src/models/qwen4exp.cpp:111
       [79e2e74] src/models/qwen4exp.cpp:111
       [ik] src/models/qwen4exp.cpp: file absent
```

Reading of each (labels):
- `glm.layer_norm_eps`: the indexer's key LayerNorm (`build_norm(…, LLM_NORM)` in `build_kpool_select`) uses
  `f_norm_eps` = 1e-6 from `glm5-next.attention.layer_norm_epsilon` (CONFIRMED present); no tiny GLM model has the key,
  so every llama.cpp parity run so far had eps 0 (N12's ik runs used ik's RMS-eps fallback). ik @5194a9e reads the same
  key (`src/llama-hparams.cpp:2465`) and falls back to the RMS eps only when it is absent, so N12's difference (2) does
  not arise on this file (CONFIRMED by reading). Numerical effect of 1e-6 vs 0 on the real model: UNKNOWN; expected to be
  small for 128-wide keys (PROVISIONAL, not computed). A tinygen option writing the key would close it.
- `glm.mtp_heads`: the real MTP block has `nextn.shared_head_norm` (no own embedding or head), so @79e2e74 the MTP graph
  normalises with it instead of `output_norm` (`glm5-next.cpp:647`); N9's MTP parity used the trunk norm. ec7630a has no
  MTP graph.
- `glm.shexp_clamp`: the real file has its own `swiglu_clamp_shexp` (10 for all 46 layers) where the tiny models fall
  back to the routed clamp; same value as the routed clamp, so the graph computes the same as the fallback (load path
  only, `glm5-next.cpp:35`).
- `qwen.recurrent_source`: no `attention.recurrent_layers`; the pattern comes from `full_attention_interval` = 4
  (`qwen4exp.cpp:149`, else-branch: full attention at layers 3, 7, …, 47), the same pattern the tiny models write as an
  array (PROVISIONAL by reading).
- `qwen.ple_image_token`: `ple.image_token_id` = 248056 is used only for image (embedding) batches
  (`qwen4exp.cpp:1225-1231@79e2e74`); text-only R1 never takes it.

Everything else the probes check is exercised by the tiny models: bounded KDA gate (-5.0), DSA on every 4th layer with
a full indexer each (no `attention.indexer.types`), k-pool tail (key absent, default true:
`src/llama-hparams.h:286@ec7630a` / `:289@79e2e74`), nope MLA (rope 0), hc 4, 3 dense layers then MoE, sigmoid gating,
renormalised and ×2.5-scaled routing without groups, fused SwiGLU clamp 10, scalar expert widths, separate KDA q/k/v,
3-D conv weights, one MTP block; for Qwen QSA on all 12 full-attention layers (no dense one), PLE on layer 1, explicit
expert widths 640, IMRoPE sections [11, 11, 10, 0] (subset of the tiny [1, 1, 1, 1]), partial rotary 64 of 256, no RoPE
scaling, separate gate/up experts, own output head. Sizes the tiny models do not have (GLM 64 heads, 288 experts top-8,
widths 4096 / 2048 / 12288, top-k 2048, 1 M context; Qwen 512 experts top-10, low rank 320, 16 PLE heads of ~20 M rows)
change no branch but are never run at that scale.

### 7.7 (a) + (f) Quant audit, ADR-012 D5

```
=== glm-5.3-flash-ud-q4_k_xl (third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl)
(a) tools/gguf_quant_audit.py on the header-only shards (files total 199707321472 B = 185.992 GiB)
  # GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf (6 file(s)), architecture 'glm5-next'
  total 320.76B params, 185.98 GiB, 4.981 bpw
    routed_experts   311.654B    176.82 GiB   4.874 bpw  [Q4_K 65.1%, Q5_K 32.6%, Q6_K 2.3%]
    shared_experts     1.082B      1.07 GiB   8.500 bpw  [Q8_0 100.0%]
    attention          6.182B      6.13 GiB   8.520 bpw  [Q8_0 99.9%, F32 0.1%]
    dense_ffn          0.453B      0.45 GiB   8.500 bpw  [Q8_0 100.0%]
    embedding          0.634B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    output_head        0.634B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    other              0.120B      0.26 GiB  18.511 bpw  [Q8_0 57.4%, F32 42.6%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: blk.3.ffn_gate_exps.weight Q4_K 4.500, blk.3.ffn_up_exps.weight Q4_K 4.500, blk.4.ffn_gate_exps.weight Q4_K 4.500, blk.4.ffn_up_exps.weight Q4_K 4.500, blk.5.ffn_gate_exps.weight Q4_K 4.500, blk.5.ffn_up_exps.weight Q4_K 4.500
=== qwen3.8-flash-next-ud-q4_k_xl (third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl)
(a) tools/gguf_quant_audit.py on the header-only shards (files total 111334654784 B = 103.688 GiB)
  # Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf (4 file(s)), architecture 'qwen4exp'
  total 176.94B params, 103.68 GiB, 5.033 bpw
    routed_experts   120.796B     71.73 GiB   5.101 bpw  [Q4_K 65.3%, Q5_1 29.9%, Q8_0 3.5%, Q5_K 1.4%]
    shared_experts     0.236B      0.23 GiB   8.512 bpw  [Q8_0 99.9%, F32 0.1%]
    attention          2.704B      2.72 GiB   8.644 bpw  [Q8_0 98.9%, BF16 0.7%, F32 0.4%]
    embedding          0.636B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    output_head        0.636B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    other             51.937B     27.74 GiB   4.587 bpw  [IQ4_NL 98.6%, Q8_0 1.3%, F32 0.1%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: per_layer_token_embd.weight IQ4_NL 4.500, blk.0.ffn_gate_exps.weight Q4_K 4.500, blk.0.ffn_up_exps.weight Q4_K 4.500, blk.1.ffn_gate_exps.weight Q4_K 4.500, blk.1.ffn_up_exps.weight Q4_K 4.500, blk.3.ffn_gate_exps.weight Q4_K 4.500
```

Both PASS the D5 floor (routed experts and every non-expert matrix ≥ 4.0 bpw); the lowest matrices are Q4_K (4.5 bpw)
and, for Qwen, the IQ4_NL PLE table (4.5 bpw). Per projection, counted over the layers (43 MoE blocks in GLM including
the MTP block, 48 in Qwen):

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n15-committed-facts.py experts
glm-5.3-flash-ud-q4_k_xl {'ffn_down_exps': {'Q5_K': 40, 'Q6_K': 3}, 'ffn_gate_exps': {'Q4_K': 42, 'Q5_K': 1}, 'ffn_up_exps': {'Q4_K': 42, 'Q5_K': 1}}
qwen3.8-flash-next-ud-q4_k_xl {'ffn_down_exps': {'Q5_1': 43, 'Q8_0': 5}, 'ffn_gate_exps': {'Q4_K': 47, 'Q5_K': 1}, 'ffn_up_exps': {'Q4_K': 47, 'Q5_K': 1}}
```

Qwen's `ffn_down_exps` input width is 640, not a multiple of the 256-element k-quant block, hence Q5_1 / Q8_0
(32-element blocks; PROVISIONAL reason, from llama-quantize's fallback rule `tensor_type_fallback`, not traced for this
file).

### 7.8 Discovery (LISTING.json)

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n15-committed-facts.py listing
fetched 2026-10-11T11:36:06Z request data/gguf-headers/discover-glm-mimo/REQUEST.json
search 'GLM-5.3-Flash': 95 repos, expanded True, matches 80, with file listing 25; architectures {'None': 3, 'clip': 1, 'controlvector': 1, 'dflash': 6, 'glm5-next': 34, 'glm5-next-vision': 1, 'glm53flash': 1, 'glm5next': 46, 'llama': 2}
search 'MiMo-V2.6-Flash': 30 repos, expanded True, matches 28, with file listing 0; architectures {'None': 2, 'mimo2': 28}
unsloth/GLM-5.3-Flash-GGUF @a38483c8cd5df544f53d70fb281afe97369d5ab6 arch glm5-next lastModified 2026-10-05T22:23:39.000Z params 320759404382 ctx 1048576 chat_template sha256 a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724
   BF16: 14 file(s), 641641064288 B = 597.57 GiB
   Q8_0: 8 file(s), 340981966208 B = 317.56 GiB
   UD-IQ1_M: 3 file(s), 97579344672 B = 90.88 GiB
   UD-IQ1_S: 3 file(s), 93087245088 B = 86.69 GiB
   UD-IQ2_XXS: 4 file(s), 101844951904 B = 94.85 GiB
   UD-IQ3_XXS: 4 file(s), 120367571840 B = 112.10 GiB
   UD-IQ4_XS: 5 file(s), 156822111200 B = 146.05 GiB
   UD-Q2_K_XL: 4 file(s), 108720071552 B = 101.25 GiB
   UD-Q3_K_XL: 4 file(s), 147535922080 B = 137.40 GiB
   UD-Q4_K_XL: 6 file(s), 199707321472 B = 185.99 GiB
   UD-Q5_K_XL: 6 file(s), 240306087008 B = 223.80 GiB
   UD-Q6_K_XL: 7 file(s), 291833111808 B = 271.79 GiB
   mmproj-BF16.gguf: 1 file(s), 1164010176 B = 1.08 GiB
   mmproj-F16.gguf: 1 file(s), 1128047296 B = 1.05 GiB
ggml-org/GLM-5.3-Flash-GGUF @c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa arch glm5-next lastModified 2026-10-08T10:56:49.000Z params 313326811966 ctx 1048576 chat_template sha256 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
   GLM-5.3-Flash-Q2_K-00001-of-00002.gguf: 1 file(s), 9430240 B = 0.01 GiB
   GLM-5.3-Flash-Q2_K-00002-of-00002.gguf: 1 file(s), 133799626432 B = 124.61 GiB
   GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf: 1 file(s), 9430240 B = 0.01 GiB
   GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf: 1 file(s), 110017922752 B = 102.46 GiB
   GLM-5.3-Flash-Q4_K-00001-of-00002.gguf: 1 file(s), 9429952 B = 0.01 GiB
   GLM-5.3-Flash-Q4_K-00002-of-00002.gguf: 1 file(s), 181363033792 B = 168.91 GiB
   mmproj-GLM-5.3-Flash-Q8_0.gguf: 1 file(s), 652829376 B = 0.61 GiB
   mtp-GLM-5.3-Flash-Q4_0.gguf: 1 file(s), 4947287104 B = 4.61 GiB
   mtp-GLM-5.3-Flash-Q8_0.gguf: 1 file(s), 9284197440 B = 8.65 GiB
ggml-org/MiMo-V2.6-Flash-RL-GGUF @b0b2d0fb0b56611c761e075425c88989faf1ed1f arch mimo2 lastModified 2026-10-11T08:23:14.000Z params 308778780864 ctx 1048576 chat_template sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059
ggml-org/MiMo-V2.6-Flash-MOPD-GGUF @e6d4bced739310a721eefad4c247ec7e0247c89d arch mimo2 lastModified 2026-10-11T07:58:04.000Z params 308778780864 ctx 1048576 chat_template sha256 11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059
```

The full table (95 + 30 repos, with `*` for an architecture match) is the first block of appendix A. Only the first 25
matching repos got a file listing (all from the GLM search), so no MiMo file listing was fetched.

### 7.9 Stretch: token-level prefix stability with the real vocabularies

`docs/evidence/mutation-scripts/2026-10-11-n15-token-prefix.py`: vocab-only GGUFs from the committed first shards
(no rename: both architectures are known at 79e2e74), llama-tokenize @79e2e74 (built in `<S>/n15/build-tok`, section 6)
with the server's flags (add_special and parse_special true, as `handle_completions_impl`,
`tools/server/server-context.cpp:4808@79e2e74`), N14's four agent sessions in the loop view, prompts from llama-server
with the **embedded** template, turns from jinja2:

```
$ cd <S>/n15/tok && python3 /home/user/wt-n15/docs/evidence/mutation-scripts/2026-10-11-n15-token-prefix.py --tokenize-bin <S>/n15/build-tok/bin/llama-tokenize --server-bin <S>/llama.cpp/build-cpu/bin/llama-server --glm-gguf <S>/n14/final/models/glm5next_bytes.gguf --qwen-gguf <S>/n14/final/models/qwen4exp_bytes.gguf --out <S>/n15/tok/out > summary.txt; echo "exit $?"; cat summary.txt
exit 0
glm: vocab 154880, vocab-only GGUF sha256 d66c39806a657d08, embedded template sha256 a4fddbbf0b432101
  tags: [gMASK] -> [154822] (type [3]); <sop> -> [154824] (type [3]); <|system|> -> [154826] (type [3]); <|user|> -> [154827] (type [3]); <|assistant|> -> [154828] (type [3]); <|observation|> -> [154829] (type [3]); <think> -> [154841] (type [4]); </think> -> [154842] (type [4]); <tool_call> -> [154843] (type [4]); </tool_call> -> [154844] (type [4]); <arg_key> -> [154847] (type [4]); </arg_key> -> [154848] (type [4]); <arg_value> -> [154849] (type [4]); </arg_value> -> [154850] (type [4]); <tool_response> -> [154845] (type [4]); </tool_response> -> [154846] (type [4])
  default: 6 request pairs, 6 measured; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  preserve_off: 6 request pairs, 6 measured; byte-extends 5, token-extends 5, prompt-token-extends 5, concat canonical 6
    request 4: bytes False (reuse 1936 of 3704), tokens False (reuse 494 of 797) at {"held": [".", "<|assistant|>", "<think>", "I", "Ġshould", "Ġlook"], "next": [".", "<|assistant|>", "<think>", "</think>", "<tool_call>", "list"], "index": 494}
  enable_thinking_false: 6 request pairs, 6 measured; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  empty_reasoning_turn: 6 request pairs, 6 measured; byte-extends 5, token-extends 5, prompt-token-extends 6, concat canonical 6
    request 3: bytes False (reuse 3320 of 3517), tokens False (reuse 732 of 758) at {"held": ["</tool_response>", "<|assistant|>", "<think>", "</think>", "<tool_call>", "run"], "next": ["</tool_response>", "<|assistant|>", "<think>", "parse", "()", "Ġsplits"], "index": 732}
qwen: vocab 248320, vocab-only GGUF sha256 c092638a93b4c6e4, embedded template sha256 12827f24b742ea4e
  tags: <|im_start|> -> [248045] (type [3]); <|im_end|> -> [248046] (type [3]); <think> -> [248068] (type [4]); </think> -> [248069] (type [4]); <tool_call> -> [248058] (type [4]); </tool_call> -> [248059] (type [4]); <tool_response> -> [248066] (type [4]); </tool_response> -> [248067] (type [4]); <function= -> [27, 1628, 28] (type [1, 1, 1]); </function> -> [510, 1628, 29] (type [1, 1, 1]); <parameter= -> [27, 15704, 28] (type [1, 1, 1]); </parameter> -> [510, 15704, 29] (type [1, 1, 1])
  default: 6 request pairs, 6 measured; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  preserve_off: 6 request pairs, 6 measured; byte-extends 5, token-extends 5, prompt-token-extends 5, concat canonical 6
    request 4: bytes False (reuse 2756 of 4651), tokens False (reuse 683 of 1170) at {"held": ["<|im_start|>", "assistant", "Ċ", "<think>", "Ċ", "I"], "next": ["<|im_start|>", "assistant", "Ċ", "<tool_call>", "Ċ", "<"], "index": 683}
  enable_thinking_false: 6 request pairs, 6 measured; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  empty_reasoning_turn: 6 request pairs, 6 measured; byte-extends 6, token-extends 5, prompt-token-extends 5, concat canonical 5
    request 3: bytes True (reuse 4425 of 4425), tokens False (reuse 1058 of 1117) at {"held": ["assistant", "Ċ", "<think>", "Ċ", "Ċ", "</think>"], "next": ["assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"], "index": 1058} canonical!=concat {"index": 1058, "canonical": ["Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"], "concat": ["Ċ", "<think>", "Ċ", "Ċ", "</think>"]}
```

What it shows (CONFIRMED for llama.cpp's tokenizer function on these vocabularies; the server's own path, its cache
and real sampling are NOT exercised): N14's byte-level prefix claims hold at token level for GLM in all 24 request pairs
(the two byte breaks of N14 break at the same place in tokens); for Qwen in 23 of 24 — after an empty reasoning turn
the held tokens end `<think>`, `Ċ`, `Ċ`, `</think>` while the next prompt tokenizes `<think>`, `ĊĊ`, `</think>`, so even
the previous prompt alone is no longer a token prefix (reuse 1058 of 1117 held tokens). Whether llama-server then
re-processes only those 59 tokens or back to an earlier checkpoint (recurrent layers) is UNKNOWN (N14 found checkpoints
only at user-message starts). The tags of both templates are single tokens except Qwen's `<function=` / `<parameter=`
(3 ordinary tokens each), which closes N14 section 8's question for these vocabularies.

### 7.10 Consequences (for the lead)

- AGENTS.md's "GGUF architecture string `glm5next`" for the host files may be outdated for the HF repo: the current
  UD-Q4_K_XL is `glm5-next` and, by its header, loadable by llama.cpp mainline at both pins (PROVISIONAL, 7.4). Which
  revision the host holds: UNKNOWN (commands in 7.3).
- The Qwen UD-Q4_K_XL has no MTP block (CONFIRMED): native-MTP speculation on it needs another source.
- R1's llama.cpp runs use the embedded templates: GLM's reasoning leak (N14 1(a)) applies; Qwen's `high` works.
- Tiny-model gaps worth closing in tinygen (PROVISIONAL, by the probes): `attention.layer_norm_epsilon`,
  `nextn.shared_head_norm`, a qwen4exp file without `attention.recurrent_layers`.

## 8. NOT RUN

- Loading either real file in llama.cpp, ik_llama.cpp or Strata (no weights here): loadability is PROVISIONAL (7.4).
- The host check of 7.3 (`head -c … | sha256sum`, `sha256sum` of the tensor shards on epyc-ai).
- Headers of the other GLM / MiMo GGUFs of the listing (e.g. `ggml-org/GLM-5.3-Flash-GGUF` Q4_K, the MiMo MOPD files):
  not requested; a new `data/gguf-headers/<name>/REQUEST.json` would fetch them the same way.
- The token-level check through llama-server's own tokenization and cache, and with a real model's sampled tokens
  (7.9 is the tokenizer function only).
- The behaviour of the GLM embedded template's removed `{% break %}` / changed `else` outside N14's inputs (7.5).
- The workflow's step logs (self-test `Ran N tests`): not readable from the sandbox (7.1).
- `analyze.py --src ik=`: ik's source layout differs (`src/models/glm5-next.cpp`, `src/llama-graph.cpp` print "file
  absent" at ik), so for ik only the architecture / tokenizer support lines are meaningful.

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
- The fetcher @`4eae257` recorded `size_check: "mismatch: header implies 9429956 B"` (GLM) and `"… 10946618 B"`
  (Qwen) for the metadata-only first shards, with a warning annotation each; the files are consistent (writer padding
  after the KV block, 7.2). Fixed in `9c5384c`; the committed PIN.json files keep what the runner recorded.
- Round 1 section 2 expected "about 15–25 MB per model" fetched and "about 2–4 MB" committed: measured 10.7 / 11.7 MB
  fetched and 2.8 / 3.3 MB committed (7.2).

## 11. Round 2: tests and mutation check

Code changed in round 2: `9c5384c` (size check, analyzer config map), `1d57273` (tests over the committed headers),
`7190e17` / `93ee70d` (evidence scripts, M19). All runs below at `93ee70d`.

```
$ cd /home/user/wt-n15 && git log --oneline -n 1 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > <S>/n15/full-suite-93ee70d.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n15/full-suite-93ee70d.log
93ee70d [model-analyst] evidence scripts: N15 committed-facts printer; mutation M19 for the metadata-only shard size check
exit 0
Ran 900 tests in 152.124s
OK (skipped=358)
$ python3 -m unittest tests.test_hooks tests.test_check_unittest 2>&1 | tail -3
Ran 54 tests in 14.151s

OK
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py LLAMA_CPP_SRC_79E2E74=<S>/llama.cpp LLAMA_TOKENIZE_BIN=<S>/n15/build-tok/bin/llama-tokenize python3 -m unittest tests.test_gguf_headers tests.test_gguf_headers_committed -v > <S>/n15/modules-93ee70d.log 2>&1; echo "exit $?"; grep -E "^(Ran|OK|FAILED)" <S>/n15/modules-93ee70d.log; grep -c " \.\.\. ok$" <S>/n15/modules-93ee70d.log; grep -c "skipped" <S>/n15/modules-93ee70d.log
exit 0
Ran 40 tests in 18.723s
OK
40
0
$ python3 -m unittest tests.test_gguf_headers_committed 2>&1 | tail -3
Ran 8 tests in 2.801s

OK (skipped=2)
$ python3 -c "import yaml; yaml.safe_load(open('.github/workflows/gguf-headers.yml')); print('YAML OK .github/workflows/gguf-headers.yml')"
YAML OK .github/workflows/gguf-headers.yml
```

900 = 891 of round 1 + `test_size_check_of_tensor_and_metadata_only_shards` + the 8 tests of
`tests/test_gguf_headers_committed.py`; the skip count is unchanged (358: the llama-tokenize test without
`LLAMA_TOKENIZE_BIN`; the committed-header tests find gguf-py and the ec7630a checkout through `LLAMA_CPP_GGUF_PY`).
Without any environment the committed-header module skips exactly its two optional tests (branches: gguf-py + numpy;
pin support: the checkouts). The workflow file is unchanged since round 1.

Mutation check (`docs/evidence/mutation-scripts/2026-10-11-n15-mutations.sh`, now with M19):

```
$ cd <S>/n15 && git -C /home/user/wt-n15 log --oneline -n 1 && bash /home/user/wt-n15/docs/evidence/mutation-scripts/2026-10-11-n15-mutations.sh > mutations-93ee70d.log 2>&1; echo "exit $?"; cut -c1-400 mutations-93ee70d.log
93ee70d [model-analyst] evidence scripts: N15 committed-facts printer; mutation M19 for the metadata-only shard size check
exit 0
HEAD 93ee70d
base exit 0 (Ran 32 tests in 16.224s, OK (skipped=1))
M1-no-data-padding KILLED  (Fetch.test_fetch_stops_at_the_header_and_pins_the_revision Parser.test_alignment_rules_of_gguf_cpp Parser.test_needmore_reports_the_exact_next_need Parser.test_size_check_of_tensor_and_metadata_only_shards Parser.test_summary_digests_big_arrays_and_keeps_the_rest Tinygen.test_offsets_agree_with_gguf_py )
M2-pad-without-tensors KILLED  (Parser.test_alignment_rules_of_gguf_cpp Parser.test_rebuild_copies_values_byte_for_byte Parser.test_size_check_of_tensor_and_metadata_only_shards VocabOnly.test_vocab_only_file_keeps_the_tokenizer_bytes_and_renames_the_arch )
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
M19-no-metadata-shard-padding KILLED  (Parser.test_size_check_of_tensor_and_metadata_only_shards )
```

**19 of 19 killed** (an intermediate run at `7190e17`, before M19, gave 18 of 18 and is not pasted).

## Appendix A. `tools/gguf_headers/analyze.py` on the committed headers, full output

```
$ cd <S>/n15/r2 && LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 /home/user/wt-n15/tools/gguf_headers/analyze.py --src ec7630a=<S>/llama-ec7630a --src 79e2e74=<S>/llama.cpp --src ik=<S>/n12/ik-src --json analyze.json > analyze.txt; echo "exit $?"
exit 0
sources: ec7630a = ec7630a640789c393694fb194f1bbbf0369fc62d (<S>/llama-ec7630a), 79e2e74 = 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (<S>/llama.cpp), ik = 5194a9e1451b885e24dd5a065f650864e8807c59 (<S>/n12/ik-src)

=== discover-glm-mimo (third_party/gguf-headers/discover-glm-mimo)
search 'GLM-5.3-Flash' (architectures ['glm5-next', 'glm5next']): 95 repos
  * huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5-next  downloads=27094  sha=c43a994637c1  47 .gguf, 1451.9 GiB
    peasantsmith/GLM-5.3-Flash-Maya-GGUF  arch=llama  downloads=57366  sha=f16ee4ba50d8
  * unsloth/GLM-5.3-Flash-GGUF  arch=glm5-next  downloads=954431  sha=a38483c8cd5d  70 .gguf, 2368.1 GiB
  * autotrust/GLM-5.3-Flash-GGUF-DGX-Spark  arch=glm5-next  downloads=35088  sha=e1599a6bc0ea  2 .gguf, 79.1 GiB
  * orcarouter/GLM-5.3-Flash-Uncensored-GGUF  arch=glm5next  downloads=17786  sha=efa699effe7e  27 .gguf, 994.7 GiB
  * BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF  arch=glm5-next  downloads=482874  sha=8a31ca664853  13 .gguf, 429.9 GiB
  * pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF  arch=glm5-next  downloads=32438  sha=892aabe2e458  3 .gguf, 238.2 GiB
  * marcorez8/GLM-5.3-Flash-Abliterated-Heretic-V2-UD-IQ4_XS  arch=glm5next  downloads=1604  sha=fa7f285225cb  8 .gguf, 148.2 GiB
    Baekpica/GLM-5.3-Flash-Uncensored-Mixed-Quant-GGUF  arch=glm5-next-vision  downloads=816  sha=62955f8e8b18
  * neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF  arch=glm5-next  downloads=8067  sha=7feedfeb5ee9  1 .gguf, 105.8 GiB
  * ggml-org/GLM-5.3-Flash-GGUF  arch=glm5-next  downloads=2718  sha=c259a9f17fe2  9 .gguf, 409.9 GiB
  * aj9o9/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=17095  sha=07c62fcdeaf1  5 .gguf, 186.0 GiB
    Anbeeld/GLM-5.3-Flash-DFlash2-GGUF  arch=dflash  downloads=1766  sha=f3569acb7f9c
  * SyndicateLabs/GLM-5.3-Flash-CYBERSECURITY-GGUF  arch=glm5-next  downloads=8228  sha=e8b51a5b8a45  7 .gguf, 946.4 GiB
  * GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF  arch=glm5-next  downloads=8152  sha=6363ade6cb1a  2 .gguf, 438.0 GiB
  * avar6/GLM-5.3-Flash-BF16-gguf  arch=glm5-next  downloads=2118  sha=7083b2085efa  17 .gguf, 591.9 GiB
  * AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF  arch=glm5next  downloads=31049  sha=b3ce393a304d  21 .gguf, 762.0 GiB
  * patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF  arch=glm5-next  downloads=12175  sha=46cb1bb88772  6 .gguf, 403.2 GiB
    Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ8e-DFlash2  arch=dflash  downloads=12386  sha=2532db3456d4
    Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2  arch=dflash  downloads=21082  sha=bffafea2ded6
  * GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF  arch=glm5next  downloads=523  sha=1d8439bff3ef  2 .gguf, 0.5 GiB
  * AliceThirty/GLM-5.3-Flash-UNCENSORED-V2-GGUF  arch=glm5next  downloads=634  sha=06a87980ce3f  5 .gguf, 186.2 GiB
  * neuralll/GLM-5.3-Flash-MTP-GGUF  arch=glm5-next  downloads=581  sha=c68e5f340cba  1 .gguf, 4.3 GiB
  * bartowski/GLM-5.3-Flash-BF16-GGUF  arch=glm5-next  downloads=11326  sha=66e5e9f0e337  121 .gguf, 3863.0 GiB
  * SixVolts/GLM-5.3-Flash-ewaste-edition-GGUF  arch=glm5-next  downloads=407  sha=f9dd17e251f5  17 .gguf, 544.6 GiB
  * peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF  arch=glm5-next  downloads=737  sha=b0a5474568ab  3 .gguf, 108.0 GiB
    alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF  arch=llama  downloads=678  sha=3518a2d20e5a
    AtomicChat/GLM-5.3-Flash-GGUF  arch=None  downloads=0  sha=dfa0fac07337
  * vcruz305/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=5014  sha=b19ca66573c0  3 .gguf, 289.4 GiB
  * DevQuasar/zai-org.GLM-5.3-Flash-GGUF  arch=glm5next  downloads=5840  sha=0be0114b087a  116 .gguf, 1355.6 GiB
  * Blackfrost-AI/GLM-5.3-Flash-DERISKED-GGUF  arch=glm5next  downloads=450  sha=b4e3bc7d7981  56 .gguf, 778.4 GiB
    msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44  arch=controlvector  downloads=40  sha=ef85b0169fd2
  * AMAImedia/GLM-5.3-Flash-FP8-GGUF  arch=glm5next  downloads=3678  sha=7764ceaa66e8  12 .gguf, 138.8 GiB
  * axiomofmind/GLM-5.3-Flash-W4A16-NVFP4-GGUF  arch=glm5next  downloads=222  sha=ad1eed18404a
  * 6block/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=548  sha=ecc6c5303921
  * antirez/glm-5.3-flash-gguf  arch=glm5-next  downloads=74444  sha=b2fa29d7a6b4
  * ddh0/GLM-5.3-Flash-GGUF  arch=glm5-next  downloads=938  sha=79b15e132622
  * Nil626yhh/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=162  sha=540c41d2cff0
    Serpen/GLM-5.3-Flash  arch=clip  downloads=9  sha=1f0c4ba27b16
    vcruz305/GLM-5.3-Flash-DFlash2-GGUF  arch=dflash  downloads=356  sha=caf6ef0cedd0
  * sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP  arch=glm5next  downloads=178  sha=6e6619ac46a7
  * meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers  arch=glm5next  downloads=18923  sha=d1b3be770eac
  * qtum/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=298  sha=3e90be1b205c
  * 0ppxnhximxr/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=233  sha=fac026d82ab1
  * lazywold/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=143  sha=48d746c3cf52
  * D9010/GLM-5.3-Flash-D9010-86GB  arch=glm5next  downloads=71  sha=856d9b46cc21
  * Hagwell/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=184  sha=c4379587064b
  * MorinoNushi/GLM-5.3-Flash-Heretic-LoRA-V1-GGUF  arch=glm5next  downloads=139  sha=4f0c44149130
  * CoboSan/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=158  sha=5ec23a1e4de9
  * ashuaria/GLM-5.3-Flash-GGUF-Finetuning  arch=glm5next  downloads=179  sha=8bd0713f091f
  * AesSedai/GLM-5.3-Flash-GGUF  arch=glm5-next  downloads=2119  sha=99fb1c6c5e87
  * batiai/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=918  sha=5d8c47def046
  * Sciguy429/GLM-5.3-Flash-BF16  arch=glm5-next  downloads=111  sha=8fc436035040
  * zurichquants/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=269  sha=c1d2cf279199
  * zurichquants/GLM-5.3-Flash-UD-Q4_K_XL-layers  arch=glm5next  downloads=1461  sha=c0bae1f4b727
  * lausannequants/GLM-5.3-Flash-GGUF  arch=glm5next  downloads=201  sha=7a065d040c3b
  * lausannequants/GLM-5.3-Flash-UD-Q4_K_XL-layers  arch=glm5next  downloads=636  sha=dae7911b928c
  * DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4  arch=glm5-next  downloads=15614  sha=235989b3a2f4
    petr567/GLM-5.3-Flash-AJ-IQ2-XXS-Strix-Halo-256K  arch=None  downloads=0  sha=07f9bcbbcfe8
  * BoldingBuilds/GLM-5.3-Flash-Uncensored-GGUF  arch=glm5-next  downloads=4308  sha=be99016c76c7
  * marcorez8/GLM-5.3-Flash-Abliterated-Heretic-V1-UD-IQ4_XS-F16  arch=glm5next  downloads=351  sha=eaeca1ca3d2a
  * backpack-run/GLM-5.3-Flash-GGUF  arch=glm5-next  downloads=489  sha=144a2c2723b8
    Solstice-AI/GLM-5.3-Flash-UNCENSORED-NVFP4-DFlash2  arch=dflash  downloads=8289  sha=9b9f2589f542
    Solstice-AI/GLM-5.3-Flash-UNCENSORED-GGUF  arch=dflash  downloads=1106  sha=f167280509ea
  * PaoAI/GLM-5.3-Flash-PaoAI-ROCmFP4-STRIX-BALANCED-GGUF  arch=glm5next  downloads=1083  sha=fe78bc85332b
  * PaoAI/GLM-5.3-Flash-PaoAI-ROCmFP4-STRIX-HYBRID-GGUF  arch=glm5next  downloads=249  sha=d1a213cf9f86
  * PaoAI/GLM-5.3-Flash-PaoAI-ROCmFP4-STRIX-MTP-GGUF  arch=glm5next  downloads=91  sha=dae7a7d01adb
  * M0STAFAz/GLM-5.3-Flash-0.1B-A0.1B-F16.gguf  arch=glm5next  downloads=129  sha=82f77811d306
  * MorinoNushi/GLM-5.3-Flash-Heretic-Abliterated-LoRA-V2-GGUF  arch=glm5next  downloads=836  sha=1ab13071490a
  * xautonomics/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF  arch=glm5next  downloads=3472  sha=f873b2463e71
  * elvezjp/GLM-5.3-Flash-ELVEZ-KDA-Q8_0-gguf  arch=glm5-next  downloads=235  sha=8a3748f4851c
  * Cypherfox/GLM-5.3-Flash-dealignai-orcarouter-merge  arch=glm5next  downloads=1825  sha=70253325de76
  * YTan2000/GLM-5.3-Flash-TQ3_4S  arch=glm5next  downloads=438  sha=4bce6a5fecbf
  * baerquants/GLM-5.3-Flash-UD-Q4_K_XL-layers  arch=glm5next  downloads=763  sha=b37667a46551
  * kyuz0/GLM-5.3-Flash-Mixed-Q2-Q4-GGUF  arch=glm5-next  downloads=2165  sha=6ec5f33b8ac7
  * onicubag/GLM-5.3-Flash-IQ6_K-GGUF  arch=glm5next  downloads=257  sha=eb013ee1238b
  * taurusduan/GLM-5.3-Flash-GGUF-DGX-Spark  arch=glm5-next  downloads=221  sha=5d637027494d
  * Oppsy/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5next  downloads=406  sha=84080f36fef9
  * taurusduan/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5next  downloads=665  sha=2ac4dcb4a2ec
  * Nbighir/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5next  downloads=344  sha=4edf69d36da4
  * bytehackr/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5next  downloads=342  sha=4d68b8b38431
  * Ihnenko/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5next  downloads=361  sha=acaba26472f9
  * Jerome0207/huihui-glm-5.3-flash-abliterated-q4-k-xl  arch=glm5next  downloads=92  sha=b8e8b5ab3dfe
  * axi0mX/GLM-5.3-Flash-Uncensored-GGUF  arch=glm5next  downloads=4294  sha=ea05e17ef542
  * Dennits/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5-next  downloads=349  sha=625420c5a882
    amd/GLM-5.3-Flash-EXTENSOR  arch=glm53flash  downloads=275  sha=20eb2529ccb0
  * cafonez/GLM-5.3-Flash-Gorgon-GGUF  arch=glm5-next  downloads=64  sha=7ce8c3507c2a
  * taurusduan/GLM-5.3-Flash-GSQ-RCO-GGUF  arch=glm5-next  downloads=1551  sha=fd0d96efa505
  * moixx/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5-next  downloads=378  sha=7dec0e5d7fdf
  * SaturnHeaven/GLM-5.3-Flash-RANMA-EXL3-GGUF  arch=glm5-next  downloads=1233  sha=89dcf818785c
    lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts  arch=None  downloads=491  sha=3eb74994c133
  * SaturnHeaven/GLM-5.3-Flash-EXL3-Spark-RANMA-EXL3-GGUF  arch=glm5-next  downloads=192  sha=20962f29dd93
  * mussonking/GLM-5.3-Flash-Phobos-GGUF  arch=glm5-next  downloads=49  sha=8180fc686744
  * muterornament/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5-next  downloads=893  sha=71d633797558
  * Dgvdhb/Huihui-GLM-5.3-Flash-abliterated-GGUF  arch=glm5-next  downloads=817  sha=d26512d71ef4
search 'MiMo-V2.6-Flash' (architectures ['mimo2']): 30 repos
  * ProCreations/MiMo-V2.6-Flash-RL-NVFP4  arch=mimo2  downloads=11177  sha=34cebc5d7574
  * ggml-org/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=15855  sha=b0b2d0fb0b56
  * MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-GGUF  arch=mimo2  downloads=344  sha=fbbb29584eb9
  * MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF  arch=mimo2  downloads=587  sha=0ecc6864d187
  * pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF  arch=mimo2  downloads=2853  sha=a37b69042dbe
  * patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF  arch=mimo2  downloads=6615  sha=8d0abe0abfbe
  * AesSedai/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=1669  sha=b6e95d58c343
  * ProCreations/MiMo-V2.6-Flash-RL-IQ3_XXS-GGUF  arch=mimo2  downloads=712  sha=9b2bf573c44e
  * TrevorJS/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=1442  sha=dce9d8c8e576
  * Baekpica/MiMo-V2.6-Flash-RL-Mixed-Quant-GGUF  arch=mimo2  downloads=3673  sha=7cbefe60ca47
  * Baekpica/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=840  sha=a485ac9ebf51
  * kernelpool/MiMo-V2.6-Flash-MXFP4-GGUF  arch=mimo2  downloads=571  sha=f4a1c0708d4a
  * jedisct1/MiMo-V2.6-Flash-RL-GGUF-100K  arch=mimo2  downloads=240  sha=a207209263a0
  * YanissAmz/MiMo-V2.6-Flash-RL-GGUF  arch=mimo2  downloads=1277  sha=e7a3bd785397
  * AliceThirty/MiMo-V2.6-Flash-RL-UNCENSORED-gguf  arch=mimo2  downloads=2028  sha=5cc3ec2959c0
  * 0xSojalSec/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF  arch=mimo2  downloads=153  sha=c47289f56ace
  * QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF  arch=mimo2  downloads=48815  sha=ac33b3743d70
  * ProCreations/MiMo-V2.6-Flash-MOPD-NVFP4  arch=mimo2  downloads=1204  sha=da654d8ed8d0
  * ddh0/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=1200  sha=b9e3e8a3370a
  * kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF  arch=mimo2  downloads=602  sha=44d9191bb7a9
  * ggml-org/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=5621  sha=e6d4bced7393
  * webmp3/Sakura-MiMo-V2.6-Flash-MOPD-P160-GGUF  arch=mimo2  downloads=691  sha=a0d8ac025575
  * fresherbz/MiMo-V2.6-Flash-MOPD-IQ2_XXS-GGUF  arch=mimo2  downloads=204  sha=b393fdeb4476
  * Baekpica/MiMo-V2.6-Flash-MOPD-Mixed-Quant-GGUF  arch=mimo2  downloads=554  sha=c2de39b1ac00
  * AMAImedia/MiMo-V2.6-Flash-MOPD-BPW2.5-GGUF  arch=mimo2  downloads=335  sha=a98d84c2f711
    CarnaK/MiMo-V2.6-Flash-RL-Heretic-MTP-Q3_K-GGUF  arch=None  downloads=0  sha=4d21819d3ede
  * cafonez/MiMo-V2.6-Flash-RL-UNCENSORED-Gorgon-GGUF  arch=mimo2  downloads=224  sha=998bdd418f7a
  * kernelpool/MiMo-V2.6-Flash-MOPD-GGUF  arch=mimo2  downloads=286  sha=3ce46a094a33
  * spiritfather/MiMo-V2.6-Flash-MOPD-Yamz-Uncensored-GGUF  arch=mimo2  downloads=1180  sha=f1adc2e15d07
    petr567/MiMo-V2.6-Flash-MOPD-Strix-Halo-Recipe  arch=None  downloads=0  sha=29500f0a8910

=== glm-5.3-flash-ud-q4_k_xl (third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl)
status ok, unsloth/GLM-5.3-Flash-GGUF @ a38483c8cd5df544f53d70fb281afe97369d5ab6
integrity: GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf OK, GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf OK, GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf OK, GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf OK, GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf OK, GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf OK
  UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf: 9429984 B, LFS sha256 b59c030090f012b38213bc8c4f78ce10767f2693d730d0dbcd3dc16a5f69e2bc, header 9429956 B (sha256 0c90bc17e2486d3ed599dfda7e2b244bdded27c48d0680569077d741c11b1016), fetched 9429984 B, committed 2487157 B gz, 0 tensors, size check exact (no tensors: the writer's padding after the KV block) (PIN.json recorded: mismatch: header implies 9429956 B), offsets contiguous
  UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf: 49198039200 B, LFS sha256 d3ecb6ff3957a99878f9a0352676a0913748a431a9a4a1b1ffa5ade9b8947a74, header 24608 B (sha256 a1e0c64d7bf1be6528c0945b06e58b9259b38ab710a3da89a93ce156022cdc1a), fetched 262144 B, committed 3679 B gz, 397 tensors, size check exact, offsets contiguous
  UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf: 49988626304 B, LFS sha256 328073c004e7c5395b208247feed26089db247b94dd8fa4ab0f2569d86c1d770, header 21376 B (sha256 34fc5c898088996d8e1ee0d76a9c10df8fa6dc7c30273612823427bf892fc317), fetched 262144 B, committed 3167 B gz, 337 tensors, size check exact, offsets contiguous
  UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf: 48700701344 B, LFS sha256 6803ab7effa6da4a0b02c9c7865952fdd6c5b2e10e53903348da369e90bd13f9, header 20256 B (sha256 7176860b6d2fdc38f36bd3cbfac4055c223201623370b97267f316c420a9fc5f), fetched 262144 B, committed 3031 B gz, 320 tensors, size check exact, offsets contiguous
  UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf: 49026026752 B, LFS sha256 852f3df9dacde3d196796f9fe738468e80c8701555ff0a8c30bc010fb8974dd8, header 21632 B (sha256 1b4a3a705d62bdf4fa36910ad01243a6843d351daa4deea4302001340a394ba1), fetched 262144 B, committed 3209 B gz, 341 tensors, size check exact, offsets contiguous
  UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf: 2784497888 B, LFS sha256 c6ba510fafc1e12cc0addbc361a329c0c592ab96980098e7e25e107d8faa983e, header 1248 B (sha256 dd1a523a0a9dcbad7cd6754fe69682cd94177f1a51f937721fc21f8ea57f019e), fetched 262144 B, committed 384 B gz, 17 tensors, size check exact, offsets contiguous

(e) identity
  architecture: glm5-next
  general.name: GLM 5.3 Flash
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: glm4
  vocab (tokenizer.ggml.tokens): 154880
  merges: 321649
  token_embd shape: [4096, 154880]
  output shape: [4096, 154880]
  tokenizer.ggml.bos_token_id: 154822
  tokenizer.ggml.eom_token_id: 154829
  tokenizer.ggml.eos_token_id: 154820
  tokenizer.ggml.eot_token_id: 154827
  tokenizer.ggml.padding_token_id: 154820
  tokenizer.ggml.unknown_token_id: 154820
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.file_type": 15, "general.languages": ["en", "zh"], "general.license": "mit", "general.quantization_version": 2, "general.quantized_by": "Unsloth", "general.repo_url": "https://huggingface...
  [ec7630a] arch glm5-next: src/llama-arch.cpp:156
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer glm4: src/llama-vocab.cpp:2290
  [79e2e74] arch glm5-next: src/llama-arch.cpp:158
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer glm4: src/llama-vocab.cpp:2407
  [ik] arch glm5-next: src/llama-arch.cpp:91
  [ik] tokenizer model gpt2: src/llama-vocab.cpp:1816
  [ik] pre-tokenizer glm4: src/llama-vocab.cpp:2074

(d) expert_group_count = 1, expert_group_used_count = 1 (keys present: [True, True]): no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)
  [ec7630a] src/llama-graph.cpp:2105
  [79e2e74] src/llama-graph.cpp:2145
  [ik] src/llama-graph.cpp: file absent

(c) chat template vs third_party/hf/glm-5.3-flash-chat/chat_template.jinja: DIFFERENT (embedded 10648 B sha256 a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724, vendored 10950 B sha256 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5, first difference at byte 3039, equal up to trailing newlines: False)
  | --- vendored
  | +++ embedded
  | @@ -70,4 +70,4 @@
  |          {%- endfor -%}
  | -    {%- elif content is not none -%}
  | -        {{- content -}}
  | +    {%- else -%}
  | +        {{- content }}
  |      {%- endif -%}
  | @@ -80,3 +80,3 @@
  |      {{- tool_response(m.content) -}}
  | -{%- elif m.content and m.content is not mapping and m.content.0.type == "tool_reference" -%}
  | +{%- elif m.content and m.content is not mapping and m.content[0].type == "tool_reference" -%}
  |      {{- tool_references_to_response(m.content) -}}
  | @@ -84,3 +84,3 @@
  |      {%- for tr in m.content -%}
  | -        {%- if tr.output is iterable and tr.output is not string and tr.output is not mapping and tr.output and tr.output.0.type == "tool_reference" -%}
  | +        {%- if tr.output is iterable and tr.output is not string and tr.output is not mapping and tr.output and tr.output[0].type == "tool_reference" -%}
  |              {{- tool_references_to_response(tr.output) -}}
  | @@ -102,3 +102,3 @@
  |  {%- macro is_list_of_outputs(m) -%}
  | -    {%- if m.content and m.content.0.output is defined -%}1{%- endif -%}
  | +    {%- if m.content and m.content[0].output is defined -%}1{%- endif -%}
  |  {%- endmacro -%}
  | @@ -161,3 +161,3 @@
  |  {%- endif %}
  | -{{- '<tool_call>' ~ tc.name -}}
  | +{{- '<tool_call>' + tc.name -}}
  |  {% set _args = tc.arguments %}{% for k, v in _args.items() %}<arg_key>{{ k }}</arg_key><arg_value>{{ v | tojson(ensure_ascii=False) if v is not string else v }}</arg_value>{% endfor %}</tool_call>{% endfor %}
  | @@ -185,3 +185,2 @@
  |          {%- for k in range(block_start, ns_blk.end + 1) -%}
  | -            {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
  |              {%- set m = messages[k] -%}
  | @@ -189,3 +188,2 @@
  |                  {%- for entry in m.content -%}
  | -                    {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
  |                      {%- set eid = id_of(entry) -%}
  | @@ -211,3 +209,2 @@
  |          {%- for i in range(ns_a.tool_calls | length) -%}
  | -            {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
  |              {%- set tc_id = id_of(ns_a.tool_calls[i]) -%}
  | @@ -219,6 +216,5 @@
  |                      {%- set ns_chk.can_sort = false -%}
  | -                    {%- break -%}
  |                  {%- endif -%}
  |              {%- endfor -%}
  | -        {%- endfor -%}
  | +    {%- endfor -%}
  |      {%- endif -%}
  | @@ -233,3 +229,3 @@
  |                          {%- if eid == tc_id -%}
  | -                            {%- if entry.output is iterable and entry.output is not string and entry.output is not mapping and entry.output and entry.output.0.type == "tool_reference" -%}
  | +                            {%- if entry.output is iterable and entry.output is not string and entry.output is not mapping and entry.output and entry.output[0].type == "tool_reference" -%}
  |                                  {{- tool_references_to_response(entry.output) -}}
  | @@ -239,4 +235,4 @@
  |                          {%- endif -%}
  | -                    {%- endfor -%}
  | -                {%- else -%}
  | +    {%- endfor -%}
  | +{%- else -%}
  |                      {%- set tk_id = id_of(m) -%}

(b) hyperparameters, real vs tiny
  attention.head_count                       | real: 64 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  attention.head_count_kv                    | real: [46] 0 x34, 1 x12 | tiny default: [12] 0 x9, 1 x3 | tiny --mtp: [13] 0 x9, 1 x4 | tiny --quant-dims: [12] 0 x9, 1 x3 | tiny --dialect ik: [12] 0 x9, 1 x3
  attention.indexer.head_count               | real: 32 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2 | tiny --dialect ik: 2
  attention.indexer.index_share_mtp          | real: true | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  attention.indexer.key_length               | real: 128 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.indexer.kpool                    | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  attention.indexer.kpool_select_tail        | real: (absent) | tiny default: true | tiny --mtp: true | tiny --quant-dims: true | tiny --dialect ik: true
  attention.indexer.top_k                    | real: 2048 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8 | tiny --dialect ik: 8
  attention.key_length                       | real: 512 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.key_length_mla                   | real: 256 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.kv_lora_rank                     | real: 512 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.layer_norm_epsilon               | real: 9.999999974752427e-07 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  attention.layer_norm_rms_epsilon           | real: 9.999999747378752e-06 | tiny default: 9.999999747378752e-06 | tiny --mtp: 9.999999747378752e-06 | tiny --quant-dims: 9.999999747378752e-06 | tiny --dialect ik: 9.999999747378752e-06
  attention.q_lora_rank                      | real: 1536 | tiny default: 32 | tiny --mtp: 32 | tiny --quant-dims: 256 | tiny --dialect ik: 32
  attention.value_length                     | real: 512 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16 | tiny --dialect ik: 16
  attention.value_length_mla                 | real: 256 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64 | tiny --dialect ik: 16
  block_count                                | real: 46 | tiny default: 12 | tiny --mtp: 13 | tiny --quant-dims: 12 | tiny --dialect ik: 12
  context_length                             | real: 1048576 | tiny default: 512 | tiny --mtp: 512 | tiny --quant-dims: 512 | tiny --dialect ik: 512
  embedding_length                           | real: 4096 | tiny default: 64 | tiny --mtp: 64 | tiny --quant-dims: 256 | tiny --dialect ik: 64
  expert_count                               | real: 288 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8 | tiny --dialect ik: 8
  expert_feed_forward_length                 | real: 2048 | tiny default: 32 | tiny --mtp: 32 | tiny --quant-dims: 256 | tiny --dialect ik: 32
  expert_gating_func                         | real: 2 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2 | tiny --dialect ik: 2
  expert_group_count                         | real: 1 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: 1
  expert_group_used_count                    | real: 1 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  expert_shared_count                        | real: 1 | tiny default: 1 | tiny --mtp: 1 | tiny --quant-dims: 1 | tiny --dialect ik: 1
  expert_shared_feed_forward_length          | real: 2048 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
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
  nextn_predict_layers                       | real: 1 | tiny default: (absent) | tiny --mtp: 1 | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  rope.dimension_count                       | real: 0 | tiny default: 0 | tiny --mtp: 0 | tiny --quant-dims: 0 | tiny --dialect ik: 0
  ssm.conv_kernel                            | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4 | tiny --dialect ik: 4
  swiglu_clamp_exp                           | real: [46] 10.0 x46 | tiny default: [12] 0.5 x12 | tiny --mtp: [13] 0.5 x13 | tiny --quant-dims: [12] 0.5 x12 | tiny --dialect ik: [12] 0.5 x12
  swiglu_clamp_shexp                         | real: [46] 10.0 x46 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  vocab_size                                 | real: 154880 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent) | tiny --dialect ik: (absent)
  config.json cross-check:
    OK     num_hidden_layers = 45  ->  layers = 45
    OK     num_nextn_predict_layers = 1  ->  nextn_predict_layers = 1
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
    OK     swiglu_limit = 10.0  ->  swiglu_clamp_exp = [46] 10.0 x46
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
    ABSENT index_kpool_always_select_tail = true  ->  attention.indexer.kpool_select_tail = null
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
       [ik] src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent
    [tiny OK] glm.layer_pattern: KDA (head_count_kv 0) vs DSA layers
       real: ['KDA + DSA every 4th layer (3, 7, ...)']  DSA layers [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45] of 46
       tiny: default: ['KDA + DSA every 4th layer (3, 7, ...)']; --mtp: ['KDA + DSA every 4th layer (3, 7, ...)']; --quant-dims: ['KDA + DSA every 4th layer (3, 7, ...)']; --dialect ik: ['KDA + DSA every 4th layer (3, 7, ...)']
       [ec7630a] src/models/glm5-next.cpp:22
       [79e2e74] src/models/glm5-next.cpp:22
       [ik] src/models/glm5-next.cpp: file absent
    [tiny OK] glm.indexer_types: DSA indexer full per layer vs shared with the previous DSA layer
       real: ['full indexer in every DSA layer']  attention.indexer.types = None
       tiny: default: ['full indexer in every DSA layer']; --mtp: ['full indexer in every DSA layer']; --quant-dims: ['full indexer in every DSA layer']; --dialect ik: ['full indexer in every DSA layer']
       [ec7630a] src/models/glm5-next.cpp:47; src/models/glm5-next.cpp:958
       [79e2e74] src/models/glm5-next.cpp:47; src/models/glm5-next.cpp:1059
       [ik] src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent
    [tiny OK] glm.kpool_select_tail: DSA selection appends the incomplete k-pool tail
       real: ['tail appended']  kpool_select_tail = None
       tiny: default: ['tail appended']; --mtp: ['tail appended']; --quant-dims: ['tail appended']; --dialect ik: ['tail appended']
       [ec7630a] src/models/glm5-next.cpp:875
       [79e2e74] src/models/glm5-next.cpp:983
       [ik] src/models/glm5-next.cpp: file absent
    [tiny OK] glm.dsa_threshold: sparse DSA selection only above top_k context tokens; at ec7630a ubatches <= 16 gather the selection
       real: ['selection active above top_k']  top_k 2048, kpool 4: at most 2051 selected cells per token (ec7630a gathers them)
       tiny: default: ['selection active above top_k']; --mtp: ['selection active above top_k']; --quant-dims: ['selection active above top_k']; --dialect ik: ['selection active above top_k']
       [ec7630a] src/models/glm5-next.cpp:328,858; src/models/glm5-next.cpp:331
       [79e2e74] src/models/glm5-next.cpp:320,966; not found
       [ik] src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent
    [tiny OK] glm.rope: MLA must be nope-only (rope.dimension_count 0)
       real: ['n_rot 0 (nope MLA)']  rope.dimension_count = 0, key_length = 512
       tiny: default: ['n_rot 0 (nope MLA)']; --mtp: ['n_rot 0 (nope MLA)']; --quant-dims: ['n_rot 0 (nope MLA)']; --dialect ik: ['n_rot 0 (nope MLA)']
       [ec7630a] src/models/glm5-next.cpp:935; src/llama-model.cpp:1448
       [79e2e74] src/models/glm5-next.cpp:1036; src/llama-model.cpp:1476
       [ik] src/models/glm5-next.cpp: file absent; not found
    [tiny OK] glm.hc: mHC with 4 streams only
       real: ['hc 4']  sinkhorn 20, eps 9.999999974752427e-07
       tiny: default: ['hc 4']; --mtp: ['hc 4']; --quant-dims: ['hc 4']; --dialect ik: ['hc 4']
       [ec7630a] src/models/glm5-next.cpp:53
       [79e2e74] src/models/glm5-next.cpp:53
       [ik] src/models/glm5-next.cpp: file absent
    [tiny OK] glm.dense_lead: leading dense FFN layers, then MoE + shared expert
       real: ['MoE + shared expert', 'dense FFN']  leading_dense_block_count = 3
       tiny: default: ['MoE + shared expert', 'dense FFN']; --mtp: ['MoE + shared expert', 'dense FFN']; --quant-dims: ['MoE + shared expert', 'dense FFN']; --dialect ik: ['MoE + shared expert', 'dense FFN']
       [ec7630a] src/models/glm5-next.cpp:616
       [79e2e74] src/models/glm5-next.cpp:735
       [ik] src/models/glm5-next.cpp: file absent
    [tiny OK] glm.gating: router gating function (absent or none = sigmoid)
       real: ['sigmoid']  expert_gating_func = 2
       tiny: default: ['sigmoid']; --mtp: ['sigmoid']; --quant-dims: ['sigmoid']; --dialect ik: ['sigmoid']
       [ec7630a] src/models/glm5-next.cpp:31
       [79e2e74] src/models/glm5-next.cpp:31
       [ik] src/models/glm5-next.cpp: file absent
    [NOT IN TINY] glm.shexp_clamp: shared-expert SwiGLU clamp: own key or the routed one
       real: ['own shexp clamp key']  swiglu_clamp_shexp = [10.0]
       tiny: default: ['shexp clamp = routed clamp']; --mtp: ['shexp clamp = routed clamp']; --quant-dims: ['shexp clamp = routed clamp']; --dialect ik: ['shexp clamp = routed clamp']
       [ec7630a] src/models/glm5-next.cpp:35; src/llama-graph.cpp:1853
       [79e2e74] src/models/glm5-next.cpp:35; src/llama-graph.cpp:1888
       [ik] src/models/glm5-next.cpp: file absent; src/llama-graph.cpp: file absent
    [NOT IN TINY] glm.layer_norm_eps: LayerNorm eps of the indexer key norm (key absent: 0)
       real: ['eps 9.999999974752427e-07']  
       tiny: default: ['eps 0 (key absent)']; --mtp: ['eps 0 (key absent)']; --quant-dims: ['eps 0 (key absent)']; --dialect ik: ['eps 0 (key absent)']
       [ec7630a] src/models/glm5-next.cpp:9; src/llama-graph.cpp:1612
       [79e2e74] src/models/glm5-next.cpp:9; src/llama-graph.cpp:1647
       [ik] src/models/glm5-next.cpp: file absent; src/llama-graph.cpp: file absent
    [tiny OK] glm.ff_exp_array: expert FFN width: one value or per layer
       real: ['scalar']  expert_feed_forward_length = 2048
       tiny: default: ['scalar']; --mtp: ['scalar']; --quant-dims: ['scalar']; --dialect ik: ['scalar']
       [ec7630a] src/models/glm5-next.cpp:25
       [79e2e74] src/models/glm5-next.cpp:25
       [ik] src/models/glm5-next.cpp: file absent
    [tiny OK] glm.kda_qkv: KDA q/k/v projections: separate (the graph reads wq/wk/wv) or fused attn_qkv
       real: ['separate attn_q/k/v']  
       tiny: default: ['separate attn_q/k/v']; --mtp: ['separate attn_q/k/v']; --quant-dims: ['separate attn_q/k/v']; --dialect ik: ['separate attn_q/k/v']
       [ec7630a] src/models/glm5-next.cpp:109; src/llama-model.cpp:3405
       [79e2e74] src/models/glm5-next.cpp:107; src/llama-model.cpp:3447
       [ik] src/models/glm5-next.cpp: file absent; not found
    [tiny OK] glm.conv_rank: KDA conv1d weights 4-D {d_conv,1,d_inner,1} or 3-D
       real: ['3-D conv1d']  
       tiny: default: ['3-D conv1d']; --mtp: ['3-D conv1d']; --quant-dims: ['3-D conv1d']; --dialect ik: ['3-D conv1d']
       [ec7630a] src/models/glm5-next.cpp:102
       [79e2e74] src/models/glm5-next.cpp:100
       [ik] src/models/glm5-next.cpp: file absent
    [tiny OK] glm.mtp: MTP (NextN) block: count; the MTP graph exists at 79e2e74 only
       real: ['nextn_predict_layers 1']  
       tiny: default: ['nextn_predict_layers 0']; --mtp: ['nextn_predict_layers 1']; --quant-dims: ['nextn_predict_layers 0']; --dialect ik: ['nextn_predict_layers 0']
       [ec7630a] not found
       [79e2e74] src/models/glm5-next.cpp:547
       [ik] src/models/glm5-next.cpp: file absent
    [NOT IN TINY] glm.mtp_heads: MTP block's own embedding / head / head norm instead of the trunk's
       real: ['own nextn.shared_head_norm']  
       tiny: default: ['trunk embedding, head and norm']; --mtp: ['trunk embedding, head and norm']; --quant-dims: ['trunk embedding, head and norm']; --dialect ik: ['trunk embedding, head and norm']
       [ec7630a] not found; not found; not found
       [79e2e74] src/models/glm5-next.cpp:584; src/models/glm5-next.cpp:647; src/models/glm5-next.cpp:658
       [ik] src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent; src/models/glm5-next.cpp: file absent
    [tiny OK] moe.groups: group-limited routing (expert_group_count > 1)
       real: ['no group step']  expert_group_count = 1, used = 1
       tiny: default: ['no group step']; --mtp: ['no group step']; --quant-dims: ['no group step']; --dialect ik: ['no group step']
       [ec7630a] src/llama-model.cpp:1334; src/llama-graph.cpp:2105
       [79e2e74] src/llama-model.cpp:1362; src/llama-graph.cpp:2145
       [ik] not found; src/llama-graph.cpp: file absent
    [tiny OK] moe.norm_w: renormalise the selected expert weights
       real: ['renormalised']  expert_weights_norm = True
       tiny: default: ['renormalised']; --mtp: ['renormalised']; --quant-dims: ['renormalised']; --dialect ik: ['renormalised']
       [ec7630a] src/llama-graph.cpp:2156
       [79e2e74] src/llama-graph.cpp:2196
       [ik] src/llama-graph.cpp: file absent
    [tiny OK] moe.scale: routed weights scaled (scale not 0 or 1)
       real: ['scaled']  expert_weights_scale = 2.5
       tiny: default: ['scaled']; --mtp: ['scaled']; --quant-dims: ['scaled']; --dialect ik: ['scaled']
       [ec7630a] src/llama-graph.cpp:2171
       [79e2e74] src/llama-graph.cpp:2211
       [ik] src/llama-graph.cpp: file absent
    [tiny OK] moe.clamp_glm: routed SwiGLU clamp (glm5-next: fused swiglu_clamp)
       real: ['fused swiglu_clamp']  swiglu_clamp_exp = [10.0]
       tiny: default: ['fused swiglu_clamp']; --mtp: ['fused swiglu_clamp']; --quant-dims: ['fused swiglu_clamp']; --dialect ik: ['fused swiglu_clamp']
       [ec7630a] src/llama-graph.cpp:2247
       [79e2e74] src/llama-graph.cpp:2290
       [ik] src/llama-graph.cpp: file absent
    [tiny OK] moe.used_array: experts per token: one value or per layer
       real: ['scalar']  expert_used_count = 8, expert_count = 288
       tiny: default: ['scalar']; --mtp: ['scalar']; --quant-dims: ['scalar']; --dialect ik: ['scalar']
       [ec7630a] src/llama-model.cpp:1333
       [79e2e74] src/llama-model.cpp:1361
       [ik] not found

(a) tools/gguf_quant_audit.py on the header-only shards (files total 199707321472 B = 185.992 GiB)
  # GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf (6 file(s)), architecture 'glm5-next'
  total 320.76B params, 185.98 GiB, 4.981 bpw
    routed_experts   311.654B    176.82 GiB   4.874 bpw  [Q4_K 65.1%, Q5_K 32.6%, Q6_K 2.3%]
    shared_experts     1.082B      1.07 GiB   8.500 bpw  [Q8_0 100.0%]
    attention          6.182B      6.13 GiB   8.520 bpw  [Q8_0 99.9%, F32 0.1%]
    dense_ffn          0.453B      0.45 GiB   8.500 bpw  [Q8_0 100.0%]
    embedding          0.634B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    output_head        0.634B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    other              0.120B      0.26 GiB  18.511 bpw  [Q8_0 57.4%, F32 42.6%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: blk.3.ffn_gate_exps.weight Q4_K 4.500, blk.3.ffn_up_exps.weight Q4_K 4.500, blk.4.ffn_gate_exps.weight Q4_K 4.500, blk.4.ffn_up_exps.weight Q4_K 4.500, blk.5.ffn_gate_exps.weight Q4_K 4.500, blk.5.ffn_up_exps.weight Q4_K 4.500

=== qwen3.8-flash-next-ud-q4_k_xl (third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl)
status ok, unsloth/Qwen3.8-Flash-Next-GGUF @ 38bb39ee97821de2c9009abb7e93950eec396e66
integrity: Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf OK, Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf OK, Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf OK, Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf OK
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf: 10946624 B, LFS sha256 4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082, header 10946618 B (sha256 acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858), fetched 10946624 B, committed 3085545 B gz, 0 tensors, size check exact (no tensors: the writer's padding after the KV block) (PIN.json recorded: mismatch: header implies 10946618 B), offsets contiguous
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf: 49859583136 B, LFS sha256 3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9, header 18720 B (sha256 92451f96cc2f45e2cdb0a3f50c42b63fc632043d93eca2f8e3e77ec65a9309d5), fetched 262144 B, committed 2945 B gz, 297 tensors, size check exact, offsets contiguous
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf: 49376141504 B, LFS sha256 56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3, header 48064 B (sha256 e80785c5c71408581be7be1812b2cb0089061cf4d66b5dd75230198c22270464), fetched 262144 B, committed 6577 B gz, 752 tensors, size check exact, offsets contiguous
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf: 12087983520 B, LFS sha256 753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a, header 11296 B (sha256 e5a1d1edda4a7c584252ca8c3dbece083c21260702a90940b46f1853f215a3db), fetched 262144 B, committed 1823 B gz, 175 tensors, size check exact, offsets contiguous

(e) identity
  architecture: qwen4exp
  general.name: Qwen3.8 Flash Next
  tokenizer.ggml.model: gpt2
  tokenizer.ggml.pre: qwen35
  vocab (tokenizer.ggml.tokens): 248320
  merges: 247587
  token_embd shape: [2560, 248320]
  output shape: [2560, 248320]
  tokenizer.ggml.add_bos_token: false
  tokenizer.ggml.bos_token_id: 248044
  tokenizer.ggml.eos_token_id: 248046
  tokenizer.ggml.padding_token_id: 248044
  chat templates: ["tokenizer.chat_template"]
  general.* keys: {"general.description": "A Preview of the Qwen4 Architecture", "general.file_type": 15, "general.quantization_version": 2, "general.quantized_by": "Unsloth", "general.repo_url": "https://huggingfac...
  [ec7630a] arch qwen4exp: src/llama-arch.cpp:43
  [ec7630a] tokenizer model gpt2: src/llama-vocab.cpp:1993
  [ec7630a] pre-tokenizer qwen35: src/llama-vocab.cpp:2266
  [79e2e74] arch qwen4exp: src/llama-arch.cpp:44
  [79e2e74] tokenizer model gpt2: src/llama-vocab.cpp:2106
  [79e2e74] pre-tokenizer qwen35: src/llama-vocab.cpp:2383
  [ik] arch qwen4exp: src/llama-arch.cpp:38
  [ik] tokenizer model gpt2: src/llama-vocab.cpp:1816
  [ik] pre-tokenizer qwen35: src/llama-vocab.cpp:2054

(d) expert_group_count = None, expert_group_used_count = None (keys present: [False, False]): no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)
  [ec7630a] src/llama-graph.cpp:2105
  [79e2e74] src/llama-graph.cpp:2145
  [ik] src/llama-graph.cpp: file absent

(c) chat template vs third_party/hf/qwen3.8-flash-next-chat/chat_template.jinja: DIFFERENT (embedded 9993 B sha256 12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce, vendored 8952 B sha256 c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041, first difference at byte 1971, equal up to trailing newlines: False)
  | --- vendored
  | +++ embedded
  | @@ -44,2 +44,14 @@
  |  {%- endif %}
  | +{%- set sysns = namespace(count=0, text='') %}
  | +{%- for message in messages %}
  | +    {%- if sysns.count == loop.index0 and (message.role == 'system' or message.role == 'developer') %}
  | +        {%- set sys_content = render_content(message.content, false, true)|trim %}
  | +        {%- if sys_content %}
  | +            {%- set sysns.text = sysns.text + ('\n' if sysns.text else '') + sys_content %}
  | +        {%- endif %}
  | +        {%- set sysns.count = sysns.count + 1 %}
  | +    {%- endif %}
  | +{%- endfor %}
  | +{%- set num_sys = sysns.count %}
  | +{%- set merged_system = sysns.text %}
  |  {%- set reasoning_instructions = '' %}
  | @@ -47,2 +59,5 @@
  |      {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
  | +    {%- if resolved_reasoning_effort == 'high' %}
  | +        {%- set resolved_reasoning_effort = 'xhigh' %}
  | +    {%- endif %}
  |      {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
  | @@ -68,7 +83,4 @@
  |      {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
  | -    {%- if messages[0].role == 'system' %}
  | -        {%- set content = render_content(messages[0].content, false, true)|trim %}
  | -        {%- if content %}
  | -            {{- '\n\n' + content }}
  | -        {%- endif %}
  | +    {%- if merged_system %}
  | +        {{- '\n\n' + merged_system }}
  |      {%- endif %}
  | @@ -76,9 +88,4 @@
  |  {%- else %}
  | -    {%- if messages[0].role == 'system' %}
  | -        {%- set content = render_content(messages[0].content, false, true)|trim %}
  | -        {%- if content %}
  | -            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
  | -        {%- elif reasoning_instructions %}
  | -            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
  | -        {%- endif %}
  | +    {%- if merged_system %}
  | +        {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + merged_system + '<|im_end|>\n' }}
  |      {%- elif reasoning_instructions %}
  | @@ -98,11 +105,7 @@
  |  {%- endfor %}
  | -{%- if ns.multi_step_tool %}
  | -    {{- raise_exception('No user query found in messages.') }}
  | -{%- endif %}
  |  {%- for message in messages %}
  | +    {%- if loop.index0 >= num_sys %}
  |      {%- set content = render_content(message.content, true)|trim %}
  | -    {%- if message.role == "system" %}
  | -        {%- if not loop.first %}
  | -            {{- raise_exception('System message must be at the beginning.') }}
  | -        {%- endif %}
  | +    {%- if message.role == "system" or message.role == "developer" %}
  | +        {{- raise_exception('System message must be at the beginning.') }}
  |      {%- elif message.role == "user" %}
  | @@ -125,2 +128,5 @@
  |                  {%- endif %}
  | +                {%- if tool_call.name is not defined or tool_call.name is none %}
  | +                    {{- raise_exception('Tool call is missing a function name.') }}
  | +                {%- endif %}
  |                  {%- if loop.first %}
  | @@ -134,3 +140,3 @@
  |                  {%- endif %}
  | -                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
  | +                {%- if tool_call.arguments is mapping %}
  |                      {%- for args_name, args_value in tool_call.arguments|items %}
  | @@ -141,2 +147,8 @@
  |                      {%- endfor %}
  | +                {%- elif tool_call.arguments is string %}
  | +                    {%- if tool_call.arguments|trim %}
  | +                        {{- raise_exception('Tool call arguments for function "' + (tool_call.name | string) + '" were passed as a JSON string. Parse them into an object before calling apply_chat_template.') }}
  | +                    {%- endif %}
  | +                {%- elif tool_call.arguments is defined and tool_call.arguments is not none %}
  | +                    {{- raise_exception('Tool call arguments for function "' + (tool_call.name | string) + '" must be an object/mapping or a JSON string.') }}
  |                  {%- endif %}
  | ... 7 more diff lines

(b) hyperparameters, real vs tiny
  attention.compress_ratios                  | real: [48] 0 x36, 4 x12 | tiny default: [8] 0 x7, 4 x1 | tiny --mtp: [9] 0 x7, 4 x2 | tiny --quant-dims: [8] 0 x7, 4 x1
  attention.head_count                       | real: 24 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4
  attention.head_count_kv                    | real: 2 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2
  attention.indexer.head_count               | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4
  attention.indexer.key_length               | real: 128 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 16
  attention.indexer.top_k                    | real: 2048 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8
  attention.key_length                       | real: 256 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64
  attention.layer_norm_rms_epsilon           | real: 9.999999974752427e-07 | tiny default: 9.999999974752427e-07 | tiny --mtp: 9.999999974752427e-07 | tiny --quant-dims: 9.999999974752427e-07
  attention.recurrent_layers                 | real: (absent) | tiny default: [8] true x6, false x2 | tiny --mtp: [9] true x6, false x3 | tiny --quant-dims: [8] true x6, false x2
  attention.value_length                     | real: 256 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64
  block_count                                | real: 48 | tiny default: 8 | tiny --mtp: 9 | tiny --quant-dims: 8
  context_length                             | real: 262144 | tiny default: 512 | tiny --mtp: 512 | tiny --quant-dims: 512
  embedding_length                           | real: 2560 | tiny default: 64 | tiny --mtp: 64 | tiny --quant-dims: 256
  embedding_length_per_layer_input           | real: 160 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64
  expert_count                               | real: 512 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8
  expert_feed_forward_length                 | real: 640 | tiny default: 32 | tiny --mtp: 32 | tiny --quant-dims: 256
  expert_shared_feed_forward_length          | real: 640 | tiny default: 48 | tiny --mtp: 48 | tiny --quant-dims: 256
  expert_used_count                          | real: 10 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2
  feed_forward_length                        | real: (absent) | tiny default: 96 | tiny --mtp: 96 | tiny --quant-dims: 256
  full_attention_interval                    | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4
  hyper_connection.count                     | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4
  hyper_connection.low_rank                  | real: 320 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8
  nextn_predict_layers                       | real: (absent) | tiny default: (absent) | tiny --mtp: 1 | tiny --quant-dims: (absent)
  ple.conv_kernel                            | real: 4 | tiny default: 3 | tiny --mtp: 3 | tiny --quant-dims: 3
  ple.eos_token_id                           | real: 248044 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2
  ple.head_offsets                           | real: [16] 0 x1, 20000003 x1, 40000026 x1, 60000059... | tiny default: [0, 37, 78, 121] | tiny --mtp: [0, 37, 78, 121] | tiny --quant-dims: [0, 37, 78, 121]
  ple.head_vocab_sizes                       | real: [16] 20000003 x1, 20000023 x1, 20000033 x1, 2... | tiny default: [37, 41, 43, 47] | tiny --mtp: [37, 41, 43, 47] | tiny --quant-dims: [37, 41, 43, 47]
  ple.heads_per_ngram                        | real: 8 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2
  ple.image_token_id                         | real: 248056 | tiny default: (absent) | tiny --mtp: (absent) | tiny --quant-dims: (absent)
  ple.layer_multipliers                      | real: [23703573157769, 20109073645365, 8052911324071] | tiny default: [1798312351201, 1038892647605, 1329187385379] | tiny --mtp: [1798312351201, 1038892647605, 1329187385379] | tiny --quant-dims: [1798312351201, 1038892647605, 1329187385379]
  ple.layers                                 | real: [1] | tiny default: [1] | tiny --mtp: [1] | tiny --quant-dims: [1]
  ple.ngram_size                             | real: 3 | tiny default: 3 | tiny --mtp: 3 | tiny --quant-dims: 3
  rope.dimension_count                       | real: 64 | tiny default: 8 | tiny --mtp: 8 | tiny --quant-dims: 8
  rope.dimension_sections                    | real: [11, 11, 10, 0] | tiny default: [1, 1, 1, 1] | tiny --mtp: [1, 1, 1, 1] | tiny --quant-dims: [1, 1, 1, 1]
  rope.freq_base                             | real: 10000000.0 | tiny default: 1000.0 | tiny --mtp: 1000.0 | tiny --quant-dims: 1000.0
  ssm.conv_kernel                            | real: 4 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4
  ssm.group_count                            | real: 16 | tiny default: 2 | tiny --mtp: 2 | tiny --quant-dims: 2
  ssm.inner_size                             | real: 6144 | tiny default: 64 | tiny --mtp: 64 | tiny --quant-dims: 256
  ssm.state_size                             | real: 128 | tiny default: 16 | tiny --mtp: 16 | tiny --quant-dims: 64
  ssm.time_step_rank                         | real: 48 | tiny default: 4 | tiny --mtp: 4 | tiny --quant-dims: 4
  config.json cross-check:
    OK     num_hidden_layers = 48  ->  layers = 48
    ABSENT mtp_num_hidden_layers = 1  ->  nextn_predict_layers = null
    OK     hidden_size = 2560  ->  embedding_length = 2560
    OK     num_attention_heads = 24  ->  attention.head_count = 24
    OK     num_key_value_heads = 2  ->  attention.head_count_kv = 2
    OK     head_dim = 256  ->  attention.key_length = 256
    OK     num_experts = 512  ->  expert_count = 512
    OK     num_experts_per_tok = 10  ->  expert_used_count = 10
    OK     moe_intermediate_size = 640  ->  expert_feed_forward_length = 640
    OK     shared_expert_intermediate_size = 640  ->  expert_shared_feed_forward_length = 640
    OK     full_attention_interval = 4  ->  full_attention_interval = 4
    OK     linear_conv_kernel_dim = 4  ->  ssm.conv_kernel = 4
    OK     linear_key_head_dim = 128  ->  ssm.state_size = 128
    OK     linear_num_key_heads = 16  ->  ssm.group_count = 16
    OK     linear_num_value_heads = 48  ->  ssm.time_step_rank = 48
    OK     hc_count = 4  ->  hyper_connection.count = 4
    OK     hc_lowrank = 320  ->  hyper_connection.low_rank = 320
    OK     indexer_n_heads = 4  ->  attention.indexer.head_count = 4
    OK     indexer_head_dim = 128  ->  attention.indexer.key_length = 128
    OK     indexer_budget = 2048  ->  attention.indexer.top_k = 2048
    OK     rms_norm_eps = 1e-06  ->  attention.layer_norm_rms_epsilon = 9.999999974752427e-07
    OK     max_position_embeddings = 262144  ->  context_length = 262144
    OK     rope_parameters.rope_theta = 10000000  ->  rope.freq_base = 10000000.0
    OK     ple_layer_ids = [1]  ->  ple.layers = [1]
    OK     ngram_size = 3  ->  ple.ngram_size = 3
    OK     heads_per_ngram = 8  ->  ple.heads_per_ngram = 8
    OK     ple_conv_kernel_size = 4  ->  ple.conv_kernel = 4
    OK     vocab_size = 248320  ->  @vocab = {"tokens": 248320, "token_embd rows": 248320}
  graph branches:
    [tiny OK] moe.groups: group-limited routing (expert_group_count > 1)
       real: ['no group step']  expert_group_count = None, used = None
       tiny: default: ['no group step']; --mtp: ['no group step']; --quant-dims: ['no group step']
       [ec7630a] src/llama-model.cpp:1334; src/llama-graph.cpp:2105
       [79e2e74] src/llama-model.cpp:1362; src/llama-graph.cpp:2145
       [ik] not found; src/llama-graph.cpp: file absent
    [tiny OK] moe.scale: routed weights scaled (scale not 0 or 1)
       real: ['unscaled']  expert_weights_scale = None
       tiny: default: ['unscaled']; --mtp: ['unscaled']; --quant-dims: ['unscaled']
       [ec7630a] src/llama-graph.cpp:2171
       [79e2e74] src/llama-graph.cpp:2211
       [ik] src/llama-graph.cpp: file absent
    [tiny OK] moe.clamp_qwen: routed SwiGLU clamp (qwen4exp is not in the fused list: generic clamp)
       real: ['no clamp']  swiglu_clamp_exp absent
       tiny: default: ['no clamp']; --mtp: ['no clamp']; --quant-dims: ['no clamp']
       [ec7630a] src/llama-graph.cpp:2247
       [79e2e74] src/llama-graph.cpp:2290
       [ik] src/llama-graph.cpp: file absent
    [tiny OK] moe.used_array: experts per token: one value or per layer
       real: ['scalar']  expert_used_count = 10, expert_count = 512
       tiny: default: ['scalar']; --mtp: ['scalar']; --quant-dims: ['scalar']
       [ec7630a] src/llama-model.cpp:1333
       [79e2e74] src/llama-model.cpp:1361
       [ik] not found
    [tiny OK] rope.scaling: RoPE scaling type (YaRN etc. change ggml_rope_multi's factors)
       real: ['rope scaling none']  factor None, original ctx None, freq_base 10000000.0
       tiny: default: ['rope scaling none']; --mtp: ['rope scaling none']; --quant-dims: ['rope scaling none']
       [ec7630a] src/llama-model.cpp:1418
       [79e2e74] src/llama-model.cpp:1446
       [ik] not found
    [tiny OK] qwen.hc: low-rank hyper-connections (count > 1)
       real: ['hyper-connections']  hc 4, low rank 320
       tiny: default: ['hyper-connections']; --mtp: ['hyper-connections']; --quant-dims: ['hyper-connections']
       [ec7630a] src/models/qwen4exp.cpp:49
       [79e2e74] src/models/qwen4exp.cpp:49
       [ik] src/models/qwen4exp.cpp: file absent
    [NOT IN TINY] qwen.recurrent_source: linear/full layer pattern from the array or the interval rule
       real: ['full_attention_interval rule']  full_attention_interval = 4
       tiny: default: ['recurrent_layers array']; --mtp: ['recurrent_layers array']; --quant-dims: ['recurrent_layers array']
       [ec7630a] src/models/qwen4exp.cpp:149
       [79e2e74] src/models/qwen4exp.cpp:149
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.full_attention: full-attention layers: QSA (compress ratio > 0) or dense
       real: ['QSA full attention']  compress_ratios = [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4]
       tiny: default: ['QSA full attention', 'dense full attention (compress ratio 0)']; --mtp: ['QSA full attention', 'dense full attention (compress ratio 0)']; --quant-dims: ['QSA full attention', 'dense full attention (compress ratio 0)']
       [ec7630a] src/models/qwen4exp.cpp:942
       [79e2e74] src/models/qwen4exp.cpp:939
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.ple: per-layer n-gram embeddings (PLE) on a linear layer
       real: ['PLE layer']  ple.layers [1], ngram 3, heads/ngram 8, conv 4
       tiny: default: ['PLE layer']; --mtp: ['PLE layer']; --quant-dims: ['PLE layer']
       [ec7630a] src/models/qwen4exp.cpp:267,453
       [79e2e74] src/models/qwen4exp.cpp:266,452
       [ik] src/models/qwen4exp.cpp: file absent
    [NOT IN TINY] qwen.ple_image_token: PLE image token id: own key or the EOS fallback
       real: ['image token id key']  
       tiny: default: ['EOS fallback']; --mtp: ['EOS fallback']; --quant-dims: ['EOS fallback']
       [ec7630a] src/models/qwen4exp.cpp:111
       [79e2e74] src/models/qwen4exp.cpp:111
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.output: own output.weight or tied to token_embd
       real: ['own output.weight']  
       tiny: default: ['own output.weight']; --mtp: ['own output.weight']; --quant-dims: ['own output.weight']
       [ec7630a] src/models/qwen4exp.cpp:191
       [79e2e74] src/models/qwen4exp.cpp:190
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.attn_scale: attention scale: 1/sqrt(head_dim) or attention.scale
       real: ['1/sqrt(head_dim)']  
       tiny: default: ['1/sqrt(head_dim)']; --mtp: ['1/sqrt(head_dim)']; --quant-dims: ['1/sqrt(head_dim)']
       [ec7630a] src/models/qwen4exp.cpp:994
       [79e2e74] src/models/qwen4exp.cpp:991
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.ff_defaults: expert / shared-expert widths: explicit keys or derived from n_ff
       real: ['explicit n_ff_exp', 'explicit n_ff_shexp']  
       tiny: default: ['explicit n_ff_exp', 'explicit n_ff_shexp']; --mtp: ['explicit n_ff_exp', 'explicit n_ff_shexp']; --quant-dims: ['explicit n_ff_exp', 'explicit n_ff_shexp']
       [ec7630a] src/models/qwen4exp.cpp:221; src/models/qwen4exp.cpp:222
       [79e2e74] src/models/qwen4exp.cpp:220; src/models/qwen4exp.cpp:221
       [ik] src/models/qwen4exp.cpp: file absent; src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.rope_sections: IMRoPE sections in use
       real: ['section h non-empty', 'section t non-empty', 'section w non-empty']  sections [11, 11, 10, 0], n_rot 64, head 256
       tiny: default: ['section e non-empty', 'section h non-empty', 'section t non-empty', 'section w non-empty']; --mtp: ['section e non-empty', 'section h non-empty', 'section t non-empty', 'section w non-empty']; --quant-dims: ['section e non-empty', 'section h non-empty', 'section t non-empty', 'section w non-empty']
       [ec7630a] src/models/qwen4exp.cpp:31
       [79e2e74] src/models/qwen4exp.cpp:31
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.partial_rope: partial rotary (n_rot < head_dim)
       real: ['partial rotary']  
       tiny: default: ['partial rotary']; --mtp: ['partial rotary']; --quant-dims: ['partial rotary']
       [ec7630a] src/models/qwen4exp.cpp:978
       [79e2e74] src/models/qwen4exp.cpp:975
       [ik] src/models/qwen4exp.cpp: file absent
    [tiny OK] qwen.gate_up_fused: routed experts: separate gate/up or fused gate_up_exps
       real: ['separate gate/up']  
       tiny: default: ['separate gate/up']; --mtp: ['separate gate/up']; --quant-dims: ['separate gate/up']
       [ec7630a] src/llama-graph.cpp:2191
       [79e2e74] src/llama-graph.cpp:2234
       [ik] src/llama-graph.cpp: file absent
    [tiny OK] qwen.mtp: MTP (NextN) block count
       real: ['nextn_predict_layers 0']  
       tiny: default: ['nextn_predict_layers 0']; --mtp: ['nextn_predict_layers 1']; --quant-dims: ['nextn_predict_layers 0']
       [ec7630a] src/models/qwen4exp.cpp:531
       [79e2e74] src/models/qwen4exp.cpp:528
       [ik] src/models/qwen4exp.cpp: file absent

(a) tools/gguf_quant_audit.py on the header-only shards (files total 111334654784 B = 103.688 GiB)
  # Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf (4 file(s)), architecture 'qwen4exp'
  total 176.94B params, 103.68 GiB, 5.033 bpw
    routed_experts   120.796B     71.73 GiB   5.101 bpw  [Q4_K 65.3%, Q5_1 29.9%, Q8_0 3.5%, Q5_K 1.4%]
    shared_experts     0.236B      0.23 GiB   8.512 bpw  [Q8_0 99.9%, F32 0.1%]
    attention          2.704B      2.72 GiB   8.644 bpw  [Q8_0 98.9%, BF16 0.7%, F32 0.4%]
    embedding          0.636B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    output_head        0.636B      0.63 GiB   8.500 bpw  [Q8_0 100.0%]
    other             51.937B     27.74 GiB   4.587 bpw  [IQ4_NL 98.6%, Q8_0 1.3%, F32 0.1%]
  verdict: PASS (routed experts >= 4.0 bpw, non-expert matrices >= 4.0 bpw: 0 failing tensors, 0 warnings)
(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): PASS; lowest-bpw matrices: per_layer_token_embd.weight IQ4_NL 4.500, blk.0.ffn_gate_exps.weight Q4_K 4.500, blk.0.ffn_up_exps.weight Q4_K 4.500, blk.1.ffn_gate_exps.weight Q4_K 4.500, blk.1.ffn_up_exps.weight Q4_K 4.500, blk.3.ffn_gate_exps.weight Q4_K 4.500
```

## Appendix B. `2026-10-11-n15-embedded-templates.py`, full summary

```
glm/vendored render (server vs jinja2): 51 of 72 equal
  default/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  default/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  default/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  default/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  default/tools_ends_with_assistant agp=True {"first_diff": 3688, "server": "plits on ','. The test passes.‖", "jinja2": "plits on ','. The test passes.‖<|assistant|><think>"}
  effort_low/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_low/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_low/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_low/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_high/reasoning_mixed agp=True {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_high/reasoning_mixed agp=False {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_high/reasoning_empty agp=True {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_high/reasoning_empty agp=False {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_medium/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_medium/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_medium/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_medium/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_none/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_none/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_none/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_none/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  prefix default/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix default/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix preserve_off/jinja2: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 1936, "nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs" -> "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"
  prefix preserve_off/server: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 1936, "nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs" -> "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"
  prefix preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 1936, "nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs" -> "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"
  prefix enable_thinking_false/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 3: byte 3320, "_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_" -> "_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t"
  switch preserve_off: {"shared_bytes": 1936, "prompt_bytes": 4068, "at": "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"}
  switch enable_thinking_false: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
  switch effort_low: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖Low<|system|>\n# Tools\n\nYou may call one or more fu"}
  switch effort_high: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖High<|system|>\n# Tools\n\nYou may call one or more f"}
  switch effort_medium: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
  switch effort_none: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
  switch one_tool_added: {"shared_bytes": 1501, "prompt_bytes": 4068, "at": "}}, \"required\": [\"verbose\"]}}\n‖{\"name\": \"web_search\", \"description\": \"Search.\", \""}
glm/embedded render (server vs jinja2): 51 of 72 equal
  default/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  default/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  default/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  default/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  default/tools_ends_with_assistant agp=True {"first_diff": 3688, "server": "plits on ','. The test passes.‖", "jinja2": "plits on ','. The test passes.‖<|assistant|><think>"}
  effort_low/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_low/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_low/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_low/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_high/reasoning_mixed agp=True {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_high/reasoning_mixed agp=False {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_high/reasoning_empty agp=True {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_high/reasoning_empty agp=False {"first_diff": 116, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_medium/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_medium/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_medium/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_medium/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_none/reasoning_mixed agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_none/reasoning_mixed agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  effort_none/reasoning_empty agp=True {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>"}
  effort_none/reasoning_empty agp=False {"first_diff": 115, "server": "<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3", "jinja2": "<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3"}
  prefix default/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix default/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix preserve_off/jinja2: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 1936, "nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs" -> "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"
  prefix preserve_off/server: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 1936, "nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs" -> "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"
  prefix preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 1936, "nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs" -> "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"
  prefix enable_thinking_false/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 3: byte 3320, "_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_" -> "_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t"
  switch preserve_off: {"shared_bytes": 1936, "prompt_bytes": 4068, "at": "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"}
  switch enable_thinking_false: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
  switch effort_low: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖Low<|system|>\n# Tools\n\nYou may call one or more fu"}
  switch effort_high: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖High<|system|>\n# Tools\n\nYou may call one or more f"}
  switch effort_medium: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
  switch effort_none: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
  switch one_tool_added: {"shared_bytes": 1501, "prompt_bytes": 4068, "at": "}}, \"required\": [\"verbose\"]}}\n‖{\"name\": \"web_search\", \"description\": \"Search.\", \""}
glm/jinja2_vendored_vs_embedded: 68 of 68 renders byte-equal
qwen/vendored render (server vs jinja2): 62 of 72 equal
  default/tools_ends_with_assistant agp=True {"first_diff": 4624, "server": "plits on ','. The test passes.‖", "jinja2": "plits on ','. The test passes.‖<|im_end|>\n<|im_start|>assistant\n<think>\n"}
  default/tools_ends_with_assistant agp=False {"first_diff": 4624, "server": "plits on ','. The test passes.‖", "jinja2": "plits on ','. The test passes.‖<|im_end|>\n"}
  effort_high/reasoning agp=True error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/reasoning agp=False error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/reasoning_mixed agp=True error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/reasoning_mixed agp=False error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/reasoning_empty agp=True error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/reasoning_empty agp=False error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/agent_session agp=True error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  effort_high/agent_session agp=False error: RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{-
  prefix default/jinja2: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
  prefix default/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
  prefix default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix preserve_off/jinja2: history breaks 3 (k=[6, 12, 16]), refused prefixes 1; loop breaks 1 of 7 requests
    break at request 4: byte 2756, "end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor" -> "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"
  prefix preserve_off/server: history breaks 3 (k=[6, 12, 16]), refused prefixes 1; loop breaks 1 of 7 requests
    break at request 4: byte 2756, "end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor" -> "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"
  prefix preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 2756, "end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor" -> "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"
  prefix enable_thinking_false/jinja2: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/jinja2: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  switch preserve_off: {"shared_bytes": 2756, "prompt_bytes": 5093, "at": "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"}
  switch enable_thinking_false: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
  switch effort_low: {"shared_bytes": 46, "prompt_bytes": 5093, "at": "em\nReasoning effort is set to ‖low. Keep your thinking brief and focused, moving "}
  switch effort_medium: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
  switch effort_high: {"error": "RuntimeError: HTTP 500 {\"error\": {\"code\": 500, \"message\": \"\\n------------\\nWhile executing CallExpression at line 49, column 28 in source:\\n...', 'low') %}↵ {{- raise_exception('U
  switch effort_none: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
  switch one_tool_added: {"shared_bytes": 1748, "prompt_bytes": 5093, "at": "}, \"required\": [\"verbose\"]}}}\n‖{\"type\": \"function\", \"function\": {\"name\": \"web_sea"}
qwen/embedded render (server vs jinja2): 70 of 72 equal
  default/tools_ends_with_assistant agp=True {"first_diff": 4624, "server": "plits on ','. The test passes.‖", "jinja2": "plits on ','. The test passes.‖<|im_end|>\n<|im_start|>assistant\n<think>\n"}
  default/tools_ends_with_assistant agp=False {"first_diff": 4624, "server": "plits on ','. The test passes.‖", "jinja2": "plits on ','. The test passes.‖<|im_end|>\n"}
  prefix default/jinja2: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix default/server: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix preserve_off/jinja2: history breaks 3 (k=[6, 12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 2756, "end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor" -> "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"
  prefix preserve_off/server: history breaks 3 (k=[6, 12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 2756, "end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor" -> "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"
  prefix preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
    break at request 4: byte 2756, "end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor" -> "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"
  prefix enable_thinking_false/jinja2: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/jinja2: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server: history breaks 1 (k=[6]), refused prefixes 0; loop breaks 0 of 7 requests
  prefix empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
  switch preserve_off: {"shared_bytes": 2756, "prompt_bytes": 5093, "at": "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"}
  switch enable_thinking_false: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
  switch effort_low: {"shared_bytes": 46, "prompt_bytes": 5093, "at": "em\nReasoning effort is set to ‖low. Keep your thinking brief and focused, moving "}
  switch effort_medium: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
  switch effort_high: {"shared_bytes": 5093, "prompt_bytes": 5093, "at": "<|im_start|>assistant\n<think>\n‖"}
  switch effort_none: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
  switch one_tool_added: {"shared_bytes": 1748, "prompt_bytes": 5093, "at": "}, \"required\": [\"verbose\"]}}}\n‖{\"type\": \"function\", \"function\": {\"name\": \"web_sea"}
qwen/jinja2_vendored_vs_embedded: 60 of 68 renders byte-equal
  differ: effort_high/reasoning agp=True {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/reasoning agp=False {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/reasoning_mixed agp=True {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/reasoning_mixed agp=False {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/reasoning_empty agp=True {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/reasoning_empty agp=False {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/agent_session agp=True {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
  differ: effort_high/agent_session agp=False {"vendored_error": "TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "embedded_error": null}
```
