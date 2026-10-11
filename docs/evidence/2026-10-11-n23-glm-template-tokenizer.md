# N23: a corrected GLM-5.3-Flash chat template for llama.cpp's Jinja engine; tokenizer.json pinned by sha256; token-level prefix stability

Date 2026-10-11. Role: parity-qa. Branch `claude/n23-glm-template-tokenizer`, cut from N19 @`e71b241` (contains N14's
`tools/chattmpl` and N16's `data/hf-requests` vendoring). Hardware-free: the CPU llama-server / llama-tokenize of the
R1 pin in the sandbox (4 cores), N14's bytes-vocabulary tiny models (random weights: they only make llama-server load a
hybrid model so that its template engine, chat parser and checkpoints run for real; they say nothing about either real
model). `<S>` stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution. No GPU, no performance claims. Round 1: everything that needs huggingface.co is **PENDING runner**.

## Result

1. **Corrected template, CONFIRMED (section 2).** `templates/glm-5.3-flash/chat_template.jinja` (12 047 B, sha256
   `b2ff07b242f6edc13337126b89285e979e9a5bcd5a17f4a6a1f9235d67349839`) = the official
   `zai-org/GLM-5.3-Flash@eb9eb208` template (sha256 `0c4099f3…6309c5`) plus a header comment, one inserted line
   (`{%- set reasoning_content = none %}`, line 153, the first statement of every assistant iteration after `content`)
   and one changed test (`reasoning_content is defined` → `is not none`, line 160). `CorrectedFile.test_only_the_two_lines_differ`
   asserts exactly that diff. The variable is now assigned in every iteration, so the scope difference between the two
   engines (jinja2: one scope per iteration; llama.cpp @79e2e74: one scope for all iterations,
   `common/jinja/runtime.cpp:486,601-618@79e2e74`) can no longer show.
2. **Byte-identical to the official format, CONFIRMED (section 3).** On all 21 conversations (N14's 10 + 11 new: missing
   / empty / `null` / whitespace-only / inline reasoning, tool loops, a two-round mixed agent loop), all 7 GLM switches,
   generation prompt on and off: jinja2 configured as transformers (`tools/chattmpl/hf_render.py`) renders ours and the
   official template to the same bytes, 294/294; transformers 5.16.0's own `render_jinja_template` (scratch venv) equals
   hf_render for both templates, 294/294 each.
3. **Through llama-server @79e2e74, CONFIRMED (section 3):** the server with `--chat-template-file <ours>` renders the
   same bytes as jinja2 with the **official** template on 279 of 298 rows; the 19 others are not template effects: 7 rows
   of N14's by-design `--prefill-assistant` case and 12 rows where the server refuses `reasoning_content: null` with
   HTTP 500 (finding 6). With the official template the server matches on 185 of 298: the same 19 plus 94 leak rows, in
   every conversation with a turn after one that has reasoning (N14's 2 and 7 of the 11 new ones). Parser, format
   detection, delimiters (checkpoint positions) and round trips are identical to the official template's (N14 table,
   7/7 requests) — CONFIRMED by `ParseHarnessN23`.
4. **Leak gone, prefix reuse equal or better, CONFIRMED live (section 4)** (`/v1/chat/completions`, every turn forced by
   a grammar, `timings.cache_n`): with ours, `cache_n` = everything the slot holds at every request of the default,
   empty-reasoning and mixed agent sessions; the only partial reuse left is the by-design one where past reasoning is
   dropped (`preserve_off`, identical numbers to the official template: `cache_n 1932`, `prompt_n 1624`; with
   `--ctx-checkpoints 4`: 1420 / 2136). With the official template: empty-reasoning session — 1 break, `cache_n 3316` of
   3517 held, 348 re-processed; mixed loop — 3 breaks (270, 247, 304 re-processed), session prompt_n total 3056 vs 2482
   with ours (−18.8 %, PROVISIONAL as a share: tiny model, invented session); and the model sees a wrong history.
5. **R1 recommendation (not applied; `scripts/r1/configs` belongs to the PR #1 session), section 8:** add
   `"--chat-template-file", "${REPO}/templates/glm-5.3-flash/chat_template.jinja"` after `"--jinja"` in
   `scripts/r1/configs/llamacpp-server-moecache.json:40` (and, once ik_llama's `--help` lists the flag, in
   `ik-llamacpp-server.json:37`). This replaces the GGUF-embedded template (Unsloth's, which keeps the same bug, N15) by
   the official format.
6. **New server finding, CONFIRMED:** llama-server @79e2e74 answers HTTP 500 `Failed to parse messages:
   [json.exception.type_error.302] type must be string, but is null` for an assistant message with
   `"reasoning_content": null` (`common/chat.cpp:422-423@79e2e74` reads it with `message.at(...)` into a string), with
   either template. jinja2 renders it as an empty think block. Agent clients must omit the field or send `""` (the server
   itself drops an empty one, `common/chat.cpp:202-204`).
7. **Whitespace-only reasoning leaks too with the official template, CONFIRMED:** a turn with reasoning `" \n "` followed
   by one without renders `<think> \n </think>a2` through the server (jinja2: `<think></think>a2`). Fixed by ours.
8. **tokenizer.json: pinned, not committed (section 6).** `scripts/vendor_hf_files.py` now takes `"lfs_files"` in a
   request (only `tokenizer.json`, ≤ 64 MiB): the runner downloads it from `resolve/<40-hex revision>`, checks size and
   sha256 against the tree API's LFS pointer, and records path / size / sha256 / LFS oid / URL plus the model card's
   license fields in `PIN.json`; the bytes never enter git. `scripts/hf_lfs_fetch.py` fetches by that pin into a cache
   outside the tree (`~/.cache/408032gb/hf-lfs/<sha256>/tokenizer.json`), re-hashing on every lookup; CI keeps the cache
   with `actions/cache` keyed on the pins. Requests: `data/hf-requests/glm-5.3-flash-tokenizer` (@`eb9eb208`) and
   `qwen3.8-flash-next-tokenizer` (@`de4b8e4d`), the revisions the templates are pinned at; `LICENSE` and `README.md`
   (model card) vendored as text. Licenses: GLM `license:mit` (HF tag, `third_party/hf/glm-5.3-flash-chat/PIN.json`,
   CONFIRMED as a tag); Qwen `license:other` (tag CONFIRMED; the terms UNKNOWN until `LICENSE` arrives) — **PENDING
   runner**. Sizes 20 217 442 B / 12 809 320 B (CONFIRMED from the vendored listings).
9. **Token level with llama.cpp's GGUF tokenizer, CONFIRMED in the sandbox (section 7)** (llama-tokenize @79e2e74 on
   vocab-only GGUFs from N15's committed Unsloth headers, `--no-bos --no-escape`): GLM — official and corrected
   template identical, token-level breaks only where bytes break (past reasoning dropped), all 16 tags single tokens, no
   merge across a prompt / turn boundary (concat canonical everywhere). Qwen — **N15's `ĊĊ` finding is real and wider
   than N15 saw:** every assistant turn with empty, missing or whitespace-only reasoning breaks token reuse at the last
   token of the previous prompt (`<think> Ċ | Ċ </think>` held vs `<think> ĊĊ </think>` re-tokenized): 1 of 6 requests
   in the empty-reasoning session, **4 of 7** in the mixed loop; with `enable_thinking=false` no break (its generation
   prompt already ends `<think>\n\n</think>\n\n`). With the HF tokenizer and HF-vs-llama.cpp id parity: **PENDING
   runner** (`tests/test_token_level.py` asserts the same expectations for both tokenizers).
10. **The Qwen break is real in llama-server too, CONFIRMED live (section 7.3)** on a tiny qwen4exp carrying the real
    Qwen vocabulary (248 320 tokens, 247 587 merges, pre `qwen35`, from N15's header) with the slot pre-filled with the
    canonical tokens of each turn: reuse breaks exactly where the tokenizer predicts, and the server restores the
    checkpoint 4 tokens before the end of the previous prompt (`cache_n` = previous prompt − 4: 1055, 826, 893, 1060,
    1133), re-processing turn + 4 tokens (58-61 per break); over the mixed loop Σ prompt_n 1021 instead of 819 (+202
    tokens). No full re-prefill. **Mitigation:** the official format cannot avoid it (any `<think>\n\n</think>`
    re-tokenizes to `ĊĊ`); recommended: accept it for R1 (bounded by the last turn), use `enable_thinking: false` where
    thinking is off anyway (no break, CONFIRMED); alternatives in section 7.3.
11. **N15 caveat closed, CONFIRMED:** N15 ran llama-tokenize without `--no-escape` (escape processing is on by default,
    `common/common.h:570@79e2e74`); none of the 97 distinct rendered texts (47 GLM, 50 Qwen) contains a backslash, so it did not matter.
    The GLM vocabulary adds no BOS either (llama-tokenize without `--no-bos`: `[gMASK]<sop>hi` → `[154822, 154824,
    6023]`; its header has no `tokenizer.ggml.add_bos_token` key), nor does Qwen's (`add_bos_token = false` in the
    header), so `--no-bos` changes nothing for these files and llama-server does not double `[gMASK]`.

Tests: `tests/test_glm_template_n23.py` 12 (6 without a build: 3 file, 3 jinja2; 1 transformers; 5 llama-server),
`tests/test_token_level.py` 12 (5 llama, 5 hf, 1 parity, 1 pin), `tests/test_hf_lfs.py` 15. Sandbox: all llama-server
and llama side tests pass; hf / parity / pin tests skip cleanly (PENDING runner). Mutation check: section 10.

## 1. Inputs

| input | value | label |
|---|---|---|
| official GLM template | `third_party/hf/glm-5.3-flash-chat/chat_template.jinja`, `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, 10 950 B, sha256 `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` (PIN.json, SHA256SUMS) | CONFIRMED |
| official Qwen template | `third_party/hf/qwen3.8-flash-next-chat/chat_template.jinja`, `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | CONFIRMED |
| llama.cpp | `<S>/llama.cpp` at `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (= `tools/engines/pins.json`, checked by `run_n23.sh`), shared CPU build `<S>/llama.cpp/build-cpu` (N14's); llama-tokenize `<S>/n15/build-tok` of the same checkout (N15) | CONFIRMED |
| tiny models | N14's `bytes_vocab_model.py` with N11's tinygen (`git archive origin/claude/n11-r1-tools-cpu-server tools/tinygen`), rebuilt by `run_n23.sh`: glm5next_bytes.gguf sha256 `3fb41bfc…4f3d`, qwen4exp_bytes.gguf `135bf3f8…946c` | CONFIRMED |
| GGUF headers | N15's committed Unsloth first shards, read with `git archive origin/claude/n15-gguf-headers` (@`67bd5baceafe63e70e1b78010181225ecebbefa4`) into `<S>/n23/n15` (nothing under third_party/): GLM `.header.gz` sha256 `f93ba472…244a`, Qwen `7f7334a9…1b5a`; vocab-only GGUFs by N15's `vocab_gguf.py`: `vocab-glm.gguf` 9 429 874 B sha256 `d66c3980…5f95` (arch glm5-next, pre glm4), `vocab-qwen.gguf` 10 946 536 B `c092638a…8f74` (qwen4exp, pre qwen35) | CONFIRMED |
| reference renderer | jinja2 3.1.6 (system), `tools/chattmpl/hf_render.py` (transformers 5.16.0 configuration, N14); plus transformers 5.16.0, tokenizers 0.23.3, jinja2 3.1.6 from PyPI in the scratch venv `<S>/n23/venv` (no torch) | CONFIRMED |
| HF tokenizer.json | not available in the sandbox (no huggingface.co) | PENDING runner |

## 2. The change

```
$ diff third_party/hf/glm-5.3-flash-chat/chat_template.jinja templates/glm-5.3-flash/chat_template.jinja
0a1,10
> {#- GLM-5.3-Flash chat template, corrected for llama.cpp's Jinja engine (408032gb, task N23).
> Source: zai-org/GLM-5.3-Flash revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a, file chat_template.jinja
> (10950 bytes, sha256 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5; vendored copy in
> third_party/hf/glm-5.3-flash-chat). License of the source repository: MIT.
> Change, and nothing else: in the assistant branch of the message loop the variable reasoning_content is reset to
> none at the start of every iteration and tested with "is not none" instead of "is defined". The source sets it only
> when the message has reasoning; jinja2 scopes a set inside a for loop to one iteration, but llama.cpp's engine (at
> 79e2e74, common/jinja/runtime.cpp for_statement::execute_impl: one scope for all iterations) keeps it, so an assistant
> turn without reasoning, or with an empty one (llama-server drops it), was rendered with the previous turn's reasoning.
> The rendered bytes under jinja2 / transformers are unchanged (tests/test_glm_template_n23.py). -#}
142a153
> {%- set reasoning_content = none %}
149c160
< {%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is defined -%}
---
> {%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is not none -%}
```

Why it is equivalent under jinja2: in the official template `reasoning_content` is either unset or a string in each
iteration (lines 143-148: `m.reasoning_content` when it is a string, else the text before `</think>` in content); ours
holds `none` exactly where it was unset and the same string otherwise, so `is not none` ⇔ `is defined`. One input
renders differently by construction: a caller that passes a **global** variable `reasoning_content` to the template
(e.g. `reasoning_content="G"` as a chat-template kwarg on `reasoning_mixed`: official `<|assistant|><think>G</think>a2`,
ours `<|assistant|><think></think>a2`, CONFIRMED with hf_render). No client in this project does that; it is not among
the 21 conversations. Why the header is safe: llama.cpp chooses some chat handlers by substrings of the template source
(`common/chat.cpp:1082-1283@79e2e74`); the header contains no `<` or `[` (`CorrectedFile.test_header_names_source_and_change`).
Both engines strip the `{#- … -#}` comment and its newline (llama.cpp lexer `common/jinja/lexer.cpp:136-147,190-195`:
`-#}` sets the rstrip of the following text); the byte equality below includes the first bytes `[gMASK]<sop>`.

## 3. Rendering

`tools/chattmpl/run_n23.py` sections `equal` (jinja2 only) and `render` (llama-server `/apply-template`), driver
`tools/chattmpl/run_n23.sh` (section 11 has its full output at the final commit). Equal section, with transformers:

```
$ <S>/n23/venv/bin/python tools/chattmpl/run_n23.py --out <S>/n23/eq-head --sections equal      (HEAD 4ed71cb)
equal jinja2 official vs corrected: 294/294 byte-equal
equal transformers 5.16.0 render_jinja_template vs hf_render (official): 294/294 byte-equal
equal transformers 5.16.0 render_jinja_template vs hf_render (ours): 294/294 byte-equal
```

Server section: `render glm_n23: 279/298 byte-equal (server vs jinja2)` (ours through the server against jinja2 with
the official template) and `render glm: 185/298` (the official template through the server). The 19 differences of
glm_n23, all asserted by `ServerRenderN23.test_server_with_ours_equals_official_format`:

| rows | case | why | label |
|---|---|---|---|
| 7 | `tools_ends_with_assistant`, generation prompt on, every switch | `--prefill-assistant` (default on) continues a trailing assistant message (N14 finding 1(b)) | CONFIRMED |
| 12 | `reasoning_null`, both generation flags, 6 switches (not `enable_thinking_false`, which strips reasoning) | HTTP 500 `type must be string, but is null` (`common/chat.cpp:422-423`) | CONFIRMED |

Examples of the leak with the official template (server vs jinja2, first differing byte marked `‖`), each asserted by
`ServerRenderN23.test_official_template_still_leaks_at_this_pin`:

```
missing_after_two                  server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4' jinja2='…<think>‖</think>a3<|user|>u4'
reasoning_whitespace_then_missing  server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3' jinja2='…<think>‖</think>a2<|user|>u3'
inline_then_missing                server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='…<think>‖</think>a2<|user|>u3'
tools_reasoning_empty              server='_response><|assistant|><think>‖R1</think><tool_call>read_file…' jinja2='…<think>‖</think><tool_call>read_file…'
agent_session_empty_reasoning      server='_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t' jinja2='…<think>‖</think><tool_call>run_tests…'
```

`inline_then_missing` leaks even with `enable_thinking=false` (GLM ignores it, N14 finding 9; the reasoning comes from
the content split, which the switch does not strip). Server warnings with ours: only N14's `chat template supports
preserving reasoning, it is enabled by default` and the 500s of the `null` rows.

## 4. Prefix stability and live B3

Prefix section (`run_n14.section_prefix` for `glm_n23`: server prompts with ours, turns as the official template writes
them; N14's 4 sessions + the mixed loop with defaults and with past reasoning dropped): loop breaks only in
`preserve_off` (request k=11) and `agent_loop_mixed_preserve_off` (k=11), every server prompt equal to the jinja2 prompt
of the official template (`PrefixN23`). With the official template through the server, `empty_reasoning_turn`
1 of 7 and `agent_loop_mixed` 3 of 8 requests break.

Live (`run_n14.section_live`, `-c 16384 -lv 4`, turns forced byte for byte; one byte = one token with these models):

| config | ours: breaks / Σ prompt_n | official: breaks / Σ prompt_n | label |
|---|---|---|---|
| default | none / 2561 | none / 2561 | CONFIRMED |
| preserve_off | request 5 (`cache_n 1932`, `prompt_n 1624`) / 4114 | same / 4114 | CONFIRMED |
| preserve_off, `--ctx-checkpoints 4` | request 5 (1420, 2136) / 4626 | same / 4626 | CONFIRMED |
| empty_reasoning_turn | none / 2561 | request 4 (`cache_n 3316` of 3517 held, `prompt_n 348`) / 2826 | CONFIRMED |
| agent_loop_mixed | none / 2482 | requests 2, 3, 6 (`prompt_n` 270, 247, 304) / 3056 | CONFIRMED |

With ours `cache_n` equals the held bytes at every request that extends (`LiveB3N23` asserts it per request). The
official template's prompts in the last two rows are also longer (they carry leaked reasoning), so its Σ prompt_n mixes
two effects; "unchanged or better" holds row by row.

## 5. Parser (`ParseHarnessN23`)

Ours through `common` @79e2e74: `using differential autoparser`, format `peg-native`, generation prompt
`<|assistant|><think>`, thinking tags `<think>` / `['</think>']`, delimiters `[('assistant', '<|assistant|>'),
('user', '<|user|>')]`, tool grammar built, 7/7 ideal turns parsed and re-rendered to the same bytes, k=9's JSON null
argument returned as `"null"`, padded reasoning reproduced, padded content not, unknown tool → parse error — every
value equal to N14's GLM row (CONFIRMED, run_n23 output in section 11).

## 6. tokenizer.json: vendoring and CI

| question | answer | label |
|---|---|---|
| commit the file? | No. 20.2 MB + 12.8 MB would stay in history forever; a pin (`PIN.json` `lfs_files`: repo, 40-hex revision, path, size, sha256 = LFS oid, URL) is enough to reproduce the bytes from `resolve/<revision>`, and the fetcher refuses any other bytes | design |
| who verifies the sha | the runner, once at vendoring (download, size and sha256 against `lfs.size` / `lfs.oid` of the tree API); `hf_lfs_fetch.py` again at every fetch and lookup | CONFIRMED by tests (fake hub, local HTTP server) |
| where tests find it | `hf_lfs_fetch.path_for(PIN.json, "tokenizer.json")` → `$HF_LFS_CACHE` or `~/.cache/408032gb/hf-lfs/<sha256>/tokenizer.json`; missing or corrupt → `None` → the class skips with "PENDING runner" | CONFIRMED by tests |
| CI | `.github/workflows/token-level.yml`: `actions/cache@v4` on that directory, key `hf-lfs-${{ hashFiles('third_party/hf/*-tokenizer/PIN.json') }}`; `hf_lfs_fetch.py` (no-op on a cache hit); llama-tokenize of the R1 pin; vocab-only GGUFs from N15's headers at the pinned commit `67bd5ba` with sha256 check; `tests.test_token_level` + transformers render, **fails on any skip**; summary as notice annotations, `token-prefix.json` as artifact | NOT RUN (runner) |
| trigger | `vendor-hf-branch.yml` calls `token-level.yml` (`workflow_call`, `ref` = the branch) right after its vendor job, because the bot's own push starts no workflow and a workflow that is not on `main` cannot be dispatched; later pushes that touch the listed paths start it directly. Before the pins exist it only warns `PENDING` | design |
| licenses | recorded in PIN.json: `license_tags` (as before) and `card_license` (cardData `license`, `license_name`, `license_link`); `LICENSE` and `README.md` vendored as text next to the pin | PENDING runner |
| what if HF removes the revision | the pin still identifies the bytes; a copy would have to come from elsewhere (bundle, release asset) and pass the same sha check | PROVISIONAL |
| cost | one 33 MB download per cache miss (pins change); no git growth | PROVISIONAL |

Requests (`tests/test_hf_lfs.py Requests` checks them against the chat pins):

```
$ cat data/hf-requests/glm-5.3-flash-tokenizer/REQUEST.json data/hf-requests/qwen3.8-flash-next-tokenizer/REQUEST.json
{"repo": "zai-org/GLM-5.3-Flash", "revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a", "files": ["LICENSE", "README.md"], "lfs_files": ["tokenizer.json"], "lfs_max_bytes": 33554432}
{"repo": "Qwen/Qwen3.8-Flash-Next", "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540", "files": ["LICENSE", "README.md"], "lfs_files": ["tokenizer.json"], "lfs_max_bytes": 33554432}
```

## 7. Token level

### 7.1 llama.cpp's GGUF tokenizer (CONFIRMED, sandbox)

`tools/chattmpl/token_prefix.py` (output in section 11): per request pair, T = llama-tokenize @79e2e74 `--no-bos
--no-escape` (special tokens parsed, as the server); held = T(prompt k) + T(turn k) (the canonical tokens of the turn
stand in for the sampled ones), next = T(prompt k+1).

| template | session | byte breaks | token breaks | label |
|---|---|---|---|---|
| GLM official = ours (same jinja2 bytes) | default, enable_thinking_false, empty_reasoning_turn, agent_loop_mixed | 0 | 0 | CONFIRMED |
| GLM official = ours | preserve_off, agent_loop_mixed_preserve_off | request 4 | request 4 (same place) | CONFIRMED |
| Qwen official | default, enable_thinking_false | 0 | 0 | CONFIRMED |
| Qwen official | preserve_off | request 4 | request 4 | CONFIRMED |
| Qwen official | empty_reasoning_turn | 0 | request 3 (reuse 1058 of 1117, prompt 1059) | CONFIRMED (= N15) |
| Qwen official | agent_loop_mixed | 0 | requests 1, 2, 5, 6 (reuse = prompt − 1 each) | CONFIRMED |
| Qwen official | agent_loop_mixed_preserve_off | request 4 | requests 1, 2, 4, 5, 6 | CONFIRMED |

GLM ids of the 16 tags: `[gMASK]` 154822 … `</tool_response>` 154846, each one token. Qwen: `<|im_start|>` 248045,
`<|im_end|>` 248046, `<think>` 248068, `</think>` 248069, `<tool_call>` 248058, `</tool_call>` 248059,
`<tool_response>` 248066, `</tool_response>` 248067 single; `<function=`, `</function>`, `<parameter=`, `</parameter>`
three tokens each. At every Qwen newline break: held `… <think>(248068) Ċ(198) | Ċ(198) </think>(248069)`, next
`… <think> ĊĊ(271) </think>`.

### 7.2 HF tokenizer and parity: PENDING runner

`tests/test_token_level.py` `HFTokenLevel` asserts the same table for `tokenizers` + tokenizer.json
(`add_special_tokens=False`, as transformers' `apply_chat_template(tokenize=True)`), `Parity` that both tokenizers give
the same ids for all 47 GLM and 50 Qwen distinct rendered texts. Expected to agree (PROVISIONAL: the GGUF vocabularies
were converted from these tokenizers; the pre-tokenizer regexes `glm4` / `qwen35` are llama.cpp's own transcriptions).

### 7.3 Qwen: is it real, and what to do

Real at the tokenizer level (CONFIRMED above): in thinking mode the generation prompt ends `<think>\n` (template
line 168) and a turn with empty reasoning is written back as `<think>\n\n</think>\n\n` (line 117, reasoning trimmed).
The model's own tokens after the prompt are UNKNOWN (it may sample `Ċ </think>` or `</think>` directly — the latter
breaks at the same place in bytes too). Where the reuse stops: at the last token of the previous prompt.

Cost, CONFIRMED live on CPU (`docs/evidence/mutation-scripts/2026-10-11-n23-qwen-newline-live.py`, output in section
11): N11's tiny qwen4exp generator (random weights, hybrid, so checkpoints are used) written with the tokenizer keys of
N15's Qwen header instead of its byte vocabulary (141 026 400 B, `n_vocab 248320, merges 247587`), the official Qwen
template, `-c 16384`. Per request: (a) `/v1/chat/completions` on conv[:k] (`max_tokens 1`) — the server renders and
tokenizes, `cache_n` is the measurement; (b) `/completion` with the token ids T(prompt k) + T(turn k) (this server's
`/tokenize`, turn cut after `<|im_end|>`), so the slot holds exactly the canonical tokens a model would have sampled.

| session | request (next) | held | predicted reuse | `cache_n` | restored checkpoint | `prompt_n` | label |
|---|---|---|---|---|---|---|---|
| default | 1-6 | — | = held | = held | none | 21-50 | CONFIRMED |
| empty_reasoning_turn | 4 | 1116 | 1058 | 1055 | `n_tokens = 1055` (= prompt 1059 − 4) | 84 | CONFIRMED |
| agent_loop_mixed | 2 | 874 | 829 | 826 | 826 (= 830 − 4) | 71 | CONFIRMED |
| agent_loop_mixed | 3 | 934 | 896 | 893 | 893 (= 897 − 4) | 61 | CONFIRMED |
| agent_loop_mixed | 6 | 1122 | 1063 | 1060 | 1060 (= 1064 − 4) | 77 | CONFIRMED |
| agent_loop_mixed | 7 | 1184 | 1136 | 1133 | 1133 (= 1137 − 4) | 66 | CONFIRMED |

Every other request reused exactly the held tokens. Extra work per break = held − `cache_n` = 58-61 tokens (the turn
+ 4); over the mixed loop Σ `prompt_n` = 1021 against 819 if every request had extended (+202 tokens, +24.7 % of this
short invented session; PROVISIONAL as a share for real sessions). Bounded by the length of the last turn; no full
re-prefill; the model's history is the correct one.

Options:

| option | effect | cost / risk | recommendation |
|---|---|---|---|
| A. accept | turn + 4 tokens per turn without reasoning | small; grows with long tool-call turns | **recommended for R1** |
| B. `enable_thinking: false` per request when reasoning is off anyway | generation prompt `<think>\n\n</think>\n\n` = history bytes → no break (CONFIRMED, row above) | only where thinking is not wanted | use when applicable |
| C. our own Qwen template with a generation prompt ending at `<think>` | the model writes `\n` itself, history re-tokenizes canonically (PROVISIONAL) | changes the official prompt bytes the model was trained on; quality UNKNOWN | not without an eval |
| D. agent harness keeps token ids (`/completion` with token arrays) | no re-tokenization at all | only for our own harness, not OpenAI-compatible clients | possible for the bench harness |
| E. llama-server patch: compare cached and new prompts by text near the end of the cached prompt | fixes every merge of this kind | an upstream patch; out of N23 | upstream idea |

GLM has no such break: `<think>` and `</think>` are adjacent single tokens and the generation prompt ends at
`<think>`, so nothing can merge across the boundary (concat canonical on every GLM request, CONFIRMED).

## 8. R1: proposed change (not applied)

`scripts/r1/configs` is owned by the PR #1 session. Proposed, after `"--jinja"`:

```
scripts/r1/configs/llamacpp-server-moecache.json:40
-             "--jinja", "--metrics", "--slot-save-path", "${SLOT_DIR}"],
+             "--jinja", "--chat-template-file", "${REPO}/templates/glm-5.3-flash/chat_template.jinja",
+             "--metrics", "--slot-save-path", "${SLOT_DIR}"],
```

`${REPO}` is one of run_r1.py's fixed variables (`scripts/r1/run_r1.py:94`). The run should record the template's
sha256 (`b2ff07b2…9839`) with the engine flags. `ik-llamacpp-server.json:37` (ik_llama.cpp, own template code): the same
flag if its `--help` lists `--chat-template-file` (UNKNOWN; `artifacts/engine/flags_check.json` on the server); whether
ik's engine has the loop-scope bug is UNKNOWN, but ours renders the same in both scoping models, so it is safe either
way (PROVISIONAL: shown for jinja2 and llama.cpp only). The Qwen configs need no template change for this bug (N14).
Also for the client side of R1's agent loop: never send `"reasoning_content": null` (finding 6).

## 9. Tests

| module | class | needs | sandbox |
|---|---|---|---|
| test_glm_template_n23 | CorrectedFile (3) | nothing | pass |
| | JinjaEquality (3) | jinja2 | pass |
| | TransformersRender (1) | transformers | pass in `<S>/n23/venv`; skip in system python |
| | ServerRenderN23 (2), ParseHarnessN23 (1), PrefixN23 (1), LiveB3N23 (1) | llama-server + tiny model (+ harness) | pass (`run_n23.sh`) |
| test_token_level | PinRecords (1) | vendored tokenizer pins | skip (PENDING runner) |
| | LlamaTokenLevel (5) | llama-tokenize + vocab GGUFs | pass |
| | HFTokenLevel (5), Parity (1) | tokenizer.json in the cache (+ llama) | skip (PENDING runner) |
| test_hf_lfs | VendorLfs (7), Fetch (7), Requests (1) | nothing (no network) | pass |

The N14 suite (`tests/test_chat_templates.py`, 14 tests) still passes with the refactored `run_n14.py` (section 11).

## 10. Mutation check (`docs/evidence/mutation-scripts/2026-10-11-n23-mutations.sh`)

Each mutation runs on a fresh `git archive HEAD` copy (HEAD `4ed71cb`) and runs the module(s) covering the mutated file
with the sandbox's binaries. The first run was stopped by the session's 30-minute limit for background commands after
M6 (other sessions' llama-servers shared the 4 cores); M7-M13 were run again with `ONLY` (the script's filter, added for
that; no base runs then). Both logs, literally:

```
$ time bash docs/evidence/mutation-scripts/2026-10-11-n23-mutations.sh       (stopped after M6, no time output)
HEAD 4ed71cb
base template exit 0 (OK (skipped=1))
base token exit 0 (OK (skipped=7))
base lfs exit 0 (OK)
M1-no-reset KILLED  (CorrectedFile.test_only_the_two_lines_differ JinjaEquality.test_byte_equal_everywhere JinjaEquality.test_turn_without_reasoning_renders_an_empty_think_block LiveB3N23.test_prompt_cache_reuse PrefixN23.test_server_prompts_extend_held_turns ServerRenderN23.test_server_with_ours_equals_official_format )
M2-is-defined KILLED  (CorrectedFile.test_only_the_two_lines_differ JinjaEquality.test_byte_equal_everywhere JinjaEquality.test_turn_without_reasoning_renders_an_empty_think_block )
M3-reset-empty-string KILLED  (CorrectedFile.test_only_the_two_lines_differ )
M4-tag-in-header KILLED  (CorrectedFile.test_header_names_source_and_change )
M5-server-loads-official KILLED  (LiveB3N23.test_prompt_cache_reuse PrefixN23.test_server_prompts_extend_held_turns ServerRenderN23.test_server_with_ours_equals_official_format )
M6-reference-is-ours SURVIVED (OK (skipped=1))
$ time ONLY='^M(7|8|9|1[0-3])-' bash docs/evidence/mutation-scripts/2026-10-11-n23-mutations.sh
HEAD 4ed71cb
M7-conv-no-missing-turn KILLED  (JinjaEquality.test_turn_without_reasoning_renders_an_empty_think_block ServerRenderN23.test_official_template_still_leaks_at_this_pin )
M8-held-drops-a-token KILLED  (LlamaTokenLevel.test_glm_tokens_break_only_where_bytes_break LlamaTokenLevel.test_qwen_newline_merge_after_empty_reasoning )
M9-llama-with-bos SURVIVED (OK (skipped=7))
M10-cache-size-only KILLED  (Fetch.test_corrupt_cache_is_replaced )
M11-vendor-no-sha-check KILLED  (VendorLfs.test_sha256_mismatch_fails )
M12-vendor-any-lfs KILLED  (VendorLfs.test_never_weights )
M13-fetch-no-length-cap KILLED  (Fetch.test_longer_body_is_cut_off )

real	11m19.761s
user	8m11.587s
sys	0m31.604s
rc=0
```

11 of 13 killed. Survivors, both equivalent mutations:

| mutation | why it cannot fail a test | label |
|---|---|---|
| M6 reference = our file instead of the official one | ours renders like the official template under jinja2 on every input (294/294); that property is tested directly by `JinjaEquality` | CONFIRMED (section 3) |
| M9 llama-tokenize without `--no-bos` | neither vocabulary adds a BOS (finding 11), so the ids are the same | CONFIRMED |

M3 (reset to `''` instead of `none`) renders the same bytes everywhere (an empty string falls into the same
`<think></think>`) and is killed only by the structural test of the file, as intended.

## 11. Outputs

### 11.1 `tools/chattmpl/run_n23.sh` at `4ed71cb`

Driver, run_n23.py summary and both test modules (`tests.test_glm_template_n23`, `tests.test_chat_templates`):
`Ran 26 tests … OK (skipped=1)` (the transformers class; it passes in the venv, 11.4), `driver rc=0`, wall 15 min 20 s.
Full output in Appendix A. Its `results.json` equals the earlier run's on the pre-commit tree field for field
(server logs excluded).

### 11.2 Token level, llama.cpp's tokenizer (`4ed71cb`)

```
$ time python3 tools/chattmpl/token_prefix.py --llama-tokenize <S>/n15/build-tok/bin/llama-tokenize --vocab-glm <S>/n23/n15/vocab-glm.gguf --vocab-qwen <S>/n23/n15/vocab-qwen.gguf --out <S>/n23/tok-llama-head
llama glm_hf: tags {"[gMASK]": [154822], "<sop>": [154824], "<|system|>": [154826], "<|user|>": [154827], "<|assistant|>": [154828], "<|observation|>": [154829], "<think>": [154841], "</think>": [154842], "<tool_call>": [154843], "</tool_call>": [154844], "<arg_key>": [154847], "</arg_key>": [154848], "<arg_value>": [154849], "</arg_value>": [154850], "<tool_response>": [154845], "</tool_response>": [154846]}
  default: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  preserve_off: 6 request pairs; byte-extends 5, token-extends 5, prompt-token-extends 5, concat canonical 6
    request 4 (k=11): bytes False (reuse 1936 of 3704), tokens False (reuse 494 of 797, prompt 777) at {"index": 494, "held": [13, 154828, 154841, 40, 1265, 1401], "next": [13, 154828, 154841, 154842, 154843, 1607]}
  enable_thinking_false: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  empty_reasoning_turn: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  agent_loop_mixed: 7 request pairs; byte-extends 7, token-extends 7, prompt-token-extends 7, concat canonical 7
  agent_loop_mixed_preserve_off: 7 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 7
    request 4 (k=11): bytes False (reuse 1921 of 3270), tokens False (reuse 490 of 685, prompt 669) at {"index": 490, "held": [13, 154828, 154841, 5890, 2176, 12402], "next": [13, 154828, 154841, 154842, 154843, 38002]}
llama glm_n23: tags {"[gMASK]": [154822], "<sop>": [154824], "<|system|>": [154826], "<|user|>": [154827], "<|assistant|>": [154828], "<|observation|>": [154829], "<think>": [154841], "</think>": [154842], "<tool_call>": [154843], "</tool_call>": [154844], "<arg_key>": [154847], "</arg_key>": [154848], "<arg_value>": [154849], "</arg_value>": [154850], "<tool_response>": [154845], "</tool_response>": [154846]}
  default: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  preserve_off: 6 request pairs; byte-extends 5, token-extends 5, prompt-token-extends 5, concat canonical 6
    request 4 (k=11): bytes False (reuse 1936 of 3704), tokens False (reuse 494 of 797, prompt 777) at {"index": 494, "held": [13, 154828, 154841, 40, 1265, 1401], "next": [13, 154828, 154841, 154842, 154843, 1607]}
  enable_thinking_false: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  empty_reasoning_turn: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  agent_loop_mixed: 7 request pairs; byte-extends 7, token-extends 7, prompt-token-extends 7, concat canonical 7
  agent_loop_mixed_preserve_off: 7 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 7
    request 4 (k=11): bytes False (reuse 1921 of 3270), tokens False (reuse 490 of 685, prompt 669) at {"index": 490, "held": [13, 154828, 154841, 5890, 2176, 12402], "next": [13, 154828, 154841, 154842, 154843, 38002]}
llama qwen_hf: tags {"<|im_start|>": [248045], "<|im_end|>": [248046], "<think>": [248068], "</think>": [248069], "<tool_call>": [248058], "</tool_call>": [248059], "<tool_response>": [248066], "</tool_response>": [248067], "<function=": [27, 1628, 28], "</function>": [510, 1628, 29], "<parameter=": [27, 15704, 28], "</parameter>": [510, 15704, 29]}
  default: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  preserve_off: 6 request pairs; byte-extends 5, token-extends 5, prompt-token-extends 5, concat canonical 6
    request 4 (k=11): bytes False (reuse 2756 of 4651), tokens False (reuse 683 of 1170, prompt 1146) at {"index": 683, "held": [248045, 74455, 198, 248068, 198, 40], "next": [248045, 74455, 198, 248058, 198, 27]}
  enable_thinking_false: 6 request pairs; byte-extends 6, token-extends 6, prompt-token-extends 6, concat canonical 6
  empty_reasoning_turn: 6 request pairs; byte-extends 6, token-extends 5, prompt-token-extends 5, concat canonical 5
    request 3 (k=9): bytes True (reuse 4425 of 4425), tokens False (reuse 1058 of 1117, prompt 1059) at {"index": 1058, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
  agent_loop_mixed: 7 request pairs; byte-extends 7, token-extends 3, prompt-token-extends 3, concat canonical 3
    request 1 (k=5): bytes True (reuse 3462 of 3462), tokens False (reuse 829 of 875, prompt 830) at {"index": 829, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
    request 2 (k=7): bytes True (reuse 3724 of 3724), tokens False (reuse 896 of 935, prompt 897) at {"index": 896, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
    request 5 (k=13): bytes True (reuse 4492 of 4492), tokens False (reuse 1063 of 1123, prompt 1064) at {"index": 1063, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
    request 6 (k=15): bytes True (reuse 4750 of 4750), tokens False (reuse 1136 of 1185, prompt 1137) at {"index": 1136, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
  agent_loop_mixed_preserve_off: 7 request pairs; byte-extends 6, token-extends 2, prompt-token-extends 2, concat canonical 3
    request 1 (k=5): bytes True (reuse 3462 of 3462), tokens False (reuse 829 of 875, prompt 830) at {"index": 829, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
    request 2 (k=7): bytes True (reuse 3724 of 3724), tokens False (reuse 896 of 935, prompt 897) at {"index": 896, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
    request 4 (k=11): bytes False (reuse 2741 of 4216), tokens False (reuse 679 of 1051, prompt 1031) at {"index": 679, "held": [248045, 74455, 198, 248068, 198, 5708], "next": [248045, 74455, 198, 248058, 198, 27]}
    request 5 (k=13): bytes True (reuse 4332 of 4332), tokens False (reuse 1021 of 1081, prompt 1022) at {"index": 1021, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}
    request 6 (k=15): bytes True (reuse 4590 of 4590), tokens False (reuse 1094 of 1143, prompt 1095) at {"index": 1094, "held": [74455, 198, 248068, 198, 198, 248069], "next": [74455, 198, 248068, 271, 248069, 271]}

real	3m22.837s
user	2m24.186s
sys	0m11.435s
rc=0
```

### 11.3 Qwen newline break, live (`docs/evidence/mutation-scripts/2026-10-11-n23-qwen-newline-live.py`, code of `4ed71cb`)

```
$ time python3 docs/evidence/mutation-scripts/2026-10-11-n23-qwen-newline-live.py --tinygen <S>/n23/n11src/tools/tinygen --gguf-py <S>/llama.cpp/gguf-py --n15 <S>/n23/n15 --server-bin <S>/llama.cpp/build-cpu/bin/llama-server --out <S>/n23/qlive/out
wrote <S>/n23/qlive/out/qwen4exp_realvocab.gguf: n_vocab 248320, merges 247587, 141026400 B
default
  request 0 k=2 prompt 685 tok | held None predicted reuse None extends None | cache_n 0 prompt_n 685
  request 1 k=4 prompt 747 tok | held 724 predicted reuse 724 extends True | cache_n 724 prompt_n 23
  request 2 k=7 prompt 925 tok | held 875 predicted reuse 875 extends True | cache_n 875 prompt_n 50
  request 3 k=9 prompt 1059 tok | held 1038 predicted reuse 1038 extends True | cache_n 1038 prompt_n 21
  request 4 k=11 prompt 1146 tok | held 1122 predicted reuse 1122 extends True | cache_n 1122 prompt_n 24
  request 5 k=13 prompt 1191 tok | held 1169 predicted reuse 1169 extends True | cache_n 1169 prompt_n 22
  request 6 k=15 prompt 1270 tok | held 1237 predicted reuse 1237 extends True | cache_n 1237 prompt_n 33
empty_reasoning_turn
  request 0 k=2 prompt 685 tok | held None predicted reuse None extends None | cache_n 0 prompt_n 685
  request 1 k=4 prompt 747 tok | held 724 predicted reuse 724 extends True | cache_n 724 prompt_n 23
  request 2 k=7 prompt 925 tok | held 875 predicted reuse 875 extends True | cache_n 875 prompt_n 50
  request 3 k=9 prompt 1059 tok | held 1038 predicted reuse 1038 extends True | cache_n 1038 prompt_n 21
  request 4 k=11 prompt 1139 tok | held 1116 predicted reuse 1058 extends False | cache_n 1055 prompt_n 84
  request 5 k=13 prompt 1184 tok | held 1162 predicted reuse 1162 extends True | cache_n 1162 prompt_n 22
  request 6 k=15 prompt 1263 tok | held 1230 predicted reuse 1230 extends True | cache_n 1230 prompt_n 33
    | restored context checkpoint (pos_min = 1054, pos_max = 1054, n_tokens = 1055, n_past = 1055, size = 0.038 MiB)
agent_loop_mixed
  request 0 k=2 prompt 681 tok | held None predicted reuse None extends None | cache_n 0 prompt_n 681
  request 1 k=5 prompt 830 tok | held 796 predicted reuse 796 extends True | cache_n 796 prompt_n 34
  request 2 k=7 prompt 897 tok | held 874 predicted reuse 829 extends False | cache_n 826 prompt_n 71
  request 3 k=9 prompt 954 tok | held 934 predicted reuse 896 extends False | cache_n 893 prompt_n 61
  request 4 k=11 prompt 1031 tok | held 1014 predicted reuse 1014 extends True | cache_n 1014 prompt_n 17
  request 5 k=13 prompt 1064 tok | held 1050 predicted reuse 1050 extends True | cache_n 1050 prompt_n 14
  request 6 k=15 prompt 1137 tok | held 1122 predicted reuse 1063 extends False | cache_n 1060 prompt_n 77
  request 7 k=17 prompt 1199 tok | held 1184 predicted reuse 1136 extends False | cache_n 1133 prompt_n 66
    | restored context checkpoint (pos_min = 825, pos_max = 825, n_tokens = 826, n_past = 826, size = 0.038 MiB)
    | restored context checkpoint (pos_min = 892, pos_max = 892, n_tokens = 893, n_past = 893, size = 0.038 MiB)
    | restored context checkpoint (pos_min = 1059, pos_max = 1059, n_tokens = 1060, n_past = 1060, size = 0.038 MiB)
    | restored context checkpoint (pos_min = 1132, pos_max = 1132, n_tokens = 1133, n_past = 1133, size = 0.038 MiB)

real	0m21.960s
user	0m17.388s
sys	0m2.898s
rc=0
```

### 11.4 Tests

```
$ <S>/n23/venv/bin/python -m unittest -v tests.test_glm_template_n23.TransformersRender tests.test_glm_template_n23.JinjaEquality tests.test_glm_template_n23.CorrectedFile   (4ed71cb; PyTorch notice line removed)
test_transformers_renders_like_hf_render (tests.test_glm_template_n23.TransformersRender.test_transformers_renders_like_hf_render) ... ok
test_agent_loops_prefix_stable (tests.test_glm_template_n23.JinjaEquality.test_agent_loops_prefix_stable) ... ok
test_byte_equal_everywhere (tests.test_glm_template_n23.JinjaEquality.test_byte_equal_everywhere) ... ok
test_turn_without_reasoning_renders_an_empty_think_block (tests.test_glm_template_n23.JinjaEquality.test_turn_without_reasoning_renders_an_empty_think_block) ... ok
test_header_names_source_and_change (tests.test_glm_template_n23.CorrectedFile.test_header_names_source_and_change) ... ok
test_only_the_two_lines_differ (tests.test_glm_template_n23.CorrectedFile.test_only_the_two_lines_differ) ... ok
test_source_is_the_pinned_template (tests.test_glm_template_n23.CorrectedFile.test_source_is_the_pinned_template) ... ok

----------------------------------------------------------------------
Ran 7 tests in 3.428s

OK
$ TOKCHECK_LLAMA_TOKENIZE=<S>/n15/build-tok/bin/llama-tokenize TOKCHECK_VOCAB_GLM=<S>/n23/n15/vocab-glm.gguf TOKCHECK_VOCAB_QWEN=<S>/n23/n15/vocab-qwen.gguf python3 -m unittest -v tests.test_token_level   (working tree before the commits; same files as fdd7fe5)
test_glm_official_and_corrected_identical (tests.test_token_level.HFTokenLevel.test_glm_official_and_corrected_identical) ... skipped 'needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner)'
test_glm_tags_single_tokens (tests.test_token_level.HFTokenLevel.test_glm_tags_single_tokens) ... skipped 'needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner)'
test_glm_tokens_break_only_where_bytes_break (tests.test_token_level.HFTokenLevel.test_glm_tokens_break_only_where_bytes_break) ... skipped 'needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner)'
test_qwen_newline_merge_after_empty_reasoning (tests.test_token_level.HFTokenLevel.test_qwen_newline_merge_after_empty_reasoning) ... skipped 'needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner)'
test_qwen_tags (tests.test_token_level.HFTokenLevel.test_qwen_tags) ... skipped 'needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner)'
test_glm_official_and_corrected_identical (tests.test_token_level.LlamaTokenLevel.test_glm_official_and_corrected_identical) ... ok
test_glm_tags_single_tokens (tests.test_token_level.LlamaTokenLevel.test_glm_tags_single_tokens) ... ok
test_glm_tokens_break_only_where_bytes_break (tests.test_token_level.LlamaTokenLevel.test_glm_tokens_break_only_where_bytes_break) ... ok
test_qwen_newline_merge_after_empty_reasoning (tests.test_token_level.LlamaTokenLevel.test_qwen_newline_merge_after_empty_reasoning) ... ok
test_qwen_tags (tests.test_token_level.LlamaTokenLevel.test_qwen_tags) ... ok
test_same_ids (tests.test_token_level.Parity.test_same_ids) ... skipped 'needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner); set TOKCHECK_LLAMA_TOKENIZE, TOKCHECK_VOCAB_GLM, TOKCHECK_VOCAB_QWEN'
test_lfs_records_match_the_template_pins (tests.test_token_level.PinRecords.test_lfs_records_match_the_template_pins) ... skipped 'tokenizer PIN.json not vendored yet (PENDING runner)'

----------------------------------------------------------------------
Ran 12 tests in 293.201s

OK (skipped=7)
rc=0
```

The mutation script's base runs repeat the three groups at `4ed71cb` (section 10: `base template exit 0 (OK (skipped=1))`,
`base token exit 0 (OK (skipped=7))`, `base lfs exit 0 (OK)`). The full unit suite (`python3 -m unittest discover -s
tests`) is in the report of this round.

## 12. NOT RUN / PENDING runner

- PENDING runner: vendoring of both tokenizer requests (PIN.json with sha256, card license, LICENSE / README.md text);
  `token-level.yml` (HF tokenizer token-level table, HF-vs-llama.cpp id parity, transformers render on the runner,
  PinRecords). Round 2 reads its annotations and artifact.
- The live Qwen probe uses canonical tokens for the turn; tokens a real model samples may differ (UNKNOWN), which can
  only add breaks.
- NOT RUN: ik_llama.cpp's template engine with our file; image / video / audio content parts; `add_vision_id`.
- UNKNOWN: which tokens the real models sample around an empty think block.
- The MIT notice of the GLM source: the header names the license; the copyright line of the upstream `LICENSE` is to be
  added to the header once the file is vendored (PENDING runner).

## Appendix A. `tools/chattmpl/run_n23.sh` at `4ed71cb`, full output

```
$ time tools/chattmpl/run_n23.sh <S>/llama.cpp <S>/llama.cpp/build-cpu <S>/n23/n11src/tools/tinygen <S>/n23/run2
llama.cpp checkout 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (pins.json llama.cpp: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6)
version: 0.6.0-dev (build 1, commit 79e2e74)
b2ff07b242f6edc13337126b89285e979e9a5bcd5a17f4a6a1f9235d67349839  /home/user/wt-n23/templates/glm-5.3-flash/chat_template.jinja
0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5  /home/user/wt-n23/third_party/hf/glm-5.3-flash-chat/chat_template.jinja
== harness
== models
wrote <S>/n23/run2/models/qwen4exp_bytes.gguf (n_vocab 260)
wrote <S>/n23/run2/models/glm5next_bytes.gguf (n_vocab 260)
3fb41bfc9e78a1b77c39fc1db07c1e01b9c3a6115c992d0cb8f4f6634ed64f3d  <S>/n23/run2/models/glm5next_bytes.gguf
135bf3f8ae16e442be9653b11a5b4490d1a7db2dfb69c0693b983e9eef43946c  <S>/n23/run2/models/qwen4exp_bytes.gguf
== run_n23.py
equal jinja2 official vs corrected: 294/294 byte-equal
equal transformers: transformers not importable: No module named 'transformers'
render glm_n23: 279/298 byte-equal (server vs jinja2)
  default tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  default reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  default reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  preserve_off tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  preserve_off reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  preserve_off reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  enable_thinking_false tools_ends_with_assistant agp=True first_diff=3469 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_low tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_low reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_low reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_high tools_ends_with_assistant agp=True first_diff=3689 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_high reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_high reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_medium tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_medium reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_medium reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_none tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_none reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_none reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_none system_user agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none system_user agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none multi_turn agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none multi_turn agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none no_system agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none no_system agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_mixed agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_mixed agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none inline_think agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none inline_think agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_empty agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_empty agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none tools agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none tools agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none tools_ends_with_assistant agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none tools_ends_with_assistant agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none agent_session agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none agent_session agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none missing_first agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none missing_first agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none missing_after_two agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none missing_after_two agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_null agp=True equal_to_jinja2=False same_as_server_default=False
  effort_none reasoning_null agp=False equal_to_jinja2=False same_as_server_default=False
  effort_none reasoning_whitespace_then_missing agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_whitespace_then_missing agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none inline_then_missing agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none inline_then_missing agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none inline_empty_then_missing agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none inline_empty_then_missing agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none tools_reasoning_empty agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none tools_reasoning_empty agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none tools_no_reasoning_at_all agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none tools_no_reasoning_at_all agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none agent_loop_mixed agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none agent_loop_mixed agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none agent_loop_mixed_open agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none agent_loop_mixed_open agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none agent_session_empty_reasoning agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none agent_session_empty_reasoning agp=False equal_to_jinja2=True same_as_server_default=True
  server_flag_reasoning_off reasoning agp=True equal_to_jinja2=True same_as_server_default=True
  server_flag_reasoning_off agent_session agp=True equal_to_jinja2=True same_as_server_default=True
  log (default): 0.00.361.819 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
  log (server_flag_reasoning_off): 0.00.880.050 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
render glm: 185/298 byte-equal (server vs jinja2)
  default reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  default reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  default reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  default reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  default tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  default missing_after_two agp=True first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4<|assistant|><think>' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4<|assistant|><think>'
  default missing_after_two agp=False first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4'
  default reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  default reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  default reasoning_whitespace_then_missing agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  default reasoning_whitespace_then_missing agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  default inline_then_missing agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  default inline_then_missing agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  default tools_reasoning_empty agp=True first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  default tools_reasoning_empty agp=False first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  default agent_loop_mixed agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  default agent_loop_mixed agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  default agent_loop_mixed_open agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  default agent_loop_mixed_open agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  default agent_session_empty_reasoning agp=True first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  default agent_session_empty_reasoning agp=False first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  preserve_off tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  preserve_off reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  preserve_off reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  preserve_off agent_loop_mixed_open agp=True first_diff=3233 server='move them.<|assistant|><think>‖Done: a.py and b.py.</think><tool_call>write_file<' jinja2='move them.<|assistant|><think>‖</think><tool_call>write_file<arg_key>path</arg_ke'
  preserve_off agent_loop_mixed_open agp=False first_diff=3233 server='move them.<|assistant|><think>‖Done: a.py and b.py.</think><tool_call>write_file<' jinja2='move them.<|assistant|><think>‖</think><tool_call>write_file<arg_key>path</arg_ke'
  enable_thinking_false tools_ends_with_assistant agp=True first_diff=3469 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  enable_thinking_false inline_then_missing agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  enable_thinking_false inline_then_missing agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_low reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_low reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_low reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_low reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_low tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_low missing_after_two agp=True first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4<|assistant|><think>' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4<|assistant|><think>'
  effort_low missing_after_two agp=False first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4'
  effort_low reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_low reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_low reasoning_whitespace_then_missing agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_low reasoning_whitespace_then_missing agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_low inline_then_missing agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_low inline_then_missing agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_low tools_reasoning_empty agp=True first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_low tools_reasoning_empty agp=False first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_low agent_loop_mixed agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_low agent_loop_mixed agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_low agent_loop_mixed_open agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_low agent_loop_mixed_open agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_low agent_session_empty_reasoning agp=True first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_low agent_session_empty_reasoning agp=False first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_high reasoning_mixed agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_high reasoning_mixed agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_high reasoning_empty agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_high reasoning_empty agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_high tools_ends_with_assistant agp=True first_diff=3689 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_high missing_after_two agp=True first_diff=158 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4<|assistant|><think>' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4<|assistant|><think>'
  effort_high missing_after_two agp=False first_diff=158 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4'
  effort_high reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_high reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_high reasoning_whitespace_then_missing agp=True first_diff=117 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_high reasoning_whitespace_then_missing agp=False first_diff=117 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_high inline_then_missing agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_high inline_then_missing agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_high tools_reasoning_empty agp=True first_diff=1981 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_high tools_reasoning_empty agp=False first_diff=1981 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_high agent_loop_mixed agp=True first_diff=2465 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_high agent_loop_mixed agp=False first_diff=2465 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_high agent_loop_mixed_open agp=True first_diff=2465 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_high agent_loop_mixed_open agp=False first_diff=2465 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_high agent_session_empty_reasoning agp=True first_diff=3305 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_high agent_session_empty_reasoning agp=False first_diff=3305 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_medium reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_medium reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_medium reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_medium reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_medium tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_medium missing_after_two agp=True first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4<|assistant|><think>' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4<|assistant|><think>'
  effort_medium missing_after_two agp=False first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4'
  effort_medium reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_medium reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_medium reasoning_whitespace_then_missing agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_medium reasoning_whitespace_then_missing agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_medium inline_then_missing agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_medium inline_then_missing agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_medium tools_reasoning_empty agp=True first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_medium tools_reasoning_empty agp=False first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_medium agent_loop_mixed agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_medium agent_loop_mixed agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_medium agent_loop_mixed_open agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_medium agent_loop_mixed_open agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_medium agent_session_empty_reasoning agp=True first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_medium agent_session_empty_reasoning agp=False first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_none reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_none reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_none reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_none reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_none tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_none missing_after_two agp=True first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4<|assistant|><think>' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4<|assistant|><think>'
  effort_none missing_after_two agp=False first_diff=157 server='<|user|>u3<|assistant|><think>‖R2</think>a3<|user|>u4' jinja2='<|user|>u3<|assistant|><think>‖</think>a3<|user|>u4'
  effort_none reasoning_null agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_none reasoning_null agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "Failed to parse messages: [json.exception.type_error.302] type must be string, but is null", "type": "server_error"}} | jinja2_error=None
  effort_none reasoning_whitespace_then_missing agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_none reasoning_whitespace_then_missing agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖ \n </think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_none inline_then_missing agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_none inline_then_missing agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_none tools_reasoning_empty agp=True first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_none tools_reasoning_empty agp=False first_diff=1980 server='_response><|assistant|><think>‖R1</think><tool_call>read_file<arg_key>path</arg_k' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_none agent_loop_mixed agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_none agent_loop_mixed agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_none agent_loop_mixed_open agp=True first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_none agent_loop_mixed_open agp=False first_diff=2464 server='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_' jinja2='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key'
  effort_none agent_session_empty_reasoning agp=True first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_none agent_session_empty_reasoning agp=False first_diff=3304 server="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t" jinja2='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_'
  effort_none system_user agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none system_user agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none multi_turn agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none multi_turn agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none no_system agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none no_system agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_mixed agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none reasoning_mixed agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none inline_think agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none inline_think agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none reasoning_empty agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none reasoning_empty agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none tools agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none tools agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none tools_ends_with_assistant agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none tools_ends_with_assistant agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none agent_session agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none agent_session agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none missing_first agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none missing_first agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none missing_after_two agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none missing_after_two agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none reasoning_null agp=True equal_to_jinja2=False same_as_server_default=False
  effort_none reasoning_null agp=False equal_to_jinja2=False same_as_server_default=False
  effort_none reasoning_whitespace_then_missing agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none reasoning_whitespace_then_missing agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none inline_then_missing agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none inline_then_missing agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none inline_empty_then_missing agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none inline_empty_then_missing agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none tools_reasoning_empty agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none tools_reasoning_empty agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none tools_no_reasoning_at_all agp=True equal_to_jinja2=True same_as_server_default=True
  effort_none tools_no_reasoning_at_all agp=False equal_to_jinja2=True same_as_server_default=True
  effort_none agent_loop_mixed agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none agent_loop_mixed agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none agent_loop_mixed_open agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none agent_loop_mixed_open agp=False equal_to_jinja2=False same_as_server_default=True
  effort_none agent_session_empty_reasoning agp=True equal_to_jinja2=False same_as_server_default=True
  effort_none agent_session_empty_reasoning agp=False equal_to_jinja2=False same_as_server_default=True
  server_flag_reasoning_off reasoning agp=True equal_to_jinja2=True same_as_server_default=True
  server_flag_reasoning_off agent_session agp=True equal_to_jinja2=True same_as_server_default=True
  log (default): 0.01.699.711 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
  log (server_flag_reasoning_off): 0.00.382.719 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
prefix glm_n23 log: 0.00.446.651 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
prefix glm_n23 switch preserve_off: {"shared_bytes": 1936, "prompt_bytes": 4068, "at": "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"}
prefix glm_n23 switch enable_thinking_false: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm_n23 switch effort_low: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖Low<|system|>\n# Tools\n\nYou may call one or more fu"}
prefix glm_n23 switch effort_high: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖High<|system|>\n# Tools\n\nYou may call one or more f"}
prefix glm_n23 switch effort_medium: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm_n23 switch effort_none: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm_n23 switch one_tool_added: {"shared_bytes": 1501, "prompt_bytes": 4068, "at": "}}, \"required\": [\"verbose\"]}}\n‖{\"name\": \"web_search\", \"description\": \"Search.\", \""}
prefix glm_n23 default/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 default/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 preserve_off/jinja2: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 1936 of 3704 held, in message 2 (assistant) old='nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs' new='nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke'
prefix glm_n23 preserve_off/server: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 1936 of 3704 held, in message 2 (assistant) old='nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs' new='nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke'
prefix glm_n23 preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 1936 of 3704 held, in message 2 (assistant) old='nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs' new='nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke'
prefix glm_n23 enable_thinking_false/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 enable_thinking_false/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 empty_reasoning_turn/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 empty_reasoning_turn/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm_n23 agent_loop_mixed/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 8 requests
prefix glm_n23 agent_loop_mixed/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 8 requests
prefix glm_n23 agent_loop_mixed/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 8 requests
prefix glm_n23 agent_loop_mixed_preserve_off/jinja2: history breaks 2 (k=[12, 18]), refused prefixes 0; loop breaks 1 of 8 requests
  request 4 (k=11): first_diff byte 1921 of 3270 held, in message 2 (assistant) old='and tests.<|assistant|><think>‖Search both trees at once.</think><tool_call>grep<' new='and tests.<|assistant|><think>‖</think><tool_call>grep<arg_key>pattern</arg_key><'
prefix glm_n23 agent_loop_mixed_preserve_off/server: history breaks 2 (k=[12, 18]), refused prefixes 0; loop breaks 1 of 8 requests
  request 4 (k=11): first_diff byte 1921 of 3270 held, in message 2 (assistant) old='and tests.<|assistant|><think>‖Search both trees at once.</think><tool_call>grep<' new='and tests.<|assistant|><think>‖</think><tool_call>grep<arg_key>pattern</arg_key><'
prefix glm_n23 agent_loop_mixed_preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 8 requests
  request 4 (k=11): first_diff byte 1921 of 3270 held, in message 2 (assistant) old='and tests.<|assistant|><think>‖Search both trees at once.</think><tool_call>grep<' new='and tests.<|assistant|><think>‖</think><tool_call>grep<arg_key>pattern</arg_key><'
prefix glm log: 0.00.276.400 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
prefix glm switch preserve_off: {"shared_bytes": 1936, "prompt_bytes": 4068, "at": "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"}
prefix glm switch enable_thinking_false: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm switch effort_low: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖Low<|system|>\n# Tools\n\nYou may call one or more fu"}
prefix glm switch effort_high: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖High<|system|>\n# Tools\n\nYou may call one or more f"}
prefix glm switch effort_medium: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm switch effort_none: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm switch one_tool_added: {"shared_bytes": 1501, "prompt_bytes": 4068, "at": "}}, \"required\": [\"verbose\"]}}\n‖{\"name\": \"web_search\", \"description\": \"Search.\", \""}
prefix glm default/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm default/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm preserve_off/jinja2: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 1936 of 3704 held, in message 2 (assistant) old='nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs' new='nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke'
prefix glm preserve_off/server: history breaks 2 (k=[12, 16]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 1936 of 3704 held, in message 2 (assistant) old='nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs' new='nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke'
prefix glm preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 1936 of 3704 held, in message 2 (assistant) old='nd fix it.<|assistant|><think>‖I should look at the layout of the repository firs' new='nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke'
prefix glm enable_thinking_false/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm enable_thinking_false/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm empty_reasoning_turn/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm empty_reasoning_turn/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix glm empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
  request 3 (k=9): first_diff byte 3320 of 3517 held, in message 9 (assistant) old='_response><|assistant|><think>‖</think><tool_call>run_tests<arg_key>verbose</arg_' new="_response><|assistant|><think>‖parse() splits on ';' but the test uses ','. Fix t"
prefix glm agent_loop_mixed/jinja2: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 8 requests
prefix glm agent_loop_mixed/server: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 8 requests
prefix glm agent_loop_mixed/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 3 of 8 requests
  request 1 (k=5): first_diff byte 2471 of 2624 held, in message 5 (assistant) old='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key' new='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_'
  request 2 (k=7): first_diff byte 2737 of 2862 held, in message 7 (assistant) old='_response><|assistant|><think>‖</think>Checking the README too.<tool_call>read_fi' new='_response><|assistant|><think>‖Search both trees at once.</think>Checking the REA'
  request 5 (k=13): first_diff byte 3362 of 3574 held, in message 13 (assistant) old='move them.<|assistant|><think>‖</think><tool_call>write_file<arg_key>path</arg_ke' new='move them.<|assistant|><think>‖Done: a.py and b.py.</think><tool_call>write_file<'
prefix glm agent_loop_mixed_preserve_off/jinja2: history breaks 2 (k=[12, 18]), refused prefixes 0; loop breaks 1 of 8 requests
  request 4 (k=11): first_diff byte 1921 of 3270 held, in message 2 (assistant) old='and tests.<|assistant|><think>‖Search both trees at once.</think><tool_call>grep<' new='and tests.<|assistant|><think>‖</think><tool_call>grep<arg_key>pattern</arg_key><'
prefix glm agent_loop_mixed_preserve_off/server: history breaks 2 (k=[12, 18]), refused prefixes 0; loop breaks 1 of 8 requests
  request 4 (k=11): first_diff byte 1921 of 3322 held, in message 2 (assistant) old='and tests.<|assistant|><think>‖Search both trees at once.</think><tool_call>grep<' new='and tests.<|assistant|><think>‖</think><tool_call>grep<arg_key>pattern</arg_key><'
prefix glm agent_loop_mixed_preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 4 of 8 requests
  request 1 (k=5): first_diff byte 2471 of 2624 held, in message 5 (assistant) old='_response><|assistant|><think>‖</think><tool_call>read_file<arg_key>path</arg_key' new='_response><|assistant|><think>‖Search both trees at once.</think><tool_call>read_'
  request 2 (k=7): first_diff byte 2737 of 2862 held, in message 7 (assistant) old='_response><|assistant|><think>‖</think>Checking the README too.<tool_call>read_fi' new='_response><|assistant|><think>‖Search both trees at once.</think>Checking the REA'
  request 4 (k=11): first_diff byte 1921 of 3322 held, in message 2 (assistant) old='and tests.<|assistant|><think>‖Search both trees at once.</think><tool_call>grep<' new='and tests.<|assistant|><think>‖</think><tool_call>grep<arg_key>pattern</arg_key><'
  request 5 (k=13): first_diff byte 3245 of 3457 held, in message 13 (assistant) old='move them.<|assistant|><think>‖</think><tool_call>write_file<arg_key>path</arg_ke' new='move them.<|assistant|><think>‖Done: a.py and b.py.</think><tool_call>write_file<'
parse glm_n23: format ['peg-native']; notools peg-native; nothink gen prompt '<|assistant|><think>'; handler ['\x1b[0mcommon_chat_templates_apply_jinja: using differential autoparser', 'common_chat_templates_apply_jinja: using differential autoparser']
  generation_prompt '<|assistant|><think>' thinking '<think>' ['</think>'] delimiters [('assistant', '<|assistant|>'), ('user', '<|user|>')]
  k=2 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=2 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": true}
  k=2 unknown_tool: {"parse_error": "The model produced output that does not match the expected peg-native format"}
  k=4 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=4 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": true}
  k=4 content_padded: {"fields_equal": {"content": false, "reasoning_content": true, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 2225, "rendered": "where `parse` is used.</think>‖Reading the tests and searching for the parser.<to", "generated": "where `parse` is used.</think>‖\nReading the tests and searching for the parser.\n<"}
  k=7 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=7 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": true}
  k=7 content_padded: {"fields_equal": {"content": false, "reasoning_content": true, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 2884, "rendered": "'. Fix the separator.\n</think>‖The separator is wrong; fixing it.<tool_call>write", "generated": "'. Fix the separator.\n</think>‖\nThe separator is wrong; fixing it.\n<tool_call>wri"}
  k=9 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": false}, "round_trip_equal": true}
  k=9 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": false}, "round_trip_reproduces_output": true}
  k=11 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=11 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": true}
  k=11 content_padded: {"fields_equal": {"content": false, "reasoning_content": true, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 3638, "rendered": ">All green; summarise.</think>‖Fixed: `parse` now splits on ','. The test passes.", "generated": ">All green; summarise.</think>‖\nFixed: `parse` now splits on ','. The test passes"}
  k=13 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=13 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": true}
  k=15 ideal: {"fields_equal": {"content": true, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=15 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": true}
  k=15 content_padded: {"fields_equal": {"content": false, "reasoning_content": true, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 4099, "rendered": "case and add the test.</think>‖Done: `parse('')` returns `[]`, test added.<|user|", "generated": "case and add the test.</think>‖\nDone: `parse('')` returns `[]`, test added.\n"}
live glm_n23 default (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 222 B predicted_n 223 finish stop 
  request 4 k=11 status 200 prompt 3625 B | held 3542 B shared 3542 B extends True | cache_n 3542 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3775 B | held 3704 B shared 3704 B extends True | cache_n 3704 prompt_n 71 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 4068 B | held 3946 B shared 3946 B extends True | cache_n 3946 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3541, pos_max = 3541, n_tokens = 3542, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3620, pos_max = 3620, n_tokens = 3621, size = 0.055 MiB)
    | created context checkpoint 12 of 32 (pos_min = 3703, pos_max = 3703, n_tokens = 3704, size = 0.055 MiB)
    | created context checkpoint 13 of 32 (pos_min = 3770, pos_max = 3770, n_tokens = 3771, size = 0.055 MiB)
    | created context checkpoint 14 of 32 (pos_min = 3945, pos_max = 3945, n_tokens = 3946, size = 0.055 MiB)
    | created context checkpoint 15 of 32 (pos_min = 4063, pos_max = 4063, n_tokens = 4064, size = 0.055 MiB)
live glm_n23 preserve_off (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 222 B predicted_n 223 finish stop 
  request 4 k=11 status 200 prompt 3625 B | held 3542 B shared 3542 B extends True | cache_n 3542 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3556 B | held 3704 B shared 1936 B extends False | cache_n 1932 prompt_n 1624 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 3849 B | held 3727 B shared 3727 B extends True | cache_n 3727 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3541, pos_max = 3541, n_tokens = 3542, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3620, pos_max = 3620, n_tokens = 3621, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 1931, pos_max = 1931, n_tokens = 1932, n_past = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2079, pos_max = 2079, n_tokens = 2080, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2162, pos_max = 2162, n_tokens = 2163, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2641, pos_max = 2641, n_tokens = 2642, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2814, pos_max = 2814, n_tokens = 2815, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3234, pos_max = 3234, n_tokens = 3235, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3315, pos_max = 3315, n_tokens = 3316, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3541, pos_max = 3541, n_tokens = 3542, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3620, pos_max = 3620, n_tokens = 3621, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 3039, pos_max = 3039, n_tokens = 3040, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 3484, pos_max = 3484, n_tokens = 3485, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 3551, pos_max = 3551, n_tokens = 3552, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 3726, pos_max = 3726, n_tokens = 3727, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3844, pos_max = 3844, n_tokens = 3845, size = 0.055 MiB)
live glm_n23 preserve_off-ctx-checkpoints-4 (-c 16384 -lv 4 --ctx-checkpoints 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 222 B predicted_n 223 finish stop 
  request 4 k=11 status 200 prompt 3625 B | held 3542 B shared 3542 B extends True | cache_n 3542 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3556 B | held 3704 B shared 1936 B extends False | cache_n 1420 prompt_n 2136 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 3849 B | held 3727 B shared 3727 B extends True | cache_n 3727 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 4 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 4 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3541, pos_max = 3541, n_tokens = 3542, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3620, pos_max = 3620, n_tokens = 3621, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 1419, pos_max = 1419, n_tokens = 1420, n_past = 1420, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3541, pos_max = 3541, n_tokens = 3542, n_swa = 0, pos_next = 1420, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3620, pos_max = 3620, n_tokens = 3621, n_swa = 0, pos_next = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 4 (pos_min = 3039, pos_max = 3039, n_tokens = 3040, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3484, pos_max = 3484, n_tokens = 3485, size = 0.055 MiB)
    | created context checkpoint 4 of 4 (pos_min = 3551, pos_max = 3551, n_tokens = 3552, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3039, pos_max = 3039, n_tokens = 3040, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3484, pos_max = 3484, n_tokens = 3485, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3726, pos_max = 3726, n_tokens = 3727, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3551, pos_max = 3551, n_tokens = 3552, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3844, pos_max = 3844, n_tokens = 3845, size = 0.055 MiB)
live glm_n23 empty_reasoning_turn (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 197 B predicted_n 198 finish stop 
  request 4 k=11 status 200 prompt 3600 B | held 3517 B shared 3517 B extends True | cache_n 3517 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3750 B | held 3679 B shared 3679 B extends True | cache_n 3679 prompt_n 71 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 4043 B | held 3921 B shared 3921 B extends True | cache_n 3921 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3516, pos_max = 3516, n_tokens = 3517, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3595, pos_max = 3595, n_tokens = 3596, size = 0.055 MiB)
    | created context checkpoint 12 of 32 (pos_min = 3678, pos_max = 3678, n_tokens = 3679, size = 0.055 MiB)
    | created context checkpoint 13 of 32 (pos_min = 3745, pos_max = 3745, n_tokens = 3746, size = 0.055 MiB)
    | created context checkpoint 14 of 32 (pos_min = 3920, pos_max = 3920, n_tokens = 3921, size = 0.055 MiB)
    | created context checkpoint 15 of 32 (pos_min = 4038, pos_max = 4038, n_tokens = 4039, size = 0.055 MiB)
live glm_n23 agent_loop_mixed (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1921 B | held None B shared None B extends None | cache_n 0 prompt_n 1921 | forced 418 B predicted_n 419 finish stop 
  request 1 k=5 status 200 prompt 2471 B | held 2339 B shared 2339 B extends True | cache_n 2339 prompt_n 132 | forced 153 B predicted_n 154 finish stop 
  request 2 k=7 status 200 prompt 2711 B | held 2624 B shared 2624 B extends True | cache_n 2624 prompt_n 87 | forced 125 B predicted_n 126 finish stop 
  request 3 k=9 status 200 prompt 2928 B | held 2836 B shared 2836 B extends True | cache_n 2836 prompt_n 92 | forced 215 B predicted_n 216 finish stop 
  request 4 k=11 status 200 prompt 3217 B | held 3143 B shared 3143 B extends True | cache_n 3143 prompt_n 74 | forced 53 B predicted_n 54 finish stop 
  request 5 k=13 status 200 prompt 3310 B | held 3270 B shared 3270 B extends True | cache_n 3270 prompt_n 40 | forced 212 B predicted_n 213 finish stop 
  request 6 k=15 status 200 prompt 3590 B | held 3522 B shared 3522 B extends True | cache_n 3522 prompt_n 68 | forced 159 B predicted_n 160 finish stop 
  request 7 k=17 status 200 prompt 3817 B | held 3749 B shared 3749 B extends True | cache_n 3749 prompt_n 68 | forced 21 B predicted_n 22 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1404, pos_max = 1404, n_tokens = 1405, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1916, pos_max = 1916, n_tokens = 1917, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2338, pos_max = 2338, n_tokens = 2339, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2466, pos_max = 2466, n_tokens = 2467, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2623, pos_max = 2623, n_tokens = 2624, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2706, pos_max = 2706, n_tokens = 2707, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 2835, pos_max = 2835, n_tokens = 2836, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 2923, pos_max = 2923, n_tokens = 2924, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3142, pos_max = 3142, n_tokens = 3143, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3212, pos_max = 3212, n_tokens = 3213, size = 0.055 MiB)
    | created context checkpoint 12 of 32 (pos_min = 3269, pos_max = 3269, n_tokens = 3270, size = 0.055 MiB)
    | created context checkpoint 13 of 32 (pos_min = 3305, pos_max = 3305, n_tokens = 3306, size = 0.055 MiB)
    | created context checkpoint 14 of 32 (pos_min = 3521, pos_max = 3521, n_tokens = 3522, size = 0.055 MiB)
    | created context checkpoint 15 of 32 (pos_min = 3585, pos_max = 3585, n_tokens = 3586, size = 0.055 MiB)
    | created context checkpoint 16 of 32 (pos_min = 3748, pos_max = 3748, n_tokens = 3749, size = 0.055 MiB)
    | created context checkpoint 17 of 32 (pos_min = 3812, pos_max = 3812, n_tokens = 3813, size = 0.055 MiB)
live glm default (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 222 B predicted_n 223 finish stop 
  request 4 k=11 status 200 prompt 3625 B | held 3542 B shared 3542 B extends True | cache_n 3542 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3775 B | held 3704 B shared 3704 B extends True | cache_n 3704 prompt_n 71 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 4068 B | held 3946 B shared 3946 B extends True | cache_n 3946 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3541, pos_max = 3541, n_tokens = 3542, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3620, pos_max = 3620, n_tokens = 3621, size = 0.055 MiB)
    | created context checkpoint 12 of 32 (pos_min = 3703, pos_max = 3703, n_tokens = 3704, size = 0.055 MiB)
    | created context checkpoint 13 of 32 (pos_min = 3770, pos_max = 3770, n_tokens = 3771, size = 0.055 MiB)
    | created context checkpoint 14 of 32 (pos_min = 3945, pos_max = 3945, n_tokens = 3946, size = 0.055 MiB)
    | created context checkpoint 15 of 32 (pos_min = 4063, pos_max = 4063, n_tokens = 4064, size = 0.055 MiB)
live glm preserve_off (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 222 B predicted_n 223 finish stop 
  request 4 k=11 status 200 prompt 3625 B | held 3542 B shared 3542 B extends True | cache_n 3542 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3556 B | held 3704 B shared 1936 B extends False | cache_n 1932 prompt_n 1624 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 3849 B | held 3727 B shared 3727 B extends True | cache_n 3727 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3541, pos_max = 3541, n_tokens = 3542, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3620, pos_max = 3620, n_tokens = 3621, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 1931, pos_max = 1931, n_tokens = 1932, n_past = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2079, pos_max = 2079, n_tokens = 2080, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2162, pos_max = 2162, n_tokens = 2163, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2641, pos_max = 2641, n_tokens = 2642, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 2814, pos_max = 2814, n_tokens = 2815, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3234, pos_max = 3234, n_tokens = 3235, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3315, pos_max = 3315, n_tokens = 3316, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3541, pos_max = 3541, n_tokens = 3542, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3620, pos_max = 3620, n_tokens = 3621, n_swa = 0, pos_next = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 3039, pos_max = 3039, n_tokens = 3040, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 3484, pos_max = 3484, n_tokens = 3485, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 3551, pos_max = 3551, n_tokens = 3552, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 3726, pos_max = 3726, n_tokens = 3727, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3844, pos_max = 3844, n_tokens = 3845, size = 0.055 MiB)
live glm preserve_off-ctx-checkpoints-4 (-c 16384 -lv 4 --ctx-checkpoints 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 222 B predicted_n 223 finish stop 
  request 4 k=11 status 200 prompt 3625 B | held 3542 B shared 3542 B extends True | cache_n 3542 prompt_n 83 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3556 B | held 3704 B shared 1936 B extends False | cache_n 1420 prompt_n 2136 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 3849 B | held 3727 B shared 3727 B extends True | cache_n 3727 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 4 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 4 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3541, pos_max = 3541, n_tokens = 3542, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3620, pos_max = 3620, n_tokens = 3621, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 1419, pos_max = 1419, n_tokens = 1420, n_past = 1420, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3541, pos_max = 3541, n_tokens = 3542, n_swa = 0, pos_next = 1420, size = 0.055 MiB)
    | erased invalidated context checkpoint (pos_min = 3620, pos_max = 3620, n_tokens = 3621, n_swa = 0, pos_next = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 4 (pos_min = 3039, pos_max = 3039, n_tokens = 3040, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3484, pos_max = 3484, n_tokens = 3485, size = 0.055 MiB)
    | created context checkpoint 4 of 4 (pos_min = 3551, pos_max = 3551, n_tokens = 3552, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3039, pos_max = 3039, n_tokens = 3040, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3484, pos_max = 3484, n_tokens = 3485, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3726, pos_max = 3726, n_tokens = 3727, size = 0.055 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3551, pos_max = 3551, n_tokens = 3552, size = 0.055 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3844, pos_max = 3844, n_tokens = 3845, size = 0.055 MiB)
live glm empty_reasoning_turn (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1936 B | held None B shared None B extends None | cache_n 0 prompt_n 1936 | forced 144 B predicted_n 145 finish stop 
  request 1 k=4 status 200 prompt 2167 B | held 2080 B shared 2080 B extends True | cache_n 2080 prompt_n 87 | forced 475 B predicted_n 476 finish stop 
  request 2 k=7 status 200 prompt 2819 B | held 2642 B shared 2642 B extends True | cache_n 2642 prompt_n 177 | forced 416 B predicted_n 417 finish stop 
  request 3 k=9 status 200 prompt 3320 B | held 3235 B shared 3235 B extends True | cache_n 3235 prompt_n 85 | forced 197 B predicted_n 198 finish stop 
  request 4 k=11 status 200 prompt 3664 B | held 3517 B shared 3320 B extends False | cache_n 3316 prompt_n 348 | forced 79 B predicted_n 80 finish stop 
  request 5 k=13 status 200 prompt 3814 B | held 3743 B shared 3743 B extends True | cache_n 3743 prompt_n 71 | forced 171 B predicted_n 172 finish stop 
  request 6 k=15 status 200 prompt 4107 B | held 3985 B shared 3985 B extends True | cache_n 3985 prompt_n 122 | forced 90 B predicted_n 91 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1419, pos_max = 1419, n_tokens = 1420, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1931, pos_max = 1931, n_tokens = 1932, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2079, pos_max = 2079, n_tokens = 2080, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2162, pos_max = 2162, n_tokens = 2163, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2641, pos_max = 2641, n_tokens = 2642, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2814, pos_max = 2814, n_tokens = 2815, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3234, pos_max = 3234, n_tokens = 3235, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 3315, pos_max = 3315, n_tokens = 3316, n_past = 3316, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3315, pos_max = 3315, n_tokens = 3316, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3659, pos_max = 3659, n_tokens = 3660, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3742, pos_max = 3742, n_tokens = 3743, size = 0.055 MiB)
    | created context checkpoint 12 of 32 (pos_min = 3809, pos_max = 3809, n_tokens = 3810, size = 0.055 MiB)
    | created context checkpoint 13 of 32 (pos_min = 3984, pos_max = 3984, n_tokens = 3985, size = 0.055 MiB)
    | created context checkpoint 14 of 32 (pos_min = 4102, pos_max = 4102, n_tokens = 4103, size = 0.055 MiB)
live glm agent_loop_mixed (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 1921 B | held None B shared None B extends None | cache_n 0 prompt_n 1921 | forced 418 B predicted_n 419 finish stop 
  request 1 k=5 status 200 prompt 2471 B | held 2339 B shared 2339 B extends True | cache_n 2339 prompt_n 132 | forced 153 B predicted_n 154 finish stop 
  request 2 k=7 status 200 prompt 2737 B | held 2624 B shared 2471 B extends False | cache_n 2467 prompt_n 270 | forced 125 B predicted_n 126 finish stop 
  request 3 k=9 status 200 prompt 2980 B | held 2862 B shared 2737 B extends False | cache_n 2733 prompt_n 247 | forced 215 B predicted_n 216 finish stop 
  request 4 k=11 status 200 prompt 3269 B | held 3195 B shared 3195 B extends True | cache_n 3195 prompt_n 74 | forced 53 B predicted_n 54 finish stop 
  request 5 k=13 status 200 prompt 3362 B | held 3322 B shared 3322 B extends True | cache_n 3322 prompt_n 40 | forced 212 B predicted_n 213 finish stop 
  request 6 k=15 status 200 prompt 3662 B | held 3574 B shared 3362 B extends False | cache_n 3358 prompt_n 304 | forced 159 B predicted_n 160 finish stop 
  request 7 k=17 status 200 prompt 3889 B | held 3821 B shared 3821 B extends True | cache_n 3821 prompt_n 68 | forced 21 B predicted_n 22 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 1404, pos_max = 1404, n_tokens = 1405, size = 0.055 MiB)
    | created context checkpoint 2 of 32 (pos_min = 1852, pos_max = 1852, n_tokens = 1853, size = 0.055 MiB)
    | created context checkpoint 3 of 32 (pos_min = 1916, pos_max = 1916, n_tokens = 1917, size = 0.055 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2338, pos_max = 2338, n_tokens = 2339, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2466, pos_max = 2466, n_tokens = 2467, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 2466, pos_max = 2466, n_tokens = 2467, n_past = 2467, size = 0.055 MiB)
    | created context checkpoint 5 of 32 (pos_min = 2466, pos_max = 2466, n_tokens = 2467, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2732, pos_max = 2732, n_tokens = 2733, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 2732, pos_max = 2732, n_tokens = 2733, n_past = 2733, size = 0.055 MiB)
    | created context checkpoint 6 of 32 (pos_min = 2732, pos_max = 2732, n_tokens = 2733, size = 0.055 MiB)
    | created context checkpoint 7 of 32 (pos_min = 2975, pos_max = 2975, n_tokens = 2976, size = 0.055 MiB)
    | created context checkpoint 8 of 32 (pos_min = 3194, pos_max = 3194, n_tokens = 3195, size = 0.055 MiB)
    | created context checkpoint 9 of 32 (pos_min = 3264, pos_max = 3264, n_tokens = 3265, size = 0.055 MiB)
    | created context checkpoint 10 of 32 (pos_min = 3321, pos_max = 3321, n_tokens = 3322, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3357, pos_max = 3357, n_tokens = 3358, size = 0.055 MiB)
    | restored context checkpoint (pos_min = 3357, pos_max = 3357, n_tokens = 3358, n_past = 3358, size = 0.055 MiB)
    | created context checkpoint 11 of 32 (pos_min = 3357, pos_max = 3357, n_tokens = 3358, size = 0.055 MiB)
    | created context checkpoint 12 of 32 (pos_min = 3657, pos_max = 3657, n_tokens = 3658, size = 0.055 MiB)
    | created context checkpoint 13 of 32 (pos_min = 3820, pos_max = 3820, n_tokens = 3821, size = 0.055 MiB)
    | created context checkpoint 14 of 32 (pos_min = 3884, pos_max = 3884, n_tokens = 3885, size = 0.055 MiB)
== tests/test_glm_template_n23.py tests/test_chat_templates.py
test_header_names_source_and_change (tests.test_glm_template_n23.CorrectedFile.test_header_names_source_and_change) ... ok
test_only_the_two_lines_differ (tests.test_glm_template_n23.CorrectedFile.test_only_the_two_lines_differ) ... ok
test_source_is_the_pinned_template (tests.test_glm_template_n23.CorrectedFile.test_source_is_the_pinned_template) ... ok
test_agent_loops_prefix_stable (tests.test_glm_template_n23.JinjaEquality.test_agent_loops_prefix_stable) ... ok
test_byte_equal_everywhere (tests.test_glm_template_n23.JinjaEquality.test_byte_equal_everywhere) ... ok
test_turn_without_reasoning_renders_an_empty_think_block (tests.test_glm_template_n23.JinjaEquality.test_turn_without_reasoning_renders_an_empty_think_block) ... ok
test_prompt_cache_reuse (tests.test_glm_template_n23.LiveB3N23.test_prompt_cache_reuse) ... ok
test_parse_like_the_official_template (tests.test_glm_template_n23.ParseHarnessN23.test_parse_like_the_official_template) ... ok
test_server_prompts_extend_held_turns (tests.test_glm_template_n23.PrefixN23.test_server_prompts_extend_held_turns) ... ok
test_official_template_still_leaks_at_this_pin (tests.test_glm_template_n23.ServerRenderN23.test_official_template_still_leaks_at_this_pin) ... ok
test_server_with_ours_equals_official_format (tests.test_glm_template_n23.ServerRenderN23.test_server_with_ours_equals_official_format) ... ok
test_transformers_renders_like_hf_render (tests.test_glm_template_n23.TransformersRender.test_transformers_renders_like_hf_render) ... skipped 'transformers is not installed'
test_prompt_cache_reuse (tests.test_chat_templates.LiveB3.test_prompt_cache_reuse) ... ok
test_parse_and_round_trip (tests.test_chat_templates.ParseHarness.test_parse_and_round_trip) ... ok
test_render_server_vs_jinja2 (tests.test_chat_templates.ServerRender.test_render_server_vs_jinja2) ... ok
test_defaults_prefix_stable_in_agent_loop (tests.test_chat_templates.TemplateProperties.test_defaults_prefix_stable_in_agent_loop) ... ok
test_dropping_past_reasoning_breaks_at_new_user_message (tests.test_chat_templates.TemplateProperties.test_dropping_past_reasoning_breaks_at_new_user_message) ... ok
test_glm_ignores_enable_thinking (tests.test_chat_templates.TemplateProperties.test_glm_ignores_enable_thinking) ... ok
test_glm_loop_scope_jinja2 (tests.test_chat_templates.TemplateProperties.test_glm_loop_scope_jinja2) ... ok
test_history_breaks_default (tests.test_chat_templates.TemplateProperties.test_history_breaks_default) ... ok
test_no_date_injection (tests.test_chat_templates.TemplateProperties.test_no_date_injection) ... ok
test_qwen_enable_thinking_false (tests.test_chat_templates.TemplateProperties.test_qwen_enable_thinking_false) ... ok
test_reasoning_effort_values (tests.test_chat_templates.TemplateProperties.test_reasoning_effort_values) ... ok
test_glm_config_facts (tests.test_chat_templates.VendoredFiles.test_glm_config_facts) ... ok
test_qwen_config_facts (tests.test_chat_templates.VendoredFiles.test_qwen_config_facts) ... ok
test_sha256sums (tests.test_chat_templates.VendoredFiles.test_sha256sums) ... ok

----------------------------------------------------------------------
Ran 26 tests in 621.680s

OK (skipped=1)

real	15m20.225s
user	10m47.542s
sys	0m14.233s
driver rc=0
```
