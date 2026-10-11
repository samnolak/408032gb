# N14: chat templates, tool calls and prefix stability of GLM-5.3-Flash and Qwen3.8-Flash-Next through llama-server @79e2e74 (gap 6)

Date 2026-10-11. Roles: parity-qa (renderer, analysis, tests, evidence), engine-integrator (parse harness against
llama.cpp `common`). Branch `claude/n14-chat-templates`, cut from `claude/n13-ci-no-silent-skips`; the vendored HF files
arrived in `12662b1`. Hardware-free: the CPU llama-server of the R1 pin in the sandbox (4 cores), tiny random-weight
models. **The random weights say nothing about either real model**: they only make llama-server load a hybrid model, so
that its template engine, chat parser and prompt cache / checkpoints run for real. `<S>` stands for the session scratch
directory `/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with
only that substitution. No GPU, no performance claims.

## Result

1. **Rendering, CONFIRMED (section 3).** llama-server's own Jinja engine (`--jinja --chat-template-file`, `POST
   /apply-template`) renders both vendored templates byte for byte like jinja2 configured as transformers 5.16.0 does,
   for system + user, multi-turn, tool calls (parallel, nested objects, numbers, booleans, null, non-ASCII), tool results,
   a `tools` list, `reasoning_content`, `add_generation_prompt` true and false, and every switch the templates read
   (switches on the reasoning, mixed-reasoning and agent-session conversations) —
   with three exceptions, all reproduced by the tests:
   (a) **GLM, a llama.cpp engine bug:** a `{% set %}` inside a `{% for %}` body survives into the next iteration in
   llama.cpp's engine (`common/jinja/runtime.cpp:485,601-618@79e2e74`: one `scope` for all iterations); the GLM template
   sets `reasoning_content` only when the message has one (template lines 143-148), so an assistant turn **without**
   reasoning — or with an **empty** one, which llama-server drops from the message (`common/chat.cpp:202-204`) — is
   rendered with the **previous turn's reasoning** (`<think>R1</think>a2` instead of `<think></think>a2`). jinja2 scopes
   it per iteration. The Qwen template resets the variable each turn (line 111) and is not affected;
   (b) by design: a conversation that ends with an assistant message is continued, not closed (`--prefill-assistant`,
   default on, `tools/server/server-common.cpp` `oaicompat_chat_params_parse`); in an agent loop the last message is a
   user or tool message, so this does not arise;
   (c) Qwen with `reasoning_effort: "high"` fails in both engines (`raise_exception`, template line 49); llama-server
   answers HTTP 500. The only warning at load is `chat template supports preserving reasoning, it is enabled by default
   (may use more tokens, disable via --no-reasoning-preserve)`; no template parse error or warning otherwise.
2. **B3 with the server defaults: both templates are prefix-stable over a whole agent session, and llama-server reuses
   everything, CONFIRMED (sections 4, 6).** llama-server @79e2e74 sends `preserve_reasoning=true` by default
   (`common/arg.cpp:964-966`), which sets `preserve_thinking=true`, `clear_thinking=false` (`common/jinja/caps.cpp:22-27`)
   — the templates' own defaults (GLM line 4, Qwen line 116). Live, every request after the first reuses exactly what the
   slot holds (prompt + forced turn): `cache_n` = held bytes at all 6 boundaries for each template.
3. **B3 with past reasoning dropped (`--no-reasoning-preserve`, or `chat_template_kwargs: {"preserve_reasoning": false}`
   per request; also the templates' `clear_thinking=true` / `preserve_thinking=false`), CONFIRMED (sections 4, 6):** at
   every new user message the template re-renders all assistant turns before it without reasoning (GLM lines 149-152:
   `<think></think>`; Qwen lines 116-120: no think block at all). The first differing byte is the reasoning of the first
   assistant turn after the **previous** user message. llama-server then restores the newest context checkpoint at or
   before that byte — **not a full re-prefill** in these runs: GLM restored `n_tokens = 1932` (= the first prompt of the
   previous round minus 4; first diff at byte 1936) and re-processed 1624 of 3556 tokens; Qwen restored `n_tokens = 2649`
   (the start of the previous user message; the diff at byte 2756 lies 6 bytes before the end of that prompt, after the
   `-4` checkpoint) and re-processed 1790 of 4439. With `--ctx-checkpoints 4` the nearer checkpoints were evicted and the
   server fell back to `prompt − 4 − n_ubatch` (1420 / 2246). Cost: the whole previous round (every assistant turn, tool
   result and its reasoning since the previous user message) is processed again at each new user message.
4. **Full re-prefill, CONFIRMED live (section 5):** changing `reasoning_effort` between two requests of one session
   (both templates put it at the very start: GLM line 3, Qwen lines 46-56, 80-85; shared prefix 40 / 46 bytes) and, on
   Qwen, adding a tool (the tools are in the first system block; shared 1748 bytes) gave `cache_n 0` and `forcing full
   prompt re-processing due to lack of cache data`. Toggling `enable_thinking` on Qwen changes byte 19 (PROVISIONAL that it
   forces a full re-prefill: same mechanism, not run live). GLM ignores `enable_thinking` entirely (finding 9).
5. **The GLM leak of finding 1(a) breaks B3 too, CONFIRMED live:** a turn the model wrote with an empty think block
   (`<think></think>`) is sent back without `reasoning_content`; the next prompt renders the previous turn's reasoning
   there; reuse stops at that turn (`shared 3320 of 3517` bytes, restored the `-4` checkpoint, 348 tokens re-processed)
   and **the model sees a wrong history** from then on (another turn's reasoning duplicated). Qwen: no break.
6. **Tool-call parsing, CONFIRMED with a harness against `common` @79e2e74 (section 6, `tools/chattmpl/chat_parse_harness.cpp`):**
   Qwen3.8 → specialized `Qwen3-Coder` handler (`common/chat.cpp:1218-1227`, `common/parsers/qwen3-coder.cpp`); GLM-5.3
   → no specialized handler, the differential autoparser (`common/chat.cpp:1381-1399`); both `peg-native`, both with a
   lazy tool-call grammar. Every assistant turn of the session, written exactly in the template's format, is parsed back
   into `reasoning_content`, `content` and structured `tool_calls` (both parallel calls of one turn included), and the
   parsed message re-renders to the same bytes (7/7 per template). Field-level exceptions: GLM returns a JSON `null`
   argument as the string `"null"` (schema type `["string","null"]`; the bytes round-trip, the tool gets a string); Qwen
   keeps a trailing `\n` in `reasoning_content` and `\n\n` at the end of `content` before a tool call (the template trims
   both, so the bytes round-trip). A call to a tool that is not in the list: GLM → parse error ("The model produced output
   that does not match the expected peg-native format"), Qwen → the call silently disappears (empty content, no
   `tool_calls`); with the lazy grammar active the model cannot emit such a name (PROVISIONAL, from the grammar triggers,
   not run with sampling).
7. **Whitespace a real model might add, CONFIRMED on the parser + server (section 6):** GLM keeps whitespace inside
   `<think>…</think>` and re-renders it verbatim; Qwen trims reasoning (line 115); both strip content (GLM 154-155, Qwen
   103). If the real model writes whitespace the template would not (e.g. `\n` after `</think>` on GLM, `<think>\n\n` on
   Qwen), the next prompt diverges inside that newest turn — cheap (the `-4` checkpoint of that request lies before it).
   Whether the real models do: UNKNOWN (random weights).
8. **Where checkpoints can be, CONFIRMED:** the server places mid-prompt checkpoints only at user-message starts found
   by the template's message delimiters (`tools/server/server-task.h:269`, `server-context.cpp:3976-3983`), plus at
   `4 + n_ubatch` and 4 tokens before the end of each prompt (`:3985-4006`). Delimiters: GLM `<|assistant|>`, `<|user|>`
   (autoparser); Qwen `<|im_start|>assistant`, `<|im_start|>user\n<tool_response>` (role tool), `<|im_start|>tool_response`
   (tool), `<|im_start|>user`, `<|im_start|>system`. Tool results are never user starts (GLM wraps them in
   `<|observation|>`, Qwen's tool delimiter matches first), so inside a tool loop only the end-of-prompt checkpoints
   exist — matching the trace (section 6).
9. **Template switches, CONFIRMED by rendering:** GLM reads `reasoning_effort` (`low`, `high`, anything else → `Max`,
   silently: `medium` gives `Max`) and `clear_thinking`; it does **not** read `enable_thinking` — `--reasoning off`,
   `chat_template_kwargs.enable_thinking=false` and `reasoning_effort: "none"` (which llama-server maps to
   `enable_thinking=false`, `tools/server/server-common.cpp:1417-1426`) leave the prompt unchanged and still end in
   `<|assistant|><think>`. Qwen reads `enable_thinking` (false → `<think>\n\n</think>\n\n`, no reasoning instructions),
   `reasoning_effort` (`xhigh` default, `medium` = no instructions, `low`; anything else → exception), `preserve_thinking`
   and `add_vision_id`. Neither template injects a date: no `strftime_now`, `datetime`, `date_string` (llama.cpp passes
   `datetime` / `date_string`, `common/chat.cpp:1052-1060`, unused). The R1 example `EXTRA_BODY` `{"chat_template_kwargs":
   {"enable_thinking": false}}` (`scripts/r1/configs/llamacpp-server.json`) therefore works for Qwen only.
10. **Model facts (section 7), CONFIRMED from the vendored `config.json`:** GLM-5.3-Flash `Glm5NextForConditionalGeneration`
    with a `vision_config`; `text_config` `n_group` 1, `topk_group` 1, `swiglu_limit` 10.0, `index_topk` 2048,
    `first_k_dense_replace` 3, 45 layers, `vocab_size` 154880, `max_position_embeddings` 1048576; the official checkpoint
    is FP8 (`quant_method` fp8, `fmt` e4m3, `activation_scheme` dynamic, `weight_block_size` [128, 128]).
    Qwen3.8-Flash-Next `Qwen4ExpForConditionalGeneration`, also VL (`vision_config`), `language_model_only: false`.
11. **N12's open point, group-limited routing:** llama.cpp's converter @79e2e74 lifts `text_config` to the root and
    writes `expert_group_count = n_group`, `expert_group_used_count = topk_group` (`conversion/base.py:1353-1355,
    1524-1529`; `Glm5NextModel`, `conversion/glm.py:407-411`, does not override them) — CONFIRMED by reading; the converter
    was NOT RUN (no torch in the sandbox). So a GGUF converted by llama.cpp from this config carries 1 / 1 (PROVISIONAL,
    from the code and the config), and grouping only happens for `n_expert_groups > 1` (`src/llama-model.cpp:1394`,
    `src/llama-graph.cpp:2145`): the "possible 7th routing difference" of N12 does not arise for such a file. The value in
    the Unsloth `glm5next` files (another converter): UNKNOWN until their header is read (gap 9).
12. **Token level: UNKNOWN.** Everything above is bytes; the real tokenizers are not vendored (`tokenizer.json` 20.2 MB /
    12.8 MB, PIN.json `not_vendored`). Request in section 8.

Tests: `tests/test_chat_templates.py`, 14 tests: 11 need only jinja2 and run in the plain `unittest discover`; 3 need the
server build and skip cleanly without it. All 14 pass with the build (section 9). Mutation check: 8/9 mutations killed,
the survivor is an equivalent mutation (section 10).

## 1. Inputs

| input | value | label |
|---|---|---|
| GLM template | `zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, `third_party/hf/glm-5.3-flash-chat/chat_template.jinja` 10950 B, sha256 `0c4099f3…6309c5` (PIN.json, SHA256SUMS; `VendoredFiles.test_sha256sums` checks them) | CONFIRMED |
| Qwen template | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540`, `third_party/hf/qwen3.8-flash-next-chat/chat_template.jinja` 8952 B | CONFIRMED |
| llama.cpp | checkout `<S>/llama.cpp` at `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (= `tools/engines/pins.json`), shared CPU build `<S>/llama.cpp/build-cpu` (`GGML_NATIVE=OFF`, Release), `version: 0.6.0-dev (build 1, commit 79e2e74)` (log line 2) | CONFIRMED |
| reference renderer | jinja2 3.1.6 (installed), configured like transformers 5.16.0 (`transformers-5.16.0-py3-none-any.whl`, sha256 `2b4adda8d99bb4c1b208da5ed967e2625ad449087827a750cb3b23ce4000b139`, downloaded into scratch only to read `utils/chat_template_utils.py:478-494,581-587` and `tokenization_utils_base.py:980-988,1344-1360,3107`; transformers itself is not installed). 5.16.0 is the version GLM's `generation_config.json` was written with | CONFIRMED |
| tiny models | `tools/chattmpl/bytes_vocab_model.py` over N11's generators (`git archive origin/claude/n11-r1-tools-cpu-server` = `6680448`, `tools/tinygen`): qwen4exp and glm5-next (12 layers, 3 DSA layers), vocabulary 256 single bytes + 3 controls + 1 unused merge, so every text has exactly one tokenization; sha256 `3fb41bfc…64f3d` (glm5next), `135bf3f8…3946c` (qwen4exp), the same in the earlier full driver run of this session (log lines 7-8; the generator is deterministic, seed 0) | CONFIRMED |
| harness | `tools/chattmpl/chat_parse_harness.cpp`, g++ 13.3.0 against the build's `libllama-common.so` (`run_n14.sh`), or `tools/chattmpl/CMakeLists.txt` out of tree | CONFIRMED |

Why a bytes-only vocabulary: with N11's `--text-vocab` (merges `" a".." z"`, `"\n\n"`) a grammar-forced turn may be
sampled as different tokens than the same bytes tokenize to in the next prompt; an exploratory run on those models
reused only up to `prompt − 4` at every request even where the text extended exactly (the generated tokens differed
from the re-tokenized ones). That measures BPE, not the template; with one byte per token the live numbers are the
template's (exploratory run not kept; the committed driver uses the bytes-only models).

## 2. Method

`tools/chattmpl/run_n14.sh <llama.cpp checkout> <build> <N11 tinygen> <work>` checks the pin, builds the harness, writes
the models, runs `tools/chattmpl/run_n14.py` (four sections) and `tests/test_chat_templates.py`. Conversations:
`tools/chattmpl/convs.py` — system + user, multi-turn (with whitespace and Cyrillic), no system, reasoning, a second
assistant turn without / with empty `reasoning_content`, reasoning inline in content, and an 18-message agent session
(system → user → assistant with reasoning + tool call → tool → assistant with two parallel calls → two tool results →
… → answer → user → … → answer → user) with five tools (string, integer, array, object, boolean, number, `["string",
"null"]` parameters, one description with non-ASCII). Messages go to llama-server in OpenAI form (tool-call `arguments`
as a JSON string), to jinja2 in HF form (arguments as a dict).

- **render**: every case × `add_generation_prompt` true/false × switch, `/apply-template` (`tools/server/server-context.cpp:5584-5598`:
  `oaicompat_chat_params_parse`, the same path as `/v1/chat/completions`, without inference) against jinja2; plus a
  server started with `--no-reasoning-preserve`.
- **prefix**: `tools/chattmpl/analysis.py` on the agent session — `history`: is `render(conv[:k])` (no generation
  prompt) a byte prefix of `render(conv[:k+1])`; `loop`: at every request (conversation ends with user/tool, the model
  writes the next assistant turn) does the next prompt start with prompt + the turn as the template renders it. Three
  renderer pairs: jinja2 alone, server alone, and server prompts with jinja2 turns (what a model trained on the
  reference template writes into llama-server). The server for this section runs with `--no-prefill-assistant` so that a
  history ending in an assistant turn renders as history. Plus "switches": the last prompt of the session under another
  setting.
- **parse**: per request point, the harness gets the conversation and candidate outputs: `ideal` (the turn exactly as
  the template renders it after the generation prompt, cut before Qwen's `<|im_end|>`), `reasoning_padded`,
  `content_padded`, `unknown_tool`; the parsed message is put back into the conversation, rendered by the server and
  compared with the original.
- **live**: `/v1/chat/completions` with the tiny hybrid model, each assistant turn forced byte for byte with a GBNF
  literal (`tool_choice: none` so that a custom grammar is allowed), `-lv 4` for the checkpoint trace; four
  configurations: default, preserve off, preserve off with `--ctx-checkpoints 4`, one turn with empty reasoning.

## 3. Rendering: llama-server vs jinja2 (log lines 10-48)

`render glm: 45/62 byte-equal`, `render qwen: 52/62 byte-equal`; every non-equal row is one of the three kinds of
Result 1, and `ServerRender.test_render_server_vs_jinja2` fails on any other difference or on a missing one (17 for GLM,
10 for Qwen). Template lines (GLM):

```
143:{%- if m.reasoning_content is string %}
144:    {%- set reasoning_content = m.reasoning_content %}
145:{%- elif '</think>' in content %}
146:    {%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}
147:    {%- set content = content.split('</think>')[-1] %}
148:{%- endif %}
149:{%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is defined -%}
150:{{ '<think>' + reasoning_content +  '</think>'}}
151:{%- else -%}
152:{{ '<think></think>' }}
```

llama.cpp `common/jinja/runtime.cpp@79e2e74`:

```
485:    context scope(ctx); // new scope for loop variables
601:    for (size_t i = 0; i < filtered_items.size(); i++) {
615:        scope_update_fns[i](scope);
618:                value val = stmt->execute(scope);
```

`common/chat.cpp:202-204@79e2e74` (`to_json_oaicompat`): `if (!reasoning_content.empty()) { jmsg["reasoning_content"] =
reasoning_content; }` — an empty reasoning is not passed to the template, so the leak also hits a turn whose reasoning
was empty, not only one the client sent without the field.

## 4. Prefix stability (log lines 49-106)

| session | GLM loop breaks | Qwen loop breaks | where |
|---|---|---|---|
| defaults (server = jinja2) | 0 of 7 | 0 of 7 | — |
| preserve off | 1 (request 4 → 5, after the 2nd user message) | 1 (same) | first assistant turn after the previous user message (message 2): GLM byte 1936 `<think>‖I should…` → `<think>‖</think><tool_call>…`; Qwen byte 2756 `<t‖hink>\nI should…` → `<t‖ool_call>…` |
| enable_thinking false (no reasoning in the session) | 0 | 0 | — |
| one turn with empty reasoning, server prompts + jinja2 turns | 1 (request 3 → 4, byte 3320 of 3517: `<think>‖</think><tool_call>run_tests…` → `<think>‖parse() splits on ';'…`) | 0 | the leak of Result 1(a) |

History view (`render(conv[:k])` prefix of `render(conv[:k+1])`): GLM none with defaults; Qwen one, at k = 6 — between
two parallel tool results, because a tool block is closed by `<|im_end|>` only after its last result (lines 154-157);
llama-server never holds that state in an agent loop (the client sends both results together), so the loop view has no
break. Qwen also refuses `conv[:1]` (system only: `No user query found in messages.`, line 100), logged by the server as
`got exception` (log lines 73-84). Switches of the last prompt (log lines 50-55, 86-91): GLM `reasoning_effort` low/high
40 shared bytes, `medium` and `enable_thinking=false` 4068 = identical; Qwen `enable_thinking=false` and `medium` 19,
`low` 46; one tool added: GLM 1501, Qwen 1748.

## 5. Switch between requests, live (`docs/evidence/mutation-scripts/2026-10-11-n14-switch-probe.py`)

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n14-switch-probe.py <S>/llama.cpp/build-cpu/bin/llama-server <S>/n14/final2/models/glm5next_bytes.gguf <S>/n14/final2/models/qwen4exp_bytes.gguf <S>/n14/switch
glm reasoning_effort_low request 0: status 200 cache_n 0 prompt_n 1936
glm reasoning_effort_low request 1: status 200 cache_n 2080 prompt_n 87
glm reasoning_effort_low request 2 (switched): status 200 cache_n 0 prompt_n 2819
    | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
glm one_tool_added request 0: status 200 cache_n 0 prompt_n 1936
glm one_tool_added request 1: status 200 cache_n 2080 prompt_n 87
glm one_tool_added request 2 (switched): status 200 cache_n 1420 prompt_n 1523
    | restored context checkpoint (pos_min = 1419, pos_max = 1419, n_tokens = 1420, n_past = 1420, size = 0.055 MiB)
qwen reasoning_effort_low request 0: status 200 cache_n 0 prompt_n 2762
qwen reasoning_effort_low request 1: status 200 cache_n 2929 prompt_n 113
qwen reasoning_effort_low request 2 (switched): status 200 cache_n 0 prompt_n 3642
    | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
qwen one_tool_added request 0: status 200 cache_n 0 prompt_n 2762
qwen one_tool_added request 1: status 200 cache_n 2929 prompt_n 113
qwen one_tool_added request 2 (switched): status 200 cache_n 0 prompt_n 3871
    | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
```

GLM "one tool added" restored `n_tokens = 1420` only because the tools block of this short session ends after the first
`prompt − 4 − n_ubatch` checkpoint (1501 > 1420 bytes); with a real first prompt (thousands of tokens of instructions
after the tools) that checkpoint lies after the tools block (PROVISIONAL), and the result is a full re-prefill as on Qwen.

## 6. Parser and live B3

Parser (log lines 107-148): see Result 6-8. GLM handler line `common_chat_templates_apply_jinja: using differential
autoparser`, Qwen `Using specialized template: Qwen3-Coder` (harness stderr with `CHATTMPL_DEBUG=1`); generation prompts
`<|assistant|><think>` and `<|im_start|>assistant\n<think>\n`; thinking end tags GLM `['</think>']`, Qwen `['\n</think>',
'</think>', '<tool_call>']`; with `enable_thinking=false` GLM's generation prompt stays `<|assistant|><think>`, Qwen's is
`<|im_start|>assistant\n<think>\n\n</think>\n\n`.

Live (log lines 149-386; tokens = bytes): with defaults every request reuses `cache_n = held` (GLM 2080, 2642, 3235,
3542, 3704, 3946; Qwen 2929, 3508, 4123, 4449, 4650, 4946), and new checkpoints appear only at the end of each prompt and
at the last user message (GLM 1853, Qwen 2649 in request 0). Preserve off: request 5 `extends False`, GLM `restored
context checkpoint (pos_min = 1931, pos_max = 1931, n_tokens = 1932…)` then `erased invalidated` for the 8 later ones,
`cache_n 1932 prompt_n 1624`; Qwen `n_tokens = 2649`, `cache_n 2649 prompt_n 1790`. `--ctx-checkpoints 4`: `erasing
context checkpoint too close to an earlier one` (`server-context.cpp:2524-2541`: when the list is full, checkpoints of
earlier tasks within `--checkpoint-min-step` of the previous kept one go first; if it is still full, the oldest goes,
`:2543-2552`), restore at 1420 / 2246.
Answer to N7 finding 4: the server re-processes from the newest surviving checkpoint at or before the first changed
token; with the defaults (`--checkpoint-min-step 8192`, `--ctx-checkpoints 32`, `common/common.h:637-639`) in a long
session that is at most one user round plus up to one min-step back after evictions (PROVISIONAL beyond the measured
5 KB session: the eviction rule is read, not run at 8192-token scale); a full re-prefill happens when no checkpoint
precedes the change — a changed system header, tools list or reasoning setting (section 5).

## 7. Model facts and the converter (`2026-10-11-n14-config-facts.py`, `2026-10-11-n14-converter-ngroup.sh`)

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n14-config-facts.py
== third_party/hf/glm-5.3-flash-chat/config.json (zai-org/GLM-5.3-Flash @ eb9eb208eb0d988989d07a6a12d0fdeb5f52574a)
  architectures            2: "architectures": [
  n_group                  234: "n_group": 1,
  topk_group               255: "topk_group": 1,
  swiglu_limit             253: "swiglu_limit": 10.0,; 278: "swiglu_limit": 10.0,
  index_topk               26: "index_topk": 2048,
  first_k_dense_replace    14: "first_k_dense_replace": 3,
  num_hidden_layers        240: "num_hidden_layers": 45,
  vocab_size               259: "vocab_size": 154880
  max_position_embeddings  181: "max_position_embeddings": 1048576,
  vision_config            261: "vision_config": {
  quantization_config      290: "quantization_config": {
  quant_method             1804: "quant_method": "fp8",
  fmt                      292: "fmt": "e4m3",
  activation_scheme        291: "activation_scheme": "dynamic",
  weight_block_size        1805: "weight_block_size": [
  architectures=['Glm5NextForConditionalGeneration'] text_config.model_type=glm5_next_text top-level keys=['architectures', 'image_end_token_id', 'image_start_token_id', 'image_token_id', 'model_type', 'quantization_config', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_end_token_id', 'video_start_token_id', 'video_token_id', 'vision_config']
== third_party/hf/qwen3.8-flash-next-chat/config.json (Qwen/Qwen3.8-Flash-Next @ de4b8e4d43b917e7706784d8bb445c9af86a3540)
  architectures            2: "architectures": [
  language_model_only      6: "language_model_only": false,
  vision_config            136: "vision_config": {
  num_hidden_layers        93: "num_hidden_layers": 1,; 103: "num_hidden_layers": 48,
  vocab_size               131: "vocab_size": 248320
  max_position_embeddings  84: "max_position_embeddings": 262144,
  quantization_config      ABSENT
  architectures=['Qwen4ExpForConditionalGeneration'] text_config.model_type=qwen4_exp_text top-level keys=['architectures', 'image_token_id', 'language_model_only', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_token_id', 'vision_config', 'vision_end_token_id', 'vision_start_token_id']
$ bash docs/evidence/mutation-scripts/2026-10-11-n14-converter-ngroup.sh <S>/llama.cpp
checkout 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
--- conversion/__init__.py:112
    "Glm5NextForConditionalGeneration": "glm",
--- conversion/glm.py:407,411
@ModelBase.register("Glm5NextForConditionalGeneration")
@ModelBase.example("zai-org/GLM-5.3-Flash")
class Glm5NextModel(TextModel):

    model_arch = gguf.MODEL_ARCH.GLM5_NEXT
--- conversion/glm.py:456,458

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
--- conversion/base.py:1340,1355


class TextModel(ModelBase):
    model_type = ModelType.TEXT
    hf_arch: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_mistral_format:
            self.hf_arch = get_model_architecture(self.hparams, self.model_type)
        else:
            self.hf_arch = ""

        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}
--- conversion/base.py:1435,1436
    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)
--- conversion/base.py:1524,1529
        if (n_expert_groups := self.hparams.get("n_group")) is not None:
            self.gguf_writer.add_expert_group_count(n_expert_groups)
            logger.info(f"gguf: expert groups count = {n_expert_groups}")
        if (n_group_used := self.hparams.get("topk_group")) is not None:
            self.gguf_writer.add_expert_group_used_count(n_group_used)
            logger.info(f"gguf: expert groups used count = {n_group_used}")
--- gguf-py/gguf/constants.py:125,126
        EXPERT_GROUP_COUNT                = "{arch}.expert_group_count"
        EXPERT_GROUP_USED_COUNT           = "{arch}.expert_group_used_count"
--- src/llama-model.cpp:1362,1363
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);
--- src/llama-model.cpp:1390,1398
    GGML_ASSERT(n_expert_used_max <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(n_expert_used_max > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
--- src/llama-graph.cpp:2143,2146
    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;
--- Glm5NextModel overrides of n_group / topk_group / expert_group (none expected):
(none)
```

## 8. Token level: UNKNOWN, and a request

What is UNKNOWN without the real tokenizers: whether `<think>`, `</think>`, `<tool_call>`, `<arg_key>`, `<arg_value>`,
`<tool_response>` are single tokens in GLM-5.3 (its `tokenizer_config.json` lists `<|system|>`, `<|user|>`,
`<|assistant|>`, `<|observation|>`, `[gMASK]`, `<sop>` among `extra_special_tokens`, not those; for Qwen
`tokenizer_config.json` lists `<think>` 248068, `</think>` 248069, `<tool_call>` 248058, `<tool_response>` 248066 as
added tokens with `special: false` — CONFIRMED from the file); whether the turn a model samples tokenizes the same way
when the next prompt re-tokenizes it (BPE boundaries around `\n</think>\n\n`, `<|im_end|>\n`); the token positions of the
breaks above. **REQUEST (infra-ops, workflow change):** a `vendor-hf` variant that, on the runner, downloads
`tokenizer.json` of both pinned revisions (20 217 442 B and 12 809 320 B), runs `tools/chattmpl` against a vocab-only
GGUF built from it (or against `tokenizers` directly) and commits only the result JSON (token counts and ids at each
break point, round-trip token equality of every forced turn) — no file over 1 MiB in the repository. Also: the R1
configs start llama-server with `--jinja` and **no** `--chat-template-file` (`scripts/r1/configs/llamacpp-server.json`),
so they use the template embedded in the Unsloth GGUF; whether it equals the HF template is UNKNOWN — vendor the GGUF
header's `tokenizer.chat_template` string (a few KB, via an HTTP Range read, as gap 9 proposes for the headers), or pass
the vendored template with `--chat-template-file`.

## 9. Tests

```
$ CHATTMPL_SERVER_BIN=... CHATTMPL_HARNESS=... CHATTMPL_GLM_GGUF=... CHATTMPL_QWEN_GGUF=... CHATTMPL_WORK=... python3 -m unittest -v tests.test_chat_templates   (last step of run_n14.sh, Appendix A)
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
Ran 14 tests in 44.879s

OK
```

Without the build (plain `unittest discover`): 11 run, 3 skip with `set CHATTMPL_SERVER_BIN, CHATTMPL_GLM_GGUF,
CHATTMPL_QWEN_GGUF (tools/chattmpl/run_n14.sh)`.

## 10. Mutation check (`docs/evidence/mutation-scripts/2026-10-11-n14-mutations.sh`)

```
$ MUTDIR=<S>/n14/mut bash docs/evidence/mutation-scripts/2026-10-11-n14-mutations.sh
HEAD 1a95c5c
base exit 0 (OK)
M1-tojson-ascii KILLED  (ParseHarness.test_parse_and_round_trip ServerRender.test_render_server_vs_jinja2 )
M2-no-trim-blocks KILLED  (LiveB3.test_prompt_cache_reuse ParseHarness.test_parse_and_round_trip ServerRender.test_render_server_vs_jinja2 )
M3-lcp-short KILLED  (LiveB3.test_prompt_cache_reuse TemplateProperties.test_defaults_prefix_stable_in_agent_loop TemplateProperties.test_dropping_past_reasoning_breaks_at_new_user_message TemplateProperties.test_qwen_enable_thinking_false )
M4-gbnf-newline SURVIVED
M4b-gbnf-quote KILLED  (LiveB3.test_prompt_cache_reuse )
M5-no-cache-prompt KILLED  (LiveB3.test_prompt_cache_reuse )
M6-glm-preserve-kwarg KILLED  (LiveB3.test_prompt_cache_reuse ServerRender.test_render_server_vs_jinja2 )
M7-harness-no-reasoning KILLED  (ParseHarness.test_parse_and_round_trip )
M8-client-drops-kwargs KILLED  (LiveB3.test_prompt_cache_reuse ServerRender.test_render_server_vs_jinja2 )
```

M4 (GBNF newline escape removed) survives as expected: a GBNF string literal accepts a raw newline
(`src/llama-grammar.cpp:180-181@79e2e74`, `parse_char` decodes any unescaped byte), so the grammar is unchanged; M4b
(quote escape removed) is killed.

## 11. NOT RUN

- Token-level effects with the real tokenizers (section 8 request); the real models; GPU; any timing.
- The Unsloth GGUF's embedded chat template (R1 uses it) — not available here.
- llama.cpp's own `test-chat` target (the harness calls the same `common` functions instead).
- ik_llama.cpp's server with these templates (optional in the task; base B / R1 row 2 of GLM).
- `convert_hf_to_gguf.py` itself (no torch); finding 11 is from reading.
- Streaming (`stream: true`) parsing and partial tool calls; sampling with the lazy tool grammar active (live runs use
  `tool_choice: none` and a custom grammar).
- Sessions long enough (> 8192 tokens per user round, > 32 checkpoints) to exercise the eviction rule at default
  settings (section 6 used `--ctx-checkpoints 4` instead).

## Appendix A. `tools/chattmpl/run_n14.sh` at `1a95c5c`, full output

```
$ bash tools/chattmpl/run_n14.sh <S>/llama.cpp <S>/llama.cpp/build-cpu <S>/n14/n11src/tools/tinygen <S>/n14/final2
llama.cpp checkout 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (pins.json llama.cpp: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6)
version: 0.6.0-dev (build 1, commit 79e2e74)
== harness
== models
wrote <S>/n14/final2/models/qwen4exp_bytes.gguf (n_vocab 260)
wrote <S>/n14/final2/models/glm5next_bytes.gguf (n_vocab 260)
3fb41bfc9e78a1b77c39fc1db07c1e01b9c3a6115c992d0cb8f4f6634ed64f3d  <S>/n14/final2/models/glm5next_bytes.gguf
135bf3f8ae16e442be9653b11a5b4490d1a7db2dfb69c0693b983e9eef43946c  <S>/n14/final2/models/qwen4exp_bytes.gguf
== run_n14.py
render glm: 45/62 byte-equal (server vs jinja2)
  default reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  default reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  default reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  default reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  default tools_ends_with_assistant agp=True first_diff=3688 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|assistant|><think>"
  effort_low reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_low reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_low reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_low reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_high reasoning_mixed agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_high reasoning_mixed agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_high reasoning_empty agp=True first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_high reasoning_empty agp=False first_diff=116 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_medium reasoning_mixed agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_medium reasoning_mixed agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  effort_medium reasoning_empty agp=True first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3<|assistant|><think>' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3<|assistant|><think>'
  effort_medium reasoning_empty agp=False first_diff=115 server='<|user|>u2<|assistant|><think>‖R1</think>a2<|user|>u3' jinja2='<|user|>u2<|assistant|><think>‖</think>a2<|user|>u3'
  log: 0.00.186.440 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
render qwen: 52/62 byte-equal (server vs jinja2)
  default tools_ends_with_assistant agp=True first_diff=4624 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|im_end|>\n<|im_start|>assistant\n<think>\n"
  default tools_ends_with_assistant agp=False first_diff=4624 server="plits on ','. The test passes.‖" jinja2="plits on ','. The test passes.‖<|im_end|>\n"
  effort_high reasoning agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high reasoning agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high reasoning_mixed agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high reasoning_mixed agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high reasoning_empty agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high reasoning_empty agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high agent_session agp=True server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  effort_high agent_session agp=False server_error=RuntimeError: HTTP 500 {"error": {"code": 500, "message": "\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "type": "server_error"}} | jinja2_error=TemplateError: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
  log: 0.00.087.736 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
  log: 0.00.701.448 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.703.316 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.704.687 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.706.105 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.707.641 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.708.953 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.711.092 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
  log: 0.00.713.070 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
prefix glm log: 0.00.213.630 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
prefix glm switch preserve_off: {"shared_bytes": 1936, "prompt_bytes": 4068, "at": "nd fix it.<|assistant|><think>‖</think><tool_call>list_files<arg_key>path</arg_ke"}
prefix glm switch enable_thinking_false: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
prefix glm switch effort_low: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖Low<|system|>\n# Tools\n\nYou may call one or more fu"}
prefix glm switch effort_high: {"shared_bytes": 40, "prompt_bytes": 4068, "at": "p><|system|>Reasoning Effort: ‖High<|system|>\n# Tools\n\nYou may call one or more f"}
prefix glm switch effort_medium: {"shared_bytes": 4068, "prompt_bytes": 4068, "at": "_response><|assistant|><think>‖"}
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
prefix qwen log: 0.00.107.149 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
prefix qwen log: 0.00.275.877 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.00.388.630 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.01.241.214 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.01.352.313 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.01.597.830 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.01.870.857 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.02.286.735 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.02.557.312 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.02.705.574 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.02.798.442 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.03.603.258 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.03.694.038 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 100, column 24 in source:\n...lti_step_tool %}↵    {{- raise_exception('No user query found in messages.') }}↵...\n                                           ^\nError: Jinja Exception: No user query found in messages.","type":"server_error"}}
prefix qwen log: 0.04.533.561 W srv    operator(): got exception: {"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 49, column 28 in source:\n...', 'low') %}↵        {{- raise_exception('Unexpected reasoning effort ' ~ reason...\n                                           ^\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"server_error"}}
prefix qwen switch preserve_off: {"shared_bytes": 2756, "prompt_bytes": 5093, "at": "end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/"}
prefix qwen switch enable_thinking_false: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
prefix qwen switch effort_low: {"shared_bytes": 46, "prompt_bytes": 5093, "at": "em\nReasoning effort is set to ‖low. Keep your thinking brief and focused, moving "}
prefix qwen switch effort_medium: {"shared_bytes": 19, "prompt_bytes": 5093, "at": "<|im_start|>system\n‖# Tools\n\nYou have access to the following function"}
prefix qwen switch effort_high: {"error": "RuntimeError: HTTP 500 {\"error\": {\"code\": 500, \"message\": \"\\n------------\\nWhile executing CallExpression at line 49, column 28 in source:\\n...', 'low') %}↵ {{- raise_exception('Unexpected reasoning effort ' ~ reason...\\n ^\\nError: Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.\", \"type\": \"server_error\"}}"}
prefix qwen switch one_tool_added: {"shared_bytes": 1748, "prompt_bytes": 5093, "at": "}, \"required\": [\"verbose\"]}}}\n‖{\"type\": \"function\", \"function\": {\"name\": \"web_sea"}
prefix qwen default/jinja2: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
prefix qwen default/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
prefix qwen default/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix qwen preserve_off/jinja2: history breaks 3 (k=[6, 12, 16]), refused prefixes 1; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 2756 of 4651 held, in message 2 (assistant) old='end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor' new='end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/'
prefix qwen preserve_off/server: history breaks 3 (k=[6, 12, 16]), refused prefixes 1; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 2756 of 4651 held, in message 2 (assistant) old='end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor' new='end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/'
prefix qwen preserve_off/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 1 of 7 requests
  request 4 (k=11): first_diff byte 2756 of 4651 held, in message 2 (assistant) old='end|>\n<|im_start|>assistant\n<t‖hink>\nI should look at the layout of the repositor' new='end|>\n<|im_start|>assistant\n<t‖ool_call>\n<function=list_files>\n<parameter=path>\n/'
prefix qwen enable_thinking_false/jinja2: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
prefix qwen enable_thinking_false/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
prefix qwen enable_thinking_false/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
prefix qwen empty_reasoning_turn/jinja2: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
prefix qwen empty_reasoning_turn/server: history breaks 1 (k=[6]), refused prefixes 1; loop breaks 0 of 7 requests
prefix qwen empty_reasoning_turn/server_prompts+jinja2_turns: history breaks 0 (k=[]), refused prefixes 0; loop breaks 0 of 7 requests
parse glm: format ['peg-native']; notools peg-native; nothink gen prompt '<|assistant|><think>'; handler ['\x1b[0mcommon_chat_templates_apply_jinja: using differential autoparser', 'common_chat_templates_apply_jinja: using differential autoparser']
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
parse qwen: format ['peg-native']; notools peg-native; nothink gen prompt '<|im_start|>assistant\n<think>\n\n</think>\n\n'; handler ['Using specialized template: Qwen3-Coder']
  generation_prompt '<|im_start|>assistant\n<think>\n' thinking '<think>' ['\n</think>', '</think>', '<tool_call>'] delimiters [('assistant', '<|im_start|>assistant'), ('tool', '<|im_start|>user\n<tool_response>'), ('tool', '<|im_start|>tool_response'), ('user', '<|im_start|>user'), ('system', '<|im_start|>system')]
  k=2 ideal: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_equal": true}
  k=2 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 2755, "rendered": "<|im_start|>assistant\n<think>\n‖I should look at the layout of the repository firs", "generated": "<|im_start|>assistant\n<think>\n‖\nI should look at the layout of the repository fir"}
  k=2 unknown_tool: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": false}, "round_trip_reproduces_output": false, "round_trip_first_diff": 2819, "rendered": " repository first.\n</think>\n\n<‖|im_end|>\n<|im_start|>user\n<tool_response>\nsrc/\nte", "generated": " repository first.\n</think>\n\n<‖tool_call>\n<function=no_such_tool>\n<parameter=path"}
  k=4 ideal: {"fields_equal": {"content": false, "reasoning_content": false, "tool_calls": true}, "round_trip_equal": true}
  k=4 reasoning_padded: {"fields_equal": {"content": false, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 3035, "rendered": "<|im_start|>assistant\n<think>\n‖Two things to check: the tests and where `parse` i", "generated": "<|im_start|>assistant\n<think>\n‖\nTwo things to check: the tests and where `parse` "}
  k=4 content_padded: {"fields_equal": {"content": false, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 3103, "rendered": "re `parse` is used.\n</think>\n\n‖Reading the tests and searching for the parser.\n\n<", "generated": "re `parse` is used.\n</think>\n\n‖\nReading the tests and searching for the parser.\n\n"}
  k=7 ideal: {"fields_equal": {"content": false, "reasoning_content": true, "tool_calls": true}, "round_trip_equal": true}
  k=7 reasoning_padded: {"fields_equal": {"content": false, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 3706, "rendered": "<|im_start|>assistant\n<think>\n‖parse() splits on ';' but the test uses ','. Fix t", "generated": "<|im_start|>assistant\n<think>\n‖\nparse() splits on ';' but the test uses ','. Fix "}
  k=7 content_padded: {"fields_equal": {"content": false, "reasoning_content": true, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 3780, "rendered": " Fix the separator.\n</think>\n\n‖The separator is wrong; fixing it.\n\n<tool_call>\n<f", "generated": " Fix the separator.\n</think>\n\n‖\nThe separator is wrong; fixing it.\n\n\n<tool_call>\n"}
  k=9 ideal: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_equal": true}
  k=9 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 4218, "rendered": "<|im_start|>assistant\n<think>\n‖Run the tests to confirm.\n</think>\n\n<tool_call>\n<f", "generated": "<|im_start|>assistant\n<think>\n‖\nRun the tests to confirm.\n\n</think>\n\n<tool_call>\n"}
  k=11 ideal: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_equal": true}
  k=11 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 4542, "rendered": "<|im_start|>assistant\n<think>\n‖All green; summarise.\n</think>\n\nFixed: `parse` now", "generated": "<|im_start|>assistant\n<think>\n‖\nAll green; summarise.\n\n</think>\n\nFixed: `parse` n"}
  k=11 content_padded: {"fields_equal": {"content": false, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 4574, "rendered": "l green; summarise.\n</think>\n\n‖Fixed: `parse` now splits on ','. The test passes.", "generated": "l green; summarise.\n</think>\n\n‖\nFixed: `parse` now splits on ','. The test passes"}
  k=13 ideal: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_equal": true}
  k=13 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 4736, "rendered": "<|im_start|>assistant\n<think>\n‖An empty string should give an empty list; the cur", "generated": "<|im_start|>assistant\n<think>\n‖\nAn empty string should give an empty list; the cu"}
  k=15 ideal: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_equal": true}
  k=15 reasoning_padded: {"fields_equal": {"content": true, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 5077, "rendered": "<|im_start|>assistant\n<think>\n‖Handle the empty case and add the test.\n</think>\n\n", "generated": "<|im_start|>assistant\n<think>\n‖\nHandle the empty case and add the test.\n\n</think>"}
  k=15 content_padded: {"fields_equal": {"content": false, "reasoning_content": false, "tool_calls": true}, "round_trip_reproduces_output": false, "round_trip_first_diff": 5127, "rendered": "e and add the test.\n</think>\n\n‖Done: `parse('')` returns `[]`, test added.<|im_en", "generated": "e and add the test.\n</think>\n\n‖\nDone: `parse('')` returns `[]`, test added.\n"}
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
live qwen default (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 2762 B | held None B shared None B extends None | cache_n 0 prompt_n 2762 | forced 167 B predicted_n 168 finish stop 
  request 1 k=4 status 200 prompt 3042 B | held 2929 B shared 2929 B extends True | cache_n 2929 prompt_n 113 | forced 466 B predicted_n 467 finish stop 
  request 2 k=7 status 200 prompt 3713 B | held 3508 B shared 3508 B extends True | cache_n 3508 prompt_n 205 | forced 410 B predicted_n 411 finish stop 
  request 3 k=9 status 200 prompt 4234 B | held 4123 B shared 4123 B extends True | cache_n 4123 prompt_n 111 | forced 215 B predicted_n 216 finish stop 
  request 4 k=11 status 200 prompt 4558 B | held 4449 B shared 4449 B extends True | cache_n 4449 prompt_n 109 | forced 92 B predicted_n 93 finish stop 
  request 5 k=13 status 200 prompt 4752 B | held 4650 B shared 4650 B extends True | cache_n 4650 prompt_n 102 | forced 194 B predicted_n 195 finish stop 
  request 6 k=15 status 200 prompt 5093 B | held 4946 B shared 4946 B extends True | cache_n 4946 prompt_n 147 | forced 103 B predicted_n 104 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 2245, pos_max = 2245, n_tokens = 2246, size = 0.038 MiB)
    | created context checkpoint 2 of 32 (pos_min = 2648, pos_max = 2648, n_tokens = 2649, size = 0.038 MiB)
    | created context checkpoint 3 of 32 (pos_min = 2757, pos_max = 2757, n_tokens = 2758, size = 0.038 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2928, pos_max = 2928, n_tokens = 2929, size = 0.038 MiB)
    | created context checkpoint 5 of 32 (pos_min = 3037, pos_max = 3037, n_tokens = 3038, size = 0.038 MiB)
    | created context checkpoint 6 of 32 (pos_min = 3507, pos_max = 3507, n_tokens = 3508, size = 0.038 MiB)
    | created context checkpoint 7 of 32 (pos_min = 3708, pos_max = 3708, n_tokens = 3709, size = 0.038 MiB)
    | created context checkpoint 8 of 32 (pos_min = 4122, pos_max = 4122, n_tokens = 4123, size = 0.038 MiB)
    | created context checkpoint 9 of 32 (pos_min = 4229, pos_max = 4229, n_tokens = 4230, size = 0.038 MiB)
    | created context checkpoint 10 of 32 (pos_min = 4448, pos_max = 4448, n_tokens = 4449, size = 0.038 MiB)
    | created context checkpoint 11 of 32 (pos_min = 4553, pos_max = 4553, n_tokens = 4554, size = 0.038 MiB)
    | created context checkpoint 12 of 32 (pos_min = 4649, pos_max = 4649, n_tokens = 4650, size = 0.038 MiB)
    | created context checkpoint 13 of 32 (pos_min = 4650, pos_max = 4650, n_tokens = 4651, size = 0.038 MiB)
    | created context checkpoint 14 of 32 (pos_min = 4747, pos_max = 4747, n_tokens = 4748, size = 0.038 MiB)
    | created context checkpoint 15 of 32 (pos_min = 4945, pos_max = 4945, n_tokens = 4946, size = 0.038 MiB)
    | created context checkpoint 16 of 32 (pos_min = 5088, pos_max = 5088, n_tokens = 5089, size = 0.038 MiB)
live qwen preserve_off (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 2762 B | held None B shared None B extends None | cache_n 0 prompt_n 2762 | forced 167 B predicted_n 168 finish stop 
  request 1 k=4 status 200 prompt 3042 B | held 2929 B shared 2929 B extends True | cache_n 2929 prompt_n 113 | forced 466 B predicted_n 467 finish stop 
  request 2 k=7 status 200 prompt 3713 B | held 3508 B shared 3508 B extends True | cache_n 3508 prompt_n 205 | forced 410 B predicted_n 411 finish stop 
  request 3 k=9 status 200 prompt 4234 B | held 4123 B shared 4123 B extends True | cache_n 4123 prompt_n 111 | forced 215 B predicted_n 216 finish stop 
  request 4 k=11 status 200 prompt 4558 B | held 4449 B shared 4449 B extends True | cache_n 4449 prompt_n 109 | forced 92 B predicted_n 93 finish stop 
  request 5 k=13 status 200 prompt 4439 B | held 4650 B shared 2756 B extends False | cache_n 2649 prompt_n 1790 | forced 194 B predicted_n 195 finish stop 
  request 6 k=15 status 200 prompt 4780 B | held 4633 B shared 4633 B extends True | cache_n 4633 prompt_n 147 | forced 103 B predicted_n 104 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 2245, pos_max = 2245, n_tokens = 2246, size = 0.038 MiB)
    | created context checkpoint 2 of 32 (pos_min = 2648, pos_max = 2648, n_tokens = 2649, size = 0.038 MiB)
    | created context checkpoint 3 of 32 (pos_min = 2757, pos_max = 2757, n_tokens = 2758, size = 0.038 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2928, pos_max = 2928, n_tokens = 2929, size = 0.038 MiB)
    | created context checkpoint 5 of 32 (pos_min = 3037, pos_max = 3037, n_tokens = 3038, size = 0.038 MiB)
    | created context checkpoint 6 of 32 (pos_min = 3507, pos_max = 3507, n_tokens = 3508, size = 0.038 MiB)
    | created context checkpoint 7 of 32 (pos_min = 3708, pos_max = 3708, n_tokens = 3709, size = 0.038 MiB)
    | created context checkpoint 8 of 32 (pos_min = 4122, pos_max = 4122, n_tokens = 4123, size = 0.038 MiB)
    | created context checkpoint 9 of 32 (pos_min = 4229, pos_max = 4229, n_tokens = 4230, size = 0.038 MiB)
    | created context checkpoint 10 of 32 (pos_min = 4448, pos_max = 4448, n_tokens = 4449, size = 0.038 MiB)
    | created context checkpoint 11 of 32 (pos_min = 4553, pos_max = 4553, n_tokens = 4554, size = 0.038 MiB)
    | restored context checkpoint (pos_min = 2648, pos_max = 2648, n_tokens = 2649, n_past = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 2757, pos_max = 2757, n_tokens = 2758, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 2928, pos_max = 2928, n_tokens = 2929, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 3037, pos_max = 3037, n_tokens = 3038, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 3507, pos_max = 3507, n_tokens = 3508, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 3708, pos_max = 3708, n_tokens = 3709, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 4122, pos_max = 4122, n_tokens = 4123, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 4229, pos_max = 4229, n_tokens = 4230, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 4448, pos_max = 4448, n_tokens = 4449, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 4553, pos_max = 4553, n_tokens = 4554, n_swa = 0, pos_next = 2649, size = 0.038 MiB)
    | created context checkpoint 3 of 32 (pos_min = 3922, pos_max = 3922, n_tokens = 3923, size = 0.038 MiB)
    | created context checkpoint 4 of 32 (pos_min = 4337, pos_max = 4337, n_tokens = 4338, size = 0.038 MiB)
    | created context checkpoint 5 of 32 (pos_min = 4434, pos_max = 4434, n_tokens = 4435, size = 0.038 MiB)
    | created context checkpoint 6 of 32 (pos_min = 4632, pos_max = 4632, n_tokens = 4633, size = 0.038 MiB)
    | created context checkpoint 7 of 32 (pos_min = 4775, pos_max = 4775, n_tokens = 4776, size = 0.038 MiB)
live qwen preserve_off-ctx-checkpoints-4 (-c 16384 -lv 4 --ctx-checkpoints 4):
  request 0 k=2 status 200 prompt 2762 B | held None B shared None B extends None | cache_n 0 prompt_n 2762 | forced 167 B predicted_n 168 finish stop 
  request 1 k=4 status 200 prompt 3042 B | held 2929 B shared 2929 B extends True | cache_n 2929 prompt_n 113 | forced 466 B predicted_n 467 finish stop 
  request 2 k=7 status 200 prompt 3713 B | held 3508 B shared 3508 B extends True | cache_n 3508 prompt_n 205 | forced 410 B predicted_n 411 finish stop 
  request 3 k=9 status 200 prompt 4234 B | held 4123 B shared 4123 B extends True | cache_n 4123 prompt_n 111 | forced 215 B predicted_n 216 finish stop 
  request 4 k=11 status 200 prompt 4558 B | held 4449 B shared 4449 B extends True | cache_n 4449 prompt_n 109 | forced 92 B predicted_n 93 finish stop 
  request 5 k=13 status 200 prompt 4439 B | held 4650 B shared 2756 B extends False | cache_n 2246 prompt_n 2193 | forced 194 B predicted_n 195 finish stop 
  request 6 k=15 status 200 prompt 4780 B | held 4633 B shared 4633 B extends True | cache_n 4633 prompt_n 147 | forced 103 B predicted_n 104 finish stop 
    | created context checkpoint 1 of 4 (pos_min = 2245, pos_max = 2245, n_tokens = 2246, size = 0.038 MiB)
    | created context checkpoint 2 of 4 (pos_min = 2648, pos_max = 2648, n_tokens = 2649, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2757, pos_max = 2757, n_tokens = 2758, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2648, pos_max = 2648, n_tokens = 2649, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 2928, pos_max = 2928, n_tokens = 2929, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2757, pos_max = 2757, n_tokens = 2758, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3037, pos_max = 3037, n_tokens = 3038, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 2928, pos_max = 2928, n_tokens = 2929, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3507, pos_max = 3507, n_tokens = 3508, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3037, pos_max = 3037, n_tokens = 3038, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 3708, pos_max = 3708, n_tokens = 3709, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3507, pos_max = 3507, n_tokens = 3508, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4122, pos_max = 4122, n_tokens = 4123, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3708, pos_max = 3708, n_tokens = 3709, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4229, pos_max = 4229, n_tokens = 4230, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 4122, pos_max = 4122, n_tokens = 4123, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4448, pos_max = 4448, n_tokens = 4449, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 4229, pos_max = 4229, n_tokens = 4230, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4553, pos_max = 4553, n_tokens = 4554, size = 0.038 MiB)
    | restored context checkpoint (pos_min = 2245, pos_max = 2245, n_tokens = 2246, n_past = 2246, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 4448, pos_max = 4448, n_tokens = 4449, n_swa = 0, pos_next = 2246, size = 0.038 MiB)
    | erased invalidated context checkpoint (pos_min = 4553, pos_max = 4553, n_tokens = 4554, n_swa = 0, pos_next = 2246, size = 0.038 MiB)
    | created context checkpoint 2 of 4 (pos_min = 3922, pos_max = 3922, n_tokens = 3923, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4337, pos_max = 4337, n_tokens = 4338, size = 0.038 MiB)
    | created context checkpoint 4 of 4 (pos_min = 4434, pos_max = 4434, n_tokens = 4435, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 3922, pos_max = 3922, n_tokens = 3923, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 4337, pos_max = 4337, n_tokens = 4338, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4632, pos_max = 4632, n_tokens = 4633, size = 0.038 MiB)
    | erasing context checkpoint too close to an earlier one (pos_min = 4434, pos_max = 4434, n_tokens = 4435, size = 0.038 MiB)
    | created context checkpoint 3 of 4 (pos_min = 4775, pos_max = 4775, n_tokens = 4776, size = 0.038 MiB)
live qwen empty_reasoning_turn (-c 16384 -lv 4):
  request 0 k=2 status 200 prompt 2762 B | held None B shared None B extends None | cache_n 0 prompt_n 2762 | forced 167 B predicted_n 168 finish stop 
  request 1 k=4 status 200 prompt 3042 B | held 2929 B shared 2929 B extends True | cache_n 2929 prompt_n 113 | forced 466 B predicted_n 467 finish stop 
  request 2 k=7 status 200 prompt 3713 B | held 3508 B shared 3508 B extends True | cache_n 3508 prompt_n 205 | forced 410 B predicted_n 411 finish stop 
  request 3 k=9 status 200 prompt 4234 B | held 4123 B shared 4123 B extends True | cache_n 4123 prompt_n 111 | forced 190 B predicted_n 191 finish stop 
  request 4 k=11 status 200 prompt 4533 B | held 4424 B shared 4424 B extends True | cache_n 4424 prompt_n 109 | forced 92 B predicted_n 93 finish stop 
  request 5 k=13 status 200 prompt 4727 B | held 4625 B shared 4625 B extends True | cache_n 4625 prompt_n 102 | forced 194 B predicted_n 195 finish stop 
  request 6 k=15 status 200 prompt 5068 B | held 4921 B shared 4921 B extends True | cache_n 4921 prompt_n 147 | forced 103 B predicted_n 104 finish stop 
    | created context checkpoint 1 of 32 (pos_min = 2245, pos_max = 2245, n_tokens = 2246, size = 0.038 MiB)
    | created context checkpoint 2 of 32 (pos_min = 2648, pos_max = 2648, n_tokens = 2649, size = 0.038 MiB)
    | created context checkpoint 3 of 32 (pos_min = 2757, pos_max = 2757, n_tokens = 2758, size = 0.038 MiB)
    | created context checkpoint 4 of 32 (pos_min = 2928, pos_max = 2928, n_tokens = 2929, size = 0.038 MiB)
    | created context checkpoint 5 of 32 (pos_min = 3037, pos_max = 3037, n_tokens = 3038, size = 0.038 MiB)
    | created context checkpoint 6 of 32 (pos_min = 3507, pos_max = 3507, n_tokens = 3508, size = 0.038 MiB)
    | created context checkpoint 7 of 32 (pos_min = 3708, pos_max = 3708, n_tokens = 3709, size = 0.038 MiB)
    | created context checkpoint 8 of 32 (pos_min = 4122, pos_max = 4122, n_tokens = 4123, size = 0.038 MiB)
    | created context checkpoint 9 of 32 (pos_min = 4229, pos_max = 4229, n_tokens = 4230, size = 0.038 MiB)
    | created context checkpoint 10 of 32 (pos_min = 4423, pos_max = 4423, n_tokens = 4424, size = 0.038 MiB)
    | created context checkpoint 11 of 32 (pos_min = 4528, pos_max = 4528, n_tokens = 4529, size = 0.038 MiB)
    | created context checkpoint 12 of 32 (pos_min = 4624, pos_max = 4624, n_tokens = 4625, size = 0.038 MiB)
    | created context checkpoint 13 of 32 (pos_min = 4625, pos_max = 4625, n_tokens = 4626, size = 0.038 MiB)
    | created context checkpoint 14 of 32 (pos_min = 4722, pos_max = 4722, n_tokens = 4723, size = 0.038 MiB)
    | created context checkpoint 15 of 32 (pos_min = 4920, pos_max = 4920, n_tokens = 4921, size = 0.038 MiB)
    | created context checkpoint 16 of 32 (pos_min = 5063, pos_max = 5063, n_tokens = 5064, size = 0.038 MiB)
== tests/test_chat_templates.py
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
Ran 14 tests in 44.879s

OK
```
