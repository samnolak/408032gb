# N11: the measurement tools of R1 against a real llama-server on CPU - 2026-10-11 (sandbox CPU, no GPU)

Task N11 of the hardware-free track: close gap 7 of `docs/reassessment/CORRECTNESS.md` section 3 ("the measurement tools
were checked only against mocks or stubs"). The tools - `scripts/r1/run_r1.py` with `scripts/r1/configs/llamacpp-server.json`,
`tools/bench/` (`prompts.py`, `run_matrix.py`, `probes.py`, the `/tokenize` dialects of `oa_client.py`),
`tools/quality/logprob_compare.py`, `tools/gateway/gateway.py`, and `tools/kv/slot_persist.py` as a step of `run_r1.py` -
were run against the real `llama-server` that R1 will use (llama.cpp `79e2e74`, `tools/engines/pins.json`), built for CPU,
on tiny random-weight models. Branch `claude/n11-r1-tools-cpu-server`, cut from `claude/n7-server-hybrid-cpu` at `c2f6c91`
(session `session_01EyB9XrhA3NTRjjQx1A9qMh`).

**Nothing here is a GPU result, a speed result or a quality result.** The models are tiny random-weight files; every
timing below is a CPU tiny-model wall time, said once as such and never compared with any target. Scores of probes are
noise by construction. In literal blocks the scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad` is shortened to `<scratch>`.

| piece | file |
|---|---|
| text-capable tiny models | `tools/tinygen/text_vocab.py`, `qwen4exp_tiny.py` / `glm5next_tiny.py` `--text-vocab --ctx N` (defaults byte-identical), commit `830c46a` |
| smoke driver (CI) | `tools/srvcheck/r1_tools_smoke.py` (stdlib only), `tools/srvcheck/run_n11.sh`, commit `56f1a0b` |
| tests | `tests/test_tinygen_text_vocab.py` (4), `tests/test_r1_tools_smoke.py` (2), new cases in `tests/test_logprob_compare.py` (+5), `tests/test_bench.py` (+2), `tests/test_slot_persist.py` (+1), `tests/test_r1_orchestration.py` (assertions added) |
| demonstrations | `docs/evidence/mutation-scripts/2026-10-11-n11-demos.py` (old tool at `c2f6c91` against the fixed one, same server) |
| CI | `.github/workflows/server-cpu.yml`, job `n7`: step `run_n11.sh` after `run_n7.sh` (same build) |

## Setup (CONFIRMED, literal)

Server binary: the CPU build of N7 at the R1 pin (`GGML_NATIVE=OFF`, `docs/evidence/2026-10-10-n7-server-hybrid-cpu.md`,
"Setup"); the CI-style run below also built its own copy from the same checkout into `<scratch>/n11/build-n11`; the
checkout itself was not written to (`git status --short` empty before and after, HEAD `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`).
```
$ <scratch>/llama.cpp/build-cpu/bin/llama-server --version
version: 0.6.0-dev (build 1, commit 79e2e74)
built with GNU 13.3.0 for Linux x86_64
```
Models (`PYTHONPATH=<scratch>/llama.cpp/gguf-py`):
```
$ python3 tools/tinygen/qwen4exp_tiny.py <scratch>/n11/models/qwen4exp_text.gguf --text-vocab --ctx 262144
$ python3 tools/tinygen/glm5next_tiny.py <scratch>/n11/models/glm5next_text.gguf --text-vocab --ctx 262144
d5304a8c2756e999005791de0348e7b9cbc0f0ed1c9bfeb83a6661da34ebe83c  n11/models/glm5next_text.gguf
e9555b378dddacabd773fe3621823b440dc1b28823d94fc7486c04ab86c7df81  n11/models/qwen4exp_text.gguf
```
Default outputs of the generators before and after the `--text-vocab` change (`<scratch>/n11/gen-before` with the
unmodified scripts, `gen-after` with the new ones), equal to N7's hashes; `cmp` reported every pair identical:
```
26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5  glm5next.gguf        (both)
6309ba52478a425e0bf6a87ac5ea1f283a584bba52b502ea56c62b44eba49a67  glm5next_l4.gguf     (both, --layers 4)
f7833b6ce213f0fe1f2130b79754931f428c83c4c46edf952321990853965681  glm5next_mtp.gguf    (both)
0fe7442121108a67b7bb01d7bc432d308f43fff41107d56e7a08c99e1ec51951  qwen4exp.gguf        (both)
23bf70ffd45b0a458bc99657b15bcebf05f2843d6a6737a528a4f2f4422976cb  qwen4exp_mtp.gguf    (both)
```
(the five lines are merged from the two `sha256sum` listings, which printed the same hashes; `tests/test_tinygen_text_vocab.py`
now pins three of them, and gguf-py of `ec7630a` writes the same bytes: `26c83267fe1a7588` for the default glm5-next.)

## 1. Inventory: what each tool needs from the server

Tool lines are on this branch (HEAD of the evidence commit); server lines are `tools/server/*.cpp@79e2e74`, each printed
and read before citing. "Provided" = CONFIRMED by reading those lines and by the runs in section 3.

| tool (entry point) | endpoints and fields it relies on | llama-server @79e2e74 | result on the real server |
|---|---|---|---|
| `tools/bench/oa_client.py` (used by `prompts.py`, `run_matrix.py`, `probes.py`) | GET `/v1/models` (:184), `/health` (:187), `/metrics` (:190); POST `/tokenize` llama.cpp dialect `{"content", "add_special": false}` (:206), vLLM dialect `{"prompt", "add_special_tokens"}` (:208), a zero count is a miss; streamed POST `/v1/chat/completions` (:259) and `/v1/completions` (:268) with `stream_options.include_usage` (:286); SSE `choices[0].delta.{content,reasoning_content,tool_calls}`, `choices[0].text`, `finish_reason`, `usage`, `timings` (:396-397); cached tokens from `usage.prompt_tokens_details.cached_tokens` or `timings.cache_n` (`runlog.py:96-105`) | routes `server.cpp:263,265,269,272,274,290`; `/tokenize` answers `{"tokens": []}` to a body without `content` (`server-context.cpp:5612-5651`); `include_usage` `server-task.cpp:494`; `usage` with `prompt_tokens_details.cached_tokens` `server-task.cpp:357-364` | provided; dialect detection picks llama.cpp, the vLLM body is a miss as designed |
| `tools/bench/prompts.py` | `/tokenize` counts; pad piece `" the"` assumed to be one token (`prompts.py:33`) | provided | works; exact fits depend on the vocabulary (finding 9) |
| `tools/bench/run_matrix.py` | forced length: vLLM `{"ignore_eos", "min_tokens"}`, llama.cpp `{"ignore_eos", "n_predict"}` (`run_matrix.py:42-44`); `tools` and `role: tool` messages in the agent loop (`run_matrix.py:50`, agent loop); cold prompts unique by a first-line nonce | `n_predict` with aliases `max_completion_tokens`, `max_tokens` (`server-schema.cpp:44-47`), `ignore_eos` (`server-schema.cpp:423`); no `min_tokens` field (absent from `server-schema.cpp`; the key is accepted and ignored, observed); tools with `--jinja` and the built-in chatml | works; **bug: cold prompts repeated across runs and were served from the cache** (fixed, finding 4) |
| `tools/bench/probes.py` | `response_format: {"type": "json_object"}` (`probes.py:338`); `tools` + `tool_choice: "auto"` (`probes.py:369`); `role: tool` result (`probes.py:395`) | `response_format` `server-common.cpp:1304-1306` (grammar) | works; every probe answered |
| `tools/quality/logprob_compare.py` | non-streamed POST `/v1/completions` `{"prompt", "max_tokens": 1, "temperature": 0, "logprobs": K}` (`logprob_compare.py:42-45`); reads `choices[0].logprobs` | `logprobs` is an alias of `n_probs` (`server-schema.cpp:179-180`); layout `choices[0].logprobs.content[0].top_logprobs = [{id, token, bytes, logprob}]` (`server-task.cpp:256-293`, `366-373`); `null` when no probabilities were kept | **bug: the tool read only the OpenAI legacy layout, every prefix failed** (fixed, findings 1-3) |
| `tools/gateway/gateway.py` | proxies `GEN_PATHS` `/v1/chat/completions`, `/v1/completions`, GET `/v1/models`, POST `/tokenize`, `/detokenize` (`gateway.py:55-58`); context guard: `/tokenize {"content", "add_special": true}` (`gateway.py:656`) after `/apply-template` for chat (`gateway.py:686`); backend `GET /health` | `server.cpp:291-292`; `/apply-template` answers `{"prompt"}` (`server-context.cpp:5585-5597`); a closed connection cancels the task (`server-queue.cpp:610` logs `cancel task`) | works: 9 smoke checks (section 3.3) and a longer session (section 3.4) |
| `tools/kv/slot_persist.py` | POST `/completion` `{"prompt", "n_predict", "cache_prompt", "id_slot", "return_tokens"}` (`slot_persist.py:42-44`), POST `/slots/0?action=erase|save|restore` (`slot_persist.py:54-58`); `timings.prompt_n`, `tokens_cached` | `server.cpp:270, 304`; `return_tokens` `server-schema.cpp:34`; `tokens_cached` `server-task.cpp:348` | works; continuation compared by token ids since this branch (finding 6) |
| `scripts/r1/run_r1.py` | engine argv flags against `--help` (`run_r1.py:1156`); GET health URL until 200 (`run_r1.py:1331`); SIGTERM to the engine's group (`run_r1.py:1378`); tool outputs under `artifacts/` | `/health` 503 `Loading model` until ready (`server-http.cpp:312-320`), then 200 | works; **bug: `artifacts/quality/` was never created, logprob_compare died writing its result** (fixed, finding 5) |

## 2. What a tiny model can and cannot exercise

- **Id-only vocabulary (the tinygen default, N2/N6/N7):** text is refused and the slot is capped at the model's 512-token
  context (D1 below). None of the tools above can run on it. CONFIRMED.
- **`--text-vocab` (new):** byte-level BPE (286 tokens: `<unk>`, `<s>`, `</s>`, the 256 bytes, 27 merges), context length
  set by `--ctx`, no chat template (llama-server falls back to its built-in chatml, `common/chat.cpp` `CHATML_TEMPLATE_SRC`
  @79e2e74). It exercises every request and response format, tokenization, prompt fitting, streaming, usage and cache
  accounting, context checkpoints, slots, logprobs including pieces that are not valid UTF-8, the gateway and the
  orchestration. CONFIRMED by the runs.
- **It cannot exercise:** quality (random weights: probe scores are 0, tools/json probes fail); speed; the real chat
  templates and their tool-call parsing (gap 6); real tokenizer properties (here `" the"` is 3 tokens; in the byte-level BPE
  vocabularies of the target models it is believed to be 1 - PROVISIONAL, the tokenizer files are not in the repository);
  262K lengths in reasonable time on this CPU (a cold 32768-token prefill took 9.8 s of wall time, CPU tiny model); MTP;
  GPU anything. These are NOT RUN, not inferred.

## 3. Runs (CONFIRMED on CPU, tiny models; literal)

### 3.1 Demonstrations, old tool (`c2f6c91`) against fixed tool on the same server

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n11-demos.py <scratch>/llama.cpp/build-cpu/bin/llama-server <scratch>/n11/demos <scratch>/n7/qwen4exp.gguf <scratch>/n11/models/qwen4exp_text.gguf
== D1 id-only vocabulary (tinygen default): text is refused, the slot is capped at context_length 512
POST /tokenize {"content": "The quick brown fox.", "add_special": false} -> 500 {"error":{"code":500,"message":"unordered_map::at","type":"server_error"}}
POST /tokenize {"prompt": "The quick", "add_special_tokens": false} -> 200 {"tokens":[]}
d_model: the slot context (1024) exceeds the training context of the model (512) - capping

== D2 text vocabulary (--text-vocab): /tokenize, logprobs layout of /v1/completions, two ids with one text
POST /tokenize 'The quick brown fox.' -> 200 {"tokens":[87,107,104,275,120,108,102,110,260,117,114,122,113,264,114,123,49]}
POST /tokenize ' the' -> 200 {"tokens":[278,107,104]}
POST /v1/completions logprobs=2 -> 200; choices[0] keys ['finish_reason', 'index', 'logprobs', 'text']; logprobs keys ['content']
  choices[0].logprobs.content[0] = {"id": 252, "token": "�", "bytes": [249], "logprob": -4.418757915496826, "top_logprobs": [{"id": 252, "token": "�", "bytes": [249], "logprob": -4.418757915496826}, {"id": 50, "token": "/", "bytes": [47], "logprob": -4.443063259124756}]}
POST /detokenize {"tokens": [200]} -> 200 '{"content":"�"}' (content '�')
POST /detokenize {"tokens": [201]} -> 200 '{"content":"�"}' (content '�')

== D3 logprob_compare before (c2f6c91) and after, ref -fa off vs cand -fa on, 10 prefixes
before: exit 1: no successful prefixes (10 errors) -> <scratch>/lc-before.json
  first error: ValueError('http://127.0.0.1:59489: response has no top_logprobs (engine does not support logprobs on /v1/completions?)')
after: exit 0: 7 prefixes (3 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0000 median 0.0000 p95 0.0000 max 0.0000 -> <scratch>/lc-after.json
  first error: ValueError("http://127.0.0.1:59489: response has no top_logprobs (a token was generated ('�') but no logprobs came back: the engine ignores logprobs, 

== D4 run_matrix before (c2f6c91) and after: the R1 template's smoke-2k, then ctx-2k's colliding cell, same server
before smoke-2k cold 2048/128 rep 0: prompt_tokens 2046, server cached_tokens 0
before ctx-2k   cold 2048/128 rep 0: prompt_tokens 1918, server cached_tokens 1530
after smoke-2k cold 2048/128 rep 0: prompt_tokens 2046, server cached_tokens 0
after ctx-2k   cold 2048/128 rep 0: prompt_tokens 1918, server cached_tokens 0
exit=0
```
(the D1 capping line is the last 90 characters of the server log line, as the script prints it; `<scratch>/lc-*.json` are
under `<scratch>/n11/demos/`, the script's own path rewrite shortened them; the D3 error text is cut at 150 characters by
the script.)

### 3.2 logprob_compare: `-fa off` (reference) against `-fa on` (candidate), 200 prefixes, both architectures

Servers `-c 8192 -np 1 -t 2 --no-warmup`, KV cache at the default f16, flash attention as stated; corpus = three
`prompts.Filler` texts (prose, code, logs; 400 units each, seed `n11-corpus`); tool at `2d59d65`.
```
$ python3 tools/quality/logprob_compare.py --ref http://127.0.0.1:18971 --cand http://127.0.0.1:18972 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n11/run/lc2-qwen4exp-faoff-vs-faon.json   # qwen4exp: ref -fa off, cand -fa on
170 prefixes (30 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0000 median 0.0000 p95 0.0000 max 0.0000 -> <scratch>/n11/run/lc2-qwen4exp-faoff-vs-faon.json
exit=0
  errors: 30, all 'incomplete UTF-8': True; merged keys ref/cand: 0/0; greedy keys: text 109, bytes: 60, id: 1
  summary: n 170 top1_agree 1.0000 ref_in_cand_topk 1.0000 kl_topk mean 4.422e-09 median 2.986e-09 p95 1.721e-08 max 3.070e-08
$ python3 tools/quality/logprob_compare.py ... --out <scratch>/n11/run/lc2-glm5next-faoff-vs-faon.json   # glm5next: ref -fa off, cand -fa on
166 prefixes (34 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0001 median 0.0000 p95 0.0000 max 0.0118 -> <scratch>/n11/run/lc2-glm5next-faoff-vs-faon.json
exit=0
  errors: 34, all 'incomplete UTF-8': True; merged keys ref/cand: 0/0; greedy keys: text 113, bytes: 51, id: 2
  summary: n 166 top1_agree 1.0000 ref_in_cand_topk 1.0000 kl_topk mean 1.382e-04 median 2.033e-07 p95 2.493e-06 max 1.177e-02
```
(the second command line is shortened with `...`: it is the first one with `glm5next` in place of `qwen4exp`; the two
indented lines after each run are a Python read-back of the result file.) Before `2d59d65` the same runs merged 2 and 3
keys (control tokens printed with an empty piece, finding 2). The glm5-next KL tail (max 1.2e-2, same greedy token on
every prefix) is of the same order as N7/N10's same-prompt precision difference of that model - not a measurement of a
real model (PROVISIONAL relevance).

### 3.3 The CI smoke: every tool through its CLI (`run_n11.sh`, fresh build)

```
$ tools/srvcheck/run_n11.sh <scratch>/llama.cpp <scratch>/n11/build-n11 <scratch>/n11/ci-out
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (pins.json)
CMAKE_BUILD_TYPE=Release
version: 0.6.0-dev (build 1, commit 79e2e74)
wrote <scratch>/n11/ci-out/qwen4exp_text.gguf
e9555b378dddacabd773fe3621823b440dc1b28823d94fc7486c04ab86c7df81  <scratch>/n11/ci-out/qwen4exp_text.gguf
tokenize oa_client detects the llama.cpp /tokenize dialect              PASS  style llamacpp, probe [('llamacpp', 36)]
tokenize prompts.py fits 2000 tokens with the server's /tokenize (<= target, within 3) PASS  tokens 2000 exit 0
matrix   m1: exit 0, no failed request, not flagged as mock             PASS  exit 0 failures 0 is_mock False
matrix   m1: usage and server cached tokens on every request            PASS  6 requests, prompt_tokens [275, 562, 842, 843]
matrix   m2: exit 0, no failed request, not flagged as mock             PASS  exit 0 failures 0 is_mock False
matrix   m2: usage and server cached tokens on every request            PASS  6 requests, prompt_tokens [274, 561, 842, 843]
matrix   second run: the cold request is not served from the cache      PASS  cached [0] of [1024]; salts c62a32f3dd90d592 / b2a5d49ce3953c1d
probes   every probe answered (no request error)                        PASS  probes ['codeqa', 'determinism', 'json', 'multihop', 'needle', 'tools'] errors 0 exit 1 (1 = scores below thresholds: random weights)
probes   json: response_format accepted by the server                   PASS  accepted True
slot     save/erase/restore: identical continuation by token ids, warm re-processes few tokens PASS  exit 0 identical True compared ['tokens'] warm prompt_n 4.0 of 2000.0
logprob  llama-server layout parsed; every error is the known incomplete-UTF-8 one PASS  28 rows, 12 errors (0 other), top1_agree 1.0, merged keys 0
gateway  /health without a key: 200, backend reachable                  PASS  HTTP 200
gateway  /v1/models: 401 without the key, the server's list with it     PASS  HTTP 401 / 200
gateway  /tokenize proxied                                              PASS  HTTP 200 b'{"tokens":[75,104,111,111,114,281,114,117,111,103,49]}'
gateway  chat stream via the gateway == direct (minus id/created/timings) PASS  events 20 / 20
gateway  context guard: completion and chat over 4096 tokens -> 400, never reach a slot PASS  HTTP 400/400 ['context_length_exceeded', 'context_length_exceeded'], slot launches 67 -> 67
gateway  client gone mid-stream: the gateway drops the backend, llama-server cancels the task PASS  slot idle after 0.1 s; server log '.10.133.038 W srv          stop: cancel task, id_task = 1235'
gateway  queue: 1 running + 2 queued, the 4th request gets 429 + Retry-After PASS  [(429, '5'), (200, None), (200, None), (200, None)]
gateway  request log: usage from the server, the disconnect recorded, no prompt text PASS  11 records, 4 generations (usage_src ['timings', 'usage']), 1 disconnected
gateway  SIGTERM: drains and exits 0                                    PASS  exit 0
r1       run_r1 started, health-checked and stopped the engine; steps as expected PASS  exit 1 (1: probes fail on random weights); unexpected none
r1       pack built and verified; verdict not PASS                      PASS  verdict FAIL (mandatory criteria FAIL: agent_api_probes)
r1       engine stopped by SIGTERM with exit code 0                     PASS  signals ['SIGTERM'] exit 0
r1_tools_smoke: 23/23 checks PASS
smoke wall time 34 s (CPU, tiny model: not a measurement)
artifacts: <scratch>/n11/ci-out
real	4m7.617s
user	12m33.821s
sys	0m51.419s
exit=0
```
(cut: the CMake warning that the web UI assets are not available offline, 5 lines, printed by every configure of this
checkout; `real` includes the from-scratch llama-server build.)

The same smoke with the four fixed PR #1 tools put back to their `c2f6c91` versions (scratch copy of HEAD, nothing else
changed) fails exactly at the four bugs:
```
$ python3 tools/srvcheck/r1_tools_smoke.py --server <scratch>/n11/build-n11/bin/llama-server --model <scratch>/n11/ci-out/qwen4exp_text.gguf --workdir <scratch>/n11/oldtree-work --out <scratch>/n11/oldtree-smoke.json   # run in <scratch>/n11/oldtree
matrix   second run: the cold request is not served from the cache      FAIL  cached [1019] of [1023]; salts None / None
slot     slot_persist.py                                                FAIL  KeyError: 'compared'
logprob  llama-server layout parsed; every error is the known incomplete-UTF-8 one FAIL  0 rows, 40 errors (40 other), top1_agree None, merged keys 0
r1       run_r1 started, health-checked and stopped the engine; steps as expected FAIL  exit 1 (1: probes fail on random weights); unexpected {'logprob_compare': ('failed', 'ok')}
r1_tools_smoke: 19/23 checks PASS
```
(cut: the 19 PASS lines, identical in wording to the run above; the `time` lines.)

### 3.4 The gateway in front of the real server

Section 3.3 lists the session as the smoke runs it (auth on with one key, `--max-concurrency 1 --max-queue 2
--max-context-tokens 4096 --set-default cache_prompt=true`). A longer interactive session earlier the same day
(scratch script, 16 checks, same server build and model) also showed: the guard's count for a chat request equals the
server's own prompt count (`guard: ok:74+24<=4096` in the gateway log, `prompt_tokens 74` = `prompt_n + cache_n` from the
server's timings); `/tokenize` in the vLLM dialect answers `{"tokens":[]}` through the gateway (a miss, as `oa_client.py`
expects); a stream cut by the client is logged by the gateway as status 200 with `client_disconnected: true` and error
`client disconnected (streaming)` (headers were already sent, `gateway.py:1144`). CONFIRMED (literal gateway log record
read back from `<scratch>/n11/run/gw/requests.jsonl`):
```
{'path': '/v1/chat/completions', 'stream': True, 'status': 200, 'queue_s': 1e-05, 'ttft_s': 0.008053, 'total_s': 0.033126, 'prompt_tokens': 74, 'completion_tokens': 24, 'cached_tokens': 70, 'usage_src': 'timings', 'guard': 'ok:74+24<=4096', 'injected_defaults': [], 'client_disconnected': False, 'error': None}
{'path': '/v1/completions', 'stream': True, 'status': 200, 'queue_s': 5e-06, 'ttft_s': 0.007417, 'total_s': 0.015594, 'prompt_tokens': None, 'completion_tokens': None, 'cached_tokens': None, 'usage_src': None, 'guard': 'ok:13+3500<=4096', 'injected_defaults': ['cache_prompt'], 'client_disconnected': True, 'error': 'client disconnected (streaming)'}
```
(two of the 14 records of the first pass of that session.) The run_matrix and probes CLIs also ran through the gateway with
the key from `OPENAI_API_KEY` (`--url http://127.0.0.1:18950/v1` and `--url http://127.0.0.1:18950`): `status ok, 11 raw
rows, 0 failures` and `status ok, all hard probes pass: False` (random weights), token counting `server-tokenize`,
`is_mock False`.

### 3.5 run_r1.py with `llamacpp-server.json`

**Run A - the template as committed, paths overridden with `--set` as its README describes.** The first model file is a
symlink named like the template's shard 1 pointing at `qwen4exp_text.gguf`:
```
$ python3 scripts/r1/run_r1.py --config scripts/r1/configs/llamacpp-server.json --out <scratch>/n11/r1runs --set LLAMA_BIN=<scratch>/llama.cpp/build-cpu/bin --set WEIGHTS_DIR=<scratch>/n11/weights --set SLOT_DIR=<scratch>/n11/state/slots --set HEALTH_TIMEOUT_S=120
[r1] run directory: <scratch>/n11/r1runs/20261011T034953Z-llamacpp-qwen38fn
[r1] weights_verify                     failed (exit 1): exit code 1 (accepted: 0)
[r1] quant_audit                        failed (exit 1): exit code 1 (accepted: 0); stdout is not JSON, kept as artifacts/quant/quant_audit.stdout.txt
[r1] roofline_prediction[ctx=2048]      ok (exit 0)
[r1] roofline_prediction[ctx=32768]     ok (exit 0)
[r1] roofline_prediction[ctx=131072]    ok (exit 0)
[r1] roofline_prediction[ctx=262144]    ok (exit 0)
[r1] roofline_prediction[record]        ok
[r1] nvidia_smi_before                  skipped: nvidia-smi not found on this host
[r1] engine_build_check                 failed: cannot read <scratch>/llama.cpp/build-cpu/build-info.json: [Errno 2] No such file or directory: '<scratch>/llama.cpp/build-cpu/build-info.json'
[r1] engine_version[version]            ok (exit 0)
[r1] engine_version[help]               ok (exit 0)
[r1] engine_start                       skipped: depends on engine_build_check, which failed
[r1] health_wait                        skipped: depends on engine_start, which was skipped (dependency failed)
[r1] health_wait[after-ready-1]         skipped: depends on engine_start, which was skipped (dependency failed)
[r1] health_wait[after-ready-2]         skipped: depends on engine_start, which was skipped (dependency failed)
[r1] matrix[smoke-2k]                   skipped: depends on engine_start, which was skipped (dependency failed)
[r1] matrix[ctx-2k]                     skipped: depends on engine_start, which was skipped (dependency failed)
[r1] matrix[ctx-32k]                    skipped: depends on engine_start, which was skipped (dependency failed)
[r1] matrix[ctx-128k]                   skipped: depends on engine_start, which was skipped (dependency failed)
[r1] matrix[filled-262k]                skipped: depends on engine_start, which was skipped (dependency failed)
[r1] probes[32k-128k-262k]              skipped: depends on engine_start, which was skipped (dependency failed)
[r1] slot_persist[prompt]               skipped: depends on engine_start, which was skipped (dependency failed)
[r1] slot_persist[persist]              skipped: depends on engine_start, which was skipped (dependency failed)
[r1] logprob_compare                    skipped: no reference endpoint ("reference": {"url": ...}): V6 stays NOT RUN
[r1] vs_roofline                        skipped: no matrix run directory with summary.json and meta.json
[r1] engine_stop                        skipped: the engine was not started by this run
[r1] nvidia_smi_after                   skipped: nvidia-smi not found on this host
[r1] pack[build]                        ok (exit 1)
[r1] pack[check]                        ok (exit 0)
[r1] steps: weights_verify=failed, quant_audit=failed, roofline_prediction=ok, nvidia_smi_before=skipped, engine_build_check=failed, engine_version=ok, engine_start=blocked, health_wait=blocked, matrix=blocked, probes=blocked, slot_persist=blocked, logprob_compare=skipped, vs_roofline=skipped, engine_stop=skipped, nvidia_smi_after=skipped, pack=ok
[r1] verification pack: FAIL (mandatory criteria FAIL: weights_integrity) -> pack/VERDICT.md
[r1] journal: <scratch>/n11/r1runs/20261011T034953Z-llamacpp-qwen38fn/steps.jsonl
```
Artifacts read back: `artifacts/weights/weights_verify.txt` ends with `SIZE ... 00001-of-00004.gguf: 3104256 != 10946624`,
three `MISSING` lines and `verify: 0/4 files OK`; `quant_audit` stopped with `FileNotFoundError: missing shards: [...00002-of-00004.gguf, ...]`;
`artifacts/engine/flags_check.json` has `"missing_from_help": []` for all 14 flags of the template's engine argv. Every
safety rule did its job: the stand-in is not the manifest's file (V2 FAIL is the correct verdict for these files), a split
set with missing shards is not audited, and an engine whose build is not proven to be the pin is never started.

**Run B - a config derived from the template for this CPU** (`r1_tools_smoke.py derive-config --lengths evidence`):
the engine argv is the template's (`-c 262144 -np 1 -ngl all -sm layer -ts 1,1,1,1 -fa on -ctk f16 -ctv f16 --jinja
--metrics --slot-save-path ...`; the CPU build ignores the three GPU flags with warnings); differences: name; model = the
SYNTHETIC file without a weights manifest; no `engine.build_info` (so `engine_build_check` is skipped and run_r1 starts
the engine itself); matrix 2K/2K/8K/16K instead of 2K/2K/32K/128K/262K (smoke-2k and ctx-2k unchanged); probes at 4K
and 16K; `slot_persist.prompt_file` = a prompt fitted earlier with `prompts.py` (8191 tokens, see finding 9);
`logprob_compare` `--max-chars 3000`. Reference endpoint: a second server on the same file with `-fa off`.
```
$ python3 scripts/r1/run_r1.py --config <scratch>/n11/configs/n11-evidence.json --out <scratch>/n11/r1runs-b2 --set LLAMA_BIN=<scratch>/llama.cpp/build-cpu/bin --set SLOT_DIR=<scratch>/n11/state-b2/slots --set HEALTH_TIMEOUT_S=120 --set REFERENCE_URL=http://127.0.0.1:18971 --set "CORPUS=<scratch>/n11/corpus/*.txt"
[r1] run directory: <scratch>/n11/r1runs-b2/20261011T040616Z-n11-llamacpp-cpu-tiny
[r1] weights_verify                     skipped: no model.weights_manifest in the config: weights integrity (V2) stays NOT RUN
[r1] quant_audit                        ok (exit 0)
[r1] roofline_prediction[ctx=2048]      ok (exit 0)
[r1] roofline_prediction[ctx=8192]      ok (exit 0)
[r1] roofline_prediction[ctx=16384]     ok (exit 0)
[r1] roofline_prediction[record]        ok
[r1] nvidia_smi_before                  skipped: nvidia-smi not found on this host
[r1] engine_build_check                 skipped: no engine.build_info: the build of pin 'llama.cpp' is not checked
[r1] engine_version[version]            ok (exit 0)
[r1] engine_version[help]               ok (exit 0)
[r1] engine_start                       ok: started (pid 13462); health is checked by health_wait
[r1] health_wait                        ok: http://127.0.0.1:53985/health answered HTTP 200 after 5.0 s
[r1] health_wait[after-ready-1]         ok (exit 0)
[r1] health_wait[after-ready-2]         skipped: nvidia-smi not found on this host
[r1] matrix[smoke-2k]                   ok (exit 0)
[r1] matrix[ctx-2k]                     ok (exit 0)
[r1] matrix[ctx-8k]                     ok (exit 0)
[r1] matrix[filled-16k]                 ok (exit 0)
[r1] probes[4k-16k]                     failed (exit 1): exit code 1 (accepted: 0)
[r1] slot_persist[persist]              ok (exit 0)
[r1] logprob_compare                    ok (exit 0)
[r1] vs_roofline                        ok (exit 0)
[r1] engine_stop                        ok (exit 0)
[r1] nvidia_smi_after                   skipped: nvidia-smi not found on this host
[r1] pack[build]                        ok (exit 1)
[r1] pack[check]                        ok (exit 0)
[r1] steps: weights_verify=skipped, quant_audit=ok, roofline_prediction=ok, nvidia_smi_before=skipped, engine_build_check=skipped, engine_version=ok, engine_start=ok, health_wait=ok, matrix=ok, probes=failed, slot_persist=ok, logprob_compare=ok, vs_roofline=ok, engine_stop=ok, nvidia_smi_after=skipped, pack=ok
[r1] verification pack: FAIL (mandatory criteria FAIL: agent_api_probes) -> pack/VERDICT.md
[r1] journal: <scratch>/n11/r1runs-b2/20261011T040616Z-n11-llamacpp-cpu-tiny/steps.jsonl

real	2m37.835s
user	8m12.803s
sys	0m8.115s
exit=1
```
Run B2 used the tools at `0f23aa6` (HEAD when it started); its config came from `r1_tools_smoke.py`, then uncommitted:
`derive-config --lengths evidence` at `5ea2ea7` writes the same bytes (`cmp` silent against the config file and against
the run's `artifacts/orchestrator/config.json`). An earlier dry run with a hand-written config that differed only
in the `model` block (`<scratch>/n11/r1runs-b/20261011T040124Z-...`, before `0f23aa6`) failed the `logprob_compare` step
with `FileNotFoundError: ... artifacts/quality/logprob_compare.json` after computing all 200 prefixes - finding 5.

Read back from Run B2 (Python one-off over its artifacts): `exit.json` `{'exit_code': 0, 'signals_sent': ['SIGTERM'],
'group_leftovers_killed': False, ...}`; flags missing from `--help`: `[]`; nonce salts of smoke-2k / ctx-2k
`54a2e15ca5539a0d` / `77a76083b66af7dd`, their cold 2048/128 cells `cached_tokens 0` and `0`; slot_persist `runs 3,
errors 0, cold_prompt_n_p50 8191, warm_prompt_n_p50 4, all_continuations_identical True`, compared by `tokens`;
logprob_compare `rows 170, errors 30` (all the incomplete-UTF-8 kind), `top1_agree 1.0`; quant_audit `PASS`, architecture
`qwen4exp`, types `['F32']`; probes: needle/multihop/codeqa at 4K and 16K score 0.0, json 0.071, tools 0.0, determinism
1.0, no request errors; every `vs_roofline` row has ratio > 1, which by that tool's own rule means its inputs are wrong
(the bound is for Qwen3.8-Flash-Next on 4x4080, the run is a tiny model on a CPU; the numbers are not quoted here).

**Pack verdict and why, criterion by criterion (Run B2):**

| criterion | verdict | why - and why none of it is a measurement |
|---|---|---|
| single_configuration | PASS | five runs share one model/engine description. True of the files; it identifies a SYNTHETIC model |
| openai_api | PASS | `/v1/models` 200, streamed requests with usage: the server's API format, CONFIRMED - the one row that is real evidence about llama-server @79e2e74 (CPU build) |
| context_probes_full_length, output_after_full_prompt | NOT RUN | not at 262144 tokens |
| agent_api_probes | **FAIL** | tools 0.0, json 0.071 < 1.0: random weights cannot call tools. A correct verdict for this model; says nothing about any real model |
| decode_filled_ctx_floor / _target, prefix_cache, decode_vs_roofline | NOT RUN | no cell at filled context (262144) |
| agent_loop | NOT RUN | reached 15097 prompt tokens < 100000 |
| quant_min_bpw | PASS | the tiny file is all F32 (32 bpw); meaningless for the target weights |
| weights_integrity | NOT RUN | no manifest in the derived config (Run A: FAIL, the stand-in is not the manifest's file) |
| exact_routing_logprobs | NOT RUN | 30 of 200 prefixes failed (llama-server sent no logprobs, finding 3); the pack never averages errors away |
| kv_slot_persist | NOT RUN | 8191-token state < 100000 |
| hardware_facts | NOT RUN | no `scripts/diag` summary |
| determinism | PASS (informational) | identical output twice at temperature 0 |
| manual_* | NOT RUN | manual |

The expected verdict NOT RUN holds only for mock artifacts: the pack recognises the mock server (`owned_by`), not a
synthetic model served by a real engine; the SYNTHETIC markers it collects are informational (finding 7).

## 4. Fixes to PR #1 tools (for the session that owns `claude/relaxed-dijkstra-1d7i5o`)

All four tools and their tests are byte-identical at this branch's base `c2f6c91` and at PR #1's head as known locally
(`259081e`; `git diff --quiet c2f6c91 origin/claude/relaxed-dijkstra-1d7i5o -- <file>` for each file, no fetch), so each
commit applies there as it is. Each fix has a test that fails before it (literal outputs in section 7).

| commit | tool | what broke on the real server | fix |
|---|---|---|---|
| `026b0e9` | `tools/quality/logprob_compare.py` | every prefix: "response has no top_logprobs" - llama-server's layout `logprobs.content[0].top_logprobs` was not read; distinct byte tokens share the text U+FFFD | read both layouts; key tokens by raw bytes (text when valid UTF-8, else `bytes:<hex>`); duplicates add probabilities; null-logprobs answer stays an error, with the real reason |
| `2d59d65` | same | control tokens come back with an empty piece, so EOS and others merged under one key | empty piece with an id -> `id:<n>`; error text names both causes of `"logprobs": null` |
| `92e3886` | `tools/bench/run_matrix.py` | a second run with the same `--seed` (and R1's ctx-2k after smoke-2k) re-sent byte-identical "cold" prompts, served from the server's cache (1019 of 1023 / 1530 of 1918 tokens) | `--nonce-salt`, random per run and recorded in `meta.json`, in the cold/warm nonce and the agent session nonces; the same salt reproduces a run. BENCH_PROTOCOL.md "cold" updated |
| `ffec637` | `tools/kv/slot_persist.py` | not a failure: the continuation check compared decoded text, a lossy view (ids 200 and 201 both read U+FFFD) | `return_tokens` and comparison by token ids when the server returns them (`compared` recorded) |
| `0f23aa6` | `scripts/r1/run_r1.py` | `artifacts/quality/` never created: logprob_compare computed 200 prefixes and died writing its result; V6 could not be produced through run_r1 | `mkdir artifacts/quality` before the step; the existing failure-path test now requires the artifact |

The commit message of `92e3886` says the BENCH_PROTOCOL.md line is in "section 3"; it is in section 4 ("Матрица и
сценарии"). History is not rewritten.

## 5. Findings

1. **logprob_compare could not read llama-server's logprobs** (CONFIRMED, D3 and section 3.2; fixed in `026b0e9`).
   Without the fix, V6 could never be measured with llama.cpp on either side.
2. **Token identity in logprobs is the byte string, not the text** (CONFIRMED: D2 shows id 252 printed as U+FFFD with
   `bytes [249]`; `/detokenize` of 200 and of 201 both answer U+FFFD; control tokens have an empty piece). Fixed in
   `026b0e9` and `2d59d65`; after the fix both 200-prefix runs merged 0 keys.
3. **llama-server sends no logprobs for a token whose text ends in an incomplete UTF-8 sequence** - `slot.add_token`
   is inside `if (!incomplete)` (`tools/server/server-context.cpp:1977-2009` @79e2e74) and `logprobs` stays `null`
   (`server-task.cpp:368-372`) - a server limitation (CONFIRMED by reading and observed: 30 and 34 of 200 prefixes on the
   random byte-level models). The tool records such a prefix as an error and the pack then says V6 NOT RUN (zero errors
   allowed). How often a real model's greedy token is such a piece: UNKNOWN; PROVISIONAL guess: rare on English and code,
   more likely on multi-byte text (Cyrillic, CJK) if the vocabulary splits characters. The rule was not changed;
   whether to choose the corpus accordingly or to tolerate these errors is the lead's decision.
4. **Cold cells were not cold across runs** (CONFIRMED, D4 and 3.3; fixed in `92e3886`). In
   `scripts/r1/configs/llamacpp-server.json` the ctx-2k cell 2048/128 rep 0 shares its key with smoke-2k; any re-run
   against an engine started outside run_r1 repeats every cold prompt and the agent loop's cold control. The pack's
   `prefix_cache` criterion takes cold TTFT at face value (`tools/verify/pack.py:691-721`), so such a run would have
   skewed the warm/cold ratio. Recommendation (not done): the pack could refuse a cold cell whose server-reported cached
   tokens exceed the template head.
5. **run_r1 never created `artifacts/quality/`** (CONFIRMED, Run B dry run; fixed in `0f23aa6`). The mock config has no
   reference endpoint, and the failure-path test expected exit 1, which the crash also produced.
6. **slot_persist compared decoded text** (CONFIRMED lossy, no false pass observed; hardened in `ffec637`).
7. **The pack does not know a synthetic model served by a real engine.** On Run B2 it says FAIL (`agent_api_probes`)
   with PASS for `single_configuration`, `openai_api`, `quant_min_bpw`, `determinism`; the `SYNTHETIC` model description
   is listed under "synthetic markers" only (`tools/verify/pack.py:1366-1374`). CONFIRMED. Not changed (verdict rules
   belong to the pack's owner); a possible rule: treat a SYNTHETIC model description like the mock (NOT RUN, reason
   `synthetic`).
8. **The R1 llama.cpp template on the pinned build:** every flag of its engine argv is in `llama-server --help`; the CPU
   build accepts `-ngl all -sm layer -ts 1,1,1,1` with warnings and gives the slot the full 262144 context of the
   text-vocab model; run_r1 starts it in its own process group, `/health` answers 200 (first poll after 5 s, the
   template's poll interval; the 503 `Loading model` phase was too short to observe with a 4 MB model - CONFIRMED only by
   reading), and SIGTERM stops it with exit code 0. CONFIRMED on CPU; on a GPU build NOT RUN.
9. **`prompts.py` exact fits rely on the pad piece `" the"` being one token** (`prompts.py:33`). With the tiny vocabulary
   it is 3 tokens (D2), fits end 1-2 tokens short (8192 -> 8191, 16384 -> 16382, 16000 -> 15999, 12000 -> 11999, 8000
   -> 7999) and the CLI exits 1, so run_r1's `slot_persist[prompt]` step would fail and skip the persist step (V7 NOT RUN).
   Run B therefore used `slot_persist.prompt_file`. Whether `" the"` is one token in the target tokenizers: UNKNOWN (not in
   the repository). Not changed.
10. **Gateway on llama-server:** the context guard (`/apply-template` + `/tokenize` with `add_special: true`) counted 74
    tokens for a chat request for which the server reported 74 prompt tokens; a client that leaves mid-stream makes the
    gateway close the backend connection and llama-server cancels the task (`stop: cancel task`, slot idle within 0.1 s)
    - the gateway README's PROVISIONAL statement is CONFIRMED on CPU at 79e2e74; queue limit, 429 + `Retry-After`, auth,
    byte-identical streaming (minus `id`/`created`/`timings`) and SIGTERM drain work against the real server. CONFIRMED.
11. **Forced decode length on llama.cpp:** `run_matrix --eos-mode auto` chose the `vllm` key set (`ignore_eos` +
    `min_tokens`): llama-server honours `ignore_eos` and ignores the unknown `min_tokens`, so lengths are forced;
    "verified" is False here because random weights never stop early. CONFIRMED (B2: `eos vllm/False` in every matrix
    summary; all 38 measured requests have `completion_tokens` = their output length, `n_short_output` 0); the mode label
    is misleading but harmless (PROVISIONAL).
12. **Several tokens per SSE chunk on byte output:** llama-server holds back an incomplete UTF-8 sequence, so a chunk can
    carry 2+ tokens (1.0-1.6 tokens per chunk per request in the first run_matrix run on the random byte output); run_matrix's decode formula already accounts for
    tokens per chunk (`runlog.py:136-141`). CONFIRMED.
13. **Prefix reuse of a hybrid model in run_matrix's warm state** is limited to the checkpoint `4 + n_ubatch` tokens before
    the end of the priming prompt (cached 508 of 1024 in the first run_matrix run), as N7 finding 4 predicts. CONFIRMED on
    qwen4exp; at 262K that leaves ~516 tokens to re-process (PROVISIONAL, from the mechanism).

## 6. CI

`.github/workflows/server-cpu.yml`, job `n7`: the unit-test step adds `tests.test_r1_tools_smoke`; after the llama.cpp
checkout a step runs `tests.test_tinygen_text_vocab` with `LLAMA_CPP_GGUF_PY=../llama.cpp/gguf-py`; after `run_n7.sh` the
step `tools/srvcheck/run_n11.sh ../llama.cpp ../build-n7 ...` reuses N7's build. Cost on this sandbox (4 cores, CPU tiny
model wall times, not measurements): smoke 34 s; model generation not timed separately; the build is N7's. On GitHub's runner:
UNKNOWN until it runs (NOT RUN: this branch is not pushed by this task). The trigger paths now include `tools/bench/**`,
`tools/quality/**`, `tools/gateway/**`, `tools/verify/**`, `scripts/r1/**`.

## 7. Tests (literal)

Before each fix (the new or extended tests against the tool as it was; filtered with `grep -E '^(FAIL|ERROR|Ran|OK|FAILED|...)'`):
```
$ python3 -m unittest tests.test_logprob_compare -v 2>&1 | tail -30          # first 3 new tests, tool at c2f6c91
...
Ran 9 tests in 4.922s

FAILED (failures=3)
$ (tool of c2f6c91 injected as module logprob_compare) unittest tests.test_logprob_compare   # the 10 tests of 026b0e9
FAIL: test_answer_without_logprobs_is_an_error_that_says_why
FAIL: test_dialects_compare_with_each_other
FAIL: test_llamacpp_logprobs_dialect
FAIL: test_tokens_with_the_same_text_but_different_bytes_stay_apart
Ran 10 tests in 5.867s
FAILED (failures=4)
$ (mutant: key = printed text) unittest tests.test_logprob_compare
FAIL: test_tokens_with_the_same_text_but_different_bytes_stay_apart
Ran 10 tests in 4.589s
FAILED (failures=1)
$ (tool of 026b0e9 injected) unittest tests.test_logprob_compare            # the 11 tests of 2d59d65
FAIL: test_control_tokens_with_an_empty_piece_stay_apart
Ran 11 tests in 5.168s
FAILED (failures=1)
$ python3 -m unittest tests.test_bench.Matrix.test_second_run_against_the_same_server_is_still_cold tests.test_bench.Matrix.test_nonce_salt_reproduces_a_run
ERROR: test_second_run_against_the_same_server_is_still_cold
KeyError: 'nonce_salt'
ERROR: test_nonce_salt_reproduces_a_run
Ran 2 tests in 0.357s
FAILED (errors=2)
$ (mutants in scratch copies) test_second_run_against_the_same_server_is_still_cold
== base     Ran 1 test in 0.366s / OK
== matrix   AssertionError: 399 not less than 8 : {'cached_tokens': 399, ... 'prompt_tokens': 400, ...}   (salt dropped from the cold nonce)
== agent    AssertionError: 576 not less than 288.5                                                     (salt dropped from the agent nonce)
$ python3 -m unittest tests.test_slot_persist      # new test, tool at 2d59d65
FAIL: test_same_text_from_other_tokens_is_not_the_same_continuation
AssertionError: True is not false
Ran 6 tests in 3.135s
FAILED (failures=1)
$ python3 -m unittest tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped   # run_r1 at ffec637
ERROR: test_failing_steps_are_recorded_and_the_engine_is_still_stopped
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/r1-fail-jlj3qb09/runs/20261011T040523Z-failing/artifacts/quality/logprob_compare.json'
Ran 1 test in 1.374s
FAILED (errors=1)
```
(the first block shows only its last lines: the three failures were the three new tests, each an `AssertionError` over
`(1, [{'prefix': 0, 'error': "ValueError('h...` - every prefix failed; the mutant lines of the run_matrix check are
shortened as shown; "injected" = the old file loaded with `importlib` as module `logprob_compare` before importing the
test module, the working tree untouched.)

After: the targeted modules, then the full suite (the command of the task, at `5ea2ea7`; 475 = 461 tests at `c2f6c91` +
14 new, counted with `unittest.defaultTestLoader.discover('tests').countTestCases()` on a `git archive` of `c2f6c91` and
on the tree):
```
$ python3 -m unittest tests.test_logprob_compare            ->  Ran 11 tests ... OK
$ python3 -m unittest tests.test_bench                      ->  Ran 40 tests in 12.962s / OK
$ python3 -m unittest tests.test_slot_persist               ->  Ran 6 tests in 3.119s / OK
$ python3 -m unittest tests.test_r1_orchestration           ->  Ran 23 tests in 11.402s / OK
$ LLAMA_CPP_GGUF_PY=<scratch>/llama-ec7630a/gguf-py python3 -m unittest tests.test_tinygen_text_vocab -v   ->  Ran 4 tests in 1.464s / OK
$ python3 -m unittest tests.test_r1_tools_smoke             ->  Ran 2 tests in 0.175s / OK
$ LLAMA_CPP_GGUF_PY=<scratch>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests
Ran 475 tests in 167.959s

OK (skipped=84)
```
(the six targeted lines are condensed from their runs' last lines; the full-suite lines are literal.)

## 8. Not covered (NOT RUN)

| what | why / where it belongs |
|---|---|
| anything on a GPU, the GPU build's `--help`, speed of anything | no GPU; R1 on the server |
| the template's matrix, probes and K3 at 32K, 128K, 262K | about 10 s per cold 32K prefill here (CPU tiny model) - hours for the template; the pack would then judge CPU tiny-model numbers; R1 on the server |
| real chat templates, tool-call parsing, reasoning output of GLM-5.3 / Qwen3.8 | gap 6; the tiny model uses llama-server's built-in chatml |
| real tokenizers (`" the"` one token?), how often logprobs are dropped on real text | tokenizer files not in the repository; R2.1 on the server |
| ik_llama.cpp, Strata, vLLM servers with these tools | only llama.cpp here (gap 5 for ik_llama.cpp) |
| the 503 `Loading model` phase of `/health` | too short with a 4 MB model; first real load on the server |
| `vs_roofline` with meaningful inputs; the pack on real artifacts (R6.2) | needs real runs |
| the CI job on GitHub | the branch is not pushed by this task |
| the gateway with HTTPS to the engine, under load | as in its README |

## Reproduce
```
git init llama.cpp && git -C llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 && git -C llama.cpp checkout FETCH_HEAD
tools/srvcheck/run_n11.sh llama.cpp llama.cpp/build-n11 /tmp/n11-out
PYTHONPATH=llama.cpp/gguf-py python3 tools/tinygen/qwen4exp_tiny.py /tmp/n11/id.gguf
python3 docs/evidence/mutation-scripts/2026-10-11-n11-demos.py llama.cpp/build-n11/bin/llama-server /tmp/n11/demos /tmp/n11/id.gguf /tmp/n11-out/qwen4exp_text.gguf
python3 tools/srvcheck/r1_tools_smoke.py derive-config --model /tmp/n11-out/qwen4exp_text.gguf --prompt-file <8K-token prompt> --lengths evidence --out /tmp/n11/evidence.json
```
