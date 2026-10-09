# Мутационные проверки инструментов: bench, gguf_quant_audit, logprob_compare, slot_persist, vs_roofline, offline weights, diag summarize, gateway (2026-10-09, облачная песочница)

Дата: 2026-10-09. Повод: docs/reassessment/REVIEW-adversarial.md, п. 11. Коммит `0daa7bc` заявлял «34 tests, 6/6 mutations
caught», но литерального вывода в `docs/` нет; ещё шесть наборов тестов добавлены без записанной мутационной проверки. Roofline
(7/7) и cachesim (4/4) уже записаны литерально в `2026-10-09-roofline-calibration-and-mutations.md` и `2026-10-09-g0-trace-cpu.md`
и здесь не повторялись.

## Условия (литерально)
`<S>` = `/tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad` (временный каталог сессии, не сохраняется).
```
$ git rev-parse --short HEAD          # at the start; the worktree was created from this commit
64fa04d
$ git -C /home/user/408032gb worktree add --detach <S>/mutwt HEAD      (first line of the output)
Preparing worktree (detached HEAD 64fa04d)
$ git rev-parse --short HEAD          # when this file was written; other agents committed in between
d89e633
$ git diff --quiet 64fa04d HEAD -- tools/bench tools/gguf_quant_audit.py tools/quality tools/kv scripts/offline scripts/diag tools/roofline.py models/specs/glm-5.3-flash.json hardware/target-4x4080-32g.unverified.json tests/fixtures; echo $?
0
$ git diff --quiet 0daa7bc 64fa04d -- tools/bench/oa_client.py tools/bench/prompts.py tools/bench/run_matrix.py tools/bench/probes.py tools/bench/runlog.py tools/bench/mock_server.py tests/test_bench.py; echo $?
0
$ python3 --version
Python 3.13.16
$ git -C <S>/llama.cpp rev-parse HEAD     # gguf-py used as LLAMA_CPP_GGUF_PY
ec7630a640789c393694fb194f1bbbf0369fc62d
```
The results therefore hold unchanged for HEAD `d89e633` (CONFIRMED: the `git diff --quiet` above). The bench tools and
`tests/test_bench.py` are byte-identical to `0daa7bc` (CONFIRMED: second `git diff --quiet`). The path `<S>/mut` named in the
task already existed as a non-git directory left by the earlier perf-bench session, so the worktree went to the sibling
`<S>/mutwt`; the old directory was not touched.

Method (each step applies to every block below):
- one mutation at a time, only in the worktree (the main checkout was never mutated): a Python `str.replace` of a fragment
  that occurs exactly once (asserted);
- `python3 -m unittest tests.<module>` from the worktree root with `PYTHONDONTWRITEBYTECODE=1` and
  `LLAMA_CPP_GGUF_PY=<S>/llama.cpp/gguf-py` (all runs except the one marked "env unset"). `__pycache__` was removed from the
  worktree before the mutation runs, because Python's pyc check uses mtime in seconds plus size, so a stale pyc from an
  earlier same-size mutation could otherwise mask one;
- recorded: the `FAIL:`/`ERROR:` lines and the `Ran`/`OK`/`FAILED` lines; then `git checkout -- <file>` and
  `git status --short -- <file>`, printed as `-> ''` when clean;
- every block shows the mutation as the `-`/`+` lines of `git diff -U0`, which is enough to redo it by hand;
- CAUGHT = the test module exits non-zero, SURVIVED = it exits 0.

Skipped tests (not counted; no counted mutation relied on them). `test_diag_kit` skips 2 tests in this sandbox:
```
test_compile_sm89 (tests.test_diag_kit.CudaCompile.test_compile_sm89) ... skipped 'nvcc not available (set NVCC=/path/to/nvcc)'
test_shellcheck (tests.test_diag_kit.StaticSafety.test_shellcheck) ... skipped 'shellcheck not available (env SHELLCHECK or PATH)'
```
`CollectAsRoot` does run here, because the sandbox runs as root. With `LLAMA_CPP_GGUF_PY` set, all 6 tests of
`test_gguf_quant_audit` run, including `test_type_table_matches_pinned_gguf_py ... ok` (verbose run). Without the variable that
test is skipped (`OK (skipped=1)`, block "G7 (env unset)").

## Сводка (CONFIRMED by the literal blocks below)
| tool | test module | caught / mutations (tests as of 64fa04d) | survived | caught after the added tests |
|---|---|---|---|---|
| `tools/bench/{oa_client,prompts,run_matrix,probes,runlog}.py` | `test_bench` | 23 / 27 | B10, B22, B23, B24 | 27 / 27 |
| `tools/gguf_quant_audit.py` | `test_gguf_quant_audit` | 7 / 8 | G8 | 8 / 8 |
| `tools/quality/logprob_compare.py` | `test_logprob_compare` | 2 / 4 | L1, L2 | 4 / 4 |
| `tools/kv/slot_persist.py` | `test_slot_persist` | 2 / 4 | S1, S3 | 4 / 4 |
| `tools/bench/vs_roofline.py` | `test_vs_roofline` | 2 / 4 | V3, V4 | 4 / 4 |
| `scripts/offline/weights.py` | `test_offline_weights` | 4 / 5 | W4 | 5 / 5 |
| `scripts/diag/summarize.py` | `test_diag_kit` | 6 / 6 | - | (no change) |
| **total** | | **46 / 58** | | **58 / 58** |

G7 is counted as CAUGHT because the pinned gguf-py was available. The extra run with the variable unset (SURVIVED) is shown
only to make that dependency visible, and is not counted.

## Мутации коммита 0daa7bc (B1-B6): откуда они и что было с исходным прогоном
The six mutations of the perf-bench session were in `<S>/mutate.py`, a session scratch file that was never in the repository.
Below is its `MUTS` list as printed by `cat <S>/mutate.py` earlier in this session (literal):
```
MUTS = [
 ("decode uses n instead of n - n/chunks", "tools/bench/runlog.py",
  'm["decode_tok_s"] = (n - n / len(gen)) / span_s', 'm["decode_tok_s"] = n / span_s'),
 ("role-only chunk counted as a token", "tools/bench/oa_client.py",
  'if n_content or n_reason or n_tool or n_text:', 'if True:'),
 ("TTFT measured at response headers", "tools/bench/runlog.py",
  'ttft_s = gen[0][0] / 1e6', 'ttft_s = (r.get("headers_ms") or 0) / 1000.0 or 1e-9'),
 ("warm uses a fresh prefix", "tools/bench/run_matrix.py",
  'contents.append(stem + SUFFIX % k)', 'contents.append(self.build(inp, out, ("cold", inp, out, conc, rep, j, "x"), k)[0])'),
 ("needle grader accepts any 6 digits", "tools/bench/probes.py",
  'correct": got == truth[k]', 'correct": got is not None'),
 ("failed requests still get timings", "tools/bench/runlog.py",
  '    if not m["ok"]:\n        return m\n', ''),
]
```
That script printed to stdout only. The scratchpad has `mutations-roofline.txt` and `mutations-cachesim.txt` but no bench
output file, and `docs/` had none (review item 11). At 23:15:09 UTC another agent overwrote `<S>/mutate.py` with its own
gateway mutation script, so the original file no longer exists. B1-B6 below were transcribed from the listing above: the same
fragments and the same replacements.

A leftover of the original run, `<S>/mut/`, has its test copy at mtime 22:40:32; commit `0daa7bc` is dated 22:43:21 +0000:
```
$ git show 0daa7bc:tests/test_bench.py > <S>/mutlogs/test_bench.0daa7bc.py
$ diff <S>/mut/tests/test_bench.py <S>/mutlogs/test_bench.0daa7bc.py | grep -c '^>'
23
$ diff <S>/mut/tests/test_bench.py <S>/mutlogs/test_bench.0daa7bc.py | grep -c '^<'
0
$ diff <S>/mut/tests/test_bench.py <S>/mutlogs/test_bench.0daa7bc.py | grep '^>.*def test_'
>     def test_variants_completions_ratio_user_format_options(self):
```
PROVISIONAL, inferred from the leftover copy: the original 6/6 run used a `test_bench.py` without
`test_variants_completions_ratio_user_format_options`, 23 lines shorter than the committed file. None of B1-B6 is caught by
that test today (blocks below), so this does not change the 6/6.

## tools/bench/{oa_client,prompts,run_matrix,probes,runlog}.py (`tests/test_bench.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_bench
Ran 34 tests in 11.063s

OK
```

Baseline in the worktree after copying in the test file with the added tests (output filtered with `grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):

```
$ python3 -m unittest tests.test_bench
Ran 38 tests in 11.348s
OK
```

| ID | file:line @64fa04d | mutation | result | caught by | after the added tests |
|---|---|---|---|---|---|
| B1 | `tools/bench/runlog.py:141` | decode uses n instead of n - n/chunks [0daa7bc set] | **CAUGHT** | `Stats.test_decode_formula_one_and_multi_token_chunks` | - |
| B2 | `tools/bench/oa_client.py:439` | role-only/empty chunk recorded as an output event (TTFT from first chunk, not first content) [0daa7bc set] | **CAUGHT** | `Client.test_completions_endpoint`<br>`Client.test_stream_timestamps_usage_and_role_chunk_not_counted` | - |
| B3 | `tools/bench/runlog.py:127` | TTFT measured at response headers instead of first output chunk [0daa7bc set] | **CAUGHT** | `Matrix.test_small_matrix_files_timings_cache`<br>`Stats.test_decode_formula_one_and_multi_token_chunks` | - |
| B4 | `tools/bench/run_matrix.py:305` | warm requests use a fresh prefix (warm == cold) [0daa7bc set] | **CAUGHT** | `Matrix.test_small_matrix_files_timings_cache` | - |
| B5 | `tools/bench/probes.py:246` | needle grader accepts any 6 digits [0daa7bc set] | **CAUGHT** | `Probes.test_wrong_answer_mutation_scores_zero` | - |
| B6 | `tools/bench/runlog.py:122` | failed requests still get timings [0daa7bc set] | **CAUGHT** | `Client.test_faults_become_errors`<br>`Stats.test_metrics_of_failed_request_are_none` | - |
| B7 | `tools/bench/runlog.py:129` | TTFC counts reasoning-only chunks as content | **CAUGHT** | `Client.test_multi_token_chunks_and_reasoning` | - |
| B8 | `tools/bench/run_matrix.py:308` | cold requests reuse one prefix per cell (cold is not cold) | **CAUGHT** | `Matrix.test_small_matrix_files_timings_cache` | - |
| B9 | `tools/bench/runlog.py:102` | server-reported cached tokens (usage.prompt_tokens_details) read as 0 | **CAUGHT** | `Matrix.test_agent_loop_prefix_cache`<br>`Matrix.test_small_matrix_files_timings_cache` | - |
| B10 | `tools/bench/runlog.py:135` | uncached_prefill_tok_s ignores the cached tokens | **SURVIVED** | - | **CAUGHT** by `Stats.test_uncached_prefill_subtracts_server_cached_tokens` |
| B11 | `tools/bench/prompts.py:310` | exact fit: single-word padding step skipped | **CAUGHT** | `Probes.test_honest_mock_passes_everything`<br>`Prompts.test_cli_reports_achieved_counts`<br>`Prompts.test_exact_counts_all_targets_via_server_tokenize` | - |
| B12 | `tools/bench/oa_client.py:221` | llama.cpp /tokenize count off by one | **CAUGHT** | `Client.test_tokenize_dialects`<br>`Prompts.test_exact_counts_all_targets_via_server_tokenize` | - |
| B13 | `tools/bench/run_matrix.py:266` | measured chat-template overhead not subtracted from the prompt target | **CAUGHT** | `Matrix.test_small_matrix_files_timings_cache` | - |
| B14 | `tools/bench/runlog.py:219` | mock server never detected (is_mock -> False) | **CAUGHT** | `Matrix.test_small_matrix_files_timings_cache`<br>`Probes.test_honest_mock_passes_everything` | - |
| B15 | `tools/bench/runlog.py:309` | MOCK SERVER RUN banner not written to summary.md | **CAUGHT** | `Matrix.test_small_matrix_files_timings_cache`<br>`Probes.test_honest_mock_passes_everything` | - |
| B16 | `tools/bench/oa_client.py:369` | stream without finish_reason (truncated) accepted as ok | **CAUGHT** | `Client.test_faults_become_errors`<br>`Matrix.test_truncated_stream_and_idle_timeout` | - |
| B17 | `tools/bench/run_matrix.py:148` | short_output never flagged | **CAUGHT** | `Matrix.test_no_length_forcing_is_flagged_not_hidden` | - |
| B18 | `tools/bench/run_matrix.py:409` | cell summary aggregates failed rows as ok | **CAUGHT** | `Matrix.test_http500_everywhere_records_failures_without_numbers`<br>`Matrix.test_partial_faults_aggregate_only_successes`<br>`Matrix.test_truncated_stream_and_idle_timeout` | - |
| B19 | `tools/bench/run_matrix.py:389` | agent cold control keeps the old session nonce (fully cached) | **CAUGHT** | `Matrix.test_agent_loop_prefix_cache` | - |
| B20 | `tools/bench/probes.py:279` | multihop grader accepts any 5-digit number | **CAUGHT** | `Probes.test_wrong_answer_mutation_scores_zero` | - |
| B21 | `tools/bench/probes.py:313` | codeqa file check always true | **CAUGHT** | `Probes.test_wrong_answer_mutation_scores_zero` | - |
| B22 | `tools/bench/probes.py:134` | JSON probe: value checks always true (valid JSON with wrong values passes) | **SURVIVED** | - | **CAUGHT** by `Graders.test_json_validator_rejects_valid_json_with_wrong_values` |
| B23 | `tools/bench/probes.py:379` | tools probe: tool name check always true | **SURVIVED** | - | **CAUGHT** by `ToolGrading.test_wrong_tool_name_with_right_city_fails` |
| B24 | `tools/bench/probes.py:386` | tools probe: city argument check always true | **SURVIVED** | - | **CAUGHT** by `ToolGrading.test_right_tool_name_with_wrong_city_fails` |
| B25 | `tools/bench/probes.py:404` | tools probe: final answer counted as using the tool result | **CAUGHT** | `Probes.test_wrong_answer_mutation_scores_zero` | - |
| B26 | `tools/bench/probes.py:186` | probe pass threshold ignored | **CAUGHT** | `Probes.test_wrong_answer_mutation_scores_zero` | - |
| B27 | `tools/bench/runlog.py:58` | percentile position uses len instead of len-1 | **CAUGHT** | 17 tests (1 FAIL, 16 ERROR), incl. `Stats.test_percentile_linear` | - |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[B1] tools/bench/runlog.py:141  decode uses n instead of n - n/chunks [0daa7bc set]
    -            m["decode_tok_s"] = (n - n / len(gen)) / span_s
    +            m["decode_tok_s"] = n / span_s
$ python3 -m unittest tests.test_bench
FAIL: test_decode_formula_one_and_multi_token_chunks (tests.test_bench.Stats.test_decode_formula_one_and_multi_token_chunks)
Ran 34 tests in 10.958s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B2] tools/bench/oa_client.py:439  role-only/empty chunk recorded as an output event (TTFT from first chunk, not first content) [0daa7bc set]
    -        if n_content or n_reason or n_tool or n_text:
    +        if True:
$ python3 -m unittest tests.test_bench
FAIL: test_completions_endpoint (tests.test_bench.Client.test_completions_endpoint)
FAIL: test_stream_timestamps_usage_and_role_chunk_not_counted (tests.test_bench.Client.test_stream_timestamps_usage_and_role_chunk_not_counted)
Ran 34 tests in 10.965s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/oa_client.py -> ''

[B3] tools/bench/runlog.py:127  TTFT measured at response headers instead of first output chunk [0daa7bc set]
    -    ttft_s = gen[0][0] / 1e6
    +    ttft_s = (r.get("headers_ms") or 0) / 1000.0 or 1e-9
$ python3 -m unittest tests.test_bench
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
FAIL: test_decode_formula_one_and_multi_token_chunks (tests.test_bench.Stats.test_decode_formula_one_and_multi_token_chunks)
Ran 34 tests in 10.951s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B4] tools/bench/run_matrix.py:305  warm requests use a fresh prefix (warm == cold) [0daa7bc set]
    -                    contents.append(stem + SUFFIX % k)
    +                    contents.append(self.build(inp, out, ("cold", inp, out, conc, rep, j, "x"), k)[0])
$ python3 -m unittest tests.test_bench
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
Ran 34 tests in 11.401s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/run_matrix.py -> ''

[B5] tools/bench/probes.py:246  needle grader accepts any 6 digits [0daa7bc set]
    -                              "correct": got == truth[k]})
    +                              "correct": got is not None})
$ python3 -m unittest tests.test_bench
FAIL: test_wrong_answer_mutation_scores_zero (tests.test_bench.Probes.test_wrong_answer_mutation_scores_zero)
Ran 34 tests in 10.953s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B6] tools/bench/runlog.py:122  failed requests still get timings [0daa7bc set]
    -    if not m["ok"]:
    -        return m
$ python3 -m unittest tests.test_bench
FAIL: test_faults_become_errors (tests.test_bench.Client.test_faults_become_errors)
FAIL: test_metrics_of_failed_request_are_none (tests.test_bench.Stats.test_metrics_of_failed_request_are_none)
Ran 34 tests in 10.890s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B7] tools/bench/runlog.py:129  TTFC counts reasoning-only chunks as content
    -    cont = [e for e in gen if e[1] or e[3] or e[4]]
    +    cont = gen
$ python3 -m unittest tests.test_bench
FAIL: test_multi_token_chunks_and_reasoning (tests.test_bench.Client.test_multi_token_chunks_and_reasoning)
Ran 34 tests in 11.004s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B8] tools/bench/run_matrix.py:308  cold requests reuse one prefix per cell (cold is not cold)
    -                    c, f = self.build(inp, out, ("cold", inp, out, conc, rep, j), k)
    +                    c, f = self.build(inp, out, ("cold", inp, out, conc, 0, 0), k)
$ python3 -m unittest tests.test_bench
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
Ran 34 tests in 10.968s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/run_matrix.py -> ''

[B9] tools/bench/runlog.py:102  server-reported cached tokens (usage.prompt_tokens_details) read as 0
    -            return det["cached_tokens"]
    +            return 0
$ python3 -m unittest tests.test_bench
FAIL: test_agent_loop_prefix_cache (tests.test_bench.Matrix.test_agent_loop_prefix_cache)
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
Ran 34 tests in 11.220s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B10] tools/bench/runlog.py:135  uncached_prefill_tok_s ignores the cached tokens
    -            m["uncached_prefill_tok_s"] = max(pt - m["cached_tokens"], 0) / ttft_s
    +            m["uncached_prefill_tok_s"] = pt / ttft_s
$ python3 -m unittest tests.test_bench
Ran 34 tests in 11.086s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/bench/runlog.py -> ''

[B11] tools/bench/prompts.py:310  exact fit: single-word padding step skipped
    -    while cur < target and len(memo) < max_calls:
    +    while False:
$ python3 -m unittest tests.test_bench
FAIL: test_honest_mock_passes_everything (tests.test_bench.Probes.test_honest_mock_passes_everything)
FAIL: test_cli_reports_achieved_counts (tests.test_bench.Prompts.test_cli_reports_achieved_counts)
FAIL: test_exact_counts_all_targets_via_server_tokenize (tests.test_bench.Prompts.test_exact_counts_all_targets_via_server_tokenize)
Ran 34 tests in 8.455s
FAILED (failures=3)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/prompts.py -> ''

[B12] tools/bench/oa_client.py:221  llama.cpp /tokenize count off by one
    -            return len(toks), r
    +            return len(toks) + 1, r
$ python3 -m unittest tests.test_bench
FAIL: test_tokenize_dialects (tests.test_bench.Client.test_tokenize_dialects)
FAIL: test_exact_counts_all_targets_via_server_tokenize (tests.test_bench.Prompts.test_exact_counts_all_targets_via_server_tokenize)
Ran 34 tests in 8.429s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/oa_client.py -> ''

[B13] tools/bench/run_matrix.py:266  measured chat-template overhead not subtracted from the prompt target
    -        target = max(target_total - self.overhead, 1)
    +        target = max(target_total, 1)
$ python3 -m unittest tests.test_bench
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
Ran 34 tests in 11.049s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/run_matrix.py -> ''

[B14] tools/bench/runlog.py:219  mock server never detected (is_mock -> False)
    -        return any(m.get("owned_by") == MOCK_OWNER for m in js.get("data", []))
    +        return False
$ python3 -m unittest tests.test_bench
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
FAIL: test_honest_mock_passes_everything (tests.test_bench.Probes.test_honest_mock_passes_everything)
Ran 34 tests in 11.139s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B15] tools/bench/runlog.py:309  MOCK SERVER RUN banner not written to summary.md
    -    if meta.get("is_mock_server"):
    +    if False:
$ python3 -m unittest tests.test_bench
FAIL: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
FAIL: test_honest_mock_passes_everything (tests.test_bench.Probes.test_honest_mock_passes_everything)
Ran 34 tests in 11.330s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B16] tools/bench/oa_client.py:369  stream without finish_reason (truncated) accepted as ok
    -        if r.error is None and r.finish_reason is None:
    +        if False:
$ python3 -m unittest tests.test_bench
FAIL: test_faults_become_errors (tests.test_bench.Client.test_faults_become_errors)
FAIL: test_truncated_stream_and_idle_timeout (tests.test_bench.Matrix.test_truncated_stream_and_idle_timeout)
Ran 34 tests in 8.767s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/oa_client.py -> ''

[B17] tools/bench/run_matrix.py:148  short_output never flagged
    -                   short_output=bool(forced and m["ok"] and requested_max and ct < requested_max))
    +                   short_output=bool(forced and m["ok"] and requested_max and ct < 0))
$ python3 -m unittest tests.test_bench
FAIL: test_no_length_forcing_is_flagged_not_hidden (tests.test_bench.Matrix.test_no_length_forcing_is_flagged_not_hidden)
Ran 34 tests in 10.956s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/run_matrix.py -> ''

[B18] tools/bench/run_matrix.py:409  cell summary aggregates failed rows as ok
    -            ok = [r for r in rows if r["metrics"]["ok"]]
    +            ok = rows
$ python3 -m unittest tests.test_bench
FAIL: test_http500_everywhere_records_failures_without_numbers (tests.test_bench.Matrix.test_http500_everywhere_records_failures_without_numbers)
FAIL: test_partial_faults_aggregate_only_successes (tests.test_bench.Matrix.test_partial_faults_aggregate_only_successes)
FAIL: test_truncated_stream_and_idle_timeout (tests.test_bench.Matrix.test_truncated_stream_and_idle_timeout)
Ran 34 tests in 9.241s
FAILED (failures=3)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/run_matrix.py -> ''

[B19] tools/bench/run_matrix.py:389  agent cold control keeps the old session nonce (fully cached)
    -            ctrl[0]["content"] = system(rng.getrandbits(64))
$ python3 -m unittest tests.test_bench
FAIL: test_agent_loop_prefix_cache (tests.test_bench.Matrix.test_agent_loop_prefix_cache)
Ran 34 tests in 10.980s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/run_matrix.py -> ''

[B20] tools/bench/probes.py:279  multihop grader accepts any 5-digit number
    -                items.append({"var": v, "expected": truth[v], "got": got, "correct": got == truth[v]})
    +                items.append({"var": v, "expected": truth[v], "got": got, "correct": got is not None})
$ python3 -m unittest tests.test_bench
FAIL: test_wrong_answer_mutation_scores_zero (tests.test_bench.Probes.test_wrong_answer_mutation_scores_zero)
Ran 34 tests in 11.088s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B21] tools/bench/probes.py:313  codeqa file check always true
    -                f_ok = bool(truth[nm]["file"]) and truth[nm]["file"] in line
    +                f_ok = True
$ python3 -m unittest tests.test_bench
FAIL: test_wrong_answer_mutation_scores_zero (tests.test_bench.Probes.test_wrong_answer_mutation_scores_zero)
Ran 34 tests in 11.145s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B22] tools/bench/probes.py:134  JSON probe: value checks always true (valid JSON with wrong values passes)
    -            checks["value_" + k] = (v.strip() == expected[k]) if isinstance(v, str) else (v == expected[k])
    +            checks["value_" + k] = True
$ python3 -m unittest tests.test_bench
Ran 34 tests in 11.052s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/bench/probes.py -> ''

[B23] tools/bench/probes.py:379  tools probe: tool name check always true
    -            call["name_ok"] = tc["function"]["name"] == "get_weather"
    +            call["name_ok"] = True
$ python3 -m unittest tests.test_bench
Ran 34 tests in 11.330s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/bench/probes.py -> ''

[B24] tools/bench/probes.py:386  tools probe: city argument check always true
    -                call["city_ok"] = city.lower() in str(args.get("city", "")).lower()
    +                call["city_ok"] = True
$ python3 -m unittest tests.test_bench
Ran 34 tests in 11.458s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/bench/probes.py -> ''

[B25] tools/bench/probes.py:404  tools probe: final answer counted as using the tool result
    -            uses = any(re.search(r"(?<![\d.])%s(?![\d])" % re.escape(f), r2.content) for f in forms)
    +            uses = True or any(re.search(r"(?<![\d.])%s(?![\d])" % re.escape(f), r2.content) for f in forms)
$ python3 -m unittest tests.test_bench
FAIL: test_wrong_answer_mutation_scores_zero (tests.test_bench.Probes.test_wrong_answer_mutation_scores_zero)
Ran 34 tests in 11.098s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B26] tools/bench/probes.py:186  probe pass threshold ignored
    -        passed = (error is None and score is not None and (thr is None or score >= thr))
    +        passed = (error is None and score is not None)
$ python3 -m unittest tests.test_bench
FAIL: test_wrong_answer_mutation_scores_zero (tests.test_bench.Probes.test_wrong_answer_mutation_scores_zero)
Ran 34 tests in 11.057s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B27] tools/bench/runlog.py:58  percentile position uses len instead of len-1
    -    pos = (len(s) - 1) * q / 100.0
    +    pos = len(s) * q / 100.0
$ python3 -m unittest tests.test_bench
ERROR: test_multi_token_chunks_and_reasoning (tests.test_bench.Client.test_multi_token_chunks_and_reasoning)
ERROR: test_stream_options_rejected_is_retried_and_recorded (tests.test_bench.Client.test_stream_options_rejected_is_retried_and_recorded)
ERROR: test_agent_loop_prefix_cache (tests.test_bench.Matrix.test_agent_loop_prefix_cache)
ERROR: test_eos_detection_llamacpp_dialect (tests.test_bench.Matrix.test_eos_detection_llamacpp_dialect)
ERROR: test_no_length_forcing_is_flagged_not_hidden (tests.test_bench.Matrix.test_no_length_forcing_is_flagged_not_hidden)
ERROR: test_partial_faults_aggregate_only_successes (tests.test_bench.Matrix.test_partial_faults_aggregate_only_successes)
ERROR: test_refuses_to_overwrite_a_run (tests.test_bench.Matrix.test_refuses_to_overwrite_a_run)
ERROR: test_small_matrix_files_timings_cache (tests.test_bench.Matrix.test_small_matrix_files_timings_cache)
ERROR: test_variants_completions_ratio_user_format_options (tests.test_bench.Matrix.test_variants_completions_ratio_user_format_options)
ERROR: test_honest_mock_passes_everything (tests.test_bench.Probes.test_honest_mock_passes_everything)
ERROR: test_nondeterminism_is_reported_not_fatal (tests.test_bench.Probes.test_nondeterminism_is_reported_not_fatal)
ERROR: test_ratio_counter_probe (tests.test_bench.Probes.test_ratio_counter_probe)
ERROR: test_response_format_rejected_is_recorded (tests.test_bench.Probes.test_response_format_rejected_is_recorded)
ERROR: test_wrong_answer_mutation_scores_zero (tests.test_bench.Probes.test_wrong_answer_mutation_scores_zero)
ERROR: test_decode_formula_one_and_multi_token_chunks (tests.test_bench.Stats.test_decode_formula_one_and_multi_token_chunks)
ERROR: test_describe_skips_missing_and_fmt_prints_na (tests.test_bench.Stats.test_describe_skips_missing_and_fmt_prints_na)
FAIL: test_percentile_linear (tests.test_bench.Stats.test_percentile_linear)
Ran 34 tests in 6.723s
FAILED (failures=1, errors=16)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''
```

Re-run of the surviving mutations with the added tests:

```
[B10] tools/bench/runlog.py:135  uncached_prefill_tok_s ignores the cached tokens
    -            m["uncached_prefill_tok_s"] = max(pt - m["cached_tokens"], 0) / ttft_s
    +            m["uncached_prefill_tok_s"] = pt / ttft_s
$ python3 -m unittest tests.test_bench
FAIL: test_uncached_prefill_subtracts_server_cached_tokens (tests.test_bench.Stats.test_uncached_prefill_subtracts_server_cached_tokens)
Ran 38 tests in 11.205s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/runlog.py -> ''

[B22] tools/bench/probes.py:134  JSON probe: value checks always true (valid JSON with wrong values passes)
    -            checks["value_" + k] = (v.strip() == expected[k]) if isinstance(v, str) else (v == expected[k])
    +            checks["value_" + k] = True
$ python3 -m unittest tests.test_bench
FAIL: test_json_validator_rejects_valid_json_with_wrong_values (tests.test_bench.Graders.test_json_validator_rejects_valid_json_with_wrong_values)
Ran 38 tests in 11.112s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B23] tools/bench/probes.py:379  tools probe: tool name check always true
    -            call["name_ok"] = tc["function"]["name"] == "get_weather"
    +            call["name_ok"] = True
$ python3 -m unittest tests.test_bench
FAIL: test_wrong_tool_name_with_right_city_fails (tests.test_bench.ToolGrading.test_wrong_tool_name_with_right_city_fails)
Ran 38 tests in 11.257s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''

[B24] tools/bench/probes.py:386  tools probe: city argument check always true
    -                call["city_ok"] = city.lower() in str(args.get("city", "")).lower()
    +                call["city_ok"] = True
$ python3 -m unittest tests.test_bench
FAIL: test_right_tool_name_with_wrong_city_fails (tests.test_bench.ToolGrading.test_right_tool_name_with_wrong_city_fails)
Ran 38 tests in 11.257s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/probes.py -> ''
```

## tools/gguf_quant_audit.py (`tests/test_gguf_quant_audit.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_gguf_quant_audit
Ran 6 tests in 0.100s

OK
```

Baseline in the worktree after copying in the test file with the added tests (output filtered with `grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):

```
$ python3 -m unittest tests.test_gguf_quant_audit
Ran 7 tests in 0.084s
OK
```

| ID | file:line @64fa04d | mutation | result | caught by | after the added tests |
|---|---|---|---|---|---|
| G1 | `tools/gguf_quant_audit.py:138` | default expert floor 4.0 -> 3.0 bpw (IQ3_S experts pass) | **CAUGHT** | `Audit.test_iq3_expert_fails` | - |
| G2 | `tools/gguf_quant_audit.py:121` | expert classification needs '_exps_' (no GGUF name matches) | **CAUGHT** | `Audit.test_pass_and_classes`<br>`Audit.test_split_files` | - |
| G3 | `tools/gguf_quant_audit.py:121` | shared experts classified as routed experts | **CAUGHT** | `Audit.test_pass_and_classes`<br>`Audit.test_split_files` | - |
| G4 | `tools/gguf_quant_audit.py:165` | non-expert floor applied only to dense_ffn (attention/shexp/embd unchecked) | **CAUGHT** | `Audit.test_low_bit_dense_fails_unless_warn_only` | - |
| G5 | `tools/gguf_quant_audit.py:166` | non-expert finding is always WARN (dense floor never FAILs) | **CAUGHT** | `Audit.test_low_bit_dense_fails_unless_warn_only` | - |
| G6 | `tools/gguf_quant_audit.py:32` | type table: Q4_K type_size 144 -> 136 (4.25 bpw) | **CAUGHT** | `Audit.test_pass_and_classes`<br>`Audit.test_type_table_matches_pinned_gguf_py` | - |
| G7 | `tools/gguf_quant_audit.py:34` | type table: IQ4_XS type_size 136 -> 120 (3.75 bpw); only the gguf-py cross-check uses IQ4_XS | **CAUGHT** | `Audit.test_type_table_matches_pinned_gguf_py` | - |
| G8 | `tools/gguf_quant_audit.py:171` | unknown ggml type ids do not fail the verdict | **SURVIVED** | - | **CAUGHT** by `Audit.test_unknown_type_id_fails` |
| G7 (env unset) | `tools/gguf_quant_audit.py:34` | type table: IQ4_XS type_size 136 -> 120 (3.75 bpw); only the gguf-py cross-check uses IQ4_XS | **SURVIVED** | - | - |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[G1] tools/gguf_quant_audit.py:138  default expert floor 4.0 -> 3.0 bpw (IQ3_S experts pass)
    -def audit(first, min_expert_bpw=4.0, min_dense_bpw=4.0, dense_warn_only=False):
    +def audit(first, min_expert_bpw=3.0, min_dense_bpw=4.0, dense_warn_only=False):
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_iq3_expert_fails (tests.test_gguf_quant_audit.Audit.test_iq3_expert_fails)
Ran 6 tests in 0.082s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G2] tools/gguf_quant_audit.py:121  expert classification needs '_exps_' (no GGUF name matches)
    -    if "_exps" in name:
    +    if "_exps_" in name:
$ python3 -m unittest tests.test_gguf_quant_audit
ERROR: test_pass_and_classes (tests.test_gguf_quant_audit.Audit.test_pass_and_classes)
ERROR: test_split_files (tests.test_gguf_quant_audit.Audit.test_split_files)
Ran 6 tests in 0.078s
FAILED (errors=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G3] tools/gguf_quant_audit.py:121  shared experts classified as routed experts
    -    if "_exps" in name:
    +    if "_exps" in name or "_shexp" in name:
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_pass_and_classes (tests.test_gguf_quant_audit.Audit.test_pass_and_classes)
FAIL: test_split_files (tests.test_gguf_quant_audit.Audit.test_split_files)
Ran 6 tests in 0.080s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G4] tools/gguf_quant_audit.py:165  non-expert floor applied only to dense_ffn (attention/shexp/embd unchecked)
    -            elif c != "routed_experts" and len(dims) >= 2 and bpw < min_dense_bpw:
    +            elif c == "dense_ffn" and len(dims) >= 2 and bpw < min_dense_bpw:
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_low_bit_dense_fails_unless_warn_only (tests.test_gguf_quant_audit.Audit.test_low_bit_dense_fails_unless_warn_only)
Ran 6 tests in 0.083s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G5] tools/gguf_quant_audit.py:166  non-expert finding is always WARN (dense floor never FAILs)
    -                findings.append({"level": "WARN" if dense_warn_only else "FAIL", "tensor": name, "type": tname, "bpw": round(bpw, 3)})
    +                findings.append({"level": "WARN", "tensor": name, "type": tname, "bpw": round(bpw, 3)})
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_low_bit_dense_fails_unless_warn_only (tests.test_gguf_quant_audit.Audit.test_low_bit_dense_fails_unless_warn_only)
Ran 6 tests in 0.080s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G6] tools/gguf_quant_audit.py:32  type table: Q4_K type_size 144 -> 136 (4.25 bpw)
    -    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    +    12: ("Q4_K", 256, 136), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_pass_and_classes (tests.test_gguf_quant_audit.Audit.test_pass_and_classes)
FAIL: test_type_table_matches_pinned_gguf_py (tests.test_gguf_quant_audit.Audit.test_type_table_matches_pinned_gguf_py)
Ran 6 tests in 0.081s
FAILED (failures=2)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G7] tools/gguf_quant_audit.py:34  type table: IQ4_XS type_size 136 -> 120 (3.75 bpw); only the gguf-py cross-check uses IQ4_XS
    -    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    +    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 120),
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_type_table_matches_pinned_gguf_py (tests.test_gguf_quant_audit.Audit.test_type_table_matches_pinned_gguf_py)
Ran 6 tests in 0.085s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G8] tools/gguf_quant_audit.py:171  unknown ggml type ids do not fail the verdict
    -    verdict = "FAIL" if any(f["level"] == "FAIL" for f in findings) or unknown else "PASS"
    +    verdict = "FAIL" if any(f["level"] == "FAIL" for f in findings) else "PASS"
$ python3 -m unittest tests.test_gguf_quant_audit
Ran 6 tests in 0.080s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/gguf_quant_audit.py -> ''

[G7] tools/gguf_quant_audit.py:34  type table: IQ4_XS type_size 136 -> 120 (3.75 bpw); only the gguf-py cross-check uses IQ4_XS  (LLAMA_CPP_GGUF_PY unset)
    -    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    +    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 120),
$ python3 -m unittest tests.test_gguf_quant_audit
Ran 6 tests in 0.004s
OK (skipped=1)
=> SURVIVED (exit 0); restored: git status --short -- tools/gguf_quant_audit.py -> ''
```

Re-run of the surviving mutations with the added tests:

```
[G8] tools/gguf_quant_audit.py:171  unknown ggml type ids do not fail the verdict
    -    verdict = "FAIL" if any(f["level"] == "FAIL" for f in findings) or unknown else "PASS"
    +    verdict = "FAIL" if any(f["level"] == "FAIL" for f in findings) else "PASS"
$ python3 -m unittest tests.test_gguf_quant_audit
FAIL: test_unknown_type_id_fails (tests.test_gguf_quant_audit.Audit.test_unknown_type_id_fails)
Ran 7 tests in 0.085s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/gguf_quant_audit.py -> ''
```

## tools/quality/logprob_compare.py (`tests/test_logprob_compare.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_logprob_compare
Ran 5 tests in 2.236s

OK
```

Baseline in the worktree after copying in the test file with the added tests (output filtered with `grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):

```
$ python3 -m unittest tests.test_logprob_compare
Ran 6 tests in 2.247s
OK
```

| ID | file:line @64fa04d | mutation | result | caught by | after the added tests |
|---|---|---|---|---|---|
| L1 | `tools/quality/logprob_compare.py:60` | KL direction reversed: KL(cand || ref) | **SURVIVED** | - | **CAUGHT** by `Compare.test_compare_exact_kl_direction_and_floor` |
| L2 | `tools/quality/logprob_compare.py:55` | missing-token floor without the -log(2) margin (ref side) | **SURVIVED** | - | **CAUGHT** by `Compare.test_compare_exact_kl_direction_and_floor` |
| L3 | `tools/quality/logprob_compare.py:61` | reference top-1 taken as argmin | **CAUGHT** | `Compare.test_compare_unit`<br>`Compare.test_identical_engines`<br>`Compare.test_softer_candidate_has_kl_but_same_top1` | - |
| L4 | `tools/quality/logprob_compare.py:134` | exit code 0 even with no successful prefix | **CAUGHT** | `Compare.test_engine_without_logprobs_is_an_error_not_a_number` | - |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[L1] tools/quality/logprob_compare.py:60  KL direction reversed: KL(cand || ref)
    -    kl = sum((p[t] / zp) * math.log((p[t] / zp) / (q[t] / zq)) for t in union)
    +    kl = sum((q[t] / zq) * math.log((q[t] / zq) / (p[t] / zp)) for t in union)
$ python3 -m unittest tests.test_logprob_compare
Ran 5 tests in 2.225s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/quality/logprob_compare.py -> ''

[L2] tools/quality/logprob_compare.py:55  missing-token floor without the -log(2) margin (ref side)
    -    fr = min(ref.values()) - math.log(2)
    +    fr = min(ref.values())
$ python3 -m unittest tests.test_logprob_compare
Ran 5 tests in 2.246s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/quality/logprob_compare.py -> ''

[L3] tools/quality/logprob_compare.py:61  reference top-1 taken as argmin
    -    r1 = max(ref, key=ref.get)
    +    r1 = min(ref, key=ref.get)
$ python3 -m unittest tests.test_logprob_compare
FAIL: test_compare_unit (tests.test_logprob_compare.Compare.test_compare_unit)
FAIL: test_identical_engines (tests.test_logprob_compare.Compare.test_identical_engines)
FAIL: test_softer_candidate_has_kl_but_same_top1 (tests.test_logprob_compare.Compare.test_softer_candidate_has_kl_but_same_top1)
Ran 5 tests in 2.238s
FAILED (failures=3)
=> CAUGHT (exit 1); restored: git status --short -- tools/quality/logprob_compare.py -> ''

[L4] tools/quality/logprob_compare.py:134  exit code 0 even with no successful prefix
    -    return 0 if rows else 1
    +    return 0
$ python3 -m unittest tests.test_logprob_compare
FAIL: test_engine_without_logprobs_is_an_error_not_a_number (tests.test_logprob_compare.Compare.test_engine_without_logprobs_is_an_error_not_a_number)
Ran 5 tests in 2.231s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/quality/logprob_compare.py -> ''
```

Re-run of the surviving mutations with the added tests:

```
[L1] tools/quality/logprob_compare.py:60  KL direction reversed: KL(cand || ref)
    -    kl = sum((p[t] / zp) * math.log((p[t] / zp) / (q[t] / zq)) for t in union)
    +    kl = sum((q[t] / zq) * math.log((q[t] / zq) / (p[t] / zp)) for t in union)
$ python3 -m unittest tests.test_logprob_compare
FAIL: test_compare_exact_kl_direction_and_floor (tests.test_logprob_compare.Compare.test_compare_exact_kl_direction_and_floor)
Ran 6 tests in 2.228s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/quality/logprob_compare.py -> ''

[L2] tools/quality/logprob_compare.py:55  missing-token floor without the -log(2) margin (ref side)
    -    fr = min(ref.values()) - math.log(2)
    +    fr = min(ref.values())
$ python3 -m unittest tests.test_logprob_compare
FAIL: test_compare_exact_kl_direction_and_floor (tests.test_logprob_compare.Compare.test_compare_exact_kl_direction_and_floor)
Ran 6 tests in 2.234s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/quality/logprob_compare.py -> ''
```

## tools/kv/slot_persist.py (`tests/test_slot_persist.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_slot_persist
Ran 3 tests in 1.559s

OK
```

Baseline in the worktree after copying in the test file with the added tests (output filtered with `grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):

```
$ python3 -m unittest tests.test_slot_persist
Ran 5 tests in 2.590s
OK
```

| ID | file:line @64fa04d | mutation | result | caught by | after the added tests |
|---|---|---|---|---|---|
| S1 | `tools/kv/slot_persist.py:52` | no erase between save and restore (warm request can hit the live cache) | **SURVIVED** | - | **CAUGHT** by `SlotPersist.test_restore_that_loads_nothing_shows_full_prefill` |
| S2 | `tools/kv/slot_persist.py:57` | continuation identity always true | **CAUGHT** | `SlotPersist.test_divergent_continuation_is_reported` | - |
| S3 | `tools/kv/slot_persist.py:102` | exit code 0 when some repeats failed | **SURVIVED** | - | **CAUGHT** by `SlotPersist.test_partial_failure_is_a_nonzero_exit` |
| S4 | `tools/kv/slot_persist.py:90` | warm prompt_n read from the cold request | **CAUGHT** | `SlotPersist.test_restore_makes_warm_prefill_free` | - |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[S1] tools/kv/slot_persist.py:52  no erase between save and restore (warm request can hit the live cache)
    -    post(url, "/slots/0?action=erase", {}, timeout)
$ python3 -m unittest tests.test_slot_persist
Ran 3 tests in 1.557s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/kv/slot_persist.py -> ''

[S2] tools/kv/slot_persist.py:57  continuation identity always true
    -            "same_continuation": cold["content"] == warm["content"]}
    +            "same_continuation": True}
$ python3 -m unittest tests.test_slot_persist
FAIL: test_divergent_continuation_is_reported (tests.test_slot_persist.SlotPersist.test_divergent_continuation_is_reported)
Ran 3 tests in 1.558s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/kv/slot_persist.py -> ''

[S3] tools/kv/slot_persist.py:102  exit code 0 when some repeats failed
    -    return 0 if runs and not errors else 1
    +    return 0 if runs else 1
$ python3 -m unittest tests.test_slot_persist
Ran 3 tests in 1.557s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/kv/slot_persist.py -> ''

[S4] tools/kv/slot_persist.py:90  warm prompt_n read from the cold request
    -            "warm_prompt_n_p50": med([r["warm"]["prompt_n"] for r in runs]),
    +            "warm_prompt_n_p50": med([r["cold"]["prompt_n"] for r in runs]),
$ python3 -m unittest tests.test_slot_persist
FAIL: test_restore_makes_warm_prefill_free (tests.test_slot_persist.SlotPersist.test_restore_makes_warm_prefill_free)
Ran 3 tests in 1.559s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/kv/slot_persist.py -> ''
```

Re-run of the surviving mutations with the added tests:

```
[S1] tools/kv/slot_persist.py:52  no erase between save and restore (warm request can hit the live cache)
    -    post(url, "/slots/0?action=erase", {}, timeout)
$ python3 -m unittest tests.test_slot_persist
FAIL: test_restore_that_loads_nothing_shows_full_prefill (tests.test_slot_persist.SlotPersist.test_restore_that_loads_nothing_shows_full_prefill)
Ran 5 tests in 2.581s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/kv/slot_persist.py -> ''

[S3] tools/kv/slot_persist.py:102  exit code 0 when some repeats failed
    -    return 0 if runs and not errors else 1
    +    return 0 if runs else 1
$ python3 -m unittest tests.test_slot_persist
FAIL: test_partial_failure_is_a_nonzero_exit (tests.test_slot_persist.SlotPersist.test_partial_failure_is_a_nonzero_exit)
Ran 5 tests in 2.590s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/kv/slot_persist.py -> ''
```

## tools/bench/vs_roofline.py (`tests/test_vs_roofline.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_vs_roofline
Ran 1 test in 0.002s

OK
```

Baseline in the worktree after copying in the test file with the added tests (output filtered with `grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):

```
$ python3 -m unittest tests.test_vs_roofline
Ran 2 tests in 0.004s
OK
```

| ID | file:line @64fa04d | mutation | result | caught by | after the added tests |
|---|---|---|---|---|---|
| V1 | `tools/bench/vs_roofline.py:35` | decode context without +output/2 | **CAUGHT** | `VsRoofline.test_ratio_against_direct_roofline` | - |
| V2 | `tools/bench/vs_roofline.py:29` | concurrency>1 cells not filtered out | **CAUGHT** | `VsRoofline.test_ratio_against_direct_roofline` | - |
| V3 | `tools/bench/vs_roofline.py:36` | --strategy/--miss not passed to roofline.decode (its defaults pp/cpu used) | **SURVIVED** | - | **CAUGHT** by `VsRoofline.test_strategy_miss_and_mock_flag_are_passed_through` |
| V4 | `tools/bench/vs_roofline.py:37` | mock run not marked in the table | **SURVIVED** | - | **CAUGHT** by `VsRoofline.test_strategy_miss_and_mock_flag_are_passed_through` |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[V1] tools/bench/vs_roofline.py:35  decode context without +output/2
    -        ctx = int((prompt or 0) + gen / 2)
    +        ctx = int(prompt or 0)
$ python3 -m unittest tests.test_vs_roofline
FAIL: test_ratio_against_direct_roofline (tests.test_vs_roofline.VsRoofline.test_ratio_against_direct_roofline)
Ran 1 test in 0.003s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/vs_roofline.py -> ''

[V2] tools/bench/vs_roofline.py:29  concurrency>1 cells not filtered out
    -        if c.get("concurrency") != 1:
    -            continue
$ python3 -m unittest tests.test_vs_roofline
FAIL: test_ratio_against_direct_roofline (tests.test_vs_roofline.VsRoofline.test_ratio_against_direct_roofline)
Ran 1 test in 0.003s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/vs_roofline.py -> ''

[V3] tools/bench/vs_roofline.py:36  --strategy/--miss not passed to roofline.decode (its defaults pp/cpu used)
    -        bound = roofline.decode(spec, hw, q, ctx, hit, strategy=strategy, miss=miss)["tok_s"]
    +        bound = roofline.decode(spec, hw, q, ctx, hit)["tok_s"]
$ python3 -m unittest tests.test_vs_roofline
Ran 1 test in 0.002s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/bench/vs_roofline.py -> ''

[V4] tools/bench/vs_roofline.py:37  mock run not marked in the table
    -        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "mock": bool(meta.get("is_mock_server")),
    +        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "mock": False,
$ python3 -m unittest tests.test_vs_roofline
Ran 1 test in 0.002s
OK
=> SURVIVED (exit 0); restored: git status --short -- tools/bench/vs_roofline.py -> ''
```

Re-run of the surviving mutations with the added tests:

```
[V3] tools/bench/vs_roofline.py:36  --strategy/--miss not passed to roofline.decode (its defaults pp/cpu used)
    -        bound = roofline.decode(spec, hw, q, ctx, hit, strategy=strategy, miss=miss)["tok_s"]
    +        bound = roofline.decode(spec, hw, q, ctx, hit)["tok_s"]
$ python3 -m unittest tests.test_vs_roofline
FAIL: test_strategy_miss_and_mock_flag_are_passed_through (tests.test_vs_roofline.VsRoofline.test_strategy_miss_and_mock_flag_are_passed_through)
Ran 2 tests in 0.005s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/vs_roofline.py -> ''

[V4] tools/bench/vs_roofline.py:37  mock run not marked in the table
    -        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "mock": bool(meta.get("is_mock_server")),
    +        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "mock": False,
$ python3 -m unittest tests.test_vs_roofline
FAIL: test_strategy_miss_and_mock_flag_are_passed_through (tests.test_vs_roofline.VsRoofline.test_strategy_miss_and_mock_flag_are_passed_through)
Ran 2 tests in 0.004s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- tools/bench/vs_roofline.py -> ''
```

## scripts/offline/weights.py (`tests/test_offline_weights.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_offline_weights
Ran 4 tests in 0.531s

OK
```

Baseline in the worktree after copying in the test file with the added tests (output filtered with `grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):

```
$ python3 -m unittest tests.test_offline_weights
Ran 5 tests in 1.041s
OK
```

| ID | file:line @64fa04d | mutation | result | caught by | after the added tests |
|---|---|---|---|---|---|
| W1 | `scripts/offline/weights.py:93` | sha256 mismatch not detected | **CAUGHT** | `Verify.test_ok_missing_size_and_corruption` | - |
| W2 | `scripts/offline/weights.py:113` | resume requests Range from byte 0 | **CAUGHT** | `Download.test_resume_and_verify` | - |
| W3 | `scripts/offline/weights.py:115` | resume truncates the partial file (open 'wb') | **CAUGHT** | `Download.test_resume_and_verify` | - |
| W4 | `scripts/offline/weights.py:116` | server that ignores Range (200) is not refused | **SURVIVED** | - | **CAUGHT** by `Download.test_server_ignoring_range_is_refused` |
| W5 | `scripts/offline/weights.py:43` | branch names accepted as revision | **CAUGHT** | `Manifest.test_build_filters_and_uses_lfs_sha` | - |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[W1] scripts/offline/weights.py:93  sha256 mismatch not detected
    -            if got != f["sha256"]:
    +            if False:
$ python3 -m unittest tests.test_offline_weights
FAIL: test_ok_missing_size_and_corruption (tests.test_offline_weights.Verify.test_ok_missing_size_and_corruption)
Ran 4 tests in 0.544s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- scripts/offline/weights.py -> ''

[W2] scripts/offline/weights.py:113  resume requests Range from byte 0
    -        headers = {"Range": f"bytes={have}-"} if have else {}
    +        headers = {"Range": "bytes=0-"} if have else {}
$ python3 -m unittest tests.test_offline_weights
FAIL: test_resume_and_verify (tests.test_offline_weights.Download.test_resume_and_verify)
Ran 4 tests in 0.540s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- scripts/offline/weights.py -> ''

[W3] scripts/offline/weights.py:115  resume truncates the partial file (open 'wb')
    -        with urllib.request.urlopen(_req(url, headers), timeout=120) as r, open(p, "ab" if have else "wb") as out:
    +        with urllib.request.urlopen(_req(url, headers), timeout=120) as r, open(p, "wb") as out:
$ python3 -m unittest tests.test_offline_weights
FAIL: test_resume_and_verify (tests.test_offline_weights.Download.test_resume_and_verify)
Ran 4 tests in 0.538s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- scripts/offline/weights.py -> ''

[W4] scripts/offline/weights.py:116  server that ignores Range (200) is not refused
    -            if have and r.status != 206:
    +            if False:
$ python3 -m unittest tests.test_offline_weights
Ran 4 tests in 0.537s
OK
=> SURVIVED (exit 0); restored: git status --short -- scripts/offline/weights.py -> ''

[W5] scripts/offline/weights.py:43  branch names accepted as revision
    -    if not re.fullmatch(r"[0-9a-f]{40}", revision):
    +    if not re.fullmatch(r"[0-9A-Za-z]+", revision):
$ python3 -m unittest tests.test_offline_weights
FAIL: test_build_filters_and_uses_lfs_sha (tests.test_offline_weights.Manifest.test_build_filters_and_uses_lfs_sha)
Ran 4 tests in 0.543s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- scripts/offline/weights.py -> ''
```

Re-run of the surviving mutations with the added tests:

```
[W4] scripts/offline/weights.py:116  server that ignores Range (200) is not refused
    -            if have and r.status != 206:
    +            if False:
$ python3 -m unittest tests.test_offline_weights
FAIL: test_server_ignoring_range_is_refused (tests.test_offline_weights.Download.test_server_ignoring_range_is_refused)
Ran 5 tests in 1.047s
FAILED (failures=1)
=> CAUGHT (exit 1); restored: git status --short -- scripts/offline/weights.py -> ''
```

## scripts/diag/summarize.py (`tests/test_diag_kit.py`)

Baseline in the worktree, unmutated, before any test was added (last 3 lines of the output):

```
$ python3 -m unittest tests.test_diag_kit
Ran 33 tests in 22.241s

OK (skipped=2)
```

| ID | file:line @64fa04d | mutation | result | caught by |
|---|---|---|---|---|
| D1 | `scripts/diag/summarize.py:83` | probe status ignored (values read from needs-root/failed probes) | **CAUGHT** | `Collect.test_summary_of_nonroot_run_does_not_invent`<br>`Summarize.test_mutation_probe_lost_becomes_unknown`<br>`Summarize.test_nonroot_fixture_unknowns` |
| D2 | `scripts/diag/summarize.py:347` | memory channels = number of DIMMs, not distinct channel ids | **CAUGHT** | `Summarize.test_infer_channels_locator_styles` |
| D3 | `scripts/diag/summarize.py:418` | inferred channel count labelled CONFIRMED instead of PROVISIONAL | **CAUGHT** | `CollectAsRoot.test_root_probes_run_and_parse`<br>`Summarize.test_confirmed_sources_exist`<br>`Summarize.test_full_fixture_values` |
| D4 | `scripts/diag/summarize.py:419` | red flag only below 4 channels (4 channels not flagged) | **CAUGHT** | `Summarize.test_full_fixture_red_flags` |
| D5 | `scripts/diag/summarize.py:985` | SYNTHETIC banner dropped from the draft | **CAUGHT** | `Summarize.test_fixtures_are_marked_synthetic` |
| D6 | `scripts/diag/summarize.py:269` | AVX-512 red flag inverted | **CAUGHT** | `Summarize.test_full_fixture_red_flags`<br>`Summarize.test_nonroot_fixture_unknowns` |

Literal output of the driver (one block per mutation: the mutation as `git diff -U0` lines, the test command, the FAIL/ERROR lines, the `Ran`/`OK`/`FAILED` lines, the verdict and the restore check):

```
[D1] scripts/diag/summarize.py:83  probe status ignored (values read from needs-root/failed probes)
    -        if p.get("status") != "ok":
    +        if False:
$ python3 -m unittest tests.test_diag_kit
FAIL: test_summary_of_nonroot_run_does_not_invent (tests.test_diag_kit.Collect.test_summary_of_nonroot_run_does_not_invent)
FAIL: test_mutation_probe_lost_becomes_unknown (tests.test_diag_kit.Summarize.test_mutation_probe_lost_becomes_unknown)
FAIL: test_nonroot_fixture_unknowns (tests.test_diag_kit.Summarize.test_nonroot_fixture_unknowns)
Ran 33 tests in 22.435s
FAILED (failures=3, skipped=2)
=> CAUGHT (exit 1); restored: git status --short -- scripts/diag/summarize.py -> ''

[D2] scripts/diag/summarize.py:347  memory channels = number of DIMMs, not distinct channel ids
    -    return len(set(ids)), "distinct (socket, channel) ids parsed from the Bank Locator/Locator strings of %d populated DIMMs: %s" % (
    +    return len(ids), "distinct (socket, channel) ids parsed from the Bank Locator/Locator strings of %d populated DIMMs: %s" % (
$ python3 -m unittest tests.test_diag_kit
FAIL: test_infer_channels_locator_styles (tests.test_diag_kit.Summarize.test_infer_channels_locator_styles)
Ran 33 tests in 22.877s
FAILED (failures=1, skipped=2)
=> CAUGHT (exit 1); restored: git status --short -- scripts/diag/summarize.py -> ''

[D3] scripts/diag/summarize.py:418  inferred channel count labelled CONFIRMED instead of PROVISIONAL
    -                F.provisional(S, "memory channels populated", n, how + " (" + src + ")")
    +                F.confirmed(S, "memory channels populated", n, how + " (" + src + ")")
$ python3 -m unittest tests.test_diag_kit
FAIL: test_root_probes_run_and_parse (tests.test_diag_kit.CollectAsRoot.test_root_probes_run_and_parse)
FAIL: test_confirmed_sources_exist (tests.test_diag_kit.Summarize.test_confirmed_sources_exist)
FAIL: test_full_fixture_values (tests.test_diag_kit.Summarize.test_full_fixture_values)
Ran 33 tests in 22.062s
FAILED (failures=3, skipped=2)
=> CAUGHT (exit 1); restored: git status --short -- scripts/diag/summarize.py -> ''

[D4] scripts/diag/summarize.py:419  red flag only below 4 channels (4 channels not flagged)
    -                if n < 8:
    +                if n < 4:
$ python3 -m unittest tests.test_diag_kit
FAIL: test_full_fixture_red_flags (tests.test_diag_kit.Summarize.test_full_fixture_red_flags)
Ran 33 tests in 22.299s
FAILED (failures=1, skipped=2)
=> CAUGHT (exit 1); restored: git status --short -- scripts/diag/summarize.py -> ''

[D5] scripts/diag/summarize.py:985  SYNTHETIC banner dropped from the draft
    -    if m.get("synthetic"):
    +    if False:
$ python3 -m unittest tests.test_diag_kit
FAIL: test_fixtures_are_marked_synthetic (tests.test_diag_kit.Summarize.test_fixtures_are_marked_synthetic)
Ran 33 tests in 22.369s
FAILED (failures=1, skipped=2)
=> CAUGHT (exit 1); restored: git status --short -- scripts/diag/summarize.py -> ''

[D6] scripts/diag/summarize.py:269  AVX-512 red flag inverted
    -            if kv["avx512_any"] == "yes":
    +            if kv["avx512_any"] != "yes":
$ python3 -m unittest tests.test_diag_kit
FAIL: test_full_fixture_red_flags (tests.test_diag_kit.Summarize.test_full_fixture_red_flags)
FAIL: test_nonroot_fixture_unknowns (tests.test_diag_kit.Summarize.test_nonroot_fixture_unknowns)
Ran 33 tests in 22.370s
FAILED (failures=2, skipped=2)
=> CAUGHT (exit 1); restored: git status --short -- scripts/diag/summarize.py -> ''
```

## Добавленные тесты (только `tests/`, инструменты не менялись)
Each test closes one SURVIVED mutation. Not committed (task constraint: no commit, no `git add`).
- `tests/test_bench.py`
  - `Stats.test_uncached_prefill_subtracts_server_cached_tokens` (B10). Covers `uncached_prefill_tok_s` with
    `usage.prompt_tokens_details.cached_tokens`, with the llama.cpp `timings.cache_n` fallback, and with no cache report
    (stays None).
  - `Graders.test_json_validator_rejects_valid_json_with_wrong_values` (B22). Valid JSON with one wrong value per key must
    not be `valid`.
  - `ToolGrading.test_wrong_tool_name_with_right_city_fails` (B23) and
    `ToolGrading.test_right_tool_name_with_wrong_city_fails` (B24). Both patch `mock_server.toy_answer` in-process for the
    first-turn tool call only, and restore it in `finally`; `tools/bench/mock_server.py` itself is unchanged.
- `tests/test_gguf_quant_audit.py`: `Audit.test_unknown_type_id_fails` (G8). Type id 99 is not in the table, so the
  verdict must be FAIL and `main` must exit 1.
- `tests/test_logprob_compare.py`: `Compare.test_compare_exact_kl_direction_and_floor` (L1, L2). The exact
  KL(ref || cand), with the floor `min - log 2` on both sides, is derived by hand in the test. The test also checks that
  this case separates the two KL directions.
- `tests/test_slot_persist.py`: `SlotPersist.test_restore_that_loads_nothing_shows_full_prefill` (S1; new stub option
  `noop_restore`) and `SlotPersist.test_partial_failure_is_a_nonzero_exit` (S3; new stub option `fail_restore_once`).
- `tests/test_vs_roofline.py`: `VsRoofline.test_strategy_miss_and_mock_flag_are_passed_through` (V3, V4). Uses `tp`/`pcie`
  (a different bound from the default `pp`/`cpu`, asserted) and a mock meta.
- `tests/test_offline_weights.py`: `Download.test_server_ignoring_range_is_refused` (W4). A server answers 200 to a Range
  request, so `RuntimeError` is expected and the partial file must be left untouched.

`tests/test_diag_kit.py` is unchanged (6/6 without additions).

Main checkout `/home/user/408032gb`, tools unmodified, with the added tests (output filtered with
`grep -E "^(FAIL|ERROR|Ran |OK|FAILED)"`):
```
$ export PYTHONDONTWRITEBYTECODE=1 LLAMA_CPP_GGUF_PY=<S>/llama.cpp/gguf-py
$ python3 -m unittest tests.test_bench
Ran 38 tests in 11.201s
OK
$ python3 -m unittest tests.test_gguf_quant_audit
Ran 7 tests in 0.106s
OK
$ python3 -m unittest tests.test_logprob_compare
Ran 6 tests in 2.221s
OK
$ python3 -m unittest tests.test_slot_persist
Ran 5 tests in 2.585s
OK
$ python3 -m unittest tests.test_vs_roofline
Ran 2 tests in 0.004s
OK
$ python3 -m unittest tests.test_offline_weights
Ran 5 tests in 1.035s
OK
```

## Выводы
- **`0daa7bc` «34 tests, 6/6 mutations caught».** Re-checked at `64fa04d`, where the bench tools and the test file are
  byte-identical to `0daa7bc`:
  - 34 tests: CONFIRMED (baseline `Ran 34 tests ... OK`).
  - The same six mutations are 6/6 CAUGHT: CONFIRMED (blocks B1-B6).
  - The original run's output was never recorded. Its script was overwritten today. By the leftover copy it ran against a test
    file one test shorter than the committed one (PROVISIONAL).
  - So the claim now holds for the current code, backed by this file. At commit time it was unverifiable.
- **The six were not the full picture.** Of 21 further bench mutations, 4 SURVIVED the committed tests:
  - B10: `uncached_prefill_tok_s` computed from all prompt tokens. On warm runs the uncached prefill rate would be inflated
    by the cached tokens.
  - B22: the JSON probe passes valid JSON with wrong values.
  - B23: the tools probe accepts a wrong function name when the city matches.
  - B24: the tools probe accepts a wrong city when the function name matches.

  The wrong-answer mock cannot tell B22-B24 apart. It answers prose instead of JSON, and calls `get_stock_price` without a
  `city` argument, so the name check and the city check fail together. With the added tests the bench set is 27/27. The
  areas named in the task:
  - prefix-cache warm/cold accounting: B4, B8, B9, B10, B19;
  - TTFT from the first content chunk: B2, B3, B7;
  - exact-token prompt fitting: B11, B12, B13;
  - probe scoring: B5, B20, B21, B22, B23, B24, B25, B26;
  - mock-run flagging: B14, B15;
  - truncated stream detection: B16, plus B18 for aggregation of failures.
- **The other suites.** Their commits claimed test counts, not mutation checks. The counts are CONFIRMED by the baselines:
  - `94cbd5f` «5 tests» -> `Ran 5 tests`;
  - `a84737b` «3 stub-server tests» -> `Ran 3 tests`;
  - `2c9c05b` «4 tests» -> `Ran 4 tests`;
  - `3c2a741` «33 tests» -> `Ran 33 tests ... OK (skipped=2)`.

  Mutation results are recorded here for the first time, against the tests as of `64fa04d`:
  - `gguf_quant_audit` 7/8. G8 survived: an unknown ggml type id did not fail the verdict.
  - `logprob_compare` 2/4. The tests checked only KL > 0 or KL ≈ 0, so the KL direction (L1) and the floor margin (L2) were
    not pinned.
  - `slot_persist` 2/4. S1: dropping the erase before restore went unnoticed, and experiment K3 would then time the live
    cache instead of the restore. S3: exit code 0 after a partially failed run.
  - `vs_roofline` 2/4. V3: `--strategy`/`--miss` were tested only at the roofline defaults. V4: the mock flag was not tested.
  - offline `weights` 4/5. W4: refusing a server that ignores Range was not tested.
  - diag `summarize` 6/6.

  After the added tests every surviving mutation is caught, each by exactly the new test aimed at it (blocks above).
  Overall: 46/58 before, 58/58 after.
- **G7** (IQ4_XS size in the type table) is caught only by `test_type_table_matches_pinned_gguf_py`, which needs
  `LLAMA_CPP_GGUF_PY`. With the variable unset the mutation survives (block "G7 (env unset)").
  - The CI configuration sets the variable: `.github/workflows/ci.yml:33@64fa04d`
    (`LLAMA_CPP_GGUF_PY=../llama.cpp/gguf-py ... tests.test_gguf_quant_audit`), after cloning llama.cpp at `ec7630a`
    (`ci.yml:12-13`). ci.yml is unchanged at `d89e633` (`git diff --quiet 64fa04d HEAD -- .github/workflows/ci.yml` -> `0`).
  - Whether that workflow actually runs on the remote is UNKNOWN: `ci.yml:1` says it has to be copied/pushed with the
    Workflows permission.
- **Limits.**
  - The mutations are single-point and chosen by hand, not exhaustive.
  - `tools/bench/mock_server.py` itself was not mutated.
  - The `tools/bench` tests depend on wall-clock timing against the mock. No flaky result appeared in the 27 + 4 runs
    recorded here and the baselines; stability on a loaded machine is PROVISIONAL.
  - No real engine or llama-server was run (NOT RUN).

## tools/gateway/gateway.py (`tests/test_gateway.py`) — перепроверка ведущим

Шлюз написан субагентом engine-integrator (его отчёт: 12/12 мутаций пойманы после добавления
`test_stream_disconnect_detected_by_failed_write_when_watch_is_off`; в первом круге M6 выживала). Ведущий повторил все 12 мутаций
своим скриптом (`<S>/scripts/gw_mut.py`, вне репозитория): временный worktree от `d89e633`, в него скопированы незакоммиченные
`tools/gateway/gateway.py` (sha256 `fe607938eef4d4da2c0838ae465bbd1dd2860c53fd81f4b4be893187748f114d`) и `tests/test_gateway.py`
(sha256 `02fb74d227c0790055bc48cc08e5611883ccaee1f5bc6b2b5380c3d456e54390`); каждая мутация — `str.replace` фрагмента, который
встречается ровно один раз (assert), затем `python3 -m unittest tests.test_gateway`, затем восстановление. Worktree удалён.
Мутации (фрагмент → замена): M1 `if len(self.queue) >= self.max_queue:` → `if False:`; M2 после SSE-события ошибки дописывается
`data: [DONE]`; M3 `if keys is None: return True` → `if True: return True`; M4 тело пересериализуется `json.dumps(obj)` вместо
исходных байтов; M5 `if gw.guard is not None:` → `if False:`; M6 после неудачной записи клиенту `break` → `continue`; M7
`self.queue.append(token)` → `appendleft`; M8 `watch.add_callback(kill_upstream)` → `pass`; M9 `--set-default` без проверки
`k not in obj`; M10 `if gone():` → `if False:`; M11 `resp.read1(65536)` → `resp.read()`; M12 `resp.status >= 500` → `>= 599`.
Литеральный вывод:
```
baseline: Ran 29 tests in 2.760s / OK
M1 queue limit disabled -> CAUGHT
    FAIL: test_429_when_queue_full (tests.test_gateway.Queueing.test_429_when_queue_full)
    Ran 29 tests in 12.704s
    FAILED (failures=1)
M2 fake [DONE] after truncation error event -> CAUGHT
    FAIL: test_truncated_stream_ends_with_error_event_and_no_done (tests.test_gateway.Failures.test_truncated_stream_ends_with_error_event_and_no_done)
    Ran 29 tests in 2.717s
    FAILED (failures=1)
M3 auth check dropped -> CAUGHT
    FAIL: test_keys_file_good_bad_missing_and_never_logged (tests.test_gateway.Auth.test_keys_file_good_bad_missing_and_never_logged)
    Ran 29 tests in 2.730s
    FAILED (failures=1)
M4 body re-serialised -> CAUGHT
    FAIL: test_byte_identical_forwarding_and_same_prefix_cache_hits (tests.test_gateway.BodyFidelity.test_byte_identical_forwarding_and_same_prefix_cache_hits)
    FAIL: test_set_default_injected_only_when_absent (tests.test_gateway.BodyFidelity.test_set_default_injected_only_when_absent)
    Ran 29 tests in 2.794s
    FAILED (failures=2)
M5 context guard skipped -> CAUGHT
    ERROR: test_chat_without_apply_template_passes_unchecked (tests.test_gateway.ContextGuard.test_chat_without_apply_template_passes_unchecked)
    ERROR: test_pass_through_when_backend_has_no_tokenize (tests.test_gateway.ContextGuard.test_pass_through_when_backend_has_no_tokenize)
    FAIL: test_chat_guard_uses_apply_template (tests.test_gateway.ContextGuard.test_chat_guard_uses_apply_template)
    FAIL: test_reject_completion_over_limit_before_engine (tests.test_gateway.ContextGuard.test_reject_completion_over_limit_before_engine)
    Ran 29 tests in 2.726s
    FAILED (failures=2, errors=2)
M6 keep relaying after failed client write -> CAUGHT
    FAIL: test_stream_disconnect_detected_by_failed_write_when_watch_is_off (tests.test_gateway.Disconnect.test_stream_disconnect_detected_by_failed_write_when_watch_is_off)
    Ran 29 tests in 9.696s
    FAILED (failures=1)
M7 LIFO queue -> CAUGHT
    FAIL: test_fifo_order (tests.test_gateway.Queueing.test_fifo_order)
    Ran 29 tests in 2.796s
    FAILED (failures=1)
M8 watcher does not close upstream -> CAUGHT
    FAIL: test_disconnect_while_engine_prefills_closes_upstream (tests.test_gateway.Disconnect.test_disconnect_while_engine_prefills_closes_upstream)
    Ran 29 tests in 7.755s
    FAILED (failures=1)
M9 --set-default overrides client keys -> CAUGHT
    FAIL: test_apply_defaults_unit (tests.test_gateway.BodyFidelity.test_apply_defaults_unit)
    FAIL: test_set_default_injected_only_when_absent (tests.test_gateway.BodyFidelity.test_set_default_injected_only_when_absent)
    Ran 29 tests in 2.784s
    FAILED (failures=2)
M10 disconnected queued client stays queued -> CAUGHT
    FAIL: test_queued_client_disconnect_leaves_queue (tests.test_gateway.Queueing.test_queued_client_disconnect_leaves_queue)
    Ran 29 tests in 7.765s
    FAILED (failures=1)
M11 stream buffered -> CAUGHT
    ERROR: test_stream_disconnect_closes_upstream (tests.test_gateway.Disconnect.test_stream_disconnect_closes_upstream)
    ERROR: test_stream_disconnect_detected_by_failed_write_when_watch_is_off (tests.test_gateway.Disconnect.test_stream_disconnect_detected_by_failed_write_when_watch_is_off)
    ERROR: test_stream_passthrough_equals_direct_and_is_not_buffered (tests.test_gateway.Passthrough.test_stream_passthrough_equals_direct_and_is_not_buffered)
    FAIL: test_truncated_stream_ends_with_error_event_and_no_done (tests.test_gateway.Failures.test_truncated_stream_ends_with_error_event_and_no_done)
    Ran 29 tests in 22.543s
    FAILED (failures=1, errors=3)
M12 upstream 5xx passed through -> CAUGHT
    FAIL: test_http500_before_first_byte_maps_to_502 (tests.test_gateway.Failures.test_http500_before_first_byte_maps_to_502)
    Ran 29 tests in 2.735s
    FAILED (failures=1)
restored identical: True
caught 12/12
```
