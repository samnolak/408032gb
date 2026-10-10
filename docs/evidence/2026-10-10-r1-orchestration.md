# Оркестрация ячейки R1 одной командой (`scripts/r1`, трек N4): тесты, сухой прогон на mock, мутационная проверка (2026-10-10, облачная песочница)

Автор: perf-bench. Ярлыки — по `AGENTS.md`. Всё ниже снято в песочнице без GPU: «движок» — `tools/bench/mock_server.py`,
«модель» — синтетический GGUF без данных (`scripts/r1/make_mock_model.py`). **Здесь нет ни одного числа о движке или модели;
ни один реальный движок через оркестратор не запускался — NOT RUN.** Документ подтверждает только поведение самого
оркестратора: порядок шагов, журнал, остановку движка, вердикт пакета на mock-прогоне.

## Условия (литерально)

`<S>` = `/tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad` (временный каталог сессии, не
сохраняется). В литеральных выводах ниже этот путь заменён на `<S>`; больше ничего не менялось.

```
$ git log --oneline -n 7          # ветка claude/n4-r1-orchestration, до коммита этого файла
bc516b4 [perf-bench] tests/test_r1_orchestration: the interrupted tool's exit code proves it got SIGINT
bb6fbab [perf-bench] tests/test_r1_orchestration: an interrupt escaping run_r1.main fails the test instead of aborting the run
da1b4bd [perf-bench] tests/test_r1_orchestration: a signal inside a shielded section is deferred, not lost
f48fc30 [perf-bench] scripts/r1/README.md: how the operator runs one R1 cell, what is checked automatically, what stays NOT RUN
870c654 [perf-bench] scripts/r1/configs: PROVISIONAL engine templates (llama.cpp, llama.cpp --moe-cache-mib, ik_llama.cpp, Strata serve)
71d9455 [perf-bench] scripts/r1: one-command R1 orchestration (run_r1.py), mock model and mock config, tests
163012a [infra-ops] collect.sh: silence the trap-handler notice for shellcheck 0.9 too (SC2317; 0.10+ calls it SC2329) - CI test_shellcheck failed on the runner's shellcheck 0.9.0; reproduced and verified clean with 0.9.0 and 0.11.0
eb59224 is an ancestor of 163012a
$ python3 --version
Python 3.13.16
```

Ветка создана от `163012a` (вершина `claude/relaxed-dijkstra-1d7i5o`), а не от `eb59224` (HEAD, с которым был создан рабочий
каталог агента): в `eb59224` нет ни `docs/reassessment/`, ни `tools/bench/`, `tools/verify/` и других инструментов, которые
задача требует вызывать. `eb59224` — предок `163012a` (последняя строка блока выше), так что ветка — fast-forward рабочего HEAD.
Существующие инструменты не менялись (CONFIRMED: список файлов, изменённых веткой до коммита этого файла):

```
$ git diff --name-only 163012a bc516b4
scripts/r1/README.md
scripts/r1/configs/ik-llamacpp-server.json
scripts/r1/configs/llamacpp-server-moecache.json
scripts/r1/configs/llamacpp-server.json
scripts/r1/configs/mock.json
scripts/r1/configs/strata-serve-qwen38fn.json
scripts/r1/make_mock_model.py
scripts/r1/run_r1.py
tests/test_r1_orchestration.py
```

## 1. Тесты оркестратора (HEAD `bc516b4`)

```
$ python3 -m unittest tests.test_r1_orchestration -v
test_engine_that_exits_early_fails_health_at_once (tests.test_r1_orchestration.FailurePaths.test_engine_that_exits_early_fails_health_at_once) ... ok
test_exception_inside_a_step_is_recorded_and_the_run_continues (tests.test_r1_orchestration.FailurePaths.test_exception_inside_a_step_is_recorded_and_the_run_continues) ... ok
test_failing_steps_are_recorded_and_the_engine_is_still_stopped (tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped) ... ok
test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine (tests.test_r1_orchestration.FailurePaths.test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine) ... ok
test_keyboard_interrupt_stops_the_engine_and_skips_the_rest (tests.test_r1_orchestration.FailurePaths.test_keyboard_interrupt_stops_the_engine_and_skips_the_rest) ... ok
test_signal_in_a_shielded_section_is_deferred_not_lost (tests.test_r1_orchestration.FailurePaths.test_signal_in_a_shielded_section_is_deferred_not_lost) ... ok
test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) ... ok
test_engine_recorded_and_stopped (tests.test_r1_orchestration.FullMockRun.test_engine_recorded_and_stopped) ... ok
test_exit_code_and_layout (tests.test_r1_orchestration.FullMockRun.test_exit_code_and_layout) ... ok
test_journal_has_every_item_in_order_with_outputs (tests.test_r1_orchestration.FullMockRun.test_journal_has_every_item_in_order_with_outputs) ... ok
test_pack_is_not_run_because_of_the_mock (tests.test_r1_orchestration.FullMockRun.test_pack_is_not_run_because_of_the_mock) ... ok
test_prediction_is_written_before_the_engine_starts (tests.test_r1_orchestration.FullMockRun.test_prediction_is_written_before_the_engine_starts) ... ok
test_config_errors_exit_2 (tests.test_r1_orchestration.Plans.test_config_errors_exit_2) ... ok
test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) ... ok
test_plan_runs_nothing (tests.test_r1_orchestration.Plans.test_plan_runs_nothing) ... ok

----------------------------------------------------------------------
Ran 15 tests in 10.094s

OK
```

Что покрыто (15 тестов): план каждого шаблона подставляет все переменные, содержит все 15 шагов в порядке `STEPS`,
предсказание roofline стоит до `engine_start`, флаги режима предсказания и `vs_roofline` совпадают, pack собирается из
`RUN/artifacts`; `--plan` ничего не запускает и не создаёт (маркеры движка, версии, `after_ready`, `mkdirs` не появляются);
ошибки шаблона — код 2; полный прогон `mock.json` (без правок, только `--set MOCK_MODEL_DIR`) даёт полный каталог, журнал по
одной записи `end` на пункт плана, предсказание до старта движка (по `seq` и времени, sha256 предсказаний сверен), пакет с
вердиктом NOT RUN и причиной `mock` у всех критериев на прогонах bench, `pack.py --check` = 0; упавшие шаги (нет файла модели,
sha256-проверка весов MISSING, аудит битности без GGUF, `logprob_compare` против mock, «мягкая» команда `after_ready`)
записаны как failed, остальное выполнено, движок остановлен; таймаут health — failed, зависимые шаги skipped с `blocked_by`,
упрямый движок (игнорирует SIGTERM) убит SIGKILL (код -9); движок, завершившийся сразу, ловится без ожидания таймаута;
исключение внутри шага записано, прогон продолжен; KeyboardInterrupt в процессе — код 130, остальное skipped, движок
остановлен; настоящий SIGINT и SIGTERM процессу оркестратора — коды 130/143, движок и работающий инструмент завершены
(инструмент получил SIGINT: его код 130), процессов прогона не осталось; сигнал, пришедший в защищённой секции, не теряется.

Соседние наборы (инструменты, которые вызывает оркестратор; они не менялись — проверка, что ничего не сломано):

```
$ python3 -m unittest tests.test_bench tests.test_verify_pack tests.test_vs_roofline tests.test_roofline tests.test_offline_weights tests.test_gguf_quant_audit tests.test_slot_persist tests.test_logprob_compare tests.test_team_config   # последние 3 строки вывода (выше в выводе — печать самих тестов и ResourceWarning)
Ran 114 tests in 38.833s

OK (skipped=1)
```
Пропущенный тест — `test_type_table_matches_pinned_gguf_py` (`skipped 'set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py'`, вывод
`python3 -m unittest tests.test_gguf_quant_audit -v` в этой сессии); к оркестратору не относится.

## 2. Сухой прогон на mock одной командой (HEAD `bc516b4`)

```
$ python3 scripts/r1/make_mock_model.py <S>/n4/ev/mock-model
SYNTHETIC mock model (not evidence): <S>/n4/ev/mock-model/SYNTHETIC-mock-model-Q4_K.gguf, manifest <S>/n4/ev/mock-model/SYNTHETIC-mock-model.manifest.json
$ python3 scripts/r1/run_r1.py --config scripts/r1/configs/mock.json --out <S>/n4/ev     # exit 0 (echo $?)
[r1] run directory: <S>/n4/ev/20261010T115645Z-mock
[r1] MOCK CONFIG: the 'engine' is tools/bench/mock_server.py; nothing in this run is evidence
[r1] weights_verify                     ok (exit 0)
[r1] quant_audit                        ok (exit 0)
[r1] roofline_prediction[ctx=1024]      ok (exit 0)
[r1] roofline_prediction[ctx=4096]      ok (exit 0)
[r1] roofline_prediction[record]        ok
[r1] nvidia_smi_before                  skipped: nvidia-smi not found on this host
[r1] engine_version[version]            ok (exit 0)
[r1] engine_version[help]               ok (exit 0)
[r1] engine_start                       ok: started (pid 20189); health is checked by health_wait
[r1] health_wait                        ok: http://127.0.0.1:33457/health answered HTTP 200 after 0.1 s
[r1] matrix[smoke-1k]                   ok (exit 0)
[r1] matrix[filled-4k]                  ok (exit 0)
[r1] probes[4k]                         ok (exit 0)
[r1] slot_persist                       skipped: engine.supports_slots is false (no /slots save/restore): V7 stays NOT RUN
[r1] logprob_compare                    skipped: no reference endpoint ("reference": {"url": ...}): V6 stays NOT RUN
[r1] vs_roofline                        ok (exit 0)
[r1] engine_stop                        ok (exit -15)
[r1] nvidia_smi_after                   skipped: nvidia-smi not found on this host
[r1] pack[build]                        ok (exit 3)
[r1] pack[check]                        ok (exit 0)
[r1] steps: weights_verify=ok, quant_audit=ok, roofline_prediction=ok, nvidia_smi_before=skipped, engine_version=ok, engine_start=ok, health_wait=ok, matrix=ok, probes=ok, slot_persist=skipped, logprob_compare=skipped, vs_roofline=ok, engine_stop=ok, nvidia_smi_after=skipped, pack=ok
[r1] verification pack: NOT RUN (mandatory criteria NOT RUN: single_configuration, openai_api, context_probes_full_length, output_after_full_prompt, agent_api_probes, decode_filled_ctx_floor, p) -> pack/VERDICT.md
[r1] criteria NOT RUN because the server was the mock: agent_api_probes, agent_loop, context_probes_full_length, decode_filled_ctx_floor, decode_filled_ctx_target, decode_vs_roofline, determinism, openai_api, output_after_full_prompt, prefix_cache, single_configuration
[r1] journal: <S>/n4/ev/20261010T115645Z-mock/steps.jsonl
$ python3 tools/verify/pack.py --check <S>/n4/ev/20261010T115645Z-mock/pack                          # exit 0 (echo $?)
check: <S>/n4/ev/20261010T115645Z-mock/pack: 51 files listed, 0 problem(s); manifest sha256 886397e632785ab67d6275ab5fa76d49e2caadea6cbe4435ee2cd162649e0b94; recorded overall verdict: NOT RUN
$ cat <S>/n4/ev/20261010T115645Z-mock/logs/pack.stdout.txt          # вывод сборки пакета внутри прогона
NOT RUN  single_configuration        mock: artifacts/matrix/filled-4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md car
NOT RUN  openai_api                  mock: artifacts/matrix/filled-4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md car
NOT RUN  context_probes_full_length  mock: artifacts/probes/4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md carries th
NOT RUN  output_after_full_prompt    mock: artifacts/probes/4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md carries th
NOT RUN  agent_api_probes            mock: artifacts/probes/4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md carries th
NOT RUN  decode_filled_ctx_floor     mock: artifacts/matrix/filled-4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md car
NOT RUN  decode_filled_ctx_target    (optional) mock: artifacts/matrix/filled-4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md car
NOT RUN  prefix_cache                mock: artifacts/matrix/filled-4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md car
NOT RUN  agent_loop                  mock: artifacts/matrix/filled-4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md car
PASS     quant_min_bpw               every tensor type >= 4.0 bpw, no findings
PASS     weights_integrity           every manifest file verified (size + sha256) in the verify log(s)
NOT RUN  exact_routing_logprobs      no tools/quality/logprob_compare.py --out file under --runs
NOT RUN  kv_slot_persist             no tools/kv/slot_persist.py --out file under --runs
NOT RUN  decode_vs_roofline          mock: artifacts/roofline/vs_roofline.json has rows of mock runs (filled-4k, smoke-1k)
NOT RUN  hardware_facts              no scripts/diag/summarize.py --json output under --runs
NOT RUN  determinism                 (optional) mock: artifacts/probes/4k (meta.is_mock_server = true, summary.is_mock_server = true, /v1/models owned_by = 408032gb-bench-mock, summary.md carries th
NOT RUN  manual_operational_faults   (optional) operational faults F1-F8 (docs/reassessment/DEPLOY.md section 6) are judged by a person from literal logs in docs/evidence/R5-faults-<date>.md; this p
NOT RUN  manual_quality_eval         (optional) the customer's evaluation set (docs/reassessment/R2-EVAL-SET.md) is graded by people and by the customer's own harness; this pack does not judge it
overall: NOT RUN (mandatory criteria NOT RUN: single_configuration, openai_api, context_probes_full_length, output_after_full_prompt, agent_api_probes, decode_filled_ctx_floor, prefix_cache, agent_loop, exact_routing_l); 51 files in <S>/n4/ev/20261010T115645Z-mock/pack/MANIFEST.sha256
```

Вердикт пакета — **NOT RUN**: каждый критерий, опирающийся на прогоны `tools/bench`, — NOT RUN с причиной `mock`
(CONFIRMED выводом выше). V1 (`quant_min_bpw`) и V2 (`weights_integrity`) — PASS **на синтетическом GGUF** без данных: так
работают шаги аудита и проверки весов, а не свойство какой-либо модели; пакет помечает эти файлы плашкой SYNTHETIC
(`synthetic_markers` в `verdict.json`). `nvidia_smi_*` — skipped: в песочнице нет `nvidia-smi`. `slot_persist` и
`logprob_compare` — skipped по шаблону (у mock нет `/slots` и `logprobs`).

## 3. Мутационная проверка `scripts/r1/run_r1.py`

Метод: дерево коммита выгружено `git archive <commit> | tar -x -C <S>/n4/mut` (не git-checkout: рабочий каталог агента ни разу
не мутировался); одна мутация за раз — `str.replace` фрагмента, который встречается в `run_r1.py` ровно один раз (assert);
`python3 -m unittest tests.test_r1_orchestration` из выгруженного дерева с `PYTHONDONTWRITEBYTECODE=1`; записываются строки
`FAIL:`/`ERROR:` и `Ran`/`OK`/`FAILED`; затем файл восстанавливается и сверяется по sha256 с исходным; процессы, которые могла
оставить сломанная мутация, ищутся по `/proc/*/cmdline` и убиваются (их число записано). CAUGHT — модуль тестов завершился
с ненулевым кодом, SURVIVED — с нулевым. Скрипт-драйвер лежал в `<S>/n4/drv/mutate_r1.py` (в репозиторий не входит); строки
`-`/`+` каждого блока достаточно, чтобы повторить мутацию вручную. sha256 `run_r1.py` во всех трёх раундах одинаков
(`b7df922094564cdb68dd060e1107f99ede3b929824a745f0cb66c6afef468f7a`, `sha256sum` в этой сессии): менялись только тесты.

| раунд | коммит (тесты) | пойманы | что выяснилось | что сделано |
|---|---|---|---|---|
| 1 | `da1b4bd` | 16 / 16 по коду выхода | M13: KeyboardInterrupt вышел из `run_r1.main` и оборвал весь прогон unittest (код -2, без строки FAIL) — «пойман», но не назван | `bb6fbab`: `run_main` в тестах превращает такое прерывание в AssertionError |
| 2 | `bb6fbab` | **15 / 16** | M10 (SIGINT не пересылается работающему инструменту) — SURVIVED, хотя в раунде 1 подтест SIGINT на ней упал (блок ниже): обнаружение зависело от того, жив ли ещё осиротевший инструмент в момент проверки — гонка (PROVISIONAL: какая именно проверка сработала в раунде 1, не записано — сохранялись только строки FAIL/ERROR) | `bc516b4`: код выхода прерванного шага матрицы обязан показать SIGINT (130 или -2); без пересылки оркестратор записывает `None` |
| 3 | `bc516b4` | **16 / 16** | — | M10 повторена ещё 2 раза: 2 / 2 CAUGHT |

### Раунд 3 (`bc516b4`) — литерально
### M1 - engine_stop skipped after a failed step (stop only on a clean run)
```
-         if p is None:
-             return self.skip(it, "the engine was not started by this run")
+         if p is None or "failed" in self.step_state.values():
+             return self.skip(it, "the engine was not started by this run")
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 9.3 s
ERROR: test_engine_that_exits_early_fails_health_at_once (tests.test_r1_orchestration.FailurePaths.test_engine_that_exits_early_fails_health_at_once)
ERROR: test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine (tests.test_r1_orchestration.FailurePaths.test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine)
FAIL: test_exception_inside_a_step_is_recorded_and_the_run_continues (tests.test_r1_orchestration.FailurePaths.test_exception_inside_a_step_is_recorded_and_the_run_continues)
FAIL: test_failing_steps_are_recorded_and_the_engine_is_still_stopped (tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped)
Ran 15 tests in 9.181s
FAILED (failures=2, errors=2)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M2 - tail steps (engine_stop, nvidia_smi_after, pack) only on the success path, not in finally
```
-         finally:
-             self.finish(tail)
+         else:
+             self.finish(tail)
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.5 s
ERROR: test_keyboard_interrupt_stops_the_engine_and_skips_the_rest (tests.test_r1_orchestration.FailurePaths.test_keyboard_interrupt_stops_the_engine_and_skips_the_rest)
ERROR: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
ERROR: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGTERM')
Ran 15 tests in 10.400s
FAILED (errors=3)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M3 - roofline prediction after the engine start and health wait (STEPS order)
```
- STEPS = ["weights_verify", "quant_audit", "roofline_prediction", "nvidia_smi_before", "engine_version",
-          "engine_start", "health_wait", "matrix",
+ STEPS = ["weights_verify", "quant_audit", "nvidia_smi_before", "engine_version",
+          "engine_start", "health_wait", "roofline_prediction", "matrix",
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.7 s
FAIL: test_signal_in_a_shielded_section_is_deferred_not_lost (tests.test_r1_orchestration.FailurePaths.test_signal_in_a_shielded_section_is_deferred_not_lost)
FAIL: test_prediction_is_written_before_the_engine_starts (tests.test_r1_orchestration.FullMockRun.test_prediction_is_written_before_the_engine_starts)
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='ik-llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server-moecache.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='mock.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='strata-serve-qwen38fn.json')
Ran 15 tests in 10.525s
FAILED (failures=7)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M4 - a failed step reported as ok (exit code ignored)
```
-         status = "ok" if rc in it.ok_codes and not timed_out else "failed"
+         status = "ok" if not timed_out else "failed"
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.1 s
FAIL: test_failing_steps_are_recorded_and_the_engine_is_still_stopped (tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped)
Ran 15 tests in 9.929s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M5 - health timeout reported as ok
```
-                 return self.end(it, t, "failed", reason="timeout after %g s; last answer: %s" % (timeout, last),
+                 return self.end(it, t, "ok", reason="timeout after %g s; last answer: %s" % (timeout, last),
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.6 s
FAIL: test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine (tests.test_r1_orchestration.FailurePaths.test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine)
Ran 15 tests in 10.508s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M6 - pack built from the wrong directory (artifacts/matrix instead of artifacts)
```
-             argv = self.tool("pack", "--runs", self.artifacts, "--out", pack_dir,
+             argv = self.tool("pack", "--runs", self.a("matrix"), "--out", pack_dir,
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.2 s
FAIL: test_pack_is_not_run_because_of_the_mock (tests.test_r1_orchestration.FullMockRun.test_pack_is_not_run_because_of_the_mock)
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='ik-llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server-moecache.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='mock.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='strata-serve-qwen38fn.json')
Ran 15 tests in 10.037s
FAILED (failures=6)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M7 - plan mode falls through and executes the run
```
-         print(json.dumps(cell.plan_dict(selected), indent=1) if a.json else cell.plan_text(selected))
-         return 0
- 
+         print(json.dumps(cell.plan_dict(selected), indent=1) if a.json else cell.plan_text(selected))
+ 
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 16.0 s
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='ik-llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server-moecache.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='mock.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='strata-serve-qwen38fn.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first)
FAIL: test_plan_runs_nothing (tests.test_r1_orchestration.Plans.test_plan_runs_nothing)
Ran 15 tests in 15.876s
FAILED (failures=7)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M8 - no SIGKILL after the stop timeout (SIGTERM again instead)
```
-                         os.killpg(p.pid, signal.SIGKILL)
-                     signals.append("SIGKILL")
+                         os.killpg(p.pid, signal.SIGTERM)
+                     signals.append("SIGKILL")
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 40.4 s
FAIL: test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine (tests.test_r1_orchestration.FailurePaths.test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine)
Ran 15 tests in 40.265s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M9 - steps run although a step they depend on failed
```
-             if st in ("failed", "interrupted", "blocked"):
+             if st in ("interrupted",):
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.9 s
FAIL: test_engine_that_exits_early_fails_health_at_once (tests.test_r1_orchestration.FailurePaths.test_engine_that_exits_early_fails_health_at_once)
FAIL: test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine (tests.test_r1_orchestration.FailurePaths.test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine)
Ran 15 tests in 10.725s
FAILED (failures=2)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M10 - Ctrl-C not forwarded to the running tool
```
-             except KeyboardInterrupt as e:
-                 rc = self.end_child(p)
+             except KeyboardInterrupt as e:
+                 rc = p.poll()
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.1 s
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGTERM')
Ran 15 tests in 9.947s
FAILED (failures=2)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M11 - health_wait does not notice an engine that exited
```
-             if self.engine is not None and self.engine.poll() is not None:
+             if False and self.engine is not None and self.engine.poll() is not None:
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 30.3 s
FAIL: test_engine_that_exits_early_fails_health_at_once (tests.test_r1_orchestration.FailurePaths.test_engine_that_exits_early_fails_health_at_once)
Ran 15 tests in 30.121s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M12 - strings inside lists of the config are not substituted
```
-     if isinstance(obj, list):
-         return [substitute(v, var, "%s[%d]" % (path, i), typed) for i, v in enumerate(obj)]
+     if isinstance(obj, list):
+         return obj
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 64.1 s
ERROR: test_failing_steps_are_recorded_and_the_engine_is_still_stopped (tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped)
ERROR: test_engine_recorded_and_stopped (tests.test_r1_orchestration.FullMockRun.test_engine_recorded_and_stopped)
FAIL: test_exception_inside_a_step_is_recorded_and_the_run_continues (tests.test_r1_orchestration.FailurePaths.test_exception_inside_a_step_is_recorded_and_the_run_continues)
FAIL: test_keyboard_interrupt_stops_the_engine_and_skips_the_rest (tests.test_r1_orchestration.FailurePaths.test_keyboard_interrupt_stops_the_engine_and_skips_the_rest)
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGTERM')
FAIL: test_exit_code_and_layout (tests.test_r1_orchestration.FullMockRun.test_exit_code_and_layout)
FAIL: test_journal_has_every_item_in_order_with_outputs (tests.test_r1_orchestration.FullMockRun.test_journal_has_every_item_in_order_with_outputs)
FAIL: test_pack_is_not_run_because_of_the_mock (tests.test_r1_orchestration.FullMockRun.test_pack_is_not_run_because_of_the_mock)
FAIL: test_config_errors_exit_2 (tests.test_r1_orchestration.Plans.test_config_errors_exit_2) (case='unknown-var')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='ik-llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server-moecache.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='mock.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='strata-serve-qwen38fn.json')
Ran 15 tests in 64.002s
FAILED (failures=13, errors=2)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M13 - KeyboardInterrupt / signals not handled by the step loop
```
-         except KeyboardInterrupt as e:
-             self.interrupted = getattr(e, "signum", signal.SIGINT)
-             cur = self.current
+         except Exception as e:
+             self.interrupted = getattr(e, "signum", signal.SIGINT)
+             cur = self.current
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.8 s
FAIL: test_keyboard_interrupt_stops_the_engine_and_skips_the_rest (tests.test_r1_orchestration.FailurePaths.test_keyboard_interrupt_stops_the_engine_and_skips_the_rest)
FAIL: test_signal_in_a_shielded_section_is_deferred_not_lost (tests.test_r1_orchestration.FailurePaths.test_signal_in_a_shielded_section_is_deferred_not_lost)
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGTERM')
Ran 15 tests in 10.687s
FAILED (failures=4)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M14 - a tool's non-JSON stdout is kept under the .json name
```
-             if os.path.exists(target):
-                 os.replace(target, alt)
-                 return alt
+             if os.path.exists(target):
+                 os.replace(target, it.stdout)
+                 return it.stdout
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.2 s
FAIL: test_failing_steps_are_recorded_and_the_engine_is_still_stopped (tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped)
Ran 15 tests in 10.104s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M15 - a signal deferred in a shielded section is never raised
```
-         if self.pending_signal and not self.critical and not self.finishing:
+         if False:
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.4 s
FAIL: test_signal_in_a_shielded_section_is_deferred_not_lost (tests.test_r1_orchestration.FailurePaths.test_signal_in_a_shielded_section_is_deferred_not_lost)
Ran 15 tests in 10.283s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M16 - vs_roofline compares with another engine mode than the prediction (--overlap yes hard-coded)
```
-         out = ["--strategy", m["strategy"], "--miss", m["miss"], "--overlap", m["overlap"], "--spec-k", str(m["spec_k"]),
+         out = ["--strategy", m["strategy"], "--miss", m["miss"], "--overlap", "yes", "--spec-k", str(m["spec_k"]),
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.2 s
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='ik-llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server-moecache.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='llamacpp-server.json')
FAIL: test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) (config='mock.json')
Ran 15 tests in 10.113s
FAILED (failures=4)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### Повтор M10 на `bc516b4` (2 раза) — литерально
### M10 - Ctrl-C not forwarded to the running tool
```
-             except KeyboardInterrupt as e:
-                 rc = self.end_child(p)
+             except KeyboardInterrupt as e:
+                 rc = p.poll()
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.1 s
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGTERM')
Ran 15 tests in 9.944s
FAILED (failures=2)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M10 - Ctrl-C not forwarded to the running tool
```
-             except KeyboardInterrupt as e:
-                 rc = self.end_child(p)
+             except KeyboardInterrupt as e:
+                 rc = p.poll()
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.3 s
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGTERM')
Ran 15 tests in 10.202s
FAILED (failures=2)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### Раунд 2 (`bb6fbab`): выжившая M10 — литерально
### M10 - Ctrl-C not forwarded to the running tool
```
-             except KeyboardInterrupt as e:
-                 rc = self.end_child(p)
+             except KeyboardInterrupt as e:
+                 rc = p.poll()
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 0, 9.9 s
Ran 15 tests in 9.799s
OK
-> SURVIVED; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### Раунд 1 (`da1b4bd`): M10 поймана, M13 без строки FAIL — литерально
### M10 - Ctrl-C not forwarded to the running tool
```
-             except KeyboardInterrupt as e:
-                 rc = self.end_child(p)
+             except KeyboardInterrupt as e:
+                 rc = p.poll()
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit 1, 10.2 s
FAIL: test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) (signal='SIGINT')
Ran 15 tests in 10.022s
FAILED (failures=1)
-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

### M13 - KeyboardInterrupt / signals not handled by the step loop
```
-         except KeyboardInterrupt as e:
-             self.interrupted = getattr(e, "signum", signal.SIGINT)
-             cur = self.current
+         except Exception as e:
+             self.interrupted = getattr(e, "signum", signal.SIGINT)
+             cur = self.current
$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_r1_orchestration   # exit -2, 4.7 s

-> CAUGHT; run_r1.py restored (sha256 equal to the original: True); leftover processes killed: 0
```

Первый запуск раунда 1 оборвался после M1 по ошибке драйвера (поиск оставшихся процессов совпал с командной строкой
запустившей его оболочки и убил её; осиротевший драйвер и его тестовый процесс остановлены вручную `kill`); драйвер исправлен (ищутся только процессы инструментов и оркестратора дерева, предки
драйвера исключены), выгрузка пересоздана (`sha256` исходный), раунд 1 повторён целиком — блок выше из повтора.

## 4. Что эти данные НЕ показывают (PROVISIONAL / UNKNOWN / NOT RUN)

- Реальные движки (llama.cpp, ik_llama.cpp, Strata) через оркестратор — **NOT RUN**: нет сервера. Флаги шаблонов не сверены с
  `--help` реальных сборок (это сделает шаг `engine_version` на сервере).
- Шаблоны движков — **PROVISIONAL**. Источники флагов (прочитаны в этой сессии):
  llama.cpp `tools/server/README.md` строки 35, 50, 56, 71–72, 82–83, 85–87, 100, 177, 193–194, 222–223, 226, 232, 472–484 —
  из рабочей копии в `<S>/eng-src/llama.cpp`, чей файл `.git/HEAD` содержит `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`
  (строки распечатаны `awk` в этой сессии; что рабочее дерево не изменено относительно этого коммита, git-командой не проверено:
  git вне рабочего каталога агенту запрещён — PROVISIONAL в этой части);
  Strata `docs/UNSLOTH_Q4.md:107-136` (JSON сервера и строка запуска), `:31-37`, `:84-87`, `:138`, `:156` — из частичного клона
  `<S>/strata-src`, у которого в `.git/packed-refs` `refs/heads/main` = `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (та же оговорка);
  ik_llama — команда хоста 3×3090 `scripts/run-glm-3090-1gpu.sh:29-34` и `PROBLEMS.md:105` (в репозитории);
  `docs/reassessment/research/R2-engines.md:61,66,116,136,137`, `docs/reassessment/ADR-012-reassessment.md:97` (в репозитории).
- Значения режима и битности в шаблонах (`HIT`, `EXPERT_BPW`, `DENSE_BPW`, `SPEC_ACCEPT`, `NCMOE`, `MOE_CACHE_MIB`) —
  PROVISIONAL входы предрегистрации (источники — `_notes` шаблонов). Для `llamacpp-server-moecache.json` GGUF архитектуры
  `glm5-next` (файл, ревизия, манифест) — UNKNOWN; для GLM UD-Q4_K_XL ревизия и манифест — UNKNOWN.
- Длительность полной ячейки на 262K, время загрузки весов и остановки реального движка — UNKNOWN до сервера.
- Тест SIGINT/SIGTERM проверяет пересылку сигнала инструменту `run_matrix`; поведение других инструментов при SIGINT в момент
  ожидания HTTP-ответа на 262K не проверялось.
