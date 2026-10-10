# Стенд агентного оценочного набора `tools/eval` (N3): тесты на моке и мутационная проверка (2026-10-10)

Роль: parity-qa. Задача N3 трека без железа (`docs/reassessment/ROADMAP.md`, раздел «Трек без железа») для R2.2.
Всё ниже — облачная песочница разработки, **без GPU и без реального движка**: стенд проверен только против скриптового
мок-сервера `tools/eval/mock_agent_server.py` (и одного прогона против `tools/bench/mock_server.py` для проверки признака мока).
Против llama.cpp / ik_llama.cpp / vLLM / SGLang / TabbyAPI / Strata — **NOT RUN**. Синтетический набор
`tools/eval/tasks-synthetic/` (15 задач) — не доказательство качества какой-либо модели.

## Условия (литерально)
Ветка `claude/n3-eval-stand` создана от `163012a` (ветка `claude/relaxed-dijkstra-1d7i5o`): HEAD рабочего дерева агента был
`eb59224` (`main`), где нет `tools/bench/` и `docs/reassessment/`, которые задача требует использовать.
```
$ git rev-parse HEAD
d52d379bf84302f3a5d80955f518e93fb4133a6f
$ git log --oneline -n 3
d52d379 [parity-qa] tests/test_eval: turn limit as the only failing tool-use sub-check (mutation M8 survived on 4bb3780)
4bb3780 [parity-qa] tools/eval: agentic evaluation stand for R2.2 (N3) - task files, runner over any OpenAI-compatible endpoint, sandboxed code checks, tool simulator, paired comparator, scripted mock, 15 SYNTHETIC tasks, tests
163012a [infra-ops] collect.sh: silence the trap-handler notice for shellcheck 0.9 too (SC2317; 0.10+ calls it SC2329) - CI test_shellcheck failed on the runner's shellcheck 0.9.0; reproduced and verified clean with 0.9.0 and 0.11.0
$ git merge-base HEAD claude/relaxed-dijkstra-1d7i5o
163012a6292a1dfe5c079ab5ebad19b4d3da3320
$ python3 --version
Python 3.13.16
$ uname -r
6.18.44-fc-v114
$ id -u
0
$ unshare --version
unshare from util-linux 2.39.3
$ python3 <S>/caps.py      # ev_sandbox.capabilities() and missing_isolation()
{"landlock_abi": 7, "landlock_fs": true, "landlock_net": true, "netns": true, "netns_prefix": ["unshare", "--net", "--pid", "--fork", "--kill-child"], "unshare": "/usr/bin/unshare"}
missing: []
```
`<S>` ниже — временный каталог сессии под `/tmp` (не сохраняется).

Что CONFIRMED этим файлом (вывод ниже): 34 теста `tests/test_eval.py` проходят на `d52d379`; полный набор `tests/`
(как в CI, `python3 -m unittest discover -s tests`) проходит на `d52d379`; `--validate-only` на синтетическом наборе — все 15 OK;
на этой машине доступны все слои песочницы (unshare network+PID namespace, Landlock ABI 7 с TCP); мутационная проверка —
15/16 на `4bb3780`, 16/16 на `d52d379` после добавленного теста.
Что UNKNOWN: доступность `unshare`/Landlock на VM заказчика (покажет первый `--validate-only` там); поведение реальных
движков (потоковые tool calls, `response_format`, `role: tool`) через этот стенд — NOT RUN. Тесты шли от root (`id -u` = 0),
поэтому вариант для обычного пользователя (`unshare --user --map-root-user --net ...`) здесь не исполнялся — NOT RUN;
Landlock от root не зависит.

## `python3 -m unittest tests.test_eval -v` на `d52d379`
```
$ git rev-parse --short HEAD
d52d379
$ python3 -m unittest tests.test_eval -v
test_extract_and_normalize (tests.test_eval.Answers.test_extract_and_normalize) ... ok
test_matchers (tests.test_eval.Answers.test_matchers) ... ok
test_identical_runs (tests.test_eval.Compare.test_identical_runs) ... ok
test_known_example (tests.test_eval.Compare.test_known_example) ... ok
test_all_correct_answers_pass_every_category (tests.test_eval.EndToEnd.test_all_correct_answers_pass_every_category) ... ok
test_dead_server_gives_request_errors_only (tests.test_eval.EndToEnd.test_dead_server_gives_request_errors_only) ... ok
test_mock_detection_matches_bench (tests.test_eval.EndToEnd.test_mock_detection_matches_bench) ... ok
test_non_streaming_mode_and_repeats (tests.test_eval.EndToEnd.test_non_streaming_mode_and_repeats) ... ok
test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately) ... ok
test_tool_loop_variants (tests.test_eval.EndToEnd.test_tool_loop_variants) ... ok
test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason) ... ok
test_bare_hunk_headers_without_numbers (tests.test_eval.Patch.test_bare_hunk_headers_without_numbers) ... ok
test_diff_with_wrong_line_number_and_counts (tests.test_eval.Patch.test_diff_with_wrong_line_number_and_counts) ... ok
test_malformed_and_context_mismatch (tests.test_eval.Patch.test_malformed_and_context_mismatch) ... ok
test_new_file_delete_and_file_block (tests.test_eval.Patch.test_new_file_delete_and_file_block) ... ok
test_paths_cannot_escape_or_touch_protected_files (tests.test_eval.Patch.test_paths_cannot_escape_or_touch_protected_files) ... ok
test_each_network_layer_blocks_alone (tests.test_eval.Sandbox.test_each_network_layer_blocks_alone) ... ok
test_missing_isolation_is_a_harness_error_not_a_fail (tests.test_eval.Sandbox.test_missing_isolation_is_a_harness_error_not_a_fail) ... ok
test_no_network_and_no_file_escape (tests.test_eval.Sandbox.test_no_network_and_no_file_escape) ... ok
test_reference_patch_passes_wrong_patch_fails (tests.test_eval.Sandbox.test_reference_patch_passes_wrong_patch_fails) ... ok
test_timeout_is_a_fail_with_reason (tests.test_eval.Sandbox.test_timeout_is_a_fail_with_reason) ... ok
test_unsupported_keywords_are_reported (tests.test_eval.Schema.test_unsupported_keywords_are_reported) ... ok
test_valid_and_invalid (tests.test_eval.Schema.test_valid_and_invalid) ... ok
test_bootstrap_is_paired (tests.test_eval.Stats.test_bootstrap_is_paired) ... ok
test_mcnemar_exact_uses_only_discordant_pairs (tests.test_eval.Stats.test_mcnemar_exact_uses_only_discordant_pairs) ... ok
test_task_outcomes_majority_and_errors (tests.test_eval.Stats.test_task_outcomes_majority_and_errors) ... ok
test_wilson_known_values (tests.test_eval.Stats.test_wilson_known_values) ... ok
test_invalid_tasks_are_rejected (tests.test_eval.TaskFiles.test_invalid_tasks_are_rejected) ... ok
test_synthetic_set_shape (tests.test_eval.TaskFiles.test_synthetic_set_shape) ... ok
test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set) ... ok
test_order_args_forbidden_and_limit (tests.test_eval.ToolScoring.test_order_args_forbidden_and_limit) ... ok
test_turn_limit_alone_fails (tests.test_eval.ToolScoring.test_turn_limit_alone_fails) ... ok
test_calculator_is_safe (tests.test_eval.ToolSim.test_calculator_is_safe) ... ok
test_canned_builtin_and_errors (tests.test_eval.ToolSim.test_canned_builtin_and_errors) ... ok

----------------------------------------------------------------------
Ran 34 tests in 6.178s

OK
exit=0
```

## `--validate-only` на синтетическом наборе (`d52d379`, без сервера)
```
$ python3 tools/eval/run_eval.py --tasks tools/eval/tasks-synthetic --validate-only; echo "exit=$?"
OK       SYNTHETIC-code_fix-001                   code_fix           unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_fix-002                   code_fix           unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_fix-003                   code_fix           unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_write-001                 code_write         unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_write-002                 code_write         unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-long_doc_qa-001                long_doc_qa        reference answer: pass number 75.0 vs 75.0 (tol 0.0)
OK       SYNTHETIC-long_doc_qa-002                long_doc_qa        reference answer: pass regex matched
OK       SYNTHETIC-long_doc_qa-003                long_doc_qa        reference answer: pass normalized
OK       SYNTHETIC-reasoning-001                  reasoning          reference answer: pass normalized
OK       SYNTHETIC-reasoning-002                  reasoning          reference answer: pass number 40.0 vs 40.0 (tol 0.0)
OK       SYNTHETIC-structured_output-001          structured_output  reference answer: pass all 7 checks passed
OK       SYNTHETIC-structured_output-002          structured_output  reference answer: pass all 6 checks passed
OK       SYNTHETIC-tool_use-001                   tool_use           reference final answer: contains_all
OK       SYNTHETIC-tool_use-002                   tool_use           reference final answer: number 429.62 vs 429.62 (tol 0.005)
OK       SYNTHETIC-tool_use-003                   tool_use           reference final answer: normalized
validated 15 tasks: all OK
exit=0
```

## Полный набор тестов репозитория на `d52d379` (последние строки)
```
$ python3 -m unittest discover -s tests > <S>/full.log 2>&1; echo "exit=$?"; tail -4 <S>/full.log
exit=0
----------------------------------------------------------------------
Ran 300 tests in 81.245s

OK (skipped=22)
```
(22 пропуска относятся к другим модулям: в выводе `-v` для `tests/test_eval.py` выше пропусков нет. Причины чужих
пропусков здесь не разбирались.)

## Мутационная проверка: метод
- Копия дерева коммита, не рабочее дерево: `git archive --format=tar -o <S>/head-<sha>.tar HEAD` и `tar -xf ... -C <S>/mutcopy`
  (для второго прогона — `<S>/mutcopy2`). Git-операций вне собственного рабочего дерева агент не делал.
- По одной мутации за раз: Python `str.replace` фрагмента, который встречается в файле ровно один раз (assert), затем
  удаление `__pycache__`, `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_eval` в корне копии, восстановление
  исходных байтов и сверка sha256 (`restored: sha256 match`).
- Печатаются строки `FAIL:`/`ERROR:`/`Ran`/`OK`/`FAILED` и код выхода. CAUGHT = код выхода ≠ 0, SURVIVED = 0.
- Прогон 1 печатал все строки, начинающиеся с `OK`; поэтому в блоках M1, M2, M13 прогона 1 есть строки вида
  `OK       SYNTHETIC-...` — это текст сообщения об ошибке упавшего теста `test_validate_only_on_synthetic_set` (вывод
  `--validate-only`, который тест прикладывает к assert), а не строки результата unittest. В прогоне 2 фильтр сужен до
  `OK` и `OK (`. Скрипт мутаций (версия прогона 2; в прогоне 1 отличалась только строка `keep = ...`, см. ниже):
```python
"""Single-point mutation check of tools/eval against tests/test_eval.py, run in a temp copy (git archive HEAD).

usage: python3 mutate_eval.py <copy_dir> [ids...]
Each mutation: assert the fragment occurs exactly once, write the mutated file, delete __pycache__ dirs,
run `python3 -m unittest tests.test_eval` with PYTHONDONTWRITEBYTECODE=1, print FAIL:/ERROR:/Ran/OK/FAILED
lines, restore the original bytes and verify the sha256. CAUGHT = exit code != 0.
"""
import difflib
import hashlib
import os
import shutil
import subprocess
import sys

E = "tools/eval/"
MUTS = [
    ("M1", "scoring always pass", E + "ev_scoring.py",
     'return {"status": "pass" if passed else "fail", "score": float(score), "reason": reason, "details": details or {}}',
     'return {"status": "pass", "score": float(score), "reason": reason, "details": details or {}}'),
    ("M2", "sandbox ignores the test exit code", E + "ev_scoring.py",
     'passed = run["rc"] == 0', 'passed = True'),
    ("M3", "tool call order not checked", E + "ev_scoring.py",
     '"ok": _in_order(calls, req)', '"ok": True'),
    ("M4", "Wilson interval without the z^2/(4n^2) term", E + "ev_stats.py",
     'half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom',
     'half = z * math.sqrt(p * (1.0 - p) / n) / denom'),
    ("M5", "McNemar on pass counts of all pairs instead of discordant pairs", E + "compare.py",
     '"mcnemar_p": ev_stats.mcnemar_exact(disc_b, disc_c)', '"mcnemar_p": ev_stats.mcnemar_exact(sum(a), sum(b))'),
    ("M6", "request error recorded as a model failure", E + "run_eval.py",
     'g = {"status": "harness_error" if resp.get("harness") else "request_error", "score": None,',
     'g = {"status": "fail", "score": 0.0,'),
    ("M7", "summary counts request errors in the pass-rate denominator", E + "run_eval.py",
     'valid = [r for r in rows if r["status"] in ("pass", "fail")]',
     'valid = [r for r in rows if r["status"] != "harness_error"]'),
    ("M8", "turn limit counted as OK", E + "ev_scoring.py",
     'subs.append({"check": "turn_limit", "ok": not turn_limit_hit,', 'subs.append({"check": "turn_limit", "ok": True,'),
    ("M9", "write through a symlinked directory not checked", E + "ev_patch.py",
     'if parent_real != root_real and not parent_real.startswith(root_real + os.sep):', 'if False:'),
    ("M10", "Landlock rule allows writes everywhere", E + "sandbox_exec.py",
     'rules = [(root, write)]', 'rules = [("/", write)]'),
    ("M11", "protected paths not enforced", E + "ev_patch.py",
     'if matches_any(norm, protected or ()):', 'if False:'),
    ("M12", "bootstrap pairing broken (B shuffled)", E + "ev_stats.py",
     'd = [float(x) - float(y) for x, y in zip(a, b)]',
     'd = [float(x) - float(y) for x, y in zip(a, random.Random(seed + 1).sample(list(b), len(b)))]'),
    ("M13", "normalized matcher compares raw strings", E + "ev_answers.py",
     'ok = any(na == normalize(a) for a in answers)', 'ok = any(answer.strip() == a for a in answers)'),
    ("M14", "compare pairs tasks whose content hash differs", E + "compare.py",
     'if a.get("task_sha256") != b.get("task_sha256") and not allow_mismatch:', 'if False:'),
    ("M15", "grader ignores the timeout flag", E + "ev_scoring.py",
     'if run["timed_out"]:', 'if False:'),
    ("M16", "JSON Schema integer accepts booleans", E + "ev_schema.py",
     '        if isinstance(x, bool):\n            return False\n        return isinstance(x, int)',
     '        if isinstance(x, bool):\n            return True\n        return isinstance(x, int)'),
]


def clean_pycache(root):
    for dp, dns, _ in os.walk(root):
        for d in list(dns):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dp, d), ignore_errors=True)
                dns.remove(d)


def run_tests(root):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_eval"], cwd=root, env=env,
                       capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    keep = [l for l in out.splitlines() if l.startswith(("FAIL:", "ERROR:", "Ran ", "OK (", "FAILED")) or l == "OK"]
    return p.returncode, keep


def main():
    root = sys.argv[1]
    only = set(sys.argv[2:])
    clean_pycache(root)
    rc, lines = run_tests(root)
    print("### baseline (no mutation)")
    print("```")
    print("\n".join(lines))
    print("exit code %d" % rc)
    print("```")
    caught = []
    for mid, desc, rel, old, new in MUTS:
        if only and mid not in only:
            continue
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8") as f:
            orig = f.read()
        n = orig.count(old)
        assert n == 1, (mid, n)
        mutated = orig.replace(old, new)
        sha = hashlib.sha256(orig.encode()).hexdigest()
        with open(path, "w", encoding="utf-8") as f:
            f.write(mutated)
        try:
            clean_pycache(root)
            rc, lines = run_tests(root)
        finally:
            with open(path, "w", encoding="utf-8") as f:
                f.write(orig)
        with open(path, encoding="utf-8") as f:
            restored = hashlib.sha256(f.read().encode()).hexdigest() == sha
        diff = [l for l in difflib.unified_diff(orig.splitlines(), mutated.splitlines(), lineterm="", n=0)
                if l.startswith(("-", "+")) and not l.startswith(("---", "+++"))]
        verdict = "CAUGHT" if rc != 0 else "SURVIVED"
        caught.append((mid, verdict))
        print("### %s %s: %s" % (mid, verdict, desc))
        print("```")
        print("# %s" % rel)
        print("\n".join(diff))
        print("$ python3 -m unittest tests.test_eval")
        print("\n".join(lines))
        print("exit code %d; restored: %s" % (rc, "sha256 match" if restored else "MISMATCH"))
        print("```")
    print("### summary")
    print("```")
    for mid, v in caught:
        print("%s %s" % (mid, v))
    print("caught %d / %d" % (sum(1 for _, v in caught if v == "CAUGHT"), len(caught)))
    print("```")


if __name__ == "__main__":
    main()
```
Строка фильтра в прогоне 1: `keep = [l for l in out.splitlines() if l.startswith(("FAIL:", "ERROR:", "Ran ", "OK", "FAILED"))]`.

## Сводка (CONFIRMED литеральными блоками ниже)
| прогон | тесты | пойманы / мутации | выжили |
|---|---|---|---|
| 1: `4bb3780` (первый коммит стенда) | 33 | 15 / 16 | M8 «лимит ходов считается OK»: в каждом тесте с лимитом ходов падала ещё и другая подпроверка (итоговый ответ, max_calls) |
| 2: `d52d379` (+ тест `test_turn_limit_alone_fails` и проверка `turn_limit` в сквозном тесте) | 34 | 16 / 16 | — |

Мутации покрывают требуемые классы: оценка всегда PASS (M1), песочница игнорирует код выхода тестов (M2) и таймаут (M15),
порядок вызовов инструментов не проверяется (M3), неверная формула интервала Уилсона (M4), Мак-Немар по всем парам вместо
несогласованных (M5), ошибка запроса записывается как провал модели (M6) или входит в знаменатель доли (M7), лимит ходов
(M8), выход за рабочую копию через симлинк (M9), Landlock разрешает запись везде (M10), защищённые тесты можно править (M11),
бутстреп без парности (M12), нормализованное сравнение (M13), сравнение задач с разным содержимым (M14), JSON Schema (M16).

## Прогон 1: `4bb3780`, 33 теста (литерально)
```
$ git rev-parse HEAD
4bb3780d32a49b4db98b76bbbe0f63df1662e7ba
$ python3 <S>/mutate_eval.py <S>/mutcopy > <S>/mut-4bb3780.log 2>&1
```
### baseline (no mutation)
```
Ran 33 tests in 6.288s
OK
exit code 0
```
### M1 CAUGHT: scoring always pass
```
# tools/eval/ev_scoring.py
-    return {"status": "pass" if passed else "fail", "score": float(score), "reason": reason, "details": details or {}}
+    return {"status": "pass", "score": float(score), "reason": reason, "details": details or {}}
$ python3 -m unittest tests.test_eval
FAIL: test_mock_detection_matches_bench (tests.test_eval.EndToEnd.test_mock_detection_matches_bench)
FAIL: test_tool_loop_variants (tests.test_eval.EndToEnd.test_tool_loop_variants)
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_reference_patch_passes_wrong_patch_fails (tests.test_eval.Sandbox.test_reference_patch_passes_wrong_patch_fails)
FAIL: test_timeout_is_a_fail_with_reason (tests.test_eval.Sandbox.test_timeout_is_a_fail_with_reason)
FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
OK       SYNTHETIC-long_doc_qa-001                long_doc_qa        reference answer: pass number 75.0 vs 75.0 (tol 0.0)
OK       SYNTHETIC-long_doc_qa-002                long_doc_qa        reference answer: pass regex matched
OK       SYNTHETIC-long_doc_qa-003                long_doc_qa        reference answer: pass normalized
OK       SYNTHETIC-reasoning-001                  reasoning          reference answer: pass normalized
OK       SYNTHETIC-reasoning-002                  reasoning          reference answer: pass number 40.0 vs 40.0 (tol 0.0)
OK       SYNTHETIC-structured_output-001          structured_output  reference answer: pass all 7 checks passed
OK       SYNTHETIC-structured_output-002          structured_output  reference answer: pass all 6 checks passed
OK       SYNTHETIC-tool_use-001                   tool_use           reference final answer: contains_all
OK       SYNTHETIC-tool_use-002                   tool_use           reference final answer: number 429.62 vs 429.62 (tol 0.005)
OK       SYNTHETIC-tool_use-003                   tool_use           reference final answer: normalized
FAIL: test_order_args_forbidden_and_limit (tests.test_eval.ToolScoring.test_order_args_forbidden_and_limit)
Ran 33 tests in 6.172s
FAILED (failures=7)
exit code 1; restored: sha256 match
```
### M2 CAUGHT: sandbox ignores the test exit code
```
# tools/eval/ev_scoring.py
-            passed = run["rc"] == 0
+            passed = True
$ python3 -m unittest tests.test_eval
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_reference_patch_passes_wrong_patch_fails (tests.test_eval.Sandbox.test_reference_patch_passes_wrong_patch_fails)
FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
OK       SYNTHETIC-long_doc_qa-001                long_doc_qa        reference answer: pass number 75.0 vs 75.0 (tol 0.0)
OK       SYNTHETIC-long_doc_qa-002                long_doc_qa        reference answer: pass regex matched
OK       SYNTHETIC-long_doc_qa-003                long_doc_qa        reference answer: pass normalized
OK       SYNTHETIC-reasoning-001                  reasoning          reference answer: pass normalized
OK       SYNTHETIC-reasoning-002                  reasoning          reference answer: pass number 40.0 vs 40.0 (tol 0.0)
OK       SYNTHETIC-structured_output-001          structured_output  reference answer: pass all 7 checks passed
OK       SYNTHETIC-structured_output-002          structured_output  reference answer: pass all 6 checks passed
OK       SYNTHETIC-tool_use-001                   tool_use           reference final answer: contains_all
OK       SYNTHETIC-tool_use-002                   tool_use           reference final answer: number 429.62 vs 429.62 (tol 0.005)
OK       SYNTHETIC-tool_use-003                   tool_use           reference final answer: normalized
Ran 33 tests in 6.508s
FAILED (failures=3)
exit code 1; restored: sha256 match
```
### M3 CAUGHT: tool call order not checked
```
# tools/eval/ev_scoring.py
-        subs.append({"check": "order", "ok": _in_order(calls, req), "detail": [c.get("name") for c in calls]})
+        subs.append({"check": "order", "ok": True, "detail": [c.get("name") for c in calls]})
$ python3 -m unittest tests.test_eval
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_order_args_forbidden_and_limit (tests.test_eval.ToolScoring.test_order_args_forbidden_and_limit)
Ran 33 tests in 6.547s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M4 CAUGHT: Wilson interval without the z^2/(4n^2) term
```
# tools/eval/ev_stats.py
-    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
+    half = z * math.sqrt(p * (1.0 - p) / n) / denom
$ python3 -m unittest tests.test_eval
FAIL: test_wilson_known_values (tests.test_eval.Stats.test_wilson_known_values)
Ran 33 tests in 6.220s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M5 CAUGHT: McNemar on pass counts of all pairs instead of discordant pairs
```
# tools/eval/compare.py
-        "b_a_only": disc_b, "c_b_only": disc_c, "mcnemar_p": ev_stats.mcnemar_exact(disc_b, disc_c),
+        "b_a_only": disc_b, "c_b_only": disc_c, "mcnemar_p": ev_stats.mcnemar_exact(sum(a), sum(b)),
$ python3 -m unittest tests.test_eval
FAIL: test_known_example (tests.test_eval.Compare.test_known_example)
Ran 33 tests in 6.219s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M6 CAUGHT: request error recorded as a model failure
```
# tools/eval/run_eval.py
-                    g = {"status": "harness_error" if resp.get("harness") else "request_error", "score": None,
+                    g = {"status": "fail", "score": 0.0,
$ python3 -m unittest tests.test_eval
FAIL: test_dead_server_gives_request_errors_only (tests.test_eval.EndToEnd.test_dead_server_gives_request_errors_only)
FAIL: test_non_streaming_mode_and_repeats (tests.test_eval.EndToEnd.test_non_streaming_mode_and_repeats)
FAIL: test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately)
Ran 33 tests in 6.350s
FAILED (failures=3)
exit code 1; restored: sha256 match
```
### M7 CAUGHT: summary counts request errors in the pass-rate denominator
```
# tools/eval/run_eval.py
-    valid = [r for r in rows if r["status"] in ("pass", "fail")]
+    valid = [r for r in rows if r["status"] != "harness_error"]
$ python3 -m unittest tests.test_eval
FAIL: test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately)
Ran 33 tests in 6.465s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M8 SURVIVED: turn limit counted as OK
```
# tools/eval/ev_scoring.py
-    subs.append({"check": "turn_limit", "ok": not turn_limit_hit,
+    subs.append({"check": "turn_limit", "ok": True,
$ python3 -m unittest tests.test_eval
Ran 33 tests in 6.548s
OK
exit code 0; restored: sha256 match
```
### M9 CAUGHT: write through a symlinked directory not checked
```
# tools/eval/ev_patch.py
-    if parent_real != root_real and not parent_real.startswith(root_real + os.sep):
+    if False:
$ python3 -m unittest tests.test_eval
FAIL: test_paths_cannot_escape_or_touch_protected_files (tests.test_eval.Patch.test_paths_cannot_escape_or_touch_protected_files)
Ran 33 tests in 6.480s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M10 CAUGHT: Landlock rule allows writes everywhere
```
# tools/eval/sandbox_exec.py
-        rules = [(root, write)]
+        rules = [("/", write)]
$ python3 -m unittest tests.test_eval
FAIL: test_no_network_and_no_file_escape (tests.test_eval.Sandbox.test_no_network_and_no_file_escape)
Ran 33 tests in 6.310s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M11 CAUGHT: protected paths not enforced
```
# tools/eval/ev_patch.py
-            if matches_any(norm, protected or ()):
+            if False:
$ python3 -m unittest tests.test_eval
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_paths_cannot_escape_or_touch_protected_files (tests.test_eval.Patch.test_paths_cannot_escape_or_touch_protected_files)
Ran 33 tests in 6.667s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M12 CAUGHT: bootstrap pairing broken (B shuffled)
```
# tools/eval/ev_stats.py
-    d = [float(x) - float(y) for x, y in zip(a, b)]
+    d = [float(x) - float(y) for x, y in zip(a, random.Random(seed + 1).sample(list(b), len(b)))]
$ python3 -m unittest tests.test_eval
FAIL: test_identical_runs (tests.test_eval.Compare.test_identical_runs)
FAIL: test_bootstrap_is_paired (tests.test_eval.Stats.test_bootstrap_is_paired)
Ran 33 tests in 6.641s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M13 CAUGHT: normalized matcher compares raw strings
```
# tools/eval/ev_answers.py
-        ok = any(na == normalize(a) for a in answers)
+        ok = any(answer.strip() == a for a in answers)
$ python3 -m unittest tests.test_eval
FAIL: test_matchers (tests.test_eval.Answers.test_matchers)
FAIL: test_all_correct_answers_pass_every_category (tests.test_eval.EndToEnd.test_all_correct_answers_pass_every_category)
FAIL: test_non_streaming_mode_and_repeats (tests.test_eval.EndToEnd.test_non_streaming_mode_and_repeats)
FAIL: test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately)
FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
OK       SYNTHETIC-code_fix-002                   code_fix           unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_fix-003                   code_fix           unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_write-001                 code_write         unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-code_write-002                 code_write         unchanged fixture: tests failed (exit code 1); reference answer: pass tests passed
OK       SYNTHETIC-long_doc_qa-001                long_doc_qa        reference answer: pass number 75.0 vs 75.0 (tol 0.0)
OK       SYNTHETIC-long_doc_qa-002                long_doc_qa        reference answer: pass regex matched
OK       SYNTHETIC-reasoning-001                  reasoning          reference answer: pass normalized
OK       SYNTHETIC-reasoning-002                  reasoning          reference answer: pass number 40.0 vs 40.0 (tol 0.0)
OK       SYNTHETIC-structured_output-001          structured_output  reference answer: pass all 7 checks passed
OK       SYNTHETIC-structured_output-002          structured_output  reference answer: pass all 6 checks passed
OK       SYNTHETIC-tool_use-001                   tool_use           reference final answer: contains_all
OK       SYNTHETIC-tool_use-002                   tool_use           reference final answer: number 429.62 vs 429.62 (tol 0.005)
OK       SYNTHETIC-tool_use-003                   tool_use           reference final answer: normalized
Ran 33 tests in 6.316s
FAILED (failures=5)
exit code 1; restored: sha256 match
```
### M14 CAUGHT: compare pairs tasks whose content hash differs
```
# tools/eval/compare.py
-        if a.get("task_sha256") != b.get("task_sha256") and not allow_mismatch:
+        if False:
$ python3 -m unittest tests.test_eval
FAIL: test_known_example (tests.test_eval.Compare.test_known_example)
Ran 33 tests in 6.519s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M15 CAUGHT: grader ignores the timeout flag
```
# tools/eval/ev_scoring.py
-            if run["timed_out"]:
+            if False:
$ python3 -m unittest tests.test_eval
FAIL: test_timeout_is_a_fail_with_reason (tests.test_eval.Sandbox.test_timeout_is_a_fail_with_reason)
Ran 33 tests in 6.425s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M16 CAUGHT: JSON Schema integer accepts booleans
```
# tools/eval/ev_schema.py
-            return False
+            return True
$ python3 -m unittest tests.test_eval
FAIL: test_valid_and_invalid (tests.test_eval.Schema.test_valid_and_invalid)
Ran 33 tests in 6.492s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### summary
```
M1 CAUGHT
M2 CAUGHT
M3 CAUGHT
M4 CAUGHT
M5 CAUGHT
M6 CAUGHT
M7 CAUGHT
M8 SURVIVED
M9 CAUGHT
M10 CAUGHT
M11 CAUGHT
M12 CAUGHT
M13 CAUGHT
M14 CAUGHT
M15 CAUGHT
M16 CAUGHT
caught 15 / 16
```

## Прогон 2: `d52d379`, 34 теста (литерально)
```
$ git rev-parse HEAD
d52d379bf84302f3a5d80955f518e93fb4133a6f
$ python3 <S>/mutate_eval.py <S>/mutcopy2 > <S>/mut-d52d379.log 2>&1
```
### baseline (no mutation)
```
Ran 34 tests in 6.182s
OK
exit code 0
```
### M1 CAUGHT: scoring always pass
```
# tools/eval/ev_scoring.py
-    return {"status": "pass" if passed else "fail", "score": float(score), "reason": reason, "details": details or {}}
+    return {"status": "pass", "score": float(score), "reason": reason, "details": details or {}}
$ python3 -m unittest tests.test_eval
FAIL: test_mock_detection_matches_bench (tests.test_eval.EndToEnd.test_mock_detection_matches_bench)
FAIL: test_tool_loop_variants (tests.test_eval.EndToEnd.test_tool_loop_variants)
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_reference_patch_passes_wrong_patch_fails (tests.test_eval.Sandbox.test_reference_patch_passes_wrong_patch_fails)
FAIL: test_timeout_is_a_fail_with_reason (tests.test_eval.Sandbox.test_timeout_is_a_fail_with_reason)
FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
FAIL: test_order_args_forbidden_and_limit (tests.test_eval.ToolScoring.test_order_args_forbidden_and_limit)
FAIL: test_turn_limit_alone_fails (tests.test_eval.ToolScoring.test_turn_limit_alone_fails)
Ran 34 tests in 6.236s
FAILED (failures=8)
exit code 1; restored: sha256 match
```
### M2 CAUGHT: sandbox ignores the test exit code
```
# tools/eval/ev_scoring.py
-            passed = run["rc"] == 0
+            passed = True
$ python3 -m unittest tests.test_eval
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_reference_patch_passes_wrong_patch_fails (tests.test_eval.Sandbox.test_reference_patch_passes_wrong_patch_fails)
FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
Ran 34 tests in 6.169s
FAILED (failures=3)
exit code 1; restored: sha256 match
```
### M3 CAUGHT: tool call order not checked
```
# tools/eval/ev_scoring.py
-        subs.append({"check": "order", "ok": _in_order(calls, req), "detail": [c.get("name") for c in calls]})
+        subs.append({"check": "order", "ok": True, "detail": [c.get("name") for c in calls]})
$ python3 -m unittest tests.test_eval
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_order_args_forbidden_and_limit (tests.test_eval.ToolScoring.test_order_args_forbidden_and_limit)
Ran 34 tests in 6.407s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M4 CAUGHT: Wilson interval without the z^2/(4n^2) term
```
# tools/eval/ev_stats.py
-    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
+    half = z * math.sqrt(p * (1.0 - p) / n) / denom
$ python3 -m unittest tests.test_eval
FAIL: test_wilson_known_values (tests.test_eval.Stats.test_wilson_known_values)
Ran 34 tests in 6.602s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M5 CAUGHT: McNemar on pass counts of all pairs instead of discordant pairs
```
# tools/eval/compare.py
-        "b_a_only": disc_b, "c_b_only": disc_c, "mcnemar_p": ev_stats.mcnemar_exact(disc_b, disc_c),
+        "b_a_only": disc_b, "c_b_only": disc_c, "mcnemar_p": ev_stats.mcnemar_exact(sum(a), sum(b)),
$ python3 -m unittest tests.test_eval
FAIL: test_known_example (tests.test_eval.Compare.test_known_example)
Ran 34 tests in 6.230s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M6 CAUGHT: request error recorded as a model failure
```
# tools/eval/run_eval.py
-                    g = {"status": "harness_error" if resp.get("harness") else "request_error", "score": None,
+                    g = {"status": "fail", "score": 0.0,
$ python3 -m unittest tests.test_eval
FAIL: test_dead_server_gives_request_errors_only (tests.test_eval.EndToEnd.test_dead_server_gives_request_errors_only)
FAIL: test_non_streaming_mode_and_repeats (tests.test_eval.EndToEnd.test_non_streaming_mode_and_repeats)
FAIL: test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately)
Ran 34 tests in 6.528s
FAILED (failures=3)
exit code 1; restored: sha256 match
```
### M7 CAUGHT: summary counts request errors in the pass-rate denominator
```
# tools/eval/run_eval.py
-    valid = [r for r in rows if r["status"] in ("pass", "fail")]
+    valid = [r for r in rows if r["status"] != "harness_error"]
$ python3 -m unittest tests.test_eval
FAIL: test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately)
Ran 34 tests in 6.515s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M8 CAUGHT: turn limit counted as OK
```
# tools/eval/ev_scoring.py
-    subs.append({"check": "turn_limit", "ok": not turn_limit_hit,
+    subs.append({"check": "turn_limit", "ok": True,
$ python3 -m unittest tests.test_eval
FAIL: test_tool_loop_variants (tests.test_eval.EndToEnd.test_tool_loop_variants)
FAIL: test_turn_limit_alone_fails (tests.test_eval.ToolScoring.test_turn_limit_alone_fails)
Ran 34 tests in 6.476s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M9 CAUGHT: write through a symlinked directory not checked
```
# tools/eval/ev_patch.py
-    if parent_real != root_real and not parent_real.startswith(root_real + os.sep):
+    if False:
$ python3 -m unittest tests.test_eval
FAIL: test_paths_cannot_escape_or_touch_protected_files (tests.test_eval.Patch.test_paths_cannot_escape_or_touch_protected_files)
Ran 34 tests in 6.751s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M10 CAUGHT: Landlock rule allows writes everywhere
```
# tools/eval/sandbox_exec.py
-        rules = [(root, write)]
+        rules = [("/", write)]
$ python3 -m unittest tests.test_eval
FAIL: test_no_network_and_no_file_escape (tests.test_eval.Sandbox.test_no_network_and_no_file_escape)
Ran 34 tests in 6.762s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M11 CAUGHT: protected paths not enforced
```
# tools/eval/ev_patch.py
-            if matches_any(norm, protected or ()):
+            if False:
$ python3 -m unittest tests.test_eval
FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
FAIL: test_paths_cannot_escape_or_touch_protected_files (tests.test_eval.Patch.test_paths_cannot_escape_or_touch_protected_files)
Ran 34 tests in 6.604s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M12 CAUGHT: bootstrap pairing broken (B shuffled)
```
# tools/eval/ev_stats.py
-    d = [float(x) - float(y) for x, y in zip(a, b)]
+    d = [float(x) - float(y) for x, y in zip(a, random.Random(seed + 1).sample(list(b), len(b)))]
$ python3 -m unittest tests.test_eval
FAIL: test_identical_runs (tests.test_eval.Compare.test_identical_runs)
FAIL: test_bootstrap_is_paired (tests.test_eval.Stats.test_bootstrap_is_paired)
Ran 34 tests in 6.250s
FAILED (failures=2)
exit code 1; restored: sha256 match
```
### M13 CAUGHT: normalized matcher compares raw strings
```
# tools/eval/ev_answers.py
-        ok = any(na == normalize(a) for a in answers)
+        ok = any(answer.strip() == a for a in answers)
$ python3 -m unittest tests.test_eval
FAIL: test_matchers (tests.test_eval.Answers.test_matchers)
FAIL: test_all_correct_answers_pass_every_category (tests.test_eval.EndToEnd.test_all_correct_answers_pass_every_category)
FAIL: test_non_streaming_mode_and_repeats (tests.test_eval.EndToEnd.test_non_streaming_mode_and_repeats)
FAIL: test_request_errors_are_recorded_separately (tests.test_eval.EndToEnd.test_request_errors_are_recorded_separately)
FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
Ran 34 tests in 6.552s
FAILED (failures=5)
exit code 1; restored: sha256 match
```
### M14 CAUGHT: compare pairs tasks whose content hash differs
```
# tools/eval/compare.py
-        if a.get("task_sha256") != b.get("task_sha256") and not allow_mismatch:
+        if False:
$ python3 -m unittest tests.test_eval
FAIL: test_known_example (tests.test_eval.Compare.test_known_example)
Ran 34 tests in 6.303s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M15 CAUGHT: grader ignores the timeout flag
```
# tools/eval/ev_scoring.py
-            if run["timed_out"]:
+            if False:
$ python3 -m unittest tests.test_eval
FAIL: test_timeout_is_a_fail_with_reason (tests.test_eval.Sandbox.test_timeout_is_a_fail_with_reason)
Ran 34 tests in 6.813s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### M16 CAUGHT: JSON Schema integer accepts booleans
```
# tools/eval/ev_schema.py
-            return False
+            return True
$ python3 -m unittest tests.test_eval
FAIL: test_valid_and_invalid (tests.test_eval.Schema.test_valid_and_invalid)
Ran 34 tests in 6.452s
FAILED (failures=1)
exit code 1; restored: sha256 match
```
### summary
```
M1 CAUGHT
M2 CAUGHT
M3 CAUGHT
M4 CAUGHT
M5 CAUGHT
M6 CAUGHT
M7 CAUGHT
M8 CAUGHT
M9 CAUGHT
M10 CAUGHT
M11 CAUGHT
M12 CAUGHT
M13 CAUGHT
M14 CAUGHT
M15 CAUGHT
M16 CAUGHT
caught 16 / 16
```

## Перепроверка ведущим (2026-10-10, после слияния в `claude/relaxed-dijkstra-1d7i5o`)
```
$ python3 -m unittest tests.test_eval
Ran 34 tests in 7.202s
OK
$ python3 tools/eval/run_eval.py --tasks tools/eval/tasks-synthetic --validate-only   (хвост)
validated 15 tasks: all OK
```
Выборочная мутационная проверка ведущим своим скриптом (`<S>/scripts/ev_mut.py`, вне репозитория) на копии ветки
(`git archive claude/n3-eval-stand`), по одной правке за прогон:
```
baseline: Ran 34 tests in 6.282s / OK
E-a Wilson half-width halved -> CAUGHT
    FAIL: test_wilson_known_values (tests.test_eval.Stats.test_wilson_known_values)
    Ran 34 tests in 6.309s
    FAILED (failures=1)
E-b McNemar k = max(b, c) -> CAUGHT
    FAIL: test_known_example (tests.test_eval.Compare.test_known_example)
    FAIL: test_mcnemar_exact_uses_only_discordant_pairs (tests.test_eval.Stats.test_mcnemar_exact_uses_only_discordant_pairs)
    Ran 34 tests in 6.628s
    FAILED (failures=2)
E-c code task passes regardless of the test exit code -> CAUGHT
    FAIL: test_wrong_answers_fail_every_task_with_a_reason (tests.test_eval.EndToEnd.test_wrong_answers_fail_every_task_with_a_reason)
    FAIL: test_reference_patch_passes_wrong_patch_fails (tests.test_eval.Sandbox.test_reference_patch_passes_wrong_patch_fails)
    FAIL: test_validate_only_on_synthetic_set (tests.test_eval.TaskFiles.test_validate_only_on_synthetic_set)
    Ran 34 tests in 6.408s
    FAILED (failures=3)
```

## N6: стенд через шлюз `tools/gateway` (mock, 2026-10-10)
`tests/test_eval_via_gateway.py` гоняет те же SYNTHETIC-задачи N3 через шлюз к mock-серверу (`tools/eval/mock_agent_server.py`)
и сравнивает с прямым прогоном: статусы задач совпадают и все `pass`; тела запросов, дошедшие до бэкенда, совпадают как
канонический JSON (идентификаторы tool calls mock-сервера содержат его глобальный счётчик запросов, поэтому они
переименовываются по порядку появления внутри каждого запроса; связь `tool_call_id` с вызовом проверяется отдельно);
TTFT записан для каждого запроса; признак mock-сервера виден через шлюз; каждый запрос есть в JSONL-журнале шлюза со
статусом 200. Неверные ответы через шлюз остаются `fail`; запрос без ключа к шлюзу с `--api-key` не засчитывается ни как
`pass`, ни как `fail` (ошибка запроса, а не ошибка модели). Это CONFIRMED только для mock; о реальном движке ничего не
говорит (NOT RUN).
```
$ python3 -m unittest tests.test_eval_via_gateway
Ran 3 tests in 2.200s

OK
```
Мутационная проверка ведущим (скрипт вне репозитория, копия `tools/` и `tests/` во временном каталоге, одна правка
`tools/gateway/gateway.py` за прогон, правка вставляется сразу после `apply_defaults` и переписывает пересылаемое тело):
```
baseline                 rc=0 ok       OK
G1 drop tool_call_id     rc=1 caught   FAILED (failures=1)
G2 drop tools            rc=1 caught   FAILED (failures=1)
G3 force non-stream      rc=1 caught   FAILED (failures=3)
G4 drop tool messages    rc=1 caught   FAILED (failures=1)
G5 inject temperature    rc=1 caught   FAILED (failures=1)
```
Полный набор после добавления (два лишних пропуска против прошлого прогона — `test_shellcheck` и `test_compile_sm89`, в этом
прогоне не заданы `SHELLCHECK`/`NVCC`; с ними оба зелёные, вывод ниже):
```
$ python3 -m unittest discover -s tests
Ran 361 tests in 92.825s

OK (skipped=40)
$ SHELLCHECK=<S>/sc09/bin/shellcheck NVCC=<S>/venv/.../nvidia/cu13/bin/nvcc python3 -m unittest tests.test_diag_kit -v   (выборка)
test_compile_sm89 (tests.test_diag_kit.CudaCompile.test_compile_sm89) ... ok
test_shellcheck (tests.test_diag_kit.StaticSafety.test_shellcheck) ... ok
Ran 33 tests in 43.835s
OK
```
