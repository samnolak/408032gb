# Скрипты мутационных проверок (evidence)

Байт-в-байт те скрипты, которые дали мутационные таблицы в `docs/evidence/` (до этого они жили только в каталоге песочницы
сессии и пропали бы вместе с контейнером — замечание evidence-auditor 2026-10-10). Каждый копирует дерево репозитория во
временный каталог, вносит одну правку за прогон и запускает тесты; рабочее дерево не меняется. Якоря правок привязаны к коду
на указанном коммите: на более позднем коде скрипт падает на `assert s.count(old) == 1`, и это ожидаемо. Файлы скрипты берут
через `git ls-files -co`, поэтому `<репозиторий>` должен быть рабочим деревом git: для выгрузки `git archive <коммит>` сначала
`git init -q && git add -A` в её каталоге.

| скрипт | что ломает | коммит кода | команда | результат |
|---|---|---|---|---|
| `2026-10-10-gateway-eval.py` | `tools/gateway/gateway.py` (тело запроса к бэкенду) против `tests/test_eval_via_gateway.py` | `9d37e28` | `SP=<временный каталог> python3 docs/evidence/mutation-scripts/2026-10-10-gateway-eval.py <репозиторий>` | 5/5, `docs/evidence/2026-10-10-eval-stand.md` (раздел N6) |
| `2026-10-10-r1-build-check.py` | `engine_build_check` в `scripts/r1/run_r1.py`, `files_sha256` в `scripts/engines/build_engine.py` | `c1a0336` | `python3 docs/evidence/mutation-scripts/2026-10-10-r1-build-check.py <репозиторий> <временный каталог>` | 16/16, `docs/evidence/2026-10-10-r1-orchestration.md` (раздел N6) |
| `2026-10-10-torch-extension-build.py` | путь `"kind": "torch-extension"` в `scripts/engines/build_engine.py` | `62afac2` | `python3 docs/evidence/mutation-scripts/2026-10-10-torch-extension-build.py <репозиторий> <временный каталог>` | 9/9, `docs/evidence/2026-10-10-engines-build.md` (раздел ExLlamaV3) |
| `2026-10-10-qwen4exp-n2-tie-hole.py` | `Q.qsa_select` (на один пул меньше) против `tests/test_qwen4exp_tiny_parity.py` без исправления `_tie_variants` и с ним; нужны дампы N2-драйвера | `ec342f7` (до исправления) | `python3 docs/evidence/mutation-scripts/2026-10-10-qwen4exp-n2-tie-hole.py <репозиторий> <каталог артефактов run_parity_qwen4exp.sh> <llama.cpp>/gguf-py` | без исправления не ловится, с ним ловится; `docs/evidence/2026-10-10-qwen4exp-tiny-parity.md` (последний раздел) |
| `2026-10-10-offline-kit.py` | `scripts/offline/engines.py` (verify) и путь `--offline-kit` в `scripts/engines/build_engine.py` против `tests/test_offline_engines.py`, `tests/test_engines_build.py` | `bd2dcb0` (повтор на `git archive bd2dcb0`) | `python3 docs/evidence/mutation-scripts/2026-10-10-offline-kit.py <репозиторий> <временный каталог>` | 7/8, после правки теста 8/8, `docs/evidence/2026-10-10-offline-engine-kit.md` |
| `2026-10-10-torch-extension-build-shim.py` | то же + шаг `link shim for libcudart` (T1 переписан под новый порядок шагов, T10–T12 новые) | `16f39d6` | `python3 docs/evidence/mutation-scripts/2026-10-10-torch-extension-build-shim.py <репозиторий> <временный каталог>` | 12/12, там же |
| `2026-10-10-glm5next-decode.py` | инкрементальный путь и MTP-блок `ref/glm5next.py`, правила тестов, зависящие от пина, и пары MTP против `tests/test_glm5next_decode_parity.py`, `tests/test_glm5next_mtp_parity.py`, `tests/test_glm5next_chunked_ref.py`; нужны дампы `run_parity_glm5next_decode.sh` (`GLM5NEXT_DECODE_WORK`) | `7d99e0b` | `python3 docs/evidence/mutation-scripts/2026-10-10-glm5next-decode.py <репозиторий> <каталог артефактов> <llama.cpp> <временный каталог>` | `ec7630a` 12/15, `79e2e74` 14/15 (каждый пропуск — путь, которого на этом пине нет), `docs/evidence/2026-10-10-glm5next-decode-parity.md` |

Проверку воспроизводимости первых двух сделал evidence-auditor на `git archive` нужного коммита: вывод совпал построчно
(отчёт аудита 2026-10-10, строки 4–5 таблицы). Запускал проверки ведущий (эта же сессия) под ролевыми префиксами коммитов
`[parity-qa]`, `[perf-bench]` и `[infra-ops]`.
