# VERIFICATION PACK — состав и правила приёмки (2026-10-09)

**Статус: состав и сборщик готовы, самого pack НЕТ (НЕ ВЫПОЛНЕНО).** На целевом сервере не снято ни одного артефакта. Сборщик
`tools/verify/pack.py` (обёртка `scripts/verify_pack.sh`) проверен только на синтетических артефактах (`tests/test_verify_pack.py`);
при отсутствии артефактов он выдаёт NOT RUN, а не PASS. Документ закрывает часть BLOCKER 2 ревью (`REVIEW-adversarial.md`) и задачу
ROADMAP R6.1; сборка на сервере — R6.2.

## Зачем
Один каталог, по которому заказчик (или evidence-auditor) без доступа к сессии агента проверяет каждое требование: что измерено, чем,
на каком коммите, с каким результатом. Число без артефакта в pack не считается (AGENTS.md: «a number without a command is invalid»).

## Критерии (обязательные — для общего PASS нужны все)
| # | требование заказчика | артефакт (инструмент) | правило вердикта |
|---|---|---|---|
| V1 | основные веса ≥ 4 бит качества | `tools/gguf_quant_audit.py --json` по каждому файлу модели | PASS, если все маршрутизируемые эксперты и не-экспертные матрицы ≥ 4.0 bpw; иначе FAIL. Не-GGUF формат — NOT RUN (инструмента нет) |
| V2 | целостность весов | `scripts/offline/weights.py verify` против манифеста ревизии | PASS при совпадении всех sha256 |
| V3 | ≥ 262 144 реально используемых токенов | `tools/bench/probes.py` на 262K (needle, multihop, code-QA; JSON, tools) | PASS, если все пробы на длине ≥ 262 144 прошли свои пороги и вывод после заполнения окна есть; прогон только на коротком контексте — NOT RUN |
| V4 | декод single-stream ≥ 50 tok/s при заполненном контексте | `tools/bench/run_matrix.py` (`raw.jsonl`, сводка), ячейка с промптом ≥ 262 144 − N | p50 ≥ 50 → PASS (≥ 80 — отмечается «цель достигнута»); < 50 → FAIL; нет ячейки на заполненном контексте → NOT RUN |
| V5 | кэш префиксов работает | `run_matrix.py` cold/warm, agent loop | TTFT тёплого ≤ 10% холодного и сервер сообщает cached tokens (B3) |
| V6 | точная маршрутизация | конфигурация движка (top-k = модельному, без пропуска экспертов) + `tools/quality/logprob_compare.py` против эталона | top-1 совпадение не ниже порога из `tools/verify/criteria.default.json` (PROVISIONAL) |
| V7 | сохранение/восстановление состояния | `tools/kv/slot_persist.py` (K3) | restore ≤ 25% префилла и совпадение следующих токенов; движок без этой функции — NOT RUN с причиной |
| V8 | факты о железе измерены | `scripts/diag/summarize.py` по архиву `collect.sh`/`bench_all.sh` | PASS, если сводка есть и снята на этом сервере (hostname/серийники); иначе NOT RUN |
| V9 | замер против границы | `tools/bench/vs_roofline.py` | информационный (не влияет на общий PASS), но обязателен как файл: показывает, где модель ошибается |
| V10 | эксплуатационные отказы | `docs/evidence/R5-faults-<date>.md` (`DEPLOY.md` §6) | ручной: PASS только с литеральными логами по каждому F1–F8 |
| V11 | качество на задачах заказчика | `R2-EVAL-SET.md`, отчёт R2.2 | ручной; автоматически не проверяется |

Сквозные правила вердикта (реализованы в сборщике и покрыты тестами — см. `tools/verify/README.md`):
- отсутствующий или нечитаемый артефакт → NOT RUN, никогда не PASS;
- артефакт, снятый против mock-сервера (`tools/bench/mock_server.py` помечает модель `owned_by: 408032gb-bench-mock`, сводки —
  `is_mock_server: true`) → NOT RUN с причиной «mock»;
- скорость декода берётся только из ячейки с заполненным контекстом, не из короткой;
- общий PASS — только если все обязательные критерии PASS; ручные (V10, V11) учитываются по отдельной отметке.

## Состав каталога pack
```
verification-pack-<date>/
  VERDICT.md            таблица V1–V11: PASS / FAIL / NOT RUN, значение, путь к артефакту (по-русски)
  verdict.json          то же для машин
  MANIFEST.sha256       sha256 каждого файла pack (проверка: scripts/verify_pack.sh --check <dir>)
  env.txt               git rev-parse HEAD, git status --short, uname -a, python3 --version, nvidia-smi -L
  artifacts/            копии входных артефактов: quant-audit, weights-verify, bench run dirs, probes, logprob, slot_persist,
                        vs_roofline, diag summary, faults log
```

## Как собрать (на сервере после R1–R5)
```
scripts/verify_pack.sh --runs <каталог с выходами инструментов> --out docs/evidence/verification-pack-<date>
scripts/verify_pack.sh --check docs/evidence/verification-pack-<date>
```
Точные флаги — `python3 tools/verify/pack.py --help` и `tools/verify/README.md` (README — источник истины). Pack коммитится в
репозиторий (кроме весов и больших трасс — для них в pack только sha256 и путь) и передаётся бандлом.

## Чего pack не доказывает
- Качество на задачах заказчика вне оценочного набора R2 (V11 — выборка, а не гарантия).
- Устойчивость под нагрузкой дольше прогона soak.
- Поведение на другой версии движка/драйвера: pack привязан к `env.txt`.
