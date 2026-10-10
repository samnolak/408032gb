# ROADMAP переоценки (гейты R0–R6), 2026-10-09, редакция 2 после ревью

Основание — `ADR-012-reassessment.md`. Старые гейты G0–G8 (`docs/PLAN.md`) — история; их классификация в `LEGACY_AUDIT.md`.
Правила гейтов из AGENTS.md сохраняются: гейт N+1 не принимается раньше N; принимает lead-architect со ссылкой на evidence;
«не запущено» = NOT RUN. Эксперименты — `EXPERIMENTS.md` (предрегистрация). Роли — `.claude/agents/`.

Сквозные критерии продукта (ADR-012 D1): веса ≥ 4-бит класса; ≥ 262 144 реально используемых токенов (пробы на полной длине);
single-stream декод при заполненном контексте 50 tok/s минимум, 80–100+ цель; точная маршрутизация; API + кэш префиксов.

---

## R0 — факты о железе (блокирует всё, что про скорость)
| id | задача | владелец | файлы | тест/приёмка | статус |
|---|---|---|---|---|---|
| R0.1 | безопасный диагностический комплект | infra-ops | `scripts/diag/*`, `tests/test_diag_kit.py`, `HW_DIAG_RUNBOOK.md` | тесты в песочнице зелёные; статическая проверка запрещённых команд | СДЕЛАНО в песочнице (33 теста); на сервере NOT RUN |
| R0.2 | заказчик запускает `collect.sh --sudo` + `bench_all.sh`, присылает tar.gz | заказчик + infra-ops | `diag-out/…` → `docs/evidence/hw/` | литеральные выводы в репозитории | NOT RUN |
| R0.3 | `HARDWARE_FACTS.md` F1–F20 заполнен из выводов; `hardware/target-…json` обновлён измеренными значениями (bw_eff, каналы, PCIe, P2P) | infra-ops, lead | `HARDWARE_FACTS.md`, hardware JSON | каждая строка — CONFIRMED с файлом-источником или UNKNOWN | NOT RUN |
| R0.4 | BIOS шаг 0 (фото «до»), затем шаги 1…по `BIOS_PLAN.md`, журнал | заказчик + infra-ops | `BIOS_PLAN.md`, `docs/evidence/bios/` | после каждого шага повтор F4–F6, F12–F14 | NOT RUN |
**Выход R0:** F1–F20 без UNKNOWN (или с причиной), карта PCIe/NUMA, эталон «до». Откат: не требуется (только чтение), BIOS — по журналу.

## R1 — бейзлайны готовых движков на шорт-листе (направление A)
| id | задача | владелец | файлы | тест/приёмка | статус |
|---|---|---|---|---|---|
| R1.1 | стенд бенчмарков OpenAI-API + пробы длинного контекста | perf-bench | `tools/bench/*`, `tests/test_bench.py`, `BENCH_PROTOCOL.md` | тесты на mock-сервере зелёные; мутация «неверные ответы» → needle = 0 | СДЕЛАНО на mock (38 тестов; мутации — `docs/evidence/2026-10-09-tool-mutations.md`); на движках NOT RUN |
| R1.2 | аудит битности каждого скачанного GGUF: маршрутизируемые эксперты и не-экспертные матрицы ≥ 4 bpw; GLM/Qwen `UD-IQ4_XS` исключены заранее (E18b, E18c); не-GGUF форматы (EXL3, AWQ, safetensors MXFP4) — отдельный инструмент, пока UNKNOWN | model-analyst | `tools/gguf_quant_audit.py` (готов, тесты) → `docs/evidence/weights-<date>.md` | файл с тензорами < 4 bpw — FAIL и не участвует в R1.5 | инструмент СДЕЛАН; на файлах NOT RUN |
| R1.3 | загрузка весов на NVMe сервера (через ПК заказчика; интернета на VM может не быть) по порядку D2: Qwen3.8-Flash-Next UD-Q4_K_XL; запасной лицензионно чистый кандидат Qwen3.5-122B-A10B (Apache-2.0, ≥ 4 бит); GLM-5.3-Flash UD-Q4_K_XL; DeepSeek-V4-Flash-0731 UD-Q4_K_XL; MiMo-V2.6-Flash-MOPD — GGUF MXFP4 известен только для `-RL`, для MOPD нужна конвертация из ~173 GB safetensors (поддержка конвертером — UNKNOWN) с последующим аудитом R1.2; sha256 всех файлов | заказчик + infra-ops | `scripts/offline/weights.py`, `models/weights/*.manifest.json` → `docs/evidence/weights-<date>.md` | sha256 совпадают с манифестом ревизии HF | NOT RUN |
| R1.4 | сборка движков на сервере по пинам: llama.cpp (не старше `c811cb8f0` — мульти-GPU `--moe-cache-mib`; код читали на `10a60cf30` и `79e2e74`, сборочный SHA записывается в evidence), ik_llama.cpp, Strata upstream (`fb58e0d` или новее), ExLlamaV3, база A (strata-glm @ed37419 + `patches/strata-glm`); vLLM/SGLang — если модель целиком в VRAM | infra-ops | `docs/evidence/engines-<date>.md` | `--version`/сборочные логи | NOT RUN |
| R1.5 | матрица B0–B6 (`EXPERIMENTS.md`): модель × движок × режим (`-sm layer/row/tensor/graph`, `--n-cpu-moe`, `--moe-cache-mib`) × контекст 2K/32K/128K/262K × декод 128/512/2048, cold/warm, 3 повтора; **для GLM — отдельная строка «база A» (B1a, `--cpu-share`)**, проверяющая условие разморозки D4 | perf-bench | `docs/evidence/R1-matrix/` | `raw.jsonl`+`meta.json` на каждую ячейку; предсказание roofline (с теми же `--spec-k`/`--overlap`) записано заранее; PASS/FAIL по критерию заказчика при 262K | NOT RUN |
| R1.6 | пробы K1 (needle/var-tracking/code-QA) на 262K для каждой пары модель×движок: промпт ≈ 262 144 − N + вывод N ≥ 512 поверх; записать, нужен ли YaRN | parity-qa | там же | доля верных по глубине; вывод после заполнения окна | NOT RUN |
**Выход R1:** таблица «лучший движок для каждой модели» с TTFT, декодом при 262K, VRAM/RAM, пробами. Откат: нечего откатывать (без изменений систем).

## R2 — выбор модели и кванта (качество)
| id | задача | владелец | приёмка | статус |
|---|---|---|---|---|
| R2.1 | Q1: KLD/top-1 к эталону Q8_0 (mmap с NVMe) для 4-бит вариантов шорт-листа на фиксированном корпусе | parity-qa | таблица + команды | NOT RUN |
| R2.2 | агентный набор заказчика (реальные сессии code-агента, tool calls, длинные документы) — собрать с заказчиком 30–100 задач; прогнать 4-бит кандидатов | parity-qa + заказчик | доля решённых, тот же харнесс | NOT RUN |
| R2.3 | K2: FP8/Q8 KV против f16 на 128K/262K | parity-qa | Δ needle ≤ 2 п.п., ΔPPL ≤ 1% | NOT RUN |
| R2.4 | решение заказчика по лицензии Qwen Community License (если Qwen3.8 в лидерах) | заказчик | письменный ответ | ОЖИДАЕТ |
| R2.5 | ADR-013: модель + квант + движок по Pareto | lead | ссылки на R1/R2 evidence | NOT RUN |

## R3 — маршрутизация и ярусы (если выбранная модель не помещается целиком в VRAM)
| id | задача | владелец | файлы | приёмка | статус |
|---|---|---|---|---|---|
| R3.1 | трассы маршрутизации на реальной модели: prompt + генерация, корпус заказчика (M1) | perf-bench | `tools/g0/expert_trace.cpp` (готов, CPU-смоук CONFIRMED) | `.trace` + `-b/-ub` в meta | NOT RUN |
| R3.2 | политики кэша на трассах (M2): static/LRU/LFU/TinyLFU/hybrid/OPT при ёмкостях из roofline | perf-bench | `tools/g0/cachesim.py` (готов, тесты) | кривая попаданий held-out | NOT RUN |
| R3.3 | если лучшая политика > LRU на ≥ 5 п.п.: патч `patches/llama.cpp/` для `--moe-cache-mib` (допуск по профилю/TinyLFU, счёт промахов на CPU вместо копии при batch=1) | engine-integrator | патч + тест + бенчмарк до/после | выигрыш декода ≥ 10%; качество: для политик, меняющих только место чтения эксперта (H1/H2/H4), — бит-экзактность логитов; для счёта промахов на CPU (H3) бит-экзактность недостижима (CPU и GPU округляют по-разному) — допуск `tools/quality/logprob_compare.py`: top-1 ≥ 0.99, KL top-K не хуже межпрогонного шума (`D1-moe-cache-policy.md` п. 3) | NOT RUN |
| R3.4 | M4: CPU-вычисление промахов vs копия по PCIe на сервере | perf-bench | — | мс на промах | NOT RUN |

## R4 — многокарточность и спекуляция
| id | задача | условие | приёмка | статус |
|---|---|---|---|---|
| R4.1 | P2: all-reduce 8–64 КБ на 4 картах (NCCL, с/без P2P) | R0 F14 | латентность p50/p99 | NOT RUN |
| R4.2 | TP/EP в готовых движках (`-sm tensor`, ik `-sm graph`, ExLlamaV3 TP без P2P) на шорт-листе | R4.1 | выигрыш к PP при batch=1 на 262K | NOT RUN |
| R4.3 | (опционально, с согласия заказчика) P2P-патч драйвера (tinygrad/aikitoria) | BAR1 ≥ VRAM (F9), ReBAR (BIOS шаг 2), не VM или клика в VM | матрица P2P + R4.1 | ОЖИДАЕТ СОГЛАСИЯ |
| R4.4 | S1/S2: MTP и n-gram/suffix-спекуляция, принятие и чистый выигрыш на 262K; DFlash2 — только при основаниях (S3) | движок с поддержкой | ≥ 1.1× или выключено | NOT RUN |

## R5 — платформа (эксплуатация)
| id | задача | владелец | файлы | приёмка | статус |
|---|---|---|---|---|---|
| R5.1 | API-шлюз перед движком: аутентификация, очередь single-stream (429 + Retry-After), защита контекста через `/tokenize`, стриминг без буферизации, честная ошибка при обрыве (без поддельного `[DONE]`), журнал JSONL без текста промптов, метрики Prometheus, health, побайтовая пересылка тела (не ломает кэш префиксов) | engine-integrator | `tools/gateway/`, `tests/test_gateway.py` | на mock: тесты + мутации; на движке: пробы `probes.py` через шлюз = напрямую, нагрузочный прогон | на mock — см. `tools/gateway/README.md`; на движке NOT RUN |
| R5.2 | кэш префиксов и сохранение/восстановление состояния (KV + рекуррентные состояния) на NVMe с версиями и проверкой целостности (K3) | perf-bench | `tools/kv/slot_persist.py` | restore ≤ 25% префилла; побитовое совпадение следующих токенов | NOT RUN |
| R5.3 | эксплуатационные отказы (мастер-промпт разд. 9): отмена запросов, рестарт, OOM, диск полон, недоступный диск, перегрев NVMe, смена модели | infra-ops | `DEPLOY.md` §«Отказы» → `docs/evidence/R5-faults-<date>.md` | матрица pass/fail с логами | NOT RUN |
| R5.4 | runbook запуска/остановки/обновления/отката | infra-ops + заказчик | `DEPLOY.md` (черновик готов, шаги NOT RUN) | каждый шаг выполнен на сервере с литеральным выводом; проверен заказчиком | черновик; NOT RUN |

## R6 — релиз и финальный отчёт
| id | задача | владелец | файлы | приёмка | статус |
|---|---|---|---|---|---|
| R6.1 | сборщик verification pack: копии артефактов, `MANIFEST.sha256`, `env.txt`, вердикт по каждому критерию заказчика (PASS / FAIL / NOT RUN; отсутствующий артефакт и mock-прогон — NOT RUN, никогда не PASS) | parity-qa | `tools/verify/`, `scripts/verify_pack.sh`, `tests/test_verify_pack.py`, `VERIFICATION_PACK.md` | тесты и мутации на синтетике; `--check` ловит подмену файла | на синтетике — см. `tools/verify/README.md`; на сервере NOT RUN |
| R6.2 | собрать pack на сервере из артефактов R0–R5 | perf-bench + infra-ops | `docs/evidence/verification-pack-<date>/` | `scripts/verify_pack.sh --check` зелёный; вердикт по всем обязательным критериям | NOT RUN |
| R6.3 | финальный отчёт: что сохранили, что заменили, какие гипотезы опровергнуты, во сколько раз ускорили, что упирается в физику | lead | `docs/reassessment/FINAL_REPORT.md` | каждое число — ссылка на pack R6.2; evidence-auditor сверил | NOT RUN |

---

## Трек без железа (пока сервер собирается; решение заказчика 2026-10-10)
Цель — снять риски R1–R5 заранее, ничего не объявляя проверенным на железе. Всё ниже — CPU, компиляция под sm_89 без GPU,
математика и стенды; результат каждого пункта — коммит с тестами и литеральным выводом, push в `claude/*`.
| id | задача | владелец | файлы | приёмка | статус |
|---|---|---|---|---|---|
| N0 | публикация работы: политика push (ADR-012.1), хуки и тесты, push ветки, Draft PR, CI, резервный бандл | infra-ops + lead | `scripts/hooks/`, `tests/test_hooks.py`, PR #1 | удалённый SHA = локальный HEAD; CI по PR разобран | СДЕЛАНО, кроме разбора CI (см. PR #1) |
| N1 | сборка движков R1.4 под sm_89 в CI (компиляция без GPU): llama.cpp (≥ `c811cb8f0`, `--moe-cache-mib`), ik_llama.cpp, Strata upstream | infra-ops | `tools/engines/pins.json`, `scripts/engines/build_engine.py`, `.github/workflows/engines-sm89.yml`, `docs/evidence/2026-10-10-engines-build.md` | бинарники собраны, `cuobjdump` показывает sm_89; SHA записаны | СДЕЛАНО (компиляция): llama.cpp `79e2e74`, ik_llama.cpp `5194a9e`, Strata `fb58e0d` собраны под sm_89 в CI на CUDA 13.0 (run 38046973798), llama.cpp ещё и локально на 13.4; флаги `--moe-cache-mib`, `--n-cpu-moe`, `-sm`, `--cache-reuse`, `--slot-save-path` подтверждены по `--help`; запуск на GPU — NOT RUN; ExLlamaV3 — см. N6 |
| N2 | CPU-эталон кандидата №1: крошечная модель архитектуры Qwen3.8-Flash-Next (граф llama.cpp `qwen4exp`), parity-тест как у GLM/MiMo | parity-qa | `ref/qwen4exp.py`, `tools/tinygen/qwen4exp_tiny.py`, `run_parity_qwen4exp.sh`, `tests/test_qwen4exp_*.py`, шаг CI | тест на CPU зелёный; оракул и дампы как в G1 | СДЕЛАНО на CPU: 17 parity + 16 property тестов, худшая ошибка 7.4e-7 при допуске 1e-4, мутации 13/13 (перепроверено ведущим); не покрыто: MTP, декод/многобатчевое состояние, квантованные веса, GPU — `docs/evidence/2026-10-10-qwen4exp-tiny-parity.md` (MTP и декод закрыты в N6) |
| N3 | стенд агентного оценочного набора R2.2: формат задач, прогон через OpenAI API, автоматическая оценка (тесты кода, проверка tool calls), проверка на mock | parity-qa | `tools/eval/`, `tests/test_eval.py`, `R2-EVAL-SET.md`, `docs/evidence/2026-10-10-eval-stand.md` | тесты на mock; формат задач описан для заказчика | СДЕЛАНО на mock: 34 теста, мутации 16/16 (ведущий повторил 3/3), 15 SYNTHETIC-задач; на реальных движках NOT RUN; песочница без root (`unshare --user`) NOT RUN; задачи заказчика — ждут материалов |
| N4 | оркестрация R1 одной командой: старт движка по шаблону, матрица, пробы, аудит битности, verification pack, остановка | perf-bench | `scripts/r1/run_r1.py`, `scripts/r1/configs/`, `tests/test_r1_orchestration.py`, `docs/evidence/2026-10-10-r1-orchestration.md` | прогон на mock-сервере даёт полный pack с вердиктом NOT RUN (mock) | СДЕЛАНО на mock: 16 тестов, мутации 16/16 у агента + 2/2 у ведущего после добавленного теста аварийной остановки; шаблоны движков PROVISIONAL (llama.cpp, llama.cpp `--moe-cache-mib`, ik_llama.cpp, Strata); на реальном движке NOT RUN; движки привязаны к `tools/engines/pins.json` в N6 |
| N5 | документы: PROJECT_STATE/ROADMAP в каждом коммите шага; R2-EVAL-SET под стенд N3 | lead | `docs/reassessment/` | — | постоянно; R2-EVAL-SET обновлён под N3 |
| N6 | следующие шаги без железа (кандидаты): сверка имён движков N4 с `pins.json` и проверка сборки из N1 в оркестраторе; ExLlamaV3 (B5) сборка в CI; CPU-эталон для MTP-слоя Qwen3.8 и для декода/многобатчевого состояния (пробелы N2); прогон стенда N3 через шлюз `tools/gateway` на mock | lead | `tests/test_eval_via_gateway.py`; `scripts/r1/run_r1.py`, `scripts/r1/configs/`, `tests/test_r1_orchestration.py`; `docs/evidence/mutation-scripts/` | по пунктам | шлюз + стенд: СДЕЛАНО на mock (3 теста, мутации шлюза 5/5, `docs/evidence/2026-10-10-eval-stand.md`); пины в R1: СДЕЛАНО — `engine.pin`, шаг `engine_build_check` (сборка ≠ пин → движок не стартует), 7 тестов, мутации 16/16, проверено на настоящей сборке llama.cpp `79e2e74` sm_89 (`docs/evidence/2026-10-10-r1-orchestration.md`); сборка движка прямо из оркестратора — NOT RUN (сделана только проверка готовой сборки); ExLlamaV3 `151539c`: СДЕЛАНО локально — расширение собрано под sm_89 (torch 2.14.1, CUDA 13.0.3 из pip; 167 cubin sm_89; загружается в torch без GPU), 13 тестов, мутации 12/12, в CI `engines-sm89` собрано тоже (run 38068809785, `0dd3b51`), `docs/evidence/2026-10-10-engines-build.md`; декод, несколько ubatch, откат (`n_rs_seq`) и MTP для `qwen4exp` на CPU: СДЕЛАНО (parity-qa, влито `ec342f7`) — 54 теста, худшая ошибка 5.4e-6 при допуске 1e-4, мутации 17/17, перепроверено ведущим; найдено: падение llama.cpp в MTP-контексте без повторного reserve планировщика (обход в инструменте дампов; затронут ли llama-server — UNKNOWN) и дыра в повторе ничьих QSA в тестах N2 (закрыта); `docs/evidence/2026-10-10-qwen4exp-decode-parity.md` |
| N7 | llama-server пина R1 (llama.cpp `79e2e74`, сборка на CPU) на крошечных гибридных моделях qwen4exp и glm5-next, каждая и с MTP-блоком: кэш префиксов (B3), откат к чекпойнтам, `--cache-reuse`, сохранение/восстановление слота с перезапуском (K3), два слота одновременно, MTP против обычного жадного декода (S1, находка 12 N6) | parity-qa (сессия `session_01EyB9XrhA3NTRjjQx1A9qMh`, ветка `claude/n7-server-hybrid-cpu`) | `tools/srvcheck/`, `tests/test_srvcheck.py`, `tests/test_n7_report.py`, `tools/tinygen/glm5next_tiny.py --mtp`, `.github/workflows/server-cpu.yml`, `docs/evidence/2026-10-10-n7-server-hybrid-cpu.md` | тёплый ответ = холодному (те же токены, top-5 log-prob ≤ 1e-3) при f32-кэшах и FA off; стенд ловит внесённые неисправности | СДЕЛАНО на CPU: все проверки зелёные на обеих архитектурах; MTP в сервере не падает и совпадает с обычным декодом (механизм находки 12 на `79e2e74` устранён в upstream); продолжение диалога переиспользует 231 из 232 токенов, расхождение внутри закэшированного промпта → полный пересчёт (чекпойнты только в начале user-сообщений чата и за 4+n_ubatch/4 токена до конца промпта); `--cache-reuse` для этих моделей бесполезен (qwen4exp — отключается, glm5-next — сдвиг KV и всё равно полный пересчёт), а в шаблонах R1 и DEPLOY.md стоит `--cache-reuse 256`; слот после перезапуска — та же последовательность; при FA on + f16 KV у glm5-next тёплый/холодный расходятся до 0.027 log-prob (токены те же) — внутри собственной погрешности этой конфигурации (0.4–0.5 к f32). GPU, реальные веса, 262K, шаблоны чата — NOT RUN |

## Зависимости
R0 → R1 → R2 → (R3, R4) → R5 → R6. R1.1/R1.2, R3.1/R3.2, R5.1 (шлюз на mock) и R6.1 (сборщик pack) — инструменты, делаются без
сервера и сделаны в песочнице. Трек N1–N5 не зависит от железа и идёт параллельно сборке сервера; R0 стартует, как только сервер
собран.
