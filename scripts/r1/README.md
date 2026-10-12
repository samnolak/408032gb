# scripts/r1 — ячейка R1 одной командой (ROADMAP, трек N4)

Автор: perf-bench, 2026-10-10. Ярлыки — по `AGENTS.md`: CONFIRMED (с источником), PROVISIONAL (как получено),
UNKNOWN, NOT RUN.

**Состояние.** Оркестратор проверен только в песочнице без GPU: «движком» был `tools/bench/mock_server.py`, «моделью» —
синтетический GGUF без данных (`tests/test_r1_orchestration.py`, литеральный вывод и мутационная проверка —
`docs/evidence/2026-10-10-r1-orchestration.md`). Ни один шаблон реального движка не запускался — **NOT RUN**. Флаги шаблонов
взяты из документации на пинах и **сверяются с `--help` на сервере** (это делает шаг `engine_version`, см. ниже).

## Зачем

Каждая ячейка R1 (модель × движок × режим, `EXPERIMENTS.md` B0–B6, K1–K4) должна измеряться одинаково и оставлять полный
проверяемый каталог: что запускали (точный argv), в каком порядке, с какими кодами выхода, что получилось и какой вердикт
выносит verification pack. Предсказание roofline записывается **до** запуска движка (правило предрегистрации
`EXPERIMENTS.md`), а сравнение с ним после замера делается с теми же флагами режима (`--overlap`, `--spec-k`, …).

## Как оператор запускает одну ячейку на сервере

```bash
cd /srv/llm/repo                                    # клон репозитория (бандл), Python 3 без pip
# 0. собрать движок на пине tools/engines/pins.json (N1) и указать на сборку ссылкой, которую ждёт шаблон;
#    на сервере без интернета исходники берутся из офлайн-набора (scripts/offline/engines.py pack на ПК, verify здесь)
python3 scripts/offline/engines.py verify /srv/llm/kits/engines-<дата>
python3 scripts/engines/build_engine.py --engine llama.cpp --arch 89 --offline-kit /srv/llm/kits/engines-<дата> \
    --src-root /srv/llm/engines-src --build-root /srv/llm/engines-build
ln -sfn /srv/llm/engines-build/llama.cpp /srv/llm/engines/current-llamacpp
# 1. выбрать шаблон, при необходимости скопировать и поправить пути/значения (или передать --set)
python3 scripts/r1/run_r1.py --config scripts/r1/configs/llamacpp-server.json --out /srv/llm/r1-runs \
    --plan                                          # 2. посмотреть ВСЕ команды; ничего не запускается и не создаётся
python3 scripts/r1/run_r1.py --config scripts/r1/configs/llamacpp-server.json --out /srv/llm/r1-runs
                                                    # 3. прогон; Ctrl-C в любой момент останавливает движок
# 4. посмотреть сводку в конце вывода, run.json и pack/VERDICT.md; каталог прогона целиком (без правок) —
#    в docs/evidence/R1-matrix/<дата>-<движок>-<модель>/ (BENCH_PROTOCOL.md, раздел 7), коммит, бандл
```

Полезные ключи:
- `--plan [--json]` — напечатать план (каждую команду с подставленными переменными). Ничего не запускает, каталогов не создаёт.
- `--only step1,step2` — только эти шаги (остальные записываются в журнал как `skipped: not selected`). `engine_start`
  всегда тянет за собой `engine_stop`. Без `engine_start` шаги замеров идут против уже запущенного движка (URL из шаблона).
- `--set KEY=VALUE` — переопределить переменную шаблона (повторяемый), например `--set WEIGHTS_DIR=/nvme1/models/...`,
  `--set HIT=0.7`, `--set EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'`.

Перед первым запуском на сервере (для шаблона llama.cpp/Strata): веса скачаны и лежат по пути `WEIGHTS_DIR`, движок собран
(ROADMAP R1.4), манифест весов перегенерирован с API HF (`models/weights/README.md`). Предрегистрацию в `EXPERIMENTS.md`
(гипотеза, порог остановки, PASS/FAIL) оркестратор не пишет — это делает человек до запуска; оркестратор записывает
предсказание границы и полный план (`artifacts/prereg/`, `artifacts/orchestrator/`).

## Шаги и что проверяется автоматически

Порядок фиксирован (`STEPS` в `run_r1.py`); любой шаг можно выключить в шаблоне (`"steps": {"<шаг>": {"enabled": false}}`)
или не выбрать через `--only`. Упавший шаг записывается, следующие шаги, которые от него не зависят, выполняются.

| # | шаг | команда / действие | что проверяется | если нечего делать |
|---|---|---|---|---|
| 1 | `weights_verify` | `scripts/offline/weights.py verify <манифест> --dest <каталог>` + копия манифеста | размер и sha256 каждого файла (V2 пакета) | нет манифеста → skipped, V2 NOT RUN |
| 2 | `quant_audit` | `tools/gguf_quant_audit.py <первый GGUF> --json` | эксперты и не-экспертные матрицы ≥ 4 bpw (V1) | нет файлов модели → skipped |
| 3 | `roofline_prediction` | `tools/roofline.py … --json` на каждую длину входа матрицы, затем `prereg/prereg.json` с sha256 каждого предсказания | предсказание записано **до** старта движка (журнал: время и порядок) | нет `roofline.spec` → skipped |
| 4 | `nvidia_smi_before` | `nvidia-smi -q` | состояние GPU до прогона | нет `nvidia-smi` → skipped |
| 4a | `engine_build_check` | `engine.build_info` (файл `build-info.json` от `scripts/engines/build_engine.py`) против `tools/engines/pins.json[engine.pin]`; копия → `artifacts/engine/build-info.json`, результат по каждой проверке → `artifacts/engine/build_check.json` | движок, репозиторий, SHA и реально выгруженный коммит = пин; зависимости (ggml у Strata) = пин; статус сборки начинается с `ok` (`ok (cubins not checked: no cuobjdump)` всё равно падает на проверке cubin); архитектура и cubin `sm_<engine.arch>` (по умолчанию 89); sha256 **каждого** собранного файла не изменился после сборки; `engine.binary` (по умолчанию `argv[0]`) — бинарник этой сборки; `git HEAD` у `engine.source_dir` = пин (Strata: Python-часть `serve/`). Любое расхождение → failed, **движок не запускается** (`engine_start` blocked) | нет `engine.pin` → skipped (mock); нет `engine.build_info` → skipped |
| 5 | `engine_version` | команда версии шаблона и `--help` | флаги argv движка, которых нет в `--help` → `artifacts/engine/flags_check.json` и WARNING в выводе (шаг не падает); код выхода `--help` сверяется с `engine.help_ok_codes` (по умолчанию `[0]`; llama-server ik_llama.cpp печатает справку и выходит с 1) | — |
| 6 | `engine_start` | argv движка в своей группе процессов; stdout+stderr → `artifacts/engine/engine.log`; argv, cwd, переменные окружения → `artifacts/engine/argv.json`; файлы конфигурации движка (`engine.config_files`) → `artifacts/engine/` | — | нет `engine.argv` → движок внешний |
| 7 | `health_wait` | GET health URL до ожидаемого кода; затем `engine.after_ready` (например `free -g`, память GPU) → `artifacts/hw/after-ready-N.txt` | таймаут → failed; движок завершился раньше → failed сразу, с кодом выхода | — |
| 8 | `matrix` | `tools/bench/run_matrix.py` по каждому прогону шаблона (smoke 2K, 2K, 32K, 128K, 262K с agent loop) | всё, что пишет run_matrix (BENCH_PROTOCOL.md) | — |
| 9 | `probes` | `tools/bench/probes.py` (32K/128K/262K) | K1 и агентные пробы | — |
| 10 | `slot_persist` | `tools/bench/prompts.py --save-text` (промпт ровно N токенов по `/tokenize` сервера), затем `tools/kv/slot_persist.py` | K3 / V7 | `engine.supports_slots: false` → skipped, V7 NOT RUN |
| 11 | `logprob_compare` | `tools/quality/logprob_compare.py --ref <эталон> --cand <движок>` | V6 | нет `reference.url` или корпуса → skipped, V6 NOT RUN |
| 12 | `vs_roofline` | `tools/bench/vs_roofline.py <прогоны матрицы> … --json` **с теми же флагами режима, что предсказание** (+ `vs_roofline.cmd.txt`) | V9 | нет прогона матрицы с `summary.json` → skipped |
| 13 | `engine_stop` | SIGTERM группе процессов движка, через `engine.stop.timeout_s` — SIGKILL; код выхода → журнал и `artifacts/engine/exit.json` | движок упал раньше времени → failed | движок не запускался → skipped |
| 14 | `nvidia_smi_after` | `nvidia-smi -q` | — | нет `nvidia-smi` → skipped |
| 15 | `pack` | `tools/verify/pack.py --runs RUN/artifacts --out RUN/pack`, затем `pack.py --check RUN/pack` | вердикт по V1–V11 (PASS / FAIL / NOT RUN) и целостность пакета | — |

Зависимости: `engine_start` зависит от `engine_build_check` (`--only engine_start` всегда добавляет его, как и
`engine_stop`); `health_wait` зависит от `engine_start`; `matrix`, `probes`, `slot_persist`, `logprob_compare` — от
`engine_start` и `health_wait`. Если зависимость **упала** (или прервана), шаг записывается как `skipped` с `blocked_by`;
если зависимость просто не выбрана или выключена, шаг выполняется (движок управляется снаружи). `after_ready` — «мягкие»
команды: их ошибка записывается и даёт код выхода 1, но не блокирует замеры.

**Движок останавливается всегда**, если его запустил этот прогон: `engine_stop` выполняется в `finally` — после ошибки шага,
после исключения в самом оркестраторе, после Ctrl-C, SIGTERM или SIGHUP (сигналы, пришедшие во время записи журнала или
остановки движка, откладываются до её конца). Инструмент, который работал в момент Ctrl-C, получает SIGINT (bench-инструменты
тогда дописывают свой каталог со статусом `interrupted`), затем SIGTERM/SIGKILL. После прерывания остальные шаги и pack не
выполняются (`skipped: interrupted`); пакет из неполного прогона при желании собирается вручную `scripts/verify_pack.sh`.

## Каталог прогона `RUNS_ROOT/<UTC>-<имя шаблона>/`

```
steps.jsonl      журнал: на каждую команду/действие запись "start" и "end" (step, sub, argv, cwd, env, utc_start/utc_end,
                 duration_s, exit_code, status ok|failed|skipped|interrupted, reason, outputs, sha256 выходов, pid, …)
run.json         итог: статус каждого шага, код выхода, вердикт пакета, git SHA репозитория, argv оркестратора
artifacts/       всё, из чего собирается pack:
  orchestrator/  config.json (байтовая копия шаблона), plan.txt и resolved.json (план с подставленными переменными),
                 steps.jsonl (журнал на момент сборки пакета)
  weights/ quant/ prereg/ hw/ engine/ matrix/<прогон>/ probes/<прогон>/ kv/ quality/ roofline/ logs/
pack/            verification pack (VERDICT.md, verdict.json, MANIFEST.sha256, artifacts/ — копия)
logs/            stdout/stderr самой сборки пакета (вне artifacts/)
slots/           SLOT_DIR по умолчанию (в шаблонах llama.cpp — /srv/llm/state/slots; состояния большие, в pack не копируются)
```

Коды выхода: 0 — все выбранные шаги ok или пропущены по шаблону; 1 — шаг упал (или пропущен из-за упавшей зависимости);
128+N — прерван сигналом N (130 = Ctrl-C, 143 = SIGTERM); 2 — ошибка шаблона или аргументов. **Вердикт пакета (PASS / FAIL /
NOT RUN) — не код выхода**: он печатается в конце и лежит в `run.json` (`pack.overall`) и `pack/VERDICT.md`.

## Шаблоны (`configs/`)

Переменные: `${NAME}` в любой строке шаблона (кроме ключей, начинающихся с `_`). Встроенные: `REPO`, `RUNS_ROOT`, `RUN_DIR`,
`ARTIFACTS`, `ENGINE_DIR`, `NAME`, `UTC`, `CONFIG_DIR`, `MODEL` (первый файл `model.files`); с умолчаниями, которые можно
переопределить: `PYTHON` (текущий интерпретатор), `HOST` (127.0.0.1), `PORT` (`auto` — свободный порт), `CTX` (262144),
`SLOT_DIR` (`${RUN_DIR}/slots`), `NVIDIA_SMI`. Неизвестная переменная, цикл, неизвестный ключ верхнего уровня, неизвестный
шаг, флаг, который ставит сам оркестратор (`--url`, `--out-dir`, …), или попытка выключить `engine_stop` — ошибка (код 2) ещё
до запуска.

| файл | ячейка | статус | что UNKNOWN |
|---|---|---|---|
| `llamacpp-server.json` | llama.cpp `llama-server`, Qwen3.8-Flash-Next UD-Q4_K_XL, всё в VRAM, `-sm layer` на 4 GPU (B1 на модели №1) | PROVISIONAL: флаги — `tools/server/README.md`@79e2e74 (строки в `_label`), NOT RUN | манифест весов PROVISIONAL (переписан из документации Strata) |
| `llamacpp-server-moecache.json` | llama.cpp с `-ncmoe` + `--moe-cache-mib` (B1, D1), GLM-5.3-Flash | PROVISIONAL, NOT RUN | **GGUF архитектуры `glm5-next`**: mainline не знает строку `glm5next` файлов Unsloth (R2-engines.md:136) — имя файла, ревизия, манифест; доля попаданий в кэш (`HIT`) |
| `ik-llamacpp-server.json` | ik_llama.cpp, GLM-5.3-Flash UD-Q4_K_XL, `-ncmoe -muge` (B1 ik) | PROVISIONAL: флаги команды, работавшей на хосте 3×3090 (`scripts/run-glm-3090-1gpu.sh:29-34`), под 4 GPU; NOT RUN | ревизия и манифест весов; `/slots` в ik_llama; `NCMOE` под 4×32 GB |
| `strata-serve-qwen38fn.json` | Strata upstream serve, Qwen3.8-Flash-Next UD-Q4_K_XL, split на 4 GPU, MTP `--spec 4` (B0) | PROVISIONAL: JSON сервера и строка запуска — Strata `docs/UNSLOTH_Q4.md:107-136`@fb58e0d с изменениями `BRINGUP-qwen38fn.md` шаг 4; NOT RUN | флаги `serve.server` (`--engine/--config/--port`), health-эндпоинт (шаблон ждёт `/v1/models`), совместимость API с `tools/bench`; RAM 128 GB против рекомендованных ~135 GB |
| `mock.json` | `tools/bench/mock_server.py` вместо движка, синтетический GGUF | MOCK: не доказательство, пакет всегда NOT RUN (mock) | — |

Значения режима и битности в шаблонах (`HIT`, `EXPERT_BPW`, `DENSE_BPW`, `SPEC_ACCEPT`, `NCMOE`, `MOE_CACHE_MIB`) — PROVISIONAL
входы предрегистрации (источник — в `_notes` каждого шаблона, в основном `ROOFLINE.md`). Отношение замер/граница > 1 в
`vs_roofline` означает, что вход неверен (так и записано в `tools/bench/vs_roofline.py`), а не что движок «быстрее физики».
Для следующего прогона битность берётся из `artifacts/quant/quant_audit.json` этого прогона.

Чтобы сделать новую ячейку: скопировать ближайший шаблон, поменять `name`, `engine`, `model`, `roofline`, `mode` и `vars`,
проверить `--plan`. `engine.name` — свободный текст для отчёта; **какой движок измеряется, задаёт `engine.pin`** — ключ
`tools/engines/pins.json` (`llama.cpp`, `ik_llama.cpp`, `strata`). Коммит тогда берётся из пина (`engine.commit` можно не
писать; если написан, он обязан быть префиксом SHA пина, иначе ошибка шаблона, код 2). Неизвестный пин, `engine.build_info`,
`binary` или `source_dir` без пина — тоже код 2. Все четыре шаблона реальных движков привязаны к пинам (это проверяет
`tests/test_r1_orchestration.py`: каждому пину — шаблон, каждому шаблону — пин); переменной `ENGINE_COMMIT` больше нет.
Сдвинуть пин — отдельный коммит в `pins.json` (N1), пересборка, и проверка шаблона пройдёт только на новой сборке.

## Сухой прогон без GPU (mock)

```bash
python3 scripts/r1/make_mock_model.py /tmp/r1/mock-model          # SYNTHETIC GGUF без данных + манифест
python3 scripts/r1/run_r1.py --config scripts/r1/configs/mock.json --out /tmp/r1   # ~3–5 с в песочнице
```

Получается полный каталог прогона и пакет с вердиктом **NOT RUN**: все критерии, опирающиеся на прогоны bench, — NOT RUN с
причиной `mock`; V1/V2 оцениваются по синтетическому GGUF (пакет помечает их плашкой SYNTHETIC: это тестовые данные).

## Что NOT RUN, пока нет сервера

**N11 (2026-10-11, CPU):** `llamacpp-server.json` прогнан как есть с путями через `--set` против CPU-сборки llama.cpp
`79e2e74` и крошечной модели: все 14 флагов движка есть в `--help`; `engine_build_check` не пускает движок без
`build-info.json` (так и задумано), V2 падает на файле не из манифеста. Выведенная из шаблона CPU-конфигурация
(`tools/srvcheck/r1_tools_smoke.py derive-config`) выполнила все шаги с настоящим сервером: старт, `/health`, матрица,
пробы, `slot_persist`, `logprob_compare`, `vs_roofline`, остановка по SIGTERM (код 0), pack (шаг probes — failed,
вердикт pack FAIL по agent_api_probes: случайные веса, не замер). Исправлено: каталог
`artifacts/quality/` (`0f23aa6`). CONFIRMED, `docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md`. Ниже — что по-прежнему
не проверено на сервере с GPU.

- Ни один реальный движок через оркестратор не запускался; флаги шаблонов не сверены с `--help` реальных сборок.
- Поведение `health_wait` на реальной загрузке весов (llama-server отвечает 503 во время загрузки, 200 после —
  `tools/server/README.md:472-484`@79e2e74, CONFIRMED чтением; Strata — UNKNOWN), длительность загрузки — UNKNOWN.
- Остановка реального движка (сколько он завершается по SIGTERM, освобождает ли VRAM) — UNKNOWN; `nvidia_smi_after`
  это покажет.
- Длительность полного прогона ячейки на 262K — UNKNOWN (BENCH_PROTOCOL.md, раздел 4: префилл 262K — минуты на запрос).
- `slot_persist` и `logprob_compare` против реального движка и эталона; подгонка промпта на 131 072 токена через `/tokenize`.
- Вердикт пакета на реальных артефактах (R6.2).

## Ограничения (инструменты не менялись)

- Проверка флагов по `--help` ищет только имена флагов из argv движка, не значения; аргументы внутри JSON Strata
  (`engine.config_files`) не проверяются.
- `logprob_compare` и `slot_persist` не пишут признак mock и описание движка (известное ограничение, `tools/verify/README.md`);
  пакет связывает их с прогоном по URL.
- Эталонный сервер для `logprob_compare` оркестратор не запускает: нужен его URL (`--set REFERENCE_URL=...`) и корпус
  (`--set CORPUS=<glob>`).
- Предсказание считается на длинах входа матрицы (`--inputs`), а `vs_roofline` — на середине вывода (промпт + вывод/2): при
  `--length-mode total` разница мала, но не нулевая.
