# Закрытие этапа «инженерия без железа» (N7–N25), 2026-10-11

Роль: lead-architect. Ветка `claude/n25-integration` (интеграция N20–N24 поверх N19, `docs/evidence/2026-10-11-n25-integration.md`),
рабочее дерево `/home/user/wt-n25`; ветка не опубликована (по заданию). Метки — по `AGENTS.md`: **CONFIRMED** (с источником или
литеральным выводом), **PROVISIONAL** (как получено), **UNKNOWN**, **NOT RUN**.

**Что этот документ не утверждает.**
- Ничего не исполнялось на GPU и на сервере 4 × RTX 4080 Super: сервера у проекта нет, к железу доступа не было, все прогоны —
  CPU песочницы и раннеров GitHub (CONFIRMED: `CORRECTNESS.md`, уровни 4–6 — NOT RUN).
- Требование заказчика «50 tok/s минимум, 80–100+ цель при заполненных 262K» (ADR-012 D1) **не измерено, не подтверждено и не
  опровергнуто**. Все tok/s в документах этапа — потолки моделей (PROVISIONAL) или чужие замеры (REPORTED).
- Сборка под sm_89 в CI (`engines-sm89`, `strata-glm-sm89`, `lt-strata-sm89`, `cuda-compile`) — это компиляция без GPU, а не
  работающий движок: ни один бинарник не запускался на sm_89.
- Ни один гейт этим документом не переводится в ACCEPTED: R0 (`ROADMAP.md`) — NOT RUN; закрываются только задачи трека без
  железа N7–N25, и только на уровне «сделано на CPU с доказательствами».

## (а) Что сделано за этап

Статус CI — последний прогон на текущей голове ветки (CONFIRMED: `gh api …/actions/runs?head_sha=…` и `?branch=…`,
приложение Б). Все PR — Draft, ничего не влито в `main`.

| N | что сделано (кратко) | PR / ветка @ голова | статус |
|---|---|---|---|
| N7 | llama-server пина R1 `79e2e74` на крошечных гибридных моделях (CPU): кэш префиксов, чекпойнты, слоты, MTP | #2 `claude/n7-server-hybrid-cpu` @`4f7be84` | CI зелёный (ci, server-cpu); здесь же исправлен докстринг пробела 11 (г) |
| N8 | повреждённые файлы слотов: сырой llama-server принимает 26/26, хранилище `tools/kv/session_store.py` отвергает 26/26 | #4 `claude/n8-session-store` @`09eb438` | CI зелёный |
| N9 | паритет glm5-next: несколько ubatch, декод, откат, MTP на двух пинах | #5 `claude/n9-glm-decode-parity` @`af4db13` | CI зелёный |
| N10 | паритет при f16 KV + flash attention и на весах Q8_0/Q4_K_M | #6 `claude/n10-quant-fa-parity` @`0d4d582` | CI зелёный |
| N11 | инструменты замеров R1 против настоящего llama-server на CPU, 4 ошибки инструментов исправлены | #7 `claude/n11-r1-tools-cpu-server` @`b74efde` | CI зелёный |
| N12 | ik_llama.cpp и диалект `glm5next` на CPU: ik считает не ту же функцию (6 отличий, в т. ч. clamp SwiGLU) | #8 `claude/n12-ik-glm5next-cpu` @`5730cc4` | CI зелёный |
| N13 | шаги паритета CI падают, если тесты не прогнались (пробел 11 (а)) | #9 `claude/n13-ci-no-silent-skips` @`6dd6b28` | CI зелёный |
| N14 | настоящие чат-шаблоны в llama-server: ошибка Jinja-движка (`set` в цикле), B3, tool calls | #10 `claude/n14-chat-templates` @`b2096c1` | CI зелёный |
| N15 | настоящие заголовки GGUF по HTTP Range: GLM на HF — `glm5-next`, у Qwen нет MTP в основном файле | #12 `claude/n15-gguf-headers` @`67bd5ba` | CI зелёный |
| N16 | паритет MiMo: декод, откат за пределы SWA-кэша, MTP, MXFP4 | #11 `claude/n16-mimo-decode-parity` @`0b0eace` | CI зелёный |
| N17 | MiMo: три MTP-головы, f16 KV + FA с синками; находка — голова 0 читает чужие K/V | #14 `claude/n17-mimo-mtp-heads-fa` @`3ce2293` | CI зелёный |
| N18 | заголовки отдельных MTP-файлов Qwen, GLM ggml-org, MiMo-V2.6 | #13 `claude/n18-gguf-headers-mtp-mimo` @`35ba5ba` | CI зелёный |
| N19 | qwen4exp: несколько последовательностей, `seq_cp`, края отката; две молчаливые ошибки llama.cpp | #15 `claude/n19-qwen4exp-multiseq` @`e71b241` | CI зелёный |
| N20 | защита рекуррентного отката: достижимость, патч llama.cpp, правило R1 `-np 1` | ветка `claude/n20-recurrent-rollback-guard` @`b7d6af1`, PR нет | CI зелёный (шаг N20 Ran 110 на обоих пинах); аудит: FAIL 1 исправлен |
| N21 | MiMo MTP-головы: тяжесть (только черновики), патч slot replay, обходы, эксперимент A/B | ветка `claude/n21-mimo-mtp-heads-fix` @`93a48d6`, PR нет | CI зелёный (в т. ч. задание `n21-patched`, Ran 57); аудит: 4 MINOR исправлены |
| N22 | ложнозелёные CI (пробел 11 (б)–(г)), таблицы ошибок по слоям (пробел 14) | ветка `claude/n22-ci-false-green` @`56202c8`, PR нет | CI зелёный (`cuda-compile` собирает `scripts/diag/*.cu` под sm_89 — только компиляция); аудит: FAIL нет |
| N23 | исправленный шаблон GLM, `tokenizer.json` по sha256, уровень токенов | ветка `claude/n23-glm-template-tokenizer` @`7541c4a`, PR нет | CI и `token-level` зелёные; аудит: исправления в `facc74d`, `7541c4a` |
| N24 | решения владельца: правило синтетики pack, V6, источники SwiGLU, проверка хоста, документы R0 | ветка `claude/n24-owner-decisions` @`76e9d56`, PR нет | CI зелёный (`ci`); аудит: FAIL нет, 3 MINOR исправлены |
| N25 | интеграция N20–N24 в одну ветку, все шаги на дампах на обоих пинах, N20 + N21 на одной копии; закрытие этапа (этот документ, статусы `CORRECTNESS.md`, план скорости Qwen) | ветка `claude/n25-integration`, локально, не опубликована | GitHub — NOT RUN (не опубликована); локальные шаги — `docs/evidence/2026-10-11-n25-integration.md` |

Проверки аудитора для N7–N19 — в их evidence (сводка: `RESEARCH-PHASE-2026-10-10.md`, раздел 2, ветка `claude/research-landscape`).
Аудит закрывающих документов этого этапа (N25) evidence-auditor ещё не проводил — NOT RUN.

## (б) Матрица рисков: найденные дефекты корректности

«Рекомендованная конфигурация R1» — шаблоны `scripts/r1/configs/*.json` на этой ветке (не менялись с `93bdfe0`; на голове PR #1
`37415fe` в `llamacpp-server*.json` те же `-np 1`, `--jinja` без `--chat-template-file`, `spec_k 0`): llama.cpp `79e2e74` для Qwen
(`llamacpp-server.json`) и GLM (`llamacpp-server-moecache.json`), ik_llama.cpp `5194a9e` для GLM (`ik-llamacpp-server.json`); все
без спекуляции, `-np 1` / `--parallel 1` (CONFIRMED: `scripts/r1/rs_rollback_rule.py`, приложение Б). Колонка «достижим» —
сейчас; в скобках — после правил раздела (в), если они меняют ответ.

Состояние: **исправлено** (патч или правка + CPU-тест), **обходится** (обход или правило конфигурации), **требует GPU**,
**открыто**.

| # | дефект | модели | достижим в рекомендованной конфигурации R1? | состояние | доказательства |
|---|---|---|---|---|---|
| 1 | Jinja-движок llama.cpp: `{% set %}` в теле `{% for %}` переживает итерацию — ход ассистента без рассуждения получает рассуждение предыдущего хода (утечка в историю, разрыв префикса) | GLM-5.3-Flash (официальный шаблон и встроенный шаблон Unsloth) | **да**, CONFIRMED: шаблон GLM R1 запускает `--jinja` без `--chat-template-file`, встроенный шаблон Unsloth несёт тот же `set` (N15); (после правила 2: нет — CONFIRMED на CPU: 279/298 строк = официальный формат, 0 строк утечки) | обходится (наш шаблон `templates/glm-5.3-flash/chat_template.jinja`; сам движок не правился) | N14, N15, N23 (`docs/evidence/2026-10-11-n23-glm-template-tokenizer.md`) |
| 2 | `"reasoning_content": null` в сообщении ассистента → HTTP 500 (`common/chat.cpp:422-423@79e2e74`) | все модели в llama-server `79e2e74`, любой шаблон | да, если клиент шлёт `null` (CONFIRMED); ошибка громкая, не молчаливая | обходится (клиент опускает поле или шлёт `""`) | N23, находка 6 |
| 3 | Qwen: пустой/отсутствующий/пробельный блок рассуждения перетокенизируется (`Ċ Ċ` → `ĊĊ`), переиспользование префикса рвётся на последнем токене прошлого промпта | Qwen3.8-Flash-Next (официальный шаблон, токенизатор Qwen) | да (CONFIRMED на крошечной модели с настоящим словарём, llama-server `79e2e74`: 41–62 токена переобработки на разрыв); для настоящей модели — PROVISIONAL. Вывод не искажается — теряется префилл | обходится (принять: ограничено последним ходом; `enable_thinking: false`, где рассуждение выключено — проверено только токенизатором) | N15, N23 п. 9–10 |
| 4 | qwen4exp, рекуррентный откат (оба пина): откат глубже последнего ubatch читает устаревший снимок (`stale`); `seq_cp` при незавершённом откате даёт состояние до отката (`cp_pending`); `last_call`, `extra_cell` (снимок **другой** последовательности), `cp_pending_order` — всё молча, логиты до 1.17 | Qwen3.8-Flash-Next; общая рекуррентная память и у других архитектур с откатом (glm5-next прогнан; остальные — PROVISIONAL) | **нет**, CONFIRMED по коду: без спекуляции `n_rs_seq` = 0, ни один частичный откат не принимается; со спекуляцией через сервер достижим только `cp_pending` при `-np` ≥ 2 (PROVISIONAL, путь decision) — правило 1 его исключает | исправлено патчем на CPU, вне R1-сборки (`scripts/engines/build_engine.py` патчи не применяет): `patches/llama.cpp/{79e2e74,ec7630a}/0001-recurrent-rollback-guard.patch`, CPU 110/110 на обоих пинах, в GitHub тоже; на GPU NOT RUN. В R1 — недостижимо при `-np 1` без спекуляции (правило 1 — предложение) | N19, N20, N25 (N20 + N21 вместе) |
| 5 | откат рекуррентной ячейки, общей для двух последовательностей, принимается и уменьшает `pos_max` другой | qwen4exp только на `ec7630a` | нет: R1 на `79e2e74`, где upstream отказывает; через сервер недостижим и на `ec7630a` (CONFIRMED по коду, N20 строка 11) | исправлено (upstream `79e2e74`; патч N20 для `ec7630a`) | N19 находка 7, N20 |
| 6 | MiMo: откат глубже, чем держит SWA-кэш, принимается и молча меняет выход | MiMo-V2.6-Flash (`mimo2`), оба пина | нет: MiMo нет в шаблонах R1; llama-server защищён проверкой `seq_pos_min` (CONFIRMED по коду, N16) | обходится (сервер; для своего движка — глубина ≤ C − n_swa + 1, PROVISIONAL) | N16 |
| 7 | MiMo MTP: после вызова, пересекающего конец SWA-кольца (или из нескольких ubatch), голова 0 читает K/V чужих позиций — **только черновые токены**, выход проверяется целью | MiMo-V2.6-Flash (3 MTP-головы), оба пина | нет, CONFIRMED: ни MiMo, ни спекуляции в шаблонах R1 | исправлено патчем на CPU, вне R1-сборки (`scripts/engines/build_engine.py` патчи не применяет): `patches/llama.cpp/0001-mtp-heads-slot-replay.patch`, CPU 57/57, в GitHub тоже; GPU NOT RUN. В R1 MTP выключен, и MiMo нет — недостижимо; без патча — MTP выключен / `--swa-full` / GGUF с одной головой | N17, N21 |
| 8 | MTP-файлы Qwen `-shared-` не загрузятся ни на одном пине (`token_embd` создаётся с флагами 0) | Qwen3.8-Flash-Next, MTP | нет (без MTP); PROVISIONAL по чтению — ничего не загружалось | обходится (правило 4: только обычные `mtp-…-{Q8_0,BF16,Q4_K_M}.gguf`) | N18 |
| 9 | у всех шести MTP-файлов Qwen `compress_ratios[48] = 0` — MTP-слой считает плотное внимание вместо QSA | Qwen3.8-Flash-Next, MTP | нет (без MTP); значение CONFIRMED по заголовкам, следствие PROVISIONAL по чтению | открыто (влияние на принятие UNKNOWN; правка метаданных — только с parity против оракула) | N18 |
| 10 | GGUF kernelpool: `mimo2.qkv_chunks = 4` — если строки `attn_qkv` не перегруппированы, llama.cpp посчитает неверные Q/K/V | MiMo-V2.6-Flash, файл kernelpool | нет (MiMo нет в R1) | открыто (UNKNOWN; проверка — сравнение байтов строк `blk.0.attn_qkv` с файлом ggml-org, чтение данных, не заголовка) | N18 |
| 11 | порядок clamp и SiLU в SwiGLU: llama.cpp — до SiLU, ik_llama.cpp и exllamav3 — после; на выходе SwiGLU до 4.54e-3 на элемент при L = 10 | GLM-5.3-Flash | **да** для `ik-llamacpp-server.json` (ik считает «после», CONFIRMED N12); какой вариант верен — UNKNOWN (кода модели у zai-org нет; transformers и vLLM — «до», PROVISIONAL как семантика по умолчанию) | требует GPU (эксперимент на настоящих весах подготовлен, NOT RUN) | N12, N24 раздел 3 |
| 12 | файл GGUF на epyc-ai: `AGENTS.md` называет архитектуру `glm5next`, пин HF (`unsloth/GLM-5.3-Flash-GGUF@a38483c8`) — `glm5-next`; llama.cpp `glm5next` не грузит | GLM-5.3-Flash на хосте epyc-ai (3×3090) | нет: R1 — сервер 4×4080, веса сверяются с манифестом sha256 (V2) | открыто (проверка хоста `tools/hostcheck/glm_gguf_check.sh` готова, NOT RUN; решение владельца — не доверять файлу до проверки) | AGENTS.md, N15, N24 раздел 4 |
| 13 | llama-server восстанавливает повреждённый файл слота без контрольной суммы (26/26 приняты, у 5 изменились токены) | все модели с `--slot-save-path` | да: шаблоны llama.cpp R1 включают `--slot-save-path` (CONFIRMED); само повреждение — внешнее событие | обходится (правило 6: восстанавливать только через `tools/kv/session_store.py`, он отвергает 26/26) | N8 |
| 14 | ложнозелёные сигналы CI: шаги паритета при пропуске тестов (а), `cuda-compile` пустой по построению (б), сборка `scripts/diag/*.cu` нигде (в), докстринг на несуществующий файл (г) | CI | не относится к движку | исправлено ((а) N13; (б)–(г) N22 и `4f7be84`; в GitHub зелёные с ожидаемыми аннотациями) | N13, N22, приложение Б |
| 15 | вердикт verification pack FAIL на синтетической модели: причина — пробы поведения модели (`tools` 0.0, `json` 0.071), а не движок | verification pack | не относится к движку | исправлено (объявление `synthetic_run` + маркер `SYNTHETIC`, исходный вердикт сохраняется, FAIL с ошибками запросов остаётся FAIL; мутации 20/20) | N11, N24 раздел 1 |
| 16 | допуск почти-ничьих V6 = 0.025 nats не измерен на настоящих весах и GPU | все модели (критерий pack) | да (используется pack R1/R6); PROVISIONAL | требует GPU (план замера N24 2.4) | N24 раздел 2 |
| 17 | `llama-quantize` квантует двумерные гаммы `ple_norm_*` у qwen4exp, после чего CPU падает на первом графе | qwen4exp, собственная квантизация | нет: R1 берёт готовый GGUF Unsloth, а не наш `llama-quantize` (PROVISIONAL); затронут ли настоящий файл — UNKNOWN | открыто | N10 (`CORRECTNESS.md`, статус пробела 2) |

## (в) Правила рекомендованной конфигурации R1 против молчаливо неверного результата

**Это предложения для сессии PR #1**, которая ведёт `scripts/r1/configs` и `DEPLOY.md`; здесь ничего из этого не применено.
Метка каждого правила — метка его основания.

1. **`-np 1` явно при любой спекуляции** с `n_rs_seq` > 0 (`--spec-type draft-mtp|draft-eagle3|draft-dflash|draft-dspark`) и при
   черновой модели без `--spec-type` (тип выводится). Умолчание llama-server — `-np` auto = 4. Проверка в оркестраторе —
   `python3 scripts/r1/rs_rollback_rule.py <config.json>` (ненулевой код — отказ запуска). PROVISIONAL (код, N20 раздел 4);
   с патчем N20 правило для (а)/(б) не нужно.
2. **GLM: наш шаблон через `--chat-template-file`**: `"--chat-template-file", "${REPO}/templates/glm-5.3-flash/chat_template.jinja"`
   после `"--jinja"` в `llamacpp-server-moecache.json:40`; sha256 шаблона (`5d47ac8e…ef5a`) записывать в артефакты прогона. Для
   ik — только если его `--help` знает флаг (UNKNOWN). CONFIRMED на CPU (N23).
3. **Клиенты и шлюз не шлют `"reasoning_content": null`** — поле опускается или `""` (шлюз `tools/gateway/` может нормализовать).
   CONFIRMED (N23, находка 6).
4. **MTP Qwen — только обычные файлы** `mtp-Qwen3.8-Flash-Next-{Q8_0,BF16,Q4_K_M}.gguf` через `-md FILE --spec-type draft-mtp`;
   никаких `-shared-`; не полагаться на `-hf … --mtp` (папку `MTP/` он не найдёт). PROVISIONAL по чтению (N18).
5. **MiMo с MTP — только сборка с патчем N21** или без MTP / с `--swa-full` / с GGUF одной головы; и до эксперимента A/B
   спекуляция включается только по замеру (ADR-011 «только по замеру, с автоотключением»; `ROADMAP.md` R4.4: ≥ 1.1× или выключено). CONFIRMED на CPU (N21).
6. **Восстановление слотов — только через `tools/kv/session_store.py`**, никогда сырым `/slots?action=restore` из
   непроверенного файла. CONFIRMED на CPU (N8).
7. **Собственный движок (не llama-server)**: откат на r — только если последний вызов подал последовательности ≥ r + 1 токенов
   (ассерт); `seq_cp` — только после декода источника; `false` от `seq_rm` — обрабатывать восстановлением чекпойнта; у MiMo
   глубина отката ≤ C − n_swa + 1; размещать вызов MTP один раз для всех голов. PROVISIONAL (из кода, N16, N19–N21).
8. **Веса**: перед первым запуском — `scripts/offline/weights.py verify` (sha256 по манифесту) и `tools/gguf_quant_audit.py`
   (≥ 4 bpw); файл с epyc-ai — только после `tools/hostcheck/glm_gguf_check.sh`. CONFIRMED как инструменты (тесты на CPU).
9. **ik_llama.cpp для GLM — не эталон паритета**: расхождения с llama.cpp ожидаемы и при верных ядрах (6 отличий N12); до
   эксперимента SwiGLU семантикой по умолчанию считается «до SiLU» (PROVISIONAL, N24).
10. **Синтетический прогон pack** объявляется только `"synthetic_run": true` в файле критериев и маркером `SYNTHETIC` в
    `model_desc`; как приёмка он не засчитывается никогда. CONFIRMED (N24, тесты).
11. **Qwen: не менять `reasoning_effort` внутри сессии** (полная переобработка, N14). `"high"` даёт HTTP 500 только с
    официальным файлом шаблона (`--chat-template-file`, N14); шаблон Qwen R1 — `--jinja` без файла, то есть встроенный шаблон
    GGUF Unsloth, который переводит `high` в `xhigh` без HTTP 500 (N15; `RESEARCH-PHASE-2026-10-10.md:98-99`, ветка
    research-landscape). Где рассуждение не нужно — `enable_thinking: false` (без разрыва `ĊĊ`, проверено токенизатором, N23).
    CONFIRMED на CPU.

## (г) План запуска на будущем сервере

Порядок — гейты `ROADMAP.md`; гейт N+1 не начинается до приёмки N (AGENTS.md). Все пороги — из источников; их метки сохранены.

| стадия | что делаем | go / no-go | источник |
|---|---|---|---|
| **R0** — факты о железе | `scripts/diag/collect.sh --sudo`, `bench_all.sh`; каналы DRAM (dmidecode/EDAC на хосте Proxmox, `bench_mem.sh`, STREAM); P2P — только штатная диагностика после прибытия железа (`nvidia-smi topo -m`/`-p2p r`, `bench_gpu.sh`, nccl-tests с `NCCL_P2P_DISABLE=1` и без), без патча драйвера; VRAM 32 GB на каждой карте; PCIe x16 Gen4; NVMe — число и модели (план 4 × 1 ТБ; поправка заказчика 2026-10-12: 3 × 1 ТБ); BIOS по журналу | F1–F20 `HARDWARE_FACTS.md` — CONFIRMED или UNKNOWN с причиной; DRAM Triad: 8 каналов ≈ 170–180 ГБ/с, 4 канала ≈ 90–100 (HYPOTHESIS, «красный флаг» < 150 / < 80); 0 ошибок шаблонов VRAM; P2P есть/нет — факт, без P2P по умолчанию конвейер (PP) | `R0-HARDWARE-CHECKS.md` (a)–(c); `CORRECTNESS.md` §2.1 (B1–B9); `HW_DIAG_RUNBOOK.md` |
| **R1** — бейзлайны готовых движков | веса на NVMe + `weights.py verify` + `gguf_quant_audit.py`; сборка движков на сервере по пинам (`build_engine.py --arch 89`), `--version` с драйвером сервера; проверки N7 на CUDA-сборке llama-server (B11); `logprob_compare.py` CPU против GPU на одном GGUF (B13, калибровка V6); `run_r1.py` по шаблонам с правилами (в) | движок стартует, `build-info.json` = пины; гейт N7 (те же токены, \|Δ log-prob\| ≤ 1e-3 при f32/FA off); декод при заполненных 262K — **p50 ≥ 50 tok/s, 0 отказов, ≥ 3 повтора** (V4: меряется здесь, не обещается); needle ≥ 0.8, tools/json = 1.0 (V3) | `ROADMAP.md` R1; `CORRECTNESS.md` B10–B11, B13, B17–B21 |
| **R2** — модель и квант | KLD/top-1 к Q8_0 (Q1); агентный набор заказчика (30–100 задач, ждём материалов); K2: FP8/Q8 KV против f16 | KLD в пределах 1.5× лучшего; Δ needle ≤ 2 п. п., ΔPPL ≤ 1 % (PROVISIONAL пороги `EXPERIMENTS.md`); ADR-013 по Pareto | `ROADMAP.md` R2; B24–B26 |
| **R3** — маршрутизация и ярусы (если модель не целиком в VRAM — GLM) | трассы реальной модели (M1), политики кэша (M2), промах на CPU против копии по PCIe (M4) | held-out hit; R9 M1: < 0.75 — стоп, ≥ 0.85 — строить (PROVISIONAL) | `ROADMAP.md` R3; B29–B31 |
| **R4** — многокарточность и спекуляция | all-reduce 8–64 КБ на 4 картах с P2P и без (R4.1); TP/EP готовых движков (R4.2); MTP / n-gram (R4.4); P2P-патч драйвера — **не делается** (ADR-012.2) | all-reduce ≤ 30 мкс или P2P PASS → TP-путь; спекуляция ≥ 1.1× или выключена | `ROADMAP.md` R4; `ARCHITECTURE-ALTERNATIVES.md` §6 |
| **R5** — эксплуатация | шлюз, сохранение/восстановление 262K-состояния (только через `session_store.py`), отказы F1–F8, runbook | restore + тёплый префилл ≤ 25 % холодного (V7); каждый шаг runbook с литеральным выводом | `ROADMAP.md` R5; B27, B34, B35 |
| **R6** — сдача | verification pack из артефактов R0–R5 | общий PASS только если все обязательные критерии PASS; синтетический прогон — никогда | `ROADMAP.md` R6; B36 |

**Подготовленные эксперименты на железе** (все NOT RUN):

| эксперимент | когда | команда / документ | что решает | метка |
|---|---|---|---|---|
| N21: A/B принятия MTP у MiMo — одна сборка `79e2e74` с патчем, A = `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` (порядок upstream), B = исправленный порядок; `-np 1`, `--spec-type draft-mtp --spec-draft-n-max 3`, промпты 100–20000 токенов | R4.4, если MiMo в шорт-листе | `docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md`, раздел 4 (точные команды) | заметна ли разница принятия и tok/s; ожидание по модели — A портит 3–16 % черновиков (PROVISIONAL) | NOT RUN |
| N24: SwiGLU на настоящих весах — две сборки llama.cpp `79e2e74` («до» и «после», патч только для эксперимента), `gate` (сколько гейтов > L = 10) и `kld` (перплексия и KLD, с опорой BF16/Q8_0, если владелец её даст) | R2 (или на epyc-ai, если там `glm5-next`) | `tools/quality/swiglu_clamp_experiment.sh`, `docs/evidence/2026-10-11-n24-owner-decisions.md` раздел 3.4 | какая семантика clamp у обученной GLM-5.3-Flash | NOT RUN |
| N24: проверка файла GGUF на epyc-ai — только чтение, размеры и sha256 шардов против пина HF, факты заголовка | до любого использования файлов хоста | `tools/hostcheck/glm_gguf_check.sh --dir /mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL --out "$HOME/glm-gguf-check-$(date -u +%Y%m%dT%H%M%SZ)" --jobs 3` | тот ли это файл, что на HF; `glm5next` или `glm5-next` | NOT RUN |
| V6: допуск почти-ничьих на GPU — тот же GGUF, CPU (f32 KV, FA off) против шаблона R1 на GPU, 3 seed × 200 префиксов; шум внутри GPU (`-ub 512`/`32`, FA on/off); допуск = 2 × p99 без дискретных событий, не ниже 0.025 | R1/R2.1 | `docs/evidence/2026-10-11-n24-owner-decisions.md` раздел 2.4 | замена PROVISIONAL 0.025 измеренным значением | NOT RUN |
| N20: патч защиты отката на GPU — копия `79e2e74` через `patches/llama.cpp/apply.sh`, сборка CUDA; Qwen с MTP (`-np 1 --spec-type draft-mtp`) длинная сессия на сборке с патчем и без: те же токены, в журнале нет `refusing a rollback`; затем `-np 2` с запросами decision с общим префиксом (путь (б)) | R4.4 для Qwen | точных команд в evidence N20 нет — это **предложение** (PROVISIONAL), подготовить до сервера | что патч не меняет вывод там, где сервер откатывает верно, и ловит путь (б) | NOT RUN |

## (д) Что требует GPU (или сервер) — явный список

1. Любое исполнение CUDA-кода: движки на sm_89, ядра, CUDA-сборка `dump-tensors` и паритет оракулов против CUDA-бэкенда (B12;
   допуски для CUDA не заданы).
2. Любые замеры скорости: декод при 262K, префилл, TTFT, все tok/s (уровни 5–6, B20–B23).
3. Калибровка V6 (шум GPU против CPU на настоящих весах) и KLD/top-1 кванта к Q8_0 (B13, B24).
4. Эксперимент SwiGLU на настоящих весах GLM (N24).
5. A/B принятия MTP у MiMo (N21) и вообще принятие любой спекуляции на настоящих весах (B28, R4.4).
6. Патч N20 и патч N21 на GPU-сборке (на CPU оба проверены).
7. P2P, all-reduce, полоса VRAM и PCIe, проверка 32 GB на каждой карте (R0 B3–B6, R4.1).
8. Сервер (не обязательно GPU): каналы DRAM, NVMe, BIOS, загрузка весов, сохранение 262K-состояния на NVMe, отказы (B2, B7, B8,
   B17, B27, B34); проверка файла на epyc-ai — доступ к хосту.
9. Поведение настоящих моделей: какие токены они выбирают вокруг пустого блока рассуждения (N23), качество, пробы 262K.

## (е) Решения владельца и открытые вопросы

Решения владельца от 2026-10-11 переданы lead-architect в задачах N24 и N25; письменного документа заказчика в репозитории
нет — UNKNOWN как первоисточник.

| решение | где записано / как выполнено |
|---|---|
| P2P — только диагностика после прибытия железа; драйвер NVIDIA не патчить | ADR-012.2; `R0-HARDWARE-CHECKS.md` (b). Устарело и не правилось здесь: `ROADMAP.md` R4.3 и `CORRECTNESS.md` B33 — «ОЖИДАЕТ СОГЛАСИЯ»; `BIOS_PLAN.md` шаги 2 и 6 как предпосылка патча |
| Qwen — только локальная разработка и тесты | ADR-012.2, `ROADMAP.md` R2.4; `QWEN-MAX-SPEED.md` риск 10 |
| NVMe — план 4 × 1 ТБ (поправка заказчика 2026-10-12: **3 × 1 ТБ = 3 ТБ**; `docs/evidence/2026-10-12-n26b-nvme-3x1tb.md`) | ADR-012.2, `hardware/4x4080s-32g.json`; `AGENTS.md:36` («3x Samsung 990 Pro») не изменён — правит владелец (агентам запрещено) |
| каналы DRAM — измерить | `R0-HARDWARE-CHECKS.md` (a) |
| FAIL на синтетике не переименовывать без причины | правило `synthetic_run` в `tools/verify/pack.py` (N24) |
| SwiGLU не выбирать произвольно | источники N24 раздел 3; эксперимент подготовлен |
| `tokenizer.json` из закреплённых ревизий в CI | N23: `vendor_hf_files.py` `lfs_files`, `hf_lfs_fetch.py`, workflow `token-level` |
| исправить шаблон GLM | `templates/glm-5.3-flash/chat_template.jinja` (N23); в шаблоны R1 — предложение (в) 2 |
| файлу epyc-ai не доверять до проверки | `tools/hostcheck/glm_gguf_check.sh` (N24), NOT RUN |

**Открытые вопросы.**
1. Сессии PR #1: принять правила (в) в `scripts/r1/configs` и `DEPLOY.md`; поддерживает ли ik `--chat-template-file` (UNKNOWN).
2. Владельцу: правка `AGENTS.md:36` и `.claude/agents/infra-ops.md:20` (NVMe); письменное подтверждение решений 2026-10-11.
   [Поправка 2026-10-12: обе правки сделаны в N26 по разрешению владельца — 3 × 1 ТБ, ADR-012.3; правила (в) внедрены в
   `scripts/r1/` (N26a), ik `--chat-template-file` поддерживает (CONFIRMED, N26a).]
3. Каким вариантом SwiGLU обучена GLM-5.3-Flash — UNKNOWN до эксперимента; даст ли владелец BF16/Q8_0-опору.
4. Файл на epyc-ai — `glm5next` или `glm5-next` — UNKNOWN до проверки хоста.
5. Какой GGUF MiMo брать (R1.3) — UNKNOWN; риск kernelpool `qkv_chunks` открыт.
6. Нет задания CI, которое применяет N20 и N21 вместе (проверено только локально в N25).
7. Для N20–N25 Draft PR нет; N25 не опубликована (по заданию). Решение ведущего о публикации и базе PR — открыто.
8. Лицензия Qwen для продуктива третьим лицам не согласована; запасной путь — Qwen3.5-122B-A10B (Apache-2.0), потолок PROVISIONAL.
9. Порог Triad DRAM расходится между документами: `CORRECTNESS.md` B2 (по `EXPERIMENTS.md:31`) — 140–170 ГБ/с и < 90 для 4 каналов,
   `R0-HARDWARE-CHECKS.md` (a) — 170–180 / 90–100 и «красный флаг» < 150 / < 80; оба — гипотезы; решает замер R0.

## (ж) Цепочка Draft PR и ветки N20–N25

Базы и головы — CONFIRMED (`gh api …/pulls`, `git ls-remote origin`; приложение Б). Проверка слияния — `git merge-tree
--write-tree` (пишет только объекты, не ссылки и не рабочее дерево) базы с головой; `base-in-head` — содержит ли голова
текущую голову базы (`git merge-base --is-ancestor`). Литеральный вывод — приложение А.

| PR / ветка | голова | база @ голова базы | база в голове | merge-tree |
|---|---|---|---|---|
| #1 `claude/relaxed-dijkstra-1d7i5o` | `37415fe` | `main` @`eb59224` | да | чисто |
| #2 `claude/n7-server-hybrid-cpu` | `4f7be84` | #1 @`37415fe` | нет | чисто |
| #3 `claude/research-landscape` | `2c64027` | #1 @`37415fe` | нет | чисто |
| #4 `claude/n8-session-store` | `09eb438` | #2 @`4f7be84` | да | чисто |
| #5 `claude/n9-glm-decode-parity` | `af4db13` | #3 @`2c64027` | нет | чисто |
| #6 `claude/n10-quant-fa-parity` | `0d4d582` | #5 @`af4db13` | да | чисто |
| #7 `claude/n11-r1-tools-cpu-server` | `b74efde` | #2 @`4f7be84` | да | чисто |
| #8 `claude/n12-ik-glm5next-cpu` | `5730cc4` | #6 @`0d4d582` | да | чисто |
| #9 `claude/n13-ci-no-silent-skips` | `6dd6b28` | #8 @`5730cc4` | да | чисто |
| #10 `claude/n14-chat-templates` | `b2096c1` | #9 @`6dd6b28` | да | чисто |
| #11 `claude/n16-mimo-decode-parity` | `0b0eace` | #10 @`b2096c1` | нет | чисто |
| #12 `claude/n15-gguf-headers` | `67bd5ba` | #10 @`b2096c1` | нет | чисто |
| #13 `claude/n18-gguf-headers-mtp-mimo` | `35ba5ba` | #12 @`67bd5ba` | да | чисто |
| #14 `claude/n17-mimo-mtp-heads-fa` | `3ce2293` | #11 @`0b0eace` | да | чисто |
| #15 `claude/n19-qwen4exp-multiseq` | `e71b241` | #14 @`3ce2293` | да | чисто |
| N20 `claude/n20-recurrent-rollback-guard` (PR нет) | `b7d6af1` | N19 @`e71b241` | да | чисто |
| N21 `claude/n21-mimo-mtp-heads-fix` (PR нет) | `93a48d6` | N19 @`e71b241` | да | чисто |
| N22 `claude/n22-ci-false-green` (PR нет) | `56202c8` | N19 @`e71b241` | да | чисто |
| N23 `claude/n23-glm-template-tokenizer` (PR нет) | `7541c4a` | N19 @`e71b241` | да | чисто |
| N24 `claude/n24-owner-decisions` (PR нет) | `76e9d56` | N19 @`e71b241` | да | чисто |
| N25 `claude/n25-integration` (локально) | `f57547a` при проверке | N19 @`e71b241` | да | чисто; содержит N20–N24 целиком |

«База в голове: нет» у #2, #3, #5, #11, #12 — база ушла вперёд после открытия PR (у #5 API показывает базу `3d7ad83`, а голова
`claude/research-landscape` теперь `2c64027`); слияние при этом чистое. N25 с ветками, которых нет в её истории: с N15, N18,
research-landscape и PR #1 — чисто; **с N11 (`b74efde`) — конфликты** в `docs/evidence/mutation-scripts/README.md`,
`tools/tinygen/glm5next_tiny.py`, `tools/tinygen/qwen4exp_tiny.py`; **с N8 (`09eb438`)** — в
`docs/evidence/mutation-scripts/README.md`, `tools/tinygen/glm5next_tiny.py` (те же три файла N24 видел между N19 и N11 —
`docs/evidence/2026-10-11-n24-owner-decisions.md`, введение). Их разрешит тот, кто будет сводить ветку N7/N8/N11 с N25.
После проверки в N25 добавлен только этот файл (новый путь), что на результаты merge-tree не влияет.

## Приложение А. merge-tree по цепочке (литерально)

```
$ cat chain.sh
#!/usr/bin/env bash
# Merge-tree check along the Draft PR chain and the N20-N25 branches (read-only: git merge-tree --write-tree writes
# objects, never refs or the worktree).
cd /home/user/wt-n25
chk() {  # chk <label> <base ref> <head ref>
  local b=$1 base=$2 head=$3 out rc anc
  out=$(git merge-tree --write-tree --name-only --no-messages "$base" "$head" 2>&1); rc=$?
  git merge-base --is-ancestor "$base" "$head" && anc=yes || anc=no
  printf '%-6s %-44s <- %-42s base-in-head=%s merge-tree rc=%s' "$b" "${base#origin/}@$(git rev-parse --short "$base")" "${head#origin/}@$(git rev-parse --short "$head")" "$anc" "$rc"
  if [ $rc -eq 0 ]; then echo " clean"; else echo " CONFLICTS: $(echo "$out" | tail -n +2 | tr '\n' ' ')"; fi
}
o=origin/claude
chk '#1'  origin/main                    $o/relaxed-dijkstra-1d7i5o
chk '#2'  $o/relaxed-dijkstra-1d7i5o     $o/n7-server-hybrid-cpu
chk '#3'  $o/relaxed-dijkstra-1d7i5o     $o/research-landscape
chk '#4'  $o/n7-server-hybrid-cpu        $o/n8-session-store
chk '#5'  $o/research-landscape          $o/n9-glm-decode-parity
chk '#6'  $o/n9-glm-decode-parity        $o/n10-quant-fa-parity
chk '#7'  $o/n7-server-hybrid-cpu        $o/n11-r1-tools-cpu-server
chk '#8'  $o/n10-quant-fa-parity         $o/n12-ik-glm5next-cpu
chk '#9'  $o/n12-ik-glm5next-cpu         $o/n13-ci-no-silent-skips
chk '#10' $o/n13-ci-no-silent-skips      $o/n14-chat-templates
chk '#11' $o/n14-chat-templates          $o/n16-mimo-decode-parity
chk '#12' $o/n14-chat-templates          $o/n15-gguf-headers
chk '#13' $o/n15-gguf-headers            $o/n18-gguf-headers-mtp-mimo
chk '#14' $o/n16-mimo-decode-parity      $o/n17-mimo-mtp-heads-fa
chk '#15' $o/n17-mimo-mtp-heads-fa       $o/n19-qwen4exp-multiseq
chk 'N20' $o/n19-qwen4exp-multiseq       $o/n20-recurrent-rollback-guard
chk 'N21' $o/n19-qwen4exp-multiseq       $o/n21-mimo-mtp-heads-fix
chk 'N22' $o/n19-qwen4exp-multiseq       $o/n22-ci-false-green
chk 'N23' $o/n19-qwen4exp-multiseq       $o/n23-glm-template-tokenizer
chk 'N24' $o/n19-qwen4exp-multiseq       $o/n24-owner-decisions
chk 'N25' $o/n19-qwen4exp-multiseq       claude/n25-integration
for h in n20-recurrent-rollback-guard n21-mimo-mtp-heads-fix n22-ci-false-green n23-glm-template-tokenizer n24-owner-decisions; do
  git merge-base --is-ancestor $o/$h claude/n25-integration && echo "N25 contains $h @$(git rev-parse --short $o/$h)" || echo "N25 does NOT contain $h"
done
# side branches that are not in N25's history
for h in n15-gguf-headers n18-gguf-headers-mtp-mimo n11-r1-tools-cpu-server n8-session-store research-landscape relaxed-dijkstra-1d7i5o; do
  out=$(git merge-tree --write-tree --name-only --no-messages claude/n25-integration $o/$h 2>&1); rc=$?
  git merge-base --is-ancestor $o/$h claude/n25-integration && anc=yes || anc=no
  echo "N25 x $h @$(git rev-parse --short $o/$h): in N25=$anc merge-tree rc=$rc $( [ $rc -eq 0 ] && echo clean || echo "CONFLICTS: $(echo "$out" | tail -n +2 | tr '\n' ' ')")"
done
$ git rev-parse --short HEAD
f57547a
$ bash chain.sh   # body below
#1     main@eb59224                                 <- claude/relaxed-dijkstra-1d7i5o@37415fe     base-in-head=yes merge-tree rc=0 clean
#2     claude/relaxed-dijkstra-1d7i5o@37415fe       <- claude/n7-server-hybrid-cpu@4f7be84        base-in-head=no merge-tree rc=0 clean
#3     claude/relaxed-dijkstra-1d7i5o@37415fe       <- claude/research-landscape@2c64027          base-in-head=no merge-tree rc=0 clean
#4     claude/n7-server-hybrid-cpu@4f7be84          <- claude/n8-session-store@09eb438            base-in-head=yes merge-tree rc=0 clean
#5     claude/research-landscape@2c64027            <- claude/n9-glm-decode-parity@af4db13        base-in-head=no merge-tree rc=0 clean
#6     claude/n9-glm-decode-parity@af4db13          <- claude/n10-quant-fa-parity@0d4d582         base-in-head=yes merge-tree rc=0 clean
#7     claude/n7-server-hybrid-cpu@4f7be84          <- claude/n11-r1-tools-cpu-server@b74efde     base-in-head=yes merge-tree rc=0 clean
#8     claude/n10-quant-fa-parity@0d4d582           <- claude/n12-ik-glm5next-cpu@5730cc4         base-in-head=yes merge-tree rc=0 clean
#9     claude/n12-ik-glm5next-cpu@5730cc4           <- claude/n13-ci-no-silent-skips@6dd6b28      base-in-head=yes merge-tree rc=0 clean
#10    claude/n13-ci-no-silent-skips@6dd6b28        <- claude/n14-chat-templates@b2096c1          base-in-head=yes merge-tree rc=0 clean
#11    claude/n14-chat-templates@b2096c1            <- claude/n16-mimo-decode-parity@0b0eace      base-in-head=no merge-tree rc=0 clean
#12    claude/n14-chat-templates@b2096c1            <- claude/n15-gguf-headers@67bd5ba            base-in-head=no merge-tree rc=0 clean
#13    claude/n15-gguf-headers@67bd5ba              <- claude/n18-gguf-headers-mtp-mimo@35ba5ba   base-in-head=yes merge-tree rc=0 clean
#14    claude/n16-mimo-decode-parity@0b0eace        <- claude/n17-mimo-mtp-heads-fa@3ce2293       base-in-head=yes merge-tree rc=0 clean
#15    claude/n17-mimo-mtp-heads-fa@3ce2293         <- claude/n19-qwen4exp-multiseq@e71b241       base-in-head=yes merge-tree rc=0 clean
N20    claude/n19-qwen4exp-multiseq@e71b241         <- claude/n20-recurrent-rollback-guard@b7d6af1 base-in-head=yes merge-tree rc=0 clean
N21    claude/n19-qwen4exp-multiseq@e71b241         <- claude/n21-mimo-mtp-heads-fix@93a48d6      base-in-head=yes merge-tree rc=0 clean
N22    claude/n19-qwen4exp-multiseq@e71b241         <- claude/n22-ci-false-green@56202c8          base-in-head=yes merge-tree rc=0 clean
N23    claude/n19-qwen4exp-multiseq@e71b241         <- claude/n23-glm-template-tokenizer@7541c4a  base-in-head=yes merge-tree rc=0 clean
N24    claude/n19-qwen4exp-multiseq@e71b241         <- claude/n24-owner-decisions@76e9d56         base-in-head=yes merge-tree rc=0 clean
N25    claude/n19-qwen4exp-multiseq@e71b241         <- claude/n25-integration@f57547a             base-in-head=yes merge-tree rc=0 clean
N25 contains n20-recurrent-rollback-guard @b7d6af1
N25 contains n21-mimo-mtp-heads-fix @93a48d6
N25 contains n22-ci-false-green @56202c8
N25 contains n23-glm-template-tokenizer @7541c4a
N25 contains n24-owner-decisions @76e9d56
N25 x n15-gguf-headers @67bd5ba: in N25=no merge-tree rc=0 clean
N25 x n18-gguf-headers-mtp-mimo @35ba5ba: in N25=no merge-tree rc=0 clean
N25 x n11-r1-tools-cpu-server @b74efde: in N25=no merge-tree rc=1 CONFLICTS: docs/evidence/mutation-scripts/README.md tools/tinygen/glm5next_tiny.py tools/tinygen/qwen4exp_tiny.py 
N25 x n8-session-store @09eb438: in N25=no merge-tree rc=1 CONFLICTS: docs/evidence/mutation-scripts/README.md tools/tinygen/glm5next_tiny.py 
N25 x research-landscape @2c64027: in N25=no merge-tree rc=0 clean
N25 x relaxed-dijkstra-1d7i5o @37415fe: in N25=no merge-tree rc=0 clean
```

Ссылки `origin/*` локального клона равны `git ls-remote origin` на момент проверки (сравнение `for-each-ref` с `ls-remote` — без
различий).

## Приложение Б. GitHub, шаблоны R1 (литерально)

Только чтение (`git`, `gh api` GET). Пути песочницы не встречаются.

```
$ git rev-parse --short HEAD
f57547a
$ git ls-remote origin 'refs/heads/*'
986da38164bf731ab1c9cb6dd19b88d418dc8ad4	refs/heads/3090
0d4d582209f41ed8131cb49525aa3a08290cb90a	refs/heads/claude/n10-quant-fa-parity
b74efdeabc882eb93ba94192de597b7be05144cb	refs/heads/claude/n11-r1-tools-cpu-server
5730cc4225c712f5aef5901758b3ac20e3cc67e9	refs/heads/claude/n12-ik-glm5next-cpu
6dd6b28eec9489a2fc343e1b3d0892fa8d25819d	refs/heads/claude/n13-ci-no-silent-skips
b2096c1a897c5ceee257b891582b4c5d97413edb	refs/heads/claude/n14-chat-templates
67bd5baceafe63e70e1b78010181225ecebbefa4	refs/heads/claude/n15-gguf-headers
0b0eace7748a413ed19232a6a5a7e590937a671e	refs/heads/claude/n16-mimo-decode-parity
3ce229374809392eccf09c2dfa81d01f004c587c	refs/heads/claude/n17-mimo-mtp-heads-fa
35ba5ba4a7d957ffc53d5fbb2354ea6f8a1e3eb8	refs/heads/claude/n18-gguf-headers-mtp-mimo
e71b241011575ccaf4a7d309f15ed54d339d614d	refs/heads/claude/n19-qwen4exp-multiseq
b7d6af11f4fbec136fd995399bd0cf3ab980823b	refs/heads/claude/n20-recurrent-rollback-guard
93a48d6bc0bfa243d36e9a4442ff213ef71bc14d	refs/heads/claude/n21-mimo-mtp-heads-fix
56202c8fe0949657d6583af4a4e5dece6141abd2	refs/heads/claude/n22-ci-false-green
7541c4a67b71213d0506215c785d0fbf4388a9f0	refs/heads/claude/n23-glm-template-tokenizer
76e9d565d1f6ac615cbdda9f5d7cbcc023976344	refs/heads/claude/n24-owner-decisions
4f7be84cbfaed37e7c74b1c7d92fe746e64a2788	refs/heads/claude/n7-server-hybrid-cpu
09eb438e28f86b292395da60380361fb386dddb0	refs/heads/claude/n8-session-store
af4db13f43f7027f20effb5401affcd0ca365a93	refs/heads/claude/n9-glm-decode-parity
37415fedcfb754941c45effda1d6c80e735ef3a9	refs/heads/claude/relaxed-dijkstra-1d7i5o
2c640273b248226ae3e51e9b6082948855d8c1d0	refs/heads/claude/research-landscape
eb59224c835581740a715ddcaa9abe101f90aa2f	refs/heads/main
$ gh api 'repos/samnolak/408032gb/pulls?state=all&per_page=50' --jq '.[] | "\(.number) \(.state) draft=\(.draft) \(.head.ref)@\(.head.sha[0:7]) -> \(.base.ref)@\(.base.sha[0:7]) merged=\(.merged_at)"' | sort -n
1 open draft=true claude/relaxed-dijkstra-1d7i5o@37415fe -> main@eb59224 merged=null
2 open draft=true claude/n7-server-hybrid-cpu@4f7be84 -> claude/relaxed-dijkstra-1d7i5o@259081e merged=null
3 open draft=true claude/research-landscape@2c64027 -> claude/relaxed-dijkstra-1d7i5o@259081e merged=null
4 open draft=true claude/n8-session-store@09eb438 -> claude/n7-server-hybrid-cpu@4f7be84 merged=null
5 open draft=true claude/n9-glm-decode-parity@af4db13 -> claude/research-landscape@3d7ad83 merged=null
6 open draft=true claude/n10-quant-fa-parity@0d4d582 -> claude/n9-glm-decode-parity@af4db13 merged=null
7 open draft=true claude/n11-r1-tools-cpu-server@b74efde -> claude/n7-server-hybrid-cpu@4f7be84 merged=null
8 open draft=true claude/n12-ik-glm5next-cpu@5730cc4 -> claude/n10-quant-fa-parity@0d4d582 merged=null
9 open draft=true claude/n13-ci-no-silent-skips@6dd6b28 -> claude/n12-ik-glm5next-cpu@5730cc4 merged=null
10 open draft=true claude/n14-chat-templates@b2096c1 -> claude/n13-ci-no-silent-skips@6dd6b28 merged=null
11 open draft=true claude/n16-mimo-decode-parity@0b0eace -> claude/n14-chat-templates@b2096c1 merged=null
12 open draft=true claude/n15-gguf-headers@67bd5ba -> claude/n14-chat-templates@b2096c1 merged=null
13 open draft=true claude/n18-gguf-headers-mtp-mimo@35ba5ba -> claude/n15-gguf-headers@67bd5ba merged=null
14 open draft=true claude/n17-mimo-mtp-heads-fa@3ce2293 -> claude/n16-mimo-decode-parity@0b0eace merged=null
15 open draft=true claude/n19-qwen4exp-multiseq@e71b241 -> claude/n17-mimo-mtp-heads-fa@3ce2293 merged=null
$ gh api 'repos/samnolak/408032gb/actions/runs?branch=claude/n20-recurrent-rollback-guard&per_page=10' --jq '.workflow_runs[] | "\(.id) \(.name) \(.head_sha[0:7]) \(.event) \(.status) \(.conclusion)"'
38175409480 ci b7d6af1 push completed success
38175409509 parity-r1-pin b7d6af1 push completed success
$ gh api 'repos/samnolak/408032gb/actions/runs?branch=claude/n21-mimo-mtp-heads-fix&per_page=10' --jq '.workflow_runs[] | "\(.id) \(.name) \(.head_sha[0:7]) \(.event) \(.status) \(.conclusion)"'
38175157536 ci 93a48d6 push completed success
38175157541 parity-r1-pin 93a48d6 push completed success
$ gh api 'repos/samnolak/408032gb/actions/runs?branch=claude/n22-ci-false-green&per_page=10' --jq '.workflow_runs[] | "\(.id) \(.name) \(.head_sha[0:7]) \(.event) \(.status) \(.conclusion)"'
38169629180 parity-r1-pin 56202c8 push completed success
38169629204 ci 56202c8 push completed success
$ gh api 'repos/samnolak/408032gb/actions/runs?branch=claude/n23-glm-template-tokenizer&per_page=10' --jq '.workflow_runs[] | "\(.id) \(.name) \(.head_sha[0:7]) \(.event) \(.status) \(.conclusion)"'
38176823636 token-level 7541c4a push completed success
38176823592 ci 7541c4a push completed success
38175655900 token-level 4447b2c push completed success
38175655895 ci 4447b2c push completed success
38169032753 token-level 96dd906 push completed success
38169032771 ci 96dd906 push completed success
38168065511 vendor-hf-branch d4583c5 push completed failure
38168065231 ci d4583c5 push completed success
38168065274 token-level d4583c5 push completed success
$ gh api 'repos/samnolak/408032gb/actions/runs?branch=claude/n24-owner-decisions&per_page=10' --jq '.workflow_runs[] | "\(.id) \(.name) \(.head_sha[0:7]) \(.event) \(.status) \(.conclusion)"'
38169631240 ci 76e9d56 push completed success
$ gh api repos/samnolak/408032gb/actions/runs/38169629204/jobs --jq '.jobs[] | "\(.id) \(.name) \(.conclusion)"'
114559323648 cpu-parity success
114559323729 strata-glm-sm89 (86) success
114559323738 cuda-compile success
114559323773 strata-glm-sm89 (89) success
114559323819 ref-torch-cpu success
114559323852 lt-strata-sm89 success
$ gh api repos/samnolak/408032gb/check-runs/114559323738/annotations --jq '.[] | select(.annotation_level=="notice") | .message'
sm_89 CUDA build:   gpu_p2p_matrix.cu: nvcc -arch=sm_89 OK, 1075888 bytes   gpu_vram_check.cu: nvcc -arch=sm_89 OK, 1024920 bytes Ran 5 tests in 7.266s OK  Cuda compilation tools, release 13.0, V13.0.88 Build cuda_13.0.r13.0/compiler.36424714_0 
$ gh api repos/samnolak/408032gb/check-runs/114559323648/annotations --jq '.[] | select(.message|test("diag|unit tests")) | "\(.annotation_level): \(.message)"'
warning: scripts/diag/*.cu NOT compiled in this job (no nvcc here): tests.test_diag_kit.CudaCompile skipped; the sm_89 build runs in job cuda-compile
notice: unit tests: Ran 1102 tests in 95.081s OK (skipped=558) 
$ gh api repos/samnolak/408032gb/check-runs/114576646023/annotations --jq '.[] | select(.annotation_level=="notice") | .message' | grep -E '^n2[01]-'
n20-qwen4exp-rs-guard @ llama.cpp ec7630a: Ran 110, skipped 0 as expected; the driver's checkout ../llama.cpp is at ec7630a
$ gh api repos/samnolak/408032gb/check-runs/114576646756/annotations --jq '.[] | select(.annotation_level=="notice") | .message' | grep -E '^n2[01]-'
n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 110, skipped 0 as expected; the driver's checkout ../llama.cpp is at 79e2e74
$ gh api repos/samnolak/408032gb/check-runs/114575893704/annotations --jq '.[] | select(.annotation_level=="notice") | .message' | grep -E '^n2[01]-'
n21-mimo2-mtp-fix @ llama.cpp 79e2e74: Ran 32, skipped 6 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.ReplayRefusal.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_independent_of_the_mtp_order); the driver's checkout ../llama.cpp is at 79e2e74
$ gh api repos/samnolak/408032gb/check-runs/114575893826/annotations --jq '.[] | select(.annotation_level=="notice") | .message' | grep -E '^n2[01]-'
n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 57, skipped 0 as expected; the driver's checkout ../llama.cpp is at 79e2e74
$ python3 scripts/r1/rs_rollback_rule.py scripts/r1/configs/*.json; echo rc=$?
scripts/r1/configs/ik-llamacpp-server.json: not covered (engine pin 'ik_llama.cpp')
scripts/r1/configs/llamacpp-server-moecache.json: ok (spec types none, draft model no, slots 1)
scripts/r1/configs/llamacpp-server.json: ok (spec types none, draft model no, slots 1)
scripts/r1/configs/mock.json: not covered (engine pin None)
scripts/r1/configs/strata-serve-qwen38fn.json: not covered (engine pin 'strata')
rc=0
$ git diff --quiet 93bdfe0 HEAD -- scripts/r1/configs; echo rc=$?
rc=0
$ for sha in 37415fe 4f7be84 2c64027 09eb438 af4db13 0d4d582 b74efde 5730cc4 6dd6b28 b2096c1 0b0eace 67bd5ba 35ba5ba 3ce2293 e71b241; do full=$(git rev-parse $sha); echo "$sha: $(gh api "repos/samnolak/408032gb/actions/runs?head_sha=$full&per_page=30" --jq '[.workflow_runs[] | "\(.name)/\(.event)=\(.conclusion)"] | join(" ")')"; done
37415fe: engines-sm89/pull_request=success ci/pull_request=success ci/push=success
4f7be84: server-cpu/pull_request=success ci/pull_request=success server-cpu/push=success ci/push=success
2c64027: parity-r1-pin/pull_request=success ci/pull_request=success ci/push=success
09eb438: server-cpu/pull_request=success ci/pull_request=success server-cpu/push=success ci/push=success
af4db13: parity-r1-pin/pull_request=success ci/pull_request=success ci/push=success
0d4d582: parity-r1-pin/pull_request=success ci/pull_request=success ci/push=success
b74efde: server-cpu/pull_request=success ci/pull_request=success server-cpu/push=success ci/push=success
5730cc4: ci/pull_request=success parity-r1-pin/pull_request=success ci/push=success parity-r1-pin/push=success
6dd6b28: ci/pull_request=success parity-r1-pin/pull_request=success ci/push=success parity-r1-pin/push=success
b2096c1: ci/pull_request=success ci/push=success
0b0eace: ci/pull_request=success parity-r1-pin/pull_request=success ci/push=success
67bd5ba: ci/pull_request=success ci/push=success
35ba5ba: ci/pull_request=success ci/push=success
3ce2293: parity-r1-pin/pull_request=success ci/pull_request=success ci/push=success parity-r1-pin/push=success
e71b241: parity-r1-pin/pull_request=success ci/pull_request=success ci/push=success parity-r1-pin/push=success
```

## Приложение В. Тесты на голове ветки (литерально)

`<S>` — каталог песочницы сессии (`/tmp/claude-0/…/scratchpad`), подставлен только он. Полный набор запускался на `f57547a` (до добавления этого файла; тесты его не читают).

```
$ git rev-parse --short HEAD
f57547a
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > <S>/n25close/discover.log 2>&1; echo exit=$? >> <S>/n25close/discover.log; tail -n 5 <S>/n25close/discover.log
----------------------------------------------------------------------
Ran 1238 tests in 178.396s

OK (skipped=608)
exit=0
$ git rev-parse --short HEAD
f57547a
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest tests.test_check_unittest tests.test_hooks 2>&1 | tail -n 4; echo exit=${PIPESTATUS[0]}
----------------------------------------------------------------------
Ran 54 tests in 26.211s

OK
exit=0
$ python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('yaml ok:', ' '.join(sys.argv[1:]))" .github/workflows/*.yml; echo exit=$?
yaml ok: .github/workflows/ci.yml .github/workflows/engines-sm89.yml .github/workflows/parity-r1-pin.yml .github/workflows/token-level.yml .github/workflows/traces-real.yml .github/workflows/vendor-deepseek.yml .github/workflows/vendor-hf-branch.yml .github/workflows/vendor-hf.yml
exit=0
```

608 пропусков `discover` — тесты на дампах без своих дампов (их гоняют шаги паритета через `tools/ci/check_unittest.py`), `CudaCompile` без nvcc и прочие по причинам из N22, раздел 9 (состав пропусков здесь не пересчитывался — PROVISIONAL). Шаги паритета на дампах в этой ветке — `docs/evidence/2026-10-11-n25-integration.md`, разделы 2–3 (на `1f35c7a`/`447d7e2`; после них менялись только документы и добавлен скрипт расчёта `docs/evidence/mutation-scripts/2026-10-11-n25-qwen-placement.py`, который ни один шаг не читает).
