# Протокол замеров: скорость, длинный контекст и агентные пробы через OpenAI API (`tools/bench/`)

Автор: perf-bench, 2026-10-09. Задача R1.1 (`docs/reassessment/ROADMAP.md:24@393cbc1`).
Ярлыки — по `AGENTS.md`: CONFIRMED (с источником), PROVISIONAL (как получено), UNKNOWN (чем подтвердить), NOT RUN.

**Состояние инструментов.** `tools/bench/*` проверены только против мок-сервера `tools/bench/mock_server.py`
в `tests/test_bench.py` (без GPU). Ни один реальный движок через эти инструменты ещё не запускался — **NOT RUN**.
Числа мок-сервера не говорят ничего о движках: `summary.md` таких прогонов начинается с плашки
«MOCK SERVER RUN», в `meta.json` стоит `"is_mock_server": true`. Такие прогоны не являются доказательством.

## 1. Инструменты

| файл | назначение |
|---|---|
| `tools/bench/oa_client.py` | клиент OpenAI API на stdlib: `/v1/chat/completions`, `/v1/completions`, SSE, время прихода каждого чанка (`perf_counter_ns`), `usage`, `finish_reason`, сборка `tool_calls`; `/tokenize` (диалекты llama.cpp и vLLM), `/v1/models`, `/health`, `/metrics` как сырой текст |
| `tools/bench/prompts.py` | детерминированные (seed) заполнители `prose` / `code` / `logs` и подгонка под точное число токенов; CLI печатает достигнутое число для 2K…262K |
| `tools/bench/run_matrix.py` | матрица вход × выход × {cold, warm} × конкурентность × повторы + сценарий «agent loop» |
| `tools/bench/probes.py` | пробы: needle, multihop (переменные), code QA, JSON, вызов инструментов, детерминизм |
| `tools/bench/mock_server.py` | мок-сервер для тестов (задержки, кэш префиксов, игрушечная «модель», инъекция отказов) |
| `tools/bench/runlog.py` | общий формат каталога прогона, метрики, статистика |

Требования к среде запуска: только Python 3 (проверено на 3.13; модули stdlib), без pip и без интернета.
Клиент лучше запускать на той же VM, что и сервер: тогда сеть почти не влияет на TTFT, и вывод `nvidia-smi`,
который пишется в `meta.json`, относится к машине сервера. Если клиент на другой машине, это нужно указать в `--engine-desc`.

## 2. Предрегистрация (заполнить ДО запуска)

Каждый прогон, числа которого пойдут в отчёт, сначала регистрируется в `docs/reassessment/EXPERIMENTS.md`
(общие условия там же, строки 7–13). Минимальный набор полей:

```text
id:              <B1|B2|B3|K1|K4|...> / <короткое имя>
гипотеза:        <что ожидаем и почему>
предсказание:    <число из tools/roofline.py + полная команда>  (PROVISIONAL, записать заранее)
бейзлайн:        <с каким прогоном сравниваем: путь к его каталогу в docs/evidence/>
модель:          <HF id + ревизия или GGUF-файлы + sha256>; квант <...>; KV dtype <...>
контекст сервера: <ctx>; слотов <n>; длины входа <...>; длины вывода <...>
движок:          <имя> @ <commit/версия>; ТОЧНАЯ команда запуска сервера; переменные окружения
железо:          4× RTX 4080 Super 32 GB; драйвер/CUDA; PCIe gen/ширина; P2P да/нет; каналы RAM (блок H)
состояние:       cold / warm (что считается холодным: свежий запуск сервера или только новый префикс)
команды клиента: полные строки run_matrix.py / probes.py
SHA репозитория: `git rev-parse HEAD` (run_matrix/probes пишут его в meta.json сами)
повторы:         >= 3; статистика p50 / p95 / min / max
порог остановки: <что прекращает ветку, напр. декод p50 < 15 tok/s при 32K (B1)>
PASS/FAIL:       <критерий до запуска; без подгонки после>
что игнорируется: <напр. сеть, если клиент на другой машине>
```

## 3. Как считаются метрики

Определения также записываются в каждый `summary.json` (ключ `definitions`, код в `tools/bench/runlog.py`).

| метрика | определение | оговорки |
|---|---|---|
| TTFT | от отправки запроса до первого SSE-чанка с выводом (content, reasoning или tool call) | включает HTTP, очередь, токенизацию, префилл и первый шаг декода. Чанк, где есть только `role`, не считается |
| TTFC | то же, но до первого content/tool-чанка | у моделей с рассуждением TTFC > TTFT |
| prefill tok/s | `usage.prompt_tokens / TTFT` | в warm сюда входят закэшированные токены: это не скорость вычисления. Для неё есть `uncached_prefill_tok_s = (prompt_tokens − cached)/TTFT`, считается, только если сервер сообщает `cached_tokens` |
| decode tok/s | `(n − n/chunks) / (t_последнего − t_первого чанка)`, где n = `usage.completion_tokens` (если его нет, число чанков) | при 1 токене на чанк это ровно `(n−1)/(t_last−t_first)`. Если спекуляция отдаёт по несколько токенов в чанке, формула предполагает равные чанки (PROVISIONAL, погрешность ≤ (k−1)/n). Токены рассуждения входят в n |
| ITL p50/p95/p99 | интервалы между чанками с выводом | равны межтокенным только при 1 токене на чанк; поле `tokens_per_chunk` показывает, так ли это |
| aggregate out tok/s | `Σ completion_tokens / (время волны: первая отправка → последнее завершение)`; только для конкурентности > 1 | в отдельной таблице; **не сравнивать** с single-stream |
| short_output | `completion_tokens < max_tokens` в ячейке, где длина вывода принудительная | строки не выбрасываются, а помечаются; режим принуждения записан в `meta.eos` |

Перцентили считаются линейной интерполяцией между порядковыми статистиками (как `numpy` `linear`).
При 3 повторах p95 почти равен максимуму, поэтому min/max печатаются рядом.

Длина вывода принуждается ключами тела запроса. Режим `--eos-mode auto` сначала шлёт короткий запрос без ключей
(модель должна остановиться рано), затем пробует два набора ключей и записывает в `meta.eos` все попытки и выбранный вариант:

- `vllm`: `{"ignore_eos": true, "min_tokens": N}`. Поля CONFIRMED: `vllm/entrypoints/openai/chat_completion/protocol.py:275-276@bc21cba` (`ignore_eos`, `min_tokens`).
- `llamacpp`: `{"ignore_eos": true, "n_predict": N}`. CONFIRMED: `tools/server/README.md:575@ec7630a` (`ignore_eos`); то, что `n_predict` перекрывает `max_tokens`, — `tools/server/server-common.cpp:1065-1066@ec7630a`.

Если не сработал ни один набор, режим будет `none`. Ячейки с коротким выводом тогда видны в колонке `short`.

Число токенов промпта берётся из `usage.prompt_tokens` — из того, что посчитал сам сервер. Подгонка промпта под цель:
1. Точно по `/tokenize` сервера. Диалект llama.cpp: `{"content": ...}` → `{"tokens": [...]}`, CONFIRMED (`tools/server/server.cpp:281@ec7630a`, `tools/server/README.md:677-681@ec7630a`). Диалект vLLM: `{"model", "prompt"}` → `{"count", "max_model_len", "tokens"}`, CONFIRMED (`vllm/entrypoints/serve/tokenize/api_router.py:35-36@bc21cba`, `vllm/entrypoints/serve/tokenize/protocol.py:20-22,157-160@bc21cba`). Накладные токены chat-шаблона измеряются одним запросом и вычитаются из цели.
2. Если `/tokenize` нет: соотношение символов на токен калибруется по `usage.prompt_tokens` двух запросов (линейная подгонка даёт и соотношение, и накладные токены шаблона). Затем — поиск по длине и по умолчанию один уточняющий запрос полной длины на каждую новую длину (`--ratio-refine`; каждый стоит одного префилла). Подгонка в этом случае приблизительная (`exact: false`), фактическое число всё равно берётся из `usage`.

Кэш префиксов на сервере виден из `usage.prompt_tokens_details.cached_tokens` (у llama.cpp CONFIRMED: `tools/server/server-task.cpp:370@ec7630a`) или из `timings.cache_n` (`tools/server/server-common.cpp:86@ec7630a`). `stream_options.include_usage` отправляется всегда. Если сервер отвечает на него 400/422, запрос повторяется без него, и это записывается (`stream_options_supported: false`). Последний чанк с `usage` и пустым `choices` в llama.cpp: `tools/server/server-task.cpp:502-504@ec7630a`.

## 4. Матрица и сценарии

```bash
python3 tools/bench/run_matrix.py --url http://127.0.0.1:<PORT> \
  --inputs 2048,32768,131072,262144 --outputs 128,512,2048 --states cold,warm --concurrency 1,<N> \
  --repeats 3 --length-mode total --filler prose --seed 0 \
  --engine-desc "<движок @ commit; команда сервера>" --model-desc "<модель, квант, файлы>" \
  --out-dir docs/evidence/R1-matrix/<YYYY-MM-DD>-<engine>-<model>
```

- **cold**: каждый запрос начинается со свежего случайного префикса (уникальный nonce в первой строке), так что кэш не может помочь. Общей остаётся только голова chat-шаблона в несколько токенов — так же и у настоящих движков. Nonce зависит и от `--nonce-salt` (по умолчанию — случайная соль на каждый прогон, пишется в `meta.json`; тот же `--nonce-salt` повторяет промпты прогона): до N11 он зависел только от `--seed` и ячейки, и повторный прогон против того же llama-server получал «холодные» промпты из кэша (CONFIRMED на CPU, `docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md`).
- **warm**: сначала прайминг (строка `phase: prime`), затем тот же длинный префикс с другим коротким хвостом. Падение TTFT относительно cold показывает повторное использование префикса. Сводка печатает таблицу warm/cold.
- **конкурентность N**: N запросов одновременно (через барьер). Метрики на запрос плюс агрегат по волне идут в отдельную таблицу. N ставится равным числу слотов сервера.
- **`--length-mode total`**: промпт = вход − вывод, и окно 262 144 не переполняется. При `prompt` (по умолчанию) ячейка 262144 + 2048 потребует от сервера окно больше 262 144 — иначе она будет записана как отказ.
- **agent loop** (`--scenarios agent`): ходы, в каждом к разговору добавляется синтетический вызов `read_file` и результат инструмента на 1–4K токенов (`--agent-tool-tokens 1024,4096`), до `--agent-target-tokens` (по умолчанию 131072) или `--agent-turns`. По каждому ходу пишутся TTFT, prompt_tokens, новые токены и `cached_tokens` сервера. Затем идёт контрольный холодный запрос: тот же финальный разговор с новым nonce сессии. Ассистентские сообщения синтетические и одинаковы для всех движков (сравнимость важнее реализма). Если сервер не принимает `role: tool`, используется `--agent-format user`.
- **Таймауты**: `--timeout` — простой сокета на запрос (по умолчанию 3600 с, потому что префилл 262K может молчать минутами; TTFT 262K ≥ 1.5–3 мин — PROVISIONAL граница из `docs/reassessment/ROOFLINE.md:78@393cbc1`), `--total-timeout` — полный лимит. Превышение записывается как отказ.
- **`--extra-body`** добавляется к каждому запросу, например `'{"chat_template_kwargs": {"enable_thinking": false}}'`, и сохраняется в `meta.args`. Включено рассуждение или нет — часть предрегистрации.

Декод «при заполненном контексте» — это ячейки с входом L и выводом ≥ 512 (`EXPERIMENTS.md:11@393cbc1`).
Продолжительность прогона по умолчанию (4 × 3 × 2 × 2 × 3 ячейки при 262K): UNKNOWN до первого замера. Ориентир — отдельные
холодные префиллы 262K по несколько минут каждый (PROVISIONAL, ROOFLINE.md:78). Поэтому рекомендуется прогонять по одной длине входа на запуск.

## 5. Пробы и что считается PASS для 262K

```bash
python3 tools/bench/probes.py --url http://127.0.0.1:<PORT> --contexts 32768,131072,262144 \
  --length-mode total --answer-tokens 1024 --seed 0 \
  --engine-desc "<...>" --model-desc "<...>" --out-dir docs/evidence/R1-probes/<YYYY-MM-DD>-<engine>-<model>
```

| проба | что проверяет | счёт | порог по умолчанию |
|---|---|---|---|
| needle | 10 пар «ключ → 6 цифр» на глубинах 5%…95%; спрашиваются все | доля точных ответов; по каждой глубине — в `details.items` | 0.8 — равен порогу K1 (`EXPERIMENTS.md:54@393cbc1`) |
| multihop | 4 спрашиваемые цепочки `VAR_X = 12345`, `VAR_Y = VAR_X` … (4 перехода) и 4 цепочки-отвлекающие | доля верных конечных значений | 0.75 — PROVISIONAL (предложение perf-bench, утверждает lead) |
| codeqa | синтетический код; 4 функции возвращают константы | 0.5 за константу + 0.5 за файл | 0.75 — PROVISIONAL (то же) |
| json | объект по схеме из 5 полей; без и с `response_format: json_object` | доля проверок (разбор, ключи, типы, значения) | 1.0; принят ли `response_format`, записывается отдельно |
| tools | 3 схемы функций, нужен один вызов; затем результат инструмента и проверка, что ответ его использует | 0.5 вызов + 0.5 использование | 1.0 |
| determinism | один промпт дважды при temperature 0 | совпадение | только информация (не «hard»); расхождение записывается вместе с позицией |

Пороги меняются флагом `--threshold needle=0.9,...`; фактические пороги записываются в `meta.thresholds`.
Оценивается только `content` (не рассуждение). Литеральные ответы лежат в `raw.jsonl` (`result.content`).

**PASS «262K реально используемого контекста»** (флаг `--ctx-size 262144` PASS не даёт — ADR-012 D1,
`docs/reassessment/ADR-012-reassessment.md:39-41@393cbc1`). Требуется всё перечисленное ниже на одном и том же запуске сервера, с той же моделью и квантом.
Состав критерия — PROVISIONAL, предложение perf-bench; утверждает lead-architect:

1. `probes.py --contexts 262144 --length-mode total`: needle, multihop и codeqa — PASS по порогам выше.
   В `summary.json` `prompt_tokens` каждой пробы ≥ 262144 − `--answer-tokens` − 16, то есть сервер не обрезал промпт.
2. `run_matrix.py --inputs 262144 --length-mode total --outputs 512,2048 --states cold,warm --concurrency 1 --repeats 3`:
   все запросы ok, `short` = 0, декод p50 ≥ 50 tok/s (пол заказчика; желательно 80–100+,
   `ADR-012-reassessment.md:11-12@393cbc1`).
3. Кэш префиксов: warm TTFT p50 ≤ 10% cold TTFT p50 на той же длине (порог B3, `EXPERIMENTS.md:38@393cbc1`).
   Agent loop до ≥ 100K токенов прошёл без отказов, TTFT по ходам приложен.
4. Пробы `tools` и `json` — PASS (агентное использование через API).

Длины 32K и 128K нужны для кривой «качество по глубине и длине» (K1) и для отношения декода 262K/2K (K4).

## 6. Шаблоны запуска движков как OpenAI-сервера — NOT RUN

Ни одна команда ниже не запускалась. Это шаблоны с подстановками `<...>`. Флаги нужно сверить с `--help` той версии,
которая стоит на сервере, а итоговую команду целиком записать в предрегистрацию и в `--engine-desc`.

**llama.cpp `llama-server`** (флаги — по `tools/server/README.md@ec7630a`: `-c` :50, `-fa` :56, `-ctk` :71, `-ot` :80,
`-ncmoe` :82, `-ngl` :84, `-sm` :85, `-ts` :86, `-cram` :167, `-np` :176, `--cache-reuse` :221, `--metrics` :222,
`--jinja` :231, `--reasoning-format` :232, `--spec-type` :271; на других версиях флаги отличаются):
```bash
llama-server -m <GGUF_первая_часть> --host 0.0.0.0 --port <PORT> \
  -c <CTX> -np <SLOTS> -ngl all -sm <layer|row> -ts <T0,T1,T2,T3> -fa on \
  -ctk <f16|q8_0> -ctv <f16|q8_0> <-ncmoe N | -ot "<regex>=CPU"> \
  --jinja --reasoning-format <deepseek|none> --metrics [--spec-type <...>]
```
`--cache-reuse` для гибридных моделей шорт-листа не ставится: на пине 79e2e74 для qwen4exp его отключает сервер (IMRoPE),
для glm5-next он сдвигает KV и всё равно пересчитывает промпт (N7, находка 5; подробнее — `DEPLOY.md`, раздел про unit движка).
Новее пина (CONFIRMED ведущим: коммиты `d6cf9acb2`, `c811cb8f0`, master `10a60cf30`): `--moe-cache-mib N` — GPU-кэш экспертов,
лежащих в RAM (LRU, подгрузка по PCIe при промахе, несколько GPU) — включать в матрицу для ярусных моделей (`D1-moe-cache-policy.md`);
`--slot-save-path` + `/slots/{id}?action=save|restore` — для K3 (`tools/kv/slot_persist.py`).
Как `-c` делится между слотами при `-np > 1` — проверить в своей версии (см. `--kv-unified-per-slot` в README :164) — UNKNOWN.
`/tokenize` есть (CONFIRMED выше), поэтому подгонка будет точной.

**ik_llama.cpp `llama-server`**: та же семья команд, свои флаги (`-ot`, `-ncmoe`, `-muge`, режимы разбиения —
`docs/reassessment/research/R2-engines.md:116@393cbc1`):
```bash
<ik_llama.cpp>/build/bin/llama-server -m <GGUF> --host 0.0.0.0 --port <PORT> -c <CTX> -ngl <...> \
  <IK_FLAGS: разбиение, -ot/-ncmoe, MLA/FA, KV dtype> --jinja
```
Совпадают ли `/tokenize` и ключи `ignore_eos`/`n_predict` с mainline — UNKNOWN; harness определит это сам и запишет в `meta`.

**vLLM** (`vllm serve`; поля API сверены на `bc21cba`, флаги CLI — PROVISIONAL, по памяти документации, сверить с `vllm serve --help`):
```bash
[NCCL_P2P_DISABLE=<0|1>] vllm serve <HF_ID_или_путь> --served-model-name <NAME> --host 0.0.0.0 --port <PORT> \
  --tensor-parallel-size 4 [--pipeline-parallel-size <...>] --max-model-len <CTX> \
  --gpu-memory-utilization <0.90> --kv-cache-dtype <auto|fp8> --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser <PARSER> --reasoning-parser <PARSER> [--speculative-config '<JSON>']
```
Без auto-tools vLLM на пине для `tool_choice: auto` возвращает обычное сообщение без `tool_calls`
(`vllm/entrypoints/openai/chat_completion/serving.py:1028@bc21cba`, непотоковая ветка; потоковая не проверялась). Тогда проба `tools`
провалится, а agent loop всё равно измерит TTFT. Для GLM-5.3-Flash на sm_89 есть блокер DeepGEMM (`R2-engines.md:118@393cbc1`).

**SGLang** (флаги PROVISIONAL, сверить с `python -m sglang.launch_server --help`):
```bash
python -m sglang.launch_server --model-path <PATH> --served-model-name <NAME> --host 0.0.0.0 --port <PORT> \
  --tp 4 --context-length <CTX> --mem-fraction-static <0.85> [--kv-cache-dtype <...>] \
  --tool-call-parser <PARSER> --reasoning-parser <PARSER>
```
Кэш префиксов (radix) по умолчанию включён — PROVISIONAL. `/tokenize` в формате, который понимает harness, — UNKNOWN; при отказе
включится калибровка по `usage` (запишется в `meta.token_counter`). Поддержка `ignore_eos`/`min_tokens` — UNKNOWN, определится автоматически.

**KTransformers**: инференс идёт через SGLang (kt-kernel) (`R2-engines.md:67@393cbc1`). Команда — по документации версии на сервере:
`<KT_SGLANG_LAUNCH> --port <PORT> ...` — UNKNOWN. Документированный путь для GLM-5.3 требует AVX-512 (`R2-engines.md:40-41@393cbc1`),
а у EPYC 7H43 его нет (`AGENTS.md`).

**ExLlamaV3 через TabbyAPI** (OpenAI-совместимый сервер; лицензия AGPL-3.0 — `R2-engines.md:72@393cbc1`): модель, `max_seq_len`,
режим кэша и разбиение по GPU задаются в `config.yml`, запуск — скриптом из README TabbyAPI. Точные ключи и команда — UNKNOWN, сверить с версией.

**Strata (вилка)**: сервер `serve/server.py` (`docs/strata-map.md:32@393cbc1`; шаблон запуска в `docs/reassessment/BRINGUP-qwen38fn.md:30-31@393cbc1`).
Совместимость с OpenAI (SSE, `usage`, `/v1/models`) — UNKNOWN. Перед матрицей проверить так:
`python3 tools/bench/prompts.py --url <URL> --targets 2048 --verify`. Если `usage` не приходит, prefill tok/s будет `n/a`, а не выдуманным числом.

Перед любым прогоном — дымовой тест на 2K:
`run_matrix.py --inputs 2048 --outputs 128 --states cold,warm --concurrency 1 --repeats 1 --scenarios matrix`.
Посмотреть `meta.eos`, `meta.token_counter`, `stream_options_supported`, колонку `short`.

## 7. Как результат становится CONFIRMED

1. Каталог прогона копируется в `docs/evidence/` **целиком и без правок**: `meta.json`, `raw.jsonl`, `summary.json`, `summary.md`,
   `metrics_before.txt`/`metrics_after.txt`, если они есть. Путь: `docs/evidence/R1-matrix/<дата>-<движок>-<модель>/` (ROADMAP R1.5,
   `docs/reassessment/ROADMAP.md:28@393cbc1`), для проб — `docs/evidence/R1-probes/...`.
2. Рядом лежат предрегистрация (ссылка на строку `EXPERIMENTS.md`) и литеральный вывод команды запуска сервера (первые строки лога загрузки).
3. В отчёте число цитируется как `docs/evidence/.../summary.json` + ключ (например `cells[3].metrics.decode_tok_s.p50`) и SHA коммита,
   в котором лежит каталог. Число без каталога недействительно.
4. evidence-auditor сверяет: SHA репозитория в `meta.json` = коммит инструментов; `is_mock_server` = false; `status` = `ok`
   (или отказы перечислены); `raw_rows` = числу строк `raw.jsonl`; пересчитанные из `raw.jsonl` p50 совпадают со `summary.json`.
5. Отказы (`ok: false`) остаются в данных. Удалять строки или перезапускать «до хорошего числа» без записи — нарушение протокола.
   Повторный прогон — это новый каталог.

## 8. Что не проверено (ограничения)

- Против настоящего llama.cpp `llama-server` (пин R1 `79e2e74`, сборка на CPU, крошечная модель со словарём для текста) —
  прогнано в N11 (CONFIRMED, `docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md`): диалект `/tokenize` llama.cpp
  определяется, vLLM-тело даёт `{"tokens": []}` (промах, как задумано); `prompts.py`, `run_matrix.py` (матрица и agent loop),
  `probes.py`, `logprob_compare.py`, шлюз, `slot_persist.py` и `run_r1.py` с pack работают; найдены и исправлены формат
  logprobs llama-server в `logprob_compare`, повтор «холодных» промптов между прогонами, отсутствующий каталог
  `artifacts/quality/` в `run_r1`. Длину ответа на llama.cpp задаёт `ignore_eos`, `min_tokens` сервер молча игнорирует.
  Числа там — время крошечной модели на CPU, не замер. Против ik_llama.cpp, vLLM, SGLang, TabbyAPI и Strata — NOT RUN;
  на GPU, на 262K и с настоящими шаблонами чата — NOT RUN.
- HTTPS-путь клиента написан, но не тестировался (мок только HTTP).
- Погрешность формулы декода при неравных многотокенных чанках (спекуляция) не оценена на реальном движке — PROVISIONAL.
- Мок-сервер исполняет запросы параллельно без батчинга. Агрегатная пропускная способность на моке ничего не говорит о реальном планировщике.
- Скорость подгонки промпта через `/tokenize` реального сервера на 262K (≈ 1 МБ текста, 5–8 вызовов) — UNKNOWN; на моке 262K подгоняется за ≈ 0.7 с.
