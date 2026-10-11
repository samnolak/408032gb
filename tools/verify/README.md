# tools/verify — сборщик verification pack (ROADMAP R6.1)

Автор: parity-qa, 2026-10-09. Ярлыки — по `AGENTS.md`: CONFIRMED (с источником), PROVISIONAL (как получено),
UNKNOWN, NOT RUN.

**Состояние.** Сборщик проверен только на синтетических артефактах (`tests/test_verify_pack.py`; артефакты делают сами
инструменты, в том числе на mock-сервере). На сервере 4×4080 pack ни разу не собирался — **NOT RUN**.

## Что это

`tools/verify/pack.py` (только stdlib; обёртка `scripts/verify_pack.sh`) берёт каталог, куда оператор сложил выводы
инструментов замеров, копирует их в один каталог-пакет, считает sha256 каждого файла и выносит вердикт по каждому
критерию приёмки заказчика: **PASS / FAIL / NOT RUN**, с измеренным значением, порогом и путём к файлу-доказательству.
Состав критериев V1–V11 и правила вердикта описаны в `docs/reassessment/VERIFICATION_PACK.md` (infra-ops, d89e633);
у каждого критерия сборщика в `verdict.json` есть поле `v` — номер строки этой таблицы.

Жёсткие правила (покрыты `tests/test_verify_pack.py`; 10 одноточечных мутаций `pack.py` пойманы тестами 2026-10-09 —
литеральный вывод повторного прогона ведущим — `docs/evidence/2026-10-09-tool-mutations.md`, раздел «tools/verify/pack.py»):

- нет артефакта, он не разбирается или его сводка не совпадает с сырыми данными (`summary.json` пересчитывается из
  `raw.jsonl` формулами `tools/bench/runlog.py`) — **NOT RUN, никогда не PASS**;
- артефакт, снятый на mock-сервере (`meta.is_mock_server`, `owned_by: "408032gb-bench-mock"` в `/v1/models`, плашка
  «MOCK SERVER RUN» в `summary.md`), делает каждый критерий, который на него опирается, **NOT RUN с причиной `mock`**;
  выводы `logprob_compare.py` и `slot_persist.py`, снятые с URL mock-прогона, — тоже `mock`;
- декод засчитывается **только при заполненном контексте** (раздел «Заполненный контекст» ниже); ячейки с коротким
  контекстом в зачёт не идут, они только перечисляются в `details`;
- итог **PASS** — только если каждый обязательный критерий PASS и каждый входной артефакт разобран; FAIL — если хоть
  один обязательный FAIL; иначе NOT RUN.

## Запуск

```bash
scripts/verify_pack.sh --runs <каталог с выводами> --out docs/evidence/verification-pack-<YYYY-MM-DD>   # сборка
scripts/verify_pack.sh --check docs/evidence/verification-pack-<YYYY-MM-DD>                           # проверка целостности
```

Дополнительно: `--criteria FILE` (JSON с порогами, по умолчанию `tools/verify/criteria.default.json`; неизвестный ключ —
ошибка), `--hardlink` (жёсткие ссылки вместо копий, при ошибке — копия). Каталог `--out` не должен существовать или должен
быть пустым: пакет никогда не перезаписывается.

Коды выхода: сборка — 0 PASS, 1 FAIL, 3 NOT RUN, 2 ошибка аргументов/ввода-вывода; `--check` — 0 всё совпало,
1 файл изменён / пропал / лишний / строка манифеста испорчена, 2 нет `MANIFEST.sha256`.

## Что лежит в пакете

| файл | содержимое |
|---|---|
| `artifacts/` | побайтовые копии всего, что было в `--runs` (структура каталогов сохраняется) |
| `criteria.json` | действующие пороги (умолчания + `--criteria`) |
| `env.txt` | литеральный вывод `git rev-parse HEAD`, `git status --short`, `uname -a`, версия Python, `nvidia-smi -L` (или `nvidia-smi: not present`) на машине, где собирался пакет. Окружение самого сервера — в `meta.json` каждого прогона и в архиве диагностики |
| `verdict.json` | по критерию: `v` (V1–V11), `status`, `reason_code`, `reason`, `measured`, `value`, `threshold`, `evidence` (пути внутри пакета), `details`; плюс итог, список артефактов, неразобранные файлы, пометки SYNTHETIC, отличия критериев от умолчаний, sha256 самого `pack.py` |
| `VERDICT.md` | то же по-русски: таблица «V / критерий / обязательный / статус / измерено / порог / доказательство / причина» |
| `MANIFEST.sha256` | sha256 каждого файла пакета (формат `sha256sum`, пути от корня пакета); `--check` ловит изменённый, пропавший и лишний файл |

`--check` проверяет только целостность. Защита от подмены самого манифеста — коммит пакета в git (sha256 манифеста
печатается при `--check`).

**Синтетический прогон (N24).** `"synthetic_run": true` в файле `--criteria` объявляет прогон на синтетической
(случайной) модели. Если в `model_desc` каждого прогона `run_matrix`/`probes` есть положительная пометка SYNTHETIC (заглавными, целым
словом, не после «non-»/«not »: «non-synthetic», «NOT SYNTHETIC», «synthetic» строчными не считаются), критерии из
`synthetic_not_applicable` (допустимы только оценивающие ответы модели: `agent_api_probes`, `context_probes_full_length`,
`manual_quality_eval`) из PASS или FAIL по порогу становятся NOT RUN с причиной `not_applicable_synthetic`; исходный
вердикт сохраняется (`details.synthetic_raw`) и перечисляется в `verdict.json` (`synthetic_run.not_applicable`) и
плашкой в `VERDICT.md`. Ошибка запроса пробы остаётся FAIL; отказ сервера принять `response_format` (HTTP 400/422,
`response_format_accepted: false` у пробы json) — тоже ошибка запроса: FAIL при любом балле и при любом объявлении. Итог такого прогона никогда не PASS. Если хотя бы один
прогон не помечен SYNTHETIC, объявление записывается, но не применяется.

Если критерии отличаются от `criteria.default.json`, в `VERDICT.md` стоит плашка «вердикт НЕ относится к требованиям
заказчика в их исходном виде», а в `verdict.json` — `criteria_is_default: false` и список отличий. Если в путях или в
`--engine-desc`/`--model-desc` есть слово SYNTHETIC — плашка «тестовые данные, а не доказательство».

## Как оператор готовит входы

Всё складывается в один каталог (ниже `runs/`); раскладка свободная, тип артефакта определяется по содержимому. Каталоги
прогонов `tools/bench` копируются **целиком и без правок** (`BENCH_PROTOCOL.md`, раздел 7) и **не переименовываются**:
`vs_roofline.py` ссылается на прогон по имени каталога. Все прогоны — на одном запуске сервера, с одной моделью и
квантом, с одинаковыми `--engine-desc` и `--model-desc` (в `--model-desc` указать имя первого файла GGUF — так пакет
связывает аудит битности с обслуживаемой моделью).

| вход | команда (из корня репозитория; флаги — по `--help` инструмента и `docs/reassessment/BENCH_PROTOCOL.md`) |
|---|---|
| матрица скорости + кэш префиксов + agent loop | `python3 tools/bench/run_matrix.py --url http://127.0.0.1:<PORT> --inputs 262144 --length-mode total --outputs 512,2048 --states cold,warm --concurrency 1 --repeats 3 --scenarios matrix,agent --agent-target-tokens 131072 --seed 0 --engine-desc "<движок @ commit; команда сервера>" --model-desc "<модель, квант, файлы GGUF>" --out-dir runs/matrix/<YYYY-MM-DD>-<engine>-<model>-262k` (длины 2K/32K/128K — отдельными прогонами, в зачёт декода они не идут) |
| пробы длинного контекста и агентного API | `python3 tools/bench/probes.py --url http://127.0.0.1:<PORT> --contexts 32768,131072,262144 --length-mode total --answer-tokens 1024 --seed 0 --engine-desc "<...>" --model-desc "<...>" --out-dir runs/probes/<YYYY-MM-DD>-<engine>-<model>` |
| целостность весов | `python3 scripts/offline/weights.py verify <манифест ревизии>.json --dest <каталог весов> > runs/weights_verify.txt` и сам манифест (`weights.py manifest ... -o`) в `runs/` — нужны оба файла; пакет сверяет каждую строку `OK` с sha256 и размером из манифеста |
| аудит битности | `python3 tools/gguf_quant_audit.py <первый файл GGUF, -00001-of-0000N.gguf> --json > runs/quant_audit.json` — с порогами по умолчанию и **без** `--dense-warn-only` (иначе критерий NOT RUN или FAIL) |
| точная маршрутизация | `python3 tools/quality/logprob_compare.py --ref http://<эталон> --cand http://<проверяемый движок> --corpus <корпус>/*.txt --prefixes 200 --top-k 20 --out runs/logprob_compare.json`; эталон — llama.cpp на **том же** GGUF со всеми экспертами (без пропуска), кандидат — развёрнутый движок |
| сохранение/восстановление KV | сервер llama-server с `--slot-save-path <каталог на NVMe>` и `-np 1`; `python3 tools/kv/slot_persist.py --url http://127.0.0.1:<PORT> --prompt-file <текст на ≥ 100K токенов> --gen 64 --repeats 3 --out runs/slot_persist.json` |
| сравнение с roofline | `python3 tools/bench/vs_roofline.py runs/matrix/<прогон 262K> --spec models/specs/<модель>.json --expert-bpw <bpw> --dense-bpw <bpw> --hit <доля попаданий> --strategy <pp\|ep\|tp> --overlap <yes\|no> --spec-k <k> --spec-accept <доля> --json > runs/vs_roofline.json` — режим движка как он работал (llama.cpp: `--overlap no`; спекуляция — её k и измеренное принятие); с 2026-10-10 входы границы пишутся в поле `inputs` каждой строки, команду всё равно положить рядом в `runs/vs_roofline.cmd.txt` |
| факты о железе | по `docs/reassessment/HW_DIAG_RUNBOOK.md`: `sudo bash scripts/diag/collect.sh`, затем `bash scripts/diag/bench_all.sh --run --out diag-out/<каталог> --mem --gpu --vram-check --nvme-all /mnt/a/d,... --size-gib 64`, затем `python3 scripts/diag/summarize.py diag-out/<каталог> --json runs/hw/<host>.facts.json`; на хосте Proxmox и в VM — по файлу; архивы `diag-out/*.tar.gz` можно положить в `runs/hw/` (копируются и хэшируются, но не разбираются) |

Текстовый вывод `gguf_quant_audit.py` и `vs_roofline.py` (без `--json`) и `HARDWARE_FACTS.draft.md` пакет не разбирает:
нужны JSON-выводы.

## Критерии и условия PASS

Пороги — `criteria.default.json`; у каждого порога в `_notes` указан источник и ярлык.

| id | V | обяз. | вход | PASS, если |
|---|---|---|---|---|
| `single_configuration` | — | да | `meta.json` всех прогонов | у всех прогонов одинаковы `model`, `engine_desc`, `model_desc` (пустые описания — NOT RUN) |
| `openai_api` | — | да | все прогоны | `GET /v1/models` = 200, есть успешные потоковые запросы, `usage` есть в каждом успешном ответе |
| `context_probes_full_length` | V3 | да | `probes.py` | needle ≥ 0.8, multihop ≥ 0.75, codeqa ≥ 0.75 на контексте ≥ 262 144, и `prompt_tokens` ≥ 262 144 − `--answer-tokens` − 16 (сервер не обрезал промпт); оценка пересчитывается по порогам критериев, а не по `--threshold` прогона |
| `output_after_full_prompt` | V3 | да | `probes.py`, `run_matrix.py` | каждый запрос с промптом на полное окно завершился и выдал ≥ 1 токен с непустым текстом |
| `agent_api_probes` | V3 | да | `probes.py` | `tools` = 1.0 и `json` = 1.0 (все попытки JSON валидны) |
| `decode_filled_ctx_floor` | V4 | да | `run_matrix.py` | в каждой single-stream ячейке при заполненном контексте с выводом ≥ 512: отказов 0, `short` 0, повторов ≥ 3, декод p50 ≥ 50 tok/s |
| `decode_filled_ctx_target` | V4 | нет | то же | то же с порогом 80 tok/s |
| `prefix_cache` | V5 | да | `run_matrix.py` | на заполненном контексте warm TTFT p50 ≤ 10% cold TTFT p50, сервер сообщает `cached_tokens` и они > 0 |
| `agent_loop` | V5 | да | `run_matrix.py --scenarios agent` | ни одного отказа и достигнуто ≥ 100 000 токенов промпта |
| `quant_min_bpw` | V1 | да | `gguf_quant_audit.py --json` | ни одного типа < 4 bpw в гистограммах классов (независимая проверка по таблице типов), нет находок FAIL/WARN, нет неизвестных типов; аудит запущен с порогами ≥ 4.0 |
| `weights_integrity` | V2 | да | `weights.py verify` + манифест | ни одной строки `MISSING`/`SIZE`/`SHA256`, `verify: N/N`, каждый файл манифеста с sha256 проверен с тем же sha256 и размером, ревизия — 40-hex коммит |
| `exact_routing_logprobs` | V6 | да | `logprob_compare.py` | ≥ 200 префиксов без ошибок, `ref` ≠ `cand`, top-1 ≥ 0.97 (порог PROVISIONAL, обоснование в `_notes`); расхождение top-1 считается совпадением, только если оба жадных токена есть в обоих top-K и `ref_gap + cand_gap` ≤ `logprob_tie_tolerance_nats` (0.025, PROVISIONAL, N24; 0 — строго); файлы без этих полей (до N24) — строго |
| `kv_slot_persist` | V7 | да | `slot_persist.py` | ≥ 3 повтора без ошибок, продолжение после восстановления совпадает, `n_written` > 0, холодный промпт ≥ 100K токенов, повторно обработано ≤ 1% промпта, восстановление + тёплый префилл ≤ 25% холодного префилла |
| `decode_vs_roofline` | V9 | да | `vs_roofline.py --json` + матрица | для каждой ячейки заполненного контекста есть строка, её число совпадает с `summary.json`, measured/bound ≤ 1. Отношение > 1 — NOT RUN (`invalid`: входы границы неверны, `vs_roofline.py` предписывает исправить входы), а не FAIL: по V9 само отношение информационное, обязателен действительный файл |
| `hardware_facts` | V8 | да | `summarize.py --json` | `synthetic` = false; измерены (не UNKNOWN) число GPU, ширина PCIe, AVX2/AVX-512, каналы памяти, полоса RAM, P2P, H2D, fio; число GPU = 4. Красные флаги и сравнение hostname диагностики с hostname клиента стенда перечисляются в `details`, но не оцениваются |
| `determinism` | — | нет | `probes.py` | два одинаковых ответа при temperature 0 (информационно) |
| `manual_operational_faults` | V10 | нет | — | всегда NOT RUN: отказы F1–F8 оценивает человек по логам (`DEPLOY.md` §6) |
| `manual_quality_eval` | V11 | нет | — | всегда NOT RUN: ручная проверка |

### Заполненный контекст

По умолчанию (`decode_filled_context_basis: "prompt_plus_output"`) ячейка считается «при заполненном контексте», если её
целевой контекст ≥ `ctx_tokens` (вход при `--length-mode total`, вход + вывод при `prompt`), вывод ≥ 512, а измеренные
сервером `usage.prompt_tokens` + вывод ≥ 262 144 − 16. Это определение `EXPERIMENTS.md:19@55db75d` («промпт ≥ 262 144 − N,
затем вывод N ≥ 512») и `BENCH_PROTOCOL.md:128@55db75d` (`--length-mode total`). Значение `"prompt"` требует, чтобы промпт
один достигал `ctx_tokens`; модель с нативным окном ровно 262 144 так не пройдёт без YaRN (`EXPERIMENTS.md:65@55db75d`).
Ведущий оставил по умолчанию `"prompt_plus_output"` (PROVISIONAL; `docs/reassessment/VERIFICATION_PACK.md`); какое прочтение
принимать для сдачи — решает заказчик.

## Что пакет не проверяет автоматически (ручная проверка)

- оценочный набор заказчика R2.2 (`docs/reassessment/R2-EVAL-SET.md`) и субъективное качество ответов (V11);
- эксплуатационные отказы F1–F8 (V10, `docs/reassessment/DEPLOY.md` §6);
- что сводка диагностики снята именно на этом сервере (V8): hostname сравнивается с hostname клиента стенда только информационно;
- что `ref` в `logprob_compare` — действительно эталон со всеми экспертами на тех же весах: в выводе есть только URL и id
  моделей;
- что аудит битности относится к обслуживаемой модели: связь только через имя файла в `--model-desc`
  (предупреждение в `single_configuration.details.warning`);
- что параметры границы roofline (поле `inputs` в JSON `vs_roofline.py`: spec, hit, strategy, overlap, спекуляция, bpw) соответствуют
  движку и модели — сборщик их не сверяет;
- красные флаги диагностики железа (`hardware_facts.details.red_flags`);
- что все прогоны сделаны на одном запуске сервера (проверяется только совпадение модели и описаний);
- предрегистрация в `EXPERIMENTS.md` и литеральный лог запуска сервера (`BENCH_PROTOCOL.md`, раздел 7) — их сверяет
  evidence-auditor.

## Ограничения форматов существующих инструментов (инструменты не менялись)

- `tools/quality/logprob_compare.py` и `tools/kv/slot_persist.py` не пишут признак mock, SHA репозитория, время и описание
  движка; mock распознаётся только по совпадению URL с mock-прогоном `tools/bench` в том же пакете (у mock-сервера нет
  `logprobs` и `/slots`, поэтому против него эти инструменты дают только ошибки, и критерий не может стать PASS).
- `tools/bench/vs_roofline.py --json` до 2026-10-10 не записывал входы границы; теперь они в поле `inputs` (сборщик их только копирует).
- `scripts/diag/summarize.py` берёт признак `synthetic` из `manifest.json` диагностики; другого признака тестовых данных нет.
- `tools/gguf_quant_audit.py --json` хранит находки только относительно порогов своего запуска; поэтому пакет дополнительно
  проверяет гистограммы типов по классам.
- `tools/bench/probes.py` перезаписывает `pass` у проб `json` и `determinism` уже после того, как строка `probe_result`
  записана в `raw.jsonl` (`tools/bench/probes.py:358-360,429-431@55db75d`), поэтому `pass` в `summary.json` и `raw.jsonl`
  может законно различаться. Пакет `pass` не сверяет и пересчитывает вердикт по `score` и своим порогам.
- `scripts/offline/weights.py verify` пишет только текст в stdout (JSON нет); пакет разбирает строки `OK`/`OK-SIZE`/`MISSING`/
  `SIZE`/`SHA256` и итог `verify: N/M files OK` (`scripts/offline/weights.py:78-101@55db75d`); сохранять вывод нужно в `.txt`.
