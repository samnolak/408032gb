# Промпт: ТЗ «GLM-5.3-Flash для Strata» с привязкой к репозиторию

**Статус:** действует с 2026-10-03 (ADR-009). ТЗ исполняется на базе B — форке lighttransport/Strata с нативным
GGUF-бэкендом GLM. Прежний конфликт «ТЗ против ADR-007» снят: база A (strata-glm, NVFP4) остаётся отдельным треком.

**РОУТИНГ: Sol.** Ведёт lead-architect; kernel-engineer, engine-integrator — Sol; parity-qa, model-analyst,
perf-bench, infra-ops — Terra; evidence-auditor — Luna.

Сначала прочитай AGENTS.md, docs/PLAN.md, docs/DECISIONS.md (ADR-001…009), docs/evidence/*,
patches/strata-glm/README.md. Источник истины — репозиторий. Разделы 1–3 привязывают ТЗ (раздел 4) к репозиторию
и имеют приоритет там, где они расходятся.

## 1. Привязка

1. База для ТЗ — lighttransport/Strata, ветка glm53f @e486a95 (MIT, ADR-009). «Backend glm5next» в ТЗ — это её
   `strata-glm-decode`: `src/program/glm_decode.cpp`, `src/kernels/cuda/glm*.cu`, `include/strata/kernels/glm*.hpp`.
   База A (sergqwer/strata-glm @ed37419, ADR-007) — отдельный трек; её патчи и CI не трогать.
2. Upstream не правим: изменения базы B — только патчами в `patches/lt-strata/` (по образцу `patches/strata-glm/`).
3. Железо: 4× RTX 4080 Super (sm_89), только CUDA. AMD HIP/Vulkan вне объёма: не трогать и не ломать сборку.
4. Форк уже заявляет сделанными этапы 1, 2, 4, 5 (частично), 7 ТЗ (docs/GLM53_FLASH.md форка). Их не переписывать, а проверять:
   каждое «сделано» либо подтверждается нашим тестом (CONFIRMED), либо остаётся заявлением автора (PROVISIONAL).
   Этап 3 ТЗ (совпадение логитов с эталоном) автором не достигнут — он сам пишет, что сквозного совпадения с llama.cpp нет
   (отн. L1 0.05923, макс. |Δlogit| 0.84482 на одном префиксе; docs/GLM53_FLASH.md:141-154 форка). Это главная работа.
   Этапы 8 (DFlash2) и 9 (vision) не начинать без решения заказчика.
5. Эталоны: llama.cpp @ec7630a, граф glm5-next, прогон `-fa off -ctk f32 -ctv f32`; numpy-эталоны в `ref/`. Форк читает GGUF
   с именем архитектуры `glm5next`, llama.cpp на пине — `glm5-next`: тензоры совместимы по словам автора форка (PROVISIONAL),
   проверить. При ничьих в top-k индексера допустим любой корректный выбор (evidence 2026-10-02-g1-parity.md, п. 5).
6. GPU сейчас нет (ADR-006). Выполнимы только:
   а) docs/GLM53_PORT_STATUS.md: таблица «пункт Definition of Done из ТЗ → статус → evidence»;
   б) CI-job `lt-strata-sm89`: сборка базы B под sm_89 и тесты `gguf_reader_test`, `model_test` — сделано, CONFIRMED (check-run 111158460817);
      осталось добавить в него `tools/test_glm_*.py` и `serve/test_glm.py`;
   в) открыть крошечную GGUF из tools/tinygen через `strata-model-inspect`: принимает ли форк геометрию, отличную от 4096/2048 (UNKNOWN);
   г) если принимает — сверка CPU-пути форка с эталонами G1; метрики помимо top-1: max/mean abs, relative, cosine, top-k overlap;
   д) тесты семантики: prefill(N)+decode(1) = prefill(N+1), откат состояния KDA и кэша DSA (у форка есть snapshot/replay — проверить
      на длине ≥ 4 окон индексера), изоляция последовательностей;
   е) дизайн-док: все эксперты Q2 в VRAM четырёх GPU (в форке один GPU, эксперты при декоде по умолчанию на CPU; есть статический
      GPU-кэш с бюджетом на слой и `--decode-experts=gpu`). По образцу docs/design/G4-glm-multigpu.md.
   Всё, что требует GPU, помечай NOT RUN и добавляй шагом в scripts/gpu_day.sh.
7. Цифры автора форка (7.3–8.4 tok/s декода, 98–158 tok/s префилла) — чужие замеры на Threadripper 1950X + RTX 5060 Ti:
   в отчётах только с меткой PROVISIONAL и без переноса на наш сервер.
8. DFlash2 (incoai/GLM-5.3-Flash-DFlash2): доступ по ручной заявке, лицензия CC-BY-NC-ND-4.0 — веса не изменять и не
   переквантовывать; блок 8, 7 спекулятивных токенов (PROVISIONAL: карточки моделей на HF). До решения заказчика только паспорт в docs/models/.
9. Правила AGENTS.md сильнее формата отчёта ТЗ: метки CONFIRMED/PROVISIONAL/UNKNOWN, литеральный вывод команд, хеши только из
   вывода git, NOT RUN вместо выдуманного результата. Отчёт — в формате ТЗ плюс вывод `git log --oneline -n 5` и `git status --short`.
10. При противоречии между ТЗ, репозиторием, llama.cpp и чекпойнтом остановись и напиши `CONFLICT REQUIRES DECISION`, как требует ТЗ.

Первое действие: п. 6в (крошечная GGUF через `strata-model-inspect` в CI), затем п. 6а и отчёт.

## 2. Справка по модели и llama.cpp (добавление заказчика от 2026-10-02, сверено)

| утверждение | метка | источник |
|---|---|---|
| 45 слоёв + 1 MTP; KDA на 34 слоях, DSA/MLA на 11 (3, 7, …, 43); слои 0–2 плотные (FF 12288) | CONFIRMED | docs/models/glm-5.3-flash.md |
| 42 MoE-слоя × 288 экспертов = 12 096; top-8 + 1 общий эксперт; FF эксперта 2048 | CONFIRMED | там же |
| маршрутизируемые эксперты ≈ 304.4 млрд параметров (12 096 × 3 × 4096 × 2048), ≈ 95% модели | PROVISIONAL | арифметика из геометрии; «всего ≈ 320 млрд» — описания PR llama.cpp #27773 (320B) и #27754 (321.3B) |
| «активно 17–18 млрд параметров на токен» | UNKNOWN | в репо не считалось; подтвердить расчётом по models/glm-5.3-flash.json |
| поддержка GLM-5.3-Flash влита в llama.cpp 2026-09-30 (коммит 649dcb103, PR #27773) и есть на пине ec7630a | CONFIRMED | `git log -- src/models/glm5-next.cpp` |
| в llama.cpp на пине есть KDA, DSA/MLA, индексер, mHC, MoE, конвертер GGUF (`conversion/glm.py`), vision (`tools/mtmd`) | CONFIRMED (файлы есть); совпадение логитов с Transformers — PROVISIONAL (заявлено в PR) | llama.cpp @ec7630a |
| MTP-графа на пине нет: тензоры NextN сохраняются, граф — отдельный черновой PR #27917 | CONFIRMED | `src/models/glm5-next.cpp:68` @ec7630a (TODO про DECODER_MTP) |
| принятие черновых токенов MTP 0.745 (3474 из 4664), средняя длина 4.13 | PROVISIONAL | замер автора PR #27917, голова в Q4, чужое железо |
| «MTP даёт +15–30% на конфигурациях с CPU-offload» | UNKNOWN, есть обратный замер | в PR #27917: 19.0 и 16.4 tok/s с MTP против 25.0 без него |
| несколько последовательностей в llama.cpp требуют `--kv-unified` | PROVISIONAL | PR #27773, Limitations; учесть в тесте изоляции последовательностей |
| Strata привязан к одной архитектуре (`qwen4exp`), GLM нельзя добавить записью в конфигурации | CONFIRMED | docs/strata-map.md |

Вывод для плана: MTP включать только после собственного замера на нашем железе (G4/G6) и с автоматическим отключением,
если скорость падает. Цифры чужих замеров в отчётах не использовать как свои.

## 3. Порядок порта из добавления заказчика → состояние в базе B

Столбец «состояние» — по документации форка @e486a95; всё, что не проверено нами, PROVISIONAL.

| шаг добавления | состояние |
|---|---|
| 1. Сначала текст, без vision и MTP | совпадает; vision в форке нет |
| 2. Обновить llama.cpp до ревизии с glm5-next | для эталона сделано: пин ec7630a. Пин ggml внутри форка не менять |
| 3. Чтение GLM GGUF и метаданных | есть: разделённые GGUF, проверки метаданных и границ тензоров |
| 4. Отдельный граф KDA → DSA/MLA → mHC → MoE | есть на CUDA (роутер, mHC, KDA, IndexPool, MLA); сквозное совпадение с эталоном не установлено |
| 5–6. Арена экспертов на 12 096 экспертов, профиль и кэш под top-8 | частично: статический GPU-кэш с бюджетом на слой, один GPU; размещение на 4 GPU — наша работа (п. 6е) |
| 7. Общий эксперт считать отдельно | заявлено: общий FFN считается на GPU, пока CPU обрабатывает промахи |
| 8. Состояние KDA и кэш/индексер DSA | есть, плюс проверки snapshot/replay |
| 9. Prefill | есть, батч до 4096 токенов |
| 10. MTP после проверки корректности, затем vision | MTP есть (GPU и CPU); по разделу 2 включать только после своего замера |

## 4. ТЗ заказчика (содержание без изменений; три блока с именами и один список свёрнуты в строку)

# Техническое задание: GLM-5.3-Flash для Strata

## Проект

Добавить в Strata поддержку модели **GLM-5.3-Flash / GLM5-Next** с сохранением основных преимуществ Strata:

- работа с моделью, превышающей объём VRAM;
- mmap весов;
- CPU/RAM/VRAM offload;
- кэширование активных MoE-экспертов;
- asynchronous prefetch;
- поддержка NVIDIA CUDA и AMD HIP/Vulkan, если это возможно в текущей архитектуре;
- OpenAI-compatible API;
- последующее подключение MTP и DFlash2 speculative decoding.

Название backend-а: `glm5next`. Целевая модель: `zai-org/GLM-5.3-Flash`.
Целевой draft для отдельного DFlash2-режима: `incoai/GLM-5.3-Flash-DFlash2`.

Не считать задачу выполненной, если модель только загружается. Обязательны корректные logits, autoregressive decode, cache rollback и тесты против эталонной реализации.

## Исходные предположения

Strata уже содержит runtime, ориентированный на крупную MoE-модель, включая scheduler, mmap/memory mapping, распределение весов между RAM и VRAM, expert cache, prefetch, API server и существующую speculative/MTP-инфраструктуру.

Сначала изучить реальную структуру репозитория, текущую ветку, commit, build system и backend-и. Не предполагать имена файлов и классов.

Перед изменением кода создать файл `docs/GLM53_PORT_STATUS.md`. В нём зафиксировать текущий commit Strata, используемый commit/ветку llama.cpp или vLLM, поддерживаемые форматы, точки интеграции, ограничения текущего backend-а, план и статус каждого этапа.

## Главные ограничения

Запрещено:

1. Писать GLM5-Next «по аналогии с Qwen» без проверки tensor map и reference implementation.
2. Менять существующее поведение Qwen backend-а без regression tests.
3. Сразу начинать с DFlash2.
4. Смешивать MTP и DFlash2 в одном execution path.
5. Считать совпадением только одинаковый выбранный token.
6. Реализовывать multimodal до корректной текстовой модели.
7. Делать необратенный cache update во время speculative decoding.
8. Прятать fallback на CPU или dense execution без сообщения в логах.
9. Добавлять фиктивную поддержку архитектуры только для прохождения проверки имени модели.
10. Удалять или ослаблять существующие Strata memory optimizations.

Обязательные принципы:

- Сначала корректность, потом производительность.
- Сначала текстовый inference, потом vision.
- Сначала обычный target decode, потом MTP, потом DFlash2.
- Каждое архитектурное предположение подтверждать исходным кодом, конфигурацией модели или тестом.
- Все неподтверждённые сведения помечать как TODO или assumption.
- При невозможности использовать конкретный kernel сделать явный fallback и вывести warning.
- Каждая оптимизация должна иметь benchmark до/после.

## Архитектура решения

Разделить систему на два слоя:

```text
Strata runtime
├── scheduler
├── memory manager
├── mmap/storage
├── CPU/GPU offload
├── expert cache
├── prefetch
├── API server
└── speculative execution framework

Model backends
├── qwen_flash
└── glm5next
    ├── model loader
    ├── tensor mapping
    ├── KDA layers
    ├── DSA/MLA layers
    ├── mHC
    ├── MoE/router
    ├── cache management
    ├── optional MTP
    └── optional DFlash2
```

Если в Strata ещё нет полноценного интерфейса backend-а, сначала выделить его без изменения вычислений Qwen.

Минимальный интерфейс:

```cpp
class model_backend {
public:
    virtual ~model_backend() = default;
    virtual model_info info() const = 0;
    virtual void load(const model_path&, memory_manager&, const runtime_config&) = 0;
    virtual void begin_sequence(sequence_id, const sequence_config&) = 0;
    virtual void prefill(sequence_id, span<const token_id>) = 0;
    virtual logits forward_decode(sequence_id, token_id) = 0;
    virtual cache_checkpoint checkpoint(sequence_id) = 0;
    virtual void rollback(sequence_id, const cache_checkpoint&) = 0;
    virtual void commit(sequence_id, const cache_checkpoint&) = 0;
};
```

Добавить:

```cpp
enum class model_arch {
    qwen_flash,
    glm5next,
};
```

Выбор backend-а должен происходить по metadata модели, а не по имени файла. Если архитектура неизвестна, завершать работу с понятной ошибкой.

## Поддержка форматов

### Фаза 1

Поддержать существующий GGUF для `glm5next`. Не добавлять собственный формат весов, пока GGUF backend не заработает.

Нужно реализовать или переиспользовать: GGUF metadata parsing; tensor name mapping; dtype detection; quantized tensor loading; mmap; tensor residency; alignment; shape validation.

На старте поддержать только реально необходимый набор dtype: reference FP16/BF16 или FP8 и выбранный production quantization path. Для неподдерживаемого dtype — явная ошибка, а не молчаливая замена.

## Реализация GLM5-Next

### Конфигурация

Создать `models/glm5next/config.hpp` и `models/glm5next/config.cpp`. Конфигурация должна читаться из metadata/checkpoint, а не быть захардкожена.

Проверять и логировать hidden size, number of layers, типы attention-слоёв, number of experts, experts per token, shared expert, vocabulary size, head dimension, query/key/value heads, RoPE/position parameters, KDA parameters, DSA/MLA parameters, mHC parameters, context length и tokenizer information.

Добавить `validate_glm5next_config(config)` до выделения больших буферов.

### Типы блоков

Не использовать один универсальный attention path. Предусмотреть:

```cpp
forward_kda_layer(...)
forward_dsa_mla_layer(...)
forward_mhc(...)
forward_moe(...)
```

Типы блоков:

```cpp
enum class glm5_block_kind {
    kda,
    dsa_mla,
};
```

Порядок операций сверить с reference implementation, не угадывать.

### KDA

Реализовать persistent state для KDA/linear-attention слоёв:

```cpp
struct kda_state {
    tensor state;
    position_t position;
    dtype_t dtype;
};
```

Требования: отдельный state на sequence, prefill, single-token decode, checkpoint, rollback, commit, memory accounting, CPU/GPU residency. Quantization state добавлять только после reference correctness.

Проверить эквивалентность `prefill(N) + decode(1)` и `prefill(N+1)`.

### DSA/MLA

Реализовать отдельный cache/state:

```cpp
struct dsa_mla_cache {
    tensor keys;
    tensor values;
    tensor compressed_state;
    size_t token_count;
    size_t capacity;
};
```

Обеспечить causal mask, позиционную обработку, prefill, decode, рост контекста, rollback speculative tokens и multi-sequence isolation.

### mHC

Реализовать mHC отдельными функциями без fusion до прохождения тестов:

```cpp
tensor mhc_pre(...);
tensor mhc_post(...);
```

Порядок операций взять из reference implementation и зафиксировать в документации.

### MoE

Добавить GLM5 MoE backend с router, top-k expert selection, routing weights, shared expert, grouped execution, expert output accumulation, CPU/RAM offload, GPU hot cache, prefetch и статистикой по слоям.

Не предполагать число experts и top-k. Читать из checkpoint и валидировать.

В debug mode логировать layer, token, selected expert IDs, routing weights, cache hit/miss, prefetch latency и expert execution latency.

## Memory manager и offload

Сохранить существующий memory manager Strata и добавить типы allocation:

```cpp
enum class allocation_kind {
    model_weight,
    expert_weight,
    shared_expert_weight,
    attention_projection,
    kda_state,
    dsa_cache,
    temporary_activation,
    logits,
    draft_state,
};
```

Отслеживать size, dtype, device, residency, owner layer/expert, last use, prefetch status, eviction status и reference count.

### Expert cache

Добавить GLM-aware expert cache с политиками LRU, frequency, router-probability и hybrid. Shared expert не должен вытесняться обычной политикой, если это ухудшает производительность.

Counters:

```text
expert_cache_hits
expert_cache_misses
expert_prefetch_hits
expert_evictions
expert_load_bytes
expert_load_time_ms
```

### Prefetch

Минимальная схема: router → top-k experts → async prefetch → независимая attention work → ожидание experts → MoE. Эффект подтвердить benchmark-ом.

## Tokenizer и chat template

Поддержать tokenizer из checkpoint. Проверить encode/decode, special tokens, BOS/EOS, Unicode, code, long prompt, chat template, thinking/reasoning markers, stop tokens и batch tokenization.

Chat template не копировать из другой GLM-модели без проверки. Добавить golden test с ожидаемыми token IDs.

## Корректность

Использовать Transformers/vLLM и/или llama.cpp glm5next как reference. Для тестовых последовательностей сохранять input_ids, rendered prompt, hidden states, router scores, selected experts, logits, top-k logits, generated tokens и cache metadata.

Сравнивать max_abs_error, mean_abs_error, relative_error, top-1 agreement, top-k overlap и cosine similarity.

Обязательные тесты:

- logits после prefill;
- logits после single-token decode;
- logits после последовательного decode;
- logits после cache rollback;
- длинный контекст;
- CPU и GPU;
- quantized weights;
- prefill/decode equivalence;
- multi-sequence isolation.

Не считать тест успешным только по top-1 token.

## API и CLI

Добавить выбор backend-а:

```bash
strata --model /models/glm53.gguf --architecture auto --backend cuda
```

Опции:

```text
--glm-expert-cache-size <bytes>
--glm-expert-cache-policy <lru|frequency|hybrid>
--glm-prefetch <auto|on|off>
--glm-kda-state-device <cpu|gpu|auto>
--glm-dsa-cache-device <cpu|gpu|auto>
--glm-mtp <auto|on|off>
--glm-dflash <auto|on|off>
--glm-speculative-tokens <N>
--glm-debug-routing
--glm-debug-cache
```

MTP и DFlash2 взаимно исключаются. При одновременном включении должна быть ошибка конфигурации.

OpenAI-compatible API должен поддерживать `/models`, `/chat/completions` и `/completions`. В capabilities нельзя заявлять vision, MTP или DFlash2, если они реально не включены.

## Спекулятивное декодирование

Порядок реализации:

1. GLM target без speculative decoding.
2. GLM target + native MTP/NextN.
3. Отдельный GLM-5.3-Flash-DFlash2.

### MTP

Добавить proposer с checkpoint до proposal, target verification, rollback rejected tokens, commit accepted tokens, RNG state и acceptance statistics. При низком acceptance rate разрешить fallback на обычный decode.

### DFlash2

DFlash2 считать отдельным block-diffusion proposer, а не обычной autoregressive draft model. Поддержать block proposal, non-causal attention mask, hidden-state interface от target, candidate selector, target verification, rollback и acceptance statistics.

Начать с batch size 1, greedy decoding и проверенного числа speculative tokens. Block size не захардкодить без validation checkpoint-а.

## Vision

Vision не реализовывать в первой версии. Сначала выпустить text-only backend. Если изображение передано в text-only версии, вернуть явную ошибку, а не игнорировать его.

## Производительность

Создать benchmark harness для short/long prompt, short/long decode, cold/warm expert cache, CPU/RAM offload, разных VRAM limits, CUDA, HIP/Vulkan, MTP и DFlash2.

Измерять prompt tokens/sec, decode tokens/sec, first-token latency, p50/p95 token latency, peak VRAM/RAM, page faults, expert cache hit rate, prefetch hit rate и rollback cost.

## Этапы реализации

### Этап 0 — исследование

Изучить Strata, найти backend interface, определить формат модели, изучить reference glm5next, составить tensor map и зафиксировать неизвестные места. Код модели не писать.

### Этап 1 — backend skeleton

Добавить `glm5next_backend`, `glm5next_config`, `glm5next_loader`. Модель определяется по metadata, config валидируется, tensor names перечисляются, memory estimate работает.

### Этап 2 — loader и tokenizer

GGUF загружается, tensor map проверяется, tokenizer работает, chat template golden test проходит.

### Этап 3 — простой CPU/reference forward

Реализовать один слой, KDA, DSA/MLA, mHC, MoE и полный text-only forward. Logits должны совпадать с reference.

### Этап 4 — обычный decode

Добавить prefill, single-token decode, persistent KDA state, DSA/MLA cache, multi-sequence, checkpoint/rollback и greedy generation.

### Этап 5 — интеграция Strata runtime

Добавить mmap, RAM/VRAM offload, expert cache, prefetch, scheduler и API server.

### Этап 6 — GPU kernels

Порядок: generic kernels, CUDA/HIP correctness, KDA optimization, DSA/MLA optimization, MoE grouped kernels, fusion.

### Этап 7 — MTP

Добавить MTP proposal, target verification, rollback, acceptance statistics, benchmark и fallback.

### Этап 8 — DFlash2

Добавить отдельный DFlash2 proposer, block/non-causal path, verification, rollback и benchmark. MTP и DFlash2 не смешивать.

### Этап 9 — vision

Отдельная фаза после text-only release.

## Definition of Done

Задача выполнена только если:

- `glm5next` определяется по metadata;
- GGUF загружается без ручного переименования;
- tokenizer и chat template корректны;
- text-only prefill и decode работают;
- KDA state сохраняется и восстанавливается;
- DSA/MLA cache корректен;
- MoE router и experts корректны;
- logits сопоставлены с reference;
- multi-sequence isolation проходит;
- Strata RAM/VRAM offload работает;
- expert cache имеет counters;
- API не ломает существующих клиентов;
- Qwen regression tests проходят;
- MTP и DFlash2 не смешиваются;
- unsupported functions явно объявляются;
- есть документация, воспроизводимая команда запуска и benchmark до/после;
- нет silent fallback-ов;
- нет TODO в критическом execution path.

Минимальный smoke test:

```bash
strata \
  --model /models/GLM-5.3-Flash.gguf \
  --architecture auto \
  --backend cuda \
  --max-context 4096 \
  --speculative none \
  --prompt "Explain Kotlin coroutines in three paragraphs."
```

## Формат отчёта домашней LLM

После каждого этапа отвечать:

```text
## Status

- Current git commit:
- Completed phase:
- Build status:
- Tests passed:
- Tests failed:
- Known limitations:

## Files changed

- path/to/file.cpp — description
- path/to/file.hpp — description

## Evidence

- exact command:
- relevant output:
- reference commit:
- comparison metrics:

## Next step

- one concrete next action
```

Если обнаружено противоречие между Strata, llama.cpp, vLLM и checkpoint, остановиться и сообщить `CONFLICT REQUIRES DECISION`, перечислив источники конфликта, предложенный вариант, последствия и минимальный тест для разрешения.
