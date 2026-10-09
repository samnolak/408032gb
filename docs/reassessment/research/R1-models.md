# R1. Ландшафт open-weight моделей для сервера 4×RTX 4080 Super 32 GB (по состоянию на 2026-10-09)

Роль: model-quality research lead. Файл только для чтения другими ролями; репозиторий не менялся, коммитов нет.

**Метки.** VERIFIED — я сам видел утверждение в первичном источнике: файл@SHA (GitHub-клон или вендоренный файл репозитория), либо вывод команды.
REPORTED — заявление вендора, агрегатора или сообщества, которое я видел только во фрагменте поисковой выдачи (страница не загружена) или которое никто независимо не перемерял.
PROVISIONAL — моя арифметика по указанным входам. HYPOTHESIS — мой вывод, который нужно проверить. DISPROVED — опровергнуто. UNKNOWN — данных нет.
У бенчмарков вендора метка VERIFIED относится только к факту публикации числа в первичном файле; само число остаётся заявлением вендора.

**Ограничения доступа (важно для доверия к числам).** Из песочницы недоступны huggingface.co, arxiv.org, artificialanalysis.ai, unsloth.ai, openrouter.ai, recipes.vllm.ai и z.ai.
Для huggingface.co и arxiv.org прокси вернул 403 (policy denial); у WebFetch для остальных хостов — `getaddrinfo ENOTFOUND`. Обходить блокировку через зеркала я не стал.
Поэтому HF-карточки и `config.json` в этом отчёте **не прочитаны**, за исключением файлов, которые CI уже вендорил в `third_party/`.
Доступен GitHub: клоны склонированы в песочницу, SHA перечислены в §10. Вендорные README с бенчмарками, рецепты vLLM, исходники llama.cpp, ik_llama.cpp, vLLM, SGLang, transformers, ExLlamaV3, TensorRT-LLM, KTransformers и Strata — **VERIFIED**.
Всё, что взято из поисковых фрагментов (оценки Artificial Analysis, таблицы unsloth, даты, лицензии), — **REPORTED**.

---

## 1. Итог (10 строк)

1. Пространство решений шире трёх моделей репозитория. Есть две модели, которых в плане нет и которые по сумме критериев не хуже: **Qwen3.8-Flash-Next** (125B + 51B n-gram, 6B active, родная модель Strata) и **DeepSeek-V4-Flash-0731** (284B/13B, родные FP4-эксперты).
2. На текущем индексе Artificial Analysis v4.3.x модели, которые физически влезают в этот сервер в 4 битах, почти не различаются (REPORTED): GLM-5.3-Flash **42**, Qwen3.8-Flash-Next **40**, DeepSeek-V4.1-Flash **39**, MiMo-V2.6-Flash **38**. Для сравнения, лучшие open-weight гиганты: MiMo-V2.6-Pro 46, GLM-5.3 45, Kimi K3 44.
3. Только Qwen3.8-Flash-Next целиком (без n-gram-таблицы) помещается в 120 GB VRAM в честном 4-битном виде (UD-Q4_K_XL ≈111 GB вместе с таблицей) с KV на 262K. Её наивный потолок ~160–190 ток/с (PROVISIONAL); сообщество намерило 70 ток/с при 128K на **одной** 32-ГБ карте + RAM (REPORTED).
4. GLM-5.3-Flash в UD-Q4_K_XL (199.7 GB) требует VRAM+RAM. Даже при 100% попаданий в VRAM конвейер из 4 карт упирается в **~60 ток/с** (PROVISIONAL), а при равномерной маршрутизации даёт 34–63 ток/с. Пол в 50 ток/с достижим только при сильном перекосе маршрутизации или со спекуляцией/TP; 80–100 ток/с без TP недостижимы.
5. У DeepSeek-V4-Flash-0731 эксперты в GGUF UD-Q4_K_XL (155 GB) бит-в-бит равны родному FP4, KLD ≈0.01 (REPORTED). Модель влезает в VRAM+RAM с долей ~0.74 в VRAM, потолок 66–112 ток/с (PROVISIONAL). Качество ниже GLM-5.3-Flash.
6. DeepSeek-V4.1-Flash (552B + 196.6B Engram) не влезает в VRAM+RAM даже в родном MXFP4: эксперты ~290–310 GB. Часть экспертов уходит на SSD, и наивный потолок падает до **12–16 ток/с**. При требовании «≥4 бит и ≥50 ток/с» модель как цель этого сервера проваливается, её роль — только эталон качества.
7. На 262K решает тип attention. У моделей с полным GQA/MLA в части слоёв (MiMo, Qwen3.5, Step-3.7, Laguna, Mistral Small 4) чтение KV на токен составляет 3–6 GB в FP8 и съедает 25–50% бюджета полосы. У разреженных и линейных (GLM DSA+KDA, Qwen3.8 QSA+GDN, DeepSeek CSA) — ≤0.4 GB.
8. Проверка утверждений репозитория: геометрия GLM-5.3-Flash (45 слоёв = 34 KDA + 11 DSA, 288×top-8, FF 2048, hidden 4096, 1M), MiMo (309B/15B, 48 слоёв, MXFP4 ≈150 GiB) и DeepSeek-V4.1 (552B + 196.6B Engram, MXFP4) — VERIFIED.
   `nvidia/GLM-5.3-Flash-NVFP4` 204 GB и unsloth UD-Q4_K_XL ≈186 GiB — REPORTED и согласуются. Неточность: строка арки `glm5next` mainline llama.cpp не знает, его понимает только ik_llama.cpp.
9. Рекомендуемый шорт-лист для первых замеров на железе: (1) Qwen3.8-Flash-Next UD-Q4_K_XL, (2) GLM-5.3-Flash UD-Q4_K_XL, (3) DeepSeek-V4-Flash-0731 UD-Q4_K_XL, (4) MiMo-V2.6-Flash-MOPD MXFP4. DeepSeek-V4.1-Flash — вне шорт-листа скорости.
10. Главные неизвестные: независимое качество Qwen3.8-Flash-Next против GLM-5.3-Flash на задачах заказчика в 4 битах и на 128–262K; реальный перекос маршрутизации и пропускная способность RAM (4 или 8 каналов); работают ли P2P и TP на 4 картах; что разрешает лицензия Qwen Community License для сценария заказчика.

---

## 2. Таблица кандидатов

### 2.1 Паспорт (класс a — тиринг VRAM+RAM(+SSD); класс b — целиком в VRAM)

| # | Модель, HF id | Класс | Релиз | Лицензия | Total / active | Слои и attention | Контекст | Спекуляция | Нативная точность |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GLM-5.3-Flash, `zai-org/GLM-5.3-Flash` (+`-BF16`) | a | веса 2026-08-27, анонс 08-26 (REPORTED [50]); в рецептах vLLM с 2026-08-27 (VERIFIED [1]) | MIT (REPORTED [50]); код на GitHub — Apache-2.0 (VERIFIED [10]) | 321B / 18B (VERIFIED [1]; README: «320B-A18B» [10]) | 45 слоёв: 34 KDA (64×128, conv 4) + 11 DSA/MLA без RoPE (`qk_rope_head_dim=0`, kv_lora 512, индексер 32×128, top-2048, k-pool), mHC ×4, 288 экспертов top-8 + 1 shared, FF 2048, hidden 4096, первые 3 слоя плотные (VERIFIED [3][7]) | 1 048 576 заявлено (VERIFIED [1][3]); RoPE в MLA нет, YaRN не нужен (VERIFIED [3]) | 1 MTP-слой, в vLLM MTP-5 (VERIFIED [1]); внешние DFlash2: `incoai/…` (CC-BY-NC-ND), `canada-quant/…-G` (Apache-2.0) (репо [18]) | FP8 (блоки 128) (VERIFIED [10]) |
| 2 | MiMo-V2.6-Flash, `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` (рекомендован) / `…-Flash-RL` | a | 2026-09-21 или 09-22 (REPORTED, конфликт [41]); в рецептах с 2026-09-22 (VERIFIED [1]) | MIT (REPORTED [41]) | 309B / 15B (VERIFIED [1]) | 48 слоёв (1 плотный + 47 MoE), 256 экспертов top-8, SWA-128 + global (VERIFIED [1]); 39 SWA + 9 global, 4 KV-головы global, QK 192 / V 128 — PROVISIONAL (дефолты transformers для V2-Flash [3]; конфиг V2.6 не прочитан) | 1M (VERIFIED [1]) | 5-слойный DFlash-драфтер, 7 токенов, лежит в `dflash/` (VERIFIED [1]) | эксперты хранятся в MXFP4, счёт в FP8; чекпойнт 173 GB (VERIFIED [1]); сторонний источник даёт 165.53 GiB (REPORTED [55]) |
| 3 | DeepSeek-V4.1-Flash, `deepseek-ai/DeepSeek-V4.1-Flash` | a (на деле SSD) | 2026-09-10 или 09-15 (REPORTED [40]); HF `last_modified` 2026-10-01 (VERIFIED [17] PIN.json) | MIT (VERIFIED [17]) | 552B бэкбон + 196.6B Engram; 8B active на префилле / 16B на декоде (VERIFIED [17]) | 40 слоёв CED (20 энкодер + 20 декодер), SWA-128 + CSA2 (сжатие 1 и 2) + двухуровневый индексер (top-512), mHC ×4, 384 эксперта top-6 + 1 shared, FF 2304, hidden 5120, Engram в слоях 1 и 14 (VERIFIED [17]) | 1M через YaRN ×16 поверх 64K (VERIFIED [17]) | DSpark, блок 5, MTP нет (VERIFIED [1]); «golden» длина принятия 3.51 (REPORTED [1]) | MXFP4-эксперты + FP8/MXFP8, масштабы UE8M0; 476 GiB на диске (VERIFIED [1][17]) |
| 4 | DeepSeek-V4-Flash-0731, `deepseek-ai/DeepSeek-V4-Flash-0731` | a | 2026-07-30 или 07-31 (REPORTED [38][39]) | MIT (REPORTED [38]) | 284B / 13B (VERIFIED [1]); 304B с DSpark (REPORTED [39]) | 43 слоя, hidden 4096, 256 экспертов top-6 + 1 shared, FF 2048, CSA+HCA (сжатие 4/128), SWA-128, mHC (PROVISIONAL: дефолты transformers для V4-Flash-Base [3]; рецепт [1]) | 1M (VERIFIED [1]) | MTP-1 и DSpark (VERIFIED [1]) | FP4-эксперты + FP8; ~160–167 GB (VERIFIED [1]) |
| 5 | **Qwen3.8-Flash-Next**, `Qwen/Qwen3.8-Flash-Next` (+`-FP8`) | **b** (n-gram-таблица в RAM/SSD) | 2026-08-26…28 (REPORTED, конфликт) | Qwen Community License 1.0 (название VERIFIED [12] HOW_IT_WORKS; условия REPORTED [47]) | 125B основная часть + 51.2B n-gram; 6B active; ~176–180B на диске (VERIFIED [1]: FP8 172.78 GiB, BF16 335.28 GiB) | 48 слоёв: 36 GDN + 12 QSA (разреженные, k-pool-индексер), gated residual ×4, 512 экспертов × FF 640, top-10 + shared, hidden 2560 (VERIFIED [1][12] layout.hpp) | **262 144 нативно**, до 1M через YaRN (VERIFIED [1]); один запрос на 262K в vLLM не тестировали (VERIFIED [1]) | встроенный MTP (vLLM MTP-3) (VERIFIED [1]); в Strata принимается 52.8–59.9% черновиков (REPORTED [12]) | BF16 и FP8 официально (VERIFIED [1]) |
| 6 | Qwen3.5-397B-A17B, `Qwen/Qwen3.5-397B-A17B` | a | 2026-02-16 (VERIFIED [11]) | Apache-2.0 (REPORTED) | 397B / 17B (VERIFIED [1]) | GDN : full = 3 : 1, 60 слоёв (REPORTED [46]), KV 2×256 (PROVISIONAL [3][46]) | 262K нативно (VERIFIED [1]) | MTP; есть вариант с DFlash (VERIFIED [1]) | BF16; GPTQ-Int4 официально (VERIFIED [1]) |
| 7 | Qwen3.5-122B-A10B, `Qwen/Qwen3.5-122B-A10B` | b | 2026-02-24 (VERIFIED [11]) | Apache-2.0 (REPORTED) | 122B / 10B (VERIFIED [1]) | GDN 3:1, 48 слоёв, 256 экспертов top-8 + shared (VERIFIED [1] про 256; остальное REPORTED [46]) | 262K нативно (VERIFIED [1]) | MTP-1 (VERIFIED [1]) | BF16; FP8 и GPTQ-Int4 официально (VERIFIED [1]) |
| 8 | Nemotron-3-Super-120B-A12B, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-{BF16,FP8,NVFP4}` | b | ~2026-03 (REPORTED [43]); в рецептах с 2026-04-28 | NVIDIA Open Model License (REPORTED) | 120.6B / 12.7B (REPORTED [43]) | 88 слоёв: гибрид Mamba-2 + ~8 attention (2 KV × 128) + LatentMoE 512 экспертов top-22 (REPORTED [43]) | 1M по заявлению; в конфиге по умолчанию 256K (REPORTED [43]); в рецепте 262 144 (VERIFIED [1]) | MTP (2 слоя с общими весами) (REPORTED [43]) | BF16, FP8, NVFP4 официально (VERIFIED [1]) |
| 9 | Step-3.7-Flash, `stepfun-ai/Step-3.7-Flash` | a/b (на грани) | 2026-05-29 (REPORTED) | Apache-2.0 (VERIFIED [14]) | 198B (196B LM + 1.8B зрение) / 11B (VERIFIED [14]) | SWA-512 : global = 3:1 (VERIFIED [1]); 45 слоёв, 8 KV × 128 (PROVISIONAL [3]) | 256K (VERIFIED [14]) | MTP-3 (VERIFIED [14]) | BF16, FP8, NVFP4 официально (VERIFIED [1]) |
| 10 | Inkling-Small, `thinkingmachines/Inkling-Small` (+`-NVFP4`) | a | 2026-07-30 (REPORTED [42]) | Apache-2.0 (REPORTED, один источник [42]) | 276B / 12B (VERIFIED [1]) | relative attention + short conv (VERIFIED [1]); KV — UNKNOWN | 1M (VERIFIED [1]) | 8 цепочечных MTP-голов (VERIFIED [1]) | BF16; NVFP4 (VERIFIED [1]) |
| 11 | MiniMax-M3, `MiniMaxAI/MiniMax-M3` | a→SSD | 2026-06 (REPORTED; в рецептах с 2026-06-12) | MiniMax Community License (REPORTED [49]) | ~428B / ~23B по README (VERIFIED [15]); 427B / 26B по рецепту (VERIFIED [1]) — конфликт | MSA (блочно-разреженный); 60 слоёв, 4 KV × 128 (PROVISIONAL, дефолты [3]) | 1M (VERIFIED [15]) | EAGLE3 от Inferact (VERIFIED [1]) | BF16; MXFP8, NVFP4, MXFP4 (VERIFIED [1]) |
| 12 | Mistral-Small-4-119B-2603 | b | 2026-03 (по суффиксу 2603) | Apache-2.0 (REPORTED) | 119B / 6.5B (VERIFIED [1]) | MLA, 36 слоёв, 128 экспертов top-4 + 1 (VERIFIED [1][3]) | 256K через YaRN ×128 поверх 8K (VERIFIED [1]) | EAGLE, 2 слоя (VERIFIED [1]) | FP8; NVFP4 официально (VERIFIED [1]) |
| 13 | Laguna-S-2.1, `poolside/Laguna-S-2.1` | b | 2026-07-21 (по рецепту) | UNKNOWN | 118B / 8B (VERIFIED [1]) | 36 SWA-512 + 12 global, 8 KV × 128, KV в FP8 (VERIFIED [1]) | 256K (VERIFIED [1]) | DFlash-драфтер (VERIFIED [1]) | BF16; FP8, NVFP4, INT4 официально (VERIFIED [1]) |
| 14 | Ling-3.0-flash, `inclusionAI/Ling-3.0-flash` | b | 2026-08/09 (UNKNOWN) | UNKNOWN | 124.4B / 5.5B (+3.1B MTP) (VERIFIED [1]) | гибрид MLA/KDA, 42 слоя, 512 экспертов top-8 (VERIFIED [1]) | 262 144 (VERIFIED [1]) | MTP + DSpark (VERIFIED [1]) | BF16; FP8, FP4, INT4 официально (VERIFIED [1]) |
| 15 | Qwen3.8-27B (плотная) | b | 2026-08-14 (VERIFIED [11]) | Apache-2.0 (REPORTED) | 27B (VERIFIED [1]) | 64 слоя, из них 48 линейных (VERIFIED [1]) | 262K нативно (VERIFIED [1]) | MTP; принятие 0.754 (REPORTED [1]) | BF16; FP8 официально |
| 16 | Hy3, `tencent/Hy3` | a | 2026-07 (по рецепту) | UNKNOWN | 295B / 21B (+3.8B MTP) (VERIFIED [1]) | 80 слоёв, полный GQA 64/8, 192 эксперта top-8 + 1 (VERIFIED [1]) | 256K (VERIFIED [1]) | MTP (VERIFIED [1]) | BF16; FP8 |

Гиганты, которые не проходят по размеру, разобраны в §9.1: Kimi K3 (2.8T), Kimi K2.6 и K2.7-Code (1T), GLM-5.3 (743B), MiMo-V2.6-Pro (1T), DeepSeek-V4-Pro (1.6T), Qwen3.8-2.4T, Nemotron-3-Ultra (550B), Mistral-Large-3 (675B), Hy4-preview (770B), Inkling (975B).

### 2.2 Качество (вендор и независимые оценки; разные харнессы между собой не сравнимы)

| Модель | AA Index v4.3.x | AA, ранние версии | Terminal-Bench 2.1 (вендор) | DeepSWE v1.1 (вендор) | SWE-bench Pro | Независимо измерено | Длинный контекст |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | **42** (41.81) REPORTED [19][25] | 57 (ранняя версия) REPORTED [23] | 84.3 (VERIFIED [10]: картинка bench_53.png) | 63.4 (VERIFIED [10]) | — | GDPval-AA v2 1773; HLE с инструментами 55.3; AutomationBench 48.8 (вендор, VERIFIED [10]) | AA-LCR 80.0% REPORTED [31]; дрейф после ~700K по отзывам сообщества REPORTED [53] |
| MiMo-V2.6-Flash | **38** REPORTED [21] | — | 87.6 REPORTED [36] | 67.9 в карточке и 65.7 в посте — конфликт, REPORTED [34] | — | Terminal-Bench 4.0: вендор 28.8, AA 22.7, Vals 24.24 (REPORTED [32][33][35]) | AA-LCR 74.3% REPORTED [32] |
| DeepSeek-V4.1-Flash | **39** REPORTED [54] | — | 90.6 (VERIFIED [17], харнесс DSH Minimal) | 74.2 (VERIFIED [17]) | — | — | LongBench-V2 base 45.2 (VERIFIED [17]); GPQA-D 90.9, TB 3.0 30.0, TB 4.0 31.2 (вендор, VERIFIED [17]) |
| DeepSeek-V4-Flash-0731 | UNKNOWN (на странице AA версии конфликтуют: 25/34) | 52 (v4.1.1) / 50 REPORTED | 82.7 (VERIFIED [17][1]) | 54.4 (VERIFIED [17][1]) | — | — | LongBench-V2 base (V4-Flash-Base) 44.7 (VERIFIED [17]) |
| Qwen3.8-Flash-Next | **40** (39.82) REPORTED [22][25] | 56 (v4.1.1, «estimate») REPORTED [22] | UNKNOWN | 58.7 REPORTED [26] | 62.5 (вендор; BenchLM) REPORTED [25][52] | LiveCodeBench v6 91.9, GPQA-D 91.7 (вендор, REPORTED [25]) | иголки 6/6 на 32K и 128K (IQ3_S, сообщество, REPORTED [12]); RULER/MRCR — UNKNOWN |
| Qwen3.5-397B-A17B | UNKNOWN | UNKNOWN | — | — | — | UD-Q4_K_XL 80.5% против 81.3% оригинала (Benjamin Marie, REPORTED [56]) | — |
| Qwen3.5-122B-A10B | UNKNOWN | 42 (эпоха v4.0); ~32 в другой версии — конфликт, REPORTED [44] | — | — | — | — | — |
| Nemotron-3-Super | UNKNOWN | 36 (на запуске) REPORTED [43][44] | — | — | — | — | — |
| Step-3.7-Flash | UNKNOWN (19 «estimate» или 31 — конфликт) | 43 (на запуске) REPORTED [45] | 59.5 (VERIFIED [14]) | — | 56.3 (VERIFIED [14]) | — | — |
| Inkling-Small | 40 (версия UNKNOWN) REPORTED [42] | — | 64.7 у вендора, 55 по перепрогону AA (REPORTED [42]) | — | — | SWE-bench Verified 80.2 (вендор, REPORTED [42]) | — |
| Mistral-Small-4 | UNKNOWN | 27 REPORTED [44] | — | — | — | — | — |
| Qwen3.8-27B | 34 (xhigh) REPORTED [57] | 52 (HN, версия неизвестна) | — | — | — | — | — |
| Для сравнения: MiMo-V2.6-Pro, GLM-5.3, Kimi K3 | 46 / 45 / 44 REPORTED [19][24] | | GLM-5.3: 88.2; K3: 88.3 (VERIFIED [17], таблица DeepSeek) | 66.9 / 67.5 (VERIFIED [17]) | | | |

Индекс AA v4.3 введён 2026-09-07. Terminal-Bench 2.1 в нём заменён на 4.0, τ³-Banking — на AutomationBench (REPORTED [20]). Значения разных версий индекса между собой **не переводятся**: GLM-5.3-Flash стоит 57 в ранней версии и 42 в v4.3.
SWE-bench Verified как общий знаменатель больше не годится: вендоры перешли на DeepSWE v1.1, SWE-bench Pro и Terminal-Bench 2.1/3.0/4.0. Кроме того, BenchLM ссылается на июльский аудит OpenAI, по которому около 30% публичных задач сломаны (REPORTED [52]).

---

## 3. KV и состояние при контексте 262 144 токена

Формулы (PROVISIONAL):
- Полный или GQA-слой: `bytes/token = n_layers_full × 2 × n_kv_heads × head_dim × bytes_el`.
- MLA: `n_layers × (kv_lora + rope_dim) × bytes_el`.
- DSA/QSA с индексером добавляет `idx_dim` на токен. В раскладке llama.cpp для GLM хранится ключ, gate и pooled, то есть 3 × 128 (VERIFIED [2] `src/llama-memory-hybrid-idx.cpp:56-58,581`).
- Объём на контекст: `KV(262K) = bytes/token × 262 144`. Рекуррентное состояние: `n_layers × heads × d_k × d_v × 4 B` (fp32).
- **Чтение KV на шаг декода** — то, что важно для заполненного контекста. У полного attention читается весь KV. У разреженного — top-k строк плюс скан индексера по pooled-ключам. У линейного — состояние читается и пишется.

| Модель | Входы | KV, B/токен (BF16 / FP8) | KV при 262K (BF16 / FP8) | Фиксированное состояние | Чтение KV на токен при 262K (FP8) |
|---|---|---|---|---|---|
| GLM-5.3-Flash | 11 DSA × (latent 512 + idx 128); в llama.cpp 12 слоёв вместе с MTP и idx 3 × 128 в f32 = 30 720 B | 14 080 / 7 040; llama.cpp — 30 720 | 3.69 / 1.85 GB; llama.cpp — 8.05 GB | KDA: 34 × 64 × 128 × 128 × 4 B = 142.6 MB + conv ≈5 MB | ≈0.4 GB: top-2048 × 512 × 11 ≈ 12 MB, скан индексера 23–92 MB (k-pool 16 или 4), запись и чтение KDA-состояния ≈0.29 GB |
| MiMo-V2.6-Flash | 9 global × 4 KV × (192 + 128) | 23 040 / 11 520 | 6.04 / 3.02 GB | SWA: 39 × 128 × 8 × 320 × 2 B ≈ 26 MB | **3.0 GB** (global-слои читаются целиком) |
| DeepSeek-V4.1-Flash | вендор: 890 B/токен в FP4 (VERIFIED [17]) | ≈3 560 / ≈1 780 (пересчёт ×4 и ×2 от FP4) | 0.93 / 0.47 GB; FP4 — 0.23 GB | SWA-окно 128 на слой | ≈0.1 GB (top-512) |
| DeepSeek-V4-Flash-0731 | вендор: V4.1 даёт «~1/4 от V4-Flash» (VERIFIED [17]) → ≈3.5 KB/токен | ≈7 000 / ≈3 500 | 1.84 / 0.92 GB | SWA-128 | ≈0.1 GB |
| Qwen3.8-Flash-Next | 12 QSA × (2 × 2 × 256 + idx 128 + pooled 128) | 30 720 / 15 360 | 8.05 / 4.03 GB | GDN: 36 × 48 × 128 × 128 × 4 B = 113 MB | ≈0.35 GB |
| Qwen3.5-397B | 15 full × 2 × 2 × 256 | 30 720 / 15 360 | 8.05 / 4.03 GB | GDN ≈0.19 GB | **4.0 GB** |
| Qwen3.5-122B | 12 full × 2 × 2 × 256 | 24 576 / 12 288 | 6.44 / 3.22 GB | GDN ≈0.1 GB | **3.2 GB** |
| Nemotron-3-Super | 8 attention × 2 × 2 × 128 | 8 192 / 4 096 | 2.15 / 1.07 GB | Mamba ≈0.17 GB (дефолты Nano) | 1.1 GB |
| Step-3.7-Flash | ≈11 global × 2 × 8 × 128 | ≈46 080 / 23 040 | 12.1 / 6.0 GB | SWA-512 ≈70 MB | **6.0 GB** |
| Mistral-Small-4 | 36 × (256 + 64) | 23 040 / 11 520 | 6.04 / 3.02 GB | — | 3.0 GB |
| Laguna-S-2.1 | 12 global × 2 × 8 × 128 | 49 152 / 24 576 | 12.9 / 6.4 GB | SWA ≈70 MB | **6.4 GB** |
| MiniMax-M3 | 60 × 2 × 4 × 128 (дефолты [3]) | 122 880 / 61 440 | **32.2 / 16.1 GB** | — | ≈1 GB (MSA, top-16 блоков) |
| Hy3 | 80 × 2 × 8 × 128 | 327 680 / 163 840 | **85.9 / 42.9 GB** | — | **42.9 GB** (не годится) |

Вывод (PROVISIONAL): у GLM-5.3-Flash, Qwen3.8-Flash-Next и DeepSeek V4/V4.1 KV на 262K занимает единицы гигабайт, а чтение на токен — ≤0.4 GB. Декод почти не замедляется с ростом контекста.
Это подтверждает и замер сообщества для Qwen3.8-Flash-Next: 70.9 / 77.3 / 71.1 ток/с при промптах 4K / 32K / 128K (REPORTED [12]). У MiMo, Qwen3.5, Step-3.7 и Laguna на 262K к весам добавляется 3–6 GB чтения на токен.
Нативность 262K: у Qwen3.8-Flash-Next, Qwen3.5 и Qwen3.8-27B она нативная (VERIFIED [1]). У DeepSeek V4.1 — YaRN ×16 поверх 64K (VERIFIED [17]). У Mistral Small 4 — YaRN ×128 поверх 8K, и это слабый признак для качества на длинном контексте (VERIFIED [1]). У GLM-5.3-Flash позиционного RoPE в MLA нет совсем (`qk_rope_head_dim=0`, VERIFIED [3]), 1M — заявленный максимум.
Независимых RULER/MRCR/HELMET на 128–256K для этих моделей я не нашёл (UNKNOWN). Единая метрика — AA-LCR (документы ~100K), она есть только для GLM-5.3-Flash (80.0%) и MiMo-V2.6-Flash (74.3%) (REPORTED).

---

## 4. Кванты и качество 4-bit

| Модель | Официальные кванты | GGUF (размер) | Потери 4-bit | Замечание |
|---|---|---|---|---|
| GLM-5.3-Flash | FP8 (родной); BF16; NVFP4: `RedHatAI/GLM-5.3-Flash-NVFP4` «требует Blackwell», `vram_minimum` 229 (VERIFIED [1]); `nvidia/GLM-5.3-Flash-NVFP4` (REPORTED [37]; 204 GB по README strata-glm, VERIFIED [13]) | unsloth: UD-IQ4_XS 156.82 GB, **UD-Q4_K_XL 199.71 GB (= 185.99 GiB, ≈4.98 bpw)**, UD-Q5_K_XL 240.31 GB, UD-IQ3_XXS ≈120 GB (REPORTED [27]) | UD-Q4_K_XL: совпадение top-1 92.22%, средний KLD 0.0493, KLD на 99.9-м перцентиле 1.49. UD-IQ4_XS: top-1 88.18%. UD-Q5_K_XL: 94.35% и 0.0271 (REPORTED [27]). strata-glm на NVFP4-экспертах: KLD 0.018, top-1 91.7% после промпта 2600 токенов (VERIFIED [13]) | Родная точность FP8, поэтому любые 4 бита — настоящее переквантование с заметной потерей |
| MiMo-V2.6-Flash | родное хранение MXFP4 (173 GB) | `ggml-org/MiMo-V2.6-Flash-RL-GGUF:MXFP4` (REPORTED [55], размер UNKNOWN). Конвертер llama.cpp перепаковывает `store_dtype=mxfp4` в GGUF MXFP4 без переквантования (VERIFIED [2] `conversion/mimo.py:177-251`) | эксперты без потерь относительно релиза (PROVISIONAL по коду конвертера) | 4-битный класс «бесплатно»; качество в отчётах AA уже содержит эффект MXFP4 |
| DeepSeek-V4.1-Flash | родные MXFP4 + MXFP8 (476 GiB); NVFP4 `LibertAIDAI/…` (репо [18]) | для mainline нет (арки нет); ik_llama.cpp поддерживает арку `deepseek41` (VERIFIED [6]) | эксперты родные | Engram 196.6B в FP8 + E8M0 ≈189 GiB — кандидат на SSD-ярус |
| DeepSeek-V4-Flash-0731 | родные FP4 + FP8; `nvidia/…-NVFP4`, `RedHatAI/…-NVFP4-FP8` (VERIFIED [1]) | unsloth UD-Q4_K_XL **155.1 GB**, эксперты бит-в-бит, не-эксперты в Q8_0 (REPORTED [28]) | KLD 0.0102, «same top token» 96.28%, PPL 4.5335 по документации [28]; в обсуждении на HF — KLD 0.01324, 96.04% (REPORTED [29]) — конфликт | Практически «родные» 4 бита |
| Qwen3.8-Flash-Next | BF16, FP8 официально; NVFP4 Inferact; AWQ-INT4 cyankiwi (REPORTED) | unsloth UD-Q4_K_XL **111.3 GB**, UD-Q5_K_XL 158.3 GB, Q8_0 188.2 GB (REPORTED [30]); UD-IQ4_XS 94 GB, в нём 59.5 GB экспертов; у UD-Q4_K_XL 77 GB экспертов; ISTA-DASLab IQ3_S 54.8 GB (VERIFIED [12]) | UD-Q4_K_XL: top-1 92.255%, KLD 0.0469; UD-Q5_K_XL: 93.68% и 0.0304 (REPORTED [30]) | Потери у Q4 того же порядка, что у GLM |
| Qwen3.5-397B | GPTQ-Int4 официально (VERIFIED [1]); NVFP4 nvidia | unsloth, размеры UNKNOWN; для 4 бит ~256 GB RAM (REPORTED [56]) | UD-Q4_K_XL 80.5 против 81.3 (REPORTED [56]) | — |
| Qwen3.5-122B | GPTQ-Int4, FP8 официально (VERIFIED [1]) | UNKNOWN | UNKNOWN | — |
| Nemotron-3-Super | FP8, NVFP4 официально; модель обучалась с NVFP4 (REPORTED [43]) | UNKNOWN | UNKNOWN | — |
| Step-3.7-Flash | FP8, NVFP4 официально; у Step-3.5 есть INT4 (VERIFIED [1]) | UNKNOWN | NVFP4 + MTP на GPQA-D: 77.81% против 78.41% (VERIFIED [14]) | — |

NVFP4-чекпойнты в рецептах помечены как Blackwell-only (VERIFIED [1]). У sm_89 нет FP4-тензорных ядер, поэтому на этом сервере они работают только через W4A16-фолбэк (HYPOTHESIS, проверить по движку).
Для Ada естественные форматы: GGUF (llama.cpp и ik_llama.cpp), EXL3, AWQ/GPTQ (vLLM Marlin), MXFP4 через Marlin или ggml.

---

## 5. Поддержка движками (HEAD на 2026-10-09; «да» — арка или модель есть в исходниках или в рецепте)

| Модель | llama.cpp mainline @79e2e74 [2] | ik_llama.cpp @5194a9e [6] | vLLM @ed7ae70 [4], [1] | SGLang @46c2585 [5] | KTransformers @2ac8cf9 [7] | TensorRT-LLM @49ec573 [9] | ExLlamaV3 @151539c [8] |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | да, `glm5-next`; строки `glm5next` нет | да, `glm5next` и `glm5-next` | да, `vllm/models/glm5next` (nightly) | да, `glm5_next.py` | частично: туториал требует AVX-512 FP8-ядро и 306 GiB RAM (VERIFIED [7]), на AVX2 — только BF16/FP8/GPTQ_INT4/RAWINT4 | да (`Glm5NextForConditionalGeneration`) | да, `glm5_next.py` |
| MiMo-V2.6-Flash | да, `mimo2` + MXFP4-перепаковка | частично (`mimo2` есть, про V2.6 UNKNOWN) | да, нужен v0.31.0 (VERIFIED [1]) | да, `mimo_v2.py` | UNKNOWN | нет | да, `mimo_v2.py` |
| DeepSeek-V4.1-Flash | **нет** (нет V41 и Engram) | **да**, `deepseek41` + Engram | да, `vllm/models/deepseek_v41` | да (`deepseek_v4.py`: Engram, V41Indexer) | UNKNOWN | нет (V4 только на Blackwell) | нет |
| DeepSeek-V4-Flash-0731 | да, `deepseek4` | да | да | да | UNKNOWN | только Blackwell (VERIFIED [9]) → на sm_89 нет | да, `deepseek_v4.py` |
| Qwen3.8-Flash-Next | да, `qwen4exp` | да | да, отдельный образ; offload n-gram-таблицы в RAM (VERIFIED [1]) | да, `qwen4_exp*.py` | UNKNOWN | да | да, `qwen4_exp.py` |
| Qwen3.5 (397B, 122B) | да, `qwen35moe` | да | да | да | да (семейство; есть AVX2-пример FP8 для 35B) | да | да |
| Nemotron-3-Super | да, `nemotron_h_moe` | нет | да | да | UNKNOWN | да | да, `nemotronh.py` |
| Step-3.7-Flash | да (`Step3p7…` → `step35`) | частично (`step35`) | nightly | да | UNKNOWN | да | да |
| Inkling-Small | **нет** | **нет** | да | да, `inkling.py` | UNKNOWN | нет | нет |
| MiniMax-M3 | да, `minimax-m3` | да | да | да | да (туториал) | да | нет |
| Mistral-Small-4 | да, `mistral4` | да | да | да | UNKNOWN | — | да |

Strata (Niko1221 @fb58e0d) поддерживает «из коробки» только Qwen3.8-Flash-Next. GLM-5.3-Flash идёт через базы A и B репозитория (strata-glm, lighttransport). Остальные модели требуют портирования (VERIFIED [12][13]; ADR репозитория [18]).
Гибридный режим GPU + CPU-эксперты, который нужен классу (a), сейчас реально дают только llama.cpp, ik_llama.cpp, KTransformers и Strata. vLLM, SGLang, TRT-LLM и ExLlamaV3 рассчитаны на VRAM. У vLLM есть `--cpu-offload-gb` через UVA, но по PCIe это медленно (HYPOTHESIS).

---

## 6. Fit и верхние границы скорости

**Допущения (PROVISIONAL).** VRAM 4 × (32 − 2) = 120 GB, из них 4 GB под буферы и драфтер. RAM 128 − 12 = 116 GB. 4.5 bpw = 0.5625 B на параметр.
Полоса: GPU 700 GB/s (конвейер, активна одна карта), RAM 150 GB/s (между 4 и 8 каналами: 102–205 GB/s теоретически), SSD 20 GB/s (3–4 × 990 Pro).
Попадание в VRAM при равномерной маршрутизации: `h = (120 − неэкспертные веса − KV_FP8(262K) − 4) / веса_экспертов`. При реальном перекосе маршрутизации h выше: strata-glm с 13% экспертов в VRAM попадал в 55–58% (VERIFIED [13]).
- (i) `tok/s = 700 / (active × 0.5625 + чтение_KV)` — все веса в VRAM.
- (ii) `t = (неэкспертные_active + h·routed_active)·B/700 + (1−h)·routed_active·B/150 [+ доля_SSD·routed_active·B/20]`. «Посл.» — сумма времён (GPU ждёт CPU), «паралл.» — максимум из них.

Таблица 6.1 — равномерные 4.5 bpw (скрипт в песочнице, вывод ниже без изменений):

```
model                      W4.5  KVbf16  KVfp8 GB/tok  +KVrd i_tok/s     h RAMneed    SSD ii_ser ii_ovl fit
GLM-5.3-Flash             180.6    3.69   2.00  10.12  10.53      67  0.61    66.6    0.0     40     81 (ii) VRAM+RAM  side=0.0GB
MiMo-V2.6-Flash           173.8    6.04   3.05   8.44  11.46      61  0.64    60.9    0.0     38     73 (ii) VRAM+RAM  side=0.0GB
DeepSeek-V4.1-Flash       310.5    0.93   0.52   9.00   9.10      77  0.36   195.0   79.0     12     16 (iii) SSD  side=110.6GB
DeepSeek-V4-Flash-0731    159.8    1.84   0.97   7.31   7.41      94  0.71    44.7    0.0     62    110 (ii) VRAM+RAM  side=0.0GB
Qwen3.8-Flash-Next         70.3    8.05   4.15   3.38   3.73     188  1.00     0.0    0.0    188    188 (i) VRAM  side=28.8GB
Qwen3.5-397B-A17B         223.3    8.05   4.22   9.56  13.59      51  0.49   111.5    0.0     32     61 (ii) VRAM+RAM  side=0.0GB
Qwen3.5-122B-A10B          68.6    6.44   3.32   5.62   8.85      79  1.00     0.0    0.0     79     79 (i) VRAM  side=0.0GB
Nemotron-3-Super-120B      67.8    2.15   1.24   7.14   8.21      85  1.00     0.0    0.0     85     85 (i) VRAM  side=0.0GB
Step-3.7-Flash            111.4   12.08   6.11   6.19  12.23      57  0.99     1.5    0.0     56     57 (ii) VRAM+RAM  side=0.0GB
Mistral-Small-4-119B       66.9    6.04   3.02   3.66   6.68     105  1.00     0.0    0.0    105    105 (i) VRAM  side=0.0GB
Laguna-S-2.1               66.4   12.88   6.51   4.50  10.94      64  1.00     0.0    0.0     64     64 (i) VRAM  side=0.0GB
MiniMax-M3                240.2   32.21  16.11  14.62  15.62      45  0.39   140.3   24.3     12     24 (iii) SSD  side=0.0GB
Hy3                       165.9   85.90  42.95  11.81  54.71      13  0.42    92.9    0.0     10     14 (ii) VRAM+RAM  side=0.0GB
```

Расшифровка столбцов. W4.5 — веса в GB без боковых таблиц; side — n-gram или Engram при 4.5 bpw, такие таблицы держатся вне GPU. GB/tok — активные веса на токен. +KVrd — то же плюс чтение KV при 262K. i — граница (i). h — доля экспертов в VRAM. RAMneed и SSD — сколько GB экспертов уходит в RAM и на SSD. ii_ser и ii_ovl — граница (ii) последовательная и параллельная.
Разбиение routed/неэкспертные (PROVISIONAL): GLM — 304.4B / 8.46B active routed (42 × 288 × 25.17M; 42 × 8 = 336 экспертов на токен, 4.76 GB в NVFP4 — VERIFIED [13]). MiMo — 302.8B / 9.46B. DeepSeek-V4.1 — 543.6B / 8.49B. DeepSeek-V4-Flash — 277.0B / 6.49B. Qwen3.8-Flash-Next — 120.8B / 2.36B. Qwen3.5-397B — 386.5B / 7.55B. Для Nemotron, Step, Mistral, Laguna, MiniMax и Hy3 — грубые оценки.

Таблица 6.2 — реальные размеры файлов (эффективный bpw = размер × 8 / параметры):

```
GLM-5.3-Flash UD-Q4_K_XL 199.71GB          bpw=4.98 E= 189.4 N= 10.3 h=0.55 RAM=  85.7 SSD=  0.0 tok/s: i=   60 ii_ser=   34 ii_ovl=   63
DSV4-Flash-0731 UD-Q4_K_XL 155.1GB         bpw=4.37 E= 151.3 N=  3.8 h=0.74 RAM=  40.1 SSD=  0.0 tok/s: i=   97 ii_ser=   66 ii_ovl=  112
MiMo-V2.6-Flash MXFP4 ~173GB               bpw=4.48 E= 169.5 N=  3.5 h=0.65 RAM=  60.0 SSD=  0.0 tok/s: i=   61 ii_ser=   38 ii_ovl=   73
Qwen3.8-FN UD-Q4_K_XL main~82GB            bpw=5.25 E=  79.2 N=  2.8 h=1.00 RAM=   0.0 SSD=  0.0 tok/s: i=  163 ii_ser=  163 ii_ovl=  163
Qwen3.8-FN UD-Q5_K_XL main~104GB(est)      bpw=6.66 E= 100.5 N=  3.5 h=1.00 RAM=   0.0 SSD=  0.0 tok/s: i=  131 ii_ser=  131 ii_ovl=  131
GLM heff=0.61 RAMbw=75: serial= 24.4 overlap= 36.5
GLM heff=0.61 RAMbw=150: serial= 36.6 overlap= 73.1
GLM heff=0.75 RAMbw=75: serial= 31.0 overlap= 57.0
GLM heff=0.75 RAMbw=150: serial= 42.6 overlap= 68.1
GLM heff=0.90 RAMbw=75: serial= 43.8 overlap= 63.2
GLM heff=0.90 RAMbw=150: serial= 51.7 overlap= 63.2
GLM heff=1.00 RAMbw=75: serial= 60.4 overlap= 60.4
GLM heff=1.00 RAMbw=150: serial= 60.4 overlap= 60.4
```

Основная часть Qwen3.8-Flash-Next ≈82 GB = 77 GB экспертов (VERIFIED [12]) плюс ≈5 GB прочего (PROVISIONAL). Таблица n-gram ≈29 GB лежит в RAM.

**Таблица fit (PROVISIONAL).**

| Категория | Модели (в качестве ≥4 бит) |
|---|---|
| (i) целиком в VRAM вместе с KV на 262K | **Qwen3.8-Flash-Next** (UD-Q4_K_XL; n-gram — в RAM), Qwen3.5-122B, Nemotron-3-Super, Mistral-Small-4, Laguna-S-2.1, Ling-3.0-flash (PROVISIONAL), Qwen3.8-27B |
| (ii) VRAM + RAM | **GLM-5.3-Flash** (h ≈ 0.55–0.61), **DeepSeek-V4-Flash-0731** (h ≈ 0.71–0.74), **MiMo-V2.6-Flash** (h ≈ 0.64), Qwen3.5-397B (h ≈ 0.49, RAM 111.5 из 116 GB — на грани), Step-3.7-Flash (почти VRAM), Inkling-Small (≈155 GB, но гибридного движка нет), Hy3 (влезает, но проваливает 262K по KV) |
| (iii) нужен SSD | **DeepSeek-V4.1-Flash** (≈79 GB экспертов + Engram), MiniMax-M3 (≈24 GB экспертов, лицензия), все гиганты из §9.1 |

Чтение границ. В конвейере (одна активная карта) модели с 18B active (GLM) и 15B (MiMo) не могут превысить **~60 ток/с** даже при 100% попаданий в VRAM. 80–100 ток/с для них требуют TP/EP на 4 картах (P2P у заказчика заявлен, но не измерен) или спекуляции с высоким принятием.
У MoE при batch 1 верификация блока подгружает объединение экспертов нескольких токенов, поэтому выигрыш меньше длины принятия (HYPOTHESIS). Qwen3.8-Flash-Next — единственный сильный кандидат, у которого 80–100+ ток/с правдоподобны без TP.
Внешняя опора: UD-Q4_K_XL на **одной** RTX PRO 4500 Blackwell 32 GB (896 GB/s) с RAM Threadripper показал 61.5 ток/с при промпте 2.7K и 70.2 ток/с при 128K (REPORTED [12] bench/results/2026-10-01-community-rtx-pro-4500).

---

## 7. Pareto-фронт и шорт-лист

**Оси:** качество (AA v4.3 плюс агентные бенчмарки вендора), скорость (граница из §6 плюс внешние замеры), стоимость 262K (чтение KV), fit в ≥4 битах, наличие гибридного движка.

**Фронт (HYPOTHESIS на основе REPORTED и PROVISIONAL):**
- **Qwen3.8-Flash-Next** — максимум скорости при качестве, близком к лидерам. AA 40 против 42 у GLM-5.3-Flash. Целиком в VRAM. Разреженное и линейное attention. 262K нативно. Готовый движок (Strata) с замерами сообщества.
  Минусы: лицензия Qwen Community License 1.0 (продуктам типа «AI Work Assistant», предлагаемым третьим лицам, по сообщениям нужна отдельная лицензия — REPORTED [47]). Агентные бенчмарки вендора ниже, чем у GLM и MiMo: DeepSWE 58.7 против 63.4 и 67.9. Модель позиционируется как «превью Qwen4».
- **GLM-5.3-Flash** — максимум качества среди тех, кто влезает (AA 42, TB2.1 84.3, DeepSWE 63.4). Attention дешёвое на 262K, MIT, самая широкая поддержка движками.
  Минусы: 4 бита — переквантование из FP8 (top-1 92.2%). Скорость в конвейере ≤60 ток/с, реалистично 35–60 ток/с.
- **DeepSeek-V4-Flash-0731** — средняя точка. Эксперты без потерь в 4 битах (KLD ≈0.01), наименьший тиринг (h ≈ 0.74), потолок 66–112 ток/с, 1M контекст при KV <1 GB, MIT.
  Минусы: качество ниже GLM (TB2.1 82.7 против 84.3, DeepSWE 54.4 против 63.4; AA v4.1.1 52 против 57).
- **MiMo-V2.6-Flash** — рядом с фронтом. Родные MXFP4 без потерь, сильные агентные числа вендора (TB2.1 87.6, DeepSWE 65.7–67.9).
  Минусы: на AA ниже GLM (38 против 42), независимые TB4.0 22.7–24.2 против 28.8 у вендора, на 262K +3 GB чтения KV на токен. Выпуск RL склонен повторять tool-call, рекомендован MOPD (VERIFIED [1]).
- Вне фронта на этом железе:
  - DeepSeek-V4.1-Flash — лучший по агентным числам вендора, но SSD-bound, ≤16 ток/с.
  - Qwen3.5-122B, Nemotron-3-Super, Mistral-Small-4, Laguna-S-2.1 — по совокупности их доминирует Qwen3.8-Flash-Next: она быстрее за счёт разреженного attention на 262K и выше по качеству; их AA в ранних версиях 27–42.
  - Step-3.7-Flash — ниже по качеству (TB2.1 59.5) и тяжёлый KV.
  - Inkling-Small — нет гибридного движка.

**Ранжированный шорт-лист для первых замеров на железе:**
1. **Qwen3.8-Flash-Next, UD-Q4_K_XL** (затем UD-Q5_K_XL, если влезет с KV). Strata целиком в VRAM; базовая линия — llama.cpp или ik_llama.cpp `qwen4exp`.
   Мерить: декод при промптах 4K / 64K / 128K / 250K, префилл, принятие MTP. Качество — задачи заказчика и long-doc QA на 128–262K в 4 битах против FP8 через API.
2. **GLM-5.3-Flash, UD-Q4_K_XL** — текущая цель репозитория. ik_llama.cpp (понимает `glm5next`) против баз A и B.
   Мерить: h_eff (перекос маршрутизации на трассах заказчика), полосу RAM (4 или 8 каналов), декод на 262K, принятие MTP и DFlash2.
3. **DeepSeek-V4-Flash-0731, UD-Q4_K_XL** (155 GB). llama.cpp mainline `deepseek4` и ik_llama.cpp. Мерить те же метрики плюс качество против GLM на агентных задачах.
4. **MiMo-V2.6-Flash-MOPD, MXFP4 GGUF.** llama.cpp `mimo2`. Мерить падение декода от 4K до 262K: проверка гипотезы о стоимости global-слоёв.
5. (только эталон качества) DeepSeek-V4.1-Flash с SSD-ярусом — без претензии на скорость.

**Что изменит ранжирование:**
- Если в 4 битах на задачах заказчика (агентный код, длинные документы) Qwen3.8-Flash-Next проиграет GLM-5.3-Flash больше ~5 п.п., а GLM при этом даст ≥50 ток/с на 262K, то GLM выходит на первое место.
- Замер полосы RAM (8 каналов) и h_eff ≥0.85 поднимают тиринговые модели (GLM, DSV4F, MiMo) к их потолкам 60 / 97 / 61 ток/с.
- Работающий TP/EP по PCIe с P2P поднимает все потолки в 2–3 раза (HYPOTHESIS). Тогда GLM может выйти за 80 ток/с, и разрыв по скорости с Qwen3.8 перестаёт быть решающим.
- Если лицензия Qwen Community License запретит сценарий заказчика, Qwen3.8-Flash-Next выпадает, и первым становится GLM-5.3-Flash либо DeepSeek-V4-Flash-0731.
- Высокое принятие спекуляции (≥3 токена) на 4-битных целях в нашем движке выгоднее всего для GLM и MiMo.

---

## 8. Проверка исторических утверждений репозитория

| Утверждение (где) | Статус | Доказательство |
|---|---|---|
| GLM-5.3-Flash: 45 слоёв, 34 KDA + 11 DSA (models/glm-5.3-flash.json, docs/models/glm-5.3-flash.md) | **VERIFIED** | transformers `configuration_glm5_next.py:105,165` (`idx % 4 != 3` → KDA), `@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")` [3]; KTransformers: «45 layers comprise 34 Linear Attention layers and 11 DSA layers» [7]; llama.cpp `case 45: LLM_TYPE_320B_A18B` [2] |
| 288 экспертов, top-8, FF 2048, hidden 4096 | **VERIFIED** | transformers `:101-119` [3]; рецепт: «8 of 288» [1]; strata-glm: «42 MoE layers × 288» [13] |
| Контекст 1M | **VERIFIED** (заявленный максимум) | рецепт `context_length: 1048576` [1]; `max_position_embeddings 1048576` [3]. Качество на 1M — UNKNOWN; есть отзывы о дрейфе после ~700K (REPORTED [53]) |
| Индексер «k-pool 4» (ADR-001) | **UNKNOWN / конфликт** | значение `index_kpool` llama.cpp берёт из config.json [2]; дефолт transformers 16 [3]; сам config.json здесь не прочитан |
| Total 320B / active 18B | **VERIFIED** | README: «320B-A18B» [10]; рецепт: «321B / 18B» [1] |
| MiMo-V2.6-Flash: 309B/15B, 48 слоёв (1 + 47), 256 × top-8, SWA-128 + global, 5-слойный DFlash на 7 токенов, 1M | **VERIFIED** | рецепт MiMo-V2.6-Flash-RL [1] |
| MiMo: 39 SWA + 9 global, KV 4 и 8 голов, QK 192 / V 128 | **REPORTED / PROVISIONAL** | совпадает с дефолтами transformers для **V2**-Flash [3]; config.json V2.6 не прочитан |
| MiMo: родные MXFP4-эксперты ≈150 GiB | **VERIFIED** (порядок) | рецепт: чекпойнт 173 GB с хранением в mxfp4 [1]; расчёт 302.8B × 4.25/8 = 160.9 GB ≈ 149.8 GiB (PROVISIONAL) |
| MiMo: `hf_repo` = `…-Flash-RL` | **устарело** | рецепт рекомендует `…-Flash-MOPD`: RL-версия повторяет tool-call (VERIFIED [1]) |
| DeepSeek-V4.1-Flash: 552B + Engram 196B | **VERIFIED** | вендоренный README и config@2cba9e4 [17]; рецепт: 196.6B ≈ 183.1 GiB в FP8 без масштабов [1]. Цифра репозитория 189.1 GiB включает E8M0/32 — согласуется |
| DeepSeek-V4.1-Flash: MXFP4-эксперты | **VERIFIED** | config `"expert_dtype": "fp4"`, `scale_fmt ue8m0` [17]; рецепт: «Routed expert weights are MXFP4» [1] |
| DeepSeek: «нет графа в llama.cpp» (на пине ec7630a) | **VERIFIED и сейчас** (mainline @79e2e74: нет `DeepseekV41`/Engram) | [2]; при этом **ik_llama.cpp поддерживает `deepseek41` с Engram** [6] — новый факт |
| DeepSeek: «требует переквантования ≤3 бит» (ADR-001) | **VERIFIED арифметически** | ≈290–310 GB экспертов > 236 GB VRAM+RAM. Но ≤3 бит нарушает требование заказчика «≥4 бит» → как цель модель **не проходит** |
| `nvidia/GLM-5.3-Flash-NVFP4` существует, 204 GB | **REPORTED** | README strata-glm называет 204 GB и даёт инструкцию скачать именно этот репозиторий (VERIFIED [13]); страница HF видна в поиске [37]. vLLM вместо него рекомендует `RedHatAI/GLM-5.3-Flash-NVFP4` (VERIFIED [1]). Размер на HF — UNKNOWN |
| unsloth GLM-5.3-Flash-GGUF UD-Q4_K_XL ≈186 GiB (AGENTS.md: 185.983 GiB) | **REPORTED, согласуется** | документация unsloth: 199.71 GB = 185.99 GiB [27] |
| Строка арки GGUF `glm5next` (AGENTS.md, ADR-010) | **VERIFIED с оговоркой** | mainline llama.cpp знает только `"glm5-next"` (`src/llama-arch.cpp:158`) [2]; ik_llama.cpp знает обе (`:90-91`) [6]. Файлы с `glm5next` в mainline, скорее всего, не загрузятся (HYPOTHESIS — проверить запуском) |
| models/qwen3.8-flash-next.json: total 125, active null | **неполно** | active 6B, n-gram 51.2B, контекст 262K нативно (VERIFIED [1]) |
| hardware: 3 × 990 Pro, P2P true (`hardware/4x4080s-32g.json`) | **устарело** | в задании «4 × 1 TB NVMe, P2P unknown». Новый `hardware/target-4x4080-32g.unverified.json` (коммит 125f35f, появился, пока шло это исследование) прямо пишет: «its NVMe count and p2p=true are outdated» [18] |
| Неэкспертные веса GLM (`models/specs/glm-5.3-flash.json`, 125f35f): KDA 4.68B, DSA 1.37B, shared 1.06B, dense FFN 0.45B | **согласуется** | моя независимая оценка в §6: ~6.1B attention, 1.06B shared, 0.45B dense (PROVISIONAL) |

---

## 9. Негативные находки и конфликты

### 9.1 Выглядит привлекательно, но проваливает жёсткое требование

- **Kimi K3** (`moonshotai/Kimi-K3`, 2.8T, MXFP4 QAT, 1M; Kimi K3 License): VERIFIED [16][1]. Веса ~1.5 TB → SSD, скорость однозначные числа ток/с.
  **Kimi K2.6 и K2.7-Code** (1T/32B, INT4 ~595 GB, 262K): VERIFIED [1]. Тоже SSD-bound.
- **GLM-5.3** (743B/39B; FP8 893 GB, NVFP4 558 GB), **MiMo-V2.6-Pro** (1T/42B; 566 GB), **DeepSeek-V4-Pro** (1.6T), **Qwen3.8-2.4T-A95B**, **Nemotron-3-Ultra** (550B/55B; NVFP4 330 GB), **Mistral-Large-3** (675B), **Hy4-preview** (770B), **Inkling** (975B): VERIFIED [1]. Все больше VRAM + RAM в 4 битах.
- **DeepSeek-V4.1-Flash**: в 4 битах SSD-bound (12–16 ток/с, PROVISIONAL). Ниже 4 бит нельзя по требованию заказчика.
- **MiniMax-M3**: ≈240–257 GB в 4 битах плюс 16 GB KV в FP8 на 262K → SSD. Лицензия MiniMax Community License с ограничениями (REPORTED [49]).
  **MiniMax M3.1 Flash** — только API (REPORTED [51]). **GLM-5.3 FlashX** — хостинговый тариф, отдельных весов нет (REPORTED [50]).
- **Hy3**: полный GQA на 80 слоях → 43 GB KV в FP8 на 262K и 43 GB чтения на токен (PROVISIONAL). 262K практически неприменим.
- **gpt-oss-120b**: контекст 131 072 (VERIFIED [1]) < 262K. **MiniMax-M2.5 и M2.7**: 196 608 (VERIFIED [1]) < 262K.
- **Mistral-Medium-3.5-128B** (плотная): ≈72 GB на токен в 4.5 bpw → ≈10 ток/с в конвейере (PROVISIONAL).
- **Qwen3.8-27B** (плотная): потолок конвейера ≈46 ток/с, AA 34 < 40–42.
- **Inkling-Small**: нет llama.cpp, ik_llama.cpp и ExLlamaV3 (VERIFIED [2][6][8]); ≈155 GB в 4 битах > 120 GB VRAM. Гибрида нет, VRAM-only движки не вмещают модель.
- **KTransformers** для GLM-5.3-Flash: нужен AVX-512 FP8 и 306 GiB RAM (VERIFIED [7]); у сервера AVX2 и 128 GB. **TRT-LLM** для DeepSeek-V4 — только Blackwell (VERIFIED [9]).
- **NVFP4-чекпойнты** (nvidia, RedHatAI) в рецептах помечены «requires Blackwell» (VERIFIED [1]); на sm_89 нет FP4-тензорных ядер.

### 9.2 Конфликты между источниками

- Индекс AA по версиям: GLM-5.3-Flash 57 против 42; Qwen3.8-Flash-Next 56 против 40 (39.82); DeepSeek-V4-Flash-0731 52 и 50 против 25 и 34 на разных страницах; Qwen3.5-122B 42 против ~32; Qwen3.8-27B 52 против 34. Всё REPORTED, страницы не загружены.
- MiMo-V2.6-Flash: DeepSWE 67.9 в карточке против 65.7 в посте. Terminal-Bench 4.0: 28.8 у вендора, 22.7 у AA, 24.24 у Vals. Чекпойнт 173 GB [1] против 165.53 GiB [55]. Релиз 09-21 или 09-22.
- Inkling-Small: TB2.1 64.7 у вендора против 55 у AA.
- DeepSeek-V4-Flash-0731: 284B [1] против 304B [39]; UD-Q4_K_XL KLD 0.0102 [28] против 0.01324 [29].
- DeepSeek-V4.1-Flash: релиз 09-10 или 09-15.
- Qwen3.8-Flash-Next: 125B, 176B или 180B; «262K нативно» [1] против «1M» в некоторых обзорах; релиз 08-26…28.
- MiniMax-M3: 23B active по README [15] против 26B по рецепту [1].
- Step-3.7-Flash: 198B [14] против 201.4B.
- Квантованные GLM-5.3-Flash NVFP4: nvidia 204 GB [13], RedHatAI 229 GB VRAM [1], сообщество 181 GiB–196 GB (REPORTED).
- GLM `index_kpool`: 4 (репо) против 16 (дефолт transformers).
- Строка арки GLM: `glm5next` против `glm5-next`.
- Железо: 3 против 4 NVMe; P2P «true» против «unknown». Старый `hardware/4x4080s-32g.json` против нового `hardware/target-4x4080-32g.unverified.json`; новый помечает старые значения как устаревшие.
- Лицензия GLM-5.3-Flash: MIT для весов (REPORTED) и Apache-2.0 для GitHub-кода — это разные объекты, а не противоречие.

---

## 10. Источники (дата обращения для всех: 2026-10-09)

Первичные (GitHub-клоны в песочнице и файлы репозитория; VERIFIED):
1. vllm-project/recipes @`8ed3d031cee2f8b7cfc5b701438aaacef6c69a18` — `models/zai-org/GLM-5.3-Flash.yaml`, `GLM-5.3.yaml`, `XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml`, `MiMo-V2.6-Pro-RL.yaml`, `deepseek-ai/DeepSeek-V4.1-Flash.yaml`, `DeepSeek-V4-Flash.yaml`, `DeepSeek-V4-Pro.yaml`, `Qwen/Qwen3.8-Flash-Next.yaml`, `Qwen3.8-27B.yaml`, `Qwen3.8-2.4T-A95B.yaml`, `Qwen3.5-397B-A17B.yaml`, `Qwen3.5-122B-A10B.yaml`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.yaml`, `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.yaml`, `stepfun-ai/Step-3.7-Flash.yaml`, `thinkingmachines/Inkling-Small.yaml`, `Inkling.yaml`, `MiniMaxAI/MiniMax-M3.yaml`, `MiniMax-M2.7.yaml`, `mistralai/Mistral-Small-4-119B-2603.yaml`, `Mistral-Medium-3.5-128B.yaml`, `Mistral-Large-3-675B-Instruct-2512.yaml`, `poolside/Laguna-S-2.1.yaml`, `inclusionAI/Ling-3.0-flash.yaml`, `tencent/Hy3.yaml`, `Hy4-preview.yaml`, `moonshotai/Kimi-K3.yaml`, `Kimi-K2.6.yaml`, `Kimi-K2.7-Code.yaml`, `openai/gpt-oss-120b.yaml`. https://github.com/vllm-project/recipes
2. ggml-org/llama.cpp @`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (2026-10-09) — `src/llama-arch.cpp:9-165`, `src/models/glm5-next.cpp:1-58`, `src/llama-memory-hybrid-idx.cpp:56-58,581-626`, `conversion/glm.py`, `conversion/mimo.py:16,177-251`, `conversion/deepseek.py:525,1037`, `conversion/qwen4exp.py`.
3. huggingface/transformers @`536ecc007387a50e77603bb5d92100e9b07514cc` — `models/glm5_next/configuration_glm5_next.py:100-202`, `mimo_v2_flash/configuration_mimo_v2_flash.py:78-133`, `deepseek_v4/configuration_deepseek_v4.py:125-159`, `qwen3_5_moe/configuration_qwen3_5_moe.py:90-112`, `minimax_m3_vl/…:81-118`, `step3p7/…:144-183`, `mistral4/…:75-109`, `nemotron_h/…:88-135`.
4. vllm-project/vllm @`ed7ae705a1f366c2008964af562a7543e91d934e` — `vllm/models/{glm5next,deepseek_v41,deepseek_v4,qwen4_exp,minimax_m3,inkling,kimi_k3,hy_v4,dots3_note}`, `vllm/model_executor/models/{mimo_v2.py,qwen3_5.py,nemotron_h.py,step3p7.py,laguna.py,mistral_large_3.py}`.
5. sgl-project/sglang @`46c2585faa4fcb6fa8febf4d2d4534a7cf33da79` — `python/sglang/srt/models/` (`glm5_next.py`, `mimo_v2.py`, `deepseek_v4.py`, `qwen4_exp.py`, `inkling.py`, `minimax_m3.py`, `step3p7.py`, `nemotron_h.py`).
6. ikawrakow/ik_llama.cpp @`5194a9e1451b885e24dd5a065f650864e8807c59` — `src/llama-arch.cpp:89-92` (`glm5next`, `glm5-next`), `deepseek41`, `qwen4exp`, `mimo2`, `minimax-m3`, `mistral4`; поддержка Engram в `src/graphs/build_deepseek4.cpp`.
7. kvcache-ai/ktransformers @`2ac8cf9c88fa7d6de106bcc3ab3ef7645e7476a0` — `doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md`, `AVX2-Tutorial.md`, `README.md:24-33`.
8. turboderp-org/exllamav3 @`151539c77abc7ab7425d30da7a4e8e3c5c154e7b` — `exllamav3/architecture/`.
9. NVIDIA/TensorRT-LLM @`49ec5730428f8eb0434e01dc9b4c532136d6c578` — `docs/source/models/supported-models.md:13-100`.
10. zai-org/GLM-5 @`c8ad661c6cf4cb0a78064987bc42f97e14355929` — `README.md:18-97`, `resources/bench_53.png` (бенчмарки GLM-5.3-Flash), `LICENSE` (Apache-2.0).
11. QwenLM/Qwen3.8 (тот же SHA отдают адреса Qwen3.5 и Qwen3.6) @`2ea10dc725823bf7c3e21ce8557cbe15245132ae` — `README.md:48-55`.
12. Niko1221/Strata @`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (HEAD; пин репозитория — c499bd1) — `README.md:12-135`, `docs/MODELS.md:25-146`, `include/strata/core/layout.hpp:28-64`, `docs/COMMUNITY_BENCHMARKS.md:14-24`, `docs/MULTI_GPU.md:84-197`, `docs/DETAILS.md:123`, `docs/HOW_IT_WORKS.md:72-77`, `bench/results/2026-10-01-community-rtx-pro-4500/README.md:1-20`.
13. sergqwer/strata-glm @`ed37419fccd0c52e07d26d526a29c2098f547843` (= пин) — `README.md:1-80, 93-94`.
14. stepfun-ai/Step-3.7-Flash @`e2132efea2836f5d1638f8145e2178b2f6c9c747` — `README.md:12-50, 414-416`, `LICENSE`.
15. MiniMax-AI/MiniMax-M3 @`79882a353ea7d8b3b52ecaf6523ba7ab2a6fb6e5` — `README.md:17-74`.
16. MoonshotAI/Kimi-K3 @`3cb39dfd32e51c3328e2e4b4af21341247d06c43` — `README.md:30-37, 601-603`, `LICENSE`.
17. Этот репозиторий @`eb59224`: `third_party/deepseek-v41-flash/{README.md,config.json,inference/config.json,LICENSE,PIN.json}` (HF-ревизия `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, вендорено CI побайтно).
18. Этот репозиторий: `AGENTS.md`, `docs/DECISIONS.md` (ADR-001, -010), `models/*.json`, `docs/models/*.md`, `hardware/4x4080s-32g.json` — все без изменений между `eb59224` и `0662572` (проверено `git diff --stat`). Добавленные в `125f35f` файлы `hardware/target-4x4080-32g.unverified.json` и `models/specs/glm-5.3-flash.json`.

Вторичные (только фрагменты WebSearch, страницы не загружены; REPORTED):

19. https://artificialanalysis.ai/models/open-source/large — v4.3.2: GLM-5.3-Flash 42; MiMo-V2.6-Pro 46, GLM-5.3 45, Kimi K3 44.
20. https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3 — v4.3 от 2026-09-07, состав оценок.
21. https://artificialanalysis.ai/models/mimo-v2-6-flash — 38.
22. https://artificialanalysis.ai/models/qwen3-8-flash-next — 40; 56 в версии v4.1.1.
23. https://artificialanalysis.ai/models/glm-5-3-flash — 57 (ранняя версия).
24. https://www.unite.ai/xiaomis-new-flagship-model-leads-open-weight-rankings-with-a-score-of-46/
25. https://codersera.com/blog/qwen-3-8-flash-complete-guide-2026/ — таблица вендора Qwen3.8-Flash-Next; 39.82 против 41.81.
26. https://www.datacamp.com/blog/qwen3-8-flash-next — DeepSWE 58.7; 262K / 1M.
27. https://unsloth.ai/docs/models/glm-5.3-flash — таблица GGUF и KLD.
28. https://unsloth.ai/docs/models/deepseek-v4 — UD-Q4_K_XL 155.1 GB, KLD 0.0102.
29. https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/discussions/11 — KLD 0.01324.
30. https://unsloth.ai/docs/models/qwen3.8-next — UD-Q4_K_XL 111.3 GB, top-1 92.255%, KLD 0.0469.
31. https://openrouter.ai/z-ai/glm-5.3-flash — AA-LCR 80.0%, контекст 1 048 576.
32. https://openrouter.ai/xiaomi/mimo-v2.6-flash — AA-LCR 74.3%, TB 22.7%.
33. https://www.vals.ai/models/xiaomi_mimo-v2.6-flash — 24.24%.
34. https://computingforgeeks.com/xiaomi-mimo-v2-6-pro-flash/ — DeepSWE 67.9 против 65.7.
35. https://www.mindstudio.ai/blog/mimo-v2-6-flash-rl-efficient-model — TB4.0 28.8.
36. https://codersera.com/blog/xiaomi-mimo-v2-6-flash-distill-9b-guide-2026/ — TB2.1 87.6.
37. https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4 — страница видна в поиске, содержимое не прочитано.
38. https://datanorth.ai/news/deepseek-releases-deepseek-v4-flash-0731 — 2026-07-31, MIT.
39. https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash-0731 — 07/30/2026, 304B.
40. https://thenextweb.com/news/deepseek-v4-1-flash-launch-v4-pro-retired-price-cut — 2026-09-10.
41. https://openrouter.ai/xiaomi/mimo-v2.6-flash и https://letsdatascience.com/news/xiaomi-releases-mimo-v26-open-weight-multimodal-models-d436ddef — 09-21 или 09-22, MIT.
42. https://runtimewire.com/article/mira-murati-thinking-machines-inkling-small-open-weight-model — Apache-2.0, AA 40, SWE-V 80.2, TB2.1 64.7 против 55.
43. https://arxiv.org/pdf/2604.12374 (только фрагмент поиска) и https://artificialanalysis.ai/articles/nvidia-nemotron-3-super-the-new-leader-in-open-efficient-intelligence — геометрия Nemotron-3-Super, AA 36.
44. https://x.com/ArtificialAnlys/status/2034960206736892365 — Mistral Small 4: 27; Qwen3.5-122B: 42.
45. https://x.com/ArtificialAnlys/status/2062381047212638697 — Step 3.7 Flash: 43.
46. https://inferencex.semianalysis.com/model/qwen-3-5, https://apxml.com/models/qwen35-122b-a10b — геометрия Qwen3.5.
47. https://pasqualepillitteri.it/en/news/14642/qwen3-8-flash-next-local-qwen4-preview — условия Qwen Community License 1.0.
48. https://benchlm.ai/blog/posts/kimi-k3-is-open-you-cannot-run-it, https://llmgateway.io/blog/kimi-k3-open-weights
49. https://artificialanalysis.ai/models/releases/minimax-m3, https://howaiworks.ai/blog/minimax-m3-open-source-hugging-face — лицензия MiniMax Community.
50. https://www.cometapi.com/models/zhipuai/glm-5-3-flashx/, https://www.orcarouter.ai/de/blog/glm-5-3-flashx-release — FlashX как хостинговый тариф; GLM-5.3-Flash под MIT, веса 08-27.
51. https://www.eesel.ai/blog/minimax-m3-1-flash — только API.
52. https://benchlm.ai/benchmarks/swe-bench-pro — Qwen3.8-Flash-Next 62.5; заметка про аудит 30% задач.
53. https://www.datacamp.com/blog/glm-5-3-flash — дрейф после ~700K.
54. https://artificialanalysis.ai/models/releases/comparisons/deepseek-v4-1-flash-vs-mimo-v2-5-pro — DeepSeek-V4.1-Flash 39.
55. https://openclawdc.com/blog/can-i-run-mimo-v2-6-flash-locally/ (165.53 GiB), https://book.st-hakky.com/data-science/mimo-v26-flash-vram-guide (`ggml-org/MiMo-V2.6-Flash-RL-GGUF:MXFP4`).
56. https://news.hada.io/topic?id=27315 — пересказ гайда unsloth по Qwen3.5: 4 бита ~256 GB RAM; 80.5 против 81.3.
57. https://artificialanalysis.ai/models/qwen3-8-27b — 34 (xhigh, v4.3.2).

Расчёты: `/tmp/claude-0/-home-user-408032gb/bb137142-5fe7-5c4d-85b1-fcda37487f8d/scratchpad/calc/fit.py`, `fit2.py`. Это файлы песочницы, их вывод вставлен в §6 без изменений; в репозиторий они не добавлены.
