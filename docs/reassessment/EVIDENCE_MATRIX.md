# Evidence Matrix — утверждения, на которых держится ADR-012 (2026-10-09, редакция 2 после ревью)

Шкала: **VERIFIED/CONFIRMED** — видели в первичном источнике (файл:строка@SHA, литеральный вывод, URL); **REPORTED** — заявление
автора/вендора/сообщества; **UNVERIFIED** — заявление заказчика о его железе, не проверенное замером (по AGENTS.md это UNKNOWN до
замера); **HYPOTHESIS/PROVISIONAL** — вывод или расчёт; **DISPROVED**; **UNKNOWN**. «Ведущий» — проверено ведущим
лично в этой сессии; иначе — субагентом (файл research/…). Колонка «закроет» — эксперимент из `EXPERIMENTS.md`/`ROADMAP.md`.
Редакция 2: правки по `REVIEW-adversarial.md` (п. 1, 3, 10, 11, 13, 14) — условия чужих замеров, база A, состав IQ4_XS-файлов,
метки «CONFIRMED формулой/арифметикой» понижены до PROVISIONAL (арифметика над PROVISIONAL-входами остаётся PROVISIONAL).

## Железо
| # | утверждение | метка | источник | закроет |
|---|---|---|---|---|
| E1 | 4× RTX 4080 Super 32 GB, sm_89 | UNVERIFIED (заявление) | мастер-промпт | H1/H3, F9–F10 |
| E2 | CPU «EPYC 7H43»: публичного SKU нет | REPORTED-отсутствие (поиск ведущего и R4) | WebSearch 2026-10-09; R4 §2.1 | F1 (CPUID) |
| E3 | каналы DDR4: 4 или 8 | UNKNOWN | hardware JSON | F4, H2 |
| E4 | P2P на штатном драйвере GeForce Ada выключен | REPORTED (NVIDIA forum) + VERIFIED чужой замер 8×4090: 4.5→25.1 ГБ/с с патчем | R4 §4.1, §4.3 (CVGL report) | F14, H6 |
| E5 | P2P-патч требует ReBAR, BAR1 ≥ VRAM, `iommu=pt`; на 4080/4080S не проверялся публично | VERIFIED (README патчей) / UNKNOWN (4080S) | R4 §4.2 | F9 (BAR1), R4.3 |
| E6 | NVMe: 3× 1 TB = 3 TB (поправка заказчика 2026-10-12; мастер-промпт говорил 4× 1 TB); модели UNKNOWN | UNVERIFIED (заявление) | мастер-промпт; поправка 2026-10-12 | F15 |
| E8 | модифицированные RTX 4080 SUPER 32 GB существуют и работают под CUDA: 32 760 MiB, Gen4 x16, H2D 24.0 ГБ/с, драйвер 616.56 | REPORTED (замер сообщества), VERIFIED как файл (ведущий) | Strata `bench/results/2026-10-04-community-rtx-4080s-iq3s/README.md`@fb58e0d | H3, H4 (наши карты) |
| E9 | Strata: split UD-Q4_K_XL рекомендует RAM ≥ GGUF + 24 GB (~135 GB), у нас 128 GB; это рекомендация установщика, а не запрет: «128 GB ran it fine (#737)» | VERIFIED как текст документации и кода (ведущий) / REPORTED (запуск на 128 GB) | Strata `docs/UNSLOTH_Q4.md:31-37`, `setup.py:1103-1124`@fb58e0d | запуск по BRINGUP-qwen38fn.md |
| E7 | GPUDirect Storage на GeForce не поддерживается | REPORTED | R4 §6.4 | — (проектируем через O_DIRECT + pinned) |

## Модели
| # | утверждение | метка | источник | закроет |
|---|---|---|---|---|
| E10 | GLM-5.3-Flash: 45 слоёв (34 KDA + 11 DSA), 288×top-8, FF 2048, hidden 4096 | VERIFIED | `docs/models/glm-5.3-flash.md`; llama.cpp `src/models/glm5-next.cpp:60-155`@ec7630a (ведущий) | — |
| E11 | GLM: всего 321.3B, активных ≈16.7B без эмбеддинга | PROVISIONAL (сумма форм тензоров ведущим) = REPORTED 321.3B из PR #27754 | `models/specs/glm-5.3-flash.json`, `tests/test_roofline.py` | — |
| E12 | GLM UD-Q4_K_XL = 199.71 GB (185.99 GiB), top-1 92.2%, KLD 0.049 | REPORTED (unsloth) + совпадение размера с замером 3090-хоста | R1 §4; `PROBLEMS.md` | R1.2, Q1 |
| E13 | Qwen3.8-Flash-Next: 125B + 51.2B n-gram, 6B active, 262K нативно | REPORTED (поиск ведущего: несколько источников согласны) + VERIFIED геометрия (Strata layout.hpp) | WebSearch 2026-10-09; `docs/strata-map.md` | — |
| E14 | Qwen3.8-Flash-Next UD-Q4_K_XL = 111.3 GB (4 файла) | VERIFIED (ведущий) | Strata `docs/UNSLOTH_Q4.md:54-57`@fb58e0d | R1.3 sha256 |
| E15 | лицензия Qwen Community License 1.0 с ограничениями для MaaS/«work assistant» | REPORTED | R1 §7; WebSearch | R2.4 (заказчик) |
| E16 | индекс AA v4.3: GLM 42, Qwen3.8-FN 40, DSV4.1 39, MiMo 38 | REPORTED (поисковая выдача; версии индекса конфликтуют) | R1 §2.2 | R2.2 (свой агентный набор) |
| E17 | DeepSeek-V4.1-Flash в 4 бит не помещается в VRAM+RAM (эксперты ≈ 276–310 GiB) | PROVISIONAL (fitplan, roofline, R1) | `ROOFLINE.md` | — |
| E18 | DeepSeek-V4-Flash-0731 UD-Q4_K_XL 155.1 GB, эксперты бит-в-бит родные FP4 | REPORTED | R1 §4 | R1.2, Q1 |
| E18a | DeepSeek-V4-Flash-0731: качество на индексе AA | UNKNOWN (на странице AA конфликт 25/34) | R1 §2.2, `research/R1-models.md:66` | R2.2 |
| E18b | Qwen3.8-Flash-Next **UD-IQ4_XS** — не 4-битный: маршрутизируемые эксперты IQ3_S gate/up (IQ4_XS в одном слое), IQ4_NL down | VERIFIED как текст (ревьюер) | Strata `docs/UNSLOTH_Q4.md:234-236`@fb58e0d; `REVIEW-adversarial.md` п. 1 | исключён из шорт-листа |
| E18c | GLM-5.3-Flash **UD-IQ4_XS** (156.82 GB): средний bpw маршрутизируемых экспертов ≤ 3.99 при любом допущении о плотной части ≥ 4.25 bpw | PROVISIONAL (арифметика ревьюера на `models/specs/glm-5.3-flash.json`; размер REPORTED) | `REVIEW-adversarial.md` п. 1 | исключён; правило — `tools/gguf_quant_audit.py` |
| E18d | Запасные модели класса «всё в VRAM, 262K нативно»: Qwen3.5-122B-A10B (Apache-2.0), Ling-3.0-flash, Nemotron-3-Super | REPORTED (карточки); качество на задачах заказчика UNKNOWN | `research/R1-models.md:46-53` | R1.3, R2.2 |
| E19 | MiMo-V2.6-Flash: рекомендован MOPD (RL повторяет tool-call) | VERIFIED (рецепт vLLM, субагент) | R1 §2.1 | — |

## Движки
| # | утверждение | метка | источник | закроет |
|---|---|---|---|---|
| E20 | llama.cpp: GPU-кэш экспертов из RAM `--moe-cache-mib`, LRU, копия по PCIe при промахе; мульти-GPU | VERIFIED (ведущий) | `d6cf9acb2`, `c811cb8f0`; `src/llama-moe-cache.cpp:15,72`@10a60cf30 | R3.3 |
| E21 | TP для `glm5-next` нет ни в llama.cpp, ни в ik_llama | VERIFIED (субагент, файл:строка) | R2 §2.3 | R4.2 |
| E22 | vLLM: custom all-reduce выключен на >2 PCIe-GPU без NVLink | VERIFIED (субагенты R2, R3, код @ed7ae70) | `custom_all_reduce.py:270-291` | R4.1 |
| E23 | vLLM DSA-индексер требует DeepGEMM (Hopper/Blackwell) → GLM на sm_89 не запустится | VERIFIED (требование) / HYPOTHESIS (следствие) | R2 §1 п. 4 | R1.4 |
| E24 | ExLlamaV3 v1.6.0: TP/EP без NCCL/P2P, GLM5-Next/MiMo-V2/DeepSeek-V4 | VERIFIED (README, субагент) / REPORTED (работа) | R2 | R4.2 |
| E25 | KTransformers для GLM-5.3: нужен AVX-512 FP8 и ≥ 350 GB RAM | VERIFIED (туториал, субагенты R1–R3) | R2/R3 | — (DISPROVED для нашего сервера как есть) |
| E26 | Strata upstream: +1552 коммита от пина, только Qwen3.8-FN | VERIFIED (ведущий: `git rev-list --count`) | `c499bd1..fb58e0d` | — |
| E27 | Strata Qwen3.8-FN UD-Q4_K_XL: 61.5/72.1/70.2 tok/s @2.7K/64K/128K на 1×RTX PRO 4500 32 GB (Blackwell, GDDR7) + TR PRO 3975WX (AVX2). **Условия:** MTP `--spec 4 --spec-min-p 0.5` включён (обязателен в serve-режиме той версии), `--kv int8`, 512 GB RAM, `--resident-budget-gib 100` (70% экспертов в RAM), окна процессов 8 192 / 81 920 / 139 264 (не 262K), синтетические code-промпты, один пользователь. Сравнивать только с границей `--spec-k`, не с k=0 | REPORTED (замер пользователя), VERIFIED как файл (ведущий; условия — ревьюер) | Strata `bench/results/2026-10-01-community-rtx-pro-4500/README.md`@fb58e0d; `REVIEW-adversarial.md` п. 3 | B1 |
| E28 | Strata Qwen3.8-FN UD-Q4_K_XL: 64–78 tok/s на 2×3090 split, 55 при 128K | REPORTED, VERIFIED как текст (ведущий) | Strata `docs/UNSLOTH_Q4.md:34-35`@fb58e0d | B1 |
| E29 | ik_llama.cpp: GLM-5.3-Flash UD-Q4_K_XL на 1×3090 + RAM: tg 15.0, pp 166.8 tok/s (8K) | PROVISIONAL (замер оператора чужого хоста) | `PROBLEMS.md` | B1 |
| E30 | база B (lighttransport) не исполняет Q4_K-экспертов (только перепаковка в Q2/Q3), ≤ 2 GPU | VERIFIED (субагент, код) | R2 | — (заморожено) |
| E31 | llama.cpp, Qwen3.8-FN UD-Q4_K_XL: 2×5090 — 53.2 / 46.1 tok/s при 10K / 87K (36.1 на 87K в другом прогоне); форк на 6×4090 (layer split, q8_0 KV) — 21.0 → 48.7 tok/s при 250K после доработок | REPORTED | `research/R2-engines.md` §4, строки 14 и 17 таблицы | B1 |
| E32 | **база A** (strata-glm-3090 `794cd69`, реальный NVFP4-пак 171.2 GB, 1×3090 + EPYC, 251 GiB): декод 6.85 → **13.89 tok/s** с `--cpu-share 1.0` (RAM-эксперты считает CPU — политика D7), 14.12 со `--spec 3` (принятие 55.6%), prefill ~381–401 tok/s | REPORTED | `PROBLEMS.md`@794cd69 форка; `research/R2-engines.md` §3.4 | B1a (R1.5) |
| E33 | повтор эксперта от предыдущего токена у GLM на реальных весах — **24.6%** («плоская» маршрутизация): довод ПРОТИВ большого выигрыша кэша | REPORTED | там же | M1 |
| E34 | GLM-5.3-Flash в «IQ4_XS»-паках на 3090-классе с выгрузкой: 6–20 tok/s | REPORTED; **к требованию ≥ 4 бит напрямую не относится** (состав файлов неизвестен, вероятно частично 3-бит, см. E18c) | `research/R2-engines.md` §4, строки 1, 4, 5 | — |

## Физика и наши инструменты
| # | утверждение | метка | источник | закроет |
|---|---|---|---|---|
| E40 | конвейер по слоям при batch=1 ≈ полоса одной карты | PROVISIONAL (модель) + REPORTED (Strata MULTI_GPU: «декод на уровне более быстрой карты») | `ROOFLINE.md`, R3 §2.1 | B1, P1 |
| E41 | GLM UD-Q4_K_XL (4.89/8.04 bpw): граница конвейера без спекуляции ≤ ~42 tok/s при любом попадании; со спекуляцией (принятие 0.7) — огибающая 50.6–52.9 при попадании ≥ ~0.8 и только с перекрытием CPU/GPU, без перекрытия ≤ ~47; при TP латент MLA реплицируется, TP@auto ≈ 41 | PROVISIONAL | `tools/roofline.py` (поле `envelope`), `ROOFLINE.md` §2–3 (ред. 3) | B1, M1 |
| E42 | промахи выгоднее считать на CPU, чем копировать по PCIe (batch=1) | PROVISIONAL (следует из полос 25 против ~140 ГБ/с; тест `test_cpu_compute_beats_pcie_copy_for_batch_one` проверяет **код формулы, не физику** — при rbw > pbw он истинен по построению). Отдельно: порог `GGML_OP_OFFLOAD_MIN_BATCH=32` в llama.cpp — VERIFIED как код (R3) | `tests/test_roofline.py`; R3 | M4 |
| E43 | калибровка: ik_llama 15.0 tok/s против границы 15.2 (4 канала × КПД 0.9) или 16.5 (8 каналов × КПД 0.5) при `--overlap no` — **каналы и КПД по одному числу не разделимы**; переносить на «всё в VRAM» и делать порогом приёмки нельзя | PROVISIONAL | `ROOFLINE.md` §1 (ред. 2); `docs/evidence/2026-10-09-roofline-calibration-and-mutations.md` | R0 membw + B1 |
| E44 | MiMo при 262K читает ≈ 5.6 GiB KV на токен (f16) | PROVISIONAL (геометрия частично PROVISIONAL) | `models/specs/mimo-v2.6-flash.json` | K4 |
| E45 | трассировщик маршрутизации и симулятор кэша работают на реальном графе llama.cpp (CPU) | CONFIRMED (ведущий) | `docs/evidence/2026-10-09-g0-trace-cpu.md` | — |
| E46 | маршрутизация зависит от формы батча (на случайных весах: 0% в слоях 1–4, до 5.7% токенов в самом глубоком слое) | CONFIRMED на крошечной модели; для реальной UNKNOWN | там же | M1 |
| E47 | таблица типов `tools/gguf_quant_audit.py` = gguf-py @ec7630a | CONFIRMED (тест) | `tests/test_gguf_quant_audit.py` | — |
| E48 | GLM в llama.cpp хранит на токен индексера `key`, `gate`, `pooled` = 3×128 в 12 слоях (с MTP): ~8.05 GB при 262K против 3.75 GiB в roofline — на 2–4 GiB меньше места под эксперты | VERIFIED как код (ревьюер: `src/llama-memory-hybrid-idx.cpp:56-58`@79e2e74) / PROVISIONAL (размер) | `REVIEW-adversarial.md` п. 14; R1 §3 | R1.5 замер VRAM |
| E49 | граница TTFT при 262K — десятки секунд (GLM ~50 с, MiMo ~98 с) при FP16/FP32-акк. тензорных 104.5 TFLOPS × 0.5; для движков с INT8-префиллом (llama.cpp MMQ) — не нижняя граница | HYPOTHESIS (TFLOPS выведены из whitepaper 4090, R4 §3.2) / PROVISIONAL (граница) | `ROOFLINE.md` §4 п. 6 | R1 замер TTFT |

## Исторические утверждения, изменившие статус
| # | было | стало | источник |
|---|---|---|---|
| E50 | «P2P works, CONFIRMED (stated by the customer)» | UNKNOWN; на штатном драйвере — против (E4) | REPO_AUDIT C2, R4 |
| E51 | «3× Samsung 990 Pro» | число 3 совпадает с поправкой 2026-10-12 (3× 1 TB); модель 990 Pro UNVERIFIED | REPO_AUDIT C1 |
| E52 | «strata-glm в архиве» (ADR-007) | на GitHub не archived (REPORTED, R2) | R2 |
| E53 | «ik_llama не знает `glm5-next`» (PROBLEMS.md) | алиас есть в исходниках @5f89bfc (VERIFIED, R2) | R2 |
| E54 | strata-glm 19.65 tok/s на 3090 как ориентир скорости | синтетический пак (40/42 слоёв нули) — не ориентир | REPO_AUDIT §2 |
| E55 | «GLM active 17–18B: UNKNOWN» (ТЗ) | ≈16.7B без эмбеддинга, 17.4B с ним (PROVISIONAL) | E11 |
| E56 | «единственный замер на реальных весах — ik_llama 15.0» (ред. 1 этой переоценки) | второй замер — база A 13.89 tok/s на том же хосте (E32) | `REVIEW-adversarial.md` п. 10 |
| E57 | «UD-IQ4_XS может пойти без новых ядер, если все его экспертные тензоры — IQ4_XS» (`research/R2-engines.md` §3.3) | DISPROVED как путь к ≥ 4 бит: у Qwen эксперты IQ3_S (E18b), у GLM средний bpw экспертов ≤ 3.99 (E18c) | `REVIEW-adversarial.md` п. 1 |
| E58 | «TTFT при 262K — минуты» (ROOFLINE ред. 1) | десятки секунд: в ред. 1 TFLOPS взяты для FP32 CUDA-ядер (52.2) вместо тензорных | `ROOFLINE.md` ред. 2; `REVIEW-adversarial.md` п. 5 |
| E59 | «Strata Qwen 61.5/72.1/70.2 — тот же порядок, что граница × 0.7» (ROOFLINE ред. 1) | сравнение несопоставимых величин (замер со спекуляцией и KV int8 против границы без спекуляции, калибровка с другого движка) — снято | `REVIEW-adversarial.md` п. 3–4 |
