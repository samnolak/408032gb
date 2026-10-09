# Evidence Matrix — утверждения, на которых держится ADR-012 (2026-10-09)

Шкала: **VERIFIED/CONFIRMED** — видели в первичном источнике (файл:строка@SHA, литеральный вывод, URL); **REPORTED** — заявление
автора/вендора/сообщества; **HYPOTHESIS/PROVISIONAL** — вывод или расчёт; **DISPROVED**; **UNKNOWN**. «Ведущий» — проверено ведущим
лично в этой сессии; иначе — субагентом (файл research/…). Колонка «закроет» — эксперимент из `EXPERIMENTS.md`/`ROADMAP.md`.

## Железо
| # | утверждение | метка | источник | закроет |
|---|---|---|---|---|
| E1 | 4× RTX 4080 Super 32 GB, sm_89 | UNVERIFIED (заявление) | мастер-промпт | H1/H3, F9–F10 |
| E2 | CPU «EPYC 7H43»: публичного SKU нет | REPORTED-отсутствие (поиск ведущего и R4) | WebSearch 2026-10-09; R4 §2.1 | F1 (CPUID) |
| E3 | каналы DDR4: 4 или 8 | UNKNOWN | hardware JSON | F4, H2 |
| E4 | P2P на штатном драйвере GeForce Ada выключен | REPORTED (NVIDIA forum) + VERIFIED чужой замер 8×4090: 4.5→25.1 ГБ/с с патчем | R4 §4.1, §4.3 (CVGL report) | F14, H6 |
| E5 | P2P-патч требует ReBAR, BAR1 ≥ VRAM, `iommu=pt`; на 4080/4080S не проверялся публично | VERIFIED (README патчей) / UNKNOWN (4080S) | R4 §4.2 | F9 (BAR1), R4.3 |
| E6 | NVMe: 4× 1 TB (не 3×990 Pro) | UNVERIFIED (заявление) | мастер-промпт | F15 |
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
| E27 | Strata Qwen3.8-FN UD-Q4_K_XL: 61.5/72.1/70.2 tok/s @2.7K/64K/128K на 1×RTX PRO 4500 + AVX2 TR | REPORTED (замер пользователя), VERIFIED как файл (ведущий) | Strata `bench/results/2026-10-01-community-rtx-pro-4500/README.md`@fb58e0d | B1 |
| E28 | Strata Qwen3.8-FN UD-Q4_K_XL: 64–78 tok/s на 2×3090 split, 55 при 128K | REPORTED, VERIFIED как текст (ведущий) | Strata `docs/UNSLOTH_Q4.md:34-35`@fb58e0d | B1 |
| E29 | ik_llama.cpp: GLM-5.3-Flash UD-Q4_K_XL на 1×3090 + RAM: tg 15.0, pp 166.8 tok/s (8K) | PROVISIONAL (замер оператора чужого хоста) | `PROBLEMS.md` | B1 |
| E30 | база B (lighttransport) не исполняет Q4_K-экспертов (только перепаковка в Q2/Q3), ≤ 2 GPU | VERIFIED (субагент, код) | R2 | — (заморожено) |

## Физика и наши инструменты
| # | утверждение | метка | источник | закроет |
|---|---|---|---|---|
| E40 | конвейер по слоям при batch=1 ≈ полоса одной карты | PROVISIONAL (модель) + REPORTED (Strata MULTI_GPU: «декод на уровне более быстрой карты») | `ROOFLINE.md`, R3 §2.1 | B1, P1 |
| E41 | GLM: ≤ ~40 tok/s в конвейере при Q8-плотной части при любом попадании | PROVISIONAL | `tools/roofline.py` | B1 |
| E42 | промахи выгоднее считать на CPU, чем копировать по PCIe (batch=1) | PROVISIONAL (формула) + VERIFIED порог `GGML_OP_OFFLOAD_MIN_BATCH=32` в llama.cpp (R3) | `tests/test_roofline.py` | M4 |
| E43 | калибровка: зрелый движок ≈ 0.70 от границы при 8 каналах | PROVISIONAL (каналы чужого хоста неизвестны) | `ROOFLINE.md` §1 | B1 |
| E44 | MiMo при 262K читает ≈ 5.6 GiB KV на токен (f16) | PROVISIONAL (геометрия частично PROVISIONAL) | `models/specs/mimo-v2.6-flash.json` | K4 |
| E45 | трассировщик маршрутизации и симулятор кэша работают на реальном графе llama.cpp (CPU) | CONFIRMED (ведущий) | `docs/evidence/2026-10-09-g0-trace-cpu.md` | — |
| E46 | маршрутизация зависит от формы батча (на случайных весах: 0% в слоях 1–4, до 5.7% токенов в самом глубоком слое) | CONFIRMED на крошечной модели; для реальной UNKNOWN | там же | M1 |
| E47 | таблица типов `tools/gguf_quant_audit.py` = gguf-py @ec7630a | CONFIRMED (тест) | `tests/test_gguf_quant_audit.py` | — |

## Исторические утверждения, изменившие статус
| # | было | стало | источник |
|---|---|---|---|
| E50 | «P2P works, CONFIRMED (stated by the customer)» | UNKNOWN; на штатном драйвере — против (E4) | REPO_AUDIT C2, R4 |
| E51 | «3× Samsung 990 Pro» | UNVERIFIED 4× 1 TB | REPO_AUDIT C1 |
| E52 | «strata-glm в архиве» (ADR-007) | на GitHub не archived (REPORTED, R2) | R2 |
| E53 | «ik_llama не знает `glm5-next`» (PROBLEMS.md) | алиас есть в исходниках @5f89bfc (VERIFIED, R2) | R2 |
| E54 | strata-glm 19.65 tok/s на 3090 как ориентир скорости | синтетический пак (40/42 слоёв нули) — не ориентир | REPO_AUDIT §2 |
| E55 | «GLM active 17–18B: UNKNOWN» (ТЗ) | ≈16.7B без эмбеддинга, 17.4B с ним (PROVISIONAL) | E11 |
