# Архитектурные решения (ADR)

Метки: CONFIRMED / PROVISIONAL / UNKNOWN (см. AGENTS.md). Пины: Strata `c499bd1`, llama.cpp `ec7630a`.

---

## ADR-001. Первая цель — GLM-5.3-Flash, затем MiMo-V2.6-Flash, затем DeepSeek-V4.1-Flash

**Статус:** принято (lead-architect, 2026-10-01).

**Почему GLM первым:** по коду он ближайший родственник модели, под которую написан Strata.

| признак | Qwen3.8-Flash-Next (Strata) | GLM-5.3-Flash | метка |
|---|---|---|---|
| схема слоёв | 36 GDN + 12 QSA, разреженный слой при `layer % 4 == 3` | 34 KDA + 11 DSA, разреженные слои 3, 7, …, 43 | CONFIRMED: `include/strata/core/layout.hpp`, HF `config.json` |
| линейное attention | GDN, скалярный gate на голову | KDA, gate на каждый канал | CONFIRMED: `ggml.h` — `ggml_gated_delta_net` принимает gate `[1,H]` или `[S_v,H]` |
| индексер разреженного attention | k-pool 4, top-k 2048, dim 128, 4 головы | k-pool 4, top-k 2048, dim 128, 32 головы | CONFIRMED: `qsa.hpp:95-112`, HF `config.json` |
| остаточный поток | hyper-connections, 4 потока | mHC, 4 потока + Sinkhorn (20 итераций) | CONFIRMED: `gr.hpp`, HF `config.json` |
| общие builder'ы в llama.cpp | `build_inp_kpool`, `build_input_k_idxs`, `build_delta_net_base`, `build_gdn_l2_norm`, `build_recurrent_attn`, `build_rs`, `build_moe_ffn` | те же | CONFIRMED: `src/models/qwen4exp.cpp`, `src/models/glm5-next.cpp` |
| GGUF-кванты | есть | есть (GGUF arch `glm5-next`; HF показывает тег `glm5next`; DevQuasar) | CONFIRMED: HF DevQuasar/zai-org.GLM-5.3-Flash-GGUF |

**Что придётся написать для GLM:** KDA-gate (поканальный), attention MLA без RoPE (`qk_rope_head_dim = 0`, `kv_lora_rank = 512`) для DSA-слоёв, Sinkhorn для mHC, 3 плотных FFN-слоя (`first_k_dense_replace = 3`), роутер sigmoid + `noaux_tc` + `e_score_correction_bias` + `routed_scaling_factor = 2.5`, shared expert, `swiglu_limit = 10`, геометрия эксперта 4096×2048 (в Strata зашито 2560×640).

**MiMo вторым:** граф `mimo2.cpp` в llama.cpp есть (CONFIRMED); attention простое (SWA 128 + global); родные MXFP4-эксперты ≈150 GiB (PROVISIONAL, расчёт `tools/fitplan.py`, сверка с публичными ~150 GiB) — это ровно режим Strata без переквантования.

**DeepSeek третьим:** графа V4.1 в llama.cpp на пине нет (CONFIRMED: нет `src/models/deepseek41*`); архитектура самая новая (CED 20+20, CSA2, DSpark); routed-эксперты в FP4 занимают 268.9 GiB (CONFIRMED публично, воспроизведено fitplan), что больше VRAM+RAM (128+128 GB) → нужен переквант ≤3 бит; Engram 189 GiB → SSD-ярус.

---

## ADR-002. Новый backend рядом с `qwen4exp`, а не параметризация всего Strata

**Статус:** принято.

Strata — однопородный движок (CONFIRMED): guard архитектуры `qwen4exp` (`gguf_reader.hpp:12`), геометрия — константы компиляции (`layout.hpp: ModelGeometry`), ядра экспертов прошиты `constexpr int H = 2560; FF = 640;` (`s2_expert_grouped.cu:37-38`), из GGUF читаются только `expert_count`, `expert_used_count` и RoPE (`generate.cpp:1749-1755`).

**Решение:** форк Strata с backend'ом `glm5next`, переиспользующим подсистемы `expert_source`, `expert_cache`, `pinned`, CPU-пул, MTP/verify и сервер. **Первый рефакторинг (G2)** — сделать геометрию эксперта (H, FF, раскладку блоба) параметром времени выполнения: это нужно всем трём моделям.

---

## ADR-003. Мульти-GPU: сначала конвейер Strata, TP/EP — отдельным гейтом

**Статус:** принято.

Конвейер по слоям в Strata уже есть, P2P ему не нужен (CONFIRMED: `docs/MULTI_GPU.md:9-10`). При одном потоке декода конвейер в каждый момент нагружает одну карту, поэтому при полностью резидентном IQ3 главный рычаг скорости — TP/EP на 4 карты с P2P (G6). Сначала корректность (G3–G4), потом скорость.

---

## ADR-004. CPU-путь — только AVX2 i-quant

**Статус:** принято.

EPYC 7H43 (Zen 3) не умеет AVX-512. CPU-ядра канонического Q2_0-пака Strata требуют AVX-512, native i-quant-паки работают на AVX2 (CONFIRMED: `generate.cpp:1725-1730`). Q2_0-путь не используем.

---

## ADR-005. Ярусность покупает качество

**Статус:** принято.

На 4×32 GB IQ2_XS/IQ3_XXS для GLM и MiMo помещаются в VRAM почти целиком (PROVISIONAL, `tools/fitplan.py`). Ярус RAM/CPU используется, чтобы запускать Q4-класс (~160+ GiB экспертов) вместо 2–3 бит. Выбор квантования делается в G5 по замерам качества (KLD/перплексия) и скорости, а не по вкусу.

---

## ADR-006. Режим разработки без GPU и без ПК заказчика

**Статус:** принято (lead-architect, 2026-10-02). Контекст: сервер с GPU сейчас недоступен, у заказчика только телефон.

| что | где выполняется | чем проверяется |
|---|---|---|
| корректность операций и графа | CPU: песочница агента или бесплатный раннер GitHub Actions | крошечные модели `tools/tinygen` + настоящий граф llama.cpp + дампы (G1 так и принят) |
| компиляция CUDA под sm_89 | GitHub Actions, nvcc без GPU | job `cuda-compile` в `ci/github-actions.yml` |
| корректность CUDA-ядер | бесплатный T4 в Kaggle или Colab, запускается с телефона | те же дампы; T4 = sm_75, поэтому только корректность |
| скорость, память, многокарточность | только сервер 4×4080 | G0, G4–G6 ждут сервер |
| запуск команды агентов | Claude Code в облаке из приложения Claude на телефоне | репозиторий подключается через GitHub, агенты из `.claude/agents` |

Следствие: G2–G3 делаются CPU-first. CPU-путь экспертов Strata (AVX2) и наш backend проверяются на
крошечных моделях без GPU; CUDA-ядра компилируются в CI, их корректность проверяется на T4.
Числа скорости до появления сервера не публикуются.

---

## ADR-007. Основа GLM-движка — sergqwer/strata-glm, а не свой backend с нуля

**Статус:** принято (lead-architect, 2026-10-02). Заменяет решение ADR-002 для GLM; для MiMo и DeepSeek ADR-002 остаётся.

**Что это.** [sergqwer/strata-glm](https://github.com/sergqwer/strata-glm) @`ed37419` (MIT, 334 коммита, в архиве):
готовый движок GLM-5.3-Flash NVFP4 на Strata под одну RTX 5090 + 128 GB RAM + 2 NVMe. По данным автора (README, PROVISIONAL для нас):
декод 16.5–18 tok/s, примерно 6 раз быстрее llama.cpp; префилл около 1000 tok/s; KL к BF16 = 0.017–0.021.

**Почему берём (CONFIRMED 2026-10-02):**
- все 7 GLM-исходников `src/glm/*.cu` компилируются под **sm_89** nvcc 13.4 без правок: NVFP4 декодируется через `__nv_cvt_fp4x2_to_halfraw2`, а не через FP4 tensor cores; остальное — WMMA fp16 и `mma.sync` f16/tf32 (sm_80+);
- размеры совпадают с нашей геометрией: 12 096 экспертов × 14.16 MB, пак ~171 GB (`tests/test_fitplan.py::StrataGlmSizes`);
- в Strata есть разбиение слоёв по нескольким GPU (`bench/results/2026-09-29-layer-split`: побайтовое совпадение с одной картой).

**Пробелы под наш сервер (что делаем мы):**

| пробел | факт | работа |
|---|---|---|
| многокарточность | в GLM-пути нет выбора устройства и P2P (CONFIRMED: 0 упоминаний в `src/glm`) | подключить GLM к разбиению слоёв Strata: 42 MoE-слоя на 4 карты, у каждой свой VRAM-ярус и свой PCIe |
| Linux | дисковый ярус только под `_WIN32`; `_fseeki64` не компилировался | патчи 0001 и 0003 (`O_DIRECT` `pread`), функциональный тест; диск нужен для одной карты (163 GiB > VRAM одной карты + RAM) |
| CPU Zen 3 без AVX-512 | AVX-512 ядра выбираются в рантайме (`cpu_avx512_ok`) | проверить, что GLM-путь не требует их при `--cpu-share 0` |
| полная сборка под sm_89 на Linux | CONFIRMED: CI 36956571033, 26 кубинов sm_89 | job `strata-glm-sm89` в CI |

**Почему это меняет картину скорости.** У автора узкое место — промахи мимо VRAM (13% экспертов в VRAM, 55–58% попаданий) и ожидание диска.
На 4×32 GB в VRAM помещается 65–69% экспертов, остальные 51–58 GiB — в RAM, диск не нужен
(PROVISIONAL: `tools/fitplan.py models/glm-5.3-flash.json --quant NVFP4`). Сколько это даст в tok/s,
покажут только замеры на сервере (G0, G4).
