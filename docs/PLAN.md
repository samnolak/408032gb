# План работ: гейты G0–G7

Двигать гейт в ACCEPTED может только lead-architect со ссылкой на доказательства в `docs/evidence/`.

| гейт | суть | где выполняется | статус |
|---|---|---|---|
| G0 | замеры железа, гистограммы экспертов GLM, бейзлайн llama.cpp | VM (GPU) | READY TO DISPATCH: инструменты собраны, промпт `docs/prompts/G0-dispatch.md` |
| G1 | эталоны numpy для всех новых операций + parity с дампами llama.cpp | песочница + VM | IN PROGRESS: KDA/GDN совпадает с ggml (2.9e-8) |
| G2 | рефакторинг Strata: геометрия эксперта в рантайме | VM | BLOCKED by G1 |
| G3 | backend `glm5next`, 1 GPU + эксперты в RAM, совпадение токенов | VM | BLOCKED |
| G4 | 4 GPU конвейер + кэш экспертов + MTP | VM | BLOCKED |
| G5 | выбор квантования по качеству и скорости | VM | BLOCKED |
| G6 | TP/EP на 4 карты через P2P | VM | BLOCKED |
| G7 | MiMo-V2.6-Flash, затем DeepSeek-V4.1-Flash | VM | BLOCKED |

---

## G0 — разведка на железе

**Цель:** заменить UNKNOWN и PROVISIONAL в планировании на замеры.

1. **Железо** → `docs/evidence/G0-hardware.md` с литеральным выводом:
   - каналы памяти: `sudo dmidecode -t memory | grep -E "Size|Locator|Speed"`;
   - PCIe Gen и ширина каждой карты: `nvidia-smi -q | grep -A3 "Link Width\|Link Gen"`;
   - матрица P2P-пропускной способности (nvbandwidth или p2pBandwidthLatencyTest);
   - случайное чтение 4K/16K с NVMe (`fio`, QD32) — нужно для SSD-яруса Engram.
2. **Гистограммы экспертов GLM-5.3-Flash** через `tools/g0/` на llama.cpp @`ec7630a`
   и GGUF-кванте, который помещается в машину. Выход: CSV + JSON-сводка
   `tools/g0/analyze.py` (кривая «сколько экспертов на слой дают 50/80/90/95% попаданий»).
3. **Бейзлайн:** `llama-bench` для того же GGUF на 4 картах (`-sm layer`), pp512 и tg128.

**Приёмка:** три файла в `docs/evidence/` с литеральным выводом; `tools/fitplan.py`
перезапущен с измеренной кривой попаданий (`--hit-curve`).

## G1 — эталоны

Для каждой новой операции — numpy-эталон в `ref/` и тест в `tests/`, затем сверка
с дампом тензора llama.cpp по имени из `cb(...)`:

| операция | эталон | имя в llama.cpp | статус |
|---|---|---|---|
| KDA (gated delta rule, поканальный gate) | `ref/kda.py` | `kda_scan_out` | эталон = ggml_gated_delta_net (CPU) до 2.9e-8, CONFIRMED; дамп полной модели NOT RUN |
| KDA gate (lower bound) | `ref/kda.py` | `kda_g1` | эталон готов, сверка NOT RUN |
| mHC Sinkhorn | — | `build_hc_sinkhorn` | TODO |
| k-pool индексер, выбор top-k | — | `build_kpool_select` | TODO |
| MLA без RoPE (DSA-слой) | — | `build_dsa_layer` | TODO |
| роутер sigmoid + noaux_tc | — | `ffn_moe_topk` | TODO |

**Приёмка:** max относительная ошибка ≤ 1e-3 (fp32) на 3 слоях каждого типа.

## G2 — рефакторинг Strata

Геометрия эксперта (H, FF, раскладка блоба, число экспертов, top-k) — параметры
времени выполнения в CUDA и CPU (AVX2) путях. **Приёмка:** собственные `*_parity`
тесты Strata зелёные на Qwen3.8 (регрессий нет) + новые тесты на геометрии 4096×2048.

## G3 — GLM на одной карте

Backend `glm5next`: плотные части на GPU0, все эксперты в RAM, CPU считает промахи.
**Приёмка:** совпадение greedy-токенов с llama.cpp на 20 промптах × 256 токенов.

## G4 — GLM на 4 картах

Конвейер по слоям, кэш экспертов, MTP-спекуляция. **Приёмка:** совпадение токенов
как в G3; tok/s декода ≥ бейзлайна llama.cpp из G0 (CONFIRMED-замером).

## G5 — режим точности

Сравнить IQ3_XXS (почти всё в VRAM) и Q4_K (ярусно) по KLD к эталонным логитам
и по скорости. **Приёмка:** таблица с командами и выводом; решение записано в ADR.

## G6 — TP/EP

Тензорный/экспертный параллелизм на 4 карты через P2P для задержки декода.
**Приёмка:** совпадение токенов + выигрыш по tok/s против G4, замер в evidence.

## G7 — следующие модели

MiMo-V2.6-Flash (граф `mimo2.cpp`, MXFP4-эксперты), затем DeepSeek-V4.1-Flash
(Engram на NVMe, CED, CSA2, DSpark; графа в llama.cpp на пине нет).
