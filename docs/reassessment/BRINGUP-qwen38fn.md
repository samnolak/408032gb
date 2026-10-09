# Запуск кандидата №1: Qwen3.8-Flash-Next UD-Q4_K_XL на 4×4080 32 GB (шаблон, NOT RUN)

Источник всех утверждений — документация Strata @`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (прочитана ведущим 2026-10-09):
`docs/MULTI_GPU.md`, `docs/UNSLOTH_Q4.md`, `docs/DETAILS.md`, `bench/results/2026-10-04-community-rtx-4080s-iq3s/README.md`.
Это заявления автора/сообщества (REPORTED), не наши замеры. Флаги сверить с `--help` той версии, которую соберёте.

## Что известно о движке (REPORTED по документации Strata)
- Разбиение по слоям на 2–4 карты (`--gpus 0,1,2,3`, `--layer-split auto` перебирает размещения для ≤ 4 карт с 0.1.40), P2P не
  нужен: один переход между картами на окно проверки через pinned RAM (`MULTI_GPU.md:3-12,134`).
- Каждая карта держит копию плотных весов и свой кэш экспертов своих слоёв; **арена всех экспертов — в RAM хоста** (`MULTI_GPU.md:143-148`).
- Для UD-Q4_K_XL на нескольких GPU setup требует RAM ≥ размер GGUF + 24 GB (~135 GB), потому что при split все 77 GB экспертов
  грузятся в RAM (`UNSLOTH_Q4.md:31-37`). **У нашего сервера 128 GB.** По коду установщика это рекомендация, а не запрет:
  «#737: a recommendation, not a wall - 128 GB ran it fine»; при нехватке setup спрашивает, `--gpus … --yes` продолжает
  (`setup.py:1103-1124`@fb58e0d, VERIFIED чтением; сам запуск на 128 GB — REPORTED, issue #737). Риск: подкачка/остановки при
  загрузке (страничный кэш GGUF) — смотреть `free -g` и `MemAvailable` во время старта.
- Замеры сообщества: 2×RTX 3090 split UD-Q4_K_XL — 64–78 tok/s, 55 tok/s на промпте 128K (`UNSLOTH_Q4.md:34-35`);
  1×RTX PRO 4500 32 GB + AVX2 Threadripper — 61.5/72.1/70.2 tok/s при 2.7K/64K/128K (bench/results …rtx-pro-4500);
  **1×RTX 4080 SUPER 32 GB (модифицированная, 32 760 MiB, Gen4 x16, H2D 24.0 ГБ/с, драйвер 616.56)** на IQ3_S — 89–122 tok/s при
  промптах ~44–48K (`DETAILS.md:601-611`; это 3-бит, к нашему требованию не относится, но подтверждает, что такие карты работают).
- KV при длинном контексте: 8-битный KV выше 4K, «KV streaming» (`--kv-resident N`) переносит KV в RAM от 64K
  (`DETAILS.md:17-19,79`); на 4 картах по 32 GB, вероятно, можно держать KV 262K в VRAM — проверить.
- `--compat-bf16` обязателен для пака из UD-Q4_K_XL; 195 малых Q8_0-проекций округляются в BF16 (макс. ошибка 0.0144) —
  это расхождение с файлом Unsloth, а не с оригиналом (`UNSLOTH_Q4.md:84-96`).

## Шаги (на сервере; всё NOT RUN)
1. Веса: `scripts/offline/weights.py verify models/weights/qwen3.8-flash-next-UD-Q4_K_XL.manifest.json --dest <nvme>/qwen38fn`
   (манифест сначала перегенерировать с API HF — см. `models/weights/README.md`), затем
   `python3 tools/gguf_quant_audit.py <nvme>/qwen38fn/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` — ожидаем PASS
   (все маршрутизируемые эксперты ≥ 4 бит: по документации Q4_K, в слое 2 Q5_K).
2. Сборка Strata на пине (Linux, CUDA, sm_89) и MTP-слоя по `docs/ORCA.md` Strata.
3. Пак: `tools/iq_pack.py --gguf …-00001-of-00004.gguf --out packs/ud-q4_k_xl --compat-bf16`.
4. Запуск сервера: шаблон JSON из `UNSLOTH_Q4.md` (разд. «The server») с изменениями: `"gpu": [0,1,2,3]`, `"layer_split": "auto"`,
   **без** `--resident-budget-gib` (split его не поддерживает), `--max-context 262144`, `--spec 4 --mtp mtp/rt`.
   Записать стартовые строки движка: `layer split: …`, `MiB of VRAM free with everything loaded`, RAM (`free -g`) после загрузки.
5. Матрица: `tools/bench/run_matrix.py` (2K/32K/128K/262K × 128/512/2048, cold/warm, 3 повтора) и `tools/bench/probes.py` на 262K.
6. Запуск через setup: `./setup.sh --family unsloth --model UD-Q4_K_XL --gpus 0,1,2,3 --yes` (флаги сверить с версией).
   Если split не стартует из-за RAM — зафиксировать литеральную ошибку; варианты: (а) llama.cpp `qwen4exp` целиком в VRAM
   (`-ngl 999 -sm layer`, n-gram-таблица — проверить, где её держит llama.cpp); (б) вопрос к заказчику о расширении RAM.

## Что считать успехом (из ADR-012 D1)
Декод при заполненных 262K ≥ 50 tok/s single-stream (цель 80–100), пробы needle/var-tracking/code-QA на 262K ≥ 80%, TTFT тёплого
повторного запроса ≤ 10% холодного.
