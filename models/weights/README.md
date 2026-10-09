# Веса шорт-листа (ADR-012 D2) — манифесты для доставки на сервер

Манифест = репозиторий HF + **закреплённая ревизия (40-hex)** + список файлов с размерами и sha256 (LFS oid). Строится на машине с
интернетом (`scripts/offline/weights.py manifest …`), проверяется на сервере офлайн (`… verify …`). После скачивания каждый GGUF
проходит `tools/gguf_quant_audit.py` (маршрутизируемые эксперты ≥ 4 бит).

| модель | репозиторий (метка) | вариант | манифест | статус |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | `unsloth/Qwen3.8-Flash-Next-GGUF` @`38bb39e…` (VERIFIED как ссылка в Strata docs/UNSLOTH_Q4.md:8) | UD-Q4_K_XL, 4 файла, 111.3 GB | `qwen3.8-flash-next-UD-Q4_K_XL.manifest.json` (PROVISIONAL: переписан из документации Strata, не из API HF) | перегенерировать с API |
| GLM-5.3-Flash | unsloth GGUF, точное имя репозитория и ревизия — UNKNOWN здесь (R1: UD-Q4_K_XL 199.71 GB, REPORTED) | UD-Q4_K_XL | — | построить на ПК |
| DeepSeek-V4-Flash-0731 | unsloth GGUF — UNKNOWN здесь (R1: UD-Q4_K_XL 155.1 GB, REPORTED) | UD-Q4_K_XL | — | построить на ПК |
| MiMo-V2.6-Flash-MOPD | GGUF MXFP4 для MOPD — UNKNOWN (есть `ggml-org/MiMo-V2.6-Flash-RL-GGUF`, REPORTED; RL не рекомендован) | MXFP4 | — | выяснить |

Из этой облачной сессии `huggingface.co` недоступен (прокси 403), поэтому манифесты с API не строились — NOT RUN.
