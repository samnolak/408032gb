# Research sweep 2026-10-09 (R1–R4)

Написаны исследовательскими субагентами ведущего; ведущий перепроверил ключевые для решения утверждения сам.

| файл | тема | что перепроверено ведущим (CONFIRMED) | оговорки |
|---|---|---|---|
| R1-models.md | ландшафт моделей, Pareto, шорт-лист | Strata HEAD `fb58e0d`; UD-Q4_K_XL Qwen3.8-Flash-Next = 4 файла, 111.3 GB (`docs/UNSLOTH_Q4.md:54-57`); замер 61.5/72.1/70.2 tok/s на 1×RTX PRO 4500 (`bench/results/2026-10-01-community-rtx-pro-4500/README.md`); 2×3090 split 64–78 tok/s (`docs/UNSLOTH_Q4.md:34-35`) | HF-карточки и config.json не читались (прокси); оценки Artificial Analysis — из поисковой выдачи (REPORTED) |
| R2-engines.md | движки, направления A–E | `d6cf9acb2` и `c811cb8f0` в llama.cpp (MoE GPU-кэш, LRU + копия по PCIe, `--moe-cache-mib`); Strata пин → HEAD = 1552 коммита | флаг `archived` у sergqwer/strata-glm получен субагентом через поиск GitHub за пределами настроенной области репозиториев сессии — считать REPORTED |
| R3-techniques.md | MoE-выгрузка, multi-GPU, KV, спекуляция, roofline | выводы о потолке конвейера совпали с независимым `tools/roofline.py` | статьи arXiv — по аннотациям (REPORTED) |
| R4-hardware.md | плата, CPU, 32 GB карты, P2P, NVMe, BIOS | «EPYC 7H43» в поиске не находится (поиск ведущего 2026-10-09) | WebFetch не работал; страницы вендоров — по выдаче (REPORTED) |
