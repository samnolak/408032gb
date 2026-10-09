# R2. Движки инференса для сервера 4×RTX 4080 Super 32 GB: состояние на 2026-10-09 и пять направлений

Роль: inference-engine research lead. Репозиторий только читался; этот файл — единственное изменение, коммитов нет.
Дата доступа ко всем источникам: 2026-10-09.

**Метки.** VERIFIED — видел сам в первичном источнике: `файл:строка@SHA` из клона, который я сделал, или литеральный вывод команды.
REPORTED — заявление автора, вендора или сообщества (README, issue, обсуждение, поисковый фрагмент); никто из нас его не перемерял.
HYPOTHESIS — мой вывод или расчёт; вход и способ вывода указаны. DISPROVED — опровергнуто первичным источником. UNKNOWN — данных нет.
Соответствие меткам репозитория (AGENTS.md): VERIFIED = CONFIRMED; REPORTED и HYPOTHESIS = PROVISIONAL; UNKNOWN = UNKNOWN.

**Железо — заявление заказчика, не проверено.** 4×RTX 4080 Super 32 GB (sm_89), без NVLink, P2P UNKNOWN; EPYC 7H43 (Zen 3, AVX2, без AVX-512);
128 GB DDR4-3200, каналов 4 или 8 — UNKNOWN; PCIe Gen4; Proxmox VM. Все прогнозы ниже условны относительно этих фактов.

**Ограничения доступа.** Из песочницы `git clone` и `git ls-remote` к github.com работают, это источник всех SHA.
У WebFetch не открылись x.com, deploy.helix.ml, smcleod.net, arxiv.org и huggingface.co (`getaddrinfo ENOTFOUND`).
`api.github.com` из Bash отвечает 403. Метаданные репозиториев (флаг archived, лицензия) взяты через GitHub MCP `search_repositories`.
Числа со страниц, которые не загрузились, помечены «только поисковый фрагмент».

---

## 1. Итог

1. **Ни один готовый движок сегодня не делает всего, что нужно этому серверу.** Нужны четыре вещи сразу: модели GLM-5.3-Flash и MiMo-V2.6-Flash, sm_89, все 4 GPU без P2P и CPU-ярус экспертов на AVX2.
   Ближе всех подходят **llama.cpp mainline** и **ik_llama.cpp**: модели есть, есть `-ot`/`-ncmoe`, MTP и DFlash.
   В mainline с 2026-10-07/08 появился GPU-кэш экспертов, лежащих в RAM, в том числе на нескольких GPU (`--moe-cache-mib`, коммиты `d6cf9ac`, `c811cb8`, VERIFIED).
   Тензорного параллелизма для GLM5-Next нет ни в одном из двух форков (VERIFIED, §2.3).
2. **Strata-семейство ушло далеко от пинов.**
   - Апстрим Niko1221/Strata: +1552 коммита за неделю, v0.1.41. Движок по-прежнему однопородный: guard `qwen4exp` (VERIFIED).
   - lighttransport/Strata glm53f: +18 своих коммитов после пина. Добавлены 2 GPU (максимум, VERIFIED в коде), паки Q2_K/Q3_K, REAP-50 и межузловой expert-TP.
     Исполнения Q4_K-экспертов по-прежнему нет: Q4_K только перепаковывается в Q2_K/Q3_K (VERIFIED).
   - sergqwer/strata-glm на GitHub **не архивирован** (`archived:false`), хотя автор называет его «Archived experiment». Новых коммитов нет.
3. **Новые проекты 2026 года, о которых в репозитории нет ни слова.**
   - **FreeToken** (FlashML-org, Apache-2.0, 14k★). Гибрид CPU/PCIe по измеренной полосе (q*), radix-кэш, GLM-5.3-Flash, DeepSeek-V4.1-Flash с Engram на NVMe.
     Ограничения: строго **один GPU** («tensor parallelism is not supported yet», VERIFIED), спекулятивного декода нет.
   - **ExLlamaV3 v1.6.0** (MIT): GLM5-Next, MiMo-V2 и DeepSeek-V4 с MTP; TP/EP с «нативным» бэкендом без NCCL и P2P; экспериментальная CPU-выгрузка MoE на AVX2 (VERIFIED).
   - **NInfer** и **Infernix** существуют, но собираются только под sm_120a (VERIFIED) — к серверу неприменимы.
4. **vLLM и SGLang модели знают, но для GLM-5.3-Flash на sm_89 есть блокер.** NVIDIA-путь индексера DSA в vLLM требует DeepGEMM, а DeepGEMM — «only Hopper and Blackwell» (VERIFIED).
   Значит, GLM-5.3-Flash на 4080 в vLLM, скорее всего, не запустится (HYPOTHESIS).
   Кроме того, кастомный all-reduce vLLM на >2 PCIe-картах отключается всегда, независимо от P2P (VERIFIED), и TP идёт через NCCL.
5. **KTransformers** официально поддерживает GLM-5.3-Flash и SM89, но документированный путь не подходит этому серверу:
   FP8-веса 306 GiB, ≥350 GB RAM и «AVX-512 FP8 CPU expert kernel» (VERIFIED, туториал). INT4 на AVX2 для GLM-5.3 не описан (UNKNOWN).
6. **Реальные замеры GLM-5.3-Flash в 4 битах на 3090-классе с выгрузкой — 6–20 tok/s** (REPORTED, §4).
   4×3090 + llama.cpp IQ4_XS + MoE-кэш: 15.75–19.6 tok/s, 16.0 на глубине 90K. 1×3090 + EPYC Milan 8ch: 6.14 tok/s на 69K.
   Лучшее однокарточное число на этом классе GPU — 28.2 tok/s, но в 3.05 bpw (ниже порога 4 бит) на EPYC 7443P того же поколения Zen 3.
7. **Мой прогноз для GLM-5.3-Flash ≥4 бит на этом сервере — 20–40 tok/s без спекуляции** (HYPOTHESIS, §5.0).
   Потолок fitplan — 61–69 tok/s при 262K; измеренные гибридные системы достигают 0.35–0.6 от идеала.
   **Пол 50 tok/s для GLM и MiMo в ≥4 битах под вопросом при любом направлении.** Для Qwen3.8-Flash-Next (125B, целиком в 128 GB VRAM в UD-Q4_K_XL) готовые движки уже дают 46–104 tok/s на сопоставимых GPU (REPORTED).
8. **Предварительный рейтинг направлений:** A (готовый движок как база и замер) > D (свой планировщик памяти поверх llama.cpp/ggml) > C (пересадка частей, как часть D) > B (патчи форков Strata) > E (свой рантайм). Обоснование — §5.
   Окончательный выбор — только после экспериментов §6: E0 (железо), E1 (матрица llama.cpp/ik_llama), E3 (CPU-ядра экспертов на 7H43), E2 (локальность маршрутизации).

---

## 2. Таблица движков

### 2.1 Идентичность и состояние

SHA — только из моего вывода `git ls-remote`, `git log` (VERIFIED, литеральный вывод — в конце раздела). Звёзды и флаг archived — из GitHub MCP `search_repositories`.

| движок | репозиторий | HEAD (дата коммита) | последний тег | лицензия | состояние |
|---|---|---|---|---|---|
| Strata (апстрим) | github.com/Niko1221/Strata | `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (2026-10-08) | v0.1.41 | MIT | активен: +1552 коммита от пина `c499bd1` за 7 дней; не архивирован |
| strata-glm (база A) | github.com/sergqwer/strata-glm | `ed37419fccd0c52e07d26d526a29c2098f547843` (2026-10-01) = пин | v0.1 | MIT | `archived:false`; описание автора: «Archived experiment»; коммитов после пина нет |
| lt-Strata glm53f (база B) | github.com/lighttransport/Strata, ветка `glm53f` | `27deb918394186590f77589773eabc99b1fa28ea` (2026-10-07) | — (только теги апстрима) | MIT | активен: 18 своих коммитов после пина `e486a95`, слит апстрим 0.1.40 |
| strata-glm-3090 | github.com/Grigory-Rylov/strata-glm-3090 | `794cd69ae270389c67ed201b855b14aafb5eb875` (2026-10-03) | — | MIT (форк) | 4 коммита после пина `b115f38`: реальные веса, `--cpu-share`, `--spec K` |
| llama.cpp | github.com/ggml-org/llama.cpp | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (2026-10-09) | b11538 | MIT | очень активен |
| ik_llama.cpp | github.com/ikawrakow/ik_llama.cpp | `5194a9e1451b885e24dd5a065f650864e8807c59` (2026-10-09) | t0002 (релизов нет) | MIT | очень активен |
| KTransformers | github.com/kvcache-ai/ktransformers | `2ac8cf9c88fa7d6de106bcc3ab3ef7645e7476a0` (2026-10-08) | v0.7.1 | Apache-2.0 | активен; инференс через SGLang (kt-kernel) |
| vLLM | github.com/vllm-project/vllm | `ed7ae705a1f366c2008964af562a7543e91d934e` (2026-10-09) | v0.31.0 | Apache-2.0 | очень активен |
| SGLang | github.com/sgl-project/sglang | `46c2585faa4fcb6fa8febf4d2d4534a7cf33da79` (2026-10-10 +0800) | v0.5.21 | Apache-2.0 | очень активен |
| TensorRT-LLM | github.com/NVIDIA/TensorRT-LLM | `49ec5730428f8eb0434e01dc9b4c532136d6c578` (2026-10-10 +0800) | v1.2.1 | Apache-2.0 | активен |
| ExLlamaV3 | github.com/turboderp-org/exllamav3 | `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` (2026-10-07) | v1.6.0 | MIT | активен |
| TabbyAPI (сервер для ExLlamaV3) | github.com/theroyallab/tabbyAPI | `884e88cc1fdd2dfcd5ed27f71ff5b41f52541e1f` (2026-10-09) | — | **AGPL-3.0** | активен |
| LMDeploy | github.com/InternLM/lmdeploy | `822dddb8b3bc895eaf6cffb45a5514768e7ecb84` (2026-10-09) | v0.18.0 | Apache-2.0 | активен |
| mistral.rs | github.com/EricLBuehler/mistral.rs | `3f2515e9b5adc2ac44949c128c50a13b15721294` (2026-10-01) | v0.9.4 | MIT | активен |
| MoE-Infinity | github.com/EfficientMoE/MoE-Infinity | `0f96d563db2dda5f1834bf667f6fcc44f45b2b9d` (2026-10-08) | v0.2.2 | Apache-2.0 | активен |
| PowerInfer | github.com/Tiiny-AI/PowerInfer (бывш. SJTU-IPADS; оба адреса дают один SHA) | `8bd56d69906c9d2dba4d3bf6899763401e01a9a4` (2026-05-11) | — | MIT | **переименован**; почти не развивается |
| Fiddler | github.com/efeslab/fiddler | `227715bfd6e8c731b29548eab01d9919c4fe9564` (**2024-04-28**) | — | Apache-2.0 | **заброшен** (исследовательский прототип под Mixtral) |
| FreeToken | github.com/FlashML-org/FreeToken | `2d8b9cc1eac2e8f2ae63eb0ae90fe419252d9dc6` (2026-10-09) | v0.1.3 | Apache-2.0 | новый (с 2026-07-20), очень активен; 14 322★ |
| NInfer | github.com/Neroued/ninfer | `81c8ce093b2c1646a87566a8e59d807fcf0ec95c` (2026-10-07) | — | Apache-2.0 | существует; **только sm_120a** |
| Infernix | github.com/Wallawalla47/Infernix | `a3edb450f373d1044a1589cf4e742a4e7121c14f` (2026-10-09) | windows-2026.10.09.2 | Apache-2.0 | существует (создан 2026-10-08, «grew from NInfer»); **только sm_120a** |
| Pulsar | github.com/giannisanni/pulsar | `18f990094e1091ee2ef3078f20be418f8c162f03` (2026-09-01) | — | MIT | новый (SSD-стриминг, Rust+CUDA) |
| LMCache | github.com/LMCache/LMCache (ветка `dev`) | `ee68e5ba955cb51147767c207fc462932b87ab04` (2026-10-09) | — | Apache-2.0 | активен (KV-слой для vLLM/SGLang) |

Литеральный вывод (`git ls-remote --symref https://github.com/<repo> HEAD`, 2026-10-09; повторяющиеся строки `ref:` сокращены):
```
Niko1221/Strata               fb58e0dbc8399662c0e47c76578c6e878b14f6cf  (refs/heads/main)
sergqwer/strata-glm           ed37419fccd0c52e07d26d526a29c2098f547843  (refs/heads/main)
lighttransport/Strata         27deb918394186590f77589773eabc99b1fa28ea  (refs/heads/glm53f)
Grigory-Rylov/strata-glm-3090 794cd69ae270389c67ed201b855b14aafb5eb875  (refs/heads/main)
ggml-org/llama.cpp            79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6  (refs/heads/master)
ikawrakow/ik_llama.cpp        5194a9e1451b885e24dd5a065f650864e8807c59  (refs/heads/main)
kvcache-ai/ktransformers       2ac8cf9c88fa7d6de106bcc3ab3ef7645e7476a0  (refs/heads/main)
vllm-project/vllm             ed7ae705a1f366c2008964af562a7543e91d934e  (refs/heads/main)
sgl-project/sglang            46c2585faa4fcb6fa8febf4d2d4534a7cf33da79  (refs/heads/main)
NVIDIA/TensorRT-LLM           49ec5730428f8eb0434e01dc9b4c532136d6c578  (refs/heads/main)
turboderp-org/exllamav3       151539c77abc7ab7425d30da7a4e8e3c5c154e7b  (refs/heads/master)
theroyallab/tabbyAPI          884e88cc1fdd2dfcd5ed27f71ff5b41f52541e1f  (refs/heads/main)
InternLM/lmdeploy             822dddb8b3bc895eaf6cffb45a5514768e7ecb84  (refs/heads/main)
EricLBuehler/mistral.rs       3f2515e9b5adc2ac44949c128c50a13b15721294  (refs/heads/master)
EfficientMoE/MoE-Infinity     0f96d563db2dda5f1834bf667f6fcc44f45b2b9d  (refs/heads/main)
Tiiny-AI/PowerInfer           8bd56d69906c9d2dba4d3bf6899763401e01a9a4  (refs/heads/main)
SJTU-IPADS/PowerInfer         8bd56d69906c9d2dba4d3bf6899763401e01a9a4  (refs/heads/main)
efeslab/fiddler               227715bfd6e8c731b29548eab01d9919c4fe9564  (refs/heads/main)
LMCache/LMCache               ee68e5ba955cb51147767c207fc462932b87ab04  (refs/heads/dev)
huggingface/mistral.rs        fatal: could not read Username for 'https://github.com': terminal prompts disabled
```
`FreeToken`, `ninfer`, `Infernix`, `pulsar`: SHA из `git log -1` после `git clone --depth 50 --filter=blob:none` (VERIFIED, тот же день).

### 2.2 Возможности (для этого сервера)

Сокращения: L = `-sm layer` (конвейер по слоям); T = тензорный параллелизм; EP = экспертный; «host» = обмен через память хоста без P2P.

| движок | sm_89 | multi-GPU / нужен ли P2P | гибрид CPU-GPU для MoE | SSD | кванты на sm_89 | 262K и квант KV | префикс-кэш | спекуляция | API |
|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | да (общий CUDA-бэкенд; REPORTED — сообщество гоняет 4080/4090) | L, row; T (`-sm tensor`, EXPERIMENTAL, **не для GLM5_NEXT, KIMI_K3, MINIMAX_M3**: `src/llama-arch.cpp:1205-1240@79e2e74`); RPC с `-sm tensor` (`a46709b`). P2P только по `GGML_CUDA_P2P` (`ggml/src/ggml-cuda/ggml-cuda.cu:386`) — VERIFIED | `-ot`, `-cmoe`, `-ncmoe` (CPU считает свои эксперты); `--moe-cache-mib` — LRU-кэш на GPU для экспертов в RAM, промахи **подгружаются по PCIe**, на нескольких GPU делится как слои; с `-sm tensor` несовместим (`src/llama-moe-cache.cpp:234-236`, `src/llama-moe-cache.h:11-12`) — VERIFIED | только mmap (специального SSD-яруса нет) — HYPOTHESIS | все GGUF: Q4_K, IQ4_XS, MXFP4; NVFP4 через cuBLAS-путь (`b56f34a`) — VERIFIED по логу | `-ctk/-ctv` q8_0/q4_0 — VERIFIED (`common/arg.cpp:2440`) | `--cache-ram` (кэш промптов в RAM), `--cache-reuse`, `--slot-save-path`, `--prompt-cache` — VERIFIED (`common/arg.cpp:1719,3602,3634,1870`) | draft, EAGLE3, MTP, DFlash, DSpark, 5 видов n-gram (`common/common.h:175-186`); MTP для GLM5-Next (`b9acf13`, 2026-10-07) — VERIFIED | llama-server, OpenAI, стриминг, jinja/tool calls (REPORTED) |
| ik_llama.cpp | да (REPORTED: PR #2376 на 3090/PRO 6000; хост 3090 репозитория) | L, attn, graph (T). Graph поддержан для MIMO2 (≤4 GPU), MINIMAX_M2/M3, GLM_DSA, DEEPSEEK2, QWEN35MOE, но **не для GLM5NEXT, DEEPSEEK4/41, QWEN4EXP** (`src/llama.cpp:4216-4250@5194a9e`) — VERIFIED | `-ot`, `-ncmoe`, `-muge` (склейка up/gate), pinned; `GGML_CUDA_NO_PINNED_WEIGHTS` (`19dfb71`) — VERIFIED по логу | mmap; чекпойнты сервера сбрасываются на диск (`912e9eb`) | Q4_K, IQ4_XS, IQ*_K, IQ*_KT, PQ2_0 — VERIFIED (лог) | q8_0 KV + индексер; 262K — REPORTED (needle ~117K, ctx 262144 в PR #2376) | чекпойнты слотов, сброс на диск | MTP (glm5next, `d741de5`), DFlash/DFlash2 (`28fbe34`), DSpark (`c84f76c`), n-gram — VERIFIED по логу | llama-server |
| KTransformers (+SGLang-KT) | да: «SM 80, 86, 89, 90» (`kt-kernel/README.md:62`), GLM-5.3: «SM89 and SM120» (туториал) — VERIFIED | TP через SGLang (`--tp-size 4` в туториале GLM-5.3); P2P — как в SGLang/NCCL | ядро идеи: эксперты на CPU, «горячие» на GPU (`--kt-num-gpu-experts`). AVX2 с 2026-03-26 (BF16/FP8/GPTQ_INT4/RAWINT4, `doc/en/kt-kernel/AVX2-Tutorial.md`), но таблица `kt-kernel/README.md:189-193` требует AVX-512 для FP8/BF16 — **документы противоречат**. GLM-5.3: «AVX-512 FP8 CPU expert kernel», FP8 306 GiB, ≥350 GB RAM — VERIFIED | старый 3-уровневый префикс-кэш GPU-CPU-диск (README, 2025-06-30) | FP8, BF16, GPTQ_INT4, RAWINT4, LLAMAFILE (GGUF на CPU) | GLM-5.3: до 1M, проверено 501025 — REPORTED | radix SGLang | как в SGLang — UNKNOWN для KT | OpenAI (SGLang), `--tool-call-parser glm47` |
| vLLM | общий — да; **GLM5-Next: индексер требует DeepGEMM** (`vllm/models/glm5next/nvidia/sparse_indexer.py:699-701`), DeepGEMM — «only Hopper and Blackwell» (`vllm/utils/deep_gemm.py:111-115@ed7ae70`) → HYPOTHESIS: не работает на sm_89; KDA-prefill на sm_89 уходит в Triton (`common/kda.py:146-158`) | TP/PP/EP/DP. Кастомный all-reduce выключается без P2P и **всегда при >2 PCIe-only GPU** (`custom_all_reduce.py:270-291`) → NCCL — VERIFIED | только выгрузка весов (`cpu_offload_gb`, prefetch-offload, `arg_utils.py:582-588`); эксперты на CPU не считаются — VERIFIED | нет (KV offload через коннекторы) | AWQ/GPTQ-Marlin, FP8 W8A8, MXFP4/NVFP4 через Marlin W4A16 (HYPOTHESIS для sm_89) | fp8 KV — VERIFIED (`--kv-cache-dtype`) | APC, `--kv-offloading-size`, LMCache | EAGLE/EAGLE3, MTP, DFlash/DFlash2 (пин репозитория `bc21cba`), n-gram | OpenAI, парсеры tool calls |
| SGLang | GLM5-Next на sm_89 в апстриме — UNKNOWN; вилка KT заявляет SM89 | TP/PP/EP/DP-attention | интеграция KT в апстриме: `python/sglang/srt/layers/moe/kt_ep_wrapper.py@46c2585` — VERIFIED | HiCache (иерархический KV: host/SSD/внешние хранилища) — VERIFIED (`mem_cache/hicache_storage.py`) | FP8, AWQ/GPTQ, MXFP4/NVFP4 | fp8 KV | RadixAttention + HiCache | EAGLE, MTP (`glm5_next_nextn.py`), DSpark (`46c2585`) | OpenAI |
| TensorRT-LLM | FP8 на Ada — REPORTED | TP/PP/EP через NCCL | нет данных о CPU-экспертах — UNKNOWN | — | FP8, INT4-AWQ | — | KV reuse | EAGLE/MTP — REPORTED | OpenAI (trtllm-serve) |
| ExLlamaV3 (+TabbyAPI) | да (CUDA; ROCm тоже) — REPORTED | TP и EP «для потребительского железа» (`README.md:11`); бэкенды NCCL или **native**: обмен через хост без NCCL «ценой PCIe-трафика и CPU» (`exllamav3/model/model_tp_backend.py:245@151539c`) — VERIFIED | `-mcl/--moe_cpu_offload`: экспериментально, **только layer-split**, эксперты в RAM считает отдельный процесс, AVX2/AVX512 (`doc/env_vars.md:315-325`, `README.md:12`) — VERIFIED | нет (у форков есть) | EXL3 (QTIP) 2–8 bpw; кэш 2–8 бит | — | страничный кэш (REPORTED); TP-кэш страниц в CPU (лог `f1cfbee`) | спекуляция, MTP-архитектуры, ядро DFlash2 (`79d0b81`) | TabbyAPI: OpenAI (**AGPL-3.0**) |
| LMDeploy | TurboMind на Ada — REPORTED | TP | нет — UNKNOWN | — | AWQ, FP8 | — | есть | — | OpenAI |
| mistral.rs | да (CUDA) — REPORTED | NCCL TP, P2P layer mapping, ring (README:246) | — | — | GGUF, GPTQ, AWQ, FP8, ISQ | — | — | — | OpenAI |
| MoE-Infinity | сборка по умолчанию sm_80/sm_90, sm_120 — по флагу (README:83); sm_89 — UNKNOWN | эксперты по GPU по кругу; P2P, если есть, иначе через хост (README:9) — VERIFIED (текст) | эксперты в RAM и на SSD, кэш горячих на GPU (fetch) | да | FP8/FP4-эксперты | — | только для пути Qwen3 | — | OpenAI |
| FreeToken | заявлены RTX 30/40/50 (README); NVFP4 на sm_80…9x — через Marlin vLLM (`layers/quantization/moe/nvfp4.py:243`) — VERIFIED | **только один GPU**: «tensor parallelism is not supported yet: --gpu takes one entry» (`python/freetoken/server/args.py:823@2d8b9cc`) — VERIFIED | `--moe-strategy fused/offload/cpu/hybrid`; hybrid делит промахи на PCIe-подгрузку и CPU по замеру `ft bench bw` (`docs/models.md:40-54`); CPU-исполнитель AVX-512-BF16 → AVX-512F → **AVX2+FMA** (`cpu_moe_ext.cpp:13-16`) — VERIFIED | Engram DeepSeek-V4.1 стримится с NVMe (`docs/models.md:92-97`) — VERIFIED (текст) | NVFP4, MXFP4, FP8, BF16; GGUF — только gemma4 (`models/gguf/config.py:19-21`) | эластичный KV/кэш экспертов без рестарта | radix (SWA/GDN-aware), «semantic anchors» для агентов | **нет** («DSpark speculative decoding is not served», `docs/models.md:116-117`; в server/engine нет кода спекуляции — VERIFIED grep) | OpenAI, Anthropic, Responses; парсеры tool calls и reasoning; `/v1/stats` |
| Strata (апстрим) | да: сборка `[75, 86, 89, 120]` в отчёте 4080 SUPER 32 GB (REPORTED) | L на 2–4 карты, P2P не нужен (`docs/MULTI_GPU.md:9-10`), авторазмещение для четырёх карт с 0.1.40 (`MULTI_GPU.md:134`); no-P2P peer-путь (опция) — VERIFIED | VRAM-кэш + CPU-пул (AVX2 i-quant) + PCIe-доля | SSD-ярус экспертов (`docs/UNSLOTH_Q4.md:162-163`) | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, UD-IQ4_XS, UD-Q4_K_XL (Q4_K/Q5_K/Q6_K) | 262K родной, 524K с YaRN (REPORTED); int8/K8V4 KV | файлы сессий, кэш по границам сообщений | MTP; lookup | OpenAI и Anthropic; tool calls |
| lt-Strata glm53f | да: CI репозитория собрал sm_89 (`docs/evidence/2026-10-03-lighttransport-fork.md`, на `e486a95`) | **1 или 2 GPU**: «select one or two GPU devices» (`src/program/glm_decode.cpp:6686-6687@27deb91`); peer — если есть, иначе обход (`:1149-1200`); межузловой expert-TP по IB/UDP (`ae343e5`) — VERIFIED | CPU считает эксперты из RAM, статический GPU-кэш, NUMA-пул | — | эксперты Q2_K, Q3_K, IQ2_S, IQ3_S, IQ4_XS (`docs/GLM53_FLASH.md:126-127`); Q4_K/Q5_K/Q6_K — **только перепаковка в Q2_K/Q3_K** (`include/strata/core/model.hpp:308-312`); EXL3 K2 (один GPU) — VERIFIED | контекст до 131072 (отчёт V100) — REPORTED | **нет** («does not reuse conversation prefixes») | MTP GPU/CPU, lookup | `serve.server --engine glm`, только greedy, **без парсинга tool calls** (README) |
| strata-glm (база A) | сборка sm_89 — CONFIRMED в репозитории (CI 36956571033) | в GLM-пути нет (ADR-007); у нас — патч 0002 (стадии) | VRAM/RAM/диск; `--cpu-share` (форк 3090) | диск | NVFP4 (pack) | 64K проверено на 3090 (REPORTED) | UNKNOWN | `--spec K` n-gram (форк 3090) | UNKNOWN |
| NInfer / Infernix | **нет**: «supports only CMAKE_CUDA_ARCHITECTURES=120a» (оба `CMakeLists.txt:3-12`) — VERIFIED | один GPU | Infernix: Qwen3.8-Flash-Next, RAM/SSD-ярусы | Infernix — да | NVFP4 и др. | Infernix: 262K, 1M с YaRN (REPORTED) | Infernix: 3 уровня, переживает рестарт | MTP, DFlash/DFlash2, n-gram | OpenAI/Anthropic |
| Pulsar | REPORTED: 5060 Ti + 4060 Ti | «zero-config multi-GPU» (README) | горячие в VRAM, «CPU lane» | **да** (основной режим) | GGUF | — | — | nextn MTP | OpenAI (REPORTED) |

### 2.3 Поддержка моделей-кандидатов (код на HEAD; «—» = не найдено в коде)

| движок | GLM-5.3-Flash | MiMo-V2.6-Flash | DeepSeek-V4.1-Flash (Engram) | DeepSeek-V4-Flash | Qwen3.8-Flash-Next | Kimi K3 | MiniMax M3 |
|---|---|---|---|---|---|---|---|
| llama.cpp | `glm5-next` (`649dcb1`, PR #27773); строку `glm5next` (файлы Unsloth UD) **не знает** | `mimo2`; V2.6 — UNKNOWN до загрузки GGUF | **—** (нет арки `deepseek41`) | `deepseek4` | `qwen4exp` | `kimi-k3` | `minimax-m3` |
| ik_llama.cpp | `glm5next` (`42a9a2f`, #2376) и алиас `glm5-next` → GLM5NEXT (`src/llama-model-loader.h:149-151`, с `d741de5`) | `mimo2` (`043ced9` «add mimo») | `deepseek41` (`a761619`, `af5817d`) + DSpark | `deepseek4` | `qwen4exp` | — | `minimax-m3` |
| vLLM | `Glm5NextForCausalLM` (registry:119) — блокер sm_89 см. 2.2 | `MiMoV2FlashForCausalLM`, `MiMoV2ForCausalLM` (:175-176) | `DeepseekV41ForCausalLM` (:370-372) | `DeepseekV4ForCausalLM` | `Qwen4ExpForCausalLM` | `KimiK3ForConditionalGeneration` | `MiniMaxM3Sparse…` |
| SGLang | `glm5_next.py` (+nextn) | `mimo_v2.py` (+nextn) | `deepseek_v4*.py`, `deepseek_v41_vit.py` | да | `qwen4_exp.py` | `kimi_k3.py` | `minimax_m3.py` |
| TensorRT-LLM | `modeling_glm5_next.py` | — | UNKNOWN (только `modeling_deepseekv4.py`) | да | `modeling_qwen4_exp.py` | `modeling_kimi_k3_vl.py` | `modeling_minimaxm3.py` |
| ExLlamaV3 | `glm5_next.py` (+mtp) | `mimo_v2.py` (+mtp) | UNKNOWN | `deepseek_v4.py` (+mtp) | `qwen4_exp.py` (+mtp) | — | — (есть `minimax_m2`) |
| KTransformers | да (FP8 путь, туториал 2026-08-26) | UNKNOWN | UNKNOWN | да (туториал 2026-05-02) | UNKNOWN | Kimi K2.x | да (2026-06-21) |
| FreeToken | да (NVFP4, `RedHatAI/GLM-5.3-Flash-NVFP4`) | — | **да**, Engram с NVMe | да | да (FP8/NVFP4) | — | да (NVFP4) |
| MoE-Infinity | да (FP8, transformers ≥5.16, eager-attention) | — | — | да (FP4) | — | — | — |
| LMDeploy | — (есть GLM-5/5.2) | — | — | DeepSeek-V4 (PyTorch engine) | — | (Kimi-K2.6) | — |
| mistral.rs | — | — | — | — | `Qwen4ExpLoader` | — | — |
| Strata апстрим | — | — | — | — | **единственная модель** | — | — |
| lt-Strata glm53f | только `glm5next` GGUF (+EXL3 K2, REAP-50) | — | — | — | (наследует Qwen-движок) | — | — |
| Pulsar | — (GLM-5.2) | — | — | да | — | (Kimi K2.7) | да |

Источники ячеек — VERIFIED листингами `git ls-tree` и grep на указанных HEAD (§8: [1]–[14]). Строки README KTransformers, MoE-Infinity и LMDeploy — VERIFIED как текст, работоспособность — REPORTED.

---

## 3. Strata-семейство относительно пинов репозитория

### 3.1 Niko1221/Strata: пин `c499bd1` (0.1.32, 2026-10-01) → HEAD `fb58e0d` (0.1.41, 2026-10-08)

- **+1552 коммита, пин — предок HEAD** (VERIFIED: `git rev-list --count c499bd1..origin/HEAD` = 1552; `git merge-base --is-ancestor` — да).
- **Однопородность не изменилась** (VERIFIED). Guard `qwen4exp`: `include/strata/artifact/gguf_reader.hpp:573-595@fb58e0d`, проверка на строке 584.
  `docs/MODELS.md:3`: «Strata runs one model, Qwen3.8-Flash-Next». Файлов glm/mimo/deepseek в дереве нет.
- Что появилось и важно для нас (VERIFIED по логу; цифры — REPORTED):
  - **Многокарточность без P2P.** Opt-in no-P2P peer-путь (слияние `4aca10c`, коммиты `81af1da`, `f4bfedc`, `b92a44b`). Авторазмещение перебирает все раскладки для четырёх карт (`fa9bd5f`, `e2b3289`; `docs/MULTI_GPU.md:134`).
    `--pipeline-windows 2`: 2×3090 без NVLink, декод 136.7 tok/s (MULTI_GPU.md:249-263, REPORTED).
  - **UD-Q4_K_XL (4 бита) на нескольких картах.** 2×3090 + 165 GiB RAM: 64–78 tok/s, 55 tok/s на промпте 128K, 31 tok/s на одной карте (`docs/UNSLOTH_Q4.md:31-35`, REPORTED).
  - **SSD-ярус экспертов.** UD-Q4_K_XL при нехватке RAM: 7–8.5 tok/s при бюджете 40 GiB (UNSLOTH_Q4.md:162-163, REPORTED).
  - **Агентный сервер.** Файлы сессий r3/r4, `--batch-groups auto`, кэш токенизатора (повторный промпт агента на 89K кодируется за 0.07 с), backlog 256.
    Плюс экспериментальный `STRATA_ROUTE_RESIDENT` — смещение маршрутизации к резидентным экспертам; **меняет вывод**, что противоречит требованию точной маршрутизации.
  - **CPU-ядра.** AVX-VNNI-формы AVX2 i-quant (`b211ba1`), gathered-декод IQ3 на AVX2 (`a733617`), опциональные Q6_K-эксперты.
- **Вывод.** Апстрим — сильный кандидат A только для Qwen3.8-Flash-Next. Для GLM, MiMo и DeepSeek его нужно форкать: это ADR-002, сегодня ещё дороже, так как дельта за неделю — +1552 коммита.

### 3.2 sergqwer/strata-glm (база A): HEAD = пин `ed37419`

- Коммитов после пина нет, тег `v0.1`; `pushed_at` 2026-10-01T07:06:36Z (GitHub MCP).
- **ADR-007 говорит «в архиве» — по флагу GitHub DISPROVED** (`"archived":false`). По словам автора — REPORTED: описание репозитория «Archived experiment: ~1000 tok/s prefill, 15-18 tok/s decode».
- Практически проект заброшен автором. Поддерживать его будем только мы.

### 3.3 lighttransport/Strata glm53f (база B): пин `e486a95` → HEAD `27deb91` (2026-10-07)

- Пин — предок HEAD; +1565 коммитов, из них 18 — Syoyo Fujita. Остальное — слияния апстрима: `3bbb469` (2026-10-03) и `7449b2b`/`27deb91` (0.1.40).
  Только GLM-файлы: 6 файлов, +6730/−361 (VERIFIED: `git diff --stat`).
- Что добавлено (VERIFIED по логу и коду; цифры автора — REPORTED):
  - **2 GPU**, максимум 2 (`glm_decode.cpp:6686-6687`). Фиксированные проекции живут на основном GPU, резидентные эксперты — на обоих, промахи считают CPU-воркеры (`docs/GLM53_V100.md:71-76`).
    2×V100 32 GB + 2×Xeon Gold 6240, UD-Q2_K_XL: 28.91 tok/s, с MTP 29.06; prefill 501.6 tok/s (4K) и 790.47 (16K) (GLM53_V100.md:5-11,110-123).
  - **NUMA-пул CPU-экспертов.** TR 1950X + 5060 Ti, 2 NUMA-узла: 9.74 tok/s обычный декод, 10.2–11.6 с MTP-2 (`docs/GLM_Q2_NUMA_DECODE.md`). Автор пишет: «15 tokens/s target remains unmet».
  - **`strata-glm-q2-pack`** (откалиброванные Q2_K gate/up + Q3_K down), «step pipeline decode», Q8-декод, AVX2-ядра Q2_K/Q3_K.
    **REAP-50** (`861e1f9`) — GGUF с 144 экспертами из 288, то есть **прунинг экспертов**: противоречит требованию «точная маршрутизация».
  - **Межузловой expert-TP по InfiniBand/RDMA с UDP-запасным путём** (библиотека `ucomm`, `bd4c31f`; `1aa7b05`, `ae343e5`).
  - **Нативный EXL3 K2/MCG** (`aef3c60`). Только один GPU, контекст до 65536, без MTP и кэшей (`docs/GLM_EXL3.md`).
- **Не изменилось (VERIFIED):**
  - Q4_K/Q5_K/Q6_K-эксперты не исполняются. Источник K-quant только перепаковывается в Q2_K/Q3_K (`model.hpp:308-312`).
  - Повторного использования префикса нет; парсинга tool calls нет (README_GLM53_FLASH.md: «GLM tool-call parsing is not implemented»).
  - Тензоры IQ4_XS исполняются (5 форматов, `docs/GLM53_FLASH.md:126-127`). Поэтому **UD-IQ4_XS может пойти без новых ядер**, если все его экспертные тензоры — IQ4_XS (UNKNOWN: состав файла не инспектирован).
- **Поправка к ADR-009** («Multi-GPU is not mentioned anywhere»): теперь до 2 GPU в коде (VERIFIED) и замер на 2×V100 (REPORTED). До 4 GPU не доходит.

### 3.4 Grigory-Rylov/strata-glm-3090: `b115f38` → HEAD `794cd69` (2026-10-03 18:51)

4 коммита, только `src/glm/glm_main.cpp`, `glm_gemm.cu`, `PROBLEMS.md`. Хост 3090 репозитория, реальный NVFP4-пак 171.2 GB (REPORTED, PROBLEMS.md@794cd69):
- первый прогон: prefill 381.59 tok/s, декод 9.25 tok/s; повтор эксперта от предыдущего токена — **24.6%** (маршрутизация «плоская»);
- `--cpu-share 1.0 --cpu-threads 90` (RAM-эксперты считает EPYC вместо копирования по PCIe) и `--dense-i8`: декод **6.85 → 13.89 tok/s**;
- `--spec 3` (n-gram prompt-lookup): 14.12 tok/s при 55.6% принятия; разброс между прогонами сравним с выигрышем. Вывод автора: «The MoE is the ceiling, not the plumbing».

Сравнение на том же хосте: ik_llama.cpp UD-Q4_K_XL дал 15.0 tok/s (PROBLEMS.md репозитория). **База A на реальных весах не быстрее готового ik_llama** (REPORTED, оба числа с одного хоста).

### 3.5 Поправка к PROBLEMS.md репозитория

PROBLEMS.md утверждает: «ik_llama знает только `glm5next` (`src/llama-arch.cpp:88`), алиаса нет», хост обновлён до `5f89bfc`.
**Как утверждение об исходниках — DISPROVED.** На `5f89bfc` (2026-10-02) уже есть `{ LLM_ARCH_GLM5NEXT_DASHED, "glm5-next" }` (`src/llama-arch.cpp:90@5f89bfc`), и загрузчик сводит его к GLM5NEXT (`src/llama-model-loader.h:149-151`, добавлено в `d741de5`, 2026-09-28).
Наблюдаемый отказ загрузить `GLM-5.3-Flash-Q4_K.gguf` остаётся фактом хоста (REPORTED). Причина — UNKNOWN. HYPOTHESIS: ключи метаданных `glm5-next.*` против `glm5next.*` (ср. `expert_group_count` в evidence 2026-10-03).

---

## 4. Реальные замеры на сопоставимом железе

Все числа — REPORTED (чужие замеры, у нас не воспроизведены). «Только поисковый фрагмент» — страница не загрузилась.
Числа ниже 4 бит приведены как ориентир скорости движка, не как кандидат по качеству.

| # | модель / квант | движок | железо | контекст | pp tok/s | tg tok/s | кто, где |
|---|---|---|---|---|---|---|---|
| 1 | GLM-5.3-Flash IQ4_XS | llama.cpp (образ club-3090, MoE-кэш; коммит не указан) | 4×RTX 3090, ~22.6 GB кэша экспертов на карту, VRAM 92 008 MiB; CPU/RAM не указаны | макс. 204 800, стресс до 188K | 344 (10K), 284 (90K) | 16.2 рассказ / 19.6 код / **16.0 на глубине 90K**; soak p50 15.75, p95 19.00 | alexanderjacuna, club-3090 issue #1543, 2026-10-03 [20] |
| 2 | GLM-5.3-Flash Q2_K (DevQuasar, 114.4 GB) + драфтер DFlash2 Q4_K_M | llama.cpp (образ club-3090) | 2×3090 Gen4 x8, без NVLink, i7-12700K, 96 GB | 65 536 | 117.78 (10K) | 19.17 (median 19.25), принятие 0.681 | moritzburgard, club-3090 #1351 [21] |
| 3 | GLM-5.3-Flash UD-Q4_K_XL (185.98 GiB) | ik_llama.cpp `5f89bfc`, `-ncmoe 39 -muge`, pinned | 1×3090 (остальные заняты), 96 потоков, 251 GiB | 8192 | 166.8 | **15.0** | хост 3090, `PROBLEMS.md` репозитория (PROVISIONAL там же) |
| 4 | GLM-5.3-Flash UD-IQ4_XS | ik_llama.cpp, ветка PR #2376 (до фикса #2399) | 1×3090, **EPYC Milan, 8 каналов DDR4-3200** | промпт 69 025 | 95.82 | **6.14** | maximus2600, PR #2376 [22] |
| 5 | GLM-5.3-Flash UD-IQ4_XS | ik_llama.cpp, PR #2376, `-ncmoe 28` | 2×3090 + A4000, i265K, 192 GB DDR5 | 16K | ~100 | ~8 | Nexesenex, PR #2376 [22] |
| 6 | GLM-5.3-Flash UD_IQ3_XXS | ik_llama.cpp, PR #2376, `-sm layer` | 3×3090, выгрузка в DDR4 | 32 768 | 65.84 | 14.57 | devMiikaK, PR #2376 [22] |
| 7 | GLM-5.3-Flash NVFP4-пак 171 GB | strata-glm-3090 `794cd69` (база A + патчи 0001–0006 + новое) | 1×3090, EPYC (96 потоков), 251 GiB | промпт 16K | 401 | 6.85 → **13.89** (`--cpu-share`), 14.12 со `--spec 3` | grishberg, PROBLEMS.md@794cd69 [5] |
| 8 | GLM-5.3-Flash EXL3 **3.05 bpw** (125 GB) | glm53-flash-offload (ExLlamaV3 + плагин) | 1×3090 PCIe 4.0 x16, **EPYC 7443P (Zen 3), 8 каналов DDR4**, 4×NVMe RAID0 | 8K / 32K / 93K | 710 / 951 / 989 | **28.2**; 28.6 на 93K; с NVMe-ярусом 55 GiB — 19.5 | 0xSero/glm53-flash-offload @`6769b27` [23] |
| 9 | GLM-5.3-Flash UD-Q2_K_XL | lt-Strata glm53f | 2×V100 32 GB PCIe Gen3, 2×Xeon Gold 6240, 160 GiB | выделено 131 072, промпт 4K / 16K | 501.6 / 790.5 | **28.91**, MTP-1: 29.06 | lt-Strata `docs/GLM53_V100.md@27deb91` [3] |
| 10 | GLM-5.3-Flash UD-Q2_K_XL | lt-Strata glm53f | TR 1950X (2 NUMA), 4×32 GB DDR4, 5060 Ti 16 GB | 8192 | — | 9.74 обычный; 10.2–11.6 MTP-2 | `docs/GLM_Q2_NUMA_DECODE.md@27deb91` [3] |
| 11 | GLM-5.3-Flash NVFP4 (181 GB экспертов) | FreeToken v0.1.2 + патчи ox-boost | 1×RTX PRO 6000 96 GB, PCIe Gen5 x16, DDR5-3600 | 0.9K; 16K/64K/128K; после 232K | 1030 (232K) | **~38**; 38.7 / 37.4 / 36.8; 25.1 после 232K; промахов ~31 на токен (0.11) | Cerynitius/freetoken-ox-boost @`fdb99c6` [24] |
| 12 | GLM-5.3-Flash NVFP4 | SGLang, TP4 | 4×RTX PRO 6000 | — | — | 63.7 на один запрос | Helix ML (только поисковый фрагмент) [25] |
| 13 | Qwen3.8-Flash-Next UD-Q4_K_XL | Strata | 2×3090, 165 GiB RAM | промпт 128K | — | 64–78 split; **55 на 128K**; 31 на одной карте | `docs/UNSLOTH_Q4.md:31-35@fb58e0d` [1] |
| 14 | Qwen3.8-Flash-Next UD-Q4_K_XL | llama.cpp `df03399` → форк lukolszewski | **6×4090** (аренда), без NVLink, layer split, q8_0 KV | 50K / 250K | 1728→7050 / 744→7403 | 46.2→57.5 / **21.0→48.7** | llama.cpp discussion #30071 [26] |
| 15 | то же | то же | 6×3090 (часть через USB4 eGPU) | 50K / 250K | 568→2127 / 263→2111 | 25.7→41.9 / 10.2→33.7 | там же [26] |
| 16 | Qwen3.8-Flash-Next IQ3_S | Strata 0.1.39 | **1×RTX 4080 SUPER 32 GB** (тот же GPU), i7-13790F (AVX2), 96 GB, Gen4 x16 (24.0 GB/s H2D) | ctx 262 144, промпт 44K | 3363 | **91.3** (103.9 «live»), 85.3–90.8 при 524K | 1314521gjy, `bench/results/2026-10-04-community-rtx-4080s-iq3s@fb58e0d` [1] |
| 17 | Qwen3.8-Flash-Next UD-Q4_K_XL | llama.cpp moecache-rc (`d9cc8af98`), кэш фактически выключен | 2×5090, TR PRO 3975WX, DDR4 (STREAM 91.5 GB/s), P2P выкл. | 10K / 87K | 1117 / 1040 | 53.2 / 46.1; 36.1 на 87K | paulp83, club-3090 #1133 [27] |
| 18 | Qwen3.8-Flash-Next Coder IQ1_M | Strata 0.1.26 | 2×3090, **EPYC 7453 (Zen 3, AVX2)** | 32K | — | 98.6 на двух картах | `bench/results/2026-09-29-rtx3090-epyc-milan@fb58e0d` [1] |
| 19 | DeepSeek-V4-Flash-0731 UD-IQ2_M (~91 GB) | llama.cpp | 1×4090 | 250K | 620–650 | ~12; 11.5 на 250K | X, analogalok (только поисковый фрагмент) [28] |

**Чего не нашлось (UNKNOWN):**
- vLLM, SGLang или KTransformers на 4×4090/4080 с выгрузкой для GLM-5.3-Flash, MiMo-V2.6 или DeepSeek-V4.1;
- любые замеры MiMo-V2.6-Flash на потребительских GPU;
- ik_llama `-sm graph` с GLM (графовый режим для glm5next не поддержан, §2.2).

Статья «I failed to make GLM-5.3 Flash usable on 4x RTX 3090» (x.com) не открылась: содержание UNKNOWN.

**Что следует из таблицы (HYPOTHESIS).**
- GLM-5.3-Flash в 4 битах с выгрузкой на GPU 3090-класса: 6–20 tok/s у готовых движков, на глубоком контексте нижняя часть диапазона.
- Ближе всего к нашему серверу строка 1: 4×24 GB против наших 4×32 GB, IQ4_XS, MoE-кэш.
- 28–38 tok/s получают либо при ≤3 бит, либо на картах с 96 GB и Gen5.

---

## 5. Направления A–E

### 5.0 Модель скорости, общая для всех направлений (HYPOTHESIS)

- **Вход.** `tools/fitplan.py` репозитория, запуск 2026-10-09 (вывод PROVISIONAL по докстрингу инструмента):
  ```
  $ python3 tools/fitplan.py models/glm-5.3-flash.json --quant Q4_K --ctx 262144 --ram-channels 8
  experts: 12384 x 25.17M params = 163.3 GiB (13.50 MiB each)
  VRAM-resident experts: 64.5% (7982)  | spill to RAM: 58.0 GiB of 116 usable
  decode traffic per token: GPU 10171 MiB, CPU 1612 MiB  (hit 64.5%, uniform routing ...)
    ceiling, 8-channel RAM: pipeline 69 tok/s | ideal TP4 121 tok/s  (upper bounds, no MTP)
  (4 канала: pipeline 61 | TP4 61;  IQ4_XS: 68.2% в VRAM, pipeline 69 | TP4 72 (4ch) / 144 (8ch))
  $ python3 tools/fitplan.py models/mimo-v2.6-flash.json --ctx 262144 --ram-channels 4|8
  experts: 12032 x 25.17M params = 149.8 GiB ... VRAM-resident experts: 72.9%
    ceiling: pipeline 75 (4ch) / 77 (8ch) | TP4 75 / 150
  $ python3 tools/fitplan.py models/deepseek-v4.1-flash.json --ctx 262144
  ... disjoint VRAM/RAM: DOES NOT FIT | NVMe spill 48.2 GiB ; n-gram/Engram tables on NVMe: 188.8 GiB
    ceiling: pipeline 38 (4ch) / 75 (8ch)   (стоимость NVMe-промахов в потолок не входит)
  ```
- **КПД гибрида (доля от идеала по полосе)** — оценка по измеренным системам §4 (REPORTED → HYPOTHESIS).
  Строка 8: 3.22 GB экспертов на токен с CPU (8×DDR4-3200, 204.8 GB/s) и ~5 GB плотной части на 3090 (936 GB/s) дают идеал ~51 tok/s, измерено 28.2 → **≈0.55**.
  Строка 1: ~55% экспертов в VRAM, 16–19.6 tok/s → **≈0.35–0.45**. Тот же порядок получается и по строкам 9 и 11.
- **Следствие для GLM-5.3-Flash ≥4 бит при 262K:** 61–69 × (0.35–0.6) ≈ **21–41 tok/s без спекуляции**. Для MiMo-V2.6-Flash (MXFP4): 75–77 × (0.35–0.6) ≈ **26–46 tok/s**.
- **Спекуляция при выгрузке экспертов** даёт мало: проверка T токенов при «плоской» маршрутизации тянет почти T× байт CPU-экспертов.
  Измерено: строки 2 (DFlash2, 0.681 принятия, 19.17 tok/s), 7 (`--spec 3`: +0–4%) и 9 (MTP-1: +0.5%). Реалистично ×1.0–1.4.
  **Итог: 50 tok/s для GLM и MiMo в ≥4 битах достижимы только при КПД ≥0.75 или сильном перекосе маршрутизации** (сейчас нет данных, это E2).
- **Плотная часть** — 7.7 GiB при Q8_0 на токен, это ~10 GB трафика GPU у fitplan. В конвейере по слоям она читается последовательно на одной карте за раз: ≈13.8 мс на токен, то есть потолок ≈72 tok/s, даже если CPU ничего не стоит.
  TP на 4 карты без P2P делит эту часть на 4. Цена — ~92 all-reduce по ~8 КБ на токен через хост, 20–60 мкс каждый → ≈2–6 мс.
  **Поэтому TP плотной части — главный рычаг сверх кэшей** (HYPOTHESIS; решает E5).
- **Qwen3.8-Flash-Next (125B, UD-Q4_K_XL) влезает в 128 GB VRAM целиком** (R1, REPORTED). Ориентиры §4, строки 13–17: 46–104 tok/s → ожидание **60–120 tok/s** (HYPOTHESIS).
- **DeepSeek-V4.1-Flash в ≥4 битах** не влезает в VRAM+RAM: 48 GiB экспертов и 189 GiB Engram уходят на NVMe. Ожидание **3–15 tok/s** (HYPOTHESIS: Pulsar на DSV4-Flash с SSD — 8.2 tok/s, [10]; потолок fitplan без учёта NVMe — 38–75).

### 5.1 Сравнение направлений

| | A. Лучший готовый движок без форка | B. Патчи баз A/B (форки Strata) | C. Пересадка частей из разных проектов | D. Свой планировщик памяти поверх рантайма | E. Новый специализированный рантайм в репозитории |
|---|---|---|---|---|---|
| что это | GLM/MiMo: llama.cpp `79e2e74` (`-sm layer`, `-ot`/`-ncmoe`, `--moe-cache-mib` на 4 GPU, MTP, DFlash) или ik_llama `5194a9e` (`-ncmoe`, `-muge`, MTP; `-sm graph` для MiMo ≤4 GPU). Qwen3.8-Flash-Next: Strata v0.1.41 на 4 картах. Запасные: ExLlamaV3+TabbyAPI; KT+SGLang, если подтвердится INT4 на AVX2 | База A (NVFP4, 4 стадии по патчу 0002, `--cpu-share`) или база B (2 GPU → 4, IQ4_XS/Q4_K-эксперты, префикс-кэш, tool calls) | Например: q*-политика FreeToken (Apache-2.0) и AVX2 i-quant ядра Strata (MIT) в llama.cpp; native TP-бэкенд ExLlamaV3 (MIT); Q4_K-ядра апстрима Strata в базу B | Поверх ggml/llama.cpp: размещение «горячих» экспертов по гистограмме G0 на каждую карту, решение «считать на CPU или везти по PCIe» по замеру полосы, TP плотной части, NUMA-пул | Свой граф и ядра GLM/MiMo/DS с нуля: конвейер, TP, ярусы, сервер (продолжение ADR-002) |
| декод GLM ≥4 бит, 262K (HYPOTHESIS) | **18–30 tok/s**: строка 1 §4 (16–19.6 на 4×24 GB) масштабирована на больше VRAM (68% против ~55% экспертов) и меньшую полосу карты (736 против 936 GB/s); КПД ≈0.3–0.45 от 61–69 | **15–35**: база A — 13.89 на 1×3090 (строка 7) при той же архитектуре ярусов, ×(4 карты, больше резидентных) ≤ КПД D; база B — 9.7–28.9 на Q2 (строки 9–10), Q4 удвоит CPU-байты | **25–40**: как D, если пересадка удастся | **25–45**: КПД 0.4–0.65 от 61–69 + TP плотной части | **30–55**: верх — КПД 0.7–0.8 при идеальном перекрытии + MTP ×1.2; низ — как D |
| Qwen3.8-Flash-Next UD-Q4_K_XL (HYPOTHESIS) | **60–120** (Strata; строки 13, 16, 17) | не применимо | — | — | — |
| стоимость интеграции | дни: сборка, параметры, бенч-скрипты; у llama-server уже есть OpenAI, стриминг, tool calls, кэш промптов | недели–месяцы: база B ограничена 2 GPU, не умеет Q4_K, без префикс-кэша и tool calls; база A — одна модель NVFP4, без сервера; дрейф от апстрима (+1552 коммита за неделю) | недели на каждый перенос; сопряжение чужих форматов весов (NVFP4/EXL3/GGUF) и численных контрактов | 1–3 месяца: код в `src/llama-moe-cache.cpp` и планировщике ggml; можно отдавать в апстрим | 6+ месяцев на одну модель; × три архитектуры (GLM, MiMo, DS) |
| лицензии | llama.cpp и ik_llama — MIT; ExLlamaV3 — MIT, **TabbyAPI — AGPL-3.0** (ок, если не модифицировать и не раздавать; иначе — обязательства); KT/SGLang/vLLM/FreeToken — Apache-2.0 | MIT | MIT + Apache-2.0 совместимы (нужен NOTICE и атрибуция); AGPL-код TabbyAPI не переносить | MIT (как llama.cpp) | наше; заимствования — по месту |
| главный риск | GLM5-Next без TP ни в одном ggml-форке; MoE-кэш — только подгрузка по PCIe, без CPU-расчёта промахов; `glm5next` против `glm5-next` по файлам (§3.5); качество/скорость могут не дотянуть до 50 | две базы, обе фактически однопользовательские; без tool calls и префикс-кэша не закрыть требования сервера; автор базы B уходит в Q2 и прунинг REAP-50 | точность и дубли кода; на каждую модель — заново | конкуренция с апстримом (llama.cpp добавляет то же само: кэш, multi-GPU — 2026-10-07/08); выигрыш ограничен «плоской» маршрутизацией | огромная стоимость при выигрыше, который по §5.0 упирается в ту же полосу; корректность (G1-эталоны есть только для GLM/MiMo) |
| что решает | E1 + E0 | E1 (с базой A и базой B UD-IQ4_XS на 2 GPU) против A на том же железе | E3 и E8: даёт ли гибрид CPU+PCIe реальный выигрыш на 7H43 | E1 (базовая линия), E2 (локальность), E3 (CPU-полоса), E5 (all-reduce через хост) | E1–E5: только если A и D упираются в КПД <0.35 при понятной причине, которую нельзя устранить в ggml |

### 5.2 Предварительный рейтинг (не финальный; данных с сервера нет)

1. **A — базовая линия и, возможно, итог.** Самая дешёвая проверка. llama.cpp и ik_llama уже исполняют все целевые модели, кроме DeepSeek-V4.1 в mainline (в ik_llama она есть).
   Если по итогам E1 на сервере ≥35 tok/s на 128K, дальше нужна только тонкая настройка и MTP/DFlash.
   Если модель меняется на Qwen3.8-Flash-Next (решение R1 и заказчика), A у Strata апстрима почти наверняка закрывает 50 tok/s (HYPOTHESIS по строкам 13–17).
2. **D — свой планировщик поверх ggml.** Единственное направление, где видны конкретные рычаги сверх A: TP плотной части, CPU-расчёт промахов вместо подгрузки, размещение по профилю. Код остаётся в экосистеме, которую поддерживает апстрим.
3. **C — как поставщик деталей для D.** Q*-политика FreeToken, AVX2-ядра Strata, native TP-бэкенд ExLlamaV3. Отдельной цели не имеет.
4. **B.** Ни одна база на реальных весах не показала преимущества над ik_llama на одном и том же хосте (§3.4). База B ограничена 2 GPU и Q2/Q3/IQ4_XS, база A — NVFP4 одной модели. Оставить как источник идей (размещение по стадиям, `--cpu-share`) и как резерв.
5. **E.** Только при доказанном потолке A и D, который нельзя поднять внутри ggml.

---

## 6. Решающие эксперименты (по порядку)

| # | эксперимент | команда, метрика | что решает |
|---|---|---|---|
| E0 | Железо (G0 из PLAN.md) | `dmidecode -t memory`, STREAM triad; `nvidia-smi -q` (Gen/ширина под нагрузкой); `nvidia-smi topo -p2p r` и `p2pBandwidthLatencyTest` **внутри VM**; `numactl -H` (NPS); `fio` 4K/16K QD32 | 4 или 8 каналов (CPU-доля ×2); возможны ли P2P и TP в VM Proxmox. Патч драйвера aikitoria/tinygrad (RTX 30/40/50, нужны ReBAR и `iommu=pt` [29]) в VM — HYPOTHESIS: сложно |
| E1 | Матрица A: GLM-5.3-Flash UD-IQ4_XS и UD-Q4_K_XL (`glm5next`, ik_llama) / `glm5-next` Q4_K (llama.cpp) | `llama-bench` pp512/tg128 и `llama-server` с агентным трассом при ctx 8K / 64K / 200K–262K; `-sm layer`, `-ncmoe N`, `--moe-cache-mib`, MTP вкл./выкл.; 3 повтора, медиана | Уровень A. Порог: ≥35 tok/s на 128K → A + доводка; ≤20 → нужна D |
| E2 | Локальность маршрутизации на реальном агентном трафике | `tools/g0` (гистограммы по слоям, кривая попаданий для 4×26 GiB), повтор эксперта t→t+1 | Есть ли перекос, на котором кэш и D дадут больше ×1.3. «Плоская» маршрутизация (24.6% повтора, строка 7; «extremely flat», [24]) ограничивает все кэши |
| E3 | Микробенч CPU-ядер экспертов на 7H43 | эффективные GB/s GEMV Q4_K/IQ4_XS: ggml AVX2 против `iq_avx2` Strata против AVX2-пути FreeToken/KT, 1…48 потоков, NPS1/NPS4 | Реальная цена CPU-яруса; сравнение «считать на CPU» против «везти по PCIe» (q*) |
| E4 | Спекуляция при выгрузке | llama.cpp: `--spec-type` MTP n=1..3 и DFlash с драфтером `canada-quant/GLM-5.3-Flash-DFlash2-G` (ADR-011); принятие и чистые tok/s | Даёт ли спекуляция ×1.2+ при выгрузке (ADR-011, п. «риски») |
| E5 | TP без P2P | ExLlamaV3 native TP-бэкенд (на модели, которая влезает в VRAM) и llama.cpp `-sm tensor` (на архитектуре с поддержкой, например MiMo/deepseek4); время all-reduce на слой | Окупается ли TP плотной части для D/E; «±» у сообщества: P2P почти ничего не дал llama.cpp, выигрыш — у all-reduce ядра vLLM [30] |
| E6 | Класс «влезает в VRAM»: Qwen3.8-Flash-Next UD-Q4_K_XL | Strata v0.1.41, 4 карты, auto layer split, 262K, агентный трасс; то же на llama.cpp | Верхняя граница A; вход для решения R1 о модели |
| E7 | MiMo-V2.6-Flash | загрузка GGUF V2.6 в llama.cpp `mimo2` и ik_llama (`-sm graph` ≤4 GPU) | Поддержан ли V2.6 на деле (сейчас UNKNOWN) |
| E8 | FreeToken на одной 4080 | `ft bench bw` и `ft serve --moe-strategy hybrid` с GLM-5.3-Flash NVFP4 (Marlin на sm_89) | Сколько даёт q*-гибрид на Ada и DDR4 — цена пересадки в D |
| E9 | KT + SGLang tp4 на AVX2 | GLM-5.3-Flash GPTQ_INT4/LLAMAFILE, `--kt-method`, `kt doctor` | Есть ли готовый TP+CPU-эксперты путь для GLM на этом CPU (сейчас UNKNOWN; документы противоречат) |

---

## 7. Негативные находки

1. **sergqwer/strata-glm не архивирован.** ADR-007 пишет «в архиве», флаг GitHub `archived:false` (DISPROVED). «Archived experiment» — только формулировка автора. Коммитов после пина нет.
2. **lt-Strata glm53f:**
   - максимум 2 GPU (`glm_decode.cpp:6686-6687`);
   - Q4_K/Q5_K/Q6_K не исполняются, только перепаковка в Q2_K/Q3_K (`model.hpp:308-312`);
   - префикс-кэша и парсинга tool calls нет, только greedy;
   - новые направления автора (REAP-50 — прунинг экспертов; Q2-паки) противоречат требованиям «≥4 бит» и «точная маршрутизация».
3. **Strata апстрим по-прежнему однопородный** (`gguf_reader.hpp:584@fb58e0d`). `STRATA_ROUTE_RESIDENT` меняет вывод.
4. **NInfer и Infernix — только sm_120a** (`CMakeLists.txt:3-12` у обоих): к 4080 неприменимы. Сообщество сделало порты NInfer на sm_89, но только для плотной Qwen3.8-27B (поиск GitHub, REPORTED).
5. **FreeToken:**
   - один GPU («tensor parallelism is not supported yet», `server/args.py:823`);
   - спекулятивного декода нет;
   - GGUF — только для gemma4;
   - MiMo не поддержан.
6. **vLLM:**
   - индексер GLM5-Next требует DeepGEMM, поддержан только на Hopper и Blackwell (`sparse_indexer.py:699-701`, `deep_gemm.py:111-115`) → GLM-5.3-Flash на sm_89, скорее всего, не запустится (HYPOTHESIS);
   - кастомный all-reduce выключен при >2 PCIe-only GPU (`custom_all_reduce.py:270-276`);
   - CPU-расчёта экспертов нет.
7. **llama.cpp:**
   - `-sm tensor` выключен для GLM5_NEXT, KIMI_K3, MINIMAX_M3 (`llama-arch.cpp:1205-1240`);
   - MoE-кэш несовместим с `-sm tensor` и на промахе только подгружает эксперт по PCIe;
   - арки `deepseek41` нет;
   - строку `glm5next` (файлы Unsloth UD) не знает.
8. **ik_llama.cpp:** `-sm graph` не поддержан для glm5next, deepseek4/41 и qwen4exp (`src/llama.cpp:4216-4250`). README предупреждает о тарабарщине при сочетании graph и частичной выгрузки (README:22).
9. **KTransformers:**
   - документированный путь GLM-5.3-Flash требует ≥350 GB RAM и AVX-512 FP8 ядро — на этом сервере неисполним как описано;
   - про AVX2 и FP8 документы противоречат друг другу (`AVX2-Tutorial.md` против `kt-kernel/README.md:192`).
10. **ExLlamaV3:** CPU-выгрузка MoE экспериментальная и только для layer-split (не вместе с TP). Сервер TabbyAPI — **AGPL-3.0**.
11. **LMDeploy:** нет GLM-5.3-Flash и MiMo. **mistral.rs:** из кандидатов только Qwen4Exp. **TensorRT-LLM:** нет MiMo-V2 и CPU-экспертов.
12. **PowerInfer** переехал в Tiiny-AI, кандидатов не поддерживает. **Fiddler** заброшен (последний коммит 2024-04-28). **MoE-Infinity** по умолчанию собирается под sm_80/sm_90, GLM в нём идёт через transformers с eager-attention.
13. **Pulsar** не поддерживает GLM-5.3-Flash, MiMo и DSV4.1 (README на `18f9900`).
14. **PROBLEMS.md:** «ik_llama: алиаса `glm5-next` нет» — для исходников DISPROVED (`src/llama-arch.cpp:90@5f89bfc`); причина отказа загрузки на хосте — UNKNOWN (§3.5).
15. Готовых замеров vLLM/SGLang/KT на 4×4090/4080 с выгрузкой для целевых моделей **не найдено**. Статья с x.com про неудачу на 4×3090 не открылась.

---

## 8. Источники

Клоны — `git clone --filter=blob:none` в песочницу, 2026-10-09; SHA — из `git ls-remote` и `git log` (§2.1).

1. Niko1221/Strata @`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` — `include/strata/artifact/gguf_reader.hpp:573-595`, `docs/MODELS.md`, `docs/MULTI_GPU.md`, `docs/UNSLOTH_Q4.md`, `bench/results/2026-10-04-community-rtx-4080s-iq3s/README.md`, `bench/results/2026-09-29-rtx3090-epyc-milan/README.md`. https://github.com/Niko1221/Strata
2. sergqwer/strata-glm @`ed37419fccd0c52e07d26d526a29c2098f547843`; метаданные (`archived:false`, MIT, описание) — GitHub MCP `search_repositories`. https://github.com/sergqwer/strata-glm
3. lighttransport/Strata, ветка glm53f @`27deb918394186590f77589773eabc99b1fa28ea` — `src/program/glm_decode.cpp`, `include/strata/core/model.hpp`, `docs/README_GLM53_FLASH.md`, `docs/GLM53_FLASH.md`, `docs/GLM53_V100.md`, `docs/GLM_Q2_NUMA_DECODE.md`, `docs/GLM_EXL3.md`; коммиты `861e1f9`, `ae343e5`, `1aa7b05`, `bd4c31f`, `aef3c60`. https://github.com/lighttransport/Strata/tree/glm53f
4. Пины и evidence репозитория: `AGENTS.md`, `docs/DECISIONS.md`, `docs/evidence/2026-10-03-lighttransport-fork.md`, `docs/evidence/2026-10-03-3090-review.md`, `PROBLEMS.md`, `tools/fitplan.py`.
5. Grigory-Rylov/strata-glm-3090 @`794cd69ae270389c67ed201b855b14aafb5eb875` — `PROBLEMS.md`. https://github.com/Grigory-Rylov/strata-glm-3090
6. ggml-org/llama.cpp @`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` — `src/llama-arch.cpp`, `src/llama-moe-cache.{h,cpp}`, `common/arg.cpp`, `common/common.h`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `tools/server/README.md`, `conversion/`; коммиты `649dcb1`, `b9acf13`, `d6cf9ac`, `c811cb8`, `a46709b`, `033df86`, `10f340d`, `22bdcc4`. https://github.com/ggml-org/llama.cpp
7. ikawrakow/ik_llama.cpp @`5194a9e1451b885e24dd5a065f650864e8807c59` (и `5f89bfc` для §3.5) — `src/llama-arch.cpp`, `src/llama-model-loader.h`, `src/llama.cpp:4216-4250`, `common/common.cpp`, `README.md`; коммиты `42a9a2f`, `d741de5`, `043ced9`, `a761619`, `af5817d`, `c84f76c`, `28fbe34`, `912e9eb`. https://github.com/ikawrakow/ik_llama.cpp
8. kvcache-ai/ktransformers @`2ac8cf9c88fa7d6de106bcc3ab3ef7645e7476a0` — `README.md`, `kt-kernel/README.md`, `doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md`, `doc/en/kt-kernel/AVX2-Tutorial.md`. https://github.com/kvcache-ai/ktransformers
9. vllm-project/vllm @`ed7ae705a1f366c2008964af562a7543e91d934e` — `vllm/model_executor/models/registry.py`, `vllm/distributed/device_communicators/custom_all_reduce.py`, `vllm/utils/deep_gemm.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`, `vllm/models/glm5next/common/kda.py`, `vllm/engine/arg_utils.py`. https://github.com/vllm-project/vllm
10. giannisanni/pulsar @`18f990094e1091ee2ef3078f20be418f8c162f03` — `README.md`. https://github.com/giannisanni/pulsar
11. sgl-project/sglang @`46c2585faa4fcb6fa8febf4d2d4534a7cf33da79` — `python/sglang/srt/models/`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/mem_cache/hicache_storage.py`. https://github.com/sgl-project/sglang
12. NVIDIA/TensorRT-LLM @`49ec5730428f8eb0434e01dc9b4c532136d6c578` — `tensorrt_llm/_torch/models/`. https://github.com/NVIDIA/TensorRT-LLM
13. turboderp-org/exllamav3 @`151539c77abc7ab7425d30da7a4e8e3c5c154e7b` — `README.md`, `doc/env_vars.md`, `exllamav3/architecture/`, `exllamav3/model/model_tp_backend.py`. https://github.com/turboderp-org/exllamav3
14. FlashML-org/FreeToken @`2d8b9cc1eac2e8f2ae63eb0ae90fe419252d9dc6` — `README.md`, `docs/models.md`, `docs/cli.md`, `python/freetoken/server/args.py`, `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`, `python/freetoken/layers/quantization/moe/nvfp4.py`, `python/freetoken/models/gguf/config.py`. https://github.com/FlashML-org/FreeToken
15. FreeToken, статья arXiv:2608.16157 («FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution», подана 2026-08-17). Сама страница не открылась; заявление «1.5–2.3× декода против llama.cpp, Ollama, KTransformers» — REPORTED по вторичным источникам: https://www.alphaxiv.org/abs/2608.16157, https://forum.level1techs.com/t/freetoken-an-llm-engine-to-max-the-bandwidth-of-all-the-things/254396
16. theroyallab/tabbyAPI @`884e88cc1fdd2dfcd5ed27f71ff5b41f52541e1f` — `LICENSE` (AGPL-3.0). https://github.com/theroyallab/tabbyAPI
17. InternLM/lmdeploy @`822dddb8b3bc895eaf6cffb45a5514768e7ecb84` — `docs/en/supported_models/supported_models.md`. EricLBuehler/mistral.rs @`3f2515e9b5adc2ac44949c128c50a13b15721294` — `README.md`, `mistralrs-core/src/pipeline/loaders/mod.rs`.
18. EfficientMoE/MoE-Infinity @`0f96d563db2dda5f1834bf667f6fcc44f45b2b9d` — `README.md`. Tiiny-AI/PowerInfer @`8bd56d69906c9d2dba4d3bf6899763401e01a9a4` — `README.md`. efeslab/fiddler @`227715bfd6e8c731b29548eab01d9919c4fe9564`. LMCache/LMCache @`ee68e5ba955cb51147767c207fc462932b87ab04`.
19. Neroued/ninfer @`81c8ce093b2c1646a87566a8e59d807fcf0ec95c` и Wallawalla47/Infernix @`a3edb450f373d1044a1589cf4e742a4e7121c14f` — `README.md`, `CMakeLists.txt:3-12`.
20. club-3090 issue #1543 «[bench] GLM 5.3 Flash with 4x 3090 / llama.cpp/glm53-flash-multi4-iq4xs-moecache»: https://github.com/noonghunna/club-3090/issues/1543 (репозиторий club-3090 HEAD `6305d83a6357bb32dc0794dce7e600919f8ea1d8`).
21. club-3090 issue #1351 «GLM-5.3-Flash Q2_K dual-3090»: https://github.com/noonghunna/club-3090/issues/1351
22. ik_llama.cpp PR #2376 «model: Add GLM-5.3-Flash (glm5next) runtime support» (слит 2026-09-14 как `42a9a2f`): https://github.com/ikawrakow/ik_llama.cpp/pull/2376
23. 0xSero/glm53-flash-offload @`6769b27652a18487758caac45a2b6999dc0a4184`: https://github.com/0xSero/glm53-flash-offload
24. Cerynitius/freetoken-ox-boost @`fdb99c6daae162094c8e2a80a5c177136c5e52f6` — `README.md`: https://github.com/Cerynitius/freetoken-ox-boost
25. Helix ML, «glm53 flash on rtx pro 6000 part 1» (только поисковый фрагмент; WebFetch: ENOTFOUND): https://deploy.helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1
26. llama.cpp discussion #30071 «Qwen3.8-Flash-Next on 6x3090 and 6x4090 without NVLink»: https://github.com/ggml-org/llama.cpp/discussions/30071 (форк lukolszewski/llama.cpp-multigpu HEAD `ee5d54afd20d94aaea3ed097837eb079b96aa674`).
27. club-3090 issue #1133 «2x 5090 + DDR4 3200 + 3975x TR / qwen38-flash-next-q4kxl-moecache»: https://github.com/noonghunna/club-3090/issues/1133
28. X, analogalok, DeepSeek V4 Flash 0731 на одной 4090 (только поисковый фрагмент): https://x.com/analogalok/status/2084274615829102618
29. aikitoria/open-gpu-kernel-modules `README.md` @`461d638b7a91f73702001e343ec91b275a43ec35` (HEAD `c8bff5400e2fc8bda4cb9a141fb596aad607a35f`): «P2P for RTX 30, 40, and 50 series», нужны ReBAR и `iommu=pt`. Основа — tinygrad/open-gpu-kernel-modules (HEAD `9e39420bc4cb50cd7f8033ac6dd5c5583fd07567`). https://github.com/aikitoria/open-gpu-kernel-modules
30. Поисковые фрагменты о TP без P2P: https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870 ; https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md ; https://www.pugetsystems.com/labs/hpc/problems-with-rtx4090-multigpu-and-amd-vs-intel-vs-rtx6000ada-or-rtx3090/ (страницы не загружались, REPORTED).
31. Соседний отчёт R1 (модели): `docs/reassessment/research/R1-models.md` — только для согласования классов моделей (Qwen3.8-Flash-Next влезает в VRAM; GLM — VRAM+RAM).
