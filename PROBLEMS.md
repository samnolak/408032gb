# PROBLEMS — GLM-5.3-Flash Q4 на 3×RTX 3090 (ветка `3090`)

Практический прогон модели на реальном железе, отличном от целевого (`hardware/4x4080s-32g.json`).
Все цифры — литеральный вывод команд 2026-10-02 на хосте `epyc-ai`; ничего не оценено, если не помечено PROVISIONAL.

## Стенд (замер)

| | |
|---|---|
| GPU | 3× RTX 3090 24 GiB, **sm_86** (не sm_89, не 4 карты, не 32 GiB) |
| занято | GPU0 23598 MiB, GPU2 24004 MiB — чужие процессы; свободна **одна** карта, index 1 |
| RAM | 251 GiB всего; свободно ~173 GiB (движок Strata/Qwen3.8 держит 59.2 GiB RSS, порт 8082) |
| swap | 8 GiB, уже 6.3 GiB занято до нашего запуска |
| CPU | 96 потоков |
| движок | `ik_llama.cpp` `build-cuda` — единственный на машине, кто понимает `glm5next`; обновлён 03.10 до `5f89bfc` (с `9cba2e3`, glm5next цел) |

## Модель (замер из GGUF)

| файл | арх | размер | статус |
|---|---|---|---|
| `GLM-5.3-Flash-GGUF/UD-Q4_K_XL/*-00001-of-00006.gguf` | `glm5next` | 185.983 GiB (4.981 BPW) | **загружается**, 46 слоёв, 42 MoE (blk.3–blk.44), 288 экспертов, top-8 |
| `GLM-5.3-Flash-Q4_K.gguf` | `glm5-next` | 148 GiB | **не загружается**: ik_llama знает только `glm5next` (`src/llama-arch.cpp:88`), алиаса нет |

`/home/grishberg/llama.cpp` @`53b1389d0` и `llama.cpp.turbo` @`ba0d2b391` графа `glm5*` не содержат вовсе.

---

## Проблема 1 — загрузка убивается oom-killer'ом, на GPU при этом 0 байт

Дважды убито ядром до этапа оффлоада, поэтому «веса не ложатся на GPU»:

```
Allocating 173.37 GiB of pinned host memory, this may take a while.
kernel: Out of memory: Killed process 1057302 (llama-server) total-vm:188078368kB, anon-rss:181793288kB
kernel: Out of memory: Killed process 1058605 (llama-server) total-vm:202571620kB, anon-rss:181829056kB
```

Причина, по шагам (подтверждается логом загрузки):

1. `-ncmoe 46` помечает **все 42 MoE-слоя** как `CUDA_Host` — `CUDA_Host buffer size = 177528.81 MiB` = 173.37 GiB;
2. `-muge` склеивает `ffn_up_exps`/`ffn_gate_exps` (`merge_up_gate_exps: merging up/gate in layer N`), то есть эксперты обязательно копируются в **анонимную** память: `file-rss` у процесса 71 MiB на 182 GiB RSS — mmap-резидентности нет, вытеснять ядру нечего;
3. ggml просит зафиксировать всё это как pinned (`ggml/src/ggml-cuda.cu:1432`), pinned не вытесняется;
4. свободной RAM 173 GiB < 173.37 GiB + ~8 GiB остального → OOM.

`GGML_CUDA_NO_PINNED=1` сам по себе **не лечит**: 173.4 GiB анонимной памяти остаются 173.4 GiB, пининг только убирает шанс, что ядро вообще доживёт до аллокации. Третья смерть (19:46:50) — тот же OOM уже с `NO_PINNED=1`.

## Проблема 2 — 24 GiB VRAM достаточно, узкое место RAM

Разрез рабочей конфигурации (`-ncmoe 40`, ctx 8192, `-ub 512`):

```
llm_load_tensors:        CPU buffer size = 169176.81 MiB   (40 MoE слоёв, 165.2 GiB)
llm_load_tensors:      CUDA0 buffer size =  16902.87 MiB   (плотная часть 8.4 GiB + 2 MoE слоя 8.2 GiB)
llama_kv_cache_init:      CUDA0 KV buffer size =   236.31 MiB
llama_init_from_model:      CUDA0 compute buffer size =   318.50 MiB
```

Потребление по ярусам (замер):

| ярус | объём |
|---|---|
| эксперты, 42 слоя | 177 528 MiB = 173.4 GiB (RAM) |
| эксперты, 1 MoE слой | ~4.4 GiB |
| плотная часть + attention + embed/lm_head | 8550 MiB = 8.4 GiB (VRAM) |
| KV @131k (`-ctk q8_0 -ctv q8_0` + indexer f16) | 1598 MiB (VRAM) |
| KV @8k | 236 MiB (VRAM) |
| GPU compute-буферы @`-ub 2048` / @`-ub 512` | 4353 MiB / 319 MiB |

**Вывод: одной 24 GiB карты хватает** (фактически 18.4 GiB из 24). Нужно ~166 GiB свободной RAM + ~18 GiB VRAM, то есть ~184 GiB суммарно. На 3090-машине это проходит только ценой срезания контекста и переноса части экспертов в VRAM.

## Проблема 3 — деградация скорости из-за свопa

RAM после запуска: `245.1 / 251.5 GiB`, свободно 6.4 GiB, swap 6.3/8.0 GiB. Замеры одного и того же запроса подряд:

```
pp 115.5 tok/s | tg  8.9 tok/s | prompt 3050 tok / 26402 ms | gen 64 tok / 7180 ms
pp   16.5 tok/s | tg  7.0 tok/s | gen 64 tok / 9079 ms
```

Первый запрос после старта давал `tg 15.2 tok/s`. Падение до 7.0 совпадает по времени с заполнением swap. Конфигурация живёт «в ноль»: любой рост соседнего Strata-движка уронит один из двух процессов.

## Проблема 4 — pinned-память (быстрый путь) недоступна рядом с другим движком

`tg 12.6–13.1 tok/s` и `pp 290 tok/s` на этой же машине сегодня уже наблюдались (лог `server-v2-161520.log`, 16:37) — тогда 173.37 GiB pinned влезли. Сейчас не влезают: RSS Strata-движка вырос. Чтобы вернуть pinned, нужно ~166 GiB **свободной** RAM, то есть остановить `serve/server.py --port 8082` + `engine/strata` (pid 921236, 59.2 GiB RSS).

## Проблема 5 — из репозитория запустить нечего

Ветка `3090` ничего не меняет в этом пункте, фиксирую как факт:

- `docs/PLAN.md`: G3 `BLOCKED`, G4 `код этапа 1 написан; NOT RUN`, G5/G6 `BLOCKED`;
- GLM-движок — форк `sergqwer/strata-glm @ed37419` + `patches/strata-glm/000{1,2,3}`, на машине не склонирован и не собран (CI проверял только сборку под sm_89, на GPU не запускал);
- `scripts/gpu_day.sh` требует `CKPT=nvidia/GLM-5.3-Flash-NVFP4` (204 GB, safetensors) — в `/mnt/data/home/grishberg/models` такого чекпойнта нет, есть только GGUF;
- `hardware/4x4080s-32g.json` и `tools/fitplan.py` описывают 4×32 GiB; на 3×24 GiB с двумя занятыми картами их расклады неприменимы.

Рабочий путь сегодня один: `ik_llama.cpp` (см. `scripts/run-glm-3090-1gpu.sh`).

---

## Рабочая конфигурация (повторить) — тюнинг 03.10

```bash
GPU=2 scripts/run-glm-3090-1gpu.sh    # порт 8090; дефолты уже GPU2/ncmoe39/ub1024/t96/pinned
```

Ключи: `-ngl 999 -sm none -ncmoe 39 -fa on --dsa -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 --ctx-size 8192 -muge -t 96 --threads-batch 90`, pinned ON.

Матрица (промпт 2027 токена, `-n 128`, temp 0; логи и `results.tsv` — `../ik-llama-run/bench-0310/`):

| конфигурация | pp tok/s | tg tok/s |
|---|---|---|
| ncmoe40 ub512 pinned-off (база 02.10 была pp 115.5 / tg 8.9) | 129.5 | 10.3 |
| ncmoe40 ub1024 pinned-off | 144.7 | 5.6 (аномалия, цифра из лога) |
| ncmoe40 ub2048 pinned-off | 131.1 | 15.0 |
| ncmoe39 ub512 pinned-off | 140.7 | 15.3 |
| **ncmoe39 ub1024 pinned-on t96/tb90** | **166.8** | **15.0** |
| ncmoe39 ub1024 pinned-on t48 | 165.0 | 15.4 |
| ncmoe39 ub2048 pinned-off | CUDA OOM (21079 + 3620 > 24576 MiB) | |
| ncmoe38 все ub | CUDA OOM (CUDA0 buffer 25255 MiB > VRAM) | |

Итог: **pp +44%, tg +68% к базе**. Pinned 161.13 GiB влез рядом с движком Strata (available RAM падал до
~9 GB, oom-killer не сработал) — проблема 4 от 02.10 снята наблюдением. По pp прежний pinned-рекорд (290)
не превзойдён: он был на другом разрезе без `-dsa`/KV-кванта; по tg — вдвое выше. Сервер на 127.0.0.1:8090
(GPU2, 22.6 GiB) оставлен работать; `/completion` отвечает, warm tg 15.4. Для сравнения: strata-glm с
портированным ярусом — 19.65 tok/s decode (`../strata-glm/PROBLEMS.md`).

---

## Вопросы

1. **Квант канонический?** `UD-Q4_K_XL` (186 GiB, `glm5next`) или `Q4_K` (148 GiB, `glm5-next`)? Второй ik_llama не читает; если он канонический — нужен либо перенос поддержки `glm5-next`, либо перепаковка в `glm5next`.
2. **Гонять ли один процесс с другим движком?** GLM + Strata/Qwen3.8 вместе не оставляют запаса: 245/251 GiB и забитый swap. Либо GLM с pinned и быстрым pp/tg, либо оба одновременно и медленно.
3. **Целевой контекст.** 8k — временная мера, чтобы влезло. 131k стоит +1.4 GiB VRAM (VRAM позволяет) и упирается в ту же нехватку RAM/swap.
4. **sm_86 в плане не учтён.** CI и патчи (`ci.yml`, `docs/evidence/2026-10-02-strata-glm-sm89.md`) считаются для sm_89. Нужна ли сборка под sm_86 и кто её верифицирует?
5. **4 карты → 1 карта.** G4 (`--layer-split 14,25,35`) на этой машине проверяем только в варианте «одна свободная карта». Проверять этап 1 в `--layer-split` на одной карте (п. 2 README патчей) или считать гейт непроверяемым здесь?
6. **Профиль экспертов (G0).** `tools/g0/` требует GGUF, который «помещается в машину». На 173 GiB свободной RAM помещается только UD-Q4_K_XL с `-ncmoe`; корректно ли снимать гистограмму маршрутизации в таком усечённом режиме?
7. **Пак для Strata.** Строить ли pack из GGUF (не из NVFP4-чекпойнта, которого нет), чтобы вообще получить `strata-glm` на этом стенде?
