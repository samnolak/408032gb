# strata-glm: ярусный путь на Linux — пошаговый запуск

> **Канонический исходник доработок:** `git@github.com:Grigory-Rylov/strata-glm-3090.git`,
> ветка `main` (форк sergqwer/strata-glm + наши 15 коммитов; патчи 0001–0004 ниже — их же
> git-format-экспорт, применять не нужно, они уже в ветке).
> **Сборка:** `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`
> `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DSTRATA_ENABLE_CUDA=ON`
> `-DSTRATA_BUILD_TESTS=OFF -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<клон llama.cpp>`
> (последний флаг — только когда сеть занята: ggml берётся из локального клона), затем
> `make strata-glm -j16`. Проверено 03.10: бинарь из чистого клона форка дал
> **437.84 tok/s @16K / 19.78 decode** — идентично локальной сборке.

Рецепт, давший на 1×RTX 3090 (GPU2) 03.10.2026 на тестовом паке (`../glm-synthetic/`):

| режим | prompt (1024 токена) | decode |
|---|---|---|
| `--chunk 0` (per-token) | 19.58 tok/s | 19.69 tok/s |
| `--chunk 512` | 66.18 tok/s | 19.85 tok/s |
| `--chunk 1024` | 124.18 tok/s | 19.81 tok/s |
| `--chunk 2048` | 132.02 tok/s (плато короткого промпта: 1024 = один чанк) | 19.83 tok/s |
| промпт 16K, `--chunk 4096 --max-context 16384` | **437.98 tok/s** | — |
| промпт 64K, `--chunk 4096 --max-context 65536` | **420.74 tok/s** | — |

Логи: `prefill_chunk{0,512,1024}.log`; разбивка `GLM_TIMING` в них же. Движок — `strata-glm` с четырьмя патчами
из `../patches/strata-glm/` (без 0004 ярусный путь на Linux не работает).

## 0. Требования

| | |
|---|---|
| GPU | одна карта >= 22 GiB свободной VRAM (у нас GPU2, `CUDA_VISIBLE_DEVICES=2`) |
| RAM | **>= ~166 GiB свободной** (ярус ~142 GiB закрепляется через `cudaHostRegister`) + dense |
| диск | NVMe под `experts.bin` (~160 GiB); при холодном кэше первая загрузка яруса дольше |
| софт | Linux, gcc 13, cmake+ninja, CUDA nvcc 13.x (`/usr/local/cuda-13.2`), арх `sm_86` |

## 1. Сборка

```bash
git clone <sergqwer/strata-glm> && cd strata-glm && git checkout ed37419
git am ../408032gb/patches/strata-glm/000{1,2,3,4}-*.patch   # 0004 = ярус на Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target strata-glm -j
```

## 2. Pack

Движок читает **только pack-формат** (`dense.txt` + `experts.bin` + опц. `native_experts.txt`);
GGUF он не открывает.

- **Боевые веса:** канонические — GGUF `UD-Q4_K_XL` (путь в `../AGENTS.md`); pack строится из них
  конвертером GGUF→pack (в разработке). Апстрим-скрипт `tools/glm_pack.py` ждёт другой формат
  входных файлов и Windows-only — на этой машине не используется.
- **Тестовый пак (на машине уже готов: `/mnt/data/apps/glm-synthetic/`):**
  `python3 gen_synth.py` строит `ckpt/weights.bin` + `pack/dense.txt` + `pack/experts.bin`
  (случайные значения, структура боевая). Для ярусного пути нужны ещё два файла:

`pack/native_experts.txt` — 42 строки (по MoE-слою), смещение и размер блоба эксперта в experts.bin:

```
# native experts (synthetic, GLM-5.3-Flash geometry: n_embd 4096 n_ff 2048 swiglu_limit 10)
<l> 40 40 <l*288*14155792> 14155792        # для l = 0..41; 40 = тип блоба (нативный), 288 экспертов
```

`profile.bin` — рейтинг «горячих» экспертов для ярусов (парсер `read_expert_profile`,
`src/core/expert_cache.cpp`): magic `STRP`, заголовок 5×u32 `(version=1, n_layers=42,
n_expert=288, slots=12096, n_ranked=12096)`, далее `n_ranked` пар uint16 `(layer, expert)`:

```python
import struct
hdr = b"STRP" + struct.pack("<5I", 1, 42, 288, 12096, 12096)
pairs = [(l, e) for l in range(42) for e in range(288)]        # тест: все эксперты «горячие»
open("profile.bin", "wb").write(hdr + b"".join(struct.pack("<HH", *p) for p in pairs))
```

## 3. Запуск

```bash
GPU=2 ../scripts/run-strata-tiered.sh            # портит все флаги, GLM_TIMING=1 включён
```

или вручную:

```bash
cd /mnt/data/apps && CUDA_VISIBLE_DEVICES=2 GLM_TIMING=1 \
  strata-glm-build/strata-glm \
  --pack glm-synthetic/pack --tokens glm-synthetic/pack/tokens.txt \
  --chunk 0 --max-new 32 --skip-disk 0 --skip-ram 0 --dense-fp4 all \
  --profile glm-synthetic/profile.bin
```

Опции яруса: `--ram-gib N` (потолок яруса; при «cannot reserve the host tier» — 120),
`--vram-experts N` (сколько экспертов держать в VRAM; по умолчанию авто от свободной VRAM).

## 4. Что должно быть в логе (замер 03.10, GPU2)

```
experts: 1233 in VRAM (16.3 GiB, filled in 2.6 s), 10755 in RAM (141.91 GiB, read in 172.5 s;
         43 of 43 slices pinned; MAP_HUGETLB unavailable ... using 4 KB pages), 108 in experts.bin
prompt 40 tokens in 2.6 s (15.23 tok/s)
decode 24 steps in 1.2 s (19.65 tok/s)
decode experts: 99.2% VRAM, 0.8% RAM to the GPU, 0.0% disk (0.0 ms a token waiting for the disk)
a decode token 49.4 ms: expert copies 13.0, expert kernels 0.4, the rest 36.1
```

Если вместо этого `the low-RAM tier is Windows-only` — не применён патч 0004.
Если `cannot reserve the host tier` — мало свободной RAM, `RAMGIB=120`.

## 5. Ограничения

- Скорости выше — на **тестовом паке** (случайные веса): механика (память/копии/ядра) настоящая,
  качество модели — нет. На реальных весах pack пока не собран (п. 2).
- Загрузка яруса ~175 s на старте (page cache тёплый); hugetlb-пул на машине не настроен,
  ярус на 4 KB-страницах — на скорость не влияет, только на время регистрации.
- **Длинные промпты — другой режим и другие цифры.** На 1024-токеновом промпте все 12 096 экспертов
  стримятся по PCIe ради 1024 токенов (~167 MB/токен) — потолок шины ~190 tok/s, отсюда «плато» 132.
  На 16K/64K тот же стрим амортизируется в 16-64 раза и узлом становятся MMQ-вычисления:
  **438 tok/s @16K, 421 tok/s @64K** (`--chunk 4096`; логи `prefill_16k_c4096.log`, `prefill_64k_c4096.log`).
- `--chunk 8192` (рецепт апстрима на 32 GB 5090) на 24 GB 3090 **не влезает**: MMQ-буферы
  `m_gu`/`m_h` = T·kK·2·kFF·4 + T·kK·kFF·4 ≈ 1.6 GB при T=8192 плюс KV — OOM в `init_prompt_mmq`
  (glm_main.cpp:646). Для 24 GB рабочий чанк — 4096.
