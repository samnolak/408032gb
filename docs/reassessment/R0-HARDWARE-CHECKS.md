# R0: проверки железа — каналы DRAM, P2P, NVMe, лицензия Qwen (план, 2026-10-11, N24)

**Статус: план. На сервере ничего не измерено (NOT RUN).** Документ дополняет `HW_DIAG_RUNBOOK.md` (как запускать
комплект `scripts/diag/`) и `HARDWARE_FACTS.md` (строки F1–F20): здесь — что именно проверить в четырёх вопросах, которые
владелец поднял в задаче N24, какие числа ожидать и что делать с ответом. Метки — по `AGENTS.md`: CONFIRMED (с источником),
PROVISIONAL (оценка, сказано как получена), UNKNOWN.

**Решения владельца (переданы lead-architect в задаче N24, 2026-10-11; письменного документа заказчика в репозитории нет —
текст решения в первоисточнике UNKNOWN, записано в `docs/DECISIONS.md`, ADR-012.2):**

1. драйвер NVIDIA **не патчить** (никаких P2P-патчей драйвера, в том числе из `BIOS_PLAN.md` шаги 2 и 6 «предпосылка
   P2P-патча»); P2P проверяется только штатными средствами;
2. NVMe: **планируемая конфигурация 3 × 1 ТБ = 3 ТБ** (поправка заказчика 2026-10-12; в N24 было записано «4 × 1 ТБ» — неверно);
   модели дисков, реальная скорость и PCIe-топология — UNKNOWN, проверяются в R0;
3. Qwen (Qwen Community License 1.0): **только локальная разработка и тесты**; коммерческое предоставление третьим лицам
   **не согласовано**.

---

## (a) Сколько каналов DRAM активно

### Что известно

| факт | значение | метка |
|---|---|---|
| память | 128 GB DDR4-3200 | заявление заказчика (UNVERIFIED), `hardware/4x4080s-32g.json` |
| плата | HUANANZHI H12D-8D rev 2.0, 8 слотов DDR4, по 1 DIMM на канал | заявление (F7) + спецификация производителя: REPORTED, `research/R4-hardware.md:76` |
| CPU | EPYC «7H43» (Zen 3, Milan, SP3: 8 каналов DDR4-3200 на сокет) | SKU публично не найден (F1); 8 каналов у платформы — REPORTED (спецификация платы, `research/R4-hardware.md:76`) |
| варианты набора 128 GB | 8 × 16 GB → 8 каналов; 4 × 32 GB → 4 канала; 2 × 64 GB → 2 канала | арифметика; какой набор стоит — **UNKNOWN** |
| `ram_channels` | `null` | UNKNOWN, `hardware/4x4080s-32g.json` |

Почему важно: CPU-ярус экспертов ограничен полосой DRAM; 4 канала вместо 8 делят её примерно вдвое (оценка R4, ниже).

### Ожидаемые числа

| каналы | теоретический пик | ожидание STREAM Triad (все ядра, 1 сокет) | метка |
|---|---|---|---|
| 8 | 8 × 3200 MT/s × 8 B = 204.8 GB/s | ≈ 170–180 GB/s | пик — арифметика (CONFIRMED); Triad — HYPOTHESIS `research/R4-hardware.md:169` (из ≈ 85 % пика на Milan, REPORTED `:167`) |
| 4 | 102.4 GB/s | ≈ 90–100 GB/s | то же; доля 54 % от 8 DIMM — REPORTED `:168` |
| 2 | 51.2 GB/s | ≈ 45–50 GB/s | то же |

Ориентир для того же класса CPU: 136 GB/s чтения на 22 потоках у EPYC 7443P (Zen 3, 8 каналов) — REPORTED
(`research/R11-verification.md:60`, glm-flash-lite). Порог «красного флага» из R4 (HYPOTHESIS, `research/R4-hardware.md:584`):
Triad < 150 GB/s при 8 DIMM или < 80 GB/s при 4 DIMM.

### План измерения (по порядку; всё только читает систему или нагружает обычную память процесса)

Где запускать: `dmidecode` и EDAC показывают реальные модули **только на хосте** (Proxmox). Внутри VM `dmidecode` описывает
виртуальные DIMM QEMU — это не ответ. Есть ли у нас доступ к хосту — UNKNOWN (вопрос заказчику). Полосу памяти мерить и
на хосте, и в VM: разница покажет, сколько теряет виртуализация (vCPU pinning, NUMA-топология гостя).

1. **Модули и частота (хост, root):**
   ```
   sudo dmidecode -t 17 | grep -E "Locator|Size|Speed|Rank|Part Number|Manufacturer"
   ```
   Число записей с `Size:` не «No Module Installed» = число DIMM; при 1 DIMM на канал (плата с 8 слотами) это и есть число
   каналов. `Configured Memory Speed` должен быть 3200 MT/s (`BIOS_PLAN.md` п. 14, если ниже). То же собирает
   `scripts/diag/collect.sh --sudo` (проба `memory.dmidecode_16_17`).
2. **Каналы по контроллерам (хост):** драйвер `amd64_edac` показывает DIMM по каналам UMC:
   ```
   ls /sys/devices/system/edac/mc/; grep -H . /sys/devices/system/edac/mc/mc*/dimm*/dimm_label /sys/devices/system/edac/mc/mc*/dimm*/size 2>/dev/null
   edac-util -v   # если пакет установлен
   ```
   Число `mc*` с ненулевым `size` — число заполненных каналов (PROVISIONAL: так раскладывает amd64_edac на Zen; проверить на месте).
3. **NUMA (хост и VM):** `lscpu`, `numactl -H`, `grep MemTotal /sys/devices/system/node/node*/meminfo`. При NPS4 и 4 каналах
   часть NUMA-узлов останется без памяти — красный флаг.
4. **Полоса, основной инструмент — наш `scripts/diag/bench_mem.sh` (membw.c, Triad):**
   ```
   bash scripts/diag/bench_mem.sh --run --out diag-out/<каталог>
   ```
   Он гоняет 1 поток; все ядра с `OMP_PROC_BIND=spread` и `close`; все потоки; по NUMA-узлам (`numactl --cpunodebind=N
   --membind=N`); `--interleave=all` (`HW_DIAG_RUNBOOK.md`, шаг 3). Дополнительно снять кривую по числу потоков
   (1, 2, 4, 8, 16, 24, 32, 48): при 4 каналах насыщение наступает раньше (PROVISIONAL: колено около 12–16 потоков против
   24–32 при 8).
5. **Альтернативы для сверки (по одной, не обязательно все):**
   - STREAM (McCalpin, `stream.c` в офлайн-бандле): `gcc -O3 -march=znver3 -fopenmp -DSTREAM_ARRAY_SIZE=400000000
     -DNTIMES=20 -mcmodel=medium stream.c -o stream && OMP_NUM_THREADS=48 OMP_PROC_BIND=spread OMP_PLACES=cores ./stream`
     (массивы 3 × 3.2 GB — больше любого кэша);
   - `likwid-bench -t load_avx -w S0:16GB:48` и `-t stream_avx` (LIKWID поддерживает Zen 3; PROVISIONAL — проверить версию);
   - Intel MLC (`mlc --max_bandwidth`) на AMD **официально не поддерживается**; если запустится, считать вторичным числом;
     матрицу задержек MLC на AMD не использовать;
   - `perf stat` по счётчикам UMC/DF (`amd_df`, `amd_umc`) — показывает трафик по каналам напрямую, если ядро хоста их
     экспортирует (UNKNOWN; в VM почти наверняка нет).
6. **Решение:** число DIMM из шага 1 — основной ответ; Triad из шага 4 — проверка. Совпадение «8 DIMM и Triad ≥ 150 GB/s»
   или «4 DIMM и 80–100 GB/s» закрывает F4–F6; расхождение (8 DIMM, но Triad < 120 GB/s) — искать причину: частота ниже 3200,
   NPS, разные ранги, канал выключен BIOS, VM без NUMA-раскладки.
7. **Запись:** `HARDWARE_FACTS.md` F4/F5/F6 с путями к выводам; `hardware/4x4080s-32g.json` → `ram_channels` и измеренный
   `ram_bw_gbs` (с меткой CONFIRMED и путём к файлу).

---

## (b) P2P между GPU — только штатная диагностика

Владелец запрещает патчить драйвер NVIDIA. Ожидание: на GeForce со штатным драйвером P2P выключен (REPORTED,
`HARDWARE_FACTS.md` F14, `research/R4-hardware.md`); внутри Proxmox-VM с пробросом GPU — тем более. Значение `p2p: true` в
`hardware/4x4080s-32g.json` устарело (`REPO_AUDIT.md` C2): статус **UNKNOWN** до замера.

### Диагностика (только чтение и штатные тесты)

На хосте Proxmox (root; если доступ есть):
```
cat /proc/cmdline                                   # iommu=pt / amd_iommu, pcie_acs_override (если есть — записать)
dmesg | grep -i -e AMD-Vi -e IOMMU | head -50
for g in /sys/kernel/iommu_groups/*; do echo "group ${g##*/}: $(ls $g/devices)"; done
lspci -tv                                           # дерево: на одном ли root complex / свитче карты
sudo lspci -vvv | grep -E "^[0-9a-f]{2}:|ACSCap|ACSCtl"   # ACS на корневых портах и мостах
qm config <vmid> | grep -E "hostpci|machine|numa|cpu"  # как GPU проброшены в VM (pcie=1, q35)
```
В VM (и на голом железе, если без виртуализации):
```
nvidia-smi topo -m                                  # PHB/NODE/SYS между картами
nvidia-smi topo -p2p r ; nvidia-smi topo -p2p w     # статус P2P чтения/записи (OK / NS / CNS)
nvidia-smi -q | grep -A3 -i "bar1"                  # размер BAR1
bash scripts/diag/bench_gpu.sh --run --out diag-out/<каталог>      # cudaDeviceCanAccessPeer + полоса D2D, H2D/D2H (gpu_p2p_matrix.cu)
p2pBandwidthLatencyTest                             # CUDA samples (собрать из офлайн-бандла)
nvbandwidth -t device_to_device_memcpy_read_ce      # NVIDIA nvbandwidth, если есть в бандле
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8K -e 64M -f 2 -g 4              # nccl-tests: какой транспорт выбрал NCCL
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 ./build/all_reduce_perf -b 8K -e 64M -f 2 -g 4   # явно без P2P: SHM через хост
```
Из лога `NCCL_DEBUG=INFO` записать транспорт каналов (`via P2P/IPC`, `via SHM`, `via NET`). Результат — строки F12–F14.

### Если P2P нет: варианты и как мерить каждый

Цифры ниже — оценки для выбора, PROVISIONAL до замеров.

| вариант | что передаётся между GPU на токен декода | как мерить | условие выбора |
|---|---|---|---|
| **A. TP с all-reduce через хост** (NCCL SHM-транспорт; в llama.cpp свой host-staged all-reduce есть только для ровно 2 GPU — REPORTED, `research/R6-local-engines.md:33-35`; ExLlamaV3 TP — all-reduce через pinned-память и CPU, REPORTED `:42`) | 2 all-reduce на слой (после внимания и после FFN), по ~8–64 КБ | `all_reduce_perf -b 8K -e 64K -f 2 -g 4` с `NCCL_P2P_DISABLE=1`: латентность p50/p99 (ROADMAP R4.1); затем `llama-bench -sm tensor` / движок, если поддерживает модель | цена на токен ≈ 2 × n_слоёв × латентность: при 46 слоях GLM и 30 мкс — 2.8 мс (14 % бюджета 20 мс при 50 tok/s), при 100 мкс — 9.2 мс (46 %): TP имеет смысл, только если латентность ≲ 30–40 мкс |
| **B. Конвейер по слоям (PP, `-sm layer`)** — текущий путь llama.cpp/Strata | одна передача активаций на границе стадии (3 границы на 4 картах) | `llama-bench -m <gguf> -sm layer -p 512 -n 128 -r 3` (стандартный бейзлайн); `nsys profile` декода: доля времени `cudaMemcpy` DtoH/HtoD на границах; H2D/D2H из `bench_gpu.sh` | без P2P — вариант по умолчанию: пересылка мала, задержка на границе ≈ две копии через хост |
| **C. Экспертный параллелизм (EP)**: эксперты слоя распределены по картам | dispatch/combine токена к картам, где лежат его top-k экспертов, через хост — 2 обмена на MoE-слой | `alltoall_perf -b 8K -e 1M -g 4` с `NCCL_P2P_DISABLE=1`; доля попаданий по картам из гистограмм маршрутизации (`tools/g0`) | при батче 1 цена как у A; осмысленно для пропускной способности при многих потоках |
| **D. «Каждая карта — свои слои + горячие эксперты, промахи на CPU»** (Strata/strata-glm: PP + VRAM-кэш экспертов + CPU-ярус) | как B | `tools/fitplan.py ... --hit-curve` с измеренной кривой; `llama-bench`/движок с `-ot exps=CPU` и кэшем | основной кандидат без P2P для 262K (ADR-012) |
| **E. Реплики (data parallel)** | ничего | — | только если модель целиком помещается на одну карту (не наш случай для GLM/MiMo/DS); для пропускной способности, не для задержки одного потока |

Решение: P2P есть (`topo -p2p r` = OK и `CanAccessPeer` = 1) → A/C на P2P мерить в R4; P2P нет → B/D по умолчанию, A — только
если латентность host-staged all-reduce на 4 картах уложится в порог выше.

---

## (c) NVMe: планируется 3 × 1 ТБ (поправка 2026-10-12)

Заказчик, 2026-10-12: планируемая конфигурация — **3 × 1 ТБ = 3 ТБ** (не 4 × 1 ТБ, как записано в N24 2026-10-11). Модели
дисков, реальная скорость, линк, PCIe-топология (к каким корневым портам/CCD подключены) и место в VM — UNKNOWN, проверяются в R0:
```
lsblk -d -o NAME,MODEL,SIZE,TRAN,ROTA ; nvme list ; sudo nvme id-ctrl /dev/nvme0 | grep -E "^mn|^fr"
lspci | grep -i "non-volatile"                      # проброшены ли контроллеры в VM или это виртуальные диски
bash scripts/diag/bench_nvme.sh --run --all /mnt/a/d,/mnt/b/d,/mnt/c/d --size-gib 64 --out diag-out/<каталог>
```
(`HW_DIAG_RUNBOOK.md` шаги 6–7; F15–F17).

Раскладка по 3 дискам (PROVISIONAL, решается замером H7/P3): диски 1+2 — веса и паки, диск 3 — состояния слотов, кэш префиксов,
логи, черновое место конвертеров; NVMe-ярус DeepSeek-V4.1-Flash (эксперты + Engram) — шардинг на все 3 диска. Ёмкость: обязательные
веса ≈ 1.35 ТБ из 3 ТБ; пересчёт всех чисел, зависевших от 4 дисков, и полный список правок — `docs/evidence/2026-10-12-n26b-nvme-3x1tb.md`.

Исторически (N24, 2026-10-11) — места, где было записано «3 NVMe / 3 × 990 Pro», и что с ними сделано тогда. После поправки
2026-10-12 число 3 снова верно; строки «4 × 1 ТБ», внесённые N24, исправлены в N26b (см. evidence выше):

| место | что сказано | действие N24 |
|---|---|---|
| `AGENTS.md:36` | «3x Samsung 990 Pro 1 TB NVMe» | **не изменено** (правит владелец / lead); предложение: «4x 1 TB NVMe (planned; earlier statement: 3x Samsung 990 Pro 1 TB); actual count and models UNKNOWN until R0» |
| `hardware/4x4080s-32g.json` (`nvme`: 3 записи) | 3 × «Samsung 990 Pro 1TB» | **изменено**: 4 записи «UNKNOWN 1 TB NVMe (earlier statement: Samsung 990 Pro 1TB)», метка «PLANNED … UNKNOWN until R0» |
| `tools/ceilings/tier_cost.py:78` | комментарий «hardware/4x4080s-32g.json still lists 3» | **изменено**: комментарий обновлён (значение `n_drives = 4` было и осталось) |
| `.claude/agents/infra-ops.md:20` | «model files on the 990 Pro drives» | не изменено (конфигурация агентов; правит lead); предложение: «on the NVMe drives (planned 4 x 1 TB; models UNKNOWN until R0)» |
| `docs/prompts/G0-dispatch.md:24` | `fio ... <file on each 990 Pro>` | не изменено (исторический промпт G0) |
| `scripts/diag/summarize.py:676`, `tests/test_diag_kit.py:632` | текст флага «990 Pro is Gen4» | не изменено (подсказка про Gen4, не про число дисков) |
| `docs/reassessment/research/R3-techniques.md:461, 488, 543, 801, 833` | расчёты «при 3 дисках», «3×7.45 ГБ/с» | не изменено (исследовательский срез; там же сказано «4×1 ТБ (UNKNOWN)» в :461) |
| `docs/reassessment/research/R7-serving-kv-longctx.md:389` | «3 NVMe striped @14 GB/s» | не изменено (срез) |
| `docs/reassessment/research/R1-models.md:157, 276`; `R4-hardware.md:413` | «3–4 × 990 Pro»; «устарело» | не изменено (срез; уже помечено устаревшим) |
| `docs/reassessment/REPO_AUDIT.md:131, 190`; `EVIDENCE_MATRIX.md:18, 78`; `HARDWARE_FACTS.md:24` | уже «4 × 1 TB (ранее 3 × 990 Pro)» | без изменений (согласовано) |
| `hardware/target-4x4080-32g.unverified.json:105` | уже «customer says 4 x 1 TB» | без изменений |

N26 (2026-10-12): `AGENTS.md` «Hardware» исправлен lead-architect по явному разрешению владельца — «3x NVMe SSD 1 TB =
3 TB total (PLANNED …; models, real speed and PCIe topology UNKNOWN until R0)»; запись решения — `docs/DECISIONS.md` ADR-012.3.

---

## (d) Лицензия Qwen

Записано в реестр решений `docs/DECISIONS.md` (ADR-012.2) и в `ROADMAP.md` R2.4. Кратко: Qwen3.8-Flash-Next (Qwen Community
License 1.0) — только локальная разработка и тестирование; предоставление модели, её выходов или возможностей третьим лицам
на коммерческой основе **не согласовано**; условие лицензии (CONFIRMED текст, `research/R11-verification.md:312-318`,
`halt95@28cbb72:LICENSE.qwen-community-1.0:9`): лицензиат, ведущий бизнес «Model as a Service or AI Work Assistant», обязан
получить отдельную лицензию Qwen до коммерческого использования; внутреннее использование освобождено, если ПО, его выходы и
возможности модели не становятся доступными третьим лицам. Полный текст лицензии (остальные пункты) в репозитории не
вендорен — UNKNOWN здесь.
