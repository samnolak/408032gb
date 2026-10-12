# HW_DIAG_RUNBOOK: диагностика железа сервера инференса (`scripts/diag/`)

**Статус (2026-10-09):** комплект написан и проверен в песочнице без GPU, без NVMe и без BMC. На целевом сервере он
**ни разу не запускался** (NOT RUN). Все сведения о сервере (4× RTX 4080 Super 32 GB, EPYC 7H43, HUANANZHI H12D-8D,
128 GB DDR4-3200, 3× NVMe 1 TB — план, поправка заказчика 2026-10-12, Proxmox) остаются заявлениями заказчика (UNVERIFIED). Комплект нужен как раз для
того, чтобы их проверить. Факты становятся CONFIRMED только из литерального вывода, который пришлёт заказчик (раздел 8).

## 1. Что это и что получится

Один основной скрипт `collect.sh` только читает систему: CPU, память, плату/BMC, PCIe, GPU, NVMe, ПО и сеть. Каждая
проверка («проба») сохраняется в отдельный файл: литеральная команда, время UTC, код возврата, полный stdout/stderr. Плюс
`manifest.json` со статусом каждой пробы и архив `.tar.gz`. Отдельно, только по явному флагу `--run`, запускаются
нагрузочные замеры: память (`bench_mem.sh`), GPU и P2P (`bench_gpu.sh`, с проверкой VRAM) и NVMe (`bench_nvme.sh`).
`summarize.py` строит из каталога черновик `HARDWARE_FACTS.draft.md`: таблица «факт / значение / метка» и список красных
флагов. Пропусков он не заполняет: если вывода нет, пишет UNKNOWN и причину.

Файлы комплекта (все в `scripts/diag/`):

| файл | что делает |
|---|---|
| `collect.sh` | точка входа, только чтение; `--out DIR`, `--sudo`, `--net-check`, `--timeout SEC` |
| `probes_sysfs.sh` | чтение `/sys` и `/proc` без root: PCIe-линки, BAR, AER, NVMe, hugepages, NUMA, EDAC, DMI |
| `summarize.py` | черновик фактов (только stdlib Python 3), сводка fio |
| `membw.c` + `bench_mem.sh` | собственный OpenMP-замер полосы памяти: copy/scale/add/triad/read |
| `gpu_p2p_matrix.cu` + `bench_gpu.sh` | список GPU, матрица P2P, H2D/D2H, D2D, одновременная загрузка всех карт, латентность 4 КиБ |
| `gpu_vram_check.cu` | заполнение почти всей свободной VRAM шаблонами и сверка (реальны ли 32 GB на карте) |
| `bench_nvme.sh` | fio: только чтение с O_DIRECT по тестовому файлу; сырое устройство только с `--readonly` |
| `bench_all.sh` | запускает выбранные замеры в тот же каталог и пересобирает архив |
| `bench_common.sh` | общий формат логов для замеров |

## 2. Гарантии безопасности

Что комплект **никогда** не делает. Это проверяют `tests/test_diag_kit.py` (статический поиск запрещённых команд и
запуск со «шпионскими» заглушками, которые записывают каждый вызов):

- не меняет настройки: никаких лимитов мощности, режима persistence, частот и сброса GPU в `nvidia-smi`; никаких
  `sysctl`, `cpupower ... set`, `setpci`, записи в `/sys` и `/proc`;
- не загружает и не выгружает модули ядра, не трогает службы (`systemctl` только `show`), не перезагружает машину;
- не пишет на блочные устройства: никаких `mkfs`, `dd of=`, `wipefs`, `nvme format/sanitize/write`, `blkdiscard`,
  разметки диска; `smartctl` только `-a`, без самотестов;
- BMC: `ipmitool -I open` только `mc info`, `fru print`, `sdr list` и `sel list`. Никаких power/chassis/raw/reset,
  `sel clear`, сетевого доступа к BMC (`-H`);
- ничего не устанавливает и не скачивает. Сеть трогает только `--net-check` (по умолчанию выключен): один HEAD-запрос
  к `https://github.com` с таймаутом 5 с. Даже `pip list` идёт с `--disable-pip-version-check`;
- root: без root пробы, которым он нужен, помечаются `needs-root` и **не запускаются**. С `--sudo` используется только
  `sudo -n` (без запроса пароля; если sudo требует пароль, пробы остаются `needs-root`);
- падение или зависание одной пробы не останавливает сбор: у каждой пробы свой таймаут (по умолчанию 120 с, `--timeout`).

Что комплект **пишет**:

| скрипт | куда пишет | что занимает |
|---|---|---|
| `collect.sh` | только `diag-out/<host>-<UTC>/` (или `--out`) и `<тот же каталог>.tar.gz` рядом | несколько МБ |
| `bench_mem.sh` | `<diag-out>/bench/mem/<UTC>/`: собранный бинарник и логи | RAM процесса: 3 × `--size-mib` (по умолчанию 6 GiB) |
| `bench_gpu.sh` | `<diag-out>/bench/gpu/<UTC>/`: бинарники и логи | на каждую GPU 2 × 256 MiB VRAM и 256 MiB закреплённой RAM |
| `gpu_vram_check` | только VRAM, выделенная им самим | почти вся свободная VRAM (минус `--margin-mib`, по умолчанию 512) |
| `bench_nvme.sh` (файл) | **новый** подкаталог `DIR/diag-fio-<host>-<UTC>/` с одним файлом `--size-gib` GiB; в конце удаляется | `--size-gib` GiB записи на SSD |
| `bench_nvme.sh --raw-readonly` | ничего (fio с `--readonly`) | — |

## 3. Что должно быть установлено

Установка без интернета в этот документ не входит: ниже только список. Если инструмента нет, проба получает статус
`not-available`, и сбор идёт дальше.

| пакет (Debian/Proxmox) | что даёт | без него |
|---|---|---|
| `pciutils` | `lspci`: топология, LnkCap/LnkSta, ACS, Resizable BAR | линки берутся из sysfs, нет ACS и дерева |
| `dmidecode` (нужен root) | DIMM: число, слоты, локаторы, частота; плата, BIOS, запись IPMI | **число каналов памяти останется UNKNOWN** |
| `nvme-cli` | `nvme list`, `id-ctrl`, `id-ns`, `smart-log`, `error-log` (root) | модель/прошивка/температура только из sysfs |
| `smartmontools` | `smartctl -a` (root) | нет второго источника SMART |
| `numactl` | `numactl -H`; привязка в `bench_mem.sh` по узлам NUMA | нет замеров полосы по узлам |
| `hwloc` (`lstopo-no-graphics`) | карта кэшей, узлов и PCI-устройств | — |
| `ipmitool` (+ `/dev/ipmi0`) | FRU, датчики, журнал SEL BMC | BMC = UNKNOWN |
| `linux-cpupower` | `cpupower frequency-info` | частоты только из sysfs |
| `edac-utils` / `rasdaemon` | счётчики ошибок памяти | только sysfs EDAC |
| `fio` | `bench_nvme.sh` | замер NVMe невозможен |
| `gcc` с OpenMP (`build-essential`, `libgomp`) | сборка `membw.c` | замер памяти невозможен |
| драйвер NVIDIA (`nvidia-smi`) | всё о GPU, `topo -p2p` | GPU = UNKNOWN (кроме PCIe из sysfs) |
| CUDA toolkit (`nvcc`) | сборка `gpu_p2p_matrix.cu`, `gpu_vram_check.cu` под `sm_89` | P2P-матрица и проверка VRAM невозможны |
| `python3` | `summarize.py` (черновик фактов, сводка fio) | сырые выводы полные, черновика нет |

`/dev/ipmi0` появляется только при загруженных модулях `ipmi_si`/`ipmi_devintf`. Комплект их **не** загружает, потому что
это изменение системы. Если загружать их нельзя, FRU и журнал SEL можно снять через веб-интерфейс BMC.

## 4. Где запускать

Сервер исторически работает как хост Proxmox с изолированной VM. Поэтому `collect.sh` нужно запустить **дважды**, и каждый
запуск даст свой архив:

1. **На хосте Proxmox, от root.** Здесь видны настоящие DIMM и каналы, плата, BIOS, BMC, IOMMU/ACS, PCIe-дерево и
   конфигурация VM (`qm config`: balloon, hostpci, NUMA, hugepages). Если GPU проброшены в VM (`vfio-pci`), `nvidia-smi`
   на хосте не будет; линки GPU всё равно видны через sysfs и `lspci`.
2. **В VM, где будет работать инференс.** Здесь видны драйвер, CUDA, GPU, P2P, memlock и hugepages гостя. Внутри VM
   данные о DIMM, BIOS и PCIe-топологии **виртуальные**: `summarize.py` в этом случае ставит красный флаг.

## 5. Порядок шагов

Каталог `scripts/diag/` переносится на сервер целиком: git-бандлом репозитория (skill bundle-sync) или простой копией
папки. Все команды ниже выполняются из корня копии репозитория (или из каталога, где лежит `scripts/diag/`).

Длительности ниже **PROVISIONAL**: это оценки по размерам замеров. На сервере их никто не измерял. В песочнице (4 vCPU,
без GPU и NVMe) два прогона `collect.sh` от root заняли 8.8 с и 3.6 с (`real`): CONFIRMED, вывод `time` в отчёте
infra-ops от 2026-10-09. В песочнице не было большинства утилит; на сервере их будет больше, и сбор пойдёт дольше.

| # | что | команда | длительность (PROVISIONAL) | что может нарушить (риск) |
|---|---|---|---|---|
| 0 | просмотреть план замеров без запуска | `bash scripts/diag/bench_all.sh --plan --out diag-out/plan --mem --gpu --vram-check` | секунды | ничего: только печать, каталоги не создаются |
| 1 | сбор без root (быстрая проверка) | `bash scripts/diag/collect.sh` | 10–60 с | ничего: только чтение |
| 2 | сбор с root (основной) | `sudo bash scripts/diag/collect.sh` или `bash scripts/diag/collect.sh --sudo` | 1–5 мин; `ipmitool sdr/sel` бывают медленными (таймаут 240 с) | только чтение; на хосте опрос BMC по KCS может на минуты занять BMC |
| 3 | полоса памяти | `bash scripts/diag/bench_mem.sh --run --out diag-out/<каталог шага 2>` | 1–5 мин | 100% CPU и ~6 GiB RAM на время замера; **не запускать при работающем инференсе** |
| 4 | GPU: P2P/PCIe-матрица | `bash scripts/diag/bench_gpu.sh --run --out diag-out/<каталог>` | сборка 1–2 мин + замер ≤ 2 мин | нагрузка на PCIe и GPU; другие задачи на GPU замедлятся; P2P включается только внутри процесса замера |
| 5 | GPU: проверка VRAM | `bash scripts/diag/bench_gpu.sh --run --vram-check --out diag-out/<каталог>` | + 1–3 мин | занимает почти всю свободную VRAM; если на GPU есть вычислительные процессы, **отказывается** (обход: `--force`, только если это осознанно) |
| 6 | NVMe по одному диску | `bash scripts/diag/bench_nvme.sh --run --dir /mnt/<диск>/<каталог> --size-gib 64 --out diag-out/<каталог>` | ~4 мин на диск + запись 64 GiB | пишет один тестовый файл 64 GiB в **новый** подкаталог и в конце удаляет его; нужно ≥ 128 GiB свободного места (проверяется); износ SSD: 64 GiB записи |
| 7 | NVMe: все диски по очереди и вместе | `bash scripts/diag/bench_nvme.sh --run --all /mnt/a/d,/mnt/b/d,/mnt/c/d --size-gib 64 --out diag-out/<каталог>` | ~4 мин × (число дисков + 1) | как шаг 6, на каждом диске |
| 7a | NVMe напрямую (только если диск не размечен) | `sudo bash scripts/diag/bench_nvme.sh --run --raw-readonly /dev/nvme0n1 --out diag-out/<каталог>` | ~4 мин | только чтение (`fio --readonly`); никогда не записанные блоки читаются быстрее реальных данных, поэтому файловый режим точнее |
| 8 | всё вместе | `bash scripts/diag/bench_all.sh --run --out diag-out/<каталог> --mem --gpu --vram-check --nvme-all /mnt/a/d,... --size-gib 64` | сумма шагов 3–7 | сумма рисков 3–7; после замеров пересобирает `diag-out/<каталог>.tar.gz` |

Подробности:

- Без `--run` любой `bench_*.sh` печатает план (что, куда, сколько) и выходит с кодом 2, ничего не запуская. С `--plan`
  печатает план и выходит с кодом 0.
- `bench_gpu.sh` собирает программы на месте: `nvcc -O2 -arch=sm_89`. Нужен toolkit, который поддерживает драйвер сервера.
  Номера CUDA-устройств совпадают с `nvidia-smi`, потому что скрипт ставит `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Во время замера
  `nvidia-smi` раз в 500 мс записывает Gen и ширину линка (`link_samples.csv`). В простое GPU снижают Gen ради экономии
  энергии (PROVISIONAL, общеизвестное поведение), поэтому Gen смотрят под нагрузкой.
- `bench_mem.sh` гоняет варианты: 1 поток; все физические ядра со `spread` и с `close`; все потоки; по каждому NUMA-узлу
  через `numactl --cpunodebind=N --membind=N` (если узлов больше одного); `--interleave=all`.
- `bench_nvme.sh` гоняет задания: randread 4k QD1; 4k QD32 × 4 задания; 128k QD32; 1M QD8; 16M QD4; seqread 1M QD32.
  Все с O_DIRECT и `--readonly`, по 30 с плюс 5 с разгона (`--runtime`, `--ramp`). Температура дисков пишется до и после
  каждого задания (`temps.tsv`).
- Если `bench_nvme.sh` убить через `kill -9`, тестовый файл может остаться. Его путь напечатан в плане (`diag-fio-*`).
  Ctrl-C удаляет его штатно.

## 6. Что прислать

Пришлите **архив `diag-out/<host>-<UTC>.tar.gz`**: по одному с хоста Proxmox и из VM. Если запускался `bench_all.sh`,
архив уже пересобран вместе с результатами замеров. Внутри: `manifest.json`, `probes/*.txt`, `bench/...`,
`HARDWARE_FACTS.draft.md`, `collect.log` и копия самого комплекта (`kit/`) с контрольными суммами
(`probes/*_meta.kit_sha256.txt`).

В архиве есть: имя хоста, IP-адреса (`ip -br a`), серийные номера DIMM/NVMe/GPU/платы, список Python-пакетов, журнал
ядра и процессы на GPU. Если что-то нужно скрыть, удалите это и напишите нам, что именно удалено: факт из удалённого
файла станет UNKNOWN, а не «примерно известным».

## 7. Красные флаги

`summarize.py` ищет их сам (раздел «Red flags» в черновике), но каждый флаг нужно перепроверить по указанному файлу.

| флаг | где видно | почему важно |
|---|---|---|
| GPU на **x8** или ширина ниже максимальной | `gpu.query_csv`, `pcie.sysfs_links`, `link_samples.csv`; в dmesg строки «limited by» | вдвое меньше полосы H2D и P2P; промахи кэша экспертов дороже |
| GPU или NVMe на **Gen3** (8.0 GT/s) или ниже | там же; для NVMe `storage.nvme_sysfs` | у 4080 и 990 Pro Gen4; Gen3 = вдвое меньше полосы |
| P2P **NS / CNS / TNS / GNS** в `nvidia-smi topo -p2p r/w`; `cudaDeviceCanAccessPeer = 0` | `gpu.topo_p2p_*`, `bench/gpu/*/p2p_matrix.csv` | нет прямого P2P: TP/EP через хост, другая архитектура движка |
| `ACSCtl: ... ReqRedir+` на мостах | `pcie.lspci_vvv_all` (root) | P2P-трафик принудительно идёт через корневой комплекс |
| **BAR1 = 256 MiB** (Resizable BAR выключен) | `gpu.q_memory`, BAR в `pcie.sysfs_links` | ограничивает P2P через BAR1 и большие отображения |
| **DIMM меньше 8** или каналов меньше 8 | `memory.dmidecode_16_17` (root, на хосте) | у EPYC SP3 8 каналов; 4 канала ≈ половина полосы RAM для экспертов на CPU |
| настроенная частота памяти ниже 3200 MT/s | там же, `Configured Memory Speed` | полоса RAM падает пропорционально |
| **NVMe > 70 °C** (при сборе или под fio) | `storage.nvme_sysfs`, `smart-log`, `bench/nvme/*/temps.tsv` | троттлинг; SSD-ярус под нагрузкой просядет |
| NVMe на x2 или ниже x4; `critical_warning` ≠ 0; `media_errors` > 0 | `pcie.sysfs_links`, `storage.nvme_smart_log.*` | полоса или здоровье диска |
| **Xid** в журнале ядра, «fallen off the bus» | `system.dmesg_filtered` (может требовать root) | нестабильный GPU, райзер или питание |
| ненулевые счётчики **AER** у GPU, NVMe или мостов | `pcie.sysfs_links` (`aer_*`), dmesg | ошибки линка: райзер, кабель, слот |
| **несовпадения в проверке VRAM** или выделено заметно меньше 32 GB | `bench/gpu/*/vram_check.txt` | память карты неисправна или не та, что заявлена (модифицированные платы) |
| `memory.total` GPU заметно меньше 32 GB | `gpu.query_csv`, `p2p_matrix.csv` | не та карта или не та прошивка |
| `ulimit -l` не `unlimited` и меньше ~100 GiB | `memory.ulimit_l`, `limits.conf`, `systemd` | закреплённые буферы движка не выделятся |
| одновременный H2D всех карт заметно меньше суммы одиночных (< 0.75) | `p2p_matrix.csv`, `*_all_simultaneous` | общие линии (свитч, бифуркация) или узкое место корневого порта |
| флаги AVX-512 у «Zen 3» | `cpu.flags` | процессор не тот, что заявлен, или флаги подменены гипервизором |
| сбор внутри VM | `system.virt` | DIMM, BIOS и PCIe в этом архиве виртуальные; нужен архив с хоста |

Что флагом **не является**: `pcie.link.gen.current = 1` у простаивающей GPU, если max = 4 и под нагрузкой (по
`link_samples.csv`) достигается Gen4.

## 8. Как результаты становятся CONFIRMED

1. Источник истины — архив заказчика. Факт становится CONFIRMED, только если он **литерально** есть в выводе пробы или
   замера. Ссылка на него оформляется так: `<имя архива>:probes/NNN_<проба>.txt`. Номер NNN зависит от числа найденных
   NVMe, поэтому искать надо по имени пробы.
2. `HARDWARE_FACTS.draft.md` — только черновик. В нём CONFIRMED означает «разобрано из названного файла»; PROVISIONAL —
   выведено (например, число каналов из имён слотов DIMM, способ указан); UNKNOWN — вывода нет. Ведущий или
   evidence-auditor переносит строки в `docs/reassessment/HARDWARE_FACTS.md` и `docs/evidence/` только после сверки с
   исходным файлом.
3. Синтетические данные в `tests/fixtures/diag/` — **SYNTHETIC TEST FIXTURE, NOT EVIDENCE**. Это вымышленные значения
   для тестов парсера, ссылаться на них нельзя.

Где искать факты F1–F20 из `docs/reassessment/HARDWARE_FACTS.md` в реальном выводе комплекта. В колонке «источник» там
теперь указаны те же фактические имена:

| факт | фактический файл (`probes/NNN_<имя>.txt` или `bench/...`) |
|---|---|
| F1 CPU | `cpu.lscpu`, `cpu.lscpu_parse`, `cpu.cpuinfo_model` |
| F2 AVX2/FMA/AVX-512 | `cpu.flags` |
| F3 NUMA | `cpu.numactl_hardware`, `cpu.numa_sysfs`, `cpu.lstopo` |
| F4–F5 DIMM, каналы, частота | `memory.dmidecode_16_17` (root); запасной источник `memory.edac_sysfs` |
| F6 полоса RAM | `bench/mem/<UTC>/membw.csv`, `bench/mem/<UTC>/run-*.txt` |
| F7 плата, BIOS, BMC | `board.dmidecode_0_1_2_3`, `board.dmidecode_38_ipmi`, `board.dmi_sysfs`, `bmc.ipmitool_*` |
| F8 ОС, гипервизор, IOMMU | `system.*`, `system.proxmox_*`, `pcie.iommu_groups`, `pcie.iommu_cmdline`, `pcie.iommu_dmesg` |
| F9 GPU, VBIOS, VRAM, BAR1 | `gpu.nvidia_smi_q`, `gpu.query_csv*`, `gpu.q_memory`, `gpu.proc_driver_nvidia` |
| F10 целостность VRAM | `bench/gpu/<UTC>/vram_check.txt` |
| F11 полоса D2D на одной карте | `bench/gpu/<UTC>/p2p_matrix.csv`, диагональ `d2d_peer`/`d2d_nopeer`. Это cudaMemcpy: байты считаются один раз, а память читает и пишет (PROVISIONAL: трафик DRAM ≈ 2× от числа) |
| F12 PCIe GPU | `gpu.query_csv`, `pcie.sysfs_links`, `pcie.lspci_vvv_nvidia` (LnkCap/LnkSta, root), `bench/gpu/<UTC>/link_samples.csv` |
| F13 H2D/D2H | `bench/gpu/<UTC>/p2p_matrix.csv` (`h2d_pinned`, `d2h_pinned`, `*_pageable`, `*_all_simultaneous`) |
| F14 P2P | `gpu.topo_p2p_r/w/n/a/p`, `p2p_matrix.csv` (`can_access_peer`, `enable_peer`, `d2d_*`, `lat_d2d_peer`) |
| F15 NVMe | `storage.nvme_sysfs`, `storage.nvme_list`, `storage.nvme_id_ctrl.*`, `storage.nvme_smart_log.*`, `storage.smartctl.*` |
| F16 NVMe fio | `bench/nvme/<UTC>/*.json`, `summary.txt`, `summary.csv`, `temps.tsv` |
| F17 монтирование, ФС, RAID | `storage.lsblk*`, `storage.findmnt`, `storage.df`, `storage.mdstat`, `storage.lvm`, `storage.zpool_*` |
| F18 драйвер, CUDA, компиляторы | `gpu.nvidia_smi`, `gpu.proc_driver_nvidia`, `system.modinfo_nvidia`, `software.*` |
| F19 memlock, hugepages, THP | `memory.ulimit_l`, `memory.limits_conf_memlock`, `memory.systemd_default_memlock`, `memory.hugepages`, `memory.meminfo` |
| F20 интернет | `network.internet_check` (только с `--net-check`), `network.ip_br_*` |

## 9. Что проверено и что нет

| что | метка | основание |
|---|---|---|
| `collect.sh` отрабатывает с кодом 0 без root (тестовый хук), от root и с `--sudo` (заглушка sudo); `manifest.json` валиден; статусы `not-available`/`needs-root`/`skipped` согласованы; таймаут и падение пробы не останавливают сбор; архив создаётся | CONFIRMED | `python3 -m unittest tests.test_diag_kit -v`, вывод в отчёте infra-ops 2026-10-09 |
| запрещённых команд в `scripts/diag/` нет; root-утилиты без root не вызываются; `nvidia-smi` вызывается только с разрешёнными аргументами; sudo только `-n` | CONFIRMED | там же (статический поиск + заглушки); 4 мутации пойманы (`--readonly` убран, `nvidia-smi -pm 1`, `dmidecode` без root, выдуманное число каналов) |
| `bench_nvme.sh` пишет только новый тестовый файл, все чтения идут с `--readonly`, файл удаляется; сырой режим только читает | CONFIRMED **с заглушкой fio** | там же; настоящий fio в песочнице NOT RUN (не установлен) |
| `membw.c` собирается (`gcc -fopenmp -Wall -Wextra -Werror`) и проходит самопроверку на маленьком массиве | CONFIRMED | там же |
| `gpu_p2p_matrix.cu` и `gpu_vram_check.cu` собираются под `sm_89` (nvcc 13.4 из PyPI-колеса `nvidia-cuda-nvcc`) | CONFIRMED (сборка) | `test_compile_sm89 ... ok` — литерально в `docs/evidence/2026-10-09-tool-mutations.md`, раздел «scripts/diag: сборка CUDA-программ» (повтор 2026-10-10) |
| запуск CUDA-программ, их числа и поведение на 4 GPU | **NOT RUN** | в песочнице нет GPU |
| форматы вывода `dmidecode`, `nvidia-smi`, `nvme-cli`, `ipmitool`, `fio`, по которым написан парсер | PROVISIONAL | в песочнице этих утилит нет; фикстуры воспроизводят формат по памяти автора; первый реальный архив — проверка парсера |
| поведение `lspci`, `numactl`, `cpupower`, `edac-util`, `ras-mc-ctl`, `qm`, `zpool`, `mdadm` | NOT RUN (проверены только вызовы через заглушки) | в песочнице не установлены |

Известные ограничения:

- Бинарник, собранный CUDA 13, требует свежий драйвер. Поэтому программы собираются на сервере его собственным `nvcc`.
  Используемый API (runtime: peer access, memcpy, events) старый и стабильный, но сборка с CUDA 12.x в песочнице
  NOT RUN.
- H2D/D2H зависят от того, на каком узле NUMA лежит закреплённая память. Есть флаг `bench_gpu.sh --numa-node N`.
- Латентность 4 КиБ = асинхронная копия + синхронизация потока: накладные расходы драйвера входят в число.
- Число каналов памяти выводится из имён слотов (`Locator`/`Bank Locator`) и поэтому помечено PROVISIONAL. Если имена не
  распознаны, будет UNKNOWN с перечислением имён.
