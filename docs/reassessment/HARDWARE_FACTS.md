# HARDWARE_FACTS — целевой сервер (заполняется только из литерального вывода диагностики)

Статус на 2026-10-09: **ничего не измерено**. Всё ниже — UNKNOWN или заявление заказчика (UNVERIFIED), пока не заполнено из
`diag-out/<host>-<время>/` (комплект `scripts/diag/`, инструкция `HW_DIAG_RUNBOOK.md`). Черновик таблицы строит
`python3 scripts/diag/summarize.py diag-out/<...>/` → `HARDWARE_FACTS.draft.md`; ведущий переносит сюда строки с меткой CONFIRMED
и ссылкой на файл вывода. Старый стенд 3×RTX 3090 (`epyc-ai`) сюда не относится (см. `PROBLEMS.md`).

| # | факт | заявлено заказчиком | значение | метка | источник (файл в diag-out) |
|---|---|---|---|---|---|
| F1 | CPU: модель, stepping, Zen 2/3, ядра/потоки, сокеты | EPYC «7H43», 48C/96T; R4: такой SKU публично не найден | — | UNKNOWN | `cpu/lscpu.txt` |
| F2 | AVX2/FMA да, AVX-512 нет | да / нет | — | UNKNOWN | `cpu/flags.txt` |
| F3 | NUMA-узлы, NPS | — | — | UNKNOWN | `cpu/numactl.txt` |
| F4 | DIMM: число, размер, слоты → число каналов | 128 GB DDR4-3200, каналы 4 или 8 | — | UNKNOWN | `memory/dmidecode-17.txt` (root) |
| F5 | настроенная частота памяти | 3200 | — | UNKNOWN | там же, «Configured Memory Speed» |
| F6 | пропускная способность RAM (triad: 1 поток / все ядра / по NUMA) | — | — | UNKNOWN | `bench/membw-*.txt` |
| F7 | плата, ревизия, BIOS версия/дата, BMC | HUANANZHI H12D-8D rev 2.0 | — | UNKNOWN | `board/dmidecode-0-3.txt`, `board/ipmitool-*.txt` |
| F8 | ОС / гипервизор / VM, ядро, параметры ядра, IOMMU | исторически Proxmox + изолированная VM | — | UNKNOWN | `system/*.txt`, `pcie/iommu-groups.txt` |
| F9 | GPU ×4: модель, PCI ID, VBIOS, VRAM total, BAR1 | 4× RTX 4080 Super 32 GB (модифицированные?) | — | UNKNOWN | `gpu/nvidia-smi-q.txt` |
| F10 | реальная целостная VRAM на каждой карте | 32 GB | — | UNKNOWN | `bench/gpu_vram_check-*.txt` |
| F11 | полоса HBM (D2D) каждой карты | ~736 ГБ/с (паспорт штатной 4080S) | — | UNKNOWN | `bench/gpu_p2p_matrix.csv` |
| F12 | PCIe каждой GPU: Gen/ширина (cap/sta под нагрузкой), корневой порт | Gen4 x16 | — | UNKNOWN | `pcie/lspci-vvv-nvidia.txt` |
| F13 | H2D/D2H pinned по одной и всем картам сразу | ~25 ГБ/с | — | UNKNOWN | `bench/gpu_p2p_matrix.csv` |
| F14 | P2P: CanAccessPeer, полоса/латентность D2D | «работает» (не измерено); R4: штатный драйвер GeForce выключает | — | UNKNOWN | `gpu/topo-p2p-*.txt`, `bench/gpu_p2p_matrix.csv` |
| F15 | NVMe ×4: модель, прошивка, ёмкость, линк, температура, SMART | 4× 1 TB (ранее «3× 990 Pro») | — | UNKNOWN | `storage/nvme-*.txt`, `storage/smartctl-*.txt` |
| F16 | NVMe: 4K QD1 p99, 4K QD32 IOPS, 16 МиБ ГБ/с; масштаб 1→4 | — | — | UNKNOWN | `bench/nvme-*.json` |
| F17 | точки монтирования, ФС, RAID/LVM | — | — | UNKNOWN | `storage/lsblk.txt`, `storage/findmnt.txt` |
| F18 | драйвер NVIDIA / CUDA / toolkit / компиляторы | — | — | UNKNOWN | `gpu/driver.txt`, `software/*.txt` |
| F19 | memlock, hugepages, THP | — | — | UNKNOWN | `memory/limits.txt` |
| F20 | доступ в интернет | исторически нет | — | UNKNOWN | `network/*.txt` (только с `--net-check`) |

## Карта PCIe/NUMA (заполнить)
```
(вывод lspci -tv и nvidia-smi topo -m сюда, литерально)
```

## Состояние «до/после» BIOS
См. `BIOS_PLAN.md` — журнал изменений; после каждого шага повторяются F4–F6, F9, F12–F14.
