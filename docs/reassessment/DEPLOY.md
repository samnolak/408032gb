# DEPLOY — runbook развёртывания на сервере 4×4080 32 GB (черновик, 2026-10-09)

**Статус: ЧЕРНОВИК. Ни один шаг не выполнялся на сервере — всё NOT RUN.** Документ закрывает часть BLOCKER 2 ревью
(`REVIEW-adversarial.md`): runbook можно написать без сервера, а проверить — только на нём (ROADMAP R5.4). Модель и движок не
выбраны (ADR-013 после R1–R2); ниже — общий скелет плюс шаблоны для двух движков, которые первыми идут в замеры: Strata upstream
(Qwen3.8-Flash-Next, `BRINGUP-qwen38fn.md`) и llama.cpp `llama-server`. Флаги движков взяты из их документации на пинах
(`BENCH_PROTOCOL.md`, раздел шаблонов) и **сверяются с `--help` собранной версии** перед первым запуском; расхождение записывается.

Что требует отдельного согласия владельца (правило мастер-промпта; агент сам не делает): разметка и форматирование NVMe, любые
изменения BIOS (`BIOS_PLAN.md`), установка/смена драйвера NVIDIA и P2P-патч (R4.3), изменение лимитов мощности GPU, остановка
работающих сервисов на хосте, открытие портов наружу.

## 0. Предусловия (гейты)
| # | условие | откуда | статус |
|---|---|---|---|
| P0 | R0 закрыт: `HARDWARE_FACTS.md` F1–F20 заполнены, известны каналы RAM, PCIe-линии, P2P, NVMe | `HW_DIAG_RUNBOOK.md` | NOT RUN |
| P1 | веса на NVMe сервера, sha256 совпадают с манифестом, `tools/gguf_quant_audit.py` — PASS | `scripts/offline/weights.py verify`, R1.2/R1.3 | NOT RUN |
| P2 | движок собран на записанном SHA под sm_89, `--version` в evidence | R1.4 | NOT RUN |
| P3 | выбрана конфигурация (модель + квант + движок + режим разбиения) по R1/R2 | ADR-013 | NOT RUN |

До P3 сервер поднимается только как стенд замеров (R1), не как продукт.

## 1. Раскладка на сервере (предложение; диски не форматируются без согласия)
```
/srv/llm/
  models/<model>/<revision>/...      веса (только чтение после verify), манифест рядом
  engines/<engine>-<sha>/            сборки движков; активная — симлинк engines/current-<engine>
  packs/<model>-<engine>/            производные паки (Strata iq_pack), со ссылкой на sha256 исходника
  state/slots/                       сохранённые состояния (llama-server --slot-save-path), отдельный NVMe
  logs/gateway/  logs/engine/        журналы (ротация logrotate, ≤ 10% диска)
  etc/                               api-keys (0600), конфиги движка и шлюза, versions.txt
```
Распределение 4 NVMe — после R0 (H7, P3): по умолчанию модели на одном томе, состояния слотов на другом, логи — на системном.
RAID и файловая система — решение владельца (UNKNOWN до R0).

Пользователь `llm` без shell-логина; группы `video`/`render` для доступа к GPU. `memlock` для закреплённой памяти:
```
# /etc/security/limits.d/llm.conf  (применяется после перелогина/перезапуска сервиса)
llm  soft  memlock  unlimited
llm  hard  memlock  unlimited
```
Для systemd-сервиса — `LimitMEMLOCK=infinity` в юните (ниже). NVIDIA persistence mode: `nvidia-smi -pm 1` (требует root;
не меняет частоты/напряжения) — записать в evidence до/после.

## 2. Движок (systemd, слушает только 127.0.0.1)

### 2a. llama.cpp `llama-server` (шаблон; значения в `<...>` — из R1)
```
# /etc/systemd/system/llm-engine.service
[Unit]
Description=LLM engine (llama-server)
After=network-online.target
[Service]
User=llm
Group=llm
LimitMEMLOCK=infinity
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
ExecStart=/srv/llm/engines/current-llamacpp/bin/llama-server \
  -m /srv/llm/models/<model>/<revision>/<file-00001-of-N>.gguf \
  --host 127.0.0.1 --port 8080 \
  -c <CTX> -np 1 -ngl all -sm <layer|row> -ts <T0,T1,T2,T3> -fa on \
  <-ot/-ncmoe/--moe-cache-mib из R1> \
  --jinja --metrics --cache-reuse 256 \
  --slot-save-path /srv/llm/state/slots
Restart=on-failure
RestartSec=10
TimeoutStopSec=60
StandardOutput=append:/srv/llm/logs/engine/engine.log
StandardError=append:/srv/llm/logs/engine/engine.log
[Install]
WantedBy=multi-user.target
```
`<CTX>`: для требования «≥ 262 144 используемых токенов + вывод» окно должно быть больше 262 144 на бюджет вывода, если модель это
позволяет (YaRN/rope-scaling — только с замером проб K1 на этой длине, см. `BRINGUP-qwen38fn.md` «Окно контекста»).

### 2b. Strata upstream `serve` (Qwen3.8-Flash-Next)
Команда и JSON-конфиг — по `BRINGUP-qwen38fn.md` шаг 4 (`"gpu": [0,1,2,3]`, `"layer_split": "auto"`, MTP `--spec 4`), привязка к
`127.0.0.1:8080`, тот же скелет юнита (`ExecStart=` на скрипт запуска Strata). Сохранение состояний на NVMe у Strata — UNKNOWN
(в прочитанной документации не найдено); контрольные точки в RAM есть (`BRINGUP-qwen38fn.md`).

## 3. Шлюз API (`tools/gateway/`, единственная точка входа для клиентов)
```
# /etc/systemd/system/llm-gateway.service
[Unit]
Description=OpenAI-compatible gateway
After=llm-engine.service
Requires=llm-engine.service
[Service]
User=llm
Group=llm
ExecStart=/usr/bin/python3 /srv/llm/repo/tools/gateway/gateway.py \
  --listen 0.0.0.0:8000 --backend http://127.0.0.1:8080 \
  --api-keys-file /srv/llm/etc/api-keys \
  --max-concurrency 1 --max-queue 8 \
  --max-context-tokens <CTX> \
  --set-default cache_prompt=true \
  --log-file /srv/llm/logs/gateway/requests.jsonl \
  --drain-s 120
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
```
Точные имена флагов — `python3 tools/gateway/gateway.py --help` и `tools/gateway/README.md` (README — источник истины, если здесь
расхождение). Шлюз не меняет тело запроса, кроме дефолтов `--set-default`, которые клиент не задал, — иначе ломается кэш префиксов.
`--max-concurrency 1`: продукт single-stream; остальные ждут в очереди FIFO, при переполнении — 429 с `Retry-After`.
Порт 8000 открывается только в изолированной сети заказчика (решение владельца).

## 4. Запуск, проверка, остановка
```
sudo systemctl daemon-reload
sudo systemctl start llm-engine          # загрузка весов: минуты; смотреть engine.log до строки готовности
curl -s http://127.0.0.1:8080/health     # движок
sudo systemctl start llm-gateway
curl -s http://127.0.0.1:8000/health     # шлюз: состояние движка и очереди
curl -s http://127.0.0.1:8000/metrics | head
```
Дымовой тест через шлюз (короткие пробы; полная матрица — R1):
```
OPENAI_API_KEY="$(head -1 /srv/llm/etc/api-keys)" python3 tools/bench/probes.py --url http://127.0.0.1:8000 \
  --contexts 8192 --out-dir /srv/llm/logs/smoke-$(date +%F)    # каталог вывода — по --help probes.py
```
Остановка: `sudo systemctl stop llm-gateway` (дожидается текущих запросов до `--drain-s`), затем `sudo systemctl stop llm-engine`.
Статус и журналы: `systemctl status llm-engine llm-gateway`, `tail -f /srv/llm/logs/engine/engine.log`,
`tail -f /srv/llm/logs/gateway/requests.jsonl`.

## 5. Обновление и откат (движок, модель, шлюз)
1. Новая сборка движка — в новый каталог `engines/<engine>-<newsha>/`; старая не удаляется.
2. На остановленном продукте (или на свободных картах — их нет: все 4 заняты одной моделью) прогон `tools/bench/run_matrix.py` и
   `probes.py` на новой сборке; сравнение с прошлым evidence (`tools/bench/vs_roofline.py`) и логпробами
   (`tools/quality/logprob_compare.py`, top-1 ≥ 0.99 к прошлой сборке при той же модели — иначе разбор, а не выкатка).
3. Переключение: `ln -sfn engines/<engine>-<newsha> engines/current-<engine>`; `systemctl restart llm-engine`; запись в
   `etc/versions.txt` (дата, SHA, модель, ревизия весов, кто).
4. Откат: симлинк на прежний каталог + рестарт; состояния слотов от новой версии не переносятся назад (формат может отличаться) —
   `state/slots/` очищается только владельцем.
Смена модели — то же, плюс P1 (verify + quant-audit) для новых весов и полный R1-прогон до переключения.

## 6. Эксплуатационные отказы (R5.3; матрица — тестировать на сервере, всё NOT RUN)
| # | отказ | ожидаемое поведение | как проверить | статус |
|---|---|---|---|---|
| F1 | клиент отменил запрос посреди стрима | шлюз закрывает upstream-соединение, движок прекращает генерацию, слот освобождается | обрыв `curl` на 100-м токене; в метриках движка генерация остановилась, следующий запрос не ждёт | на mock — тест шлюза; на движке NOT RUN |
| F2 | рестарт движка во время стрима | клиент получает SSE-событие ошибки без поддельного `[DONE]`; шлюз отвечает 502/503, пока движок не поднимется; systemd перезапускает | `systemctl restart llm-engine` во время запроса | на mock — тест шлюза; на движке NOT RUN |
| F3 | OOM VRAM (контекст/батч выше рассчитанного) | движок отклоняет запрос или падает и перезапускается; шлюз заранее отклоняет промпт > `--max-context-tokens` | промпт на 1.2× окна | NOT RUN |
| F4 | диск состояний полон | сохранение слота — ошибка в ответе, сервис работает дальше | заполнить тестовый том (не продуктовый!) | NOT RUN |
| F5 | NVMe с весами недоступен/ошибки чтения | при mmap — падение при обращении; движок перезапускается и не стартует; health шлюза показывает недоступность | только с согласия владельца (физическое действие) | NOT RUN |
| F6 | перегрев NVMe / троттлинг | рост латентности, видно в SMART и метриках; алерт по температуре | `bench_nvme.sh` длительный прогон, `nvme smart-log` | NOT RUN |
| F7 | смена модели | по разделу 5; клиенты получают 503 на время переключения | — | NOT RUN |
| F8 | перезагрузка хоста | оба сервиса стартуют сами (`WantedBy`), первые запросы ждут загрузки | `reboot` в окно обслуживания | NOT RUN |

Каждый пункт после проверки получает литеральный лог в `docs/evidence/R5-faults-<date>.md` и попадает в verification pack
(`VERIFICATION_PACK.md`).

## 7. Безопасность и данные
- Движок слушает только `127.0.0.1`; наружу — только шлюз с ключами. Ключи — файл `0600` владельца `llm`; в журналах не пишутся.
- Тексты промптов в журнал шлюза по умолчанию не пишутся (`--log-prompts` — только по решению заказчика).
- Интернета на сервере нет (заявление; проверяется R0): обновления — бандлами и через ПК заказчика (`scripts/offline/weights.py`).
