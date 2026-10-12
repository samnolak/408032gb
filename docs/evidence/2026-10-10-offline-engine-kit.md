# N6: офлайн-набор движков R1 для сервера без интернета (2026-10-10)

Автор: lead-architect (сессия ведущего), роль кода — infra-ops. Ярлыки — по `AGENTS.md`. Всё ниже — песочница без GPU;
на сервере — **NOT RUN**.

## Зачем
`scripts/engines/build_engine.py` (N1) берёт исходники движков на пинах с GitHub (`git fetch --depth 1 <repo> <sha>`), а у
сервера интернета нет (заявление заказчика; проверяется R0). Без офлайн-пути сборка R1.4 на сервере невозможна. ExLlamaV3
вдобавок требует колёса Python (torch 2.14.1 и CUDA 13.0.3 из pip, ~3 ГБ).

## Что сделано
- `scripts/offline/engines.py` (пара к `weights.py`):
  - `pack` (ПК с интернетом): для каждого движка `tools/engines/pins.json` и каждой его зависимости — неглубокая выборка ровно
    пинованного коммита и git-бандл из одного коммита `<kit>/sources/<key>.bundle` со ссылкой `refs/pins/<key>`
    (`key` = движок или `<движок>-dep-<зависимость>`); для `"kind": "torch-extension"` — `pip download` списка
    `python.install` пина под Python и платформу сервера в `<kit>/wheels/<движок>/`; `manifest.json` с коммитом каждого
    бандла и размером и sha256 каждого файла;
  - `verify` (сервер, без сети): sha256 и размер каждого файла; каждый бандл выбирается в пустой репозиторий и обязан дать
    ровно пинованный коммит (`git fsck` чистый) — хеш коммита доказывает содержимое; коммиты набора = `pins.json`;
  - `install-python` (сервер, без сети): venv и `pip install --no-index --find-links <kit>/wheels/<движок>` списка пина.
- `build_engine.py --offline-kit <kit>`: выборка из бандлов вместо GitHub; отказ (код 2), если набор собран под другие
  коммиты; новая проверка: коммит каждой зависимости после выборки = пин (раньше проверялся только сам движок).
- Особенность git (CONFIRMED опытом в песочнице, git 2.43.0): бандл неглубокой выборки не принимается пустым репозиторием
  («did not send all necessary objects»), `git fetch --depth 1` из бандла — тоже. Принимается, если до выборки записать
  пинованный коммит в `.git/shallow` (граница неглубокой истории); так и делают `build_engine.py` и `verify`.
- Платформы колёс: `pip download --platform` берёт теги буквально, а колёса помечены минимальным glibc (torch 2.14.1 —
  `manylinux_2_28`, ninja 1.13.2 — `manylinux2014`/`manylinux_2_17`, nvidia-nccl-cu13 2.30.7 — `manylinux_2_18`; pypi.org,
  прочитано 2026-10-10). По умолчанию передаются все теги manylinux до 2.28 — то есть предполагается glibc сервера ≥ 2.28
  (PROVISIONAL до R0: `ldd --version` на сервере). Две первые попытки упаковки упали ровно на этом, после чего список стал
  полным диапазоном (литеральный вывод — раздел «Неудачные попытки» ниже).
- Python колёс: `--python-version 3.12` выбран потому, что 3.12 — системный Python Ubuntu 24.04 (так собирает CI
  `engines-sm89`, раннер ubuntu-24.04); ОС и Python сервера — UNKNOWN до R0. Если там другой Python, набор перепаковывается
  с его версией (тег cpXY — часть ABI колёс torch и собранного расширения).
- Сборка движка «прямо из оркестратора R1» (кандидат N6) — **отклонена по замыслу**: прогон R1 — это замер; сборка — отдельный
  шаг (`build_engine.py`, теперь и офлайн), а оркестратор проверяет её (`engine_build_check`, `c1a0336`).

## Тесты и мутации (CONFIRMED, песочница)
`tests/test_offline_engines.py` (9 тестов, локальные поддельные upstream-репозитории; в пинах для `build_engine` адреса
недостижимы, поэтому успешная сборка доказывает, что использован только набор):
```
$ python3 -m unittest tests.test_offline_engines tests.test_engines_build      (код bd2dcb0)
Ran 22 tests in 7.887s

OK (skipped=1)
```
Мутации (`docs/evidence/mutation-scripts/2026-10-10-offline-kit.py`). Первый прогон поймал 7/8: K1 выжил — тест подменённого
бандла падал уже на выборке (у коммита были родители), и сравнение коммита не проверялось; тест переписан на коммит без
родителей, с проверкой текста. Вывод первого прогона (рабочее дерево до правки теста; файл с выводом не сохранён — строки
перенесены из журнала сессии ведущего):
```
baseline                                   rc=0 ok       OK (skipped=1)
K1 verify: commit check always ok          rc=0 SURVIVED OK (skipped=1)
K2 verify: sha256 ignored                  rc=1 caught   FAILED (failures=1, skipped=1)
K3 verify: pins.json not compared          rc=1 caught   FAILED (failures=1, skipped=1)
K4 build: kit for other commits accepted   rc=1 caught   FAILED (failures=1, skipped=1)
K5 build: dependency checkout unchecked    rc=1 caught   FAILED (failures=1, skipped=1)
K6 build: no shallow boundary              rc=1 caught   FAILED (failures=2, errors=1, skipped=1)
K7 pack: bundle under a branch ref         rc=1 caught   FAILED (failures=4, skipped=1)
K8 build: kit ignored (network fetch)      rc=1 caught   FAILED (failures=3, skipped=1)
caught 7/8
```
Повтор на `git archive bd2dcb0` с `git init` и `git add -A` в каталоге выгрузки (скрипт берёт файлы через
`git ls-files -co`, поэтому каталог должен быть рабочим деревом git; код этого раздела):
```
baseline                                   rc=0 ok       OK (skipped=1)
K1 verify: commit check always ok          rc=1 caught   FAILED (failures=1, skipped=1)
K2 verify: sha256 ignored                  rc=1 caught   FAILED (failures=1, skipped=1)
K3 verify: pins.json not compared          rc=1 caught   FAILED (failures=1, skipped=1)
K4 build: kit for other commits accepted   rc=1 caught   FAILED (failures=1, skipped=1)
K5 build: dependency checkout unchecked    rc=1 caught   FAILED (failures=1, skipped=1)
K6 build: no shallow boundary              rc=1 caught   FAILED (failures=2, errors=1, skipped=1)
K7 pack: bundle under a branch ref         rc=1 caught   FAILED (failures=5, skipped=1)
K8 build: kit ignored (network fetch)      rc=1 caught   FAILED (failures=3, skipped=1)
caught 8/8
```

## Настоящий набор (CONFIRMED, песочница; `<S>` — каталог песочницы)
`python3 scripts/offline/engines.py pack --out <S>/kit --python-version 3.12` (18:00:36 → 18:01:38 UTC):
```
source  exllamav3                          151539c77abc  16dc46aaa0060868
wheels  exllamav3                          36 files, 3113047930 bytes
source  ik_llama.cpp                       5194a9e1451b  4d9b733ae09891fa
source  llama.cpp                          79e2e74eb110  085e1df70dc026af
source  strata-dep-ggml                    3cf03257f219  b26eebcd49815ac5
source  strata                             fb58e0dbc839  81a32e39025a82de
kit: <S>/kit (4 engines)
rc=0
```
Бандлы: exllamav3 8914713, ik_llama.cpp 35404233, llama.cpp 38378921, strata-dep-ggml 37776166, strata 17185662 байт (сумма
137659695 байт; `du -sh` печатает `132M` — это его округлённые МиБ);
колёса 2.9 ГБ, крупнейшие — torch 2.14.1 cp312 554618695, nvidia_cudnn_cu13 553099438, nvidia_cublas 423138758 байт.
`manifest.json`: схема `408032gb-engine-kit/1`, `pins_sha256` 79b71f7a1b8131cb… = sha256 текущего `tools/engines/pins.json`.

`GIT_ALLOW_PROTOCOL=file python3 scripts/offline/engines.py verify <S>/kit` (12 с; 36 строк колёс `ok` опущены):
```
ok   pins exllamav3                               = tools/engines/pins.json
ok   source exllamav3 sha256                      16dc46aaa0060868
ok   source exllamav3 commit                      commit 151539c77abc, fsck ok
ok   pins ik_llama.cpp                            = tools/engines/pins.json
ok   source ik_llama.cpp sha256                   4d9b733ae09891fa
ok   source ik_llama.cpp commit                   commit 5194a9e1451b, fsck ok
ok   pins llama.cpp                               = tools/engines/pins.json
ok   source llama.cpp sha256                      085e1df70dc026af
ok   source llama.cpp commit                      commit 79e2e74eb110, fsck ok
ok   pins strata                                  = tools/engines/pins.json
ok   source strata sha256                         81a32e39025a82de
ok   source strata commit                         commit fb58e0dbc839, fsck ok
ok   source strata-dep-ggml sha256                b26eebcd49815ac5
ok   source strata-dep-ggml commit                commit 3cf03257f219, fsck ok
verify: ok
rc=0
```
Выборка шагами `build_engine.py --offline-kit` без сети: короткий скрипт ведущего (текст не сохранён) брал шаги `fetch…` из
`build_engine.plan(..., kit=<S>/kit)` и выполнял их; запуск — с `GIT_ALLOW_PROTOCOL=file HTTPS_PROXY=http://127.0.0.1:9
https_proxy=http://127.0.0.1:9` (команда — из журнала сессии ведущего, в артефактах не записана). Его вывод, как напечатан:
```
llama.cpp          HEAD 79e2e74eb110 tree e09f8293280c  = pin
ik_llama.cpp       HEAD 5194a9e1451b tree bc9494c393af  = pin
strata             HEAD fb58e0dbc839 tree b5a210b63013  = pin
strata-dep-ggml    HEAD 3cf03257f219 tree d255198f04f9  = pin
online N1 checkout llama.cpp: 79e2e74eb110 tree e09f8293280c
```
Дерево llama.cpp из набора совпадает с деревом онлайн-выборки, из которой в N1 собрана локальная сборка.

`engines.py install-python <S>/kit --engine exllamav3 --venv <S>/venv-kit312 --python /usr/bin/python3.12` (без индекса,
58 с): `rc=0`, затем `torch 2.14.1+cu130 cuda 13.0`, nvcc 13.0 в `nvidia/cu13/bin` venv.

Полная сборка ExLlamaV3 только из набора (код `bd2dcb0`). Команда — из журнала сессии ведущего; переменные окружения в
артефактах сборки (`out`, лог, `build-info.json`) не записаны, поэтому «без сети» подтверждается этой командой и тем, что
лог сборки показывает выборку из `<S>/kit/sources/exllamav3.bundle refs/pins/exllamav3` и `pip wheel --no-deps
--no-build-isolation` (индекс не нужен):
```
export PATH=<V>/bin:<S>/cuobjdump-bin:$PATH GIT_ALLOW_PROTOCOL=file PIP_NO_INDEX=1 HTTPS_PROXY=http://127.0.0.1:9 \
       HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9
python3 scripts/engines/build_engine.py --engine exllamav3 --arch 89 --offline-kit <S>/kit \
    --nvcc <V>/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --python <V>/bin/python \
    --src-root <S>/kitbuild/src --build-root <S>/kitbuild/build --jobs 4
```
(`<V>` = `<S>/venv-kit312`; `cuobjdump` — CUDA 13.4 из другого pip-venv песочницы.) Файлы `start` и `out`, как записаны:
```
2026-10-10T18:04:24Z
fetch exllamav3 @ 151539c77abc           rc=0
fetch exllamav3 @ 151539c77abc           rc=0
fetch exllamav3 @ 151539c77abc           rc=0
fetch exllamav3 @ 151539c77abc           rc=0
check torch 2.14.1 with CUDA 13.0        rc=0
link shim for libcudart                  rc=0
build wheel                              rc=0
unpack exllamav3_ext                     rc=0
{"engine": "exllamav3", "sha": "151539c77abc7ab7425d30da7a4e8e3c5c154e7b", "arch": 89, "status": "ok"}
cubins ext/exllamav3_ext.so                     sm_89
rc=0
2026-10-10T18:20:42Z
```
`build-info.json`: `status ok`, `checked_out 151539c…`, `offline_kit <S>/kit`, `nvcc_version Build cuda_13.0.r13.0/compiler.36424714_0`,
файлы `dist/exllamav3-1.6.0-cp312-cp312-linux_x86_64.whl` и `ext/exllamav3_ext.so`; `cuobjdump --list-elf`: `167 sm_89`, других
архитектур нет; расширение загружается в torch из набора: `ext loaded with torch 2.14.1+cu130 - 189 symbols`. То есть весь путь
«набор → venv → сборка → проверка» прошёл без сети. На GPU — NOT RUN.

## Неудачные попытки упаковки (литеральный вывод, файлы задач сессии ведущего)
Только `--platform manylinux_2_28_x86_64` (17:56:03):
```
source  exllamav3                          151539c77abc  16dc46aaa0060868
ERROR: Could not find a version that satisfies the requirement ninja (from versions: none)
ERROR: No matching distribution found for ninja
pip download failed for exllamav3
rc=1
```
Десять тегов manylinux, без `manylinux_2_18` (17:58:39):
```
ERROR: Could not find a version that satisfies the requirement nvidia-nccl-cu13==2.30.7; platform_system == "Linux" (from torch) (from versions: 0.0.0a0, 2.27.7, 2.32.3)
ERROR: No matching distribution found for nvidia-nccl-cu13==2.30.7; platform_system == "Linux"
pip download failed for exllamav3
rc=1
```

## Полный набор тестов и аудит
```
$ python3 -m unittest discover -s tests      (дерево b6f4fe3, песочница)
Ran 435 tests in 100.602s

OK (skipped=86)
```
Аудит доказательств (evidence-auditor, только чтение, на выгрузках `git archive`): всё воспроизведено — 22 теста, мутации 8/8,
`verify` (вывод совпал с записанным), sha256 пинов в манифесте, размеры, опыт с `.git/shallow` на git 2.43.0, 167 cubin
sm_89, загрузка расширения, теги колёс (sha256 трёх колёс совпали с PyPI), выборка без сети = пины, 435/86. Шесть мелких
замечаний закрыты в этой правке: условие `git init` для скрипта мутаций; точная сумма байт бандлов вместо «132 МБ»;
литеральность блоков вывода (четыре строки `fetch` вместо свёрнутой, команды помечены как взятые из журнала сессии,
переменные окружения — как не записанные в артефактах); литеральный вывод неудачных попыток упаковки и первого прогона
мутаций 7/8; обоснование cp312; вывод полного набора тестов. Сверх того: комментарий в `build_engine.py` уточнён (бандл несёт
пинованный коммит без его родителей), а `verify` теперь называет движки `pins.json`, которых нет в наборе (набор может быть
подмножеством — это не ошибка; тест `test_verify_names_pinned_engines_the_kit_leaves_out`).

## Не проверено
- На сервере — NOT RUN (Python и glibc сервера UNKNOWN до R0; набор собран под cp312 и glibc ≥ 2.28).
- Сборка llama.cpp, ik_llama.cpp и Strata из набора — только выборка исходников (выше); их компиляция из тех же коммитов
  проверена в CI N1 онлайн-выборкой.
- Упаковка на Windows-ПК заказчика (git для Windows, `pip download`) — NOT RUN.
