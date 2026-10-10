# N1: сборка движков R1 под sm_89 без GPU (2026-10-10)

Задача — ROADMAP, «Трек без железа», N1 (снять риск R1.4 заранее). Скрипт `scripts/engines/build_engine.py`, пины
`tools/engines/pins.json`, CI `.github/workflows/engines-sm89.yml`. **Сборка ничего не говорит о скорости или корректности на GPU:
GPU нет ни в песочнице, ни на раннере — это R1 на сервере (NOT RUN).**

## Локально (облачная песочница, 4 ядра, без GPU): llama.cpp @ 79e2e74

Тулкит — pip-колёса CUDA 13.4 (`nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cublas`, `nvidia-cuda-cuobjdump`) в venv песочницы;
в нём нет заглушки драйвера `libcuda.so`, поэтому локально добавлен `-DGGML_CUDA_NO_VMM=ON` (убирает только линковку с драйвером
ради VMM-пула; ядра CUDA те же). В CI ставится `cuda-driver-dev-13-0` и собирается с пиновыми флагами без этой добавки.
Первая попытка дошла до линковки и упала: `undefined reference to cudaMemset@libcudart.so.13` и т. п. — компоновщик не видел
библиотек pip-раскладки; со `LD_LIBRARY_PATH`/`LIBRARY_PATH` на каталог колеса сборка завершилась (особенность песочницы, не движка).
```
$ PATH=<cu13>/bin:$PATH LD_LIBRARY_PATH=<cu13>/lib LIBRARY_PATH=<cu13>/lib python3 scripts/engines/build_engine.py --engine llama.cpp \
    --arch 89 --nvcc <cu13>/bin/nvcc --src-root <S>/eng-src --build-root <S>/eng-build --jobs 4 --extra-cmake=-DGGML_CUDA_NO_VMM=ON
fetch llama.cpp @ 79e2e74eb110           rc=0
configure                                rc=0
build                                    rc=0
{"engine": "llama.cpp", "sha": "79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6", "arch": 89, "status": "ok"}
cubins bin/libggml-cuda.so                      sm_89
cubins bin/libggml-cuda.so.0                    sm_89
cubins bin/libggml-cuda.so.0.26.0               sm_89
```
`build-info.json` (выдержка): `"checked_out": "79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6"`, `"nvcc_version": "Build cuda_13.4.r13.4/compiler.38855100_0"`,
`"bins": {"bin/llama-server": true, "bin/llama-bench": true}`. `cuobjdump --list-elf bin/libggml-cuda.so` — 144 cubin, все `sm_89`
(первые строки: `ELF file    1: libggml-cuda.1.sm_89.cubin`).

Флаги собранного сервера (CONFIRMED на этой сборке — снимает PROVISIONAL с шаблонов `BENCH_PROTOCOL.md` для этих флагов):
```
$ llama-server --version
version: 0.6.0-dev (build 1, commit 79e2e74)
built with GNU 13.3.0 for Linux x86_64
$ llama-server --help   (выдержка)
--moe-cache-mib N                       GPU cache size in MiB for the MoE experts kept in the CPU. with
                                        multiple GPUs, it is split among them like the layers (--tensor-split)
                                        (default: 0, disabled)
                                        (env: LLAMA_ARG_MOE_CACHE_MIB)
-ot,   --override-tensor <tensor name pattern>=<buffer type>,...
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the
-sm,   --split-mode {none,layer,row,tensor}
--cache-reuse N                         min chunk size to attempt reusing from the cache via KV shifting,
--slot-save-path PATH                   path to save slot kv cache (default: disabled)
```

## CI (GitHub Actions, engines-sm89): llama.cpp, ik_llama.cpp, Strata — все три собраны
Первый полный прогон: run 38046973798 (push, `1448706`), раннер ubuntu-24.04 без GPU, apt-тулкит CUDA 13.0 с заглушкой драйвера
(`cuda-driver-dev-13-0`), пиновые флаги без добавок (в том числе VMM-путь llama.cpp). Литеральные строки из логов джобов (GitHub API):
```
build (llama.cpp)      job 114198348004  11:03:46 -> 11:14:59
{"engine": "llama.cpp", "sha": "79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6", "arch": 89, "status": "ok"}
cubins bin/libggml-cuda.so                      sm_89
cubins bin/libggml-cuda.so.0                    sm_89
cubins bin/libggml-cuda.so.0.26.0               sm_89
##[notice]llama.cpp 79e2e74eb110 sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0

build (ik_llama.cpp)   job 114198348056  11:03:48 -> 11:29:50
{"engine": "ik_llama.cpp", "sha": "5194a9e1451b885e24dd5a065f650864e8807c59", "arch": 89, "status": "ok"}
cubins ggml/src/libggml.so                      sm_89
##[notice]ik_llama.cpp 5194a9e1451b sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0

build (strata)         job 114198347901  11:04:17 -> 11:09:27
{"engine": "strata", "sha": "fb58e0dbc8399662c0e47c76578c6e878b14f6cf", "arch": 89, "status": "ok"}
cubins strata                                   sm_89
##[notice]strata fb58e0dbc839 sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0
```
`build-info.json` и логи сборки сохранены как артефакты прогона (`build-llama.cpp-sm89`, `build-ik_llama.cpp-sm89`,
`build-strata-sm89`). Вывод: на CUDA 13.0 все три движка R1 на пинах компилируются под sm_89 (CONFIRMED сборкой); работоспособность
на RTX 4080 Super, скорость и корректность — NOT RUN до сервера. ExLlamaV3 (B5) сюда не входит: это расширение PyTorch, его сборка —
отдельный шаг (NOT RUN).

## ExLlamaV3 (B5): сборка расширения PyTorch под sm_89 (N6, 2026-10-10)
Пин `exllamav3` @ `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` (v1.6.0, `docs/reassessment/research/R2-engines.md:71,96`) с
`"kind": "torch-extension"` в `tools/engines/pins.json`. Расширение `exllamav3_ext` собирает `torch.utils.cpp_extension`
(`setup.py`@151539c), поэтому в пин входят torch 2.14.1 и его CUDA 13.0: по метаданным PyPI torch 2.14.1 требует
`cuda-toolkit==13.0.3` (`https://pypi.org/pypi/torch/2.14.1/json`, прочитано 2026-10-10 — CONFIRMED по этому URL); nvcc берётся из
того же pip-тулкита. Архитектура задаётся только `TORCH_CUDA_ARCH_LIST` (`exllamav3/util/arch_list.py`@151539c), скрипт ставит
`8.9`. `scripts/engines/build_engine.py`: проверка версии torch/CUDA → `pip wheel --no-deps --no-build-isolation` →
распаковка `exllamav3_ext*.so` из колеса в `ext/exllamav3_ext.so` → те же проверки cubin `sm_89` и `files_sha256` (колесо тоже
хешируется). У движка нет шаблона R1 (`"r1_template"` в пине: TabbyAPI — AGPL-3.0, квант EXL3 ≥ 4 bpw — UNKNOWN до выбора весов).

Первая локальная попытка (CONFIRMED, лог сборки в песочнице): без компонента `cccl` тулкита компиляция падает —
```
.../nvidia/cu13/include/cuda_fp16.h:4492:10: fatal error: nv/target: No such file or directory
```
поэтому в `python.install` пина стоит `cuda-toolkit[nvcc,cudart,cublas,cccl]==13.0.3`.

Тесты пути torch-extension (план и поддельный тулкит; настоящий nvcc не вызывается):
```
$ python3 -m unittest tests.test_engines_build
Ran 13 tests in 1.336s

OK (skipped=1)
```
Мутационная проверка (`docs/evidence/mutation-scripts/2026-10-10-torch-extension-build.py`, код на `62afac2`):
```
baseline                             rc=0 ok       OK (skipped=1)
T1 torch check step dropped          rc=1 caught   FAILED (failures=2, skipped=1)
T2 arch list fixed to 8.6            rc=1 caught   FAILED (failures=2, skipped=1)
T3 PYTORCH_NVCC not set              rc=1 caught   FAILED (failures=1, errors=1, skipped=1)
T4 MAX_JOBS ignored                  rc=1 caught   FAILED (failures=2, skipped=1)
T5 unpack takes any .so              rc=1 caught   FAILED (failures=1, skipped=1)
T6 wheel not hashed                  rc=1 caught   FAILED (failures=1, skipped=1)
T7 --extra-cmake accepted            rc=1 caught   FAILED (failures=1, skipped=1)
T8 torch builds take the cmake path  rc=1 caught   FAILED (errors=5, skipped=1)
T9 build isolation on                rc=1 caught   FAILED (failures=3, skipped=1)
caught 9/9
```
Настоящая компиляция под sm_89 (локально, песочница без GPU, 4 ядра, Python 3.13, torch 2.14.1+cu130, nvcc 13.0 из pip):
- попытка 2 (с `cccl`): все 188 объектов скомпилированы с `-gencode=arch=compute_89,code=sm_89` за 16:17:53 → 16:37:35 UTC
  (~20 мин), затем линковка упала (CONFIRMED, лог сборки):
  ```
  /usr/bin/ld: cannot find -lcudart: No such file or directory
  collect2: error: ld returned 1 exit status
  ```
  `cpp_extension` передаёт `-lcudart`, а pip-рантайм CUDA 13 содержит только `libcudart.so.13` (в `nvidia/cu13/lib`:
  `libcudart.so.13`, `libcudart_static.a`). Исправление (`16f39d6`): шаг `link shim for libcudart` создаёт
  `<build>-linkshim/libcudart.so` → `libcudart.so.13` (только если в тулките нет `libcudart.so`) и кладёт его в `LIBRARY_PATH`.
- попытка 3 (код `16f39d6`, объекты из кэша ninja, только линковка):
  ```
  check torch 2.14.1 with CUDA 13.0        rc=0
  link shim for libcudart                  rc=0
  build wheel                              rc=0
  unpack exllamav3_ext                     rc=0
  {"engine": "exllamav3", "sha": "151539c77abc7ab7425d30da7a4e8e3c5c154e7b", "arch": 89, "status": "ok"}
  cubins ext/exllamav3_ext.so                     sm_89
  rc=0
  ```
  Проверка результата (CONFIRMED командами `readelf -d`, `cuobjdump --list-elf | … | uniq -c` и загрузкой расширения в torch):
  ```
  dist/exllamav3-1.6.0-cp313-cp313-linux_x86_64.whl   57461120 байт
  ext/exllamav3_ext.so                                151110072 байт
  NEEDED: libc10.so libtorch_cpu.so libtorch_python.so libcudart.so.13 libc10_cuda.so libtorch_cuda.so libstdc++.so.6 …
      167 sm_89
  torch 2.14.1+cu130 cuda available False
  ext loaded: exllamav3_ext 189 symbols
  ```
  То есть расширение собрано только под sm_89 (167 cubin, других архитектур нет), зависит от `libcudart.so.13` того же torch
  и загружается этим torch без GPU. Колесо — под CPython 3.13 этой песочницы; на сервере колесо собирается тем Python, что там
  стоит (тег cp3XX — часть ABI). Работа ядер на RTX 4080 Super, скорость и корректность — **NOT RUN**.

Мутации после шима (`docs/evidence/mutation-scripts/2026-10-10-torch-extension-build-shim.py`, код на `16f39d6`):
```
baseline                             rc=0 ok       OK (skipped=1)
T1 torch check step dropped          rc=1 caught   FAILED (failures=2, skipped=1)
T2 arch list fixed to 8.6            rc=1 caught   FAILED (failures=2, skipped=1)
T3 PYTORCH_NVCC not set              rc=1 caught   FAILED (failures=1, errors=1, skipped=1)
T4 MAX_JOBS ignored                  rc=1 caught   FAILED (failures=2, skipped=1)
T5 unpack takes any .so              rc=1 caught   FAILED (failures=1, skipped=1)
T6 wheel not hashed                  rc=1 caught   FAILED (failures=1, skipped=1)
T7 --extra-cmake accepted            rc=1 caught   FAILED (failures=1, skipped=1)
T8 torch builds take the cmake path  rc=1 caught   FAILED (errors=5, skipped=1)
T9 build isolation on                rc=1 caught   FAILED (failures=3, skipped=1)
T10 link shim step dropped           rc=1 caught   FAILED (failures=1, errors=1, skipped=1)
T11 shim not on LIBRARY_PATH         rc=1 caught   FAILED (failures=2, skipped=1)
T12 shim links the wrong file        rc=1 caught   FAILED (failures=1, skipped=1)
caught 12/12
```
CI: `exllamav3` добавлен в матрицу `.github/workflows/engines-sm89.yml` (`16f39d6`; venv из `python.install` пина, nvcc из того
же pip-тулкита, колесо — артефакт прогона на 30 дней). Результат CI — в следующем разделе после прогона (пока NOT RUN).
