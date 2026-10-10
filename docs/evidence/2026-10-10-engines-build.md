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
