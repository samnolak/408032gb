# strata-glm on sm_89: compile evidence (2026-10-02, sandbox, no GPU)

Source: sergqwer/strata-glm ed37419fccd0c52e07d26d526a29c2098f547843. Compiler: nvcc from pip (nvidia-cuda-nvcc).

```
$ nvcc --version | tail -1
Build cuda_13.4.r13.4/compiler.38855100_0
$ for f in src/glm/*.cu; do nvcc -std=c++17 -O2 -arch=sm_89 --expt-relaxed-constexpr -Iinclude -Isrc -I$CUDA/include -c $f; done
OK glm_dense.cu
OK glm_kernels.cu
OK glm_kernels_rows.cu
OK glm_mla.cu
OK glm_moe.cu
OK glm_select.cu
OK glm_gemm
```

Arch-specific constructs found (grep): WMMA fp16 (glm_mla.cu, glm_select.cu), mma.sync f16/tf32 and ldmatrix (Strata kernels, sm_80+),
`__nv_cvt_fp4x2_to_halfraw2` / `__nv_cvt_fp8x2_to_halfraw2` software-path conversions (glm_dense.cu). No FP4 tensor-core MMA.

## Full engine build for sm_89 on Linux (CI, GitHub-hosted runner, no GPU)

| run | commit | patches | result |
|---|---|---|---|
| 36928522223 .. 36929757058 | b09ccbf .. facfaf7 | none | build step falsely green (exit code lost in a pipe), binary missing; step rewritten |
| 36955809531 | 0203557 | none | FAIL: `_fseeki64` not declared (glm_main.cpp:345, 1678, 2404) - MSVC-only |
| 36956174871 | 22c5609 | 0001 | PASS, binary contains sm_89 code |
| 36956444172 | 2f9c7a6 | 0001-0002 | PASS |
| 36956571033 | f52a50c | 0001-0003 | PASS; annotations: `binary build/strata-glm, 33198128 bytes`, `strata-glm build OK, 69 CUDA objects`, `cubin archs: 26 sm_89`; DiskReader functional test passed |

CONFIRMED: sergqwer/strata-glm @ed37419 + patches/strata-glm/0001-0003 builds and links for sm_89 on Linux (CUDA 13.0, Ubuntu 24.04).
NOT RUN: any execution on a GPU.
