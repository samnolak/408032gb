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

Full engine build for sm_89 on Linux: CI job `strata-glm-sm89` (result linked in docs/PLAN.md when done).
