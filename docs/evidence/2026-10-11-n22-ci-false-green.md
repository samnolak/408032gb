# N22: CI false-green signals, gap 11 (b), (c), (d), and the per-layer error report, gap 14

Date 2026-10-11. Role: infra-ops. Branch `claude/n22-ci-false-green`, cut from `claude/n19-qwen4exp-multiseq` at
`e71b241`. Hardware-free: a CPU sandbox (4 cores), no GPU; CUDA 13.0 `nvcc` from NVIDIA's pip wheels in a scratch venv
(section 3). `<S>` stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution (and long tables shortened where marked).

## Result

| gap | before (CONFIRMED, cited at `e71b241`) | now | status |
|---|---|---|---|
| 11 (b) | `cuda-compile` globbed `kernels/*.cu`, printed "no kernels yet" and ran `exit 0` (`.github/workflows/ci.yml:93-94@e71b241`); there is no `kernels/` directory, so the job was green by construction | the job installs nvcc 13.0 + cuobjdump and runs `tests.test_diag_kit.CudaCompile` (every `scripts/diag/*.cu` built with `-arch=sm_89`, `cuobjdump --list-elf` must show an `sm_89` cubin) and `CudaSourcesInCi`; a skip, an empty run or a failure fails the step (`ci.yml:91-116@8242eec`) | closed in the repo; the job on GitHub Actions NOT RUN |
| 11 (c) | `CudaCompile` skipped silently in `cpu-parity` (no nvcc), so `scripts/diag/*.cu` was compiled by no job (`tests/test_diag_kit.py:557@e71b241`) | the build moved to `cuda-compile` (above); in `cpu-parity` the skip stays (no nvcc there), its reason names the job that builds it, and the `unit tests` step turns it into a `::warning::` annotation plus a `::notice::` with the unittest summary (`ci.yml:25-35@8242eec`, `tests/test_diag_kit.py:561@8242eec`) | closed in the repo; annotations on GitHub NOT RUN |
| 11 (d) | docstring of `tests/test_srvcheck.py:5` names a non-existent `tests/test_srvcheck_llama.py` | **not fixable on this branch**: the file does not exist here (section 6). Patch for the three branches that carry it: `<S>/n22/srvcheck-docstring.patch` | CLOSED by the lead after this report: `4f7be84` on `claude/n7-server-hybrid-cpu`, merged into N8 (`09eb438`) and N11 (`b74efde`) |
| 14 | `tests/test_glm_tiny_parity.py` (G1 + N9 whole model) and `tests/test_mimo_tiny_parity.py` (MiMo G1) only asserted tolerances | both print a per-component, per-layer table (max abs, rel, tolerance, fraction of tolerance used) and the worst entry per layer; tolerances unchanged (`03decf4`) | closed; output in section 4 |

Other CUDA jobs, CONFIRMED by reading the workflows at `e71b241`: `strata-glm-sm89` (base A, sm_89 and sm_86),
`lt-strata-sm89` (base B) and `.github/workflows/engines-sm89.yml` (the R1 engines) build the forks; none of them
compiles `scripts/diag` (`grep -n "scripts/diag\|test_diag_kit" .github/workflows/*.yml` after the change matches only
`ci.yml`, section 5). The new job does not duplicate them.

The only tracked CUDA sources outside `third_party/` are `scripts/diag/gpu_p2p_matrix.cu` and
`scripts/diag/gpu_vram_check.cu` (CONFIRMED, `git ls-files '*.cu' '*.cuh'`, section 5). `CudaSourcesInCi` fails when a
new one appears that `CudaCompile` does not build (e.g. a future `kernels/*.cu`), in every job, nvcc or not.

## 1. Changes

- `8242eec` `.github/workflows/ci.yml`, `tests/test_diag_kit.py`: gap 11 (b), (c).
  - `CudaCompile` builds every `scripts/diag/*.cu` once in `setUpClass` with the old flags
    (`nvcc -O2 -arch=sm_89 -Xcompiler -Wall`), `test_compile_sm89` checks each build, `test_binaries_hold_sm89_code`
    runs `cuobjdump --list-elf` (next to nvcc) and requires `.sm_89.cubin` (skips if cuobjdump is absent; in CI a skip
    fails the job, so the package `cuda-cuobjdump-13-0` is installed).
  - `CudaSourcesInCi` (no nvcc): tracked CUDA sources == the ones `CudaCompile` builds; `cuda-compile` has no `exit 0`
    and no `kernels/` glob outside comments, installs nvcc and cuobjdump, runs `CudaCompile` with
    `NVCC=/usr/local/cuda-13.0/bin/nvcc`, fails on a skip (`skipped=` or `... skipped '`) and on an empty run; the
    `cpu-parity` `unit tests` step is `shell: bash` (pipefail through `tee`) and emits the warning.
  - The skip detection does not use a bare `grep skipped` (as `ref-torch-cpu` does): test names here contain the
    word, which made the first version of the step fail on a clean run (section 3, found while simulating the step).
- `03decf4` `tests/test_glm_tiny_parity.py`, `tests/test_mimo_tiny_parity.py`: gap 14. Every comparison is recorded;
  `tearDownModule` prints the table to stderr after the tests (same place as N2's `tearDownClass`, so
  `tools/ci/check_unittest.py` still finds one summary). Router weights (absolute `delta=1e-5`) are recorded as `abs`.
  The helpers now also require `got.size == want.size` and read `got` in `want`'s shape: before, a numpy broadcast
  between mismatched shapes would have been compared silently. No comparison broadcast in the runs of section 4
  (they pass with the size check).

## 2. Unit suites and YAML

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests      (at 03decf4)
----------------------------------------------------------------------
Ran 1102 tests in 210.214s

OK (skipped=555)
exit=0

$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest tests.test_check_unittest tests.test_hooks
exit=0
Ran 54 tests in 30.580s

OK

$ python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('yaml ok:', ' '.join(sys.argv[1:]))" .github/workflows/*.yml
yaml ok: .github/workflows/ci.yml .github/workflows/engines-sm89.yml .github/workflows/parity-r1-pin.yml .github/workflows/traces-real.yml .github/workflows/vendor-deepseek.yml .github/workflows/vendor-hf-branch.yml .github/workflows/vendor-hf.yml
```

The 555 skips of `discover` are the dump-parity tests without their dumps (by design, N13 section 6) plus the two
`CudaCompile` tests (no nvcc in the sandbox's PATH).

## 3. The CUDA build, locally, with a real nvcc 13.0 (gap 11 (b), (c))

The sandbox has no CUDA toolkit. NVIDIA's pip wheels were installed into a scratch venv (`<S>/n22/venv-cuda`):
`nvidia-cuda-nvcc 13.0.88`, `nvidia-cuda-runtime 13.0.96`, `nvidia-nvvm 13.0.88`, `nvidia-cuda-crt 13.0.88`,
`nvidia-cuda-cuobjdump 13.4.92` (no 13.0 cuobjdump wheel exists). Two local-only adjustments, not in the repo: nvvm and
crt pinned to 13.0 (the default 13.4 nvvm emits PTX 9.4, which ptxas 13.0 rejects: `Unsupported .version 9.4; current
version is '9.0'`), and a symlink `cu13/lib64 -> lib` (nvcc looks for `libcudart_static.a` / `libcudadevrt.a` in
`lib64`; the wheel ships them in `lib`). The CI job installs NVIDIA's apt packages for 13.0 instead, the same toolkit
the `strata-glm-sm89` / `lt-strata-sm89` jobs use; that combination is NOT RUN here.

```
$ <S>/n22/venv-cuda/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0

$ NVCC=<S>/n22/venv-cuda/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc python3 -m unittest tests.test_diag_kit.CudaCompile tests.test_diag_kit.CudaSourcesInCi -v
test_binaries_hold_sm89_code (tests.test_diag_kit.CudaCompile.test_binaries_hold_sm89_code) ... ok
test_compile_sm89 (tests.test_diag_kit.CudaCompile.test_compile_sm89) ... 
  gpu_p2p_matrix.cu: nvcc -arch=sm_89 OK, 1075888 bytes
  gpu_vram_check.cu: nvcc -arch=sm_89 OK, 1024920 bytesok
test_cpu_parity_reports_the_skipped_cuda_build (tests.test_diag_kit.CudaSourcesInCi.test_cpu_parity_reports_the_skipped_cuda_build) ... ok
test_cuda_compile_job_runs_the_build_and_fails_on_a_skip (tests.test_diag_kit.CudaSourcesInCi.test_cuda_compile_job_runs_the_build_and_fails_on_a_skip) ... ok
test_every_tracked_cuda_source_is_built_by_cuda_compile (tests.test_diag_kit.CudaSourcesInCi.test_every_tracked_cuda_source_is_built_by_cuda_compile) ... ok

----------------------------------------------------------------------
Ran 5 tests in 8.383s

OK
```
(That run was before two cosmetic edits of the same commit: a newline before `ok`, and the test name
`test_cpu_parity_warns_when_the_cuda_build_does_not_run`.)

**The step body itself.** The `run:` text of the new `cuda-compile` step was extracted from `ci.yml` with PyYAML and
run with `bash --noprofile --norc -eo pipefail` (what `shell: bash` runs) in the worktree:

```
=== A: cuda-compile step as written, no /usr/local/cuda-13.0 on this machine
exit=1
::notice::sm_89 CUDA build: Ran 5 tests in 0.006s OK (skipped=2)  
::error::the sm_89 CUDA build was skipped (nvcc, cuobjdump or git not found)
=== B: same step, nvcc path replaced by the scratch CUDA 13.0 nvcc
exit=0
::notice::sm_89 CUDA build:   gpu_p2p_matrix.cu: nvcc -arch=sm_89 OK, 1075888 bytes   gpu_vram_check.cu: nvcc -arch=sm_89 OK, 1024920 bytes Ran 5 tests in 9.091s OK  Cuda compilation tools, release 13.0, V13.0.88 Build cuda_13.0.r13.0/compiler.36424714_0 
=== C: B with the test mutated to build sm_86
exit=1
::notice::sm_89 CUDA build:   gpu_p2p_matrix.cu: nvcc -arch=sm_89 OK, 1075888 bytes   gpu_vram_check.cu: nvcc -arch=sm_89 OK, 1024920 bytes Ran 5 tests in 9.267s FAILED (failures=2)  Cuda compilation 
::error::2:  test_binaries_hold_sm89_code (tests.test_diag_kit.CudaCompile.test_binaries_hold_sm89_code) (src='gpu_p2p_matrix.cu') ... FAIL
::error::3:  test_binaries_hold_sm89_code (tests.test_diag_kit.CudaCompile.test_binaries_hold_sm89_code) (src='gpu_vram_check.cu') ... FAIL
::error::13:FAIL: test_binaries_hold_sm89_code (tests.test_diag_kit.CudaCompile.test_binaries_hold_sm89_code) (src='gpu_p2p_matrix.cu')
::error::19:AssertionError: Regex didn't match: '\\.sm_89\\.cubin' not found in 'ELF file    1: gpu_p2p_matrix.1.sm_86.cubin\nELF file    2: gpu_p2p_matrix.2.sm_86.cubin\n' : ELF file    1: gpu_p2p_ma
...
::error::37:FAILED (failures=2)
```
(C: the `-arch` flag of the test was `sed`-ed to `sm_86` and restored afterwards; the `notice` text still said sm_89
because the label was hard-coded then; it now prints `CudaCompile.ARCH`.) Before the skip detection was narrowed,
B exited 1: `grep -q 'skipped'` matched the test name `test_cpu_parity_reports_the_skipped_cuda_build` on a clean run.

**The `cpu-parity` `unit tests` step body**, run the same way (`LLAMA_CPP_GGUF_PY` set, at `03decf4`):

```
exit=0
::notice::unit tests: Ran 1102 tests in 224.650s OK (skipped=555) 
::warning::scripts/diag/*.cu NOT compiled in this job (no nvcc here): tests.test_diag_kit.CudaCompile skipped; the sm_89 build runs in job cuda-compile
```

## 4. Gap 14: the per-layer tables, through `tools/ci/check_unittest.py` at both pins

Build: `<S>/n22/build-ec` (`cmake -S tools/g0 -DLLAMA_CPP_DIR=<S>/llama-ec7630a`, target `dump-tensors`) and
`<S>/n22/build-r1` (configured by the drivers on `<S>/llama.cpp` at `79e2e74`). The N17 / N19 builds were not reused:
their `CMakeCache.txt` has `CMAKE_HOME_DIRECTORY=/home/user/wt-n19/tools/g0`, another source tree (identical content,
`diff -r` shows only `__pycache__`), and CMake refuses a cache from a different source directory. The drivers ran with
`TMPDIR=<S>/n22/tmp`.

```
$ python3 tools/ci/check_unittest.py --step g1-glm --checkout <S>/llama-ec7630a -- tools/tinygen/run_parity.sh <S>/llama-ec7630a <S>/n22/build-ec
g1-glm exit=0
::notice::g1-glm @ llama.cpp ec7630a: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
$ python3 tools/ci/check_unittest.py --step g1-mimo --checkout <S>/llama-ec7630a -- tools/tinygen/run_parity_mimo.sh <S>/llama-ec7630a <S>/n22/build-ec
g1-mimo exit=0
::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
$ python3 tools/ci/check_unittest.py --step g1-glm --checkout <S>/llama.cpp -- tools/tinygen/run_parity.sh <S>/llama.cpp <S>/n22/build-r1
g1-glm exit=0
::notice::g1-glm @ llama.cpp 79e2e74: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
$ python3 tools/ci/check_unittest.py --step g1-mimo --checkout <S>/llama.cpp -- tools/tinygen/run_parity_mimo.sh <S>/llama.cpp <S>/n22/build-r1
g1-mimo exit=0
::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
```

The tables at `ec7630a` and `79e2e74` are byte-identical (`diff` of the table sections: GLM 158 lines, MiMo 37 lines).
The checker's counts are unchanged (Ran 9 / Ran 5, no skips): the table does not disturb its parsing.

GLM (G1 op-by-op classes + N9 `GlmTinyWholeModel`), ec7630a, worst entry per layer (the full table has 155 rows: one
per component and layer):
```
worst per layer:
  L-1  GlmTinyWholeModel.logits: max abs 1.910e-05, rel 5.474e-06, 0.055 of tol 1e-04 rel
  L0   GlmTinyParity.hc_comb: max abs 2.501e-07, rel 2.502e-07, 0.025 of tol 1e-05 rel
  L1   GlmTinyParity.hc_post: max abs 7.728e-07, rel 3.864e-07, 0.039 of tol 1e-05 rel
  L2   GlmTinyParity.hc_comb: max abs 4.306e-07, rel 4.306e-07, 0.043 of tol 1e-05 rel
  L3   GlmTinyParity.hc_comb: max abs 3.615e-07, rel 3.615e-07, 0.036 of tol 1e-05 rel
  L4   GlmTinyWholeModel.l_out: max abs 3.637e-05, rel 5.025e-06, 0.050 of tol 1e-04 rel
  L5   GlmTinyWholeModel.l_out: max abs 3.118e-05, rel 4.330e-06, 0.043 of tol 1e-04 rel
  L6   GlmTinyParity.hc_post: max abs 9.214e-07, rel 4.607e-07, 0.046 of tol 1e-05 rel
  L7   GlmTinyParity.hc_post: max abs 9.313e-07, rel 4.657e-07, 0.047 of tol 1e-05 rel
  L8   GlmTinyWholeModel.l_out: max abs 3.487e-05, rel 4.476e-06, 0.045 of tol 1e-04 rel
  L9   GlmTinyWholeModel.l_out: max abs 3.876e-05, rel 4.779e-06, 0.048 of tol 1e-04 rel
  L10  GlmTinyWholeModel.l_out: max abs 4.966e-05, rel 5.819e-06, 0.058 of tol 1e-04 rel
  L11  GlmTinyWholeModel.l_out: max abs 6.637e-05, rel 6.606e-06, 0.066 of tol 1e-04 rel
```
Excerpt of the full GLM table (layer 3 = a DSA layer, layer 11 = the last):
```
class              component              layer    max abs        rel          tol  of tol
GlmTinyDsaParity   indexer key                3  5.160e-07  1.800e-07    1e-04 rel   0.002
GlmTinyDsaParity   indexer scores             3  3.661e-07  2.552e-07    1e-04 rel   0.003
GlmTinyDsaParity   q absorbed                 3  7.298e-07  1.638e-07    1e-05 rel   0.016
GlmTinyDsaParity   kqv_out                    3  9.021e-07  2.648e-07    1e-04 rel   0.003
GlmTinyParity      kda scan                  10  3.163e-08  9.964e-08    1e-04 rel   0.001
GlmTinyParity      hc_comb                   11  5.072e-07  5.072e-07    1e-05 rel   0.051
GlmTinyParity      router weights            11  1.665e-07  1.042e-07    1e-05 abs   0.017
GlmTinyParity      swiglu clamp              11  1.988e-08  1.277e-07    1e-05 rel   0.013
GlmTinyWholeModel  l_out                     11  6.637e-05  6.606e-06    1e-04 rel   0.066
```
MiMo G1, ec7630a, full table (34 rows) shortened to the worst per layer:
```
worst per layer:
  L0   kqv_out (global): max abs 3.989e-07, rel 2.134e-07, 0.021 of tol 1e-05 rel
  L1   attn_out: max abs 2.711e-07, rel 1.251e-07, 0.013 of tol 1e-05 rel
  L2   Qcur rope: max abs 5.431e-07, rel 1.452e-07, 0.015 of tol 1e-05 rel
  L3   attn_out: max abs 3.512e-07, rel 1.406e-07, 0.014 of tol 1e-05 rel
  L4   Qcur rope: max abs 6.397e-07, rel 1.861e-07, 0.019 of tol 1e-05 rel
  L5   attn_out: max abs 3.844e-07, rel 1.927e-07, 0.019 of tol 1e-05 rel
```
Reading (PROVISIONAL, from the tables above): the closest any G1 comparison comes to its tolerance is 6.6 % (GLM
`l_out` of the last layer in the whole-model oracle, error growing with depth from 0.5 % at L0); every op-level check
uses at most 5.3 % of its tolerance (GLM `hc_comb` L10), MiMo at most 2.1 %. Full logs: `<S>/n22/g1-{glm,mimo}-{ec,r1}.log`.

## 5. Inventory commands

```
$ git ls-files '*.cu' '*.cuh' | grep -v '^third_party/'
scripts/diag/gpu_p2p_matrix.cu
scripts/diag/gpu_vram_check.cu
$ grep -n "scripts/diag\|test_diag_kit" .github/workflows/*.yml | cut -c1-60      (at 03decf4)
.github/workflows/ci.yml:25:      # No nvcc in this job: tes
.github/workflows/ci.yml:34:            echo "::warning::scr
.github/workflows/ci.yml:93:    # kernels/ directory; today 
.github/workflows/ci.yml:105:      - name: scripts/diag/*.cu
.github/workflows/ci.yml:109:          NVCC=/usr/local/cuda-
$ git show e71b241:.github/workflows/ci.yml | grep -n "kernels\|cuda-compile:"
82:  cuda-compile:
91:      - name: compile kernels for sm_89
93:          shopt -s nullglob; files=(kernels/*.cu)
94:          if [ ${#files[@]} -eq 0 ]; then echo "no kernels yet"; exit 0; fi
$ git show e71b241:tests/test_diag_kit.py | sed -n 557p
@unittest.skipUnless(which_nvcc(), "nvcc not available (set NVCC=/path/to/nvcc)")
```

## 6. Gap 11 (d): not fixable on this branch

`tests/test_srvcheck.py` does not exist on `claude/n22-ci-false-green` (nor on `main` or N19): `git show HEAD:tests/test_srvcheck.py`
-> `fatal: path 'tests/test_srvcheck.py' does not exist in 'HEAD'`. CONFIRMED, it exists with the same stale line 5
(`... The run against the real llama-server is tests/test_srvcheck_llama.py (needs a build).`) on exactly three
branches: `claude/n7-server-hybrid-cpu` (`c2f6c91`), `claude/n8-session-store` (`c4c9015`) and
`claude/n11-r1-tools-cpu-server` (`6680448`), local and origin alike. The real run is `tools/srvcheck/run_n7.sh`, called by
the step `N7 llama-server checks on tiny qwen4exp and glm5-next (CPU)` of `.github/workflows/server-cpu.yml` on those
branches. Adding the file here would import N7 into this branch, so the fix is a patch for those branches, kept in
scratch: `<S>/n22/srvcheck-docstring.patch`; `patch -p1 --dry-run` applies cleanly to the file of all three branches.

```
-pass everything. The run against the real llama-server is tests/test_srvcheck_llama.py (needs a build).
+pass everything. The run against the real llama-server is tools/srvcheck/run_n7.sh (needs a llama.cpp build; CI:
+the N7 step of .github/workflows/server-cpu.yml).
```

## 7. NOT RUN

- Both changed jobs on GitHub Actions (this task does not push). On push: `cuda-compile` must end green with the notice
  `sm_89 CUDA build: ... nvcc -arch=sm_89 OK ... Ran 5 tests ... OK` and no `skipped`; `cpu-parity` must show the
  warning annotation and the `unit tests: Ran ... OK (skipped=...)` notice. Both are readable through the GitHub
  check-run annotations API.
- The apt CUDA 13.0 packages (`cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cuobjdump-13-0`) on the runner; the same
  `nvcc`/`cuobjdump` packages are installed by `strata-glm-sm89` and `lt-strata-sm89` at `e71b241`.
- Gap 11 (d): applied after this report by the lead (`4f7be84` on N7, merged into N8 `09eb438` and N11 `b74efde`); GitHub CI on those heads: see the PRs.
- Running the diagnostics binaries: no GPU.

## 9. After the audit (lead, 2026-10-11)

evidence-auditor: no FAIL. MINOR 1 (11 (d) status) fixed above. MINOR 3: the 555 skips of section 2 are not only the
dump-parity tests and the two CudaCompile tests; by skip reason (auditor's `-v` rerun) they also include shellcheck (1),
torch/vLLM (2), GGML_GDN_HARNESS (3), CHATTMPL_* (3) and LLAMA_CPP_FULL_CLONE (1) (CONFIRMED by the auditor's count; the
sentence in section 2 is to be read with this correction). MINOR 2 (CORRECTNESS statuses for gaps 11 and 14) is done by
the lead in the integration branch.
