# lighttransport/Strata, branch glm53f: recon, 2026-10-03 (sandbox, no GPU)

Reported by the customer; verified here by cloning the repository. Nothing was built or run.

## The fork exists and what it is based on
```
$ git clone --filter=blob:none --no-checkout https://github.com/lighttransport/Strata lt-strata && cd lt-strata
$ git branch -r
  origin/HEAD -> origin/glm53f
  origin/glm53f
  origin/main
$ git log -1 --format='glm53f HEAD %H %ad %an | %s' --date=short origin/glm53f
glm53f HEAD e486a95d78876989b853b16c7056bcd96880afc3 2026-10-01 Syoyo Fujita | glm: refresh initial Q2 measurements and decode demo
$ git log --format='%h %ad %s' --date=short $(git merge-base origin/main origin/glm53f)..origin/glm53f
e486a95 2026-10-01 glm: refresh initial Q2 measurements and decode demo
da5a54d 2026-10-01 Rename project to Strata-GLM53F
50ded4f 2026-10-01 glm: add terminal demos and live decode README animation
aaca5d2 2026-10-01 glm: optimize Q2 decode and document quality validation
c9229b3 2026-10-01 glm: add GLM-5.3-Flash inference and benchmarks
merge-base main..glm53f: a790805 2026-09-29 Engine 0.1.27: CJK answers faster ...
pin c499bd1 2026-10-01 docs (AMD): images through the CPU encoder and the Monitor's AMD readings (0.1.32, #304
pin..merge-base commits: 0 ; merge-base..pin: 253
$ git diff --stat <merge-base> origin/glm53f | tail -1
 111 files changed, 34557 insertions(+), 93 deletions(-)
$ git diff --stat <merge-base> origin/glm53f -- src include tools CMakeLists.txt cmake serve setup.py | tail -1
 47 files changed, 6904 insertions(+), 93 deletions(-)
$ git show origin/glm53f:LICENSE | head -3
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors
```
CONFIRMED: a real fork of Niko1221/Strata, 5 commits on top of Strata a790805 (Engine 0.1.27); MIT.
CONFIRMED: our Strata pin c499bd1 (0.1.32) is 253 commits ahead of the fork's base.
Largest code files: src/program/glm_decode.cpp (2606 lines), src/kernels/cuda/glm.cu (497), src/kernels/cuda/glm_prefill.cu (484),
src/kernels/glm_parity.cpp (302), include/strata/core/model.hpp (+276), src/kernels/glm_prefill_parity.cpp (200).

## sm_89
CMakeLists.txt:60-76 @e486a95: default architecture 120; anything below 75 is refused; the comment lists RTX 40 (89) as building.
CONFIRMED by reading. Build for sm_89: NOT RUN.

## What the author states (PROVISIONAL for us: the author's claims and measurements, not ours)
docs/README_GLM53_FLASH.md, docs/GLM53_FLASH.md @e486a95.
- Hardware: Threadripper 1950X, 160 GB DDR4, RTX 5060 Ti 16 GB. Model: UD-Q2_K_XL, four shards.
- Decode: 7.33 tok/s median (short chat), 7.26 (4096-token C++ prompt), 8.35 with GPU MTP depth 1; three trials each.
- Prefill of the 4096-token prompt: 98.29 tok/s at batch 2048, 158.25 tok/s at batch 4096.
- Q2 payload: main experts 99.03 GB, fixed weights 6.89 GB, MTP block 2.79 GB, total 108.71 GB / 101.25 GiB;
  selected expert weights per decoded token 2.751 GB.
- Implemented: split-GGUF reader with validation; layer description from metadata; mixed expert quantisation
  (Q2_K/Q3_K grouped execution, IQ3_S GPU down projection); CUDA router, mHC Sinkhorn, KDA, IndexPool, MLA; prefill and
  greedy decode with CPU routed experts (default) or `--decode-experts=gpu`; optional static GPU expert cache with a
  budget per MoE layer; snapshot/replay checks; GLM4 BPE tokenizer and chat template; `serve.server --engine glm`; GPU and CPU MTP.
- Not implemented: prefix reuse, multi-user batching, tool-call parsing. Multi-GPU is not mentioned anywhere (UNKNOWN; one GPU tested).
- The C++ answers of both modes compiled and passed 400,532 cases each.

## The author does not claim numerical parity (CONFIRMED: the author's own text)
docs/GLM53_FLASH.md:141-154 @e486a95: against a CPU llama.cpp reference at 1665c0e (timkhronos' branch) one prefix gave the same
greedy token, full-vocabulary relative L1 difference 0.05923 and maximum absolute logit difference 0.84482; "This single-prefix
comparison is not an accuracy benchmark or a parity pass"; "These checks do not establish end-to-end numerical parity or
model-quality benchmarks." Also line 79: "CPU and GPU experts quantize activations differently, so cache hits can change logits."
The artifact's GGUF architecture string is `glm5next`; llama.cpp at our pin uses `glm5-next` (docs/models/glm-5.3-flash.md).

## What it means for the 4 x 32 GB server (PROVISIONAL, tools/fitplan.py, model not measurement)
```
$ python3 tools/fitplan.py models/glm-5.3-flash.json --ctx 32768 --quant IQ2_XXS | grep -E '^experts|VRAM-resident'
experts: 12384 x 25.17M params = 74.8 GiB (6.19 MiB each)
VRAM-resident experts: 100.0% (12384)  | spill to RAM: 0.0 GiB of 116 usable
$ ... --quant IQ3_XXS
experts: 12384 x 25.17M params = 111.1 GiB (9.19 MiB each)
VRAM-resident experts: 100.0% (12384)  | spill to RAM: 0.0 GiB of 116 usable
$ ... --quant Q4_K
experts: 12384 x 25.17M params = 163.3 GiB (13.50 MiB each)
VRAM-resident experts: 68.5% (8479)  | spill to RAM: 51.5 GiB of 116 usable
```
The fork's Q2 artifact (101.25 GiB in total by the author's count) is smaller than the server's 128 GiB of VRAM, so the CPU expert
path that limits the author's machine is not needed on ours if experts are placed on all four GPUs. That placement does not
exist in the fork (one GPU) and is our work. No speed figure follows from this until it runs on the GPUs.

## Build for sm_89 on Linux and the fork's CPU-only tests (CI, no GPU) - CONFIRMED
Job `lt-strata-sm89` on commit b2401c0, check-run 111158460817, conclusion success. Annotations, literal:
```
[notice] lt-strata at e486a95d78876989b853b16c7056bcd96880afc3
[notice] configure OK
[notice] lt-strata build OK, 62 CUDA objects
[notice] binary build-lt/strata-glm-decode, 32478656 bytes
[notice] cubin archs:      18 sm_89
[notice] gguf_reader_test: 1/1 Test #1: gguf_reader_test .................   Passed    0.00 sec 100% tests passed, 0 tests failed out of 1
[notice] model_test: 1/1 Test #2: model_test .......................   Passed    0.00 sec 100% tests passed, 0 tests failed out of 1
```
Targets built: strata-glm-decode, strata-model-inspect, gguf_reader_test, model_test (CUDA 13.0 toolkit, -DCMAKE_CUDA_ARCHITECTURES=89,
no patches of ours). Not covered: glm_parity and the other GPU tests (need a GPU), any run of the decoder, any model file.

## Probe: base B on our tiny GLM GGUF (CI, no GPU) - CONFIRMED
Job `lt-strata-sm89` on commit 69f8c32, check-run 111168779834 (success; the probe step is exploratory and never fails the job).
`tools/tinygen/glm5next_tiny.py --layers 12 --arch <arch>`, then `strata-model-inspect`. Annotations, literal:
```
[notice] strata-model-inspect, tiny GGUF arch=glm5next (4157568 bytes): exit 1
[notice] arch=glm5next output head: model: missing or invalid integer glm5next.expert_group_count|
[notice] strata-model-inspect, tiny GGUF arch=glm5-next (4157600 bytes): exit 1
[notice] arch=glm5-next output head: model: unsupported architecture glm5-next|
```
There are two GGUF dialects of GLM-5.3-Flash and they are not interchangeable by renaming the architecture string:
- `glm5-next`: llama.cpp master (PR #27773), our reference graph and the G1 oracles; base B refuses it by name;
- `glm5next`: the Unsloth UD files; read by base B and by ik_llama.cpp. Its metadata has keys our generator does not write
  (first one hit: `glm5next.expert_group_count`), so the schemas differ beyond the name.
Consequence: a parity run of base B against our reference needs the same weights in both dialects - either a second tinygen
writer for the `glm5next` schema (keys from the fork's include/strata/core/model.hpp) or a converter. Whether tensor names and
layouts also differ: UNKNOWN (the fork's author states only the name mapping was needed for the reverse direction).
The same split is seen on the 3090 host: ik_llama.cpp loads the `glm5next` UD-Q4_K_XL and refuses a `glm5-next` Q4_K file
(PROBLEMS.md, PROVISIONAL: measured there).
