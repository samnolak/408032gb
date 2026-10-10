# R9. Adversarial review of the decode ceilings (ROOFLINE.md rev. 3, ADR-012 context item 3, tools/roofline.py)

Date: 2026-10-10. Role: independent adversarial architecture reviewer (research brief of 2026-10-10). Nothing was run on a
GPU; **no speed in this file is measured on the target server**. Every tok/s number below is a ceiling computed by
`tools/ceilings/arch_ceilings.py` (new, with tests) from stated parameters, or a number reported by someone else on other
hardware (then it says so).

Labels (brief, rule 3): **VERIFIED** = I read the code/file at the recorded SHA or ran the command and quote its output;
**REPORTED** = stated by a README/doc/issue/search result (source given; it says on which hardware); **HYPOTHESIS** = my
inference or model output (formula and inputs given); **UNKNOWN**; **DISPROVED** (with the disproving source). Repository
mapping (AGENTS.md): VERIFIED = CONFIRMED; REPORTED/HYPOTHESIS = PROVISIONAL.

Worktree state: the brief names `b6f4fe3` as the head of `claude/research-landscape`; during this review three commits by
other roles landed (`5100adb`, `6715f40`, `8981351`: an expert-prediction probe and a CI workflow for real-model routing
traces). I read them (section 2.1). I committed nothing.

---

## 0. Verdict in brief

| # | claim under review (ROOFLINE.md rev. 3 / ADR-012 ctx. 3) | verdict |
|---|---|---|
| C1 | GLM-5.3-Flash UD-Q4_K_XL: layer pipeline (PP) without speculation ≤ ~42 tok/s at any hit rate | **CONFIRMED in substance**, number is engine-dependent. Physics ceiling of a strict PP at 262K at the 736 GB/s spec and 100% bandwidth efficiency: **52.5** (hit 1.0) / **54.4** (hit 0.907). 50 tok/s needs a GEMV efficiency ≥ 0.98 of peak with modest fixed costs: not excluded by physics, done by no engine (HYPOTHESIS, Tables E, F). |
| C2 | with speculation 50.6–52.9 only with CPU/GPU overlap and hit ≥ ~0.8; hit ≥ 0.8 needs a routing skew "against which base A's 24.6% reuse speaks" | numbers **CONFIRMED** by an independent formulation (PP + MTP reaches 50 from hit ≥ 0.78–0.80, Table F). The premise is **DISPROVED as an argument**: 24.6% previous-token reuse is 8.9× the 2.78% of uniform routing, and the same source line says the top 1,310 experts (10.8%) carry 42–64% of the routed mass. A second, independent count file gives a frequency-sorted placement at our 60.7% residency an in-sample hit of **0.907** (section 2.1). hit ≈ 0.8–0.9 is the expected range if that concentration carries over to the customer's workload (UNKNOWN until traced). |
| C3 | TP@auto ≈ 41 tok/s | **REFUTED as a ceiling of TP.** "auto" = uniform routing, contradicted above; and the ×4 replication of the MLA latent is a placement choice that an existing engine does not make (ExLlamaV3 runs MLA/DSA layers whole on one GPU, VERIFIED). At uniform routing TP is DRAM-bound and ~41 is right (Table A); at the profile hit the same TP gives **78 (base) / 110 (optimistic)** tok/s without speculation and **109 / 147** with MTP k=2 (8 DRAM channels; Tables A, B). |
| C4 | target 80–100 unreachable without TP of the dense part | **CONFIRMED for strict PP and EP** (the dense part runs on one GPU at a time: ≤ 54.4 at peak, Table E). Two qualifications: (a) **pipelined speculative windows** use the idle stages without collectives: upper bound **84** (base, MTP k=1, W=4) — HYPOTHESIS, no engine does it beyond 2 stages; (b) ADR-012 D2 "TP of the dense part — not in ready engines" is **DISPROVED at code level**: ExLlamaV3 `151539c` declares `supports_tp` for `glm5_next` with a no-P2P backend. What no examined engine has is **TP + CPU-computed experts** for GLM on AVX2 with 128 GB RAM. |
| C5 | Qwen3.8-Flash-Next all in VRAM: PP 84–87, with speculation 95–101 | **CONFIRMED as PP numbers** (Table C). Not the ceiling of the server: TP (ExLlamaV3 also declares `supports_tp` for `qwen4_exp`, VERIFIED) gives **168–173** without speculation; **100 tok/s needs an all-reduce ≤ 59 µs, 150 needs ≤ 24 µs** (Table F2). |

What decides the GLM answer, in order (sensitivity, Table G): **(1) the VRAM hit share on the customer's workload**
(uniform 0.61 → TP 41; profile 0.91 → TP 78); **(2) DRAM channels** (4 vs 8: TP 62 vs 78 at the profile hit, 26 vs 41
at uniform); **(3) all-reduce latency and per-layer fixed GPU cost** (80 tok/s without speculation needs ≤ 11 µs and
≤ 12.5 µs respectively; with MTP k=2, ≤ 97 µs and ≤ 184 µs); (4) GEMV bandwidth efficiency; (5) CPU expert-kernel
throughput when it is below DRAM bandwidth.

---

## 1. What the current ceiling model assumes (code reading, reproduction)

`tools/roofline.py` @ worktree (VERIFIED, read in full):
- Time per step = bytes / bandwidth; **no kernel-launch, sync or compute terms** (docstring lines 14–16).
- PP: `t = (w_dense + w_kv)/gbw + L_moe·layer(e_gpu, e_off + 2·lat) + (n_gpu−1)·lat + w_draft/gbw` (lines 260–263): one
  GPU active at a time.
- TP: everything dense divided by `n_gpu`, KV per GPU with undividable groups (MLA latent, indexer key) **read and stored
  whole on every GPU** (lines 64–67, 270–273; placement replicates them, lines 174, 76–87), `n_layers·2·allreduce_us`.
- EP: dense part as in PP, experts divided by `n_gpu` × top-k imbalance, `2·lat` per MoE layer (lines 264–269).
- Hit `auto` = resident share of bytes (**uniform routing**), lines 356–357; misses on the CPU at
  `channels × 25.6 GB/s × 0.7` (lines 219–220), overlapped per layer or serial (lines 255–258).
- Speculation: union of experts under **independent uniform** routing (lines 155–157), MTP draft bytes charged
  (lines 126–135), tokens/step `(1−a^(k+1))/(1−a)`.
- Hardware: `hardware/target-4x4080-32g.unverified.json`: 736 GB/s × 0.75, 8 ch × 0.7, 25 GB/s PCIe, 10 µs, 30 µs all-reduce.

Reproduction (VERIFIED, command and literal output, trimmed to the rows used):
```
$ python3 tools/roofline.py models/specs/glm-5.3-flash.json --ctx 262144 --expert-bpw 4.89 --dense-bpw 8.04 --sweep-hit auto,0.9,1.0 --strategies pp,tp
  VRAM fixed 20.3 GiB, expert cache 107.7 GiB -> resident 60.7% | RAM 39.3% | NVMe 0.0% (0.0 GiB)
  under tp: KV+state 14.58 GiB (undividable KV replicated on every GPU) -> resident 54.6% | RAM 45.4% | NVMe 0.0%
  ctx  262144 hit VRAM 0.61 RAM 0.39 SSD 0.00 pp/cpu  RAM 8ch:   32.2 tok/s (31.05 ms/step; read GiB dense 7.76 routed 4.81 kv 0.478)
  ctx  262144 hit VRAM 0.55 RAM 0.45 SSD 0.00 tp/cpu  RAM 8ch:   41.3 tok/s (24.19 ms/step; read GiB dense 7.76 routed 4.81 kv 0.478)
  ctx  262144 hit VRAM 0.90 RAM 0.10 SSD 0.00 pp/cpu  RAM 8ch:   40.9 tok/s (24.47 ms/step; ...) | envelope 42.2 tok/s at hit 0.82
  ctx  262144 hit VRAM 0.90 RAM 0.10 SSD 0.00 tp/cpu  RAM 8ch:   87.5 tok/s (11.43 ms/step; ...)
  ctx  262144 hit VRAM 1.00 RAM 0.00 SSD 0.00 pp/cpu  RAM 8ch:   39.4 tok/s (25.41 ms/step; ...) | envelope 42.2 tok/s at hit 0.82
  ctx  262144 hit VRAM 1.00 RAM 0.00 SSD 0.00 tp/cpu  RAM 8ch:  107.2 tok/s (9.33 ms/step; ...) | envelope 108.0 tok/s at hit 0.97
```
The published table of ROOFLINE.md §2–3 reproduces exactly; the new tool reproduces `tools/roofline.py` to ≤ 3e-16
relative error for pp/tp/ep, overlap on/off, MTP on/off (test `RooflineConsistency`, section 9).

The model is internally sound. The refutable parts are its **inputs and its menu of architectures**: uniform routing as the
"auto" case, the replicated-latent TP placement, no fixed costs (which makes TP look better than it is at high hit), the
absence of pipelined speculation, and a CPU path priced at DRAM bandwidth even when the kernel cannot reach it.

---

## 2. Evidence that changes the inputs

### 2.1 GLM-5.3-Flash routing is strongly concentrated (published data); "24.6% = flat" is a misreading

- strata-glm-3090 `794cd69` `PROBLEMS.md:110-113` (REPORTED, real NVFP4 weights, 16 real segments code/docs/prose):
  "expert reuse from the previous token 24.6% (the synthetic pack: ~60%), per-layer top-1310 covers 42-64% of the routed
  mass". Under uniform independent top-8-of-288 routing the reuse would be 8/288 = **2.78%** and the top 10.8% of experts
  would carry 10.8% of the mass (arithmetic). The author's word "spread" compares against the synthetic pack, not against
  uniform. ROOFLINE.md:31, ROOFLINE.md:82, ADR-012:18 and EVIDENCE_MATRIX E33 read it as "flat routing — an argument
  against a large cache gain": **DISPROVED** as an inference by the same source line.
- 0xSero/glm53-flash-offload `6769b27`, `data/stats_own_dec.json` (REPORTED data; sha256 `79bff1e3…`; 42 layers × 288
  counts, 74,000 picks per layer ≈ 9,250 decoded tokens; workload not documented; the repo warm-starts its VRAM cache from
  it, `docs/how-it-works.md:53-54`). My computation (VERIFIED as arithmetic, script in Appendix A):
  ```
  layers 42 experts/layer {288}
  picks per layer: min 74000 max 74000 mean 74000 -> ~9250 tokens (top-8)
  capacity 0.107: per-layer static hit mean 0.371 (min 0.173 max 0.487) | global-budget static hit 0.378
  capacity 0.546: per-layer static hit mean 0.874 (min 0.666 max 0.953) | global-budget static hit 0.887
  capacity 0.607: per-layer static hit mean 0.907 (min 0.723 max 0.971) | global-budget static hit 0.923
  capacity 0.700: per-layer static hit mean 0.946 (min 0.803 max 0.987) | global-budget static hit 0.962
  CV of per-expert counts: mean 1.097 min 0.318 max 1.461
  CV expected from sampling noise alone under uniform routing: 0.062
  ```
  **In-sample** (the counts both choose and score the placement): an upper reference for a static placement, not a
  forecast for agentic coding at 262K. Cross-check (REPORTED): the same repo's live CLOCK cache with ~1,300 slots (≈10.7%
  of experts) serves "about 120 of the ~318 unique picks per token … (38-39 %)" (`docs/how-it-works.md:58`), next to the
  0.371–0.378 of the static curve at 0.107: the concentration held on their live prompts.
- Static skew alone explains only part of the temporal reuse: the counts predict 6.4% previous-token reuse (Appendix A),
  the fork measured 24.6% → additional temporal locality (HYPOTHESIS), which helps both caching and speculation unions.
- What the concentration does **not** buy (computed from the same counts, Appendix A): the union of experts in a
  verification window shrinks only 5% at n=4 (29.15 vs 30.69) and 10% at n=8, so speculation still multiplies the
  **GPU** expert reads. It does shrink the **CPU** side: at 60.7% residency, distinct missed experts per layer are 0.74 /
  2.91 / 5.68 for n = 1 / 4 / 8 tokens (uniform: 3.14 / 12.06 / 22.84) — misses are cold and add up linearly.
- Project status (VERIFIED, `8981351`): `data/traces/real/REQUEST.json` asks CI for traces of **proxy** models
  (GLM-4.7-Flash, Qwen3-Coder-30B-A3B); no trace data is committed yet. Real GLM-5.3-Flash traces on agentic prompts are
  still UNKNOWN.

### 2.2 ExLlamaV3 already implements most of a "TP without P2P" GLM engine — except the CPU tier

turboderp-org/exllamav3 `151539c` (2026-10-07; = pinned in `tools/engines/pins.json`; MIT). All VERIFIED by reading code:
- `glm5_next.py:350` and `qwen4_exp.py:311`: `"supports_tp": True`. "TP runs one Python process per participating CUDA
  device … The output device … runs synchronously in the main process … A final CPU worker slot uses device -1 for backend
  helper work such as native CPU-based reductions" (`model/model_tp.py:40-53`), so the ranks launch their kernels in
  parallel.
- Native backend, no NCCL/P2P (`model/model_tp_backend.py:221-246`): five named shared-memory regions registered as pinned,
  **mapped** host memory; `all_reduce()` "routes through pg_all_reduce_cpu(): GPU workers publish their contributions into
  the R buffer, a designated CPU helper performs the reduction over host memory, and workers copy the reduced result back"
  (lines 243–246, 458–472). Decode-size payloads: "two ordinary kernels in stream order (send, then recv). Measured 1-3 us/op
  faster than the cooperative launch … CUDA-graph-capturable" (`exllamav3_ext/parallel/all_reduce_cpu.cu:437-441`); the
  CPU reducer spins on a job queue (`run_cpu_reduce_jobs`, same file) with AVX2/AVX-512 accumulate. **Latency per
  all-reduce: not stated anywhere I found (UNKNOWN).**
- MLA/DSA under TP: "the layer runs whole on one device. The latent K/V cache is MQA … the DSA/k-pool indexers are
  headless, so a head split would replicate the whole KV side for little gain … (max_devices = 1)"
  (`modules/mla_attn.py:1195-1244`). This is the "owner" KV policy of section 3 (no ×4 latent replication).
- MoE under TP: expert-parallel by default (`moe_tensor_split` option off → `channel_unit = "experts"`,
  `modules/block_sparse_mlp.py:1536-1547`); **replicated routing** on every rank, "which saves two broadcasts per MoE layer
  per token" (`block_sparse_mlp.py:1646`, `doc/env_vars.md:835`); one all-reduce per MoE layer.
- CPU experts: "Layer-split mode only" (`doc/env_vars.md:315-325`) — **no TP with CPU experts.** In layer-split mode there is
  an "experimental per-layer expert split: … forward() then submits the tail partial to the worker right after routing and
  adds it to the GPU partial, so the CPU GEMMs overlap the layer's own GPU expert compute" (`modules/block_sparse_mlp_cpu.py:
  461-469`), a persistent worker with a job ring in pinned shared memory and stream-ordered flag waits
  (`model/moe_cpu_host.py:25-40`), dynamic hot/cold swaps between generations (`block_sparse_mlp_cpu.py:497-516`). The CPU
  kernel uses int8 activations: "Accuracy matches the GPU int8-GEMV mode-2 class (~0.9% per-call output RMS)"
  (`exllamav3_ext/cpu/moe_mul1.cpp:36-48`).
- Engine sensitivity, REPORTED in its docs (`doc/env_vars.md:704-716`, Ryzen 9 7900X + RTX 4090, Qwen3.8-Flash-Next,
  408 of 512 experts per layer on the CPU): "decode 11–32 tok/s with the host unpinned, 30–36 tok/s with a reserved core".
  Thread placement alone moves decode up to 3×.

### 2.3 Measured efficiency anchors from real engines (all on other hardware)

| anchor | value | label / source |
|---|---|---|
| non-MoE GPU GEMVs of GLM-5.3-Flash in ExLlamaV3 EXL3, 1×3090 | "KDA/DSA attention GEMVs at 50-57 % of VRAM bandwidth" | REPORTED, glm53-flash-offload `README.md:152-153` @6769b27 |
| same system, 1×3090 + EPYC 7443P (24c Zen 3) + 8-ch DDR4, all experts in RAM, EXL3 3.05 bpw | 28.2 tok/s decode; the author's own all-lanes-at-floor estimate ≈ 41 | REPORTED, `README.md:8,14,154-157` |
| AVX2 EXL3 trellis expert kernel, same CPU, 1–3 tokens × 1–6 experts per call | 30.9–69.5 GB/s of weights (compute-bound, IPC 1.1–2.4) | REPORTED, `results/cpu_kernel/n135_batchC_2026-10-09.log` |
| CPU-lane handoff | "Publish to "seen by the host" takes 10 µs at p50" | REPORTED, `docs/how-it-works.md:150` |
| CPU lane numerics | decode KL 0.0047, top-1 0.986 vs the exact GPU path | REPORTED, `docs/how-it-works.md:113` |
| ik_llama.cpp Q4_K experts on CPU, epyc-ai host | calibration ≈ 92–102 GB/s effective (4 ch × 0.9 or 8 ch × 0.5) | PROVISIONAL, ROOFLINE.md §1 |
| SGLang, GLM-5.3-Flash NVFP4, 4× RTX PRO 6000 (one engine, 4 GPUs) | 63.7 tok/s single request, 82.8 with EAGLE | REPORTED, search snippet of deploy.helix.ml (page not reachable from the sandbox; TP size not visible in the snippet) |
| Strata 2-stage pipelined speculative windows | +16% (72.4→84.0 tok/s) | REPORTED via research/R3 §2.5 |

The SGLang row is a warning about TP at batch 1: four workstation GPUs, each with more memory bandwidth than a 4080 Super,
deliver 64 tok/s on one stream — far below any bandwidth roofline of that system (HYPOTHESIS: fixed costs and collectives
dominate; the snippet does not say why, and its TP size is not visible).

### 2.4 All-reduce latency without P2P: what is and is not known

- No measurement of an 8–16 KB all-reduce over 4 GeForce GPUs without P2P was found (searches 2026-10-10: nccl-tests small
  message PCIe no-P2P; ExLlamaV3 native backend latency) — **UNKNOWN**.
- Nearby REPORTED points: nccl-tests 8-byte all-reduce on 2× RTX 4090 with `NCCL_P2P_DISABLE=1` ≈ 6.7 µs (PyTorch forum,
  anecdotal); `p2pBandwidthLatencyTest` with P2P disabled ≈ 10–11 µs GPU-to-GPU (research/R4 §4.3); NCCL busbw without P2P on
  Milan 0.8–4.5 GB/s (R4 §4.3, VERIFIED there).
- Floor (HYPOTHESIS, protocol arithmetic for the ExLlamaV3-style mapped-memory scheme): posted PCIe write of 8–16 KB
  ≈ 1–2 µs, CPU spin detection < 1 µs, CPU sum of 4 × 8 KB ≈ 1 µs, GPU acquire-poll over PCIe 1–2 µs per read plus the
  read-back ≈ 1–2 µs → **≈ 5–8 µs** per all-reduce at best. 90 all-reduces per GLM token → 0.5–0.7 ms at the floor,
  2.7 ms at the roofline's 30 µs, 5.4 ms at 60 µs.

### 2.5 Hardware facts used

- RTX 4080 Super: 23 Gbps × 256 bit = **736 GB/s** — REPORTED (spec pages via search; arithmetic). The modded 32 GB boards
  (clamshell, R4 §3.3) may run the memory at another clock: UNKNOWN.
- DDR4-3200: 25.6 GB/s per channel; 4 or 8 channels UNKNOWN (R4 §2.3).
- PCIe 4.0 x16 ≈ 24–26 GB/s pinned H2D — REPORTED (R4 §5).

---

## 3. Competing architectures: critical-path formulas

Notation per decode step verifying n = 1 + k tokens: D = active dense bytes (GLM UD-Q4_K_XL: 8.33 GB incl. lm_head);
K(n) = KV/state bytes (0.51 GB at 262K, n = 1); e = bytes per expert (15.38 MB); L = 45 layers, L_m = 42 MoE layers;
u(n) = distinct experts per MoE layer; u_hit, u_miss its VRAM and RAM parts; B_g = 736·η_g GB/s per GPU;
B_c = min(DRAM peak·η_d, CPU kernel cap); t_ar all-reduce; t_h one-way GPU↔host handoff; f_l, f_s fixed GPU cost per layer
and per step; f_c CPU job cost per MoE layer; p = probability that a layer has CPU work; τ = tokens per step. All
expectations are taken inside max(), which can only overstate tok/s, so each formula is a ceiling.

**A. Layer pipeline (current baseline; llama.cpp `-sm layer`, Strata, base A).** One GPU active at a time.
```
t = (D + K)/B_g + L_m·comb(u_hit·e/B_g, u_miss·e/B_c + p·(2·t_h + f_c)) + (G−1)·t_hop + L·f_l + f_s + t_draft
comb = max (CPU/GPU overlap inside a layer: Strata, base A --cpu-share, ExLlamaV3 split) | + (llama.cpp -ot)
```
Hardware limits: B_g of one card; DRAM for misses. Engine limits: η_g, f_l, overlap. `pp_serial`, `pp` in the tables.

**B. Expert parallel, dense pipelined (also covers "KV/state and attention on one GPU, experts on the others").**
```
t = (D + K)/B_g + L_m·[comb(imb(u_hit, G)·u_hit·e/(G·B_g), cpu path) + 2·t_x] + ...
```
The dense part stays on one GPU per layer, so B gains nothing on the 8.33 GB that dominate GLM's step; each MoE layer adds a
scatter and a gather (t_x, ≈ 2 host hops without P2P). Moving all attention/KV to one GPU is the same formula. `ep`.

**C. Tensor parallel over T GPUs (+ EP or TP-split experts, + CPU misses).** 2 all-reduces per layer.
```
t = (D_div/T + D_rep + D_own + K_crit)/B_g + L_m·comb(imb·u_hit·e/(T·B_g), cpu path) + 2·L·t_ar + L·f_l + f_s + t_draft/T
replicate (tools/roofline.py): D_own = 0, K_crit = K_kda/T + K_mla (every GPU stores and reads the whole latent)
owner (ExLlamaV3):             D_own = DSA attention weights on one GPU, K_crit = K_kda/T + K_mla on that GPU only
```
`tp_repl`, `tp_owner`; `tp_owner_hostcpu` = the CPU computes the misses on the host-staged reduction path: the CPU helper
already holds the hidden state and sums partials (section 2.2), so the miss path costs no extra PCIe hops, only a router on
the CPU (1.2 MB per layer). `pp2_tp2` = two pipeline stages of TP-2. Hardware limits: aggregate VRAM bandwidth, DRAM;
engine limits: t_ar, f_l (90 collectives and ~45 × 15–30 kernels per token).

**D. CPU/GPU co-execution critical path (KTransformers / Strata / FreeToken / ExLlamaV3-split / glm53-flash-offload).**
Per MoE layer: `t_moe = t_router + max(t_gpu_hits, t_publish + t_wake + max(miss/B_cpu_kernel, miss/B_dram) + t_readback)`.
Implementations that keep the CPU in stream order without host syncs: KTransformers `CPUInfer::submit_with_cuda_stream`
→ `cudaLaunchHostFunc` (`kt-kernel/cpu_backend/cpuinfer.h:89-122` @5d048e5, VERIFIED); ExLlamaV3 memops
`cuStreamWaitValue32` on a worker flag (`doc/env_vars.md` `EXL3_MOE_MEMOPS`, VERIFIED); glm53-flash-offload per-expert
"landed" flags (REPORTED). Layer-ahead **prefetch** variant: a share r (recall) of RAM misses, predicted by running layer
l+1's router on layer l's input, is copied over PCIe and computed by the GPU; precision q wastes r·(1/q − 1) of the miss
bytes: `t_cpu = max((1−r)·miss/B_cpu, (miss·(1 + r/q − r))/B_dram)`, PCIe copy hidden if ≤ one layer. It only helps when
the CPU kernel, not DRAM, binds (Table G, prefetch row).

**E. Speculation (MTP, DFlash2 block drafts, n-gram/suffix) and what a verification window costs on an MoE.**
```
tok/s = τ / t(n = k+1);  τ = (1 − a^(k+1))/(1 − a)  or  1 + Σ P(≥ i accepted)
t(n): D once; GPU experts u_hit(n)·e per layer (u(n) ≈ n·8 for n ≤ 4: union shrinks ≤ 5% from skew);
      CPU experts u_miss(n) = n·8·(1 − h) (frequency-sorted placement: cold experts do not repeat);
      KV: DSA selected latents per token; draft cost (MTP: 0.98 GB per draft at these bpw; DFlash2 drafter ~2.47B params/step)
```
Consequence (HYPOTHESIS, arithmetic): the CPU bytes per accepted token grow by n/τ (×1.48 for MTP k=3, a = 0.745), so
speculation helps only the GPU-bound part; under uniform routing it makes GLM slower (Table B, uniform columns).
**Pipelined speculative windows** (`pp_pipewin`; Strata `--pipeline-windows` with 2 stages): stage 1 starts window w+1
while w is in stage 2, guessing that w is fully accepted. With π = P(all k accepted)·P(bonus guessed) ≈ a^(k+1) and
W windows in flight, a renewal argument gives
```
E[N] = 1/(1 − π);   tok/s = τ·E[N] / max((E[N] + W − 1)·t/W,  (E[N] + (W − 1)/2)·L_m·t_cpu)
```
(the second term: the CPU also computes misses of the discarded windows). The tool picks the best W ≤ 4.

**F. Other variants.**
- F1 quantisation inside the ≥ 4 bpw rule (ADR-012 D5): experts 4.25 bpw (IQ4_XS/MXFP4 class) raise residency to 0.70;
  dense Q6_K/Q5_K cut D by 18–32% (Table D). Quality UNKNOWN (KLD audit needed).
- F2 FP8 dense on Ada tensor cores: same bytes as 8-bit → **no decode gain**; prefill only (FP8 with FP32 accumulate
  ≈ 2× the FP16 rate, R4 §3.2 HYPOTHESIS).
- F3 asymmetric layer split: Σ_l bytes_l/B_g does not depend on the split → no decode gain on identical GPUs.
- F4 CPU joins the dense GEMVs in PP (row split GPU + CPU): t_dense ≈ (D + K)/(B_g + B_c) + 2·L·t_h ≈ 12.7 + 0.9 ms vs
  16.0 ms (base, 8 ch) → PP static ≈ 43 instead of 39 tok/s (HYPOTHESIS, hand arithmetic from these formulas; not in the tool).
- F5 out of policy, for the record: EXL3 3.05 bpw experts (the 125 GB checkpoint used by glm53-flash-offload) leave 97%
  of experts resident: TP 97 tok/s without and 170 with MTP k=3 (Table D, last row). It fails D5's literal ≥ 4 bpw rule;
  its quality vs a 4-bit GGUF is UNKNOWN. Listed because "≥ 4-bit quality" could be judged by KLD instead of bit count —
  the customer's call, not an engineering one.

---

## 4. Results (`tools/ceilings/arch_ceilings.py`, every number HYPOTHESIS)

Presets (all labelled in `PARAM_NOTES` of the tool): **roofline** = `tools/roofline.py` exactly; **base** = 736 GB/s × 0.75,
DRAM × 0.7, all-reduce 15 µs, handoff 10 µs, 20 µs per layer GPU fixed, 10 µs per CPU job, 300 µs per step, exact EP
imbalance, MTP head costs 2 all-reduces per draft under TP; **optimistic** = η_g 0.9, DRAM × 0.85, all-reduce 5 µs, handoff
5 µs, 5 µs per layer, 100 µs per step; **pessimistic** = η_g 0.55, DRAM × 0.55, all-reduce 50 µs, 60 µs per layer, 1 ms per
step. Hit models: **uniform** = resident share (roofline "auto"); **static** = the in-sample GLM profile curve (2.1) at the
resident share; **mid** = their midpoint (a placeholder for out-of-sample loss). GLM at 262K, UD-Q4_K_XL 4.89/8.04 bpw,
KV f16.

Regenerate every table: `python3 tools/ceilings/arch_ceilings.py --report r9` (≈ 1 s; Tables A–E as Markdown came from the
same functions, Appendix B).

**Table A. GLM-5.3-Flash, no speculation, tok/s ceiling**

| architecture | base 8ch uniform | base 8ch mid | base 8ch static | base 4ch uniform | base 4ch static | optimistic 8ch uniform | optimistic 8ch static | optimistic 4ch static | pessimistic 8ch uniform | pessimistic 8ch static |
|---|---|---|---|---|---|---|---|---|---|---|
| pp_serial | 26.1 | 29.3 | 33.6 | 19.1 | 30.2 | 32.6 | 41.9 | 37.6 | 18.7 | 23.5 |
| pp | 30.7 | 36.9 | 38.9 | 21.4 | 38.9 | 38.5 | 48.2 | 48.2 | 21.9 | 26.9 |
| ep | 29.2 | 34.8 | 43.6 | 20.6 | 38.1 | 37.6 | 56.9 | 49.5 | 20.4 | 29.1 |
| tp_repl | 40.7 | 53.9 | 81.4 | 24.4 | 59.4 | 54.0 | 116.5 | 81.2 | 27.0 | 46.9 |
| tp_owner | 41.3 | 53.6 | 77.9 | 26.1 | 61.8 | 54.7 | 110.0 | 84.4 | 27.2 | 45.2 |
| tp_owner_hostcpu | 42.2 | 54.8 | 79.5 | 26.2 | 62.1 | 55.1 | 110.8 | 83.8 | 27.8 | 46.1 |
| pp2_tp2 | 36.4 | 45.6 | 61.2 | 24.0 | 51.4 | 47.7 | 81.6 | 68.7 | 24.2 | 37.0 |

Hit values: uniform 0.607 (0.546 for tp_repl, whose replicated latent takes 10.8 GiB of cache), mid 0.757, static 0.907
(0.874 for tp_repl). Reading: at uniform routing every TP variant is DRAM-bound at ~41 (8 ch) / ~25 (4 ch); at the profile
hit TP roughly doubles. `tp_repl` vs `tp_owner` is a wash (the owner GPU reads 1.38 GB of DSA weights alone; the replicated
latent costs 6 points of residency); a hybrid (TP-split DSA projections, latent attention on one GPU) is not modelled.

**Table B. GLM-5.3-Flash with speculation, tok/s ceiling**

| speculation | pp base 8ch uniform | pp base 8ch static | pp base 8ch static reuse .246 | pp_pipewin base 8ch static | tp_owner base 8ch uniform | tp_owner base 8ch static | tp_owner base 4ch static | tp_owner optimistic 8ch static | pp_pipewin optimistic 8ch static |
|---|---|---|---|---|---|---|---|---|---|
| none | 30.7 | 38.9 | 38.9 | 38.9 | 41.3 | 77.9 | 61.8 | 110.0 | 48.2 |
| MTP k=1, a=0.745 | 36.2 | 48.8 | 51.8 | 83.6 (W=4) | 45.3 | 102.5 | 73.6 | 140.6 | 102.7 (W=4) |
| MTP k=2 | 36.3 | 50.5 | 55.2 | 73.3 (W=4) | 43.9 | 109.3 | 74.0 | 146.9 | 89.6 (W=4) |
| MTP k=3 | 34.7 | 49.3 | 54.7 | 64.1 (W=4) | 41.1 | 108.6 | 70.7 | 143.9 | 78.2 (W=4) |
| DFlash2 K=7, tau 3.68 (1K traces) | 29.5 | 45.1 | 50.9 | 50.0 (W=4) | 31.7 | 91.8 | 55.0 | 117.7 | 60.6 (W=4) |
| DFlash2 K=7, tau 3.48 (4K traces) | 27.9 | 42.7 | 48.1 | 46.6 (W=4) | 30.0 | 86.9 | 52.1 | 111.4 | 56.6 (W=4) |
| n-gram k=3, a=0.556 | 27.9 | 40.9 | 45.8 | 44.0 (W=4) | 31.5 | 86.4 | 55.1 | 114.5 | 53.7 (W=4) |

Inputs: MTP acceptance 0.745 (REPORTED, llama.cpp PR #27917 via ROOFLINE.md §3); DFlash2 per-position acceptance and
drafter size from `third_party/hf/glm-5.3-flash-dflash2-g/README.md` (REPORTED, B300, thinking traces, not agentic code);
n-gram 0.556 = base A `--spec 3` on a 16K docs+code prompt (REPORTED, R2 §3.4); "reuse .246" = union with 24.6%
previous-token reuse instead of independent routing. Block drafts (K = 7) lose to MTP k = 1–2 here because an 8-token
window reads ~7× the experts of one token. Agentic copy-heavy segments can have far higher n-gram acceptance (REPORTED:
SuffixDecoding up to ×5.3, R3 §8.4): try `--spec ngram:7:0.8`.

**Table C. Qwen3.8-Flash-Next UD-Q4_K_XL, all experts in VRAM, tok/s ceiling**

| architecture | roofline none | base none | base mtp:1:0.55 | base mtp:2:0.55 | optimistic none | optimistic mtp:2:0.55 | pessimistic none | pessimistic mtp:2:0.55 |
|---|---|---|---|---|---|---|---|---|
| pp | 84.3 | 76.2 | 88.5 | 84.9 | 97.9 | 106.2 | 49.8 | 57.9 |
| ep | 88.9 | 74.3 | 96.8 | 100.0 | 103.4 | 133.6 | 45.9 | 64.5 |
| tp_owner | 168.0 | 173.2 | 223.5 | 229.1 | 295.8 | 352.6 | 77.7 | 114.3 |
| pp2_tp2 | 113.5 | 115.9 | 142.8 | 141.8 | 173.7 | 197.1 | 59.6 | 80.5 |

Qwen's dense part (~10 bpw, 4.24 GiB/token) is PROVISIONAL (ROOFLINE.md §2, from file sizes); it dominates PP. Under TP
the step is 2.4 ms of dense + 0.7 ms of experts against 1.4–4.8 ms of collectives and 1.3–3.9 ms of fixed costs, so the
Qwen TP ceiling is set by engine latencies, not bandwidth.

**Table D. GLM quantisation variants (preset base, 8 channels)**

| experts / dense bpw | resident | pp uniform none | pp static none | pp static mtp:3:0.745 | tp_owner uniform none | tp_owner static none | tp_owner static mtp:3:0.745 |
|---|---|---|---|---|---|---|---|
| 4.89 / 8.04 | 0.607 | 30.7 | 38.9 | 49.3 | 41.3 | 77.9 | 108.6 |
| 4.25 / 8.04 | 0.699 | 35.9 | 40.1 | 52.3 | 51.5 | 90.9 | 148.3 |
| 4.89 / 6.5625 | 0.616 | 33.9 | 43.5 | 52.6 | 43.8 | 86.1 | 117.2 |
| 4.25 / 5.5 | 0.716 | 44.4 | 49.5 | 59.1 | 58.7 | 111.5 | 174.0 |
| 3.05 / 8.04 | 0.973 | 43.6 | 43.3 | 60.7 | 97.6 | 97.2 | 170.4 |

The last row is out of policy (F5). Even at 4.25/5.5 bpw (inside D5) a strict PP stays < 50 without speculation.

**Table E. Physics bounds — no architecture can exceed these (peak 736 GB/s per GPU, peak DDR4, no fixed costs)**

| model | hit | speculation | DRAM ch | DRAM bound | aggregate VRAM bound | single-GPU PP bound | GPU GB/step | DRAM GB/step |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | uniform (0.607) | none | 4 | 50.4 | 245.8 | 61.4 | 11.98 | 2.03 |
| GLM-5.3-Flash | uniform (0.607) | none | 8 | 100.9 | 245.8 | 61.4 | 11.98 | 2.03 |
| GLM-5.3-Flash | uniform (0.607) | mtp:3:0.745 | 4 | 35.7 | 335.8 | 83.9 | 23.79 | 7.79 |
| GLM-5.3-Flash | uniform (0.607) | mtp:3:0.745 | 8 | 71.3 | 335.8 | 83.9 | 23.79 | 7.79 |
| GLM-5.3-Flash | static (0.907) | none | 4 | 213.4 | 217.6 | 54.4 | 13.53 | 0.48 |
| GLM-5.3-Flash | static (0.907) | none | 8 | 426.8 | 217.6 | 54.4 | 13.53 | 0.48 |
| GLM-5.3-Flash | static (0.907) | mtp:3:0.745 | 4 | 144.8 | 269.3 | 67.3 | 29.66 | 1.92 |
| GLM-5.3-Flash | static (0.907) | mtp:3:0.745 | 8 | 289.5 | 269.3 | 67.3 | 29.66 | 1.92 |
| GLM-5.3-Flash | 1.0 (1.000) | none | 4 | inf | 210.1 | 52.5 | 14.01 | 0.00 |
| GLM-5.3-Flash | 1.0 (1.000) | none | 8 | inf | 210.1 | 52.5 | 14.01 | 0.00 |
| GLM-5.3-Flash | 1.0 (1.000) | mtp:3:0.745 | 4 | inf | 253.0 | 63.2 | 31.58 | 0.00 |
| GLM-5.3-Flash | 1.0 (1.000) | mtp:3:0.745 | 8 | inf | 253.0 | 63.2 | 31.58 | 0.00 |
| Qwen3.8-Flash-Next | 1.0 (1.000) | none | 4 | inf | 450.6 | 112.7 | 6.53 | 0.00 |
| Qwen3.8-Flash-Next | 1.0 (1.000) | none | 8 | inf | 450.6 | 112.7 | 6.53 | 0.00 |
| Qwen3.8-Flash-Next | 1.0 (1.000) | mtp:2:0.55 | 4 | inf | 481.9 | 120.5 | 11.32 | 0.00 |
| Qwen3.8-Flash-Next | 1.0 (1.000) | mtp:2:0.55 | 8 | inf | 481.9 | 120.5 | 11.32 | 0.00 |

"Single-GPU PP bound" covers the GPU side only; a PP's real bound is the lower of it and the DRAM bound. It applies to a
plain pipeline with speculation (one window in flight); pipelined windows (E) are not bounded by it.

---

## 5. Sensitivity and "what must be true"

**Table F. Thresholds (solver `--solve`, other parameters at preset base; the solver scans then bisects, so bounds that are
not monotone in the parameter are handled)**

| model / arch / hit / speculation / DRAM ch | parameter (others at preset base) | for 50 tok/s | for 80 tok/s | for 100 tok/s |
|---|---|---|---|---|
| glm / pp / static / none / 8 | vram_eff | >= 0.979 | unreachable | unreachable |
| glm / pp / static / mtp:2:0.745 / 8 | vram_eff | >= 0.742 | unreachable | unreachable |
| glm / pp_pipewin / static / mtp:1:0.745 / 8 | vram_eff | >= 0.442 | >= 0.716 | >= 0.903 |
| glm / tp_owner / static / none / 8 | vram_eff | >= 0.361 | >= 0.792 | unreachable |
| glm / tp_owner / static / none / 8 | allreduce_us | <= 94.6 | <= 11.3 | unreachable |
| glm / tp_owner / static / none / 8 | gpu_layer_fixed_us | <= 179 | <= 12.5 | unreachable |
| glm / tp_owner / static / mtp:2:0.745 / 8 | allreduce_us | <= 281 | <= 97.1 | <= 35.9 |
| glm / tp_owner / static / mtp:2:0.745 / 8 | gpu_layer_fixed_us | always | <= 184 | <= 61.8 |
| glm / tp_owner / static / mtp:2:0.745 / 8 | vram_eff | always | >= 0.385 | >= 0.59 |
| glm / tp_owner / static / none / 4 | allreduce_us | <= 57.4 | unreachable | unreachable |
| glm / tp_owner / static / mtp:2:0.745 / 4 | allreduce_us | <= 174 | unreachable | unreachable |
| glm / tp_owner / static / none / 8 | cpu_compute_gbs | >= 45.7 | unreachable | unreachable |
| glm / tp_owner / uniform / none / 8 | dram_eff | >= 0.995 | unreachable | unreachable |
| glm / tp_owner / static / mtp k=2 / 8 | MTP acceptance a | >= 0.0494 | >= 0.466 | >= 0.663 |
| glm / pp / fixed hit / none / 8 | VRAM hit share | unreachable | unreachable | unreachable |
| glm / pp / fixed hit / mtp:1:0.745 / 8 | VRAM hit share | >= 0.797 | unreachable | unreachable |
| glm / pp / fixed hit / mtp:2:0.745 / 8 | VRAM hit share | >= 0.779 | unreachable | unreachable |
| glm / tp_owner / fixed hit / none / 8 | VRAM hit share | >= 0.721 | >= 0.915 | unreachable |
| glm / tp_owner / fixed hit / mtp:1:0.745 / 8 | VRAM hit share | >= 0.663 | >= 0.843 | >= 0.902 |
| glm / tp_owner / fixed hit / mtp:2:0.745 / 8 | VRAM hit share | >= 0.678 | >= 0.837 | >= 0.889 |
| glm / pp_pipewin / fixed hit / mtp:1:0.745 / 8 | VRAM hit share | >= 0.622 | >= 0.807 | unreachable |
| glm / pp_pipewin / fixed hit / mtp:2:0.745 / 8 | VRAM hit share | >= 0.683 | unreachable | unreachable |
| glm / pp / fixed hit / mtp:2:0.745 / 4 | VRAM hit share | >= 0.889 | unreachable | unreachable |
| glm / tp_owner / fixed hit / none / 4 | VRAM hit share | >= 0.857 | >= 0.954 | unreachable |
| glm / tp_owner / fixed hit / mtp:1:0.745 / 4 | VRAM hit share | >= 0.831 | >= 0.92 | >= 0.949 |
| glm / tp_owner / fixed hit / mtp:2:0.745 / 4 | VRAM hit share | >= 0.839 | >= 0.918 | >= 0.944 |
| glm / pp_pipewin / fixed hit / mtp:1:0.745 / 4 | VRAM hit share | >= 0.81 | >= 0.916 | unreachable |

("unreachable" = not within the scanned range: vram_eff ≤ 1, hit ≤ 1, all-reduce ≥ 0 µs, layer cost ≥ 0. Rows with
"fixed hit" set hit as an input with additive misses; PP and EP bounds peak below hit 1.0 because of the overlap effect
documented in ROOFLINE.md rev. 3.)

**Table F2. Qwen3.8-Flash-Next (all in VRAM)**

| model / arch / speculation | parameter | for 100 tok/s | for 150 tok/s | for 200 tok/s |
|---|---|---|---|---|
| qwen / tp_owner / none | allreduce_us | <= 59 | <= 24.3 | <= 6.95 |
| qwen / tp_owner / none | gpu_layer_fixed_us | <= 108 | <= 38.6 | <= 3.9 |
| qwen / tp_owner / mtp:2:0.55 | allreduce_us | <= 119 | <= 57.6 | <= 26.8 |
| qwen / tp_owner / mtp:2:0.55 | gpu_layer_fixed_us | <= 229 | <= 105 | <= 43.5 |
| qwen / pp2_tp2 / none | allreduce_us | <= 29.3 | unreachable | unreachable |
| qwen / pp2_tp2 / mtp:2:0.55 | allreduce_us | <= 69.6 | <= 7.89 | unreachable |
| qwen / pp / none | vram_eff | unreachable | unreachable | unreachable |
| qwen / pp / mtp:2:0.55 | vram_eff | >= 0.894 | unreachable | unreachable |

**Table G. One-at-a-time sensitivity around preset base, 8 channels (tok/s per value, in the order listed)**

| parameter | values | glm/pp/static/none | glm/tp_owner/uniform/none | glm/tp_owner/static/none | glm/tp_owner/static/mtp:2:0.745 | qwen/pp/1.0/none | qwen/tp_owner/1.0/none |
|---|---|---|---|---|---|---|---|
| vram_eff | [0.55, 0.65, 0.75, 0.85, 0.95] | 29/34/39/44/49 | 38/40/41/43/44 | 66/72/78/83/87 | 97/104/109/114/118 | 57/67/76/85/94 | 145/160/173/185/195 |
| dram_channels | [4, 8] | 39/39 | 26/41 | 62/78 | 74/109 | 76/76 | 173/173 |
| dram_eff | [0.5, 0.6, 0.7, 0.8, 0.9] | 39/39/39/39/39 | 33/38/41/45/48 | 71/75/78/81/83 | 92/101/109/116/122 | 76/76/76/76/76 | 173/173/173/173/173 |
| cpu_compute_gbs | [40, 60, 90, 120, None] | 33/39/39/39/39 | 16/23/31/37/41 | 47/57/67/74/78 | 49/66/85/100/109 | 76/76/76/76/76 | 173/173/173/173/173 |
| allreduce_us | [5, 10, 15, 30, 60, 100] | 39/39/39/39/39/39 | 43/42/41/39/35/31 | 84/81/78/70/59/49 | 114/112/109/102/91/79 | 76/76/76/76/76/76 | 208/189/173/139/99/72 |
| gpu_layer_fixed_us | [0, 10, 20, 40, 80] | 40/40/39/38/35 | 43/42/41/40/37 | 84/81/78/73/64 | 114/112/109/105/96 | 82/79/76/71/62 | 208/189/173/149/116 |
| handoff_us | [2, 5, 10, 20, 40] | 39/39/39/39/39 | 42/42/41/40/38 | 80/79/78/75/71 | 113/111/109/106/99 | 76/76/76/76/76 | 173/173/173/173/173 |
| pcie_gbs | [12.5, 25.0] | 39/39 | 41/41 | 78/78 | 109/109 | 76/76 | 173/173 |
| step_fixed_us | [0, 300, 1000, 3000] | 39/39/38/35 | 42/41/40/37 | 80/78/74/64 | 111/109/106/97 | 78/76/72/63 | 183/173/154/118 |
| prefetch (cpu_compute_gbs 60) | recall,precision 0,1 / 0.3,0.9 / 0.6,0.9 / 0.6,0.5 / 0.9,0.9 | 39/38/38/38/38 | 23/30/40/31/40 | 57/67/79/69/79 | 66/83/108/86/109 | 76/76/76/76/76 | 173/173/173/173/173 |
| hit (fixed; overrides the column's hit) | 0.6/0.7/0.8/0.9/0.95/1.0 | 30/34/39/39/38/38 | 41/48/59/76/90/90 | 41/48/59/76/90/90 | 42/53/70/105/141/138 | 62/67/73/78/77/76 | 93/104/119/142/162/173 |
| MTP acceptance k=2 (overrides speculation; GLM columns only) | 0.5/0.6/0.7/0.745/0.8/0.9 | 38/43/48/51/54/60 | 33/37/42/44/47/52 | 83/93/104/109/116/129 | 83/93/104/109/116/129 | — | — |

Reading: PCIe bandwidth does not matter for decode in any of these architectures (it matters for prefill and for
prefetch with one link); the handoff latency matters little; DRAM channels and the hit share dominate GLM; collective
latency and fixed costs dominate Qwen under TP and GLM under TP once the hit is high. Prefetch with a predictor (recall
0.6, precision 0.9) is worth +40% to TP only when the CPU kernel is compute-bound (60 GB/s cap here); with precision 0.5 it
loses most of that.

---

## 6. Classification of the conclusions

**Physically impossible on this hardware** (at the 736 GB/s spec, DDR4-3200, UD-Q4_K_XL file sizes, 262K; Table E):
1. A strict layer pipeline without speculation above **52.5–54.4 tok/s** for GLM (61.4 only if the misses were free).
2. Above **50.4 tok/s (4 channels) / 100.9 (8 channels)** for GLM with any architecture **if routing is uniform** (hit =
   residency 0.607), no speculation; with MTP k=3 under independent routing the bound falls to 35.7 / 71.3, because the
   window's misses all cross DRAM.
3. Speculation cannot lower the DRAM bytes per accepted token below `u_miss(k+1)·e/τ`; with a frequency-sorted placement
   they rise by `(k+1)/τ` versus no speculation.
4. Qwen3.8 in a strict layer pipeline above **112.7 tok/s** (no speculation; 120.5 with MTP k=2, a = 0.55).
5. Under TP, 90 (GLM) / 96 (Qwen) collectives per token at the ~5–8 µs protocol floor cost 0.5–0.8 ms per token —
   negligible as a bandwidth limit, decisive as latency only above ~150 tok/s.

**No existing engine does it** (as read at the recorded SHAs; "not found" = not in the code I read, not proven absent
elsewhere):
1. TP of `glm5-next` dense/attention **together with CPU-computed expert misses** on AVX2 with 128 GB RAM: ExLlamaV3 has TP
   (no P2P) and a CPU split, but the CPU path is "Layer-split mode only"; KTransformers has TP + CPU experts for GLM but needs
   the AVX-512 FP8 kernel and ≥ 350 GB RAM (`doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md:13-16` @5d048e5), and its AVX2 path
   needs RAM ≥ the model weights (`AVX2-Tutorial.md:34`); llama.cpp / ik_llama have no TP for `glm5-next` (R2, VERIFIED there).
2. Pipelined speculative windows across more than 2 pipeline stages (Strata: 2 stages, REPORTED via R3 §2.5).
3. A host-staged all-reduce that also absorbs the CPU expert partials (`tp_owner_hostcpu`): a design, HYPOTHESIS.
4. ≥ 4 bpw EXL3 weights of GLM-5.3-Flash for ExLlamaV3: only a 3.05 bpw checkpoint is referenced (glm53-flash-offload
   README); a ≥ 4 bpw EXL3 conversion: UNKNOWN whether one exists.

**Unknown until measured** (and which experiment decides it, section 8):
1. **VRAM hit share on the customer's agentic-coding workload at long context** (static profile, LRU/CLOCK, global vs
   per-layer budgets). Moves GLM TP between ~41 and ~110.
2. DRAM channel count and the effective CPU expert-kernel throughput of the 7H43 (kernel-bound vs DRAM-bound).
3. 8–16 KB all-reduce latency over 4 GPUs without P2P (ExLlamaV3 native, NCCL SHM), in the VM if it stays a VM.
4. Per-layer fixed GPU cost and η_g of the chosen kernels on the 4080 Super (and the modded boards' memory clock).
5. MTP / DFlash2 / n-gram acceptance on agentic traces at 128K–262K (all inputs above come from short contexts or B300).
6. Out-of-sample stability of the routing profile across sessions (placement must not thrash).

---

## 7. The most promising architecture for GLM: concrete algorithm and what it takes

**`tp_owner` (+ CPU lane, + profile placement, + MTP k = 1–2)** — ceiling 78 (no speculation) / 102–109 (MTP) at the
profile hit with 8 channels (base preset); 62 / 74 with 4 channels. Decode step (pseudocode; HYPOTHESIS design, pieces
VERIFIED in the engines cited):
```
per token (or verification window of n = k+1 tokens), ranks r = 0..3, one process per GPU:
  x = embed(tokens)                                   # every rank, replicated (mHC streams too)
  for layer l in 0..44:
    if l is KDA:  partial_r = KDA_heads[r](x)         # 16 of 64 heads per rank, state split by head
    if l is DSA:  partial = (r == owner(l)) ? MLA_DSA(x) : 0     # whole layer on one GPU, latent KV only there
    x = mhc_merge(x, all_reduce(partial))             # host-staged: GPU writes 8-16 KB to mapped pinned memory,
                                                      #   CPU helper sums, GPU acquire-polls the result
    ids, w = router(x)                                # replicated on every rank (no broadcast)
    hot   = {e in ids : resident(l, e)}               # profile-placed, EP shards per rank
    cold  = ids - hot
    cpu_job(l, x, cold, w)  -> host                   # doorbell in mapped memory, CPU pool starts at once
    partial_r = sum_{e in hot ∩ shard_r} w_e·FFN_e(x) + shared_expert_slice_r(x)
    total = all_reduce(partial_r) + cpu_result(l)     # the CPU adds its partial inside the host-side reduction
    x = mhc_merge(x, total)
  logits = lm_head_slice_r(x) -> gather -> sample; MTP head = one more TP layer per draft
between generations: swap hottest CPU experts into coldest GPU slots (ExLlamaV3 _split_sweep_layer pattern)
```
What it takes on our server:
- **Start from ExLlamaV3 `151539c`** (MIT): TP native backend, MLA owner placement, EP MoE with replicated routing, MTP head
  for `glm5_next`, CPU worker with a job ring and stream-ordered flag waits — all VERIFIED in code. Missing: the CPU worker
  in TP mode. Patch: register the cold tail of each MoE layer with the CPU worker on the output rank; let the native
  backend's CPU reducer add the worker's partial before releasing the MoE all-reduce (one extra job type in
  `run_cpu_reduce_jobs`). Size: weeks, not months (HYPOTHESIS).
- Weights: ExLlamaV3 reads EXL3 only. A ≥ 4 bpw EXL3 of GLM-5.3-Flash needs a conversion (`convert.py`, GPU time) or must
  be found; its KLD against the reference must be measured (ADR-012 D5). GGUF UD-Q4_K_XL cannot be used.
- Numerics: the CPU path quantises activations to int8 (~0.9% RMS per call, VERIFIED comment); the comparable lane in
  glm53-flash-offload gives decode KL 0.005 (REPORTED). Bit-exactness across placements is lost; "correct math" must be
  defined as a KL/top-1 budget (`tools/quality/logprob_compare.py`).
- Risks: the native all-reduce latency is unmeasured; Python multi-process orchestration and thread placement swing decode
  up to 3× (ExLlamaV3 docs, section 2.2); the in-sample hit (0.907) may not hold on agentic prompts.
- Fallback inside llama.cpp: no TP for `glm5-next` (`llm_arch_supports_sm_tensor`, R2): adding KDA/DSA/mHC to `-sm tensor`
  is a larger project; the PP + CPU path of llama.cpp (`-ot`, `--moe-cache-mib`) stays at ≤ ~39 (Table A, pp).

**Second candidate without collectives: `pp_pipewin`** (Strata-style windows over 4 stages): 84 tok/s upper bound at the
profile hit with MTP k = 1 (base). Requires per-stage recurrent-state snapshots for rollback (34 KDA layers, R3 §8.6) and a
deep enough draft chain; the bound falls fast with acceptance (π = a²: 0.555 at a = 0.745). Implemented nowhere beyond 2
stages; its CPU lane serves all in-flight windows, so it collapses when misses are many (uniform routing: 48.9 with W = 2).

**Qwen3.8-Flash-Next**: TP is the lever (Table C) and ExLlamaV3 declares it for `qwen4_exp` (VERIFIED); the n-gram table
stays in RAM or is streamed (`modules/ngram_embedding.py`, VERIFIED). EXL3 weights ≥ 4 bpw: UNKNOWN. Whether 150 tok/s is
reachable is purely a question of all-reduce latency (≤ 24 µs) and per-layer cost (≤ 39 µs) (Table F2).

---

## 8. What to measure first (each item names the table cell it replaces)

| # | measurement (no GPU code needed except where stated) | replaces | stop / go |
|---|---|---|---|
| M1 | Real GLM-5.3-Flash routing traces on agentic-coding prompts at 32K–128K (the 3090 host can trace with ik_llama + `tools/g0`; proxies in `data/traces/real/REQUEST.json` are not GLM-5.3) → static, CLOCK/LRU and global-budget hit curves at 0.55–0.72 capacity, split train/test by session | "static" vs "uniform" in Tables A/B | test-set hit at 0.607 capacity < 0.75 → TP-for-GLM path not worth building (Table F: 80 needs ≥ 0.84 with MTP); ≥ 0.85 → build it |
| M2 | `dmidecode`, STREAM, and a CPU expert GEMV (Q4_K and EXL3) at 1/24/48 threads on the server | dram_channels, dram_eff, cpu_compute_gbs | 4 channels → GLM ≥ 80 needs hit ≥ 0.92 (Table F) |
| M3 | 8/16 KB all-reduce p50/p99 on 4 GPUs: ExLlamaV3 native backend microbenchmark and nccl-tests with `NCCL_P2P_DISABLE=1` | allreduce_us | > 60 µs → Qwen TP < 100, GLM TP needs MTP |
| M4 | ExLlamaV3 TP smoke on a model that fits (Qwen3.8 EXL3 or any TP-capable MoE), per-layer time with Nsight | gpu_layer_fixed_us, vram_eff | per-layer fixed > 40 µs → TP ceilings fall 10–20% |
| M5 | MTP / n-gram acceptance on long agentic traces | MTP acceptance | a < 0.6 → speculation adds < 20% (Table G) |

---

## 9. Reproducibility and tests

New files: `tools/ceilings/arch_ceilings.py` (presets `roofline`, `base`, `optimistic`, `pessimistic`; `--params`,
`--set`, `--sweep`, `--solve`, `--bounds`, `--report r9`, `--dump-params`, `--json`) and `tests/test_arch_ceilings.py`
(30 tests: exact consistency with `tools/roofline.py`, placement equality, monotonicity in every efficiency/latency/hit
parameter where it must hold, the documented non-monotone overlap peak, TP of one GPU = PP, physics bounds never
exceeded by any architecture/preset, zero acceptance never helps, prefetch neutrality and DRAM bound, pipelined-window
limits, solver on non-monotone bounds, CLI). Mutation check on the final code (seven mutants in a scratch copy, loaded in
place of the module): unmutated `failures=0 errors=0 of 30`; TP dense not divided 4 failures; free CPU path 8; no union
overlap 2; overlap as min 8; acceptance ignored 3; all-reduce not charged 6; pipelined windows without the bubble 3.

Literal output (2026-10-10, final code):
```
$ python3 -m unittest tests.test_arch_ceilings tests.test_roofline 2>&1 | tail -4
----------------------------------------------------------------------
Ran 56 tests in 1.987s

OK
$ git log --oneline -n 5
8981351 [infra-ops] ci: workflow traces-real runs scripts/traces/run_real_traces.sh on a runner with huggingface.co access and commits data/traces/real/<model>/ back to the claude/* branch
6715f40 [perf-bench] scripts/traces: real-model routing traces - agent-style corpus from repository files, chat-template rendering, HF fetch with sha256, per-model driver
5100adb [parity-qa] tools/g0: expert-prediction probe (router input, logits, selected experts) and predict_eval with three self-checks
b6f4fe3 [infra-ops] evidence: offline engine kit packed, verified and used for an offline ExLlamaV3 sm_89 build
bd2dcb0 [infra-ops] engines: offline kit for a server without internet (pack/verify/install-python, --offline-kit)
$ git status --short
?? docs/reassessment/research/R10-discovery-sweep.md
?? docs/reassessment/research/R5-moe-offload-engines.md
?? docs/reassessment/research/R6-local-engines.md
?? docs/reassessment/research/R7-serving-kv-longctx.md
?? docs/reassessment/research/R8-memory-tiers-expert-scheduling.md
?? docs/reassessment/research/R9-adversarial-ceilings.md
?? docs/reassessment/research/matrix/
?? tests/test_arch_ceilings.py
?? tools/ceilings/
```
Mine: `R9-adversarial-ceilings.md`, `matrix/R9.json` (inside `matrix/`, next to siblings' R5/R8), `tests/test_arch_ceilings.py`,
`tools/ceilings/`. The other untracked files belong to sibling research tasks. Nothing was committed, per the brief.

---

## 10. Limitations of this analysis

- Bandwidth-and-latency model: no compute terms for decode (DSA top-k over 65K pooled keys per layer, KDA recurrence,
  sampling), no dequantisation cost on the GPU beyond η_g, expectations inside max() (stragglers ignored → optimistic).
- Recurrent-state handling for speculation: one state read/write per window (as `tools/roofline.py`); storing per-position
  states for rollback would add up to (k+1)·153 MB per step (34 KDA layers × 4.5 MB) — not charged.
- KV size of real engines (llama.cpp stores more than the model, ROOFLINE.md §5) and activation/workspace VRAM are in the
  2 GiB/GPU reserve only.
- The static hit curve is in-sample, from one undocumented workload of ~9K tokens on an EXL3 3.05 bpw quantisation; the
  out-of-sample loss is UNKNOWN ("mid" is a placeholder, not an estimate).
- DFlash2 and MTP acceptances are from other hardware/contexts; acceptance at 262K agentic sessions is UNKNOWN.
- Fixed-cost presets are HYPOTHESIS ranges, not measurements; Table F gives the thresholds so a measurement can be placed
  directly against them.

---

## Appendix A. Scripts behind section 2.1 (run against the cloned file; outputs above)

```python
# static_hit.py <stats_own_dec.json>: frequency-sorted static hit curve, per layer and global budget
import json, sys, math, statistics as st
d = json.load(open(sys.argv[1])); keys = sorted(d, key=lambda k: int(k.split('.layers.')[1].split('.')[0]))
for c in (0.107, 0.546, 0.607, 0.700):
    hs = [sum(sorted(d[k], reverse=True)[:int(round(c * 288))]) / sum(d[k]) for k in keys]
    allv = sorted((x / sum(d[k]) for k in keys for x in d[k]), reverse=True)
    print(c, sum(hs) / len(hs), min(hs), max(hs), sum(allv[:int(round(c * len(allv)))]) / len(keys))
cv = [st.pstdev(d[k]) / st.mean(d[k]) for k in keys]; N = 74000; p = 1 / 288
print("CV", sum(cv) / len(cv), "noise-only CV", math.sqrt((1 - p) / (N * p)))
# union_skew.py: per-token inclusion q_e = count_e / tokens; union(n) = sum_e 1 - (1 - q_e)^n;
# missed union with the top-c share resident = same sum over non-resident experts; static reuse = sum_e q_e^2 / 8
```
Output of the union script (VERIFIED, run 2026-10-10):
```
n= 4: union uniform 30.69 | union static-skew mean 29.15 (ratio 0.950)
n= 8: union uniform 58.11 | union static-skew mean 52.33 (ratio 0.900)
resident 0.607 n=1: distinct MISSED experts per layer 0.74 (uniform: 3.14)
resident 0.607 n=4: distinct MISSED experts per layer 2.91 (uniform: 12.06)
resident 0.607 n=8: distinct MISSED experts per layer 5.68 (uniform: 22.84)
static-model expected reuse from previous token: mean 0.064 (uniform 0.0278)
```
The 21-point curves embedded in the tool (`GLM_STATIC_CURVE`, `GLM_STATIC_CURVE_GLOBAL`) come from the same computation at
capacities 0, 0.05, …, 1.0.

## Appendix B. How the Markdown tables were produced

Tables A–E: functions `step()` / `physics_bounds()` of the tool with the presets and arguments named in the column
headers; Table F/F2: `solve()`; Table G: section T7 of `--report r9`. No number in sections 4–5 was typed by hand.

---

## Sources

Repositories (cloned into the review's scratch directory; `git log -1` recorded):

| repo | URL | SHA | commit date | license | used for |
|---|---|---|---|---|---|
| ExLlamaV3 | https://github.com/turboderp-org/exllamav3 | 151539c77abc7ab7425d30da7a4e8e3c5c154e7b | 2026-10-07 18:47:56 +0200 | MIT | TP native backend, MLA owner, EP, CPU MoE split, docs |
| glm53-flash-offload | https://github.com/0xSero/glm53-flash-offload | 6769b27652a18487758caac45a2b6999dc0a4184 | 2026-10-09 09:03:52 -0400 | MIT | GLM routing counts, efficiency anchors, CPU kernel log |
| strata-glm-3090 | https://github.com/Grigory-Rylov/strata-glm-3090 | 794cd69ae270389c67ed201b855b14aafb5eb875 | 2026-10-03 20:11:15 +0300 | MIT | routing profile (24.6% reuse, top-1310 42–64%) |
| KTransformers | https://github.com/kvcache-ai/ktransformers | 5d048e516507e1603e0ed19645611b4501476a50 | 2026-10-10 16:16:45 +0800 | Apache-2.0 | cudaLaunchHostFunc CPU MoE, GLM/AVX2 requirements |
| this repo | (worktree `claude/research-landscape`) | 8981351d4248102ec286c92cb59cc9e593fd60d5 | 2026-10-10 18:41:23 +0000 | — | tools/roofline.py, specs, ROOFLINE.md, ADR-012, R2–R4, DFlash2 card |

Web (searches 2026-10-10; WebFetch to deploy.helix.ml failed with ENOTFOUND, so those numbers are snippet-level REPORTED):
- RTX 4080 Super memory spec: https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4080-Super-Benchmarks-and-Specs.806212.0.html ,
  https://www.pugetsystems.com/parts/Video-Card/NVIDIA-GeForce-RTX-4080-SUPER-16GB-Founders-Edition-15351 ,
  https://www.igorslab.de/en/the-nvidia-rtx-4xxx-super-portfolio-is-complete-faster-ram-a-little-more-clock-speed-and-speculation-about-prices/
- NCCL without P2P on 2× 4090 (anecdotal 8-byte all-reduce ≈ 6.7 µs): https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366/10 ;
  SHM vs patched P2P on 2× 5090 (+2.7% end-to-end): https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870
- PCIe polling/flag latency (older reports, 2–10 µs vs 56 µs hiccups): https://forums.developer.nvidia.com/t/massive-hiccups-when-transferring-flags-back-to-host-challenging-the-commonly-quoted-2-10us-latency/20503 ,
  https://arxiv.org/pdf/2404.19381 (abstract-level)
- SGLang GLM-5.3-Flash on 4× RTX PRO 6000, 63.7 / 82.8 tok/s single request: https://deploy.helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1 (snippet only)
- KTransformers GLM-5.3-Flash discussion (18.3 tok/s single stream, hardware not stated): https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/38 (snippet only)
- Not found: any measurement of an 8–16 KB all-reduce latency over 4 GeForce GPUs without P2P; any ExLlamaV3 TP benchmark
  without P2P (searches: "exllamav3 tensor parallel native backend no P2P all-reduce latency benchmark", "NCCL
  all_reduce_perf small message latency PCIe without P2P 4 GPUs").
