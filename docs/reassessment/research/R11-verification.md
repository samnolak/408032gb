# R11. Independent verification of R5–R10 (deep review, 2026-10-10)

Role: independent verification team. Scope: the claims in `R5-moe-offload-engines.md`, `R6-local-engines.md`,
`R7-serving-kv-longctx.md`, `R8-memory-tiers-expert-scheduling.md`, `R9-adversarial-ceilings.md`,
`R10-discovery-sweep.md` that would most change the architecture decision if wrong. Every verdict below comes from a fresh
clone made for this review (scratch dir `.../scratchpad/research/r11`, SHAs and commit dates in "Sources"), from a command
run here, or from a web search result quoted as such. Nothing was built, nothing ran on a GPU, nothing was committed.

Verdicts: **CONFIRMED** (seen at `repo@sha:path:line` or URL, quote given) · **CORRECTED** (what is actually true, with
source) · **DISPROVED** · **UNVERIFIABLE** (what was tried). "Label too strong" is noted where a report wrote VERIFIED for
something that is REPORTED or HYPOTHESIS. Mapping to AGENTS.md: CONFIRMED = CONFIRMED; CORRECTED facts carry their own
CONFIRMED source; derived arithmetic here is PROVISIONAL.

Known error, re-checked and not counted as new: R10 §0 item 2 "also runs without P2P" for halt95 — the 155.5 tok/s run used
the aikitoria P2P patch (`halt95/qwen38-flash-next-3090s@28cbb72:docs/reference.md:289`); see C07.

---

## Summary table

| id | report:section | claim (one line) | verdict | primary source |
|---|---|---|---|---|
| C01 | R9 §2.1 | strata-glm-3090: 24.6% previous-token reuse, top-1310 pairs carry 42–64% of routed mass; uniform would be 2.78% / 10.8% | CONFIRMED | `Grigory-Rylov/strata-glm-3090@794cd69:PROBLEMS.md:111-113`, `tools/glm_profile.py:10-12,113-121` |
| C02 | R9 §2.1, App. A | glm-flash-lite routing counts give static hit 0.907 at 60.7% residency (in-sample), CV 1.10 vs 0.062 noise | CONFIRMED (reproduced to 3 decimals) | `sybil-solutions/glm-flash-lite@6769b27:data/stats_own_dec.json` (sha256 `79bff1e3…`) |
| C03 | R9 §0 C2/C3, Tables A/B "static" | the in-sample 0.907 is the hit to plan TP ceilings (78/110 tok/s) on | CORRECTED: the only published **held-out** GLM-5.3-Flash placement scores are 0.701 (code) – 0.773 (agentic) at 55.6% residency vs 0.880 in-sample at the same residency; R9's "mid" column, not "static", is the evidence-backed case | `anoane/glm53-flash-exl3-recipe@116e6a2:moe_profiles/README.md:13,26-27` |
| C04 | R10 §0 item 8, §2.4 | "LRU on GLM routing: 9% hit vs 5.6% random"; GLM routing "much flatter" | CORRECTED: the 9% is a 16-slot (5.6%-capacity) LRU; the same source says routing is "strongly skewed within each layer — the hottest ~36% … serve 75%" and withdraws its own "near-uniform" conclusion as an aggregation error | `pctablet505/glm53-flash-single-gpu@865edb6:README.md:157-158`, `LAB-NOTEBOOK.md:159-176` |
| C05 | R10 §0 item 8 | "MTP with offload lost in three independent GLM runs" | CORRECTED: pctablet505 ships MTP-2 as a gain (8.7 → 10.6–12.1 tok/s), only depth 4 lost; anoane: 0.63× short context but **1.08× at 256K** | pctablet505 `README.md:176-177,662`; anoane `README.md:159-160` |
| C06 | R5 §0.5 | lt-Strata ideal per-layer LRU: 41% @10%, 58% @20%, 76% @40%, 84% @60%, ceiling 88.5%; 23% adjacent reuse | CONFIRMED; R5's extrapolation "hit ≈ 0.9–0.95 at ~60% coverage" exceeds the curve's own 0.844 at 0.60 and its 0.885 ceiling — HYPOTHESIS, not derived from the data | `lighttransport/Strata@e4dbb18:tools/sim/data/lru_curve.json`, `docs/GLM_DECODE_RESEARCH.md:61-68` |
| C07 | R10 §2.1, §3.1 Q1/Q2 | halt95: Qwen3.8-FN 155.5 tok/s median at 261K on 4×3090 (INT4 experts, MTP K=3); "also runs without P2P" | numbers CONFIRMED; with P2P (aikitoria patch); the no-P2P data point is a 24-token prompt on another host (x8, 2 NUMA nodes), 127–141 tok/s incl. TTFT — summary line DISPROVED (known error); host is a Proxmox **LXC**, not a VM | `halt95/qwen38-flash-next-3090s@28cbb72:benchmarks/2026-10-06/BENCH-CARD.md:3,115-121`, `docs/reference.md:99-105,288-290` |
| C08 | R10 §2.4, §3.1 Q3 | DominikBucko: AVX2 INT4 CPU expert tier in vLLM, our CPU class (Zen 3, 8-ch DDR4-3200): 103.1 tok/s after 260K | CORRECTED: the 103.1 profile (128 GB) does **not** compute experts on the CPU — "Decode, cold experts: Copied into the GPU cache on a miss" — and "needs working bidirectional CUDA P2P"; the CPU-expert profile (64 GB, no P2P) gives 84.2–88.8 tok/s at 131K | `DominikBucko/qwen38-flash-next-2x3090@939905b:docs/memory.md:52-58`, `configs/fast-256k.env:1-4`, `README.md:88-98,306-308` |
| C09 | R10 §2.5, §3.2 G1 | anoane: GLM-5.3-Flash EXL3 4.05 bpw, ExLlamaV3 CPU tier + profile, 39.7–40.2 prose / 33.7–34.0 code tok/s at 262,144, no speculation; `step_ms ≈ 10.8 + 0.77·cold%`; no LICENSE | CONFIRMED (REPORTED by author; 1×RTX PRO 6000 + DDR5 2-ch; CPU kernel at 53 GB/s is AVX-512) | anoane `README.md:3-6,23-36,53`; repo has no LICENSE file (`ls`) |
| C10 | R9 §6 "no engine" 4, §7 | "≥4 bpw EXL3 of GLM-5.3-Flash: UNKNOWN whether one exists; only 3.05 bpw referenced" | CORRECTED: `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (~165 GB) is used end-to-end by anoane (REPORTED; HF not reachable here) | anoane `README.md:3,6,23` |
| C11 | R6 §4.2, R9 §2.2 | ExLlamaV3: `supports_tp` for glm5_next / qwen4_exp / mimo_v2; native TP backend without NCCL/P2P (CPU-reduced all-reduce over mapped pinned memory); MLA/DSA layer runs whole on one GPU; CPU MoE split/offload layer-split only (also on dev) | CONFIRMED | `turboderp-org/exllamav3@151539c`: `architecture/glm5_next.py:350`, `qwen4_exp.py:311`, `mimo_v2.py:389`, `model/model_tp_backend.py:221-246,458-472`, `modules/mla_attn.py:1195-1244`, `model_init.py:212-217`; dev `60c0162:model_init.py:216` |
| C12 | R6 §0 item 5, §4.2 | 6×RTX 3090 PCIe Gen3, Qwen3.8-FN 6.05 bpw, native TP: decode 76 tok/s | CONFIRMED; context was "three 32K cold prompts", temperature 1.0; it is the only measurement of the backend R9 recommends and sits at ~0.45× R9's Qwen `tp_owner` ceiling (168–173) — R9 did not use it | exllamav3 commit `8114c49` message |
| C13 | R6 §4.3 | ExLlamaV3 AVX2 trellis CPU kernel 15.6–21 GB/s on a 20-core Xeon; GLM 3.05 bpw 232/288 experts on CPU 4.3–4.9 → 7.9 tok/s | CONFIRMED; the 4.3–4.9 is the AVX2 tier, 7.9 the new AVX-512BW tier (our CPU has no AVX-512) | exllamav3 commit `94c05ba` message lines 20-25 |
| C14 | R6 §2.2–2.3 | llama.cpp: internal no-P2P all-reduce only for 2 GPUs; GLM5_NEXT excluded from `-sm tensor`; MoE cache syncs per layer, throws under `-sm tensor`, disables pipeline parallelism; mmap → pageable host experts | CONFIRMED (GLM5_NEXT exclusion is at line 1235, not 1240) | `ggml-org/llama.cpp@abee0c8`: `ggml/src/ggml-cuda/allreduce.cu:400-406`, `src/llama-arch.cpp:1235`, `src/llama-moe-cache.cpp:234-236,549-550`, `src/llama-context.cpp:434-440`, `src/llama-model-loader.cpp:1269-1276` |
| C15 | R5 §T4, §6 | "`--moe-cache-mib` needs every expert in host RAM → does not fit 128 GB for GLM UD-Q4_K_XL" | CORRECTED: the cache only covers layers whose experts are **all** host weights; other layers can keep all experts on the GPU, so exclusivity is available per layer; per-layer arithmetic gives the same ~62% VRAM share with ~107 GiB of host experts (PROVISIONAL) | `llama.cpp@abee0c8:src/llama-moe-cache.cpp:241-266` |
| C16 | R7 §0 item 1, item 4 | stock vLLM: GLM indexer requires DeepGEMM; DeepGEMM only SM90/100/120; custom all-reduce off on >2 PCIe GPUs | CONFIRMED | `vllm-project/vllm@187a0eb`: `vllm/models/glm5next/nvidia/sparse_indexer.py:699-702`, `vllm/platforms/cuda.py:688-695,136-153`, `custom_all_reduce.py:270-276` |
| C17 | R7 §0 item 3 | "vLLM no longer has a GGUF loader — our UD-Q4_K_XL files cannot be served by vLLM at all" | CORRECTED: GGUF moved out of tree to `vllm-project/vllm-gguf-plugin` (docs still in tree); the plugin has no glm5next/qwen4exp mapping found by grep, so the practical conclusion (our GGUFs: no vLLM path today) stands | vllm `@187a0eb:docs/features/quantization/gguf.md:6-10`; `vllm-gguf-plugin@e2b8ad5` grep |
| C18 | R5 §2.6, R10 §2.3 | yhfgyyf vLLM fork: SM89 Triton indexer path; GLM-5.3-Flash FP8 on 8×4090 48 GB, TP8, `NCCL_P2P_DISABLE=1`, 262K | CONFIRMED (launch config from issue #74; no sm_89 tok/s published); the fork also opens the SM120 sparse-MLA backend to (8,9) | `yhfgyyf/vllm-deepseek-v4-sm89@3f3bbae:vllm/model_executor/layers/sparse_attn_indexer.py:61-66`, `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:199-200`, `README.md:584-615` |
| C19 | R7 §0 item 2; R5 §0.10 | SGLang-KT fork: SM86/SM89 Triton GLM indexer; GLM CPU experts require `--kt-method FP8`. R5: "no GLM-5.3 INT4 path documented (UNKNOWN)" | CONFIRMED; R5's UNKNOWN is resolved: non-FP8 methods raise `NotImplementedError` for GLM CPU experts in that fork | `kvcache-ai/sglang@18c0de7:python/sglang/srt/models/glm5_next_moe.py:178-185`, `layers/attention/nsa/glm5_next_indexer_triton.py:21-24` |
| C20 | R5 §0.1, §2.2, §T1 | Maya: Q4_K/Q5_K/Q6_K GPU expert path (slow per-row form); 4×T4 head/tail spec decode 15.0 → 22.6; activations hop through pinned host (no P2P); other quants "experimental" | CONFIRMED (README now quotes 14.1 → 21.0 for the same feature) | `mw00/project-maya@10b7b07:src/kernels/cuda/glm_fast.cu:925-932`, `CHANGELOG.md:270-271`, `README.md:147-148,248`, `src/core/glm_fast_path.cu:3787-3799` |
| C21 | R5 §0.2 | Strata 0.1.42 makes lossy `STRATA_ROUTE_TAIL_SKIP=7` default on CUDA serve | CONFIRMED; not applied when every expert is VRAM-resident, with helper/peer GPUs, or `--batch` | `Niko1221/Strata@61b3fb5:src/core/expert_source.cpp:3592-3612`, `src/program/generate.cpp:8066-8085`, `docs/DETAILS.md:1560-1571` |
| C22 | R5 §0.4 | Strata helper-GPU expert caches: "REPORTED 2×4090 +63% mixed / +132% code" | CORRECTED: those gains are `--remote-expert-opt` **over the plain helper path**, not helpers vs one GPU; 3–4 cards "have not been validated" | Strata `@61b3fb5:docs/SECOND_GPU.md:93-96,126` |
| C23 | R10 §2.6, §4 C | freetoken-rdna3 host-memory all-reduce "~12 µs per decode message inside a graph" | CORRECTED: ~12 µs is RCCL's proxy path being replaced; the host all-reduce takes 6 µs, +2.5–4.8% decode, two ranks, ROCm | `Cedriceuh/freetoken-rdna3@a4469ae:docs/rdna3/journey.md:103-104`, `docs/rdna3/options.md:33` |
| C24 | R8 §0 item 3, §7.4 | "PCIe prefetch cannot pay structurally: link lands ~0.7 experts/layer vs ~3.1 misses/layer" | CORRECTED (premise): 0.72 experts/layer is right, but 3.1 misses/layer is the uniform-routing value; with the measured skew it is 0.74 (in-sample) to ~2.0 (held-out ≈0.75), so the gap is 1–3×, not 4×. Field results (Strata Foresight 0.91–1.00×) still stand | arithmetic here; C02, C03; Strata `docs/DETAILS.md:1650-1658` |
| C25 | R10 §8 (licence column), R1 E15 | Qwen3.8-Flash-Next checkpoints carry the Qwen Community License 1.0 | CONFIRMED text (copy in halt95): clause 2 requires a separate Qwen licence for commercial use by a licensee that "conducts a Model as a Service or AI Work Assistant business"; internal use exempt if not made available to third parties. R10 did not analyse it | `halt95@28cbb72:LICENSE.qwen-community-1.0:7-11` |

Additional checks (shorter):

| id | report:section | claim | verdict | source |
|---|---|---|---|---|
| A1 | R6 §2.6 | PR #30277 (Gerganov) mirrors KDA/MLA attention under `-sm tensor`; PR #27861 hybrid is `n_tokens == 1` only | CONFIRMED | llama.cpp PR heads `bcb1101` (message), `bccbacd` (diff vs `4e97ac8`, `src/llama-graph.cpp` hunk) |
| A2 | R5 §2.5, R8 §7.1 | KTransformers GLM tutorial: FP8 306 GiB, ≥350 GB RAM, AVX-512 FP8; AVX2 needs RAM ≥ weights; placement table 52.7 → 70.2 / 76.2 → 81.2 | CONFIRMED; the placement table was measured with a **Xeon Gold 6454S** (AVX-512/AMX), not stated in R5/R8 | `ktransformers@5d048e5:doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md:12-16`, `AVX2-Tutorial.md:31-34`, `experts-sched-Tutorial.md:176-190` |
| A3 | R10 §2.9 | P2P in Proxmox Q35/OVMF guest (3090, 610.57.04-p2p-v3): +21.9/+16.3% prefill, +6.3/+7.1% decode; small-BAR 4090 KVM fix +2.5–2.6% at C1 | CONFIRMED; the +2.5% is for a dense 27B FP8 model vs a prior W8A8 deployment | `noonghunna/club-3090@c8b6a73:docs/PCIE_P2P.md:481-545`; `xiaoyanzi191/rtx4090-small-bar-p2p@dbd0cdf:README.md:5,23,26`, `docs/RESULTS.md:9` |
| A4 | R6 §5, §3 | fastllm cache comment + "No publishable … GLM-5.3-Flash throughput"; DwarfStar GLM streaming only Metal/ROCm; ik_llama `-sm graph` excludes glm5next/qwen4exp/deepseek4, MiMo2 capped at 4 GPUs | CONFIRMED | `ztxz16/fastllm@a2ff521:src/devices/cuda/moe/fastllm-moe-cache.cu:1-9`, `docs/glm5_en.md:59`; `antirez/ds4@fc80bd6:ds4.c:455-462`; `ikawrakow/ik_llama.cpp@89bba35:src/llama.cpp:4216-4249,4575-4581` |
| A5 | R8 §0 item 2 | Strata 2×3080 Qwen UD-Q4_K_XL 86 tok/s with the PLE table read `--ple-io direct`, "measured speed-neutral" | CORRECTED: "speed neutral" was measured on **prefill reads at 25K**, on two different binaries; decode 80.7–90.6 tok/s at 4K–127K (262K not measured); fork build `strata-w7` | Strata `@61b3fb5:bench/results/2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool/README.md:10-11,29-38,106` |
| A6 | R8 §0 item 6, §4.1 | "io_uring: not used by any of them (grep returned nothing)" | CORRECTED (minor): KTransformers' archived `balance_serve/kvc2` KV store has an optional io_uring engine | `ktransformers@5d048e5:archive/csrc/balance_serve/kvc2/src/async_store.cpp:140-188`, `CMakeLists.txt:38-41` |
| A7 | R9 §2.3 | SGLang GLM-5.3-Flash NVFP4 on 4×RTX PRO 6000: 63.7 tok/s single request, 82.8 with EAGLE | CONFIRMED (search snippet); the page calls the day-zero stack "not ready for production", so R9's reading "fixed costs and collectives dominate" is HYPOTHESIS with an equally good alternative (immature stack) | WebSearch result for deploy.helix.ml (page not loadable) |
| A8 | R5 §2.3, R9 §2.3 (new fact) | glm-flash-lite "exclusive tiers" | CORRECTED/added: its CPU-lane mode `fast` keeps **two** RAM copies of all experts (234 GB process RAM); the 119 GB `exact` mode has no CPU lane. Same host measured 136 GB/s DRAM read at 22 threads and ~3.6 GB/s per core for its AVX2 lane (EPYC 7443P, Zen 3, 8-ch) | `glm-flash-lite@6769b27:docs/reference.md:123-126` |

Totals (primary verdict per row): 33 claims checked — **CONFIRMED 20** (C01, C02, C06, C07, C09, C11–C14, C16, C18–C21,
C25, A1–A4, A7), **CORRECTED 13** (C03, C04, C05, C08, C10, C15, C17, C22, C23, C24, A5, A6, A8), **DISPROVED 0 new** (the
known halt95 "without P2P" error is re-confirmed inside C07; R5's UNKNOWN about a GLM INT4 KT path is answered "not supported"
inside C19), **UNVERIFIABLE**: sub-claims listed in "Could not verify". Label notes: C06 (R5 extrapolation is HYPOTHESIS, not
data), C17 (R7 wording too strong), C18 (sm_89 GLM run is REPORTED from a README pointing to an issue, not VERIFIED).

---

## Per-claim detail

### C01 — GLM routing concentration in strata-glm-3090 (R9 §2.1) — CONFIRMED

`PROBLEMS.md:110-113@794cd69` (section "2026-10-03: real weights - the routing spread is the decode ceiling"):
> "Routing profile from 16 real segments (code/docs/prose, glm_profile.py): expert reuse from the previous token 24.6% (the
> synthetic pack: ~60%), per-layer top-1310 covers 42-64% of the routed mass."

How the numbers are made (`tools/glm_profile.py`): `stats` prints "the routed mass the top-N (layer, expert) pairs cover
(N = the VRAM and VRAM + RAM tiers)" (lines 10-12, default `--vram 1310`, line 5) and "per layer, share of routed mass in its
top 31 / 72 / 144 experts" (lines 113-116). 1310 pairs = 10.8% of 42×288 = 12,096; 31/288 = 10.8%. So "top-1310 / 42-64%"
is the per-layer top-31 share range — R9's reading "the top 10.8% carry 42–64%" is correct. Reuse is computed as
`(onehot[1:] & onehot[:-1]).sum() / ((T-1)*K)` (lines 87-91); under independent uniform top-8-of-288 the expectation is
8/288 = 2.78% (arithmetic). Caveats R9 did not state: the routes come from `--max-new 0` runs (prefill over corpus text,
`run()` lines 51-53), 16 segments of 4,096 tokens, NVFP4 weights. Teacher-forced routing over text equals decode routing on
the same tokens for a causal model, so this is not a flaw, but the sample is ~65K tokens.

### C02 — static hit 0.907 from glm-flash-lite counts (R9 §2.1) — CONFIRMED

Script `r11_scripts/skew.py` (run with `python3 -I`) on `data/stats_own_dec.json` (sha256 printed
`79bff1e39f9c887b6f12b242b2ee5d685630f418d5a84e318c082ee2fcdcf094`, same prefix as R9):
```
shape (42, 288) picks/layer min/max 74000.0 74000.0
cap 0.107: per-layer mean 0.371 min 0.173 max 0.487 global 0.378
cap 0.546: per-layer mean 0.874 min 0.666 max 0.953 global 0.887
cap 0.607: per-layer mean 0.907 min 0.723 max 0.971 global 0.923
cap 0.7: per-layer mean 0.946 min 0.803 max 0.987 global 0.962
CV mean 1.097013738616377 min 0.31821531179428814 max 1.4607454714492139
noise CV 0.062276627866145566
static reuse 0.06402608876830498
```
Every R9 number reproduces. Workload of the counts is undocumented; the file is used to warm-start the repo's cache
(`docs/how-it-works.md:53-54`). R9 labels it in-sample — correct.

### C03 — which hit to plan on (R9 Tables A/B, C2/C3) — CORRECTED

anoane's profile catalogue scores static placements on **held-out windows** for GLM-5.3-Flash 4.05 bpw at 160 of 288
experts resident (55.6%) (`moe_profiles/README.md:11-16`):
> "`held-out` is capture on windows the ranking never saw … `uniform` (55.6% at 160 resident of 288) is what random placement scores."

Rows: `code_long` 70.1% (line 26), `agentic` 77.3%, headroom 77% (line 27); systems/security profiles 73.5–86.9%. From the
headroom definition, the agentic in-sample oracle is 55.6 + (77.3 − 55.6)/0.77 ≈ 83.8% (PROVISIONAL arithmetic). For
comparison at the same 0.556 capacity: glm-flash-lite in-sample per-layer static 0.880 (this review, same script); lt-Strata
ideal LRU 0.828 (linear interpolation of `lru_curve.json`). pctablet505 adds that re-profiling on mixed traffic at T>0 was
"Worse on everything: 13.02 vs 15.77 prose. Sampling at T>0 flattens routing" (`README.md:661`).

Consequence: R9's direction (routing is far from uniform; the uniform "auto" case understates hit) is CONFIRMED, but the
number to plan with at ~0.6 residency is ≈ 0.75–0.85, i.e. R9's "mid" (0.757) column, not "static" (0.907). R9 Table A
`tp_owner` base 8 ch: mid 53.6 vs static 77.9 tok/s. R9's own M1 gate ("test-set hit at 0.607 < 0.75 → TP-for-GLM path
not worth building") sits right at the boundary of this published held-out evidence (code 0.70, agentic 0.77 at 0.556).

### C04 — "LRU 9% vs 5.6% random", "much flatter" routing (R10) — CORRECTED

`OPTIMIZING-GLM53-ON-ONE-GPU.md:526-527@865edb6`: "A 16-slot-per-layer LRU over 288 experts measured a 9.0% hit rate (10188 /
113208 fetches) against 5.6% at random." The capacity is 16/288 = 5.6%. The same repository, `README.md:157-158`: "Routing is
in fact strongly skewed **within** each layer — the hottest ~36% of a layer's experts serve 75% of its fetches", and
`LAB-NOTEBOOK.md:159,173-176`: "the most expensive analytical mistake: 'routing is near-uniform' … That conclusion was wrong
… The error was in the aggregation … The histogram summed per-expert counts across layers." Cross-check here: glm-flash-lite
counts at 104/288 = 0.361 give 0.733 per-layer (≈ the 75% reported). R10's "GLM routing … much flatter, see §4" is
contradicted by its own cited source. What survives: temporal LRU with a tiny cache is weak; frequency placement is what works.

### C05 — "MTP with offload lost in three independent GLM runs" (R10 §0 item 8) — CORRECTED

pctablet505 `README.md:176-177`: "marlin, no MTP | 8.7 … + MTP-2 | 10.6 – 12.1 | one expert fetch verifies ~2.2 tokens"; only
"MTP depth 4 … Worse (10.77 → 8.72)" (`:662`). anoane `README.md:159-160`: "MTP / speculative decoding. 0.63x at short
context, 1.08x at 256k … acceptance (0.462 at 32k) does not pay". So: MTP-2 helped one offload engine (+22–39%), helped
slightly at 256K in another, hurt at short context. The third source (colibri "32% loss around 85% expert hit") was not
re-checked (UNVERIFIABLE here, clone not made). The claim as written overstates the negative; R9's ceilings (MTP gains only
on the GPU-bound part) are consistent with this mixed picture.

### C06 — lt-Strata routing facts (R5 §0.5) — CONFIRMED (+ label note)

`tools/sim/data/lru_curve.json@e4dbb18` source: "ideal per-layer LRU replayed on build-q2-v3 routing traces (q23, 288 experts)";
points include `[0.1007, 0.4119], [0.2014, 0.5773], [0.3993, 0.7589], [0.6007, 0.8443], [1.0, 0.8848]`.
`docs/GLM_DECODE_RESEARCH.md:61-68`: "Adjacent-token expert reuse: 23 % … (chance 2.8 %, i.e. 8x chance)"; "Distinct experts
per window of 2 / 3 / 4 / 8 / 16 tokens: 1.78 / 2.46 / 3.10 / 5.3 / 8.7x one token." All as R5 states. Note: lt-Strata's
glm53f branch has moved since (HEAD `999616c`, 2026-10-11); R5's SHA was checked out explicitly. R5's "hit ≈ 0.9–0.95 at our
~60% coverage" is above this curve (0.844 at 0.60, 0.885 at 1.0 — the gap is cold misses of a short trace); it is a
HYPOTHESIS that long sessions remove cold misses, not something the data shows.

### C07 — halt95 Qwen on 4×3090 (R10 §2.1) — CONFIRMED numbers; summary DISPROVED (known)

`BENCH-CARD.md:121@28cbb72`: "261K | 5,236 [5,234–5,238] | +2.3 % | 155.5 [139.5–160.8] | 160.8 / 139.5 / 155.5 | 20.04 …";
line 3 provenance: "4× RTX 3090, 220 W, PCIe P2P, no NVLink". `docs/reference.md:289`: "P2P over the aikitoria open-kernel-module
patch". No-P2P (`reference.md:99-105`): measured on v2.2.0, "on a host that also had a PCIe x8 link per card and its cards
split across two NUMA nodes … Three 512-token decodes of a 24-token prompt took 3.6–4.0 s wall each (127–141 tok/s including
time to first token; not stream-timed, so not comparable with the bench card)". Also: the reference host is "a privileged
Debian 13 LXC" on Proxmox (`reference.md:155`) — a container, not a VFIO VM, so it says nothing about P2P inside our VM.
Decode tokens/event 2.8–3.4 with MTP K=3 (`BENCH-CARD.md:115-121`): the 155 tok/s depends on high draft acceptance on that
benchmark's prompts.

### C08 — DominikBucko "103 tok/s with an AVX2 CPU tier on our CPU class" (R10 §2.4) — CORRECTED

`docs/memory.md:52-58@939905b`:
> "| Decode, cold experts | Copied into the GPU cache on a miss | Computed by the CPU (one thread pool per GPU), ~30% by the GPU over PCIe |"
(columns: 128 GB profiles | 64 GB profile). `configs/fast-256k.env:3`: "Requires working bidirectional CUDA P2P";
`README.md:306-308`: the fast 256K profile "needs working bidirectional CUDA P2P between the two cards". The CPU-expert
profile (`configs/2x3090-64gb.env`, `QWEN38_CPU_EXPERTS_CPUS=auto`, `DISABLE_CUSTOM_ALL_REDUCE=1`) is the one that "CUDA P2P is
not needed" (`README.md:98`), with 84.2–88.8 tok/s at 131,099 + 512 (`README.md:88`). Hardware (`docs/hardware.md:25-31`):
TR PRO 5975WX, 8×16 GB DDR4-3200, P2P checks passed. So the evidence for "AVX2 CPU tier, Zen 3 8-ch, no P2P" is
**84–89 tok/s at 131K for Qwen3.8-FN**, not 103 at 260K. The 103 tok/s case is evidence that, for Qwen's 3.1 MB experts,
copying misses into a GPU LRU over PCIe (with P2P all-reduce between 2 GPUs) works — the opposite lane to the one R10 credits.

### C09 — anoane GLM 4.05 bpw at 262K (R10 §2.5) — CONFIRMED

`README.md:23-36`: "exllamav3 dev `0531096`. GLM-5.3-Flash-exl3 4.05bpw … RTX PRO 6000 Blackwell 96 GB + Ryzen 9 9950X3D /
128 GB DDR5 (dual channel, ~57.5 GB/s measured)"; "prose | 262,144 | … | 40.24 / 39.71"; "code | 262,144 | … | 33.66 / 34.01".
`README.md:53`: "step_ms ~= 10.8 + 0.77 x cold_percent"; next line: the slope is "42 layers x 8 topk x 12.62 MB / 100 / 53 GB/s
-- the fit recovers the measured AVX-512 CPU kernel bandwidth". Repo listing has no LICENSE file. For us: the kernel figure is
AVX-512 on Zen 5; ExLlamaV3's AVX2 tier measured 15.6–21 GB/s on 20 Skylake cores (C13), so R10's "35–55 tok/s for our box"
needs an AVX2 Zen 3 measurement before it is used.

### C10 — existence of a ≥4 bpw EXL3 GLM-5.3-Flash (R9 §6, §7) — CORRECTED

anoane `README.md:3`: "Serving recipe for `turboderp/GLM-5.3-Flash-exl3` at 4.05bpw"; `:6`: "The model is ~165 GB of weights".
R6 also listed community 4.0 bpw EXL3 quants (search snippets). So the "weights" blocker in R9 §7 ("A ≥ 4 bpw EXL3 of
GLM-5.3-Flash needs a conversion … or must be found") is resolved at REPORTED level (HF not reachable here; size not
verified by download). ~165 GB > 128 GB VRAM, so TP-only ExLlamaV3 still cannot hold it; the TP + CPU-tier gap of R9 §6 stands.

### C11 — ExLlamaV3 TP and CPU-tier facts (R6 §4, R9 §2.2) — CONFIRMED

- `glm5_next.py:350`, `qwen4_exp.py:311`, `mimo_v2.py:389`: `"supports_tp": True`. `deepseek_v4.py` has none; it inherits
  `self.caps = {"supports_tp": True}` (`model/model.py:26-28`). No DeepSeek-V4.1/Engram architecture file (`ls architecture`).
- `model/model_tp_backend.py:242-246`: "all_reduce() currently routes through pg_all_reduce_cpu(): GPU workers publish their
  contributions into the R buffer, a designated CPU helper performs the reduction over host memory … This avoids relying on
  NCCL for the native backend"; `:458-472` the GPU-ring alternative is commented out.
- `modules/mla_attn.py:1195-1200`: "Tensor-parallel: the layer runs whole on one device … (max_devices = 1)".
- `model_init.py:212-217`: `assert not args.tensor_parallel, "--moe_cpu_split currently requires layer-split mode"`; same
  assertion on dev `60c0162` (2026-10-10, newer than R6's `d334b03`) at line 216 — R6's HYPOTHESIS "unchanged on dev" is now
  CONFIRMED. `doc/env_vars.md:315-325`: "Layer-split mode only".
- `exllamav3_ext/cpu/moe_mul1.cpp:43`: "Accuracy matches the GPU int8-GEMV mode-2 class (~0.9% per-call output RMS)".
R9's "DISPROVED: TP of the dense part not in ready engines" is correct at code level (declared, not run on GLM here).

### C12 — ExLlamaV3 native TP measured (R6 §4.2) — CONFIRMED with context

Commit `8114c49330a2…` (2026-10-04): "Measured on six RTX 3090 (PCIe Gen3, Threadripper PRO 3945WX, EXL3 6.05 bpw
Qwen3.8-Flash-Next, native TP, chunk 1024, three 32K cold prompts each): cold prefill 1,238 -> 1,458 tok/s, decode unchanged
(76.1 vs 76.2 tok/s at temperature 1.0) … Microbenchmark of the accumulate alone, six ranks, fp16 wire: 9.8 GB/s on one
thread, 15.7 GB/s on four." Not stated: MTP on/off, P2P (irrelevant to this backend). This is the only end-to-end number for
the backend that R9 §7 builds on. R9 Table C gives `tp_owner` 168–173 tok/s for Qwen on 4 GPUs; R9 Table G implies that 76 tok/s
corresponds to ~100 µs-class all-reduce or ~80 µs per-layer fixed cost (PROVISIONAL reading of R9's own sensitivity rows).
6 ranks on Gen3 is not our 4 ranks on Gen4, but R9 should cite it as the current empirical anchor.

### C13 — ExLlamaV3 CPU kernel speeds (R6 §4.3) — CONFIRMED

Commit `94c05babb123…` (2026-09-08): "Measured on a Xeon Gold 6148 (20C, AVX-512F/BW/DQ/VL, no VNNI) … 4096x2048 K3 15.6 ->
24.0, K4 21 -> 35 … End to end (TabbyAPI): GLM-5.3-Flash 3.05bpw, 232/288 experts per layer on CPU, 4.3-4.9 -> 7.9 T/s;
Qwen3.8-Flash-Next 4.05bpw, 340/512 on CPU, 17.8 -> 24.1 T/s." The left values are the AVX2 tier (what a Zen 3 runs).

### C14 — llama.cpp internals (R6 §2.2–2.3) — CONFIRMED

At `abee0c8` (checked out by SHA; master has moved to `23b0202`): `allreduce.cu:402-405` "if (n_devices != 2) { … internal
AllReduce only supports n_devices=2 … falling back"; `llama-arch.cpp:1235` `case LLM_ARCH_GLM5_NEXT:` inside
`llm_arch_supports_sm_tensor` (returns false); `llama-moe-cache.cpp:234-235` "MoE cache does not support tensor parallelism";
`:549-550` `ggml_backend_tensor_get_async(...); ggml_backend_synchronize(backend);`; `llama-context.cpp:434-440`
`cparams.moe_cache_size == 0; // not supported by the MoE cache`; `llama-model-loader.cpp:1269-1276` "avoid using a host buffer
when using mmap". Line reference 1240 in R6 is off by 5.

### C15 — `--moe-cache-mib` and 128 GB RAM (R5 §T4, §6) — CORRECTED

`llama-moe-cache.cpp:241-266`: "only cache layers that keep all of their experts in host memory … if (experts.empty() ||
!std::all_of(... llama_moe_cache_is_host_weight)) continue;". Layers kept on the GPU (no `-ot`/`-ncmoe` for them) never need a
host copy. Arithmetic (PROVISIONAL, sizes from `PROBLEMS.md:62` of this repo, 173.4 GiB experts = 4.13 GiB/layer, ~107.7 GiB
VRAM for experts per ROOFLINE): 16 layers fully on GPU (66 GiB) + 26 host layers (107 GiB of host experts) with ~41 GiB of
cache slots → ~62% of expert bytes VRAM-resident, host RAM ~107 GiB. That fits 128 GB only tightly (page cache + OS + KV
checkpoints), so R5's practical concern is real, but "needs every expert in host RAM" is wrong; per-layer exclusivity exists.
What is true: within a cached layer the VRAM slots duplicate host bytes.

### C16 — stock vLLM on sm_89 (R7) — CONFIRMED

`sparse_indexer.py:699-702@187a0eb`: `if current_platform.is_cuda() and not has_deep_gemm(): raise RuntimeError("Sparse
Attention Indexer CUDA op requires DeepGEMM to be installed.")`; `platforms/cuda.py:688-695` `support_deep_gemm` = SM90 or
SM100/SM120 families ("Currently, only Hopper and Blackwell GPUs are supported."); `cuda.py:136-153` non-SM12x list
`FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA, *sparse_tail`; `custom_all_reduce.py:270-276` "Custom allreduce is
disabled because it's not supported on more than two PCIe-only GPUs". Note: R7's claim is literally "requires DeepGEMM
installed"; whether DeepGEMM installs on sm_89 and fails later is HYPOTHESIS in R7 and stays so.

### C17 — vLLM GGUF (R7 §0 item 3) — CORRECTED

`docs/features/quantization/gguf.md:6-10@187a0eb`: "GGUF support has migrated to OOT vllm-gguf-plugin … Make sure you have
GGUF plugin installed before serving a GGUF model"; in-tree tests `tests/plugins_tests/gguf/*`. `vllm-gguf-plugin@e2b8ad5`
(2026-09-24): grep for `glm5|qwen4|mimo|deepseek` in `*.py` finds only `deepseek_v3/deepseek_v2 → deepseek2` mapping
(`weights_adapter/transformers.py:65-66`). So: vLLM can still load some GGUFs via a plugin; for our architectures no support was
found. R7's operational conclusion holds; its wording ("no GGUF loader … cannot be served at all") is too strong.

### C18 — yhfgyyf sm_89 fork (R5 §2.6, R10 §2.3) — CONFIRMED

`sparse_attn_indexer.py:61-66@3f3bbae`: `return has_deep_gemm() or (HAS_TRITON and not use_fp4_cache and
current_platform.is_device_capability((8, 9)))`. `flashinfer_mla_sparse.py:199-200`: `return capability.major == 12 or
capability == DeviceCapability(8, 9)` — this is why the README's sm_89 command uses `--attention-backend
FLASHINFER_MLA_SPARSE_SM120`. `README.md:584-615` (§6.1 SM89 8× RTX 4090 48GB): `export NCCL_P2P_DISABLE=1`,
`--tensor-parallel-size 8`, `--kv-cache-dtype fp8`, MTP 5, `--max-model-len 262144`, "参考社区用户在 Issue #74 … 使用官方 FP8 权重完成的部署验证".
No sm_89 tok/s in the README; issue #74 itself not readable here (UNVERIFIABLE beyond the README).

### C19 — SGLang-KT GLM CPU experts FP8-only (R7; R5 §0.10) — CONFIRMED; R5 UNKNOWN resolved

`glm5_next_moe.py:178-185@18c0de7`: `if method == "FP8": return` / `raise NotImplementedError("GLM-5-Next KT CPU experts require
--kt-method FP8 … the current {method} KT kernel discards swiglu_limit and would produce incorrect activations")`.
`glm5_next_indexer_triton.py:21-24`: `return torch.cuda.get_device_capability(device) in ((8, 6), (8, 9))`. So in the code that
runs GLM with KTransformers CPU experts, INT4/GPTQ/RAWINT4 are refused for GLM — the R5 UNKNOWN "no GLM-5.3 INT4 checkpoint path
documented" is answered "not supported" (DISPROVED as a possibility in this fork; upstream kt-kernel was not re-read for a later fix).

### C20 — Maya (R5) — CONFIRMED

`glm_fast.cu:925-927@10b7b07`: "Q4_K / Q5_K / Q6_K experts (K-quant GGUFs other than Maya's): no shared-activation form yet -
each row takes the dense GEMV's dot (row_dot)". `CHANGELOG.md:270-271`: "Speculative decode on three GPUs or more (#31 by
@ksanislo): the parts split into a head and a tail group (4x Tesla T4: decode 15.0 -> 22.6 tok/s)." README line 248 quotes
"4x Tesla T4, Maya-S: decode 14.1 -> 21.0 tok/s" for the same option (both ×1.5). `glm_fast_path.cu:3787-3799`
`hop_to_next`: D2H `cudaMemcpyAsync` into `hop_h`, event, H2D on the next device — no peer copy. `README.md:147-148`: "other
quants than these four are experimental: they run, but Maya is not measured with them". R5's "No measurement at ≥4 bit" stands.

### C21 — Strata tail skip (R5 §0.2) — CONFIRMED

`generate.cpp:8068-8076@61b3fb5`: "STRATA_ROUTE_TAIL_SKIP: ON BY DEFAULT on CUDA builds (owner decision, 0.1.42) … =0 restores
0.1.41's bytes … Nothing to do … when every expert is in VRAM; off with --batch slots … and … with a peer or helper GPU";
`tail_default_ok = tse == nullptr && !all_experts_resident && o.batch == 0 && o.peer_device < 0`. For Qwen UD-Q4_K_XL fully
in 128 GB VRAM it would not engage; for any partial-VRAM run it does, so parity runs must set `=0` (R5 is right).

### C22 — Strata helper GPUs gain (R5 §0.4) — CORRECTED

`docs/SECOND_GPU.md:93-96@61b3fb5`: "Measured by the PR's author: dual RTX 4090 +63% mixed / +132% code decode over the plain
helper path; RTX 5090 + 4090 +28% / +63%." Line 126: "Three/four cards have not been validated." The gains are of the
`--remote-expert-opt` optimisation relative to helpers without it; no number in this doc gives helpers vs a single GPU. R5's
T3 ceiling (55–65 tok/s) therefore has no measured multi-GPU anchor.

### C23 — freetoken-rdna3 host all-reduce (R10 §2.6) — CORRECTED

`docs/rdna3/journey.md:103-104@a4469ae`: "Host-memory all-reduce: RCCL stages each 5 KB decode all-reduce through host memory
with its proxies (~12 µs in a graph). A shared host region and a one-block kernel do it in 6 µs with the same bits: +2.5-4.8 %
decode." `options.md:33`: "`1` (two ranks, ROCm) … 6 µs instead of ~12 µs per decode all-reduce". This is a useful 2-rank
anchor for R9's all-reduce parameter (6 µs host-staged on AMD; 4 ranks and CUDA unmeasured).

### C24 — PCIe prefetch argument (R8 §0 item 3, §7.4) — CORRECTED premise

Recomputed: 25e9 B/s × 0.444e-3 s / 15.38e6 B = 0.72 experts per layer per link at 50 tok/s (R8 right). R8's "~3.1 of 8 experts
miss per layer at 61% residency" is 8 × 0.39 (uniform). With frequency placement: 0.74 distinct misses/layer at n=1 in-sample
(R9 App. A, reproduced inputs in C02) and 8 × (1 − 0.75) ≈ 2.0 at the held-out ≈0.75 of C03. The structural gap shrinks from ~4×
to ~1–3×; whether prefetch pays then depends on recall/precision at d=1 and DRAM contention, which R8's cost model handles.
The empirical negatives R8 cites stand (Strata `DETAILS.md:1650-1658`: Foresight slots 0.91–1.00× "in every box measured";
5070 Ti probe `bench/results/2026-10-04-community-rtx-5070ti-windows/README.md:117`: "recall 27–34%, precision ~8% … closed").
R8's own cost-model defaults (h_v 0.61, uniform) inherit the same issue R9 raised against ROOFLINE.

### C25 — Qwen Community License 1.0 (R10; E15) — CONFIRMED text

`LICENSE.qwen-community-1.0:9@28cbb72`: "If the licensee or any of its affiliates conducts a Model as a Service or AI Work
Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works
for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided
that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party."
Line 11 defines "AI Work Assistant" as "an independent AI-powered product primarily designed for AI-assisted coding or office
productivity". Clause 1: attribution above 100 M MAU or US$20 M monthly revenue. This is a third-party copy (halt95 says the
checkpoint derivatives "carry the Qwen Community License 1.0", `NOTICE:247-248`); the vendor's own text was not reachable (HF
blocked). For Ribrad: internal coding-agent use without third-party access is permitted; offering it as a coding-assistant
service to others needs a separate licence. EVIDENCE_MATRIX E15 can move from REPORTED to CONFIRMED-text (third-party copy).

### A1–A8 (short)

- A1: PR #30277 head `bcb1101` (Georgi Gerganov, 2026-10-10): "mirror the attention part of models with n_embd_head_kda != 0 …
  their MLA layers use an absorbed latent, so neither can be head-split. FFN and output still split". PR #27861 head `bccbacd`
  (csantiago78, 2026-08-28), diff vs `4e97ac8`: `if (n_tokens == 1 && !gate_up_exps && gate_exps && down_exps && …` and the CPU
  `src[3]` id table in `ggml-cpu.c`. GLM5-next uses separate gate/up expert tensors (`src/models/glm5-next.cpp:166-168`), so
  the condition is met.
- A2: KT `GLM-5.3-Flash-Tutorial.md:12-16`: "approximately 306 GiB. Reserve at least 350 GB … NVIDIA SM89 and SM120 … The
  AVX-512 FP8 CPU expert kernel"; `AVX2-Tutorial.md:31-34` "Memory: At least the size of the model weights";
  `experts-sched-Tutorial.md:176`: "Qwen3-Next-80B-A3B-Instruct-FP8 with 4 x RTX 4090, Intel Xeon Gold 6454S, tensor parallel
  size 4, using ShareGPT dataset".
- A3: club-3090 `PCIE_P2P.md:483` "field-validated on two Proxmox + Q35/OVMF + VFIO guests (3× and 2× RTX 3090)", `:542-545`
  +21.9% / +16.3% prefill, +6.3% / +7.1% decode; xiaoyanzi191 `docs/RESULTS.md:9` "Baseline: previous A3 W8A8-tuned deployment",
  `docs/REPRODUCE.md:109` model "Qwen3.8-27B-FP8".
- A4: quotes in the table rows; ds4 `ds4.c:455-462` returns true for CUDA only under `DS4_ROCM_BUILD`.
- A5: Strata 2×3080 report `:10-11` "This report is a fork build on the 0.1.40.3 base. It does not reproduce from upstream
  `main`"; decode table `:29-35` (4K–127K); `:106` "`--ple-io direct` … speed neutral (25K reads 1,458 / 1,468 … against
  1,451 / 1,465 …; two different binaries)".
- A6: `async_store.cpp:186-188`: `int ev_engine = use_io_uring ? photon::INIT_EVENT_IOURING : photon::INIT_EVENT_EPOLL;`
  (`# set(USE_IO_URING ON)` commented in `CMakeLists.txt:38`). Archived KV-cache code, not expert I/O: no decision impact.
- A7: search result for https://deploy.helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1: "A single NUMA-local four-GPU SGLang
  engine running the community NVFP4 checkpoint reached 63.7 output tokens per second for one request … EAGLE … 82.8 tok/s
  … Helix calls the day-zero serving stack not ready for production". Page not loadable here.
- A8: `glm-flash-lite docs/reference.md:123`: "host RAM, `fast` | ~218 GiB (~234 GB) for the process today: pinned home copy of
  all routed experts … + the CPU tier's block-contiguous second copy (114.2 GB)"; `:124` "`exact` | ~111 GiB … 128 GiB total is
  the minimum"; `:125` "~3.6 GB/s per core under all-core load | EPYC 7443P 24C"; `:126` "measured 136 GB/s read at 22 threads".

---

## Impact on decisions

1. **GLM hit-rate input (R9, R8, R6, ROOFLINE).** Non-uniform routing is confirmed by three independent sources (C01, C02, C04).
   But the planning value at ~0.6 residency is the held-out ≈ 0.75 (C03), not the in-sample 0.907. R9's headline "TP 78 / 110
   tok/s without speculation, 102–109 with MTP" should be restated on the "mid" column (≈ 54 / 70–80 range in R9's own Table
   A/B); R9's M1 stop/go threshold (0.75) is right at the published evidence, so M1 (our own held-out GLM traces) remains the
   deciding experiment. R8's "PCIe prefetch can't pay" and R6's ExLlamaV3 GLM 25–37 tok/s estimate both assume uniform routing
   and should be re-run with the skewed curve.
2. **Qwen3.8-FN numbers used as "comparable hardware" (R10 §3.1).** Q1 (155.5) needs working P2P; Q3 (103.1) needs P2P and does
   not use the CPU tier (C07, C08). For a no-P2P VM, the comparable anchors are DominikBucko's 84–89 tok/s at 131K (CPU tier,
   2×3090) and the ExLlamaV3 native-TP 76 tok/s at 32K on 6×3090 (C12). P2P inside our Proxmox VM (A3) therefore moves from
   "single-digit percent" to "gate for the best Qwen path" — E0/G0 P2P validation priority goes up.
3. **ExLlamaV3 as the GLM TP base (R9 §7).** Code facts confirmed (C11); the ≥4 bpw weight blocker is resolved (C10, 4.05 bpw
   exists, ~165 GB). Unchanged blockers: no TP + CPU tier, AVX2 tier of the CPU kernel is 3–4× slower than the AVX-512 numbers
   R10 used (C09, C13), int8 activations (~0.9% RMS per call) on the CPU path, and fp16-only KV on the GLM DSA path
   (anoane `README.md:161-162`). The only measured native-TP number (C12) is ~0.45× of R9's Qwen TP ceiling — R9 Table C/F2 needs
   that anchor before Qwen-on-ExLlamaV3-TP is ranked above Strata.
4. **Speculation with offload (R10 vs R9).** MTP-2 is not a demonstrated loss with offload (C05); keep MTP k=1–2 in the GLM plan
   and drop R10's blanket negative.
5. **llama.cpp path (R5 T4).** `--moe-cache-mib` is not ruled out by RAM size per se (C15); it is ruled down by the per-layer host
   sync, no CPU lane and no PP (C14). R5's "concrete argument against stock llama.cpp `--moe-cache-mib`" should be reworded.
6. **Strata helper GPUs (R5 T3).** No measured scaling vs one GPU exists (C22); T3's 55–65 tok/s ceiling is unanchored.
7. **vLLM for GLM (R7 vs R5/R10).** Both are right at their scope: upstream blocked on sm_89 (C16), forks open it (C18, C19);
   GGUF route through vLLM is not available for our archs (C17). No sm_89 GLM tok/s exists anywhere.
8. **Licence (C25).** Qwen3.8-Flash-Next is usable for internal coding agents; any plan where Ribrad serves it to third parties
   as a coding assistant needs a separate Qwen licence — the customer question R2.4 in ROADMAP stays open but now has the exact text.
9. **glm-flash-lite as a model for tiers (A8).** Its CPU lane needs a second full RAM copy of the experts; not portable to a
   128 GB host as-is. Its Zen 3 numbers (136 GB/s read, ~3.6 GB/s/core) are the best CPU-tier anchor for our CPU class and
   support R8's 8-channel B_dram·η ≈ 120–140 GB/s assumption.

## Could not verify

- GitHub issue/discussion contents (yhfgyyf #74, club-3090 #1454/#1615, llama.cpp PR threads): only README/commit text.
- Hugging Face cards (turboderp 4.05 bpw size, `wtdcode/GLM-5.3-Flash-AWQ-W4A16` size, Qwen official licence page): blocked.
- colibri "MTP 32% loss around 85% expert hit" (not cloned), rtx6kpro Discord summaries, x.com reports.
- ExLlamaV3 native all-reduce latency and NCCL-in-VM behaviour: no measurement exists; nothing to verify against.
- Whether halt95 / DominikBucko / vllm-backport trees build for sm_89 (builds > 20 min, no GPU).
- R7's PROVISIONAL KV/state byte formulas and R9's tool outputs were not re-derived (they are arithmetic on stated inputs; R9's
  tests were not re-run).

## Sources

Repositories (cloned 2026-10-10 into `scratchpad/research/r11`; SHA and commit date from `git log -1 --format='%H %ci'`):

| repo | SHA | commit date |
|---|---|---|
| https://github.com/Grigory-Rylov/strata-glm-3090 | 794cd69ae270389c67ed201b855b14aafb5eb875 | 2026-10-03 20:11:15 +0300 |
| https://github.com/sybil-solutions/glm-flash-lite | 6769b27652a18487758caac45a2b6999dc0a4184 | 2026-10-09 09:03:52 -0400 |
| https://github.com/pctablet505/glm53-flash-single-gpu | 865edb6ec46f9c7cd72ee2ff137355a80b783f8d | 2026-09-02 10:57:04 +0530 |
| https://github.com/anoane/glm53-flash-exl3-recipe | 116e6a2e0ada95b5f3b592581ac0cd1ff2788adf | 2026-09-01 13:05:13 +0800 |
| https://github.com/halt95/qwen38-flash-next-3090s | 28cbb7293f134c244b443fb43bbf6e74ff4d8e24 | 2026-10-06 11:35:09 -0400 |
| https://github.com/DominikBucko/qwen38-flash-next-2x3090 | 939905b0dab8c6eb3852ff35cbbd42d098af2451 | 2026-10-02 11:56:00 +0200 |
| https://github.com/Cedriceuh/freetoken-rdna3 | a4469ae01d60180df72b427d88358a04b44661e5 | 2026-10-10 18:43:49 +0200 |
| https://github.com/yhfgyyf/vllm-deepseek-v4-sm89 | 3f3bbaea916707c851ca027b87b97fb6605998b8 | 2026-09-16 20:44:30 +0800 |
| https://github.com/mw00/project-maya | 10b7b078c11f455ff16f3ea3bbea5170c879e767 | 2026-10-10 18:23:54 +0100 |
| https://github.com/xiaoyanzi191/rtx4090-small-bar-p2p | dbd0cdf15c76548626ab224dbd9e34097d866ea3 | 2026-09-16 14:49:52 +0800 |
| https://github.com/antirez/ds4 | fc80bd695da76ee14bcb86820bf900a5cdfda806 | 2026-10-07 21:43:05 +0200 |
| https://github.com/turboderp-org/exllamav3 (master) | 151539c77abc7ab7425d30da7a4e8e3c5c154e7b | 2026-10-07 18:47:56 +0200 |
| exllamav3 dev (fetched) | 60c0162345aa8a273b51e2ee16e36ecdd186ddb1 | 2026-10-10 20:38:30 +0200 |
| exllamav3 commit 8114c49 | 8114c49330a29f891df3ec3f8416a3567b5a8c8c | 2026-10-04 23:13:12 -0500 |
| exllamav3 commit 94c05ba | 94c05babb1239c7ef79f42de3144c3707c0719cc | 2026-09-08 23:17:04 +0530 |
| https://github.com/Niko1221/Strata | 61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406 | 2026-10-10 14:49:37 +0200 |
| https://github.com/lighttransport/Strata (glm53f; checked out by SHA) | e4dbb18a0cc4de20359c2c74b6ea419776d2e8f5 | 2026-10-10 17:17:31 +0900 (branch HEAD now 999616cef543eb32445d8b16618bdb0c87fdada1, 2026-10-11 03:39:05 +0900) |
| https://github.com/ggml-org/llama.cpp (checked out by SHA) | abee0c8476b585b0d081d580b5ce21d0addc22c4 | 2026-10-10 20:00:56 +0200 (master now 23b0202a189c44a54625aadcb37a946dd1d6278d, 2026-10-10 21:25:38 +0200) |
| llama.cpp PR #30277 head | bcb110179a1513c983c8f90f50fa3623951b76c6 | 2026-10-10 12:18:09 +0300 |
| llama.cpp PR #27861 head (parent 4e97ac86ebe2c4cb8212d98d2641ad6768810896) | bccbacdb8945680f1cfc7e6bffd1e59014705750 | 2026-08-28 04:20:04 -0400 |
| https://github.com/ikawrakow/ik_llama.cpp | 89bba35c8841618502b2be8273fb5265b405aca1 | 2026-10-10 08:32:03 +0200 |
| https://github.com/kvcache-ai/ktransformers | 5d048e516507e1603e0ed19645611b4501476a50 | 2026-10-10 16:16:45 +0800 |
| https://github.com/kvcache-ai/sglang (blobless fetch by SHA) | 18c0de72e59359baba793f7a5675f983aa656fc9 | 2026-09-30 21:29:17 +0800 |
| https://github.com/vllm-project/vllm (blobless fetch by SHA) | 187a0eb98aa42341d703f83421d693fa7585581b | 2026-10-10 16:36:03 +0000 |
| https://github.com/vllm-project/vllm-gguf-plugin | e2b8ad532b8b5ea175100202c30430c1d2b5e6a8 | 2026-09-24 11:44:44 +0800 |
| https://github.com/ztxz16/fastllm | a2ff521f61820e1972a84e494cda58c0c2572c5f | 2026-10-10 11:12:53 +0800 |
| https://github.com/noonghunna/club-3090 | c8b6a737988640bd0985e7d8ab6d0571f81a49fc | 2026-10-10 21:31:18 +0500 |
| https://github.com/wtdcode/vllm-backport | 29e66dad461b11caae6039f510eaa7913f9dfb19 | 2026-09-27 00:16:02 +0800 |
| https://github.com/daraskme/Strata-GLM | 9e0b779d963c2b1ccc1925864a2247c92e16b43a | 2026-10-09 05:25:01 +0900 |

Also checked (not separate claims): vllm-backport `docker/Dockerfile:292` (`'7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'`) and
`README.md:128` ("Verified with MTP-3 on 4x A100-80GB (sm80)" — GLM is not validated on sm_89 there); daraskme `README.md:9-18`
(15.9 / 17.1 tok/s at 8,180 / 260,090 prompt tokens, RTX PRO 6000 + DDR4-3200, "50 tokens/sは未達") — both CONFIRMED as R5/R10
describe them.

Web (search only; pages not loadable): https://deploy.helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1 (query "helix.ml
GLM-5.3-Flash RTX PRO 6000 SGLang tok/s single request EAGLE", 2026-10-10).

In-repo files read (read-only): `/home/user/wt-research` R5–R10, `matrix/`, `PROBLEMS.md:60-66,105-118`, R1 line 44/224,
`EVIDENCE_MATRIX.md:31`. Scripts: `scratchpad/research/r11_scripts/skew.py` (C02/C03 arithmetic).
