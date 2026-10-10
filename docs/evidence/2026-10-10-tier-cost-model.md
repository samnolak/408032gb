# 2026-10-10 — tier cost model (R8 §8) as a tool: three fixes and the sensitivity analysis

Branch `claude/research-landscape`. Role: perf-bench. `tools/ceilings/tier_cost.py` + `tests/test_tier_cost.py`.
Everything below is model output (PROVISIONAL): a bandwidth-and-latency model with stated parameters, run on the
sandbox CPU. **No number here is a measurement, and none of it is a promise of tok/s on the server.**

## 1. What the tool adds

R8 (docs/reassessment/research/R8-memory-tiers-expert-scheduling.md §8) specified a per-layer decode cost of one MoE
step across VRAM / RAM (CPU compute) / NVMe / PCIe with expert prediction, prefetch, speculation and table lookups,
and left a reference implementation in the agent's scratch directory. `tools/roofline.py` and
`tools/ceilings/arch_ceilings.py` do not model the NVMe tier per layer, prediction/prefetch with its DRAM and PCIe
contention, or lookup latency. The tool ports R8's model, derives per-model constants from `models/specs/*.json`
through `tools/roofline.py` (expert bytes, dense bytes, KV/state read at 262,144 tokens), and adds sweeps and a
one-at-a-time sensitivity ("tornado") analysis.

Hardware defaults are the ones R8 used (labels in the dataclass), except: NVMe per-drive bandwidth 7.45 GB/s x 0.6
(the vendor figure in `hardware/4x4080s-32g.json`; R8 used 7.0 x 0.6), and 4 drives (the customer's specification of
2026-10-10 says 4 x 1 TB; `hardware/4x4080s-32g.json` and AGENTS.md still say 3 — a discrepancy to resolve in R0;
`n_drives` is in the tornado).

## 2. Fixes against R8's scratch implementation (CONFIRMED by tests; each is a test in tests/test_tier_cost.py)

1. **Phantom NVMe latency.** The term for predicted NVMe reads that overrun the lead time was
   `max(0, (pred + wasted) * per + L_s - W)`, which is `L_s` (80 µs) per layer even when nothing is predicted.
   R8's rows "1% on NVMe" and "5% on NVMe" carried 42 x 80 µs = 3.4 ms per step that no read causes
   (`test_no_prediction_costs_no_background_latency`). Literal, with R8's own constants (`--r8-layer --set
   B_s_drive=4.2e9`): R8 printed 27.9 and 20.4 tok/s (8 channels), the fixed model gives

   ```
   h_v .61, 1% on NVMe | 8ch | 30.83 | 7.9e-05
   h_v .61, 5% on NVMe | 8ch | 21.93 | 0.000393
   ```
   (columns: case, DRAM, tok/s, NVMe seconds per layer). Every other R8 row is reproduced to 0.05 tok/s
   (`test_r8_rows_reproduced_with_r8_constants`). R8's conclusion stands with a smaller penalty: 1% of routed bytes from
   NVMe costs 34.4 -> 30.8 tok/s in this model (R8: 34.4 -> 27.9).
2. **Link-rate prefetch made a faster link slower.** With R8's rule (copy predicted experts at link rate; wasted
   copies still cost DMA that steals DRAM bandwidth), PCIe x16 came out slower than x8 for low-precision predictions;
   the property test `x8 never faster than x16` found a case (4.54 vs 4.52 tok/s). A scheduler can always copy less, so
   the ceiling now takes the copy count that minimises the layer time (`prefetch_opt`, default; every term is linear
   in the count, the layer time convex, ternary search). R8's rule stays available (`prefetch_opt=False`) and the R8
   table uses it.
3. **All-or-nothing NVMe staging made more recall slower.** The tornado around a 5%-NVMe point with prediction showed
   recall 0.3 -> 25.6 tok/s but 0.95 -> 23.1 (more recall, more wasted background reads). The scheduler now stages the
   best part of the predicted set; the cost is concave on each side of the point where reads start to overrun the lead
   time, so the optimum is at 0, that point, or all — checked against a 401-point brute force
   (`test_nvme_staging_optimum_matches_a_fine_grid`) and by `test_better_prediction_is_never_slower`.

Also: GPU prefetch takes only RAM-resident experts (an NVMe expert would have to reach RAM first, which the model does
not chain).

## 3. Tests (CONFIRMED, literal)

```
$ python3 -m unittest tests.test_tier_cost -v
...
Ran 26 tests in 1.086s

OK
```

Closed forms (all-VRAM layer, DRAM-bound CPU layer), agreement with `tools/roofline.py` (`spec_tokens_per_step`,
`distinct_experts`), the GLM preset against R8's constants (S_e within 0.2%, t_pre within 5%, t_other within 2%), the
properties R8 §8.3 asked for (prediction off == no prediction; more NVMe never faster; overlap never slower; x8 never
faster than x16; more DRAM never slower; useless prefetch never faster than none; zero-acceptance drafts never pay),
the three fixes, lookups, and the CLI.

## 4. Sensitivity analysis (PROVISIONAL model output, literal)

Base point: layer pipeline, 61% of distinct routed experts in VRAM, the rest in RAM computed by the CPU in overlap,
one MTP draft (n_tok 2) at acceptance 0.7, 8 DRAM channels, 262,144-token context. Each row moves one parameter
between the bounds shown, all else at the base.

```
$ python3 tools/ceilings/tier_cost.py --model glm --tornado
# GLM-5.3-Flash: n_moe 42, S_e 15.38 MB, K 8, E 288, t_pre 0.308 ms, t_shared 0.046 ms, t_other 1.61 ms, ctx 262144; PROVISIONAL model output
# tornado around: {'h_v': 0.61, 'h_r': 0.39, 'miss_path': 'cpu', 'overlap': True, 'p_recall': 0.0, 'pi_prec': 1.0, 'W': 0.0, 'prefetch_to': 'gpu', 'prefetch_opt': True, 'n_tok': 2, 'accept': 0.7, 'U': None}  -> base 39.4 tok/s
param | low | high | tok/s at low | tok/s at high | swing
h_v | 0.45 | 0.82 | 31.2 | 53.5 | 22.3
B_dram | 1.024e+11 | 2.048e+11 | 24.0 | 39.4 | 15.4
eta_dram | 0.55 | 0.85 | 33.5 | 44.5 | 10.9
accept | 0.55 | 0.85 | 36.0 | 42.9 | 7.0
t_pre | 0.0002464 | 0.000385 | 41.9 | 36.7 | 5.3
n_cpu | 16 | 46 | 36.5 | 39.4 | 2.9
t_sync | 5e-06 | 6e-05 | 40.0 | 38.0 | 2.1
b_core | 5e+09 | 1.2e+10 | 39.4 | 39.4 | 0.0
B_v | 4.416e+11 | 6.256e+11 | 39.4 | 39.4 | 0.0
B_p | 1.25e+10 | 2.5e+10 | 39.4 | 39.4 | 0.0
n_drives | 3 | 4 | 39.4 | 39.4 | 0.0
p_recall | 0.3 | 0.95 | 39.4 | 39.4 | 0.0
pi_prec | 0.2 | 0.9 | 39.4 | 39.4 | 0.0
L_p | 5e-06 | 3e-05 | 39.4 | 39.4 | 0.0

$ python3 tools/ceilings/tier_cost.py --model qwen --tornado
# Qwen3.8-Flash-Next: n_moe 48, S_e 3.13 MB, K 10, E 512, t_pre 0.190 ms, t_shared 0.000 ms, t_other 0.45 ms, ctx 262144; PROVISIONAL model output
# tornado around: (same policy)  -> base 91.3 tok/s
param | low | high | tok/s at low | tok/s at high | swing
h_v | 0.45 | 0.82 | 77.5 | 119.2 | 41.8
B_dram | 1.024e+11 | 2.048e+11 | 63.6 | 91.3 | 27.7
t_pre | 0.0001518 | 0.0002373 | 101.2 | 81.3 | 19.8
eta_dram | 0.55 | 0.85 | 81.6 | 98.9 | 17.3
accept | 0.55 | 0.85 | 83.2 | 99.3 | 16.1
t_sync | 5e-06 | 6e-05 | 95.0 | 82.7 | 12.2
n_cpu | 16 | 46 | 86.7 | 91.3 | 4.6
(b_core, B_v, B_p, n_drives, p_recall, pi_prec, L_p: swing 0.0)
```

The same with 5% of the routed experts on NVMe and prediction on (recall 0.85, precision 0.5, one layer of lead time):

```
$ python3 tools/ceilings/tier_cost.py --model glm --tornado --set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3 --set h_r=0.34
# tornado around: {... 'h_r': 0.34, 'p_recall': 0.85, 'pi_prec': 0.5, 'W': 0.00044, ...}  -> base 25.9 tok/s
h_v | 0.45 | 0.82 | 22.1 | 33.6 | 11.5
B_dram | 1.024e+11 | 2.048e+11 | 18.2 | 25.9 | 7.7
eta_dram | 0.55 | 0.85 | 23.2 | 28.0 | 4.8
accept | 0.55 | 0.85 | 23.6 | 28.2 | 4.6
pi_prec | 0.2 | 0.9 | 24.2 | 28.7 | 4.6
n_drives | 3 | 4 | 22.6 | 25.9 | 3.3
```

Residency x DRAM channels (one MTP draft, acceptance 0.7; columns h_v, DRAM B/s, tok/s, ms per step):

```
GLM-5.3-Flash                         Qwen3.8-Flash-Next
0.45 | 1.024e+11 | 18.16 | 93.6       0.45 | 1.024e+11 | 50.93 | 33.38
0.45 | 2.048e+11 | 31.2 | 54.49       0.45 | 2.048e+11 | 77.45 | 21.95
0.61 | 1.024e+11 | 24 | 70.84         0.61 | 1.024e+11 | 63.6 | 26.73
0.61 | 2.048e+11 | 39.43 | 43.11      0.61 | 2.048e+11 | 91.28 | 18.62
0.75 | 1.024e+11 | 33.38 | 50.93      0.75 | 1.024e+11 | 81.3 | 20.91
0.75 | 2.048e+11 | 51.27 | 33.16      0.75 | 2.048e+11 | 108.2 | 15.72
0.82 | 1.024e+11 | 41.48 | 40.98      0.82 | 1.024e+11 | 94.44 | 18
0.82 | 2.048e+11 | 53.5 | 31.78       0.82 | 2.048e+11 | 119.2 | 14.26
0.9 | 1.024e+11 | 51.12 | 33.25       0.9 | 1.024e+11 | 115.8 | 14.68
0.9 | 2.048e+11 | 51.12 | 33.25       0.9 | 2.048e+11 | 116.4 | 14.61
```
(two runs of `--sweep h_v=0.45,0.61,0.75,0.82,0.9 --sweep B_dram=102.4e9,204.8e9 --set n_tok=2 --set accept=0.7`,
columns cut to four, placed side by side.)

What the model says (HYPOTHESIS, conditional on its parameters):
* In a layer pipeline the two inputs that move the ceiling most are **VRAM residency of the routed experts** (cache
  policy and placement) and **DRAM channels**; both are knowable before tuning anything else (`tools/g0/cachesim.py`
  on real traces; `dmidecode` in R0). Prediction quality, PCIe width and NVMe drive count change nothing unless experts
  live on NVMe or prediction is on — consistent with R8 finding 3 (prefetch of 15 MB experts over PCIe cannot pay at
  batch 1).
* GLM-5.3-Flash stays below 50 tok/s in a pipeline unless residency reaches ~0.75 with 8 channels or ~0.9 with 4; the
  model cannot tell whether that residency is reachable — the cache simulator on real GLM traces can (pending:
  traces-real run on proxy models; GLM-5.3-Flash itself needs the server).
* Qwen3.8-Flash-Next clears 50 tok/s at every point of the grid and 80 at h_v >= 0.61 with 8 channels or >= 0.75 with 4.
* With overlap, more residency is not always faster (GLM 0.82 -> 53.5, 0.90 -> 51.1): the GPU becomes the bottleneck
  while the CPU idles; a scheduler holding RAM copies could rebalance (tools/roofline.py documents the same effect).
* A pipeline is one of the architectures; tensor/expert parallel and their fixed costs are in
  `tools/ceilings/arch_ceilings.py` (R9). These numbers are not a forecast for R1.

## 4b. Follow-up to R11 C24: does PCIe prefetch pay with skewed routing? (PROVISIONAL model output, literal)

R11 C24: R8's "prefetch cannot pay" compared the link's ~0.7 experts per layer with 3.1 misses per layer, a uniform-routing
value; with the held-out GLM hit ~0.75 there are ~2 misses per layer. In this model the answer does not hinge on the miss
count but on **whether the CPU expert path is DRAM-bound or compute-bound**: a DMA copy reads the same expert bytes from
DRAM that the CPU would read to compute it, so when DRAM is the CPU's limit a copy frees nothing and the best copy count is
0. With the defaults (32 threads x 8 GB/s = 256 GB/s of kernel throughput > 143 GB/s usable DRAM) prediction gives nothing
even at recall 0.95 / precision 0.9 / 3 layers of lead time, at hit 0.61 and 0.75.

The per-core rate is the open input. The best anchor for our CPU class is ~3.6 GB/s per core for an AVX2 expert lane on an
EPYC 7443P (Zen 3, 8 channels), with 136 GB/s DRAM read at 22 threads (R11 A8, REPORTED by glm-flash-lite
`docs/reference.md:123-126`@6769b27). With that rate:

```
$ for n in 32 46; do for hv in 0.61 0.75; do for pf in "" "--set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3" "--set p_recall=0.95 --set pi_prec=0.9 --set W=1.3e-3"; do ... python3 tools/ceilings/tier_cost.py --model glm --set b_core=3.6e9 --set n_cpu=$n --set h_v=$hv --set h_r=<1-h_v> $pf --json ...; done; done; done
n_cpu=32  h_v=0.61 no prediction                                           30.4 tok/s copies 0.00
n_cpu=32  h_v=0.61 --set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3   32.3 tok/s copies 0.68
n_cpu=32  h_v=0.61 --set p_recall=0.95 --set pi_prec=0.9 --set W=1.3e-3    33.6 tok/s copies 0.63
n_cpu=32  h_v=0.75 no prediction                                           37.6 tok/s copies 0.00
n_cpu=32  h_v=0.75 --set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3   39.4 tok/s copies 0.44
n_cpu=32  h_v=0.75 --set p_recall=0.95 --set pi_prec=0.9 --set W=1.3e-3    40.7 tok/s copies 0.40
n_cpu=46  h_v=0.61 no prediction                                           34.0 tok/s copies 0.00
n_cpu=46  h_v=0.61 --set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3   34.0 tok/s copies 0.00
n_cpu=46  h_v=0.61 --set p_recall=0.95 --set pi_prec=0.9 --set W=1.3e-3    34.0 tok/s copies 0.00
n_cpu=46  h_v=0.75 no prediction                                           41.0 tok/s copies 0.00
n_cpu=46  h_v=0.75 --set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3   41.0 tok/s copies 0.00
n_cpu=46  h_v=0.75 --set p_recall=0.95 --set pi_prec=0.9 --set W=1.3e-3    41.0 tok/s copies 0.00
```

Reading (HYPOTHESIS, conditional on the model): with 32 expert threads at 3.6 GB/s (115 GB/s < 143 GB/s DRAM) the CPU
is compute-bound and predicted PCIe copies add 6–10% (30.4 -> 33.6 at hit 0.61, 37.6 -> 40.7 at 0.75); with 46 threads the
CPU reaches the DRAM limit and prefetch adds nothing. PCIe prefetch is therefore a substitute for CPU cores, worth at most
~10% here, not a lever on the ceiling; R8's recommendation (compute misses on the CPU in place, spend prediction on
NVMe -> RAM warming and cache adaptation) stands, for this reason rather than the link-capacity one. The deciding
measurements on the server: Q4_K expert throughput per Zen 3 core, the number of cores free for experts, DRAM channels.

## 5. Not done

No measurement of any parameter (all are defaults with labels in the code); per-layer residency distributions
(the tool takes one h_v; feed per-layer values from `cachesim.py` when real traces exist); DRAM contention between
several links at once; KV growth during a session.
