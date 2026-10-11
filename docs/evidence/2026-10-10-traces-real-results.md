# 2026-10-10 — traces-real: routing of two real proxy MoE models, cache policies and expert prediction

Branch `claude/research-landscape`. Role: perf-bench. Data: workflow `traces-real` run 38079738013, committed by the bot as
`0914244` (`data/traces/real/{qwen3-coder-30b-a3b,glm-4.7-flash}/`). Analysis here: agent sandbox CPU, no GPU, no model
weights; `predict_eval` results are CONFIRMED-from-CI (the probes and GGUFs were not kept), every `cachesim` number was
recomputed here from the committed traces. Tools added for this file: `scripts/traces/cachesim_sensitivity.py` (commits
`830106e`, `8435e8d`) with `tests/test_cachesim_sensitivity.py`. Literal outputs of every run are in
`docs/evidence/2026-10-10-traces-real-results/` (file names below).

**Scope of the claims.** The two models are PROXIES (Qwen3-Coder-30B-A3B for the Qwen family, GLM-4.7-Flash for the GLM
family), run at Q4_K_M on CPU over 16 CONSTRUCTED agent-style conversations with at most 128 greedy tokens each. Every
statement about GLM-5.3-Flash or Qwen3.8-Flash-Next below is PROVISIONAL. Every tok/s is a PROVISIONAL model ceiling of
`tools/ceilings/tier_cost.py`, never a measurement and never a forecast.

## 0. Findings

1. **Self-checks pass** (CONFIRMED-from-CI, `predict.txt` line 1): router weights reproduce the recorded logits to 2.9e-4 /
   1.7e-4 relative error; the selection rule reproduces 72,384 of 72,384 (Qwen) and 69,367 of 69,368 (GLM) recorded top-k
   rows; the probe run selects exactly the experts of the trace run on every recorded row. The single GLM row is not the
   group rule (GLM-4.7-Flash has one group); a near-tie under fp32 vs float64 arithmetic is the PROVISIONAL explanation
   (section 3).
2. **`cachesim` reproduces exactly**: the committed `cachesim.txt` and `cachesim.json` of both models are byte-identical to
   a re-run here from the committed traces (CONFIRMED, `reproduce.txt`).
3. **At the GLM-5.3-Flash residency of 0.607** (UD-Q4_K_XL, ARCHITECTURE-ALTERNATIVES 9.12), the GLM proxy's held-out
   **static** hit is **0.770** (0.754-0.786 over the 4 hold-out folds), inside the planning band 0.70/0.75/0.80 and just
   above R9's 0.75 stop/go line; at 0.546 it is 0.708, beside the only published held-out number for GLM-5.3-Flash itself
   (0.701-0.773 at 0.556). The Qwen proxy is higher everywhere (static 0.908 at 0.607). (CONFIRMED for the proxies,
   `runs-*.txt`; the mapping to the target is PROVISIONAL.)
4. **Better policies exist on the proxies at the same residency**: a static set ranked on **decode** tokens 0.899, LRU
   0.86-0.91 depending on how it is warmed, hybrid (half static, half LRU) 0.87-0.90, Belady bound 0.965 (GLM proxy,
   0.607). How the prompt is replayed moves per-layer LRU by ~5 points and which tokens train the static profile moves it
   by ~13 points; GPU count does not matter (CONFIRMED for the proxies; section 5).
5. **`lru_global` / `lru_device` plateau at exactly 0.900 (GLM) / 0.922 (Qwen)**, the compulsory-miss bound of a cache that
   is empty at the first of 100-128 generated tokens (CONFIRMED, `se_cold_last_full`). With `prompt_readonly` they start
   every generation empty, so from capacity ~0.75 on those columns measure generation length, not the policy (within
   2 points of the bound already at 0.607).
6. **Prediction**: cross-layer pre-gating names 81-86% of the next MoE layer's experts with m = k candidates and 95-98% with
   m = 2k (d = 1); 74-80% / 90-95% at d = 2 (CONFIRMED-from-CI). In the cost model a PCIe prefetch with these values gains
   nothing while the CPU expert path is DRAM-bound and +4-10% when it is compute-bound (32 threads x 3.6 GB/s), about what
   the old assumptions gave; m = 2k is worse than m = k there because precision halves. Recall on the experts that actually
   miss, the number prefetch needs, is UNKNOWN (PROVISIONAL model output; section 7).
7. **Qwen3.8-Flash-Next keeps every expert in VRAM** on this server (residency 1.0 in ARCHITECTURE-ALTERNATIVES 9.12 and in
   `fitplan`), so expert-cache policy and prefetch do not apply to it (PROVISIONAL placement model).

## 1. Provenance

| item | value | label, source |
|---|---|---|
| workflow run | `traces-real` run 38079738013, one job `traces` (id 114293998620), conclusion success; step 8 "traces, probes, predictors and cache policies per model" 19:33:24-23:53:13 UTC | CONFIRMED: `gh run view 38079738013 --repo samnolak/408032gb --json ...` (read-only, this session), `provenance.txt`; the lead read the same times in the job log |
| code that ran | run head `7a0bf02`; between `7a0bf02` and `506c14c` (parent of the results commit) only `.github/workflows/traces-real.yml` and `scripts/traces/smoke_tiny.sh` changed, not `tools/g0`, `scripts/traces/run_real_traces.sh`, `REQUEST.json` | CONFIRMED: `git diff --stat 7a0bf02 0914244^ -- tools/g0 scripts/traces data/traces/real/REQUEST.json .github/workflows/traces-real.yml`, `provenance.txt` |
| results commit | `091424485311dec9faed34a3f6d946ab49e5b9aa` (github-actions bot), parent `506c14c68de3e7b7bd6b52f34ecfd3775becf30d` | CONFIRMED: `git rev-parse 0914244 0914244^`, `provenance.txt` |
| runner | ubuntu-24.04, 4 vCPU AMD EPYC 7763; 16 GB RAM | CPU CONFIRMED (`MANIFEST.json` "cpu", "threads": 4); RAM PROVISIONAL (GitHub's documented spec for public repositories, not measured) |
| models | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` rev `b17cb02`, Q4_K_M, 18,556,689,568 B (17.28 GiB), arch `qwen3moe`, 128 experts, top-8, 48 MoE layers (0-47), n_embd 2048. `unsloth/GLM-4.7-Flash-GGUF` rev `0d32489`, Q4_K_M, 18,312,339,808 B (17.05 GiB), arch `deepseek2`, 64 experts, top-4, 46 MoE layers (1-46), n_embd 2048. Both sha256 equal the HF LFS sha256 | CONFIRMED: `MANIFEST.json`, `trace.txt`, `probe.txt` (n_embd), trace headers (layer ids) |
| paging | both files are larger than the runner's RAM and were paged from disk; this affects run time, not routing | PROVISIONAL (RAM not measured) |
| llama.cpp | `ec7630a640789c393694fb194f1bbbf0369fc62d`, `-c 8192 -b 512 -ub 512`, CPU build with `GGML_NATIVE=ON` | CONFIRMED: `MANIFEST.json`, `run_real_traces.sh:51,62` |
| corpus | 16 CONSTRUCTED coding-agent conversations built from this repository's files (12 English, 4 Russian): one system prompt, a task from 9 templates, "I will read X", the file (<= 7,000 chars) as a tool result, a follow-up; rendered with each model's chat template by llama-server; 1,314-3,300 prompt tokens; greedy generation of at most 128 tokens | CONFIRMED: `scripts/traces/build_agent_prompts.py:19-77`, `CORPUS.json`, `ids/RENDER.json`, `trace.txt` |
| hold-out split | test = files with index % 4 == 3, i.e. c03, c07, c11, c15 in both models; train = the other 12 (static ranking, `popularity`, `transition`) | CONFIRMED: `run_real_traces.sh:56`, `cachesim.py:387-388` |
| scored generated tokens | 484 per model (c03 100 + 3 x 128) | CONFIRMED: `trace.txt` |
| probes | the 4 test conversations only: last 256 prompt tokens + every generated token = 1,508 recorded tokens per model; the `.probe` files were not packaged (`run_real_traces.sh:7`) and the GGUF was deleted (`:89`), so `predict_eval` cannot be re-run here | CONFIRMED: `probe.txt`, `predict.json` `recorded_tokens` |

## 2. What every table means (CONFIRMED: code at `0914244`, each cited line printed with `sed -n` before citing)

### 2.1 `predict.txt` (`tools/g0/predict_eval.py`)

* **Question.** While MoE layer l runs, how well can the experts that layer L = l + d will select be named in advance
  (`predict_eval.py:2`, `:17-23`)? `d` counts MoE layers: `L = pr["layers"][i + d]` (`:206`).
* **The number** in every predictor column is **recall of the true top-k**: per (token, layer) pair, |candidates ∩ true
  top-k| / k, averaged over pairs and weighted by pair count (`recall`, `:165-168`; weighting `:242`). It is not a cache hit
  rate and not precision. For a set of exactly m candidates, precision = recall x k / m (arithmetic from the definition).
* **m** = candidate set size, default {k, 2k, 4k} (`:174`): 8/16/32 for the Qwen proxy, 4/8/16 for the GLM proxy.
  **random** = m / n_expert (`:239`). **part**: `prompt` = the recorded prompt tail (last 256 prompt tokens,
  `expert_probe.cpp:212`), `gen` = generated tokens (`predict_eval.py:197-199`); decode prefetch depends on `gen`.
* **Predictors** (`:17-22`, code `:208-236`):
  * `pregate`: top-m of layer L's selection rule applied to W_L x_l, where x_l is layer l's router input (`ffn_norm`
    output, recorded as fp16, `expert_probe.cpp:2-4,106`) and W_L is layer L's router from the GGUF (`:208-213`). It needs
    nothing but the model.
  * `prev_token`: the experts layer L chose for the previous token, padded with popular experts up to m (`:218-222`).
  * `popularity`: the m most frequent experts of layer L in the 12 training traces (`:214-217`).
  * `transition`: training co-occurrence counts C[e, e'] between layer l's chosen experts and layer L's, summed over
    l's k experts (`:233-236`); **d = 1 only** (`:233`), hence `-` at d = 2.
  * `pregate+prev`: the first m - k pregate candidates united with the previous token's k experts (`:228`). **At m = k it
    holds no pregate candidate and equals `prev_token`**; the identical columns at m = k are this construction.
  * `prev_token` and `pregate+prev` are scored on recorded tokens with a recorded predecessor (`idx_prev`, `:200`), the
    others on all recorded tokens (the two sets differ by one token per file and part).
* Training data for `popularity` / `transition`: the 12 training traces (`run_real_traces.sh:56,64-65`). **Lead time**:
  with d = 1 the candidates are known one MoE layer before layer L routes (`:23`).

### 2.2 `cachesim.txt` (`tools/g0/cachesim.py`)

* **capacity** = fraction of each MoE layer's routed experts the VRAM tier holds; the simulator keeps
  `cap = round(capacity x n_expert)` experts per layer (`:316`): 13/32/64/96 of 128 (Qwen proxy) and 6/16/32/48 of 64 (GLM
  proxy) for 0.1/0.25/0.5/0.75. Global policies get one budget of `cap x n_layers` (layer, expert) slots (`:323`). The text
  table prints capacity with two decimals (`:415`), so 0.546/0.55 and 0.746/0.75 print alike; the JSON keys are exact
  (`fine_table.py` below reads them).
* **"hit rate on gen tokens"** = Σ hits / (k x MoE layers x generated tokens) over the held-out traces, pooled (every
  generated token weighs the same; `:339-347` per-layer, `:324-330` global): the share of routed-expert reads during decode
  that VRAM serves, not the share of tokens with all experts resident. `layer_min` / `layer_max` in the JSON are per-layer
  rates over **all** tokens of the test traces, prompt included (`:348-356`).
* **Policies** (`:8-22`): `static` = top-C experts per layer by frequency in the 12 training traces, all their tokens;
  `lru`; `lfu` (counts halved every 4,096 accesses of the layer, `:184-186`; never evicts an expert of the current token,
  `:191`); `tinylfu` (LRU with a frequency admission filter, `:169-170`); `hybrid` = C/2 static + C - C/2 LRU slots; `opt` =
  Belady MIN with bypass and a free preload of the first-used experts (`:105-137`), an offline bound for any per-layer policy;
  `lru_global` = one LRU over all layers; `lru_device` = one LRU per device over a contiguous layer split with the budget
  split by layer count (`:245-248`), the shape of llama.cpp `--moe-cache-mib` at that project's commit `10a60cf30`
  (`:18-21`, not our pin); `opt_global` = Belady bound for the shared budget. All are exact routing (`:8`).
* **Admission.** `lru`, `lfu` and the LRU half of `hybrid` admit every missed expert (`:164-172`, `:187-196`); `tinylfu` and
  `opt` may bypass; `static` never changes. In an engine every admission is one copy of one expert into VRAM (section 6.4).
* **warm**: per-layer dynamic policies start with the static set (`:147-150`, `:176`); global policies ignore it.
  **session**: the 4 test traces are concatenated in file order and every cache persists across them (`:209-222`,
  `:312-313`). **devices**: GPUs for `lru_device` (`:245`). **prompt_readonly** (default): prompt tokens do not update
  `lru_global` / `lru_device` (llama.cpp skips cache updates for ubatches > 32 tokens, `:23-24`, `:252-264`); **per-layer
  policies replay every prompt token, one token at a time, before generation**, so they enter decode warmed by the prompt.
* **Hold-out split**: test = files i % 4 == 3 (`:387-388`). Only `static` and the static half of `hybrid` use training data.
* **"distinct experts per layer per window"**: mean distinct experts per layer touched by w consecutive tokens, over all 16
  traces (`:395-396`, `:274-289`), against the independent-routing value E(1 - (1 - k/E)^w): the expert traffic of
  verifying a w-token speculation window at batch 1.

## 3. Self-checks (CONFIRMED-from-CI: `predict.txt` line 1 of each model)

| check | Qwen3-Coder-30B-A3B | GLM-4.7-Flash | what it tests (code) |
|---|---|---|---|
| `router_logits_max_rel_err` | 2.94e-4 | 1.65e-4 | max over (file, layer) of max abs(W_l x_l - logits_l) / max abs(logits_l); W dequantized from the GGUF, x stored fp16; fails above 0.02 (`predict_eval.py:179-182,189`) |
| `selection_rule_match` | 1.0 | 0.9999855841310115 | our selection rule on the recorded fp32 logits reproduces the recorded top-k per (token, layer) row; fails below 0.99 (`:183-186,189`) |
| `gating`, `groups` | 1 (softmax), [0, 0] | 2 (sigmoid), [1, 1] | read from the GGUF (`:80-86`) |
| `probe_equals_trace_rows` | 1.0 | 1.0 | the probe run (a second, separate run) selected the same experts as the trace run on every recorded row (`:148-162`) |

Rows (CONFIRMED arithmetic): 1,508 recorded tokens x 48 layers = 72,384 (Qwen); 1,508 x 46 = 69,368 (GLM);
1 - 1/69,368 = 0.9999855841310115, equal to the printed value in every digit, so **exactly one GLM row differs**.

Which row is UNKNOWN (no probe files). What can be said:

* It is **not** the group rule: `groups = [1, 1]` is one group, and both llama.cpp (`src/llama-graph.cpp:2105`
  `if (hparams.n_expert_groups > 1 && n_tokens > 0)` @ec7630a) and `predict_eval.py:113` skip group masking then. The rule in
  force is sigmoid + selection bias + top-4: `ggml_sigmoid` (`llama-graph.cpp:2069`), fp32 `ggml_add` of `exp_probs_b`
  (`:2088`), `ggml_argsort_top_k` (`:2131`), recorded at `cb(selected_experts, "ffn_moe_topk", il)` (`:2134`).
* `predict_eval` recomputes sigmoid + bias in float64 from the recorded fp32 logits (`predict_eval.py:99-104`) and breaks
  exact ties toward the lower index (stable argsort, `:124-125`); ggml computes in fp32. PROVISIONAL: a near-tie (or an
  exact fp32 tie) between the 4th and 5th score, resolved differently by the two arithmetics or by the tie order of
  `ggml_argsort_top_k` - the mechanism seen on the tiny qwen3moe, where the flipped rows were the 2nd and 3rd smallest
  k-th/(k+1)-th margins of 432 (`2026-10-10-traces-real-fix.md` §4).
* Effect: none on any reported number. `cachesim` reads the trace (what llama.cpp chose), and `predict_eval` scores every
  predictor against the recorded top-k (`:207`), not against its own rule; the row only shows that our rule and ggml's can
  disagree at a near-tie, which `pregate` inherits at the same rate (1 in 69,368).

**Generation lengths below 128** (CONFIRMED from code; the generated text is UNKNOWN):

* `expert_trace.cpp:204-214` takes the argmax token and stops **before feeding it** when `llama_vocab_is_eog` is true; the
  traced count is prompt + generated (`:221-222`); `expert_probe.cpp:230` does the same. The only other exit is a decode
  error, which returns 1 (`expert_trace.cpp:216-218`) and would have failed the job. So c03 (both models, 100 tokens) and
  GLM c00 (30 tokens) ended on a greedily chosen end-of-generation token. Traces store expert ids, not token ids, so why
  both models stop c03 at exactly 100 is UNKNOWN.
* Test side: c03 is held out in both models, so it weighs 100/484 = 20.7% of the scored tokens instead of 25%. Measured
  effect (section 5.6): negligible - GLM static at 0.607 is 0.770 pooled vs 0.7685 as an equal-weight mean of the 4 traces
  (LRU 0.917 vs 0.9175), although c03 is the lowest of the 4 for GLM static (0.740) and the highest for GLM LRU (0.924).
* Train side: GLM c00 is a training trace; its 30 generated tokens are 1.5% of its 1,954 tokens and only feed `static`,
  `hybrid`, `popularity`, `transition`.
* Corpus side (PROVISIONAL): early EOG means short answers; long agent outputs are not represented (section 8). Generation
  length itself matters a lot for cold-started caches (section 5.6).

## 4. `cachesim` re-run: exact reproduction (CONFIRMED)

Traces: `gunzip -c data/traces/real/<model>/traces/*.trace.gz` into `$S/<model>/traces/`. Same arguments as
`run_real_traces.sh:68-69`. Literal (`reproduce.txt`):

```
$ for m in qwen3-coder-30b-a3b:128 glm-4.7-flash:64; do ... python3 tools/g0/cachesim.py $S/$n/traces --n-experts $e --capacity 0.1,0.25,0.5,0.75 --holdout-every 4 -o $S/$n/base.json > $S/$n/base.txt ...; diff $S/$n/base.txt data/traces/real/$n/cachesim.txt && echo "$n: cachesim.txt identical"; cmp $S/$n/base.json data/traces/real/$n/cachesim.json && echo "$n: cachesim.json identical"; done
rc=0 seconds=205
qwen3-coder-30b-a3b: cachesim.txt identical
qwen3-coder-30b-a3b: cachesim.json identical
rc=0 seconds=83
glm-4.7-flash: cachesim.txt identical
glm-4.7-flash: cachesim.json identical
```
(the loop body is shortened here; the full command line is in `reproduce.txt`). No difference to explain.

## 5. Sensitivity (CONFIRMED for the proxies: commands and literal outputs in `runs-<model>.txt`, runs at `830106e` and `8435e8d`)

Driver: `jobs.sh` / `jobs2.sh` (one line per run) through `runjob.sh` (writes the command, the output and
`[exit=0 seconds=... commit=...]`); all 60 runs exited 0. `cachesim_sensitivity.py` calls `cachesim.evaluate` unchanged; its
last fold equals the `cachesim` CLI exactly (unit test `test_last_fold_reproduces_cachesim_cli`). "4 folds pooled" = every
conversation held out once (rotations i % 4 == r, r = 0..3), scored together; "min-max" = range over the 4 folds.

### 5.1 Fine capacity grid (`cs_fine`, cachesim's own split), gen-token hit

```
$ python3 tools/g0/cachesim.py $S/glm-4.7-flash/traces --n-experts 64 --capacity 0.05,0.1,...,0.5,0.546,0.55,0.6,0.607,0.65,0.7,0.746,0.75,0.8,0.85,0.9 --holdout-every 4 -o ...
capacity      static         lru         lfu     tinylfu      hybrid         opt  lru_global  lru_device  opt_global
    0.05       0.149       0.116       0.274       0.119       0.107       0.421       0.000       0.000       0.463
    0.10       0.225       0.366       0.386       0.185       0.282       0.587       0.331       0.337       0.617
    0.20       0.355       0.571       0.492       0.311       0.536       0.770       0.556       0.556       0.789
    0.30       0.456       0.692       0.576       0.412       0.645       0.854       0.674       0.673       0.869
    0.40       0.566       0.801       0.665       0.516       0.749       0.914       0.778       0.778       0.927
    0.50       0.661       0.867       0.736       0.609       0.830       0.945       0.843       0.839       0.956
    0.61       0.770       0.917       0.816       0.712       0.902       0.969       0.884       0.878       0.977
    0.70       0.851       0.947       0.881       0.800       0.941       0.981       0.898       0.894       0.987
    0.80       0.922       0.968       0.935       0.888       0.966       0.989       0.900       0.899       0.992
    0.90       0.978       0.989       0.981       0.967       0.988       0.996       0.900       0.900       0.998
$ python3 tools/g0/cachesim.py $S/qwen3-coder-30b-a3b/traces --n-experts 128 --capacity <same> ...
    0.05       0.149       0.130       0.382       0.086       0.098       0.520       0.000       0.000       0.554
    0.10       0.282       0.538       0.514       0.168       0.410       0.730       0.526       0.525       0.748
    0.20       0.464       0.751       0.625       0.323       0.692       0.875       0.748       0.744       0.886
    0.30       0.599       0.855       0.722       0.473       0.813       0.930       0.842       0.839       0.938
    0.40       0.720       0.912       0.816       0.629       0.893       0.959       0.886       0.885       0.965
    0.50       0.820       0.943       0.884       0.759       0.935       0.975       0.906       0.906       0.979
    0.61       0.903       0.963       0.936       0.866       0.961       0.986       0.915       0.915       0.988
    0.70       0.949       0.976       0.965       0.928       0.975       0.991       0.921       0.920       0.993
    0.80       0.975       0.987       0.983       0.967       0.986       0.995       0.922       0.922       0.996
    0.90       0.992       0.995       0.995       0.991       0.995       0.998       0.922       0.922       0.998
```
(excerpt: 10 of 21 rows per model; the `# cachesim` and `hit rate` lines, the second column-name line and the window lines
cut; capacity list shortened in the commands; the row "0.61" is capacity 0.607. Full output in `runs-*.txt`, exact-capacity rows for 0.546 / 0.607 / 0.746 in `derived.txt`.)

* At capacity 0.05 a layer holds fewer experts than one token uses (3 < 4 of 64 for GLM, 6 < 8 of 128 for Qwen), so the
  LRUs thrash: per-layer LRU falls below static, and the global LRU (prompt read-only) scores 0.000.
* EXPERIMENTS.md M1 (`EXPERIMENTS.md:56`) expects "25-35% of a layer's experts give >= 80% of accesses". Held-out with a
  static set the proxies need ~0.48 (Qwen) and ~0.64 (GLM) of the experts for 0.80 of decode reads; an LRU reaches 0.80 at
  ~0.24 (Qwen) and ~0.40 (GLM) (PROVISIONAL: linear interpolation between neighbouring rows of the full output, e.g. GLM
  static 0.755 at 0.60 and 0.809 at 0.65).

### 5.2 Hold-out variance (4 folds pooled; `se_folds_full`), capacity 0.607

| | static | lru | lfu | tinylfu | hybrid | opt | lru_global | lru_device | opt_global |
|---|---|---|---|---|---|---|---|---|---|
| GLM proxy, pooled | 0.770 | 0.908 | 0.837 | 0.752 | 0.895 | 0.965 | 0.872 | 0.867 | 0.972 |
| GLM, fold min-max | 0.754-0.786 | 0.902-0.917 | 0.816-0.856 | 0.712-0.793 | 0.890-0.902 | 0.962-0.969 | 0.867-0.884 | 0.860-0.878 | 0.968-0.977 |
| Qwen proxy, pooled | 0.908 | 0.969 | 0.953 | 0.906 | 0.967 | 0.988 | 0.922 | 0.922 | 0.991 |
| Qwen, fold min-max | 0.886-0.935 | 0.960-0.978 | 0.936-0.965 | 0.866-0.933 | 0.956-0.977 | 0.984-0.992 | 0.913-0.934 | 0.913-0.933 | 0.988-0.994 |

cachesim's own split (fold 3) is inside every range. Fold-to-fold spread of `static` is ±1.6 (GLM) / ±2.5 (Qwen) points.

### 5.3 R8 §0 item 4, pitfall 1: replay semantics (prompt replay vs cold vs warm start), 4 folds pooled

R8 (`R8-memory-tiers-expert-scheduling.md:40-45`) cites a paper (REPORTED via snippet) in which per-access vs fused-event
trace replay inflates recency policies by 27-29%. In `cachesim`, per-layer policies replay every prompt token one by one
before decode (section 2.2), the most favourable possible warm-up for recency; llama.cpp-like engines do not update the cache
on large prefill ubatches. Three views of the same traces (`se_folds_full`, `se_folds_genonly`, `se_folds_genonly_warm`,
`se_cold_folds_genonly`):

| GLM proxy, capacity 0.607 | lru | lfu | tinylfu | hybrid | lru_global / lru_device |
|---|---|---|---|---|---|
| full view (cachesim: prompt replayed per token) | 0.908 | 0.837 | 0.752 | 0.895 | 0.872 / 0.867 |
| gen only, cold (cache empty at the first generated token) | 0.860 | 0.862 | 0.851 | 0.874 | 0.872 / 0.867 |
| gen only, warm (static set preloaded, prompt unseen) | 0.901 | 0.903 | 0.892 | 0.890 | 0.872 / 0.867 |
| cold-start bound (`cold` column) | 0.895 | | | | |

| Qwen proxy, capacity 0.607 | lru | lfu | tinylfu | hybrid | lru_global / lru_device |
|---|---|---|---|---|---|
| full view | 0.969 | 0.953 | 0.906 | 0.967 | 0.922 / 0.922 |
| gen only, cold | 0.919 | 0.919 | 0.914 | 0.946 | 0.922 / 0.922 |
| gen only, warm | 0.966 | 0.966 | 0.961 | 0.964 | 0.922 / 0.922 |
| cold-start bound | 0.926 | | | | |

* Per-token prompt replay is worth ~5 points of LRU over a cold start (GLM 0.908 vs 0.860; Qwen 0.969 vs 0.919), about 5%
  relative - much less than the paper's 27-29% (different semantics; REPORTED there). A static preload recovers almost all
  of it without seeing the prompt (0.901 / 0.966). `--warm` in the full view changes nothing at all (`cs_warm` equals the
  baseline in every cell, both models): the prompt replay overwrites the preload.
* Cold LRU cannot beat the bound 0.895 / 0.926 here: over a GLM-proxy test generation a layer meets ~48 of its 64 experts
  for the first time (1 - 0.900 = 0.100 of the 4 x 121 reads per layer of an average test trace; `se_cold_last_full`).
  That bound depends on generation length, not on the policy (5.6).
* `lfu` and `tinylfu` look much worse than `lru` in the full view and equal to it when cold. PROVISIONAL explanation: their
  frequency counts are dominated by the ~2,000 prompt tokens, whose routing differs from decode routing (5.4).

### 5.4 R8 §0 item 4, pitfall 2: template contamination / which tokens train the static profile (`se_folds_train*`)

All 16 conversations share one system prompt and 9 task templates (`build_agent_prompts.py:19-37`), so prompts of train and
test overlap; the scored tokens are generated ones only, so template tokens are never scored. They do enter the static
ranking, which `cachesim` trains on all tokens of the training traces (~94% of them prompt tokens). Static hit at 0.607,
4 folds pooled:

| static ranking trained on | GLM proxy | Qwen proxy |
|---|---|---|
| all tokens (cachesim) | 0.770 (0.754-0.786) | 0.908 (0.886-0.935) |
| prompt tokens only | 0.733 (0.711-0.757) | 0.887 (0.867-0.910) |
| generated tokens only | **0.899** (0.858-0.931) | **0.943** (0.915-0.958) |

* Prefill routing and decode routing differ: the all-token profile (~94% prefill tokens) loses 12.9 (GLM) / 3.5 (Qwen)
  points of decode hit against a decode-token profile, a prefill-only one 16.6 / 5.6. The same shows in the baseline: GLM
  static scores 0.773 on the test prompt tokens but 0.661 on the generated ones at 0.5 (`cachesim.json` "prompt" vs "gen").
  The swing is of the same order as the paper's template-contamination effect (19-32 points, REPORTED).
* PROVISIONAL caveat in the other direction: the decode-trained profile is trained on answers to the same 9 templates under
  the same system prompt; on diverse real sessions it may score lower. UNKNOWN by how much.
* Hybrid moves less (GLM 0.892 / 0.895 / 0.915; Qwen 0.966 / 0.967 / 0.969 for prompt / all / gen training).

### 5.5 Warm, session, prompt updates, devices (`cs_*`, cachesim's split)

| GLM proxy, capacity 0.75 (0.5) | per-layer policies | lru_global | lru_device |
|---|---|---|---|
| baseline | as `cachesim.txt` | 0.900 (0.843) | 0.898 (0.839) |
| `--warm` | identical to baseline | 0.900 | 0.898 |
| `--session` | lfu +0.002, tinylfu +0.009, others identical | 0.958 (0.868) | 0.955 (0.864) |
| `--prompt-updates` | identical | 0.967 (0.880) | 0.964 (0.874) |

Qwen proxy at 0.75: `--session` lru_global 0.922 -> 0.967, `--prompt-updates` 0.922 -> 0.984; per-layer policies change by
at most 0.011 (tinylfu with `--session` at 0.5).

**Devices** (`--policies lru_global,lru_device --devices D`, capacity 0.607): GLM lru_device 0.884 (1 device) / 0.881 (2) /
0.877 (3) / 0.878 (4) / 0.876 (8); with `--prompt-updates` 0.932 / 0.929 / 0.924 / 0.925 / 0.924. Qwen: 0.915 for every D
(0.966-0.965 with prompt updates). Splitting one LRU per GPU costs at most 0.8 points.

**The 0.900 / 0.922 plateau** (`se_cold_last_full`, literal):

```
$ python3 scripts/traces/cachesim_sensitivity.py $S/glm-4.7-flash/traces --n-experts 64 --capacity 0.607,0.75,1.0 --policies lru_global,lru_device
fold  test                                gen  capacity  cap  lru_global  lru_device        cold
   3  c03,c07,c11,c15                     484     0.607   39       0.884       0.878       0.900
   3  c03,c07,c11,c15                     484     0.750   48       0.900       0.898       0.900
   3  c03,c07,c11,c15                     484     1.000   64       0.900       0.900       0.900
$ python3 scripts/traces/cachesim_sensitivity.py $S/qwen3-coder-30b-a3b/traces --n-experts 128 --capacity 0.607,0.75,1.0 --policies lru_global,lru_device
   3  c03,c07,c11,c15                     484     0.607   78       0.915       0.915       0.922
   3  c03,c07,c11,c15                     484     0.750   96       0.922       0.922       0.922
   3  c03,c07,c11,c15                     484     1.000  128       0.922       0.922       0.922
```
(header lines cut from the second block.) With `prompt_readonly` these caches are empty when generation starts, so from
capacity ~0.75 on they sit exactly on the compulsory-miss bound: the columns measure 100-128-token generations, not the
policy. With caches kept across requests (`--session`) or filled by the prompt they reach 0.958-0.984.

### 5.6 Per trace and generation length (`se_pertrace`, `se_folds_*_gencap*`, `se_cold_*`)

Per test trace, capacity 0.607 (cachesim's split): GLM static c03 0.740 / c07 0.772 / c11 0.787 / c15 0.775, lru 0.924 /
0.914 / 0.918 / 0.914; Qwen static 0.939 / 0.948 / 0.854 / 0.878, lru 0.988 / 0.986 / 0.947 / 0.938. c03 (100 tokens) is
the lowest GLM static and the highest GLM LRU trace, within the spread of the others for Qwen.

Every test generation cut to its first N tokens, capacity 0.607, 4 folds pooled:

| | N = 30 | N = 64 | all (<= 128) |
|---|---|---|---|
| GLM static | 0.782 | 0.779 | 0.770 |
| GLM lru, full view | 0.878 | 0.894 | 0.908 |
| GLM lru, cold / cold bound | 0.695 / 0.699 | 0.801 / 0.819 | 0.860 / 0.895 |
| GLM lru_device (cachesim semantics) | 0.698 | 0.807 | 0.867 |
| Qwen static | 0.925 | 0.920 | 0.908 |
| Qwen lru, cold / cold bound | 0.816 / 0.816 | 0.882 / 0.884 | 0.919 / 0.926 |

Static drifts down by ~1-2 points as generation goes on (PROVISIONAL reading: decode routing moves away from the
prompt-dominated profile); cold-started caches gain 10-17 points from 30 to 128 tokens and are still rising. **So with
128-token generations the cold-start rows understate dynamic policies for long agent outputs, and the full-view rows depend
on per-token prompt replay; the truth for long sessions lies between them and is UNKNOWN until long generations are
traced.**

### 5.7 What changes the conclusions

1. **Static vs dynamic ranking flips with the warm-up assumption.** Full view: lru > hybrid > static at every target
   capacity. Cold gen-only: GLM hybrid 0.874 > lru 0.860 > static 0.770; Qwen hybrid 0.946 > lru 0.919 ≈ static 0.908, and
   at 0.746 Qwen static 0.966 beats cold lru 0.924. With 30-token generations static beats cold LRU on both proxies.
   **`hybrid` is never far from the best in any view** (GLM 0.895 / 0.874 / 0.890; Qwen 0.967 / 0.946 / 0.964).
2. **Training the static profile on decode tokens** raises GLM static from 0.770 to 0.899 at 0.607 - level with a warmed
   LRU (0.901-0.908) at zero admission traffic (section 6.4).
3. `lru_global` / `lru_device` in `cachesim.txt` are cold-start numbers (5.5); with sessions or prompt updates they rise by
   4-7 points at 0.75.
4. Device count, `--warm` in the full view, and the c03 weighting change nothing material (<= 0.8 points).

## 6. Mapping to the targets (PROVISIONAL as a whole: proxy routing applied to target placement)

### 6.1 The proxy gap

| | Qwen3-Coder-30B-A3B (proxy) | GLM-4.7-Flash (proxy) | GLM-5.3-Flash (target 1) | Qwen3.8-Flash-Next (target) |
|---|---|---|---|---|
| llama.cpp graph | `qwen3moe` | `deepseek2` | `glm5-next` | `qwen4exp` |
| routed experts / top-k | 128 / 8 | 64 / 4 | 288 / 8 | 512 / 10 |
| k / E | 6.25% | 6.25% | 2.78% | 1.95% |
| MoE layers | 48 | 46 | 42 | 48 |
| router | softmax (gating 1) | sigmoid + selection bias, one group (gating 2, groups [1, 1]) | sigmoid + `noaux_tc` bias, routed scale 2.5 | softmax |
| hidden | 2048 | 2048 | 4096 | 2560 |
| previous-token expert reuse | 0.471 (`prev_token`, gen, m = k) | 0.336 | 24.6% (REPORTED) | UNKNOWN |
| source | MANIFEST.json, trace.txt, probe.txt, predict.json (CONFIRMED) | same | `models/specs/glm-5.3-flash.json:2-15`, `docs/models/glm-5.3-flash.md:14-15` (HF config, CONFIRMED there); reuse: `docs/reassessment/ROOFLINE.md:29-31` (base A on real weights, 3x3090 host, REPORTED) | `models/specs/qwen3.8-flash-next.json:2-15` (Strata layout, CONFIRMED there); router: `ref/qwen4exp.py:63` (oracle of the llama.cpp graph) |

Also different, none measured: corpus (CONSTRUCTED, 16 conversations, 1.3-3.3K-token prompts), generation (greedy,
<= 128 tokens), weights (Q4_K_M), numerics (CPU). A 288-expert top-8 layer keeps 2.25x more experts resident than a
64-expert layer at the same fraction and has a smaller k/E; whether its held-out hit at a given fraction is higher or lower
than the proxies' is UNKNOWN. The one target number available, 24.6% previous-token reuse, is below both proxies (0.336,
0.471): less token-to-token locality in absolute terms, which is what LRU lives on. The two proxies differ from each other
by 2-14 points at the same capacity (opt to static, section 5.2), a rough lower bound on how far a third model can sit.

### 6.2 Expert capacity per layer in VRAM on 4x RTX 4080 Super 32 GB (PROVISIONAL placement models)

| target, weights, layout | residency (= capacity per layer if the expert cache is split evenly) | source |
|---|---|---|
| GLM-5.3-Flash UD-Q4_K_XL (4.89 / 8.04 bpw), 262,144 tokens, KV once | **0.607** (107.7 of 177.4 GiB) | ARCHITECTURE-ALTERNATIVES 9.12 (`ARCHITECTURE-ALTERNATIVES.md:1358`) |
| same, KV replicated on 4 GPUs (TP layouts) | **0.546** | same (`:1359`) |
| GLM-5.3-Flash EXL3 4.05 bpw | **0.746** | same (`:1360`) |
| GLM-5.3-Flash Q4_K 4.5 bpw (uniform), `fitplan` | 0.645 | `fitplan.txt` (`python3 tools/fitplan.py models/glm-5.3-flash.json --quant Q4_K --ctx 262144`) |
| Qwen3.8-Flash-Next UD-Q4_K_XL / Q4_K | **1.000** (71.7 GiB of experts, 108.0 GiB free) | ARCHITECTURE-ALTERNATIVES 9.12 (`:1361`); `fitplan.txt` |

`fitplan.txt` is the literal output with the two `ceiling` lines filtered out (`| grep -v ceiling`): they are bandwidth-only
upper bounds that ignore compute, launch and PCIe costs, and this file does not use them.

### 6.3 Hit at that capacity, per policy, against the planning assumption (proxy values CONFIRMED for the proxies; transfer PROVISIONAL)

4 folds pooled (fold range for `static`), gen tokens; "cold / warm" = gen-only views of 5.3; "decode-trained" = 5.4.

| capacity (target layout) | planning hit | GLM proxy static | static, decode-trained | hybrid full / cold | lru full / cold / warm | lru_device (cachesim) | opt bound | Qwen proxy static / lru |
|---|---|---|---|---|---|---|---|---|
| 0.607 (UD-Q4_K_XL) | 0.70 / 0.75 / 0.80 (`ARCHITECTURE-ALTERNATIVES.md:100`); R9's stop/go 0.75 (`R11-verification.md:118-119`) | **0.770** (0.754-0.786) | 0.899 | 0.895 / 0.874 | 0.908 / 0.860 / 0.901 | 0.867 | 0.965 | 0.908 / 0.969 |
| 0.546 (TP, KV x4) | GLM-5.3 itself, static held-out at 0.556: 0.701 (code) - 0.773 (agentic), third-party measurement (R11 C03, `R11-verification.md:25,104-119`) | **0.708** (0.690-0.727) | 0.867 | 0.859 / 0.841 | 0.881 / 0.840 / 0.872 | 0.847 | 0.952 | 0.866 / 0.959 |
| 0.746 (EXL3 4.05) | 0.80 / 0.85 / 0.90 (`ARCHITECTURE-ALTERNATIVES.md:113-120`, extrapolated there) | **0.895** (0.881-0.904) | 0.951 | 0.950 / 0.922 | 0.953 / 0.885 / 0.950 | 0.891 | 0.984 | 0.966 / 0.985 |

Reading (PROVISIONAL):

* With the placement the plan assumes (static, held-out), **the GLM proxy lands inside the planning band** (0.770 at 0.607,
  0.895 at 0.746) and at 0.546 on top of the only GLM-5.3-Flash measurement (0.708 vs 0.701-0.773). Nothing here says the
  band is too optimistic; nothing here measures the target.
* **The band is conservative for anything better than a prefill-trained static set**: decode-trained static, hybrid and
  warmed LRU give 0.87-0.91 at 0.607 on the GLM proxy and 0.94-0.97 on the Qwen proxy. Whether GLM-5.3-Flash gets the same
  uplift is UNKNOWN (6.1: less previous-token locality on the target).
* Per-layer spread is wide: GLM static at 0.607 ranges 0.653-0.904 over layers, LRU 0.709-0.944 (all test tokens,
  `derived.txt`), so a byte budget split evenly by layer leaves hit on the table (R8 §7.4 already proposes byte-sized,
  per-layer slots).
* For Qwen3.8-Flash-Next the question does not arise at residency 1.0; the Qwen-proxy curves would matter only if experts
  stopped fitting (more KV, another quant).

What the cost model makes of these hits, `tools/ceilings/tier_cost.py` GLM-5.3-Flash preset, one token per step, overlap on
(PROVISIONAL model ceilings, `tier_cost.txt`; columns h_v, DRAM B/s, tok/s, ms per step; excerpt: 12 of the 30 rows of the
first of two commands, its column-name line cut):

```
$ python3 tools/ceilings/tier_cost.py --model glm  --sweep h_v=0.546,0.607,0.70,0.75,0.77,0.80,0.86,0.867,0.895,0.899,0.901,0.908,0.922,0.965,0.969 --sweep B_dram=102.4e9,204.8e9 | cut -d'|' -f1-4
# GLM-5.3-Flash: n_moe 42, S_e 15.38 MB, K 8, E 288, t_pre 0.308 ms, t_shared 0.046 ms, t_other 1.61 ms, ctx 262144; PROVISIONAL model output
0.607 | 1.024e+11 | 22.87 | 43.72
0.607 | 2.048e+11 | 33.84 | 29.55
0.75 | 1.024e+11 | 29.93 | 33.41
0.75 | 2.048e+11 | 40.99 | 24.39
0.77 | 1.024e+11 | 31.28 | 31.97
0.77 | 2.048e+11 | 41.94 | 23.84
0.867 | 1.024e+11 | 40.05 | 24.97
0.867 | 2.048e+11 | 40.4 | 24.75
0.899 | 1.024e+11 | 39.92 | 25.05
0.899 | 2.048e+11 | 39.92 | 25.05
0.908 | 1.024e+11 | 39.78 | 25.14
0.908 | 2.048e+11 | 39.78 | 25.14
```
With 8 channels the model's ceiling peaks near hit 0.77-0.80 (41.9 tok/s) and then falls slightly: past that point the GPU
side of each layer is the bottleneck and extra residency only moves work onto it (`tier_cost.py:36-39`, a known property of
the overlap model). With 4 channels the hit matters up to ~0.87 (22.9 at 0.607 -> 29.9 at 0.75 -> 40.1 at 0.867). In this
model the proxy's static hit already reaches the 8-channel balance point; the dynamic policies' extra hit pays only with
4 channels or a slower CPU (the run with 32 threads x 3.6 GB/s gives 30.3 / 37.6 / 38.9 / 41.1 at 0.607 / 0.75 / 0.77 / 0.80,
8 channels, same file). What the model ignores: admission traffic (6.4), per-layer spread (one h_v for all layers), DRAM
contention between links, kernel efficiency.

### 6.4 Admission traffic of dynamic policies (PROVISIONAL arithmetic, `admission.py` -> `derived.txt`)

LRU-type caches admit every miss (section 2.2), i.e. copy one expert to VRAM per miss, and the CPU that computes the miss in
place reads the same bytes again. GLM-5.3-Flash (k 8, 42 MoE layers, 15.38 MB per expert):

```
hit   misses/layer/token  copies/token  GB/token  GB/s@30tok/s  GB/s@40tok/s  per link (4 links) @40
0.770                1.84          77.3     1.189          35.7          47.5       11.9
0.860                1.12          47.0     0.723          21.7          28.9        7.2
0.908                0.74          30.9     0.475          14.3          19.0        4.8
```
(3 of 6 rows; 30 and 40 tok/s are inputs, not measurements.) At the warmed-LRU hit of the GLM proxy that is ~0.5 GB of
copies per token: within PCIe 4.0 x16 per link, but every copy is also a DRAM read the CPU expert path competes with
(~19 GB/s against ~143 GB/s usable at 8 channels in tier_cost's defaults; double share at 4 channels). A static (or
decode-trained static) placement has none of it; Strata's adaptive tier rate-limits it (R8 §9.3). `tier_cost` does not model
admission traffic.

## 7. Prediction for prefetch

### 7.1 Measured recall and implied precision (recall CONFIRMED-from-CI; precision = recall x k / m, arithmetic)

`pregate`, generated tokens, 4 held-out conversations (484 generated tokens per model):

| | d = 1, m = k | d = 1, m = 2k | d = 2, m = k | d = 2, m = 2k |
|---|---|---|---|---|
| Qwen proxy (k 8, E 128): recall / precision | 0.858 / 0.858 | 0.978 / 0.489 | 0.804 / 0.804 | 0.953 / 0.477 |
| GLM proxy (k 4, E 64): recall / precision | 0.814 / 0.814 | 0.945 / 0.473 | 0.742 / 0.742 | 0.895 / 0.448 |
| best predictor without pregate, same cell (Qwen / GLM) | prev_token 0.471 / transition 0.343 | transition 0.646 / 0.523 | prev_token 0.475 / 0.341 | prev_token 0.568 / 0.433 |

Reading (PROVISIONAL unless marked):

* Cross-layer pre-gating has 1.5-2.4x the recall of the best predictor without it and 4-5x that of `popularity` at m = k,
  on both proxies (CONFIRMED-from-CI arithmetic on the table), and sits inside the range R8 §7.1 collects for other models
  (REPORTED there: 78.8% top-k one layer ahead; 0.85-0.90 for top-8 within top-16 one layer ahead, 0.76-0.83 two layers
  ahead).
* d = 2 costs 5-7 points of recall at m = k and 2.5-5 at m = 2k (CONFIRMED-from-CI arithmetic).
* **Prefetch needs recall on the experts that are not already resident** - only those are copied; `tier_cost`'s
  `p_recall` is defined that way (`tier_cost.py:99`). The numbers above are recall over all routed experts; conditional on a
  miss they are lower by an UNKNOWN amount, because misses are the experts the cache did not expect. One case is exact:
  with per-layer LRU and capacity >= k the previous token's k experts are always resident, so `prev_token` has recall 0 on
  LRU misses (CONFIRMED by construction, `cachesim.py:161-172`). The only measured conditional value is Strata's community
  probe: recall 27-34%, precision ~8% on the non-resident fetch set (REPORTED, R8 §0 item 3). The joint pregate-plus-cache
  simulation was not possible here (probes not packaged; section 8).
* Link capacity: one PCIe 4.0 x16 link lands ~0.7 GLM-5.3-Flash experts per 0.44 ms layer (R11 C24 arithmetic; `tier_cost`
  copies 0.63-0.70 per layer at h_v 0.61 below). Misses per layer are 8 (1 - hit): 3.1 at 0.61, 2.0 at 0.75, 1.8 at 0.77, 0.74 at 0.908.
  A d = 1 PCIe prefetch can cover a minority of misses at planning hits and most of them only near hit 0.9 - where, in the
  cost model, the CPU side is no longer the bottleneck (7.2).

### 7.2 The cost model with measured proxy values next to the old assumptions (PROVISIONAL model output; not a measurement, not a forecast)

`tools/ceilings/tier_cost.py`, GLM-5.3-Flash preset (S_e 15.38 MB, 42 MoE layers, 262,144-token context), one token per
step, 8 DRAM channels, `prefetch_opt` on (the scheduler copies the best number of predicted experts, `tier_cost.py:103,
184-196`), W = d x 0.44 ms (R8's one-layer lead at 50 tok/s; shorter than the layer time at the speeds below, so it understates
the lead). Proxy rows use the GLM-proxy `pregate` recall as `p_recall` and recall x k / m as `pi_prec`, i.e. they assume the
all-expert recall holds on the off-VRAM experts (optimistic, 7.1); the last row is Strata's off-VRAM probe. Driver
`tier_pred.sh` (a loop over `python3 tools/ceilings/tier_cost.py --model glm --set ... --json`); literal excerpt of
`tier_cost.txt` (three of ten blocks; the seven cut blocks are the 32 x 8 GB/s default CPU and 46 threads x 3.6 GB/s, each
at h_v 0.61 / 0.75 / 0.908, and 32 x 3.6 GB/s at h_v 0.908 - every row of each cut block equals its block's `none` row:
34.0, 41.0, 39.8 tok/s):

```
$ HV="0.61 0.75 0.908" bash tier_pred.sh
## CPU experts: 32 thr x 3.6 GB/s; h_v 0.61 h_r 0.39; 8 ch DRAM; GPU prefetch (prefetch_opt)
none                                             30.4 tok/s  step  32.88 ms  copies/layer 0.00  t_cpu 0.437 ms  t_ssd 0.000 ms
old R8: p .85 pi .50 d=1                         32.3 tok/s  step  30.97 ms  copies/layer 0.68  t_cpu 0.391 ms  t_ssd 0.000 ms
old 4b: p .95 pi .90 W 1.3 ms                    33.6 tok/s  step  29.72 ms  copies/layer 0.63  t_cpu 0.361 ms  t_ssd 0.000 ms
proxy pregate d=1 m=k: p .814 pi .814            33.4 tok/s  step  29.97 ms  copies/layer 0.64  t_cpu 0.367 ms  t_ssd 0.000 ms
proxy pregate d=1 m=2k: p .945 pi .473           32.2 tok/s  step  31.07 ms  copies/layer 0.68  t_cpu 0.393 ms  t_ssd 0.000 ms
proxy pregate d=2 m=k: p .742 pi .742            33.1 tok/s  step  30.19 ms  copies/layer 0.65  t_cpu 0.373 ms  t_ssd 0.000 ms
proxy pregate d=2 m=2k: p .895 pi .448           32.1 tok/s  step  31.15 ms  copies/layer 0.69  t_cpu 0.395 ms  t_ssd 0.000 ms
Strata probe, off-VRAM set: p .30 pi .08 d=1     30.7 tok/s  step  32.56 ms  copies/layer 0.70  t_cpu 0.429 ms  t_ssd 0.000 ms
## CPU experts: 32 thr x 3.6 GB/s; h_v 0.75 h_r 0.25; 8 ch DRAM; GPU prefetch (prefetch_opt)
none                                             37.6 tok/s  step  26.60 ms  copies/layer 0.00  t_cpu 0.287 ms  t_ssd 0.000 ms
old R8: p .85 pi .50 d=1                         39.4 tok/s  step  25.38 ms  copies/layer 0.44  t_cpu 0.258 ms  t_ssd 0.000 ms
old 4b: p .95 pi .90 W 1.3 ms                    40.7 tok/s  step  24.57 ms  copies/layer 0.40  t_cpu 0.239 ms  t_ssd 0.000 ms
proxy pregate d=1 m=k: p .814 pi .814            40.4 tok/s  step  24.74 ms  copies/layer 0.41  t_cpu 0.243 ms  t_ssd 0.000 ms
proxy pregate d=1 m=2k: p .945 pi .473           39.3 tok/s  step  25.44 ms  copies/layer 0.44  t_cpu 0.259 ms  t_ssd 0.000 ms
proxy pregate d=2 m=k: p .742 pi .742            40.2 tok/s  step  24.88 ms  copies/layer 0.41  t_cpu 0.246 ms  t_ssd 0.000 ms
proxy pregate d=2 m=2k: p .895 pi .448           39.2 tok/s  step  25.49 ms  copies/layer 0.44  t_cpu 0.261 ms  t_ssd 0.000 ms
Strata probe, off-VRAM set: p .30 pi .08 d=1     37.9 tok/s  step  26.38 ms  copies/layer 0.48  t_cpu 0.282 ms  t_ssd 0.000 ms
## NVMe -> RAM staging only (prefetch_to=ram): h_v 0.61, h_r 0.34, 5% of routed experts on NVMe, defaults
none                                             22.2 tok/s  step  45.03 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.371 ms
old R8: p .85 pi .50 d=1                         27.1 tok/s  step  36.93 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.178 ms
old 4b: p .95 pi .90 W 1.3 ms                    33.1 tok/s  step  30.23 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.019 ms
proxy pregate d=1 m=k: p .814 pi .814            30.9 tok/s  step  32.37 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.070 ms
proxy pregate d=1 m=2k: p .945 pi .473           26.8 tok/s  step  37.37 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.189 ms
proxy pregate d=2 m=k: p .742 pi .742            29.8 tok/s  step  33.50 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.097 ms
proxy pregate d=2 m=2k: p .895 pi .448           32.2 tok/s  step  31.10 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.039 ms
Strata probe, off-VRAM set: p .30 pi .08 d=1     22.9 tok/s  step  43.74 ms  copies/layer 0.00  t_cpu 0.355 ms  t_ssd 0.340 ms
```

Reading (PROVISIONAL, conditional on the model and on recall holding on misses):

* **PCIe prefetch to VRAM**: nothing while the CPU expert path is DRAM-bound (default 32 x 8 GB/s, and 46 x 3.6 GB/s: every
  row equals `none`); with a compute-bound CPU (32 x 3.6 GB/s) the measured proxy values give +9.9% (h_v 0.61) / +7.4%
  (0.75), against +6.3% / +4.8% for R8's old p 0.85 / pi 0.5 and +10.5% / +8.2% for the optimistic 0.95 / 0.9; nothing at
  h_v 0.908. **m = k beats m = 2k** in every GPU-prefetch block: doubling the candidate set buys 13 points of recall but
  halves precision, and wasted copies cost DRAM bandwidth. d = 2 at m = k is within 0.3 tok/s of d = 1. With Strata's off-VRAM recall the gain
  is +1%. R8's conclusion (compute misses on the CPU in place; prediction is not a lever on the GLM decode ceiling) stands
  with the measured values; the 4b result (prefetch substitutes for missing CPU cores, at most ~11%) is reproduced - the
  "old 4b" rows equal 4b's 33.6 / 40.7.
* **NVMe -> RAM staging** is where recall pays in this model (5% of experts on NVMe: 22.2 -> 30.9 tok/s at d = 1, m = k;
  32.2 at d = 2, m = 2k, where background reads have the longer lead - here, unlike GPU prefetch, m = 2k can win).
  GLM-5.3-Flash at UD-Q4_K_XL has no NVMe-resident experts in the placement of 6.2; this matters for DeepSeek-V4.1-Flash
  (48.6 GiB of experts on NVMe in the same table, `ARCHITECTURE-ALTERNATIVES.md:1365`).
* `accept` / speculation: the proxies' 2-token windows touch 6.72 of 7.75 (GLM) and 12.24 of 15.50 (Qwen) independent-routing
  distinct experts (`cachesim.txt` window line). For GLM-5.3-Flash the cost model's `U` for one MTP draft is 15.78 under
  independent routing; with the REPORTED 24.6% previous-token reuse it is 8 + 8 x 0.754 = 14.03, with the GLM proxy's overlap
  13.43. `tier_cost.txt` (n_tok 2, accept 0.7, h_v 0.61, 8 channels): 39.43 / 42.46 / 43.61 tok/s for U = independent / 14.03 /
  13.43 (PROVISIONAL model output).

## 8. What these traces cannot say (NOT RUN / UNKNOWN)

| question | status | what would answer it |
|---|---|---|
| routing of GLM-5.3-Flash (288 experts, top-8, sigmoid + noaux_tc) and Qwen3.8-Flash-Next (512, top-10, softmax) | NOT RUN: the targets do not fit a CI runner; the proxies differ in E, k/E, layer count and training | `g0-expert-trace` on target weights (GLM-5.3 UD-Q4_K_XL exists on host `epyc-ai`; it needs a llama.cpp build with `glm5next` there) |
| hit on the customer's real agent sessions (EXPERIMENTS.md M1/M2) | NOT RUN: the corpus is CONSTRUCTED from this repository's files with one system prompt and 9 task templates | the customer's own sessions rendered to ids (`render_ids.py`) and replayed |
| long contexts (32K-262K tokens) | UNKNOWN: prompts are 1,314-3,300 tokens; routing drift with context length not measured | the same tools with long prompts and `-c` to match |
| long generations (agent patches of 1-4K tokens) | UNKNOWN: at most 128 generated tokens; 3 of 32 traces (c03 in both models, GLM c00) stopped earlier at EOG; cold-start rows are still rising at 128 (5.6) | `G0_GEN_TOKENS` 1024+ |
| sampling at T > 0 | UNKNOWN: greedy only; a third-party report says sampling flattens routing (R11 C03, REPORTED) | a sampled run (the tools would need a sampler) |
| routing of the BF16/FP8 originals or other quants | UNKNOWN: Q4_K_M only (`cachesim.json` label: "routing of a quantized model may differ from the original") | the same ids on a second quant, traces compared row by row |
| GPU numerics vs these CPU traces | UNKNOWN on real models; batch-size rounding alone flipped 4 of 780 rows on a tiny model (`2026-10-10-traces-real-fix.md` §4) | a GPU run of the same ids |
| concurrent sessions / batch > 1 | NOT RUN: `--session` is one user's sequential requests | interleaved traces |
| predictor recall **on the experts that miss** VRAM (what prefetch needs) | UNKNOWN: probes not packaged, so pregate cannot be joined with a cache simulation | keep the `.probe` files, or compute the conditional recall inside the CI job |
| MTP / draft-token routing | NOT RUN: the tools trace the main decoder's MoE layers only | trace the MTP layer on GLM-5.3 |
| time: tok/s, prefill, cost of cache admission and prefetch copies | NOT MEASURED: everything in 6-7 is model output | the server (R0/R1) |
| MiMo-V2.6-Flash, DeepSeek-V4.1-Flash | NOT RUN | - |

## 9. Files and tests

`docs/evidence/2026-10-10-traces-real-results/`: `provenance.txt` (run metadata, git checks), `reproduce.txt` (section 4),
`runs-qwen3-coder-30b-a3b.txt` and `runs-glm-4.7-flash.txt` (all 30 runs per model, literal, with command, exit code,
seconds, commit), `cachesim-fine-<model>.json` (fine grid, exact capacities), `derived.txt` (`fine_table.py`, `admission.py`),
`tier_cost.txt` (prediction blocks, h_v sweeps, MTP `U` rows), `fitplan.txt`, and the drivers `jobs.sh`, `jobs2.sh`,
`runjob.sh`, `tier_pred.sh` (their paths point at the agent's scratch directory `$S`).

Unit tests: `tests/test_cachesim_sensitivity.py` (8 tests, synthetic traces); full suite in the final report of this task.
