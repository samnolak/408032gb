# N24: owner decisions - synthetic-run rule of the verification pack, V6 tolerance, GLM SwiGLU clamp, epyc-ai GGUF host check, R0 planning docs - 2026-10-11 (sandbox CPU, no GPU)

Task N24 (role perf-bench). Branch `claude/n24-owner-decisions`, cut from `claude/n19-qwen4exp-multiseq` at `e71b241`
(session `session_01EyB9XrhA3NTRjjQx1A9qMh`). The owner's decisions were relayed by the lead-architect in the task text;
there is no written customer document for them in the repository (UNKNOWN as a primary source). `<scratch>` stands for
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; literal blocks are pasted with only
that substitution. **Nothing here is a GPU, speed or real-weight quality result**: every run is a CPU run on tiny
random-weight models or on header bytes; wall times are not measurements.

| piece | file | commit |
|---|---|---|
| synthetic-run rule, V6 near-tie tolerance (pack); gap/noise fields (logprob_compare) | `tools/verify/pack.py`, `tools/verify/criteria.default.json`, `tools/quality/logprob_compare.py`, `tests/test_verify_pack_n24.py` | `344a55d`, test fix `bcc4ea6` |
| epyc-ai GGUF host check | `tools/hostcheck/glm_gguf_check.sh`, `tools/hostcheck/glm_gguf_check.py`, `tests/test_hostcheck_glm_gguf.py` | `70c01aa`, test fix `bcc4ea6` |
| SwiGLU clamp experiment | `patches/llama.cpp/n24-swiglu-clamp-after-silu.patch`, `tools/quality/swiglu_clamp_experiment.sh`, `tools/quality/swiglu_gate_stats.py`, `tests/test_swiglu_clamp_tools.py` | `ed7bc30` |
| planning docs (Russian) | `docs/reassessment/R0-HARDWARE-CHECKS.md`, `docs/DECISIONS.md` (ADR-012.2), `docs/reassessment/ROADMAP.md` (R2.4), `hardware/4x4080s-32g.json`, `tools/ceilings/tier_cost.py` (comment) | `a88abe9` |
| evidence scripts | `docs/evidence/mutation-scripts/2026-10-11-n24-mutations.py`, `-n24-swiglu-patch-check.sh`, `-n24-v6-summary.py` (+ README rows) | the commit of this file |
| docs of the pack rules; corrected tolerance note | `tools/verify/README.md`, `docs/reassessment/VERIFICATION_PACK.md`, `tools/verify/criteria.default.json` `_notes` | `4d41c87` |

N11's tools are not on this branch (it is cut from N19; N11 lives on `claude/n11-r1-tools-cpu-server` = `6680448`). The
runs below use a `git archive 6680448` copy in `<scratch>/n24/n11tree` (byte-identical tools), and for V6 a copy of N11's
`logprob_compare.py` with this branch's `compare()`/`summarize()` hunks applied (`<scratch>/n24/lc/logprob_compare.py`,
sha256 `21f207bb...4214dcb`). `git merge-tree --write-tree HEAD claude/n11-r1-tools-cpu-server` conflicts only in
`docs/evidence/mutation-scripts/README.md`, `tools/tinygen/glm5next_tiny.py`, `tools/tinygen/qwen4exp_tiny.py` - the same
three files as from the base `e71b241` (checked); `tools/quality/logprob_compare.py` merges cleanly.

---

## 1. Verification pack FAIL on the synthetic model (decision 1)

### 1.1 Cause (CONFIRMED)

N11 Run B2 (`docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md` on `claude/n11-r1-tools-cpu-server`, section 3.5) ended
`verification pack: FAIL (mandatory criteria FAIL: agent_api_probes)`. The only FAIL criterion is `agent_api_probes`
(`tools/verify/pack.py` `crit_agent_api_probes`): the probes `tools` and `json` of `tools/bench/probes.py` must score
>= 1.0 (`criteria.default.json` `probe_thresholds`, `agent_probes_required`). A random-weight model cannot call a tool
or write the schema's JSON: the engine and API worked (no request error, `response_format` accepted), the model's
answers failed. Every other criterion was PASS or NOT RUN. The pack has no notion of a synthetic model served by a real
engine: SYNTHETIC markers are informational only (N11 finding 7; `pack.py` `evaluate()` `synthetic_markers`).

Reproduced here with N11's `probes.py` against the real `llama-server` @79e2e74 (CPU build `<scratch>/llama.cpp/build-cpu`,
the R1 template's flags minus the three GPU flags, plus `-t 2 --no-warmup`) on a freshly generated text-vocab model
(`tools/tinygen/qwen4exp_tiny.py --text-vocab --ctx 262144` from the N11 copy; sha256 `e9555b37...c7df81`, byte-identical
to N11's `qwen4exp_text.gguf`):
```
$ llama-server -m <scratch>/n24/repro/weights/SYNTHETIC-qwen4exp-text-tiny.gguf --host 127.0.0.1 --port 18982 -c 16384 -np 1 -fa on -ctk f16 -ctv f16 --jinja --metrics -t 2 --no-warmup
$ python3 tools/bench/probes.py --url http://127.0.0.1:18982 --probes json,tools,determinism --seed 0 --answer-tokens 1024 --engine-desc "llama.cpp llama-server 79e2e74 CPU build, template flags minus GPU flags, -t 2 (N24 repro)" --model-desc "SYNTHETIC tiny random-weight model SYNTHETIC-qwen4exp-text-tiny.gguf (tools/tinygen --text-vocab), sha256 e9555b378dddacabd773fe3621823b440dc1b28823d94fc7486c04ab86c7df81, NOT real weights" --no-nvidia-smi --out-dir <scratch>/n24/repro/runs/probes-agent
run dir: <scratch>/n24/repro/runs/probes-agent (status ok, all hard probes pass: False)
exit=1
```
(`<scratch>/n24/repro/probes.txt`; the `time` lines of that file are left out.) Read back from `summary.json`: `tools`
score 0.0 with `n_tool_calls 0` (the final content is random bytes); `json` score 0.0714 = mean of the plain attempt
(0.0: not parsed) and the `response_format` attempt (0.1429: grammar-constrained, parsed as an object, wrong keys,
`accepted: true`); `determinism` 1.0; no `error` in any result. The same numbers as N11 B2 (tools 0.0, json 0.071).

The pack before this change (N11's copy, identical to `e71b241`'s `tools/verify/pack.py`) and after it with the default
criteria (`<scratch>/n24/repro/pack-runs.txt`, first two blocks; lines of the other criteria, all NOT RUN "missing", cut):
```
$ python3 <scratch>/n24/n11tree/tools/verify/pack.py --runs <scratch>/n24/repro/runs --out <scratch>/n24/repro/pack-n11pack
PASS     single_configuration        1 run(s), one configuration
PASS     openai_api                  /v1/models 200; streamed requests with usage
FAIL     agent_api_probes            artifacts/probes-agent tools: score 0.000 < 1.00; artifacts/probes-agent json: score 0.071 < 1.00
PASS     determinism                 (optional) identical output twice at temperature 0
overall: FAIL (mandatory criteria FAIL: agent_api_probes); 10 files in <scratch>/n24/repro/pack-n11pack/MANIFEST.sha256
exit=1
```
`tools/verify/pack.py` of this branch with the default criteria gives the same four lines and `overall: FAIL` (section 7).
The full N11 Run B2 replay through `scripts/r1/run_r1.py` was started and stopped: on this sandbox, shared with other
sessions (load average 16-18 on 4 cores), its decode ran at 4.7 s per token (smoke-2k: 128 tokens in 599.6 s, server
timing in `raw.jsonl`), so the targeted run above replaces it. NOT RUN: the whole B2 replay.

### 1.2 Rule (implemented, `344a55d`)

* **Declaration in the config, not inferred.** New criteria keys: `"synthetic_run": false` (default) and
  `"synthetic_not_applicable": ["agent_api_probes", "context_probes_full_length", "manual_quality_eval"]`. Only ids in
  `pack.py` `BEHAVIOUR_CRITERIA` - the criteria that score the model's answers - are allowed; `load_criteria` rejects any
  other id (e.g. `weights_integrity`, `decode_filled_ctx_floor`, `exact_routing_logprobs`), a non-boolean flag, or a
  non-list. The declaration is a deviation from `criteria.default.json`, so VERDICT.md already prints the "criteria
  differ from the customer's" banner.
* **The declaration must match the artifacts.** It is applied only if every matrix/probes run's `model_desc` contains
  "synthetic" (any case). Otherwise it is recorded (`synthetic_run.applied: false`, `unmarked_runs`) and the behaviour
  criteria are judged as usual: a real-model run cannot hide a FAIL behind a flag.
* **What changes.** A PASS or a score FAIL of a listed criterion becomes NOT RUN with reason code
  `not_applicable_synthetic` ("не применимо: прогон объявлен синтетическим, критерий проверяет поведение модели"); the
  raw verdict is kept (`details.synthetic_raw`) and listed in `verdict.json` `synthetic_run.not_applicable` and in a
  VERDICT.md banner line per criterion. NOT RUN results (missing, mock, unparseable, not filled) are left as they are.
  A FAIL that contains probe request errors (`details.request_errors`) stays FAIL: that is the engine or the API.
* **Overall.** A declared synthetic run is never PASS: if every mandatory criterion is PASS the overall verdict is
  NOT RUN "declared synthetic run: never acceptance evidence"; an engine/tool FAIL (decode floor, weights integrity,
  quant audit, ...) still makes it FAIL.
* **Not renamed silently.** FAIL is never turned into PASS; NOT RUN always carries the reason and the raw verdict.

The repro with the declaration (`<scratch>/n24/repro/pack-synthetic.txt`; the NOT RUN "missing" lines of the other
criteria cut):
```
$ cat <scratch>/n24/repro/criteria-synthetic.json
{"synthetic_run": true}
$ python3 tools/verify/pack.py --runs <scratch>/n24/repro/runs --out <scratch>/n24/repro/pack-n24pack-synthetic --criteria <scratch>/n24/repro/criteria-synthetic.json
PASS     single_configuration        1 run(s), one configuration
PASS     openai_api                  /v1/models 200; streamed requests with usage
NOT RUN  agent_api_probes            not applicable to the declared synthetic model (agent_api_probes judges model behaviour); raw verdict FAIL: artifacts/probes-agent tools: score 0.000 
PASS     determinism                 (optional) identical output twice at temperature 0
overall: NOT RUN (mandatory criteria NOT RUN: context_probes_full_length, output_after_full_prompt, agent_api_probes, decode_filled_ctx_floor, prefix_cache, agent_loop, quant_min_bpw, weights_integrity, exact_routing_l); 10 files in <scratch>/n24/repro/pack-n24pack-synthetic/MANIFEST.sha256
exit=3
```
VERDICT.md lines 7-8 of that pack:
```
> **Прогон объявлен СИНТЕТИЧЕСКИМ в файле критериев (`synthetic_run: true`): итог никогда не PASS.** Правило применено. declared synthetic in the criteria file and every bench run's model description says SYNTHETIC
> - `agent_api_probes`: не применимо к синтетической модели; исходный вердикт **FAIL** (threshold): artifacts/probes-agent tools: score 0.000 < 1.00; artifacts/probes-agent json: score 0.071 < 1.00
```

Tests (`tests/test_verify_pack_n24.py`, class `SyntheticRun`, artifacts of the real tools against the mock server, edited
consistently in summary.json and raw.jsonl): the N11 cause reproduced without a declaration (FAIL, exit 1); with it NOT RUN
+ listed + engine criteria still PASS + exit 3; never PASS overall even with nothing declared not applicable; a run whose
model description lacks SYNTHETIC keeps the FAIL; a probe request error stays FAIL; a decode-floor FAIL stays FAIL; a
subset list leaves `agent_api_probes` FAIL. `CriteriaValidation` covers the allowed ids and types.

**Proposal for the PR #1 session (not done; `scripts/r1/` and N11's `tools/srvcheck/` are not on this branch):**
`r1_tools_smoke.py derive-config` should write `"steps": {"pack": {"criteria": "<file with {\"synthetic_run\": true}>"}}`
into the derived config (`scripts/r1/run_r1.py` already passes `steps.pack.criteria` to `pack.py --criteria`,
`run_r1.py:712-714`@6680448), and the smoke's expectation "verdict not PASS" stays as it is.

---

## 2. V6 tolerance (decision 2)

### 2.1 What V6 judges and where a tolerance belongs

V6 (`exact_routing_logprobs`) compares the candidate engine with a reference engine on the same GGUF through
`tools/quality/logprob_compare.py`: one token at temperature 0, top-K log-probabilities per prefix; PASS needs >= 200
prefixes, no failed prefix, top-1 agreement >= 0.97 (`criteria.default.json`, all PROVISIONAL since `c10afe7`). Two
engines that compute the same function in different order differ by rounding noise; when the reference's best two
tokens are nearly tied that noise can swap them. The 0.97 bar was set to leave about 3 % of prefixes for such flips
(`_notes.top1_agreement_min`). N24 adds a separate, explicit rule for the flips that noise alone can explain.

**Rule (PROVISIONAL, `344a55d`).** For a prefix whose greedy tokens differ (ref picks `a`, cand picks `b`):
`ref_gap = lp_ref(a) - lp_ref(b) >= 0` and `cand_gap = lp_cand(b) - lp_cand(a) >= 0`. Their sum is exactly the change
of the logit difference `l(a) - l(b)` between the engines (the log-sum-exp terms cancel), so if every token's
log-probability moves by at most `d` between the engines, `ref_gap + cand_gap <= 2 d`. The flip counts as agreement
when both tokens are in both top-K lists and `ref_gap + cand_gap <= logprob_tie_tolerance_nats`; otherwise it is a
disagreement (also when a gap is unknown because a token is outside the other side's top-K, or for files written
before N24, which have no gaps). Then top-1 agreement (with near-tie flips counted as agreement) must still be
>= 0.97. `logprob_compare.py` now writes `ref_gap`, `cand_gap`, `max_abs_dlogprob` (= `d` measured over the tokens in
both top-K lists) and `n_common` per prefix, and the median/p95/p99/max of `max_abs_dlogprob` in the summary: the same
file that is judged also measures the noise.

### 2.2 Data (CPU, tiny random-weight models; CONFIRMED as measurements of these files, not of real weights)

* **N10** (`docs/evidence/2026-10-10-quant-fa-parity.md:126-134`@9c3bdac): llama.cpp f16 KV + FA against f32 + no FA on
  the same tokens, full log-softmax: qwen4exp max |d log-softmax| 2.8e-4; glm5-next 2.7e-3 to **1.15e-2 with no discrete
  choice changed** (at 79e2e74 the same runs flip nothing); 0.603 where exactly one MoE decision flipped (ec7630a,
  traced near-tie: margin 2.9e-6 in the precision-aware oracle).
* **N11** (section 3.2 there): `-fa off` vs `-fa on`, f16 KV: KL(top-K) max 3.1e-8 (qwen4exp), 1.2e-2 (glm5-next); top-1 1.0.
* **N24** (this task): three llama-server @79e2e74 configurations of each text-vocab model at once (`-c 8192 -np 1 -t 2
  --no-warmup`; A `-ctk f32 -ctv f32 -fa off`, B f16 / `-fa off`, C f16 / `-fa on` = the R1 default), the N11 corpus,
  200 prefixes, top-20, `--max-chars 3000`, with N11's `logprob_compare.py` plus this branch's hunks
  (`<scratch>/n24/v6/run_v6.sh`; output `<scratch>/n24/v6/run_v6.txt`, scratch path shortened):
```
e9555b378dddacabd773fe3621823b440dc1b28823d94fc7486c04ab86c7df81  <scratch>/n11/models/qwen4exp_text.gguf
$ python3 <scratch>/n24/lc/logprob_compare.py --ref http://127.0.0.1:18992 --cand http://127.0.0.1:18993 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n24/v6/lc-qwen4exp-f16fa-off_vs_f16fa-on.json
170 prefixes (30 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0000 median 0.0000 p95 0.0000 max 0.0000 -> <scratch>/n24/v6/lc-qwen4exp-f16fa-off_vs_f16fa-on.json
exit=0
$ python3 <scratch>/n24/lc/logprob_compare.py --ref http://127.0.0.1:18991 --cand http://127.0.0.1:18993 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n24/v6/lc-qwen4exp-f32fa-off_vs_f16fa-on.json
170 prefixes (30 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0000 median 0.0000 p95 0.0000 max 0.0001 -> <scratch>/n24/v6/lc-qwen4exp-f32fa-off_vs_f16fa-on.json
exit=0
$ python3 <scratch>/n24/lc/logprob_compare.py --ref http://127.0.0.1:18993 --cand http://127.0.0.1:18993 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n24/v6/lc-qwen4exp-same-server.json
170 prefixes (30 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0000 median 0.0000 p95 0.0000 max 0.0000 -> <scratch>/n24/v6/lc-qwen4exp-same-server.json
exit=0
d5304a8c2756e999005791de0348e7b9cbc0f0ed1c9bfeb83a6661da34ebe83c  <scratch>/n11/models/glm5next_text.gguf
$ python3 <scratch>/n24/lc/logprob_compare.py --ref http://127.0.0.1:18992 --cand http://127.0.0.1:18993 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n24/v6/lc-glm5next-f16fa-off_vs_f16fa-on.json
166 prefixes (34 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0001 median 0.0000 p95 0.0000 max 0.0118 -> <scratch>/n24/v6/lc-glm5next-f16fa-off_vs_f16fa-on.json
exit=0
$ python3 <scratch>/n24/lc/logprob_compare.py --ref http://127.0.0.1:18991 --cand http://127.0.0.1:18993 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n24/v6/lc-glm5next-f32fa-off_vs_f16fa-on.json
166 prefixes (34 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0041 median 0.0000 p95 0.0137 max 0.2038 -> <scratch>/n24/v6/lc-glm5next-f32fa-off_vs_f16fa-on.json
exit=0
$ python3 <scratch>/n24/lc/logprob_compare.py --ref http://127.0.0.1:18993 --cand http://127.0.0.1:18993 --corpus '<scratch>/n11/corpus/*.txt' --prefixes 200 --top-k 20 --max-chars 3000 --out <scratch>/n24/v6/lc-glm5next-same-server.json
166 prefixes (34 errors): top-1 agree 1.0000, ref top-1 in cand top-20 1.0000, KL(top-K) mean 0.0000 median 0.0000 p95 0.0000 max 0.0000 -> <scratch>/n24/v6/lc-glm5next-same-server.json
exit=0

real	55m26.297s
user	36m59.044s
sys	0m9.500s
```
```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n24-v6-summary.py <scratch>/n24/v6/lc-*.json
lc-glm5next-f16fa-off_vs_f16fa-on.json: n 166, errors 34 (incomplete UTF-8 34), top-1 1.0000, flips 0 []
    max_abs_dlogprob: median 1.426e-03 p95 4.445e-03 p99 3.286e-02 max 3.520e-02; common tokens min 19; KL top-K max 1.177e-02
    share with max_abs_dlogprob <= 0.0125 (TOL/2): 162/166 = 0.976
lc-glm5next-f32fa-off_vs_f16fa-on.json: n 166, errors 34 (incomplete UTF-8 34), top-1 1.0000, flips 0 []
    max_abs_dlogprob: median 1.711e-03 p95 1.029e-01 p99 4.901e-01 max 6.166e-01; common tokens min 13; KL top-K max 2.038e-01
    share with max_abs_dlogprob <= 0.0125 (TOL/2): 144/166 = 0.867
lc-glm5next-same-server.json: n 166, errors 34 (incomplete UTF-8 34), top-1 1.0000, flips 0 []
    max_abs_dlogprob: median 0.000e+00 p95 0.000e+00 p99 0.000e+00 max 0.000e+00; common tokens min 20; KL top-K max 0.000e+00
    share with max_abs_dlogprob <= 0.0125 (TOL/2): 166/166 = 1.000
lc-qwen4exp-f16fa-off_vs_f16fa-on.json: n 170, errors 30 (incomplete UTF-8 30), top-1 1.0000, flips 0 []
    max_abs_dlogprob: median 1.616e-04 p95 3.936e-04 p99 5.527e-04 max 5.827e-04; common tokens min 20; KL top-K max 3.070e-08
    share with max_abs_dlogprob <= 0.0125 (TOL/2): 170/170 = 1.000
lc-qwen4exp-f32fa-off_vs_f16fa-on.json: n 170, errors 30 (incomplete UTF-8 30), top-1 1.0000, flips 0 []
    max_abs_dlogprob: median 1.869e-04 p95 5.856e-04 p99 8.336e-03 max 2.181e-02; common tokens min 20; KL top-K max 6.056e-05
    share with max_abs_dlogprob <= 0.0125 (TOL/2): 169/170 = 0.994
lc-qwen4exp-same-server.json: n 170, errors 30 (incomplete UTF-8 30), top-1 1.0000, flips 0 []
    max_abs_dlogprob: median 0.000e+00 p95 0.000e+00 p99 0.000e+00 max 0.000e+00; common tokens min 20; KL top-K max 0.000e+00
    share with max_abs_dlogprob <= 0.0125 (TOL/2): 170/170 = 1.000
```
  The B-vs-C files reproduce N11 exactly (same prefixes, errors and KL maxima). No pair flipped a top-1 token, so the
  near-tie rule itself is exercised only by the stub tests (`TieTolerance`), not by these data. Every failed prefix is
  the known incomplete-UTF-8 kind (N11 finding 3), so V6 on these files is NOT RUN by `logprob_max_errors` = 0
  regardless of the tolerance.

Reading (PROVISIONAL): the bulk of the noise is continuous and small (medians 1.6e-4 to 1.7e-3); the tails are
discrete: in the glm5-next f32-vs-f16 pair 22 of 166 prefixes move by more than 1.25e-2 and up to 0.62, the size of
N10's single traced MoE flip (0.603) - routing or DSA selection decisions that fall on the other side of a near-tie.
The prefixes behind these tails were not traced here (UNKNOWN which decision flipped in each).

### 2.3 The value: 0.025 nats (PROVISIONAL until measured on real weights and GPU)

`logprob_tie_tolerance_nats = 0.025` = 2 x 1.15e-2 (rounded up): twice the largest same-weights |d log-softmax|
measured with **no discrete choice changed** (N10, traced). It covers every prefix of the pairs without discrete
events and 86.7-100 % of the prefixes of every N24 pair (share <= TOL/2 above). It deliberately does **not** cover the
discrete tails: "2 x the largest observed" would be 1.23 nats and would excuse real disagreements. A top-1 flip caused
by a discrete near-tie upstream (routing, DSA selection) is legitimate numerics too, but its size is not bounded, so it
stays a disagreement and uses the 3 % budget of `top1_agreement_min`.

ULP reasoning for the f16 KV cache (binary16 unit roundoff u = 2^-11 = 4.88e-4): every cached K/V element carries a
relative error <= u, so an attention score `q.k * scale` moves by at most `u * scale * sum|q_i k_i| <= u * scale * |q| |k|`
and each attention log-probability by at most twice the largest score move; N10 showed the f16 accumulator of the CPU
flash-attention path adds about one binary16 spacing at the attention output (finding 4). How these per-layer errors grow
through 46 layers (GLM-5.3-Flash `block_count`, N15) is not bounded by this argument; only measurement gives `d`. One
more ULP fact sets a floor for real weights: a logit `l` that passes through binary16 anywhere carries `|dl| <= u |l|`,
so with logits of magnitude 20-30 (typical of trained LMs, PROVISIONAL; the tiny random models have nearly flat logits,
top-1 log-probability about -4.4, N11 D2) the logit difference of two tokens can move by up to 2 x 30 x 4.88e-4 = 0.029
nats from that single rounding. llama.cpp keeps the output logits in f32 on CPU and CUDA (UNKNOWN for the candidate
engines); the CUDA path quantises activations to q8_1 in 32-element blocks (MMQ) where the CPU uses q8_K in 256-element
blocks for k-quants - a relative error per element of order 1/254, about 8 times the f16 unit roundoff (PROVISIONAL from
the formats). **So the GPU-vs-CPU noise on real weights is expected to be larger than these CPU tiny-model numbers, and
0.025 is a lower bound to be replaced by measurement.** What 0.025 ignores: GPU kernels and their activation
quantisation, real logit scales, longer prefixes (3000 characters here), the candidate engine's own kernels, and
discrete-tail flips.

### 2.4 GPU measurement plan (R2.1 / R6.2; NOT RUN)

1. Same GGUF (the R1 candidate quant), llama.cpp @79e2e74 twice on the server: reference = CPU build or `-ngl 0`, f32
   KV, `-fa off`; candidate = the R1 template (`-ngl all -sm layer -fa on -ctk f16 -ctv f16`). The R2 corpus (natural
   text in English, Russian and code; chosen so that few greedy tokens end in an incomplete UTF-8 sequence, or the lead
   decides to tolerate that error kind - N11 finding 3), `logprob_compare.py --prefixes 200 --top-k 20` with three
   seeds (`--seed 1/2/3`, 600 prefixes).
2. Within-GPU noise: the candidate against itself with `-ub 512` vs `-ub 32` (MMQ vs MMVQ paths) and `-fa on` vs
   `-fa off`.
3. From the `max_abs_dlogprob` of steps 1-2 (now in every file): set the tolerance to 2 x the 99th percentile of the
   prefixes without a discrete event (identify them by tracing the largest ones with `tools/dump` routing dumps, as N10
   did), never below 0.025; record the share of prefixes above TOL/2 and every top-1 flip with its gaps.
4. Only then judge the expert-skipping candidate (the real V6) against the all-experts reference; update
   `criteria.default.json` with the measured value and this evidence's successor, label CONFIRMED (measured), and
   re-check `top1_agreement_min` against the number of discrete-event flips seen in steps 1-2.


---

## 3. SwiGLU clamp of GLM-5.3-Flash (decision 3)

### 3.1 The two semantics (CONFIRMED by reading, N12 and here)

* **pre** (clamp before the SiLU): `silu(min(g, L)) * clamp(u, -L, L)`. llama.cpp @79e2e74
  `ggml/src/ggml-cpu/ops.cpp:3614-3616` (f32) and `:3673-3675` (f16), `ggml/src/ggml-cuda/unary.cuh:118-123`; the same at
  ec7630a (ops.cpp:3614, 3673; the patch applies at both, `git apply --check`).
* **post** (clamp after the SiLU): `min(silu(g), L) * clamp(u, -L, L)`. ik_llama.cpp @5194a9e `ggml/src/ggml.c:17374-17377`.
* They differ only where `g > L`; there the gate factor differs by at most `L * sigmoid(-L)` = 4.54e-4 for L = 10 and the
  SwiGLU output by at most `L^2 * sigmoid(-L)` = 4.54e-3 (N12, CONFIRMED arithmetic). GLM-5.3-Flash: `swiglu_limit` 10.0,
  `hidden_act` silu (vendored `third_party/hf/glm-5.3-flash-chat/config.json` text_config; the GGUF header has
  `glm5-next.swiglu_clamp_exp` / `_shexp` = 10.0 per layer, N15 summary).

### 3.2 Which one is "official"

| source | semantics | label and citation |
|---|---|---|
| zai-org/GLM-5.3-Flash HF repo @eb9eb208 | **no modeling code**: the upstream listing has no `.py` file, `config.json` has no `auto_map`; `architectures` `Glm5NextForConditionalGeneration`, `model_type` `glm5_next`, `transformers_version` "5.16.0" | CONFIRMED (`third_party/hf/glm-5.3-flash-chat/PIN.json`, `config.json`; output in 3.3) |
| HF transformers 5.16.0 (the config's version) | has **no** `models/glm5_next/` | CONFIRMED (`unzip -l` count 0; wheel sha256 = PyPI's) |
| HF transformers 5.17.0, 5.18.0, 5.19.0 `models/glm5_next/modeling_glm5_next.py` | **pre**: `Glm5NextTextMLP.forward` (dense and shared expert) and `Glm5NextTextExperts._apply_gate` (routed experts; every experts backend calls `_apply_gate`, `integrations/moe.py:153`) clamp the gate, then `act_fn(gate) * up` / `F.silu(gate) * up` | CONFIRMED (5.19.0 `modeling_glm5_next.py:102-104, 139-142`; same lines in 5.17.0/5.18.0 at :103-105, :140-143; wheels' sha256 = PyPI's) |
| vLLM @bc21cba (AGENTS.md pin) `vllm/models/glm5next/common/model.py` | **pre**: shared expert / dense MLP `SiluAndMulWithClamp` (`activation.py:208-214, 239-243`: gate = clamp(x, max=L), up = clamp; `gate * sigmoid(gate) * up`); routed experts pass `swiglu_limit` to `FusedMoEFactory`; fused-MoE activation `silu_and_mul_with_clamp` (`fused_moe/activation.py:293-311`, "silu(clamp(gate)) * clamp(up)") | CONFIRMED for the Python paths, fetched from raw.githubusercontent.com at the pin; the CUDA kernel `torch.ops._C.silu_and_mul_with_clamp` was not read (`csrc/activation_kernels.cu` 404 at that path) - its semantics UNKNOWN beyond the docstring |
| llama.cpp @79e2e74 / ec7630a | pre | CONFIRMED (3.1) |
| fastllm @a2ff521 `src/models/glm5_next.cpp:2364-2367` | pre (`Glm5NextClamp(gate, ...upper)`, clamp up, then `Silu`, `MulTo`); docs/glm5.md:79 says the same | CONFIRMED by reading (dense path; its CUDA MoE kernels not read) |
| DeepSeek-V4.1-Flash reference (vendored) | pre (`third_party/deepseek-v41-flash/inference/model.py:845-848`) | CONFIRMED (a different model) |
| ik_llama.cpp @5194a9e | **post** (CPU `ggml.c:17374-17377`; `iqk_mul_mat.cpp:155-157, 169-170`; CUDA `unary.cu` `fused_mul_silu_f32` kernels take `limit` - not read in detail) | CONFIRMED for the CPU lines; ik's own comment says the opposite (N12) |
| exllamav3 @151539c | **post** (`exllamav3_ext/cpu/moe_mul1.cpp:2413-2430`: `min(silu(g), lim) * clamp(u)`; `modules/mlp.py` slopes the same) although `architecture/glm5_next.py:26` names "Reference: transformers models/glm5_next" | CONFIRMED by reading; the contradiction with transformers is its own |

**Answer.** Every reference implementation of the architecture that is tied to the published config (HF transformers,
the only modeling code for `model_type: glm5_next`, in all three versions that have it; vLLM at our pin) computes **pre**,
which is what llama.cpp computes. ik_llama.cpp and exllamav3 compute post. **What GLM-5.3-Flash was trained with is
UNKNOWN**: zai-org publishes no modeling or training code in the repo; transformers' code may be a port by a third party
(the file header says "Copyright 2026 the HuggingFace Team"; who wrote it is not in the file). PROVISIONAL reading: pre
is the semantics of record. The experiment in 3.4 can falsify it: if post gives a clearly lower perplexity on natural
text with the same weights, the trained function is more likely post.

### 3.3 Literal sources

`<scratch>/n24/clamp-sources.txt` (transformers) and `<scratch>/n24/clamp-sources-2.txt` (the rest), excerpts:
```
$ curl -sS https://pypi.org/pypi/transformers/<v>/json | <digests of the wheel>
5.16.0 transformers-5.16.0-py3-none-any.whl 2b4adda8d99bb4c1b208da5ed967e2625ad449087827a750cb3b23ce4000b139
5.17.0 transformers-5.17.0-py3-none-any.whl 78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801
5.18.0 transformers-5.18.0-py3-none-any.whl d79e5a515a572ee3deb33eb0d578be47d91e70c75d54cf21bf6eede6edef14a9
5.19.0 transformers-5.19.0-py3-none-any.whl afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518
$ unzip -l transformers-5.16.0-py3-none-any.whl | grep -c "models/glm5_next/"
0
$ unzip -p transformers-5.19.0-py3-none-any.whl transformers/models/glm5_next/modeling_glm5_next.py | sed -n "97,104p;136,142p"

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up
$ sed -n "239,243p" vllm/model_executor/layers/activation.py      # @bc21cba
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)
# zai-org/GLM-5.3-Flash upstream listing (third_party/hf/glm-5.3-flash-chat/PIN.json @ eb9eb208): any .py? auto_map in config.json?
revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
.py files: []
False False ['Glm5NextForConditionalGeneration'] glm5_next 5.16.0 10.0 silu
```
The 5.17.0/5.18.0 wheels were downloaded for this task into `<scratch>/n24/tf/` (PyPI, `pip download --no-deps`); 5.16.0
and 5.19.0 are N14's downloads in `<scratch>/n14/tf-dl/`. vLLM files: `<scratch>/n24/vllm/` (sha256 in
`clamp-sources-2.txt`). Nothing of these is installed or vendored.

### 3.4 Real-weight experiment (prepared, NOT RUN)

**Patch validated on CPU.** `patches/llama.cpp/n24-swiglu-clamp-after-silu.patch` changes three places (CPU f32 loop,
CPU f16 loop, CUDA `ggml_cuda_op_swiglu_clamp_single`, which `unary.cu`, `mmvq.cu` and `mmvf.cu` call). A copy of
ec7630a with the patch, built as `tools/g0` `dump-tensors` (`<scratch>/n24/build-post`), against the stock N19 build
(`<scratch>/n19/build-ec`), on one tiny glm5-next model (L = 0.5) and the G1 tokens:
```
$ bash docs/evidence/mutation-scripts/2026-10-11-n24-swiglu-patch-check.sh /home/user/wt-n24 <scratch>/n19/build-ec <scratch>/n24/build-post <scratch>/llama-ec7630a <scratch>/n24/swiglu-check
26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5  <scratch>/n24/swiglu-check/tiny.gguf
== stock: <scratch>/llama-ec7630a
dump-tensors exit 0
Ran 9 tests in 0.671s
OK
ffn_moe_swiglu_limited vs pre formula: max abs err 2.121e-08; vs post formula: 9.439e-02; gate > L in 14001 of 45056 elements
== post: <scratch>/n24/llama-ec-post
dump-tensors exit 0
FAIL: test_swiglu_clamp (tests.test_glm_tiny_parity.GlmTinyParity.test_swiglu_clamp)
FAIL: test_forward_matches_dump (tests.test_glm_tiny_parity.GlmTinyWholeModel.test_forward_matches_dump)
Ran 9 tests in 0.674s
FAILED (failures=2)
ffn_moe_swiglu_limited vs pre formula: max abs err 9.439e-02; vs post formula: 2.052e-08; gate > L in 13874 of 45056 elements
exit=0
```
With the patch, llama.cpp's dumped `ffn_moe_swiglu_limited` equals the post formula to 2.1e-8 and differs from the pre
formula by 9.439e-2 = `L^2 * sigmoid(-L)` for L = 0.5 (the N12 bound, attained); only `test_swiglu_clamp` and the
whole-model forward (downstream of it) fail; every other op still equals the oracle on llama.cpp's own inputs.
CONFIRMED on CPU at ec7630a; on 79e2e74 and on CUDA the patch applies (`git apply --check`) but was not built (NOT RUN).

**Driver** `tools/quality/swiglu_clamp_experiment.sh` (`build` / `kld` / `gate`, every command printed). Both variants are
the same llama.cpp tree @79e2e74, so weights, graph, kernels and flags are equal and every difference is the clamp. The
GGUF must be the `glm5-next` dialect (llama.cpp does not load `glm5next`, N15): on epyc-ai first run the host check of
section 4; if the files there are `glm5next`, llama.cpp cannot run them and nothing may be downloaded there (AGENTS.md),
so the experiment then needs the 4x4080 server or the owner's decision.

Exact commands (server with CUDA; on epyc-ai `-DCMAKE_CUDA_ARCHITECTURES=86`, on the 4080 server `89`):
```
git -C ~/src/llama.cpp fetch && git -C ~/src/llama.cpp cat-file -e 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
tools/quality/swiglu_clamp_experiment.sh build ~/src/llama.cpp ~/n24-clamp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON
cat ~/n24-clamp/variants.txt                      # sha256 of the two patched files per variant, the N24 markers only in "post"
# corpus: fixed natural text, never the probe fillers; e.g. the R2 Q1 corpus file; its sha256 goes into the result
export RUN_ARGS="-ngl 99 -ot exps=CPU -t 32 -fa on"   # experts on CPU as in the tiered setup; same for every run
CTX=512 CHUNKS=32 tools/quality/swiglu_clamp_experiment.sh gate ~/n24-clamp ~/n24-clamp/out-gate <GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf> <corpus.txt> 10
CTX=512 CHUNKS=32 tools/quality/swiglu_clamp_experiment.sh kld  ~/n24-clamp ~/n24-clamp/out-q4 <GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf> <corpus.txt>
# with a higher-precision GGUF of the same model (BF16 or Q8_0, if the owner provides one; not on epyc-ai):
CTX=512 CHUNKS=32 tools/quality/swiglu_clamp_experiment.sh kld  ~/n24-clamp ~/n24-clamp/out-ref <GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf> <corpus.txt> <GLM-5.3-Flash-BF16-00001-of-00014.gguf>
```
What each step answers:

1. `gate`: how many expert/shared gate pre-activations exceed L = 10 on the corpus (`swiglu_gate_stats.py`); 0 means the
   two semantics are identical on this text (no evidence either way). Bound per element: 4.54e-3 on the SwiGLU output.
2. `kld` without a reference: perplexity of the same Q4 weights under pre and post, and KLD(post || pre) over the full
   vocabulary (`llama-perplexity --kl-divergence`): the size of the effect. A perplexity clearly lower under one
   semantics (beyond the run-to-run spread, which is 0 for a deterministic CPU/GPU path - check by repeating "pre") is
   evidence that the model was trained with it (PROVISIONAL criterion: the difference must exceed what the Q4
   quantisation itself moves, see 3).
3. `kld` with a BF16/Q8_0 reference: the reference is produced by **each** build (`ref-pre.kld`, `ref-post.kld`); Q4-pre
   vs ref-pre and Q4-post vs ref-post measure the quantisation under a consistent semantics; the cross terms measure
   the clamp. A reference from HF transformers or vLLM (FP8/BF16 safetensors) would be "pre" by construction (3.2), so
   it can only show which llama.cpp build reproduces those implementations, not the trained function.
4. ik_llama.cpp (post) can be added as a third arm only on its own `glm5next` file and differs from llama.cpp in five
   more places (N12), so it does not isolate the clamp; it is not part of the driver.

Cost on the server: UNKNOWN until R0 (32 chunks x 512 tokens per run; 2 runs without a reference, 6 with one).

---

## 4. Read-only host check for epyc-ai (decision 4)

`tools/hostcheck/glm_gguf_check.sh --dir <dir> --out <new dir> [--jobs N] [--no-hash]` (a `python3 -I` wrapper of
`glm_gguf_check.py`; stdlib only; the weights are opened `"rb"` only; `--out` must be new or empty and outside `--dir`;
no network or download code: the test asserts the imported modules are `argparse, concurrent, hashlib, json, os, struct,
sys, time`). Embedded pins = N15's `PIN.json` (`git show origin/claude/n15-gguf-headers:third_party/gguf-headers/
glm-5.3-flash-ud-q4_k_xl/PIN.json`, `origin/claude/n15-gguf-headers` = `67bd5ba`): per shard size, `lfs_sha256`,
`header_bytes`, `header_sha256`; shard-1 facts architecture `glm5-next`, `expert_group_count` 1,
`expert_group_used_count` 1, chat template sha256 `a4fddbbf...aa724` (sha256 of the template's UTF-8 bytes in the
pinned header, 10648 B, computed here from N15's summary.json), `split.count` 6; the vendored zai-org template's sha256
(N14) is reported for comparison only. The parser is this tool's own (N15's `tools/gguf_headers/gguf_header.py` is on
`claude/n15-gguf-headers`, not on this branch); it reads the header sequentially and never the tensor data.

On the real header bytes (N15 committed them gzipped; `git show 67bd5ba:third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/
<shard>.header.gz | gunzip > <scratch>/n24/hdr/<shard>` for the six shards), the parser reproduces every pinned
`header_bytes` and `header_sha256` and the shard-1 facts; the sizes differ by construction (header bytes only), so the
verdict is FAIL with exit 1 (`<scratch>/n24/hdr-check.txt`):
```
$ tools/hostcheck/glm_gguf_check.sh --dir <scratch>/n24/hdr --out <scratch>/n24/hdr-out --no-hash
glm_gguf_check: FAIL: 6 of 6 shard(s) missing or different from the pin
pins: claude/n15-gguf-headers@67bd5ba:third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/PIN.json (unsloth/GLM-5.3-Flash-GGUF @ a38483c8cd5df544f53d70fb281afe97369d5ab6)
dir: <scratch>/n24/hdr

MISMATCH   GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
           size 9429956 (pin 9429984)  DIFFERS
           header == pin; architecture 'glm5-next', split 0/6, n_tensors 0
MISMATCH   GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf
           size 24608 (pin 49198039200)  DIFFERS
           header == pin; architecture None, split 1/6, n_tensors 397
MISMATCH   GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf
           size 21376 (pin 49988626304)  DIFFERS
           header == pin; architecture None, split 2/6, n_tensors 337
MISMATCH   GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf
           size 20256 (pin 48700701344)  DIFFERS
           header == pin; architecture None, split 3/6, n_tensors 320
MISMATCH   GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf
           size 21632 (pin 49026026752)  DIFFERS
           header == pin; architecture None, split 4/6, n_tensors 341
MISMATCH   GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf
           size 1248 (pin 2784497888)  DIFFERS
           header == pin; architecture None, split 5/6, n_tensors 17

header facts of shard 1 (expected = the pinned Hugging Face file):
  ok   general.architecture: found 'glm5-next', expected 'glm5-next'
  ok   expert_group_count: found 1, expected 1
  ok   expert_group_used_count: found 1, expected 1
  ok   chat_template_sha256: found 'a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724', expected 'a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724'
  ok   split.count: found 6, expected 6
  DIFF chat template == vendored zai-org template (N14), informational: found 'a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724', expected '0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5' (informational)
exit=1
```
Expected on epyc-ai (PROVISIONAL): AGENTS.md says the host files are architecture `glm5next`; then every shard's size or
sha256 differs from these pins (they are the `glm5-next` files of unsloth @a38483c8) and the report shows which
architecture string, expert-group count and template the host files carry. The real check: NOT RUN (no access).
Command for the host (writes only into the new directory):
```
tools/hostcheck/glm_gguf_check.sh --dir /mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL --out "$HOME/glm-gguf-check-$(date -u +%Y%m%dT%H%M%SZ)" --jobs 3
```
Cost: one sequential read of 185.99 GiB of files (199 707 321 472 bytes, N15 sizes); wall time UNKNOWN (NVMe and
SHA-256 speed of that host).

---

## 5. Planning docs (decision 5)

`docs/reassessment/R0-HARDWARE-CHECKS.md` (Russian): (a) DRAM channels - dmidecode and EDAC on the Proxmox host (inside
the VM they describe virtual DIMMs), numactl, `scripts/diag/bench_mem.sh` as the instrument, STREAM / likwid-bench as
cross-checks, Intel MLC unsupported on AMD; expected 204.8 / 102.4 GB/s peaks (CONFIRMED arithmetic) and R4's Triad
expectations 170-180 / 90-100 GB/s (HYPOTHESIS there); (b) P2P - standard diagnostics only (IOMMU groups, ACS, `qm config`,
`nvidia-smi topo -m` / `-p2p r|w`, `bench_gpu.sh`, p2pBandwidthLatencyTest, nvbandwidth, nccl-tests with and without
`NCCL_P2P_DISABLE=1`), no driver patch, and five no-P2P layouts with how to measure each (the TP cost model
2 x layers x all-reduce latency is PROVISIONAL); (c) every place that says 3 NVMe and the action; (d) the Qwen licence.
`docs/DECISIONS.md` ADR-012.2: licence registry entry (Qwen Community License 1.0: local development and testing only;
commercial provision to third parties not agreed; clause 2 text CONFIRMED via `research/R11-verification.md` C25), the
no-driver-patch and NVMe decisions. `docs/reassessment/ROADMAP.md` R2.4: partially answered. `hardware/4x4080s-32g.json`:
four planned 1 TB NVMe, models UNKNOWN (no code reads the `nvme` list of this file: `tools/roofline.py:223` reads it from
the `--hw` file, default `hardware/target-4x4080-32g.unverified.json` (`tools/roofline.py:31`, also in the R1 configs); `tests.test_fitplan`, `test_g0_analyze`,
`test_tier_cost` pass, section 7).

**Not edited, proposals for the owner / lead:**
* `AGENTS.md:36` "- 3x Samsung 990 Pro 1 TB NVMe" -> "- 4x 1 TB NVMe (planned; earlier statement: 3x Samsung 990 Pro
  1 TB); actual count and models UNKNOWN until R0".
* `.claude/agents/infra-ops.md:20` "Storage: model files on the 990 Pro drives" -> "on the NVMe drives (planned 4 x 1 TB;
  models UNKNOWN until R0)".
* `docs/reassessment/BIOS_PLAN.md` steps 2 and 6 are motivated as prerequisites of a P2P driver patch; with the owner's
  "no driver patch" they lose that reason (Re-Size BAR may still matter for other reasons: UNKNOWN).
* `docs/reassessment/ADR-012-reassessment.md` "Что решает заказчик" item 2 can point to ADR-012.2.

---

## 6. Mutation check

`docs/evidence/mutation-scripts/2026-10-11-n24-mutations.py`: 18 single-point mutants of the new code against the new
tests. Round 1 at `a88abe9` (`<scratch>/n24/mut-round1.txt`): 16/18; L2 and H1 survived (the test moved both common
tokens by the same amount; no test changed a header byte at equal size). `bcc4ea6` strengthened both tests. Round 2:
```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n24-mutations.py /home/user/wt-n24 <scratch>/n24/mut   # at bcc4ea6
CAUGHT   P1 declaration applied even when a run's model description does not say SYNTHETIC  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   P2 probe request errors turned into NOT RUN too  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   P3 a declared synthetic run may be PASS overall  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   P4 any criterion may be declared not applicable  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   P5 the not-applicable list is ignored (every criterion converted)  [tests.test_verify_pack_n24: FAILED (failures=4)]
CAUGHT   P6 near_tie ignores the tolerance  [tests.test_verify_pack_n24: FAILED (failures=3)]
CAUGHT   P7 near_tie accepts negative gaps  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   P8 near_tie treats a missing gap as 0  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   P9 the threshold judged on strict top-1 (near-tie flips not counted)  [tests.test_verify_pack_n24: FAILED (failures=2)]
CAUGHT   L1 cand_gap with the wrong sign  [tests.test_verify_pack_n24: FAILED (failures=3)]
CAUGHT   L2 max_abs_dlogprob over one common token only  [tests.test_verify_pack_n24: FAILED (failures=1)]
CAUGHT   H1 header check ignores the header sha256  [tests.test_hostcheck_glm_gguf: FAILED (failures=1)]
CAUGHT   H2 --out allowed inside --dir  [tests.test_hostcheck_glm_gguf: FAILED (failures=1)]
CAUGHT   H3 MATCH without comparing the sha256  [tests.test_hostcheck_glm_gguf: FAILED (failures=1)]
CAUGHT   H4 --no-hash reported as a pass  [tests.test_hostcheck_glm_gguf: FAILED (failures=1)]
CAUGHT   H5 architecture-prefixed keys looked up under a fixed 'glm5-next.'  [tests.test_hostcheck_glm_gguf: FAILED (failures=2)]
CAUGHT   G1 output difference computed with the same semantics on both sides  [tests.test_swiglu_clamp_tools: FAILED (failures=1)]
CAUGHT   G2 gate values above the limit not counted  [tests.test_swiglu_clamp_tools: FAILED (failures=2)]
18/18 caught

real	5m12.980s
user	0m46.986s
sys	0m19.808s
exit=0
```

---

## 7. Tests (literal)

Filtered with `grep -E "^(Ran|OK|FAILED|FAIL:|ERROR:)"` (`<scratch>/n24/tests-final.txt`):
```
# at 4d41c87
$ python3 -m unittest tests.test_verify_pack_n24
Ran 20 tests in 27.549s
OK
$ python3 -m unittest tests.test_hostcheck_glm_gguf
Ran 12 tests in 1.267s
OK
$ python3 -m unittest tests.test_swiglu_clamp_tools
Ran 5 tests in 0.134s
OK
$ python3 -m unittest tests.test_verify_pack
Ran 19 tests in 27.074s
OK
$ python3 -m unittest tests.test_logprob_compare
Ran 6 tests in 2.847s
OK
$ python3 -m unittest tests.test_fitplan
Ran 15 tests in 0.304s
OK (skipped=1)
$ python3 -m unittest tests.test_g0_analyze
Ran 4 tests in 1.723s
OK
$ python3 -m unittest tests.test_tier_cost
Ran 26 tests in 1.471s
OK
$ python3 -m unittest tests.test_bench
Ran 38 tests in 17.803s
OK
$ python3 -m unittest discover -s tests -p "test_verify_pack*"
Ran 39 tests in 53.210s
OK
```
The one skip of `tests.test_fitplan` is `test_matches_pinned_gguf_py` ("set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py"), the
same at the base `e71b241`. The whole suite, at HEAD `4d41c87` and at the base (a `git archive e71b241` copy), same filter:
```
$ python3 -m unittest discover -s tests        # at 4d41c87
Ran 1135 tests in 206.255s
OK (skipped=557)
$ python3 -m unittest discover -s tests        # git archive e71b241
Ran 1098 tests in 160.161s
OK (skipped=557)
```
1135 - 1098 = 37 = the 20 + 12 + 5 new tests; no new skip. An earlier whole-suite run at `bcc4ea6`, while three other
background jobs and other sessions loaded the 4 cores (load average 16-18), failed one timing assertion
(`test_bench.Matrix.test_small_matrix_files_timings_cache`: `413.63335539377897 not greater than 600.0`, mock decode
tok/s); `tools/bench` and `tests/test_bench.py` are unchanged by this branch (`git diff --stat e71b241 HEAD -- tools/bench
tests/test_bench.py` empty) and `tests.test_bench` passes at the base and at HEAD (above).

The pack on the repro artifacts at the final pack code (`<scratch>/n24/repro/pack-head.txt`, criterion lines other than
`agent_api_probes` cut):
```
# at bcc4ea6
$ python3 tools/verify/pack.py --runs <scratch>/n24/repro/runs --out <scratch>/n24/repro/pack-head-default
FAIL     agent_api_probes            artifacts/probes-agent tools: score 0.000 < 1.00; artifacts/probes-agent json: score 0.071 < 1.00
overall: FAIL (mandatory criteria FAIL: agent_api_probes); 10 files in <scratch>/n24/repro/pack-head-default/MANIFEST.sha256
exit=1
$ python3 tools/verify/pack.py --runs <scratch>/n24/repro/runs --out <scratch>/n24/repro/pack-head-synth --criteria <scratch>/n24/repro/criteria-synthetic.json
NOT RUN  agent_api_probes            not applicable to the declared synthetic model (agent_api_probes judges model behaviour); raw verdict FAIL: artifacts/probes-agent tools: score 0.000 
overall: NOT RUN (mandatory criteria NOT RUN: context_probes_full_length, output_after_full_prompt, agent_api_probes, decode_filled_ctx_floor, prefix_cache, agent_loop, quant_min_bpw, weights_integrity, exact_routing_l); 10 files in <scratch>/n24/repro/pack-head-synth/MANIFEST.sha256
exit=3
```
(`pack.py` has not changed since `bcc4ea6`; `4d41c87` changed only `criteria.default.json` `_notes` text.)

`git log --oneline -n 8` and `git status --short` at the end are in the report to the lead.

---

## 8. NOT RUN and UNKNOWN

* Everything on a GPU, on real GLM weights, on epyc-ai or on the 4x4080 server: the host check, the clamp experiment,
  the V6 calibration, all R0 measurements.
* The full N11 Run B2 replay through `run_r1.py` (stopped; section 1.1).
* What GLM-5.3-Flash was trained with (pre or post): UNKNOWN (3.2).
* vLLM's CUDA `silu_and_mul_with_clamp` kernel and ik's / exllamav3's CUDA kernels: not read.
* The full text of the Qwen Community License beyond clause 2 / line 11: not in the repository.
* Whether the owner's decisions exist in writing from the customer: UNKNOWN (relayed by the lead).
