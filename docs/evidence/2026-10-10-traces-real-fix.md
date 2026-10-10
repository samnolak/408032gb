# 2026-10-10 — traces-real: why the first run wrote no trace, and the fix

Branch `claude/research-landscape`. Role: perf-bench (tools/g0, scripts/traces), parity-qa (smoke, checks).
Everything here ran on the agent sandbox CPU with llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d
(`$S/llama-ec7630a`, `$S/build-g0` = `tools/g0` built against it). No real model was run here; the real models
run in workflow `traces-real` (GitHub runner), see the end of this file.

## 1. Symptom (CONFIRMED: run conclusion and failing step via the GitHub API; the log line below was read from the job
log through the GitHub MCP tool during the session and not saved — the sandbox's own `gh` gets `Forbidden` for logs)

Workflow `traces-real` run 38076798213 on 8981351: both GGUFs downloaded with matching sha256 and the corpus
rendered (e.g. `c00_en_cachesim_py: 1967 tokens`), then for each model:

```
cp: cannot stat '../work-qwen3-coder-30b-a3b/traces/c00_en_cachesim_py.trace': No such file or directory
```

(same for `glm-4.7-flash`). `g0-expert-trace` wrote no trace and its error was not in the log.

## 2. Cause (CONFIRMED, code @ec7630a)

* The tools fed prompts with `common_batch_get_one`, which marks only the batch's last token as an output
  (`common/common.cpp:2186-2199`: `const bool output = i == n_tokens - 1;`).
* The real graphs keep only the output rows before the **last layer's** FFN:
  `src/models/qwen3moe.cpp:123-125` (`if (il == n_layer - 1 && inp_out_ids) { cur = ggml_get_rows(...)`) and
  `src/models/deepseek2.cpp:636-638` (GLM-4.7-Flash is arch `deepseek2`). The last MoE layer then sees 1 row per
  prompt batch instead of n, the tools detect a row-count mismatch ("inconsistent capture") and write nothing.
  `glm5-next.cpp:655-672` crops only after the last layer, which is why the CI smoke on the tiny glm5next model
  never failed. qwen4exp crops like qwen3moe (`qwen4exp.cpp:473-479`).
* The error line was lost twice over: the driver filtered the tool's output with `grep -E "generated|ERR|error" || true`,
  and the tools returned from `main` without flushing llama.cpp's asynchronous logger (`common/log.h:61`: "Call
  common_log_flush() before exit if you want to ensure all logs are flushed").

## 3. Fix

* `tools/g0/expert_trace.cpp`, `tools/g0/expert_probe.cpp`: `prompt_batch()` marks every prompt token as an output, so
  no rows are dropped (routing is unchanged, see 4); `main` flushes the logger before returning.
* `scripts/traces/run_real_traces.sh`: each tool runs through `logged` (full log kept; on failure the tail is shown and
  the script stops); the summary lines go to `trace.txt` / `probe.txt` in the results.
* `tools/tinygen/qwen3moe_tiny.py`: a tiny random-weight `qwen3moe` GGUF (the graph of the first proxy model).
* `scripts/traces/smoke_tiny.sh`: trace + probe + `predict_eval` self-checks + `-b 64` vs `-b 1` on that model; a step of
  `ci.yml` (every push) and of `traces-real.yml` (before any download).

## 4. Checks (CONFIRMED, commands and literal output; `R=$S/trfix`, `ids/` = 2 or 3 random-id prompts of 150-224 tokens)

Old binaries (before the fix) reproduce the CI failure on the tiny qwen4exp and qwen3moe models:

```
$ LD_LIBRARY_PATH=$S/build-g0/bin G0_TRACES_DIR=$R/ids G0_OUT_DIR=$R/old-tr G0_GEN_TOKENS=8 $R/bin-old/g0-expert-trace -m $S/n7/final2/qwen4exp.gguf -c 512 -b 64 -ub 64 -t 4 -lv 4
rc=1
0.00.056.920 E .../trfix/ids/c00.ids: inconsistent capture (k=2, layers=8); no trace written
```

New binaries, same arguments:

```
== NEW qwen4exp trace
rc=0
c00.ids: 150 prompt + 8 generated tokens, 8 MoE layers, k=2
c01.ids: 187 prompt + 8 generated tokens, 8 MoE layers, k=2
== NEW glm5next trace
rc=0
c00.ids: 150 prompt + 8 generated tokens, 11 MoE layers, k=2
c01.ids: 187 prompt + 8 generated tokens, 11 MoE layers, k=2
```

Routing unchanged — glm5next (no last-layer crop, so the old binary was valid): old vs new byte-identical; qwen4exp:
old binary with `-b 1 -ub 1` (every token its own batch and output) vs new `-b 64`:

```
glm5next c00: old (last-token output) == new (all outputs) byte-identical
glm5next c01: old (last-token output) == new (all outputs) byte-identical
old binary -b 1 rc=0
qwen4exp c00: old -b 1 (every token its own output) == new -b 64 byte-identical
qwen4exp c01: old -b 1 (every token its own output) == new -b 64 byte-identical
```

Tiny qwen3moe (`tools/tinygen/qwen3moe_tiny.py`, 4 layers, 16 experts, top-4): old trace and probe rc=1, new rc=0;
`-b 1` vs `-b 64`: `c00` byte-identical, `c01` differs in 4 of 780 (token, layer) rows — layer 1 token 125, layer 3
tokens 80, 86, 106. Layer 1 is not touched by the last-layer crop, so batch-size numerics alone flip rows. The two of
those rows inside the probe window are the near-ties:

```
token 106 layer 3: k-th/(k+1)-th logit margin 0.00451; rank of that margin among all 432 rows: 2
token 125 layer 1: k-th/(k+1)-th logit margin 0.00428; rank of that margin among all 432 rows: 1
median margin 1.209643840789795
```

(rank = number of rows with a smaller margin: they are the 3rd and 2nd smallest of 432.) PROVISIONAL: batch-size
dependent rounding (the KV cache is f16) is the explanation consistent with this; not traced further.

Logger flush: probe on qwen4exp (no `ffn_norm` callback -> always an inconsistent capture), 3 runs each:

```
old run 1: rc=1 'inconsistent capture' lines=0
new run 1: rc=1 'inconsistent capture' lines=1
old run 2: rc=1 'inconsistent capture' lines=1
new run 2: rc=1 'inconsistent capture' lines=1
old run 3: rc=1 'inconsistent capture' lines=0
new run 3: rc=1 'inconsistent capture' lines=1
```

Smoke script, new binaries:

```
$ LD_LIBRARY_PATH=$S/build-g0/bin scripts/traces/smoke_tiny.sh $S/llama-ec7630a $S/build-g0 $S/trfix/smoke
wrote .../trfix/smoke/tiny.gguf
c00.ids: 150 prompt + 8 generated tokens, 4 MoE layers, k=4
c01.ids: 187 prompt + 8 generated tokens, 4 MoE layers, k=4
c02.ids: 224 prompt + 8 generated tokens, 4 MoE layers, k=4
c00.ids: 150 prompt + 8 generated tokens, recorded 108 (from 50), 4 MoE layers, k=4, n_embd=64, n_expert=16
c01.ids: 187 prompt + 8 generated tokens, recorded 108 (from 87), 4 MoE layers, k=4, n_embd=64, n_expert=16
c02.ids: 224 prompt + 8 generated tokens, recorded 108 (from 124), 4 MoE layers, k=4, n_embd=64, n_expert=16
{"router_logits_max_rel_err": 0.00027022845461033285, "selection_rule_match": 1.0, "gating": 1, "groups": [0, 0], "probe_equals_trace_rows": 1.0}
-b 64 vs -b 1: 2334/2340 (token, layer) rows select the same experts
smoke ok
exit=0
```

Smoke script, old binaries (it must fail):

```
FAILED (1): env G0_TRACES_DIR=.../smoke-old/ids G0_OUT_DIR=.../smoke-old/tr64 G0_GEN_TOKENS=8 .../bin-old/g0-expert-trace -m .../smoke-old/tiny.gguf -c 512 -b 64 -ub 64 -t 2
0.00.022.092 E .../smoke-old/ids/c00.ids: inconsistent capture (k=4, layers=4); no trace written
exit=1
```

Unit tests (worktree, includes the untracked tests/test_arch_ceilings.py; output not saved to a file):

```
$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests
Ran 477 tests in 113.756s
OK (skipped=84)
```

## 5. Cost of the fix (PROVISIONAL, arithmetic)

All prompt tokens now go through the output head: per 512-token batch the logits buffer is 512 x n_vocab x 4 B
and the head adds about 2 x n_embd x n_vocab FLOP per token. For Qwen3-30B-A3B, taking n_vocab = 151 936 and
n_embd = 2048 from the published model config (not read from the GGUF in this sandbox): 311 MB of logits per batch and
2 x 2048 x 151 936 = 0.62 GFLOP per token, against about 6 GFLOP per token for 3B active parameters (~10% more work).

## 5b. Follow-up: the smoke step was flaky under unbuffered Python (second evidence audit, same day)

`smoke_tiny.sh` piped `predict_eval.py` into `head -n 1` under `set -euo pipefail`. With `PYTHONUNBUFFERED=1` (set in the
agent sandbox, not on GitHub's runners) Python writes the later lines after `head` has closed the pipe, gets
`BrokenPipeError`, never writes `predict.json`, and the step fails; the audit saw 1 pass in 7 runs. CI stayed green only
because the runners do not set the variable. Fix: the output goes to `$WORK/predict.txt`, then `head -n 1` of the file.

```
$ for i in 1 2 3 4; do PYTHONUNBUFFERED=1 scripts/traces/smoke_tiny.sh ...; done      # before the fix
before fix, run 1: exit=1 1 BrokenPipeError lines
before fix, run 2: exit=1 1 BrokenPipeError lines
before fix, run 3: exit=1 1 BrokenPipeError lines
before fix, run 4: exit=1 1 BrokenPipeError lines
$ for i in 1 2 3 4 5; do PYTHONUNBUFFERED=1 scripts/traces/smoke_tiny.sh ...; done   # after the fix
after fix, run 1: exit=0 smoke ok
after fix, run 2: exit=0 smoke ok
after fix, run 3: exit=0 smoke ok
after fix, run 4: exit=0 smoke ok
after fix, run 5: exit=0 smoke ok
```
(logs `$S/trfix/smoke-flake-*.log`, `$S/trfix/smoke-fixed-*.log`.) `run_real_traces.sh` pipes into `tee`, which reads to the
end, so it is not affected.

## 6. Not done here

The real-model traces: NOT RUN in the sandbox (no huggingface.co access). They come from the next `traces-real` run on
the fixed commit; its run id and results go into this branch's next evidence file.
