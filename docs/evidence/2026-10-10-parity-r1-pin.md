# 2026-10-10 — oracle parity at the llama.cpp commit R1 deploys (79e2e74), not only at the reference pin (ec7630a)

Branch `claude/research-landscape`. Role: parity-qa. Closes gap 1 of `docs/reassessment/CORRECTNESS.md`.
CPU only, tiny random-weight models, f32 caches, flash attention off (the parity contract). No GPU.

## Why

All oracle parity checks (G1 GLM, MiMo, N2 and N6 qwen4exp) compare `ref/` with llama.cpp at `ec7630a` (AGENTS.md
upstream pin, `ci.yml`), but R1 deploys llama.cpp `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`
(`tools/engines/pins.json`). The CORRECTNESS inventory ran the same scripts at `79e2e74` once (its section 4.7):
G1 8/8, MiMo 5/5, N2 17/17, N6 52/54. The two N6 failures were only the "indexer_pool_k_new (padding rows)" comparison
in the decode and rollback runs (rel. error 1.39 and 1.36).

## Cause (CONFIRMED, code)

Upstream changed which cell a padded pool entry re-pools:
* `ec7630a` `src/llama-memory-hybrid-idx.cpp:947-967`: all padded entries share one cell (the first member of pool 0
  if some pool is new, else the cell of the ubatch's first token);
* `79e2e74` same file, comment `:888-889`, code `:890-905`: "Padded entries re-pool cells whose pooled slot is never read: only the reps of
  complete pools are read. Each entry takes its own cell, entries sharing one would write it from several threads in
  the scatter." — the smallest cells that are no rep; a pool's rep is its last member's cell (`:857-859` "The pooled
  key lives in the last member's row").
In both versions the padded slots are never read (the test also checks that padded pools stay masked in the scores).

## Change

* `tests/test_qwen4exp_decode_parity.py`: the padding-row expectation depends on `PARITY_LLAMA_COMMIT` (default
  `ec7630a` → the unchanged shared-cell rule); any other commit → each padded row equals the pooled key of its own cell,
  the smallest cells not in {kp·j + kp − 1 : j < complete pools} (KV cell = position for this single sequence). No
  other check changed. Off `ec7630a` the check also asserts that the expected padded cells are distinct and none is a
  complete pool's rep (the property the upstream change protects: a padded write never lands on a slot that is read);
  since each padded row must equal the key of its expected cell, a padding that re-pooled a rep or shared a cell fails
  the value comparison (asked for by the PR #1 session, 2026-10-10).
* `tools/tinygen/run_parity_qwen4exp_decode.sh` exports `PARITY_LLAMA_COMMIT=$(git -C <llama.cpp> rev-parse HEAD)`.
* New workflow `.github/workflows/parity-r1-pin.yml`: G1, MiMo, N2, N6 at the `pins.json` llama.cpp commit (ci.yml keeps
  running them at `ec7630a`).

## Checks (CONFIRMED, literal)

Before the change, N6 at `79e2e74` (the CORRECTNESS inventory's run, `$S/corr/pin79/qwen4exp_decode.log`, filtered with
`grep -E "^(FAIL|AssertionError|Ran|OK|FAILED)"`):

```
FAIL: test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Decode.test_qsa_pool_cache_and_selection)
AssertionError: 1.3864799363845095 not less than 0.0001 : decode indexer_pool_k_new (padding rows) L3 graph 1: rel err 1.39e+00 (abs 3.06e+00)
FAIL: test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Rollback.test_qsa_pool_cache_and_selection)
AssertionError: 1.3627376980950934 not less than 0.0001 : rollback indexer_pool_k_new (padding rows) L3 graph 2: rel err 1.36e+00 (abs 2.87e+00)
Ran 54 tests in 3.231s
FAILED (failures=2)
```

After the change:

N6 runner at `79e2e74` with the final code (`$S/llama.cpp` = 79e2e74, build of `tools/g0` against it):

```
$ tools/tinygen/run_parity_qwen4exp_decode.sh $S/llama.cpp $S/corr/build-tools-79
rc=0   (printed by the wrapper around the command, not part of the log)
decode   indexer_pool_k_new (padding rows, own non-rep cells)           3     0  6.279e-07  3.404e-07
rollback indexer_pool_k_new (padding rows, own non-rep cells)           3     0  6.279e-07  3.404e-07
ub10     indexer_pool_k_new (padding rows, own non-rep cells)           3     0  6.279e-07  3.404e-07
ub16     indexer_pool_k_new (padding rows, own non-rep cells)           3     0  6.279e-07  3.404e-07
Ran 54 tests in 4.632s
OK
```

Each rule against the other pin's dumps (same test file; dumps of the two runner runs; `$S/pinaware/cross2.log`):

```
== dumps 79
Ran 54 tests in 3.850s
OK
== dumps 79-with-ec-rule
AssertionError: 1.3864799363845095 not less than 0.0001 : decode indexer_pool_k_new (padding rows) L3 graph 1: rel err 1.39e+00 (abs 3.06e+00)
AssertionError: 1.3627376980950934 not less than 0.0001 : rollback indexer_pool_k_new (padding rows) L3 graph 2: rel err 1.36e+00 (abs 2.87e+00)
Ran 54 tests in 3.871s
FAILED (failures=2)
== dumps ec
Ran 54 tests in 4.030s
OK
== dumps ec-with-79-rule
AssertionError: 1.6602412249169942 not less than 0.0001 : decode indexer_pool_k_new (padding rows, own non-rep cells) L3 graph 1: rel err 1.66e+00 (abs 3.06e+00)
AssertionError: 1.5555450674249762 not less than 0.0001 : rollback indexer_pool_k_new (padding rows, own non-rep cells) L3 graph 2: rel err 1.56e+00 (abs 2.87e+00)
Ran 54 tests in 3.478s
FAILED (failures=2)
```

(the "dumps ec" line: the `ec7630a` dumps with the default rule — unchanged behaviour at the reference pin.)

G1, MiMo and N2 at `79e2e74` (the CORRECTNESS inventory's run, `$S/corr/pin79/*.log`):

```
glm_parity.log:      Ran 8 tests in 0.229s / OK
mimo_parity.log:     Ran 5 tests in 0.076s / OK
qwen4exp_parity.log: Ran 17 tests in 0.327s / OK
```

Unit suite (dump-based tests skip without dumps):

```
$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests   # on 3b1460f; output not saved to a file
Ran 522 tests in 117.480s
OK (skipped=84)
```

The new workflow's first runs (GitHub Actions, after this was written): green on 3b1460f (runs 38083882183 push,
38083886087 pull_request) and on ca76938 (38084099965); 38084102547 was cancelled by the workflow's concurrency group
(superseded). The cross-rule block above was produced before 98adf82 added the explicit distinct / no-rep assertions;
the second evidence audit re-ran it on the later test and got the same four outcomes and errors.

## Meaning

The graphs R1 will run (`79e2e74`, CPU) agree with the numpy oracles on everything the four suites compare, for GLM
(prefill), MiMo (prefill) and qwen4exp (prefill, several ubatches, decode, rollback, MTP). Still not covered (gaps 2-4
of CORRECTNESS.md): quantized weights, f16 KV and flash attention, GLM decode/rollback/MTP.
