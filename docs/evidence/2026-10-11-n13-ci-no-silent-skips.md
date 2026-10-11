# N13: a CI parity step fails when its tests did not run (gap 11 (a))

Date 2026-10-11. Roles: parity-qa (checker, expectations, evidence), infra-ops (workflow wiring). Branch
`claude/n13-ci-no-silent-skips`, cut from `claude/n12-ik-glm5next-cpu` at `5730cc4`. Hardware-free: CPU builds of the
pinned llama.cpp commits and of ik_llama.cpp in the sandbox (4 cores), tiny random-weight models. `<S>` below stands
for the session scratch directory `/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution.

## Result

- **The gap, CONFIRMED.** Every dump-parity module skips when its environment is missing
  (`@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)`, e.g. `tests/test_glm5next_decode_parity.py:620@5730cc4`) or
  when a run directory is missing (`raise unittest.SkipTest(f"no run {cls.RUN} under {cls.ROOT}")`,
  `tests/test_glm5next_decode_parity.py:164@5730cc4`), and `python3 -m unittest` then exits 0; the drivers end with that
  command, so the GitHub step was green. Section 4: 26 of the 34 broken runs (dump variable unset or pointed at an
  empty directory; every step, at both pins where it runs at both) exit 0 without the checker (`before_rc=0`). Before
  N13 only `ref-torch-cpu` failed on skips (`.github/workflows/ci.yml:63@5730cc4`; CORRECTNESS.md cites `:57`, the line has moved).
- **The fix.** `tools/ci/check_unittest.py` wraps each parity step and fails it unless the step's unittest run is OK with
  exactly `Ran N` and exactly the listed skips (test id and reason) that `tools/ci/parity_expectations.json` gives for
  the step at the full commit of the checkout the dumps come from. All 17 parity steps of `ci.yml` (`cpu-parity`, 9)
  and `parity-r1-pin.yml` (`parity` 7, `ik-glm5next` 1) go through it. No test module, driver, oracle or tolerance was
  changed (`git diff --stat 5730cc4 abaea12` in section 7 lists only the checker, its expectations, its test and the
  two workflows; the later commits add only docs and the scripts of this report).
- **Normal runs, CONFIRMED (section 3):** 17/17 steps pass through the checker with the expected counts, at
  `ec7630a` (ci.yml) and at `79e2e74` / ik `5194a9e` (parity-r1-pin.yml).
- **It bites, CONFIRMED (section 4):** all 34 broken runs now fail (`after_rc=1`), at both pins; the one legitimate
  pin-dependent skip (glm5-next MTP at `ec7630a`) passes there and fails when forced at `79e2e74`.
- **Unit tests, CONFIRMED (section 5):** `tests/test_check_unittest.py`, 27 tests, no dumps; 13/13 mutations of the
  checker, the expectations and the workflows make it fail (at `abaea12`; at `441e61b`, after the audit: 33 tests,
  13/13 and 6/6 for the audit fixes, section 9).
- **Corrected after the audit of `19d09d2` (section 9):** the command was not bound to the commit it was checked at
  (FAIL 1, fixed in `60272fb`); four MINOR items; the counts of commit `4a0f097`'s message.
- **NOT RUN:** the new steps on GitHub Actions (this task does not push). Gap 11 (b), (c) and (d): out of scope,
  section 6.

## 1. Inventory: steps, drivers and the expected unittest result

The expectations file is `tools/ci/parity_expectations.json`; every number below comes from the N13 runs of section 3
(and the baseline runs without the checker, which gave the same counts), cross-checked with the earlier evidence
where it has the same command. "skips" lists the expected skipped tests; `-` = none.

| step (`--step`) | workflows / job (line at HEAD) | driver | modules (tests each) | `ec7630a` | `79e2e74` (ik: `5194a9e`) | earlier evidence |
|---|---|---|---|---|---|---|
| `ggml-gdn` | ci.yml cpu-parity (`ci.yml:22-29`) | inline, `tests/ggml_parity/gdn_harness` | test_kda_vs_ggml 3, test_fitplan 15 | Ran 18, - | not run there | none with this module pair (N13 only) |
| `g1-glm` | ci.yml cpu-parity (`:31`), parity-r1-pin parity (`:51`) | `tools/tinygen/run_parity.sh` | test_glm_tiny_parity 9 | Ran 9, - | Ran 9, - | 9: `docs/evidence/2026-10-10-glm5next-decode-parity.md:163` (with `GlmTinyWholeModel`, N9); 8 before N9: `2026-10-02-g1-parity.md:24`, `2026-10-10-parity-r1-pin.md:93` |
| `g1-mimo` | ci.yml (`:33`), parity-r1-pin (`:53`) | `run_parity_mimo.sh` | test_mimo_tiny_parity 5 | Ran 5, - | Ran 5, - | `2026-10-02-mimo-g1-parity.md:15`; `2026-10-10-parity-r1-pin.md:94` (79e2e74) |
| `n2-qwen4exp` | ci.yml (`:35`), parity-r1-pin (`:55`) | `run_parity_qwen4exp.sh` | test_qwen4exp_tiny_parity 17 | Ran 17, - | Ran 17, - | `2026-10-10-qwen4exp-tiny-parity.md:196` (ec7630a); `2026-10-10-parity-r1-pin.md:95` (79e2e74) |
| `n6-qwen4exp-decode` | ci.yml (`:37`), parity-r1-pin (`:57`) | `run_parity_qwen4exp_decode.sh` | decode 40, mtp 6, chunked_ref 8 | Ran 54, - | Ran 54, - | `2026-10-10-qwen4exp-decode-parity.md:322` (ec7630a, `:7`); `2026-10-10-parity-r1-pin.md:63` (79e2e74) |
| `n9-glm5next-decode` | ci.yml (`:39`), parity-r1-pin (`:59`) | `run_parity_glm5next_decode.sh` | decode 45, mtp 6, chunked_ref 7 | Ran 52, skips: `tests.test_glm5next_mtp_parity.Mtp.setUpClass` "no MTP run under ... (llama.cpp ec7630a: no glm5-next MTP graph at ec7630a, ...)" | Ran 58, - | `2026-10-10-glm5next-decode-parity.md:220-228` (ec7630a, same skip line), `:275-277` (79e2e74) |
| `n10-f16fa` | ci.yml (`:41`), parity-r1-pin (`:61`) | `run_parity_f16fa.sh` (defaults `PARITY_KV=f16 PARITY_FA=on`) | qwen4exp_f16fa 75, glm5next_f16fa 84, fattn_ref 11 | Ran 170, - | Ran 170, - | `2026-10-10-quant-fa-parity.md:412` (ec7630a), `:543` (79e2e74) |
| `n10-quant` | ci.yml (`:43`), parity-r1-pin (`:63`) | `run_parity_quant.sh` | qwen4exp_quant 48, glm5next_quant 40, quant_ref 11 | Ran 99, - | Ran 99, - | `2026-10-10-quant-fa-parity.md:1632` (ec7630a), `:1660` (79e2e74) |
| `gguf-audit` | ci.yml cpu-parity (`:50-53`) | inline, the pinned gguf-py | roofline 26, cachesim 14, gguf_quant_audit 7, offline_weights 5 | Ran 52, - | not run there | none with this module set (N13 only) |
| `n12-ik-glm5next` | parity-r1-pin ik-glm5next (`:85-88`) | `run_parity_ik_glm5next.sh` | test_ik_glm5next_parity 43 | not run there | Ran 43, - (ik `5194a9e`) | `2026-10-11-n12-ik-glm5next-cpu.md:487` |

The per-module split is from the test lines of the logs (`grep -oE '^[a-z_A-Z0-9]+ \((tests\.[a-z0-9_]+)\.'`, counted
per module); the checker compares only the total and the skips. Steps of the two jobs that run no unittest are not
wrapped and need not be: `tools/g0/run_trace_smoke.sh` and `scripts/traces/smoke_tiny.sh` run under
`set -euo pipefail` (`tools/g0/run_trace_smoke.sh:5`, `scripts/traces/smoke_tiny.sh:12`) and check with `assert`, so
they cannot pass by skipping. The `unit tests` step (`discover`) is not wrapped: it skips the dump tests by design;
see section 6.

## 2. Mechanism and why this one

Chosen: a checker with a committed expectations file (the first option of the task); no `PARITY_REQUIRE_DUMPS` flag.

- **One place, nothing in the tests changes.** The flag would need an edit in each of the 12 dump-parity modules (and
  `tests/test_kda_vs_ggml.py`), and a new module that forgot it would skip silently again. The checker reads what
  unittest prints, so the tests check exactly what they checked before, with the same tolerances.
- **It sees every way a test can go missing**, not only a missing variable: an empty or wrong dump directory, a run the
  driver did not produce (one class skipped), a module dropped from the driver's `python3 -m unittest` line (`Ran`
  drops), a test that stops being collected. Section 4 shows unset and empty-directory breaks for every step;
  `tests/test_check_unittest.py` adds the partial-dump case.
- **Legitimate skips are explicit and pinned**: an expected skip is a test id plus a reason regex, for one commit. The
  glm5-next MTP skip is allowed at `ec7630a` only; the same skip at `79e2e74` fails (row `nomtp`, section 4).
- **Pin-following**: the key is the full sha of `--checkout` (`git rev-parse HEAD`; the drivers compute the same value
  for `PARITY_LLAMA_COMMIT` / `PARITY_IK_COMMIT`, but export it only to their own unittest process). The pin variable
  is compared only if it is set in the checker's own environment; the drivers' export is not visible to it, so in CI
  that comparison is inert. Corrected after the audit (section 9): at `abaea12` nothing tied the wrapped command to
  `--checkout`; since `60272fb` the checker compares, before running, the commit of the driver's checkout argument (or,
  for an inline unittest step, of the checkout holding `LLAMA_CPP_GGUF_PY`) with it. A commit without an entry fails
  the step with a message naming the step and the commit, so a pin bump in `tools/engines/pins.json` needs a reviewed
  entry; `tests/test_check_unittest.py` already fails in the `unit tests` step when an entry for a workflow's pin is
  missing (mutation `pin_bump`, section 5).

Rules of `check()` (`tools/ci/check_unittest.py:153@abaea12`): exactly one unittest summary in the output; status `OK`; no
`failures`, `errors`, `expected failures`, `unexpected successes`; `Ran N` equal; `skipped=K` equal to the number of
expected skips; K skip lines found in the output (an unattributed skip fails); each skip matched to a distinct expected
entry by test id and `re.fullmatch` of the reason. If the wrapped command itself fails, its status is returned unchanged
(`:271@abaea12`). Since `60272fb` the command is also bound to the commit before it runs (section 9). Cost: adding a test to a parity module needs the count updated in the same change; the step says which.
Limits (PROVISIONAL, by construction): tests that ran are counted, not named; the parser reads the verbose format of
Python 3.11+ (`name (module.Class.name) ... skipped '<reason>'`), checked here under 3.11.17, 3.12.3 and 3.13.16
(the runner `ubuntu-24.04` has a 3.12 system Python, PROVISIONAL from the Ubuntu release; its exact version is UNKNOWN
here); the `n10-f16fa` entries are for the driver's defaults (`PARITY_KV=f16 PARITY_FA=on`, what CI runs); the other three combinations also gave 170 in
N10 (`2026-10-10-quant-fa-parity.md:671-748`) but are not keyed.

Wiring: each step's command is prefixed, e.g. `ci.yml:39`:

```
      - name: N9 glm5-next several ubatches, token-by-token decode and rollback against the real glm5-next graph (CPU; this pin has no glm5-next MTP graph)
        run: python3 tools/ci/check_unittest.py --step n9-glm5next-decode --checkout ../llama.cpp -- tools/tinygen/run_parity_glm5next_decode.sh ../llama.cpp ../build-tools
```

## 3. Normal runs through the checker (both pins)

Every step as the workflows run it now (`python3 tools/ci/check_unittest.py --step <id> --checkout <checkout> --
<driver or unittest command>`), from a scratch cwd, at HEAD `abaea12` (`git rev-parse HEAD` printed right before the script started:
`abaea128ac7a9a7e576813bc25438edcbd861d31`), builds reused from N10/N12 copied and re-pointed at this worktree
(`cmake` reconfigure + incremental build). Script `<S>/n13/run_wrapped.sh` (`OUTDIR=wrapped-final`); summary lines:

```
ec g1-glm wrapper_rc=0 wall_s=3 | Ran 9 tests in 0.376s OK | ::notice::g1-glm @ llama.cpp ec7630a: Ran 9, skipped 0 as expected 
ec g1-mimo wrapper_rc=0 wall_s=1 | Ran 5 tests in 0.082s OK | ::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected 
ec n2-qwen4exp wrapper_rc=0 wall_s=2 | Ran 17 tests in 0.382s OK | ::notice::n2-qwen4exp @ llama.cpp ec7630a: Ran 17, skipped 0 as expected 
ec n6-qwen4exp-decode wrapper_rc=0 wall_s=22 | Ran 54 tests in 3.376s OK | ::notice::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 54, skipped 0 as expected 
ec n9-glm5next-decode wrapper_rc=0 wall_s=20 | Ran 52 tests in 5.467s OK (skipped=1) | ::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass) 
ec n10-f16fa wrapper_rc=0 wall_s=49 | Ran 170 tests in 21.173s OK | ::notice::n10-f16fa @ llama.cpp ec7630a: Ran 170, skipped 0 as expected 
ec n10-quant wrapper_rc=0 wall_s=113 | Ran 99 tests in 65.885s OK | ::notice::n10-quant @ llama.cpp ec7630a: Ran 99, skipped 0 as expected 
ec ggml-gdn wrapper_rc=0 wall_s=0 | Ran 18 tests in 0.333s OK | ::notice::ggml-gdn @ llama.cpp ec7630a: Ran 18, skipped 0 as expected 
ec gguf-audit wrapper_rc=0 wall_s=3 | Ran 52 tests in 2.100s OK | ::notice::gguf-audit @ llama.cpp ec7630a: Ran 52, skipped 0 as expected 
r1 g1-glm wrapper_rc=0 wall_s=1 | Ran 9 tests in 0.401s OK | ::notice::g1-glm @ llama.cpp 79e2e74: Ran 9, skipped 0 as expected 
r1 g1-mimo wrapper_rc=0 wall_s=2 | Ran 5 tests in 0.076s OK | ::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected 
r1 n2-qwen4exp wrapper_rc=0 wall_s=2 | Ran 17 tests in 0.390s OK | ::notice::n2-qwen4exp @ llama.cpp 79e2e74: Ran 17, skipped 0 as expected 
r1 n6-qwen4exp-decode wrapper_rc=0 wall_s=27 | Ran 54 tests in 3.659s OK | ::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected 
r1 n9-glm5next-decode wrapper_rc=0 wall_s=29 | Ran 58 tests in 5.718s OK | ::notice::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 58, skipped 0 as expected 
r1 n10-f16fa wrapper_rc=0 wall_s=48 | Ran 170 tests in 20.800s OK | ::notice::n10-f16fa @ llama.cpp 79e2e74: Ran 170, skipped 0 as expected 
r1 n10-quant wrapper_rc=0 wall_s=111 | Ran 99 tests in 65.652s OK | ::notice::n10-quant @ llama.cpp 79e2e74: Ran 99, skipped 0 as expected 
ik n12-ik-glm5next wrapper_rc=0 wall_s=20 | Ran 43 tests in 13.201s OK | ::notice::n12-ik-glm5next @ ik_llama.cpp 5194a9e: Ran 43, skipped 0 as expected 
```

The same counts came from the drivers without the checker (`<S>/n13/run_baseline.sh`, `<S>/n13/run_base_ik.sh`):

```
ec run_parity rc=0 wall_s=2 Ran 9 tests in 0.521s OK 
ec run_parity_mimo rc=0 wall_s=1 Ran 5 tests in 0.073s OK 
ec run_parity_qwen4exp rc=0 wall_s=1 Ran 17 tests in 0.306s OK 
ec run_parity_qwen4exp_decode rc=0 wall_s=7 Ran 54 tests in 3.471s OK 
ec run_parity_glm5next_decode rc=0 wall_s=19 Ran 52 tests in 6.391s OK (skipped=1) 
ec run_parity_f16fa rc=0 wall_s=55 Ran 170 tests in 25.661s OK 
ec run_parity_quant rc=0 wall_s=111 Ran 99 tests in 66.386s OK 
r1 run_parity rc=0 wall_s=2 Ran 9 tests in 0.383s OK 
r1 run_parity_mimo rc=0 wall_s=1 Ran 5 tests in 0.071s OK 
r1 run_parity_qwen4exp rc=0 wall_s=2 Ran 17 tests in 0.380s OK 
r1 run_parity_qwen4exp_decode rc=0 wall_s=28 Ran 54 tests in 3.731s OK 
r1 run_parity_glm5next_decode rc=0 wall_s=29 Ran 58 tests in 5.726s OK 
r1 run_parity_f16fa rc=0 wall_s=58 Ran 170 tests in 20.202s OK 
r1 run_parity_quant rc=0 wall_s=103 Ran 99 tests in 65.690s OK 
ik run_parity_ik_glm5next rc=0 wall_s=20 Ran 43 tests in 13.001s OK 
ec-ggml-gdn.log:Ran 18 tests in 0.301s
ec-ggml-gdn.log:OK
ec-gguf-audit.log:Ran 52 tests in 2.159s
ec-gguf-audit.log:OK
```

## 4. It bites: broken copies of every step, both pins

Commit (added after the audit, MINOR 5): this run was made on the working tree while `1e92135` and `862b1ab` were
being committed; the checker and the expectations it used are those of `862b1ab` (`git diff 862b1ab 19d09d2 --
tools/ci/` is empty). The checker changed in `60272fb`; the run repeated at `441e61b` is in section 9, and its outputs
moved from `<S>/n13/bite/` to `<S>/n13/bite-first/` to make room for the repeat (the paths below are as written by
the script).

`<S>/n13/run_bites.sh`. Each driver is copied to `<S>/n13/bite/copies/<variant>-<driver>.sh` with two edits only:
`ROOT` fixed to the worktree and the break; `diff -u` of every copy is saved next to it. Variants: `unset` (the dump
variable removed from the unittest line), `empty` (the dump variable set to a fresh `mktemp -d`), `nomtp`
(`run_parity_glm5next_decode.sh` with its MTP run disabled). For f16fa and quant the glm5-next half is broken
(`GLM5NEXT_F16FA_DUMP`, `GLM5NEXT_QUANT_DUMP`): half the step still runs, which is the case a count catches and an "all
skipped" test would not. The two inline steps are broken in the command line (`GGML_GDN_HARNESS`, `LLAMA_CPP_GGUF_PY`).
`before_rc` is what the step returned without the checker (the command's own status, which the checker returns
unchanged when it is not 0, `tools/ci/check_unittest.py:271@abaea12`; 0 when the checker went on to compare), `after_rc` what
the step returns now. Two of the diffs:

```
# <S>/n13/bite/copies/unset-run_parity_glm5next_decode.sh.diff (the ---/+++ header lines removed)
@@ -9,7 +9,7 @@
 # GLM5NEXT_DECODE_WORK=<dir> keeps the artifacts in <dir> instead of a fresh temporary directory.
 set -euo pipefail
 LLAMA=$(cd "${1:?llama.cpp checkout}" && pwd); BUILD=${2:-build-tools}
-ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${GLM5NEXT_DECODE_WORK:-$(mktemp -d)}
+ROOT=/home/user/wt-n13; WORK=${GLM5NEXT_DECODE_WORK:-$(mktemp -d)}
 mkdir -p "$WORK"
 cmake -S "$ROOT/tools/g0" -B "$BUILD" -DLLAMA_CPP_DIR="$LLAMA" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >/dev/null
 cmake --build "$BUILD" -j"$(nproc)" --target dump-tensors >/dev/null
@@ -55,7 +55,7 @@
 fi
 # the commit decides which DSA decode path and which k-pool padding the dumps show (see the test)
 export PARITY_LLAMA_COMMIT; PARITY_LLAMA_COMMIT=$(git -C "$LLAMA" rev-parse HEAD)
-cd "$ROOT" && GLM5NEXT_DECODE_DUMP="$WORK/n9" GLM5NEXT_TINY_GGUF="$WORK/glm5next.gguf" \
+cd "$ROOT" && GLM5NEXT_TINY_GGUF="$WORK/glm5next.gguf" \
   GLM5NEXT_MTP_GGUF="$WORK/glm5next_mtp.gguf" \
   python3 -m unittest tests.test_glm5next_decode_parity tests.test_glm5next_mtp_parity tests.test_glm5next_chunked_ref -v
 echo "artifacts: $WORK"
# <S>/n13/bite/copies/empty-run_parity_ik_glm5next.sh.diff (the ---/+++ header lines removed)
@@ -9,7 +9,7 @@
 # artifacts in <dir> instead of a fresh temporary directory.
 set -euo pipefail
 IK=$(cd "${1:?ik_llama.cpp checkout}" && pwd); BUILD=${2:-build-ik}
-ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${IK_GLM5NEXT_WORK:-$(mktemp -d)}
+ROOT=/home/user/wt-n13; WORK=${IK_GLM5NEXT_WORK:-$(mktemp -d)}
 mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
 : "${LLAMA_CPP_GGUF_PY:?set LLAMA_CPP_GGUF_PY to a llama.cpp gguf-py (the writer and the test read GGUF with it)}"
 PIN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['engines']['ik_llama.cpp']['sha'])" \
@@ -59,6 +59,6 @@
 # the same weights under the llama.cpp name "glm5-next" (ik's alias): logits only
 MODEL="$WORK/llama.gguf" PFX="" run alias 64 "" "${REF[@]}" --dsa
 export PARITY_IK_COMMIT="$HEAD"
-cd "$ROOT" && IK_GLM5NEXT_DUMP="$WORK/n12" IK_GLM5NEXT_GGUF="$WORK/ik.gguf" \
+cd "$ROOT" && IK_GLM5NEXT_DUMP="$(mktemp -d)" IK_GLM5NEXT_GGUF="$WORK/ik.gguf" \
   python3 -m unittest tests.test_ik_glm5next_parity -v
 echo "artifacts: $WORK"
```

Summary (`<S>/n13/bite/summary.txt` as written by the script: one line per run, after the second `|` the first
three annotations of the checker, each cut at 260 characters; the full outputs are in `<S>/n13/bite/logs/`):

```
ec g1-glm unset before_rc=0 after_rc=1 wall_s=2 | Ran 9 tests in 0.000s OK (skipped=9) | ::error::g1-glm @ llama.cpp ec7630a: skipped=9, expected skipped=0 ::error::g1-glm @ llama.cpp ec7630a: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_indexer_and_selection: 'run tools/tinygen/run_parity.sh' ::error::g1-glm @ llama.cpp ec7630a: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_mla_attention: 'run tools/tinygen/run_parity.sh' 
ec g1-glm empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.162s FAILED (errors=3) | ::error::g1-glm @ llama.cpp ec7630a: the command exited with status 1 
ec g1-mimo unset before_rc=0 after_rc=1 wall_s=1 | Ran 5 tests in 0.000s OK (skipped=5) | ::error::g1-mimo @ llama.cpp ec7630a: skipped=5, expected skipped=0 ::error::g1-mimo @ llama.cpp ec7630a: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_attention_swa_sinks_gqa: 'run tools/tinygen/run_parity.sh mimo' ::error::g1-mimo @ llama.cpp ec7630a: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_plain_swiglu: 'run tools/tinygen/run_parity.sh mimo' 
ec g1-mimo empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.064s FAILED (errors=1) | ::error::g1-mimo @ llama.cpp ec7630a: the command exited with status 1 
ec n2-qwen4exp unset before_rc=0 after_rc=1 wall_s=2 | Ran 17 tests in 0.000s OK (skipped=17) | ::error::n2-qwen4exp @ llama.cpp ec7630a: skipped=17, expected skipped=0 ::error::n2-qwen4exp @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attention_core_and_gate: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' ::error::n2-qwen4exp @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attn_projections_norms_rope: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' 
ec n2-qwen4exp empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.069s FAILED (errors=1) | ::error::n2-qwen4exp @ llama.cpp ec7630a: the command exited with status 1 
ec n6-qwen4exp-decode unset before_rc=0 after_rc=1 wall_s=23 | Ran 54 tests in 0.369s OK (skipped=46) | ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skipped=46, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_attention_over_the_kv_cache: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_carried_state: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh' 
ec n6-qwen4exp-decode empty before_rc=0 after_rc=1 wall_s=25 | Ran 8 tests in 0.377s OK (skipped=5) | ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 8 tests, expected 54 ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skipped=5, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.ywL1qUvZdY' 
ec n9-glm5next-decode unset before_rc=0 after_rc=1 wall_s=5 | Ran 58 tests in 0.390s OK (skipped=51) | ::error::n9-glm5next-decode @ llama.cpp ec7630a: Ran 58 tests, expected 52 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skipped=51, expected skipped=1 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_decode_parity.Decode.test_dsa_pool_cache_and_selection: 'set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_glm5next_decode.s 
ec n9-glm5next-decode empty before_rc=0 after_rc=1 wall_s=15 | Ran 7 tests in 0.328s OK (skipped=6) | ::error::n9-glm5next-decode @ llama.cpp ec7630a: Ran 7 tests, expected 52 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skipped=6, expected skipped=1 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.Wy920CiAfD' 
ec n10-f16fa unset before_rc=0 after_rc=1 wall_s=32 | Ran 170 tests in 8.931s OK (skipped=84) | ::error::n10-f16fa @ llama.cpp ec7630a: skipped=84, expected skipped=0 ::error::n10-f16fa @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_output_vs_plain_oracle: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' ::error::n10-f16fa @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_path: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' 
ec n10-f16fa empty before_rc=0 after_rc=1 wall_s=38 | Ran 86 tests in 8.922s OK (skipped=6) | ::error::n10-f16fa @ llama.cpp ec7630a: Ran 86 tests, expected 170 ::error::n10-f16fa @ llama.cpp ec7630a: skipped=6, expected skipped=0 ::error::n10-f16fa @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.1m3xeKY5Lc' 
ec n10-quant unset before_rc=0 after_rc=1 wall_s=56 | Ran 59 tests in 31.863s OK (skipped=8) | ::error::n10-quant @ llama.cpp ec7630a: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp ec7630a: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'set GLM5NEXT_QUANT_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_quant.sh' 
ec n10-quant empty before_rc=0 after_rc=1 wall_s=81 | Ran 59 tests in 24.747s OK (skipped=8) | ::error::n10-quant @ llama.cpp ec7630a: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp ec7630a: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'no run Q4_K_M/decode under <S>/n13/tmp/tmp.R5fkua65Kv' 
ec n9-glm5next-decode nomtp before_rc=0 after_rc=0 wall_s=18 | Ran 52 tests in 6.110s OK (skipped=1) | ::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass) 
r1 g1-glm unset before_rc=0 after_rc=1 wall_s=1 | Ran 9 tests in 0.000s OK (skipped=9) | ::error::g1-glm @ llama.cpp 79e2e74: skipped=9, expected skipped=0 ::error::g1-glm @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_indexer_and_selection: 'run tools/tinygen/run_parity.sh' ::error::g1-glm @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_mla_attention: 'run tools/tinygen/run_parity.sh' 
r1 g1-glm empty before_rc=1 after_rc=1 wall_s=3 | Ran 0 tests in 0.264s FAILED (errors=3) | ::error::g1-glm @ llama.cpp 79e2e74: the command exited with status 1 
r1 g1-mimo unset before_rc=0 after_rc=1 wall_s=1 | Ran 5 tests in 0.000s OK (skipped=5) | ::error::g1-mimo @ llama.cpp 79e2e74: skipped=5, expected skipped=0 ::error::g1-mimo @ llama.cpp 79e2e74: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_attention_swa_sinks_gqa: 'run tools/tinygen/run_parity.sh mimo' ::error::g1-mimo @ llama.cpp 79e2e74: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_plain_swiglu: 'run tools/tinygen/run_parity.sh mimo' 
r1 g1-mimo empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.062s FAILED (errors=1) | ::error::g1-mimo @ llama.cpp 79e2e74: the command exited with status 1 
r1 n2-qwen4exp unset before_rc=0 after_rc=1 wall_s=1 | Ran 17 tests in 0.000s OK (skipped=17) | ::error::n2-qwen4exp @ llama.cpp 79e2e74: skipped=17, expected skipped=0 ::error::n2-qwen4exp @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attention_core_and_gate: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' ::error::n2-qwen4exp @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attn_projections_norms_rope: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' 
r1 n2-qwen4exp empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.071s FAILED (errors=1) | ::error::n2-qwen4exp @ llama.cpp 79e2e74: the command exited with status 1 
r1 n6-qwen4exp-decode unset before_rc=0 after_rc=1 wall_s=21 | Ran 54 tests in 0.343s OK (skipped=46) | ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skipped=46, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_attention_over_the_kv_cache: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_carried_state: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh' 
r1 n6-qwen4exp-decode empty before_rc=0 after_rc=1 wall_s=23 | Ran 8 tests in 0.399s OK (skipped=5) | ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 8 tests, expected 54 ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skipped=5, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.UXQZTpzRQ7' 
r1 n9-glm5next-decode unset before_rc=0 after_rc=1 wall_s=14 | Ran 58 tests in 0.322s OK (skipped=51) | ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=51, expected skipped=0 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_decode_parity.Decode.test_dsa_pool_cache_and_selection: 'set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_glm5next_decode.s ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_decode_parity.Decode.test_dsa_projections: 'set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_glm5next_decode.sh' 
r1 n9-glm5next-decode empty before_rc=0 after_rc=1 wall_s=19 | Ran 7 tests in 0.320s OK (skipped=6) | ::error::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 7 tests, expected 58 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=6, expected skipped=0 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.QryRrcnB2s' 
r1 n10-f16fa unset before_rc=0 after_rc=1 wall_s=38 | Ran 170 tests in 8.319s OK (skipped=84) | ::error::n10-f16fa @ llama.cpp 79e2e74: skipped=84, expected skipped=0 ::error::n10-f16fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_output_vs_plain_oracle: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' ::error::n10-f16fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_path: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' 
r1 n10-f16fa empty before_rc=0 after_rc=1 wall_s=39 | Ran 86 tests in 8.259s OK (skipped=6) | ::error::n10-f16fa @ llama.cpp 79e2e74: Ran 86 tests, expected 170 ::error::n10-f16fa @ llama.cpp 79e2e74: skipped=6, expected skipped=0 ::error::n10-f16fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.ZwDqYIw8rN' 
r1 n10-quant unset before_rc=0 after_rc=1 wall_s=67 | Ran 59 tests in 25.589s OK (skipped=8) | ::error::n10-quant @ llama.cpp 79e2e74: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp 79e2e74: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'set GLM5NEXT_QUANT_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_quant.sh' 
r1 n10-quant empty before_rc=0 after_rc=1 wall_s=73 | Ran 59 tests in 24.669s OK (skipped=8) | ::error::n10-quant @ llama.cpp 79e2e74: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp 79e2e74: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'no run Q4_K_M/decode under <S>/n13/tmp/tmp.U2W00fHdLT' 
r1 n9-glm5next-decode nomtp before_rc=0 after_rc=1 wall_s=11 | Ran 52 tests in 5.671s OK (skipped=1) | ::error::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 52 tests, expected 58 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=1, expected skipped=0 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_mtp_parity.Mtp.setUpClass: 'no MTP run under <S>/n13/tmp/tmp.MO0GGCbHfd/n9 (llama.cpp 79e2 
ec ggml-gdn unset(GGML_GDN_HARNESS) before_rc=0 after_rc=1 wall_s=1 | Ran 18 tests in 0.241s OK (skipped=3) | ::error::ggml-gdn @ llama.cpp ec7630a: skipped=3, expected skipped=0 ::error::ggml-gdn @ llama.cpp ec7630a: skip not allowed here: tests.test_kda_vs_ggml.OracleVsGgml.test_gdn_scalar_gate: 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)' ::error::ggml-gdn @ llama.cpp ec7630a: skip not allowed here: tests.test_kda_vs_ggml.OracleVsGgml.test_glm_head_geometry: 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)' 
ec ggml-gdn empty(GGML_GDN_HARNESS=<empty dir>/gdn_harness) before_rc=1 after_rc=1 wall_s=0 | Ran 18 tests in 0.234s FAILED (errors=3) | ::error::ggml-gdn @ llama.cpp ec7630a: the command exited with status 1 
ec gguf-audit unset(LLAMA_CPP_GGUF_PY) before_rc=0 after_rc=1 wall_s=3 | Ran 52 tests in 2.130s OK (skipped=1) | ::error::gguf-audit @ llama.cpp ec7630a: skipped=1, expected skipped=0 ::error::gguf-audit @ llama.cpp ec7630a: skip not allowed here: tests.test_gguf_quant_audit.Audit.test_type_table_matches_pinned_gguf_py: 'set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py' 
ec gguf-audit empty(LLAMA_CPP_GGUF_PY=<empty dir>) before_rc=1 after_rc=1 wall_s=2 | Ran 52 tests in 1.982s FAILED (errors=1) | ::error::gguf-audit @ llama.cpp ec7630a: the command exited with status 1 
ik n12-ik-glm5next unset before_rc=0 after_rc=1 wall_s=7 | Ran 43 tests in 0.001s OK (skipped=43) | ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skipped=43, expected skipped=0 ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skip not allowed here: tests.test_ik_glm5next_parity.IkAlias.test_glm5_next_name_gives_identical_logits: 'set IK_GLM5NEXT_DUMP, IK_GLM5NEXT_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_ik_glm5next.sh' ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skip not allowed here: tests.test_ik_glm5next_parity.IkDecode.test_cache_stores: 'set IK_GLM5NEXT_DUMP, IK_GLM5NEXT_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_ik_glm5next.sh' 
ik n12-ik-glm5next empty before_rc=0 after_rc=1 wall_s=8 | Ran 1 test in 0.001s OK (skipped=8) | ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: Ran 1 tests, expected 43 ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skipped=8, expected skipped=0 ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skip not allowed here: tests.test_ik_glm5next_parity.IkAlias.test_glm5_next_name_gives_identical_logits: 'no run alias under <S>/n13/ 
```

Tally (`python3` over the summary lines):

```
36 rows
broken - before_rc=0 after_rc=1 26
broken - before_rc=1 after_rc=1 8
nomtp ec before_rc=0 after_rc=0 1
nomtp r1 before_rc=0 after_rc=1 1
```

- 26 broken runs were green before and are red now: every `unset` (17 of them: 7 drivers x 2 pins, the two inline
  steps, ik) and the `empty` runs of N6, N9, f16fa, quant (both pins) and ik.
- 8 broken runs were red already (`empty` for G1, MiMo, N2 at both pins and both inline steps): their tests open the
  files and raise. They stay red; the checker reports the command's status.
- `nomtp` at `ec7630a` passes (that is the normal state there: no MTP graph); at `79e2e74` it fails. The full
  annotation set of that run:

```
mtp: not run, this llama.cpp has no glm5-next MTP graph (src/models/glm5-next.cpp)
Ran 52 tests in 5.671s
OK (skipped=1)
::error::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 52 tests, expected 58
::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=1, expected skipped=0
::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_mtp_parity.Mtp.setUpClass: 'no MTP run under <S>/n13/tmp/tmp.MO0GGCbHfd/n9 (llama.cpp 79e2e74: no glm5-next MTP graph at ec7630a, see the module docstring)'
```

## 5. Unit tests (no dumps)

```
$ cd <S>/n13/cwd && PYTHONPATH=/home/user/wt-n13 python3 -m unittest tests.test_check_unittest -v
test_count_mismatch (tests.test_check_unittest.Check.test_count_mismatch) ... ok
test_exact_match_passes (tests.test_check_unittest.Check.test_exact_match_passes) ... ok
test_extra_skip (tests.test_check_unittest.Check.test_extra_skip) ... ok
test_failures_and_other_outcomes (tests.test_check_unittest.Check.test_failures_and_other_outcomes) ... ok
test_missing_expected_skip (tests.test_check_unittest.Check.test_missing_expected_skip) ... ok
test_right_reason_on_another_test (tests.test_check_unittest.Check.test_right_reason_on_another_test)
The allowed reason does not license the skip of another class (both id and reason must match). ... ok
test_unattributed_skip (tests.test_check_unittest.Check.test_unattributed_skip) ... ok
test_unknown_commit (tests.test_check_unittest.Check.test_unknown_commit) ... ok
test_wrong_reason (tests.test_check_unittest.Check.test_wrong_reason) ... ok
test_ci_pin_is_the_agents_md_pin (tests.test_check_unittest.ExpectationsFile.test_ci_pin_is_the_agents_md_pin) ... ok
test_every_parity_step_goes_through_the_checker (tests.test_check_unittest.ExpectationsFile.test_every_parity_step_goes_through_the_checker) ... ok
test_expectations_exist_for_the_pin_each_step_runs_at (tests.test_check_unittest.ExpectationsFile.test_expectations_exist_for_the_pin_each_step_runs_at) ... ok
test_real_modules_without_their_environment_fail_at_every_pin (tests.test_check_unittest.ExpectationsFile.test_real_modules_without_their_environment_fail_at_every_pin)
What a broken driver would leave: the step's test modules with no dump / model / gguf-py variables. ... ok
test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives (tests.test_check_unittest.ExpectationsFile.test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives)
At ec7630a the N9 driver produces no MTP run; tests/test_glm5next_mtp_parity.py then skips with a reason ... ok
test_one_summary_required (tests.test_check_unittest.Parse.test_one_summary_required) ... ok
test_real_verbose_output_of_this_python (tests.test_check_unittest.Parse.test_real_verbose_output_of_this_python) ... ok
test_skip_without_description_is_kept_unattributed (tests.test_check_unittest.Parse.test_skip_without_description_is_kept_unattributed) ... ok
test_synthetic_failed_summary (tests.test_check_unittest.Parse.test_synthetic_failed_summary) ... ok
test_bad_commits (tests.test_check_unittest.ResolveCommit.test_bad_commits) ... ok
test_checkout_head_and_pin_variable (tests.test_check_unittest.ResolveCommit.test_checkout_head_and_pin_variable) ... ok
test_command_status_is_kept (tests.test_check_unittest.Wrapper.test_command_status_is_kept) ... ok
test_complete_dumps_pass_at_the_pin_that_runs_everything (tests.test_check_unittest.Wrapper.test_complete_dumps_pass_at_the_pin_that_runs_everything) ... ok
test_legitimate_skip_passes_at_its_pin_only (tests.test_check_unittest.Wrapper.test_legitimate_skip_passes_at_its_pin_only) ... ok
test_missing_environment_fails_at_every_pin (tests.test_check_unittest.Wrapper.test_missing_environment_fails_at_every_pin) ... ok
test_partial_or_empty_dumps_fail (tests.test_check_unittest.Wrapper.test_partial_or_empty_dumps_fail) ... ok
test_saved_log_mode (tests.test_check_unittest.Wrapper.test_saved_log_mode) ... ok
test_unknown_commit_fails (tests.test_check_unittest.Wrapper.test_unknown_commit_fails) ... ok

----------------------------------------------------------------------
Ran 27 tests in 5.871s

OK
```

The stdlib-only classes under the other interpreters (the CI runner is `ubuntu-24.04`):

```
$ PYTHONPATH=/home/user/wt-n13 python3.12 -m unittest tests.test_check_unittest.Parse tests.test_check_unittest.Check tests.test_check_unittest.Wrapper tests.test_check_unittest.ResolveCommit   # Python 3.12.3
Ran 22 tests in 1.526s

OK
$ PYTHONPATH=/home/user/wt-n13 python3.11 -m unittest tests.test_check_unittest.Parse tests.test_check_unittest.Check tests.test_check_unittest.Wrapper tests.test_check_unittest.ResolveCommit   # Python 3.11.17
Ran 22 tests in 1.196s

OK
```

Mutations on `git archive HEAD` copies (`<S>/n13/mutate_tests.sh`, `MUTDIR=mut-final`; `rc=1` = the unit tests caught
it). On the first round (`abaea12^`) the mutation `any_id` survived; `Check.test_right_reason_on_another_test` was added
in `abaea12` and catches it:

```
baseline rc=0 | Ran 27 tests in 6.754s OK | | no mutation
unwrap_g1 rc=1 | Ran 27 tests in 6.525s FAILED (failures=1, errors=1) | ERROR: test_expectations_exist_for_the_pin_each_step_runs_at FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/ci.yml', job='cpu-parity', step=None) | ci.yml: the G1 step runs its driver without the checker
new_step rc=1 | Ran 27 tests in 6.269s FAILED (failures=1, errors=1) | ERROR: test_expectations_exist_for_the_pin_each_step_runs_at FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/parity-r1-pin.yml', job='ik-glm5next', step=None) | parity-r1-pin.yml: a new driver step appended to job ik-glm5next without the checker
no_ec_n9 rc=1 | Ran 27 tests in 5.945s FAILED (failures=1, errors=1) | ERROR: test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/ci.yml', step='n9-glm5next-decode') | expectations: the ec7630a entry of n9-glm5next-decode removed
pin_bump rc=1 | Ran 27 tests in 6.200s FAILED (failures=7) | FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='g1-glm') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='g1-mimo') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n10-f16fa') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n10-quant') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n2-qwen4exp') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n6-qwen4exp-decode') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n9-glm5next-decode') | tools/engines/pins.json: llama.cpp R1 pin changed, expectations not updated
mtp_reason rc=1 | Ran 27 tests in 6.138s FAILED (errors=1) | ERROR: test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives | expectations: the MTP skip reason regex names the wrong commit
modules rc=1 | Ran 27 tests in 6.012s FAILED (failures=2) | FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/ci.yml', job='cpu-parity', step='n6-qwen4exp-decode') FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/parity-r1-pin.yml', job='parity', step='n6-qwen4exp-decode') | expectations: n6 lists one module fewer than its driver runs
no_count rc=1 | Ran 27 tests in 6.183s FAILED (failures=2) | FAIL: test_count_mismatch FAIL: test_legitimate_skip_passes_at_its_pin_only | checker: Ran N not compared
no_skipcount rc=1 | Ran 27 tests in 6.145s FAILED (failures=2) | FAIL: test_extra_skip FAIL: test_missing_expected_skip | checker: skipped=K not compared
any_id rc=1 | Ran 27 tests in 6.332s FAILED (failures=1) | FAIL: test_right_reason_on_another_test | checker: a skip matches an expected entry by reason only
no_status rc=1 | Ran 27 tests in 6.342s FAILED (failures=1) | FAIL: test_failures_and_other_outcomes | checker: status not compared
no_unattributed rc=1 | Ran 27 tests in 6.217s FAILED (failures=1) | FAIL: test_unattributed_skip | checker: skip lines not counted against skipped=K
env_ignored rc=1 | Ran 27 tests in 6.164s FAILED (failures=2) | FAIL: test_checkout_head_and_pin_variable FAIL: test_saved_log_mode | checker: PARITY_LLAMA_COMMIT disagreeing with the checkout is ignored
rc_lost rc=1 | Ran 2 tests in 0.001s FAILED (failures=2, skipped=2) Ran 27 tests in 6.086s FAILED (failures=2) | FAIL: test_a (fake_parity.Prefill.test_a) FAIL: test_b (fake_parity.Prefill.test_b) FAIL: test_command_status_is_kept FAIL: test_partial_or_empty_dumps_fail | checker: a failing command is reported but the step returns 0
```

## 6. Out of scope (gap 11 (b), (c), (d)) and what stays unchecked

- (b) `cuda-compile` is green by construction: there is no `kernels/` directory and the step exits 0 on "no kernels
  yet" (`.github/workflows/ci.yml:76-77@5730cc4`, `:82-83` at HEAD). Unchanged.
- (c) the sm_89 build of `scripts/diag/*.cu` is skipped in CI: `tests/test_diag_kit.py:557@5730cc4` skips without
  nvcc, and that test runs in the `unit tests` step (`discover`), which N13 does not wrap: its skips are by design
  (every dump test skips there), so a pinned skip list for it would have to name the diag-kit skip as allowed; the fix
  belongs to (c) (nvcc in that job). Unchanged.
- (d) the docstring of `tests/test_srvcheck.py` (branch n7; the file is not on this branch). Unchanged.
- `ref-torch-cpu` keeps its own `grep -q 'skipped'` check (`ci.yml:69` at HEAD). Unchanged.
- Other workflows on this branch (`engines-sm89.yml`, `traces-real.yml`, `vendor-*.yml`) run no parity unittest
  (`grep -ln "unittest\|run_parity" .github/workflows/*.yml` lists only `ci.yml` and `parity-r1-pin.yml`).

## 7. Full suite and the change set

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests   (in /home/user/wt-n13)
----------------------------------------------------------------------
Ran 837 tests in 125.141s

OK (skipped=354)
```

```
$ git diff --stat 5730cc4 abaea12
 .github/workflows/ci.yml            |  22 +-
 .github/workflows/parity-r1-pin.yml |  25 ++-
 tests/test_check_unittest.py        | 424 ++++++++++++++++++++++++++++++++++++
 tools/ci/check_unittest.py          | 290 ++++++++++++++++++++++++
 tools/ci/parity_expectations.json   | 161 ++++++++++++++
 5 files changed, 904 insertions(+), 18 deletions(-)
```

## 8. Reproduce

The scripts that produced sections 3-5 are committed byte-identical (`cmp` against the scratch files) in
`docs/evidence/mutation-scripts/`: `2026-10-11-n13-run-baseline.sh`, `2026-10-11-n13-run-base-ik.sh`,
`2026-10-11-n13-run-wrapped.sh`, `2026-10-11-n13-bites.sh`, `2026-10-11-n13-mutations.sh` (= `<S>/n13/run_baseline.sh`,
`run_base_ik.sh`, `run_wrapped.sh`, `run_bites.sh`, `mutate_tests.sh`). They hold the sandbox paths (`S=`, `N=`, `WT=`);
replace them to rerun. Builds they expect:

```
S=<scratch>; WT=<worktree>
cmake -S $WT/tools/g0 -B $S/n13/build-ec -DLLAMA_CPP_DIR=$S/llama-ec7630a -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON
cmake -S $WT/tools/g0 -B $S/n13/build-r1 -DLLAMA_CPP_DIR=$S/llama.cpp     -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON
cmake -S $WT/tools/ik -B $S/n13/build-ik -DIK_LLAMA_DIR=$S/n12/ik-src -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
# the ggml harness: the cmake lines of ci.yml on a copy of $S/llama-ec7630a/ggml (with ggml.pc.in touched there), then
$WT/tests/ggml_parity/build.sh $S/llama-ec7630a <ggml build dir>
# one step by hand, as CI runs it:
python3 $WT/tools/ci/check_unittest.py --step n9-glm5next-decode --checkout $S/llama.cpp -- \
    $WT/tools/tinygen/run_parity_glm5next_decode.sh $S/llama.cpp $S/n13/build-r1
# a saved log:
python3 $WT/tools/ci/check_unittest.py --step n9-glm5next-decode --commit ec7630a640789c393694fb194f1bbbf0369fc62d --log <log>
```

## 9. Corrections after the audit of `19d09d2`

The evidence-auditor checked N13 at `19d09d2` and confirmed the rest (fail-closed parsing on crafted logs, every count
against the earlier evidence, five steps through the checker, all 17 steps wrapped, the bites table and the scripts
byte-identical, 13/13 mutations, the full suite). It found 1 FAIL and 5 MINOR items, fixed in `60272fb` (checker),
`3e8cc3f` (ci.yml comment), `441e61b` (tests), this evidence commit and the CORRECTNESS commit after it.

- **FAIL 1.** `docs/reassessment/CORRECTNESS.md:311@19d09d2` said that `--checkout` is the value the drivers export as
  `PARITY_LLAMA_COMMIT` / `PARITY_IK_COMMIT` and that a mismatch is an error. False: the drivers export the variable only
  to their own unittest process, the checker reads its own environment (`tools/ci/check_unittest.py:205@19d09d2`), and
  in CI that is never set. The auditor's case `--checkout <ec7630a> -- run_parity_mimo.sh <79e2e74 checkout> build-r1`
  returned 0 with the `ec7630a` expectations. Fix `60272fb`: before running, `check_command_commit()` compares the
  step's commit with the commit of the checkout the command runs on (the driver's first argument when it is a
  directory; for an inline unittest command of a llama.cpp step, the checkout holding `LLAMA_CPP_GGUF_PY`); a mismatch
  exits 2 and runs nothing. Unit tests `CommandCommit` (5). The same section 2 sentence (MINOR 2) is reworded. The
  auditor's case and two variants at `441e61b`:

```
$ git -C /home/user/wt-n13 rev-parse --short HEAD
441e61b
$ python3 $WT/tools/ci/check_unittest.py --step g1-mimo --checkout <S>/llama-ec7630a -- $WT/tools/tinygen/run_parity_mimo.sh <S>/llama.cpp <S>/n13/build-r1; echo rc=$?
::error::check_unittest: the driver's checkout <S>/llama.cpp is at 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, but the step's commit (--checkout / --commit) is ec7630a640789c393694fb194f1bbbf0369fc62d: the expectations would be those of another commit
rc=2
$ python3 $WT/tools/ci/check_unittest.py --step g1-mimo --commit ec7630a640789c393694fb194f1bbbf0369fc62d -- $WT/tools/tinygen/run_parity_mimo.sh <S>/llama.cpp <S>/n13/build-r1; echo rc=$?
::error::check_unittest: the driver's checkout <S>/llama.cpp is at 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, but the step's commit (--checkout / --commit) is ec7630a640789c393694fb194f1bbbf0369fc62d: the expectations would be those of another commit
rc=2
$ LLAMA_CPP_GGUF_PY=<S>/llama.cpp/gguf-py python3 $WT/tools/ci/check_unittest.py --step gguf-audit --checkout <S>/llama-ec7630a -- python3 -m unittest tests.test_roofline tests.test_cachesim tests.test_gguf_quant_audit tests.test_offline_weights -v; echo rc=$?
::error::check_unittest: LLAMA_CPP_GGUF_PY <S>/llama.cpp/gguf-py is at 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, but the step's commit (--checkout / --commit) is ec7630a640789c393694fb194f1bbbf0369fc62d: the expectations would be those of another commit
rc=2
```

- **MINOR 2.** Section 2, "Pin-following": reworded (the pin variable is compared only when set in the checker's own
  environment; the drivers' export is not visible, so in CI that comparison is inert; the binding is now the
  command's checkout).
- **MINOR 3.** `tests/test_check_unittest.py` (`441e61b`): the wrapped driver's first argument must equal the
  `--checkout` value and an inline step must read `LLAMA_CPP_GGUF_PY=<checkout>/gguf-py`; every step of the three jobs
  must be wrapped or in `ALLOWED_UNWRAPPED` with its reason (checkouts, deps, the discover step, the two G0 smoke
  scripts), stale entries fail; workflow comments no longer count as step text.
- **MINOR 4.** `.github/workflows/ci.yml` (`3e8cc3f`): the checker note now says that the `unit tests` step right below
  it is not wrapped, and where the other unwrapped steps are listed.
- **MINOR 5.** Section 4 now states the commit of the first bites run; the bites are repeated at `441e61b` below.
- **MINOR 6.** The message of commit `4a0f097` says "36 broken runs (dump variable unset or an empty directory for
  every step at both pins, the MTP run forced off)". Correct: 36 runs = 34 broken (`unset` / `empty`) + 2 `nomtp`;
  the 7 drivers run at both pins, `ggml-gdn` and `gguf-audit` at `ec7630a` only (ci.yml) and `n12-ik-glm5next` at ik
  `5194a9e` only. The commit is not amended (published-history rule); this line is the correction.

Checks at `441e61b` (the commit with all code and test fixes; the evidence and CORRECTNESS commits after it change
only docs and the scripts directory).

Unit tests and the workflow YAML:

```
$ PYTHONPATH=/home/user/wt-n13 python3 -m unittest tests.test_check_unittest
.................................
----------------------------------------------------------------------
Ran 33 tests in 7.671s

OK
rc=0
$ python3 -c "import yaml; [yaml.safe_load(open(f)) for f in ('.github/workflows/ci.yml','.github/workflows/parity-r1-pin.yml')]"   (in /home/user/wt-n13)
rc=0
```

Mutations: the audit fixes (`docs/evidence/mutation-scripts/2026-10-11-n13-mutations-audit.sh`, byte-identical copy of
`<S>/n13/mutate_audit.sh`, default `MUTDIR=mut-audit-final`) and the original 13 again (`MUTDIR=mut-441e61b bash
<S>/n13/mutate_tests.sh`):

```
baseline rc=0 | Ran 33 tests in 7.211s OK | | no mutation
binding_off rc=1 | Ran 33 tests in 7.174s FAILED (failures=3) | FAIL: test_driver_on_another_checkout_fails_before_running FAIL: test_driver_on_the_checkout_of_the_step_passes FAIL: test_inline_command_is_bound_through_llama_cpp_gguf_py | checker: the command is not bound to the step's commit (check_command_commit returns None)
gguf_binding_off rc=1 | Ran 33 tests in 7.559s FAILED (failures=1) | FAIL: test_inline_command_is_bound_through_llama_cpp_gguf_py | checker: inline commands not bound through LLAMA_CPP_GGUF_PY
g1_other_checkout rc=1 | Ran 33 tests in 7.803s FAILED (failures=1) | FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/ci.yml', job='cpu-parity', step='g1-glm') | ci.yml: the G1 driver runs on ../llama-other while the step checks ../llama.cpp
gguf_other_checkout rc=1 | Ran 33 tests in 8.123s FAILED (failures=1) | FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/ci.yml', job='cpu-parity', step='gguf-audit') | ci.yml: the GGUF-audit step reads gguf-py of ../llama-other
ik_tool_step rc=1 | Ran 33 tests in 7.422s FAILED (failures=1) | FAIL: test_every_step_is_wrapped_or_allow_listed (wf='.github/workflows/parity-r1-pin.yml', job='ik-glm5next', step='a new ik driver outside tools/tinygen, unwrapped') | parity-r1-pin.yml: an unwrapped tools/ik/*.sh step appended to job ik-glm5next
renamed_step rc=1 | Ran 33 tests in 7.436s FAILED (failures=1) | FAIL: test_every_step_is_wrapped_or_allow_listed | parity-r1-pin.yml: an allow-listed step renamed (stale entry, unlisted step)
======
baseline rc=0 | Ran 33 tests in 7.251s OK | | no mutation
unwrap_g1 rc=1 | Ran 33 tests in 8.224s FAILED (failures=2, errors=1) | ERROR: test_expectations_exist_for_the_pin_each_step_runs_at FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/ci.yml', job='cpu-parity', step=None) FAIL: test_every_step_is_wrapped_or_allow_listed (wf='.github/workflows/ci.yml', job='cpu-parity', step='G1 tiny-model parity against the real glm5-next graph (CPU)') | ci.yml: the G1 step runs its driver without the checker
new_step rc=1 | Ran 33 tests in 8.079s FAILED (failures=2, errors=1) | ERROR: test_expectations_exist_for_the_pin_each_step_runs_at FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/parity-r1-pin.yml', job='ik-glm5next', step=None) FAIL: test_every_step_is_wrapped_or_allow_listed (wf='.github/workflows/parity-r1-pin.yml', job='ik-glm5next', step='a new parity step without the checker') | parity-r1-pin.yml: a new driver step appended to job ik-glm5next without the checker
no_ec_n9 rc=1 | Ran 33 tests in 7.526s FAILED (failures=1, errors=1) | ERROR: test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/ci.yml', step='n9-glm5next-decode') | expectations: the ec7630a entry of n9-glm5next-decode removed
pin_bump rc=1 | Ran 33 tests in 7.185s FAILED (failures=7) | FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='g1-glm') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='g1-mimo') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n10-f16fa') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n10-quant') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n2-qwen4exp') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n6-qwen4exp-decode') FAIL: test_expectations_exist_for_the_pin_each_step_runs_at (wf='.github/workflows/parity-r1-pin.yml', step='n9-glm5next-decode') | tools/engines/pins.json: llama.cpp R1 pin changed, expectations not updated
mtp_reason rc=1 | Ran 33 tests in 8.348s FAILED (errors=1) | ERROR: test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives | expectations: the MTP skip reason regex names the wrong commit
modules rc=1 | Ran 33 tests in 7.421s FAILED (failures=2) | FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/ci.yml', job='cpu-parity', step='n6-qwen4exp-decode') FAIL: test_every_parity_step_goes_through_the_checker (wf='.github/workflows/parity-r1-pin.yml', job='parity', step='n6-qwen4exp-decode') | expectations: n6 lists one module fewer than its driver runs
no_count rc=1 | Ran 33 tests in 7.298s FAILED (failures=2) | FAIL: test_count_mismatch FAIL: test_legitimate_skip_passes_at_its_pin_only | checker: Ran N not compared
no_skipcount rc=1 | Ran 33 tests in 7.614s FAILED (failures=2) | FAIL: test_extra_skip FAIL: test_missing_expected_skip | checker: skipped=K not compared
any_id rc=1 | Ran 33 tests in 7.450s FAILED (failures=1) | FAIL: test_right_reason_on_another_test | checker: a skip matches an expected entry by reason only
no_status rc=1 | Ran 33 tests in 7.186s FAILED (failures=1) | FAIL: test_failures_and_other_outcomes | checker: status not compared
no_unattributed rc=1 | Ran 33 tests in 7.228s FAILED (failures=1) | FAIL: test_unattributed_skip | checker: skip lines not counted against skipped=K
env_ignored rc=1 | Ran 6 tests in 0.001s OK Ran 33 tests in 7.917s FAILED (failures=3) | FAIL: test_checkout_head_and_pin_variable FAIL: test_pin_variable_of_the_checker_environment_must_agree FAIL: test_saved_log_mode | checker: PARITY_LLAMA_COMMIT disagreeing with the checkout is ignored
rc_lost rc=1 | Ran 2 tests in 0.001s FAILED (failures=2, skipped=2) Ran 33 tests in 7.271s FAILED (failures=2) | FAIL: test_a (fake_parity.Prefill.test_a) FAIL: test_b (fake_parity.Prefill.test_b) FAIL: test_command_status_is_kept FAIL: test_partial_or_empty_dumps_fail | checker: a failing command is reported but the step returns 0
```

All 17 steps through the checker again (`OUTDIR=wrapped-441e61b <S>/n13/run_wrapped.sh`; the notices now name the
checkout that was compared):

```
ec g1-glm wrapper_rc=0 wall_s=1 | Ran 9 tests in 0.398s OK | ::notice::g1-glm @ llama.cpp ec7630a: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec g1-mimo wrapper_rc=0 wall_s=2 | Ran 5 tests in 0.082s OK | ::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec n2-qwen4exp wrapper_rc=0 wall_s=2 | Ran 17 tests in 0.372s OK | ::notice::n2-qwen4exp @ llama.cpp ec7630a: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec n6-qwen4exp-decode wrapper_rc=0 wall_s=8 | Ran 54 tests in 3.619s OK | ::notice::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec n9-glm5next-decode wrapper_rc=0 wall_s=18 | Ran 52 tests in 5.646s OK (skipped=1) | ::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass); the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec n10-f16fa wrapper_rc=0 wall_s=47 | Ran 170 tests in 22.677s OK | ::notice::n10-f16fa @ llama.cpp ec7630a: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec n10-quant wrapper_rc=0 wall_s=114 | Ran 99 tests in 66.880s OK | ::notice::n10-quant @ llama.cpp ec7630a: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a 
ec ggml-gdn wrapper_rc=0 wall_s=1 | Ran 18 tests in 0.249s OK | ::notice::ggml-gdn @ llama.cpp ec7630a: Ran 18, skipped 0 as expected; LLAMA_CPP_GGUF_PY <S>/llama-ec7630a/gguf-py is at ec7630a 
ec gguf-audit wrapper_rc=0 wall_s=2 | Ran 52 tests in 2.137s OK | ::notice::gguf-audit @ llama.cpp ec7630a: Ran 52, skipped 0 as expected; LLAMA_CPP_GGUF_PY <S>/llama-ec7630a/gguf-py is at ec7630a 
r1 g1-glm wrapper_rc=0 wall_s=2 | Ran 9 tests in 0.427s OK | ::notice::g1-glm @ llama.cpp 79e2e74: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
r1 g1-mimo wrapper_rc=0 wall_s=2 | Ran 5 tests in 0.073s OK | ::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
r1 n2-qwen4exp wrapper_rc=0 wall_s=2 | Ran 17 tests in 0.405s OK | ::notice::n2-qwen4exp @ llama.cpp 79e2e74: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
r1 n6-qwen4exp-decode wrapper_rc=0 wall_s=29 | Ran 54 tests in 3.483s OK | ::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
r1 n9-glm5next-decode wrapper_rc=0 wall_s=33 | Ran 58 tests in 5.992s OK | ::notice::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 58, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
r1 n10-f16fa wrapper_rc=0 wall_s=62 | Ran 170 tests in 22.404s OK | ::notice::n10-f16fa @ llama.cpp 79e2e74: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
r1 n10-quant wrapper_rc=0 wall_s=108 | Ran 99 tests in 65.712s OK | ::notice::n10-quant @ llama.cpp 79e2e74: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74 
ik n12-ik-glm5next wrapper_rc=0 wall_s=19 | Ran 43 tests in 12.368s OK | ::notice::n12-ik-glm5next @ ik_llama.cpp 5194a9e: Ran 43, skipped 0 as expected; the driver's checkout <S>/n12/ik-src is at 5194a9e 
```

The bites again (`<S>/n13/run_bites.sh`, unchanged and byte-identical to the committed copy; outputs in
`<S>/n13/bite/`, the first run's moved to `<S>/n13/bite-first/`), summary and tally:

```
ec g1-glm unset before_rc=0 after_rc=1 wall_s=3 | Ran 9 tests in 0.000s OK (skipped=9) | ::error::g1-glm @ llama.cpp ec7630a: skipped=9, expected skipped=0 ::error::g1-glm @ llama.cpp ec7630a: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_indexer_and_selection: 'run tools/tinygen/run_parity.sh' ::error::g1-glm @ llama.cpp ec7630a: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_mla_attention: 'run tools/tinygen/run_parity.sh' 
ec g1-glm empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.210s FAILED (errors=3) | ::error::g1-glm @ llama.cpp ec7630a: the command exited with status 1 
ec g1-mimo unset before_rc=0 after_rc=1 wall_s=1 | Ran 5 tests in 0.000s OK (skipped=5) | ::error::g1-mimo @ llama.cpp ec7630a: skipped=5, expected skipped=0 ::error::g1-mimo @ llama.cpp ec7630a: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_attention_swa_sinks_gqa: 'run tools/tinygen/run_parity.sh mimo' ::error::g1-mimo @ llama.cpp ec7630a: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_plain_swiglu: 'run tools/tinygen/run_parity.sh mimo' 
ec g1-mimo empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.061s FAILED (errors=1) | ::error::g1-mimo @ llama.cpp ec7630a: the command exited with status 1 
ec n2-qwen4exp unset before_rc=0 after_rc=1 wall_s=2 | Ran 17 tests in 0.000s OK (skipped=17) | ::error::n2-qwen4exp @ llama.cpp ec7630a: skipped=17, expected skipped=0 ::error::n2-qwen4exp @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attention_core_and_gate: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' ::error::n2-qwen4exp @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attn_projections_norms_rope: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' 
ec n2-qwen4exp empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.072s FAILED (errors=1) | ::error::n2-qwen4exp @ llama.cpp ec7630a: the command exited with status 1 
ec n6-qwen4exp-decode unset before_rc=0 after_rc=1 wall_s=21 | Ran 54 tests in 0.362s OK (skipped=46) | ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skipped=46, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_attention_over_the_kv_cache: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_carried_state: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh' 
ec n6-qwen4exp-decode empty before_rc=0 after_rc=1 wall_s=25 | Ran 8 tests in 0.470s OK (skipped=5) | ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 8 tests, expected 54 ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skipped=5, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.ym762xoKU1' 
ec n9-glm5next-decode unset before_rc=0 after_rc=1 wall_s=21 | Ran 58 tests in 0.339s OK (skipped=51) | ::error::n9-glm5next-decode @ llama.cpp ec7630a: Ran 58 tests, expected 52 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skipped=51, expected skipped=1 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_decode_parity.Decode.test_dsa_pool_cache_and_selection: 'set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_glm5next_decode.s 
ec n9-glm5next-decode empty before_rc=0 after_rc=1 wall_s=21 | Ran 7 tests in 0.274s OK (skipped=6) | ::error::n9-glm5next-decode @ llama.cpp ec7630a: Ran 7 tests, expected 52 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skipped=6, expected skipped=1 ::error::n9-glm5next-decode @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.ClbKwk6BBq' 
ec n10-f16fa unset before_rc=0 after_rc=1 wall_s=37 | Ran 170 tests in 8.700s OK (skipped=84) | ::error::n10-f16fa @ llama.cpp ec7630a: skipped=84, expected skipped=0 ::error::n10-f16fa @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_output_vs_plain_oracle: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' ::error::n10-f16fa @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_path: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' 
ec n10-f16fa empty before_rc=0 after_rc=1 wall_s=45 | Ran 86 tests in 8.494s OK (skipped=6) | ::error::n10-f16fa @ llama.cpp ec7630a: Ran 86 tests, expected 170 ::error::n10-f16fa @ llama.cpp ec7630a: skipped=6, expected skipped=0 ::error::n10-f16fa @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.CBQoWZatSu' 
ec n10-quant unset before_rc=0 after_rc=1 wall_s=86 | Ran 59 tests in 27.734s OK (skipped=8) | ::error::n10-quant @ llama.cpp ec7630a: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp ec7630a: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'set GLM5NEXT_QUANT_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_quant.sh' 
ec n10-quant empty before_rc=0 after_rc=1 wall_s=82 | Ran 59 tests in 25.943s OK (skipped=8) | ::error::n10-quant @ llama.cpp ec7630a: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp ec7630a: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp ec7630a: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'no run Q4_K_M/decode under <S>/n13/tmp/tmp.VUFtpPF1s5' 
ec n9-glm5next-decode nomtp before_rc=0 after_rc=0 wall_s=10 | Ran 52 tests in 5.488s OK (skipped=1) | ::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass); the driver's checkout <S>/llama-ec7630a is at ec7630a 
r1 g1-glm unset before_rc=0 after_rc=1 wall_s=2 | Ran 9 tests in 0.000s OK (skipped=9) | ::error::g1-glm @ llama.cpp 79e2e74: skipped=9, expected skipped=0 ::error::g1-glm @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_indexer_and_selection: 'run tools/tinygen/run_parity.sh' ::error::g1-glm @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm_tiny_parity.GlmTinyDsaParity.test_dsa_mla_attention: 'run tools/tinygen/run_parity.sh' 
r1 g1-glm empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.180s FAILED (errors=3) | ::error::g1-glm @ llama.cpp 79e2e74: the command exited with status 1 
r1 g1-mimo unset before_rc=0 after_rc=1 wall_s=1 | Ran 5 tests in 0.000s OK (skipped=5) | ::error::g1-mimo @ llama.cpp 79e2e74: skipped=5, expected skipped=0 ::error::g1-mimo @ llama.cpp 79e2e74: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_attention_swa_sinks_gqa: 'run tools/tinygen/run_parity.sh mimo' ::error::g1-mimo @ llama.cpp 79e2e74: skip not allowed here: tests.test_mimo_tiny_parity.MimoTinyParity.test_plain_swiglu: 'run tools/tinygen/run_parity.sh mimo' 
r1 g1-mimo empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.062s FAILED (errors=1) | ::error::g1-mimo @ llama.cpp 79e2e74: the command exited with status 1 
r1 n2-qwen4exp unset before_rc=0 after_rc=1 wall_s=1 | Ran 17 tests in 0.000s OK (skipped=17) | ::error::n2-qwen4exp @ llama.cpp 79e2e74: skipped=17, expected skipped=0 ::error::n2-qwen4exp @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attention_core_and_gate: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' ::error::n2-qwen4exp @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_tiny_parity.Qwen4expTinyParity.test_attn_projections_norms_rope: 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh' 
r1 n2-qwen4exp empty before_rc=1 after_rc=1 wall_s=2 | Ran 0 tests in 0.070s FAILED (errors=1) | ::error::n2-qwen4exp @ llama.cpp 79e2e74: the command exited with status 1 
r1 n6-qwen4exp-decode unset before_rc=0 after_rc=1 wall_s=22 | Ran 54 tests in 0.404s OK (skipped=46) | ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skipped=46, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_attention_over_the_kv_cache: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.test_carried_state: 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh' 
r1 n6-qwen4exp-decode empty before_rc=0 after_rc=1 wall_s=26 | Ran 8 tests in 0.386s OK (skipped=5) | ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 8 tests, expected 54 ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skipped=5, expected skipped=0 ::error::n6-qwen4exp-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_qwen4exp_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.WAr96txjVk' 
r1 n9-glm5next-decode unset before_rc=0 after_rc=1 wall_s=25 | Ran 58 tests in 0.295s OK (skipped=51) | ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=51, expected skipped=0 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_decode_parity.Decode.test_dsa_pool_cache_and_selection: 'set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_glm5next_decode.s ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_decode_parity.Decode.test_dsa_projections: 'set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_glm5next_decode.sh' 
r1 n9-glm5next-decode empty before_rc=0 after_rc=1 wall_s=25 | Ran 7 tests in 0.347s OK (skipped=6) | ::error::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 7 tests, expected 58 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=6, expected skipped=0 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_decode_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.rD5eLqU6rf' 
r1 n10-f16fa unset before_rc=0 after_rc=1 wall_s=38 | Ran 170 tests in 8.247s OK (skipped=84) | ::error::n10-f16fa @ llama.cpp 79e2e74: skipped=84, expected skipped=0 ::error::n10-f16fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_output_vs_plain_oracle: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' ::error::n10-f16fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.test_attention_path: 'set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh' 
r1 n10-f16fa empty before_rc=0 after_rc=1 wall_s=45 | Ran 86 tests in 8.500s OK (skipped=6) | ::error::n10-f16fa @ llama.cpp 79e2e74: Ran 86 tests, expected 170 ::error::n10-f16fa @ llama.cpp 79e2e74: skipped=6, expected skipped=0 ::error::n10-f16fa @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_f16fa_parity.Decode.setUpClass: 'no run decode under <S>/n13/tmp/tmp.NyYbhN7csW' 
r1 n10-quant unset before_rc=0 after_rc=1 wall_s=81 | Ran 59 tests in 27.785s OK (skipped=8) | ::error::n10-quant @ llama.cpp 79e2e74: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp 79e2e74: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'set GLM5NEXT_QUANT_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_quant.sh' 
r1 n10-quant empty before_rc=0 after_rc=1 wall_s=82 | Ran 59 tests in 26.058s OK (skipped=8) | ::error::n10-quant @ llama.cpp 79e2e74: Ran 59 tests, expected 99 ::error::n10-quant @ llama.cpp 79e2e74: skipped=8, expected skipped=0 ::error::n10-quant @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_quant_parity.Glm_Q4_K_M_decode.setUpClass: 'no run Q4_K_M/decode under <S>/n13/tmp/tmp.PEPmEtTIFA' 
r1 n9-glm5next-decode nomtp before_rc=0 after_rc=1 wall_s=11 | Ran 52 tests in 5.527s OK (skipped=1) | ::error::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 52 tests, expected 58 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skipped=1, expected skipped=0 ::error::n9-glm5next-decode @ llama.cpp 79e2e74: skip not allowed here: tests.test_glm5next_mtp_parity.Mtp.setUpClass: 'no MTP run under <S>/n13/tmp/tmp.DYr8QvgHjQ/n9 (llama.cpp 79e2 
ec ggml-gdn unset(GGML_GDN_HARNESS) before_rc=0 after_rc=1 wall_s=0 | Ran 18 tests in 0.208s OK (skipped=3) | ::error::ggml-gdn @ llama.cpp ec7630a: skipped=3, expected skipped=0 ::error::ggml-gdn @ llama.cpp ec7630a: skip not allowed here: tests.test_kda_vs_ggml.OracleVsGgml.test_gdn_scalar_gate: 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)' ::error::ggml-gdn @ llama.cpp ec7630a: skip not allowed here: tests.test_kda_vs_ggml.OracleVsGgml.test_glm_head_geometry: 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)' 
ec ggml-gdn empty(GGML_GDN_HARNESS=<empty dir>/gdn_harness) before_rc=1 after_rc=1 wall_s=1 | Ran 18 tests in 0.222s FAILED (errors=3) | ::error::ggml-gdn @ llama.cpp ec7630a: the command exited with status 1 
ec gguf-audit unset(LLAMA_CPP_GGUF_PY) before_rc=0 after_rc=1 wall_s=2 | Ran 52 tests in 1.923s OK (skipped=1) | ::error::gguf-audit @ llama.cpp ec7630a: skipped=1, expected skipped=0 ::error::gguf-audit @ llama.cpp ec7630a: skip not allowed here: tests.test_gguf_quant_audit.Audit.test_type_table_matches_pinned_gguf_py: 'set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py' 
ec gguf-audit empty(LLAMA_CPP_GGUF_PY=<empty dir>) before_rc=0 after_rc=2 wall_s=0 | | ::error::check_unittest: cannot read the commit of <S>/n13/tmp/tmp.BKNd2SZ0Q8: Command '['git', '-C', '/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchp 
ik n12-ik-glm5next unset before_rc=0 after_rc=1 wall_s=8 | Ran 43 tests in 0.000s OK (skipped=43) | ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skipped=43, expected skipped=0 ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skip not allowed here: tests.test_ik_glm5next_parity.IkAlias.test_glm5_next_name_gives_identical_logits: 'set IK_GLM5NEXT_DUMP, IK_GLM5NEXT_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_ik_glm5next.sh' ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skip not allowed here: tests.test_ik_glm5next_parity.IkDecode.test_cache_stores: 'set IK_GLM5NEXT_DUMP, IK_GLM5NEXT_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_ik_glm5next.sh' 
ik n12-ik-glm5next empty before_rc=0 after_rc=1 wall_s=8 | Ran 1 test in 0.001s OK (skipped=8) | ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: Ran 1 tests, expected 43 ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skipped=8, expected skipped=0 ::error::n12-ik-glm5next @ ik_llama.cpp 5194a9e: skip not allowed here: tests.test_ik_glm5next_parity.IkAlias.test_glm5_next_name_gives_identical_logits: 'no run alias under <S>/n13/ 
```

```
36 rows
broken - before_rc=0 after_rc=1 26
broken - before_rc=0 after_rc=2 1
broken - before_rc=1 after_rc=1 7
nomtp ec before_rc=0 after_rc=0 1
nomtp r1 before_rc=0 after_rc=1 1
```

Row by row against the first run (pin, step, variant, `before_rc`, `after_rc`, and the unittest summary without its
time). The one different row is the new binding at work: `gguf-audit` with `LLAMA_CPP_GGUF_PY` pointed at an empty
directory (not a git checkout) now stops before running, rc 2, "cannot read the commit of ..."; its `before_rc=0` in
the repeat is an artifact of the script, which derives `before_rc` from the checker's "the command exited with status"
annotation and finds none because the command never ran (in the first run the same command returned 1). All other 35
rows are identical:

```
rows: first run 36, repeat 36, same keys: True
DIFFERENT ec gguf-audit empty(LLAMA_CPP_GGUF_PY=<empty dir>): first ('1', '1', 'Ran 52 tests FAILED (errors=1)') repeat ('0', '2', '')
identical (before_rc, after_rc, unittest summary): 35 of 36
```

Full suite before the last commit:

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests   (in /home/user/wt-n13)
rc=0 seconds=125
----------------------------------------------------------------------
Ran 843 tests in 124.821s

OK (skipped=354)
```
