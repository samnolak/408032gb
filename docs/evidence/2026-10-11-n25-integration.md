# N25: one integration branch with every audited fix (N20, N21, N22, N24, N23), 2026-10-11 (sandbox CPU, no GPU)

Branch `claude/n25-integration`, cut from N19 at `e71b241`, worktree `/home/user/wt-n25`. Role: engine-integrator.
`<S>` below stands for the session scratch directory
`<S>`; outputs are pasted with only that
substitution. Not pushed (task instruction).

Pins: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md) at `<S>/llama-ec7630a` and
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine, tools/engines/pins.json) at `<S>/llama.cpp`. Both checkouts
are unchanged before and after every run (lines `checkout ... status lines 0` below); every patched tree is a copy.

## 1. Merges (CONFIRMED, output in section 5)

Every merge is `git merge --no-ff` of the origin head, in the order the lead gave; N23 twice, because its audit-fix
head 7541c4a arrived after its first merge.

| # | merge commit | merged head | conflicts | resolution |
|---|---|---|---|---|
| 1 | `63162c3` | `claude/n20-recurrent-rollback-guard` @ `b7d6af1` | none | - |
| 2 | `fb3ce4e` | `claude/n21-mimo-mtp-heads-fix` @ `93a48d6` | `.github/workflows/ci.yml` (job cpu-parity: N20 step against N21 step); `.github/workflows/parity-r1-pin.yml` (header comment N20 against N21; job parity: N20 step against N21 step plus the new job `n21-patched`) | both sides kept, N20 first, with the Edit tool; every step stays wrapped by `tools/ci/check_unittest.py`. `tools/ci/parity_expectations.json` auto-merged (n20-qwen4exp-rs-guard, n21-mimo2-mtp-fix, n21-mimo2-mtp-fix-patched; valid JSON, counts unchanged). `patches/llama.cpp/README.md` (new on N21) did not conflict, since N20 has no README, but it was wrong after the merge: it says "the other jobs run the unpatched pins", while N20's step runs a patched copy in cpu-parity and parity. Rewritten to list both layouts (N20 `patches/llama.cpp/<pin>/*.patch` + `apply.sh`; N21 top-level `0001-mtp-heads-slot-replay.patch`, applied by job `n21-patched` and not by apply.sh), N20's patch row added, N21's row unchanged |
| 3 | `87e10e3` | `claude/n22-ci-false-green` @ `56202c8` | none (ci.yml auto-merged: N22 changes the `unit tests` step and job `cuda-compile`) | - |
| 4 | `21eac9b` | `claude/n24-owner-decisions` @ `76e9d56` | `docs/evidence/mutation-scripts/README.md` (script table: N20 row against three N24 rows) | both kept, N20 first. `patches/llama.cpp/README.md` also lists N24's experiment patch `n24-swiglu-clamp-after-silu.patch`, which only `tools/quality/swiglu_clamp_experiment.sh` applies (`patch -p1`, line 48), not CI and not apply.sh |
| 5 | `1f35c7a` | `claude/n23-glm-template-tokenizer` @ `4447b2c` | none | - (brings `third_party/hf/*` from the branch) |
| 6 | `447d7e2` | `claude/n23-glm-template-tokenizer` @ `7541c4a` (audit fixes: `facc74d` hf_lfs_fetch hex check, `7541c4a` evidence) | none | - |

`tools/dump/dump_tensors.cpp`: only N21 changed it; the merged file equals N21's (`git diff --stat 93a48d6 HEAD --
tools/dump/` is empty). The two llama.cpp patches touch disjoint files (N20: `src/llama-memory-recurrent.{cpp,h}`; N21:
eight files in `common/` and `src/`, none of them N20's). CONFIRMED by both patches applying to one copy, below.

## 2. Step table (CONFIRMED: checker lines in section 5)

Every row ran through `tools/ci/check_unittest.py --step <id> --checkout <tree>` against
`tools/ci/parity_expectations.json` as merged. "wrapper_rc=0" = Ran N and the skips exactly as expected.

Unpatched pins (`<S>/n25/runs.sh`; fresh `tools/g0` builds `<S>/n25/build-ec`, `<S>/n25/build-r1` of this worktree,
`GGML_NATIVE=OFF`, `G0_QUANTIZE=ON`; build ec 206 s, r1 166 s):

| step | driver | ec7630a | 79e2e74 |
|---|---|---|---|
| g1-glm (`PYTHONPATH`=worktree) | run_parity.sh | Ran 9, 0 skips | Ran 9, 0 skips |
| g1-mimo | run_parity_mimo.sh | Ran 5, 0 skips | Ran 5, 0 skips |
| n2-qwen4exp | run_parity_qwen4exp.sh | Ran 17, 0 skips | Ran 17, 0 skips |
| n6-qwen4exp-decode | run_parity_qwen4exp_decode.sh | Ran 54, 0 skips | Ran 54, 0 skips |
| n9-glm5next-decode | run_parity_glm5next_decode.sh | Ran 52, 1 expected skip (Mtp.setUpClass) | Ran 58, 0 skips |
| n10-f16fa | run_parity_f16fa.sh | Ran 170, 0 skips | Ran 170, 0 skips |
| n10-quant | run_parity_quant.sh | Ran 99, 0 skips | Ran 99, 0 skips |
| n16-mimo2-decode | run_parity_mimo2_decode.sh | Ran 92, 0 skips | Ran 92, 0 skips |
| n17-mimo2-mtp-heads-fa | run_parity_mimo2_n17.sh | Ran 70, 0 skips | Ran 70, 0 skips |
| n19-qwen4exp-multiseq | run_parity_qwen4exp_multiseq.sh | Ran 72, 0 skips | Ran 72, 0 skips |
| n20-qwen4exp-rs-guard, `QWEN4EXP_RS_GUARD_MODE=upstream` | run_parity_qwen4exp_rs_guard.sh | Ran 110, 0 skips | Ran 110, 0 skips |
| n21-mimo2-mtp-fix (unpatched, `patched: false`) | run_parity_mimo2_n21.sh | Ran 32, 6 expected skips | Ran 32, 6 expected skips |

Patched copies (`<S>/n25/patched.sh`):

| step | tree | result |
|---|---|---|
| n20-qwen4exp-rs-guard, patched mode | apply.sh copy of ec7630a (`2 files changed, 56 insertions(+), 2 deletions(-)`) | Ran 110, 0 skips |
| n20-qwen4exp-rs-guard, patched mode | apply.sh copy of 79e2e74 (`2 files changed, 52 insertions(+), 2 deletions(-)`) | Ran 110, 0 skips |
| n21-mimo2-mtp-fix-patched (`patched: true`) | copy of 79e2e74 + N21 patch (`8 files changed, 142 insertions(+), 1 deletion(-)`) | Ran 57, 0 skips |

The runs of `runs.sh` started on `1f35c7a`; merge 6 (`447d7e2`) was committed while they ran. It changes only
`scripts/hf_lfs_fetch.py`, `tests/test_hf_lfs.py` and the N23 evidence (`git diff --stat 1f35c7a 447d7e2`, section 5),
none of which a dump step reads. `patched.sh` and `unit.sh` ran on `447d7e2`.

## 3. Combined patch: N20 + N21 on one copy of 79e2e74 (CONFIRMED, section 5)

`patches/llama.cpp/apply.sh <S>/llama.cpp <S>/n25/llama-r1-both` (N20), then `git apply --check` and `git apply` of
`patches/llama.cpp/0001-mtp-heads-slot-replay.patch` on that copy: both apply, `10 files changed, 194 insertions(+),
3 deletions(-)` (= 52 + 142 insertions, 2 + 1 deletions), `git apply --reverse --check` of each patch passes on the
combined tree, HEAD stays 79e2e74. One build (`<S>/n25/build-r1-both`, 153 s), then:

| step | result on the combined copy |
|---|---|
| n20-qwen4exp-rs-guard (guard expectations; same 13 plans, same seq_rm results and refusal counts as the N20-only copy) | Ran 110, 0 skips |
| n21-mimo2-mtp-fix-patched (`patched: true`) | Ran 57, 0 skips |
| n6-qwen4exp-decode | Ran 54, 0 skips |
| n16-mimo2-decode | Ran 92, 0 skips |
| n17-mimo2-mtp-heads-fa | Ran 70, 0 skips |
| n19-qwen4exp-multiseq with `QWEN4EXP_RS_GUARD=1` (the N19 tests' guard expectations, N20's switch) | Ran 72, 0 skips |

So the two fixes compose: the guard's behaviour is unchanged by the slot-replay patch, the slot-replay fix is unchanged
by the guard, and the unpatched N6 / N16 / N17 expectations still hold with both applied.

How the N20 step ran on the combined copy: the committed driver cannot do it. In patched mode it calls apply.sh, which
refuses a directory that is not exactly the N20 patches (CONFIRMED, `apply.sh rc=1` below); in upstream mode it refuses a
tree that carries the N20 patch. A scratch variant `<S>/n25/drv/run_parity_qwen4exp_rs_guard_combined.sh` was used. Its
whole diff against `tools/tinygen/run_parity_qwen4exp_rs_guard.sh` is in section 5: ROOT is the worktree, and apply.sh is
replaced by a check that the given copy carries both patches and is at the pin. Plans, build, dumps and tests are the
committed driver's.

## 4. Observations, not fixed here (left to the lead)

* No CI job applies N20 and N21 together. The combined check above is local only. A job like `n21-patched` that also runs
  apply.sh, plus a driver option to accept extra patches, would close that gap (PROVISIONAL suggestion).
* Both fixes are called `0001-*` but live in different layouts (per pin directory vs top level). apply.sh never sees the
  N21 patch. Documented in `patches/llama.cpp/README.md`, not renamed, because CI, the expectations and the N21 evidence
  cite the path.
* `docs/evidence/mutation-scripts/README.md` does not list the five N21 scripts or the two N23 scripts. That was already
  so on those branches (`NOT listed` lines in section 5).
* The README's N24 mutation row says 18/18 (first round), while the N24 evidence ends at 20/20
  (docs/evidence/2026-10-11-n24-owner-decisions.md:522 "18/18 caught", :667 "20/20 caught"). As on the N24 branch.

NOT RUN: GitHub CI for this branch (not pushed); GPU, real weights; steps ggml-gdn, gguf-audit, n12-ik-glm5next and
the two G0 smoke scripts (not in the task's list); N21 with ec7630a (its patch does not apply there, per N21);
byte identity of the dump tool against a pre-N21 binary (not re-run here: the merged dump tool is N21's file, whose own
byte-identity check is in the N21 evidence).

## 5. Literal outputs

### 5.1 Merge commits and the heads used

```
$ git log --oneline --graph -n 8
*   447d7e2 [engine-integrator] merge claude/n23-glm-template-tokenizer @7541c4a into claude/n25-integration: N23 audit fixes (hf_lfs_fetch hex check, evidence)
|\  
| * 7541c4a [parity-qa] evidence: N23 audit fixes - 41-62 tokens per Qwen break, enable_thinking=false tokenizer-level only, real-model bounds PROVISIONAL, line numbers 177/184 and the round-2 diff, test counts, sha256 annotation and the third token-level run pasted from gh api; common.h escape line stays 570 (checked at the pin)
| * facc74d [parity-qa] vendor: hf_lfs_fetch refuses a non-hex sha256 or revision in a PIN (N23 audit, optional item 8)
* | 1f35c7a [engine-integrator] merge claude/n23-glm-template-tokenizer into claude/n25-integration: GLM template, tokenizer pins, token-level workflow
|\| 
| * 4447b2c [parity-qa] evidence: N23 round 2b - HF tokenizer token level and HF = llama.cpp ids CONFIRMED on the runner (run 38169032753), summary step false green fixed, summary/artifact NOT RUN until the next run
| * 008c8c7 [parity-qa] token level: absolute work directory and --out in token_prefix.py; the summary step fails the job (no false green)
| * 96dd906 [parity-qa] evidence: N23 round 2 - tokenizer pins and licenses (GLM MIT, Qwen Community License 1.0), token-level run failed on a relative path (HF side NOT RUN, fixed), MIT notice in the template
| * 374ec07 [parity-qa] templates: upstream MIT notice (Copyright (c) 2026 Z.AI Co., Ltd) verbatim in the header of the corrected GLM template
$ git log --first-parent --format='%h parents=%p %s' e71b241..HEAD
447d7e2 parents=1f35c7a 7541c4a [engine-integrator] merge claude/n23-glm-template-tokenizer @7541c4a into claude/n25-integration: N23 audit fixes (hf_lfs_fetch hex check, evidence)
1f35c7a parents=21eac9b 4447b2c [engine-integrator] merge claude/n23-glm-template-tokenizer into claude/n25-integration: GLM template, tokenizer pins, token-level workflow
21eac9b parents=87e10e3 76e9d56 [engine-integrator] merge claude/n24-owner-decisions into claude/n25-integration: verify pack, V6, hostcheck, SwiGLU, R0 docs
87e10e3 parents=fb3ce4e 56202c8 [engine-integrator] merge claude/n22-ci-false-green into claude/n25-integration: cuda-compile honesty, gap 14 tables
fb3ce4e parents=63162c3 93a48d6 [engine-integrator] merge claude/n21-mimo-mtp-heads-fix into claude/n25-integration: MiMo MTP slot-replay patch, N21 step and job n21-patched, dump tool
63162c3 parents=e71b241 b7d6af1 [engine-integrator] merge claude/n20-recurrent-rollback-guard into claude/n25-integration: qwen4exp recurrent rollback guard patch, N20 CI step
$ git ls-remote origin claude/n20-recurrent-rollback-guard claude/n21-mimo-mtp-heads-fix claude/n22-ci-false-green claude/n23-glm-template-tokenizer claude/n24-owner-decisions
b7d6af11f4fbec136fd995399bd0cf3ab980823b	refs/heads/claude/n20-recurrent-rollback-guard
93a48d6bc0bfa243d36e9a4442ff213ef71bc14d	refs/heads/claude/n21-mimo-mtp-heads-fix
56202c8fe0949657d6583af4a4e5dece6141abd2	refs/heads/claude/n22-ci-false-green
7541c4a67b71213d0506215c785d0fbf4388a9f0	refs/heads/claude/n23-glm-template-tokenizer
76e9d565d1f6ac615cbdda9f5d7cbcc023976344	refs/heads/claude/n24-owner-decisions
$ git diff --stat 93a48d6 HEAD -- tools/dump/; echo "rc=$? (no output: the merged dump tool is N21's)"
rc=0 (no output: the merged dump tool is N21's)
$ git diff --stat 1f35c7a 447d7e2
 .../2026-10-11-n23-glm-template-tokenizer.md       | 182 ++++++++++++++++++---
 scripts/hf_lfs_fetch.py                            |   5 +-
 tests/test_hf_lfs.py                               |   9 +
 3 files changed, 173 insertions(+), 23 deletions(-)
```

### 5.2 Conflict resolutions (combined diffs of the merge commits)

```
$ git show --cc --format=%h fb3ce4e -- .github/workflows
fb3ce4e

diff --cc .github/workflows/ci.yml
index c2610ad,da35f9b..f5c407f
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@@ -52,8 -52,8 +52,10 @@@ jobs
          run: python3 tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n17.sh ../llama.cpp ../build-tools
        - name: N19 qwen4exp several sequences in one context, seq_cp, seq_rm and rollback edges against the real qwen4exp graph (CPU)
          run: python3 tools/ci/check_unittest.py --step n19-qwen4exp-multiseq --checkout ../llama.cpp -- tools/tinygen/run_parity_qwen4exp_multiseq.sh ../llama.cpp ../build-tools
 +      - name: N20 recurrent rollback guard (patches/llama.cpp, applied to a copy) - the N19 plans and four guard plans on the patched qwen4exp graph (CPU)
 +        run: python3 tools/ci/check_unittest.py --step n20-qwen4exp-rs-guard --checkout ../llama.cpp -- tools/tinygen/run_parity_qwen4exp_rs_guard.sh ../llama.cpp ../build-tools-rs-guard
+       - name: N21 mimo2 MTP heads - upstream placement (ring wrap, several ubatches) and llama.cpp's own common/speculative.cpp against the replay (CPU; the runs of the patch skip at this unpatched pin)
+         run: python3 tools/ci/check_unittest.py --step n21-mimo2-mtp-fix --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n21.sh ../llama.cpp ../build-tools
        - name: G0 per-token routing tracer + cache simulator on the tiny glm5-next model (CPU smoke)
          run: tools/g0/run_trace_smoke.sh ../llama.cpp ../build-tools
        - name: G0 trace + probe on a tiny qwen3moe (last layer keeps only output rows; predict_eval self-checks) (CPU smoke)
diff --cc .github/workflows/parity-r1-pin.yml
index 101d691,c010aa6..0794fae
--- a/.github/workflows/parity-r1-pin.yml
+++ b/.github/workflows/parity-r1-pin.yml
@@@ -21,9 -21,11 +21,14 @@@
  # rollback edges; the pins differ on purpose in three places (ec7630a: shared cells re-pool every k-pool, a rollback of a
  # shared recurrent cell is accepted, padded entries of indexer_pool_k_new share one cell; 79e2e74: one pooled row per rep
  # cell, that rollback refused, every padded entry its own non-rep cell), tools/tinygen/run_parity_qwen4exp_multiseq.sh.
 +# N20: the recurrent rollback guard, patches/llama.cpp/79e2e74 applied to a copy of the checkout (patches/llama.cpp/apply.sh,
 +# the checkout itself is not touched), the N19 plans and four guard plans on the patched build,
 +# tools/tinygen/run_parity_qwen4exp_rs_guard.sh (its own build directory: a second CPU build of llama.cpp).
+ # N21: mimo2's MTP heads, the upstream placement (head 0 reads foreign K/V after a ring wrap; several ubatches) and
+ # llama.cpp's own common/speculative.cpp against the replay, tools/tinygen/run_parity_mimo2_n21.sh, in job parity at the
+ # pin and in job n21-patched on a copy of it with patches/llama.cpp/0001-mtp-heads-slot-replay.patch applied (the fix;
+ # a second llama.cpp build: about 15 min on 4 cores in the N21 sandbox, docs/evidence/2026-10-11-n21-mimo-mtp-heads-
+ # fix.md; on a GitHub runner UNKNOWN).
  # CPU only, tiny random-weight models.
  name: parity-r1-pin
  on:
@@@ -78,8 -82,29 +85,31 @@@ jobs
          run: python3 tools/ci/check_unittest.py --step n17-mimo2-mtp-heads-fa --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n17.sh ../llama.cpp ../build-tools
        - name: N19 qwen4exp several sequences, seq_cp, seq_rm and rollback edges at the R1 pin (CPU)
          run: python3 tools/ci/check_unittest.py --step n19-qwen4exp-multiseq --checkout ../llama.cpp -- tools/tinygen/run_parity_qwen4exp_multiseq.sh ../llama.cpp ../build-tools
 +      - name: N20 recurrent rollback guard (patches/llama.cpp, applied to a copy) on the patched qwen4exp graph at the R1 pin (CPU)
 +        run: python3 tools/ci/check_unittest.py --step n20-qwen4exp-rs-guard --checkout ../llama.cpp -- tools/tinygen/run_parity_qwen4exp_rs_guard.sh ../llama.cpp ../build-tools-rs-guard
+       - name: N21 mimo2 MTP heads - upstream placement and llama.cpp's own common/speculative.cpp against the replay at the R1 pin (CPU; the runs of the patch skip here, job n21-patched runs them)
+         run: python3 tools/ci/check_unittest.py --step n21-mimo2-mtp-fix --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n21.sh ../llama.cpp ../build-tools
+   n21-patched:
+     runs-on: ubuntu-24.04
+     timeout-minutes: 60
+     steps:
+       - uses: actions/checkout@v4
+       - name: deps
+         run: pip install numpy pyyaml
+       - name: llama.cpp at the R1 engine pin (tools/engines/pins.json), fetched into ../llama.cpp-pin
+         run: |
+           SHA=$(python3 -c "import json; print(json.load(open('tools/engines/pins.json'))['engines']['llama.cpp']['sha'])")
+           git init -q ../llama.cpp-pin
+           git -C ../llama.cpp-pin fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$SHA"
+           git -C ../llama.cpp-pin checkout -q FETCH_HEAD
+           git -C ../llama.cpp-pin rev-parse HEAD
+       - name: ../llama.cpp = a copy of ../llama.cpp-pin with patches/llama.cpp/0001-mtp-heads-slot-replay.patch (the fetched checkout stays untouched; HEAD stays the pin)
+         run: |
+           cp -a ../llama.cpp-pin ../llama.cpp
+           git -C ../llama.cpp apply "$PWD/patches/llama.cpp/0001-mtp-heads-slot-replay.patch"
+           git -C ../llama.cpp diff --stat
+       - name: N21 mimo2 MTP heads with the patch - the fixed order against the oracles, the real code against the replay (CPU)
+         run: python3 tools/ci/check_unittest.py --step n21-mimo2-mtp-fix-patched --checkout ../llama.cpp -- tools/tinygen/run_parity_mimo2_n21.sh ../llama.cpp ../build-tools
    ik-glm5next:
      runs-on: ubuntu-24.04
      timeout-minutes: 30
$ git diff 93a48d6 fb3ce4e -- patches/llama.cpp/README.md   # the N21 README against the merged one
diff --git a/patches/llama.cpp/README.md b/patches/llama.cpp/README.md
index 13b7d60..56138d9 100644
--- a/patches/llama.cpp/README.md
+++ b/patches/llama.cpp/README.md
@@ -1,10 +1,22 @@
 # Patches on ggml-org/llama.cpp
 
-Upstream stays untouched (AGENTS.md "Upstream pins"); a patch is applied to a copy of the checkout:
-`git -C <copy of llama.cpp @base> apply <patch>`. CI applies 0001 in job `n21-patched` of
-`.github/workflows/parity-r1-pin.yml` (to a copy of the R1 pin, step `n21-mimo2-mtp-fix-patched`); the other jobs run
-the unpatched pins.
+Upstream stays untouched (AGENTS.md "Upstream pins"); a patch is applied to a copy of the checkout, never in place.
+Two layouts exist (merged in claude/n25-integration, docs/evidence/2026-10-11-n25-integration.md):
+
+* `patches/llama.cpp/<short sha>/*.patch` (N20): one directory per pin; `patches/llama.cpp/apply.sh <checkout> <new dir>`
+  applies every patch of the checkout's pin to a `git clone --shared` copy (HEAD stays the pin). CI applies the N20
+  guard in step `n20-qwen4exp-rs-guard` of job `cpu-parity` (`.github/workflows/ci.yml`, ec7630a) and of job `parity`
+  (`.github/workflows/parity-r1-pin.yml`, R1 pin), through `tools/tinygen/run_parity_qwen4exp_rs_guard.sh`, in its own
+  build directory `../build-tools-rs-guard`.
+* `patches/llama.cpp/0001-mtp-heads-slot-replay.patch` (N21): `git -C <copy of llama.cpp @base> apply <patch>`. CI
+  applies it in job `n21-patched` of `.github/workflows/parity-r1-pin.yml` (to a copy of the R1 pin, step
+  `n21-mimo2-mtp-fix-patched`). apply.sh does not pick it up (it reads only the pin directories).
+
+Every other step runs the unpatched pins. The two patches touch disjoint files (N20: `src/llama-memory-recurrent.{cpp,h}`;
+N21: `common/speculative.cpp`, `src/llama-context.cpp`, `src/llama-ext.h`, `src/llama-kv-cache.{cpp,h}`,
+`src/llama-kv-cache-iswa.{cpp,h}`, `src/llama-memory.h`); no CI job applies both together.
 
 | patch | base | what | verified |
 |---|---|---|---|
+| 79e2e74/0001-recurrent-rollback-guard, ec7630a/0001-recurrent-rollback-guard | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin), `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) | `src/llama-memory-recurrent.cpp` / `.h`: `mem_cell::rs_n` counts the snapshot planes a cell's rows hold; `seq_rm` refuses (returns false, logs "refusing a rollback") a rollback `r >= rs_n`; `seq_cp` gives the destination the source's pending `rs_idx`; `s_copy` resets the pending rollback of every sequence of the cell (N20 audit FAIL 1). The ec7630a patch also adds 79e2e74's shared-cell refusal. Details: docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md section 3. | CPU only, tiny qwen4exp model: `tools/tinygen/run_parity_qwen4exp_rs_guard.sh <checkout> <build>` (patched copy by apply.sh), docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md. GPU and real weights: NOT RUN |
 | 0001-mtp-heads-slot-replay | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin) | `common/speculative.cpp` draft-mtp `process()`: each ubatch of a target batch is placed once for all NextN heads (heads 1..H-1 re-use head 0's KV cells through the new staging API `llama_memory_set_slot_replay`, `src/llama-ext.h`; `llama_kv_cache::prepare()` honours it). Fixes the N17 finding (head 0 of a multi-head MTP context reads K/V of other positions after a batch that wraps the SWA ring) and the multi-ubatch variant found in N21. `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` restores the upstream order for A/B runs (only a value starting with `1` counts; `0`, `true`, `yes` or empty keep the fix). | CPU only, tiny mimo2 model with 3 MTP heads: `tools/tinygen/run_parity_mimo2_n21.sh <patched copy> <build>`, docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md. `git apply --check`: applies at 79e2e74; at ec7630a `src/llama-ext.h` does not apply (other context), the other 7 files do. GPU, llama-server binary and real MiMo weights: NOT RUN |
$ git show --cc --format=%h 21eac9b -- docs/evidence/mutation-scripts/README.md patches/llama.cpp/README.md
21eac9b

diff --cc docs/evidence/mutation-scripts/README.md
index ac64e19,6a93e7c..442fbe2
--- a/docs/evidence/mutation-scripts/README.md
+++ b/docs/evidence/mutation-scripts/README.md
@@@ -25,7 -25,9 +25,10 @@@
  | `2026-10-11-n16-mimo2-mutations.py` | оракул mimo2 `ref/mimo2.py` (окно SWA, размер SWA-кэша и повторное использование ячеек, очистка `apply_ubatch`, длина и объём отката, движение head, sinks, база RoPE, GQA, value scale, MTP-блок: W1–W5, S1–S2, R1–R3, V1, G1, M1–M2), пары MTP в тесте (M3), MXFP4 в `ref/quant.py` (масштаб E8M0, порядок полубайтов, квантование активаций: X1–X3) и деквантование MXFP4 в копии gguf-py пина (P1) против `tests/test_mimo2_decode_parity.py`, `tests/test_mimo2_mtp_parity.py`, `tests/test_mimo2_chunked_ref.py`, `tests/test_mimo2_quant_parity.py`, `tests/test_mxfp4_ref.py`; нужны артефакты `run_parity_mimo2_decode.sh` (`MIMO2_DECODE_WORK`) | см. `docs/evidence/2026-10-11-n16-mimo-decode-parity.md` | `python3 docs/evidence/mutation-scripts/2026-10-11-n16-mimo2-mutations.py <репозиторий> <каталог артефактов> <llama.cpp> <временный каталог>` | см. там же, раздел «Mutation check» |
  | `2026-10-11-n17-mutations.py` | оракул нескольких MTP-голов и точности `ref/mimo2.py` (смещение головы в process и в черновике: H1–H2; откуда голова берёт h в process и в черновике: P1–P3; `seq_rm` перед каждой головой и каждым шагом черновика: Q1–Q2; учёт, для какой позиции посчитаны K/V ячейки в каждом слое: K1; округление f16 при записи в кэш: R2; масштаб kq в f32: E2; путаница размеров голов K и V: V1–V2), эмуляцию flash attention CPU `ref/fattn.py` (синк в one_chunk пропущен или применён до цикла: F1–F2; f16-аккумулятор: R1; Q в f16: R3; аргумент `expf` в f32: E1; полосы AVX2 в скалярном произведении f16: E3) и воспроизведение отказа в тесте (pending-строка `accept()`: T1) против `tests/test_mimo2_mtp_heads_parity.py`, `tests/test_mimo2_f16fa_parity.py`, `tests/test_mimo2_heads_ref.py`, `tests/test_fattn_sinks_ref.py`; нужны артефакты `run_parity_mimo2_n17.sh` (`MIMO2_N17_WORK`); копирует `ref/` и `tests/*.py` рабочего дерева (не `git ls-files`) | `d5b3537` | `python3 docs/evidence/mutation-scripts/2026-10-11-n17-mutations.py <репозиторий> <каталог артефактов> <llama.cpp> <временный каталог>` | `ec7630a` 19/19, `79e2e74` 19/19 (одни и те же падающие тесты на обоих пинах), `docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md` (раздел 7.7) |
  | `2026-10-11-n19-qwen4exp-multiseq-mutations.py` | оракул нескольких последовательностей в `ref/qwen4exp.py` (строка состояния чужой последовательности, `seq_cp` без разделения рекуррентной ячейки, откаты со сдвигом на один, плоскости снимков отката, правила отказа обоих пинов, копия «лишних» строк, модель потоков KV, разделение и дедупликация k-pool, `split_equal`, ячейка дополнения: O1–O18), ожидания планов в `tools/tinygen/qwen4exp_multiseq_plans.py` (T1–T2) и сторона llama.cpp, смоделированная правкой копии дампов (D1–D4), против `tests/test_qwen4exp_multiseq_parity.py`, `tests/test_qwen4exp_multiseq_ref.py`; нужны артефакты `run_parity_qwen4exp_multiseq.sh` (`QWEN4EXP_MULTISEQ_WORK`) | см. `docs/evidence/2026-10-11-n19-qwen4exp-multiseq.md` | `python3 docs/evidence/mutation-scripts/2026-10-11-n19-qwen4exp-multiseq-mutations.py <репозиторий> <каталог артефактов> <llama.cpp> <временный каталог>` | `ec7630a` 22/24, `79e2e74` 23/24 (выживают только мутанты правил другого пина), там же, раздел «Mutation check» |
 +| `2026-10-11-n20-rs-guard-mutations.py` | патч `patches/llama.cpp/<пин>/0001-recurrent-rollback-guard.patch` на настоящей сборке: копия llama.cpp с патчем, по одному отключённому куску патча (C1–C7) или весь патч снят (C0), пересборка `dump-tensors` и все 13 прогонов (9 N19 + 4 N20) заново; модель защиты в оракуле `ref/qwen4exp.py` (P1–P4, P6), переключатель `QWEN4EXP_RS_GUARD` в тесте (P5) и ожидания планов (T1–T4) против `tests/test_qwen4exp_multiseq_parity.py`, `tests/test_qwen4exp_rs_guard_parity.py`, `tests/test_qwen4exp_multiseq_ref.py`, `tests/test_qwen4exp_rs_guard_ref.py` с `QWEN4EXP_RS_GUARD=1`; в конце копия возвращается к патчу и пересобирается | см. `docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md` | `python3 docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py <репозиторий> <llama.cpp на пине> <копия с патчем> <её каталог сборки> <временный каталог> [C\|P\|CP]` | там же, раздел 5.4 |
+ | `2026-10-11-n24-mutations.py` | правило синтетического прогона и допуск почти-ничьих V6 в `tools/verify/pack.py` (P1–P9), поля разрывов и шума в `tools/quality/logprob_compare.py` (L1–L2), проверку GGUF на хосте `tools/hostcheck/glm_gguf_check.py` (H1–H5) и `tools/quality/swiglu_gate_stats.py` (G1–G2) против `tests/test_verify_pack_n24.py`, `tests/test_hostcheck_glm_gguf.py`, `tests/test_swiglu_clamp_tools.py`; копирует `git ls-files -co` без `third_party/` | `bcc4ea6` (первый круг `a88abe9`: 16/18) | `python3 docs/evidence/mutation-scripts/2026-10-11-n24-mutations.py <репозиторий> <временный каталог>` | 18/18, `docs/evidence/2026-10-11-n24-owner-decisions.md` (раздел 6) |
+ | `2026-10-11-n24-swiglu-patch-check.sh` | не мутации: проверка, что `patches/llama.cpp/n24-swiglu-clamp-after-silu.patch` меняет только порядок clamp/SiLU (G1-поток на крошечной glm5-next, штатная и пропатченная сборки `dump-tensors` ec7630a) | `ed7bc30` | `bash docs/evidence/mutation-scripts/2026-10-11-n24-swiglu-patch-check.sh <репозиторий> <штатная сборка> <пропатченная сборка> <llama.cpp ec7630a> <каталог>` | там же, раздел 3.4 |
+ | `2026-10-11-n24-v6-summary.py` | не мутации: сводка файлов `logprob_compare.py` для допуска V6 (разрывы, распределение `max_abs_dlogprob`) | `ed7bc30` | `python3 docs/evidence/mutation-scripts/2026-10-11-n24-v6-summary.py <lc-*.json>` | там же, раздел 2 |
  
  Пробы N9 (не мутационные: печатают числа, вставленные в `docs/evidence/2026-10-10-glm5next-decode-parity.md`) —
  байт-в-байт копии из песочницы сессии; внутри абсолютные пути песочницы и рабочего дерева (`S=`, `REPO=`, `sys.path`),
diff --cc patches/llama.cpp/README.md
index 56138d9,0000000..88ee0b2
mode 100644,000000..100644
--- a/patches/llama.cpp/README.md
+++ b/patches/llama.cpp/README.md
@@@ -1,22 -1,0 +1,26 @@@
 +# Patches on ggml-org/llama.cpp
 +
 +Upstream stays untouched (AGENTS.md "Upstream pins"); a patch is applied to a copy of the checkout, never in place.
 +Two layouts exist (merged in claude/n25-integration, docs/evidence/2026-10-11-n25-integration.md):
 +
 +* `patches/llama.cpp/<short sha>/*.patch` (N20): one directory per pin; `patches/llama.cpp/apply.sh <checkout> <new dir>`
 +  applies every patch of the checkout's pin to a `git clone --shared` copy (HEAD stays the pin). CI applies the N20
 +  guard in step `n20-qwen4exp-rs-guard` of job `cpu-parity` (`.github/workflows/ci.yml`, ec7630a) and of job `parity`
 +  (`.github/workflows/parity-r1-pin.yml`, R1 pin), through `tools/tinygen/run_parity_qwen4exp_rs_guard.sh`, in its own
 +  build directory `../build-tools-rs-guard`.
 +* `patches/llama.cpp/0001-mtp-heads-slot-replay.patch` (N21): `git -C <copy of llama.cpp @base> apply <patch>`. CI
 +  applies it in job `n21-patched` of `.github/workflows/parity-r1-pin.yml` (to a copy of the R1 pin, step
 +  `n21-mimo2-mtp-fix-patched`). apply.sh does not pick it up (it reads only the pin directories).
++* `patches/llama.cpp/n24-swiglu-clamp-after-silu.patch` (N24): an experiment patch, NOT for production (its own header);
++  applied (`patch -p1`, to a copy) only by `tools/quality/swiglu_clamp_experiment.sh`, never by CI or apply.sh;
++  `docs/evidence/mutation-scripts/2026-10-11-n24-swiglu-patch-check.sh` compares a stock and a patched build it is
++  given; docs/evidence/2026-10-11-n24-owner-decisions.md section 3.4.
 +
- Every other step runs the unpatched pins. The two patches touch disjoint files (N20: `src/llama-memory-recurrent.{cpp,h}`;
++Every other step runs the unpatched pins. The N20 and N21 patches touch disjoint files (N20: `src/llama-memory-recurrent.{cpp,h}`;
 +N21: `common/speculative.cpp`, `src/llama-context.cpp`, `src/llama-ext.h`, `src/llama-kv-cache.{cpp,h}`,
 +`src/llama-kv-cache-iswa.{cpp,h}`, `src/llama-memory.h`); no CI job applies both together.
 +
 +| patch | base | what | verified |
 +|---|---|---|---|
 +| 79e2e74/0001-recurrent-rollback-guard, ec7630a/0001-recurrent-rollback-guard | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin), `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) | `src/llama-memory-recurrent.cpp` / `.h`: `mem_cell::rs_n` counts the snapshot planes a cell's rows hold; `seq_rm` refuses (returns false, logs "refusing a rollback") a rollback `r >= rs_n`; `seq_cp` gives the destination the source's pending `rs_idx`; `s_copy` resets the pending rollback of every sequence of the cell (N20 audit FAIL 1). The ec7630a patch also adds 79e2e74's shared-cell refusal. Details: docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md section 3. | CPU only, tiny qwen4exp model: `tools/tinygen/run_parity_qwen4exp_rs_guard.sh <checkout> <build>` (patched copy by apply.sh), docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md. GPU and real weights: NOT RUN |
 +| 0001-mtp-heads-slot-replay | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin) | `common/speculative.cpp` draft-mtp `process()`: each ubatch of a target batch is placed once for all NextN heads (heads 1..H-1 re-use head 0's KV cells through the new staging API `llama_memory_set_slot_replay`, `src/llama-ext.h`; `llama_kv_cache::prepare()` honours it). Fixes the N17 finding (head 0 of a multi-head MTP context reads K/V of other positions after a batch that wraps the SWA ring) and the multi-ubatch variant found in N21. `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` restores the upstream order for A/B runs (only a value starting with `1` counts; `0`, `true`, `yes` or empty keep the fix). | CPU only, tiny mimo2 model with 3 MTP heads: `tools/tinygen/run_parity_mimo2_n21.sh <patched copy> <build>`, docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md. `git apply --check`: applies at 79e2e74; at ec7630a `src/llama-ext.h` does not apply (other context), the other 7 files do. GPU, llama-server binary and real MiMo weights: NOT RUN |
$ sed -n 48p tools/quality/swiglu_clamp_experiment.sh
    if [ "$v" = post ]; then (cd "$WORK/src-$v" && run patch -p1 --forward < "$PATCH"); fi
$ grep -n "18/18 caught\|20/20 caught" docs/evidence/2026-10-11-n24-owner-decisions.md
522:18/18 caught
667:20/20 caught
$ for f in $(ls docs/evidence/mutation-scripts/ | grep -E "n2[0-4]"); do grep -q "$f" docs/evidence/mutation-scripts/README.md && echo "listed $f" || echo "NOT listed $f"; done
listed 2026-10-11-n20-aggregate.py
listed 2026-10-11-n20-extra-cell-probe.py
listed 2026-10-11-n20-final-runs.sh
listed 2026-10-11-n20-oracle-probe.py
listed 2026-10-11-n20-rs-guard-mutations.py
listed 2026-10-11-n20-source-lines.sh
NOT listed 2026-10-11-n21-byteid.sh
NOT listed 2026-10-11-n21-final-runs.sh
NOT listed 2026-10-11-n21-mutations.py
NOT listed 2026-10-11-n21-probes.sh
NOT listed 2026-10-11-n21-source-lines.sh
NOT listed 2026-10-11-n23-mutations.sh
NOT listed 2026-10-11-n23-qwen-newline-live.py
listed 2026-10-11-n24-mutations.py
listed 2026-10-11-n24-swiglu-patch-check.sh
listed 2026-10-11-n24-v6-summary.py
```

### 5.3 Unpatched pins: builds and every dump step through the checker

```
$ cat <S>/n25/runs.sh
#!/usr/bin/env bash
# N25 integration (worktree /home/user/wt-n25, branch claude/n25-integration): fresh tools/g0 dump-tools builds at both
# pins, then every dump-based CI step through tools/ci/check_unittest.py at both pins on the unpatched checkouts
# (N20 in QWEN4EXP_RS_GUARD_MODE=upstream, N21 unpatched). Work directories under <S>/n25/tmp, deleted after each step.
set -uo pipefail
S=<S>
WT=/home/user/wt-n25
cd $S/n25/cwd
echo "wt-n25 HEAD $(git -C $WT rev-parse --short HEAD); changed or untracked: $(git -C $WT status --short | wc -l)"
pin() { if [ "$1" = ec ]; then echo $S/llama-ec7630a; else echo $S/llama.cpp; fi; }
for p in ec r1; do L=$(pin $p); echo "checkout $p: $(git -C $L rev-parse HEAD), status lines $(git -C $L status --short | wc -l)"; done
for p in ec r1; do
  L=$(pin $p); B=$S/n25/build-$p
  rm -rf $B; t0=$(date +%s)
  cmake -S $WT/tools/g0 -B $B -DLLAMA_CPP_DIR=$L -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON > $S/n25/cmake-$p.log 2>&1 \
    && cmake --build $B -j4 --target dump-tensors > $S/n25/build-$p.log 2>&1
  echo "build $p rc=$? $(( $(date +%s) - t0 )) s"
done
for p in ec r1; do
  L=$(pin $p); B=$S/n25/build-$p
  for st in g1-glm:run_parity.sh g1-mimo:run_parity_mimo.sh n2-qwen4exp:run_parity_qwen4exp.sh \
            n6-qwen4exp-decode:run_parity_qwen4exp_decode.sh n9-glm5next-decode:run_parity_glm5next_decode.sh \
            n10-f16fa:run_parity_f16fa.sh n10-quant:run_parity_quant.sh n16-mimo2-decode:run_parity_mimo2_decode.sh \
            n17-mimo2-mtp-heads-fa:run_parity_mimo2_n17.sh n19-qwen4exp-multiseq:run_parity_qwen4exp_multiseq.sh \
            n20-qwen4exp-rs-guard:run_parity_qwen4exp_rs_guard.sh n21-mimo2-mtp-fix:run_parity_mimo2_n21.sh; do
    step=${st%%:*}; drv=${st##*:}
    rm -rf $S/n25/tmp; mkdir -p $S/n25/tmp; W=$S/n25/tmp/w
    t0=$(date +%s)
    # PYTHONPATH: run_parity.sh (g1-glm) imports tests.* from the current directory, which is a scratch directory here.
    # QWEN4EXP_RS_GUARD_MODE=upstream: the N20 driver on the unpatched checkout (the pin's own behaviour expected).
    TMPDIR=$S/n25/tmp PYTHONPATH=$WT QWEN4EXP_RS_GUARD_MODE=upstream QWEN4EXP_RS_GUARD_WORK=$W MIMO2_N21_WORK=$W \
      MIMO2_N17_WORK=$W QWEN4EXP_MULTISEQ_WORK=$W GLM5NEXT_DECODE_WORK=$W F16FA_WORK=$W QUANT_WORK=$W MIMO2_DECODE_WORK=$W \
      python3 $WT/tools/ci/check_unittest.py --step $step --checkout $L -- $WT/tools/tinygen/$drv $L $B \
      > $S/n25/step-$step-$p.log 2>&1
    echo "$step $p wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
    tail -n 1 $S/n25/step-$step-$p.log
  done
done
rm -rf $S/n25/tmp
for p in ec r1; do L=$(pin $p); echo "checkout $p after: $(git -C $L rev-parse HEAD), status lines $(git -C $L status --short | wc -l)"; done
echo RUNS-DONE
$ cd <S>/n25/cwd && bash <S>/n25/runs.sh > <S>/n25/runs.out 2>&1; cat <S>/n25/runs.out
wt-n25 HEAD 1f35c7a; changed or untracked: 0
checkout ec: ec7630a640789c393694fb194f1bbbf0369fc62d, status lines 0
checkout r1: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, status lines 0
build ec rc=0 206 s
build r1 rc=0 166 s
g1-glm ec wrapper_rc=0 wall_s=1
::notice::g1-glm @ llama.cpp ec7630a: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
g1-mimo ec wrapper_rc=0 wall_s=2
::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n2-qwen4exp ec wrapper_rc=0 wall_s=1
::notice::n2-qwen4exp @ llama.cpp ec7630a: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n6-qwen4exp-decode ec wrapper_rc=0 wall_s=7
::notice::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n9-glm5next-decode ec wrapper_rc=0 wall_s=15
::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass); the driver's checkout <S>/llama-ec7630a is at ec7630a
n10-f16fa ec wrapper_rc=0 wall_s=40
::notice::n10-f16fa @ llama.cpp ec7630a: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n10-quant ec wrapper_rc=0 wall_s=98
::notice::n10-quant @ llama.cpp ec7630a: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n16-mimo2-decode ec wrapper_rc=0 wall_s=48
::notice::n16-mimo2-decode @ llama.cpp ec7630a: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n17-mimo2-mtp-heads-fa ec wrapper_rc=0 wall_s=70
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n19-qwen4exp-multiseq ec wrapper_rc=0 wall_s=26
::notice::n19-qwen4exp-multiseq @ llama.cpp ec7630a: Ran 72, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n20-qwen4exp-rs-guard ec wrapper_rc=0 wall_s=27
::notice::n20-qwen4exp-rs-guard @ llama.cpp ec7630a: Ran 110, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n21-mimo2-mtp-fix ec wrapper_rc=0 wall_s=18
::notice::n21-mimo2-mtp-fix @ llama.cpp ec7630a: Ran 32, skipped 6 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.ReplayRefusal.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_independent_of_the_mtp_order); the driver's checkout <S>/llama-ec7630a is at ec7630a
g1-glm r1 wrapper_rc=0 wall_s=2
::notice::g1-glm @ llama.cpp 79e2e74: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
g1-mimo r1 wrapper_rc=0 wall_s=1
::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n2-qwen4exp r1 wrapper_rc=0 wall_s=2
::notice::n2-qwen4exp @ llama.cpp 79e2e74: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n6-qwen4exp-decode r1 wrapper_rc=0 wall_s=19
::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n9-glm5next-decode r1 wrapper_rc=0 wall_s=24
::notice::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 58, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n10-f16fa r1 wrapper_rc=0 wall_s=49
::notice::n10-f16fa @ llama.cpp 79e2e74: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n10-quant r1 wrapper_rc=0 wall_s=87
::notice::n10-quant @ llama.cpp 79e2e74: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n16-mimo2-decode r1 wrapper_rc=0 wall_s=47
::notice::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n17-mimo2-mtp-heads-fa r1 wrapper_rc=0 wall_s=70
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n19-qwen4exp-multiseq r1 wrapper_rc=0 wall_s=17
::notice::n19-qwen4exp-multiseq @ llama.cpp 79e2e74: Ran 72, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n20-qwen4exp-rs-guard r1 wrapper_rc=0 wall_s=28
::notice::n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 110, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n21-mimo2-mtp-fix r1 wrapper_rc=0 wall_s=20
::notice::n21-mimo2-mtp-fix @ llama.cpp 79e2e74: Ran 32, skipped 6 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.ReplayRefusal.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_independent_of_the_mtp_order); the driver's checkout <S>/llama.cpp is at 79e2e74
checkout ec after: ec7630a640789c393694fb194f1bbbf0369fc62d, status lines 0
checkout r1 after: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, status lines 0
RUNS-DONE
$ grep -h "^mode " <S>/n25/step-n20-*.log; grep -h "^patched:" <S>/n25/step-n21-*.log
mode upstream: llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d, sources <S>/llama-ec7630a ()
mode upstream: llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, sources <S>/llama.cpp ()
patched: false
patched: false
```

### 5.4 Patched copies and the combined patch

```
$ diff tools/tinygen/run_parity_qwen4exp_rs_guard.sh <S>/n25/drv/run_parity_qwen4exp_rs_guard_combined.sh
21c21
< ROOT=$(cd "$(dirname "$0")/../.." && pwd); WORK=${QWEN4EXP_RS_GUARD_WORK:-$(mktemp -d)}
---
> ROOT=/home/user/wt-n25; WORK=${QWEN4EXP_RS_GUARD_WORK:-$(mktemp -d)}
30c30
<     bash "$ROOT/patches/llama.cpp/apply.sh" "$SRC" "$LLAMA"
---
>     LLAMA=${QWEN4EXP_RS_GUARD_SRC:?combined copy}; for P in "$PATCH" "$ROOT/patches/llama.cpp/0001-mtp-heads-slot-replay.patch"; do git -C "$LLAMA" apply --reverse --check "$P" || { echo "combined copy $LLAMA does not carry $P" >&2; exit 1; }; done; [ "$(git -C "$LLAMA" rev-parse HEAD)" = "$SHA" ] || exit 1
$ cat <S>/n25/patched.sh
#!/usr/bin/env bash
# N25 integration, patched llama.cpp copies (the pinned checkouts are never edited):
#  (1) N20 guard: tools/tinygen/run_parity_qwen4exp_rs_guard.sh in patched mode (patches/llama.cpp/apply.sh copy) at both pins;
#  (2) N21: a copy of 79e2e74 with patches/llama.cpp/0001-mtp-heads-slot-replay.patch, step n21-mimo2-mtp-fix-patched;
#  (3) combined: ONE copy of 79e2e74 with apply.sh (N20) and then the N21 patch; on it n20 (driver variant
#      <S>/n25/drv/run_parity_qwen4exp_rs_guard_combined.sh: apply.sh replaced by a check that the copy carries both
#      patches), n21 patched, n6, n16, n17, n19 (QWEN4EXP_RS_GUARD=1: the guard's behaviour expected), all through the checker.
set -uo pipefail
S=<S>
WT=/home/user/wt-n25
P21=$WT/patches/llama.cpp/0001-mtp-heads-slot-replay.patch
cd $S/n25/cwd
echo "wt-n25 HEAD $(git -C $WT rev-parse --short HEAD); changed or untracked: $(git -C $WT status --short | wc -l)"
pin() { if [ "$1" = ec ]; then echo $S/llama-ec7630a; else echo $S/llama.cpp; fi; }
step() {  # step <id> <checkout> <driver> <build> [env...]
  local id=$1 L=$2 drv=$3 B=$4 tag=$5; shift 5
  rm -rf $S/n25/tmp; mkdir -p $S/n25/tmp; local W=$S/n25/tmp/w t0=$(date +%s)
  env TMPDIR=$S/n25/tmp PYTHONPATH=$WT QWEN4EXP_RS_GUARD_WORK=$W MIMO2_N21_WORK=$W MIMO2_N17_WORK=$W \
    QWEN4EXP_MULTISEQ_WORK=$W MIMO2_DECODE_WORK=$W "$@" \
    python3 $WT/tools/ci/check_unittest.py --step $id --checkout $L -- $drv $L $B > $S/n25/pstep-$tag.log 2>&1
  echo "$tag wrapper_rc=$? wall_s=$(( $(date +%s) - t0 ))"
  tail -n 1 $S/n25/pstep-$tag.log
  grep -E '^mode |^patched: |guard refusals' $S/n25/pstep-$tag.log | head -20
}
echo "== (1) N20 guard, patched mode (apply.sh copy), both pins"
for p in ec r1; do
  L=$(pin $p)
  rm -rf $S/n25/llama-$p-guard $S/n25/build-$p-guard
  step n20-qwen4exp-rs-guard $L $WT/tools/tinygen/run_parity_qwen4exp_rs_guard.sh $S/n25/build-$p-guard n20-guard-$p \
    QWEN4EXP_RS_GUARD_SRC=$S/n25/llama-$p-guard
  echo "copy $p: HEAD $(git -C $S/n25/llama-$p-guard rev-parse --short HEAD); $(git -C $S/n25/llama-$p-guard diff --stat | tail -n 1)"
done
rm -rf $S/n25/llama-ec-guard $S/n25/build-ec-guard $S/n25/llama-r1-guard $S/n25/build-r1-guard
echo "== (2) N21 patched, copy of 79e2e74"
D=$S/n25/llama-r1-n21; rm -rf $D $S/n25/build-r1-n21
git clone -q --shared --no-checkout $S/llama.cpp $D && git -C $D -c advice.detachedHead=false checkout -q --detach 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 \
  && git -C $D apply $P21 && echo "copy: HEAD $(git -C $D rev-parse --short HEAD); $(git -C $D diff --stat | tail -n 1)"
step n21-mimo2-mtp-fix-patched $D $WT/tools/tinygen/run_parity_mimo2_n21.sh $S/n25/build-r1-n21 n21-patched-r1
rm -rf $D $S/n25/build-r1-n21
echo "== (3) combined: N20 (apply.sh) + N21 on one copy of 79e2e74"
D=$S/n25/llama-r1-both; B=$S/n25/build-r1-both; rm -rf $D $B
bash $WT/patches/llama.cpp/apply.sh $S/llama.cpp $D | tail -n 2
git -C $D apply --check $P21 && echo "N21 patch: git apply --check on the N20 copy OK"
git -C $D apply $P21 && echo "N21 patch applied"
git -C $D diff --stat | cat
for P in $WT/patches/llama.cpp/79e2e74/0001-recurrent-rollback-guard.patch $P21; do
  git -C $D apply --reverse --check $P && echo "carries $(basename $(dirname $P))/$(basename $P)"; done
echo "HEAD $(git -C $D rev-parse HEAD)"
echo "apply.sh on the combined copy (expected refusal: it is not exactly the N20 patches):"
bash $WT/patches/llama.cpp/apply.sh $S/llama.cpp $D; echo "apply.sh rc=$?"
t0=$(date +%s)
cmake -S $WT/tools/g0 -B $B -DLLAMA_CPP_DIR=$D -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON > $S/n25/cmake-both.log 2>&1 \
  && cmake --build $B -j4 --target dump-tensors > $S/n25/build-both.log 2>&1
echo "build combined rc=$? $(( $(date +%s) - t0 )) s"
step n20-qwen4exp-rs-guard $S/llama.cpp $S/n25/drv/run_parity_qwen4exp_rs_guard_combined.sh $B both-n20-guard QWEN4EXP_RS_GUARD_SRC=$D
step n21-mimo2-mtp-fix-patched $D $WT/tools/tinygen/run_parity_mimo2_n21.sh $B both-n21-patched
step n6-qwen4exp-decode $D $WT/tools/tinygen/run_parity_qwen4exp_decode.sh $B both-n6
step n16-mimo2-decode $D $WT/tools/tinygen/run_parity_mimo2_decode.sh $B both-n16
step n17-mimo2-mtp-heads-fa $D $WT/tools/tinygen/run_parity_mimo2_n17.sh $B both-n17
step n19-qwen4exp-multiseq $D $WT/tools/tinygen/run_parity_qwen4exp_multiseq.sh $B both-n19-guard QWEN4EXP_RS_GUARD=1
rm -rf $S/n25/tmp
for p in ec r1; do L=$(pin $p); echo "checkout $p after: $(git -C $L rev-parse HEAD), status lines $(git -C $L status --short | wc -l)"; done
echo PATCHED-DONE
$ cd <S>/n25/cwd && bash <S>/n25/patched.sh > <S>/n25/patched.out 2>&1; cat <S>/n25/patched.out
wt-n25 HEAD 447d7e2; changed or untracked: 0
== (1) N20 guard, patched mode (apply.sh copy), both pins
n20-guard-ec wrapper_rc=0 wall_s=190
::notice::n20-qwen4exp-rs-guard @ llama.cpp ec7630a: Ran 110, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
mode patched: llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d, sources <S>/n25/llama-ec-guard (2 files changed, 56 insertions(+), 2 deletions(-))
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
multi_str: 21 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
stale: 11 graphs, 13 steps, seq_rm ok: [False, False], guard refusals logged: 2
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [False], guard refusals logged: 0
last_call: 7 graphs, 11 steps, seq_rm ok: [False, True, True, True], guard refusals logged: 1
extra_cell: 5 graphs, 7 steps, seq_rm ok: [True, False], guard refusals logged: 1
cp_pending_order: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_extra: 3 graphs, 6 steps, seq_rm ok: [True, True], guard refusals logged: 0
copy ec: HEAD ec7630a;  2 files changed, 56 insertions(+), 2 deletions(-)
n20-guard-r1 wrapper_rc=0 wall_s=184
::notice::n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 110, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
mode patched: llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, sources <S>/n25/llama-r1-guard (2 files changed, 52 insertions(+), 2 deletions(-))
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
multi_str: 21 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
stale: 11 graphs, 13 steps, seq_rm ok: [False, False], guard refusals logged: 2
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [False], guard refusals logged: 0
last_call: 7 graphs, 11 steps, seq_rm ok: [False, True, True, True], guard refusals logged: 1
extra_cell: 5 graphs, 7 steps, seq_rm ok: [True, False], guard refusals logged: 1
cp_pending_order: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_extra: 3 graphs, 6 steps, seq_rm ok: [True, True], guard refusals logged: 0
copy r1: HEAD 79e2e74;  2 files changed, 52 insertions(+), 2 deletions(-)
== (2) N21 patched, copy of 79e2e74
copy: HEAD 79e2e74;  8 files changed, 142 insertions(+), 1 deletion(-)
n21-patched-r1 wrapper_rc=0 wall_s=182
::notice::n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 57, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-n21 is at 79e2e74
patched: true
== (3) combined: N20 (apply.sh) + N21 on one copy of 79e2e74
 src/llama-memory-recurrent.h   |  4 ++++
 2 files changed, 52 insertions(+), 2 deletions(-)
N21 patch: git apply --check on the N20 copy OK
N21 patch applied
 common/speculative.cpp         | 55 +++++++++++++++++++++++++++++++++++++++++-
 src/llama-context.cpp          |  4 +++
 src/llama-ext.h                |  7 ++++++
 src/llama-kv-cache-iswa.cpp    |  7 ++++++
 src/llama-kv-cache-iswa.h      |  2 ++
 src/llama-kv-cache.cpp         | 54 +++++++++++++++++++++++++++++++++++++++++
 src/llama-kv-cache.h           |  9 +++++++
 src/llama-memory-recurrent.cpp | 50 ++++++++++++++++++++++++++++++++++++--
 src/llama-memory-recurrent.h   |  4 +++
 src/llama-memory.h             |  5 ++++
 10 files changed, 194 insertions(+), 3 deletions(-)
carries 79e2e74/0001-recurrent-rollback-guard.patch
carries llama.cpp/0001-mtp-heads-slot-replay.patch
HEAD 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
apply.sh on the combined copy (expected refusal: it is not exactly the N20 patches):
apply.sh: <S>/n25/llama-r1-both exists and is not 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 with exactly the patches of /home/user/wt-n25/patches/llama.cpp/79e2e74
apply.sh rc=1
build combined rc=0 153 s
both-n20-guard wrapper_rc=0 wall_s=18
::notice::n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 110, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
mode patched: llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, sources <S>/n25/llama-r1-both (10 files changed, 194 insertions(+), 3 deletions(-))
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
multi_str: 21 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
stale: 11 graphs, 13 steps, seq_rm ok: [False, False], guard refusals logged: 2
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [False], guard refusals logged: 0
last_call: 7 graphs, 11 steps, seq_rm ok: [False, True, True, True], guard refusals logged: 1
extra_cell: 5 graphs, 7 steps, seq_rm ok: [True, False], guard refusals logged: 1
cp_pending_order: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_extra: 3 graphs, 6 steps, seq_rm ok: [True, True], guard refusals logged: 0
both-n21-patched wrapper_rc=0 wall_s=43
::notice::n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 57, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
patched: true
both-n6 wrapper_rc=0 wall_s=22
::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
both-n16 wrapper_rc=0 wall_s=52
::notice::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 92, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
both-n17 wrapper_rc=0 wall_s=67
::notice::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 70, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
both-n19-guard wrapper_rc=0 wall_s=24
::notice::n19-qwen4exp-multiseq @ llama.cpp 79e2e74: Ran 72, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
checkout ec after: ec7630a640789c393694fb194f1bbbf0369fc62d, status lines 0
checkout r1 after: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, status lines 0
PATCHED-DONE
```

### 5.5 Unit discovery, checker and hook tests, YAML parse of every workflow

```
$ cat <S>/n25/unit.sh
#!/usr/bin/env bash
# N25 integration: unit discovery, checker and hook tests, YAML parse of every workflow, the steps of the parity jobs
# and the expectations of the new steps, on wt-n25 HEAD.
set -uo pipefail
S=<S>
WT=/home/user/wt-n25
cd $WT
echo "HEAD $(git rev-parse --short HEAD) (parents $(git rev-parse --short HEAD^1) $(git rev-parse --short HEAD^2))"
echo '$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests'
TMPDIR=$S/n25/cwd LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > $S/n25/discover.log 2>&1
echo "rc=$?"; tail -n 3 $S/n25/discover.log
echo '$ python3 -m unittest tests.test_check_unittest tests.test_hooks'
python3 -m unittest tests.test_check_unittest tests.test_hooks > $S/n25/check.log 2>&1
echo "rc=$?"; tail -n 3 $S/n25/check.log
echo '$ python3 - (yaml.safe_load of every workflow; every step of the parity jobs and its checker step id)'
python3 - <<'PY'
import glob, json, yaml
for wf in sorted(glob.glob(".github/workflows/*.yml")):
    d = yaml.safe_load(open(wf))
    print(f"{wf}: parsed, jobs {list(d['jobs'])}")
PAR = {".github/workflows/ci.yml": ["cpu-parity"], ".github/workflows/parity-r1-pin.yml": ["parity", "ik-glm5next", "n21-patched"]}
for wf, jobs in PAR.items():
    d = yaml.safe_load(open(wf))
    for j in jobs:
        for s in d["jobs"][j]["steps"]:
            r = s.get("run", "")
            sid = r.split("--step ")[1].split()[0] if "check_unittest.py" in r else "(not wrapped)"
            print(f"  {wf.split('/')[-1]} [{j}] {sid:28s} {str(s.get('name', s.get('uses')))[:70]}")
e = json.load(open("tools/ci/parity_expectations.json"))
st = e["steps"] if "steps" in e else e
for k in ("n19-qwen4exp-multiseq", "n20-qwen4exp-rs-guard", "n21-mimo2-mtp-fix", "n21-mimo2-mtp-fix-patched"):
    print(k, {c[:7]: (v["ran"], len(v["skipped"])) for c, v in st[k]["expect"].items()})
PY
echo "status: $(git status --short | wc -l) changed or untracked files"
$ bash <S>/n25/unit.sh > <S>/n25/unit.out 2>&1; cat <S>/n25/unit.out
HEAD 447d7e2 (parents 1f35c7a 7541c4a)
$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests
rc=0
Ran 1238 tests in 170.856s

OK (skipped=608)
$ python3 -m unittest tests.test_check_unittest tests.test_hooks
rc=0
Ran 54 tests in 24.641s

OK
$ python3 - (yaml.safe_load of every workflow; every step of the parity jobs and its checker step id)
.github/workflows/ci.yml: parsed, jobs ['cpu-parity', 'ref-torch-cpu', 'cuda-compile', 'strata-glm-sm89', 'lt-strata-sm89']
.github/workflows/engines-sm89.yml: parsed, jobs ['build']
.github/workflows/parity-r1-pin.yml: parsed, jobs ['parity', 'n21-patched', 'ik-glm5next']
.github/workflows/token-level.yml: parsed, jobs ['tokens']
.github/workflows/traces-real.yml: parsed, jobs ['traces']
.github/workflows/vendor-deepseek.yml: parsed, jobs ['vendor']
.github/workflows/vendor-hf-branch.yml: parsed, jobs ['vendor', 'token-level']
.github/workflows/vendor-hf.yml: parsed, jobs ['vendor']
  ci.yml [cpu-parity] (not wrapped)                actions/checkout@v4
  ci.yml [cpu-parity] (not wrapped)                llama.cpp at the pinned commit
  ci.yml [cpu-parity] (not wrapped)                deps
  ci.yml [cpu-parity] (not wrapped)                unit tests
  ci.yml [cpu-parity] ggml-gdn                     ggml delta-net parity (CPU)
  ci.yml [cpu-parity] g1-glm                       G1 tiny-model parity against the real glm5-next graph (CPU)
  ci.yml [cpu-parity] g1-mimo                      MiMo G1 tiny-model parity against the real mimo2 graph (CPU)
  ci.yml [cpu-parity] n2-qwen4exp                  N2 qwen4exp (candidate 1) tiny-model parity against the real qwen4exp 
  ci.yml [cpu-parity] n6-qwen4exp-decode           N6 qwen4exp several ubatches, token-by-token decode, rollback and MTP 
  ci.yml [cpu-parity] n9-glm5next-decode           N9 glm5-next several ubatches, token-by-token decode and rollback agai
  ci.yml [cpu-parity] n10-f16fa                    N10 R1 default precision (-ctk f16 -ctv f16 -fa on) - qwen4exp and glm
  ci.yml [cpu-parity] n10-quant                    N10 quantized weights (Q8_0, Q4_K_M by the pinned llama-quantize) - ev
  ci.yml [cpu-parity] n16-mimo2-decode             N16 mimo2 (MiMo) ubatches, decode past the SWA cache, rollbacks, MTP a
  ci.yml [cpu-parity] n17-mimo2-mtp-heads-fa       N17 mimo2 three MTP heads (nextn_layer_offset 0-2, drafts, rejections)
  ci.yml [cpu-parity] n19-qwen4exp-multiseq        N19 qwen4exp several sequences in one context, seq_cp, seq_rm and roll
  ci.yml [cpu-parity] n20-qwen4exp-rs-guard        N20 recurrent rollback guard (patches/llama.cpp, applied to a copy) - 
  ci.yml [cpu-parity] n21-mimo2-mtp-fix            N21 mimo2 MTP heads - upstream placement (ring wrap, several ubatches)
  ci.yml [cpu-parity] (not wrapped)                G0 per-token routing tracer + cache simulator on the tiny glm5-next mo
  ci.yml [cpu-parity] (not wrapped)                G0 trace + probe on a tiny qwen3moe (last layer keeps only output rows
  ci.yml [cpu-parity] gguf-audit                   roofline, cache-simulator and GGUF-audit unit tests (type table checke
  parity-r1-pin.yml [parity] (not wrapped)                actions/checkout@v4
  parity-r1-pin.yml [parity] (not wrapped)                deps
  parity-r1-pin.yml [parity] (not wrapped)                llama.cpp at the R1 engine pin (tools/engines/pins.json)
  parity-r1-pin.yml [parity] g1-glm                       G1 tiny-model parity against the glm5-next graph at the R1 pin (CPU)
  parity-r1-pin.yml [parity] g1-mimo                      MiMo tiny-model parity against the mimo2 graph at the R1 pin (CPU)
  parity-r1-pin.yml [parity] n2-qwen4exp                  N2 qwen4exp prefill parity at the R1 pin (CPU)
  parity-r1-pin.yml [parity] n6-qwen4exp-decode           N6 qwen4exp ubatches, decode, rollback and MTP parity at the R1 pin (C
  parity-r1-pin.yml [parity] n9-glm5next-decode           N9 glm5-next ubatches, decode, rollback and MTP parity at the R1 pin (
  parity-r1-pin.yml [parity] n10-f16fa                    N10 R1 default precision (-ctk f16 -ctv f16 -fa on) of qwen4exp and gl
  parity-r1-pin.yml [parity] n10-quant                    N10 quantized weights (Q8_0, Q4_K_M by the R1 pin's llama-quantize), q
  parity-r1-pin.yml [parity] n16-mimo2-decode             N16 mimo2 ubatches, decode past the SWA cache, rollbacks, MTP and MXFP
  parity-r1-pin.yml [parity] n17-mimo2-mtp-heads-fa       N17 mimo2 three MTP heads and the R1 default precision (-ctk f16 -ctv 
  parity-r1-pin.yml [parity] n19-qwen4exp-multiseq        N19 qwen4exp several sequences, seq_cp, seq_rm and rollback edges at t
  parity-r1-pin.yml [parity] n20-qwen4exp-rs-guard        N20 recurrent rollback guard (patches/llama.cpp, applied to a copy) on
  parity-r1-pin.yml [parity] n21-mimo2-mtp-fix            N21 mimo2 MTP heads - upstream placement and llama.cpp's own common/sp
  parity-r1-pin.yml [ik-glm5next] (not wrapped)                actions/checkout@v4
  parity-r1-pin.yml [ik-glm5next] (not wrapped)                deps
  parity-r1-pin.yml [ik-glm5next] (not wrapped)                llama.cpp at the R1 engine pin (its gguf-py writes and reads the tiny 
  parity-r1-pin.yml [ik-glm5next] (not wrapped)                ik_llama.cpp at its R1 engine pin (tools/engines/pins.json)
  parity-r1-pin.yml [ik-glm5next] n12-ik-glm5next              N12 ik_llama.cpp glm5next parity on CPU (prefill, ubatches, decode, de
  parity-r1-pin.yml [n21-patched] (not wrapped)                actions/checkout@v4
  parity-r1-pin.yml [n21-patched] (not wrapped)                deps
  parity-r1-pin.yml [n21-patched] (not wrapped)                llama.cpp at the R1 engine pin (tools/engines/pins.json), fetched into
  parity-r1-pin.yml [n21-patched] (not wrapped)                ../llama.cpp = a copy of ../llama.cpp-pin with patches/llama.cpp/0001-
  parity-r1-pin.yml [n21-patched] n21-mimo2-mtp-fix-patched    N21 mimo2 MTP heads with the patch - the fixed order against the oracl
n19-qwen4exp-multiseq {'ec7630a': (72, 0), '79e2e74': (72, 0)}
n20-qwen4exp-rs-guard {'ec7630a': (110, 0), '79e2e74': (110, 0)}
n21-mimo2-mtp-fix {'ec7630a': (32, 6), '79e2e74': (32, 6)}
n21-mimo2-mtp-fix-patched {'79e2e74': (57, 0)}
status: 0 changed or untracked files
```
