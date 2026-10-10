# N8: corrupted slot files in llama-server, and a verified crash-safe session store — 2026-10-10 (sandbox CPU, no GPU)

Task N8 of the hardware-free track (docs/reassessment/ROADMAP.md, row N8), follow-up of N7 (slot save/restore across a
restart works, K3). Question: R1 will keep long agent sessions (up to 262K tokens) on NVMe and restore them instead of
prefilling again. What does llama-server do with a slot file that is damaged (bit rot, a torn write, a file written by
another model or KV setting), and can a thin layer around the documented endpoints make restore safe? Branch
`claude/n8-session-store`, cut from `claude/n7-server-hybrid-cpu` at `986519a` (session
`session_01EyB9XrhA3NTRjjQx1A9qMh`).

Nothing here ran on a GPU, on real weights or at long context. No timings are reported (tiny random-weight models).

| piece | file |
|---|---|
| store | tools/kv/session_store.py (stdlib only): sha256 + size + identity per state file, crash-consistent save (server writes a temporary, fsync, rename to `<sid>.<gen>.slot`, then manifest commit), quarantine on a failed size / sha256 / n_restored check or a server refusal (an identity mismatch keeps the file; an unreachable server raises and judges nothing), LRU disk budget, `recover()`, CLI |
| probe | tools/kv/slot_corruption_probe.py: real llama-server, damaged files restored raw and through the store |
| tests | tests/test_session_store.py (20, against the stand-in server); tests/test_srvcheck.py +1 (null log-prob) |
| stand-in server | tests/fixtures/srvcheck/stub_llama_server.py: faults `save_short`, `restore_refuse`, `restore_partial` added |
| harness fix | tools/srvcheck/hybrid_server_checks.py `compare()`: a null log-prob (llama-server's JSON writer emits `null` for any non-finite value: NaN or ±inf, `vendor/nlohmann/json.hpp:19523-19525`) counts as an infinite difference instead of crashing |
| CI | .github/workflows/server-cpu.yml: session-store tests + the probe on both tiny models after the N7 step |
| mutation check | docs/evidence/mutation-scripts/2026-10-10-n8-session-store.py (S1–S13) |

## 1. What llama-server checks on restore (CONFIRMED, code @79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6)

* There is **no checksum** anywhere on the restore path. What is checked is structure:
  * `src/llama-context.cpp:3413-3465` `state_seq_load_file`: magic and version; the token count against the file size
    and against the caller's buffer (the server sizes that buffer from the file's own count, `server-context.cpp:
    2968-2971`); that the state reader consumed no more than the file holds.
  * the memory readers: layer count, cell count, V transposition, key/value rotation, tensor types, row and element
    sizes, GQA/state embedding sizes (`src/llama-kv-cache.cpp:2561-2683`, `src/llama-memory-recurrent.cpp:1122-1236`,
    one `LLAMA_LOG_ERROR` per check);
  * the server (`tools/server/server-context.cpp:2944-3010`, `SERVER_TASK_TYPE_SLOT_RESTORE`): the token-state format
    and version (`server-common.cpp:639-656`), that the tokens fit the slot context and are valid vocabulary ids
    (`restored.validate`, 2980-2986).
  So a flip in a header or size field is refused, while a flip in the tensor data itself passes every check — which is
  what all 26 damaged files of section 2 did (none hit a checked field).
* `tools/server/server-context.cpp:2658-2714` `load_slot_checkpoints`: a damaged appendix magic returns 0 silently
  (2669-2671); a bad version or a truncated appendix returns 0 with a warning (`truncated context checkpoint appendix in
  '...' - ignored`, 2687); in every case the restore itself still succeeds.
* Save writes the final file name directly: `state_seq_save_file` opens it with `llama_file file(filepath, "wb");`
  (`src/llama-context.cpp:3468`), then the server appends the checkpoint appendix (`server-context.cpp:2906-2926`);
  no temporary + rename, no fsync. HYPOTHESIS from this code path: a crash during save can leave a torn file under the
  real name; a torn write was not simulated against the real server.

## 2. Real server, damaged files (CONFIRMED, literal)

Server: `$S/llama.cpp/build-cpu/bin/llama-server` (llama.cpp 79e2e74, CPU build of N7), models: the N7 tiny GGUFs
(`tools/tinygen/qwen4exp_tiny.py`, `glm5next_tiny.py`), arguments `-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32
--no-warmup --slot-save-path <dir>`. 200-token prompt, 16 greedy tokens, top-5 log-probs. Cases: `bit6@f` flips bit 6
of the byte at fraction f of the file; `exp@f` flips bit 6 of the highest byte of the aligned 32-bit word there (sign /
exponent of an f32); `truncated-1000` drops the last 1000 bytes. `raw`: POST `/slots/0?action=restore` with the damaged
file, then the same prompt; `store`: `SessionStore.restore` with the same damage, then the same prompt. `prompt_n` = prompt
tokens the server processed (4 = restored up to the last checkpoint, 200 = full prefill).

```
$ for M in qwen4exp glm5next; do echo "== $M"; timeout 900 python3 tools/kv/slot_corruption_probe.py --server $S/llama.cpp/build-cpu/bin/llama-server --model $S/n7/final2/$M.gguf --workdir $S/n8final/$M --out $S/n8final/$M.json; echo "exit=$?"; done 2>&1 | tee $S/n8final/probe.log
== qwen4exp
intact           raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: restored                  prompt_n   4 same True
bit6@0.1         raw: accepted prompt_n   4 same_tokens True  max|d| 4.77e-07  nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.1          raw: accepted prompt_n   4 same_tokens True  max|d| 0.0106    nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.2         raw: accepted prompt_n   4 same_tokens True  max|d| 2.38e-07  nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.2          raw: accepted prompt_n   4 same_tokens False max|d| inf       nulls 25  | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.3         raw: accepted prompt_n   4 same_tokens True  max|d| 4.29e-06  nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.3          raw: accepted prompt_n   4 same_tokens False max|d| 0.639     nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.4         raw: accepted prompt_n   4 same_tokens True  max|d| 2.38e-07  nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.4          raw: accepted prompt_n   4 same_tokens False max|d| 0.736     nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.5         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.5          raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.6         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.6          raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
truncated-1000   raw: accepted prompt_n 200 same_tokens True  max|d| 0         nulls 0   | store: refused  size 255724 != 256724 prompt_n 200 same True
exit=0
== glm5next
intact           raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: restored                  prompt_n   4 same True
bit6@0.1         raw: accepted prompt_n   4 same_tokens True  max|d| 0.00368   nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.1          raw: accepted prompt_n   4 same_tokens False max|d| 1.18      nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.2         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.2          raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.3         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.3          raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.4         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.4          raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.5         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.5          raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
bit6@0.6         raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.6          raw: accepted prompt_n   4 same_tokens False max|d| 0.529     nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
truncated-1000   raw: accepted prompt_n 200 same_tokens True  max|d| 0         nulls 0   | store: refused  size 286020 != 287020 prompt_n 200 same True
exit=0
```

An earlier run of the same probe (`$S/n8/probe.log`, before the eviction fix of section 4, which does not touch restore)
printed identical output (`diff $S/n8/probe.log $S/n8final/probe.log` empty, 32 lines).

Counted from the table (26 damaged files, 13 per model):

* raw llama-server accepted **26/26**. **5** changed the generated tokens (qwen4exp exp@0.2, exp@0.3, exp@0.4; glm5next
  exp@0.1, exp@0.6); in qwen4exp exp@0.2, 25 of the 80 top-5 log-probs (16 steps x 5) were null, i.e. non-finite (NaN or
  ±inf: the JSON writer does not distinguish). **6** more kept the tokens but changed
  log-probs (qwen4exp bit6@0.1–0.4 and exp@0.1, glm5next bit6@0.1; up to 0.0106; four of them are <= 4.29e-06, below the
  harness tolerance of 1e-3). **15** had no visible effect, among
  them both truncations: the server answered `"n_restored": 215` (success) and logged
  `W slot load_slot_ch: id  0 | task -1 | truncated context checkpoint appendix in '.../raw-truncated-1000.slot' - ignored`,
  then prefilled the whole prompt (prompt_n 200) — a silent loss of the restore, not a wrong answer.
* the store refused **26/26** before the server saw the file (24 by sha256, the 2 truncations by size; the server's log
  shows one load of a store file per model, the intact one), quarantined the file,
  erased the slot, and the next request prefilled (prompt_n 200) and gave the reference tokens (`same True`) in all 26;
  the intact file was restored in both models (prompt_n 4, same tokens).

Which bytes matter depends on the file layout (header, tokens, per-layer KV / recurrent state, checkpoint appendix) and
on which cells the next request reads; the probe does not map offsets to tensors. PROVISIONAL: a flip in an unused or
overwritten region explains the 15 no-effect cases; not traced.

## 3. Unit tests (CONFIRMED, literal)

```
$ python3 -m unittest tests.test_session_store tests.test_srvcheck tests.test_n7_report -v
test_bad_session_id ... ok
test_budget_evicts_least_recently_used ... ok
test_concurrent_saves_under_a_tight_budget_do_not_deadlock ... ok
test_crash_at_every_save_step_leaves_a_restorable_store ... ok
test_flipped_byte_is_refused_before_the_server_sees_it ... ok
test_new_generation_replaces_the_old_file ... ok
test_other_identity_is_refused_and_the_file_kept ... ok
test_previous_generation_kept_while_its_save_is_starting ... ok
test_repeated_rejects_of_the_same_file_name_are_all_kept ... ok
test_restore_count_mismatch_quarantines ... ok
test_round_trip_across_a_restart ... ok
test_save_started_after_the_eviction_snapshot_keeps_the_previous_generation ... ok
test_second_writer_is_refused ... ok
test_server_refusal_quarantines ... ok
test_session_being_saved_by_another_thread_is_not_evicted ... ok
test_session_just_saved_is_never_evicted ... ok
test_session_resaved_after_the_eviction_snapshot_is_kept ... ok
test_short_write_keeps_the_previous_generation ... ok
test_truncated_file_is_refused ... ok
test_unreachable_server_does_not_judge_the_file ... ok
test_identical_passes ... ok
test_logprob_beyond_tol_fails_with_same_tokens ... ok
test_nothing_to_compare_is_not_a_pass ... ok
test_null_logprob_is_an_infinite_difference ... ok
test_step_count_mismatch_is_not_a_pass ... ok
test_token_flip_fails_and_reports_margin ... ok
test_cli_writes_json_and_fails_on_fault ... ok
test_healthy_server_passes_and_reports_reuse ... ok
test_mtp_crash_is_caught ... ok
test_mtp_divergence_is_caught ... ok
test_mtp_that_never_drafts_is_not_a_pass ... ok
test_noisy_warm_logprobs_are_caught ... ok
test_nondeterministic_server_is_caught ... ok
test_restore_noop_is_caught ... ok
test_restore_wrong_state_is_caught ... ok
test_serialized_server_is_no_isolation_evidence ... ok
test_slot_crosstalk_is_caught ... ok
test_stale_reuse_is_caught ... ok
test_stale_state_after_kv_shift_is_caught ... ok
test_attr_compares_against_the_first_run ... ok
test_cli_usage ... ok
test_lines_print_stored_values ... ok
Ran 42 tests in 66.295s
OK
```

(full log `$S/n8final/unittests-v2.log`; test ids shortened by `sed` for the first 39 lines; the crash test covers four crash points: after the server's write,
after the rename, after the manifest temporary is written, after the manifest commit; each followed by `recover(deep=True)`
and a restore.)

## 4. Mutation check and the bug it found (CONFIRMED, literal)

`python3 docs/evidence/mutation-scripts/2026-10-10-n8-session-store.py <tree> <scratch>`: each mutant edits a copy of
session_store.py and runs `tests.test_session_store` (300 s timeout; a hang counts as caught).

Round 1 (first version of the tests):

```
baseline                                                 rc=0 ok       OK
S1 no sha256 check                                       rc=1 caught   FAILED (errors=1)
S2 no size check                                         rc=1 caught   FAILED (failures=1)
S3 no n_restored check                                   rc=1 caught   FAILED (failures=1)
S4 old generation deleted before the manifest commit     rc=1 caught   FAILED (errors=1)
S5 recover keeps temporaries                             rc=0 SURVIVED OK
S6 transient error judges the file                       rc=1 caught   FAILED (failures=1)
S7 identity not checked                                  rc=1 caught   FAILED (failures=1)
S8 eviction may evict the session just saved             rc=0 SURVIVED OK
S9 slot not erased after a refused restore               rc=1 caught   FAILED (failures=1)
S10 save size check skipped                              rc=1 caught   FAILED (failures=1)
```

(Mutant S5 removes only the temporaries branch of `recover`; `tmp-*.slot` files are still deleted by the orphan branch,
so what survives is `.<sid>.manifest.*.tmp`, which the crash test then catches.)

Tests added: the crash test asserts that no temporaries remain after `recover` (S5); `test_session_just_saved_is_never_evicted`
(budget smaller than any state file, S8); `test_concurrent_saves_under_a_tight_budget_do_not_deadlock` (two servers, two
sessions, every save evicts). The next mutation run stopped responding at S8 and was killed. Cause, in the unmutated code:
`save()` called `_evict()` while still holding its own session lock, and `_evict()` → `drop(other)` takes the other
session's lock. Thread A (lock a, wants b) and thread B (lock b, wants a) deadlock — a real bug for any two concurrent
saves under a budget. Fix: `_evict()` runs after the session lock is released (session_store.py `save`, comment "evict with
no session lock held"). The test threads are daemons with a 60 s join each, so the deadlock fails the test instead of
hanging it.

The pre-fix code (`$S/n8-old`, a reconstruction: the committed tree with `self._evict(keep=sid)` / `return m` moved back
inside the session lock and the comment above them removed; `diff -r` shows nothing else in tools/ or tests/) with the
final test file. Abridged from `$S/n8final/deadlock-old.log` (separators, `Traceback` and caret lines dropped, path
shortened; the log does not record the command, which was the one shown):

```
$ timeout 200 python3 -m unittest tests.test_session_store.Store.test_concurrent_saves_under_a_tight_budget_do_not_deadlock
FAIL: test_concurrent_saves_under_a_tight_budget_do_not_deadlock (tests.test_session_store.Store.test_concurrent_saves_under_a_tight_budget_do_not_deadlock)
  File ".../n8-old/tests/test_session_store.py", line 219, in test_concurrent_saves_under_a_tight_budget_do_not_deadlock
    self.assertFalse(any(t.is_alive() for t in ts), "deadlock: a saving thread did not finish")
AssertionError: True is not false : deadlock: a saving thread did not finish
Ran 1 test in 120.666s
FAILED (failures=1)
rc=1
```

One saved run (120.666 s: both 60 s joins expired). NOT SAVED as files, seen only in the session transcript: an earlier
identical run (120.662 s), and, before the test threads were made daemons, 3 of 3 runs that hung until `timeout 200`
killed them.

A complete run on the same code as round 3 also gave 10/10 (`$S/mut-n8c/round2.log`, not reproduced here). Round 3, on
the code of commit c5f78ea:

```
baseline                                                 rc=0 ok       OK
S1 no sha256 check                                       rc=1 caught   FAILED (errors=1)
S2 no size check                                         rc=1 caught   FAILED (failures=1)
S3 no n_restored check                                   rc=1 caught   FAILED (failures=1)
S4 old generation deleted before the manifest commit     rc=1 caught   FAILED (errors=2)
S5 recover keeps temporaries                             rc=1 caught   FAILED (failures=1)
S6 transient error judges the file                       rc=1 caught   FAILED (failures=1)
S7 identity not checked                                  rc=1 caught   FAILED (failures=1)
S8 eviction may evict the session just saved             rc=1 caught   FAILED (errors=1)
S9 slot not erased after a refused restore               rc=1 caught   FAILED (failures=1)
S10 save size check skipped                              rc=1 caught   FAILED (failures=1)
exit=0
```

## 4b. Audit of the first version and the fixes (CONFIRMED, literal)

The evidence audit of commit 33cf4e8 (evidence-auditor, same session) found 3 FAILs in this report (fixed in the text
above: the sha256/size split, "no checks" where only a checksum is missing, unsaved timings presented as literal) and two
defects in the store:

* **M11 — rejects overwrote each other in quarantine.** Destination names were `<seconds>-<file>`; the same
  `<sid>.<gen>.slot` rejected twice within a second (the probe does this 13 times) kept only the last file — 2–3 of 13
  per model survived (`$S/n8`: 2 and 3, `$S/n8final`: 2 and 2). Now `<time_ns>-<random>-<file>`.
* **M12 — a concurrent save could evict a session another thread was still saving.** Eviction excluded only the
  caller's own session. Now `save()` registers the session in an in-process "saving" set for its whole duration and
  `_evict()` skips every session in it. (Incomplete — the second audit found an interleaving that still evicts; fixed in
  section 4c.)

New tests `test_repeated_rejects_of_the_same_file_name_are_all_kept` and
`test_session_being_saved_by_another_thread_is_not_evicted` fail on the committed store of 33cf4e8 (copy in
`$S/n8-pre-audit` = `git archive 33cf4e8 tools tests` + the new test file) and pass on the fixed one:

```
$ timeout 300 python3 -m unittest tests.test_session_store.Store.test_repeated_rejects_of_the_same_file_name_are_all_kept tests.test_session_store.Store.test_session_being_saved_by_another_thread_is_not_evicted   # in $S/n8-pre-audit
FAIL: test_repeated_rejects_of_the_same_file_name_are_all_kept (tests.test_session_store.Store.test_repeated_rejects_of_the_same_file_name_are_all_kept)
AssertionError: 2 != 3 : ['1791661748-s1.1.slot', '1791661749-s1.1.slot']
FAIL: test_session_being_saved_by_another_thread_is_not_evicted (tests.test_session_store.Store.test_session_being_saved_by_another_thread_is_not_evicted)
AssertionError: unexpectedly None : b's eviction removed a while a was being saved
Ran 2 tests in 16.209s
FAILED (failures=2)
rc=1
```
(lines filtered with `grep -E "^(FAIL|ERROR|AssertionError|Ran|OK|FAILED|rc=)"` from `$S/n8final/pre-audit-new-tests.log`.)

The mutation script now targets the new code: S8 removes the in-flight exclusion (`busy = set()`), S11 restores the
old quarantine name. Round 4, on the code of commit 25c61f3:

```
baseline                                                 rc=0 ok       OK
S1 no sha256 check                                       rc=1 caught   FAILED (errors=2)
S2 no size check                                         rc=1 caught   FAILED (failures=1)
S3 no n_restored check                                   rc=1 caught   FAILED (failures=1)
S4 old generation deleted before the manifest commit     rc=1 caught   FAILED (errors=2)
S5 recover keeps temporaries                             rc=1 caught   FAILED (failures=1)
S6 transient error judges the file                       rc=1 caught   FAILED (failures=1)
S7 identity not checked                                  rc=1 caught   FAILED (failures=1)
S8 eviction may evict a session being saved              rc=1 caught   FAILED (failures=1, errors=1)
S9 slot not erased after a refused restore               rc=1 caught   FAILED (failures=1)
S10 save size check skipped                              rc=1 caught   FAILED (failures=1)
S11 quarantine names not unique                          rc=1 caught   FAILED (failures=1)
exit=0
```

The real-server probe of section 2 on the fixed store (`$S/n8final2/probe.log`): the 28 case lines are identical to
section 2 (`diff` against `$S/n8final/probe.log` empty apart from the added counts), and every reject is kept. The counts
are printed by the loop, not by the probe:

```
$ for M in qwen4exp glm5next; do echo "== $M"; timeout 900 python3 tools/kv/slot_corruption_probe.py ... --workdir $S/n8final2/$M --out $S/n8final2/$M.json; echo "exit=$?"; echo "quarantined .slot files: $(ls $S/n8final2/$M/slots/quarantine/ | grep -c '\.slot$')"; done > $S/n8final2/probe.log 2>&1
quarantined .slot files: 13
quarantined .slot files: 13
```

Not changed by the audit: the store holds an exclusive fcntl lock for any instance, readers included (one process per
directory), and `recover()` deletes every unreferenced `*.slot` in the server's slot directory — point the store at a
directory used by nothing else.

## 4c. Second audit of the fixes (evidence-auditor, same day) and the eviction re-check (CONFIRMED, literal)

The second audit confirmed F1–F3 and M11 fixed and found M12 still false: `_evict()` takes its snapshot of the saving
sessions once, and `drop()` did not look again. Interleaving (the auditor's probe `$S/audit-n8b/race/race_m12.py`, on
43c2171): b's eviction chooses a from its snapshot → a's save starts, commits gen 2 → b drops a → `manifest('a') = None`
after both saves. Effect: a cached session silently lost (the next request prefills), not a wrong state.

Fix (`tools/kv/session_store.py`): eviction drops a victim only through `_evict_one(sid, gen)`, which under the victim's
lock skips it if a save of it is in progress or if its manifest is no longer the generation eviction chose. Two test hooks
(`save_started`, `before_evict_drop`) let the tests force the interleavings. Three tests:
* `test_session_resaved_after_the_eviction_snapshot_is_kept` — a re-saved (gen 2) after b's snapshot; pins the generation
  re-check (S13);
* `test_save_started_after_the_eviction_snapshot_keeps_the_previous_generation` — a's save starts after b's snapshot;
  pins the in-progress re-check (S12): gen 1 must stay, it is what a failed save falls back to;
* `test_previous_generation_kept_while_its_save_is_starting` — a's save started before b's snapshot.
On the store of 43c2171 these tests cannot pause it (the hooks do not exist there), so the demonstration that the defect
is real is the auditor's probe above; that each half of the fix is needed is shown by the mutants S12 and S13.

Round 5 (code with the fix, before the third test):

```
baseline                                                 rc=0 ok       OK
S1 no sha256 check                                       rc=1 caught   FAILED (errors=2)
S2 no size check                                         rc=1 caught   FAILED (failures=1)
S3 no n_restored check                                   rc=1 caught   FAILED (failures=1)
S4 old generation deleted before the manifest commit     rc=1 caught   FAILED (errors=2)
S5 recover keeps temporaries                             rc=1 caught   FAILED (failures=1)
S6 transient error judges the file                       rc=1 caught   FAILED (failures=1)
S7 identity not checked                                  rc=1 caught   FAILED (failures=1)
S8 eviction may evict a session being saved              rc=0 SURVIVED OK
S9 slot not erased after a refused restore               rc=1 caught   FAILED (failures=1)
S10 save size check skipped                              rc=1 caught   FAILED (failures=1)
S11 quarantine names not unique                          rc=1 caught   FAILED (failures=1)
S12 eviction drop does not re-check a save in progress   rc=0 SURVIVED OK
S13 eviction drop does not re-check the generation       rc=1 caught   FAILED (failures=1)
exit=0
```

S12 survived: no test let a save start between the snapshot and the drop. With the third test, round 6 (the committed
code):

```
baseline                                                 rc=0 ok       OK
S1 no sha256 check                                       rc=1 caught   FAILED (errors=2)
S2 no size check                                         rc=1 caught   FAILED (failures=1)
S3 no n_restored check                                   rc=1 caught   FAILED (failures=1)
S4 old generation deleted before the manifest commit     rc=1 caught   FAILED (errors=2)
S5 recover keeps temporaries                             rc=1 caught   FAILED (failures=1)
S6 transient error judges the file                       rc=1 caught   FAILED (failures=1)
S7 identity not checked                                  rc=1 caught   FAILED (failures=1)
S8 eviction may evict a session being saved              rc=0 SURVIVED OK
S9 slot not erased after a refused restore               rc=1 caught   FAILED (failures=1)
S10 save size check skipped                              rc=1 caught   FAILED (failures=1)
S11 quarantine names not unique                          rc=1 caught   FAILED (failures=1)
S12 eviction drop does not re-check a save in progress   rc=1 caught   FAILED (errors=1)
S13 eviction drop does not re-check the generation       rc=1 caught   FAILED (failures=1)
exit=0
```

S8 (no exclusion in the snapshot) is now an equivalent mutant: the re-check under the victim's lock prevents the same
evictions on its own; the snapshot exclusion stays as a first layer that avoids taking locks for busy sessions.
The commit message of 25c61f3 ("a session being saved is never evicted by another thread") was not true until this fix;
history is not rewritten.

## 5. What this means for R1 (PROVISIONAL)

* Never point R1 at `/slots/{id}?action=restore` directly for sessions that matter: llama-server @79e2e74 accepts damaged
  state silently, and a damaged state can change the answer (5/26 here) or produce NaN. Use the store (or the same checks
  in the gateway): verify, then restore; on any doubt, prefill.
* The identity dict must name everything that changes the state layout or meaning: model sha256, engine commit, KV types
  (`-ctk/-ctv`), FA on/off, context size, MTP/draft settings (the checkpoint appendix holds draft state, N7 evidence).
* Cost of the check: one sha256 pass over the file per restore. The size of a 262K-token slot file of the target models
  and the hash and NVMe read rates on the server are UNKNOWN; K3 measures them. If hashing turns out to cost as much as
  the read, verify while reading (one pass) or use a faster hash.

## 6. CI and not done

CI: workflow `server-cpu` (session-store tests + this probe on both tiny models) green on 33cf4e8, the commit before the
audit fixes: runs 38080609883 (push) and 38080628252 (pull_request, PR #4); and green on ebecad4, the head with the audit
fixes: runs 38081791575 (push) and 38081793624 (pull_request).


GPU, real weights, 262K-token states, NVMe timing (K3), torn writes against the real server, concurrent restore + save of
the same session from two processes (the store allows one writer process per directory by an fcntl lock): NOT RUN.
