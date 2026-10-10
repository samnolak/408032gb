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
| store | tools/kv/session_store.py (stdlib only): sha256 + size + identity per state file, crash-consistent save (server writes a temporary, fsync, rename to `<sid>.<gen>.slot`, then manifest commit), quarantine on any verification failure, LRU disk budget, `recover()`, CLI |
| probe | tools/kv/slot_corruption_probe.py: real llama-server, damaged files restored raw and through the store |
| tests | tests/test_session_store.py (15, against the stand-in server); tests/test_srvcheck.py +1 (null log-prob) |
| stand-in server | tests/fixtures/srvcheck/stub_llama_server.py: faults `save_short`, `restore_refuse`, `restore_partial` added |
| harness fix | tools/srvcheck/hybrid_server_checks.py `compare()`: a null log-prob (llama-server sends null for NaN) counts as an infinite difference instead of crashing |
| CI | .github/workflows/server-cpu.yml: session-store tests + the probe on both tiny models after the N7 step |
| mutation check | docs/evidence/mutation-scripts/2026-10-10-n8-session-store.py (S1–S10) |

## 1. What llama-server checks on restore (CONFIRMED, code @79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6)

* `src/llama-context.cpp:3412-3460` `state_seq_load_file`: magic and version, token count against the file size and the
  slot capacity, and that the state reader consumed no more than the file holds. There is no checksum: any bit flip
  inside the state payload passes.
* `tools/server/server-context.cpp:2944-3010` (`SERVER_TASK_TYPE_SLOT_RESTORE`): additionally only that the tokens fit
  the slot context and are valid vocabulary ids (`restored.validate`).
* `tools/server/server-context.cpp:2658-2700` `load_slot_checkpoints`: a damaged or truncated checkpoint appendix is
  ignored with a warning (`truncated context checkpoint appendix in '...' - ignored`) and the restore still succeeds.
* Save writes the final file name directly: `state_seq_save_file` opens it with `llama_file(filepath, "wb")`
  (`src/llama-context.cpp:3466-3467`), then the server appends the checkpoint appendix (`server-context.cpp:2906-2926`);
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
$ for M in qwen4exp glm5next; do python3 tools/kv/slot_corruption_probe.py --server $S/llama.cpp/build-cpu/bin/llama-server --model $S/n7/final2/$M.gguf --workdir $S/n8final/$M --out $S/n8final/$M.json; echo "exit=$?"; done
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
  exp@0.1, exp@0.6); in qwen4exp exp@0.2, 25 of the top-5 log-probs were null (NaN). **6** more kept the tokens but changed
  log-probs (qwen4exp bit6@0.1–0.4 and exp@0.1, glm5next bit6@0.1; up to 0.0106). **15** had no visible effect, among
  them both truncations: the server answered `"n_restored": 215` (success) and logged
  `W slot load_slot_ch: id  0 | task -1 | truncated context checkpoint appendix in '.../raw-truncated-1000.slot' - ignored`,
  then prefilled the whole prompt (prompt_n 200) — a silent loss of the restore, not a wrong answer.
* the store refused **26/26** before the server saw the file (25 by sha256, the truncations by size), quarantined the file,
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
test_restore_count_mismatch_quarantines ... ok
test_round_trip_across_a_restart ... ok
test_second_writer_is_refused ... ok
test_server_refusal_quarantines ... ok
test_session_just_saved_is_never_evicted ... ok
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
test_attr_compares_against_the_first_run (tests.test_n7_report.Report.test_attr_compares_against_the_first_run) ... ok
test_cli_usage (tests.test_n7_report.Report.test_cli_usage) ... ok
test_lines_print_stored_values (tests.test_n7_report.Report.test_lines_print_stored_values) ... ok
Ran 37 tests in 64.031s
OK
```

(test ids shortened by `sed` for the first 34 lines; the crash test covers four crash points: after the server's write,
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

Tests added: the crash test asserts that no temporaries remain after `recover` (S5); `test_session_just_saved_is_never_evicted`
(budget smaller than any state file, S8); `test_concurrent_saves_under_a_tight_budget_do_not_deadlock` (two servers, two
sessions, every save evicts). The next mutation run stopped responding at S8 and was killed. Cause, in the unmutated code:
`save()` called `_evict()` while still holding its own session lock, and `_evict()` → `drop(other)` takes the other
session's lock. Thread A (lock a, wants b) and thread B (lock b, wants a) deadlock — a real bug for any two concurrent
saves under a budget. Fix: `_evict()` runs after the session lock is released (session_store.py `save`, comment "evict with
no session lock held"). The test threads are daemons with a 60 s join each, so the deadlock fails the test instead of
hanging it.

The pre-fix code (a copy in `$S/n8-old`: the only difference in `save` is the indentation of `self._evict(keep=sid)` /
`return m`) with the final test file:

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

(two runs, 120.662 s and 120.666 s; with the earlier non-daemon threads the same test hung until `timeout 200` killed it
in 3 of 3 runs.)

Round 3, on the final code (the tree committed with this file):

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

## 5. What this means for R1 (PROVISIONAL)

* Never point R1 at `/slots/{id}?action=restore` directly for sessions that matter: llama-server @79e2e74 accepts damaged
  state silently, and a damaged state can change the answer (5/26 here) or produce NaN. Use the store (or the same checks
  in the gateway): verify, then restore; on any doubt, prefill.
* The identity dict must name everything that changes the state layout or meaning: model sha256, engine commit, KV types
  (`-ctk/-ctv`), FA on/off, context size, MTP/draft settings (the checkpoint appendix holds draft state, N7 evidence).
* Cost of the check: one sha256 pass over the file per restore. At 262K tokens a slot file is large (UNKNOWN for the target
  models; K3 measures it on the server); sha256 on one Zen 3 core runs at roughly 1–2 GB/s (PROVISIONAL, not measured here),
  so verification may cost about as much as reading the file from NVMe. If that matters, verify while reading (one pass)
  or use a faster hash; measure in K3.

## 6. Not done

GPU, real weights, 262K-token states, NVMe timing (K3), torn writes against the real server, concurrent restore + save of
the same session from two processes (the store allows one writer process per directory by an fcntl lock): NOT RUN.
