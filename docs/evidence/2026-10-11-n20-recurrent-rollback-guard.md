# N20: llama.cpp recurrent rollback guard (qwen4exp): reachability, patch, CPU tests, 2026-10-11 (sandbox CPU, no GPU)

Task N20 of the hardware-free track. N19 (docs/evidence/2026-10-11-n19-qwen4exp-multiseq.md, findings 5-7) found three
places where llama.cpp's recurrent memory is silently wrong for qwen4exp at both pins: (a) a rollback deeper than the
sequence's last ubatch is accepted and reads a stale snapshot plane; (b) `seq_cp` while a rollback is pending gives the
copy the pre-rollback state; (c) at ec7630a only, a rollback of a cell two sequences share. N20 asks whether llama-server
can reach them, adds a guard as a llama.cpp patch (applied to a copy, never in place), tests it on CPU with the N19
tooling, and gives a rule that keeps R1 safe without the patch. Branch `claude/n20-recurrent-rollback-guard`, cut from
N19 at `e71b241`. Role: engine-integrator. `<S>` = the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad` (outputs pasted with only that
substitution).

Pins: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md) at `<S>/llama-ec7630a`, and
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine, tools/engines/pins.json) at `<S>/llama.cpp`; both unchanged
(pin check below). Abbreviations, all @79e2e74 unless "@ec7630a" is written: "SC" = tools/server/server-context.cpp,
"RC" = src/llama-memory-recurrent.cpp, "CC" = common/common.cpp. Every cited line was printed from the pinned checkout by
docs/evidence/mutation-scripts/2026-10-11-n20-source-lines.sh before citing (output: docs/evidence/2026-10-11-n20-source-lines.txt,
section 8).

## 1. Summary

| item | result | label |
|---|---|---|
| (a), (b), (c) through llama-server with the R1 config (`-np 1`, no `--spec-type`) | unreachable: `n_rs_seq` = 0, no rollback is ever accepted | CONFIRMED (code; N19 probe `rs0`) |
| (a) through llama-server with `--spec-type draft-mtp` (any `-np`) | unreachable: every server rollback is inside its last verify call | CONFIRMED (code) |
| (b) through llama-server with `draft-mtp` / `eagle3` / `dflash` / `dspark` and `-np` >= 2 | reachable, narrow: a decision request whose shared prompt prefix equals the cached tokens of a slot whose previous completion ended on a partly rejected draft | path CONFIRMED (code); not run on a server |
| (c) through llama-server (ec7630a) | unreachable: a slot is rolled back only right after its own verify decode, which un-shares its cell | CONFIRMED (code) |
| new, found by N20's plans (both pins, unpatched) | a rollback of all tokens of the last call reads a delta-rule plane that call never wrote (`last_call`); a second rollback after the extra-row copy applied the first reads plane 1 of a row that holds ANOTHER sequence's snapshot (`extra_cell`); the copy made while a rollback is pending reads the pre-rollback state also when it decodes before the source (`cp_pending_order`); all silent | CONFIRMED (runs, section 5.2) |
| the patch | `patches/llama.cpp/{79e2e74,ec7630a}/0001-recurrent-rollback-guard.patch`, 40 / 44 lines in RC + 4 in the header; `patches/llama.cpp/apply.sh` | - |
| N19 plans on the patched build | 72/72 at both pins; `stale`: both rollbacks refused, no divergence; `cp_pending`: no divergence; `cp_shared_rb` at ec7630a: refused; the other 6 plans byte-identical dumps | see 5.1, 5.3 |
| N20 step (12 plans, 102 tests) on the patched build | OK at both pins (and OK on the unpatched checkout with the pin's own expectations, which assert the divergences) | see 5.1, 5.2 |
| patch reverted / each hunk disabled | the tests fail, except for the `state_read` hunk (no plan restores state) | see 5.4 |
| R1 without the patch | rule: a speculative type that gives the target `n_rs_seq` > 0 only with `-np 1`; `scripts/r1/rs_rollback_rule.py`, both R1 llama.cpp configs pass | PROVISIONAL (code reading) |


## 2. Reachability at 79e2e74 (CONFIRMED by reading unless labelled; lines printed in section 8)

How a qwen4exp context gets rollback snapshots at all: `n_rs_seq` of the target context is
`params.speculative.need_n_rs_seq()` (CC:1700), which is the draft length `draft.n_max` (default 3, common/common.h:329)
when a speculative type is `draft-mtp`, `draft-eagle3`, `draft-dflash` or `draft-dspark`, else 0
(common/common.h:399-405). Draft contexts always get 0 (common/speculative.cpp:2618). With `n_rs_seq` = 0 every partial
`seq_rm` is refused (RC:214, rollback <= 0 fails) and `rs_idx` is only ever set by an accepted partial `seq_rm` (RC:215),
so no stale plane and no pending rollback can exist (N19 probe `rs0` showed the refusal at both pins). The server learns
the type once: `n_rs_seq` > 0 is "RS", bounded partial removal (CC:1562-1565); a recurrent memory with `n_rs_seq` = 0
refuses the probe rollback and becomes "FULL", which makes the server use state checkpoints (CC:1587-1592,
SC:1373-1380, 3351-3352). `common_memory::seq_rm` aborts the process on any refusal (CC:1601-1606, used for target and
draft, CC:1623-1628). The qwen4exp MTP draft context has no recurrent layer (src/llama-model.cpp:2752-2759), so its
`seq_rm` only moves a position (RC:206-210) and none of (a)-(c) applies to it.

| # | scenario | (a) deeper than the last ubatch | (b) `seq_cp` with a pending rollback | (c) shared-cell rollback (ec7630a) | evidence |
|---|---|---|---|---|---|
| 1 | R1 config `scripts/r1/configs/llamacpp-server.json` / `-moecache.json`: `-np 1`, no `--spec-type`, `--slot-save-path` | no | no | no | no speculative type, so `n_rs_seq` = 0 (common/common.h:399-405, CC:1700); argv in the configs (`python3 scripts/r1/rs_rollback_rule.py`, section 5.6) |
| 2 | speculative `draft-simple` (separate draft model) or any `ngram-*` type alone | no | no | no | `n_rs_seq` = 0: target is FULL, every partial acceptance restores a checkpoint (SC:4433-4464) |
| 3 | `--spec-type draft-mtp` (also eagle3 / dflash / dspark), one slot | no | no | no | the verify batch of a slot is `[sampled, draft...]`, all at consecutive positions in one batch (SC:539-580); split across llama_process calls it throws, not silently (SC:4277-4285); the split keeps the last `n_rs_seq + 1` tokens of every sequence in one ubatch (src/llama-memory-hybrid-idx.cpp:94-99, src/llama-batch.cpp:698-751; `embd_all` = split_seq only with `--embeddings`, src/llama-context.cpp:1751, 1830); the rollback `n_rollback = draft + 1 - accepted <= draft` (SC:4431); a draft longer than `n_rs_seq` takes a checkpoint before the verify and restores it instead of rolling back (SC:3428-3448, 4433-4464); the partial rollback is `seq_rm(slot, pos_next, -1)` right after that verify (SC:4505). So the plane read is < the tokens of this sequence in its last ubatch. A refused partial rollback here would abort (CC:1601-1606), not corrupt |
| 4 | same, several slots `-np` > 1, `kv_unified` (the server default is `-np` auto = 4 slots with `kv_unified`, common/arg.cpp:1406, tools/server/server.cpp:168-173) | no | **yes, narrow** (row 10) | no | per slot as row 3; a slot left out of a batch (SC:3340-3344, `can_batch_with`) keeps its pending rollback until its cell is an extra row of another ubatch, where build_rs applies it into plane 0 (src/llama-graph.cpp:3706-3711, s_copy RC:1368-1386): exact (N19 finding 4). The slot's next action is its own verify decode, never a second rollback |
| 5 | prompt-cache reuse (`cache_prompt`, default) | no | no (row 10 aside) | no | the common prefix is cut with `seq_rm(slot, p0, -1)` (SC:3826-3831). For a hybrid memory `seq_pos_min` = max of the two caches' minima = the recurrent cell's position (src/llama-memory-hybrid.cpp:172-180); a prefix shorter than the cache (`pos_min >= pos_min_thold`, SC:3668-3678, 3726) restores a context checkpoint with `pos_max < pos_next` (SC:3735-3738: `cur.pos_max <= pos_next` and `cur.pos_min < pos_min_thold`, with pos_min = pos_max) or re-processes from 0 (SC:3766-3771); after a restore `pos_next = min(pos_next, pos_max + 1)` (SC:3761), so the cut is past the restored cell. A full-prefix match cuts at the cell's position + 1 (no-op). A pending rollback left by the slot's last verify survives into the next task and is applied by its first decode (correct: it is the cached state) |
| 6 | `--cache-reuse N` (KV shifting), context shift | no | no | no | disabled for qwen4exp: IMRoPE (src/llama-model.cpp:3245-3247) gives `n_pos_per_embd` 4 (src/llama-hparams.cpp:285-288), so `get_can_shift` is false (src/llama-kv-cache.cpp:1208-1217, hybrid: src/llama-memory-hybrid.cpp:133-135) and the server turns both off (SC:1318-1328) |
| 7 | context checkpoints (`--ctx-checkpoints`) and speculative checkpoints | no | no | no | a checkpoint saves the logical state, plane `rs_idx[seq]` (state_write RC:803-825), and a restore resets the pending rollback (RC:895-897) and writes plane 0 only (state_read_data RC:1157-1160); the server never rolls back right after a restore (rows 3, 5). Upstream would accept a rollback by 1 right after a restore and read plane 1 (not reached by the server; refused by the patch) |
| 8 | `n_cmpl` > 1 (children copied from the parent, `copy_state_to`) | no | no | no | the copy happens after the parent's prompt was decoded (SC:4243-4266, 768-780): the parent's first prompt decode applied any pending rollback (the server always evaluates >= 1 token, SC:3789-3794) |
| 9 | slot save / restore (`/slots?action=save|restore`, `--slot-save-path`) | no | no | no | per-sequence state_write / state_read (row 7); `seq_cp` is called only in `copy_prompt_to` / `copy_state_to` (SC:763, 772); `seq_keep` is never called by the server (grep, section 8) |
| 10 | decision requests with a shared prompt prefix (`server_decision_group_tasks`, needs >= 2 slots) and a speculative type with `n_rs_seq` > 0 | no | **yes, narrow** | no | the parent stops at `n_tokens_shared` and the children are copied with `copy_prompt_to` (`seq_rm(child, -1, -1)` + `seq_cp(parent, child)`, SC:761-766) at SC:3835-3843, which runs right after the prompt cut and BEFORE any decode of this task. If the parent slot's cached tokens are exactly the shared prefix (`n_past` = `n_tokens_shared`, SC:3591-3596, 3836), the parent's pending rollback from its previous completion (a verify with a partly rejected draft leaves one: SC:4505, then release on stop) is still pending, and the children read plane 0 of the shared cell: the state with the rejected draft tokens (N19 finding 6). Grouping: tools/server/server-decision.cpp:1139-1161 (groups of `n_slots` tasks, none with one slot, :1154), SC:5792-5794. PROVISIONAL that a real client produces the exact prefix length; no server was run |
| 11 | ec7630a server | no | no | no | ec7630a has the same verify / checkpoint logic (SC:3926-3928@ec7630a, 3998@ec7630a) and no shared-prefix copy (`n_tokens_shared` / `copy_prompt_to`: 0 matches, section 8); `copy_state_to` only (SC:692-703@ec7630a). (c) needs a rollback of a sequence whose cell is shared, i.e. a `seq_cp` between its last decode and the rollback; the server rolls a slot back only right after that slot's verify decode (row 3), and that decode moves a sequence that shares its cell to a cell of its own (RC:605-638 copy on write) |
| 12 | other callers: `common/speculative.cpp` | no | no | no | every memory call there is on the draft context (grep, section 8) |
| 13 | examples (`examples/speculative-simple` etc.), custom C API users | UNKNOWN | UNKNOWN | UNKNOWN | not read in full; speculative-simple ignores the result of its partial `seq_rm` (examples/speculative-simple/speculative-simple.cpp:331); any caller that rolls back across calls or copies a sequence with a pending rollback hits (a) / (b) |

New hazard found while reading (not in N19, not fixed by the patch; PROVISIONAL, derived from RC and not run): a
sequence that loses its cell without a whole-sequence `seq_rm` - `seq_keep` of another sequence (RC:302-332) or
`seq_rm(s, 0, p1)` with a finite `p1` past its position (RC:184-187 resets `rs_idx` only for `p0 = 0, p1 = max`) - keeps
a pending `rs_idx`; its next cell is a fresh one read through `rs_z`, whose plane 0 only is zeroed by build_rs, so it
would start from plane `rs_idx` of row `rs_z`. Not reachable through the server for qwen4exp: it never calls `seq_keep`
(section 8), and its only cuts from position 0 with a finite end are cache reuse and context shift (SC:3646, 3301), both
off (row 6).

## 3. The patch

Files: `patches/llama.cpp/79e2e74/0001-recurrent-rollback-guard.patch`, `patches/llama.cpp/ec7630a/0001-recurrent-rollback-guard.patch`
(the ec7630a one adds 79e2e74's shared-cell refusal, RC:202-205), `patches/llama.cpp/apply.sh <checkout> <new dir>`
(`git clone --shared` of the checkout, detached at its HEAD, `git apply` of the pin's patches; HEAD stays the pin;
an existing directory is reused only if it is exactly that; tested fresh, reused, tampered (refused), from a shallow
fetch like parity-r1-pin.yml's, section 5.5). The upstream checkouts are never edited.

What it does (only src/llama-memory-recurrent.{h,cpp}):

* `mem_cell::rs_n`: how many snapshot planes the cell's rows hold. `find_slot` sets it for every cell of the range it
  stages: a ubatch cell `min(n_seq_tokens, 1 + n_rs_seq)` (ops.cpp:11183-11184 writes slots 0..n-1 only; the conv /
  PLE slot j > n starts at the ubatch's first row, qwen4exp.cpp:1322-1343); an extra cell keeps its value only if it
  reads its own row with no pending rollback, else 1 (build_rs copies one row into plane 0, src/llama-graph.cpp:3706-3711).
  `state_read` sets 1 (plane 0 rows only, RC:1157-1160); a per-sequence restore already gets 1 from `find_slot`
  (RC:1028-1052), so this hunk matters for a whole-context restore (RC:1061-1104). The pending index of an extra cell is
  read in `find_slot` before `s_copy` consumes it (src/llama-context.cpp:1415 apply, then 1472 set_inputs, 2070 next
  ubatch).
* `seq_rm`: a rollback `r` with `r >= rs_n` of the sequence's cell returns false (LLAMA_LOG_WARN "refusing a rollback of
  r for sequence s: its last ubatch wrote n snapshot plane(s)") and changes nothing. That is the precise condition
  "plane r was not written": a real fix (keeping older planes) would need graph changes in every recurrent model.
* `seq_cp`: the destination takes the source's `rs_idx` (0 when the source has no cell). This is the fix of (b), not a
  refusal: `llama_memory_seq_cp` returns void (include/llama.h), and the shared cell's plane `rs_idx` is exactly the
  source's logical state, which `s_copy` then reads for both (copy on write moves whichever decodes first). It also
  drops a pending rollback of the old destination (unpatched, a copy into a sequence whose own rollback is pending kept
  that index and read that plane of the source's cell); callers that clear the destination first, as llama-server does
  (SC:762, 771), never hit that, because the whole-sequence `seq_rm` resets the index (RC:184-187).
* ec7630a only: the shared-cell refusal of 79e2e74 (fixes (c)).

The rule "plane j valid iff j < the tokens of the sequence in the last ubatch" is the semantics of the snapshot planes
for qwen4exp (GDN + PLE, CONFIRMED by the cited code and the runs). The memory is shared by every arch with recurrent
rollback (src/llama-arch.cpp:1171-1186: kimi-k3, qwen35, qwen35moe, qwen4exp, deepseek4, glm5-next, nemotron-h(-moe),
lfm2(-moe), bailingmoe3); that their kernels write planes the same way is PROVISIONAL (the split rule's comment
"[TAG_RECURRENT_ROLLBACK_SPLITS] ... so that the rollback snapshots remain valid" says so for all; only qwen4exp and
glm5-next (N9 rollback run, section 5.3) were run).

Callers on a refusal: llama-server would abort through `common_memory::seq_rm` (CC:1601-1606), not fall back to a
re-prefill; the only fallbacks it has are the checkpoint restore when the rejected draft is longer than `n_rs_seq`
(SC:4433-4464) and the full re-processing of a prompt without a usable checkpoint (SC:3766-3771). By rows 3-11 the server
never issues a rollback the guard refuses, so no server change is needed; an explicit abort is acceptable where the
upstream code silently continues. The patch does not touch the server.

### 3.1 The patch at 79e2e74 (diff part of `patches/llama.cpp/79e2e74/0001-recurrent-rollback-guard.patch`)

```diff
diff --git a/src/llama-memory-recurrent.cpp b/src/llama-memory-recurrent.cpp
index 2893d81..a54fd2a 100644
--- a/src/llama-memory-recurrent.cpp
+++ b/src/llama-memory-recurrent.cpp
@@ -211,6 +211,13 @@ bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos
                 const llama_pos rollback = cell.pos - (p0 - 1);
                 // pending rollback is single-use
                 const bool pending = rs_idx[seq_id] != 0;
+                // plane `rollback` must have been written by the last ubatch of this cell: the graph writes plane j
+                // only for j < its tokens of the sequence (ggml_gated_delta_net: "older slots are caller-owned")
+                if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq && rollback >= (llama_pos) cell.rs_n) {
+                    LLAMA_LOG_WARN("%s: refusing a rollback of %d for sequence %d: its last ubatch wrote %u snapshot plane(s)\n",
+                            __func__, rollback, seq_id, cell.rs_n);
+                    return false;
+                }
                 if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                     set_rs_idx(seq_id, (uint32_t) rollback);
                     cell.pos = p0 - 1;
@@ -296,6 +303,11 @@ void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id
             cell_src.seq_id.insert(seq_id_dst);
             tail_dst.tail = tail_src.tail;
         }
+        // the destination continues from the source's logical state: a pending rollback of the source is part of
+        // it (s_copy reads plane rs_idx[seq] of the shared cell), and a pending rollback of the old destination is not
+        if ((uint32_t) seq_id_dst < rs_idx.size() && (uint32_t) seq_id_src < rs_idx.size()) {
+            rs_idx[seq_id_dst] = tail_src.tail >= 0 ? rs_idx[seq_id_src] : 0;
+        }
     }
 }
 
@@ -715,6 +727,27 @@ bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
                 // and still make these values available when setting the inputs
                 cells[i].src0 = cells[i].src;
             }
+
+            // the snapshot planes row i holds after this graph: a ubatch cell gets planes 0 .. n_seq_tokens - 1
+            // written (ggml_gated_delta_net writes the last min(n_seq_tokens, 1 + n_rs_seq) snapshot slots, the conv
+            // histories of the slots past n_seq_tokens start at the ubatch's first row); an extra cell between them
+            // gets plane 0 only (build_rs copies row s_copy(i) = plane rs_idx of src0 into it), so its other planes
+            // stay valid only if nothing moved: same row, no pending rollback applied
+            if ((uint32_t) (i - min) < n_seqs) {
+                cells[i].rs_n = std::min(n_seq_tokens, 1 + n_rs_seq);
+            } else {
+                uint32_t idx = 0;
+                if (n_rs_seq > 0 && !cells[i].seq_id.empty()) {
+                    const llama_seq_id seq = *cells[i].seq_id.begin(); // the sequence s_copy reads the plane of
+                    if (seq >= 0 && (size_t) seq < rs_idx.size()) {
+                        idx = rs_idx[seq];
+                    }
+                }
+                if (cells[i].src0 != i || idx != 0) {
+                    cells[i].rs_n = 1;
+                }
+            }
+
             cells[i].src = i; // avoid moving or clearing twice
         }
     }
@@ -895,6 +928,13 @@ void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_i
     if (n_rs_seq != 0) {
         set_rs_idx(seq_id, 0);
     }
+
+    // a restored state is written to plane 0 only (state_read_data), so no rollback can go behind it
+    for (auto & cell : cells) {
+        if (seq_id == -1 ? !cell.is_empty() : cell.has_seq_id(seq_id)) {
+            cell.rs_n = 1;
+        }
+    }
 }
 
 void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
diff --git a/src/llama-memory-recurrent.h b/src/llama-memory-recurrent.h
index 08489a4..63afa53 100644
--- a/src/llama-memory-recurrent.h
+++ b/src/llama-memory-recurrent.h
@@ -91,6 +91,10 @@ public:
         int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
         int32_t   tail = -1;
 
+        // snapshot planes 0 .. rs_n - 1 of this cell's rows hold the state 0 .. rs_n - 1 tokens before pos (set by
+        // find_slot); a plane the last write did not reach is stale, so a rollback onto it must be refused
+        uint32_t  rs_n = 0;
+
         std::set<llama_seq_id> seq_id;
 
         bool has_seq_id(const llama_seq_id & id) const {
```

The ec7630a patch is the same diff (offsets -4 lines) plus this hunk, 79e2e74's RC:202-205:

```diff
@@ -199,6 +199,10 @@ bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos
 
             // partial rollback via per-token snapshot index (bounded by n_rs_seq)
             if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
+                // a cell shared by several sequences cannot move back for only one of them (as at 79e2e74)
+                if (cell.seq_id.size() > 1) {
+                    return false;
+                }
                 // the filter kept no layer (e.g. an MTP draft context), so only the position moves back
                 if (is_empty()) {
                     cell.pos = p0 - 1;
```

## 4. R1 without the patch

Rule (PROVISIONAL: derived from the code cited in section 2; no llama-server was run): **a llama.cpp llama-server that
runs a speculative type with recurrent rollback snapshots (`draft-mtp`, `draft-eagle3`, `draft-dflash`, `draft-dspark`:
`n_rs_seq` = draft length > 0) must run with exactly one slot, `-np 1` given explicitly; every other configuration
(`n_rs_seq` = 0) is free.** Why: with `n_rs_seq` = 0 no rollback is ever accepted (row 1); with `n_rs_seq` > 0 the
server's rollbacks are always inside its last verify call (row 3) and the one copy path that can see a pending rollback
is the decision prefix sharing, which needs at least two slots (row 10). The slot count must be explicit because
llama-server's default is `-np` auto = 4 slots (common/arg.cpp:1406, tools/server/server.cpp:168-173). The rule says
nothing about other engines (ik_llama.cpp has its own code) or about clients of the C API (row 13: such a client must
roll a sequence back only right after a call that fed it at least r + 1 tokens, and must decode a sequence before
copying it).

The current R1 configs satisfy it trivially (no speculative type, `-np 1`): `scripts/r1/rs_rollback_rule.py` checks the
argv (and `LLAMA_ARG_SPEC_TYPE` / `LLAMA_ARG_N_PARALLEL` when the flag is absent) of every `scripts/r1/configs/*.json`
whose engine pin is `llama.cpp`; `tests/test_r1_rs_rollback_rule.py` (5 tests, stdlib) covers the rule and runs it on
the configs (output in section 5.6). With the patch applied the rule is not needed for (a) and (b) at either pin.

## 5. Runs (sandbox CPU: Intel Xeon @ 2.80GHz, 4 CPUs shared with other agents; c++ 13.3.0, Python 3.13.16, numpy 2.5.3, cmake 3.28.3)

Model and flags as N19: the N2/N6 tiny qwen4exp model (tools/tinygen/qwen4exp_tiny.py, seed 0), `-c 512 -b 512 -t 1
-fa off -ctk f32 -ctv f32`, `-np` / `-ub` / `n_rs_seq` per plan, f32 everywhere, CPU. Builds: `tools/g0` target
`dump-tensors`, `GGML_NATIVE=OFF`, four build directories (patched copy and unpatched checkout at each pin). The N20
driver runs the 9 N19 plans and the 3 N20 plans (`last_call`, `extra_cell`, `cp_pending_order`), each graph checked row
by row by the N19 tests against (A) every sequence alone and (B) the cell model, now with the guard modelled.

### 5.1 - 5.3 Final runs, both pins (the literal output of the final-runs script)

What they show (CONFIRMED by the outputs below):

* **5.1 patched build, through the CI checker: Ran 102, OK, no skips, at ec7630a and 79e2e74.** `stale`: both rollbacks
  (2 after two one-token calls, 1 after a one-token call) refused with the guard's message, no divergence; `cp_pending`:
  accepted rollback, copy, no divergence; `cp_shared_rb`: refused at both pins; `last_call`: the rollback of all 2 tokens
  of the last call refused, the exact ones (1 of 3, 3 of 4, 1 of 2) accepted; `extra_cell`: the second rollback refused;
  `cp_pending_order`: no divergence. The N19 runs `rb_uni` / `rb_str` keep every result
  (`[True, True, True, False, True, False, True]`), so no exact rollback of N19 is refused.
* **5.2 unpatched checkout, the same driver in upstream mode: Ran 102, OK** with the pin's own expectations, i.e. the
  divergences are asserted: N19's `stale` and `cp_pending` as in N19, and the three N20 runs - `last_call` (seq 0 from
  graph 2: delta-rule plane 2 was never written by the 2-token call), `extra_cell` (seq 1 from graph 3: it reads plane 1
  of row 2, which holds sequence 2's snapshot, probe below), `cp_pending_order` (the copy decoding first reads the
  pre-rollback plane 0). These are silent: no error, logits off by up to 1.169 (relative) against the sequence alone.
* **5.3 no regression:** the dumps of the plans the guard does not touch are byte-identical patched vs unpatched
  (`multi_uni`, `multi_str`, `cp_uni`, `cp_str`, `rb_uni`, `rb_str` at both pins, `cp_shared_rb` at 79e2e74, which
  already refused that rollback); only the plans whose behaviour the guard changes differ. N2 (Ran 17), N6 (Ran 54) and
  N9 glm5-next decode with its rollback run (Ran 52 + the expected MTP skip at ec7630a, Ran 58 at 79e2e74) pass on the
  patched build with their CI expectations. The unchanged N19 step on the patched build fails exactly in `stale` (its
  oracle accepts what the patch refuses: setUpClass error), `cp_pending` and, at ec7630a, `cp_shared_rb`: the intended
  changes, which the N20 step expects (5.1); the 6 other N19 runs pass there too.

```
$ bash docs/evidence/mutation-scripts/2026-10-11-n20-final-runs.sh <repo> <S>/llama-ec7630a <S>/n20 ec
repo db9da7c (7 changed files); llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d (0 changed files); copy ec7630a,  2 files changed, 48 insertions(+)
== 1 n20-qwen4exp-rs-guard (patched) through the checker
n20 patched ec rc=0 wall_s=31
mode patched: llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d, sources <S>/n20/llama-ec-guard (2 files changed, 48 insertions(+))
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
Ran 102 tests in 15.210s
OK
::notice::n20-qwen4exp-rs-guard @ llama.cpp ec7630a: Ran 102, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
== 2 the N20 driver, upstream mode (unpatched checkout) through the checker
n20 upstream ec rc=0 wall_s=30
mode upstream: llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d, sources <S>/llama-ec7630a ()
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
multi_str: 21 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
stale: 11 graphs, 13 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
last_call: 7 graphs, 11 steps, seq_rm ok: [True, True, True, True], guard refusals logged: 0
extra_cell: 5 graphs, 7 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_pending_order: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
Ran 102 tests in 14.084s
OK
::notice::n20-qwen4exp-rs-guard @ llama.cpp ec7630a: Ran 102, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
== 3 dumps patched vs upstream, per plan
cp_pending: differs (551 files)
cp_pending_order: differs (422 files)
cp_shared_rb: differs (563 files)
cp_str: identical (6858 files)
cp_uni: identical (5964 files)
extra_cell: differs (284 files)
last_call: differs (692 files)
multi_str: identical (6262 files)
multi_uni: identical (5070 files)
rb_str: identical (5438 files)
rb_uni: identical (5152 files)
stale: differs (990 files)
== 4 earlier steps on the patched build (its copy as the checkout) through the checker
n2-qwen4exp patched ec rc=0 wall_s=3
::notice::n2-qwen4exp @ llama.cpp ec7630a: Ran 17, skipped 0 as expected; the driver's checkout <S>/n20/llama-ec-guard is at ec7630a
n6-qwen4exp-decode patched ec rc=0 wall_s=33
::notice::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 54, skipped 0 as expected; the driver's checkout <S>/n20/llama-ec-guard is at ec7630a
n9-glm5next-decode patched ec rc=0 wall_s=17
::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass); the driver's checkout <S>/n20/llama-ec-guard is at ec7630a
== 5 the N19 step unchanged on the patched build (QWEN4EXP_RS_GUARD unset)
n19 on the patched build ec rc=1 wall_s=22
ERROR: setUpClass (tests.test_qwen4exp_multiseq_parity.Run_stale)
FAIL: test_attention_cells_and_no_leakage (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_attention_cells_and_no_leakage)
FAIL: test_carried_state (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_carried_state)
FAIL: test_edge_behaviour (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_edge_behaviour)
FAIL: test_gdn_and_ple_rows (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_gdn_and_ple_rows)
FAIL: test_moe_head_and_logits (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_moe_head_and_logits)
FAIL: test_qsa_pool_layout (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_qsa_pool_layout)
FAIL: test_attention_cells_and_no_leakage (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_attention_cells_and_no_leakage)
FAIL: test_carried_state (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_carried_state)
FAIL: test_edge_behaviour (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_edge_behaviour)
FAIL: test_gdn_and_ple_rows (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_gdn_and_ple_rows)
FAIL: test_graphs_rows_and_memory_positions (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_graphs_rows_and_memory_positions)
FAIL: test_moe_head_and_logits (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_moe_head_and_logits)
FAIL: test_qsa_pool_layout (tests.test_qwen4exp_multiseq_parity.Run_cp_shared_rb.test_qsa_pool_layout)
Ran 65 tests in 10.192s
FAILED (failures=13, errors=1)
```

```
$ bash docs/evidence/mutation-scripts/2026-10-11-n20-final-runs.sh <repo> <S>/llama.cpp <S>/n20 r1
repo db9da7c (7 changed files); llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (0 changed files); copy 79e2e74,  2 files changed, 44 insertions(+)
== 1 n20-qwen4exp-rs-guard (patched) through the checker
n20 patched r1 rc=0 wall_s=36
mode patched: llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, sources <S>/n20/llama-r1-guard (2 files changed, 44 insertions(+))
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
Ran 102 tests in 17.576s
OK
::notice::n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 102, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
== 2 the N20 driver, upstream mode (unpatched checkout) through the checker
n20 upstream r1 rc=0 wall_s=29
mode upstream: llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, sources <S>/llama.cpp ()
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
multi_str: 21 graphs, 15 steps, seq_rm ok: [False], guard refusals logged: 0
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True], guard refusals logged: 0
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True], guard refusals logged: 0
stale: 11 graphs, 13 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [False], guard refusals logged: 0
last_call: 7 graphs, 11 steps, seq_rm ok: [True, True, True, True], guard refusals logged: 0
extra_cell: 5 graphs, 7 steps, seq_rm ok: [True, True], guard refusals logged: 0
cp_pending_order: 6 graphs, 8 steps, seq_rm ok: [True], guard refusals logged: 0
Ran 102 tests in 13.832s
OK
::notice::n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 102, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
== 3 dumps patched vs upstream, per plan
cp_pending: differs (546 files)
cp_pending_order: differs (417 files)
cp_shared_rb: identical (1708 files)
cp_str: identical (6812 files)
cp_uni: identical (5924 files)
extra_cell: differs (282 files)
last_call: differs (686 files)
multi_str: identical (6220 files)
multi_uni: identical (5036 files)
rb_str: identical (5400 files)
rb_uni: identical (5116 files)
stale: differs (979 files)
== 4 earlier steps on the patched build (its copy as the checkout) through the checker
n2-qwen4exp patched r1 rc=0 wall_s=4
::notice::n2-qwen4exp @ llama.cpp 79e2e74: Ran 17, skipped 0 as expected; the driver's checkout <S>/n20/llama-r1-guard is at 79e2e74
n6-qwen4exp-decode patched r1 rc=0 wall_s=29
::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/n20/llama-r1-guard is at 79e2e74
n9-glm5next-decode patched r1 rc=0 wall_s=28
::notice::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 58, skipped 0 as expected; the driver's checkout <S>/n20/llama-r1-guard is at 79e2e74
== 5 the N19 step unchanged on the patched build (QWEN4EXP_RS_GUARD unset)
n19 on the patched build r1 rc=1 wall_s=33
ERROR: setUpClass (tests.test_qwen4exp_multiseq_parity.Run_stale)
FAIL: test_attention_cells_and_no_leakage (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_attention_cells_and_no_leakage)
FAIL: test_carried_state (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_carried_state)
FAIL: test_edge_behaviour (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_edge_behaviour)
FAIL: test_gdn_and_ple_rows (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_gdn_and_ple_rows)
FAIL: test_moe_head_and_logits (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_moe_head_and_logits)
FAIL: test_qsa_pool_layout (tests.test_qwen4exp_multiseq_parity.Run_cp_pending.test_qsa_pool_layout)
Ran 65 tests in 12.336s
FAILED (failures=6, errors=1)
```

Worst errors and divergences (the tables the tests print, aggregated by
docs/evidence/mutation-scripts/2026-10-11-n20-aggregate.py; the logs are the driver logs of steps 1 and 2 above): with
the patch the worst f32 error is 1.314e-05 (`final_output`, cp_uni, as in N19) and nothing diverges; without it the five
divergences.

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-aggregate.py <S>/n20/n20-ec-guard.log
664 table rows from 12 runs; worst components:
  final_output [B]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  final_output [A]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  indexer_score (own visible pools) [B]              multi_str        layer  3 graph   0 max abs 2.722e-07 rel 9.095e-06
  linear_attn_out [B]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  linear_attn_out [A]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  tie replays: ['qsa selection: tie replays = 0']
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-aggregate.py <S>/n20/n20-ec-up.log
670 table rows from 12 runs; worst components:
  final_output [B]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  final_output [A]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  indexer_score (own visible pools) [B]              multi_str        layer  3 graph   0 max abs 2.722e-07 rel 9.095e-06
  linear_attn_out [B]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  linear_attn_out [A]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  cp_pending: diverged seq 1: state off by 6.64e-01, logits vs (A) up to 1.137e+00 (graph 2)
  stale: diverged seq 0: state off by 1.02e+00, logits vs (A) up to 1.169e+00 (graph 3)
  stale: diverged seq 1: state off by 7.35e-01, logits vs (A) up to 8.415e-01 (graph 8)
  cp_pending_order: diverged seq 1: state off by 6.64e-01, logits vs (A) up to 1.137e+00 (graph 2)
  extra_cell: diverged seq 1: state off by 9.34e-01, logits vs (A) up to 1.056e+00 (graph 3)
  last_call: diverged seq 0: state off by 7.76e-01, logits vs (A) up to 6.263e-01 (graph 2)
  tie replays: ['qsa selection: tie replays = 0']
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-aggregate.py <S>/n20/n20-r1-guard.log
664 table rows from 12 runs; worst components:
  final_output [B]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  final_output [A]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  indexer_score (own visible pools) [B]              multi_str        layer  3 graph   0 max abs 2.424e-07 rel 8.099e-06
  linear_attn_out [B]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  linear_attn_out [A]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  tie replays: ['qsa selection: tie replays = 0']
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-aggregate.py <S>/n20/n20-r1-up.log
670 table rows from 12 runs; worst components:
  final_output [B]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  final_output [A]                                   cp_uni           layer  6 graph   7 max abs 2.048e-05 rel 1.314e-05
  indexer_score (own visible pools) [B]              multi_str        layer  3 graph   0 max abs 2.424e-07 rel 8.099e-06
  linear_attn_out [B]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  linear_attn_out [A]                                cp_uni           layer  6 graph   7 max abs 9.608e-06 rel 7.524e-06
  cp_pending: diverged seq 1: state off by 6.64e-01, logits vs (A) up to 1.137e+00 (graph 2)
  stale: diverged seq 0: state off by 1.02e+00, logits vs (A) up to 1.169e+00 (graph 3)
  stale: diverged seq 1: state off by 7.35e-01, logits vs (A) up to 8.415e-01 (graph 8)
  cp_pending_order: diverged seq 1: state off by 6.64e-01, logits vs (A) up to 1.137e+00 (graph 2)
  extra_cell: diverged seq 1: state off by 9.34e-01, logits vs (A) up to 1.056e+00 (graph 3)
  last_call: diverged seq 0: state off by 7.76e-01, logits vs (A) up to 6.263e-01 (graph 2)
  tie replays: ['qsa selection: tie replays = 0']
```

Which rows `extra_cell` reads (the oracle replay the dump tests check llama.cpp against; `test_carried_state` passes in
both modes, so llama.cpp read the same rows): without the patch, graph 3 has sequence 1 read (plane 1, row 2); row 2 is
sequence 1's cell only since graph 2's gather swapped it in, and its plane 1 was last written by sequence 2's 4-token
call in graph 1. With the patch the rollback is refused and sequence 1 reads (plane 0, row 2), its own state.

```
$ QWEN4EXP_MULTISEQ_DUMP=<S>/n20/w-r1-up/n20 QWEN4EXP_TINY_GGUF=<that dir>/qwen4exp.gguf LLAMA_CPP_GGUF_PY=<S>/llama.cpp/gguf-py PARITY_LLAMA_COMMIT=79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 python3 docs/evidence/mutation-scripts/2026-10-11-n20-extra-cell-probe.py .
extra_cell at r1; seq_rm results [True, True]
graph 0 call 0: seqs [0, 1, 2] x 10, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 0), []), 2: ((0, 0), [])}, diverged {0: False, 1: False, 2: False}
graph 1 call 1: seqs [0, 1, 2] x 4, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 1), []), 2: ((0, 2), [])}, diverged {0: False, 1: False, 2: False}
graph 2 call 3: seqs [0, 2] x 1, recurrent head 0 n 3, extra [(2, (2, 1), [1])], reads (plane, row) {0: ((0, 0), []), 2: ((0, 2), [])}, diverged {0: False, 2: False}
graph 3 call 5: seqs [0, 1, 2] x 1, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((1, 2), ['conv', 'ple', 'state']), 2: ((0, 1), [])}, diverged {0: False, 1: True, 2: False}
graph 4 call 6: seqs [0, 1, 2] x 1, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 1), []), 2: ((0, 2), [])}, diverged {0: False, 1: True, 2: False}
$ (the same on <S>/n20/w-r1-guard/n20 with QWEN4EXP_RS_GUARD=1)
extra_cell at r1 + guard; seq_rm results [True, False]
graph 0 call 0: seqs [0, 1, 2] x 10, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 0), []), 2: ((0, 0), [])}, diverged {0: False, 1: False, 2: False}
graph 1 call 1: seqs [0, 1, 2] x 4, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 1), []), 2: ((0, 2), [])}, diverged {0: False, 1: False, 2: False}
graph 2 call 3: seqs [0, 2] x 1, recurrent head 0 n 3, extra [(2, (2, 1), [1])], reads (plane, row) {0: ((0, 0), []), 2: ((0, 2), [])}, diverged {0: False, 2: False}
graph 3 call 5: seqs [0, 1, 2] x 1, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 2), []), 2: ((0, 1), [])}, diverged {0: False, 1: False, 2: False}
graph 4 call 6: seqs [0, 1, 2] x 1, recurrent head 0 n 3, extra [], reads (plane, row) {0: ((0, 0), []), 1: ((0, 1), []), 2: ((0, 2), [])}, diverged {0: False, 1: False, 2: False}
```

### 5.4 Mutation check, both pins

docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py. C0-C6 are real llama.cpp builds (the patched
copy with the patch removed or one hunk disabled, `dump-tensors` rebuilt, all 12 plans dumped again, tests with
`QWEN4EXP_RS_GUARD=1`); P1-P5 and T1-T4 edit copies of the oracle, the test switch and the plan expectations.
**The patch reverted (C0) fails at both pins (27 failures + 1 error at 79e2e74, 34 + 1 at ec7630a, where `cp_shared_rb` fails too).** Every hunk is caught except C5 at both pins:
the `state_read` hunk survives because no plan saves or restores state. By the code it only matters for a whole-context
restore (RC:1061-1104 sets cells without `find_slot`; a per-sequence restore goes through `find_slot` with a one-token
ubatch, RC:1028-1052, which already gives `rs_n` = 1). NOT RUN, listed in section 6.
The first run's T3 mutant was broken (its anchor left part of a comment behind, a syntax error: `rc=1 ? | ?`, no test
ran) and the first script version counted it as caught. The script now reports such a run as BROKEN; T3 was fixed
and P1-T4 rerun at both pins (second block per pin): 9/9 caught. Final count: C 6/7 per pin (C5 survives), P/T 9/9 per
pin. The C blocks come from the first script version; the fix touched only the T3 anchor and the verdict of a run without
a summary (every C run printed a summary), so the C verdicts stand. The first run's T3 lines:

```
$ grep -A2 "T3 plans" <S>/n20/mut-r1.final.log <S>/n20/mut-ec.final.log      # logs of the first run, both pins
<S>/n20/mut-r1.final.log:caught: T3 plans: cp_pending expected to diverge with the guard
<S>/n20/mut-r1.final.log-    rc=1 ? | ?
<S>/n20/mut-r1.final.log-    failing: none
--
<S>/n20/mut-ec.final.log:caught: T3 plans: cp_pending expected to diverge with the guard
<S>/n20/mut-ec.final.log-    rc=1 ? | ?
<S>/n20/mut-ec.final.log-    failing: none
```

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py <repo> <S>/llama.cpp <S>/n20/llama-r1-guard <S>/n20/build-r1-guard <S>/n20/mut-r1      # first run, C part (its P/T part had the broken T3, below)
llama.cpp 79e2e74; patched build: 14 objects compiled
baseline (patched build, no mutation): rc=0 Ran 102 tests in 13.492s | OK
caught: C0 the patch reverted (llama.cpp at the pin) (14 objects rebuilt)
    rc=1 Ran 78 tests in 15.276s | FAILED (failures=27, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_last_call.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_last_call.test_carried_state, Run_stale.test_carried_state, Run_last_call.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_last_call.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_last_call.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_last_call.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout, Run_last_call.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, GuardLog.test_refusal_messages
caught: C1 seq_rm: the plane rule dropped (rollback onto an unwritten plane accepted) (14 objects rebuilt)
    rc=1 Ran 78 tests in 17.259s | FAILED (failures=17, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, Run_last_call.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_last_call.test_carried_state, Run_stale.test_carried_state, Run_last_call.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_last_call.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_last_call.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_last_call.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_last_call.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, GuardLog.test_refusal_messages
caught: C2 seq_cp: the pending rollback not handed to the copy (14 objects rebuilt)
    rc=1 Ran 85 tests in 16.336s | FAILED (failures=10)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout
caught: C3 find_slot: an extra cell keeps its planes after the extra-row copy (14 objects rebuilt)
    rc=1 Ran 78 tests in 22.518s | FAILED (failures=1, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, GuardLog.test_refusal_messages
caught: C4 find_slot: a ubatch cell gets 1 + n_rs_seq planes whatever its tokens (14 objects rebuilt)
    rc=1 Ran 85 tests in 11.266s | FAILED (failures=16)
    failing: Run_last_call.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_last_call.test_carried_state, Run_stale.test_carried_state, Run_last_call.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_last_call.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_last_call.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_last_call.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_last_call.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, GuardLog.test_refusal_messages
SURVIVED: C5 state_read: a restored cell keeps the planes it had (14 objects rebuilt)
    rc=0 Ran 85 tests in 13.046s | OK
    failing: none
caught: C6 seq_rm: a rollback of a shared cell accepted (the ec7630a hunk; upstream code at 79e2e74) (14 objects rebuilt)
    rc=1 Ran 85 tests in 11.488s | FAILED (failures=7)
    failing: Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_carried_state, Run_cp_shared_rb.test_edge_behaviour, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_shared_rb.test_qsa_pool_layout
restored the patched copy: 14 objects rebuilt; diff vs patch: same
```

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py <repo> <S>/llama.cpp <S>/n20/llama-r1-guard <S>/n20/build-r1-guard <S>/n20/mutP-r1 P
llama.cpp 79e2e74; patched build: 14 objects compiled
baseline (patched build, no mutation): rc=0 Ran 102 tests in 14.864s | OK
restored the patched copy: 14 objects rebuilt; diff vs patch: same
caught: P1 oracle: the plane rule dropped
    rc=1 Ran 81 tests in 12.908s | FAILED (failures=4, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, test_qwen4exp_rs_guard_parity.setUpClass, GuardRules.test_a_rollback_needs_a_plane_the_last_ubatch_wrote, GuardRules.test_an_extra_cell_that_did_not_move_keeps_its_planes, GuardOracle.test_stale_rollbacks_are_refused_and_rows_stay_exact, GuardRules.test_the_extra_row_copy_leaves_plane_0_only
caught: P2 oracle: seq_cp does not hand the pending rollback over
    rc=1 Ran 102 tests in 15.146s | FAILED (failures=15)
    failing: GuardOracle.test_a_copy_made_while_a_rollback_is_pending_continues_the_kept_prefix, Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_pending_order.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout, GuardRules.test_seq_cp_hands_the_pending_rollback_to_the_copy, GuardRules.test_seq_cp_into_a_pending_rollback
caught: P3 oracle: an extra cell keeps its planes
    rc=1 Ran 95 tests in 18.550s | FAILED (failures=2, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, GuardRules.test_an_extra_cell_that_did_not_move_keeps_its_planes, GuardRules.test_the_extra_row_copy_leaves_plane_0_only
caught: P4 oracle: a ubatch cell gets 1 + n_rs_seq planes
    rc=1 Ran 88 tests in 15.306s | FAILED (failures=2, errors=2)
    failing: test_qwen4exp_multiseq_parity.setUpClass, test_qwen4exp_rs_guard_parity.setUpClass, GuardRules.test_a_rollback_needs_a_plane_the_last_ubatch_wrote, GuardOracle.test_stale_rollbacks_are_refused_and_rows_stay_exact
caught: P5 tests: QWEN4EXP_RS_GUARD ignored (the pin's own behaviour expected)
    rc=1 Ran 81 tests in 12.210s | FAILED (failures=15, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, test_qwen4exp_rs_guard_parity.setUpClass, Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_pending_order.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout, GuardLog.test_refusal_messages
caught: T1 plans: stale expected to accept both rollbacks with the guard
    rc=1 Ran 102 tests in 16.306s | FAILED (failures=1)
    failing: Run_stale.test_edge_behaviour
caught: T2 plans: stale expected to log no guard refusal
    rc=1 Ran 102 tests in 14.876s | FAILED (failures=1)
    failing: GuardLog.test_refusal_messages
caught: T3 plans: cp_pending expected to diverge with the guard
    rc=1 Ran 102 tests in 13.337s | FAILED (failures=1)
    failing: Run_cp_pending.test_edge_behaviour
caught: T4 plans: extra_cell's second rollback expected accepted with the guard
    rc=1 Ran 102 tests in 14.055s | FAILED (failures=1)
    failing: Run_extra_cell.test_edge_behaviour
caught 9/9
```

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py <repo> <S>/llama-ec7630a <S>/n20/llama-ec-guard <S>/n20/build-ec-guard <S>/n20/mut-ec      # first run, C part (its P/T part had the broken T3, below)
llama.cpp ec7630a; patched build: 13 objects compiled
baseline (patched build, no mutation): rc=0 Ran 102 tests in 15.707s | OK
caught: C0 the patch reverted (llama.cpp at the pin) (13 objects rebuilt)
    rc=1 Ran 78 tests in 23.089s | FAILED (failures=34, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_last_call.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_last_call.test_carried_state, Run_stale.test_carried_state, Run_cp_shared_rb.test_edge_behaviour, Run_last_call.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_last_call.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_last_call.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_last_call.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_last_call.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, GuardLog.test_refusal_messages
caught: C1 seq_rm: the plane rule dropped (rollback onto an unwritten plane accepted) (13 objects rebuilt)
    rc=1 Ran 78 tests in 17.127s | FAILED (failures=17, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, Run_last_call.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_last_call.test_carried_state, Run_stale.test_carried_state, Run_last_call.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_last_call.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_last_call.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_last_call.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_last_call.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, GuardLog.test_refusal_messages
caught: C2 seq_cp: the pending rollback not handed to the copy (13 objects rebuilt)
    rc=1 Ran 85 tests in 22.170s | FAILED (failures=10)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout
caught: C3 find_slot: an extra cell keeps its planes after the extra-row copy (13 objects rebuilt)
    rc=1 Ran 78 tests in 13.305s | FAILED (failures=1, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, GuardLog.test_refusal_messages
caught: C4 find_slot: a ubatch cell gets 1 + n_rs_seq planes whatever its tokens (13 objects rebuilt)
    rc=1 Ran 85 tests in 12.847s | FAILED (failures=16)
    failing: Run_last_call.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_last_call.test_carried_state, Run_stale.test_carried_state, Run_last_call.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_last_call.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_last_call.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_last_call.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_last_call.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, GuardLog.test_refusal_messages
SURVIVED: C5 state_read: a restored cell keeps the planes it had (13 objects rebuilt)
    rc=0 Ran 85 tests in 12.002s | OK
    failing: none
caught: C6 seq_rm: a rollback of a shared cell accepted (the ec7630a hunk; upstream code at 79e2e74) (13 objects rebuilt)
    rc=1 Ran 85 tests in 11.358s | FAILED (failures=7)
    failing: Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_carried_state, Run_cp_shared_rb.test_edge_behaviour, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_shared_rb.test_qsa_pool_layout
restored the patched copy: 13 objects rebuilt; diff vs patch: same
```

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py <repo> <S>/llama-ec7630a <S>/n20/llama-ec-guard <S>/n20/build-ec-guard <S>/n20/mutP-ec P
llama.cpp ec7630a; patched build: 13 objects compiled
baseline (patched build, no mutation): rc=0 Ran 102 tests in 19.954s | OK
restored the patched copy: 13 objects rebuilt; diff vs patch: same
caught: P1 oracle: the plane rule dropped
    rc=1 Ran 81 tests in 11.048s | FAILED (failures=4, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, test_qwen4exp_rs_guard_parity.setUpClass, GuardRules.test_a_rollback_needs_a_plane_the_last_ubatch_wrote, GuardRules.test_an_extra_cell_that_did_not_move_keeps_its_planes, GuardOracle.test_stale_rollbacks_are_refused_and_rows_stay_exact, GuardRules.test_the_extra_row_copy_leaves_plane_0_only
caught: P2 oracle: seq_cp does not hand the pending rollback over
    rc=1 Ran 102 tests in 13.317s | FAILED (failures=15)
    failing: GuardOracle.test_a_copy_made_while_a_rollback_is_pending_continues_the_kept_prefix, Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_pending_order.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout, GuardRules.test_seq_cp_hands_the_pending_rollback_to_the_copy, GuardRules.test_seq_cp_into_a_pending_rollback
caught: P3 oracle: an extra cell keeps its planes
    rc=1 Ran 95 tests in 12.229s | FAILED (failures=2, errors=1)
    failing: test_qwen4exp_rs_guard_parity.setUpClass, GuardRules.test_an_extra_cell_that_did_not_move_keeps_its_planes, GuardRules.test_the_extra_row_copy_leaves_plane_0_only
caught: P4 oracle: a ubatch cell gets 1 + n_rs_seq planes
    rc=1 Ran 88 tests in 11.299s | FAILED (failures=2, errors=2)
    failing: test_qwen4exp_multiseq_parity.setUpClass, test_qwen4exp_rs_guard_parity.setUpClass, GuardRules.test_a_rollback_needs_a_plane_the_last_ubatch_wrote, GuardOracle.test_stale_rollbacks_are_refused_and_rows_stay_exact
caught: P5 tests: QWEN4EXP_RS_GUARD ignored (the pin's own behaviour expected)
    rc=1 Ran 81 tests in 10.335s | FAILED (failures=22, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, test_qwen4exp_rs_guard_parity.setUpClass, Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_pending_order.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_pending_order.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_pending_order.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_pending_order.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_cp_pending_order.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_pending_order.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, GuardLog.test_refusal_messages
caught: T1 plans: stale expected to accept both rollbacks with the guard
    rc=1 Ran 102 tests in 10.968s | FAILED (failures=1)
    failing: Run_stale.test_edge_behaviour
caught: T2 plans: stale expected to log no guard refusal
    rc=1 Ran 102 tests in 12.587s | FAILED (failures=1)
    failing: GuardLog.test_refusal_messages
caught: T3 plans: cp_pending expected to diverge with the guard
    rc=1 Ran 102 tests in 14.331s | FAILED (failures=1)
    failing: Run_cp_pending.test_edge_behaviour
caught: T4 plans: extra_cell's second rollback expected accepted with the guard
    rc=1 Ran 102 tests in 13.604s | FAILED (failures=1)
    failing: Run_extra_cell.test_edge_behaviour
caught 9/9
```

### 5.5 apply.sh

Fresh copy, reuse, a tampered copy refused, the ec7630a copy equal to the build copy, the source checkout untouched; then
from a shallow fetch of the pin like parity-r1-pin.yml's checkout (run before the redundant `rs_n` swap was dropped from
the patch, hence 45 / 49 insertions here against 44 / 48 in the final patch; the script was not changed since):

```
$ cd <S>/n20; rm -rf t-apply; mkdir t-apply; bash <repo>/patches/llama.cpp/apply.sh <S>/llama.cpp t-apply/r1; echo rc=$?; bash <repo>/patches/llama.cpp/apply.sh <S>/llama.cpp t-apply/r1; echo rc=$?; diff <(git -C t-apply/r1 diff) <(git -C llama-r1-guard diff) && echo SAME-AS-BUILD-COPY; echo x >> t-apply/r1/README.md; bash <repo>/patches/llama.cpp/apply.sh <S>/llama.cpp t-apply/r1; echo rc=$?; bash <repo>/patches/llama.cpp/apply.sh <S>/llama-ec7630a t-apply/ec | tail -2; diff <(git -C t-apply/ec diff) <(git -C llama-ec-guard diff) && echo EC-SAME; git -C <S>/llama.cpp status --short | wc -l; rm -rf t-apply
apply.sh: applied 0001-recurrent-rollback-guard.patch
 src/llama-memory-recurrent.cpp | 41 +++++++++++++++++++++++++++++++++++++++++
 src/llama-memory-recurrent.h   |  4 ++++
 2 files changed, 45 insertions(+)
rc=0
apply.sh: reusing t-apply/r1 (79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 + 1 patch(es))
rc=0
SAME-AS-BUILD-COPY
apply.sh: t-apply/r1 exists and is not 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 with exactly the patches of <repo>/patches/llama.cpp/79e2e74
rc=1
 src/llama-memory-recurrent.h   |  4 ++++
 2 files changed, 49 insertions(+)
EC-SAME
0
$ cd <S>/n20 && rm -rf t-shallow && mkdir t-shallow && SHA=79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 && git init -q t-shallow/src && git -C t-shallow/src -c protocol.file.allow=always fetch -q --depth 1 file://<S>/llama.cpp "$SHA" 2>&1 | tail -2; git -C t-shallow/src checkout -q FETCH_HEAD 2>&1 | tail -1; git -C t-shallow/src rev-parse HEAD; cat t-shallow/src/.git/shallow; bash <repo>/patches/llama.cpp/apply.sh t-shallow/src t-shallow/copy 2>&1 | tail -3; git -C t-shallow/copy rev-parse HEAD; git -C <S>/llama.cpp status --short | wc -l; rm -rf t-shallow
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
 src/llama-memory-recurrent.cpp | 40 ++++++++++++++++++++++++++++++++++++++++
 src/llama-memory-recurrent.h   |  4 ++++
 2 files changed, 44 insertions(+)
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
0
```

The patch files apply to the pristine pins (after the last change of the patch):

```
$ for p in ec7630a:llama-ec7630a 79e2e74:llama.cpp; do d=${p%%:*}; git -C <S>/${p##*:} apply --check <repo>/patches/llama.cpp/$d/0001-recurrent-rollback-guard.patch && echo "$d ok"; done
ec7630a ok
79e2e74 ok
```

### 5.6 Unit suites, the R1 rule, the workflows

```
$ LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests      # end of the log
rc=0
----------------------------------------------------------------------
Ran 1133 tests in 177.483s

OK (skipped=576)
$ python3 -m unittest tests.test_check_unittest tests.test_hooks      # end of the log
rc=0
Ran 54 tests in 19.265s

OK
$ python3 -m unittest tests.test_qwen4exp_rs_guard_ref tests.test_r1_rs_rollback_rule tests.test_qwen4exp_multiseq_ref      # end of the log
rc=0
Ran 22 tests in 2.505s

OK
$ python3 scripts/r1/rs_rollback_rule.py scripts/r1/configs/*.json
scripts/r1/configs/ik-llamacpp-server.json: not covered (engine pin 'ik_llama.cpp')
scripts/r1/configs/llamacpp-server-moecache.json: ok (spec types none, slots 1)
scripts/r1/configs/llamacpp-server.json: ok (spec types none, slots 1)
scripts/r1/configs/mock.json: not covered (engine pin None)
scripts/r1/configs/strata-serve-qwen38fn.json: not covered (engine pin 'strata')
rc=0
$ python3 - (yaml.safe_load of both workflows, the N20 steps)
.github/workflows/ci.yml [cpu-parity] python3 tools/ci/check_unittest.py --step n20-qwen4exp-rs-guard --checkout ../llama.cpp -- tools/tinygen/run_parity_qwen4exp_rs_guard.sh ../llama.cpp ../build-tools-rs-guard
.github/workflows/parity-r1-pin.yml [parity] python3 tools/ci/check_unittest.py --step n20-qwen4exp-rs-guard --checkout ../llama.cpp -- tools/tinygen/run_parity_qwen4exp_rs_guard.sh ../llama.cpp ../build-tools-rs-guard
```

## 6. Not covered (NOT RUN / UNKNOWN)

| what | status | why / what would close it |
|---|---|---|
| a llama-server run of the decision shared-prefix path (row 10) with MTP | NOT RUN | needs a server build with a qwen4exp model that has an MTP head and a client that sends `/decision` with a prefix equal to the slot's cache; the path is CONFIRMED by reading only |
| the guard on a GPU backend, the real Qwen3.8-Flash-Next or GLM-5.3 GGUF | NOT RUN | no GPU in the sandbox; the patch is backend-independent host code (RC), the planes are written by the graph |
| whole-context state save / restore after the patch (mutant C5) | NOT RUN | the dump tool has no state save / load step; per-sequence restores go through `find_slot` (RC:1028-1052) |
| the rule "plane j valid iff j < the ubatch's tokens" for the other archs with rollback (kimi-k3, qwen35, deepseek4, nemotron-h, lfm2, bailingmoe3) | PROVISIONAL | only qwen4exp (all runs) and glm5-next (N9's rollback run on the patched build, 5.3) were run |
| the hazard of a sequence that loses its cell with a pending rollback (`seq_keep`, `seq_rm(s, 0, p1)`), section 2 | PROVISIONAL, not fixed | unreachable through the server for qwen4exp; would need a plan step for `seq_keep` |
| a copy into a sequence with a pending rollback (the patch clears that index) | oracle unit test only | the KV `seq_cp` keeps the destination's own cells, so no dump plan isolates it; real callers clear the destination first |
| examples / C API callers (row 13) | UNKNOWN | not read in full |
| the N20 CI step in GitHub Actions | NOT RUN | the branch is not pushed by this task; locally through tools/ci/check_unittest.py (5.1); the clone of ci.yml's blob-filtered checkout by apply.sh was not tried (a shallow fetch was, 5.5) |
| ik_llama.cpp (R1's second GLM engine) | NOT RUN | other code; the rule of section 4 does not cover it |
| the N19 mutation script on this tree | NOT RUN | its T2 anchor (`stale`'s diverge dict) changed here (a `"guard": {}` key), so it stops at its anchor assert, as its README says for later code |

## 7. Pin check and git

```
$ for L in llama-ec7630a llama.cpp; do (cd <S>/$L && git rev-parse HEAD && git status --short | wc -l); done
ec7630a640789c393694fb194f1bbbf0369fc62d
0
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
0
$ for p in ec r1; do git -C <S>/n20/llama-$p-guard rev-parse HEAD; git -C <S>/n20/llama-$p-guard diff --stat | tail -n 1; done
ec7630a640789c393694fb194f1bbbf0369fc62d
 2 files changed, 48 insertions(+)
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
 2 files changed, 44 insertions(+)
$ git log --oneline -n 5      # the code commits the runs used (the final runs print "repo db9da7c")
db9da7c [engine-integrator] scripts/r1: rs_rollback_rule.py, the R1 rule without the patch (n_rs_seq > 0 only with -np 1), tests
aad28b8 [engine-integrator] ci: N20 step n20-qwen4exp-rs-guard in cpu-parity and parity (R1 pin), expectations Ran 102
097bc0f [engine-integrator] ref, tests, tinygen: the guard in the oracle, three N20 plans, driver, guard switch for the N19 tests
9e5f3b5 [engine-integrator] patches/llama.cpp: recurrent rollback guard for 79e2e74 and ec7630a, apply.sh applies it to a copy
e71b241 [parity-qa] evidence: N19 "Merge with N17" - every dump step at both pins on the merged dump tool, byte identity vs 0b0eace, n16compat, unit suites
```

## 8. Source lines

Every line range cited above, printed with true line numbers from both pinned checkouts:
`docs/evidence/2026-10-11-n20-source-lines.txt` (literal output of
`bash docs/evidence/mutation-scripts/2026-10-11-n20-source-lines.sh <S>/llama.cpp <S>/llama-ec7630a`, 797 lines; both
checkouts at their pins with 0 changed files, first lines of that file).
