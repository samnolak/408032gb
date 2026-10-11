# N19: qwen4exp several sequences in one context, seq_cp, seq_rm and rollback edges, CPU parity, 2026-10-11 (sandbox CPU, no GPU)

Task N19 of the hardware-free track: close gap 10 of docs/reassessment/CORRECTNESS.md (several sequences, cells shared
between sequences (`cache_safe` = false), a `seq_rm` that leaves a stale complete pool, a rollback longer than
`n_rs_seq`, a rollback inside a multi-ubatch call; N6's "Not covered" rows,
docs/evidence/2026-10-10-qwen4exp-decode-parity.md:436-442). Branch `claude/n19-qwen4exp-multiseq`, cut from
`claude/n16-mimo-decode-parity` at `0b0eace`. Role: parity-qa. `<S>` below stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution.

References: llama.cpp `ec7630a640789c393694fb194f1bbbf0369fc62d` (AGENTS.md pin) and
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (R1 engine pin, tools/engines/pins.json); CPU builds of `tools/g0` target
`dump-tensors` from this worktree in `<S>/n19/build-ec`, `<S>/n19/build-r1` (`GGML_NATIVE=OFF`). Abbreviations (all
@ec7630a unless "@79e2e74" is written; every cited line was printed from the pinned checkout before citing):
"G" = src/models/qwen4exp.cpp, "HI" = src/llama-memory-hybrid-idx.cpp, "RC" = src/llama-memory-recurrent.cpp,
"KV" = src/llama-kv-cache.cpp, "B." = src/llama-batch.cpp.

| piece | file |
|---|---|
| oracle | ref/qwen4exp.py, appended section "several sequences in one context" (nothing above it changed except `import copy`): `split_equal` (replica of B.542-711), cell models `KVCells` (one stream or one per sequence; seq_cp sharing or cross-stream copy), `RecurrentCells` (RC find_slot, s_copy, seq_rm / seq_cp with the refusal rules of both pins), `KPoolLayout` (HI kpool_layout_update / kpool_build_state of both pins), `MultiSeqContext` (every sequence through `forward_chunk` with its own cache; the recurrent rows it reads come from modelled snapshot planes tagged with the token prefix they hold) |
| dump parity tests | tests/test_qwen4exp_multiseq_parity.py: 7 tests per run, 9 runs = 63; skip without `QWEN4EXP_MULTISEQ_DUMP`, `QWEN4EXP_TINY_GGUF`, `LLAMA_CPP_GGUF_PY` |
| property tests | tests/test_qwen4exp_multiseq_ref.py: 9 tests, numpy only (also in the CI `unit tests` step) |
| plans | tools/tinygen/qwen4exp_multiseq_plans.py: the nine runs, their tokens, and the behaviour each run must show at each pin (refusals, divergence) |
| driver | tools/tinygen/run_parity_qwen4exp_multiseq.sh (builds dump-tensors, writes the N2 tiny model, runs the nine plans, runs both test modules; `QWEN4EXP_MULTISEQ_WORK` keeps the artifacts) |
| dump tool | tools/dump/dump_tensors.cpp: several-sequence `DUMP_CALLS` steps (parts `<n>@<s>[\|<s2>][:<p>]` joined by `+`, `r<n>@<s>`, `c<a>:<b>`, `x<s>`), `DUMP_KV_UNIFIED`; every step's seq_rm result and `llama_memory_seq_pos_min/_max` of every sequence recorded; plans without these steps byte-identical (below) |
| generator | tools/tinygen/qwen4exp_tiny.py unchanged |
| CI | step `n19-qwen4exp-multiseq` in `.github/workflows/ci.yml` job `cpu-parity` (ec7630a) and in `.github/workflows/parity-r1-pin.yml` job `parity` (R1 pin), both through `tools/ci/check_unittest.py`; `tools/ci/parity_expectations.json`: Ran 72, no skips, at both commits |
| mutation check, probes, byte identity | docs/evidence/mutation-scripts/2026-10-11-n19-qwen4exp-multiseq-mutations.py (24 mutants), 2026-10-11-n19-probes.sh, 2026-10-11-n19-byteid.sh |

## What llama.cpp does (CONFIRMED from the source; the runs below confirm every item that has a run)

* **Memory.** A qwen4exp context holds `llama_memory_hybrid_idx` (src/llama-model.cpp:2745-2763): an attention KV
  cache of `n_ctx_seq` cells per stream, one stream with `kv_unified`, else one per sequence (KV:84, 148-155); an
  indexer cache that takes the attention cache's slots (HI:128-132); a recurrent cache of `max(1, n_seq_max)` cells
  (src/llama-model.cpp:2756) with `1 + n_rs_seq` rows per cell (RC:101). A call is split by `split_equal` with
  `sequential` = more than one stream and the last `n_rs_seq + 1` tokens of every sequence kept in one ubatch when
  `n_rs_seq > 0` (HI:84-112, B.637-701 "[TAG_RECURRENT_ROLLBACK_SPLITS]"); one stream per sequence also needs the
  sequences of a ubatch to be consecutive ascending ids (B.569-572).
* **Recurrent rows.** A ubatch's sequences are gathered into cells head .. head + n_seqs - 1 (RC:639-662); each reads
  the row `s_copy` names, plane `rs_idx[first sequence of the cell]` of cell `src0`, and that read resets the pending
  rollback (RC:1364-1381), also for the cells between the ubatch's cells (`n > n_seqs`, the "extra" rows copied to
  plane 0, src/llama-graph.cpp:3546-3551). After the graph, plane j of the delta-rule state holds the state j tokens
  before the ubatch's end only for j < min(n_seq_tokens, 1 + n_rs_seq); "older slots are caller-owned"
  (ggml/src/ggml-cpu/ops.cpp:11184-11185, src/models/delta-net-base.cpp:563-603). The GDN conv history and the PLE
  history are written to every plane, the start clamped at the ubatch's first row (G:1324-1345).
* **seq_cp** (whole sequence). Recurrent: the destination drops its cell and shares the source's tail cell; nothing is
  copied and `rs_idx` is not touched (RC:261-296, the same at both pins); the next ubatch moves a sequence that shares
  its cell to an empty one (copy on write, RC:600-637). KV: one stream - the cells gain the destination (KV:463-492);
  two streams - the destination stream is reset, takes the source's cells at the same indices and its head, the
  buffers are copied at the next update (KV:494-541, 827-855). k-pool at ec7630a: a cell held by two sequences sets
  `shared`, and any shared cell makes `cache_safe` false (HI:463-473): every pool is re-pooled every ubatch and
  nothing is written to the pooled slots (HI:728-733, G:793-803); both source and destination are staled (HI:206-215).
  At 79e2e74 there is no sharing state: only the destination is staled and a pool is pooled once per rep cell, the
  copies read that one row (HI:200-212@79e2e74, HI:653-662@79e2e74); a partial seq_cp is an assertion
  (HI:202@79e2e74).
* **seq_rm.** The recurrent cache decides first and can refuse; a refusal changes nothing (HI:185-189). A partial cut
  (0 < p0 <= last position < p1) is a rollback by `pos - p0 + 1` and is accepted iff 1 <= rollback <= n_rs_seq and no
  rollback is pending (RC:200-215): with n_rs_seq = 0 every partial cut is refused; a second rollback before a decode
  is refused. 79e2e74 also refuses it when another sequence shares the cell (RC:202-205@79e2e74). Acceptance does not
  check that the needed snapshot exists (see finding 5). A whole-sequence removal (p0 < 0) is always accepted; at
  ec7630a it stales every sequence's pools while some cell is shared (HI:196-200). A pool can only be "stale and
  complete" after a tail cut if the cut leaves complete pools that the edit regrouped: with tail cuts that happens when
  the end of sharing restales every pool (ec7630a) and for a copy's destination (both pins); both are runs below.
* **Callers.** `common_memory::seq_rm` aborts on a refusal (common/common.cpp:1536-1541@ec7630a, :1601-1606@79e2e74);
  the server restores a checkpoint instead of rolling back when the rejected draft is longer than n_rs_seq
  (tools/server/server-context.cpp:3926-3928@ec7630a) and uses seq_cp only for whole sequences (`copy_state_to`,
  :692-703). Tokens that belong to several sequences at once hit
  `GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences")`
  (G:1248, G:1246@79e2e74) with one KV stream, and are refused by the split with one stream per sequence (B.543-547).

## Runs

Model: the N2/N6 tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0: GDN layers 0-2 and 4-6, PLE on layer 1, QSA on
layer 3 with kpool 4 and 2 pools + tail, dense attention on layer 7). Flags as N6: `-c 512 -b 512 -t 1 -fa off -ctk f32
-ctv f32`, plus `-np` and `-ub` per run. Token ids per sequence from `base_tokens(s)` (EOS at 3-4 positions each),
rejected draft tokens shifted by 17 (tools/tinygen/qwen4exp_multiseq_plans.py). Graph counts are the same at both pins
(driver output below).

| run | -np, KV | n_rs_seq, -ub | plan | what it adds | graphs |
|---|---|---|---|---|---|
| multi_uni | 3, one stream | 0, 512 | `16@0+12@1+20@2`, 8 x `1@0+1@1+1@2`, 2 x `1@0+1@2`, `2@1+1@0`, `r1@2`, 2 x `1@2+1@0+1@1` | an unequal prefill in one call (ubatches 3 x 12, 2 x 4, 1 x 4), interleaved one-token decode, an idle sequence, batch orders 1,0 and 2,0,1, a partial seq_rm with n_rs_seq 0 | 17 |
| multi_str | 3, one per sequence | 0, 512 | as multi_uni | the sequential split (sets of a ubatch consecutive ascending) | 21 |
| cp_uni | 3, one stream | 0, 512 | `22@0`, `c0:1`, 6 x `1@0+1@1`, `c0:2`, 4 x three, `x1`, 2 x `1@0+1@2`, `x2`, 6 x `1@0`, `c0:1`, `c0:2`, `3@1+3@0+3@2` | shared cells, continuing copies, dropping copies, two copies before one decode | 20 |
| cp_str | 3, one per sequence | 0, 512 | as cp_uni | cross-stream copies (buffer copy at the next update) | 23 |
| rb_uni | 3, one stream | 3, 8 | `10@0+10@1+10@2`, drafts of 4 (2/3/1 wrong), `r2@0`, `r3@1`, `r1@2`, ..., `4@2` (4 wrong), `r4@2` (refused), `r3@2`, `r1@2` (refused, pending), ..., `10@0` (3 wrong; 6 + 4), `r3@0`, ... | rollbacks of 1-3 from the last ubatch, a rollback > n_rs_seq, a pending rollback, a rollback inside a two-ubatch call, a pending rollback applied through the extra rows | 18 |
| rb_str | 3, one per sequence | 3, 8 | as rb_uni | the same with per-sequence streams | 19 |
| stale | 2, one stream | 3, 512 | `24@0`, `1@0`, `1@0`, `r2@0`, 3 x `1@0`, `20@1`, `1@1`, `r1@1`, 3 x `1@1` | rollbacks deeper than the last ubatch of the sequence | 11 |
| cp_pending | 2, one stream | 3, 512 | `20@0`, `4@0`, `r2@0`, `c0:1`, 4 x `1@0+1@1` | seq_cp while a rollback is pending | 6 |
| cp_shared_rb | 2, one stream | 3, 512 | `20@0`, `4@0`, `c0:1`, `r2@0`, 3 x `1@0+1@1`, `1@1+1@0` | a rollback of a recurrent cell two sequences share | 6 |

## Method

`DUMP_PER_GRAPH=1` writes every graph (one per ubatch) into `g<NNN>/`; `manifest.json` lists every llama_process call
with its parts, return code and graphs, every seq_rm / seq_cp with its result, and `llama_memory_seq_pos_min/_max` of
every sequence after every step. Graph rows are in ubatch order (sequence sets concatenated, `Q.split_equal`);
logits.bin in batch order. Every graph is compared row by row with

* (A) per sequence alone: `Q.forward(final tokens of that sequence)` at the row's position, one "life" per sequence id
  (a seq_cp into it or dropping it starts a new one); only rows not rolled back later, of sequences that have not
  diverged;
* (B) `Q.MultiSeqContext` replaying the same calls with cell models of the KV cache, the recurrent cache and the k-pool
  layout. A sequence "diverges" when it reads a recurrent row whose token prefix differs from its KV history; (B)
  then computes with that row, (A) cannot.

Checked per graph: rows and token ids (`hc_init` exact), graph counts per call, seq_rm results and the memory's
`seq_pos_min/_max` after every step (equal to the cell models'); the recurrent state each sequence read
(`conv_state_at` GDN and PLE, `state_predelta`) and wrote (`new_state`, n_rs_seq 0); every GDN, PLE, attention, QSA and
MoE component of N6 and the head; **cross-sequence leakage**: `kq_soft_max` of both attention layers must be exactly 0
on every cell that is not a visible cell of the row's own sequence (QSA: the selected cells; dense: positions <= the
row's) and equal (B)'s softmax there, `indexer_score` must be finite exactly on the row's own sequence's visible pools,
the QSA selection (`kq_mask_qsa`, cell indices) must map to positions of the row's own sequence; the k-pool lists
(`indexer_pool_k_new` with its padded length and pin-dependent padding cells, `indexer_k` = every pool of every
sequence in layout order). `test_edge_behaviour` asserts what each run is for at the pin (refusals, which sequences
diverge from which call, that (A) is then off, the extra-row copy, the sharing / dedupe pattern, the positions after
the shared-cell rollback).

Tolerances unchanged from N2/N6: normwise relative error < 1e-4 for f32 paths; exact for the gathers, the zero state of
a new sequence and the masked attention probabilities; MoE ids as sets; QSA selections equal or a validated tie
(0 tie replays occurred).

## Findings (CONFIRMED by the literal outputs below unless labelled)

1. **No llama.cpp-vs-oracle mismatch at either pin.** The driver through the CI checker: Ran 72, OK, at ec7630a and at
   79e2e74. Worst f32 error over all runs 1.314e-05 (`final_output`, the gated norm of the delta-rule output, layer 6,
   cp_uni graph 7), about 7.6x inside 1e-4; logits 2.124e-06. (B) also matches llama.cpp on every row of the
   diverged sequences of findings 5 and 6 (stale run logits 8.703e-07). Per-component table below.
2. **Several sequences in one context compute what each computes alone**, with one KV stream and with one per
   sequence: every row of a sequence that has not diverged equals (A); `kq_soft_max` is exactly 0 on every other sequence's cell (and on every masked or
   unselected cell) in all graphs of all runs; `indexer_score` is finite only on the row's own pools; every QSA
   selection maps to the row's own positions; each sequence reads its own recurrent state. One stream: ubatches
   mix sequences in batch order (3 x 12 rows in one graph; orders 1,0 and 2,0,1); one stream per sequence: the
   sequential split cuts the same calls into more graphs (21 against 17).
3. **seq_cp then both continue** equals the copied history continued alone, at both pins, one stream or two. At
   ec7630a with one stream every graph while two sequences share cells re-pools every pool and caches none
   (cp_uni graphs 1-12 and 19: `cache_safe` false, all pools new); dropping the last sequence that shares cells stales
   every pool (HI:196-200): graph 13 re-pools all 8 pools of sequence 0 and writes them, the next graphs read them back
   from the cache (1 or 0 new of 9-10). This is the "seq_rm that leaves complete pools stale" case: pools 5-7 of
   sequence 0 (positions 20-31) were completed while its cells were shared and never written to their slots, so had
   llama.cpp not re-pooled, graph 13 on would read slots that were never written; it does re-pool, and the values
   equal (B). At 79e2e74 a pool
   is pooled once per rep cell: after two back-to-back copies, 10 of the 30 pools of graph 19 are pooled (the three
   sequences share 10 reps). Rep dedupe is otherwise unreachable for qwen4exp (coupled sequence ids abort, finding 8).
4. **Rollbacks that llama.cpp supports restore exactly**: rollbacks of 1-3 tokens from the last ubatch of a sequence
   (several sequences in one context); a rollback of 3 after a 10-token call split 6 + 4 (the split keeps the last
   n_rs_seq + 1 = 4 together, so the rollback reads a snapshot of the last ubatch); a pending rollback of a sequence
   whose recurrent cell lies between the cells of an ubatch it is not in, applied by the extra-row copy (rb_uni
   graph 7: cell 2 of sequence 1 takes plane 3 of cell 1, the cells having been swapped by the gather). A rollback of
   4 > n_rs_seq and a second rollback before a decode return false and change nothing (the next graphs equal the
   un-rolled sequence); with n_rs_seq = 0 every partial seq_rm returns false. A caller that ignores the false and
   decodes at the position the rollback would have left is refused by llama_process ("inconsistent sequence
   positions", probe `refused_then_decode`, both pins).
5. **A rollback deeper than the sequence's last ubatch is accepted and silently changes the results (both pins).**
   seq_rm checks only the rollback length and the pending flag (RC:200-215), but the delta-rule snapshot plane r
   exists only if the last ubatch of that sequence had more than r tokens (ops.cpp:11184-11185); the conv / PLE planes
   r > n hold the history before that ubatch (G:1324-1345). Run `stale`: after `24@0`, `1@0`, `1@0`, `r2@0` returns
   true; graph 3 reads plane 2 of cell 0, whose GDN conv and PLE histories are those of the 25-token prefix (one
   rejected token included) and whose delta-rule state is the one the 24-token call left 2 tokens before its end;
   after `20@1`, `1@1`, `r1@1` returns true and graph 8 reads delta-rule plane 1 (stale: written by the 20-token call)
   with the right conv history. The state read is off from the kept prefix's by 1.02 and 0.735 (relative), the logits
   of every later row by up to 1.169 and 0.842 against (A); (B), which reads the same stale planes, matches every row
   (8.7e-07). For MTP / n-gram verification this means: a rollback by r is exact only if the last llama_process call
   of that sequence fed it at least r + 1 tokens (then they are in its last ubatch by the split rule); one-token
   decode calls followed by a rollback, or a rollback across calls, are wrong without an error (PROVISIONAL rule
   derived from the cited code; the two cases above are CONFIRMED).
6. **seq_cp while a rollback is pending is accepted and the copy continues from the state before the rollback (both
   pins).** Run `cp_pending`: `r2@0` (pending), `c0:1`: in graph 2 sequence 0 moves to a new cell and reads plane 2
   (right), sequence 1 keeps the shared cell and reads plane 0 (`rs_idx[1]` = 0, RC:1364-1381), the state of all 24
   tokens including the 2 rejected; its KV history is the 22 kept tokens. State off by 0.664, logits by up to 1.137
   against (A); (B) matches.
7. **ec7630a accepts a rollback of a cell two sequences share; 79e2e74 refuses it.** Run `cp_shared_rb`: after `c0:1`,
   `r2@0` returns true at ec7630a and moves the shared cell back for both sequences (RC:207-213): the values stay right
   (the copy on write gives each sequence its own plane), but `llama_memory_seq_pos_max(1)` drops from 23 to 21 while
   sequence 1's KV cells still hold 22 and 23 (manifest `pos_max` [21, 21]); probe `shared_rb_then_pos_max`: a decode
   of sequence 1 at position 22 (= the reported max + 1) is then accepted (ret 0; what it computes is UNKNOWN, not
   modelled). At 79e2e74 the rollback returns false (RC:202-205@79e2e74) and that decode is refused.
8. **Tokens shared by several sequences (coupled ids) are not usable with qwen4exp's PLE**: with one KV stream the
   first graph aborts (G:1248 / G:1246@79e2e74, exit status 134), with one stream per sequence llama_process returns 1
   ("sequential split is not supported when there are coupled sequences"), both pins (probes `coupled_uni`,
   `coupled_str`). Whether a real Qwen3.8-Flash-Next GGUF has PLE layers is UNKNOWN (gap 9).
9. **Dump tool unchanged for old plans**: the N2 default call and the N6 per-graph, rollback and MTP modes write the
   same files byte for byte as the binary built from dump_tensors.cpp @0b0eace, at both pins.

## Mutation check

24 mutants (docs/evidence/mutation-scripts/2026-10-11-n19-qwen4exp-multiseq-mutations.py): 18 single-point mutations of
the oracle (O1-O18), 2 of the plan expectations (T1, T2), 4 simulated llama.cpp regressions on a copy of the dumps
(D1-D4). Caught 22/24 at ec7630a and 23/24 at 79e2e74 (literal output in the last section). Every survivor
changes only the other pin's behaviour and is caught there: at ec7630a O13 (79e2e74's one-row-per-rep rule) and T1 (the
expectation that 79e2e74 refuses the shared-cell rollback); at 79e2e74 O17 (ec7630a's padding cell). The four dump
mutants (attention mass on another sequence's cell, another sequence's state read, a rollback > n_rs_seq accepted, a
non-stale state after a rollback across calls) are caught at both pins. O13 needed a plan change to become reachable:
in the first round it survived at 79e2e74 too, because no ubatch marked two pools with the same rep; coupled sequence
ids, the case the upstream comment names, abort with PLE (finding 8), so cp_uni now copies twice before one decode
(both destinations stale, every pool of theirs shares its rep), after which O13 is caught. The first-round lines at
79e2e74 (its log file was overwritten by the rerun; lines copied from the session output):

```
SURVIVED: O13 79e2e74: a pool marked once per pool, not once per rep cell
    rc=0 Ran 72 tests in 10.021s | OK
SURVIVED: O17 ec7630a padding cell: always the first token's cell
    rc=0 Ran 72 tests in 9.870s | OK
caught 22/24
```

## Not covered (UNKNOWN / NOT RUN)

| what | status | why / what would close it |
|---|---|---|
| rollback in the MTP context (optional item 4) | NOT RUN | the qwen4exp MTP context's recurrent memory keeps no layer (src/llama-model.cpp:2714-2720), so its seq_rm only moves the position back (RC:202-206), without the n_rs_seq bound (CONFIRMED by the code only); the dump tool's MTP mode has no rollback step |
| seq_add / K-shift, seq_keep, seq_div, cuts that are not a tail (head or middle), partial seq_cp | NOT RUN | not modelled (B keeps each sequence's positions contiguous from 0). The attention cache of qwen4exp cannot shift (KV:1193-1195, n_pos_per_embd > 1); a partial seq_cp is an assertion at 79e2e74 (HI:202@79e2e74) and copies the whole recurrent state at ec7630a (RC:261-296 ignores the range) |
| what the accepted decode of finding 7 computes | UNKNOWN | the oracle keeps positions contiguous |
| more than 3 sequences, state save / load per sequence, multimodal batches | NOT RUN | |
| quantized weights, f16 KV cache, flash attention, GPU backends, the real model | NOT RUN | as N6; all f32, -fa off, CPU |
| the new CI step in GitHub Actions | NOT RUN | the branch is not pushed by this task |

## Commands and literal output

Environment: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs; c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; Python 3.13.16;
numpy 2.5.3; cmake 3.28.3; Linux 6.18.44-fc-v114. Scratch helpers named below are not committed.

### Pin check

```
$ for L in llama-ec7630a llama.cpp; do (cd <S>/$L && git rev-parse HEAD && git status --short | wc -l); done
ec7630a640789c393694fb194f1bbbf0369fc62d
0
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
0
```

### Driver through the CI checker, both pins

Lines of the driver's output (the verbose per-test lines and the per-run table of the test module are omitted here; the aggregate below is computed from that table):

```
$ QWEN4EXP_MULTISEQ_WORK=<S>/n19/w-ec python3 tools/ci/check_unittest.py --step n19-qwen4exp-multiseq --checkout <S>/llama-ec7630a -- tools/tinygen/run_parity_qwen4exp_multiseq.sh <S>/llama-ec7630a <S>/n19/build-ec
CMAKE_BUILD_TYPE=Release
wrote <S>/n19/w-ec/n19/qwen4exp.gguf
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False] (3 sequences, one KV stream (kv_unified): unequal prefill in one call, interleaved decode)
multi_str: 21 graphs, 15 steps, seq_rm ok: [False] (the same plan with one KV stream per sequence (sequential split))
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True] (seq_cp 0->1, 0->2 (shared cells), both continue, drop 1 and 2, copy twice again (kv_unified))
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True] (the same plan with one KV stream per sequence (cross-stream buffer copies))
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True] (n_rs_seq 3, -ub 8: rollbacks of 1-3 per sequence, refused ones, inside a multi-ubatch call)
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True] (the same plan with one KV stream per sequence)
stale: 11 graphs, 13 steps, seq_rm ok: [True, True] (rollbacks deeper than the last call's last ubatch (accepted, silently read stale snapshot planes))
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True] (seq_cp while a rollback is pending: the copy reads the state before the rollback)
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [True] (a rollback of one of two sequences that share a recurrent cell after seq_cp)
Ran 72 tests in 10.419s
OK
artifacts: <S>/n19/w-ec
::notice::n19-qwen4exp-multiseq @ llama.cpp ec7630a: Ran 72, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
```

```
$ QWEN4EXP_MULTISEQ_WORK=<S>/n19/w-r1 python3 tools/ci/check_unittest.py --step n19-qwen4exp-multiseq --checkout <S>/llama.cpp -- tools/tinygen/run_parity_qwen4exp_multiseq.sh <S>/llama.cpp <S>/n19/build-r1
CMAKE_BUILD_TYPE=Release
wrote <S>/n19/w-r1/n19/qwen4exp.gguf
multi_uni: 17 graphs, 15 steps, seq_rm ok: [False] (3 sequences, one KV stream (kv_unified): unequal prefill in one call, interleaved decode)
multi_str: 21 graphs, 15 steps, seq_rm ok: [False] (the same plan with one KV stream per sequence (sequential split))
cp_uni: 20 graphs, 26 steps, seq_rm ok: [True, True] (seq_cp 0->1, 0->2 (shared cells), both continue, drop 1 and 2, copy twice again (kv_unified))
cp_str: 23 graphs, 26 steps, seq_rm ok: [True, True] (the same plan with one KV stream per sequence (cross-stream buffer copies))
rb_uni: 18 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True] (n_rs_seq 3, -ub 8: rollbacks of 1-3 per sequence, refused ones, inside a multi-ubatch call)
rb_str: 19 graphs, 19 steps, seq_rm ok: [True, True, True, False, True, False, True] (the same plan with one KV stream per sequence)
stale: 11 graphs, 13 steps, seq_rm ok: [True, True] (rollbacks deeper than the last call's last ubatch (accepted, silently read stale snapshot planes))
cp_pending: 6 graphs, 8 steps, seq_rm ok: [True] (seq_cp while a rollback is pending: the copy reads the state before the rollback)
cp_shared_rb: 6 graphs, 8 steps, seq_rm ok: [False] (a rollback of one of two sequences that share a recurrent cell after seq_cp)
Ran 72 tests in 10.442s
OK
artifacts: <S>/n19/w-r1
::notice::n19-qwen4exp-multiseq @ llama.cpp 79e2e74: Ran 72, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
```

### Worst error per component over the nine runs

Aggregated from the table the test module prints (scratch helper `aggregate.py`: maximum over runs; rows of diverged
sequences have [B] only).

llama.cpp ec7630a:

```
$ python3 <S>/n19/scripts/aggregate.py <S>/n19/ci-ec.log
component                                                  run           layer graph    max abs        rel
final_output [B]                                           cp_uni            6     7  2.048e-05  1.314e-05
final_output [A]                                           cp_uni            6     7  2.048e-05  1.314e-05
indexer_score (own visible pools) [B]                      multi_str         3     0  2.722e-07  9.095e-06
linear_attn_out [B]                                        cp_uni            6     7  9.608e-06  7.524e-06
linear_attn_out [A]                                        cp_uni            6     7  9.608e-06  7.524e-06
ffn_out [B]                                                cp_uni            6     7  1.138e-06  4.236e-06
ffn_out [A]                                                cp_uni            6     7  1.138e-06  4.236e-06
attn_output [B]                                            multi_str         6     7  1.818e-07  3.906e-06
attn_output [A]                                            multi_str         6     7  1.818e-07  3.906e-06
Qcur [B]                                                   rb_str            7     5  8.401e-06  3.232e-06
kq_soft_max (visible cells) [B]                            cp_uni            7     7  4.958e-07  3.232e-06
state_predelta (delta-rule state) [B]                      multi_uni         6     4  1.369e-06  2.862e-06
state_predelta (delta-rule state) [A]                      multi_uni         6     4  1.369e-06  2.862e-06
new_state (written back) [B]                               multi_uni         6     3  1.369e-06  2.862e-06
Qcur [A]                                                   cp_uni            7     7  6.703e-06  2.818e-06
conv_output_silu [B]                                       multi_uni         6     2  3.627e-06  2.653e-06
conv_output_silu [A]                                       multi_uni         6     2  3.627e-06  2.653e-06
attn_pregate [B]                                           rb_uni            7     5  1.361e-06  2.408e-06
Vcur [B]                                                   multi_uni         7     2  2.941e-06  2.336e-06
Vcur [A]                                                   multi_uni         7     2  2.941e-06  2.336e-06
Kcur [B]                                                   cp_uni            7     7  5.654e-06  2.326e-06
Kcur [A]                                                   cp_uni            7     7  5.654e-06  2.326e-06
result_norm [B]                                            cp_uni           -1     7  3.294e-06  2.242e-06
result_norm [A]                                            cp_uni           -1     7  3.294e-06  2.242e-06
conv_state_at (GDN conv history) [B]                       rb_str            5     4  2.641e-06  2.172e-06
conv_state_at (GDN conv history) [A]                       rb_str            5     4  2.641e-06  2.172e-06
result_output [B]                                          cp_uni           -1     7  2.969e-06  2.124e-06
result_output [A]                                          cp_uni           -1     7  2.969e-06  2.124e-06
logits.bin (batch order) [B]                               cp_uni           -1     7  2.969e-06  2.124e-06
logits.bin (batch order) [A]                               cp_uni           -1     7  2.969e-06  2.124e-06
attn_pregate [A]                                           multi_uni         3     2  1.080e-06  2.088e-06
hc_combine [B]                                             multi_uni         5     2  1.097e-05  1.890e-06
hc_combine [A]                                             multi_uni         5     2  1.097e-05  1.890e-06
l_last [B]                                                 multi_uni         5     2  1.043e-05  1.765e-06
l_last [A]                                                 multi_uni         5     2  1.043e-05  1.765e-06
conv_output_raw [B]                                        multi_uni         6     2  3.506e-06  1.674e-06
conv_output_raw [A]                                        multi_uni         6     2  3.506e-06  1.674e-06
indexer_q [B]                                              cp_str            3     1  4.175e-06  1.624e-06
indexer_q [A]                                              cp_str            3     1  4.175e-06  1.624e-06
indexer_k_raw [B]                                          multi_str         3    12  8.737e-07  1.287e-06
indexer_k_raw [A]                                          multi_str         3    12  8.737e-07  1.287e-06
indexer_pool_k_new (padding rows)                          multi_str         3    12  1.834e-06  1.136e-06
ple_gated_value [B]                                        stale             1     7  1.751e-06  8.160e-07
ple_gate [B]                                               stale             1     7  5.095e-07  8.096e-07
indexer_pool_k_new (pooled here) [B]                       rb_uni            3    12  1.304e-06  7.560e-07
indexer_k (every pool, layout order) [B]                   stale             3     6  1.562e-06  5.117e-07
ple_gate [A]                                               rb_str            1     4  3.701e-07  4.725e-07
ple_gated_value [A]                                        cp_str            1    12  3.781e-07  3.288e-07
ple_conv_out [B]                                           cp_str            1    17  6.759e-07  3.103e-07
ple_conv_out [A]                                           cp_str            1    17  6.759e-07  3.103e-07
conv_state_at (PLE conv history) [B]                       cp_str            1     8  6.296e-07  2.112e-07
conv_state_at (PLE conv history) [A]                       cp_str            1     8  6.296e-07  2.112e-07
ple_embd = table[rows] (exact) [B]                         cp_pending       -1     0  0.000e+00  0.000e+00
ple_embd = table[rows] (exact) [A]                         cp_pending       -1     0  0.000e+00  0.000e+00
hc_init (exact)                                            cp_pending       -1     0  0.000e+00  0.000e+00
cp_pending: diverged seq 1: state off by 6.64e-01, logits vs (A) up to 1.137e+00 (graph 2)
stale: diverged seq 0: state off by 1.02e+00, logits vs (A) up to 1.169e+00 (graph 3)
stale: diverged seq 1: state off by 7.35e-01, logits vs (A) up to 8.415e-01 (graph 8)
tie replays: ['qsa selection: tie replays = 0']
```

llama.cpp 79e2e74:

```
$ python3 <S>/n19/scripts/aggregate.py <S>/n19/ci-r1.log
component                                                  run           layer graph    max abs        rel
final_output [B]                                           cp_uni            6     7  2.048e-05  1.314e-05
final_output [A]                                           cp_uni            6     7  2.048e-05  1.314e-05
indexer_score (own visible pools) [B]                      multi_str         3     0  2.424e-07  8.099e-06
linear_attn_out [B]                                        cp_uni            6     7  9.608e-06  7.524e-06
linear_attn_out [A]                                        cp_uni            6     7  9.608e-06  7.524e-06
ffn_out [B]                                                cp_uni            6     7  1.138e-06  4.236e-06
ffn_out [A]                                                cp_uni            6     7  1.138e-06  4.236e-06
attn_output [B]                                            multi_str         6     7  1.818e-07  3.906e-06
attn_output [A]                                            multi_str         6     7  1.818e-07  3.906e-06
Qcur [B]                                                   rb_str            7     5  8.401e-06  3.232e-06
kq_soft_max (visible cells) [B]                            cp_uni            7     7  4.958e-07  3.232e-06
state_predelta (delta-rule state) [B]                      multi_uni         6     4  1.369e-06  2.862e-06
state_predelta (delta-rule state) [A]                      multi_uni         6     4  1.369e-06  2.862e-06
new_state (written back) [B]                               multi_uni         6     3  1.369e-06  2.862e-06
Qcur [A]                                                   cp_uni            7     7  6.703e-06  2.818e-06
conv_output_silu [B]                                       multi_uni         6     2  3.627e-06  2.653e-06
conv_output_silu [A]                                       multi_uni         6     2  3.627e-06  2.653e-06
attn_pregate [B]                                           rb_uni            7     5  1.361e-06  2.408e-06
Vcur [B]                                                   multi_uni         7     2  2.941e-06  2.336e-06
Vcur [A]                                                   multi_uni         7     2  2.941e-06  2.336e-06
Kcur [B]                                                   cp_uni            7     7  5.654e-06  2.326e-06
Kcur [A]                                                   cp_uni            7     7  5.654e-06  2.326e-06
result_norm [B]                                            cp_uni           -1     7  3.294e-06  2.242e-06
result_norm [A]                                            cp_uni           -1     7  3.294e-06  2.242e-06
conv_state_at (GDN conv history) [B]                       rb_str            5     4  2.641e-06  2.172e-06
conv_state_at (GDN conv history) [A]                       rb_str            5     4  2.641e-06  2.172e-06
result_output [B]                                          cp_uni           -1     7  2.969e-06  2.124e-06
result_output [A]                                          cp_uni           -1     7  2.969e-06  2.124e-06
logits.bin (batch order) [B]                               cp_uni           -1     7  2.969e-06  2.124e-06
logits.bin (batch order) [A]                               cp_uni           -1     7  2.969e-06  2.124e-06
attn_pregate [A]                                           multi_uni         3     2  1.080e-06  2.088e-06
hc_combine [B]                                             multi_uni         5     2  1.097e-05  1.890e-06
hc_combine [A]                                             multi_uni         5     2  1.097e-05  1.890e-06
l_last [B]                                                 multi_uni         5     2  1.043e-05  1.765e-06
l_last [A]                                                 multi_uni         5     2  1.043e-05  1.765e-06
conv_output_raw [B]                                        multi_uni         6     2  3.506e-06  1.674e-06
conv_output_raw [A]                                        multi_uni         6     2  3.506e-06  1.674e-06
indexer_q [B]                                              cp_str            3     1  4.175e-06  1.624e-06
indexer_q [A]                                              cp_str            3     1  4.175e-06  1.624e-06
indexer_k_raw [B]                                          multi_str         3    12  8.737e-07  1.287e-06
indexer_k_raw [A]                                          multi_str         3    12  8.737e-07  1.287e-06
ple_gated_value [B]                                        stale             1     7  1.751e-06  8.160e-07
ple_gate [B]                                               stale             1     7  5.095e-07  8.096e-07
indexer_pool_k_new (pooled here) [B]                       rb_uni            3    12  1.304e-06  7.560e-07
indexer_pool_k_new (padding rows)                          cp_str            3     8  1.044e-06  5.125e-07
indexer_k (every pool, layout order) [B]                   stale             3     6  1.562e-06  5.117e-07
ple_gate [A]                                               rb_str            1     4  3.701e-07  4.725e-07
ple_gated_value [A]                                        cp_str            1    12  3.781e-07  3.288e-07
ple_conv_out [B]                                           cp_str            1    17  6.759e-07  3.103e-07
ple_conv_out [A]                                           cp_str            1    17  6.759e-07  3.103e-07
conv_state_at (PLE conv history) [B]                       cp_str            1     8  6.296e-07  2.112e-07
conv_state_at (PLE conv history) [A]                       cp_str            1     8  6.296e-07  2.112e-07
ple_embd = table[rows] (exact) [B]                         cp_pending       -1     0  0.000e+00  0.000e+00
ple_embd = table[rows] (exact) [A]                         cp_pending       -1     0  0.000e+00  0.000e+00
hc_init (exact)                                            cp_pending       -1     0  0.000e+00  0.000e+00
cp_pending: diverged seq 1: state off by 6.64e-01, logits vs (A) up to 1.137e+00 (graph 2)
stale: diverged seq 0: state off by 1.02e+00, logits vs (A) up to 1.169e+00 (graph 3)
stale: diverged seq 1: state off by 7.35e-01, logits vs (A) up to 8.415e-01 (graph 8)
tie replays: ['qsa selection: tie replays = 0']
```

### Byte identity of the dump tool's old modes

```
$ for p in ec r1; do bash docs/evidence/mutation-scripts/2026-10-11-n19-byteid.sh <repo> 0b0eace <S>/n19/build-$p <llama.cpp of the pin> <S>/n19/byteid-$p; done
llama.cpp ec7630a; base dump_tensors.cpp @0b0eace vs 981a390
default: 900 files; diff -r: identical
ub10: 6289 files; diff -r: identical
rollback: 12407 files; diff -r: identical
mtp: 12428 files; diff -r: identical
llama.cpp 79e2e74; base dump_tensors.cpp @0b0eace vs 981a390
default: 900 files; diff -r: identical
ub10: 6289 files; diff -r: identical
rollback: 12407 files; diff -r: identical
mtp: 12376 files; diff -r: identical
```

### Probes

```
$ for p in ec r1; do echo "#### llama.cpp $p"; bash docs/evidence/mutation-scripts/2026-10-11-n19-probes.sh <S>/n19/build-$p/dump-tensors <S>/n19/work/qwen4exp.gguf <S>/n19/probes-$p; done
#### llama.cpp ec
== coupled_uni (np 2, kv_unified 1, n_rs_seq 0, DUMP_CALLS 12@0|1): exit status 134
  log: GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences") failed
== coupled_str (np 2, kv_unified 0, n_rs_seq 0, DUMP_CALLS 12@0|1): exit status 0
  decode  [{"seq": 0, "n": 12, "pos0 ret 1    pos_max [-1, -1]
  log: E split_equal: sequential split is not supported when there are coupled sequences in the input batch (you may need to use the -kvu flag)
  log: W decode: failed to find a memory slot for batch of size 12
== rs0 (np 1, kv_unified 1, n_rs_seq 0, DUMP_CALLS 10@0,r1@0): exit status 0
  decode  [{"seq": 0, "n": 10, "pos0 ret 0    pos_max [9]
  seq_rm  [0, 9]                     ok False pos_max [9]
== refused_then_decode (np 1, kv_unified 1, n_rs_seq 3, DUMP_CALLS 20@0,4@0,r4@0,1@0:20): exit status 0
  decode  [{"seq": 0, "n": 20, "pos0 ret 0    pos_max [19]
  decode  [{"seq": 0, "n": 4, "pos0" ret 0    pos_max [23]
  seq_rm  [0, 20]                    ok False pos_max [23]
  decode  [{"seq": 0, "n": 1, "pos0" ret -1   pos_max [23]
  log: E decode: failed to initialize batch
  log: E init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
== shared_rb_then_pos_max (np 2, kv_unified 1, n_rs_seq 3, DUMP_CALLS 20@0,4@0,c0:1,r2@0,1@1:22): exit status 0
  decode  [{"seq": 0, "n": 20, "pos0 ret 0    pos_max [19, -1]
  decode  [{"seq": 0, "n": 4, "pos0" ret 0    pos_max [23, -1]
  seq_cp  [0, -1]                    ok True  pos_max [23, 23]
  seq_rm  [0, 22]                    ok True  pos_max [21, 21]
  decode  [{"seq": 1, "n": 1, "pos0" ret 0    pos_max [21, 22]
#### llama.cpp r1
== coupled_uni (np 2, kv_unified 1, n_rs_seq 0, DUMP_CALLS 12@0|1): exit status 134
  log: GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences") failed
== coupled_str (np 2, kv_unified 0, n_rs_seq 0, DUMP_CALLS 12@0|1): exit status 0
  decode  [{"seq": 0, "n": 12, "pos0 ret 1    pos_max [-1, -1]
  log: E split_equal: sequential split is not supported when there are coupled sequences in the input batch (you may need to use the -kvu flag)
  log: W decode: failed to find a memory slot for batch of size 12
== rs0 (np 1, kv_unified 1, n_rs_seq 0, DUMP_CALLS 10@0,r1@0): exit status 0
  decode  [{"seq": 0, "n": 10, "pos0 ret 0    pos_max [9]
  seq_rm  [0, 9]                     ok False pos_max [9]
== refused_then_decode (np 1, kv_unified 1, n_rs_seq 3, DUMP_CALLS 20@0,4@0,r4@0,1@0:20): exit status 0
  decode  [{"seq": 0, "n": 20, "pos0 ret 0    pos_max [19]
  decode  [{"seq": 0, "n": 4, "pos0" ret 0    pos_max [23]
  seq_rm  [0, 20]                    ok False pos_max [23]
  decode  [{"seq": 0, "n": 1, "pos0" ret -1   pos_max [23]
  log: E decode: failed to initialize batch
  log: E init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
== shared_rb_then_pos_max (np 2, kv_unified 1, n_rs_seq 3, DUMP_CALLS 20@0,4@0,c0:1,r2@0,1@1:22): exit status 0
  decode  [{"seq": 0, "n": 20, "pos0 ret 0    pos_max [19, -1]
  decode  [{"seq": 0, "n": 4, "pos0" ret 0    pos_max [23, -1]
  seq_cp  [0, -1]                    ok True  pos_max [23, 23]
  seq_rm  [0, 22]                    ok False pos_max [23, 23]
  decode  [{"seq": 1, "n": 1, "pos0" ret -1   pos_max [23, 23]
  log: E decode: failed to initialize batch
  log: E init: the tokens of sequence 1 in the input batch have inconsistent sequence positions:
```

(bash also printed one job-control line `... Aborted   env DUMP_DIR=...` per pin for the `coupled_uni` probe, dropped above; its exit status 134 is printed by the probe.)

### Mutation check, both pins

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n19-qwen4exp-multiseq-mutations.py <repo> <S>/n19/w-ec <S>/llama-ec7630a <S>/n19/mut-ec
llama.cpp ec7630a; baseline (no mutation): rc=0 Ran 72 tests in 10.190s | OK
caught: O1 every sequence of a ubatch reads the first sequence's recurrent row
    rc=1 Ran 72 tests in 5.054s | FAILED (failures=52)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_str.test_attention_cells_and_no_leakage, Run_cp_uni.test_attention_cells_and_no_leakage, Run_multi_str.test_attention_cells_and_no_leakage, Run_multi_uni.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_cp_str.test_carried_state, Run_cp_uni.test_carried_state, Run_multi_str.test_carried_state, Run_multi_uni.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_cp_str.test_edge_behaviour, Run_cp_uni.test_edge_behaviour, Run_multi_str.test_edge_behaviour, Run_multi_uni.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_str.test_gdn_and_ple_rows, Run_cp_uni.test_gdn_and_ple_rows, Run_multi_str.test_gdn_and_ple_rows, Run_multi_uni.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_str.test_moe_head_and_logits, Run_cp_uni.test_moe_head_and_logits, Run_multi_str.test_moe_head_and_logits, Run_multi_uni.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_cp_str.test_qsa_pool_layout, Run_cp_uni.test_qsa_pool_layout, Run_multi_str.test_qsa_pool_layout, Run_multi_uni.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_then_both_continue, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending, MultiSeqOracle.test_sequences_are_independent
caught: O2 seq_cp does not share the recurrent state (the copy starts from zeros)
    rc=1 Ran 72 tests in 8.347s | FAILED (failures=30)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_str.test_attention_cells_and_no_leakage, Run_cp_uni.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_cp_str.test_carried_state, Run_cp_uni.test_carried_state, Run_cp_shared_rb.test_edge_behaviour, Run_cp_str.test_edge_behaviour, Run_cp_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_str.test_gdn_and_ple_rows, Run_cp_uni.test_gdn_and_ple_rows, Run_cp_pending.test_graphs_rows_and_memory_positions, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_cp_str.test_graphs_rows_and_memory_positions, Run_cp_uni.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_str.test_moe_head_and_logits, Run_cp_uni.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_cp_str.test_qsa_pool_layout, Run_cp_uni.test_qsa_pool_layout, RecurrentRefusals.test_rollback_rules, MultiSeqOracle.test_seq_cp_then_both_continue, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O3 rollback reads the snapshot one token too recent
    rc=1 Ran 72 tests in 9.352s | FAILED (failures=33)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, RecurrentRefusals.test_rollback_rules, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O4 rollback keeps one position too many in the KV history
    rc=1 Ran 72 tests in 8.109s | FAILED (failures=37, errors=1)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_graphs_rows_and_memory_positions, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_rb_str.test_graphs_rows_and_memory_positions, Run_rb_uni.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O5 every delta-rule snapshot plane written (as if a short ubatch kept them all)
    rc=1 Ran 72 tests in 9.281s | FAILED (failures=7)
    failing: Run_stale.test_attention_cells_and_no_leakage, Run_stale.test_carried_state, Run_stale.test_edge_behaviour, Run_stale.test_gdn_and_ple_rows, Run_stale.test_moe_head_and_logits, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes
caught: O6 conv / PLE snapshot planes all hold the newest history
    rc=1 Ran 72 tests in 8.252s | FAILED (failures=32)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O7 s_copy ignores a pending rollback (plane 0 always)
    rc=1 Ran 72 tests in 8.223s | FAILED (failures=32)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O8 a pending rollback is not single-use
    rc=1 Ran 58 tests in 8.287s | FAILED (failures=1, errors=2)
    failing: test_qwen4exp_multiseq_parity.setUpClass, RecurrentRefusals.test_rollback_rules
caught: O9 no refusal of a rollback of a shared recurrent cell (R1 rule dropped)
    rc=1 Ran 72 tests in 10.204s | FAILED (failures=1)
    failing: RecurrentRefusals.test_rollback_rules
caught: O10 the extra recurrent rows are not copied (a pending rollback lost)
    rc=1 Ran 72 tests in 8.044s | FAILED (failures=13)
    failing: Run_multi_uni.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_multi_uni.test_carried_state, Run_rb_uni.test_carried_state, Run_multi_uni.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_multi_uni.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_multi_uni.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_multi_uni.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, MultiSeqOracle.test_sequences_are_independent
caught: O11 one KV stream for all sequences even without kv_unified
    rc=1 Ran 51 tests in 5.423s | FAILED (failures=2, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, MultiSeqOracle.test_cross_stream_copy, MultiSeqOracle.test_sequences_are_independent
caught: O12 ec7630a: shared cells never detected (cache_safe always true)
    rc=1 Ran 72 tests in 9.813s | FAILED (failures=5)
    failing: Run_cp_uni.test_edge_behaviour, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_cp_uni.test_qsa_pool_layout, MultiSeqOracle.test_seq_cp_then_both_continue
SURVIVED: O13 79e2e74: a pool marked once per pool, not once per rep cell
    rc=0 Ran 72 tests in 9.113s | OK
    failing: none
caught: O14 ec7630a: dropping a sharing sequence does not stale every sequence
    rc=1 Ran 72 tests in 8.643s | FAILED (failures=3)
    failing: Run_cp_uni.test_edge_behaviour, Run_cp_uni.test_qsa_pool_layout, MultiSeqOracle.test_seq_cp_then_both_continue
caught: O15 split: the sets of a ubatch need not be consecutive with one stream per sequence
    rc=1 Ran 51 tests in 7.522s | FAILED (failures=2, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, SplitEqual.test_examples_walked_through_the_code, SplitEqual.test_invariants_on_random_batches
caught: O16 split: the last n_rs_seq + 1 tokens of a sequence may straddle ubatches
    rc=1 Ran 58 tests in 8.827s | FAILED (failures=3, errors=2)
    failing: test_qwen4exp_multiseq_parity.setUpClass, SplitEqual.test_examples_walked_through_the_code, SplitEqual.test_invariants_on_random_batches, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix
caught: O17 ec7630a padding cell: always the first token's cell
    rc=1 Ran 72 tests in 9.751s | FAILED (failures=7)
    failing: Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_multi_str.test_qsa_pool_layout, Run_multi_uni.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout
caught: O18 seq_cp: the destination's pools are not marked stale
    rc=1 Ran 72 tests in 9.569s | FAILED (failures=1)
    failing: Run_cp_str.test_qsa_pool_layout
SURVIVED: T1 plans: the rollback of a shared cell expected to be accepted at both pins
    rc=0 Ran 72 tests in 9.922s | OK
    failing: none
caught: T2 plans: the stale rollback expected not to diverge
    rc=1 Ran 72 tests in 9.414s | FAILED (failures=2)
    failing: Run_stale.test_carried_state, Run_stale.test_edge_behaviour
caught: D1 llama.cpp puts attention probability on another sequence's cell
    rc=1 Ran 72 tests in 10.183s | FAILED (failures=1)
    failing: Run_multi_uni.test_attention_cells_and_no_leakage
caught: D2 llama.cpp reads another sequence's delta-rule state
    rc=1 Ran 72 tests in 9.176s | FAILED (failures=1)
    failing: Run_multi_uni.test_carried_state
caught: D3 llama.cpp accepts a rollback longer than n_rs_seq
    rc=1 Ran 72 tests in 9.614s | FAILED (failures=2)
    failing: Run_rb_uni.test_edge_behaviour, Run_rb_uni.test_graphs_rows_and_memory_positions
caught: D4 llama.cpp reads another state than the stale snapshot after a rollback across calls
    rc=1 Ran 72 tests in 9.882s | FAILED (failures=1)
    failing: Run_stale.test_carried_state
caught 22/24
```

```
$ python3 docs/evidence/mutation-scripts/2026-10-11-n19-qwen4exp-multiseq-mutations.py <repo> <S>/n19/w-r1 <S>/llama.cpp <S>/n19/mut-r1
llama.cpp 79e2e74; baseline (no mutation): rc=0 Ran 72 tests in 11.225s | OK
caught: O1 every sequence of a ubatch reads the first sequence's recurrent row
    rc=1 Ran 72 tests in 5.376s | FAILED (failures=52)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_str.test_attention_cells_and_no_leakage, Run_cp_uni.test_attention_cells_and_no_leakage, Run_multi_str.test_attention_cells_and_no_leakage, Run_multi_uni.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_cp_str.test_carried_state, Run_cp_uni.test_carried_state, Run_multi_str.test_carried_state, Run_multi_uni.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_cp_shared_rb.test_edge_behaviour, Run_cp_str.test_edge_behaviour, Run_cp_uni.test_edge_behaviour, Run_multi_str.test_edge_behaviour, Run_multi_uni.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_str.test_gdn_and_ple_rows, Run_cp_uni.test_gdn_and_ple_rows, Run_multi_str.test_gdn_and_ple_rows, Run_multi_uni.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_str.test_moe_head_and_logits, Run_cp_uni.test_moe_head_and_logits, Run_multi_str.test_moe_head_and_logits, Run_multi_uni.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_cp_str.test_qsa_pool_layout, Run_cp_uni.test_qsa_pool_layout, Run_multi_str.test_qsa_pool_layout, Run_multi_uni.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_then_both_continue, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending, MultiSeqOracle.test_sequences_are_independent
caught: O2 seq_cp does not share the recurrent state (the copy starts from zeros)
    rc=1 Ran 72 tests in 8.468s | FAILED (failures=30)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_str.test_attention_cells_and_no_leakage, Run_cp_uni.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_cp_shared_rb.test_carried_state, Run_cp_str.test_carried_state, Run_cp_uni.test_carried_state, Run_cp_shared_rb.test_edge_behaviour, Run_cp_str.test_edge_behaviour, Run_cp_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_str.test_gdn_and_ple_rows, Run_cp_uni.test_gdn_and_ple_rows, Run_cp_pending.test_graphs_rows_and_memory_positions, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_cp_str.test_graphs_rows_and_memory_positions, Run_cp_uni.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_str.test_moe_head_and_logits, Run_cp_uni.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_cp_str.test_qsa_pool_layout, Run_cp_uni.test_qsa_pool_layout, RecurrentRefusals.test_rollback_rules, MultiSeqOracle.test_seq_cp_then_both_continue, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O3 rollback reads the snapshot one token too recent
    rc=1 Ran 72 tests in 8.922s | FAILED (failures=27)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, RecurrentRefusals.test_rollback_rules, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O4 rollback keeps one position too many in the KV history
    rc=1 Ran 72 tests in 10.739s | FAILED (failures=30, errors=1)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_stale.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_graphs_rows_and_memory_positions, Run_rb_str.test_graphs_rows_and_memory_positions, Run_rb_uni.test_graphs_rows_and_memory_positions, Run_stale.test_graphs_rows_and_memory_positions, Run_cp_pending.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O5 every delta-rule snapshot plane written (as if a short ubatch kept them all)
    rc=1 Ran 72 tests in 10.978s | FAILED (failures=7)
    failing: Run_stale.test_attention_cells_and_no_leakage, Run_stale.test_carried_state, Run_stale.test_edge_behaviour, Run_stale.test_gdn_and_ple_rows, Run_stale.test_moe_head_and_logits, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes
caught: O6 conv / PLE snapshot planes all hold the newest history
    rc=1 Ran 72 tests in 9.366s | FAILED (failures=26)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O7 s_copy ignores a pending rollback (plane 0 always)
    rc=1 Ran 72 tests in 10.585s | FAILED (failures=26)
    failing: Run_cp_pending.test_attention_cells_and_no_leakage, Run_rb_str.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_stale.test_attention_cells_and_no_leakage, Run_cp_pending.test_carried_state, Run_rb_str.test_carried_state, Run_rb_uni.test_carried_state, Run_stale.test_carried_state, Run_cp_pending.test_edge_behaviour, Run_rb_str.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_cp_pending.test_gdn_and_ple_rows, Run_rb_str.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_stale.test_gdn_and_ple_rows, Run_cp_pending.test_moe_head_and_logits, Run_rb_str.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_stale.test_moe_head_and_logits, Run_cp_pending.test_qsa_pool_layout, Run_rb_str.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, Run_stale.test_qsa_pool_layout, MultiSeqOracle.test_rollback_deeper_than_the_last_ubatch_reads_stale_planes, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix, MultiSeqOracle.test_seq_cp_while_a_rollback_is_pending
caught: O8 a pending rollback is not single-use
    rc=1 Ran 58 tests in 8.947s | FAILED (failures=1, errors=2)
    failing: test_qwen4exp_multiseq_parity.setUpClass, RecurrentRefusals.test_rollback_rules
caught: O9 no refusal of a rollback of a shared recurrent cell (R1 rule dropped)
    rc=1 Ran 72 tests in 10.496s | FAILED (failures=7)
    failing: Run_cp_shared_rb.test_attention_cells_and_no_leakage, Run_cp_shared_rb.test_carried_state, Run_cp_shared_rb.test_gdn_and_ple_rows, Run_cp_shared_rb.test_graphs_rows_and_memory_positions, Run_cp_shared_rb.test_moe_head_and_logits, Run_cp_shared_rb.test_qsa_pool_layout, RecurrentRefusals.test_rollback_rules
caught: O10 the extra recurrent rows are not copied (a pending rollback lost)
    rc=1 Ran 72 tests in 10.291s | FAILED (failures=13)
    failing: Run_multi_uni.test_attention_cells_and_no_leakage, Run_rb_uni.test_attention_cells_and_no_leakage, Run_multi_uni.test_carried_state, Run_rb_uni.test_carried_state, Run_multi_uni.test_edge_behaviour, Run_rb_uni.test_edge_behaviour, Run_multi_uni.test_gdn_and_ple_rows, Run_rb_uni.test_gdn_and_ple_rows, Run_multi_uni.test_moe_head_and_logits, Run_rb_uni.test_moe_head_and_logits, Run_multi_uni.test_qsa_pool_layout, Run_rb_uni.test_qsa_pool_layout, MultiSeqOracle.test_sequences_are_independent
caught: O11 one KV stream for all sequences even without kv_unified
    rc=1 Ran 51 tests in 7.148s | FAILED (failures=2, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, MultiSeqOracle.test_cross_stream_copy, MultiSeqOracle.test_sequences_are_independent
caught: O12 ec7630a: shared cells never detected (cache_safe always true)
    rc=1 Ran 72 tests in 12.538s | FAILED (failures=1)
    failing: MultiSeqOracle.test_seq_cp_then_both_continue
caught: O13 79e2e74: a pool marked once per pool, not once per rep cell
    rc=1 Ran 72 tests in 12.617s | FAILED (failures=2)
    failing: Run_cp_uni.test_edge_behaviour, Run_cp_uni.test_qsa_pool_layout
caught: O14 ec7630a: dropping a sharing sequence does not stale every sequence
    rc=1 Ran 72 tests in 11.075s | FAILED (failures=1)
    failing: MultiSeqOracle.test_seq_cp_then_both_continue
caught: O15 split: the sets of a ubatch need not be consecutive with one stream per sequence
    rc=1 Ran 51 tests in 9.395s | FAILED (failures=2, errors=3)
    failing: test_qwen4exp_multiseq_parity.setUpClass, SplitEqual.test_examples_walked_through_the_code, SplitEqual.test_invariants_on_random_batches
caught: O16 split: the last n_rs_seq + 1 tokens of a sequence may straddle ubatches
    rc=1 Ran 58 tests in 10.638s | FAILED (failures=3, errors=2)
    failing: test_qwen4exp_multiseq_parity.setUpClass, SplitEqual.test_examples_walked_through_the_code, SplitEqual.test_invariants_on_random_batches, MultiSeqOracle.test_rollback_inside_the_last_call_restores_the_kept_prefix
SURVIVED: O17 ec7630a padding cell: always the first token's cell
    rc=0 Ran 72 tests in 12.006s | OK
    failing: none
caught: O18 seq_cp: the destination's pools are not marked stale
    rc=1 Ran 72 tests in 12.353s | FAILED (failures=5)
    failing: Run_cp_uni.test_edge_behaviour, Run_cp_pending.test_qsa_pool_layout, Run_cp_shared_rb.test_qsa_pool_layout, Run_cp_str.test_qsa_pool_layout, Run_cp_uni.test_qsa_pool_layout
caught: T1 plans: the rollback of a shared cell expected to be accepted at both pins
    rc=1 Ran 72 tests in 10.334s | FAILED (failures=1)
    failing: Run_cp_shared_rb.test_edge_behaviour
caught: T2 plans: the stale rollback expected not to diverge
    rc=1 Ran 72 tests in 12.669s | FAILED (failures=2)
    failing: Run_stale.test_carried_state, Run_stale.test_edge_behaviour
caught: D1 llama.cpp puts attention probability on another sequence's cell
    rc=1 Ran 72 tests in 11.508s | FAILED (failures=1)
    failing: Run_multi_uni.test_attention_cells_and_no_leakage
caught: D2 llama.cpp reads another sequence's delta-rule state
    rc=1 Ran 72 tests in 10.974s | FAILED (failures=1)
    failing: Run_multi_uni.test_carried_state
caught: D3 llama.cpp accepts a rollback longer than n_rs_seq
    rc=1 Ran 72 tests in 11.763s | FAILED (failures=2)
    failing: Run_rb_uni.test_edge_behaviour, Run_rb_uni.test_graphs_rows_and_memory_positions
caught: D4 llama.cpp reads another state than the stale snapshot after a rollback across calls
    rc=1 Ran 72 tests in 11.303s | FAILED (failures=1)
    failing: Run_stale.test_carried_state
caught 23/24
```
