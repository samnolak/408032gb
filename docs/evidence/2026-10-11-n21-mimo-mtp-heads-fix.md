# N21: mimo2 MTP heads - how often llama-server hits the N17 placement bug, a llama.cpp fix at 79e2e74, a workaround, severity for R1, 2026-10-11 (sandbox CPU, no GPU)

Task N21 of the hardware-free track, follow-up of N17 finding 2 (`docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md`):
common/speculative.cpp `process()` re-places each target batch once per MTP head after `seq_rm(first position)`, the
three mimo2 MTP head layers share one set of SWA cells, and a batch that crosses the end of the SWA ring leaves head 0
reading K/V of other positions. Branch `claude/n21-mimo-mtp-heads-fix`, cut from N19 at `e71b241` (contains N17). Role:
engine-integrator. `<S>` stands for the session scratch directory
`/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted with only that
substitution. "SP" = `common/speculative.cpp`, "SC" = `tools/server/server-context.cpp`, "SA" = `common/sampling.cpp`,
"KV" = `src/llama-kv-cache.cpp`, "KVI" = `src/llama-kv-cache-iswa.cpp`, "ARG" = `common/arg.cpp`, all at llama.cpp
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (the R1 engine pin, tools/engines/pins.json) unless marked @ec7630a; every
cited line is printed in 6.9 (script `docs/evidence/mutation-scripts/2026-10-11-n21-source-lines.sh`).

| piece | file |
|---|---|
| llama.cpp patch | `patches/llama.cpp/0001-mtp-heads-slot-replay.patch` (base 79e2e74; `patches/llama.cpp/README.md`) |
| oracle | `ref/mimo2.py`: `mtp_process(fixed=True)` (the patched order), `place(..., replay)`, `attn["missing"]` (a row whose window has a position no cell holds) |
| placement-only model | `ref/mtp_cells.py` (`MtpCells`, `serve`, `summary`): the MTP context's SWA cells under llama-server's calls, both orders, no weights |
| exposure estimate | `tools/mtp_ring_exposure.py` (MiMo-V2.6-Flash geometry; PROVISIONAL numbers, 1.3) |
| dump tool | `tools/dump/dump_tensors.cpp`: `DUMP_MTP_FIX=1` (replay of the patched order; refused without the patch), `DUMP_MTP_SPEC=1` (the MTP calls through llama.cpp's own `common_speculative_*`) |
| driver | `tools/tinygen/run_parity_mimo2_n21.sh` (runs the fixed variants only on a patched checkout) |
| tests | `tests/test_mimo2_mtp_fix_parity.py` (dump parity; the fixed classes skip without a patched build), `tests/test_mimo2_mtp_fix_ref.py` (numpy only), `tests/test_gguf_mtp_heads.py`; N17's `tests/test_mimo2_mtp_heads_parity.py` generalized (root, fixed order, incomplete windows) |
| workaround tool | `tools/gguf_mtp_heads.py` (a mimo2 GGUF with the first K NextN heads) |
| CI | step `n21-mimo2-mtp-fix` in `ci.yml` (ec7630a) and `parity-r1-pin.yml` (79e2e74), job `n21-patched` in `parity-r1-pin.yml` (a copy of 79e2e74 with the patch); `tools/ci/parity_expectations.json`; `tests/test_check_unittest.py` allow-list. GitHub Actions: NOT RUN (the branch is not pushed in this task) |
| scripts | `docs/evidence/mutation-scripts/2026-10-11-n21-*`: final-runs, mutations, probes, source-lines, byteid |

## 0. Answers (labels as given)

1. **Severity for R1: the recommended configuration is not at risk of silently wrong output from this bug.**
   * R1's llama.cpp template runs no speculation at all: its argv has no `--spec-type` and no draft model, its `env` sets
     only `CUDA_DEVICE_ORDER`, and its mode says `spec_k 0` / "no speculation" (`scripts/r1/configs/llamacpp-server.json:10, 36-39, 42, 66`).
     No llama.cpp-family R1 template runs MiMo or speculation: `llamacpp-server.json` serves Qwen3.8-Flash-Next,
     `llamacpp-server-moecache.json` and `ik-llamacpp-server.json` GLM-5.3-Flash, all with `"spec_k": 0`
     (`llamacpp-server.json:51, 66`, `llamacpp-server-moecache.json:52, 64`, `ik-llamacpp-server.json:50, 62`) and no
     `--spec*` / `--draft*` / `--mtp` flag in their argv (6.9).
     llama-server turns MTP on only through `--spec-type` / `LLAMA_ARG_SPEC_TYPE` (ARG:4284-4292), or by inference when a
     draft model path or a Hugging Face MTP sidecar is given and no type is (ARG:546-550, 566-573); the default is
     `{NONE}` (`common/common.h:376`, ARG:357-359). So the bug is **not reachable in R1 as configured** (CONFIRMED from
     those lines; the server binary was not run: NOT RUN).
   * Where MTP is turned on (R4.4 / S1, `docs/reassessment/ROADMAP.md:55`), the bug changes **draft tokens only, never the
     target model's state or the emitted tokens' distribution** (CONFIRMED from the source, 1.4 and 1.5): drafts are always
     verified by the target, by exact match (greedy) or by rejection sampling against the draft's own distribution
     (temperature > 0 with `--spec-draft-sampling probabilistic`). What it costs is **draft acceptance, i.e. speed**: in
     the placement model at MiMo-V2.6-Flash geometry, 3.3 % to 15.9 % of the drafts have their first (head-0) token
     computed from K/V of other positions (PROVISIONAL, 1.3); the effect of that on acceptance with real weights is
     UNKNOWN (hardware experiment, section 4). On the tiny model the bug changed exactly one greedy draft of five in
     N17's plan and one of two in the several-ubatch plan (real code with and without the fix, CONFIRMED, 2.3 item 4).
   * A residual, generic caveat that is not this bug: a run with speculation computes the target in different batch
     shapes than a run without it, so on a GPU its greedy tokens can differ from the non-speculative run at near-ties
     (UNKNOWN whether the CUDA kernels of 79e2e74 are batch-invariant); the bug changes which batch shapes occur, not the
     correctness of any of them.
2. **Reachability (task 1)**: every process call whose placement crosses the end of the 768-cell MTP SWA ring
   (pad(min(n_ctx, 128 + 512), 256), KVI:73, window 128 from `config.json:303`), and every process call of more than one
   ubatch (a prompt batch of up to `-b` 2048 tokens is 4 ubatches of 512) can trigger it (CONFIRMED on the tiny model,
   2.3); in steady decode the ring end is crossed roughly once per 768 placed positions and the corruption lasts for the
   window that follows (PROVISIONAL reading of the placement rule, 1.2; counted by the model, 1.3). Only the MTP context
   is affected; the target context is never touched (CONFIRMED, 1.4).
3. **Fix (task 2)**: `patches/llama.cpp/0001-mtp-heads-slot-replay.patch` at 79e2e74: `process()` runs each ubatch through
   all heads before the next one, heads 1 .. H-1 in exactly head 0's cells (new staging API
   `llama_memory_set_slot_replay`). Proven on CPU on the tiny 3-head model (2.3): with the patch no row of any head reads
   foreign K/V or an incomplete window, every MTP tensor equals the full-sequence oracle (A) within 1e-4 (f32) and
   TOL16 (f16 / flash attention), the 11 head-0 rows N17 found wrong now match (A) (kqv_out 5.8e-07, logits 3.6e-07
   against 2.05 and 0.40 upstream), the real patched `common_speculative` computes bit for bit what the replay of the
   patched order computes, all N16 / N17 and other dump steps still pass at both pins, and 17 of 18 mutants are caught
   (2.4; the survivor is a defensive check no N21 run reaches). GPU, the llama-server binary and real weights: NOT RUN.
4. **Workaround without the patch (task 3)**: `--swa-full` (no placement problem; +6.41 GiB of KV at 32K, +52.35 GiB at
   262K, PROVISIONAL from the config), or a one-head GGUF (`tools/gguf_mtp_heads.py --keep 1`; acceptance cost UNKNOWN),
   or no MTP (R1's default). `--spec-draft-n-max 1` does not help (process still runs all three heads; 10.0 % to 12.4 %
   of drafts exposed in the model, PROVISIONAL). Section 3.
5. **Hardware experiment**: section 4 (exact commands; one patched build, A/B by `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1`).

## 1. Reachability under llama-server MTP use at 79e2e74 (task 1)

### 1.1 The calls (CONFIRMED from the source, 6.9)

* llama-server decodes the target, then calls `common_speculative_process(spec, batch.view)` for every target batch
  (SC:4229-4233): prompt batches of up to `-b` tokens (default 2048, `common/common.h:456`) and verify batches
  `[id_last, d_1 .. d_n]`.
* `process()` pairs each token with the target's h row of the previous position (SP:1583-1593), then, because the MTP
  context of mimo2 has 3 heads and does not share the target's memory (`chain_heads`, SP:1481-1482; the MTP context
  gets `ctx_other = ctx_tgt`, SP:2619, but llama_context keeps it only for GEMMA4_ASSISTANT / EAGLE3 / DFLASH,
  `src/llama-context.cpp:147-166`), runs **for head 0, 1, 2: `seq_rm(first position)`, `set_nextn_layer_offset(head)`,
  `llama_process(whole batch)`** (SP:1595-1621; the same code at ec7630a, SP:1533-1557 @ec7630a). `llama_process`
  splits the batch into ubatches of `n_ubatch` (512 by default, `common/common.h:457`).
* `draft()`: step k = 0 .. H-1 runs head k on `[id_last, d_1 .. d_k]` at pos0 .. pos0+k after `seq_rm(pos0)`
  (SP:1696-1708); then the server removes the draft positions, `seq_rm(ctx_dft, pos_max + 1)` (SC:3416-3426).
* After verification `common_speculative_accept` moves the pending h row (SC:4471, SP:1814-1827) and the server cuts both
  contexts after the last accepted token (SC:4497-4505).
* KV placement: `seq_rm` moves the cache head back to the **lowest freed cell index** (KV:428-437); `find_slot` walks from
  the head (from 0 when head > used + 2 n, KV:1019-1025), taking empty cells and cells whose position is masked by the
  window (KV:1061-1078); `apply_ubatch` purges every position up to the largest one it overwrote and sets head = last
  cell + 1 (KV:1185-1205). The MTP context's SWA cache has pad(min(n_ctx, n_swa + n_ubatch), 256) cells (KVI:73).

### 1.2 When it happens (CONFIRMED on the tiny model, 2.3; the MiMo numbers derived)

(a) **A call whose head-0 placement crosses the end of the ring** (N17's case): head 0 takes e.g. cells [765, 766, 767,
0]; `seq_rm` frees them and the head moves to cell 0 (the lowest index), so heads 1 and 2 take [0, 1, 2, 3]; the cells now
name heads 1-2's placement, and head 0's layer holds, under them, K/V of other positions. Every head-0 row whose window
reaches those cells (the next n_swa - 1 positions, process and draft) attends to them. Any call of 2 or more tokens can
straddle; with MiMo the ring is 768 cells (n_swa 128 from `config.json:303`; n_ubatch 512), so in steady decode the ring
end comes round about once per 768 placed positions, and a verify batch of 4 straddles it unless it happens to end there
(PROVISIONAL reading of the placement rule; the model of 1.3 counts it).

(b) **A call of more than one ubatch** (new in N21): head 0's later ubatches may take cells that the window of the first
ubatch still needs (positions just before the call, masked only for the later ubatches), before heads 1 and 2 run that
first ubatch; their first rows then attend to an incomplete window, and the final placement (head 2's) again differs from
head 0's. CONFIRMED on the tiny model (run mtp3b: heads 1-2 rows 44..50 with an incomplete window, head-0 rows 304..310
with foreign K/V, 2.3). Every prompt longer than one ubatch (512 tokens) makes such calls: in the model of 1.3 the
first drafts after a 5000-token prompt (calls of 2048, 2048, 904) are exposed, after a 2000-token prompt (one call from
position 0, nothing before it to lose) not. The process rows themselves are not outputs (`process()` marks none,
SP:1586), and an MTP layer's K/V come from its input projection (eh_proj of the token and the h row, not from attention),
so (b)'s incomplete windows matter only through the placement they leave behind: what the later draft rows read.

Drafts are the only consumer: draft step k reads layer k (head k) for the positions before pos0. In both cases only the
heads that did not place last are inconsistent; in every run here that is head 0 (the first draft token), and once also
the first rows of heads 1-2 in process calls of case (b).

### 1.3 How often (PROVISIONAL: the placement model ref/mtp_cells.py, not llama.cpp)

`ref/mtp_cells.py` replays llama-server's calls on the cells alone with ref/mimo2.py's `Cells` (KV placement as above,
checked against llama.cpp's own cells since N16 and, here, graph by graph against llama.cpp's dumps of both plans and of
llama.cpp's own common_speculative: `test_cell_model_agrees`, `test_cell_model_counts`, 2.3). `tools/mtp_ring_exposure.py`
runs it at MiMo-V2.6-Flash geometry: 3 heads, window 128, ring 768, `-b 2048 -ub 512`, drafts of 3, verify batches of 4,
i.i.d. acceptance per draft token (0.3 / 0.6 / 0.85, a stand-in: the bug itself would lower it), 4000 generated tokens,
seeds 1-3 (literal output 6.7):

| mode | head-0 drafts on foreign K/V | heads 1-2 | note |
|---|---|---|---|
| upstream 79e2e74 | 3.3 % to 15.9 % of drafts (prompt 100 .. 5000) | 0 | the first exposed draft at position 766 .. 2305, or right after a 5000-token prompt |
| patched | 0 | 0 | |
| one head (GGUF with 1 MTP layer) | 0 | - | |
| `--swa-full` (ring = context) | 0 | 0 | |
| upstream, `--spec-draft-n-max 1` | 10.0 % to 12.4 % (prompt 700) | - | process still runs 3 heads |

So with MTP on and no fix, roughly one draft in 6 to 30 has a first token computed from wrong K/V (PROVISIONAL). Whether
such a draft is rejected more often, and by how much the acceptance and the speed drop, depends on the real weights:
UNKNOWN (section 4).

### 1.4 Only the drafts, never the target (CONFIRMED)

`process()` and `draft()` write only the MTP context's memory (`llama_get_memory(ctx_dft)`, SP:1595, 1700-1704); the target's
h rows are read (SP:1581-1590, 1636-1638), never written; the MTP context does not share the target's memory (1.1). On
CPU: the target dumps of every spec run equal those of the replay bit for bit, and the target dumps of the upstream and the
fixed replays of the same plan are identical (`SpecReplay.test_target_independent_of_the_mtp_order`, 6.2, 6.3).

### 1.5 The verify path (CONFIRMED from the source)

* Greedy (temperature 0, or `--spec-draft-sampling greedy`, the default, `common/common.h:337`, ARG:4233-4241): the server
  verifies with `common_sampler_sample_and_accept_n` (SC:4424-4425): at each position the **target** samples its own token
  and the draft token is accepted only if equal; the emitted tokens are the target's (SA:689-721).
* Sampling (temperature > 0, `use_spec_rejection`, SC:504-506) with `--spec-draft-sampling probabilistic`: rejection
  sampling, "Preserves the target distribution exactly" (SA:732-735, 794-803), with q = the draft sampler's own candidate
  distribution, from which the draft token was drawn (SP:1728-1740, 1757-1759). Exactness needs q to be the distribution
  actually sampled, not a good one; wrong K/V only make q worse. With greedy drafting and temperature > 0 the server falls
  back to sample-and-match (SP:1666-1669, SC:4412, 4424-4425), again the target's tokens.
* Synthetic acceptance (`server_sample_and_accept_synth`, SC:4415-4419) is a benchmarking mode that accepts drafts at
  random; it is only active when synthetic rates are configured (`--spec-synth-len`, `--spec-synth-rates`, ARG:4177,
  4190), not in R1.

## 2. The fix: patches/llama.cpp/0001-mtp-heads-slot-replay.patch (task 2)

### 2.1 What it changes (8 files, 142 insertions, 1 deletion, 6.1)

* `llama_memory_i::set_slot_replay(bool)` (default: unsupported), implemented by `llama_kv_cache` and forwarded by
  `llama_kv_cache_iswa` to both caches; staging API `llama_memory_set_slot_replay(llama_memory_t, bool)` in
  `src/llama-ext.h` with `#define LLAMA_EXT_HAS_SLOT_REPLAY 1`.
* `llama_kv_cache::prepare()`: while replay is on, returns the slot infos of the last batch prepared with replay off,
  provided they fit (same number and shape of ubatches, same streams, every cell empty or taken by an earlier ubatch of the
  same replay); otherwise searches as before and logs a warning.
* `common/speculative.cpp` draft-mtp: `process_heads_once()` cuts the batch into chunks of `n_ubatch` tokens and runs, per
  chunk, head 0 after `seq_rm(chunk's first position)` and heads 1 .. H-1 after the same `seq_rm` with replay on. Used when
  the heads are chained and the memory supports replay; `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` keeps the upstream loop (for A/B
  runs). `draft()` is unchanged (each step writes and reads its own layer for the draft positions, which the server
  removes afterwards).
* Invariant: for every position in the MTP cache, every head layer holds that position's K/V under the cell that holds
  the position. Cost: the same number of graphs per process call (H x ubatches; 87 and 54 MTP graphs in both orders,
  2.3); on CPU no other cost was measured (timings NOT RUN); on GPU UNKNOWN.
* Scope: other memory types (hybrid, DSA, DSv4, recurrent) keep the upstream order (the setter returns false). Whether
  other architectures with several NextN heads (glm5-next, deepseek4, step35, bailingmoe3, hy-v3 use
  `nextn_layer_offset`, `src/models/*.cpp`) hit the same placement problem: UNKNOWN (it needs a ring, i.e. sliding-window
  MTP layers, or holes in the cells; not analysed).
* `git apply --check`: applies at 79e2e74; at ec7630a `src/llama-ext.h` does not apply (other context), the other files do
  (6.5).

### 2.2 How it is tested (CPU, tiny model)

Model: N17's `n17.gguf` (6 trunk layers, window 8, MTP heads 6, 7, 8; `tools/tinygen/mimo2_tiny.py --fused-qkv --sinks swa
--value-scale 0.75 --mtp-layers 3`), `-c 512 -ub 64 -t 1`, SWA rings of 256 cells. Plans: **mtp3** = N17's plan (the verify
call 253..256 wraps the ring: case (a)); **mtp3b** = 40, 4 x 1, a call of 260 tokens (5 ubatches; head 0's fourth wraps the
ring and takes the cells of positions 0..43: case (b) and (a)), 3 x 1, d3, 4 (1 wrong), r1, 2 x 1, a final draft of 3.

| run | how the MTP calls are made | needs the patch |
|---|---|---|
| `mtp3`, `mtp3b` | the upstream order, replayed by the dump tool (`DUMP_MTP=1`, N17's replay) | no |
| `spec/mtp3`, `spec/mtp3b` | llama.cpp's own `common_speculative_process/_draft/_accept` (`DUMP_MTP_SPEC=1`): upstream code at an unpatched pin, the fixed code on the patched copy | no |
| `fix/mtp3`, `fix/mtp3b`, `fix/fa-mtp3` | the patched order, replayed (`DUMP_MTP_FIX=1`); fa = `-ctk f16 -ctv f16 -fa on` | yes |
| `spec-off/mtp3`, `spec-off/mtp3b` | the patched `common_speculative` with `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1` | yes |

Oracles: (A) the full-sequence block per head (`mtp_forward_chunk` over the final tokens with the full target's h), (B) the
cell oracle (`mtp_process`, `fixed=True` for the fixed runs, `mtp_draft`), as N17 (`docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md`
section 3). Every graph_mtp component, kq on all n_kv cells, kq_soft_max, the f16 cache views and the flash-attention op
(fa run), logits and h_nextn rows, draft tokens.

### 2.3 Results (CONFIRMED by the outputs in 6.2 to 6.4)

1. **Upstream at both pins (Ran 32 per pin, 5 patch-only skips as expected; 79e2e74 and ec7630a).** mtp3 reproduces N17
   exactly (head-0 rows 257..263, 11 rows). mtp3b shows case (b): rows 44..50 of heads 1 and 2 (process) attend to an
   incomplete window (14 rows) and head-0 rows 304..310 (process and draft, 9 rows) read foreign K/V; those rows differ
   from (A) (kqv_out up to 1.74, relative), all others match (A) and (B). The placement-only model flags the same rows
   and cells as (B), graph by graph.
2. **The real code equals the replay, bit for bit, at both pins and patched**: target dumps (8021 / 5349 files) identical;
   every MTP graph tensor identical (2742 / 1692) except the 3 post-crop output tensors of each process graph, where the real
   `process()` marks no output (0 rows) and the replay every row; draft tokens identical. So N17's and N21's replays are
   what llama-server's `common_speculative` computes.
3. **With the patch (54 of 54 on the copy of 79e2e74, CI-style: HEAD 79e2e74, patch applied with git apply):** no row of any
   head reads foreign K/V or an incomplete window, the heads of a chunk write the same cells (mtp3: [253, 254, 255, 0] for
   all three), every component matches (A) and (B) (f32 1e-4; f16 / flash attention TOL16 = 7.8e-3, op within one binary16
   spacing), and the rows upstream got wrong now match (A): mtp3 8 compared rows (of 11; 3 are rejected positions) kqv_out
   5.83e-07, logits 3.64e-07 (upstream 2.05, 0.40); mtp3b 22 of 23 rows, kqv_out 7.47e-07, logits 6.30e-07 (upstream 1.74,
   0.65).
4. **The fix changes drafts exactly where the bug was**: patched vs `LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1`, real code: the draft
   at 259 in mtp3 is [37, 66, 43, 85] fixed vs [37, 32, 77, 94] upstream (the first draft token differs, so the chained
   steps do too), the draft at 307 in mtp3b [95, 18, 19, 66] vs [95, 20, 4, 55]; the 4 + 1 other drafts are identical; the
   fixed drafts equal the oracle's own argmax over (A)-consistent caches (`test_draft_tokens`).
5. **Nothing else moved**: the N16 and N17 steps on the patched copy pass with N16's and N17's counts; every other CI step
   whose driver uses the dump tool or ref/mimo2.py passes at both pins (6.6); the dump tool writes byte-identical files to
   the one of `e71b241` in every pre-existing mode at both pins (N19's qwen4exp modes and N17 / N21 plans with `DUMP_MTP=1`,
   6.6).

### 2.4 Mutation check (CONFIRMED, 6.8)

| mutant (one edit each) | caught by |
|---|---|
| O1 oracle: fixed process without replay | 17 tests (MtpFix*, MtpFixRef) |
| O2 oracle: fixed process seq_rm from the call's start | setUpClass of the 3 fixed classes, MtpFixRef |
| O3 oracle: fixed process without chunks | the same |
| O4 oracle: incomplete windows never reported | MtpUpstreamB (3), MtpFixRef (2) |
| O5 oracle: replay accepted onto a taken cell | test_replay_refused_on_a_taken_cell |
| C1 cell model: fixed behaves as upstream | 7 tests |
| C2 cell model: window one position wider | 6 tests |
| C3 cell model: no seq_rm before each draft step | 11 tests |
| T1 test: fixed records paired head-major | 14 tests |
| T2 test: chunk h rows taken from the call's start | test_graphs_heads_and_pairing of the 3 fixed classes |
| P1 patch: `prepare()` ignores the replay | 14 tests |
| P2 patch: `process_heads_once` without replay | SpecReplay (3): the real code no longer equals the fixed replay |
| P3 patch: `process_heads_once` without chunks | SpecReplay.test_mtp_graphs_identical |
| P4 patch: seq_rm from the batch's start, not the chunk's | the run itself (`llama_process(ctx_dft) head=0 failed rc=-1 (pos=108)`) |
| P5 patch: `slot_replay_fits` does not check that a cell is free | **survived**: every replay in these runs targets cells the `seq_rm` just before freed, so the check never decides; a defensive guard, not exercised (NOT RUN: a case that replays onto a taken cell) |
| D1 dump tool: DUMP_MTP_FIX without replay | 15 tests |
| D2 dump tool: DUMP_MTP_FIX without chunks | setUpClass of the 3 fixed classes, SpecReplay |
| D3 dump tool: DUMP_MTP_SPEC accept() off by one | SpecReplay (2) |

**17 of 18 caught**; after the C++ mutants the sources were restored (the patched scratch repo clean, the worktree's
dump tool unchanged) and the driver passed again (6.8). P2 and P3 are caught only by SpecReplay: the fixed replay alone
would not notice a broken real `process()`, which is why the step compares the two.

## 3. Guard / workaround without the patch (task 3)

| option | effect on the bug | cost | evidence |
|---|---|---|---|
| keep MTP off (R1's template) | not reachable | no MTP speed-up | CONFIRMED, section 0 |
| the patch (recommended when MTP is wanted) | fixed | none measured on CPU (same graph count); GPU UNKNOWN | 2.3 |
| `--swa-full` (both contexts: the draft context's params are a copy of the target's, `common/speculative.cpp:2527-2531`, `common/common.cpp:1725`; ARG:1693-1699) | no ring, no wrap: 0 exposed drafts in the model | the SWA caches grow from 768 cells to the context: +0.153 GiB per SWA layer at 32K, +1.246 GiB at 262K (f16, 8 KV heads x (192 + 128), `config.json` swa_*); 39 target SWA layers (`hybrid_layer_pattern`, `config.json:47`) + 3 MTP layers: **+6.41 GiB at 32K, +26.10 GiB at 128K, +52.35 GiB at 262K** | PROVISIONAL (computed from the config, 6.5; 0 exposure from the model, 1.3) |
| one-head GGUF: `python3 tools/gguf_mtp_heads.py --keep 1 in.gguf out.gguf` | no chained heads (`chain_heads` needs > 1 head, SP:1482): one decode per process call, drafts grow head 0's own KV; 0 exposed drafts in the model | drafts 2 and 3 come from head 0 run autoregressively instead of the trained heads 1 and 2: acceptance cost UNKNOWN; process does 1 decode instead of 3 (cheaper) | the tool: on the tiny models `--keep 1` of the 3-head n17.gguf is N16's validated 1-head n16.gguf byte for byte (CONFIRMED, 6.5); `--override-kv mimo2.nextn_predict_layers=int:1` is refused by the loader ("wrong number of tensors; expected 95, got 87", CONFIRMED, 6.5); real MiMo GGUF NOT RUN (split files must be merged first) |
| `--spec-draft-n-max 1` | does not help: process still runs all heads (SP:1439, 1598) | - | PROVISIONAL, 10.0 % to 12.4 % exposed in the model (6.5) |

## 4. Hardware experiment (exact commands; NOT RUN)

On the 4x4080 server, with a MiMo-V2.6-Flash GGUF that carries its 3 NextN layers (`mimo2.nextn_predict_layers` 3;
which GGUF: R1.3, UNKNOWN today) at `$MODEL`, and llama.cpp at the R1 pin in `/srv/llm/engines-src/llama.cpp`:

```
# REPO = this repository's checkout on the server; one patched build serves both arms:
# A = upstream order (env set), B = the fix (default)
cp -a /srv/llm/engines-src/llama.cpp /srv/llm/engines-src/llama.cpp-n21
git -C /srv/llm/engines-src/llama.cpp-n21 rev-parse HEAD        # must print 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
git -C /srv/llm/engines-src/llama.cpp-n21 apply "$REPO/patches/llama.cpp/0001-mtp-heads-slot-replay.patch"
cmake -S /srv/llm/engines-src/llama.cpp-n21 -B /srv/llm/engines-build/llama.cpp-n21 -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build /srv/llm/engines-build/llama.cpp-n21 -j --target llama-server
BIN=/srv/llm/engines-build/llama.cpp-n21/bin/llama-server

run_arm() {   # run_arm <name> [env...]: one server, the prompts, the server log kept
  env "${@:2}" $BIN -m "$MODEL" --host 127.0.0.1 --port 8080 -c 32768 -np 1 -ngl all -sm layer -ts 1,1,1,1 \
      -fa on -ctk f16 -ctv f16 --jinja --metrics --spec-type draft-mtp --spec-draft-n-max 3 \
      > arm-$1.log 2>&1 &
  local pid=$!
  until curl -sf http://127.0.0.1:8080/health; do sleep 5; done
  for f in prompts/*.txt; do   # fixed prompts of about 100, 700, 2000, 5000 and 20000 tokens
    jq -n --rawfile p "$f" '{prompt: $p, n_predict: 4096, temperature: 0, cache_prompt: false}' |
      curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d @- > "arm-$1-$(basename $f .txt).json"
  done
  kill $pid; wait $pid
}
run_arm A LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1    # upstream order
run_arm B                                    # the fix
for f in arm-[AB]-*.json; do   # tools/server/server-common.cpp:99-102: timings.draft_n / draft_n_accepted
  echo "$f $(jq -c '.timings | {draft_n, draft_n_accepted, predicted_n, predicted_per_second}' "$f")"
done
grep -h "draft acceptance" arm-A.log arm-B.log   # SC:715 (an info-level slot log line)
for f in arm-A-*.json; do cmp <(jq -r .content "$f") <(jq -r .content "${f/arm-A/arm-B}") && echo "same text: $f"; done
```

Read-out: per arm and prompt `draft_n_accepted / draft_n` and `predicted_per_second` of the response's timings. Expected
(PROVISIONAL from 1.3): A exposes about 3 % to 16 % of the drafts, B none; whether acceptance and tokens/s differ
measurably is the question (UNKNOWN). The texts of A and B can differ only through GPU batch-shape numerics (section 0,
item 1, last point); a difference must be explained before B is adopted. Optional arms: C = the pristine pin (the
`current-llamacpp` build) to show that A equals upstream (same acceptance up to the run-to-run spread), D = `--swa-full` on
the pristine pin at `-c 32768` (+6.41 GiB, 3), E = the one-head GGUF (`tools/gguf_mtp_heads.py --keep 1` on a merged
single-file GGUF) on the pristine pin.

## 5. Not covered (NOT RUN unless labelled)

* GPU, the llama-server binary (its call sequence is driven through llama.cpp's own `common_speculative` by the dump tool,
  not through the server's HTTP loop), real MiMo weights or GGUF, several slots (`-np` > 1), checkpoint-based MTP contexts
  (`use_ckpt_dft`; the iSWA cache takes partial seq_rm, N16), probabilistic drafting.
* The guard `slot_replay_fits` of the patch is not reached by any N21 run (every replay targets cells the `seq_rm` just
  before freed): mutant P5 survives (2.4).
* Other architectures with several NextN heads: UNKNOWN (2.1).
* Timings of the patched process (CPU or GPU): NOT RUN.
* GitHub Actions for the new step and job: NOT RUN (branch not pushed).

## 6. Literal outputs

`<S>` replaces the scratch directory in every output; nothing else is edited.

### 6.1 Environment, builds, the patch

```
$ python3 --version; cmake --version | head -1; c++ --version | head -1; nproc
Python 3.13.16
cmake version 3.28.3
c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
4
```

Builds of the `tools/g0` target `dump-tensors` from this worktree (`GGML_NATIVE=OFF`): the unpatched pins (`<S>/n21/build.sh`) and the CI-style patched copy `<S>/n21/llama-r1-patched` (`git clone --shared` of the 79e2e74 checkout, then `git apply` of the patch; HEAD stays the pin):

```
r1 cmake 4 s, build 1075 s
ec cmake 5 s, build 702 s
BUILDS-DONE
r1p rc=0 build 587 s
```

```
$ git -C <S>/n21/llama-r1-patched rev-parse HEAD; git -C <S>/n21/llama-r1-patched diff --stat
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
 common/speculative.cpp      | 55 ++++++++++++++++++++++++++++++++++++++++++++-
 src/llama-context.cpp       |  4 ++++
 src/llama-ext.h             |  7 ++++++
 src/llama-kv-cache-iswa.cpp |  7 ++++++
 src/llama-kv-cache-iswa.h   |  2 ++
 src/llama-kv-cache.cpp      | 54 ++++++++++++++++++++++++++++++++++++++++++++
 src/llama-kv-cache.h        |  9 ++++++++
 src/llama-memory.h          |  5 +++++
 8 files changed, 142 insertions(+), 1 deletion(-)
$ sha256sum patches/llama.cpp/0001-mtp-heads-slot-replay.patch
1bced1b46dc3dd9ff8298df3ebbc5669355a3bfe597113834255ae32a0af52b1  patches/llama.cpp/0001-mtp-heads-slot-replay.patch
```

The mutation check (6.8) builds from `<S>/n21/llama-r1-fix`, a scratch git repo holding 79e2e74 as one commit and the patch as the next (its `git format-patch` equals the patch file apart from the index lines), build `<S>/n21/build-r1fix` (15 min 25 s).

### 6.2 The N21 step at both pins and on the patched copy, through tools/ci/check_unittest.py

`ONLY='^n21-' bash docs/evidence/mutation-scripts/2026-10-11-n21-final-runs.sh` on `3918f04` (the last commit that changed code or tests; the one untracked file was this evidence file):

```
worktree HEAD 3918f04719003df8cb3752bd5fa44f3c39a5f6b4; status: 1 changed or untracked files
n21-mimo2-mtp-fix @ec (ec7630a) wrapper_rc=0 wall_s=18: Ran 32 tests in 9.326s OK (skipped=5) ::notice::n21-mimo2-mtp-fix @ llama.cpp ec7630a: Ran 32, skipped 5 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_independent_of_the_mtp_order); the driver's checkout <S>/llama-ec7630a is at ec7630a
n21-mimo2-mtp-fix @r1 (79e2e74) wrapper_rc=0 wall_s=18: Ran 32 tests in 8.921s OK (skipped=5) ::notice::n21-mimo2-mtp-fix @ llama.cpp 79e2e74: Ran 32, skipped 5 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_independent_of_the_mtp_order); the driver's checkout <S>/llama.cpp is at 79e2e74
n21-mimo2-mtp-fix-patched @r1p (79e2e74 + patches/llama.cpp/0001) wrapper_rc=0 wall_s=59: Ran 54 tests in 20.654s OK ::notice::n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/n21/llama-r1-patched is at 79e2e74
FINAL-DONE
```

### 6.3 The patched step in detail (`<S>/n21/final-n21-mimo2-mtp-fix-patched-r1p.log`, excerpts)

The driver's runs:
```
mtp3: 87 MTP graphs, {'draft': 15, 'process': 63, 'rollback': 3, 'seq_rm': 4} (replay)
mtp3b: 54 MTP graphs, {'draft': 6, 'process': 36, 'rollback': 1, 'seq_rm': 1} (replay)
spec/mtp3: 87 MTP graphs, {'rollback': 3, 'seq_rm': 4, 'spec_draft': 5, 'spec_process': 21} (DUMP_MTP_SPEC=1)
spec/mtp3b: 54 MTP graphs, {'rollback': 1, 'seq_rm': 1, 'spec_draft': 2, 'spec_process': 12} (DUMP_MTP_SPEC=1)
fix/mtp3: 87 MTP graphs, {'draft': 15, 'process': 72, 'rollback': 3, 'seq_rm': 4} (DUMP_MTP_FIX=1)
fix/mtp3b: 54 MTP graphs, {'draft': 6, 'process': 48, 'rollback': 1, 'seq_rm': 1} (DUMP_MTP_FIX=1)
fix/fa-mtp3: 87 MTP graphs, {'draft': 15, 'process': 72, 'rollback': 3, 'seq_rm': 4} (DUMP_MTP_FIX=1)
spec-off/mtp3: 87 MTP graphs, {'rollback': 3, 'seq_rm': 4, 'spec_draft': 5, 'spec_process': 21} (DUMP_MTP_SPEC=1 LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1)
spec-off/mtp3b: 54 MTP graphs, {'rollback': 1, 'seq_rm': 1, 'spec_draft': 2, 'spec_process': 12} (DUMP_MTP_SPEC=1 LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1)
patched: true
```

The tests' own summary lines (stderr), in order:
```
fix/mtp3: 11 rows wrong upstream (mtp3), 8 compared with (A): fixed kqv_out 5.832e-07, logits 3.639e-07 (8 output rows); upstream kqv_out 2.053e+00, logits 4.039e-01
fix/mtp3b: 23 rows wrong upstream (mtp3b), 22 compared with (A): fixed kqv_out 7.469e-07, logits 6.301e-07 (22 output rows); upstream kqv_out 1.735e+00, logits 6.518e-01
mtp3: head-0 rows reading foreign K/V: 8 graphs, positions [257, 258, 259, 260, 261, 262, 263]; worst kqv_out vs (A) 2.053e+00, logits 4.039e-01
mtp3: draft at 259: fixed [37, 66, 43, 85], upstream [37, 32, 77, 94]
mtp3b: draft at 307: fixed [95, 18, 19, 66], upstream [95, 20, 4, 55]
spec/mtp3 vs fix/mtp3: 87 MTP graphs, 2742 tensors bit-identical, 216 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
spec/mtp3b vs fix/mtp3b: 54 MTP graphs, 1692 tensors bit-identical, 144 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
spec-off/mtp3 vs mtp3: 87 MTP graphs, 2742 tensors bit-identical, 216 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
spec-off/mtp3b vs mtp3b: 54 MTP graphs, 1692 tensors bit-identical, 144 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
```

Selected rows of the error table (run, component, head, graph, max abs, rel):
```
fix/mtp3 h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
fix/mtp3 8 rows upstream got wrong: kqv_out vs (A), fixed   -1    -1  0.000e+00  5.832e-07
fix/mtp3 8 of them outputs: logits vs (A), fixed        -1    -1  0.000e+00  3.639e-07
fix/mtp3 the same rows upstream (mtp3): kqv_out vs (A) (expected to differ)   -1    -1  0.000e+00  2.053e+00
fix/mtp3 the same rows upstream (mtp3): logits vs (A) (expected to differ)   -1    -1  0.000e+00  4.039e-01
fix/mtp3b h rows fed, bit-exact pairing checked on 60864 elements   -1    -1  0.000e+00  0.000e+00
fix/mtp3b 22 rows upstream got wrong: kqv_out vs (A), fixed   -1    -1  0.000e+00  7.469e-07
fix/mtp3b 22 of them outputs: logits vs (A), fixed       -1    -1  0.000e+00  6.301e-07
fix/mtp3b the same rows upstream (mtp3b): kqv_out vs (A) (expected to differ)   -1    -1  0.000e+00  1.735e+00
fix/mtp3b the same rows upstream (mtp3b): logits vs (A) (expected to differ)   -1    -1  0.000e+00  6.518e-01
fix/fa-mtp3 flash attention op (fa_tiled, own inputs)       0     0  3.864e-07  2.182e-07
fix/fa-mtp3 flash attention op (fa_tiled, own inputs)       1     7  5.426e-07  2.285e-07
fix/fa-mtp3 flash attention op (fa_tiled, own inputs)       2     2  4.655e-07  1.984e-07
fix/fa-mtp3 flash attention op (one_chunk): elements > 0.5 spacing 0/17280   -1    -1  0.000e+00  0.000e+00
fix/fa-mtp3 h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
fix/fa-mtp3 cache store: 67680 elements equal to the rounded pre-store value   -1    -1  0.000e+00  0.000e+00
mtp3     h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
mtp3     head 0 kqv_out vs (A), 11 rows reading foreign K/V (expected to differ)    0    -1  0.000e+00  2.053e+00
mtp3     head 0 logits vs (A), the same rows (expected to differ)    0    -1  0.000e+00  4.039e-01
mtp3b    h rows fed, bit-exact pairing checked on 60864 elements   -1    -1  0.000e+00  0.000e+00
mtp3b    kqv_out vs (A), rows with foreign K/V or an incomplete window (expected to differ)   -1    -1  0.000e+00  1.735e+00
```

Worst error per run and layer (every component, both oracles):
```
MTP heads: worst error per run and layer (head k = layer n_layer + k; the head-0 rows that read foreign K/V against (B) only), llama.cpp 79e2e74:
run      layer component                                    graph    max abs        rel
fix/fa-mtp3     6 kqv_out [full]                                  66  9.229e-04  1.317e-03
fix/fa-mtp3     7 kqv_out [full]                                  82  2.186e-03  1.597e-03
fix/fa-mtp3     8 kqv_out [full]                                  26  1.330e-03  1.289e-03
fix/mtp3     6 mtp_Kcur (RoPE) [full]                          78  3.284e-06  1.104e-06
fix/mtp3     7 mtp_Qcur (RoPE) [full]                          25  2.489e-06  1.090e-06
fix/mtp3     8 mtp_Qcur (RoPE) [full]                          71  3.749e-06  1.321e-06
fix/mtp3b     6 mtp_Kcur (RoPE) [full]                          36  4.061e-06  1.824e-06
fix/mtp3b     7 mtp_Kcur (RoPE) [full]                          37  5.467e-06  2.370e-06
fix/mtp3b     8 mtp_Qcur (RoPE) [chunk]                         38  3.779e-06  1.749e-06
mtp3         6 mtp_Kcur (RoPE) [full]                          78  3.284e-06  1.104e-06
mtp3         7 mtp_Qcur (RoPE) [full]                          25  2.489e-06  1.090e-06
mtp3         8 mtp_Qcur (RoPE) [full]                          71  3.749e-06  1.321e-06
mtp3b        6 mtp_Kcur (RoPE) [full]                          36  4.061e-06  1.824e-06
mtp3b        7 mtp_Kcur (RoPE) [full]                          37  5.467e-06  2.370e-06
mtp3b        8 mtp_Qcur (RoPE) [chunk]                         38  3.779e-06  1.749e-06
```

The tests (54; docstrings and results omitted, every one ok per the summary):
```
test_draft_tokens (tests.test_mimo2_mtp_fix_parity.MtpFix.test_draft_tokens)
test_graphs_heads_and_pairing (tests.test_mimo2_mtp_fix_parity.MtpFix.test_graphs_heads_and_pairing)
test_head0_reads_foreign_kv (tests.test_mimo2_mtp_fix_parity.MtpFix.test_head0_reads_foreign_kv)
test_kv_cells (tests.test_mimo2_mtp_fix_parity.MtpFix.test_kv_cells)
test_mtp_heads_block (tests.test_mimo2_mtp_fix_parity.MtpFix.test_mtp_heads_block)
test_n17_rows (tests.test_mimo2_mtp_fix_parity.MtpFix.test_n17_rows) ... ok
test_run_exercises_its_mechanisms (tests.test_mimo2_mtp_fix_parity.MtpFix.test_run_exercises_its_mechanisms) ... ok
test_upstream_rows_now_match_full (tests.test_mimo2_mtp_fix_parity.MtpFix.test_upstream_rows_now_match_full) ... 
test_draft_tokens (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_draft_tokens)
test_graphs_heads_and_pairing (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_graphs_heads_and_pairing)
test_head0_reads_foreign_kv (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_head0_reads_foreign_kv)
test_kv_cells (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_kv_cells)
test_mtp_heads_block (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_mtp_heads_block)
test_run_exercises_its_mechanisms (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_run_exercises_its_mechanisms) ... ok
test_upstream_rows_now_match_full (tests.test_mimo2_mtp_fix_parity.MtpFixB.test_upstream_rows_now_match_full) ... 
test_draft_tokens (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_draft_tokens)
test_flash_attention_op (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_flash_attention_op)
test_graphs_heads_and_pairing (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_graphs_heads_and_pairing)
test_head0_reads_foreign_kv (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_head0_reads_foreign_kv)
test_kv_cells (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_kv_cells)
test_mtp_heads_block (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_mtp_heads_block)
test_run_exercises_its_mechanisms (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.test_run_exercises_its_mechanisms) ... ok
test_cell_model_agrees (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_cell_model_agrees)
test_draft_tokens (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_draft_tokens)
test_graphs_heads_and_pairing (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_graphs_heads_and_pairing)
test_head0_reads_foreign_kv (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_head0_reads_foreign_kv)
test_kv_cells (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_kv_cells)
test_mtp_heads_block (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_mtp_heads_block)
test_run_exercises_its_mechanisms (tests.test_mimo2_mtp_fix_parity.MtpUpstream.test_run_exercises_its_mechanisms)
test_cell_model_agrees (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_cell_model_agrees)
test_draft_tokens (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_draft_tokens)
test_graphs_heads_and_pairing (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_graphs_heads_and_pairing)
test_head0_reads_foreign_kv (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_head0_reads_foreign_kv) ... ok
test_kv_cells (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_kv_cells)
test_mtp_heads_block (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_mtp_heads_block)
test_run_exercises_its_mechanisms (tests.test_mimo2_mtp_fix_parity.MtpUpstreamB.test_run_exercises_its_mechanisms) ... ok
test_cell_model_counts (tests.test_mimo2_mtp_fix_parity.SpecReplay.test_cell_model_counts)
test_draft_tokens_identical (tests.test_mimo2_mtp_fix_parity.SpecReplay.test_draft_tokens_identical) ... ok
test_fix_changes_drafts (tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts)
test_mtp_graphs_identical (tests.test_mimo2_mtp_fix_parity.SpecReplay.test_mtp_graphs_identical) ... 
test_target_identical (tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_identical) ... ok
test_target_independent_of_the_mtp_order (tests.test_mimo2_mtp_fix_parity.SpecReplay.test_target_independent_of_the_mtp_order)
test_long_prompt_first_drafts (tests.test_mimo2_mtp_fix_ref.MtpCellsModel.test_long_prompt_first_drafts)
test_n17_plan (tests.test_mimo2_mtp_fix_ref.MtpCellsModel.test_n17_plan)
test_ring_size (tests.test_mimo2_mtp_fix_ref.MtpCellsModel.test_ring_size)
test_server_exposure_mimo_geometry (tests.test_mimo2_mtp_fix_ref.MtpCellsModel.test_server_exposure_mimo_geometry)
test_call_of_several_ubatches (tests.test_mimo2_mtp_fix_ref.MtpFixRef.test_call_of_several_ubatches)
test_cell_model_agrees_with_the_oracle (tests.test_mimo2_mtp_fix_ref.MtpFixRef.test_cell_model_agrees_with_the_oracle)
test_fixed_equals_upstream_without_wrap (tests.test_mimo2_mtp_fix_ref.MtpFixRef.test_fixed_equals_upstream_without_wrap)
test_replay_refused_on_a_taken_cell (tests.test_mimo2_mtp_fix_ref.MtpFixRef.test_replay_refused_on_a_taken_cell) ... ok
test_wrap_of_one_call (tests.test_mimo2_mtp_fix_ref.MtpFixRef.test_wrap_of_one_call)
test_keep_one_head_is_n16 (tests.test_gguf_mtp_heads.GgufMtpHeads.test_keep_one_head_is_n16) ... ok
test_keep_two_heads (tests.test_gguf_mtp_heads.GgufMtpHeads.test_keep_two_heads) ... ok
test_refuses_more_heads_than_the_file_has (tests.test_gguf_mtp_heads.GgufMtpHeads.test_refuses_more_heads_than_the_file_has) ... ok
```
```
----------------------------------------------------------------------
Ran 54 tests in 20.654s

OK
artifacts: <S>/n21/tmp/work-n21-mimo2-mtp-fix-patched-r1p
::notice::n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/n21/llama-r1-patched is at 79e2e74
```

### 6.4 The unpatched step at 79e2e74 and ec7630a (excerpts)

79e2e74:
```
mtp3: 87 MTP graphs, {'draft': 15, 'process': 63, 'rollback': 3, 'seq_rm': 4} (replay)
mtp3b: 54 MTP graphs, {'draft': 6, 'process': 36, 'rollback': 1, 'seq_rm': 1} (replay)
spec/mtp3: 87 MTP graphs, {'rollback': 3, 'seq_rm': 4, 'spec_draft': 5, 'spec_process': 21} (DUMP_MTP_SPEC=1)
spec/mtp3b: 54 MTP graphs, {'rollback': 1, 'seq_rm': 1, 'spec_draft': 2, 'spec_process': 12} (DUMP_MTP_SPEC=1)
patched: false
setUpClass (tests.test_mimo2_mtp_fix_parity.MtpFix) ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
setUpClass (tests.test_mimo2_mtp_fix_parity.MtpFixB) ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
setUpClass (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA) ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
mtp3: head-0 rows reading foreign K/V: 8 graphs, positions [257, 258, 259, 260, 261, 262, 263]; worst kqv_out vs (A) 2.053e+00, logits 4.039e-01
On the patched build: the fixed and the upstream code draft different tokens exactly where the upstream head ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
spec/mtp3 vs mtp3: 87 MTP graphs, 2742 tensors bit-identical, 216 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
spec/mtp3b vs mtp3b: 54 MTP graphs, 1692 tensors bit-identical, 144 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
On the patched build: the target's dumps (every graph, logits, h_nextn) of the upstream and the fixed replay ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
MTP heads: worst error per run, component and head over the MTP graphs, llama.cpp 79e2e74:
mtp3     h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
mtp3     head 0 kqv_out vs (A), 11 rows reading foreign K/V (expected to differ)    0    -1  0.000e+00  2.053e+00
mtp3     head 0 logits vs (A), the same rows (expected to differ)    0    -1  0.000e+00  4.039e-01
mtp3b    h rows fed, bit-exact pairing checked on 60864 elements   -1    -1  0.000e+00  0.000e+00
mtp3b    kqv_out vs (A), rows with foreign K/V or an incomplete window (expected to differ)   -1    -1  0.000e+00  1.735e+00
Ran 32 tests in 8.921s
OK (skipped=5)
```

ec7630a:
```
mtp3: 87 MTP graphs, {'draft': 15, 'process': 63, 'rollback': 3, 'seq_rm': 4} (replay)
mtp3b: 54 MTP graphs, {'draft': 6, 'process': 36, 'rollback': 1, 'seq_rm': 1} (replay)
spec/mtp3: 87 MTP graphs, {'rollback': 3, 'seq_rm': 4, 'spec_draft': 5, 'spec_process': 21} (DUMP_MTP_SPEC=1)
spec/mtp3b: 54 MTP graphs, {'rollback': 1, 'seq_rm': 1, 'spec_draft': 2, 'spec_process': 12} (DUMP_MTP_SPEC=1)
patched: false
setUpClass (tests.test_mimo2_mtp_fix_parity.MtpFix) ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
setUpClass (tests.test_mimo2_mtp_fix_parity.MtpFixB) ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
setUpClass (tests.test_mimo2_mtp_fix_parity.MtpFixF16FA) ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
mtp3: head-0 rows reading foreign K/V: 8 graphs, positions [257, 258, 259, 260, 261, 262, 263]; worst kqv_out vs (A) 2.053e+00, logits 4.039e-01
On the patched build: the fixed and the upstream code draft different tokens exactly where the upstream head ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
spec/mtp3 vs mtp3: 87 MTP graphs, 2742 tensors bit-identical, 216 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
spec/mtp3b vs mtp3b: 54 MTP graphs, 1692 tensors bit-identical, 144 post-crop tensors of process graphs with 0 rows (real code) instead of every row (replay)
On the patched build: the target's dumps (every graph, logits, h_nextn) of the upstream and the fixed replay ... skipped 'no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch'
MTP heads: worst error per run, component and head over the MTP graphs, llama.cpp ec7630a:
mtp3     h rows fed, bit-exact pairing checked on 54144 elements   -1    -1  0.000e+00  0.000e+00
mtp3     head 0 kqv_out vs (A), 11 rows reading foreign K/V (expected to differ)    0    -1  0.000e+00  2.053e+00
mtp3     head 0 logits vs (A), the same rows (expected to differ)    0    -1  0.000e+00  4.039e-01
mtp3b    h rows fed, bit-exact pairing checked on 60864 elements   -1    -1  0.000e+00  0.000e+00
mtp3b    kqv_out vs (A), rows with foreign K/V or an incomplete window (expected to differ)   -1    -1  0.000e+00  1.735e+00
Ran 32 tests in 9.326s
OK (skipped=5)
```

### 6.5 Probes: git apply --check, --override-kv, the one-head GGUF, --swa-full, one draft step

`bash docs/evidence/mutation-scripts/2026-10-11-n21-probes.sh` on `64009b9`:
```
== git apply --check of 0001-mtp-heads-slot-replay.patch (sha256 1bced1b46dc3dd9f...)
-- at 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
Checking patch common/speculative.cpp...
Checking patch src/llama-context.cpp...
Checking patch src/llama-ext.h...
Checking patch src/llama-kv-cache-iswa.cpp...
Checking patch src/llama-kv-cache-iswa.h...
Checking patch src/llama-kv-cache.cpp...
Checking patch src/llama-kv-cache.h...
Checking patch src/llama-memory.h...
-- at ec7630a640789c393694fb194f1bbbf0369fc62d
Checking patch common/speculative.cpp...
Checking patch src/llama-context.cpp...
Checking patch src/llama-ext.h...
error: while searching for:
error: patch failed: src/llama-ext.h:100
error: src/llama-ext.h: patch does not apply
Checking patch src/llama-kv-cache-iswa.cpp...
Checking patch src/llama-kv-cache-iswa.h...
Checking patch src/llama-kv-cache.cpp...
Checking patch src/llama-kv-cache.h...
Checking patch src/llama-memory.h...
== --override-kv mimo2.nextn_predict_layers=int:1 on the 3-head n17.gguf (dump tool, llama.cpp 79e2e74)
W load: SPM vocabulary, but newline token not found: unordered_map::at! Using special_pad_id instead.0.00.004.794 E llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 95, got 87
E llama_model_load_from_file_impl: failed to load model
E failed to init
exit status 1
== tools/gguf_mtp_heads.py --keep 1 n17.gguf (3 heads) vs N16's n16.gguf (mimo2_tiny.py --mtp)
{'arch': 'mimo2', 'n_layer': 6, 'heads': (3, 1), 'block_count': (9, 7), 'tensors': (95, 71), 'arrays_cut': ['mimo2.attention.head_count_kv', 'mimo2.attention.sliding_window_pattern']}
cmp: byte-identical (1644640 bytes)
== --swa-full: extra f16 KV per SWA layer, MiMo-V2.6-Flash config
SWA layers 39 of 48, MTP layers 3, KV heads 8, K 192, V 128, ring 768 cells
ctx 32768: per layer +163840000 B = 0.153 GiB; target 39 layers +5.95 GiB; MTP 3 layers +0.46 GiB; total +6.41 GiB
ctx 131072: per layer +667156480 B = 0.621 GiB; target 39 layers +24.23 GiB; MTP 3 layers +1.86 GiB; total +26.10 GiB
ctx 262144: per layer +1338245120 B = 1.246 GiB; target 39 layers +48.61 GiB; MTP 3 layers +3.74 GiB; total +52.35 GiB
== placement model, upstream, one draft step (--spec-draft-n-max 1): process still runs 3 heads
prompt 700, accept 0.3: drafts 9173, head-0 drafts on foreign K/V 918 (10.0%)
prompt 700, accept 0.6: drafts 7488, head-0 drafts on foreign K/V 917 (12.2%)
prompt 700, accept 0.85: drafts 6486, head-0 drafts on foreign K/V 802 (12.4%)
```

### 6.6 Every other dump-based CI step at both pins, N16 / N17 on the patched copy, dump-tool byte identity

`bash docs/evidence/mutation-scripts/2026-10-11-n21-final-runs.sh` on `8402c34` (the commits after it add one test to tests/test_mimo2_mtp_fix_parity.py with its expectations, re-run in 6.2, and change scripts and docs):

```
worktree HEAD 8402c342f1d1f806a5cb8265fb045d500c09e0a4; status: 0 changed or untracked files
n21-mimo2-mtp-fix @ec (ec7630a) wrapper_rc=0 wall_s=33: Ran 31 tests in 14.680s OK (skipped=4) ::notice::n21-mimo2-mtp-fix @ llama.cpp ec7630a: Ran 31, skipped 4 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts); the driver's checkout <S>/llama-ec7630a is at ec7630a
n21-mimo2-mtp-fix @r1 (79e2e74) wrapper_rc=0 wall_s=43: Ran 31 tests in 17.277s OK (skipped=4) ::notice::n21-mimo2-mtp-fix @ llama.cpp 79e2e74: Ran 31, skipped 4 as expected (tests.test_mimo2_mtp_fix_parity.MtpFix.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixB.setUpClass, tests.test_mimo2_mtp_fix_parity.MtpFixF16FA.setUpClass, tests.test_mimo2_mtp_fix_parity.SpecReplay.test_fix_changes_drafts); the driver's checkout <S>/llama.cpp is at 79e2e74
n21-mimo2-mtp-fix-patched @r1p (79e2e74 + patches/llama.cpp/0001) wrapper_rc=0 wall_s=89: Ran 53 tests in 38.579s OK ::notice::n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 53, skipped 0 as expected; the driver's checkout <S>/n21/llama-r1-patched is at 79e2e74
n17-mimo2-mtp-heads-fa @r1p (79e2e74 + patches/llama.cpp/0001) wrapper_rc=0 wall_s=64: Ran 70 tests in 50.171s OK ::notice::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 70, skipped 0 as expected; the driver's checkout <S>/n21/llama-r1-patched is at 79e2e74
n16-mimo2-decode @r1p (79e2e74 + patches/llama.cpp/0001) wrapper_rc=0 wall_s=45: Ran 92 tests in 19.546s OK ::notice::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 92, skipped 0 as expected; the driver's checkout <S>/n21/llama-r1-patched is at 79e2e74
g1-glm @ec (ec7630a) wrapper_rc=0 wall_s=3: Ran 9 tests in 0.616s OK ::notice::g1-glm @ llama.cpp ec7630a: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
g1-mimo @ec (ec7630a) wrapper_rc=0 wall_s=2: Ran 5 tests in 0.102s OK ::notice::g1-mimo @ llama.cpp ec7630a: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n2-qwen4exp @ec (ec7630a) wrapper_rc=0 wall_s=3: Ran 17 tests in 0.694s OK ::notice::n2-qwen4exp @ llama.cpp ec7630a: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n6-qwen4exp-decode @ec (ec7630a) wrapper_rc=0 wall_s=20: Ran 54 tests in 6.470s OK ::notice::n6-qwen4exp-decode @ llama.cpp ec7630a: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n9-glm5next-decode @ec (ec7630a) wrapper_rc=0 wall_s=24: Ran 52 tests in 6.943s OK (skipped=1) ::notice::n9-glm5next-decode @ llama.cpp ec7630a: Ran 52, skipped 1 as expected (tests.test_glm5next_mtp_parity.Mtp.setUpClass); the driver's checkout <S>/llama-ec7630a is at ec7630a
n10-f16fa @ec (ec7630a) wrapper_rc=0 wall_s=61: Ran 170 tests in 24.918s OK ::notice::n10-f16fa @ llama.cpp ec7630a: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n10-quant @ec (ec7630a) wrapper_rc=0 wall_s=232: Ran 99 tests in 167.484s OK ::notice::n10-quant @ llama.cpp ec7630a: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n16-mimo2-decode @ec (ec7630a) wrapper_rc=0 wall_s=37: Ran 92 tests in 18.857s OK ::notice::n16-mimo2-decode @ llama.cpp ec7630a: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n17-mimo2-mtp-heads-fa @ec (ec7630a) wrapper_rc=0 wall_s=98: Ran 70 tests in 62.694s OK ::notice::n17-mimo2-mtp-heads-fa @ llama.cpp ec7630a: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
n19-qwen4exp-multiseq @ec (ec7630a) wrapper_rc=0 wall_s=24: Ran 72 tests in 8.853s OK ::notice::n19-qwen4exp-multiseq @ llama.cpp ec7630a: Ran 72, skipped 0 as expected; the driver's checkout <S>/llama-ec7630a is at ec7630a
g1-glm @r1 (79e2e74) wrapper_rc=0 wall_s=3: Ran 9 tests in 0.419s OK ::notice::g1-glm @ llama.cpp 79e2e74: Ran 9, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
g1-mimo @r1 (79e2e74) wrapper_rc=0 wall_s=2: Ran 5 tests in 0.112s OK ::notice::g1-mimo @ llama.cpp 79e2e74: Ran 5, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n2-qwen4exp @r1 (79e2e74) wrapper_rc=0 wall_s=2: Ran 17 tests in 0.371s OK ::notice::n2-qwen4exp @ llama.cpp 79e2e74: Ran 17, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n6-qwen4exp-decode @r1 (79e2e74) wrapper_rc=0 wall_s=28: Ran 54 tests in 4.115s OK ::notice::n6-qwen4exp-decode @ llama.cpp 79e2e74: Ran 54, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n9-glm5next-decode @r1 (79e2e74) wrapper_rc=0 wall_s=36: Ran 58 tests in 6.893s OK ::notice::n9-glm5next-decode @ llama.cpp 79e2e74: Ran 58, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n10-f16fa @r1 (79e2e74) wrapper_rc=0 wall_s=60: Ran 170 tests in 28.017s OK ::notice::n10-f16fa @ llama.cpp 79e2e74: Ran 170, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n10-quant @r1 (79e2e74) wrapper_rc=0 wall_s=239: Ran 99 tests in 189.940s OK ::notice::n10-quant @ llama.cpp 79e2e74: Ran 99, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n16-mimo2-decode @r1 (79e2e74) wrapper_rc=0 wall_s=32: Ran 92 tests in 12.776s OK ::notice::n16-mimo2-decode @ llama.cpp 79e2e74: Ran 92, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n17-mimo2-mtp-heads-fa @r1 (79e2e74) wrapper_rc=0 wall_s=59: Ran 70 tests in 37.608s OK ::notice::n17-mimo2-mtp-heads-fa @ llama.cpp 79e2e74: Ran 70, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
n19-qwen4exp-multiseq @r1 (79e2e74) wrapper_rc=0 wall_s=25: Ran 72 tests in 10.022s OK ::notice::n19-qwen4exp-multiseq @ llama.cpp 79e2e74: Ran 72, skipped 0 as expected; the driver's checkout <S>/llama.cpp is at 79e2e74
FINAL-DONE
```

Dump tool of this tree against `e71b241` (`bash docs/evidence/mutation-scripts/2026-10-11-n21-byteid.sh /home/user/wt-n21 e71b241 <build> <checkout> <out>` on `5fe2ea5`; `tools/dump/dump_tensors.cpp` has not changed since `add5811`), 79e2e74 then ec7630a:

```
llama.cpp 79e2e74; base dump_tensors.cpp @e71b241 vs 5fe2ea5
default: 900 files; diff -r: identical
ub10: 6289 files; diff -r: identical
rollback: 12407 files; diff -r: identical
mtp: 12376 files; diff -r: identical
mtp3: 17174 files; diff -r: identical
mtp3b: 11124 files; diff -r: identical
fa-mtp3: 14498 files; diff -r: identical
```
```
llama.cpp ec7630a; base dump_tensors.cpp @e71b241 vs 5fe2ea5
default: 900 files; diff -r: identical
ub10: 6289 files; diff -r: identical
rollback: 12407 files; diff -r: identical
mtp: 12428 files; diff -r: identical
mtp3: 17174 files; diff -r: identical
mtp3b: 11124 files; diff -r: identical
fa-mtp3: 14498 files; diff -r: identical
```

### 6.7 Exposure estimate (PROVISIONAL model numbers)

`python3 tools/mtp_ring_exposure.py` on `3918f04`:
```
SWA ring of the MTP context: 768 cells (pad(min(262144, 128 + 512), 256)); 3 heads, verify batches of 4, prompt calls of 2048, gen 4000, seeds [1, 2, 3]
mode      prompt accept  drafts  head-0 drafts on foreign K/V   share  first heads 1.. proc ub incompl. window
upstream     100   0.30    8389                           786    9.4%    766         0                       0
upstream     100   0.60    5522                           415    7.5%    766         0                       0
upstream     100   0.85    3773                           123    3.3%   1532         0                       0
upstream     700   0.30    8389                          1294   15.4%    767         0                       0
upstream     700   0.60    5522                           392    7.1%   1534         0                       0
upstream     700   0.85    3773                           304    8.1%    768         0                       0
upstream    2000   0.30    8389                          1172   14.0%   2302         0                       0
upstream    2000   0.60    5522                           295    5.3%   2305         0                       0
upstream    2000   0.85    3773                           316    8.4%   2302         0                       0
upstream    5000   0.30    8389                          1334   15.9%   5000         0                      12
upstream    5000   0.60    5522                           476    8.6%   5000         0                      12
upstream    5000   0.85    3773                           319    8.5%   5000         0                      12
fixed        100   0.30    8389                             0    0.0%      -         0                       0
fixed        100   0.60    5522                             0    0.0%      -         0                       0
fixed        100   0.85    3773                             0    0.0%      -         0                       0
fixed        700   0.30    8389                             0    0.0%      -         0                       0
fixed        700   0.60    5522                             0    0.0%      -         0                       0
fixed        700   0.85    3773                             0    0.0%      -         0                       0
fixed       2000   0.30    8389                             0    0.0%      -         0                       0
fixed       2000   0.60    5522                             0    0.0%      -         0                       0
fixed       2000   0.85    3773                             0    0.0%      -         0                       0
fixed       5000   0.30    8389                             0    0.0%      -         0                       0
fixed       5000   0.60    5522                             0    0.0%      -         0                       0
fixed       5000   0.85    3773                             0    0.0%      -         0                       0
one-head     100   0.30    8389                             0    0.0%      -         0                       0
one-head     100   0.60    5522                             0    0.0%      -         0                       0
one-head     100   0.85    3773                             0    0.0%      -         0                       0
one-head     700   0.30    8389                             0    0.0%      -         0                       0
one-head     700   0.60    5522                             0    0.0%      -         0                       0
one-head     700   0.85    3773                             0    0.0%      -         0                       0
one-head    2000   0.30    8389                             0    0.0%      -         0                       0
one-head    2000   0.60    5522                             0    0.0%      -         0                       0
one-head    2000   0.85    3773                             0    0.0%      -         0                       0
one-head    5000   0.30    8389                             0    0.0%      -         0                       0
one-head    5000   0.60    5522                             0    0.0%      -         0                       0
one-head    5000   0.85    3773                             0    0.0%      -         0                       0
swa-full     100   0.30    8389                             0    0.0%      -         0                       0
swa-full     100   0.60    5522                             0    0.0%      -         0                       0
swa-full     100   0.85    3773                             0    0.0%      -         0                       0
swa-full     700   0.30    8389                             0    0.0%      -         0                       0
swa-full     700   0.60    5522                             0    0.0%      -         0                       0
swa-full     700   0.85    3773                             0    0.0%      -         0                       0
swa-full    2000   0.30    8389                             0    0.0%      -         0                       0
swa-full    2000   0.60    5522                             0    0.0%      -         0                       0
swa-full    2000   0.85    3773                             0    0.0%      -         0                       0
swa-full    5000   0.30    8389                             0    0.0%      -         0                       0
swa-full    5000   0.60    5522                             0    0.0%      -         0                       0
swa-full    5000   0.85    3773                             0    0.0%      -         0                       0

real	8m7.661s
user	7m56.329s
sys	0m0.405s
```

### 6.8 Mutation check (18 mutants)

`python3 docs/evidence/mutation-scripts/2026-10-11-n21-mutations.py /home/user/wt-n21 <S>/n21/llama-r1-fix <S>/n21/build-r1fix <S>/n21/mut` on `3918f04` (the Python mutants run 51 tests: the two dump modules; the C++ mutants run the driver, 54):
```
patched llama.cpp <S>/n21/llama-r1-fix: cb9cd77 common/speculative: place each MTP process ubatch once for all heads (slot replay); build <S>/n21/build-r1fix; worktree 3918f04719003df8cb3752bd5fa44f3c39a5f6b4
baseline (no mutation), driver: rc=0 Ran 54 tests in 23.775s | OK
O1 oracle: fixed process without replay (heads 1..H-1 search their own cells)
    ref/mimo2.py: 'prec=prec, replay=rep)' -> 'prec=prec, replay=None)'
    rc=1 Ran 51 tests in 21.799s | FAILED (failures=17)
    test_mimo2_mtp_fix_parity: 14 failing: MtpFix.test_draft_tokens, MtpFix.test_graphs_heads_and_pairing, MtpFix.test_head0_reads_foreign_kv, MtpFix.test_kv_cells, MtpFix.test_mtp_heads_block, MtpFixB.test_head0_reads_foreign_kv ...
    test_mimo2_mtp_fix_ref: 3 failing: MtpFixRef.test_call_of_several_ubatches, MtpFixRef.test_cell_model_agrees_with_the_oracle, MtpFixRef.test_wrap_of_one_call
O2 oracle: fixed process seq_rm from the call's first position, not the chunk's
    ref/mimo2.py: 'cache_truncate(cache, p0 + j)' -> 'cache_truncate(cache, p0)'
    rc=1 Ran 29 tests in 19.499s | FAILED (failures=1, errors=4)
    test_mimo2_mtp_fix_parity: 3 failing: MtpFix.setUpClass, MtpFixB.setUpClass, MtpFixF16FA.setUpClass
    test_mimo2_mtp_fix_ref: 2 failing: MtpFixRef.test_call_of_several_ubatches, MtpFixRef.test_cell_model_agrees_with_the_oracle
O3 oracle: fixed process without chunks (one ubatch per head for the whole call)
    ref/mimo2.py: 'for j in range(0, len(tokens), n_ubatch):\n            rep = None' -> 'n_ubatch = len(tokens)\n        for j in range(0, len(tokens), n_ubatch):\n            rep ='
    rc=1 Ran 29 tests in 18.629s | FAILED (errors=5)
    test_mimo2_mtp_fix_parity: 3 failing: MtpFix.setUpClass, MtpFixB.setUpClass, MtpFixF16FA.setUpClass
    test_mimo2_mtp_fix_ref: 2 failing: MtpFixRef.test_call_of_several_ubatches, MtpFixRef.test_cell_model_agrees_with_the_oracle
O4 oracle: incomplete windows never reported (attn["missing"] all False)
    ref/mimo2.py: 'missing = held[pos + 1] - held[lo] < pos - lo + 1' -> 'missing = np.zeros(len(pos), dtype=bool)'
    rc=1 Ran 51 tests in 20.929s | FAILED (failures=5)
    test_mimo2_mtp_fix_parity: 3 failing: MtpUpstreamB.test_cell_model_agrees, MtpUpstreamB.test_head0_reads_foreign_kv, MtpUpstreamB.test_mtp_heads_block
    test_mimo2_mtp_fix_ref: 2 failing: MtpFixRef.test_call_of_several_ubatches, MtpFixRef.test_cell_model_agrees_with_the_oracle
O5 oracle: replay accepted onto a taken cell
    ref/mimo2.py: 'if len(idxs) != n or any(cache[which].pos[i] >= 0 for i in idxs):' -> 'if len(idxs) != n:'
    rc=1 Ran 51 tests in 21.601s | FAILED (failures=1)
    test_mimo2_mtp_fix_ref: 1 failing: MtpFixRef.test_replay_refused_on_a_taken_cell
C1 cell model: the fixed order behaves as upstream
    ref/mtp_cells.py: 'if not self.fixed:\n            for head in range(self.H):' -> 'if True:\n            for head in range(self.H):'
    rc=1 Ran 51 tests in 22.349s | FAILED (failures=7)
    test_mimo2_mtp_fix_parity: 3 failing: MtpFix.test_head0_reads_foreign_kv, MtpFixB.test_head0_reads_foreign_kv, MtpFixF16FA.test_head0_reads_foreign_kv
    test_mimo2_mtp_fix_ref: 4 failing: MtpCellsModel.test_long_prompt_first_drafts, MtpCellsModel.test_n17_plan, MtpCellsModel.test_server_exposure_mimo_geometry, MtpFixRef.test_cell_model_agrees_with_the_oracle
C2 cell model: the window one position wider (p - pos_cell <= n_swa)
    ref/mtp_cells.py: 'vis = (cp >= 0) & (cp <= p) & (p - cp < self.c.n_swa)' -> 'vis = (cp >= 0) & (cp <= p) & (p - cp <= self.c.n_swa)'
    rc=1 Ran 51 tests in 20.693s | FAILED (failures=6)
    test_mimo2_mtp_fix_parity: 4 failing: MtpFix.test_n17_rows, MtpFixB.test_upstream_rows_now_match_full, MtpUpstream.test_cell_model_agrees, MtpUpstreamB.test_cell_model_agrees
    test_mimo2_mtp_fix_ref: 2 failing: MtpCellsModel.test_n17_plan, MtpFixRef.test_cell_model_agrees_with_the_oracle
C3 cell model: no seq_rm before each draft step
    ref/mtp_cells.py: 'for k in range(min(n_steps, self.H)):\n                self.seq_rm(pos0)' -> 'for k in range(min(n_steps, self.H)):\n                pass'
    rc=1 Ran 51 tests in 19.551s | FAILED (failures=11)
    test_mimo2_mtp_fix_parity: 8 failing: MtpFix.test_head0_reads_foreign_kv, MtpFix.test_n17_rows, MtpFix.test_upstream_rows_now_match_full, MtpFixB.test_head0_reads_foreign_kv, MtpFixB.test_upstream_rows_now_match_full, MtpFixF16FA.test_head0_reads_foreign_kv ...
    test_mimo2_mtp_fix_ref: 3 failing: MtpCellsModel.test_long_prompt_first_drafts, MtpCellsModel.test_n17_plan, MtpCellsModel.test_server_exposure_mimo_geometry
T1 test: fixed records paired head-major instead of chunk-major
    tests/test_mimo2_mtp_heads_parity.py: 'c, head = divmod(k, cls.H)' -> 'head, c = divmod(k, n_chunks)'
    rc=1 Ran 51 tests in 19.533s | FAILED (failures=6, errors=8)
    test_mimo2_mtp_fix_parity: 14 failing: MtpFix.test_graphs_heads_and_pairing, MtpFix.test_head0_reads_foreign_kv, MtpFix.test_kv_cells, MtpFix.test_mtp_heads_block, MtpFixB.test_graphs_heads_and_pairing, MtpFixB.test_head0_reads_foreign_kv ...
T2 test: the fixed run's h rows of chunk c taken from the call's start
    tests/test_mimo2_mtp_heads_parity.py: 'want = full[off:off + rec["n"]]' -> 'want = full[:rec["n"]]'
    rc=1 Ran 51 tests in 21.529s | FAILED (failures=3)
    test_mimo2_mtp_fix_parity: 3 failing: MtpFix.test_graphs_heads_and_pairing, MtpFixB.test_graphs_heads_and_pairing, MtpFixF16FA.test_graphs_heads_and_pairing
P1 llama_kv_cache::prepare ignores the slot replay (always searches)
    src/llama-kv-cache.cpp: 'if (slot_replay) {\n        if (slot_replay_fits(ubatches)) {' -> 'if (false) {\n        if (slot_replay_fits(ubatches)) {'
    rc=1 Ran 54 tests in 28.404s | FAILED (failures=14)
    test_mimo2_mtp_fix_parity: 14 failing: MtpFix.test_draft_tokens, MtpFix.test_graphs_heads_and_pairing, MtpFix.test_kv_cells, MtpFix.test_mtp_heads_block, MtpFix.test_upstream_rows_now_match_full, MtpFixB.test_kv_cells ...
P2 process_heads_once without replay (heads 1..H-1 search their own cells)
    common/speculative.cpp: 'llama_memory_set_slot_replay(mem_dft, head > 0);' -> 'llama_memory_set_slot_replay(mem_dft, false);'
    rc=1 Ran 54 tests in 26.072s | FAILED (failures=3)
    test_mimo2_mtp_fix_parity: 3 failing: SpecReplay.test_draft_tokens_identical, SpecReplay.test_fix_changes_drafts, SpecReplay.test_mtp_graphs_identical
P3 process_heads_once without chunks (every head on the whole batch, with replay)
    common/speculative.cpp: 'const int32_t n_ub = std::max<int32_t>(1, (int32_t) llama_n_ubatch(ctx_dft));' -> 'const int32_t n_ub = std::max<int32_t>(1, n_tokens);'
    rc=1 Ran 54 tests in 24.378s | FAILED (failures=1)
    test_mimo2_mtp_fix_parity: 1 failing: SpecReplay.test_mtp_graphs_identical
P4 process_heads_once: seq_rm from the batch's first position instead of the chunk's
    common/speculative.cpp: 'llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[k].pos[0], -1);' -> 'llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[0].pos[0], -1);'
    rc=1 no unittest summary: the driver stopped before the tests: 0.01.759.875 E spec process_head: llama_process(ctx_dft) head=0 failed rc=-1 (pos=108) / 0.01.759.882 E spec_process failed / CMAKE_BUILD_TYPE=Release
P5 slot_replay_fits does not check that a replayed cell is free
    src/llama-kv-cache.cpp: 'if (idx >= cells.size() || (!cells.is_empty(idx) && taken[strm].count(idx) == 0)) {' -> 'if (idx >= cells.size()) {'
    rc=0 Ran 54 tests in 28.847s | OK
D1 DUMP_MTP_FIX replays without slot replay
    tools/dump/dump_tensors.cpp: 'if (!llama_memory_set_slot_replay(mem, head > 0)) {' -> 'if (!llama_memory_set_slot_replay(mem, false)) {'
    rc=1 Ran 54 tests in 25.912s | FAILED (failures=15)
    test_mimo2_mtp_fix_parity: 15 failing: MtpFix.test_draft_tokens, MtpFix.test_graphs_heads_and_pairing, MtpFix.test_kv_cells, MtpFix.test_mtp_heads_block, MtpFix.test_upstream_rows_now_match_full, MtpFixB.test_kv_cells ...
D2 DUMP_MTP_FIX without chunks (every head on the whole call, with replay)
    tools/dump/dump_tensors.cpp: 'const int n_ub = (int) llama_n_ubatch(mtp_ctx.get());' -> 'const int n_ub = (int) ids.size();'
    rc=1 Ran 32 tests in 19.127s | FAILED (failures=1, errors=3)
    test_mimo2_mtp_fix_parity: 4 failing: MtpFix.setUpClass, MtpFixB.setUpClass, MtpFixF16FA.setUpClass, SpecReplay.test_mtp_graphs_identical
D3 DUMP_MTP_SPEC: accept() told one draft more than the rejection kept
    tools/dump/dump_tensors.cpp: 'common_speculative_accept(spec.get(), 0, (uint16_t) (last_call_n - n - 1));' -> 'common_speculative_accept(spec.get(), 0, (uint16_t) (last_call_n - n));'
    rc=1 Ran 54 tests in 27.835s | FAILED (failures=2)
    test_mimo2_mtp_fix_parity: 2 failing: SpecReplay.test_draft_tokens_identical, SpecReplay.test_mtp_graphs_identical
after the C++ mutants, restored sources (patched llama.cpp status: clean; worktree tools/dump/dump_tensors.cpp: unchanged), driver: rc=0 Ran 54 tests in 24.373s | OK
caught 17/18
```

### 6.9 Source lines

`bash docs/evidence/mutation-scripts/2026-10-11-n21-source-lines.sh` (worktree lines at `b3a728f`):
```
== common/speculative.cpp:1481-1486 @79e2e74
1481	        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
1482	        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
1483	
1484	        if (chain_heads) {
1485	            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);
1486	
== common/speculative.cpp:1583-1593 @79e2e74
1583	            for (int k = 0; k < n_tokens; ++k) {
1584	                const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
1585	
1586	                const int32_t idx = batch.add(batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false);
1587	
1588	                const float * h_row = k == i_batch_beg[seq_id]
1589	                    ? pending_h[seq_id].data()
1590	                    : h_tgt + (size_t) (k - 1) * n_embd;
1591	
1592	                batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
1593	            }
== common/speculative.cpp:1595-1621 @79e2e74
1595	            auto * mem_dft = llama_get_memory(ctx_dft);
1596	
1597	            bool ok = true;
1598	            for (int head = 0; head < n_mtp_layers; ++head) {
1599	                if (chain_heads) {
1600	                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
1601	                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
1602	                        if (i_batch_beg[seq_id] < 0) {
1603	                            continue;
1604	                        }
1605	                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
1606	                    }
1607	                    llama_set_nextn_layer_offset(ctx_dft, head);
1608	                }
1609	
1610	                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
1611	                if (rc != 0) {
1612	                    SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
1613	                            head, (int) rc, (int) batch_in.tokens[0].pos[0]);
1614	                    ok = false;
1615	                    break;
1616	                }
1617	            }
1618	
1619	            if (chain_heads) {
1620	                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
1621	            }
== common/speculative.cpp:1627-1642 @79e2e74
1627	        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
1628	            if (i_batch_end[seq_id] < 0) {
1629	                continue;
1630	            }
1631	
1632	            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
1633	            verify_h_rows[seq_id] = n_rows;
1634	            verify_h[seq_id].resize((size_t) n_rows * n_embd);
1635	
1636	            for (int32_t i = 0; i < n_rows; ++i) {
1637	                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
1638	                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
1639	            }
1640	
1641	            std::memcpy(pending_h[seq_id].data(),
1642	                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
== common/speculative.cpp:1663-1669 @79e2e74
1663	
1664	            n_drafting++;
1665	            drafting[seq_id] = true;
1666	            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
1667	            if (!params.probabilistic) {
1668	                dp.result_q = nullptr;
1669	            }
== common/speculative.cpp:1696-1708 @79e2e74
1696	            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
1697	            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
1698	            // and select head i so it rebuilds its own layer's KV there; decoding just the
1699	            // latest token would leave its attention reading cells only another head wrote.
1700	            if (chain_heads) {
1701	                auto * mem_dft = llama_get_memory(ctx_dft);
1702	                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
1703	                    if (drafting[seq_id]) {
1704	                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
1705	                    }
1706	                }
1707	                llama_set_nextn_layer_offset(ctx_dft, i);
1708	            }
== common/speculative.cpp:1728-1752 @79e2e74
1728	                const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
1729	                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);
1730	
1731	                const auto * cur_p = common_sampler_get_candidates(smpl, true);
1732	
1733	                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
1734	                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
1735	                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
1736	                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
1737	                }
1738	
1739	                // add drafted token for each sequence
1740	                const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;
1741	
1742	                // only collect very high-confidence draft tokens
1743	                if (cur_p->data[0].p < params.p_min) {
1744	                    drafting[seq_id] = false;
1745	                    n_drafting--;
1746	
1747	                    continue;
1748	                }
1749	
1750	                common_sampler_accept(smpl, id, true);
1751	
1752	                auto & dp = dparams.at(seq_id);
== common/speculative.cpp:1814-1827 @79e2e74
1814	    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
1815	        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
1816	            return;
1817	        }
1818	
1819	        const int32_t n_rows = verify_h_rows[seq_id];
1820	        if (n_rows <= 0) {
1821	            return;
1822	        }
1823	
1824	        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
1825	        const size_t row_bytes = (size_t) n_embd * sizeof(float);
1826	        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
1827	    }
== common/speculative.cpp:2612-2619 @79e2e74
2612	
2613	    // the draft context holds as many tokens per sequence as the target context
2614	    cparams.n_ctx = llama_n_ctx(ctx_tgt);
2615	
2616	    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
2617	    //       the extra memory for small models is likely negligible?
2618	    cparams.n_rs_seq  = 0;
2619	    cparams.ctx_other = ctx_tgt;
== common/common.h:329-337 @79e2e74
329	    int32_t n_max = 3; // maximum number of tokens to draft during speculative decoding
330	    int32_t n_min = 0; // minimum number of draft tokens to use for speculative decoding
331	
332	    float p_split = 0.1f; // speculative decoding split probability
333	    float p_min   = 0.0f; // minimum speculative decoding probability (greedy)
334	
335	    bool backend_sampling = true; // offload draft sampling to the backend (default: on)
336	
337	    bool probabilistic = false; // sample the draft and verify by rejection, instead of argmax and match
== common/common.h:376-376 @79e2e74
376	    std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };
== common/common.h:456-457 @79e2e74
456	    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
457	    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
== common/speculative.cpp:76-76 @79e2e74
76	    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
== common/speculative.cpp:1439-1439 @79e2e74
1439	        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));
== common/speculative.cpp:1757-1759 @79e2e74
1757	                if (dp.result_q) {
1758	                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
1759	                }
== common/speculative.cpp:2527-2531 @79e2e74
2527	common_params common_base_params_to_speculative(const common_params & params) {
2528	    const bool has_draft = params.speculative.has_dft();
2529	
2530	    const auto & params_spec = params.speculative.draft;
2531	    common_params result = params;
== common/speculative.cpp:2641-2646 @79e2e74
2641	    } else if (spec_mtp) {
2642	        model_path = params.model.path;
2643	
2644	        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());
2645	
2646	        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
== common/arg.cpp:357-359 @79e2e74
357	static bool spec_types_is_default(const common_params & params) {
358	    return params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE};
359	}
== common/arg.cpp:546-550 @79e2e74
546	    // infer the speculative type from the sidecar shipped by the draft repo when none is requested
547	    if (spec_types_is_default(params)) {
548	        if (!plan_spec.mtp.local_path.empty()) {
549	            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
550	            plan_spec.dspark = {};
== common/arg.cpp:566-573 @79e2e74
566	    // infer the speculative type from the draft GGUF metadata when none is requested
567	    // note: reads only the first split - sharded drafts need an explicit --spec-type
568	    if (spec_types_is_default(params) && !params.speculative.draft.mparams.path.empty()) {
569	        const auto types_gguf = common_speculative_types_from_gguf(params.speculative.draft.mparams.path);
570	        if (!types_gguf.empty()) {
571	            params.speculative.types = types_gguf;
572	        }
573	    }
== common/arg.cpp:1693-1699 @79e2e74
1693	        {"--swa-full"},
1694	        string_format("use full-size SWA cache (default: %s)\n"
1695	            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa_full ? "true" : "false"),
1696	        [](common_params & params) {
1697	            params.swa_full = true;
1698	        }
1699	    ).set_env("LLAMA_ARG_SWA_FULL"));
== common/arg.cpp:4160-4167 @79e2e74
4160	        {"--spec-draft-n-max"}, "N",
4161	        string_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.draft.n_max),
4162	        [](common_params & params, int value) {
4163	            if (value < 0) {
4164	                throw std::invalid_argument("invalid value");
4165	            }
4166	            params.speculative.draft.n_max = value;
4167	        }
== common/arg.cpp:4177-4177 @79e2e74
4177	        {"--spec-synth-len"}, "L",
== common/arg.cpp:4190-4190 @79e2e74
4190	        {"--spec-synth-rates"}, "P0,P1,...",
== common/arg.cpp:4233-4241 @79e2e74
4233	        {"--spec-draft-sampling"}, "{greedy,probabilistic}",
4234	        string_format("how the draft is sampled: greedy takes its argmax, probabilistic samples it and has "
4235	                      "the target verify by rejection sampling (default: %s)",
4236	                      params.speculative.draft.probabilistic ? "probabilistic" : "greedy"),
4237	        [](common_params & params, const std::string & value) {
4238	            if (value == "greedy") {
4239	                params.speculative.draft.probabilistic = false;
4240	            } else if (value == "probabilistic") {
4241	                params.speculative.draft.probabilistic = true;
== common/arg.cpp:4284-4292 @79e2e74
4284	        {"--spec-type"}, common_speculative_all_types_str(),
4285	        string_format("comma-separated list of types of speculative decoding to use (default: %s)\n",
4286	            common_speculative_type_name_str(params.speculative.types).c_str()),
4287	        [](common_params & params, const std::string & value) {
4288	            const auto types_str = string_split<std::string>(value, ',');
4289	            auto types = common_speculative_types_from_names(types_str);
4290	            params.speculative.types.insert(params.speculative.types.end(), types.begin(), types.end());
4291	        }
4292	    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_TYPE"));
== common/common.cpp:1725-1725 @79e2e74
1725	    cparams.swa_full          = params.swa_full;
== common/sampling.cpp:689-721 @79e2e74
689	std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
690	    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
691	
692	    std::vector<llama_token> result;
693	    result.reserve(idxs.size());
694	
695	    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
696	
697	    size_t i = 0;
698	    for (; i < draft.size(); i++) {
699	        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
700	
701	        common_sampler_accept(gsmpl, id, true);
702	
703	        result.push_back(id);
704	
705	        // do not accept draft tokens after an EOG - they are not output but would stay in the context
706	        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
707	        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
708	            break;
709	        }
710	    }
711	
712	    if (i == draft.size()) {
713	        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
714	
715	        common_sampler_accept(gsmpl, id, true);
716	
717	        result.push_back(id);
718	    }
719	
720	    return result;
721	}
== common/sampling.cpp:732-735 @79e2e74
732	// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
733	// Preserves the target distribution exactly, and accepts more often than matching does when the
734	// draft samples instead of taking its argmax.
735	std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
== common/sampling.cpp:790-803 @79e2e74
790	                break;
791	            }
792	        }
793	
794	        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
795	            common_sampler_accept(gsmpl, draft[i], true);
796	            result.push_back(draft[i]);
797	            continue;
798	        }
799	
800	        // rejected: tokens outside q's support keep all of p
801	        residual.clear();
802	        float sum = 0.0f;
803	        for (size_t k = 0; k < cur_p->size; ++k) {
== tools/server/server-context.cpp:504-506 @79e2e74
504	    bool use_spec_rejection() const {
505	        return task && task->params.sampling.temp > 0.0f;
506	    }
== tools/server/server-context.cpp:713-716 @79e2e74
713	
714	            SLT_INF(*this,
715	                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
716	                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
== tools/server/server-common.cpp:99-102 @79e2e74
99	    if (n_draft_tokens > 0) {
100	        base["draft_n"]          = n_draft_tokens;
101	        base["draft_n_accepted"] = n_draft_accepted;
102	    }
== tools/server/server-context.cpp:3400-3405 @79e2e74
3400	
3401	        // generate the actual drafts (if any)
3402	        if (!drafting.empty()) {
3403	            queue_tasks.yield_to_queue([&]() {
3404	                common_speculative_draft(spec.get());
3405	            });
== tools/server/server-context.cpp:3416-3426 @79e2e74
3416	            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
3417	
3418	            if (ctx_dft) {
3419	                if (use_ckpt_dft) {
3420	                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
3421	                }
3422	
3423	                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
3424	                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
3425	                }
3426	            }
== tools/server/server-context.cpp:4225-4233 @79e2e74
4225	
4226	        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
4227	        //       for now, always re-evaluate for simplicity
4228	        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
4229	        if (spec) {
4230	            bool ok = true;
4231	            queue_tasks.yield_to_queue([&]() {
4232	                ok = common_speculative_process(spec.get(), batch.view);
4233	            });
== tools/server/server-context.cpp:4410-4427 @79e2e74
4410	
4411	                // drafters that fill no distribution fall back here
4412	                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();
4413	
4414	                std::vector<llama_token> accepted;
4415	                if (!synth_probs.empty()) {
4416	                    // synthetic acceptance replaces verification entirely, so it comes first
4417	                    accepted = server_sample_and_accept_synth(
4418	                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
4419	                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
4420	                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
4421	                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
4422	                } else if (use_rejection) {
4423	                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_draft_q);
4424	                } else {
4425	                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
4426	                }
4427	                slot.spec_i_batch.clear();
== tools/server/server-context.cpp:4468-4472 @79e2e74
4468	                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
4469	                }
4470	
4471	                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
4472	
== tools/server/server-context.cpp:4497-4505 @79e2e74
4497	
4498	            // add accepted tokens to the prompt
4499	            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
4500	            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});
4501	
4502	            slot.sampled = ids.back(); // last accepted token
4503	            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);
4504	
4505	            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
== src/llama-kv-cache.cpp:428-437 @79e2e74
428	            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
429	                if (new_head == cells.size()) {
430	                    new_head = i;
431	                }
432	            }
433	        }
434	
435	        // If we freed up a slot, set head to it so searching can start there.
436	        if (new_head != cells.size() && new_head < head) {
437	            head = new_head;
== src/llama-kv-cache.cpp:1019-1025 @79e2e74
1019	        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];
1020	
1021	        // if we have enough unused cells before the current head ->
1022	        //   better to start searching from the beginning of the cache, hoping to fill it
1023	        if (head_cur > cells.get_used() + 2*n_tokens) {
1024	            head_cur = 0;
1025	        }
== src/llama-kv-cache.cpp:1061-1078 @79e2e74
1061	
1062	                if (!can_use && cells.seq_count(idx) == 1) {
1063	                    const llama_pos pos_cell = cells.pos_get(idx);
1064	
1065	                    // (disabled) causal mask
1066	                    // note: it's better to purge any "future" tokens beforehand
1067	                    //if (cells.seq_has(idx, seq_id)) {
1068	                    //    can_use = pos_cell >= pos;
1069	                    //}
1070	
1071	                    if (!can_use) {
1072	                        const llama_seq_id seq_id_cell = cells.seq_get(idx);
1073	
1074	                        // SWA mask
1075	                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
1076	                            can_use = true;
1077	                        }
1078	                    }
== src/llama-kv-cache.cpp:1185-1205 @79e2e74
1185	            continue;
1186	        }
1187	
1188	        GGML_ASSERT(s < seq_to_stream.size());
1189	
1190	        auto & cells = v_cells[seq_to_stream[s]];
1191	
1192	        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
1193	            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
1194	                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);
1195	
1196	            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
1197	        }
1198	    }
1199	
1200	    // move the head at the end of the slot
1201	    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
1202	        auto & head = v_heads[sinfo.strm[s]];
1203	
1204	        head = sinfo.idxs[s].back() + 1;
1205	    }
== src/llama-kv-cache-iswa.cpp:70-81 @79e2e74
70	
71	    // note: the SWA cache is always padded to 256 for performance
72	    //       https://github.com/ggml-org/llama.cpp/issues/17037
73	    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
74	
75	    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
76	    if (swa_full) {
77	        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
78	                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
79	
80	        size_swa = size_base;
81	    }
== src/llama-context.cpp:147-166 @79e2e74
147	    cparams.ctx_other = nullptr;
148	
149	    // TODO: more generic
150	    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
151	        if (params.ctx_other == nullptr) {
152	            // TODO: change from runtime_error to llama_exception to avoid printing error message
153	            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
154	        }
155	
156	        cparams.ctx_other = params.ctx_other;
157	    }
158	
159	    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
160	        if (model.tok_embd == nullptr || model.output == nullptr) {
161	            if (params.ctx_other == nullptr) {
162	                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
163	            }
164	            cparams.ctx_other = params.ctx_other;
165	        }
166	    }
== src/models/mimo2.cpp:267-270 @79e2e74
267	    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
268	    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
269	                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
270	                "nextn_layer_offset out of range [0, n_layer_nextn)");
== conversion/mimo.py:134-136 @79e2e74
134	        for weight_name, (w_fn, s_fn, bid) in qkv_overrides.items():
135	            # MTP layers (bid >= n_layer_text) use SWA-style attention dims
136	            is_swa = True if bid >= n_layer_text else hybrid[bid] == 1
== conversion/mimo.py:153-160 @79e2e74
153	        # Extend the per-layer pattern with SWA entries for the MTP blocks so the
154	        # runtime arrays (sized to extended block_count) are fully populated.
155	        hybrid = list(self.hparams["hybrid_layer_pattern"]) + [1] * self._n_nextn
156	        n_head_kv_arr = [n_head_kv_swa if use_swa == 1 else n_head_kv for use_swa in hybrid]
157	        self.gguf_writer.add_head_count_kv(n_head_kv_arr)
158	
159	        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
160	        self.gguf_writer.add_sliding_window_pattern(hybrid)
== common/speculative.cpp:1533-1557 @ec7630a
1533	            bool ok = true;
1534	            for (int head = 0; head < n_mtp_layers; ++head) {
1535	                if (chain_heads) {
1536	                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
1537	                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
1538	                        if (i_batch_beg[seq_id] < 0) {
1539	                            continue;
1540	                        }
1541	                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
1542	                    }
1543	                    llama_set_nextn_layer_offset(ctx_dft, head);
1544	                }
1545	
1546	                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
1547	                if (rc != 0) {
1548	                    SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
1549	                            head, (int) rc, (int) batch_in.tokens[0].pos[0]);
1550	                    ok = false;
1551	                    break;
1552	                }
1553	            }
1554	
1555	            if (chain_heads) {
1556	                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
1557	            }
== third_party/hf/mimo-v2.6-flash-chat/config.json:162-162 @b3a728f
162	  "num_nextn_predict_layers": 3,
== third_party/hf/mimo-v2.6-flash-chat/config.json:303-303 @b3a728f
303	  "sliding_window": 128,
== scripts/r1/configs/llamacpp-server.json:10-10 @b3a728f
10	    "mode": "overlap no (llama.cpp: no CPU/GPU overlap inside a layer, tools/roofline.py decode() docstring), hit 1.0 (PROVISIONAL: every expert in VRAM, ROOFLINE.md table 2 'in VRAM 1.00'), no speculation",
== scripts/r1/configs/llamacpp-server.json:36-39 @b3a728f
36	    "argv": ["${LLAMA_BIN}/llama-server", "-m", "${MODEL}", "--host", "${HOST}", "--port", "${PORT}",
37	             "-c", "${CTX}", "-np", "1", "-ngl", "all", "-sm", "layer", "-ts", "${TENSOR_SPLIT}", "-fa", "on",
38	             "-ctk", "${KV_TYPE}", "-ctv", "${KV_TYPE}", "--jinja", "--metrics",
39	             "--slot-save-path", "${SLOT_DIR}"],
== scripts/r1/configs/llamacpp-server.json:42-42 @b3a728f
42	    "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
== scripts/r1/configs/llamacpp-server.json:51-51 @b3a728f
51	    "desc": "Qwen3.8-Flash-Next UD-Q4_K_XL (unsloth/Qwen3.8-Flash-Next-GGUF @38bb39ee97821de2c9009abb7e93950eec396e66, 4 files)",
== scripts/r1/configs/llamacpp-server.json:66-66 @b3a728f
66	  "mode": {"overlap": "no", "strategy": "pp", "miss": "cpu", "hit": "${HIT}", "spec_k": 0, "spec_accept": 0.0},
== scripts/r1/configs/llamacpp-server-moecache.json:52-52 @b3a728f
52	    "desc": "GLM-5.3-Flash, glm5-next GGUF (file, quant and revision UNKNOWN: set GLM_GGUF)",
== scripts/r1/configs/llamacpp-server-moecache.json:64-64 @b3a728f
64	  "mode": {"overlap": "no", "strategy": "pp", "miss": "pcie", "hit": "${HIT}", "spec_k": 0, "spec_accept": 0.0},
== scripts/r1/configs/ik-llamacpp-server.json:50-50 @b3a728f
50	    "desc": "GLM-5.3-Flash UD-Q4_K_XL (Unsloth GGUF, 6 shards, architecture glm5next; HF revision UNKNOWN)",
== scripts/r1/configs/ik-llamacpp-server.json:62-62 @b3a728f
62	  "mode": {"overlap": "no", "strategy": "pp", "miss": "cpu", "hit": "${HIT}", "spec_k": 0, "spec_accept": 0.0},
== third_party/hf/mimo-v2.6-flash-chat/config.json:47-47 @b3a728f
47	  "hybrid_layer_pattern": [
== docs/reassessment/ROADMAP.md:55-55 @b3a728f
55	| R4.4 | S1/S2: MTP и n-gram/suffix-спекуляция, принятие и чистый выигрыш на 262K; DFlash2 — только при основаниях (S3) | движок с поддержкой | ≥ 1.1× или выключено | NOT RUN |
```

No speculative flag in the argv of R1's llama.cpp-family templates (counts of matching lines per file):
```
$ grep -c -E -- '--spec|--draft|--mtp' scripts/r1/configs/llamacpp-server.json scripts/r1/configs/llamacpp-server-moecache.json scripts/r1/configs/ik-llamacpp-server.json
scripts/r1/configs/llamacpp-server.json:0
scripts/r1/configs/llamacpp-server-moecache.json:0
scripts/r1/configs/ik-llamacpp-server.json:0
```

### 6.10 Unit suites on `64009b9`

```
$ python3 -m unittest discover -s tests
Ran 1110 tests in 152.290s

OK (skipped=566)

real	2m33.228s
user	1m10.209s
sys	0m19.408s
```

```
$ python3 -m unittest tests.test_hooks
----------------------------------------------------------------------
Ran 21 tests in 9.357s

OK
```
