# N7: llama-server of R1 on hybrid-memory models, CPU: prefix reuse, checkpoints, --cache-reuse, slot save/restore, two slots, MTP — 2026-10-10 (sandbox CPU, no GPU)

Task N7 of the hardware-free track (docs/reassessment/ROADMAP.md, row N7): settle on a CPU, with the real server binary R1
will run, the server-level risks that N2/N6 (graph parity) do not cover: does prefix caching work for hybrid memory
(recurrent state + KV + indexer cache) and give the same answer as a cold run (B3), does slot save/restore survive a
restart (K3), are two concurrent slots isolated, and does the in-model MTP head run in llama-server without the crash of
N6 finding 12 and with output equal to plain greedy decoding (S1). Branch `claude/n7-server-hybrid-cpu`, cut from
`claude/relaxed-dijkstra-1d7i5o` at `3f32a66` (session `session_01EyB9XrhA3NTRjjQx1A9qMh`, in parallel with the lead
session of PR #1; coordinated by a message to that session before starting).

Nothing here was run on a GPU, on real weights, or at long context. Speed numbers are not reported: the models are tiny
random-weight files and every timing would be meaningless for R1.

| piece | file |
|---|---|
| harness | tools/srvcheck/hybrid_server_checks.py (stdlib only; uses tools/kv/slot_persist.py unchanged) |
| driver | tools/srvcheck/run_n7.sh (pin check against tools/engines/pins.json, CPU build, 4 tiny GGUFs, gate + characterization) |
| report views | tools/srvcheck/n7_report.py (`lines`, `attr`) |
| tests | tests/test_srvcheck.py (18: comparison logic + the harness end to end against a stand-in server with 11 injected faults;
the message of commit `7f96724` says "nine" - at that commit there were 10 faults and 17 tests, counted with
`git show 7f96724:<file>`; history is not rewritten), tests/test_n7_report.py (3) |
| stand-in server | tests/fixtures/srvcheck/stub_llama_server.py |
| generator | tools/tinygen/glm5next_tiny.py `--mtp` (new, default output byte-identical); tools/tinygen/qwen4exp_tiny.py `--mtp` (N6) |
| CI | .github/workflows/server-cpu.yml (job `n7`; new file, so no edit of ci.yml): runs on PR #2; result recorded in the PR, not here |
| mutation check | docs/evidence/mutation-scripts/2026-10-10-n7-srvcheck.py |

## Setup (CONFIRMED, literal)

llama.cpp at the R1 pin, fetched by commit (no remote added), CPU build:
```
$ git -C llama.cpp rev-parse HEAD
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
$ cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF
-- ggml commit:  79e2e74
$ cmake --build build-cpu -j4 --target llama-server llama-cli
[223/223] Linking CXX executable bin/llama-cli
$ build-cpu/bin/llama-server --version
version: 0.6.0-dev (build 1, commit 79e2e74)
built with GNU 13.3.0 for Linux x86_64
```
`tools/srvcheck/run_n7.sh` refuses a checkout whose HEAD differs from `tools/engines/pins.json` (`llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (pins.json)` is its first output line).

Models (written by the driver with the checkout's gguf-py; qwen4exp = the N2/N6 model, glm5-next = the G1 model with 12
layers, 3 of them DSA; `_mtp` = the same trunk plus one MTP block):
```
26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5  glm5next.gguf
f7833b6ce213f0fe1f2130b79754931f428c83c4c46edf952321990853965681  glm5next_mtp.gguf
0fe7442121108a67b7bb01d7bc432d308f43fff41107d56e7a08c99e1ec51951  qwen4exp.gguf
23bf70ffd45b0a458bc99657b15bcebf05f2843d6a6737a528a4f2f4422976cb  qwen4exp_mtp.gguf
```
`glm5next_tiny.py --mtp` is new in this branch (commit `73fa752`): the MTP block is layer `n_layer` as llama.cpp @79e2e74
loads it (src/models/glm5-next.cpp:79-80 `is_nextn` flags, 85-92 mHC tensors only for `i < n_layer`, 143 full indexer,
175-182 `nextn.*`; graph_mtp 543-666). Default output unchanged: sha256 of the default 12-layer file and of `--layers 4`
before and after the edit — `26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5` and
`6309ba52478a425e0bf6a87ac5ea1f283a584bba52b502ea56c62b44eba49a67` both times (`cmp` silent).

Method. All prompts are token-id lists (`/completion`, `"prompt": [ids]`), greedy (`temperature 0`), `ignore_eos`,
`n_probs 5`, `return_tokens`; slot 0 unless stated. A check passes only if the tokens are identical AND every log-prob that
both answers list in a step's top-5 differs by at most `--tol` = 1e-3 ("not only top-1"). Prompt: 200 ids (seed 7),
tail 24 ids, 32 generated tokens. Every server gets `--slot-save-path` (POST /slots answers 501 without it, observed:
`{"error":{"code":501,"message":"This server does not support slots action. Start it with `--slot-save-path`","type":"not_supported_error"}}`).

| check | what is sent | pass criterion |
|---|---|---|
| determinism | prompt, erase, prompt | identical |
| prefix_extend | prompt; then prompt + answer + tail (agent loop) | warm = cold run of the second request; reuse reported |
| prefix_diverge | prompt; then prompt[:120] + tail | warm = cold; reuse reported |
| cache_reuse | server with `--cache-reuse 8 -lv 4`; head 64 + body 120; then head + body[16:] (old tool output dropped) | warm = cold; log: flag disabled / chunk shifted / forced full re-processing |
| slot_persist | `slot_persist.run_once` (cold, save, erase, restore, same prompt); then save, stop server, new server, restore, same prompt | continuation identical in both |
| isolation | server `-np 2`; each prompt alone in its slot, then both at once from two threads, `GET /slots` polled every 5 ms meanwhile | each concurrent answer = its lone answer, and the server reported both slots `is_processing` at the same moment (`max_busy_slots` 2) |
| mtp | `_mtp` file without and with `--spec-type draft-mtp --spec-draft-n-max 3`, twice with MTP | identical, clean exit, `draft_n > 0` |

## Results (CONFIRMED on CPU, tiny models; literal)

```
$ tools/srvcheck/run_n7.sh <llama.cpp @79e2e74> <build-cpu> <out>; echo "run_n7.sh exit=$?"
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (pins.json)
version: 0.6.0-dev (build 1, commit 79e2e74)
== qwen4exp gate (-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup)
== glm5next gate (-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup)
== qwen4exp characterization (-c 1024 -np 1 -t 2 --no-warmup)
== glm5next characterization (-c 1024 -np 1 -t 2 --no-warmup)
artifacts: <out>
run_n7.sh exit=0
```
(the lines above are the driver's own lines, `grep -E "^llama.cpp|^version|^== |^artifacts|exit="` of its log; between
them it prints a JSON summary per run, whose values are the ones `n7_report.py lines` prints from the result files:)
```
$ python3 tools/srvcheck/n7_report.py lines <out>/qwen4exp-gate.json <out>/glm5next-gate.json <out>/qwen4exp-defaults.json <out>/glm5next-defaults.json
## qwen4exp.gguf: server_args -c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup
verdict {"determinism": true, "prefix_extend_correct": true, "prefix_diverge_correct": true, "slot_persist_same_process": true, "slot_persist_after_restart": true, "cache_reuse_correct": true, "isolation_two_slots": true, "server_exit_clean": true, "mtp_equals_plain_decode": true, "all_pass": true}
determinism      max|d| 0.0 same_tokens True
prefix_extend    n_next 256 reusable 232 warm prompt_n 25 cache_n 231 reused 231 | max|d| 4.76837158203125e-07 same_tokens True
prefix_diverge   n_next 144 shared 120 warm prompt_n 144 cache_n 0 reused 0 | max|d| 0.0 same_tokens True
cache_reuse      n_next 168 deleted 16 warm prompt_n 168 reused 0 flag_disabled True chunk_shifted False forced_full True | max|d| 0.0
slot_persist     same-process identical True warm prompt_n 4 | restart: same_tokens True max|d| 0.0 warm prompt_n 4 | file 269844 B | exits [0, 0]
isolation        server max_busy_slots 2 client overlap 0.109 s same_tokens [True, True] max|d| [1.1920928955078125e-06, 1.9073486328125e-06]
mtp              --spec-type draft-mtp --spec-draft-n-max 3 on_exit 0 off_exit 0 | draft_n 81 accepted 2 | same_tokens True max|d| 7.152557373046875e-07 | repeat same_tokens True
mtp server log   common_speculative_init_result: creating MTP draft context against the target model 'qwen4exp_mtp.gguf'
mtp server log   slot print_timing: id  0 | task 1 | draft acceptance = 0.02469 (    2 accepted /    81 generated), mean len =  1.07
mtp server log   slot print_timing: id  0 | task 34 | draft acceptance = 0.02469 (    2 accepted /    81 generated), mean len =  1.07

## glm5next.gguf: server_args -c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup
verdict {"determinism": true, "prefix_extend_correct": true, "prefix_diverge_correct": true, "slot_persist_same_process": true, "slot_persist_after_restart": true, "cache_reuse_correct": true, "isolation_two_slots": true, "server_exit_clean": true, "mtp_equals_plain_decode": true, "all_pass": true}
determinism      max|d| 0.0 same_tokens True
prefix_extend    n_next 256 reusable 232 warm prompt_n 25 cache_n 231 reused 231 | max|d| 7.152557373046875e-06 same_tokens True
prefix_diverge   n_next 144 shared 120 warm prompt_n 144 cache_n 0 reused 0 | max|d| 0.0 same_tokens True
cache_reuse      n_next 168 deleted 16 warm prompt_n 168 reused 0 flag_disabled False chunk_shifted True forced_full True | max|d| 0.0
slot_persist     same-process identical True warm prompt_n 4 | restart: same_tokens True max|d| 0.0 warm prompt_n 4 | file 299756 B | exits [0, 0]
isolation        server max_busy_slots 2 client overlap 0.185 s same_tokens [True, True] max|d| [4.38690185546875e-05, 1.9788742065429688e-05]
mtp              --spec-type draft-mtp --spec-draft-n-max 3 on_exit 0 off_exit 0 | draft_n 87 accepted 0 | same_tokens True max|d| 5.7220458984375e-06 | repeat same_tokens True
mtp server log   common_speculative_init_result: creating MTP draft context against the target model 'glm5next_mtp.gguf'
mtp server log   slot print_timing: id  0 | task 1 | draft acceptance = 0.00000 (    0 accepted /    87 generated), mean len =  1.00
mtp server log   slot print_timing: id  0 | task 36 | draft acceptance = 0.00000 (    0 accepted /    87 generated), mean len =  1.00

## qwen4exp.gguf: server_args -c 1024 -np 1 -t 2 --no-warmup
verdict {"determinism": true, "prefix_extend_correct": true, "prefix_diverge_correct": true, "slot_persist_same_process": true, "slot_persist_after_restart": true, "cache_reuse_correct": true, "isolation_two_slots": true, "server_exit_clean": true, "mtp_equals_plain_decode": true, "all_pass": true}
determinism      max|d| 0.0 same_tokens True
prefix_extend    n_next 256 reusable 232 warm prompt_n 25 cache_n 231 reused 231 | max|d| 2.4080276489257812e-05 same_tokens True
prefix_diverge   n_next 144 shared 120 warm prompt_n 144 cache_n 0 reused 0 | max|d| 0.0 same_tokens True
cache_reuse      n_next 168 deleted 16 warm prompt_n 168 reused 0 flag_disabled True chunk_shifted False forced_full True | max|d| 0.0
slot_persist     same-process identical True warm prompt_n 4 | restart: same_tokens True max|d| 0.0 warm prompt_n 4 | file 181140 B | exits [0, 0]
isolation        server max_busy_slots 2 client overlap 0.099 s same_tokens [True, True] max|d| [2.288818359375e-05, 1.6689300537109375e-05]
mtp              --spec-type draft-mtp --spec-draft-n-max 3 on_exit 0 off_exit 0 | draft_n 81 accepted 2 | same_tokens True max|d| 2.384185791015625e-07 | repeat same_tokens True
mtp server log   common_speculative_init_result: creating MTP draft context against the target model 'qwen4exp_mtp.gguf'
mtp server log   slot print_timing: id  0 | task 1 | draft acceptance = 0.02469 (    2 accepted /    81 generated), mean len =  1.07
mtp server log   slot print_timing: id  0 | task 34 | draft acceptance = 0.02469 (    2 accepted /    81 generated), mean len =  1.07

## glm5next.gguf: server_args -c 1024 -np 1 -t 2 --no-warmup
verdict {"determinism": true, "prefix_extend_correct": false, "prefix_diverge_correct": true, "slot_persist_same_process": true, "slot_persist_after_restart": true, "cache_reuse_correct": true, "isolation_two_slots": false, "server_exit_clean": true, "mtp_equals_plain_decode": true, "all_pass": false}
determinism      max|d| 0.0 same_tokens True
prefix_extend    n_next 256 reusable 232 warm prompt_n 25 cache_n 231 reused 231 | max|d| 0.026694536209106445 same_tokens True
prefix_diverge   n_next 144 shared 120 warm prompt_n 144 cache_n 0 reused 0 | max|d| 0.0 same_tokens True
cache_reuse      n_next 168 deleted 16 warm prompt_n 168 reused 0 flag_disabled False chunk_shifted True forced_full True | max|d| 0.0
slot_persist     same-process identical True warm prompt_n 4 | restart: same_tokens True max|d| 0.0 warm prompt_n 4 | file 211052 B | exits [0, 0]
isolation        server max_busy_slots 2 client overlap 0.170 s same_tokens [True, True] max|d| [0.0015256404876708984, 0.0022292137145996094]
mtp              --spec-type draft-mtp --spec-draft-n-max 3 on_exit 0 off_exit 0 | draft_n 87 accepted 0 | same_tokens True max|d| 2.86102294921875e-06 | repeat same_tokens True
mtp server log   common_speculative_init_result: creating MTP draft context against the target model 'glm5next_mtp.gguf'
mtp server log   slot print_timing: id  0 | task 1 | draft acceptance = 0.00000 (    0 accepted /    87 generated), mean len =  1.00
mtp server log   slot print_timing: id  0 | task 36 | draft acceptance = 0.00000 (    0 accepted /    87 generated), mean len =  1.00
```
Gate (f32 caches, flash attention off): every check passes on both architectures. Characterization (server defaults
for KV type and flash attention): qwen4exp passes everything; glm5-next fails `prefix_extend_correct` and
`isolation_two_slots` on the 1e-3 log-prob tolerance with identical tokens — explained under F7.

Precision attribution (gate and defaults rows: the driver run above, harness `0a20e20`; the two middle glm5-next rows:
`--extra "-c 1024 -np 1 -t 2 --no-warmup <flags>"` with harness `7f96724` - the change between the two is the isolation
proof only, which does not touch any comparison):
```
$ python3 tools/srvcheck/n7_report.py attr "gate f32 fa-off=..." "f16 fa-off=..." "f32 fa-on=..." "defaults (f16 fa-on)=..."
config                 extend      isolation   determin.   mtp         | cold 256-token prompt vs the first run: first differing token, |d| at step 0, max |d| up to it
gate f32 fa-off        7.15e-06    4.39e-05    0           5.72e-06    | 32, 0, 0
f16 fa-off             0.000244    0.000536    0           -           | 3, 0.414, 0.859
f32 fa-on              0.00142     0.00223     0           -           | 0, 0.516, 0.516
defaults (f16 fa-on)   0.0267      0.00223     0           2.86e-06    | 3, 0.416, 0.86
$ python3 tools/srvcheck/n7_report.py attr "gate f32 fa-off=..." "defaults (f16 fa-on)=..."      # qwen4exp
config                 extend      isolation   determin.   mtp         | cold 256-token prompt vs the first run: first differing token, |d| at step 0, max |d| up to it
gate f32 fa-off        4.77e-07    1.91e-06    0           7.15e-07    | 32, 0, 0
defaults (f16 fa-on)   2.41e-05    2.29e-05    0           2.38e-07    | 32, 3e-05, 0.000248
```

## Findings

1. **MTP runs in llama-server @79e2e74 on both architectures, no crash, output = plain greedy decoding** (CONFIRMED,
   CPU, tiny models). One flag turns it on: `--spec-type draft-mtp`; the draft context is created on the target model
   from the same GGUF (`creating MTP draft context against the target model`, common/speculative.cpp:2641-2646) and
   the MTP tensors are loaded only then (common/common.cpp:1690). Rejected drafts are rolled back from recurrent
   snapshots: `n_rs_seq` = `--spec-draft-n-max` when MTP is on (common/common.cpp:1700, common/common.h:399-405;
   QWEN4EXP and GLM5_NEXT are in `llm_arch_supports_rs_rollback`, src/llama-arch.cpp:1171-1184), rollback up to
   `n_rs_seq` tokens (src/llama-memory-recurrent.cpp:200-219), longer ones through a state checkpoint
   (tools/server/server-context.cpp:4431-4435). With random weights almost every draft is rejected (qwen4exp 2 of 81
   accepted, glm5-next 0 of 87), so these runs exercise the rollback path heavily and the acceptance path barely
   (qwen4exp only); acceptance numbers say nothing about real models.
2. **N6 finding 12 (crash of the first MTP graph without a scheduler re-reserve) does not apply at the R1 pin.**
   At `ec7630a` `llama_context::set_embeddings_nextn` only sets the flags (src/llama-context.cpp:1230-1235 @ec7630a,
   printed from raw.githubusercontent.com); at `79e2e74` it requests a reserve when they change
   (src/llama-context.cpp:1244-1247: `// these flags change the graph shape` / `sched_need_reserve = true;`), and the MTP
   path calls it for both contexts (common/speculative.cpp:1478-1479). CONFIRMED by reading and by finding 1 (server
   path on CPU); on a GPU: NOT RUN.
3. **Agent loop (next prompt = previous prompt + answer + new tail): reuse works** (CONFIRMED): 231 of the 232 reusable
   tokens reused (`cache_n 231`, `prompt_n 25` = 24 new + the answer's last token), answer equal to a cold run within
   4.8e-7 (qwen4exp) and 7.2e-6 (glm5-next).
4. **A prompt that diverges inside the cached part is re-processed from the start** (CONFIRMED): shared prefix 120 of
   144 tokens, reused 0, answers correct. Recurrent state cannot be cut at an arbitrary position; the server falls back
   to a context checkpoint at or before the divergence, else to full re-processing
   (tools/server/server-context.cpp:3726-3771, `forcing full prompt re-processing due to lack of cache data`).
   Checkpoints are made only while a prompt is processed, at user-message starts of a chat request (spaced by
   `--checkpoint-min-step`, default 8192, except the last user message; at most `--ctx-checkpoints` 32 per slot,
   common/common.h:637-639) and `4 + n_ubatch` and 4 tokens before the end of the prompt
   (tools/server/server-context.cpp:3976-4005, 4032-4036, 4051-4055). A raw-token `/completion` has no user messages,
   so it gets only the two end-of-prompt checkpoints: observed also with `--checkpoint-min-step 16` and with
   `-b 32 -ub 32` (shared 120, reused 0 on both models; runs in the session log, not repeated at the final commit).
   **Consequence for B3 (PROVISIONAL, from the code):** in a chat agent loop, if the chat template renders earlier turns
   differently on the next request (for example drops the reasoning of earlier assistant turns), a hybrid model
   re-processes everything after the last checkpoint at or before the first changed token — with default flags those
   are user-message starts at least 8192 tokens apart. Whether the GLM-5.3-Flash and Qwen3.8-Flash-Next templates do
   that: UNKNOWN (neither template is in this repository nor in llama.cpp `models/templates/` at the pin).
   Needed: the B3 probe through `/v1/chat/completions` with the real template (R1), and on CPU a tiny model with a
   byte-level vocabulary so that rendered chat text survives tokenization (candidate next step).
5. **`--cache-reuse` buys nothing for these hybrid models at the pin; on glm5-next it costs a KV shift** (CONFIRMED,
   CPU): qwen4exp — disabled at start (`cache_reuse is not supported by this context, it will be disabled`; IMROPE gives
   `n_pos_per_embd > 1`, so `get_can_shift()` is false: src/llama-kv-cache.cpp:1208-1217,
   tools/server/server-context.cpp:1318-1327). glm5-next (no RoPE) — not disabled; the server shifts the reused chunk
   (`reusing chunk with size 104, shifting KV cache [80, 184) -> [64, 168)`, logged at TRACE) and then, since no
   checkpoint covers the new position, forces full re-processing; answer equal to cold, reused 0. The R1 templates
   `scripts/r1/configs/llamacpp-server.json:37`, `llamacpp-server-moecache.json:39` and `docs/reassessment/DEPLOY.md:63`
   pass `--cache-reuse 256`. Recommendation to the lead (not changed here — those files belong to PR #1): drop it for
   hybrid models or record that it is inert; correctness is not at risk on this evidence.
6. **Slot save/restore, also across a server restart, gives the identical continuation** (CONFIRMED, CPU):
   `slot_persist.run_once` identical; after save → stop → new process → restore, same tokens with |d| = 0. The same
   prompt then re-processes only 4 tokens: the slot file carries the context checkpoints as an appendix
   (tools/server/server-context.cpp:2586-2714) and the restored N-4 checkpoint is used. The file holds the target
   context only, not the MTP draft context (tools/server/server-context.cpp:2917-2919). File sizes (tiny models):
   269 844 / 299 756 B with f32 caches, 181 140 / 211 052 B with defaults.
7. **Precision: with the server defaults (flash attention on, f16 KV) glm5-next warm and cold answers differ by up
   to 0.027 in log-prob (tokens identical); qwen4exp by 2.4e-5.** (CONFIRMED as measured.) Attribution (table above):
   flash attention with f32 caches 1.4e-3, f16 KV without flash attention 2.4e-4, both 2.7e-2. These differences are far
   inside each configuration's own deviation from the f32 / flash-attention-off answer on the same prompt: 0.41-0.52 at
   the first generated step and a different token by step 0-3. So they are not evidence of a cache bug; they show that
   the tiny glm5-next is ~10^4 times more sensitive to KV precision and flash attention than the tiny qwen4exp.
   Why (PROVISIONAL, not traced): likely the DSA k-pool top-k selection flipping on near ties, which turns rounding into
   a different attention set. Whether real GLM-5.3-Flash weights are as sensitive: UNKNOWN — R2.1 (KLD to Q8_0) and K2
   (KV type) must measure it, not assume it. The gate stays at f32 caches + flash attention off, as in N2/N6.
8. **Two slots decoding at once are isolated** (CONFIRMED, CPU): both answers equal the lone runs (|d| ≤ 4.4e-5 in the
   gate run), and the server itself reported both slots processing at the same moment (`max_busy_slots 2` in all four
   runs). The concurrent |d| varies a little between runs (4.53e-5 and 4.39e-5 for glm5-next on two runs of the same
   commit) because which tokens share a batch depends on request timing.
9. **Determinism**: two cold runs identical bit for bit in every configuration (|d| = 0).

## Harness tests and mutation check

Mutation check (each mutant edits a copy of tools/srvcheck/hybrid_server_checks.py and runs tests.test_srvcheck; the
working tree is never edited; script docs/evidence/mutation-scripts/2026-10-10-n7-srvcheck.py).

Round 1, code `7f96724`, literal:
```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n7-srvcheck.py <repository> <scratch dir>
baseline                                                 rc=0 ok       OK
M1 compare ignores tokens                                rc=1 caught   FAILED (failures=3)
M2 compare ignores the tolerance                         rc=1 caught   FAILED (failures=2)
M3 nothing to compare counts as equal                    rc=1 caught   FAILED (failures=2)
M4 determinism compares a run with itself                rc=1 caught   FAILED (failures=1)
M5 prefix_extend erases before the warm request          rc=1 caught   FAILED (failures=3)
M6 cache_reuse erases before the warm request            rc=1 caught   FAILED (failures=1)
M7 restart run never restores                            rc=1 caught   FAILED (failures=4)
M8 isolation runs the two requests one after the other   rc=1 caught   FAILED (failures=9)
M9 isolation does not require overlap                    rc=0 SURVIVED OK
M10 MTP pass without any draft                           rc=1 caught   FAILED (failures=1)
M11 MTP failure to answer counts as pass                 rc=1 caught   FAILED (failures=1)
M12 a crashed server reports a clean exit                rc=1 caught   FAILED (failures=1)
M13 verdict drops the restart check                      rc=1 caught   FAILED (failures=3)
M14 verdict drops the cache_reuse check                  rc=1 caught   FAILED (failures=1)
exit=0
```
M9 SURVIVED: the isolation check measured overlap on the client side, which a server that decodes the two requests
one after the other also shows (both requests are in flight while the first runs). Fixed in `0a20e20`: the check polls
`GET /slots` (field `is_processing`; `--slots, --no-slots  expose slots monitoring endpoint (default: enabled)` in
`llama-server --help` @79e2e74) and needs both slots busy at once; the stand-in got a `serial` fault (one completion
at a time, answers right, client overlap > 0) and the test `test_serialized_server_is_no_isolation_evidence`. In the
script M9 is re-anchored to the new condition and M15 (trust client overlap instead of /slots) is new.

Round 2, code `0a20e20`, literal (15 of 15 caught):
```
$ python3 docs/evidence/mutation-scripts/2026-10-10-n7-srvcheck.py <repository> <scratch dir>
baseline                                                 rc=0 ok       OK
M1 compare ignores tokens                                rc=1 caught   FAILED (failures=3)
M2 compare ignores the tolerance                         rc=1 caught   FAILED (failures=2)
M3 nothing to compare counts as equal                    rc=1 caught   FAILED (failures=2)
M4 determinism compares a run with itself                rc=1 caught   FAILED (failures=1)
M5 prefix_extend erases before the warm request          rc=1 caught   FAILED (failures=3)
M6 cache_reuse erases before the warm request            rc=1 caught   FAILED (failures=1)
M7 restart run never restores                            rc=1 caught   FAILED (failures=4)
M8 isolation runs the two requests one after the other   rc=1 caught   FAILED (failures=10)
M9 isolation does not require both slots busy at once    rc=1 caught   FAILED (failures=1)
M15 isolation trusts client-side overlap instead of GET /slots rc=1 caught   FAILED (failures=1)
M10 MTP pass without any draft                           rc=1 caught   FAILED (failures=1)
M11 MTP failure to answer counts as pass                 rc=1 caught   FAILED (failures=1)
M12 a crashed server reports a clean exit                rc=1 caught   FAILED (failures=1)
M13 verdict drops the restart check                      rc=1 caught   FAILED (failures=3)
M14 verdict drops the cache_reuse check                  rc=1 caught   FAILED (failures=1)
exit=0
```

Unit tests (sandbox, Python 3, after the merge of PR #1 head `3f07850` into this branch), literal:
```
$ python3 -m unittest tests.test_srvcheck tests.test_n7_report -v
test_identical_passes (tests.test_srvcheck.Compare.test_identical_passes) ... ok
test_logprob_beyond_tol_fails_with_same_tokens (tests.test_srvcheck.Compare.test_logprob_beyond_tol_fails_with_same_tokens) ... ok
test_nothing_to_compare_is_not_a_pass (tests.test_srvcheck.Compare.test_nothing_to_compare_is_not_a_pass) ... ok
test_step_count_mismatch_is_not_a_pass (tests.test_srvcheck.Compare.test_step_count_mismatch_is_not_a_pass) ... ok
test_token_flip_fails_and_reports_margin (tests.test_srvcheck.Compare.test_token_flip_fails_and_reports_margin) ... ok
test_cli_writes_json_and_fails_on_fault (tests.test_srvcheck.Harness.test_cli_writes_json_and_fails_on_fault) ... ok
test_healthy_server_passes_and_reports_reuse (tests.test_srvcheck.Harness.test_healthy_server_passes_and_reports_reuse) ... ok
test_mtp_crash_is_caught (tests.test_srvcheck.Harness.test_mtp_crash_is_caught) ... ok
test_mtp_divergence_is_caught (tests.test_srvcheck.Harness.test_mtp_divergence_is_caught) ... ok
test_mtp_that_never_drafts_is_not_a_pass (tests.test_srvcheck.Harness.test_mtp_that_never_drafts_is_not_a_pass) ... ok
test_noisy_warm_logprobs_are_caught (tests.test_srvcheck.Harness.test_noisy_warm_logprobs_are_caught) ... ok
test_nondeterministic_server_is_caught (tests.test_srvcheck.Harness.test_nondeterministic_server_is_caught) ... ok
test_restore_noop_is_caught (tests.test_srvcheck.Harness.test_restore_noop_is_caught) ... ok
test_restore_wrong_state_is_caught (tests.test_srvcheck.Harness.test_restore_wrong_state_is_caught) ... ok
test_serialized_server_is_no_isolation_evidence (tests.test_srvcheck.Harness.test_serialized_server_is_no_isolation_evidence) ... ok
test_slot_crosstalk_is_caught (tests.test_srvcheck.Harness.test_slot_crosstalk_is_caught) ... ok
test_stale_reuse_is_caught (tests.test_srvcheck.Harness.test_stale_reuse_is_caught) ... ok
test_stale_state_after_kv_shift_is_caught (tests.test_srvcheck.Harness.test_stale_state_after_kv_shift_is_caught) ... ok
test_attr_compares_against_the_first_run (tests.test_n7_report.Report.test_attr_compares_against_the_first_run) ... ok
test_cli_usage (tests.test_n7_report.Report.test_cli_usage) ... ok
test_lines_print_stored_values (tests.test_n7_report.Report.test_lines_print_stored_values) ... ok

----------------------------------------------------------------------
Ran 21 tests in 54.219s

OK
exit=0
```
Full suite on the same tree (the last lines; 447 = 426 reported for PR #1 head `3f07850` + the 21 above):
```
$ python3 -m unittest discover -s tests
Ran 447 tests in 160.847s

OK (skipped=86)
exit=0
```

## Not covered (NOT RUN)

| what | why / where it belongs |
|---|---|
| anything on a GPU (CUDA kernels, flash attention on GPU, multi-GPU, `--moe-cache-mib`) | no GPU; R1 |
| real weights, quantized weights, long context (262K), speed | tiny random models only; R1/K1/B3 |
| `/v1/chat/completions` with the real chat templates (checkpoints at user messages, re-rendered history) | finding 4; needs the templates and a byte-vocab tiny model |
| MTP acceptance path under realistic acceptance (random weights accept ~0) | R1 S1 on real weights |
| ik_llama.cpp and Strata servers | other engines of R1; only llama.cpp here |
| more than two slots, prompt cache in RAM across slot switches (`--cache-ram`) | not exercised |

## Reproduce
```
git init llama.cpp && git -C llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 && git -C llama.cpp checkout FETCH_HEAD
tools/srvcheck/run_n7.sh llama.cpp llama.cpp/build-n7 /tmp/n7-out
python3 tools/srvcheck/n7_report.py lines /tmp/n7-out/*-gate.json /tmp/n7-out/*-defaults.json
python3 -m unittest tests.test_srvcheck tests.test_n7_report -v
```
