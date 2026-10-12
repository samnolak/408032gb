# AGENTS.md — operating rules for every agent in this repo

Read this file completely before doing anything. It applies to Claude Code, Codex,
OpenCode and any other agent. Human-facing docs are Russian (README.md, docs/PLAN.md,
docs/DECISIONS.md); machine-facing files (this file, .claude/**, code, JSON) are English.

## Mission

Port the techniques of Strata (tiered MoE inference: VRAM expert cache + RAM experts
computed on CPU + SSD n-gram tables + native-MTP speculation) to a 4x RTX 4080 Super 32 GB
server, for these models (the order of ADR-001 below is SUPERSEDED: models are chosen by quality, speed and 262K
context per ADR-012 D2 and the gates of docs/reassessment/ROADMAP.md; Qwen3.8-Flash-Next is used for local
development and tests only, ADR-012.2 licence registry):

- GLM-5.3-Flash (320B / 18B active)
- MiMo-V2.6-Flash (309B / 15B active)
- DeepSeek-V4.1-Flash (552B + 196B Engram)
- Qwen3.8-Flash-Next (development and tests only, see above)

The customer is Ribrad. The team is fully autonomous inside the gates of docs/reassessment/ROADMAP.md (R0–R6; docs/PLAN.md G0–G8 is
HISTORICAL).

## Weights on host `epyc-ai` (3x RTX 3090, branch `3090`; decision of that host's operator, 2026-10-03)

On that host the only real GLM-5.3-Flash weights are the GGUF files
`/mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-*-of-00006.gguf`
(185.983 GiB, GGUF architecture string `glm5next` as reported by that host's operator; UNVERIFIED — owner decision
2026-10-11: the file is not trusted until the read-only check `tools/hostcheck/glm_gguf_check.sh` (sizes, sha256 against
the Hugging Face pin, header facts) has run on that host; NOT RUN). Nothing else is to be downloaded there, and no safetensors
checkpoint is to be described as weights present on that host.
Scope (lead-architect, 2026-10-03): this section does not choose the weights for the 4x4080 server; that choice is
ADR-005 / ADR-009 and is an open question to the customer (docs/evidence/2026-10-03-3090-review.md).
Neither engine base reads this GGUF today: base A needs its own pack (GGUF->pack converter pending; it re-quantises),
base B has no Q4_K expert kernels (PROVISIONAL, read from its docs: Q2_K, Q3_K, IQ2_S, IQ3_S, IQ4_XS only).

## Hardware (stated by the customer, NOT yet measured; see hardware/4x4080s-32g.json)

Nothing below is CONFIRMED on the real server until R0 (docs/reassessment/R0-HARDWARE-CHECKS.md) records it with
literal command output. Owner decisions of 2026-10-11/12 (docs/DECISIONS.md ADR-012.2, ADR-012.3) override every older statement.

- 4x RTX 4080 Super 32 GB (sm_89, Ada: FP8 tensor cores yes, FP4 tensor cores NO); 32 GB per card is checked in R0
- GPU-to-GPU P2P over PCIe: UNKNOWN until measured in R0 with standard tools only (`nvidia-smi topo -m`,
  `nvidia-smi topo -p2p r`, `bench_gpu.sh`, nccl-tests). The NVIDIA driver is NOT patched (ADR-012.2) without a new
  explicit owner permission. Plans must work without P2P (pipeline / layer split by default).
- AMD EPYC 7H43, 48C/96T, Zen 3: AVX2 only, NO AVX-512, NO AMX (to be confirmed by CPUID in R0)
- 128 GB DDR4-3200; number of populated memory channels (4 or 8) and the NUMA/PCIe topology are UNKNOWN — measure
  them in R0 (dmidecode/EDAC on the Proxmox host, STREAM, `scripts/diag/bench_mem.sh`)
- Storage: 3x NVMe SSD 1 TB = 3 TB total (PLANNED; customer correction 2026-10-12, replaces both the old
  "3x Samsung 990 Pro" and the 2026-10-11 "4x 1 TB"). Drive models, real speed and PCIe topology are UNKNOWN until R0
- Proxmox host; the inference VM is on an isolated network with NO internet access

## Upstream pins (never edit upstream code in place)

| upstream | commit | why |
|---|---|---|
| Niko1221/Strata | c499bd102e7a4135c0de389dcfe38c399759ccc8 | engine we port from (MIT) |
| ggml-org/llama.cpp | ec7630a640789c393694fb194f1bbbf0369fc62d | reference graphs: glm5-next.cpp, mimo2.cpp, qwen4exp.cpp |
| sergqwer/strata-glm | ed37419fccd0c52e07d26d526a29c2098f547843 | GLM engine base A: NVFP4, tiers VRAM/RAM/disk (MIT, ADR-007); patches in patches/strata-glm |
| lighttransport/Strata (branch glm53f) | e486a95d78876989b853b16c7056bcd96880afc3 | GLM engine base B: native GGUF backend in Strata (MIT, ADR-009); patches go to patches/lt-strata |
| Grigory-Rylov/strata-glm-3090 | b115f38f4c1b924665e053a686d7579407db60fe | base A with patches 0001-0006 applied as commits, used on the 3x3090 test host; `src/` and `tools/` trees equal pin + patches (docs/evidence/2026-10-03-3090-review.md). patches/ stays the source of truth |
| vllm-project/vllm | bc21cba9673cfc2256a2726b4bc32f062237cd35 | DFlash2 reference implementation (Apache-2.0, ADR-011): `vllm/model_executor/models/qwen3_dflash.py`, `qwen3_dflash2.py`, `vllm/v1/worker/gpu/spec_decode/dflash2/` |

Strata transcribes llama.cpp's `qwen4exp` graph; we transcribe `glm5-next` the same way.
Changes to upstream code live as patches or as our own sources, never as silent edits.

Engine pins of the R1 baselines (the source of truth is `tools/engines/pins.json`; `scripts/engines/build_engine.py`
builds them and `scripts/r1/run_r1.py` refuses an engine whose `build-info.json` differs). R1 builds apply NO patch
from `patches/llama.cpp/` (N20/N21 fixes are CPU-verified only; a patch enters R1 only with a separate verifiable need):

| engine | commit |
|---|---|
| ggml-org/llama.cpp (R1 pin; differs from the reference-graph pin above) | 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 |
| ikawrakow/ik_llama.cpp | 5194a9e1451b885e24dd5a065f650864e8807c59 |
| Niko1221/Strata (R1 baseline B0) | fb58e0dbc8399662c0e47c76578c6e878b14f6cf |
| turboderp-org/exllamav3 | 151539c77abc7ab7425d30da7a4e8e3c5c154e7b |

## Evidence rules (hard)

Every number or factual claim in a doc, report, commit message or PR carries a label:

- **CONFIRMED** — with a source: `path:line@commit`, a URL, or a command whose literal
  output is pasted into the report or into docs/evidence/.
- **PROVISIONAL** — derived or estimated; state how.
- **UNKNOWN** — say so; never fill the gap with a plausible value.

An agent that reports "done" pastes the literal output of:

```
git log --oneline -n 5
git status --short
<the test command it ran>
```

Sources of truth, in this order (a newer CONFIRMED measurement or owner decision beats an older document; older
ADRs and plans — ADR-001..ADR-011, docs/PLAN.md G0–G8 — have NO priority over newer confirmed research; their status
is in docs/DECISIONS.md "Пометки к прежним решениям" and docs/reassessment/LEGACY_AUDIT.md):

1. literal measurements on the real hardware (R0 onward) and owner decisions recorded in docs/DECISIONS.md;
2. docs/reassessment/PROJECT_STATE.md (current state and how to continue), docs/reassessment/ROADMAP.md (gates R0–R6),
   docs/reassessment/CORRECTNESS.md, the newest docs/reassessment/PHASE-CLOSE-*.md and docs/evidence/;
3. pins: the tables in this file and tools/engines/pins.json; R1 configurations: scripts/r1/configs/*.json, checked
   by scripts/r1/safety_rules.py;
4. everything else (research snapshots in docs/reassessment/research/, legacy docs) is context, not a decision.

A simulated GPU, a simulated SSD, a CPU run or a CI compilation is never reported as a hardware result or as a working
inference engine. No tok/s figure is stated as a fact without a `tools/` benchmark run on the real server.

Never type a commit hash from memory. Never claim a test passed without its output.
If you did not run it, write "NOT RUN" — that is acceptable; a fabricated result is not.

## Gate rules

- Work happens inside the current gate of docs/reassessment/ROADMAP.md. Starting gate N+1 before gate N
  is accepted is not allowed.
- Only the lead-architect role moves a gate to ACCEPTED, and only by citing evidence.
- Every new kernel ships with a parity test against `ref/` (numpy oracle) AND against
  llama.cpp tensor dumps (tensor names from `cb(..., "<name>", il)` in the reference graph).
- Performance claims need `tools/` benchmark output; a number without a command is invalid.

## Git and sync

- The VM has no internet. Sync is by `git bundle` through the customer's Windows PC.
- Publication policy (customer decision 2026-10-10, docs/reassessment/ADR-012.1-git-publication-policy.md):
  - Agents may create `claude/*` branches, commit intermediate results there and publish them with a plain
    fast-forward `git push [-u] origin claude/<name>` to `origin` = github.com/samnolak/408032gb, without asking
    each time. Agents may open **Draft** pull requests from `claude/*` into `main`.
  - Never: push to `main` or any branch outside `claude/*`; force-push (`--force`, `-f`, `--force-with-lease`,
    `+refspec`); delete remote branches or tags; `--mirror`/`--all`/`--tags`; push to any other remote, fork or URL;
    change remotes, refspecs, `insteadOf` or aliases to get around these rules; merge pull requests or enable
    auto-merge (the owner merges); publish secrets (tokens, keys, credentials, `.env` files).
  - Enforced by hooks (`.claude/settings.json`): `scripts/hooks/guard-bash.sh` (git, gh, curl), `guard-github.sh`
    (GitHub tools), `secret_scan.py` (outgoing commits); tests in `tests/test_hooks.py`. Never bypass or weaken a
    hook; a change to these rules needs a documented customer decision and updated tests.
- Never rewrite published history (no `reset --hard`, interactive rebase, `filter-branch`, amend of pushed commits).
- One logical change per commit; prefix the subject with the role, e.g.
  `[parity-qa] ref: KDA oracle`.

## Roles and routing (Sol = complex agentic, Terra = everyday, Luna = quick lookups)

| role (.claude/agents/) | routing | owns |
|---|---|---|
| lead-architect | Sol | plan, gates, ADRs, acceptance |
| model-analyst | Terra | configs, tensor maps, fact sheets |
| kernel-engineer | Sol | CUDA kernels (KDA, MLA-DSA, mHC, MoE GEMV) |
| engine-integrator | Sol | Strata backend wiring, loader, pipeline/TP |
| parity-qa | Terra | oracles, dumps, parity tests |
| perf-bench | Terra | fitplan, benchmarks, profiling |
| infra-ops | Terra | VM, drivers, builds, bundles |
| evidence-auditor | Luna | verifies claims, git log, labels |

## Definition of done (any task)

1. Code + tests committed with a role-prefixed message.
2. Tests run; literal output in the report (or "NOT RUN" with the reason).
3. Docs updated; every new claim labeled.
4. evidence-auditor has checked the report against `git log` and the files.
