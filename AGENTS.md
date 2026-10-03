# AGENTS.md — operating rules for every agent in this repo

Read this file completely before doing anything. It applies to Claude Code, Codex,
OpenCode and any other agent. Human-facing docs are Russian (README.md, docs/PLAN.md,
docs/DECISIONS.md); machine-facing files (this file, .claude/**, code, JSON) are English.

## Mission

Port the techniques of Strata (tiered MoE inference: VRAM expert cache + RAM experts
computed on CPU + SSD n-gram tables + native-MTP speculation) to a 4x RTX 4080 Super 32 GB
server, for these models, in this order (see docs/DECISIONS.md ADR-001):

1. GLM-5.3-Flash (320B / 18B active)        — first target
2. MiMo-V2.6-Flash (309B / 15B active)      — second
3. DeepSeek-V4.1-Flash (552B + 196B Engram) — third

The customer is Ribrad. The team is fully autonomous inside the gates in docs/PLAN.md.

## Hardware (stated by the customer, see hardware/4x4080s-32g.json)

- 4x RTX 4080 Super 32 GB (sm_89, Ada: FP8 tensor cores yes, FP4 tensor cores NO)
- GPU-to-GPU P2P over PCIe works
- AMD EPYC 7H43, 48C/96T, Zen 3: AVX2 only, NO AVX-512, NO AMX
- 128 GB DDR4-3200; number of populated channels is UNKNOWN (4 or 8) — measure it in G0
- 3x Samsung 990 Pro 1 TB NVMe
- Proxmox host; the inference VM is on an isolated network with NO internet access

## Upstream pins (never edit upstream code in place)

| upstream | commit | why |
|---|---|---|
| Niko1221/Strata | c499bd102e7a4135c0de389dcfe38c399759ccc8 | engine we port from (MIT) |
| ggml-org/llama.cpp | ec7630a640789c393694fb194f1bbbf0369fc62d | reference graphs: glm5-next.cpp, mimo2.cpp, qwen4exp.cpp |
| sergqwer/strata-glm | ed37419fccd0c52e07d26d526a29c2098f547843 | GLM engine base A: NVFP4, tiers VRAM/RAM/disk (MIT, ADR-007); patches in patches/strata-glm |
| lighttransport/Strata (branch glm53f) | e486a95d78876989b853b16c7056bcd96880afc3 | GLM engine base B: native GGUF backend in Strata (MIT, ADR-009); patches go to patches/lt-strata |

Strata transcribes llama.cpp's `qwen4exp` graph; we transcribe `glm5-next` the same way.
Changes to upstream code live as patches or as our own sources, never as silent edits.

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

Never type a commit hash from memory. Never claim a test passed without its output.
If you did not run it, write "NOT RUN" — that is acceptable; a fabricated result is not.

## Gate rules

- Work happens inside the current gate in docs/PLAN.md. Starting gate N+1 before gate N
  is accepted is not allowed.
- Only the lead-architect role moves a gate to ACCEPTED, and only by citing evidence.
- Every new kernel ships with a parity test against `ref/` (numpy oracle) AND against
  llama.cpp tensor dumps (tensor names from `cb(..., "<name>", il)` in the reference graph).
- Performance claims need `tools/` benchmark output; a number without a command is invalid.

## Git and sync

- The VM has no internet. Sync is by `git bundle` through the customer's Windows PC.
- Never push to third-party remotes. Never rewrite published history.
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
