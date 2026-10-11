# Qwen3.8-Flash-Next max-speed plan: fitplan and placement re-run (N25 phase close, 2026-10-11, sandbox CPU, no GPU)

Role: lead-architect (phase close on `claude/n25-integration`). Literal outputs behind
`docs/reassessment/QWEN-MAX-SPEED.md` sections 2-3 and appendix A. Nothing here ran on a GPU or on the server; every
number is a model output, PROVISIONAL (see `tools/fitplan.py` and `tools/ceilings/arch_ceilings.py` docstrings).

What was checked:

* The draft (perf-bench, scratch file) took both outputs in `/home/user/wt-research` @`2c64027`. The four inputs are
  byte-identical at this branch's `93bdfe0` (`git diff --quiet` rc=0 below), so the re-run is the same computation.
* `tools/fitplan.py` output equals the draft's appendix A block line for line (CONFIRMED by comparison). It models KV and
  recurrent state of this model as 0 and uses a uniform 4.5 bpw for experts instead of the file's 5.101 bpw, so its
  "pipeline 127 tok/s" is a bandwidth bound without KV, overheads or speculation, and "ideal TP4 508" ignores all-reduce
  (PROVISIONAL; the plan does not use it for planning).
* The draft's second command was a placeholder (`python3 -c '<placement ...>'`), not a reproducible command. Its body was
  restored from `scratchpad/archalt/placement.py` as printed in `docs/reassessment/ARCHITECTURE-ALTERNATIVES.md`
  (research-landscape `2c64027`, lines 1500-1517): its two Qwen rows (5.1/10/2.0 and 4.25/10/1.0, AA:1502-1505) plus a
  third Qwen row 5.1/10/1.0 that AA does not have, and committed as
  `docs/evidence/mutation-scripts/2026-10-11-n25-qwen-placement.py`. Its three lines equal the draft's three lines
  (CONFIRMED by comparison). The working tree was not yet committed when it ran (status lines below: the plan, this
  script and two doc edits; none of them is read by the tools).
* After the N25 audit (MINOR 6) the header comment of the script was corrected ("Qwen rows only" -> the rows above);
  its code is unchanged, so the literal `cat` below shows the comment as it was when the outputs were taken.

```
$ git rev-parse --short HEAD
93bdfe0
$ git status --short
 M docs/evidence/mutation-scripts/README.md
 M docs/reassessment/CORRECTNESS.md
?? docs/evidence/mutation-scripts/2026-10-11-n25-qwen-placement.py
?? docs/reassessment/QWEN-MAX-SPEED.md
$ git diff --quiet origin/claude/research-landscape HEAD -- tools/fitplan.py models/qwen3.8-flash-next.json tools/ceilings/arch_ceilings.py models/specs/qwen3.8-flash-next.json; echo rc=$?
rc=0
$ git rev-parse origin/claude/research-landscape
2c640273b248226ae3e51e9b6082948855d8c1d0
$ python3 tools/fitplan.py models/qwen3.8-flash-next.json --quant Q4_K --ctx 262144; echo rc=$?
# fitplan: Qwen3.8-Flash-Next (Strata reference)  experts=Q4_K (4.5000 bpw)  dense=Q8_0  ctx=262144
All figures PROVISIONAL (see tools/fitplan.py docstring).
experts: 24576 x 4.92M params = 63.3 GiB (2.64 MiB each)
dense 4.2 GiB | KV 0.00 GiB | recurrent state 0.00 GiB
  GPU0: 32.0 GiB = fixed 2.79 + expert cache 29.21
  GPU1: 32.0 GiB = fixed 2.79 + expert cache 29.21
  GPU2: 32.0 GiB = fixed 2.79 + expert cache 29.21
  GPU3: 32.0 GiB = fixed 2.79 + expert cache 29.21
VRAM-resident experts: 100.0% (24576)  | spill to RAM: 0.0 GiB of 116 usable
Strata mode (all experts in RAM): fits | disjoint VRAM/RAM: fits
n-gram/Engram tables on NVMe: 26.8 GiB
decode traffic per token: GPU 5525 MiB, CPU 0 MiB  (hit 100.0%, uniform routing (worst case for a frequency cache))
  ceiling, 4-channel RAM: pipeline 127 tok/s | ideal TP4 508 tok/s  (upper bounds, no MTP)
  ceiling, 8-channel RAM: pipeline 127 tok/s | ideal TP4 508 tok/s  (upper bounds, no MTP)
rc=0
$ python3 docs/evidence/mutation-scripts/2026-10-11-n25-qwen-placement.py; echo rc=$?
5.1 10.0 2.0 {'expert_mib': 2.988, 'experts_gib': 71.719, 'dense_gib': 4.937, 'kv_state_gib': 7.043, 'vram_expert_cache_gib': 108.02, 'resident_frac': 1.0, 'ram_frac': 0.0, 'ssd_frac': 0.0, 'dense_kv_fit_vram': True}
5.1 10.0 1.0 {'expert_mib': 2.988, 'experts_gib': 71.719, 'dense_gib': 4.937, 'kv_state_gib': 4.043, 'vram_expert_cache_gib': 111.02, 'resident_frac': 1.0, 'ram_frac': 0.0, 'ssd_frac': 0.0, 'dense_kv_fit_vram': True}
4.25 10.0 1.0 {'expert_mib': 2.49, 'experts_gib': 59.766, 'dense_gib': 4.937, 'kv_state_gib': 4.043, 'vram_expert_cache_gib': 111.02, 'resident_frac': 1.0, 'ram_frac': 0.0, 'ssd_frac': 0.0, 'dense_kv_fit_vram': True}
rc=0
$ cat docs/evidence/mutation-scripts/2026-10-11-n25-qwen-placement.py
# Byte placement of Qwen3.8-Flash-Next at 262,144 tokens (arch_ceilings.placement, preset "base"); run from the repo root.
# Body = scratchpad/archalt/placement.py of docs/reassessment/ARCHITECTURE-ALTERNATIVES.md (research-landscape 2c64027,
# lines 1500-1517), Qwen rows only; prints the expert/dense bpw, KV bytes per element and placement()'s whole dict.
import sys; sys.path.insert(0, "tools/ceilings"); import arch_ceilings as ac
for eb, db, kvb in ((5.1, 10.0, 2.0), (5.1, 10.0, 1.0), (4.25, 10.0, 1.0)):
    s, _ = ac.load_model("qwen")
    p = ac.resolve_params("base", {"quant": {"expert_bpw": eb, "dense_bpw": db}, "kv_bytes_per_elem": kvb})
    pl = ac.placement(s, p, ac.quant_of({}, p), 1)
    print(eb, db, kvb, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pl.items()})
```
