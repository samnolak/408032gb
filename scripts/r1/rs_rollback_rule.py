#!/usr/bin/env python3
"""N20 workaround rule for llama.cpp llama-server WITHOUT patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch
(docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md, section "R1 without the patch").

Rule: a llama.cpp (mainline) llama-server that runs a speculative type which gives the target context recurrent
rollback snapshots (n_rs_seq = draft n_max > 0 for draft-mtp, draft-eagle3, draft-dflash, draft-dspark; 0 for every
other type, common/common.h:399-405@79e2e74, common/common.cpp:1700@79e2e74) must run with exactly one slot (-np 1).
Why (labels as in the evidence file): with n_rs_seq = 0 no rollback is ever accepted, so no stale plane and no pending
rollback exist (CONFIRMED by the code and N19's rs0 probe); with n_rs_seq > 0 the server's own rollbacks are always
inside its last verify call (CONFIRMED by reading), and the one server path found that copies a sequence while a
rollback is pending is the shared prompt prefix of decision tasks (tools/server/server-context.cpp:3826-3849@79e2e74),
which needs at least two slots (tools/server/server-decision.cpp:1139-1161@79e2e74) (PROVISIONAL: by reading, no server
run). llama-server's default is -np auto = 4 slots (common/arg.cpp:1406@79e2e74, tools/server/server.cpp:168-173@79e2e74),
so the slot count must be given.

check(argv, env) -> list of problems (empty: the rule holds). argv is llama-server's argument list; env its environment
(LLAMA_ARG_SPEC_TYPE, LLAMA_ARG_N_PARALLEL are read when the flag is absent, as common/arg.cpp's set_env does).
CLI: rs_rollback_rule.py <scripts/r1 config.json>... checks every config whose engine.pin is "llama.cpp" (other engines:
"not covered"); exit 1 on a violation.
"""
import json
import sys

RS_TYPES = {"draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark"}   # common/common.h:399-405@79e2e74
AUTO_SLOTS = 4                                                             # tools/server/server.cpp:168-173@79e2e74


def _values(argv, names):
    """Values of a flag given as `--flag v` or `--flag=v`, in order."""
    out = []
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                out.append(argv[i + 1])
            elif a.startswith(n + "="):
                out.append(a[len(n) + 1:])
    return out


def spec_types(argv, env=None):
    vals = _values(argv, ["--spec-type"])
    if not vals and env and env.get("LLAMA_ARG_SPEC_TYPE"):
        vals = [env["LLAMA_ARG_SPEC_TYPE"]]
    return [t.strip() for v in vals for t in v.split(",") if t.strip()]


def n_slots(argv, env=None):
    vals = _values(argv, ["-np", "--parallel"])
    if not vals and env and env.get("LLAMA_ARG_N_PARALLEL"):
        vals = [env["LLAMA_ARG_N_PARALLEL"]]
    if not vals:
        return AUTO_SLOTS
    n = int(vals[-1])
    return AUTO_SLOTS if n < 0 else n


def check(argv, env=None):
    rs = sorted(set(spec_types(argv, env)) & RS_TYPES)
    if not rs:
        return []
    n = n_slots(argv, env)
    if n != 1:
        return [f"--spec-type {','.join(rs)} gives the target recurrent rollback snapshots (n_rs_seq > 0); "
                f"without the N20 patch run it with -np 1, not {n} slot(s)"]
    return []


def main(paths):
    bad = 0
    for p in paths:
        with open(p) as f:
            cfg = json.load(f)
        eng = cfg.get("engine", {})
        if eng.get("pin") != "llama.cpp":
            print(f"{p}: not covered (engine pin {eng.get('pin')!r})")
            continue
        probs = check(eng.get("argv", []), eng.get("env", {}))
        types = spec_types(eng.get("argv", []), eng.get("env", {}))
        print(f"{p}: {'VIOLATES: ' + '; '.join(probs) if probs else 'ok'} "
              f"(spec types {types or 'none'}, slots {n_slots(eng.get('argv', []), eng.get('env', {}))})")
        bad += bool(probs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
