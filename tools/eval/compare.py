#!/usr/bin/env python3
"""Paired comparison of two evaluation runs of the same task set (stdlib only).

    python3 tools/eval/compare.py RUN_A RUN_B [--aggregate majority] [--n-boot 10000] [--seed 0] [--json OUT]

Unit = task. Each run's repeats are collapsed to one outcome per task by --aggregate (majority by default:
more than half of the valid repeats pass; a tie is a fail). A task is PAIRED when both runs have a valid
outcome for it and the task content hash (task_sha256) is the same in both runs; tasks with a different hash,
present in one run only, or with request/harness errors only, are excluded and listed.

For every category and overall:
  b = A pass and B fail, c = A fail and B pass (discordant pairs)
  McNemar exact two-sided p = min(1, 2 * P[Binomial(b + c, 1/2) <= min(b, c)])   (ev_stats.mcnemar_exact)
  diff = pass rate A - pass rate B over the paired tasks = (b - c) / n
  95% CI of diff: percentile bootstrap over paired tasks (same resampled tasks for A and B), seeded
A difference is reported as significant only by the p-value and CI; with b + c < 6 no p-value can fall
below 0.05 (2 * 0.5^5 = 0.0625), which the output states.
Exit code 0 (the comparison itself never fails on the result); 2 when a run directory is unreadable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ev_stats  # noqa: E402
import ev_tasks  # noqa: E402


def load_run(d):
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    rows = []
    with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return meta, rows


def pair(out_a, out_b, allow_mismatch=False):
    """Returns (paired list of (task_id, category, a_pass, b_pass), excluded dict reason -> [ids])."""
    paired = []
    excl = {"only_in_a": [], "only_in_b": [], "task_hash_differs": [], "no_valid_outcome": []}
    for tid in sorted(set(out_a) | set(out_b)):
        a, b = out_a.get(tid), out_b.get(tid)
        if a is None:
            excl["only_in_b"].append(tid)
            continue
        if b is None:
            excl["only_in_a"].append(tid)
            continue
        if a.get("task_sha256") != b.get("task_sha256") and not allow_mismatch:
            excl["task_hash_differs"].append(tid)
            continue
        if a["pass"] is None or b["pass"] is None:
            excl["no_valid_outcome"].append(tid)
            continue
        paired.append((tid, a.get("category"), bool(a["pass"]), bool(b["pass"])))
    return paired, excl


def compare_block(pairs, n_boot=10000, seed=0):
    n = len(pairs)
    a = [1 if p[2] else 0 for p in pairs]
    b = [1 if p[3] else 0 for p in pairs]
    disc_b = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    disc_c = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    boot = ev_stats.bootstrap_paired(a, b, n_boot=n_boot, seed=seed)
    return {
        "n_paired": n, "a_passes": sum(a), "b_passes": sum(b),
        "a_pass_rate": sum(a) / n if n else None, "b_pass_rate": sum(b) / n if n else None,
        "diff": boot["diff"], "diff_ci95": [boot["lo"], boot["hi"]],
        "b_a_only": disc_b, "c_b_only": disc_c, "mcnemar_p": ev_stats.mcnemar_exact(disc_b, disc_c),
        "low_power": disc_b + disc_c < 6,
        "a_ci95": list(ev_stats.wilson(sum(a), n)), "b_ci95": list(ev_stats.wilson(sum(b), n)),
        "discordant": {"a_only": [p[0] for p in pairs if p[2] and not p[3]],
                       "b_only": [p[0] for p in pairs if p[3] and not p[2]]},
    }


def compare(meta_a, rows_a, meta_b, rows_b, aggregate="majority", n_boot=10000, seed=0, allow_mismatch=False):
    out_a = ev_stats.task_outcomes(rows_a, aggregate)
    out_b = ev_stats.task_outcomes(rows_b, aggregate)
    paired, excl = pair(out_a, out_b, allow_mismatch)
    cats = [c for c in ev_tasks.CATEGORIES if any(p[1] == c for p in paired)]
    res = {"aggregate": aggregate, "n_boot": n_boot, "seed": seed,
           "categories": {c: compare_block([p for p in paired if p[1] == c], n_boot, seed) for c in cats},
           "overall": compare_block(paired, n_boot, seed), "excluded": excl, "warnings": []}
    ea, eb = meta_a.get("eval") or {}, meta_b.get("eval") or {}
    w = res["warnings"]
    for name, m in (("A", meta_a), ("B", meta_b)):
        if m.get("is_mock_server"):
            w.append("run %s was made against a MOCK server: not engine evidence" % name)
        if (m.get("eval") or {}).get("synthetic_task_set"):
            w.append("run %s used a SYNTHETIC task set: not evidence of model quality" % name)
        if m.get("status") != "ok":
            w.append("run %s has status %r" % (name, m.get("status")))
    for k in ("task_set_sha256", "temperature", "max_tokens_default", "repeats", "seed", "extra_body"):
        if ea.get(k) != eb.get(k):
            w.append("%s differs: A=%r B=%r" % (k, ea.get(k), eb.get(k)))
    if any(excl.values()):
        w.append("excluded tasks: %s" % {k: len(v) for k, v in excl.items() if v})
    res["runs"] = {name: {"engine_desc": m.get("engine_desc"), "model_desc": m.get("model_desc"),
                          "model": m.get("model"), "git_sha": (m.get("git") or {}).get("sha"),
                          "is_mock_server": m.get("is_mock_server"), "utc_start": m.get("utc_start")}
                   for name, m in (("A", meta_a), ("B", meta_b))}
    return res


def _f(x, pct=True):
    if x is None:
        return "n/a"
    return "%.1f%%" % (100 * x) if pct else "%.4g" % x


def render(res, path_a, path_b):
    lines = ["# Paired comparison", "", "- A: `%s` (%s; %s)" % (path_a, res["runs"]["A"]["engine_desc"],
                                                               res["runs"]["A"]["model_desc"]),
             "- B: `%s` (%s; %s)" % (path_b, res["runs"]["B"]["engine_desc"], res["runs"]["B"]["model_desc"]),
             "- unit: task; repeats collapsed by `%s`; bootstrap %d resamples, seed %s" % (
                 res["aggregate"], res["n_boot"], res["seed"]), ""]
    for w in res["warnings"]:
        lines.append("**WARNING:** %s" % w)
    if res["warnings"]:
        lines.append("")
    lines.append("| category | paired | A pass | B pass | A-B | 95% CI (bootstrap) | b (A only) | c (B only) | McNemar p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, b in list(res["categories"].items()) + [("**overall**", res["overall"])]:
        ci = b["diff_ci95"]
        lines.append("| %s | %d | %s | %s | %s | %s | %d | %d | %s%s |" % (
            name, b["n_paired"], _f(b["a_pass_rate"]), _f(b["b_pass_rate"]),
            "n/a" if b["diff"] is None else "%+.1f pp" % (100 * b["diff"]),
            "n/a" if ci[0] is None else "[%+.1f, %+.1f] pp" % (100 * ci[0], 100 * ci[1]),
            b["b_a_only"], b["c_b_only"], _f(b["mcnemar_p"], pct=False), " (low power)" if b["low_power"] else ""))
    o = res["overall"]
    lines += ["", "Discordant tasks - A only: %s" % (", ".join(o["discordant"]["a_only"]) or "none"),
              "Discordant tasks - B only: %s" % (", ".join(o["discordant"]["b_only"]) or "none")]
    for k, v in res["excluded"].items():
        if v:
            lines.append("Excluded (%s): %s" % (k, ", ".join(v)))
    lines.append("")
    lines.append("low power = fewer than 6 discordant pairs: the exact McNemar test cannot reach p < 0.05.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--aggregate", choices=ev_stats.AGGREGATES, default="majority")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-task-mismatch", action="store_true",
                    help="pair tasks even when their content hash differs between the runs")
    ap.add_argument("--json", default=None, help="also write the result as JSON to this path")
    a = ap.parse_args(argv)
    try:
        meta_a, rows_a = load_run(a.run_a)
        meta_b, rows_b = load_run(a.run_b)
    except (OSError, ValueError) as e:
        print("cannot read run directory: %s" % e, file=sys.stderr)
        return 2
    res = compare(meta_a, rows_a, meta_b, rows_b, a.aggregate, a.n_boot, a.seed, a.allow_task_mismatch)
    res["run_a"], res["run_b"] = os.path.abspath(a.run_a), os.path.abspath(a.run_b)
    sys.stdout.write(render(res, a.run_a, a.run_b))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
