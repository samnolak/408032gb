#!/usr/bin/env python3
"""N24 V6 tolerance data: summary of logprob_compare.py result files (N11 tool + the N24 gap/noise fields).
Prints per file: prefixes, errors (and how many are the known incomplete-UTF-8 kind), strict top-1, flips with their
gaps, the distribution of max_abs_dlogprob (max over the tokens in both top-K lists of |lp_ref - lp_cand|) and the share
of prefixes with max_abs_dlogprob <= TOL / 2 (TOL = the criteria default logprob_tie_tolerance_nats, 0.025).
usage: python3 docs/evidence/mutation-scripts/2026-10-11-n24-v6-summary.py <lc-*.json> ...
Results: docs/evidence/2026-10-11-n24-owner-decisions.md (section 2)."""
import json
import os
import sys


def pct(v, q):
    return v[min(len(v) - 1, int(q * len(v)))]


TOL = 0.025
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    rows, errs = d["rows"], d["errors"]
    utf = sum("incomplete UTF-8" in e["error"] for e in errs)
    flips = [r for r in rows if not r["top1_agree"]]
    m = sorted(r["max_abs_dlogprob"] for r in rows if r.get("max_abs_dlogprob") is not None)
    kl = sorted(r["kl_topk"] for r in rows)
    print("%s: n %d, errors %d (incomplete UTF-8 %d), top-1 %.4f, flips %d %s" % (
        os.path.basename(path), len(rows), len(errs), utf, sum(r["top1_agree"] for r in rows) / len(rows), len(flips),
        [(r["prefix"], r.get("ref_gap"), r.get("cand_gap")) for r in flips][:10]))
    print("    max_abs_dlogprob: median %.3e p95 %.3e p99 %.3e max %.3e; common tokens min %d; KL top-K max %.3e" % (
        pct(m, 0.5), pct(m, 0.95), pct(m, 0.99), m[-1], min(r["n_common"] for r in rows), kl[-1]))
    print("    share with max_abs_dlogprob <= %.4f (TOL/2): %d/%d = %.3f" % (
        TOL / 2, sum(x <= TOL / 2 for x in m), len(m), sum(x <= TOL / 2 for x in m) / len(m)))
