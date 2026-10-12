"""Statistics for the agentic evaluation stand (stdlib only).

wilson(k, n)            Wilson score interval for a binomial proportion (95% by default).
mcnemar_exact(b, c)     exact two-sided McNemar test: binomial test of b against b + c at p = 1/2,
                        using ONLY the discordant pairs (b = A pass / B fail, c = A fail / B pass).
bootstrap_paired(a, b)  percentile bootstrap CI of mean(a) - mean(b), resampling paired units
                        (tasks) with replacement, the same index for both runs.
task_outcomes(rows)     collapse per-repeat results to one binary outcome per task.

References for the formulas (textbook, no code copied):
  Wilson, E. B. (1927) JASA 22:209-212 -- score interval.
  McNemar, Q. (1947) Psychometrika 12:153-157; exact form = two-sided binomial test on the
  discordant pairs (Agresti, Categorical Data Analysis, 3rd ed., sec. 11.1).
  Efron & Tibshirani (1993) -- percentile bootstrap.
"""
from __future__ import annotations

import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from runlog import percentile  # noqa: E402  (reuse: linear interpolation, numpy 'linear')

Z95 = 1.959963984540054  # standard normal quantile 0.975


def wilson(k, n, z=Z95):
    """Wilson score interval (lo, hi) for k successes out of n; (None, None) when n == 0."""
    if n <= 0:
        return None, None
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    # at k = 0 / k = n the bound is exactly 0 / 1 (center -/+ half reduces to it algebraically)
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value from the discordant counts b and c.

    p = min(1, 2 * P[X <= min(b, c)]), X ~ Binomial(b + c, 1/2). Concordant pairs do not enter.
    Returns 1.0 when there are no discordant pairs.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)
    return float(min(Fraction(1), 2 * tail))


def bootstrap_paired(a, b, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of mean(a) - mean(b) over paired units.

    a, b: equal-length sequences of per-unit values (0/1 or fractions). Each resample draws
    unit indices with replacement and uses the SAME indices for a and b (paired design).
    Returns dict(diff, lo, hi, n, n_boot, seed, alpha); lo/hi None when n == 0.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    if n == 0:
        return {"diff": None, "lo": None, "hi": None, "n": 0, "n_boot": n_boot, "seed": seed, "alpha": alpha}
    d = [float(x) - float(y) for x, y in zip(a, b)]
    obs = sum(d) / n
    rng = random.Random(seed)
    idx = range(n)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for i in rng.choices(idx, k=n):
            s += d[i]
        means.append(s / n)
    lo = percentile(means, 100.0 * alpha / 2.0)
    hi = percentile(means, 100.0 * (1.0 - alpha / 2.0))
    return {"diff": obs, "lo": lo, "hi": hi, "n": n, "n_boot": n_boot, "seed": seed, "alpha": alpha}


AGGREGATES = ("majority", "all", "any", "first")


def aggregate_pass(passes, rule="majority"):
    """One binary outcome from the valid (non-error) repeats of a task, in repeat order.

    majority: pass iff more than half of the valid repeats pass (a tie is a FAIL);
    all / any: every / at least one valid repeat passes; first: the first valid repeat.
    None when there is no valid repeat.
    """
    if not passes:
        return None
    if rule == "majority":
        return sum(1 for p in passes if p) * 2 > len(passes)
    if rule == "all":
        return all(passes)
    if rule == "any":
        return any(passes)
    if rule == "first":
        return bool(passes[0])
    raise ValueError("unknown aggregate rule %r" % rule)


def task_outcomes(rows, rule="majority"):
    """rows: results.jsonl rows. Returns {task_id: dict(category, task_sha256, n_repeats, n_valid,
    n_pass, n_request_errors, n_harness_errors, pass)} where pass is None if no repeat was valid.
    Only rows with status pass/fail are valid; request and harness errors never count as a pass or a fail."""
    out = {}
    for r in sorted(rows, key=lambda x: (x["task_id"], x.get("repeat", 0))):
        t = out.setdefault(r["task_id"], {"category": r.get("category"), "task_sha256": r.get("task_sha256"),
                                          "n_repeats": 0, "n_valid": 0, "n_pass": 0, "n_request_errors": 0,
                                          "n_harness_errors": 0, "_passes": []})
        t["n_repeats"] += 1
        st = r.get("status")
        if st in ("pass", "fail"):
            t["n_valid"] += 1
            t["_passes"].append(st == "pass")
            t["n_pass"] += 1 if st == "pass" else 0
        elif st == "request_error":
            t["n_request_errors"] += 1
        else:
            t["n_harness_errors"] += 1
    for t in out.values():
        t["pass"] = aggregate_pass(t.pop("_passes"), rule)
    return out
