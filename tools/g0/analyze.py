#!/usr/bin/env python3
"""Gate 0 analysis of expert routing histograms written by tools/g0/expert_hist.cpp.

Question it answers: if each GPU caches the most-used fraction f of every layer's experts,
what fraction of routed expert reads hit VRAM? That hit rate replaces fitplan's
uniform-routing worst case (`tools/fitplan.py --hit-curve <output>`).

To avoid fooling ourselves, the cache contents are chosen on TRAIN traces and the hit rate is
measured on HELD-OUT traces (every k-th file). The in-sample curve is reported too; a large
gap between the two means the cache must adapt online (Strata's cache does), not be static.

Usage:
  python3 tools/g0/analyze.py out/ --n-experts 288 --holdout-every 4 -o docs/evidence/g0-hit-curve.json
"""
import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict

FRACTIONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.67, 0.75, 0.8, 0.9, 1.0]


def read_counts(path):
    """layer -> {expert: count} from one CSV."""
    counts = defaultdict(dict)
    with open(path, newline="") as f:
        rows = (line for line in f if not line.startswith("#"))
        for r in csv.DictReader(rows):
            counts[int(r["layer"])][int(r["expert"])] = int(r["count"])
    return counts


def merge(items):
    total = defaultdict(lambda: defaultdict(int))
    for c in items:
        for layer, m in c.items():
            for e, n in m.items():
                total[layer][e] += n
    return total


def hit_curve(train, test, n_experts, fractions=FRACTIONS):
    """Global hit rate when each layer caches its top round(f * n_experts) train experts."""
    curve = []
    test_total = sum(sum(m.values()) for m in test.values())
    for f in fractions:
        n_cache = round(f * n_experts)
        hits = 0
        for layer, m in test.items():
            seen = train.get(layer, {})
            # rank every expert of the layer; ones never seen in train rank last (count 0)
            ranked = sorted(range(n_experts), key=lambda e: (-seen.get(e, 0), e))
            cached = set(ranked[:n_cache])
            hits += sum(n for e, n in m.items() if e in cached)
        curve.append([f, hits / test_total if test_total else 0.0])
    return curve


def layer_stats(m, n_experts):
    total = sum(m.values())
    probs = sorted((n / total for n in m.values()), reverse=True)
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    out = {"routings": total, "entropy_bits": entropy, "max_entropy_bits": math.log2(n_experts),
           "n_eff": 2 ** entropy, "unused_experts": n_experts - len(m)}
    for q in (0.5, 0.8, 0.9, 0.95):
        acc, k = 0.0, 0
        for p in probs:
            acc += p
            k += 1
            if acc >= q:
                break
        out[f"k{int(q * 100)}"] = k
    return out


def analyze(csv_dir, n_experts, holdout_every=4):
    files = sorted(pathlib.Path(csv_dir).glob("*.csv"))
    if len(files) < 2:
        raise SystemExit("need at least 2 trace CSVs for a held-out estimate")
    per_file = [read_counts(p) for p in files]
    test_idx = {i for i in range(len(files)) if i % holdout_every == holdout_every - 1} or {len(files) - 1}
    train = merge(c for i, c in enumerate(per_file) if i not in test_idx)
    test = merge(c for i, c in enumerate(per_file) if i in test_idx)
    every = merge(per_file)
    return {
        "n_experts": n_experts, "traces": len(files),
        "train_traces": len(files) - len(test_idx), "heldout_traces": len(test_idx),
        "global_curve": hit_curve(train, test, n_experts),
        "in_sample_curve": hit_curve(every, every, n_experts),
        "layers": {str(layer): layer_stats(m, n_experts) for layer, m in sorted(every.items())},
        "_label": "CONFIRMED for these traces and this GGUF; PROVISIONAL as a forecast for other workloads",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_dir")
    ap.add_argument("--n-experts", type=int, required=True)
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)
    res = analyze(a.csv_dir, a.n_experts, a.holdout_every)
    text = json.dumps(res, indent=2)
    if a.output:
        pathlib.Path(a.output).write_text(text + "\n")
    print("cache fraction -> held-out hit rate (in-sample)")
    for (f, h), (_, hi) in zip(res["global_curve"], res["in_sample_curve"]):
        print(f"  {f:5.2f} -> {h:6.3f} ({hi:6.3f})")


if __name__ == "__main__":
    main()
