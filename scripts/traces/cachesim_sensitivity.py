#!/usr/bin/env python3
"""Sensitivity runs of tools/g0/cachesim.py on routing traces, for the variants its CLI does not offer. The policies,
the scoring and the static ranking are cachesim's own code (cachesim.evaluate); this script only changes WHICH traces and
WHICH tokens go into a run:

  --fold all | r     every rotation of the hold-out split: test = files i with i % h == r (cachesim always uses r = h - 1,
                     cachesim.py main); with `all`, every conversation is tested exactly once and the summary pools them
  --test-view gen_only   test traces cut to their generated tokens: no policy sees the prompt, every dynamic cache is cold
                     (or holds the static set with --warm) at the first generated token. cachesim's per-layer policies
                     otherwise replay every prompt token one by one (cachesim.py simulate_layer), which warms them
  --gen-cap N        keep only the first N generated tokens of every test trace (generation length sensitivity)
  --train-part prompt | gen   train the static ranking on the prompt tokens / generated tokens of the train traces only
                     (default all tokens, as cachesim)
  --per-trace        score every test trace on its own
--warm, --session, --devices, --prompt-updates, --policies and --capacity mean what they mean in cachesim.py.
Input: a directory of *.trace files (gunzip data/traces/real/<model>/traces/*.trace.gz first).
Every printed number is a gen-token hit rate (cachesim's definition); the JSON (-o) also has prompt/all and per-layer
min/max. Usage:
  python3 scripts/traces/cachesim_sensitivity.py traces/ --n-experts 64 --capacity 0.5 --fold all --test-view gen_only
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "tools", "g0"))
import cachesim  # noqa: E402


def slice_trace(tr, lo, hi, n_prompt):
    """Tokens [lo, hi) of a trace as a new trace whose first n_prompt tokens are prompt tokens."""
    return {"name": tr.get("name"), "layer_ids": tr["layer_ids"], "k": tr["k"], "n_tokens": hi - lo,
            "n_prompt": n_prompt, "per_layer": {lid: seq[lo:hi] for lid, seq in tr["per_layer"].items()}}


def n_gen(tr):
    return tr["n_tokens"] - tr["n_prompt"]


def test_view(tr, view, gen_cap=None):
    """'full' (as cachesim) or 'gen_only'; gen_cap keeps only the first gen_cap generated tokens."""
    hi = tr["n_tokens"] if gen_cap is None else tr["n_prompt"] + min(gen_cap, n_gen(tr))
    if view == "gen_only":
        return slice_trace(tr, tr["n_prompt"], hi, 0)
    if view == "full":
        return slice_trace(tr, 0, hi, tr["n_prompt"])
    raise ValueError(view)


def train_view(tr, part):
    if part == "all":
        return tr
    if part == "prompt":
        return slice_trace(tr, 0, tr["n_prompt"], tr["n_prompt"])
    if part == "gen":
        return slice_trace(tr, tr["n_prompt"], tr["n_tokens"], 0)
    raise ValueError(part)


def folds(traces, every, which="last"):
    """[(r, train, test)]: test = traces i with i % every == r. which: 'last' (r = every - 1, cachesim's split), 'all',
    or an int."""
    rs = {"last": [every - 1], "all": list(range(every))}.get(which, [which])
    out = []
    for r in rs:
        test = [t for i, t in enumerate(traces) if i % every == int(r)]
        train = [t for i, t in enumerate(traces) if i % every != int(r)]
        out.append((int(r), train, test))
    return out


def run(traces, n_experts, fractions, policies, fold="last", every=4, view="full", gen_cap=None, train_part="all",
        per_trace=False, warm=False, session=False, devices=4, prompt_readonly=True):
    rows = []
    for r, train, test in folds(traces, every, fold):
        tr_train = [train_view(t, train_part) for t in train]
        groups = [[t] for t in test] if per_trace else [test]
        for grp in groups:
            tv = [test_view(t, view, gen_cap) for t in grp]
            res = cachesim.evaluate(tr_train, tv, n_experts, fractions, policies, warm=warm, session=session,
                                    n_devices=devices, prompt_readonly=prompt_readonly)
            for f in fractions:
                rows.append({"fold": r, "test": [t["name"] for t in grp], "gen_tokens": sum(n_gen(t) for t in tv),
                             "capacity": f, "cap": round(f * n_experts), "rates": res[str(f)]})
    return rows


def pooled(rows, policies):
    """Per capacity: gen-token hit pooled over the rows (weights = generated tokens; same k and layers in every trace),
    and the min / max over rows."""
    out = {}
    for f in sorted({r["capacity"] for r in rows}):
        sel = [r for r in rows if r["capacity"] == f and r["gen_tokens"] > 0]
        w = sum(r["gen_tokens"] for r in sel)
        out[f] = {p: {"pooled": sum(r["rates"][p]["gen"] * r["gen_tokens"] for r in sel) / w if w else None,
                      "min": min(r["rates"][p]["gen"] for r in sel) if sel else None,
                      "max": max(r["rates"][p]["gen"] for r in sel) if sel else None} for p in policies}
    return out


def load(trace_dir):
    files = sorted(pathlib.Path(trace_dir).glob("*.trace"))
    if not files:
        raise SystemExit(f"no *.trace files in {trace_dir}")
    traces = []
    for f in files:
        tr = cachesim.read_trace(f)
        tr["name"] = f.stem
        traces.append(tr)
    return traces


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir")
    ap.add_argument("--n-experts", type=int, required=True)
    ap.add_argument("--capacity", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--policies", default=",".join(cachesim.POLICIES))
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--fold", default="last", help="'last' (cachesim's split), 'all', or a residue r")
    ap.add_argument("--test-view", choices=["full", "gen_only"], default="full")
    ap.add_argument("--gen-cap", type=int)
    ap.add_argument("--train-part", choices=["all", "prompt", "gen"], default="all")
    ap.add_argument("--per-trace", action="store_true")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--devices", type=int, default=4)
    ap.add_argument("--prompt-updates", action="store_true")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    traces = load(a.trace_dir)
    pols = a.policies.split(",")
    fr = [float(x) for x in a.capacity.split(",")]
    fold = a.fold if a.fold in ("last", "all") else int(a.fold)
    rows = run(traces, a.n_experts, fr, pols, fold, a.holdout_every, a.test_view, a.gen_cap, a.train_part,
               a.per_trace, a.warm, a.session, a.devices, not a.prompt_updates)
    summ = pooled(rows, pols)
    head = (f"# {len(traces)} traces, fold={a.fold} (holdout every {a.holdout_every}), test_view={a.test_view}, "
            f"gen_cap={a.gen_cap}, train_part={a.train_part}, per_trace={a.per_trace}, warm={a.warm}, "
            f"session={a.session}, devices={a.devices}, prompt_readonly={not a.prompt_updates}; hit rate on gen tokens")
    print(head)
    print("fold  test                                gen  capacity  cap  " + "  ".join(f"{p:>10}" for p in pols))

    def cell(v):
        return f"{v:10.3f}" if v is not None else f"{'n/a':>10}"
    for r in rows:
        name = ",".join(t[:3] for t in r["test"])
        print(f"{r['fold']:4d}  {name:34s} {r['gen_tokens']:4d}  {r['capacity']:8.3f}  {r['cap']:3d}  " +
              "  ".join(cell(r["rates"][p]["gen"]) for p in pols))
    if len(rows) > len(fr):
        for stat in ("pooled", "min", "max"):
            for f in fr:
                print(f"{stat:>6}  {'':34s} {'':4s}  {f:8.3f}  {round(f * a.n_experts):3d}  " +
                      "  ".join(cell(summ[f][p][stat]) for p in pols))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({"header": head, "args": vars(a), "rows": rows,
                                                   "summary": {str(k): v for k, v in summ.items()}}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
