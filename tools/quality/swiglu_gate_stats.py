#!/usr/bin/env python3
"""How often the two clamped-SwiGLU semantics differ on real activations (N24, tools/quality/swiglu_clamp_experiment.sh).

  pre   silu(min(g, L)) * clamp(u, -L, L)    llama.cpp, HF transformers glm5_next, vLLM
  post  min(silu(g), L) * clamp(u, -L, L)    ik_llama.cpp, exllamav3
The two are equal wherever g <= L: then silu(g) <= g <= L for g >= 0 and silu(g) < 0 for g < 0, so both gate factors
are silu(g). For g > L the gate factor is silu(L) (pre) or min(silu(g), L) in (silu(L), L] (post): they differ by at most
L - silu(L) = L * sigmoid(-L) (4.54e-4 for L = 10; N12).

Input: a dump directory of tools/dump/dump_tensors.cpp with the prefixes ffn_moe_gate, ffn_moe_up, ffn_gate, ffn_up
(files <name>-<layer>__<k>.bin + .json). Pairs each gate tensor with the up tensor of the same layer and occurrence.
Output: per tensor kind, the element count, the count and share of gate values above L, the largest gate value, and
the largest |post - pre| of the SwiGLU output. Stdlib only (array module), so it runs on the server without numpy.
"""
import argparse
import array
import json
import math
import os
import re
import sys

NAME = re.compile(r"^(ffn_moe_gate|ffn_moe_up|ffn_gate|ffn_up)-(\d+)__(\d+)\.json$")
PAIRS = {"ffn_moe_gate": "ffn_moe_up", "ffn_gate": "ffn_up"}


def silu(x):
    if x < -80.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def load(path_json):
    with open(path_json) as f:
        meta = json.load(f)
    if meta.get("type") != "f32":
        raise ValueError("%s: type %r, expected f32" % (path_json, meta.get("type")))
    a = array.array("f")
    with open(path_json[:-5] + ".bin", "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    n = 1
    for d in meta["ne"]:
        n *= d
    if len(a) != n:
        raise ValueError("%s: %d values, ne %s" % (path_json, len(a), meta["ne"]))
    return a


def stats(dump_dir, limit):
    found = {}
    for fn in sorted(os.listdir(dump_dir)):
        m = NAME.match(fn)
        if m:
            found[(m.group(1), int(m.group(2)), int(m.group(3)))] = os.path.join(dump_dir, fn)
    out = {}
    for (name, il, k), pj in sorted(found.items()):
        if name not in PAIRS:
            continue
        up_p = found.get((PAIRS[name], il, k))
        g = load(pj)
        u = load(up_p) if up_p else None
        if u is not None and len(u) != len(g):
            u = None
        s = out.setdefault(name, {"tensors": 0, "elements": 0, "above_limit": 0, "max_gate": -math.inf,
                                  "max_abs_output_diff": 0.0, "layers_above": [], "paired_with_up": 0})
        s["tensors"] += 1
        s["elements"] += len(g)
        above = 0
        for i, x in enumerate(g):
            if x > s["max_gate"]:
                s["max_gate"] = x
            if x > limit:
                above += 1
                if u is not None:
                    uc = max(-limit, min(limit, u[i]))
                    d = abs((min(silu(x), limit) - silu(min(x, limit))) * uc)
                    if d > s["max_abs_output_diff"]:
                        s["max_abs_output_diff"] = d
        if u is not None:
            s["paired_with_up"] += 1
        if above:
            s["layers_above"].append({"layer": il, "k": k, "above": above})
        s["above_limit"] += above
    for s in out.values():
        s["share_above"] = s["above_limit"] / s["elements"] if s["elements"] else None
    return {"dump_dir": os.path.abspath(dump_dir), "limit": limit,
            "bound_gate_factor": limit / (1.0 + math.exp(limit)), "kinds": out,
            "_note": "pre and post differ only at gate values above the limit; ffn_gate also covers dense layers, "
                     "whose clamp setting is the GGUF's (swiglu_clamp_exp / swiglu_clamp_shexp)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("--limit", type=float, default=10.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = stats(a.dump_dir, a.limit)
    if not r["kinds"]:
        print("no ffn_moe_gate / ffn_gate dumps in %s" % a.dump_dir, file=sys.stderr)
        return 1
    for name, s in sorted(r["kinds"].items()):
        print("%-13s tensors %d, elements %d, gate > %g: %d (%.3g), max gate %.4g, max |post - pre| %.3g" % (
            name, s["tensors"], s["elements"], a.limit, s["above_limit"], s["share_above"], s["max_gate"],
            s["max_abs_output_diff"]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
