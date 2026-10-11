#!/usr/bin/env python3
"""N10 gap 2: summary of the tables tests/test_qwen4exp_quant_parity.py and tests/test_glm5next_quant_parity.py print
(the output of tools/tinygen/run_parity_quant.sh): per model, the worst emulated errors of the quantized products, the
range without activation quantization, the F32 products, the bounded products and their row agreement. Read-only.

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-summary.py <driver log> [...]"""
import re
import sys

NUM = r"(-|[\d.e+-]+)"
ROW = re.compile(r"^(Q8_0|Q4_K_M)\s+(\S+)\s+(.+?)\s+(\S+)\s+" + NUM + r"\s+" + NUM + r"\s+(-?\d+)\s+(-?\d+)\s+"
                 r"(?:-|([\d.e+-]+) \.\. ([\d.e+-]+))\s+(?:-|([\d.e+-]+) \.\. ([\d.e+-]+))\s*$")


def f(x):
    return float("nan") if x in (None, "-") else float(x)


for path in sys.argv[1:]:
    model, rows, agree = None, {}, {}
    for ln in open(path):
        m = re.match(r"^(qwen4exp|glm5-next): quantized products", ln)
        if m:
            model = m.group(1)
            continue
        m = re.match(r"^(Q8_0|Q4_K_M)\s+(\S+)\s+(.+): rows within TOL_OP (\d+)/(\d+)", ln)
        if m and model:
            a = agree.setdefault(model, {}).setdefault(m.group(3), [0, 0])
            a[0] += int(m.group(4))
            a[1] += int(m.group(5))
            continue
        m = ROW.match(ln)
        if m and model:
            g = m.groups()
            rows.setdefault(model, []).append(dict(qt=g[0], run=g[1], site=g[2], ty=g[3], es=f(g[4]), e=f(g[5]),
                                                   il=int(g[6]), g=int(g[7]), psmin=f(g[8]), psmax=f(g[9]),
                                                   pmin=f(g[10]), pmax=f(g[11])))
    print(path.split("/")[-1])
    for model, rs in rows.items():
        def pick(sel, key, fn=max):
            v = [r for r in rs if sel(r) and r[key] == r[key]]
            return fn(v, key=lambda r: r[key]) if v else None

        def show(label, r, key):
            print(f"  {label}: {r[key]:.2e} ({r['qt']} {r['run']} {r['site']} {r['ty']} L{r['il']} g{r['g']})"
                  if r else f"  {label}: none")
        quant = lambda r: (r["ty"] != "F32" and r["es"] == r["es"] and r["psmin"] == r["psmin"]  # noqa: E731
                           and "gather" not in r["site"])
        bounded = lambda r: r["psmin"] != r["psmin"] and r["pmin"] == r["pmin"] and "ple_" not in r["site"]  # noqa
        print(f" {model}: {len(rs)} table rows")
        show("quantized products (scaled check): worst scaled error", pick(quant, "es"), "es")
        show("quantized products (scaled check): worst normwise error", pick(quant, "e"), "e")
        show("  without activation quantization: smallest scaled error of a graph", pick(quant, "psmin", min), "psmin")
        show("  without activation quantization: smallest per-run worst scaled error", pick(quant, "psmax", min),
             "psmax")
        show("  without activation quantization: smallest normwise error of a graph", pick(quant, "pmin", min), "pmin")
        show("  without activation quantization: largest normwise error", pick(quant, "pmax"), "pmax")
        show("F32 products: worst scaled error", pick(lambda r: r["ty"] == "F32", "es"), "es")
        show("F32 products: worst normwise error", pick(lambda r: r["ty"] == "F32", "e"), "e")
        show("bounded products: largest share of the bound", pick(bounded, "es"), "es")
        show("bounded products: worst normwise error", pick(bounded, "e"), "e")
        show("PLE module: worst normwise error", pick(lambda r: "ple_" in r["site"], "e"), "e")
        for site, (ok, n) in agree.get(model, {}).items():
            print(f"  rows within TOL_OP, all runs: {site}: {ok}/{n}")
