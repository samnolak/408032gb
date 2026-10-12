"""Exact-capacity rows of the fine cachesim grid (cachesim.json keys are exact; its text table rounds the capacity to two
decimals, so 0.546/0.55 and 0.746/0.75 print alike). Usage: python3 fine_table.py <dir with cachesim-fine-<model>.json>"""
import json
import sys

D = sys.argv[1]
for m, E in (("qwen3-coder-30b-a3b", 128), ("glm-4.7-flash", 64)):
    j = json.load(open(f"{D}/cachesim-fine-{m}.json"))
    print(f"== {m}: gen-token hit (layer_min..layer_max over all test tokens for per-layer policies)")
    for c in ("0.546", "0.607", "0.746"):
        r = j["hit_rate_by_capacity"][c]
        print(f"capacity {c} cap {round(float(c) * E)}/{E}: " + "; ".join(
            f"{p} {v['gen']:.3f}" + (f" [{v['layer_min']:.3f}..{v['layer_max']:.3f}]" if v["layer_min"] is not None else "")
            for p, v in r.items()))
