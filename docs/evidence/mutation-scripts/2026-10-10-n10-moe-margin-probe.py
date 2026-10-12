#!/usr/bin/env python3
"""N10 probe (gap 3, N7 finding 7; written for the evidence audit of bd715da): the MoE decisions of one dumped run of
tests/test_*_f16fa_parity.py near a tie, from both oracles of the test class (B32: the plain f32 oracle, B: the
precision-aware oracle, incremental replay, as 2026-10-10-n10-margin-probe.py).

1. Every MoE row where llama.cpp's expert set (dump ffn_moe_topk-<il>) differs from an oracle's set: llama.cpp's set
   and, per oracle, the k+1 best candidates with their scores (9 decimals), the oracle's set and its margin between
   the k-th and the (k+1)-th candidate. The aware oracle replays a near-tie flip of llama.cpp (MOE_TIE_RTOL), so its
   set equals llama.cpp's there; its scores are those it chose from (the scores of a layer do not depend on the
   layer's choice).
2. The change of the margin by the precision over the MoE decisions (a = the k-th, b = the (k+1)-th candidate of B32;
   delta = (s_B[a] - s_B32[a]) - (s_B[b] - s_B32[b]), as 2026-10-10-n10-margin-probe.py, which prints |delta|):
   median and mean of |delta|, and the mean of max(0, -delta) (only a decrease of the margin can flip a decision),
   over all decisions and, if llama.cpp differs from B32 somewhere, over the decisions upstream of the first difference
   (earlier positions, and earlier or equal layers at its position: after a flip the oracles part).
   Then the expected number of flips per decision if the margins near 0 are spread uniformly with the run's density
   (number with margin < 1e-3, divided by the number of decisions, per 1e-3 of margin): density x X for X = median
   |delta|, mean |delta|, mean max(0, -delta); and times the decisions of a 288-token run (288 x the MoE layers).
Scores: glm5-next the selection score sigmoid(logit) + exp_probs_b (ref/glm5next.py:172, key "sel"); qwen4exp the
routing probability (ref/qwen4exp.py route_softmax, key "probs").
Usage (env as for the test module): 2026-10-10-n10-moe-margin-probe.py <test module> <run>
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))


def main(mod_name, run):
    mod = __import__(f"tests.{mod_name}", fromlist=["x"])
    cls = [c for c in vars(mod).values() if isinstance(c, type) and getattr(c, "RUN", None) == run
           and hasattr(c, "run_against_f32")][0]
    cls.setUpClass()
    hp, k = cls.hp, cls.hp["n_expert_used"]
    glm = "kda" in hp
    moe_layers = cls.MOE if glm else list(range(hp["n_layer"]))
    key, skey = ("ffn", "sel") if glm else ("moe", "probs")
    name = "glm5next" if glm else "qwen4exp"
    rows, first = [], None
    for gr in cls.R.graphs:
        g = gr["g"]
        for il in moe_layers:
            got = np.sort(cls.R.dump(g, f"ffn_moe_topk-{il}")[0].reshape(gr["n"], k).astype(int), 1)
            for r in range(gr["n"]):
                pos = gr["pos"][r]
                sets = {t: sorted(np.asarray(B[g]["layers"][il][key]["ids"][r]).astype(int).tolist())
                        for t, B in (("plain", cls.B32), ("aware", cls.B))}
                if got[r].tolist() != sets["plain"] and (first is None or (pos, il) < first):
                    first = (pos, il)
                if got[r].tolist() != sets["plain"] or got[r].tolist() != sets["aware"]:
                    rows.append((g, il, pos, r, got[r].tolist(), sets))
    for g, il, pos, r, ll, sets in rows:
        print(f"{name} {run} g{g} L{il} pos {pos}: llama.cpp {ll}")
        for t, B in (("plain", cls.B32), ("aware", cls.B)):
            s = np.asarray(B[g]["layers"][il][key][skey][r], dtype=np.float64)
            order = np.argsort(-s, kind="stable")[:k + 1]
            cand = ", ".join(f"{int(e)} {s[e]:.9f}" for e in order)
            print(f"    {t:5s} oracle set {sets[t]}; best {k + 1}: {cand}; "
                  f"margin k/k+1 {s[order[k - 1]] - s[order[k]]:.3e}")
    if not rows:
        print(f"{name} {run}: llama.cpp's expert sets equal both oracles' in every MoE row")
    dec = []
    for gr in cls.R.graphs:
        g = gr["g"]
        for il in moe_layers:
            for r in range(gr["n"]):
                s32 = np.asarray(cls.B32[g]["layers"][il][key][skey][r], dtype=np.float64)
                s16 = np.asarray(cls.B[g]["layers"][il][key][skey][r], dtype=np.float64)
                order = np.argsort(-s32, kind="stable")
                a, b = order[k - 1], order[k]
                dec.append((gr["pos"][r], il, s32[a] - s32[b], (s16[a] - s32[a]) - (s16[b] - s32[b])))
    per_run = 288 * len(moe_layers)
    subsets = [("all", dec)]
    if first is not None:
        subsets.append((f"upstream of the first llama.cpp / plain difference (pos {first[0]}, L{first[1]})",
                        [d for d in dec if (d[0], d[1]) <= first]))
    for label, ds in subsets:
        m = np.array([d[2] for d in ds])
        dl = np.array([d[3] for d in ds])
        dens = (m < 1e-3).sum() / len(m) / 1e-3
        stats = (float(np.median(np.abs(dl))), float(np.abs(dl).mean()), float(np.maximum(0.0, -dl).mean()))
        print(f"{name} {run} MoE decisions, {label}: {len(m)}, margin < 1e-3 in {int((m < 1e-3).sum())} "
              f"(density {dens:.3e} per unit margin); |delta| median {stats[0]:.3e}, mean {stats[1]:.3e}; "
              f"mean max(0, -delta) {stats[2]:.3e}")
        print(f"    expected flips per decision (density x median |delta|, x mean |delta|, x mean decrease): "
              + ", ".join(f"{dens * x:.3e}" for x in stats)
              + f"; per 288-token run ({per_run} decisions): " + ", ".join(f"{dens * x * per_run:.3f}" for x in stats))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
