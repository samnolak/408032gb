#!/usr/bin/env python3
"""N10 probe (gap 3, N7 finding 7): how close the discrete choices of the tiny models are to a tie, against how much the
R1 default precision moves the scores they are made from. For one dumped run of tests/test_*_f16fa_parity.py: every
MoE decision (token x MoE layer) and every sparse-attention decision (token x selection layer) of the plain f32 oracle
(B32): the margin between the last chosen and the first left-out candidate, and the change of those two scores when
the oracle runs at the run's precision (B, the precision-aware oracle, which tracks llama.cpp). A decision can flip when
the change exceeds the margin. Exact ties (margin 0: pools whose rectified scores are all 0) are counted apart: their
order is ggml_top_k's, not a precision effect. After a flip the two oracles part, so changes downstream of it are large.
Usage (env as for the test module): 2026-10-10-n10-margin-probe.py <test module> <run>
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
    sel_layers = cls.DSA if glm else cls.QSA
    out = {}
    for kind in ("moe", "sel"):
        margins, moves = [], []
        for gr in cls.R.graphs:
            for il in (moe_layers if kind == "moe" else sel_layers):
                for r in range(gr["n"]):
                    if kind == "moe":
                        F32 = cls.B32[gr["g"]]["layers"][il]["ffn" if glm else "moe"]
                        F16 = cls.B[gr["g"]]["layers"][il]["ffn" if glm else "moe"]
                        s32 = F32["sel"][r] if glm else F32["probs"][r]
                        s16 = F16["sel"][r] if glm else F16["probs"][r]
                        order = np.argsort(-s32, kind="stable")
                        a, b = order[k - 1], order[k]
                    else:
                        D32 = cls.B32[gr["g"]]["layers"][il]["dsa" if glm else "attn"]
                        D16 = cls.B[gr["g"]]["layers"][il]["dsa" if glm else "attn"]
                        s32, s16 = D32["scores"][r], D16["scores"][r]
                        vis = np.where(np.isfinite(s32))[0]
                        n_top = hp["idx_top_k"] // hp["kpool"]
                        if len(vis) <= n_top:
                            continue                # every visible pool is taken: no choice
                        order = vis[np.argsort(-s32[vis], kind="stable")]
                        a, b = order[n_top - 1], order[n_top]
                    margins.append(s32[a] - s32[b])
                    moves.append(abs((s16[a] - s32[a]) - (s16[b] - s32[b])))
        m, mv = np.array(margins), np.array(moves)
        med = float(np.median(mv)) if len(mv) else 0.0
        out[kind] = (len(m), int((m == 0).sum()), int((m < mv).sum()), float(np.median(m)) if len(m) else 0.0, med,
                     float(mv.max()) if len(mv) else 0.0, int(((m > 0) & (m < med)).sum()), int((m < 1e-3).sum()))
    for kind, (n, ties, flips, med_m, med_mv, max_mv, below_med, small) in out.items():
        print(f"{mod_name.split('_')[1]} {run:8s} {'MoE experts' if kind == 'moe' else 'sparse selection'}: "
              f"{n} decisions, exact ties (margin 0) {ties}, median margin {med_m:.3e}; change of the margin by the "
              f"precision: median {med_mv:.3e}, max {max_mv:.3e}; 0 < margin < median change in {below_med}; "
              f"margin < change in {flips}; margin < 1e-3 in {small}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
