#!/usr/bin/env python3
"""N17 probe: ref/fattn.py one_chunk on llama.cpp's own inputs (tests/test_mimo2_f16fa_parity.py _op) under emulation
variants, per run: elements more than 0.5 / 1 binary16 spacing from llama.cpp's kqv_out, the largest distance, and the
share of bit-equal elements. Variants: "N10" (the attention scale float64(1/sqrt(d)) rounded to f32 and expf of the
float64 difference, as ref/fattn.py had it before N17), "f32 scale" (llama.cpp's 1.0f/sqrtf(d), still the float64 exp
argument), "final" (both as llama.cpp: ref/fattn.py now), "final, double dot" (the 192-wide heads without the AVX2
lanes), "final + fma S" (the softmax sum S = fma(S, ms, vs), one rounding: not in ref/fattn.py, PROVISIONAL explanation
of the f32-level rest).
usage: 2026-10-11-n17-fa-op-probe.py <repo> <work>/n17 <llama.cpp>"""
import os
import sys

repo, dump, llama = sys.argv[1:4]
os.environ.update(MIMO2_N17_DUMP=dump, LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"))
sys.path.insert(0, repo)
import numpy as np  # noqa: E402
from ref import fattn as F  # noqa: E402
from tests import test_mimo2_f16fa_parity as T  # noqa: E402
from tests.f16fa import ulp_diff  # noqa: E402

final = F._online_f16
final_dot = F._dot_f16
final_scale = F.kq_scale_f32


def online(exp_arg_f32=True, fma_s=False):
    def fn(qh, kc, vc, scale, sink=None):
        f32, f16 = F.f32, F.f16
        G, DV = qh.shape[0], vc.shape[1]
        qq = f16(qh)
        s_all = (f32(F._dot_f16(qq, kc)).astype(np.float32) * np.float32(scale)).astype(np.float64)
        M, S, acc = np.full(G, -np.inf), np.zeros(G), np.zeros((G, DV))
        arg = f32 if exp_arg_f32 else (lambda x: x)
        with np.errstate(invalid="ignore", over="ignore"):
            for i in range(kc.shape[0]):
                s = s_all[:, i]
                new = s > M
                ms = np.where(new, f32(np.exp(arg(np.where(new, M - s, 0.0)))), 1.0)
                vs = np.where(new, 1.0, f32(np.exp(arg(np.where(new, 0.0, s - M)))))
                M = np.where(new, s, M)
                acc = np.where(new[:, None], f16(acc * ms[:, None]), acc)
                acc = f16(acc + vc[i][None, :] * vs[:, None])
                S = f32(S * ms + vs) if fma_s else f32(f32(S * ms) + vs)
            if sink is not None:
                s = f32(sink)
                new = s > M
                ms = np.where(new, f32(np.exp(arg(np.where(new, M - s, 0.0)))), 1.0)
                vs = np.where(new, 1.0, f32(np.exp(arg(np.where(new, 0.0, s - M)))))
                acc = np.where(new[:, None], f32(acc * ms[:, None]), acc)
                S = f32(S * ms + vs) if fma_s else f32(f32(S * ms) + vs)
        inv = np.where(S == 0.0, 0.0, f32(1.0 / np.where(S == 0.0, 1.0, S)))
        return f32(acc * inv[:, None]), S
    return fn


VARIANTS = [("N10", dict(scale=lambda d: float(np.float32(1 / np.sqrt(d))), online=online(False))),
            ("f32 scale", dict(scale=final_scale, online=online(False))),
            ("final", dict(scale=final_scale, online=final)),
            ("final, double dot", dict(scale=final_scale, online=final, dot=lambda qq, kc: qq @ kc.T)),
            ("final + fma S", dict(scale=final_scale, online=online(True, True)))]
print(f"{'run':12s} {'variant':18s} {'> 0.5 sp':>9s} {'> 1 sp':>7s} {'max sp':>8s} {'bit-equal':>18s}")
for cls in (T.FaUbatch64, T.FaDecode, T.FaRollback, T.FaHeadDims192):
    cls.setUpClass()
    t = cls("test_flash_attention_op")
    for name, v in VARIANTS:
        F.kq_scale_f32, F._online_f16, F._dot_f16 = v["scale"], v["online"], v.get("dot", final_dot)
        try:
            half = one = eq = n = 0
            mx = 0.0
            for gr in t.R.graphs:
                for il in range(t.hp["n_layer"]):
                    got, want, mode = t._op(gr, il)
                    if mode != "fa_chunk":
                        continue
                    u = ulp_diff(got, want)
                    half += int((u > 0.5).sum())
                    one += int((u > 1.0).sum())
                    eq += int((got == want).sum())
                    n += u.size
                    mx = max(mx, float(u.max()))
        finally:
            F.kq_scale_f32, F._online_f16, F._dot_f16 = final_scale, final, final_dot
        print(f"{t.RUN:12s} {name:18s} {half:9d} {one:7d} {mx:8.3f} {eq:8d}/{n:<9d}")
