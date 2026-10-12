#!/usr/bin/env python3
"""N10 gap 2 probe: how far the qwen4exp oracle on quantized weights is from llama.cpp when it is not fed llama.cpp's
own inputs (docs/evidence/2026-10-10-quant-fa-parity.md, part B "Why per product"). On the pf64 run (one 64-token
ubatch) of each quantization type of tools/tinygen/run_parity_quant.sh:
  (1) the whole oracle from the token ids (ref/qwen4exp.py forward) against every layer output l_last-<il> and the
      logits, with three weight sets: dequantized + activation quantization (ref/quant.py), dequantized only, and the
      unquantized f32 model the types were made from; MoE expert sets that differ from llama.cpp's are listed with
      the oracle's margin between its 2nd and 3rd expert;
  (2) one layer at a time from llama.cpp's dumped layer input (hc_init / l_last-<il-1>, ple_embd), the oracle's own
      choices except where an expert set differs (then replayed from the dump and marked);
  (3) inside one layer (Q4_K_M layer 4): the component errors in computation order, and the f16 inputs of the
      hc_gate up projection that llama.cpp rounded differently (recovered from the dumped hc_gate by least squares
      over the 1024 outputs, then rounded to f16).
Read-only; numpy + the pinned gguf-py.

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-e2e-probe.py <repo root> <QUANT_WORK> <gguf-py>"""
import sys

import numpy as np

wt, work, gguf_py = sys.argv[1:4]
sys.path.insert(0, wt)
from ref import quant as QT, qwen4exp as Q  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402
from tests.test_qwen4exp_tiny_parity import read_hparams  # noqa: E402

sys.path.append(gguf_py)       # after the tests package: gguf-py has its own tests package
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def rel(got, want):
    got = np.asarray(got, float).reshape(np.shape(want))
    return float(np.abs(got - want).max() / (np.abs(want).max() + 1e-30))


def one_layer(res, emb, W, hp, il, ids=None):
    w = Q.layer_weights(W, il)
    eps = hp["eps"]
    if il in hp["ple_layers"]:
        res = Q.ple_forward(emb, res, w["ple_key.weight"], w["ple_value.weight"], w["ple_norm_key.weight"],
                            w["ple_norm_query.weight"], w["ple_norm_conv.weight"], w["ple_conv1d.weight"], eps,
                            hp["ple_ngram"])["out"]
    m = Q.hc_mix(res, w["hc_attn_norm.weight"], w["hc_attn_down.weight"], w["hc_attn_up.weight"],
                 w["hc_attn_inject.weight"], eps)
    blk = Q.gdn_layer(m["mixed"], w, hp) if hp["recurrent"][il] else \
        Q.attn_layer(m["mixed"], w, hp, hp["compress"][il] > 0)
    res = Q.hc_combine(res, blk["out"], m["inject"])
    m = Q.hc_mix(res, w["hc_ffn_norm.weight"], w["hc_ffn_down.weight"], w["hc_ffn_up.weight"],
                 w["hc_ffn_inject.weight"], eps)
    moe = Q.moe_layer(m["mixed"], w, hp["n_expert_used"], ids)
    return Q.hc_combine(res, moe["out"], m["inject"]), moe


f32 = GGUFReader(f"{work}/qwen4exp-f32.gguf")
Wf = {t.name: np.array(t.data, dtype=np.float64) for t in f32.tensors}
for qt in ("Q8_0", "Q4_K_M"):
    r = GGUFReader(f"{work}/{qt}/model.gguf")
    hp = read_hparams({f.name: f for f in r.fields.values()})
    R = Run("pf64", f"{work}/{qt}")
    n = R.graphs[0]["n"]
    sets = (("emulated", QT.load(r, dequantize, True)), ("no act quant", QT.load(r, dequantize, False)),
            ("f32 weights", Wf))
    print(f"== {qt} pf64 (1) whole oracle from the token ids: l_last rel error per layer 0..{hp['n_layer'] - 1}, "
          f"logits, greedy tokens equal, MoE sets differing (layer, position, llama.cpp, oracle, oracle margin)")
    for label, W in sets:
        A = Q.forward(R.final, W, hp)
        errs, mism = [], []
        for il in range(hp["n_layer"]):
            errs.append(rel(R.dump(0, f"l_last-{il}")[0], A["layers"][il]["combine_ffn"]))
            ids = np.sort(R.dump(0, f"ffn_moe_topk-{il}")[0].reshape(n, -1).astype(int), 1)
            own = np.sort(A["layers"][il]["moe"]["ids"], 1)
            for t in np.where((ids != own).any(1))[0]:
                p = np.sort(A["layers"][il]["moe"]["probs"][t])[::-1]
                mism.append((il, int(t), ids[t].tolist(), own[t].tolist(), f"{(p[1] - p[2]) / p[1]:.1e}"))
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(n, -1)
        print(f"  {label:13s} " + " ".join(f"{e:.1e}" for e in errs) + f" | logits {rel(lo, A['logits']):.2e} | "
              f"greedy {int((lo.argmax(1) == A['logits'].argmax(1)).sum())}/{n} | MoE sets differing {len(mism)}: "
              f"{mism[:3]}{' ...' if len(mism) > 3 else ''}")
    print(f"== {qt} pf64 (2) one layer at a time from llama.cpp's dumped layer input: l_last rel error per layer")
    emb = R.dump(0, "ple_embd")[0].reshape(n, -1)
    for label, W in sets[:2]:
        out = []
        for il in range(hp["n_layer"]):
            res = (R.dump(0, "hc_init")[0] if il == 0 else R.dump(0, f"l_last-{il - 1}")[0]).reshape(n, hp["hc"], -1)
            new, moe = one_layer(res, emb, W, hp, il)
            ids = R.dump(0, f"ffn_moe_topk-{il}")[0].reshape(n, -1).astype(int)
            flip = np.where((np.sort(ids, 1) != np.sort(moe["ids"], 1)).any(1))[0]
            mark = ""
            if len(flip):
                new, _ = one_layer(res, emb, W, hp, il, ids)
                mark = f"(MoE replayed at {flip.tolist()})"
            out.append(f"L{il} {rel(R.dump(0, f'l_last-{il}')[0], new):.1e}{mark}")
        print(f"  {label:13s} " + " ".join(out))

qt, il = "Q4_K_M", 4
r = GGUFReader(f"{work}/{qt}/model.gguf")
hp = read_hparams({f.name: f for f in r.fields.values()})
R = Run("pf64", f"{work}/{qt}")
n = 64
W = QT.load(r, dequantize, True)
w = Q.layer_weights(W, il)
eps = hp["eps"]


def D(name, k=0):
    return R.dump(0, name, k)[0]


print(f"== {qt} pf64 (3) layer {il} from its dumped input, component errors in computation order (rel)")
res = D(f"l_last-{il - 1}").reshape(n, hp["hc"], -1)
m = Q.hc_mix(res, w["hc_attn_norm.weight"], w["hc_attn_down.weight"], w["hc_attn_up.weight"],
             w["hc_attn_inject.weight"], eps)
g = Q.gdn_layer(m["mixed"], w, hp)
res2 = Q.hc_combine(res, g["out"], m["inject"])
m2 = Q.hc_mix(res2, w["hc_ffn_norm.weight"], w["hc_ffn_down.weight"], w["hc_ffn_up.weight"],
              w["hc_ffn_inject.weight"], eps)
moe = Q.moe_layer(m2["mixed"], w, hp["n_expert_used"])
for name, got, want in (("hc_norm[0]", D(f"hc_norm-{il}", 0), m["norm"]), ("hc_gate[0]", D(f"hc_gate-{il}", 0), m["gate"]),
                        ("hc_mixed[0]", D(f"hc_mixed-{il}", 0), m["mixed"]),
                        ("linear_attn_qkv_mixed", D(f"linear_attn_qkv_mixed-{il}"), g["qkv"]),
                        ("z", D(f"z-{il}"), g["z"]), ("final_output", D(f"final_output-{il}"), g["final"]),
                        ("linear_attn_out", D(f"linear_attn_out-{il}"), g["out"]),
                        ("hc_combine", D(f"hc_combine-{il}"), res2), ("hc_mixed[1]", D(f"hc_mixed-{il}", 1), m2["mixed"]),
                        ("ffn_out", D(f"ffn_out-{il}"), moe["out"])):
    print(f"  {name:22s} {rel(got, want):.1e}")
xn = D(f"hc_norm-{il}", 0).reshape(n, -1)
a16 = QT.act_quant(Q.silu((xn @ w["hc_attn_down.weight"].T) / hp["hc"]), "F16")
got = D(f"hc_gate-{il}", 0).reshape(n, -1)
Wu = np.asarray(w["hc_attn_up.weight"])
llama_in = QT.act_quant(np.linalg.lstsq(Wu, got.T, rcond=None)[0].T, "F16")
diff = np.argwhere(llama_in != a16)
print(f"  hc_gate[0] up-projection inputs (f16) that differ: {len(diff)} of {a16.size}: " +
      ", ".join(f"(token {t}, input {j}: oracle {a16[t, j]:.9g}, llama.cpp {llama_in[t, j]:.9g})" for t, j in diff))
