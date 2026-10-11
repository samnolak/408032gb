#!/usr/bin/env python3
"""N12 probe (not a mutation check): how large each deviation of ik_llama.cpp from llama.cpp's glm5-next graph
(ref/glm5next_ik.py items 1-4) is on the tiny model, measured on ik's own dumps (tools/tinygen/run_parity_ik_glm5next.sh),
and the whole-model effect: ik's logits against the llama.cpp-semantics oracle ref/glm5next.py and against the
ik-semantics one (with the test's tie replays). Run from the repository root:
  IK_GLM5NEXT_DUMP=<work>/n12 IK_GLM5NEXT_GGUF=<work>/ik.gguf LLAMA_CPP_GGUF_PY=<gguf-py> \
      python3 docs/evidence/mutation-scripts/2026-10-11-n12-deviations.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import test_ik_glm5next_parity as T  # noqa: E402
from ref import glm5next as G, glm5next_ik as K  # noqa: E402
from ref.dsa import layer_norm  # noqa: E402

for run, cls in (("p64", T.IkP64), ("decode", T.IkDecode), ("fa16", T.IkFa16)):
    cls.setUpClass()
    R, W, hp, cfg = cls.R, cls.W, cls.hp, cls.cfg
    e1 = {"ik": 0.0, "llama.cpp": 0.0}
    e2 = {"rms eps": 0.0, "eps 0": 0.0}
    e4, pad_rows = {"ik cells": 0.0, "without cell 0": 0.0}, 0
    clamp_hits = 0
    for gr in R.graphs:
        g, n = gr["g"], gr["n"]
        for il in range(hp["n_layer"]):
            w = G.layer_weights(W, il)
            x = R.dump(g, f"ffn_norm-{il}")[0].reshape(n, -1)
            got = R.dump(g, f"ffn_out-{il}")[0]
            with K.semantics(cfg, hp):
                e1["ik"] = max(e1["ik"], T.rel(got, G.ffn_layer(x, w, hp, il)["out"]))
            f = G.ffn_layer(x, w, hp, il)
            e1["llama.cpp"] = max(e1["llama.cpp"], T.rel(got, f["out"]))
            if "gate" in f:
                clamp_hits += int((f["gate"] > hp["clamp_exp"][il]).sum())
        for il in cls.DSA:
            w = G.layer_weights(W, il)
            x = R.dump(g, f"attn_norm-{il}")[0].reshape(n, -1)
            ik_k = R.dump(g, f"dsa_indexer_k-{il}")[0]
            for name, eps in (("rms eps", hp["rms_eps"]), ("eps 0", 0.0)):
                want = layer_norm(x @ w["indexer.attn_k.weight"].T, w["indexer.k_norm.weight"],
                                  w["indexer.k_norm.bias"], eps)
                e2[name] = max(e2[name], T.rel(ik_k, want))
            qd, qne = R.dump(g, f"Qcur-{il}")
            qd = qd.reshape(qne[2], qne[1], qne[0]).transpose(1, 0, 2)
            cache, cne = R.dump(g, f"kv_cache-{il}")
            lat = cache.reshape(cne[1], cne[0])
            got = R.dump(g, f"kqv_out-{il}")[0]
            modes = ["fa_chunk"] * n if cfg["fa"] else None
            for name in e4:
                cells = []
                for t in gr["pos"]:
                    c = cls.seen[il][t]
                    kp = hp["kpool"]          # cell 0 by padding: not in the token's own tail, pool 0 not chosen
                    padded = 0 in c and 0 not in G.tail_cells(t, kp) and not set(range(kp)) <= set(c)
                    if name == "without cell 0" and padded:
                        c = [x_ for x_ in c if x_ != 0]
                        pad_rows += 1
                    cells.append(c)
                kqv = G.mla_attention(qd, lat, cells, np.asarray(w["attn_v_b.weight"]), hp["k_mla"], modes)
                e4[name] = max(e4[name], T.rel(got, kqv @ w["attn_output.weight"].T))
    lg = R.logits()
    ik_or = np.concatenate([cls.B[gr["g"]]["logits"] for gr in R.graphs])
    ml_or = G.forward(R.toks, W, T.model(T.GGUF)[1])["logits"]          # llama.cpp hparams (indexer eps 0)
    print(f"{run}: item 1 SwiGLU clamp, ffn_out from ik's ffn_norm: ik semantics {e1['ik']:.2e}, llama.cpp "
          f"{e1['llama.cpp']:.2e} (expert gate values above the limit in llama.cpp's own route: {clamp_hits})")
    print(f"{run}: item 2 indexer LayerNorm eps, from ik's attn_norm: rms eps {e2['rms eps']:.2e}, eps 0 {e2['eps 0']:.2e}")
    print(f"{run}: item 4 attention over ik's cells {e4['ik cells']:.2e}; without the padded cell 0 {e4['without cell 0']:.2e} "
          f"({pad_rows} row-layers padded)")
    print(f"{run}: logits ik vs ik-semantics oracle {T.rel(lg, ik_or):.2e}, argmax equal {(lg.argmax(1) == ik_or.argmax(1)).sum()}"
          f"/{len(lg)}; vs llama.cpp-semantics oracle {T.rel(lg, ml_or):.2e}, argmax equal "
          f"{(lg.argmax(1) == ml_or.argmax(1)).sum()}/{len(lg)}")
L = 10.0
print(f"item 1 bound for GLM-5.3-Flash (swiglu_limit {L}): max |min(silu(g), L) - silu(min(g, L))| = L * sigmoid(-L) = "
      f"{L / (1 + np.exp(L)):.3e}; for the tiny model's 0.5: {0.5 / (1 + np.exp(0.5)):.3e}")
a = np.random.default_rng(0).standard_normal((100000, 16))
d = np.abs(a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-5)
           - a / np.sqrt((a * a).sum(1, keepdims=True) + 1e-6)).max()
print(f"item 3 L2 norm, 1e5 random 16-vectors: max |ik - llama.cpp| = {d:.2e}")
