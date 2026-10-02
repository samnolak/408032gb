#!/usr/bin/env python3
"""Write a tiny random-weight GGUF with llama.cpp architecture "mimo2" (MiMo-V2.x-Flash shape).

Exercises: GQA with different K/V head dims (like 192/128), partial NEOX RoPE with a separate base on SWA layers,
per-layer KV-head counts (SWA layers have more, like 8 vs 4), sliding-window + global layers, attention sinks,
a leading dense FFN, MoE with sigmoid + bias routing and no shared expert.
Keys/tensors: llama.cpp @ec7630a src/models/mimo2.cpp, src/llama-arch.cpp. Needs gguf-py on PYTHONPATH.
"""
import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "mimo2"
CFG = dict(n_vocab=96, n_embd=64, n_layer=6, n_head=4, kv_global=1, kv_swa=2, k_dim=24, v_dim=16, n_rot=8,
           rope_base=10000.0, rope_base_swa=5000.0, n_swa=8, n_ff=96, n_expert=8, n_expert_used=2, n_ff_exp=32,
           rms_eps=1e-5, ctx=512)
GLOBAL_LAYERS = (0, 5)   # the rest are sliding-window layers


def build(path, seed=0):
    c, rng, p = CFG, np.random.default_rng(seed), ARCH
    E, L, H = c["n_embd"], c["n_layer"], c["n_head"]
    swa = [0 if i in GLOBAL_LAYERS else 1 for i in range(L)]
    kv = [c["kv_swa"] if s else c["kv_global"] for s in swa]
    w = GGUFWriter(path, ARCH)
    w.add_uint32(f"{p}.context_length", c["ctx"])
    w.add_uint32(f"{p}.embedding_length", E)
    w.add_uint32(f"{p}.block_count", L)
    w.add_uint32(f"{p}.feed_forward_length", c["n_ff"])
    w.add_uint32(f"{p}.attention.head_count", H)
    w.add_array(f"{p}.attention.head_count_kv", kv)
    w.add_uint32(f"{p}.attention.key_length", c["k_dim"])
    w.add_uint32(f"{p}.attention.value_length", c["v_dim"])
    w.add_float32(f"{p}.attention.layer_norm_rms_epsilon", c["rms_eps"])
    w.add_uint32(f"{p}.rope.dimension_count", c["n_rot"])
    w.add_float32(f"{p}.rope.freq_base", c["rope_base"])
    w.add_float32(f"{p}.rope.freq_base_swa", c["rope_base_swa"])
    w.add_uint32(f"{p}.attention.sliding_window", c["n_swa"])
    w.add_array(f"{p}.attention.sliding_window_pattern", [bool(s) for s in swa])
    w.add_uint32(f"{p}.expert_count", c["n_expert"])
    w.add_uint32(f"{p}.expert_used_count", c["n_expert_used"])
    w.add_uint32(f"{p}.expert_feed_forward_length", c["n_ff_exp"])
    toks = ["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(3, c["n_vocab"])]
    w.add_tokenizer_model("llama")
    w.add_token_list(toks)
    w.add_token_scores([0.0] * len(toks))
    w.add_token_types([2, 3, 3] + [1] * (len(toks) - 3))
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0); w.add_add_bos_token(False)

    def t(name, *shape, scale=None, ones=False):
        a = ((1.0 + 0.05 * rng.standard_normal(shape)) if ones else
             (scale if scale is not None else 1.0 / np.sqrt(shape[-1])) * rng.standard_normal(shape))
        w.add_tensor(name, a.astype(np.float32))

    t("token_embd.weight", c["n_vocab"], E, scale=1.0)
    t("output_norm.weight", E, ones=True)
    t("output.weight", c["n_vocab"], E)
    for i in range(L):
        b = f"blk.{i}"
        t(f"{b}.attn_norm.weight", E, ones=True)
        t(f"{b}.ffn_norm.weight", E, ones=True)
        t(f"{b}.attn_q.weight", H * c["k_dim"], E)
        t(f"{b}.attn_k.weight", kv[i] * c["k_dim"], E)
        t(f"{b}.attn_v.weight", kv[i] * c["v_dim"], E)
        t(f"{b}.attn_output.weight", E, H * c["v_dim"])
        t(f"{b}.attn_sinks.weight", H, scale=1.0)
        if i == 0:
            t(f"{b}.ffn_gate.weight", c["n_ff"], E)
            t(f"{b}.ffn_up.weight", c["n_ff"], E)
            t(f"{b}.ffn_down.weight", E, c["n_ff"])
        else:
            X, F = c["n_expert"], c["n_ff_exp"]
            t(f"{b}.ffn_gate_inp.weight", X, E)
            t(f"{b}.exp_probs_b.bias", X, scale=0.1)
            t(f"{b}.ffn_gate_exps.weight", X, F, E)
            t(f"{b}.ffn_up_exps.weight", X, F, E)
            t(f"{b}.ffn_down_exps.weight", X, E, F)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.seed)
    print("wrote", a.out)
