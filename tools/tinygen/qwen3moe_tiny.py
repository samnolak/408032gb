#!/usr/bin/env python3
"""Write a tiny random-weight GGUF with llama.cpp architecture "qwen3moe" (the Qwen3-30B-A3B family).

Purpose: a seconds-long CPU stand-in for the real proxy model of data/traces/real/REQUEST.json, for testing the
routing-trace and expert-probe tools (tools/g0) on the real qwen3moe graph. That graph drops all but the output rows
before the LAST layer's FFN (src/models/qwen3moe.cpp:123-125 @ec7630a) and names the router input `ffn_norm`
(qwen3moe.cpp:134), the two properties the tools depend on; the tiny glm5next / qwen4exp models do not have both.
Every layer is MoE (softmax gating, top-k weights renormalised: the build_moe_ffn call at qwen3moe.cpp:137), no
shared expert.

Keys and tensor names: llama.cpp @ec7630a src/llama-arch.cpp; required keys: src/models/qwen3moe.cpp
load_arch_hparams (lines 3-12) and llama_model_base::load_hparams (src/llama-model.cpp); shapes: qwen3moe.cpp
load_arch_tensors (lines 14-56). numpy shapes are ggml ne reversed. Needs gguf-py from the same llama.cpp checkout
on PYTHONPATH.
"""
import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "qwen3moe"

DEFAULT = dict(n_vocab=96, n_embd=64, n_layer=4, ctx=1024, rms_eps=1e-6, n_head=4, n_head_kv=2, head_dim=16,
               rope_base=10000.0, n_expert=16, n_expert_used=4, n_ff_exp=32, n_ff=128)


def build(path, cfg=DEFAULT, seed=0):
    c = dict(cfg)
    rng = np.random.default_rng(seed)
    w = GGUFWriter(path, ARCH)
    p = ARCH
    E, H, Hkv, D, L = c["n_embd"], c["n_head"], c["n_head_kv"], c["head_dim"], c["n_layer"]
    w.add_uint32(f"{p}.context_length", c["ctx"])
    w.add_uint32(f"{p}.embedding_length", E)
    w.add_uint32(f"{p}.block_count", L)
    w.add_uint32(f"{p}.feed_forward_length", c["n_ff"])
    w.add_uint32(f"{p}.attention.head_count", H)
    w.add_uint32(f"{p}.attention.head_count_kv", Hkv)
    w.add_uint32(f"{p}.attention.key_length", D)
    w.add_uint32(f"{p}.attention.value_length", D)
    w.add_float32(f"{p}.attention.layer_norm_rms_epsilon", c["rms_eps"])
    w.add_float32(f"{p}.rope.freq_base", c["rope_base"])
    w.add_uint32(f"{p}.expert_count", c["n_expert"])
    w.add_uint32(f"{p}.expert_used_count", c["n_expert_used"])
    w.add_uint32(f"{p}.expert_feed_forward_length", c["n_ff_exp"])
    # minimal SPM vocab: tokens are fed as ids, the strings never matter
    toks = ["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(3, c["n_vocab"])]
    w.add_tokenizer_model("llama")
    w.add_token_list(toks)
    w.add_token_scores([0.0] * len(toks))
    w.add_token_types([2, 3, 3] + [1] * (len(toks) - 3))
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)

    def t(name, *shape, scale=None, ones=False):
        if ones:
            a = (1.0 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
        else:
            s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
            a = (s * rng.standard_normal(shape)).astype(np.float32)
        w.add_tensor(name, a)

    V, X, F = c["n_vocab"], c["n_expert"], c["n_ff_exp"]
    t("token_embd.weight", V, E, scale=1.0)
    t("output_norm.weight", E, ones=True)
    t("output.weight", V, E)
    for i in range(L):
        b = f"blk.{i}"
        t(f"{b}.attn_norm.weight", E, ones=True)
        t(f"{b}.attn_q.weight", H * D, E)
        t(f"{b}.attn_k.weight", Hkv * D, E)
        t(f"{b}.attn_v.weight", Hkv * D, E)
        t(f"{b}.attn_output.weight", E, H * D)
        t(f"{b}.attn_q_norm.weight", D, ones=True)
        t(f"{b}.attn_k_norm.weight", D, ones=True)
        t(f"{b}.ffn_norm.weight", E, ones=True)
        t(f"{b}.ffn_gate_inp.weight", X, E, scale=1.0)          # wide logits: well separated top-k
        t(f"{b}.ffn_gate_exps.weight", X, F, E)
        t(f"{b}.ffn_up_exps.weight", X, F, E)
        t(f"{b}.ffn_down_exps.weight", X, E, F)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, seed=a.seed)
    print("wrote", a.out)
