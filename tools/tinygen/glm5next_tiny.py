#!/usr/bin/env python3
"""Write a tiny random-weight GGUF with llama.cpp architecture "glm5-next" (GLM-5.3-Flash).

Purpose: exercise every GLM mechanism (mHC + Sinkhorn, KDA, DSA with k-pool indexer, sigmoid
noaux_tc router, shared expert, clamped SwiGLU, leading dense layer) through llama.cpp's real
graph on a CPU in seconds, so oracles in ref/ and our engine can be checked without a GPU.

Key and tensor names: llama.cpp @ec7630a src/llama-arch.cpp; shapes: src/models/glm5-next.cpp
load_arch_tensors. numpy shapes are ggml ne reversed.
Needs gguf-py from the same llama.cpp checkout on PYTHONPATH.
"""
import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "glm5-next"

DEFAULT = dict(
    n_vocab=96, n_embd=64, n_layer=4, n_head=4, ctx=512,
    kda_head_dim=16, d_conv=4, kda_lower_bound=-5.0,
    q_lora=32, kv_lora=16, k_mla=16, v_mla=16,              # nope only: rope dim 0, like GLM
    idx_heads=2, idx_dim=16, idx_top_k=8, idx_kpool=4,
    n_expert=8, n_expert_used=2, n_ff_exp=32, n_shared=1, n_dense_lead=1, n_ff=96,
    w_scale=2.5, swiglu_limit=0.5, hc=4, sinkhorn_iters=20, hc_eps=1e-6, rms_eps=1e-5,
)


def layer_is_dsa(i):
    return i % 4 == 3                                        # GLM pattern: layers 3, 7, ...


def build(path, cfg=DEFAULT, seed=0):
    c = dict(cfg)
    rng = np.random.default_rng(seed)
    w = GGUFWriter(path, ARCH)
    p = ARCH
    E, L, H = c["n_embd"], c["n_layer"], c["n_head"]
    hc = c["hc"]
    w.add_uint32(f"{p}.context_length", c["ctx"])
    w.add_uint32(f"{p}.embedding_length", E)
    w.add_uint32(f"{p}.block_count", L)
    w.add_uint32(f"{p}.feed_forward_length", c["n_ff"])
    w.add_uint32(f"{p}.attention.head_count", H)
    w.add_array(f"{p}.attention.head_count_kv", [1 if layer_is_dsa(i) else 0 for i in range(L)])  # MLA latent = 1 MQA head; 0 = recurrent
    w.add_float32(f"{p}.attention.layer_norm_rms_epsilon", c["rms_eps"])
    w.add_uint32(f"{p}.attention.key_length_mla", c["k_mla"])
    w.add_uint32(f"{p}.attention.value_length_mla", c["v_mla"])
    w.add_uint32(f"{p}.attention.key_length", c["kv_lora"])
    w.add_uint32(f"{p}.attention.value_length", c["kv_lora"])
    w.add_uint32(f"{p}.attention.q_lora_rank", c["q_lora"])
    w.add_uint32(f"{p}.attention.kv_lora_rank", c["kv_lora"])
    w.add_uint32(f"{p}.rope.dimension_count", 0)
    w.add_uint32(f"{p}.ssm.conv_kernel", c["d_conv"])
    w.add_uint32(f"{p}.kda.head_dim", c["kda_head_dim"])
    w.add_float32(f"{p}.kda.gate_lower_bound", c["kda_lower_bound"])
    w.add_uint32(f"{p}.expert_count", c["n_expert"])
    w.add_uint32(f"{p}.expert_used_count", c["n_expert_used"])
    w.add_uint32(f"{p}.expert_feed_forward_length", c["n_ff_exp"])
    w.add_uint32(f"{p}.expert_shared_count", c["n_shared"])
    w.add_uint32(f"{p}.leading_dense_block_count", c["n_dense_lead"])
    w.add_float32(f"{p}.expert_weights_scale", c["w_scale"])
    w.add_bool(f"{p}.expert_weights_norm", True)
    w.add_uint32(f"{p}.expert_gating_func", 2)               # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
    w.add_array(f"{p}.swiglu_clamp_exp", [float(c["swiglu_limit"])] * L)
    w.add_uint32(f"{p}.attention.indexer.head_count", c["idx_heads"])
    w.add_uint32(f"{p}.attention.indexer.key_length", c["idx_dim"])
    w.add_uint32(f"{p}.attention.indexer.top_k", c["idx_top_k"])
    w.add_uint32(f"{p}.attention.indexer.kpool", c["idx_kpool"])
    w.add_bool(f"{p}.attention.indexer.kpool_select_tail", True)
    w.add_uint32(f"{p}.hyper_connection.count", hc)
    w.add_uint32(f"{p}.hyper_connection.sinkhorn_iterations", c["sinkhorn_iters"])
    w.add_float32(f"{p}.hyper_connection.epsilon", c["hc_eps"])
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

    def t(name, *shape, scale=None, ones=False, value=None):
        if value is not None:
            a = np.full(shape, value, dtype=np.float32)
        elif ones:
            a = (1.0 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
        else:
            s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
            a = (s * rng.standard_normal(shape)).astype(np.float32)
        w.add_tensor(name, a)

    V, D = c["n_vocab"], c["kda_head_dim"]
    d_inner = D * H
    t("token_embd.weight", V, E, scale=1.0)
    t("output_norm.weight", E, ones=True)
    t("output.weight", V, E)
    mix = (2 + hc) * hc
    for i in range(L):
        b = f"blk.{i}"
        t(f"{b}.attn_norm.weight", E, ones=True)
        t(f"{b}.ffn_norm.weight", E, ones=True)
        for k in ("attn", "ffn"):
            t(f"{b}.hc_{k}_fn.weight", mix, hc * E, scale=0.3)
            t(f"{b}.hc_{k}_base.weight", mix, scale=0.5)
            t(f"{b}.hc_{k}_scale.weight", 3, ones=True)
        if not layer_is_dsa(i):                               # KDA
            for q in ("q", "k", "v"):
                t(f"{b}.ssm_conv1d_{q}.weight", d_inner, 1, c["d_conv"], scale=0.5)
                t(f"{b}.attn_{q}.weight", d_inner, E)
            t(f"{b}.ssm_f_a.weight", D, E)
            t(f"{b}.ssm_f_b.weight", d_inner, D)
            t(f"{b}.ssm_beta.weight", H, E)
            t(f"{b}.ssm_a", H, scale=0.5)
            t(f"{b}.ssm_dt.bias", d_inner, scale=0.5)
            t(f"{b}.ssm_g_a.weight", D, E)
            t(f"{b}.ssm_g_b.weight", d_inner, D)
            t(f"{b}.ssm_norm.weight", D, ones=True)
            t(f"{b}.attn_output.weight", E, d_inner)
        else:                                                 # DSA (MLA, nope)
            ql, kl, km, vm = c["q_lora"], c["kv_lora"], c["k_mla"], c["v_mla"]
            t(f"{b}.attn_q_a_norm.weight", ql, ones=True)
            t(f"{b}.attn_kv_a_norm.weight", kl, ones=True)
            t(f"{b}.attn_q_a.weight", ql, E)
            t(f"{b}.attn_q_b.weight", H * km, ql)
            t(f"{b}.attn_kv_a_mqa.weight", kl, E)
            t(f"{b}.attn_k_b.weight", H, kl, km)
            t(f"{b}.attn_v_b.weight", H, vm, kl)
            t(f"{b}.attn_output.weight", E, H * vm)
            ih, idim, kp = c["idx_heads"], c["idx_dim"], c["idx_kpool"]
            t(f"{b}.indexer.k_norm.weight", idim, ones=True)
            t(f"{b}.indexer.k_norm.bias", idim, scale=0.1)
            t(f"{b}.indexer.proj.weight", ih, E)
            t(f"{b}.indexer.attn_k.weight", idim, E)
            t(f"{b}.indexer.attn_q_b.weight", ih * idim, ql)
            t(f"{b}.indexer_compressor_gate.weight", idim, E)
            t(f"{b}.indexer_compressor_ape.weight", kp, idim, scale=0.5)
        if i < c["n_dense_lead"]:
            t(f"{b}.ffn_gate.weight", c["n_ff"], E)
            t(f"{b}.ffn_up.weight", c["n_ff"], E)
            t(f"{b}.ffn_down.weight", E, c["n_ff"])
        else:
            X, F, S = c["n_expert"], c["n_ff_exp"], c["n_shared"]
            t(f"{b}.ffn_gate_inp.weight", X, E)
            t(f"{b}.exp_probs_b.bias", X, scale=0.1)
            t(f"{b}.ffn_gate_exps.weight", X, F, E)
            t(f"{b}.ffn_up_exps.weight", X, F, E)
            t(f"{b}.ffn_down_exps.weight", X, E, F)
            t(f"{b}.ffn_gate_shexp.weight", F * S, E)
            t(f"{b}.ffn_up_shexp.weight", F * S, E)
            t(f"{b}.ffn_down_shexp.weight", E, F * S)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=12, help="12 gives 3 DSA layers (3, 7, 11)")
    a = ap.parse_args()
    build(a.out, cfg=dict(DEFAULT, n_layer=a.layers), seed=a.seed)
    print("wrote", a.out)
