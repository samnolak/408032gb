#!/usr/bin/env python3
"""Write a tiny random-weight GGUF with llama.cpp architecture "mimo2" (MiMo-V2.x-Flash shape).

Exercises: GQA with different K/V head dims (like 192/128), partial NEOX RoPE with a separate base on SWA layers,
per-layer KV-head counts (SWA layers have more, like 8 vs 4), sliding-window + global layers, attention sinks,
a leading dense FFN, MoE with sigmoid + bias routing and no shared expert.
Keys/tensors: llama.cpp @ec7630a src/models/mimo2.cpp, src/llama-arch.cpp. Needs gguf-py on PYTHONPATH.

Options (N16, docs/reassessment/CORRECTNESS.md gap 12); without any of them the output is byte-identical to the G1
generator (tools/tinygen/run_parity_mimo.sh), and every option draws the same random numbers in the same order as the
default, so the trunk weights stay those of the default model:
  --fused-qkv        write blk.<i>.attn_qkv = [attn_q; attn_k; attn_v] (rows stacked) instead of the three tensors:
                     the fused path of mimo2.cpp:129-142 (create_tensor_qkv, src/llama-model.cpp:3424), which a GGUF
                     converted from a checkpoint with qkv_proj takes (conversion/mimo.py:100-141, MiMo-V2.5 and later)
  --sinks swa        attention sinks on the sliding-window layers only (TENSOR_NOT_REQUIRED, mimo2.cpp:57); the sinks
                     of the global layers are drawn but not written
  --value-scale V    write mimo2.attention.value_scale = V (mimo2.cpp:14-17; the graph scales attn_out by V,
                     mimo2.cpp:182-185)
  --mtp              append one MTP (NextN) block as layer n_layer (block_count n_layer + 1, nextn_predict_layers 1;
                     the converter's layout, conversion/mimo.py:153-160, 174-175, 287-298): a sliding-window layer with
                     the SWA KV-head count, fused attn_qkv (graph_mtp asserts it, mimo2.cpp:285), sinks, a dense FFN
                     (mimo2.cpp:375), nextn.eh_proj / enorm / hnorm and layer_output_norm (the head norm of the
                     fallback chain shared_head_norm -> layer_out_norm -> output_norm, mimo2.cpp:392-394; HF
                     final_layernorm maps to it, gguf-py tensor_mapping LAYER_OUT_NORM). Drawn after the trunk.
"""
import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "mimo2"
CFG = dict(n_vocab=96, n_embd=64, n_layer=6, n_head=4, kv_global=1, kv_swa=2, k_dim=24, v_dim=16, n_rot=8,
           rope_base=10000.0, rope_base_swa=5000.0, n_swa=8, n_ff=96, n_expert=8, n_expert_used=2, n_ff_exp=32,
           rms_eps=1e-5, ctx=512)
GLOBAL_LAYERS = (0, 5)   # the rest are sliding-window layers


def build(path, seed=0, fused_qkv=False, sinks="all", value_scale=None, mtp=False):
    c, rng, p = CFG, np.random.default_rng(seed), ARCH
    E, L, H = c["n_embd"], c["n_layer"], c["n_head"]
    swa = [0 if i in GLOBAL_LAYERS else 1 for i in range(L)] + ([1] if mtp else [])
    kv = [c["kv_swa"] if s else c["kv_global"] for s in swa]
    w = GGUFWriter(path, ARCH)
    w.add_uint32(f"{p}.context_length", c["ctx"])
    w.add_uint32(f"{p}.embedding_length", E)
    w.add_uint32(f"{p}.block_count", L + (1 if mtp else 0))
    if mtp:
        w.add_uint32(f"{p}.nextn_predict_layers", 1)
    w.add_uint32(f"{p}.feed_forward_length", c["n_ff"])
    w.add_uint32(f"{p}.attention.head_count", H)
    w.add_array(f"{p}.attention.head_count_kv", kv)
    w.add_uint32(f"{p}.attention.key_length", c["k_dim"])
    w.add_uint32(f"{p}.attention.value_length", c["v_dim"])
    w.add_float32(f"{p}.attention.layer_norm_rms_epsilon", c["rms_eps"])
    if value_scale is not None:
        w.add_float32(f"{p}.attention.value_scale", float(value_scale))
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

    def draw(*shape, scale=None, ones=False):
        return ((1.0 + 0.05 * rng.standard_normal(shape)) if ones else
                (scale if scale is not None else 1.0 / np.sqrt(shape[-1])) * rng.standard_normal(shape))

    def t(name, *shape, scale=None, ones=False):
        w.add_tensor(name, draw(*shape, scale=scale, ones=ones).astype(np.float32))

    def qkv(b, nkv, fused):
        q = draw(H * c["k_dim"], E)
        k = draw(nkv * c["k_dim"], E)
        v = draw(nkv * c["v_dim"], E)
        if fused:
            w.add_tensor(f"{b}.attn_qkv.weight", np.concatenate([q, k, v]).astype(np.float32))
        else:
            for nm, a in (("attn_q", q), ("attn_k", k), ("attn_v", v)):
                w.add_tensor(f"{b}.{nm}.weight", a.astype(np.float32))

    t("token_embd.weight", c["n_vocab"], E, scale=1.0)
    t("output_norm.weight", E, ones=True)
    t("output.weight", c["n_vocab"], E)
    for i in range(L):
        b = f"blk.{i}"
        t(f"{b}.attn_norm.weight", E, ones=True)
        t(f"{b}.ffn_norm.weight", E, ones=True)
        qkv(b, kv[i], fused_qkv)
        t(f"{b}.attn_output.weight", E, H * c["v_dim"])
        sk = draw(H, scale=1.0)
        if sinks == "all" or swa[i]:
            w.add_tensor(f"{b}.attn_sinks.weight", sk.astype(np.float32))
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
    if mtp:
        b = f"blk.{L}"
        t(f"{b}.attn_norm.weight", E, ones=True)
        t(f"{b}.ffn_norm.weight", E, ones=True)
        qkv(b, kv[L], True)
        t(f"{b}.attn_output.weight", E, H * c["v_dim"])
        t(f"{b}.attn_sinks.weight", H, scale=1.0)
        t(f"{b}.ffn_gate.weight", c["n_ff"], E)
        t(f"{b}.ffn_up.weight", c["n_ff"], E)
        t(f"{b}.ffn_down.weight", E, c["n_ff"])
        t(f"{b}.nextn.eh_proj.weight", E, 2 * E)
        t(f"{b}.nextn.enorm.weight", E, ones=True)
        t(f"{b}.nextn.hnorm.weight", E, ones=True)
        t(f"{b}.layer_output_norm.weight", E, ones=True)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fused-qkv", action="store_true", help="fused attn_qkv instead of attn_q/k/v (see above)")
    ap.add_argument("--sinks", choices=("all", "swa"), default="all", help="layers that carry attention sinks")
    ap.add_argument("--value-scale", type=float, default=None, help="write mimo2.attention.value_scale")
    ap.add_argument("--mtp", action="store_true", help="append one MTP (NextN) block (see above)")
    a = ap.parse_args()
    build(a.out, a.seed, a.fused_qkv, a.sinks, a.value_scale, a.mtp)
    print("wrote", a.out)
