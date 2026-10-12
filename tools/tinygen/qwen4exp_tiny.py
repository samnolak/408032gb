#!/usr/bin/env python3
"""Write a tiny random-weight GGUF with llama.cpp architecture "qwen4exp" (reassessment candidate #1).

Purpose: run every mechanism of llama.cpp's qwen4exp graph through the real graph on a CPU in seconds,
so the numpy oracle ref/qwen4exp.py can be checked against tensor dumps without a GPU:
  * low-rank hyper-connections (hc streams) in place of every layer norm, incl. the final mixer;
  * gated delta net (linear attention) layers with n_k_heads < n_v_heads and a sigmoid output gate;
  * full-attention layers with a gated query, q/k RMS norms, partial interleaved M-RoPE (IMRoPE),
    GQA, and QSA block-sparse selection (k-pool indexer: mean-pooled keys, top blocks + tail);
    one full-attention layer has compress ratio 0 and therefore takes the dense path;
  * softmax-gated MoE (top-k, renormalised, no scale) with a sigmoid-gated shared expert;
  * one PLE layer: n-gram hash embedding (uint64 multiply/xor hash, EOS reset), gated value,
    grouped norms and a dilated causal depthwise conv.
No MTP block by default (nextn_predict_layers = 0). --mtp appends one (nextn_predict_layers = 1, block_count + 1):
a full-attention QSA layer with blk.<L>.nextn.{eh_proj,enorm,hnorm,hc_head_*} (qwen4exp.cpp:290-302, the converter
conversion/qwen4exp.py:28-36, 84-88); its weights are drawn after the trunk's, so the trunk equals the default model.
--quant-dims (N10, gap 2 of docs/reassessment/CORRECTNESS.md) writes the same architecture with the input width of
every quantizable matrix a multiple of QK_K = 256, so that llama-quantize can give it k-quants (it falls back to a
32-block type or F16 for rows not divisible by the block size, src/llama-quant.cpp tensor_type_fallback): n_embd 256,
head_dim 64 (n_head * head_dim = 256), ssm_state 64 (ssm inner size 4 * 64 = 256), expert and shared-expert widths
256. The hyper-connection rank stays 8, so hc_*_up (input width 8) stays F16 under every quantization type.
--text-vocab writes a byte-level BPE vocabulary instead of the id-only one (tools/tinygen/text_vocab.py) so that a
llama-server on the model accepts text; --ctx sets context_length (the server caps each slot at it). Defaults unchanged.

Keys and tensor names: llama.cpp @ec7630a src/llama-arch.cpp; required keys: src/models/qwen4exp.cpp
load_arch_hparams (lines 26-169) and llama_model_base::load_hparams (src/llama-model.cpp:1298); shapes:
src/models/qwen4exp.cpp load_arch_tensors (lines 171-303). numpy shapes are ggml ne reversed.
Needs gguf-py from the same llama.cpp checkout on PYTHONPATH.
"""
import argparse

import numpy as np
from gguf import GGUFWriter

from text_vocab import N_TEXT_VOCAB, write_byte_bpe_vocab

ARCH = "qwen4exp"

DEFAULT = dict(
    n_vocab=96, n_embd=64, n_layer=8, ctx=512, rms_eps=1e-6,
    hc=4, hc_low_rank=8,
    # full attention: head dim 16, partial rotary 8 (4 pairs); sections [1,1,1,1] put pair 3 in the 4th
    # IMRoPE section, which text queries rotate with position 0 (see ref/qwen4exp.py imrope)
    n_head=4, n_head_kv=2, head_dim=16, n_rot=8, rope_sections=[1, 1, 1, 1], rope_base=1000.0,
    full_attn_interval=4, dense_attn_layers=(7,),          # layer 3: QSA, layer 7: compress ratio 0 -> dense
    idx_heads=4, idx_dim=16, idx_top_k=8, kpool=4,
    # gated delta net: 2 key heads, 4 value heads, head dim 16
    d_conv=4, ssm_state=16, ssm_groups=2, ssm_v_heads=4,
    n_expert=8, n_expert_used=2, n_ff_exp=32, n_ff_shexp=48, n_ff=96,
    # PLE on layer 1: 3-grams, 2 heads per n-gram -> 4 heads of n_embd/4 = 16
    ple_layer=1, ple_ngram=3, ple_heads_per_ngram=2, ple_kernel=3, ple_eos=2,
    ple_vocab_sizes=[37, 41, 43, 47],
    # 45-bit odd multipliers like the converter comment says (conversion/qwen4exp.py:62); products with
    # token ids < 2^18 stay below 2^63, so uint64 (llama.cpp) and int64 (reference) arithmetic agree
    ple_multipliers=[0x1A2B3C4D5E1, 0x0F1E2D3C4B5, 0x13579BDF023],
)


# --quant-dims: every quantizable matrix has an input width divisible by 256 except hc_*_up (rank 8 -> F16) and the PLE
# table (row width n_embd / 4 = 64 -> a 32-block type); see the module docstring
QUANT = dict(DEFAULT, n_embd=256, head_dim=64, ssm_state=64, n_ff_exp=256, n_ff_shexp=256, n_ff=256)


def is_recurrent(i, c):
    return (i + 1) % c["full_attn_interval"] != 0          # qwen4exp.cpp:154 (default rule)


def build(path, cfg=DEFAULT, seed=0, mtp=False, text_vocab=False):
    c = dict(cfg)
    rng = np.random.default_rng(seed)
    w = GGUFWriter(path, ARCH)
    p = ARCH
    E, L, H, D = c["n_embd"], c["n_layer"], c["n_head"], c["head_dim"]
    hc, lr = c["hc"], c["hc_low_rank"]
    hc_dim = hc * E
    recr = [is_recurrent(i, c) for i in range(L)]
    n_all = L + int(mtp)                                      # block_count counts the MTP block
    recr_all = recr + [False] * int(mtp)
    n_ple_heads = (c["ple_ngram"] - 1) * c["ple_heads_per_ngram"]
    ple_dim = E // n_ple_heads                                # ple_key is {n_embd, hc_dim}: n_heads*dim = n_embd
    assert ple_dim * n_ple_heads == E and len(c["ple_vocab_sizes"]) == n_ple_heads
    assert recr[c["ple_layer"]], "PLE must sit on a linear attention layer (qwen4exp.cpp:159-163)"

    w.add_uint32(f"{p}.context_length", c["ctx"])
    w.add_uint32(f"{p}.embedding_length", E)
    w.add_uint32(f"{p}.block_count", n_all)
    if mtp:
        w.add_nextn_predict_layers(1)
    w.add_uint32(f"{p}.feed_forward_length", c["n_ff"])
    w.add_uint32(f"{p}.attention.head_count", H)
    w.add_uint32(f"{p}.attention.head_count_kv", c["n_head_kv"])
    w.add_uint32(f"{p}.attention.key_length", D)
    w.add_uint32(f"{p}.attention.value_length", D)
    w.add_float32(f"{p}.attention.layer_norm_rms_epsilon", c["rms_eps"])
    w.add_uint32(f"{p}.rope.dimension_count", c["n_rot"])
    w.add_float32(f"{p}.rope.freq_base", c["rope_base"])
    w.add_rope_dimension_sections(c["rope_sections"])
    w.add_uint32(f"{p}.expert_count", c["n_expert"])
    w.add_uint32(f"{p}.expert_used_count", c["n_expert_used"])
    w.add_uint32(f"{p}.expert_feed_forward_length", c["n_ff_exp"])
    w.add_uint32(f"{p}.expert_shared_feed_forward_length", c["n_ff_shexp"])
    w.add_uint32(f"{p}.ssm.conv_kernel", c["d_conv"])
    w.add_uint32(f"{p}.ssm.state_size", c["ssm_state"])
    w.add_uint32(f"{p}.ssm.group_count", c["ssm_groups"])
    w.add_uint32(f"{p}.ssm.time_step_rank", c["ssm_v_heads"])
    w.add_uint32(f"{p}.ssm.inner_size", c["ssm_state"] * c["ssm_v_heads"])
    w.add_recurrent_layers(recr_all)                          # as conversion/qwen.py:386-388 writes it
    w.add_full_attention_interval(c["full_attn_interval"])
    w.add_hyper_connection_count(hc)
    w.add_hyper_connection_low_rank(lr)
    w.add_uint32(f"{p}.attention.indexer.head_count", c["idx_heads"])
    w.add_uint32(f"{p}.attention.indexer.key_length", c["idx_dim"])
    w.add_uint32(f"{p}.attention.indexer.top_k", c["idx_top_k"])
    w.add_attention_compress_ratios(
        [0 if recr[i] or i in c["dense_attn_layers"] else c["kpool"] for i in range(L)]
        + [c["kpool"]] * int(mtp))                            # the MTP block is QSA too (conversion/qwen4exp.py:84-88)
    w.add_ple_layers([c["ple_layer"]])
    w.add_ple_ngram_size(c["ple_ngram"])
    w.add_ple_heads_per_ngram(c["ple_heads_per_ngram"])
    w.add_ple_conv_kernel(c["ple_kernel"])
    w.add_ple_eos_token_id(c["ple_eos"])
    w.add_embedding_length_per_layer_input(ple_dim)
    offsets = [int(x) for x in np.cumsum([0] + c["ple_vocab_sizes"][:-1])]
    w.add_ple_layer_multipliers(c["ple_multipliers"])
    w.add_ple_head_offsets(offsets)
    w.add_ple_head_vocab_sizes(c["ple_vocab_sizes"])
    if text_vocab:                                            # byte-level BPE: the model accepts text (text_vocab.py)
        assert write_byte_bpe_vocab(w, bos=1, eos=c["ple_eos"]) == c["n_vocab"]
    else:
        # minimal SPM vocab: tokens are fed as ids, the strings never matter
        toks = ["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(3, c["n_vocab"])]
        w.add_tokenizer_model("llama")
        w.add_token_list(toks)
        w.add_token_scores([0.0] * len(toks))
        w.add_token_types([2, 3, 3] + [1] * (len(toks) - 3))
        w.add_bos_token_id(1)
        w.add_eos_token_id(c["ple_eos"])
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

    V = c["n_vocab"]
    t("token_embd.weight", V, E, scale=1.0)
    t("output.weight", V, E)
    t("output_hc_norm.weight", hc, E, ones=True)              # ggml {n_embd, hc}
    t("output_hc_down.weight", lr, hc_dim)                    # ggml {hc_dim, hc_lr}
    t("output_hc_up.weight", hc_dim, lr)                      # ggml {hc_lr, hc_dim}
    t("per_layer_token_embd.weight", sum(c["ple_vocab_sizes"]), ple_dim, scale=1.0)

    Sk, Hk, Hv = c["ssm_state"], c["ssm_groups"], c["ssm_v_heads"]
    key_dim, value_dim = Sk * Hk, Sk * Hv
    X, F, Fs = c["n_expert"], c["n_ff_exp"], c["n_ff_shexp"]
    for i in range(n_all):
        b = f"blk.{i}"
        for k in ("attn", "ffn"):
            t(f"{b}.hc_{k}_norm.weight", hc, E, ones=True)
            t(f"{b}.hc_{k}_down.weight", lr, hc_dim)
            t(f"{b}.hc_{k}_up.weight", hc_dim, lr)
            t(f"{b}.hc_{k}_inject.weight", hc, hc_dim)
        if recr_all[i]:                                       # gated delta net
            t(f"{b}.attn_qkv.weight", 2 * key_dim + value_dim, E)
            t(f"{b}.attn_gate.weight", value_dim, E)
            t(f"{b}.ssm_conv1d.weight", 2 * key_dim + value_dim, c["d_conv"], scale=0.5)
            t(f"{b}.ssm_dt.bias", Hv, scale=0.5)
            w.add_tensor(f"{b}.ssm_a", (-np.exp(0.5 * rng.standard_normal(Hv))).astype(np.float32))  # -exp(A_log)
            t(f"{b}.ssm_beta.weight", Hv, E)
            t(f"{b}.ssm_alpha.weight", Hv, E)
            t(f"{b}.ssm_norm.weight", Sk, ones=True)
            t(f"{b}.ssm_out.weight", E, value_dim)
        else:                                                 # full attention (+ QSA indexer)
            t(f"{b}.attn_q.weight", 2 * D * H, E)               # [q | gate] interleaved per head
            t(f"{b}.attn_k.weight", D * c["n_head_kv"], E)
            t(f"{b}.attn_v.weight", D * c["n_head_kv"], E)
            t(f"{b}.attn_output.weight", E, D * H)
            t(f"{b}.attn_q_norm.weight", D, ones=True)
            t(f"{b}.attn_k_norm.weight", D, ones=True)
            t(f"{b}.indexer.q_proj.weight", c["idx_heads"] * c["idx_dim"], E)
            t(f"{b}.indexer.k_proj.weight", c["idx_dim"], E)
            t(f"{b}.indexer.q_norm.weight", c["idx_dim"], ones=True)
            t(f"{b}.indexer.k_norm.weight", c["idx_dim"], ones=True)
        if i == c["ple_layer"]:
            t(f"{b}.ple_key.weight", hc_dim, E)
            t(f"{b}.ple_value.weight", E, E)
            t(f"{b}.ple_norm_key.weight", hc, E, ones=True)
            t(f"{b}.ple_norm_query.weight", hc, E, ones=True)
            t(f"{b}.ple_norm_conv.weight", hc, E, ones=True)
            t(f"{b}.ple_conv1d.weight", hc_dim, c["ple_kernel"], scale=0.5)
        t(f"{b}.ffn_gate_inp.weight", X, E)
        t(f"{b}.ffn_gate_exps.weight", X, F, E)
        t(f"{b}.ffn_up_exps.weight", X, F, E)
        t(f"{b}.ffn_down_exps.weight", X, E, F)
        t(f"{b}.ffn_gate_inp_shexp.weight", E)
        t(f"{b}.ffn_gate_shexp.weight", Fs, E)
        t(f"{b}.ffn_up_shexp.weight", Fs, E)
        t(f"{b}.ffn_down_shexp.weight", E, Fs)
        if i == L:                                            # MTP block (qwen4exp.cpp:290-302)
            t(f"{b}.nextn.eh_proj.weight", E, 2 * E)            # ggml {2*n_embd, n_embd}: [e ; h_s] -> n_embd
            t(f"{b}.nextn.enorm.weight", E, ones=True)
            t(f"{b}.nextn.hnorm.weight", hc, E, ones=True)         # ggml {n_embd, hc}
            t(f"{b}.nextn.hc_head_norm.weight", hc, E, ones=True)
            t(f"{b}.nextn.hc_head_down.weight", lr, hc_dim)
            t(f"{b}.nextn.hc_head_up.weight", hc_dim, lr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mtp", action="store_true", help="append the MTP block (nextn_predict_layers = 1)")
    ap.add_argument("--quant-dims", action="store_true",
                    help="input widths divisible by QK_K = 256 so that llama-quantize gives k-quants (QUANT)")
    ap.add_argument("--text-vocab", action="store_true",
                    help="byte-level BPE vocabulary (%d tokens, tools/tinygen/text_vocab.py) instead of the minimal "
                         "id-only one, so that a server on this model accepts text; changes token_embd/output shapes "
                         "and therefore every weight drawn after them" % N_TEXT_VOCAB)
    ap.add_argument("--ctx", type=int, default=DEFAULT["ctx"],
                    help="context_length written to the GGUF (llama-server caps each slot at it); default %(default)s")
    a = ap.parse_args()
    cfg = dict(QUANT if a.quant_dims else DEFAULT, ctx=a.ctx, **({"n_vocab": N_TEXT_VOCAB} if a.text_vocab else {}))
    build(a.out, cfg=cfg, seed=a.seed, mtp=a.mtp, text_vocab=a.text_vocab)
    print("wrote", a.out)
