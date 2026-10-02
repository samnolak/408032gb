#!/usr/bin/env python3
"""Run DeepSeek's unchanged reference `inference/model.py` on a CPU and dump what it computes.

    python3 tools/ds41/run_reference.py --mode reference --out fixture.npz
    python3 tools/ds41/run_reference.py --mode exact     --out fixture_exact.npz

What it does (ADR-008):
  * installs tools/ds41/kernel_cpu.py as the module `kernel` (the reference kernels are TileLang GPU
    code), then imports third_party/deepseek-v41-flash/inference/model.py as is;
  * builds a tiny model whose config exercises every V4.1 mechanism: sliding-window-only layers,
    ratio-2 and ratio-1 compressed layers, KV sources / index sources / reuse layers, the candidate
    pre-filter (two-level indexer), Engram on two layers, FP8 dense and FP4 expert weights;
  * fills every parameter from a seeded generator (the reference leaves them uninitialised);
  * runs one prefill and several single-token decode steps, records the inputs and outputs of the
    kernels and of the main modules, and checks that decode step t reproduces row t of a longer
    prefill (the reference's own consistency);
  * writes weights (dequantised to float32, exactly representable), config, tokens and records to
    one .npz that ref/ds41.py (numpy, no torch) is tested against.

The model runs in float32 instead of the reference's bf16 default: bf16-typed parameters are
created as float32, so results differ from a GPU bf16 run by bf16 rounding only. Quantisation that
is part of the model function (FP8 activations, FP4 index q/k, FP4 compressed KV) is kept in
--mode reference and removed in --mode exact.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REF = os.path.join(ROOT, "third_party", "deepseek-v41-flash", "inference")

sys.path.insert(0, HERE)
import kernel_cpu  # noqa: E402

sys.modules["kernel"] = kernel_cpu
sys.path.insert(0, REF)
import engram as R_engram  # noqa: E402
import model as R  # noqa: E402

TINY = dict(
    max_batch_size=1, max_seq_len=64, temperature=0, dtype="fp8", expert_dtype="fp4",
    vocab_size=128, dim=64, moe_inter_dim=64, n_layers=8, n_mtp_layers=0, n_heads=4,
    n_routed_experts=8, n_shared_experts=1, n_activated_experts=3, score_func="sqrtsoftplus",
    route_scale=1.5, swiglu_limit=0.5,
    q_lora_rank=32, head_dim=64, rope_head_dim=16, norm_eps=1e-20, o_groups=2, o_lora_rank=32,
    window_size=8,
    # 0: window only | 1-3: ratio 2 | 4-7: ratio 1.  KV sources 1, 3, 4; layer 2 reuses 1; 5 reuses 4;
    # 6 re-indexes inside the candidates chosen at 4; 7 reuses 6.
    compress_ratios=(0, 2, 2, 2, 1, 1, 1, 1), kv_source_layers=(1, 3, 4), index_source_layers=(1, 3, 4, 6),
    compress_rope_theta=160000.0, original_seq_len=16, rope_theta=10000.0, rope_factor=4, beta_fast=32, beta_slow=1,
    # 16 index heads: with 4, exact-zero scores (all heads rectified to 0) tie at the top-k cut and the
    # reference picks differently for a 37- and a 43-token prefill (docs/evidence, DeepSeek G7 track)
    index_n_heads=16, index_head_dim=32, index_topk=6,
    candidate_source_layer=4, candidate_topk_blocks=3, candidate_block_size=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 3), engram_max_ngram_size=4, engram_vocab_size=50, engram_n_heads=2, engram_head_dim=32,
    engram_pad_id=2,
)
N_PREFILL, N_DECODE = 37, 6


class FakeBackend:
    """Stands in for the Rust tokenizer in engram.build_compressed_token_map: enough variety to hit
    every branch (case/space/accent folding, the single-space sentinel, partial UTF-8 bytes)."""

    def __init__(self, n):
        fixed = ["<bos>", "<eos>", "<pad>", " ", "\n", "\ufffd", "", "\u00e9"]
        self.text = fixed + [(f"w{(i - 8) // 3}", f" W{(i - 8) // 3}", f"w{(i - 8) // 3}\u0301 \n")[(i - 8) % 3] for i in range(8, n)]

    def decode(self, ids, skip_special_tokens=False):
        return self.text[ids[0]]

    def id_to_token(self, i):
        return f"<0x{i:02X}>"


class FakeTokenizer:
    def __init__(self, n):
        self.backend_tokenizer = FakeBackend(n)
        self.n = n

    def __len__(self):
        return self.n


def make_args():
    tok = FakeTokenizer(TINY["vocab_size"])
    _, cvocab = R_engram.build_compressed_token_map(tok)
    args = R.ModelArgs(**TINY, engram_compressed_vocab_size=cvocab, engram_num_embeddings=(0, 0))
    layout = R_engram.EngramLayout.from_args(args)
    # table rows = sum of the bucket primes of that layer, as in the released config
    args.engram_num_embeddings = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    return args, tok


def init_params(net, seed):
    """Deterministic values for every parameter; quantised storage is produced from float values."""
    g = torch.Generator().manual_seed(seed)
    done = set()

    def rn(*shape, std=1.0):
        return torch.randn(*shape, generator=g) * std

    for mod in net.modules():
        if isinstance(mod, R.Linear):
            out_f, in_f = mod.weight.shape[0], mod.in_features
            w = rn(out_f, in_f, std=1.0 / math.sqrt(in_f))
            if mod.weight.dtype == torch.float4_e2m1fn_x2:
                s = kernel_cpu.pow2_ceil(w.unflatten(-1, (-1, 32)).abs().amax(-1) / 6.0)
                codes = kernel_cpu.fp4_codes(w / s.repeat_interleave(32, 1))
                mod.weight.data = kernel_cpu.fp4_pack(codes)
                mod.scale.data = s.to(torch.float8_e8m0fnu)
                done.add(id(mod.scale))
            elif mod.weight.dtype == torch.float8_e4m3fn:
                rows, cols = mod.scale.shape
                pad = torch.zeros(rows * 32, cols * 32)
                pad[:out_f, :in_f] = w
                s = kernel_cpu.pow2_ceil(pad.view(rows, 32, cols, 32).abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0)
                full = s.repeat_interleave(32, 0)[:out_f].repeat_interleave(32, 1)[:, :in_f]
                mod.weight.data = kernel_cpu.fp8_round(w / full).to(torch.float8_e4m3fn)
                mod.scale.data = s.to(torch.float8_e8m0fnu)
                done.add(id(mod.scale))
            else:
                mod.weight.data = w                                  # bf16 / fp32 in the checkpoint -> float32 here
            done.add(id(mod.weight))
        elif isinstance(mod, R.ParallelEngramEmbedding):
            rows, dim = mod.weight.shape
            w = rn(rows, dim)
            s = kernel_cpu.pow2_ceil(w.unflatten(-1, (-1, 32)).abs().amax(-1) / 448.0)
            mod.weight.data = kernel_cpu.fp8_round(w / s.repeat_interleave(32, 1)).to(torch.float8_e4m3fn)
            mod.scale.data = s.to(torch.float8_e8m0fnu)
            done.update((id(mod.weight), id(mod.scale)))
        elif isinstance(mod, R.RMSNorm):
            mod.weight.data = 1.0 + rn(mod.dim, std=0.2)
            done.add(id(mod.weight))

    for name, p in net.named_parameters():
        if id(p) in done:
            continue
        leaf = name.split(".")[-1]
        if name == "embed.weight":
            v = rn(*p.shape)
        elif name == "head.weight":
            v = rn(*p.shape, std=1.0 / math.sqrt(p.shape[1]))
        elif leaf == "attn_sink":
            v = rn(*p.shape, std=0.5)
        elif name.endswith("ffn.gate.weight"):
            v = rn(*p.shape, std=3.0 / math.sqrt(p.shape[1]))
        elif name.endswith("ffn.gate.bias"):
            v = rn(*p.shape, std=0.3)
        elif leaf in ("hc_attn_fn", "hc_ffn_fn"):
            v = rn(*p.shape, std=1.0 / math.sqrt(p.shape[1]))
        elif leaf in ("hc_attn_base", "hc_ffn_base"):
            v = rn(*p.shape, std=0.5)
        elif leaf in ("hc_attn_scale", "hc_ffn_scale"):
            v = torch.tensor([1.0, 0.8, 1.2]) + rn(3, std=0.1)
        elif leaf in ("q_weight", "k_weight"):
            v = 1.0 + rn(*p.shape, std=0.2)
        else:
            raise RuntimeError(f"no initialiser for parameter {name} {tuple(p.shape)} {p.dtype}")
        p.data = v.to(torch.float32)
        done.add(id(p))


def export_weights(net):
    """Dequantised float32 weights keyed by parameter name (scales folded in)."""
    out = {}
    for name, mod in net.named_modules():
        if isinstance(mod, R.Linear):
            w = mod.weight
            if w.dtype == torch.float4_e2m1fn_x2:
                v = kernel_cpu.fp4_unpack(w.data) * mod.scale.data.float().repeat_interleave(32, 1)
            elif w.dtype == torch.float8_e4m3fn:
                o, i = w.shape
                v = w.data.float() * mod.scale.data.float().repeat_interleave(32, 0)[:o].repeat_interleave(32, 1)[:, :i]
            else:
                v = w.data.float()
            out[name + ".weight"] = v.numpy()
        elif isinstance(mod, R.ParallelEngramEmbedding):
            out[name + ".weight"] = (mod.weight.data.float() * mod.scale.data.float().repeat_interleave(32, 1)).numpy()
    for name, p in net.named_parameters():
        if name.endswith(".scale") or name in out:
            continue
        out[name] = p.data.float().numpy()
    return out


class Recorder:
    """Records kernel calls and module inputs/outputs under names 'L<layer>.<what>'."""

    def __init__(self):
        self.rec, self.layer, self.on = {}, None, True

    def put(self, name, value):
        if self.on:
            key = name if self.layer is None else f"L{self.layer}.{name}"
            assert key not in self.rec, key
            self.rec[key] = value.detach().clone().float().numpy() if value.is_floating_point() else value.detach().clone().numpy()

    def install(self, net):
        rec = self
        real_sparse, real_hc = R.sparse_attn, R.hc_split_sinkhorn

        def sparse_attn(q, kv, sink, idxs, scale):
            o = real_sparse(q, kv, sink, idxs, scale)
            for n, v in (("attn.q", q), ("attn.kv", kv), ("attn.topk_idxs", idxs), ("attn.o", o)):
                rec.put(n, v)
            return o

        def hc_split_sinkhorn(mixes, *a, **k):
            pre, post, comb = real_hc(mixes, *a, **k)
            which = rec.hc_which.pop(0)
            for n, v in (("mixes", mixes), ("pre", pre), ("post", post), ("comb", comb)):
                rec.put(f"hc_{which}.{n}", v)
            return pre, post, comb

        R.sparse_attn, R.hc_split_sinkhorn = sparse_attn, hc_split_sinkhorn

        def pre_block(mod, a):
            rec.layer, rec.hc_which = mod.layer_id, ["attn", "ffn"]
            rec.put("in", a[0])
            rec.put("pre_mix", a[2])

        def post_block(mod, a, out):
            rec.put("out", out[0])
            rec.put("ffn_pre", out[1])
            rec.layer = None

        for blk in net.layers:
            blk.register_forward_pre_hook(pre_block)
            blk.register_forward_hook(post_block)
            blk.attn.register_forward_hook(lambda m, a, o: (rec.put("attn.x", a[0]), rec.put("attn.out", o)) and None)
            blk.attn.wq_a.register_forward_hook(lambda m, a, o: rec.put("attn.wq_a", o))
            blk.attn.q_norm.register_forward_hook(lambda m, a, o: rec.put("attn.qr", o))
            blk.ffn.register_forward_hook(lambda m, a, o: (rec.put("ffn.x", a[0]), rec.put("ffn.out", o)) and None)
            blk.ffn.gate.register_forward_hook(lambda m, a, o: (rec.put("ffn.weights", o[0]), rec.put("ffn.indices", o[1])) and None)
            blk.ffn.shared_experts.register_forward_hook(lambda m, a, o: rec.put("ffn.shared", o))
            if blk.attn.compressor is not None:
                # clone: Attention rotates and quantises the returned latent in place afterwards
                def post_compressor(m, a, o):
                    if o is not None:
                        rec.put("attn.latent", o)
                blk.attn.compressor.register_forward_hook(post_compressor)
            if blk.attn.indexer is not None:
                def post_indexer(m, a, o):
                    rec.put("attn.index_idxs", o)
                    if m.is_candidate_source:
                        rec.put("attn.candidates", R.shared_attn.candidates)
                blk.attn.indexer.register_forward_hook(post_indexer)
            if blk.engram is not None:
                def engram_hooks(layer_id):
                    def pre(m, a):
                        rec.layer = layer_id
                        rec.put("engram.in", a[0])
                        rec.put("engram.hash_ids", a[1])

                    def post(m, a, o):
                        rec.put("engram.out", o)
                        rec.layer = None
                    return pre, post
                pre, post = engram_hooks(blk.layer_id)
                blk.engram.register_forward_pre_hook(pre)
                blk.engram.register_forward_hook(post)
                blk.engram.embed.register_forward_hook(lambda m, a, o: rec.put("engram.rows", o))
                blk.engram.wkv.register_forward_hook(lambda m, a, o: rec.put("engram.kv", o))


def run(mode, seed):
    kernel_cpu.MODE = mode
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    args, tok = make_args()
    net = R.Transformer(args, tok)
    init_params(net, seed)
    net.eval()
    tokens = torch.randint(3, args.vocab_size, (1, N_PREFILL + N_DECODE), generator=torch.Generator().manual_seed(seed + 1))
    # a repeated 4-gram, so the same Engram rows are hit twice at different positions
    tokens[0, 20:24] = tokens[0, 5:9]

    rec = Recorder()
    rec.install(net)

    # (1) the long prefill: this is what gets recorded
    rec.on = True
    _, logits_full, _ = net(tokens, 0)
    full = {k: v for k, v in rec.rec.items()}
    with torch.inference_mode():
        hidden = net.layers[-1].hc_pre(torch.from_numpy(full[f"L{args.n_layers - 1}.out"]),
                                       torch.from_numpy(full[f"L{args.n_layers - 1}.ffn_pre"]))
        logits_all = net.head(net.norm(hidden), full_logits=True)
    # all-position logits go through a batched matmul, so equal only up to fp32 summation order
    assert torch.allclose(logits_all[:, -1], logits_full, rtol=1e-5, atol=1e-6), (logits_all[:, -1] - logits_full).abs().max()

    # (2) the reference's own consistency: prefill of N tokens, then N_DECODE single-token steps
    rec.on = False
    _, logits, _ = net(tokens[:, :N_PREFILL], 0)
    steps = [logits]
    for t in range(N_PREFILL, N_PREFILL + N_DECODE):
        _, logits, _ = net(tokens[:, t:t + 1], t)
        steps.append(logits)
    steps = torch.cat(steps).numpy()
    want = logits_all[0, N_PREFILL - 1:].numpy()
    decode_err = float(np.abs(steps - want).max() / np.abs(want).max())

    out = {"w." + k: v for k, v in export_weights(net).items()}
    out.update({"r." + k: v for k, v in full.items()})
    out["r.logits"] = logits_all[0].numpy()
    out["r.logits_decode"] = steps
    out["tokens"] = tokens[0].numpy()
    eh = net.engram_hash
    out["engram.token_map"] = eh.token_map.numpy()
    out["engram.primes"] = eh.primes.numpy()
    out["engram.offsets"] = eh.offsets.numpy()
    out["engram.multipliers"] = eh.multipliers.numpy()
    out["engram.pad_id"] = np.int64(eh.pad_id)
    cfg = {k: getattr(args, k) for k in args.__dataclass_fields__}
    meta = {"mode": mode, "seed": seed, "n_prefill": N_PREFILL, "n_decode": N_DECODE, "torch": torch.__version__,
            "decode_vs_prefill_max_rel": decode_err, "pin": json.load(open(os.path.join(REF, "..", "PIN.json")))["revision"]}
    out["config_json"] = np.array(json.dumps(cfg))
    out["meta_json"] = np.array(json.dumps(meta))
    return out, meta, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "exact"], default="reference")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, meta, _ = run(a.mode, a.seed)
    np.savez_compressed(a.out, **out)
    print(json.dumps(meta))
    print(f"{len(out)} arrays, {os.path.getsize(a.out)} bytes -> {a.out}")


if __name__ == "__main__":
    main()
