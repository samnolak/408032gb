#!/usr/bin/env python3
"""roofline: physical upper bounds for batch-1 decode (and a rough TTFT) of a model on a given server.

What it adds over tools/fitplan.py (which stays as the historical tool):
  * per-token reads of the KV cache and recurrent state at context L, per attention kind
    (full, sliding window, indexed sparse such as DSA/CSA, linear/recurrent);
  * three ways to serve an expert missing from VRAM: computed by the CPU from RAM, copied over PCIe,
    read from NVMe; plus the per-layer synchronisation cost of handing work to the CPU;
  * three multi-GPU decode strategies: layer pipeline (PP), expert parallel inside each MoE layer (EP),
    tensor parallel (TP); with the expected load imbalance of top-k routing over n GPUs;
  * speculative decoding as a function of acceptance, with the extra distinct experts a k-token
    verification touches (uniform-routing model, pessimistic for real, correlated routing).

Everything it prints is PROVISIONAL: a memory-bandwidth model with stated efficiencies, no kernel launch
overheads, no compute limits on GPU decode. It answers "can this configuration possibly reach X tok/s",
not "how fast will it run". Inputs carry their own labels (models/specs/*.json, hardware/*.json).

Usage:
  python3 tools/roofline.py models/specs/glm-5.3-flash.json --ctx 262144 --expert-bpw 4.5
  python3 tools/roofline.py models/specs/glm-5.3-flash.json --ctx 262144 --sweep-hit 0.6,0.8,0.95,1.0
  python3 tools/roofline.py models/specs/*.json --ctx 262144 --json
"""
import argparse
import itertools
import json
import math
import sys

GIB = 1024 ** 3
DEFAULT_HW = "hardware/target-4x4080-32g.unverified.json"


# ---------------------------------------------------------------------------------------------- model

def expert_params(spec):
    m = spec["moe"]
    return m["expert_mats"] * spec["hidden"] * m["expert_ff"]


def routed_active_params(spec):
    m = spec["moe"]
    return m["top_k"] * m["layers"] * expert_params(spec)


def routed_stored_params(spec):
    m = spec["moe"]
    return (m["layers"] + m.get("mtp_layers", 0)) * m["n_experts"] * expert_params(spec) + m.get("extra_stored_params", 0)


def dense_active_params(spec):
    """Non-routed parameters read for every decoded token (embedding row excluded)."""
    return sum(v for k, v in spec["dense_params"].items() if k not in ("embedding", "mtp", "vision"))


def dense_stored_params(spec, include_mtp=True):
    d = spec["dense_params"]
    s = dense_active_params(spec) + d.get("embedding", 0)
    if include_mtp:
        s += d.get("mtp", 0)
    return s


def kv_store_bytes(spec, ctx, kv_bytes_per_elem=2.0):
    """Bytes held for a context of `ctx` tokens: KV caches + recurrent states (one sequence)."""
    override = spec.get("kv_store_bytes_per_token")         # architectures with KV shared across layers
    total = override * ctx if override is not None else 0.0
    for g in spec["attention"]:
        kind, n = g["kind"], g["layers"]
        if kind == "linear":
            total += n * g["state_bytes"]
            continue
        per_tok = g["kv_elems_per_token"] * g.get("kv_bytes_per_elem", kv_bytes_per_elem) + g.get("aux_bytes_per_token", 0)
        if kind == "sliding":
            total += n * per_tok * min(ctx, g["window"])
        elif override is None:  # full, indexed
            total += n * per_tok * ctx / g.get("compress_ratio", 1)
    return total


def kv_read_bytes(spec, ctx, kv_bytes_per_elem=2.0):
    """Bytes of KV / state touched to decode one token with `ctx` tokens of context."""
    total = 0.0
    for g in spec["attention"]:
        kind, n = g["kind"], g["layers"]
        if kind == "linear":
            total += n * 2 * g["state_bytes"]                      # read and write the state
        elif kind == "sliding":
            total += n * g["kv_elems_per_token"] * g.get("kv_bytes_per_elem", kv_bytes_per_elem) * min(ctx, g["window"])
        elif kind == "full":
            total += n * (g["kv_elems_per_token"] * g.get("kv_bytes_per_elem", kv_bytes_per_elem)
                          + g.get("aux_bytes_per_token", 0)) * ctx / g.get("compress_ratio", 1)
        elif kind == "indexed":
            entries = ctx / g.get("compress_ratio", 1)
            scanned = min(entries / g.get("pool", 1), g.get("scan_cap_entries", float("inf")))
            scan = g["scan_bytes_per_entry"] * scanned
            sel = min(entries, g["selected_entries"]) * g["kv_elems_per_token"] * g.get("kv_bytes_per_elem", kv_bytes_per_elem)
            total += n * (scan + sel)
        else:
            raise ValueError(f"unknown attention kind {kind!r}")
    return total


def lookup_bytes_per_token(spec):
    lk = spec.get("lookup")
    if not lk:
        return 0.0, 0
    return lk["rows_per_token"] * lk["row_bytes"], lk["rows_per_token"]


# ---------------------------------------------------------------------------------------------- helpers

def ep_imbalance(top_k, n_gpu):
    """E[max experts on one GPU] / (top_k / n_gpu) when each of top_k experts lands on a uniformly random GPU."""
    if n_gpu <= 1:
        return 1.0
    total = 0
    for combo in itertools.product(range(n_gpu), repeat=top_k):
        counts = [0] * n_gpu
        for c in combo:
            counts[c] += 1
        total += max(counts)
    exp_max = total / n_gpu ** top_k
    return exp_max / (top_k / n_gpu)


def distinct_experts(n_experts, top_k, n_tokens):
    """Expected distinct experts per layer touched by n_tokens tokens, independent uniform routing."""
    return n_experts * (1 - (1 - top_k / n_experts) ** n_tokens)


def spec_tokens_per_step(accept, k):
    """Expected tokens emitted per verification step with k drafted tokens and per-token acceptance `accept`."""
    if accept >= 1.0:
        return k + 1
    return (1 - accept ** (k + 1)) / (1 - accept)


# ---------------------------------------------------------------------------------------------- placement

def placement(spec, hw, q, ctx):
    n_gpu = len(hw["gpus"])
    e_bytes = expert_params(spec) * q["expert_bpw"] / 8
    experts_bytes = routed_stored_params(spec) * q["expert_bpw"] / 8
    dense_bytes = dense_stored_params(spec) * q["dense_bpw"] / 8
    kv_bytes = kv_store_bytes(spec, ctx, q["kv_bytes_per_elem"])
    vram = sum(g["vram_gib"] for g in hw["gpus"]) * GIB
    fixed = n_gpu * hw["gpu_reserve_gib"] * GIB + dense_bytes + kv_bytes + q.get("extra_vram_gib", 0) * GIB
    cache = max(0.0, vram - fixed)
    resident = min(experts_bytes, cache)
    spill = experts_bytes - resident
    ram_usable = (hw["ram"]["gib"] - hw["ram"]["reserve_gib"]) * GIB
    ram_experts = min(spill, ram_usable)
    ssd_experts = spill - ram_experts
    return {
        "expert_mib": e_bytes / 2 ** 20,
        "experts_gib": experts_bytes / GIB, "dense_gib": dense_bytes / GIB, "kv_state_gib": kv_bytes / GIB,
        "vram_fixed_gib": fixed / GIB, "vram_expert_cache_gib": cache / GIB,
        "resident_frac": resident / experts_bytes if experts_bytes else 1.0,
        "ram_frac": ram_experts / experts_bytes if experts_bytes else 0.0,
        "ssd_frac": ssd_experts / experts_bytes if experts_bytes else 0.0,
        "fits_vram_only": spill <= 0, "dense_kv_fit_vram": fixed <= vram,
        "spill_gib": spill / GIB, "ram_usable_gib": ram_usable / GIB, "ssd_gib": ssd_experts / GIB,
    }


# ---------------------------------------------------------------------------------------------- decode

def decode(spec, hw, q, ctx, hit_vram, hit_ram=None, strategy="pp", miss="cpu", spec_k=0, spec_accept=0.0,
           ram_channels=None):
    """Seconds per token and tok/s bound. hit_vram: share of routed-expert reads served from VRAM.

    hit_ram: share served from RAM (default: all the rest); the remainder comes from NVMe.
    miss: "cpu" (CPU computes from RAM) or "pcie" (copy the expert to the GPU first).
    """
    n_gpu = len(hw["gpus"])
    if hit_ram is None:
        hit_ram = 1.0 - hit_vram
    hit_ssd = max(0.0, 1.0 - hit_vram - hit_ram)
    m = spec["moe"]
    L_moe = m["layers"]
    n_layers = spec["n_layers"]

    gbw = hw["gpus"][0]["hbm_bw_gbs"] * 1e9 * hw["gpus"][0]["bw_eff"]
    ch = str(ram_channels or hw["ram"]["channels"] or hw["ram"]["assume_channels"])
    rbw = hw["ram"]["bw_gbs_by_channels"][ch] * 1e9 * hw["ram"]["bw_eff"]
    pbw = hw["pcie"]["h2d_gbs"] * 1e9
    lat = hw["pcie"]["latency_us"] * 1e-6
    ssd_bw = sum(d["seq_read_gbs"] for d in hw["nvme"]) * 1e9 * hw["nvme_eff"]

    n_tok = 1 + spec_k
    d_routed = distinct_experts(m["n_experts"], m["top_k"], n_tok) if n_tok > 1 else m["top_k"]
    w_dense = dense_active_params(spec) * q["dense_bpw"] / 8
    w_routed = d_routed * L_moe * expert_params(spec) * q["expert_bpw"] / 8
    w_kv = kv_read_bytes(spec, ctx, q["kv_bytes_per_elem"])
    lk_bytes, lk_rows = lookup_bytes_per_token(spec)

    gpu_routed = hit_vram * w_routed
    ram_routed = hit_ram * w_routed
    ssd_routed = hit_ssd * w_routed

    # per MoE layer (uniform over layers)
    a_total = (w_dense + w_kv) / gbw                         # attention, dense FFN, shared expert, head
    e_gpu_l = gpu_routed / L_moe / gbw
    if miss == "cpu":
        e_off_l = ram_routed / L_moe / rbw
    elif miss == "pcie":
        e_off_l = ram_routed / L_moe / pbw + (ram_routed / L_moe) / gbw
    else:
        raise ValueError(miss)
    e_off_l += ssd_routed / L_moe / ssd_bw + (ssd_routed / L_moe / rbw if miss == "cpu" else 0.0)
    off_sync = (2 * lat) if (hit_ram + hit_ssd) > 0 else 0.0

    if strategy == "pp":
        moe = L_moe * max(e_gpu_l, e_off_l + off_sync)
        comm = (n_gpu - 1) * lat
        t = a_total + moe + comm
    elif strategy == "ep":
        imb = ep_imbalance(m["top_k"], n_gpu) if n_tok == 1 else 1.0
        e_gpu_ep = e_gpu_l / n_gpu * imb
        moe = L_moe * (max(e_gpu_ep, e_off_l + off_sync) + 2 * lat)
        comm = (n_gpu - 1) * lat
        t = a_total + moe + comm
    elif strategy == "tp":
        moe = L_moe * max(e_gpu_l / n_gpu, e_off_l + off_sync)
        comm = n_layers * 2 * hw["pcie"]["allreduce_us"] * 1e-6
        t = a_total / n_gpu + moe + comm
    else:
        raise ValueError(strategy)

    # table lookups (e.g. Engram): addresses depend only on input token ids, so they can be prefetched one
    # token ahead; the bound counts their bandwidth, not their latency.
    t_lookup = lk_bytes / ssd_bw if spec.get("lookup", {}).get("tier") == "ssd" else 0.0
    t = max(t, t_lookup)

    tokens = spec_tokens_per_step(spec_accept, spec_k) if spec_k else 1.0
    return {
        "strategy": strategy, "miss": miss, "ctx": ctx, "hit_vram": hit_vram, "hit_ram": hit_ram, "hit_ssd": hit_ssd,
        "bytes_per_step_gib": {"dense": w_dense / GIB, "routed": w_routed / GIB, "kv_state": w_kv / GIB,
                               "lookup": lk_bytes / GIB},
        "ms_per_step": t * 1e3, "tokens_per_step": tokens, "tok_s": tokens / t,
        "lookup_rows_per_token": lk_rows, "ram_channels": int(ch),
    }


def ttft(spec, hw, q, ctx, resident_frac, chunk=4096):
    """Very rough prefill bound: GEMM compute on all GPUs + streaming the non-resident experts once per chunk."""
    n_gpu = len(hw["gpus"])
    flops = 2 * (dense_active_params(spec) + routed_active_params(spec)) * ctx
    attn = 0.0
    for g in spec["attention"]:
        attn += g["layers"] * g.get("prefill_flops_per_token", 0) * ctx      # linear-in-context part
        f = g.get("prefill_flops_coef")                                       # per (query, key) pair
        if f is None:
            continue
        if g["kind"] in ("full", "indexed"):
            attn += g["layers"] * f * ctx * ctx / 2
        elif g["kind"] == "sliding":
            attn += g["layers"] * f * ctx * min(ctx, g["window"])
    tf = hw["gpus"][0]["tflops_fp16"] * 1e12 * hw["gpus"][0]["mma_eff"] * n_gpu
    t_compute = (flops + attn) / tf
    experts_bytes = routed_stored_params(spec) * q["expert_bpw"] / 8
    stream = (1 - resident_frac) * experts_bytes
    t_stream = math.ceil(ctx / chunk) * stream / (hw["pcie"]["h2d_gbs"] * 1e9 * n_gpu)
    return {"ctx": ctx, "chunk": chunk, "t_compute_s": t_compute, "t_stream_s": t_stream,
            "ttft_s_bound": max(t_compute, t_stream), "attn_flops_share": attn / (flops + attn) if flops + attn else 0}


# ---------------------------------------------------------------------------------------------- CLI

def quant_from_args(a):
    return {"expert_bpw": a.expert_bpw, "dense_bpw": a.dense_bpw, "kv_bytes_per_elem": a.kv_bytes}


def report(spec, hw, q, ctx, hits, strategies, misses, spec_k, spec_accept, ram_channels):
    p = placement(spec, hw, q, ctx)
    rows = []
    off = p["ram_frac"] + p["ssd_frac"]
    for h in hits:
        if h == "auto":                       # uniform routing: accesses follow bytes
            hv, hr = p["resident_frac"], p["ram_frac"]
        else:                                 # a better cache: misses split between RAM and NVMe like the bytes
            hv = float(h)
            hr = (1 - hv) * (p["ram_frac"] / off if off > 0 else 1.0)
        for s in strategies:
            for mi in misses:
                rows.append(decode(spec, hw, q, ctx, hv, hr, s, mi, spec_k, spec_accept, ram_channels))
    return {"model": spec["name"], "quant": q, "placement": p, "decode": rows,
            "ttft": ttft(spec, hw, q, ctx, p["resident_frac"])}


def render(r):
    p = r["placement"]
    out = [f"# {r['model']}  experts {r['quant']['expert_bpw']} bpw, dense {r['quant']['dense_bpw']} bpw, "
           f"KV {r['quant']['kv_bytes_per_elem']} B/elem  (PROVISIONAL model, see tools/roofline.py)",
           f"  experts {p['experts_gib']:.1f} GiB ({p['expert_mib']:.2f} MiB each) | dense {p['dense_gib']:.1f} GiB | "
           f"KV+state @ctx {p['kv_state_gib']:.2f} GiB",
           f"  VRAM fixed {p['vram_fixed_gib']:.1f} GiB, expert cache {p['vram_expert_cache_gib']:.1f} GiB -> resident "
           f"{100 * p['resident_frac']:.1f}% | RAM {100 * p['ram_frac']:.1f}% | NVMe {100 * p['ssd_frac']:.1f}% "
           f"({p['ssd_gib']:.1f} GiB)"]
    for d in r["decode"]:
        b = d["bytes_per_step_gib"]
        out.append(f"  ctx {d['ctx']:>7} hit VRAM {d['hit_vram']:.2f} RAM {d['hit_ram']:.2f} SSD {d['hit_ssd']:.2f} "
                   f"{d['strategy']:>2}/{d['miss']:<4} RAM {d['ram_channels']}ch: {d['tok_s']:6.1f} tok/s "
                   f"({d['ms_per_step']:.2f} ms/step; read GiB dense {b['dense']:.2f} routed {b['routed']:.2f} "
                   f"kv {b['kv_state']:.3f})")
    t = r["ttft"]
    out.append(f"  TTFT bound @ctx {t['ctx']}: {t['ttft_s_bound']:.0f} s (compute {t['t_compute_s']:.0f} s, "
               f"expert streaming {t['t_stream_s']:.0f} s, attention share of FLOPs {100 * t['attn_flops_share']:.0f}%)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--hw", default=DEFAULT_HW)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--expert-bpw", type=float, default=4.5, help="Q4_K 4.5, IQ4_XS 4.25, MXFP4 4.25, NVFP4 4.5")
    ap.add_argument("--dense-bpw", type=float, default=8.5, help="Q8_0 8.5, Q6_K 6.5625, Q4_K 4.5")
    ap.add_argument("--kv-bytes", type=float, default=2.0, help="bytes per KV element: f16 2, fp8/q8_0 ~1.06")
    ap.add_argument("--sweep-hit", default="auto", help="comma list of VRAM hit shares, or 'auto' (uniform routing)")
    ap.add_argument("--strategies", default="pp,ep,tp")
    ap.add_argument("--miss", default="cpu", help="cpu,pcie")
    ap.add_argument("--spec-k", type=int, default=0)
    ap.add_argument("--spec-accept", type=float, default=0.0)
    ap.add_argument("--ram-channels", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    with open(a.hw) as f:
        hw = json.load(f)
    q = quant_from_args(a)
    hits = a.sweep_hit.split(",")
    res = []
    for path in a.specs:
        with open(path) as f:
            spec = json.load(f)
        res.append(report(spec, hw, q, a.ctx, hits, a.strategies.split(","), a.miss.split(","), a.spec_k,
                          a.spec_accept, a.ram_channels))
    if a.json:
        json.dump(res, sys.stdout, indent=1)
        print()
    else:
        print("\n\n".join(render(r) for r in res))


if __name__ == "__main__":
    main()
