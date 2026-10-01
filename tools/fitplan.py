#!/usr/bin/env python3
"""fitplan: where a MoE model's bytes go on the 4x4080S server, and what that implies.

Everything this prints is PROVISIONAL: uniform bits per weight for all routed experts, no
per-tensor overrides, dense weights at --dense-quant, an even layer split across GPUs, and
decode ceilings that assume ideal memory bandwidth with no compute, launch or PCIe cost.
The model and hardware facts it reads carry their own labels in the JSON files.

Usage:
  python3 tools/fitplan.py models/glm-5.3-flash.json --quant IQ3_XXS --ctx 131072
  python3 tools/fitplan.py models/glm-5.3-flash.json --quant Q4_K --hit-curve curve.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_sizes import bpw  # noqa: E402

GIB = 1024 ** 3


def expert_params(m):
    moe = m["moe"]
    return moe["expert_mats"] * m["hidden_size"] * moe["expert_ff"]


def routed_expert_params_total(m, include_mtp=True):
    moe = m["moe"]
    layers = moe["moe_layers"] + (moe["mtp_moe_layers"] if include_mtp else 0)
    return layers * moe["n_routed"] * expert_params(m) + moe["extra_expert_params_b"] * 1e9


def dense_params(m):
    """Non-routed text parameters (shared experts included).

    Uses params_b.dense when the model card gives it; otherwise total - routed - vision.
    """
    p = m["params_b"]
    if p.get("dense") is not None:
        return p["dense"] * 1e9
    return max(0.0, p["total"] * 1e9 - routed_expert_params_total(m) - p.get("vision", 0.0) * 1e9)


def active_dense_params(m):
    """Dense bytes read per decoded token: active minus routed-active, capped by dense."""
    p = m["params_b"]
    dense = dense_params(m) - p.get("embedding", 0.0) * 1e9
    if p.get("active") is None:
        return dense
    moe = m["moe"]
    routed_active = moe["top_k"] * moe["moe_layers"] * expert_params(m)
    return max(0.0, min(dense, p["active"] * 1e9 - routed_active))


def interp_hit(curve, frac):
    """Hit rate for a cache holding `frac` of each layer's experts, from a G0 curve."""
    pts = sorted((float(a), float(b)) for a, b in curve)
    if frac <= pts[0][0]:
        return pts[0][1] * (frac / pts[0][0]) if pts[0][0] > 0 else pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= frac <= x1:
            return y0 + (y1 - y0) * (frac - x0) / (x1 - x0) if x1 > x0 else y1
    return pts[-1][1]


def plan(model, hw, quant, ctx, dense_quant="Q8_0", state_copies=4, hit_curve=None,
         ram_channels=None):
    e_bpw = bpw(quant)
    d_bpw = bpw(dense_quant)
    n_gpu = len(hw["gpus"])

    e_params = expert_params(model)
    e_bytes = e_params * e_bpw / 8
    experts_total_params = routed_expert_params_total(model)
    experts_bytes = experts_total_params * e_bpw / 8
    n_experts = experts_total_params / e_params

    dense_bytes = dense_params(model) * d_bpw / 8
    kv = model["kv"]
    kv_bytes = kv["bytes_per_token"] * ctx + kv["fixed_bytes"]
    state_bytes = kv["recurrent_state_bytes"] * state_copies

    per_gpu = []
    cache_bytes = 0.0
    for g in hw["gpus"]:
        vram = g["vram_gib"] * GIB
        fixed = hw["gpu_reserve_gib"] * GIB + (dense_bytes + kv_bytes + state_bytes) / n_gpu
        free = max(0.0, vram - fixed)
        per_gpu.append({"vram_gib": vram / GIB, "fixed_gib": fixed / GIB, "expert_cache_gib": free / GIB})
        cache_bytes += free

    resident_bytes = min(experts_bytes, cache_bytes)
    resident_frac = resident_bytes / experts_bytes if experts_bytes else 1.0
    spill_bytes = experts_bytes - resident_bytes

    ram_usable = (hw["ram_gib"] - hw["ram_reserve_gib"]) * GIB
    strata_mode_fits = experts_bytes <= ram_usable          # Strata keeps ALL experts in RAM
    disjoint_fits = spill_bytes <= ram_usable                # RAM holds only non-resident experts
    nvme_spill = max(0.0, spill_bytes - ram_usable)

    engram = model.get("engram")
    engram_bytes = engram["params_b"] * 1e9 * engram["native_bpw"] / 8 if engram else 0.0

    if hit_curve is not None:
        hit = interp_hit(hit_curve, resident_frac)
        hit_src = "G0 curve"
    else:
        hit = resident_frac
        hit_src = "uniform routing (worst case for a frequency cache)"

    moe = model["moe"]
    routed_active_bytes = moe["top_k"] * moe["moe_layers"] * e_bytes
    # shared experts are part of the dense parameters (read every token, stored at dense_quant)
    gpu_tok = active_dense_params(model) * d_bpw / 8 + hit * routed_active_bytes
    cpu_tok = (1 - hit) * routed_active_bytes

    gpu_bw = hw["gpus"][0]["mem_bw_gbs"] * 1e9
    ram_bw = hw["ram_bw_gbs_by_channels"]
    ceilings = {}
    chans = [str(ram_channels)] if ram_channels else sorted(ram_bw)
    for ch in chans:
        cpu_t = cpu_tok / (ram_bw[ch] * 1e9)
        pp_t = max(gpu_tok / gpu_bw, cpu_t)                  # pipeline: one GPU busy at a time
        tp_t = max(gpu_tok / (gpu_bw * n_gpu), cpu_t)        # ideal 4-way TP/EP, zero comm cost
        ceilings[ch] = {"pipeline_tok_s": 1 / pp_t, "tp_ideal_tok_s": 1 / tp_t}

    return {
        "model": model["name"], "quant": quant, "expert_bpw": e_bpw, "dense_quant": dense_quant, "ctx": ctx,
        "expert_params_m": e_params / 1e6, "expert_mib": e_bytes / 2 ** 20, "n_experts": n_experts,
        "experts_gib": experts_bytes / GIB, "dense_gib": dense_bytes / GIB,
        "kv_gib": kv_bytes / GIB, "state_gib": state_bytes / GIB,
        "per_gpu": per_gpu, "expert_cache_gib": cache_bytes / GIB,
        "resident_frac": resident_frac, "resident_experts": resident_frac * n_experts,
        "spill_gib": spill_bytes / GIB, "ram_usable_gib": ram_usable / GIB,
        "strata_mode_fits": strata_mode_fits, "disjoint_fits": disjoint_fits, "nvme_spill_gib": nvme_spill / GIB,
        "engram_gib": engram_bytes / GIB,
        "hit_rate": hit, "hit_source": hit_src,
        "decode_gpu_mib_per_tok": gpu_tok / 2 ** 20, "decode_cpu_mib_per_tok": cpu_tok / 2 ** 20,
        "decode_ceilings_by_ram_channels": ceilings,
    }


def render(p):
    out = []
    a = out.append
    a(f"# fitplan: {p['model']}  experts={p['quant']} ({p['expert_bpw']:.4f} bpw)  dense={p['dense_quant']}  ctx={p['ctx']}")
    a("All figures PROVISIONAL (see tools/fitplan.py docstring).")
    a(f"experts: {p['n_experts']:.0f} x {p['expert_params_m']:.2f}M params = {p['experts_gib']:.1f} GiB ({p['expert_mib']:.2f} MiB each)")
    a(f"dense {p['dense_gib']:.1f} GiB | KV {p['kv_gib']:.2f} GiB | recurrent state {p['state_gib']:.2f} GiB")
    for i, g in enumerate(p["per_gpu"]):
        a(f"  GPU{i}: {g['vram_gib']:.1f} GiB = fixed {g['fixed_gib']:.2f} + expert cache {g['expert_cache_gib']:.2f}")
    a(f"VRAM-resident experts: {100 * p['resident_frac']:.1f}% ({p['resident_experts']:.0f})  | spill to RAM: {p['spill_gib']:.1f} GiB of {p['ram_usable_gib']:.0f} usable")
    a(f"Strata mode (all experts in RAM): {'fits' if p['strata_mode_fits'] else 'DOES NOT FIT'}"
      f" | disjoint VRAM/RAM: {'fits' if p['disjoint_fits'] else 'DOES NOT FIT'}"
      + (f" | NVMe spill {p['nvme_spill_gib']:.1f} GiB" if p['nvme_spill_gib'] > 0 else ""))
    if p["engram_gib"]:
        a(f"n-gram/Engram tables on NVMe: {p['engram_gib']:.1f} GiB")
    a(f"decode traffic per token: GPU {p['decode_gpu_mib_per_tok']:.0f} MiB, CPU {p['decode_cpu_mib_per_tok']:.0f} MiB  (hit {100 * p['hit_rate']:.1f}%, {p['hit_source']})")
    for ch, c in p["decode_ceilings_by_ram_channels"].items():
        a(f"  ceiling, {ch}-channel RAM: pipeline {c['pipeline_tok_s']:.0f} tok/s | ideal TP4 {c['tp_ideal_tok_s']:.0f} tok/s  (upper bounds, no MTP)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--hw", default=os.path.join(os.path.dirname(__file__), "..", "hardware", "4x4080s-32g.json"))
    ap.add_argument("--quant", default=None, help="expert format; default: the model's native format")
    ap.add_argument("--dense-quant", default="Q8_0")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--state-copies", type=int, default=4, help="recurrent state copies kept for MTP verify")
    ap.add_argument("--hit-curve", default=None, help="JSON from tools/g0/analyze.py")
    ap.add_argument("--ram-channels", type=int, default=None, choices=[4, 8])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    model = json.load(open(a.model))
    hw = json.load(open(a.hw))
    curve = json.load(open(a.hit_curve))["global_curve"] if a.hit_curve else None
    p = plan(model, hw, a.quant or model["native_expert_quant"], a.ctx, a.dense_quant,
             a.state_copies, curve, a.ram_channels)
    print(json.dumps(p, indent=2) if a.json else render(p))


if __name__ == "__main__":
    main()
