#!/usr/bin/env python3
"""arch_ceilings: decode ceilings of competing multi-GPU architectures, as explicit critical-path formulas.

Written for the R9 adversarial review (docs/reassessment/research/R9-adversarial-ceilings.md). It re-uses the model
geometry of tools/roofline.py (expert size, dense bytes, KV/state reads, VRAM placement) and adds what that tool does
not model:

  * architectures: layer pipeline (pp), expert parallel with the dense part pipelined (ep), tensor parallel over a
    group of tp_size GPUs (tp; tp_size < n_gpu gives PP over TP groups) with two KV policies: 'replicate' (every rank
    keeps an undividable MLA latent / indexer key, as tools/roofline.py assumes) and 'owner' (layers whose KV cannot
    be split by head run whole on one GPU, as ExLlamaV3 does for MLA/DSA: exllamav3@151539c
    exllamav3/modules/mla_attn.py:1195-1244);
  * fixed costs that bandwidth models leave out: per-layer GPU launch/sync gaps, per-MoE-layer CPU job cost, the
    GPU<->host handoff of the CPU-expert path, all-reduce latency, per-step cost;
  * a CPU expert throughput cap below the DRAM bandwidth (kernel-bound CPUs);
  * hit-rate models: 'uniform' (hit = resident share, tools/roofline.py 'auto'), a static frequency-sorted profile
    curve measured from published GLM-5.3-Flash routing counts, a midpoint between the two, or a fixed value;
  * verification windows: union of experts per layer (independent routing or a previous-token reuse rho) and, for a
    frequency-sorted placement, additive misses (cold experts rarely repeat inside a window);
  * speculation: MTP (k drafts, the head's bytes per draft), DFlash2-style block drafts (one drafter pass per step,
    per-position acceptance list), n-gram/suffix drafts (CPU, near zero cost);
  * physics bounds that no architecture can beat (DRAM bytes of the misses, aggregate VRAM bandwidth);
  * sweeps over any parameter and a solver for the threshold value that reaches a target tok/s.

Everything printed is PROVISIONAL / HYPOTHESIS: a bandwidth-and-latency model with stated parameters, not a measured
speed. Expected values are used inside max() terms, which can only overstate tok/s, so every number is a ceiling for
the stated parameters. Parameter labels and sources: PARAM_NOTES below and the R9 report.

Usage:
  python3 tools/ceilings/arch_ceilings.py --preset base --model glm --hit uniform,mid,static
  python3 tools/ceilings/arch_ceilings.py --preset base --model glm --arch tp_owner --spec mtp:3:0.745
  python3 tools/ceilings/arch_ceilings.py --preset base --model glm --arch tp_owner --hit static \
      --sweep allreduce_us=5,15,30,60 --sweep dram_channels=4,8
  python3 tools/ceilings/arch_ceilings.py --preset base --model glm --arch tp_owner --hit static \
      --solve allreduce_us --target 80
  python3 tools/ceilings/arch_ceilings.py --params my.json --model models/specs/glm-5.3-flash.json --json
  python3 tools/ceilings/arch_ceilings.py --dump-params --preset base > my.json
  python3 tools/ceilings/arch_ceilings.py --report r9      # every table of the R9 report
"""
import argparse
import copy
import functools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import roofline as rl  # noqa: E402  (geometry shared with the project's roofline; read-only use)

GIB = 1024 ** 3
SPEC_DIR = os.path.join(ROOT, "models", "specs")
MODEL_ALIASES = {
    "glm": ("glm-5.3-flash.json", {"expert_bpw": 4.89, "dense_bpw": 8.04}),
    "qwen": ("qwen3.8-flash-next.json", {"expert_bpw": 5.10, "dense_bpw": 10.0}),
    "mimo": ("mimo-v2.6-flash.json", {"expert_bpw": 4.25, "dense_bpw": 8.5}),
    "dsv4f": ("deepseek-v4-flash-0731.json", {"expert_bpw": 4.25, "dense_bpw": 8.5}),
    "dsv41": ("deepseek-v4.1-flash.json", {"expert_bpw": 4.25, "dense_bpw": 8.5}),
}

# Static hit curves: share of routed picks served by the top-c share of experts per layer (frequency-sorted), computed
# from the per-layer decode selection counts published in github.com/0xSero/glm53-flash-offload @6769b27
# data/stats_own_dec.json (sha256 79bff1e3...; 42 layers x 288 experts, 74,000 picks per layer ~ 9,250 tokens; the
# workload behind the counts is not documented). IN-SAMPLE: the same counts choose and score the placement, so these
# are upper references for a static placement, not a prediction for another workload. 'global' = one slot budget
# shared across layers (top shares over all layer-expert pairs). Script: R9 report, appendix A.
GLM_STATIC_CURVE = [
    (0.0, 0.0), (0.05, 0.2217), (0.1, 0.3556), (0.15, 0.4518), (0.2, 0.5372), (0.25, 0.6059), (0.3, 0.6662),
    (0.35, 0.7225), (0.4, 0.7685), (0.45, 0.8115), (0.5, 0.8463), (0.55, 0.8764), (0.6, 0.904), (0.65, 0.9259),
    (0.7, 0.9456), (0.75, 0.961), (0.8, 0.9736), (0.85, 0.9843), (0.9, 0.9917), (0.95, 0.9972), (1.0, 1.0)]
GLM_STATIC_CURVE_GLOBAL = [
    (0.0, 0.0), (0.05, 0.2374), (0.1, 0.3631), (0.15, 0.4589), (0.2, 0.5385), (0.25, 0.6074), (0.3, 0.6688),
    (0.35, 0.7234), (0.4, 0.7726), (0.45, 0.8167), (0.5, 0.8558), (0.55, 0.89), (0.6, 0.9191), (0.65, 0.9428),
    (0.7, 0.9616), (0.75, 0.976), (0.8, 0.9864), (0.85, 0.9933), (0.9, 0.9974), (0.95, 0.9994), (1.0, 1.0)]

# DFlash2 drafter for GLM-5.3-Flash: P(at least i drafts accepted), K=7, held-out 1K-token traces, B300
# (third_party/hf/glm-5.3-flash-dflash2-g/README.md in this repo; REPORTED by the drafter card). 4K-token traces:
DFLASH2_GLM_K7 = [0.753, 0.562, 0.422, 0.324, 0.253, 0.201, 0.161]
DFLASH2_GLM_K7_4K = [0.729, 0.526, 0.388, 0.292, 0.225, 0.177, 0.141]

DEFAULTS = {
    "ctx": 262144,
    "quant": None,                  # None: per-model default of MODEL_ALIASES (bpw of the files in ROOFLINE.md)
    "kv_bytes_per_elem": 2.0,
    "n_gpu": 4,
    "vram_gib": 32.0,
    "gpu_reserve_gib": 2.0,
    "extra_vram_gib": 0.0,
    "vram_gbs": 736.0,
    "vram_eff": 0.75,
    "ram_gib": 128.0,
    "ram_reserve_gib": 12.0,
    "dram_channels": 8,
    "dram_gbs_per_channel": 25.6,
    "dram_eff": 0.7,
    "cpu_compute_gbs": None,
    "nvme_gbs": 21.0,               # 3 x 7.0 GB/s placeholder (customer correction 2026-10-12: 3 x 1 TB; was 4 x 7.0 = 28.0)
    "nvme_eff": 0.6,
    "handoff_us": 10.0,
    "stage_hop_us": 10.0,
    "allreduce_us": 30.0,
    "ep_xfer_us": 10.0,
    "gpu_layer_fixed_us": 0.0,
    "cpu_layer_fixed_us": 0.0,
    "step_fixed_us": 0.0,
    "cpu_fixed_when": "any_miss",
    "overlap": True,
    "hit": "uniform",
    "union": "independent",
    "miss_union": "auto",
    "spec": "none",
    "draft_gb": None,
    "draft_ar_per_token": 0,
    "drafter_params": 2.47e9,
    "drafter_bpw": 4.85,
    "drafter_layers": 8,
    "ngram_draft_ms": 0.05,
    "tp_size": 4,
    "tp_kv": "replicate",
    "tp_experts": "split",
    "tp_replicated_keys": [],
    "owner_dense_keys": ["dsa_attention"],
    "cpu_on_reduce_path": False,
    "ep_imbalance": "roofline",
    "pcie_gbs": 25.0,
    "prefetch_recall": 0.0,
    "prefetch_precision": 1.0,
    "pipeline_windows": 1,
}

PARAM_NOTES = {
    "vram_gbs": "REPORTED: stock RTX 4080 Super 23 Gbps x 256-bit = 736 GB/s; modded 32 GB boards UNKNOWN",
    "vram_eff": "PROVISIONAL: share of peak a batch-1 GEMV reaches; anchors 0.50-0.57 (ExLlamaV3 EXL3 dense GEMVs on a "
                "3090, 0xSero profile, REPORTED) to ~0.9 (llama.cpp tg on 4090s, research/R3 estimate)",
    "dram_eff": "PROVISIONAL: share of DDR4 peak the CPU expert path reaches; STREAM ~0.85 of peak (REPORTED, R4)",
    "cpu_compute_gbs": "PROVISIONAL cap on the CPU expert kernel: EXL3 trellis AVX2 31-70 GB/s per call on a 24-core "
                       "Zen 3 (0xSero results/cpu_kernel, REPORTED); ~90-100 GB/s effective for ik_llama Q4_K on the "
                       "epyc-ai host (ROOFLINE.md section 1 calibration, PROVISIONAL)",
    "handoff_us": "PROVISIONAL: one-way GPU<->host handoff of the CPU-expert path; publish->seen 10 us p50 (0xSero, "
                  "REPORTED)",
    "allreduce_us": "HYPOTHESIS: 8-16 KB all-reduce over 4 GPUs without P2P; floor ~5 us (mapped-memory protocol), "
                    "ExLlamaV3 native backend UNKNOWN, NCCL SHM tens of us",
    "gpu_layer_fixed_us": "HYPOTHESIS: kernel launch / dependency gaps per layer; ~1-2 us per graph node x 15-30 nodes",
    "cpu_layer_fixed_us": "HYPOTHESIS: CPU pool wake + barrier per MoE layer that has CPU work",
    "step_fixed_us": "HYPOTHESIS: sampling, host sync and scheduler per decode step",
    "draft_ar_per_token": "HYPOTHESIS: all-reduces of the MTP layer per drafted token under TP (2: attention + MoE); "
                          "tools/roofline.py charges 0",
    "drafter_params": "PROVISIONAL: DFlash2 GLM drafter ~1.84B own parameters + untied lm_head 0.634B read per step "
                      "(third_party/hf/glm-5.3-flash-dflash2-g/README.md)",
    "drafter_bpw": "PROVISIONAL: Q4_K_M drafter as in research/R2 section 4 row 2",
    "prefetch_recall": "HYPOTHESIS: share of a layer's CPU misses predicted one layer ahead (next-layer router on the "
                       "current input) and copied over PCIe so the GPU computes them; 0xSero measured NVMe misses "
                       "40 -> 22 per token with it (REPORTED); router-free predictors recall 7-20% (REPORTED)",
    "prefetch_precision": "HYPOTHESIS: share of prefetched experts that are actually used; the rest wastes DRAM and "
                          "PCIe bandwidth",
    "pcie_gbs": "PROVISIONAL: pinned H2D per Gen4 x16 link, ~24-26 GB/s measured on 4090/4080S class (R4)",
    "pipeline_windows": "HYPOTHESIS: speculative windows in flight in a layer pipeline (Strata --pipeline-windows, "
                        "2 stages only, +16% measured REPORTED in research/R3 2.5); a window started early is useful "
                        "only if every earlier in-flight window is fully accepted and its bonus token guessed",
}

PRESETS = {
    # Exactly tools/roofline.py with hardware/target-4x4080-32g.unverified.json: used by the consistency tests.
    "roofline": {},
    # R9 central engine assumptions (HYPOTHESIS; see PARAM_NOTES and the report).
    "base": {"allreduce_us": 15.0, "ep_xfer_us": 20.0, "gpu_layer_fixed_us": 20.0, "cpu_layer_fixed_us": 10.0,
             "step_fixed_us": 300.0, "cpu_fixed_when": "poisson", "tp_replicated_keys": ["routers", "mhc"],
             "ep_imbalance": "exact", "draft_ar_per_token": 2},
    # Physics-leaning: what tuned kernels and a mapped-memory all-reduce could approach (HYPOTHESIS).
    "optimistic": {"vram_eff": 0.9, "dram_eff": 0.85, "allreduce_us": 5.0, "ep_xfer_us": 8.0, "handoff_us": 5.0,
                   "stage_hop_us": 5.0, "gpu_layer_fixed_us": 5.0, "cpu_layer_fixed_us": 5.0, "step_fixed_us": 100.0,
                   "cpu_fixed_when": "poisson", "tp_replicated_keys": ["routers", "mhc"], "ep_imbalance": "exact",
                   "draft_ar_per_token": 2},
    # Engine-pessimistic: measured-class efficiencies of today's engines (HYPOTHESIS).
    "pessimistic": {"vram_eff": 0.55, "dram_eff": 0.55, "allreduce_us": 50.0, "ep_xfer_us": 40.0, "handoff_us": 15.0,
                    "stage_hop_us": 20.0, "gpu_layer_fixed_us": 60.0, "cpu_layer_fixed_us": 20.0,
                    "step_fixed_us": 1000.0, "cpu_fixed_when": "poisson", "tp_replicated_keys": ["routers", "mhc"],
                    "ep_imbalance": "exact", "draft_ar_per_token": 2},
}

ARCHS = {
    # name: (strategy, overrides)
    "pp_serial": ("pp", {"overlap": False}),
    "pp": ("pp", {}),
    "ep": ("ep", {}),
    "tp_repl": ("tp", {"tp_kv": "replicate", "tp_size": None}),
    "tp_owner": ("tp", {"tp_kv": "owner", "tp_size": None}),
    "tp_owner_hostcpu": ("tp", {"tp_kv": "owner", "tp_size": None, "cpu_on_reduce_path": True}),
    "pp2_tp2": ("tp", {"tp_kv": "owner", "tp_size": 2}),
    "pp_pipewin": ("pp", {"pipeline_windows": 4}),
}


# ---------------------------------------------------------------------------------------------- params

def resolve_params(preset="base", overrides=None):
    p = copy.deepcopy(DEFAULTS)
    p.update(copy.deepcopy(PRESETS[preset]))
    for k, v in (overrides or {}).items():
        set_param(p, k, v)
    return p


def set_param(p, key, value):
    """key may be dotted ('quant.expert_bpw')."""
    parts = key.split(".")
    d = p
    for part in parts[:-1]:
        if d.get(part) is None:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def parse_value(s):
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load_model(name):
    """'glm' / 'qwen' / ... or a path to a models/specs JSON; returns (spec, default quant)."""
    if name in MODEL_ALIASES:
        fn, q = MODEL_ALIASES[name]
        path = os.path.join(SPEC_DIR, fn)
    else:
        path, q = name, {"expert_bpw": 4.5, "dense_bpw": 8.5}
        base = os.path.basename(name)
        for fn, qq in MODEL_ALIASES.values():
            if base == fn:
                q = qq
    with open(path) as f:
        spec = json.load(f)
    return spec, dict(q)


def quant_of(spec_q, p):
    q = dict(spec_q)
    if p.get("quant"):
        q.update(p["quant"])
    q["kv_bytes_per_elem"] = p["kv_bytes_per_elem"]
    return q


# ---------------------------------------------------------------------------------------------- geometry

def kv_owner_split(spec, ctx, kvb, tp, n_tok):
    """KV/state bytes per step under TP with the 'owner' policy: groups that cannot be split by head (kv_heads 1)
    are read whole by their owner GPU; the others are split over tp GPUs. Returns (owner_bytes, per_gpu_split)."""
    own = [g for g in spec["attention"] if g.get("kv_heads", tp) <= 1]
    rest = [g for g in spec["attention"] if g.get("kv_heads", tp) > 1]
    b_own = rl.kv_read_bytes({"attention": own}, ctx, kvb, 1, n_tok) if own else 0.0
    b_rest = rl.kv_read_bytes({"attention": rest}, ctx, kvb, tp, n_tok) if rest else 0.0
    return b_own, b_rest


def placement(spec, p, q, kv_replication):
    """VRAM/RAM/NVMe split of the routed experts (mirrors tools/roofline.py placement; kv_replication = TP group
    size whose undividable KV is replicated, 1 for none)."""
    n_gpu = p["n_gpu"]
    e_bytes = rl.expert_params(spec) * q["expert_bpw"] / 8
    experts_bytes = rl.routed_stored_params(spec) * q["expert_bpw"] / 8
    dense_bytes = rl.dense_stored_params(spec) * q["dense_bpw"] / 8
    kv_bytes = rl.kv_store_bytes(spec, p["ctx"], q["kv_bytes_per_elem"], n_gpu_tp=kv_replication)
    vram = n_gpu * p["vram_gib"] * GIB
    fixed = n_gpu * p["gpu_reserve_gib"] * GIB + dense_bytes + kv_bytes + p.get("extra_vram_gib", 0.0) * GIB
    cache = max(0.0, vram - fixed)
    resident = min(experts_bytes, cache)
    spill = experts_bytes - resident
    ram_usable = (p["ram_gib"] - p["ram_reserve_gib"]) * GIB
    ram_experts = min(spill, ram_usable)
    ssd = spill - ram_experts
    return {"expert_mib": e_bytes / 2 ** 20, "experts_gib": experts_bytes / GIB, "dense_gib": dense_bytes / GIB,
            "kv_state_gib": kv_bytes / GIB, "vram_expert_cache_gib": cache / GIB,
            "resident_frac": resident / experts_bytes if experts_bytes else 1.0,
            "ram_frac": ram_experts / experts_bytes if experts_bytes else 0.0,
            "ssd_frac": ssd / experts_bytes if experts_bytes else 0.0,
            "dense_kv_fit_vram": fixed <= vram}


def interp(curve, x):
    x = min(1.0, max(0.0, x))
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def resolve_hit(hit, pl, spec):
    """Returns (hit_vram, label). 'uniform': the resident share (tools/roofline.py 'auto'); 'static'/'static_global':
    the in-sample GLM profile curve at the resident share; 'mid': midpoint of uniform and static; a float: fixed."""
    r = pl["resident_frac"]
    if isinstance(hit, (int, float)):
        return float(hit), f"{float(hit):.3f}"
    if hit == "uniform":
        return r, f"uniform {r:.3f}"
    if hit in ("static", "static_global", "mid"):
        curve = GLM_STATIC_CURVE_GLOBAL if hit == "static_global" else GLM_STATIC_CURVE
        s = interp(curve, r)
        tag = "" if spec["name"].startswith("GLM") else "(GLM curve as proxy)"
        if hit == "mid":
            return (r + s) / 2, f"mid{tag} {(r + s) / 2:.3f}"
        return s, f"{hit}{tag} {s:.3f}"
    raise ValueError(f"unknown hit model {hit!r}")


def union_experts(spec, n_tok, union):
    """Distinct experts per MoE layer touched by n_tok tokens. 'independent': uniform independent routing
    (tools/roofline.py distinct_experts); a float rho: each new token re-uses a share rho of experts already in the
    window (previous-token reuse, e.g. 0.246 measured on GLM by strata-glm-3090 PROBLEMS.md@794cd69)."""
    m = spec["moe"]
    k, e = m["top_k"], m["n_experts"]
    if n_tok <= 1:
        return float(k)
    ind = rl.distinct_experts(e, k, n_tok)
    if union == "independent":
        return ind
    rho = float(union)
    return min(ind, k + (n_tok - 1) * k * (1.0 - rho))


@functools.lru_cache(maxsize=None)
def expected_max_load(m, g):
    """E[max bin count] for m balls thrown uniformly into g bins (exact, m <= 160; normal approximation above)."""
    if g <= 1 or m <= 0:
        return float(m)
    if m > 160:
        mu = m / g
        return mu + math.sqrt(2 * math.log(g) * mu * (1 - 1 / g))
    logfact = [math.lgamma(i + 1) for i in range(m + 1)]
    ex = 0.0
    prev = 0.0
    for c in range(0, m + 1):
        # P(all bins <= c) = m! / g^m * [x^m] (sum_{j<=c} x^j / j!)^g
        poly = [0.0] * (m + 1)
        poly[0] = 1.0
        for _ in range(g):
            new = [0.0] * (m + 1)
            for t, v in enumerate(poly):
                if v == 0.0:
                    continue
                for j in range(0, min(c, m - t) + 1):
                    new[t + j] += v * math.exp(-logfact[j])
            poly = new
        pc = math.exp(logfact[m] - m * math.log(g)) * poly[m] if poly[m] > 0 else 0.0
        ex += c * (pc - prev)
        prev = pc
        if pc > 1 - 1e-12:
            break
    return ex


def imbalance(p, spec, u_hit, n_tok, g):
    if g <= 1:
        return 1.0
    if p["ep_imbalance"] == "roofline":
        return rl.ep_imbalance(spec["moe"]["top_k"], g) if n_tok == 1 else 1.0
    m = max(1, int(round(u_hit)))
    return expected_max_load(m, g) / (m / g)


def spec_plan(spec, p, q):
    """Speculation: (k drafts, tokens per step tau, draft seconds on GPU bytes, label, extra CPU seconds, draft layers).
    p['spec']: 'none' | 'mtp:K:ALPHA' | 'dflash2:K' (GLM per-position list) | 'dflash2_4k:K' | 'ngram:K:ALPHA' |
    dict {kind, k, accept (float or cumulative list), draft_gb, draft_ms}."""
    s = p["spec"]
    if isinstance(s, str):
        parts = s.split(":")
        kind = parts[0]
        if kind == "none":
            return {"kind": "none", "k": 0, "tau": 1.0, "gpu_bytes": 0.0, "cpu_s": 0.0, "layers": 0, "label": "none"}
        if kind in ("dflash2", "dflash2_4k"):
            k = int(parts[1]) if len(parts) > 1 else 7
            base = DFLASH2_GLM_K7 if kind == "dflash2" else DFLASH2_GLM_K7_4K
            s = {"kind": "dflash2", "k": k, "accept": base[:k]}
        else:
            s = {"kind": kind, "k": int(parts[1]), "accept": float(parts[2])}
    kind, k = s["kind"], int(s["k"])
    acc = s["accept"]
    if isinstance(acc, (list, tuple)):
        tau = 1.0 + sum(acc[:k])
        alabel = f"tau {tau:.2f}"
    else:
        tau = rl.spec_tokens_per_step(float(acc), k)
        alabel = f"a {float(acc):.3f}"
    if kind == "mtp":
        per = p["draft_gb"] * 1e9 if p.get("draft_gb") is not None else rl.draft_bytes_per_token(spec, q)
        if per is None:
            raise ValueError(f"{spec['name']}: no MTP head in the spec; set draft_gb")
        return {"kind": kind, "k": k, "tau": tau, "gpu_bytes": k * per, "cpu_s": 0.0, "layers": k,
                "label": f"mtp k={k} {alabel}"}
    if kind == "dflash2":
        gb = p["draft_gb"] * 1e9 if p.get("draft_gb") is not None else p["drafter_params"] * p["drafter_bpw"] / 8
        return {"kind": kind, "k": k, "tau": tau, "gpu_bytes": gb, "cpu_s": 0.0, "layers": p["drafter_layers"],
                "label": f"dflash2 K={k} {alabel}"}
    if kind == "ngram":
        return {"kind": kind, "k": k, "tau": tau, "gpu_bytes": 0.0, "cpu_s": p["ngram_draft_ms"] * 1e-3, "layers": 0,
                "label": f"ngram k={k} {alabel}"}
    raise ValueError(f"unknown speculation kind {kind!r}")


def dense_parts(spec, q, p, strategy):
    """Per-step dense bytes split into (divisible over TP, replicated on every TP rank, owner-only)."""
    dp = spec["dense_params"]
    active = {k: v for k, v in dp.items() if k not in ("embedding", "mtp", "vision")}
    total = sum(active.values()) * q["dense_bpw"] / 8
    if strategy != "tp":
        return total, 0.0, 0.0
    rep_keys = set(p.get("tp_replicated_keys") or [])
    own_keys = set(p.get("owner_dense_keys") or []) if p["tp_kv"] == "owner" else set()
    rep = sum(v for k, v in active.items() if k in rep_keys) * q["dense_bpw"] / 8
    own = sum(v for k, v in active.items() if k in own_keys) * q["dense_bpw"] / 8
    return total - rep - own, rep, own


# ---------------------------------------------------------------------------------------------- decode step

def step(spec, spec_q, p, arch):
    """One decode (or verification) step of architecture `arch` under params p. Returns a dict with tok/s, the ms
    breakdown and the binding term. All times are expected values on the critical path."""
    strategy, over = ARCHS[arch]
    p = copy.deepcopy(p)
    for k, v in over.items():
        p[k] = v
    if p["tp_size"] is None:
        p["tp_size"] = p["n_gpu"]
    q = quant_of(spec_q, p)
    m = spec["moe"]
    L_m, L = m["layers"], spec["n_layers"]
    g = p["n_gpu"]
    tp = p["tp_size"] if strategy == "tp" else 1
    kv_rep = tp if (strategy == "tp" and p["tp_kv"] == "replicate") else 1
    pl = placement(spec, p, q, kv_rep)
    hv, hit_label = resolve_hit(p["hit"], pl, spec)
    off = pl["ram_frac"] + pl["ssd_frac"]
    hr = (1 - hv) * (pl["ram_frac"] / off if off > 0 else 1.0) if hv < 1 else 0.0
    hs = max(0.0, 1.0 - hv - hr)

    sp = spec_plan(spec, p, q)
    n_tok = 1 + sp["k"]
    u = union_experts(spec, n_tok, p["union"])
    mu = p["miss_union"]
    if mu == "auto":
        mu = "additive" if p["hit"] in ("static", "static_global", "mid") else "proportional"
    if mu == "additive":
        u_miss = min(u, n_tok * m["top_k"] * (1 - hv))
    else:
        u_miss = (1 - hv) * u
    u_hit = u - u_miss

    e = rl.expert_params(spec) * q["expert_bpw"] / 8
    B_g = p["vram_gbs"] * 1e9 * p["vram_eff"]
    B_dram = p["dram_channels"] * p["dram_gbs_per_channel"] * 1e9 * p["dram_eff"]
    B_c = min(B_dram, p["cpu_compute_gbs"] * 1e9) if p.get("cpu_compute_gbs") else B_dram
    B_ssd = p["nvme_gbs"] * 1e9 * p["nvme_eff"]

    w_hit_l = u_hit * e
    miss_l = u_miss * e
    ram_l = miss_l * (hr / (hr + hs)) if (hr + hs) > 0 else 0.0
    ssd_l = miss_l - ram_l
    # layer-ahead prefetch of predicted RAM misses over PCIe (the GPU then computes them); every byte still crosses
    # DRAM once, wrong predictions cross it too
    gpu_div = {"pp": 1, "ep": g, "tp": tp}[strategy]
    pf_l = p["prefetch_recall"] * ram_l
    waste_l = pf_l * (1.0 / max(1e-9, p["prefetch_precision"]) - 1.0)
    ram_cpu_l = ram_l - pf_l
    B_p = p["pcie_gbs"] * 1e9
    t_pcie_l = (pf_l + waste_l) / (gpu_div * B_p)
    t_cpu_compute = ram_cpu_l / B_c + ssd_l / B_ssd + ssd_l / B_c
    t_dram_l = (ram_cpu_l + ssd_l + pf_l + waste_l) / B_dram
    t_cpu_l = max(t_cpu_compute, t_dram_l) if miss_l > 0 else 0.0
    w_hit_l += pf_l
    cpu_bytes_l = ram_cpu_l + ssd_l               # what the CPU itself computes in this layer
    u_cpu = u_miss * (cpu_bytes_l / miss_l) if miss_l > 0 else 0.0
    if cpu_bytes_l > 0:
        p_cpu = 1.0 if p["cpu_fixed_when"] == "any_miss" else 1.0 - math.exp(-u_cpu)
    else:
        p_cpu = 0.0
    d_div, d_rep, d_own = dense_parts(spec, q, p, strategy)
    w_router_l = spec["dense_params"].get("routers", 0.0) * q["dense_bpw"] / 8 / max(1, L_m)
    if strategy == "tp" and p["cpu_on_reduce_path"]:
        cpu_sync = p_cpu * (w_router_l / B_c + p["cpu_layer_fixed_us"] * 1e-6)
    else:
        cpu_sync = p_cpu * (2 * p["handoff_us"] + p["cpu_layer_fixed_us"]) * 1e-6
    if cpu_bytes_l > 0:
        t_cpu_path = t_cpu_l + cpu_sync
    else:                                          # prefetch took every miss: only the DRAM reads remain
        t_cpu_path = t_dram_l if miss_l > 0 else 0.0

    def comb(a, b):
        if b <= 0:
            return a
        return max(a, b) if p["overlap"] else a + b

    kvb = q["kv_bytes_per_elem"]
    ctx = p["ctx"]
    fixed = L * p["gpu_layer_fixed_us"] * 1e-6 + p["step_fixed_us"] * 1e-6
    comm = 0.0
    if strategy in ("pp", "ep"):
        w_kv = rl.kv_read_bytes(spec, ctx, kvb, 1, n_tok)
        t_dense = (d_div + w_kv) / B_g
        if strategy == "pp":
            t_gpu_l = w_hit_l / B_g
            moe_l = comb(t_gpu_l, t_cpu_path)
        else:                                      # scatter + gather of each MoE layer: on the critical path
            imb = imbalance(p, spec, u_hit, n_tok, g)
            t_gpu_l = imb * w_hit_l / (g * B_g)
            moe_l = comb(t_gpu_l, t_cpu_path)
            comm += L_m * 2 * p["ep_xfer_us"] * 1e-6
        comm += (g - 1) * p["stage_hop_us"] * 1e-6
        t_draft = sp["gpu_bytes"] / B_g + sp["layers"] * p["gpu_layer_fixed_us"] * 1e-6
    else:
        if p["tp_kv"] == "owner":
            kv_own, kv_split = kv_owner_split(spec, ctx, kvb, tp, n_tok)
            w_kv_crit = kv_own + kv_split
        else:
            w_kv_crit = rl.kv_read_bytes(spec, ctx, kvb, tp, n_tok)
        t_dense = (d_div / tp + d_rep + d_own + w_kv_crit) / B_g
        if p["tp_experts"] == "ep":
            imb = imbalance(p, spec, u_hit, n_tok, tp)
        else:
            imb = 1.0
        t_gpu_l = imb * w_hit_l / (tp * B_g)
        moe_l = comb(t_gpu_l, t_cpu_path)
        comm += L * 2 * p["allreduce_us"] * 1e-6 + (g // tp - 1) * p["stage_hop_us"] * 1e-6
        if sp["kind"] == "mtp":                    # the MTP head is one more TP layer per draft
            comm += sp["k"] * p["draft_ar_per_token"] * p["allreduce_us"] * 1e-6
            t_draft = sp["gpu_bytes"] / (tp * B_g) + sp["layers"] * p["gpu_layer_fixed_us"] * 1e-6
        else:                                      # a separate drafter runs whole on one GPU
            t_draft = sp["gpu_bytes"] / B_g + sp["layers"] * p["gpu_layer_fixed_us"] * 1e-6
    t_draft += sp["cpu_s"]
    if t_pcie_l > 0:                               # a prefetch copy longer than one layer delays the layer it feeds
        layer_t = (t_dense + L_m * moe_l + comm + fixed) / L
        moe_l += max(0.0, t_pcie_l - layer_t)
    moe = L_m * moe_l
    t = t_dense + moe + comm + fixed + t_draft
    lk = spec.get("lookup")
    t_lookup = rl.lookup_bytes_per_token(spec)[0] / B_ssd if (lk and lk.get("tier") == "ssd") else 0.0
    t = max(t, t_lookup)
    tok_s = sp["tau"] / t
    win_mult, pi, best_w = 1.0, 0.0, 1
    if strategy == "pp" and p.get("pipeline_windows", 1) > 1 and sp["k"] > 0:
        # W windows in flight, a new one entering stage 1 every t/W. A window started early is valid only if the
        # window before it was fully accepted and its bonus token guessed: pi = P(all k accepted) * P(bonus guessed).
        # Renewal cycle: a chain of N valid windows (E[N] = 1 / (1 - pi), the last one being the partly accepted
        # window) followed by a bubble of W - 1 issue slots (the discarded in-flight windows). Tokens per cycle
        # tau * E[N]; cycle time (E[N] + W - 1) * t / W; the CPU also computes the misses of the discarded windows
        # (on average (W - 1) / 2 windows' worth of layers per cycle).
        W = min(int(p["pipeline_windows"]), g)
        s = p["spec"]
        if isinstance(s, dict):
            acc = s["accept"]
        elif s.startswith("dflash2_4k"):
            acc = DFLASH2_GLM_K7_4K
        elif s.startswith("dflash2"):
            acc = DFLASH2_GLM_K7
        else:
            acc = float(s.split(":")[2])
        if isinstance(acc, (list, tuple)):
            lst = list(acc)[:sp["k"]]
            pi = lst[-1] * (lst[-1] / lst[-2] if len(lst) > 1 else lst[-1])
        else:
            pi = float(acc) ** (sp["k"] + 1)
        en = 1.0 / (1.0 - min(pi, 1.0 - 1e-9))
        best_w = 1
        for w in range(2, W + 1):                  # a scheduler picks the depth; W = 1 is the plain pipeline
            t_cycle = max((en + w - 1) * t / w, (en + (w - 1) / 2.0) * L_m * t_cpu_path)
            if en * t / t_cycle > win_mult:
                win_mult, best_w = en * t / t_cycle, w
        tok_s = sp["tau"] * win_mult / t
    parts = {"dense_kv": t_dense, "moe_crit": moe, "moe_gpu": L_m * t_gpu_l, "moe_cpu": L_m * t_cpu_path,
             "comm": comm, "fixed": fixed, "draft": t_draft}
    bind = "CPU/DRAM experts" if (t_cpu_path > t_gpu_l and moe > 0.25 * t) else max(
        (("GPU dense+KV", t_dense), ("GPU experts", L_m * t_gpu_l), ("collectives/hops", comm), ("fixed overheads", fixed),
         ("draft", t_draft)), key=lambda kv: kv[1])[0]
    return {"model": spec["name"], "arch": arch, "spec": sp["label"], "hit": hit_label, "hit_vram": hv,
            "resident": pl["resident_frac"], "kv_state_gib": pl["kv_state_gib"], "n_tok": n_tok, "tau": sp["tau"],
            "union": u, "u_miss": u_miss, "ms": t * 1e3, "tok_s": tok_s,
            "ms_parts": {k: v * 1e3 for k, v in parts.items()},
            "cpu_cap_tok_s": sp["tau"] / (L_m * t_cpu_path) if t_cpu_path > 0 else float("inf"),
            "binding": bind, "dram_gbs_eff": B_dram / 1e9, "cpu_gbs_eff": B_c / 1e9,
            "pipeline_window_mult": win_mult, "window_valid_prob": pi, "pipeline_windows_used": best_w}


def physics_bounds(spec, spec_q, p, kv_replication=1):
    """Bounds no architecture can beat at the given hit share (peak bandwidths, no fixed costs, no collectives):
    misses cross DRAM once; resident bytes are read at best by all GPUs in parallel; a layer pipeline reads at one
    GPU's peak at a time."""
    q = quant_of(spec_q, p)
    pl = placement(spec, p, q, kv_replication)
    hv, _ = resolve_hit(p["hit"], pl, spec)
    sp = spec_plan(spec, p, q)
    n_tok = 1 + sp["k"]
    m = spec["moe"]
    u = union_experts(spec, n_tok, p["union"])
    mu = p["miss_union"]
    if mu == "auto":
        mu = "additive" if p["hit"] in ("static", "static_global", "mid") else "proportional"
    u_miss = min(u, n_tok * m["top_k"] * (1 - hv)) if mu == "additive" else (1 - hv) * u
    e = rl.expert_params(spec) * q["expert_bpw"] / 8
    dram_peak = p["dram_channels"] * p["dram_gbs_per_channel"] * 1e9
    vram_peak = p["vram_gbs"] * 1e9
    w_dense = sum(v for k, v in spec["dense_params"].items() if k not in ("embedding", "mtp", "vision")) \
        * q["dense_bpw"] / 8
    w_kv = rl.kv_read_bytes(spec, p["ctx"], q["kv_bytes_per_elem"], 1, n_tok)
    gpu_bytes = w_dense + w_kv + m["layers"] * (u - u_miss) * e + sp["gpu_bytes"]
    waste = p["prefetch_recall"] * (1.0 / max(1e-9, p["prefetch_precision"]) - 1.0)
    miss_bytes = m["layers"] * u_miss * e * (1.0 + waste)
    return {"hit_vram": hv, "tau": sp["tau"],
            "dram_bound_tok_s": sp["tau"] * dram_peak / miss_bytes if miss_bytes > 0 else float("inf"),
            "aggregate_vram_bound_tok_s": sp["tau"] * p["n_gpu"] * vram_peak / gpu_bytes,
            "single_gpu_pp_bound_tok_s": sp["tau"] * vram_peak / gpu_bytes,
            "gpu_gb_per_step": gpu_bytes / 1e9, "dram_gb_per_step": miss_bytes / 1e9}


# ---------------------------------------------------------------------------------------------- driver

def run(models, archs, hits, specs, p):
    rows = []
    for mname in models:
        spec, sq = load_model(mname)
        for arch in archs:
            for h in hits:
                for s in specs:
                    pp = copy.deepcopy(p)
                    pp["hit"] = h
                    pp["spec"] = s
                    try:
                        rows.append(step(spec, sq, pp, arch))
                    except ValueError as ex:
                        rows.append({"model": spec["name"], "arch": arch, "error": str(ex)})
    return rows


def render(rows, title=None):
    out = []
    if title:
        out.append(title)
    hdr = ("model", "arch", "spec", "hit", "resid", "tok/s", "ms/step", "tau", "dense+kv", "moe_crit", "moe_gpu",
           "moe_cpu", "comm", "fixed", "draft", "CPU-cap", "binding")
    out.append(" | ".join(hdr))
    for r in rows:
        if "error" in r:
            out.append(f"{r['model']} | {r['arch']} | ERROR {r['error']}")
            continue
        mp = r["ms_parts"]
        cap = r["cpu_cap_tok_s"]
        arch = r["arch"] + (f"(W={r['pipeline_windows_used']})" if r.get("pipeline_windows_used", 1) > 1 else "")
        out.append(" | ".join([
            r["model"], arch, r["spec"], r["hit"], f"{r['resident']:.3f}", f"{r['tok_s']:.1f}", f"{r['ms']:.2f}",
            f"{r['tau']:.2f}", f"{mp['dense_kv']:.2f}", f"{mp['moe_crit']:.2f}", f"{mp['moe_gpu']:.2f}",
            f"{mp['moe_cpu']:.2f}", f"{mp['comm']:.2f}", f"{mp['fixed']:.2f}", f"{mp['draft']:.2f}",
            "inf" if cap == float("inf") else f"{cap:.0f}", r["binding"]]))
    return "\n".join(out)


def solve(spec, sq, p, arch, key, target, lo, hi, iters=50, grid=100):
    """Value of parameter `key` in [lo, hi] at which tok/s first crosses `target`, scanning from lo (a grid scan
    then bisection, so bounds that are not monotone in the key, e.g. the overlapped pipeline in the hit share, are
    handled). Returns (value, 'min') when tok/s rises through the target (the least value that reaches it),
    (value, 'max') when it falls through it (the largest value that still reaches it), or (None, reason)."""
    def f(x):
        pp = copy.deepcopy(p)
        set_param(pp, key, x)
        return step(spec, sq, pp, arch)["tok_s"]
    xs = [lo + (hi - lo) * i / grid for i in range(grid + 1)]
    fs = [f(x) for x in xs]
    if max(fs) < target:
        return None, f"not reached in [{lo}, {hi}] (max tok/s {max(fs):.1f})"
    if min(fs) >= target:
        return None, f"exceeded over the whole interval (min tok/s {min(fs):.1f})"
    for i in range(grid):
        if (fs[i] - target) * (fs[i + 1] - target) <= 0 and fs[i] != fs[i + 1]:
            a, b, fa = xs[i], xs[i + 1], fs[i]
            for _ in range(iters):
                mid = (a + b) / 2
                if (f(mid) - target) * (fa - target) > 0:
                    a = mid
                else:
                    b = mid
            return (a + b) / 2, ("min" if fs[i + 1] > fs[i] else "max")
    return None, "no crossing found"


# ---------------------------------------------------------------------------------------------- R9 report tables

def report_r9():
    """Every table of docs/reassessment/research/R9-adversarial-ceilings.md, reproducible from this function."""
    out = []
    glm_archs = ["pp_serial", "pp", "ep", "tp_repl", "tp_owner", "tp_owner_hostcpu", "pp2_tp2"]
    for preset in ("base", "optimistic", "pessimistic"):
        for ch in (8, 4):
            p = resolve_params(preset, {"dram_channels": ch})
            rows = run(["glm"], glm_archs, ["uniform", "mid", "static"], ["none"], p)
            out.append(render(rows, f"\n## T1 GLM-5.3-Flash UD-Q4_K_XL (4.89/8.04 bpw), 262K, no speculation; preset "
                                    f"{preset}, DRAM {ch} channels"))
    specs = ["none", "mtp:1:0.745", "mtp:2:0.745", "mtp:3:0.745", "dflash2:7", "dflash2_4k:7", "ngram:3:0.556"]
    for preset in ("base", "optimistic"):
        p = resolve_params(preset, {"dram_channels": 8})
        rows = run(["glm"], ["pp", "pp_pipewin", "tp_owner"], ["uniform", "static"], specs, p)
        out.append(render(rows, f"\n## T2 GLM speculation (union: independent routing); preset {preset}, 8 channels"))
        p = resolve_params(preset, {"dram_channels": 8, "union": 0.246})
        rows = run(["glm"], ["pp", "pp_pipewin", "tp_owner"], ["static"], specs, p)
        out.append(render(rows, f"\n## T2b GLM speculation, union with previous-token reuse 0.246; preset {preset}"))
    p = resolve_params("base", {"dram_channels": 4})
    rows = run(["glm"], ["pp", "pp_pipewin", "tp_owner"], ["uniform", "static"], ["mtp:2:0.745", "dflash2:7"], p)
    out.append(render(rows, "\n## T2c GLM speculation; preset base, 4 channels"))
    for preset in ("roofline", "base", "optimistic", "pessimistic"):
        p = resolve_params(preset, {})
        rows = run(["qwen"], ["pp", "ep", "tp_repl", "tp_owner", "pp2_tp2"], [1.0],
                   ["none", "mtp:1:0.55", "mtp:2:0.55", "mtp:4:0.55"], p)
        out.append(render(rows, f"\n## T3 Qwen3.8-Flash-Next UD-Q4_K_XL (5.10/~10 bpw), all experts in VRAM; preset "
                                f"{preset}"))
    # T4: alternative quantisations of GLM within the >= 4 bpw rule (experts 4.25, dense 6.56 / 5.5)
    for quant in ({"expert_bpw": 4.25, "dense_bpw": 8.04}, {"expert_bpw": 4.89, "dense_bpw": 6.5625},
                  {"expert_bpw": 4.25, "dense_bpw": 5.5}):
        p = resolve_params("base", {"dram_channels": 8, "quant": quant})
        rows = run(["glm"], ["pp", "tp_owner"], ["uniform", "static"], ["none", "mtp:3:0.745"], p)
        out.append(render(rows, f"\n## T4 GLM with experts {quant['expert_bpw']} / dense {quant['dense_bpw']} bpw; "
                                f"preset base, 8 channels"))
    # T4b: out of policy (D5 requires >= 4 bpw per tensor): an EXL3-3.05-like expert size, for reference only
    p = resolve_params("base", {"dram_channels": 8, "quant": {"expert_bpw": 3.05, "dense_bpw": 8.04}})
    rows = run(["glm"], ["pp", "tp_owner"], ["uniform", "static"], ["none", "mtp:2:0.745"], p)
    out.append(render(rows, "\n## T4b OUT OF POLICY reference: GLM experts 3.05 / dense 8.04 bpw (fails the D5 >= 4 bpw "
                            "rule; quality UNKNOWN); preset base, 8 channels"))
    # T5: physics bounds
    out.append("\n## T5 physics bounds (peak bandwidths, no fixed costs): tok/s")
    out.append("model | hit | spec | DRAM ch | DRAM bound | aggregate VRAM bound | single-GPU PP bound")
    for mname in ("glm", "qwen"):
        spec, sq = load_model(mname)
        for h in (["uniform", "static"] if mname == "glm" else [1.0]):
            for s in ("none", "mtp:3:0.745" if mname == "glm" else "mtp:2:0.55"):
                for ch in (4, 8):
                    p = resolve_params("base", {"dram_channels": ch, "hit": h, "spec": s})
                    b = physics_bounds(spec, sq, p)
                    out.append(f"{spec['name']} | {h} {b['hit_vram']:.3f} | {s} | {ch} | "
                               f"{b['dram_bound_tok_s']:.1f} | {b['aggregate_vram_bound_tok_s']:.1f} | "
                               f"{b['single_gpu_pp_bound_tok_s']:.1f}")
    # T6: thresholds
    out.append("\n## T6 thresholds (bisection, preset base unless stated): value needed for the target tok/s")
    spec, sq = load_model("glm")
    cases = [
        ("glm", "tp_owner", "static", "none", 8, "allreduce_us", 0.0, 300.0),
        ("glm", "tp_owner", "static", "none", 4, "allreduce_us", 0.0, 300.0),
        ("glm", "tp_owner", "static", "none", 8, "vram_eff", 0.3, 1.0),
        ("glm", "tp_owner", "static", "none", 8, "gpu_layer_fixed_us", 0.0, 400.0),
        ("glm", "tp_owner", 0.75, "none", 8, "dram_eff", 0.2, 1.0),
        ("glm", "tp_owner", "uniform", "none", 8, "dram_eff", 0.2, 1.0),
        ("glm", "tp_owner", 0.75, "none", 4, "dram_eff", 0.2, 1.0),
        ("glm", "pp", "static", "none", 8, "vram_eff", 0.3, 1.0),
        ("glm", "pp", "static", "mtp:3:0.745", 8, "vram_eff", 0.3, 1.0),
    ]
    out.append("model | arch | hit | spec | ch | parameter | 50 tok/s | 80 tok/s | 100 tok/s")
    for mname, arch, h, s, ch, key, lo, hi in cases:
        p = resolve_params("base", {"dram_channels": ch, "hit": h, "spec": s})
        cells = []
        for target in (50, 80, 100):
            v, how = solve(spec, sq, p, arch, key, target, lo, hi)
            cells.append(f"{how} {v:.3g}" if v is not None else how)
        out.append(f"{mname} | {arch} | {h} | {s} | {ch} | {key} | " + " | ".join(cells))
    # hit threshold (as a fixed float)
    out.append("\nminimum fixed hit share for the target (preset base):")
    for ch in (8, 4):
        for arch in ("pp", "tp_owner"):
            for s in ("none", "mtp:3:0.745"):
                p = resolve_params("base", {"dram_channels": ch, "spec": s, "miss_union": "additive"})
                cells = []
                for target in (50, 80, 100):
                    v, how = solve(spec, sq, p, arch, "hit", target, 0.0, 1.0)
                    cells.append(f"{how} {v:.3f}" if v is not None else how)
                out.append(f"glm | {arch} | {s} | {ch}ch | hit | " + " | ".join(cells))
    spec, sq = load_model("qwen")
    for arch in ("tp_owner", "pp2_tp2"):
        for s in ("none", "mtp:2:0.55"):
            p = resolve_params("base", {"hit": 1.0, "spec": s})
            cells = []
            for target in (100, 150, 200):
                v, how = solve(spec, sq, p, arch, "allreduce_us", target, 0.0, 300.0)
                cells.append(f"{how} {v:.3g}" if v is not None else how)
            out.append(f"qwen | {arch} | {s} | allreduce_us for 100/150/200 tok/s | " + " | ".join(cells))
    # T7: one-at-a-time sensitivity around preset base (8 channels)
    out.append("\n## T7 one-at-a-time sensitivity, preset base, 8 channels: tok/s per value")
    sweeps = [
        ("vram_eff", [0.55, 0.65, 0.75, 0.85, 0.95]),
        ("dram_channels", [4, 8]),
        ("dram_eff", [0.5, 0.6, 0.7, 0.8, 0.9]),
        ("cpu_compute_gbs", [40, 60, 90, 120, None]),
        ("allreduce_us", [5, 10, 15, 30, 60, 100]),
        ("gpu_layer_fixed_us", [0, 10, 20, 40, 80]),
        ("handoff_us", [2, 5, 10, 20, 40]),
        ("pcie_gbs", [12.5, 25.0]),
        ("step_fixed_us", [0, 300, 1000, 3000]),
    ]
    cases = [("glm", "pp", "static", "none"), ("glm", "tp_owner", "uniform", "none"),
             ("glm", "tp_owner", "static", "none"), ("glm", "tp_owner", "static", "mtp:2:0.745"),
             ("qwen", "pp", 1.0, "none"), ("qwen", "tp_owner", 1.0, "none")]
    out.append("parameter | values | " + " | ".join(f"{m}/{a}/{h}/{s}" for m, a, h, s in cases))
    for key, vals in sweeps:
        cells = []
        for mname, arch, h, s in cases:
            spec, sq = load_model(mname)
            ts = []
            for v in vals:
                pp = resolve_params("base", {"dram_channels": 8, "hit": h, "spec": s, key: v})
                ts.append(f"{step(spec, sq, pp, arch)['tok_s']:.0f}")
            cells.append("/".join(ts))
        out.append(f"{key} | {vals} | " + " | ".join(cells))
    out.append("prefetch (cpu_compute_gbs 60) | recall,precision 0,1 / 0.3,0.9 / 0.6,0.9 / 0.6,0.5 / 0.9,0.9 | "
               + " | ".join(
                   "/".join(f"{step(load_model(mname)[0], load_model(mname)[1], resolve_params('base', {'dram_channels': 8, 'hit': h, 'spec': s, 'cpu_compute_gbs': 60, 'prefetch_recall': r, 'prefetch_precision': pr}), arch)['tok_s']:.0f}"
                            for r, pr in ((0, 1), (0.3, 0.9), (0.6, 0.9), (0.6, 0.5), (0.9, 0.9)))
                   for mname, arch, h, s in cases))
    out.append("hit (fixed) | 0.6/0.7/0.8/0.9/0.95/1.0 | " + " | ".join(
        "/".join(f"{step(load_model(mname)[0], load_model(mname)[1], resolve_params('base', {'dram_channels': 8, 'hit': hv, 'spec': s, 'miss_union': 'additive'}), arch)['tok_s']:.0f}"
                 for hv in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
        for mname, arch, h, s in cases))
    out.append("MTP acceptance (k=2) | 0.5/0.6/0.7/0.745/0.8/0.9 | " + " | ".join(
        "/".join(f"{step(load_model(mname)[0], load_model(mname)[1], resolve_params('base', {'dram_channels': 8, 'hit': h, 'spec': f'mtp:2:{a}'}), arch)['tok_s']:.0f}"
                 for a in (0.5, 0.6, 0.7, 0.745, 0.8, 0.9))
        for mname, arch, h, s in cases if mname == "glm"))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS))
    ap.add_argument("--params", help="JSON file with parameter overrides (applied after the preset)")
    ap.add_argument("--set", action="append", default=[], help="key=value override, dotted keys allowed")
    ap.add_argument("--model", default="glm", help="comma list: glm,qwen,mimo,dsv4f,dsv41 or spec paths")
    ap.add_argument("--arch", default="pp_serial,pp,ep,tp_repl,tp_owner,tp_owner_hostcpu,pp2_tp2")
    ap.add_argument("--hit", default="uniform", help="comma list: uniform, static, static_global, mid, or floats")
    ap.add_argument("--spec", default="none", help="comma list: none, mtp:K:ALPHA, dflash2:K, dflash2_4k:K, "
                                                    "ngram:K:ALPHA")
    ap.add_argument("--sweep", action="append", default=[], help="key=v1,v2,... (repeatable; grid)")
    ap.add_argument("--solve", help="parameter to solve for (needs --target)")
    ap.add_argument("--target", type=float, default=80.0)
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--bounds", action="store_true", help="print physics bounds for each model/hit/spec")
    ap.add_argument("--report", choices=["r9"], help="print every table of the R9 report")
    ap.add_argument("--dump-params", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        print(report_r9())
        return
    over = {}
    if a.params:
        with open(a.params) as f:
            over.update(json.load(f))
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = parse_value(v)
    p = resolve_params(a.preset, over)
    if a.dump_params:
        json.dump(p, sys.stdout, indent=1)
        print()
        return
    models = a.model.split(",")
    archs = a.arch.split(",")
    hits = [parse_value(h) for h in a.hit.split(",")]
    specs = a.spec.split(",")
    if a.solve:
        lo = a.lo if a.lo is not None else 0.0
        hi = a.hi if a.hi is not None else (1.0 if a.solve in ("hit", "vram_eff", "dram_eff") else 300.0)
        for mname in models:
            spec, sq = load_model(mname)
            for arch in archs:
                for h in hits:
                    for s in specs:
                        pp = copy.deepcopy(p)
                        pp["hit"], pp["spec"] = h, s
                        v, how = solve(spec, sq, pp, arch, a.solve, a.target, lo, hi)
                        print(f"{spec['name']} | {arch} | hit {h} | {s} | {a.solve} for {a.target:g} tok/s: "
                              + (f"{how} {v:.4g}" if v is not None else how))
        return
    grids = [{}]
    for sw in a.sweep:
        k, vs = sw.split("=", 1)
        grids = [dict(g, **{k: parse_value(v)}) for g in grids for v in vs.split(",")]
    all_rows = []
    for gset in grids:
        pp = copy.deepcopy(p)
        for k, v in gset.items():
            set_param(pp, k, v)
        rows = run(models, archs, hits, specs, pp)
        for r in rows:
            r["sweep"] = gset
        all_rows += rows
        if not a.json:
            title = "# preset " + a.preset + ("  sweep " + json.dumps(gset) if gset else "")
            print(render(rows, title))
            if a.bounds:
                for mname in models:
                    spec, sq = load_model(mname)
                    for h in hits:
                        for s in specs:
                            b = physics_bounds(spec, sq, dict(pp, hit=h, spec=s))
                            print(f"  bounds {spec['name']} hit {h} {s}: DRAM {b['dram_bound_tok_s']:.1f} | "
                                  f"aggregate VRAM {b['aggregate_vram_bound_tok_s']:.1f} | single-GPU PP "
                                  f"{b['single_gpu_pp_bound_tok_s']:.1f} tok/s")
            print()
    if a.json:
        json.dump(all_rows, sys.stdout, indent=1, default=str)
        print()


if __name__ == "__main__":
    main()
