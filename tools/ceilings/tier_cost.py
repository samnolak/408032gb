#!/usr/bin/env python3
"""tier_cost: decode cost of one MoE step across VRAM, RAM (CPU compute), NVMe and PCIe, with expert prediction,
prefetch, speculation and Engram/PLE table lookups - the per-layer cost model of R8 (docs/reassessment/research/
R8-memory-tiers-expert-scheduling.md section 8) as a tested tool, plus one-at-a-time sensitivity analysis.

Scope next to the other models: tools/roofline.py (placement, whole-model bounds) and tools/ceilings/arch_ceilings.py
(multi-GPU architectures, fixed costs, hit-rate curves) do not model the NVMe tier per layer, expert prediction and
prefetch with their DRAM/PCIe contention, the CPU compute cap per format, or table-lookup latency. This tool does, for a
layer pipeline (one GPU busy at a time, every layer on the critical path), which is what R1 runs first.

Per decode step of n_tok tokens (1 + draft length), per MoE layer (all times in seconds):
  U      distinct experts touched (K for one token; E(1-(1-K/E)^n_tok) for independent routing, or given)
  n_v, n_r, n_s = U h_v, U h_r, U h_s      resident in VRAM / RAM / NVMe
  NVMe   an expert is striped over all drives and staged into RAM, then computed by the CPU; demand reads cost
         n_s S_e / (B_s n_drives) plus one service latency if any read happens; predicted reads (recall p, precision
         pi, issued W early) only cost what overruns W
  PCIe   prefetch (prefetch_to="gpu"): predicted RAM-resident experts are copied into spare VRAM slots; a link lands
         B_p W / (S_e + B_p L_p) experts within W; wasted copies still cost DMA. prefetch_opt (default): the scheduler
         copies the number that minimises the layer time (0 when prefetch does not pay); prefetch_opt=False copies at
         link rate as R8 did. miss_path="pcie": every off-VRAM expert is copied on demand instead of computed by the CPU
  GPU    t_shared + n_gpu S_e / B_v + t_kern
  CPU    max(n_cpu S_e (1 + (m-1) gamma_mt) / (n_cpu_threads b_core), (n_cpu S_e + DMA) / (B_dram eta_dram)) + t_sync
         (m = tokens per distinct expert; DMA bytes of the same interval steal DRAM bandwidth, first order)
  layer  t_pre + max(GPU, CPU-or-copy + NVMe)   (overlap)   or   t_pre + GPU + CPU-or-copy + NVMe   (no overlap)
  step   n_moe layer + t_other;  tok/s = tokens(accept, k) / step, tokens = (1 - a^(k+1)) / (1 - a)

Changes against the R8 scratch implementation (each covered by a test):
  * the "predicted NVMe reads overrunning the lead time" term was charged one NVMe latency even with no prediction
    (max(0, 0 + L_s - W) = L_s at p = 0): R8's "1% on NVMe" row carried 80 us per layer that no read causes;
  * prefetch_opt: with link-rate copying a faster link can be slower (more wasted copies steal DRAM bandwidth;
    found by the x8-vs-x16 property test), which no rational scheduler would do; the ceiling uses the best copy count;
  * all-or-nothing NVMe staging made more recall slower; the scheduler stages the best part of the predicted set
    (exact optimum at 0, the overrun point or all);
  * GPU prefetch takes only RAM-resident experts (an NVMe-resident expert would have to reach RAM first, which this
    model does not chain); predicted NVMe experts are staged into RAM in the background whatever prefetch_to says.
With overlap the result is not monotone in h_v: past the GPU/CPU balance point more VRAM residency makes the GPU the
bottleneck while the CPU idles (GLM, n_tok 2: h_v 0.82 -> 53.5, 0.90 -> 51.1 tok/s). A scheduler holding RAM copies of
resident experts could hand some of them to the CPU (tools/roofline.py decode() note: take the max over h' <= h_v);
this tool does not, so read a sweep's maximum as the ceiling for "at least this residency".
Everything this prints is PROVISIONAL: a bandwidth-and-latency model with stated parameters, every default is to be
replaced by a measurement on the server (R0/K-series). Expected values inside max() can only overstate tok/s.

Usage:
  python3 tools/ceilings/tier_cost.py --model glm --cases r8              # R8's table, corrected
  python3 tools/ceilings/tier_cost.py --model glm --set h_v=0.75 --set h_r=0.25 --set B_dram=102.4e9
  python3 tools/ceilings/tier_cost.py --model glm --sweep h_v=0.5,0.6,0.7,0.8 --sweep B_dram=102.4e9,204.8e9
  python3 tools/ceilings/tier_cost.py --model glm --tornado               # sensitivity around the base point
  python3 tools/ceilings/tier_cost.py --model qwen --set n_tok=4 --set accept=0.8 --json
"""
import argparse
import dataclasses
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import roofline as rl  # noqa: E402  (model geometry, read-only use)


@dataclass
class HW:
    B_v: float = 736e9 * 0.75      # effective VRAM bandwidth for expert GEMV (hardware/4x4080s-32g.json mem_bw_gbs x 0.75)
    t_kern: float = 4e-6           # launch cost of the grouped expert kernel(s); 0 under one CUDA graph (HYPOTHESIS)
    B_dram: float = 204.8e9        # theoretical DRAM: 8 ch DDR4-3200; 102.4e9 with 4 ch (channel count UNKNOWN)
    eta_dram: float = 0.70         # achievable DRAM share for an AVX2 GEMV stream (HYPOTHESIS; STREAM suggests <= 0.85)
    n_cpu: int = 32                # CPU expert threads
    b_core: float = 8e9            # expert-weight bytes/s one Zen 3 core consumes for Q4_K (HYPOTHESIS)
    t_sync: float = 20e-6          # GPU -> CPU hand-off and results back per layer (HYPOTHESIS)
    B_p: float = 25e9              # pinned H2D per GPU link, PCIe 4.0 x16 (REPORTED 23-26 GB/s); 12.5e9 at x8
    L_p: float = 10e-6             # per-copy latency
    n_links_free: int = 1          # links able to deliver prefetch into this layer's GPU (no P2P: 1)
    B_s_drive: float = 7.45e9 * 0.6  # one NVMe, large O_DIRECT reads (vendor 7.45 GB/s x 0.6, PROVISIONAL)
    n_drives: int = 4              # customer spec 2026-10-10: 4 x 1 TB (hardware/4x4080s-32g.json still lists 3)
    L_s: float = 80e-6             # service time of one random read at moderate queue depth in a VM (HYPOTHESIS)
    iops_drive: float = 1.0e6      # 4 KiB random-read IOPS per drive at high QD (vendor 1.2M at QD32, PROVISIONAL)
    gamma_mt: float = 0.2          # extra CPU compute per extra token sharing an expert (Strata, REPORTED 2.0-2.4x/3)


@dataclass
class Layer:
    S_e: float = 15.38e6           # bytes of one routed expert (gate+up+down)
    K: int = 8                     # experts per token
    E: int = 288                   # routed experts per layer
    t_pre: float = 0.30e-3         # serial GPU work of the layer before the experts (attention, norms, router, KV)
    t_shared: float = 0.045e-3     # GPU work that can overlap the CPU experts (shared expert)


@dataclass
class Policy:
    h_v: float = 0.61              # share of the distinct routed experts resident in VRAM
    h_r: float = 0.39              # share in RAM; the rest (1 - h_v - h_r) is on NVMe
    miss_path: str = "cpu"         # "cpu": the CPU computes RAM-resident misses; "pcie": copy them to the GPU first
    overlap: bool = True           # off-VRAM expert work overlaps the GPU's resident experts (Strata, strata-glm)
    p_recall: float = 0.0          # share of off-VRAM experts predicted W seconds early
    pi_prec: float = 1.0           # precision of the predicted set
    W: float = 0.0                 # prediction lead time (one layer ahead ~ one layer's time)
    prefetch_to: str = "gpu"       # "gpu": PCIe copies of predicted RAM experts; "ram": only NVMe -> RAM staging
    prefetch_opt: bool = True      # the scheduler copies the number of predicted experts that minimises the layer time
                                   # and stages the best part of the predicted NVMe set (False: all of it, as R8 did)
    n_tok: int = 1                 # tokens verified per step (1 + draft length)
    accept: float = 0.0            # per-position draft acceptance
    U: float = None                # distinct experts in the step; None = independent-routing estimate


FIELDS = {f.name: cls for cls in (HW, Layer, Policy) for f in dataclasses.fields(cls)}


def distinct(E, K, n):
    return E * (1 - (1 - K / E) ** n)


def tokens_per_step(accept, k):
    if k <= 0:
        return 1.0
    if accept >= 1.0:
        return k + 1.0
    return (1 - accept ** (k + 1)) / (1 - accept)


def layer_time(hw, ly, po):
    """Seconds for one MoE layer of one decode step, and the parts."""
    if po.h_v < 0 or po.h_r < 0 or po.h_v + po.h_r > 1 + 1e-9:
        raise ValueError(f"need h_v, h_r >= 0 and h_v + h_r <= 1 (got {po.h_v}, {po.h_r})")
    if po.miss_path not in ("cpu", "pcie"):
        raise ValueError(f"miss_path must be cpu or pcie (got {po.miss_path!r})")
    U = po.U if po.U is not None else (ly.K if po.n_tok == 1 else distinct(ly.E, ly.K, po.n_tok))
    h_s = max(0.0, 1.0 - po.h_v - po.h_r)
    h_s = h_s if h_s > 1e-12 else 0.0
    n_v, n_r, n_s = U * po.h_v, U * po.h_r, U * h_s
    predicting = po.p_recall > 0 and po.W > 0
    # NVMe: demand reads, then background reads of the predicted set that overrun the lead time
    t_ssd = 0.0
    if n_s > 0:
        per = ly.S_e / (hw.B_s_drive * hw.n_drives)
        pred_s = po.p_recall * n_s if predicting else 0.0
        waste_per = (1 / po.pi_prec - 1) if po.pi_prec > 0 else 0.0

        def ssd(x):                             # x of the predicted NVMe experts staged in the background
            unpred = n_s - x
            t = unpred * per + (hw.L_s * (1 - (1 - unpred / U) ** U) if unpred > 0 else 0.0)
            if x > 0:
                t += max(0.0, x * (1 + waste_per) * per + hw.L_s - po.W)
            return t
        t_ssd = ssd(pred_s)
        if po.prefetch_opt and pred_s > 0:
            # the scheduler stages the best part x of the predicted set: ssd(x) is concave on (0, knee] and on
            # [knee, pred_s] (knee: background reads start to overrun W), so the minimum is at 0, the knee or pred_s
            knee = (po.W - hw.L_s) / ((1 + waste_per) * per)
            t_ssd = min(ssd(x) for x in (0.0, min(max(knee, 0.0), pred_s), pred_s))

    def parts(copies, useful):
        """Layer time when `copies` predicted RAM experts are copied to the GPU, `useful` of them used."""
        dma = copies * ly.S_e
        n_gpu, n_cpu = n_v + useful, n_r + n_s - useful
        t_copy = 0.0
        if po.miss_path == "pcie":
            t_copy = n_cpu * (ly.S_e / hw.B_p + hw.L_p)
            dma += n_cpu * ly.S_e
            n_gpu, n_cpu = n_gpu + n_cpu, 0.0
        t_gpu = ly.t_shared + n_gpu * ly.S_e / hw.B_v + hw.t_kern
        t_cpu = 0.0
        if n_cpu > 1e-12:
            m = po.n_tok * ly.K / max(U, 1e-12)
            b_cmp = hw.n_cpu * hw.b_core / (1 + (m - 1) * hw.gamma_mt)
            bytes_cpu = n_cpu * ly.S_e
            t_cpu = max(bytes_cpu / b_cmp, (bytes_cpu + dma) / (hw.B_dram * hw.eta_dram)) + hw.t_sync
        off = max(t_cpu, t_copy) + t_ssd
        t = ly.t_pre + (max(t_gpu, off) if po.overlap else t_gpu + off)
        return t, {"U": U, "n_v": n_v, "n_gpu": n_gpu, "n_cpu": n_cpu, "t_gpu": t_gpu, "t_cpu": t_cpu, "t_ssd": t_ssd,
                   "t_copy": t_copy, "dma_bytes": dma, "copies": copies}

    if not (po.prefetch_to == "gpu" and predicting and n_r > 0):
        return parts(0.0, 0.0)
    n_pred = po.p_recall * n_r
    cap = hw.n_links_free * hw.B_p * po.W / (ly.S_e + hw.B_p * hw.L_p)    # experts a link can land within W
    if not po.prefetch_opt:                                                  # R8: copy at link rate
        n_wasted = n_pred * (1 / po.pi_prec - 1) if po.pi_prec > 0 else math.inf
        return parts(min(n_pred + n_wasted, cap), min(n_pred, cap * po.pi_prec))
    # the scheduler picks c copies in [0, min(cap, n_pred / pi)] (beyond that nothing more is useful); useful = c pi,
    # DMA = c S_e: every term is linear in c, the layer time a max/sum of them - convex, so a ternary search is exact
    hi = min(cap, n_pred / po.pi_prec) if po.pi_prec > 0 else 0.0
    f = lambda c: parts(c, c * po.pi_prec)[0]   # noqa: E731
    lo_c, hi_c = 0.0, hi
    for _ in range(100):
        a, b = lo_c + (hi_c - lo_c) / 3, hi_c - (hi_c - lo_c) / 3
        if f(a) <= f(b):
            hi_c = b
        else:
            lo_c = a
    best = min((0.0, hi, (lo_c + hi_c) / 2), key=f)
    return parts(best, best * po.pi_prec)


def step(hw, ly, po, n_moe, t_other):
    t_l, info = layer_time(hw, ly, po)
    t_step = n_moe * t_l + t_other
    tok = tokens_per_step(po.accept, po.n_tok - 1)
    return {"tok_s": tok / t_step, "step_ms": t_step * 1e3, "tokens_per_step": tok, **info}


def lookup_exposed(hw, rows, W_lookup, page=4096, cache_hit=0.0):
    """Exposed latency of `rows` table rows (Engram / PLE) read as `page`-byte random reads over all drives, issued
    W_lookup seconds before they are needed; cache_hit = share served from a RAM row cache."""
    n = rows * (1 - cache_hit)
    if n <= 0:
        return 0.0
    t = hw.L_s + n / (hw.iops_drive * hw.n_drives) + n * page / (hw.B_s_drive * hw.n_drives)
    return max(0.0, t - W_lookup)


# ------------------------------------------------------------------------------------------------- model presets

MODELS = {   # spec file, (expert bpw, dense bpw) of the UD-Q4_K_XL-class files used in tools/ceilings/arch_ceilings.py
    "glm": ("glm-5.3-flash.json", 4.89, 8.04),
    "qwen": ("qwen3.8-flash-next.json", 5.10, 10.0),
    "mimo": ("mimo-v2.6-flash.json", 4.25, 8.5),
    "dsv41": ("deepseek-v4.1-flash.json", 4.25, 8.5),
}


def preset(model, hw=None, ctx=262144, kv_bytes=2.0, t_fixed=0.45e-3):
    """Layer, n_moe and t_other from models/specs/<model>.json. Dense (non-routed) bytes and the KV/state read of a
    `ctx`-token context are spread evenly over the MoE layers (all on the critical path of a pipeline, so the step sum
    is the same); the shared experts go to t_shared, the LM head to t_other; t_fixed = sampling, embeddings and
    pipeline hops (HYPOTHESIS: R8's GLM t_other 1.6 ms minus its LM head). Specs that do not itemise `shared_experts`
    / `lm_head` (qwen, dsv41: aggregate keys) keep them in t_pre: same step sum, no overlap credit (conservative)."""
    hw = hw or HW()
    name, ebpw, dbpw = MODELS[model]
    with open(os.path.join(ROOT, "models", "specs", name)) as f:
        spec = json.load(f)
    m = spec["moe"]
    n_moe = m["layers"]
    S_e = rl.expert_params(spec) * ebpw / 8
    d = spec["dense_params"]
    shared = d.get("shared_experts", 0) * dbpw / 8
    head = d.get("lm_head", 0) * dbpw / 8
    dense = rl.dense_active_params(spec) * dbpw / 8 - shared - head
    kv = rl.kv_read_bytes(spec, ctx, kv_bytes)
    ly = Layer(S_e=S_e, K=m["top_k"], E=m["n_experts"], t_pre=(dense + kv) / hw.B_v / n_moe,
               t_shared=shared / hw.B_v / n_moe)
    return ly, n_moe, head / hw.B_v + t_fixed, spec["name"]


# --------------------------------------------------------------------------------------------------------- CLI

def parse_value(key, s):
    cls = FIELDS.get(key)
    if cls is None:
        raise SystemExit(f"unknown parameter {key!r}; known: {', '.join(sorted(FIELDS))}")
    t = {f.name: f.type for f in dataclasses.fields(cls)}[key]
    if t in (bool, "bool"):
        return s.lower() in ("1", "true", "yes", "on")
    if t in (str, "str"):
        return s
    if key == "U" and s.lower() == "none":
        return None
    v = float(s)
    return int(v) if t in (int, "int") else v


def apply(hw, ly, po, key, value):
    cls = FIELDS[key]
    if cls is HW:
        return replace(hw, **{key: value}), ly, po
    if cls is Layer:
        return hw, replace(ly, **{key: value}), po
    return hw, ly, replace(po, **{key: value})


def with_ram_split(po, h_v):
    """Set h_v and keep the NVMe share: the RAM share takes the difference."""
    h_s = max(0.0, 1.0 - po.h_v - po.h_r)
    return replace(po, h_v=h_v, h_r=max(0.0, 1.0 - h_v - h_s))


R8_CASES = [   # (label, policy changes) - the rows of R8 section 8.3; W = 0.44 ms ~ one GLM layer at 50 tok/s
    ("all VRAM", dict(h_v=1.0, h_r=0.0)),
    ("h_v .61 CPU misses overlap", {}),
    ("h_v .61 CPU misses no overlap", dict(overlap=False)),
    ("h_v .61 PCIe misses", dict(miss_path="pcie")),
    ("h_v .82 CPU overlap", dict(h_v=0.82, h_r=0.18)),
    ("h_v .61 + p=.85 next-layer prefetch to GPU", dict(p_recall=0.85, pi_prec=0.5, W=0.44e-3, prefetch_opt=False)),
    ("h_v .61 + p=.85 prefetch, W=3 layers", dict(p_recall=0.85, pi_prec=0.5, W=1.3e-3, prefetch_opt=False)),
    ("  same, scheduler copies the best number", dict(p_recall=0.85, pi_prec=0.5, W=1.3e-3)),
    ("MTP k=1 (n_tok 2), h_v .61", dict(n_tok=2)),
    ("h_v .61, 1% on NVMe", dict(h_r=0.38)),
    ("h_v .61, 5% on NVMe", dict(h_r=0.34)),
    ("h_v .61, 5% NVMe, p=.5 W=1 layer to RAM", dict(h_r=0.34, p_recall=0.5, pi_prec=0.5, W=0.44e-3, prefetch_to="ram")),
]

TORNADO = [   # (parameter, low, high) around the base point; ranges are the plausible spans named in R8 / ROOFLINE
    ("B_dram", 102.4e9, 204.8e9),       # 4 or 8 populated channels (UNKNOWN until R0)
    ("eta_dram", 0.55, 0.85),
    ("b_core", 5e9, 12e9),              # i-quant vs Q4_K-class CPU kernels
    ("n_cpu", 16, 46),
    ("h_v", 0.45, 0.82),                # uniform-routing residency .. adaptive cache on skewed routing
    ("t_sync", 5e-6, 60e-6),
    ("B_v", 736e9 * 0.6, 736e9 * 0.85),
    ("t_pre", None, None),              # x0.8 .. x1.25 of the preset (attention/KV kernels)
    ("accept", 0.55, 0.85),             # with n_tok = 2 (one MTP draft) in the tornado base
    ("B_p", 12.5e9, 25e9),              # x8 vs x16 (only matters with prefetch or PCIe misses)
    ("n_drives", 3, 4),                 # only matters with an NVMe share
    ("p_recall", 0.3, 0.95),            # only matters with prediction on (W > 0)
    ("pi_prec", 0.2, 0.9),
    ("L_p", 5e-6, 30e-6),
]


def run_case(hw, ly, po, n_moe, t_other):
    r = step(hw, ly, po, n_moe, t_other)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}


def tornado(hw, ly, po, n_moe, t_other):
    base = step(hw, ly, po, n_moe, t_other)["tok_s"]
    rows = []
    for key, lo, hi in TORNADO:
        if key == "t_pre":
            lo, hi = ly.t_pre * 0.8, ly.t_pre * 1.25
        out = []
        for v in (lo, hi):
            if key == "h_v":
                h2, l2, p2 = hw, ly, with_ram_split(po, v)
            else:
                h2, l2, p2 = apply(hw, ly, po, key, int(v) if key in ("n_cpu", "n_drives") else v)
            out.append(step(h2, l2, p2, n_moe, t_other)["tok_s"])
        rows.append({"param": key, "low": lo, "high": hi, "tok_s_low": out[0], "tok_s_high": out[1],
                     "swing": abs(out[1] - out[0])})
    rows.sort(key=lambda r: -r["swing"])
    return base, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glm", choices=sorted(MODELS))
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--kv-bytes", type=float, default=2.0, help="bytes per KV element (2 = f16, 1 = q8)")
    ap.add_argument("--t-fixed", type=float, default=0.45e-3)
    ap.add_argument("--r8-layer", action="store_true", help="use R8's literal GLM layer constants instead of the preset")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2,...")
    ap.add_argument("--cases", choices=["r8"])
    ap.add_argument("--tornado", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    hw = HW()
    ly, n_moe, t_other, mname = preset(a.model, hw, a.ctx, a.kv_bytes, a.t_fixed)
    if a.r8_layer:
        ly, n_moe, t_other = Layer(), 42, 1.6e-3
    po = Policy()
    for kv in a.set:
        k, _, v = kv.partition("=")
        hw, ly, po = apply(hw, ly, po, k, parse_value(k, v))
    head = (f"# {mname}: n_moe {n_moe}, S_e {ly.S_e / 1e6:.2f} MB, K {ly.K}, E {ly.E}, t_pre {ly.t_pre * 1e3:.3f} ms, "
            f"t_shared {ly.t_shared * 1e3:.3f} ms, t_other {t_other * 1e3:.2f} ms, ctx {a.ctx}; PROVISIONAL model output")
    rows = []
    if a.cases == "r8":
        for label, ch in R8_CASES:
            for chan, B in (("8ch", 204.8e9), ("4ch", 102.4e9)):
                p2 = replace(po, accept=0.7 if ch.get("n_tok", po.n_tok) > 1 else po.accept, **ch)
                rows.append({"case": label, "dram": chan, **run_case(replace(hw, B_dram=B), ly, p2, n_moe, t_other)})
    elif a.tornado:
        po_t = replace(po, n_tok=max(po.n_tok, 2), accept=po.accept or 0.7)
        base, trows = tornado(hw, ly, po_t, n_moe, t_other)
        if a.json:
            print(json.dumps({"header": head, "base_tok_s": base, "base_policy": dataclasses.asdict(po_t),
                              "rows": trows}, indent=1))
            return 0
        print(head)
        print(f"# tornado around: {dataclasses.asdict(po_t)}  -> base {base:.1f} tok/s")
        print("param | low | high | tok/s at low | tok/s at high | swing")
        for r in trows:
            print(f"{r['param']} | {r['low']:.4g} | {r['high']:.4g} | {r['tok_s_low']:.1f} | {r['tok_s_high']:.1f} | "
                  f"{r['swing']:.1f}")
        return 0
    else:
        grids = []
        for sw in a.sweep:
            k, _, vs = sw.partition("=")
            grids.append([(k, parse_value(k, v)) for v in vs.split(",")])
        for combo in itertools.product(*grids) if grids else [()]:
            h2, l2, p2 = hw, ly, po
            for k, v in combo:
                if k == "h_v":
                    p2 = with_ram_split(p2, v)
                else:
                    h2, l2, p2 = apply(h2, l2, p2, k, v)
            rows.append({**{k: v for k, v in combo}, **run_case(h2, l2, p2, n_moe, t_other)})
    if a.json:
        print(json.dumps({"header": head, "rows": rows}, indent=1))
        return 0
    print(head)
    cols = [c for c in rows[0] if c not in ("n_v",)]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
