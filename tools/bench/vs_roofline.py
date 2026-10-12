#!/usr/bin/env python3
"""Measured single-stream decode (tools/bench/run_matrix.py run dirs) against the roofline bound (tools/roofline.py).

For every single-stream cell of a run: the bound is computed at the cell's mean decode context (prompt + output/2) with
the given model spec, quantisation, VRAM hit share, strategy, overlap mode and speculation; the table shows measured p50 /
bound. Describe the engine as it ran: an engine that speculates (e.g. Strata serve with MTP) is compared with --spec-k/
--spec-accept, llama.cpp without CPU/GPU overlap inside a layer with --overlap no (docs/reassessment/ROOFLINE.md, review
item 3). With overlap the bound is the envelope over hit' <= hit (tools/roofline.py envelope()); "bound_at_hit" is the value
at the given hit. A ratio above 1 says the inputs (hit share, bandwidths, bpw, speculation) are wrong for this configuration -
fix the inputs, not the result. There is no calibrated "good" ratio yet: the only calibration (ROOFLINE.md section 1) is from
another host and engine and must not be used as a threshold; calibrate on the server (R0 membw + R1) first.
Every JSON row records the bound's inputs.

Usage:
  python3 tools/bench/vs_roofline.py RUN_DIR [RUN_DIR...] --spec models/specs/glm-5.3-flash.json \
      --expert-bpw 4.9 --dense-bpw 8.0 --hit 0.66 --strategy pp [--hw hardware/target-4x4080-32g.unverified.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import roofline  # noqa: E402  (tools/roofline.py)


def rows_for(run_dir, spec, hw, q, hit, strategy, miss, overlap=True, spec_k=0, spec_accept=0.0, draft_gb=None,
             inputs=None):
    with open(os.path.join(run_dir, "summary.json")) as f:
        summ = json.load(f)
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    out = []
    for c in summ.get("cells", []):
        if c.get("concurrency") != 1:
            continue
        m = c.get("metrics", {})
        dec = (m.get("decode_tok_s") or {}).get("p50")
        prompt = (m.get("prompt_tokens") or {}).get("p50") or c.get("input")
        gen = (m.get("completion_tokens") or {}).get("p50") or c.get("output") or 0
        ctx = int((prompt or 0) + gen / 2)
        at_hit = roofline.decode(spec, hw, q, ctx, hit, strategy=strategy, miss=miss, spec_k=spec_k,
                                 spec_accept=spec_accept, overlap=overlap, draft_gb=draft_gb)["tok_s"]
        env, env_hit = roofline.envelope(spec, hw, q, ctx, hit, None, strategy, miss, spec_k, spec_accept, None, overlap,
                                         draft_gb)
        bound = max(env, at_hit)
        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "mock": bool(meta.get("is_mock_server")),
                    "input": c.get("input"), "output": c.get("output"), "state": c.get("state"), "ctx": ctx,
                    "measured_decode_p50": dec, "bound": bound, "ratio": (dec / bound) if dec else None,
                    "bound_at_hit": at_hit, "bound_hit": env_hit if env > at_hit else hit,
                    "engine": meta.get("engine_desc"), "model": meta.get("model_desc"), "inputs": inputs})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--hw", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                  "hardware", "target-4x4080-32g.unverified.json"))
    ap.add_argument("--expert-bpw", type=float, default=4.5)
    ap.add_argument("--dense-bpw", type=float, default=8.5)
    ap.add_argument("--kv-bytes", type=float, default=2.0)
    ap.add_argument("--hit", type=float, required=True, help="VRAM hit share of routed-expert reads in this configuration")
    ap.add_argument("--strategy", default="pp", choices=["pp", "ep", "tp"])
    ap.add_argument("--miss", default="cpu", choices=["cpu", "pcie"])
    ap.add_argument("--overlap", choices=["yes", "no"], default="yes",
                    help="CPU/GPU overlap inside a layer: yes (Strata, strata-glm --cpu-share), no (llama.cpp)")
    ap.add_argument("--spec-k", type=int, default=0, help="drafted tokens per step of the engine (0 = no speculation)")
    ap.add_argument("--spec-accept", type=float, default=0.0, help="per-token acceptance of the drafts")
    ap.add_argument("--draft-gb", type=float, default=None, help="GB read per drafted token (overrides the spec)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    with open(a.spec) as f:
        spec = json.load(f)
    with open(a.hw) as f:
        hw = json.load(f)
    q = {"expert_bpw": a.expert_bpw, "dense_bpw": a.dense_bpw, "kv_bytes_per_elem": a.kv_bytes}
    inputs = {"spec": os.path.basename(a.spec), "hw": os.path.basename(a.hw), "expert_bpw": a.expert_bpw,
              "dense_bpw": a.dense_bpw, "kv_bytes": a.kv_bytes, "hit": a.hit, "strategy": a.strategy, "miss": a.miss,
              "overlap": a.overlap, "spec_k": a.spec_k, "spec_accept": a.spec_accept, "draft_gb": a.draft_gb}
    rows = []
    for r in a.runs:
        rows += rows_for(r, spec, hw, q, a.hit, a.strategy, a.miss, a.overlap == "yes", a.spec_k, a.spec_accept,
                         a.draft_gb, inputs)
    if a.json:
        json.dump(rows, sys.stdout, indent=1)
        print()
        return 0
    print(f"# {spec['name']}: measured single-stream decode vs roofline ({a.strategy}/{a.miss}, hit {a.hit}, overlap "
          f"{a.overlap}, spec k={a.spec_k} accept {a.spec_accept}, experts {a.expert_bpw} bpw, dense {a.dense_bpw} bpw); "
          f"bound is PROVISIONAL (envelope over hit' <= hit), measured is what the run recorded")
    print("| run | mock | input | output | state | ctx | measured p50 tok/s | bound tok/s | ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md = f"{r['measured_decode_p50']:.1f}" if r["measured_decode_p50"] else "n/a"
        ra = f"{r['ratio']:.2f}" if r["ratio"] else "n/a"
        print(f"| {r['run']} | {'MOCK' if r['mock'] else ''} | {r['input']} | {r['output']} | {r['state']} | {r['ctx']} | "
              f"{md} | {r['bound']:.1f} | {ra} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
