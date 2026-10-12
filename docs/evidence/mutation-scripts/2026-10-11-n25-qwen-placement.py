# Byte placement of Qwen3.8-Flash-Next at 262,144 tokens (arch_ceilings.placement, preset "base"); run from the repo root.
# Body = scratchpad/archalt/placement.py of docs/reassessment/ARCHITECTURE-ALTERNATIVES.md (research-landscape 2c64027,
# lines 1500-1517): its two Qwen rows (5.1/10/2.0, 4.25/10/1.0; AA:1502-1505) plus a third Qwen row 5.1/10/1.0 that AA
# does not have; prints the expert/dense bpw, KV bytes per element and placement()'s whole dict.
import sys; sys.path.insert(0, "tools/ceilings"); import arch_ceilings as ac
for eb, db, kvb in ((5.1, 10.0, 2.0), (5.1, 10.0, 1.0), (4.25, 10.0, 1.0)):
    s, _ = ac.load_model("qwen")
    p = ac.resolve_params("base", {"quant": {"expert_bpw": eb, "dense_bpw": db}, "kv_bytes_per_elem": kvb})
    pl = ac.placement(s, p, ac.quant_of({}, p), 1)
    print(eb, db, kvb, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pl.items()})
