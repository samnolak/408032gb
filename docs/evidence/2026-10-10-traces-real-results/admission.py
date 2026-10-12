"""Admission traffic of a cache that admits every miss (cachesim lru, lfu, the LRU half of hybrid), GLM-5.3-Flash geometry:
k = 8 routed experts per token, 42 MoE layers (models/specs/glm-5.3-flash.json), S_e = 15.38 MB per expert (R8 section 2,
tools/ceilings/tier_cost.py Layer.S_e). Misses per layer per token = k (1 - hit); each admitted miss is one H2D copy and one
extra DRAM read of S_e bytes (the CPU reads the same bytes to compute the miss in place). 30 and 40 tok/s are inputs (the
range of the tier_cost ceilings in this evidence), not measurements."""
k, L, S = 8, 42, 15.38e6
print("hit   misses/layer/token  copies/token  GB/token  GB/s@30tok/s  GB/s@40tok/s  per link (4 links) @40")
for h in (0.75, 0.77, 0.86, 0.90, 0.908, 0.95):
    m = k * (1 - h)
    gb = m * L * S / 1e9
    print(f"{h:5.3f}  {m:18.2f}  {m * L:12.1f}  {gb:8.3f}  {gb * 30:12.1f}  {gb * 40:12.1f}  {gb * 40 / 4:9.1f}")
