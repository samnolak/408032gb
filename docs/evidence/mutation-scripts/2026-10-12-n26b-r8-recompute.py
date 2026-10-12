# Recompute R8's drive-count-dependent numbers with R8's own per-drive constants (7.0 GB/s x 0.6 = 4.2 GB/s,
# vendor 7.45 GB/s for the theoretical line, L_s 80 us, iops 1.0e6, 4 KiB pages), for 4 (as in R8) and 3 drives.
import sys
sys.path.insert(0, "tools/ceilings")
from dataclasses import replace
import tier_cost as tc
per_drive = 7.0e9 * 0.6
S = {"GLM": 3663e-6 * per_drive, "Qwen": 746e-6 * per_drive, "DS-V4.1": 4476e-6 * per_drive}  # bytes, back-solved from R8's 1-drive row
for n in (4, 3):
    agg = n * per_drive
    print(f"n_drives={n}: theoretical {n * 7.45:.2f} GB/s; at R8's 7.0x0.6: {agg / 1e9:.1f} GB/s; at 7.45x0.6 (tier_cost): {n * 7.45 * 0.6:.2f} GB/s")
    print("  striped expert read: " + ", ".join(f"{k} {v / 1e6:.2f} MB -> {v / agg * 1e6:,.0f} us" for k, v in S.items()))
    print(f"  KV restore 3.75 GB -> {3.75e9 / agg:.2f} s, 8 GB -> {8e9 / agg:.2f} s; 150 MB checkpoint -> {150e6 / agg * 1e3:.1f} ms")
    hw = replace(tc.HW(), B_s_drive=per_drive, n_drives=n)
    print(f"  Engram 48 cold 4 KiB reads, lead 0: {tc.lookup_exposed(hw, 48, 0.0) * 1e6:.1f} us exposed; lead 0.25 ms: {tc.lookup_exposed(hw, 48, 0.25e-3) * 1e6:.1f} us")
    print(f"  GLM 5% of routed bytes need 12.9 / 25.8 GB/s at 50 / 100 tok/s vs {agg / 1e9:.1f} GB/s aggregate")
for n in (4, 3):
    hw = replace(tc.HW(), B_s_drive=per_drive, n_drives=n)
    print(f"n_drives={n}: Qwen PLE 16 cold 4 KiB reads, lead 0: {tc.lookup_exposed(hw, 16, 0.0) * 1e6:.1f} us exposed")
