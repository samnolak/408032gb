# Capacity check for 3 x 1 TB NVMe (PROVISIONAL arithmetic; sources in the evidence file table).
GiB = 2**30
raw = 3 * 1e12                      # 1 TB = 1e12 B per drive (vendor decimal TB)
rows = [  # (item, bytes, required?)
    ("GLM-5.3-Flash UD-Q4_K_XL GGUF", 185.983 * GiB, True),
    ("GLM base-A NVFP4 pack (derived)", 171e9, True),
    ("Qwen3.8-Flash-Next UD-Q4_K_XL GGUF", 111334654784, True),
    ("Qwen MTP Q8_0 + shared Q8_0", (4.14 + 2.79) * 1e9, True),
    ("MiMo-V2.6-Flash MXFP4 (safetensors)", 173e9, True),
    ("MiMo-V2.6-Flash MXFP4 GGUF (converted, ~same size)", 173e9, True),
    ("DeepSeek-V4.1-Flash native MXFP4+MXFP8 incl. Engram", 476 * GiB, True),
    ("DeepSeek-V4-Flash-0731 UD-Q4_K_XL (optional, R1.3 order)", 155.1e9, False),
    ("Q8_0 reference for R2.1, largest one at a time (GLM 320B x 8.5 bpw)", 320e9 * 8.5 / 8, False),
]
req = sum(b for _, b, r in rows if r)
opt = sum(b for _, b, r in rows if not r)
for name, b, r in rows:
    print(f"{name:72s} {b / 1e9:8.1f} GB {b / GiB:8.1f} GiB {'required' if r else 'optional'}")
print(f"{'raw capacity 3 x 1 TB':72s} {raw / 1e9:8.1f} GB {raw / GiB:8.1f} GiB")
print(f"{'required weights total':72s} {req / 1e9:8.1f} GB {req / GiB:8.1f} GiB  ({req / raw:.0%} of raw)")
print(f"{'required + optional':72s} {(req + opt) / 1e9:8.1f} GB {(req + opt) / GiB:8.1f} GiB  ({(req + opt) / raw:.0%} of raw)")
fs = 0.95  # filesystem + reserved-blocks allowance (assumption)
usable = raw * fs
for label, used in (("required", req), ("required + optional", req + opt)):
    print(f"headroom after {label} at {fs:.0%} usable: {(usable - used) / 1e9:.1f} GB ({(usable - used) / GiB:.1f} GiB)")
slot = 8e9  # upper per-slot state size: llama.cpp layout up to ~8 GB at 262K (R8:117); ROOFLINE KV+state 0.3-7.0 GiB
print(f"262K slot states at <= 8 GB each that fit in the headroom after required+optional: {int((usable - req - opt) // slot)}")
# two-drive model volume (placement option A) vs one drive for states/logs
two = 2e12 * fs
print(f"option A: models on 2 drives ({two / 1e9:.0f} GB usable) - required {req / 1e9:.0f} GB, +optional {(req + opt) / 1e9:.0f} GB -> "
      f"free {(two - req) / 1e9:.0f} / {(two - req - opt) / 1e9:.0f} GB; drive 3 ({1e12 * fs / 1e9:.0f} GB) for states/logs/scratch: {int(1e12 * fs // slot)} slots of 8 GB")
for n in (1, 2, 3):
    bw = n * 7.0e9 * 0.6
    print(f"cold load of GLM GGUF (199.7 GB) from {n} drive(s) at {bw / 1e9:.1f} GB/s: {185.983 * GiB / bw:.0f} s")
