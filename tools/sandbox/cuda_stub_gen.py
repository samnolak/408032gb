#!/usr/bin/env python3
"""Sandbox helper: make the PyPI torch wheel importable on a CPU-only machine without NVIDIA libraries.

The only x86_64 torch wheel on PyPI is the CUDA build; its libtorch_cuda.so needs ~3 GB of NVIDIA
libraries just to be loaded. The agent sandbox has no GPU, little disk and cannot reach
download.pytorch.org (CPU wheels). This script builds tiny stub libraries with the right sonames,
symbol names and version nodes; every stubbed function returns a non-zero error code and is never
used for computation (torch.cuda.is_available() is False).

    pip download torch --no-deps -d /tmp/tw --only-binary=:all: && pip install --no-deps /tmp/tw/torch-*.whl
    pip install filelock typing-extensions sympy networkx jinja2 fsspec tokenizers
    mkdir -p ~/cudastub && cd ~/cudastub && python3 <repo>/tools/sandbox/cuda_stub_gen.py
    LD_LIBRARY_PATH=~/cudastub python3 tools/ds41/run_reference.py --mode reference --out /tmp/f.npz

CONFIRMED 2026-10-02 with torch 2.14.1+cu130, Python 3.12 (docs/evidence/2026-10-02-deepseek-ref-cpu.md).
CI does not need this: runners install the CPU wheel from download.pytorch.org.
"""
import subprocess, re, os, glob, collections
import importlib.util
TL = os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin), "lib")
libs = [os.path.join(TL, f) for f in os.listdir(TL) if f.endswith(".so") or ".so." in f]
def nm(path, flag):
    out = subprocess.run(["nm", "-D", flag, path], capture_output=True, text=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    if flag == "--undefined-only":
        return [r[-1] for r in rows if r[-2] == "U"]          # strong references only, not weak
    return [r[-1] for r in rows]
needed = collections.OrderedDict()
for p in libs:
    out = subprocess.run(["readelf", "-d", p], capture_output=True, text=True).stdout
    for m in re.finditer(r"NEEDED.*\[(.*?)\]", out):
        n = m.group(1)
        if re.match(r"lib(cu|nv|nccl)", n) and n != "libcuda.so.1":
            needed[n] = 1
needed["libcuda.so.1"] = 1
undef = set()
for p in libs:
    undef.update(nm(p, "--undefined-only"))
defined = set()
sysl = sum([glob.glob(g) for g in ["/lib/x86_64-linux-gnu/libc.so.6", "/lib/x86_64-linux-gnu/libm.so.6",
       "/lib/x86_64-linux-gnu/libstdc++.so.6", "/lib/x86_64-linux-gnu/libgcc_s.so.1", "/lib/x86_64-linux-gnu/libpthread.so.0",
       "/lib/x86_64-linux-gnu/libdl.so.2", "/lib/x86_64-linux-gnu/librt.so.1", "/lib64/ld-linux-x86-64.so.2",
       "/usr/lib/x86_64-linux-gnu/libpython3.12.so*"]], [])
for p in libs + sysl:
    for s in nm(p, "--defined-only"):
        defined.add(s.split("@")[0])
py = set(l.split()[-1].split("@")[0] for l in subprocess.run(["nm", "-D", "--defined-only", "/usr/bin/python3.12"], capture_output=True, text=True).stdout.splitlines() if l.strip())
defined |= py
per = collections.defaultdict(set)
vernode = {}
common = set()
for s in undef:
    name, _, ver = s.partition("@")
    if name in defined:
        continue
    if ver in needed:
        per[ver].add(name)
    elif ver == "":
        common.add(name)
    elif ver == "NVSHMEM":
        per["libnvshmem_host.so.3"].add(name); vernode["libnvshmem_host.so.3"] = "NVSHMEM"
    else:
        print("skip versioned non-cuda:", s)
print("needed:", list(needed))
print("unversioned missing:", len(common), sorted(common)[:12])
for n in needed:
    syms = sorted(per.get(n, set()) | (common if n not in per else set()))
    base = n.replace(".", "_")
    with open(base + ".c", "w") as f:
        for s in sorted(per.get(n, set()) | common):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
                f.write(f"int {s}(void) {{ return 3; }}\n")
    with open(base + ".map", "w") as f:
        f.write(f"{vernode.get(n, n)} {{ global: *; }};\n")
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-w", "-o", n, base + ".c", f"-Wl,-soname,{n}", f"-Wl,--version-script={base}.map"], capture_output=True, text=True)
    print(n, len(per.get(n, set())), "versioned syms", "OK" if r.returncode == 0 else r.stderr[:300])
