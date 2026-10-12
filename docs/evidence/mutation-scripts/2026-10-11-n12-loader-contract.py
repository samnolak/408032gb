#!/usr/bin/env python3
"""N12 probe (not a mutation check): the GGUF metadata contract of a glm5next file, read from the sources.
For ik_llama.cpp: the generic hparams reads (src/llama-hparams.cpp:140-245) and the LLM_ARCH_GLM5NEXT block
(:2458-2574); for llama.cpp at the R1 pin: src/models/glm5-next.cpp:7-59 (the arch-specific reads). Each LLM_KV_*
is mapped to its key string through the LLM_KV_NAMES table of the same checkout (src/llama-arch.cpp); a read is
"required" unless its last argument is false (or it sits in an `if (!ml.get_key(...))` fallback); "fallback" = read only
when an earlier key left the value 0 (one of the three lines before is `if (... == 0)`). Then every listed
GGUF file is checked: which keys it has, and which keys of the file nobody reads.
usage: 2026-10-11-n12-loader-contract.py <ik checkout> <llama.cpp checkout> <gguf-py> <file.gguf>..."""
import re
import sys

ik, ml, gguf_py = sys.argv[1:4]
files = sys.argv[4:]
sys.path.append(gguf_py)
from gguf import GGUFReader  # noqa: E402


def kv_names(arch_cpp):
    names = {}
    for m in re.finditer(r'\{\s*(LLM_KV_\w+)\s*,\s*"([^"]+)"\s*\}', open(arch_cpp).read()):
        names[m.group(1)] = m.group(2)
    return names


def reads(path, lo, hi):
    out = []
    lines = open(path).read().splitlines()
    for n in range(lo, hi + 1):
        ln = lines[n - 1]
        for m in re.finditer(r"get_(key|key_or_arr|arr_n)\((LLM_KV_\w+)\s*,([^;]*)", ln):
            args = m.group(3)
            optional = bool(re.search(r",\s*(false|0)\s*\)", args)) or "if (!ml.get" in ln or "||" in ln
            prev = " ".join(lines[max(n - 4, 0):n - 1])           # the three lines before
            if not optional and re.search(r"if \([^)]*== 0", prev):
                optional = "fallback"                 # read only when an earlier key left the value 0
            out.append((m.group(2), optional is False, n) if optional != "fallback" else (m.group(2), "fallback", n))
    return out


IKN, MLN = kv_names(f"{ik}/src/llama-arch.cpp"), kv_names(f"{ml}/src/llama-arch.cpp")
ik_reads = reads(f"{ik}/src/llama-hparams.cpp", 140, 245) + reads(f"{ik}/src/llama-hparams.cpp", 2458, 2574)
ml_reads = reads(f"{ml}/src/models/glm5-next.cpp", 7, 59)
ml_by = {MLN.get(k, k): (req, n) for k, req, n in ml_reads}
print("key (%s = arch prefix)          ik @5194a9e (line)          llama.cpp glm5-next.cpp @79e2e74 (line; arch-specific "
      "reads only: the generic ones of src/llama-model.cpp are not tabulated, '-')")
seen = set()
rows = []
for k, req, n in ik_reads:
    name = IKN.get(k, k)
    if name in seen:
        continue
    seen.add(name)
    mr = ml_by.get(name)
    rows.append(name)
    tag = "fallback" if req == "fallback" else "required" if req else "optional"
    print(f"  {name:44s} {tag:9s} H:{n:<6d}   "
          f"{('required' if mr[0] else 'optional') + ' :' + str(mr[1]) if mr else '-'}")
for name, (req, n) in ml_by.items():
    if name not in seen:
        print(f"  {name:44s} {'-':9s} {'':8s}   {'required' if req else 'optional'} :{n}   (llama.cpp only)")
        rows.append(name)
for f in files:
    r = GGUFReader(f)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    have = {k for k in r.fields if k.startswith(arch + ".")}
    want = {n.replace("%s", arch) for n in rows}
    print(f"\n{f.split('/')[-1]}: general.architecture = {arch}, {len(have)} {arch}.* keys, {len(r.tensors)} tensors")
    print("  keys the file has that neither table reads:", sorted(have - want))
    print("  ik-required keys missing:", sorted(IKN.get(k, k).replace("%s", arch) for k, req, n in ik_reads
                                               if req is True and IKN.get(k, k).replace("%s", arch) not in r.fields
                                               and IKN.get(k, k).startswith("%s")))
