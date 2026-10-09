#!/usr/bin/env python3
"""Audit the storage type of every tensor in a (split) GGUF and enforce the customer's ">= quality 4-bit" floor.

Reads only GGUF headers (metadata and tensor infos), never tensor data, so it runs in seconds on 200 GB models.
Stdlib only: works on an offline server without gguf-py.

Classes: routed experts (`*_exps*`), shared experts (`*_shexp*`), attention/linear-attention/indexer (`attn_*`, `ssm_*`,
`indexer*`), dense FFN (`ffn_*` without exps/shexp), embedding (`token_embd`), output head (`output.weight`), other
(norms, router, mHC...). For each class: parameters, bytes, effective bits per weight, histogram of types.

Policy (ADR-012 D5: main weights at no less than a quality 4-bit class): every routed-expert tensor must be
>= --min-expert-bpw (default 4.0) and every non-expert matrix >= --min-dense-bpw (default 4.0), otherwise FAIL (exit 1);
--dense-warn-only downgrades the non-expert check to WARN (for exploration only, never for a delivered configuration). Note: by this rule a "IQ4_XS"-named pack
whose experts mix IQ3_S tensors (3.44 bpw) fails, which is the point.

Usage:
  python3 tools/gguf_quant_audit.py /models/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
  python3 tools/gguf_quant_audit.py model.gguf --json > audit.json
"""
import argparse
import json
import os
import re
import struct
import sys

# ggml_type id -> (name, block_size, type_size_bytes). CONFIRMED: llama.cpp gguf-py/gguf/constants.py
# GGMLQuantizationType + GGML_QUANT_SIZES @ec7630a640789c393694fb194f1bbbf0369fc62d (QK_K = 256 expanded).
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}

# GGUF metadata value types
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, b)[0]

    def string(self):
        n = self.take("<Q")
        if n > 1 << 30:
            raise ValueError(f"implausible string length {n}")
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF string")
        return b.decode("utf-8", errors="replace")

    def value(self, vtype, keep):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype = self.take("<I")
            count = self.take("<Q")
            if etype in _SCALAR and not keep:
                self.f.seek(struct.calcsize(_SCALAR[etype]) * count, os.SEEK_CUR)
                return f"<array of {count}>"
            items = [self.value(etype, keep) for _ in range(count)]
            return items if keep else f"<array of {count}>"
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.take("<I")
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        n_tensors = r.take("<Q")
        n_kv = r.take("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.take("<I")
            keep = vtype != ARRAY or key.startswith("split.")
            meta[key] = r.value(vtype, keep)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.take("<I")
            dims = [r.take("<Q") for _ in range(n_dims)]
            ttype = r.take("<I")
            r.take("<Q")  # offset
            tensors.append((name, dims, ttype))
    return {"version": version, "meta": meta, "tensors": tensors}


def shard_paths(first):
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    stem, total = m.group(1), int(m.group(3))
    paths = [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing shards: {missing}")
    return paths


def classify(name):
    if "_exps" in name:
        return "routed_experts"
    if "_shexp" in name:
        return "shared_experts"
    if name.startswith("token_embd"):
        return "embedding"
    if name.startswith("output.") or name == "output":
        return "output_head"
    if "norm" in name:
        return "other"
    if re.search(r"\.(attn_|ssm_|indexer|kda_|wkv|wq|wk_|wv_|wo)", name):
        return "attention"
    if re.search(r"\.ffn_(gate|up|down)(\.|$)", name):
        return "dense_ffn"
    return "other"


def audit(first, min_expert_bpw=4.0, min_dense_bpw=4.0, dense_warn_only=False):
    paths = shard_paths(first)
    classes = {}
    findings = []
    arch = None
    unknown = []
    for p in paths:
        h = read_header(p)
        arch = arch or h["meta"].get("general.architecture")
        for name, dims, ttype in h["tensors"]:
            n = 1
            for d in dims:
                n *= d
            if ttype not in GGML_TYPES:
                unknown.append((name, ttype))
                continue
            tname, block, size = GGML_TYPES[ttype]
            nbytes = n * size / block
            bpw = size * 8.0 / block
            c = classify(name)
            st = classes.setdefault(c, {"tensors": 0, "params": 0, "bytes": 0.0, "types": {}})
            st["tensors"] += 1
            st["params"] += n
            st["bytes"] += nbytes
            st["types"][tname] = st["types"].get(tname, 0) + n
            if c == "routed_experts" and bpw < min_expert_bpw:
                findings.append({"level": "FAIL", "tensor": name, "type": tname, "bpw": round(bpw, 3)})
            elif c != "routed_experts" and len(dims) >= 2 and bpw < min_dense_bpw:
                findings.append({"level": "WARN" if dense_warn_only else "FAIL", "tensor": name, "type": tname, "bpw": round(bpw, 3)})
    for st in classes.values():
        st["bpw"] = st["bytes"] * 8 / st["params"] if st["params"] else None
    total_p = sum(s["params"] for s in classes.values())
    total_b = sum(s["bytes"] for s in classes.values())
    verdict = "FAIL" if any(f["level"] == "FAIL" for f in findings) or unknown else "PASS"
    return {"files": [os.path.basename(p) for p in paths], "architecture": arch, "classes": classes,
            "total_params": total_p, "total_gib": total_b / 2 ** 30, "total_bpw": total_b * 8 / total_p if total_p else None,
            "min_expert_bpw": min_expert_bpw, "min_dense_bpw": min_dense_bpw,
            "findings": findings, "unknown_types": unknown, "verdict": verdict}


def render(r):
    out = [f"# {r['files'][0]} ({len(r['files'])} file(s)), architecture {r['architecture']!r}",
           f"total {r['total_params'] / 1e9:.2f}B params, {r['total_gib']:.2f} GiB, {r['total_bpw']:.3f} bpw"]
    for c in ("routed_experts", "shared_experts", "attention", "dense_ffn", "embedding", "output_head", "other"):
        st = r["classes"].get(c)
        if not st:
            continue
        mix = ", ".join(f"{t} {100 * n / st['params']:.1f}%" for t, n in sorted(st["types"].items(), key=lambda x: -x[1]))
        out.append(f"  {c:15s} {st['params'] / 1e9:8.3f}B  {st['bytes'] / 2 ** 30:8.2f} GiB  {st['bpw']:6.3f} bpw  [{mix}]")
    n_fail = sum(1 for f in r["findings"] if f["level"] == "FAIL")
    n_warn = sum(1 for f in r["findings"] if f["level"] == "WARN")
    for f in r["findings"][:20]:
        out.append(f"  {f['level']}: {f['tensor']} {f['type']} {f['bpw']} bpw")
    if len(r["findings"]) > 20:
        out.append(f"  ... {len(r['findings']) - 20} more findings")
    for name, t in r["unknown_types"]:
        out.append(f"  FAIL: {name}: unknown ggml type id {t} (update GGML_TYPES from the pinned gguf-py)")
    out.append(f"verdict: {r['verdict']} (routed experts >= {r['min_expert_bpw']} bpw, non-expert matrices >= "
               f"{r['min_dense_bpw']} bpw: {n_fail} failing tensors, {n_warn} warnings)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", help="a .gguf file or the first shard (-00001-of-0000N.gguf)")
    ap.add_argument("--min-expert-bpw", type=float, default=4.0)
    ap.add_argument("--min-dense-bpw", type=float, default=4.0)
    ap.add_argument("--dense-warn-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = audit(a.gguf, a.min_expert_bpw, a.min_dense_bpw, a.dense_warn_only)
    if a.json:
        json.dump(r, sys.stdout, indent=1)
        print()
    else:
        print(render(r))
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
