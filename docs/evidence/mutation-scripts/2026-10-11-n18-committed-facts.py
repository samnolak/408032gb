#!/usr/bin/env python3
"""N18 round 2: facts printed from the committed third_party/gguf-headers files (bot commit df9bea1), offline.

  pins       per request: status, repo, revision, bytes fetched / committed, per file size, LFS sha256, header bytes,
             Range requests, gzip bytes, size check, etag / commit checks
  metashard  the metadata-only first shard of the ggml-org Q4_K: sha256(header + writer padding) against its LFS sha256
  qwenmtp    the 6 Qwen MTP files: tensors, non-layer tensors, the shared-target key, compress ratio of the NextN layer,
             embedded chat template against the vendored HF one and the UD-Q4_K_XL one, plain vs -shared- differences
  glmmtp     the ggml-org trunk and its mtp- files against the unsloth UD-Q4_K_XL NextN layer blk.45
  matrices   non-expert matrices (2+ dims) of the ggml-org Q4_K trunk that are not Q8_0
  listing    the MiMo listing: named repos with full revision and files; identical LFS sha256 across files
  mimopins   the MiMo requests of round 3 (bot commit 6161763): as pins
  mimo       the three MiMo MXFP4 GGUFs and the ggml-org mtp file compared: general keys, arch KV, tensor names, shapes,
             types (trunk layers, NextN layers, non-layer tensors)
  mimotiny   NEEDS LLAMA_CPP_GGUF_PY: the mimo2 probes on the real files against N16's and N17's tiny models, built from
             tools/tinygen/mimo2_tiny.py of origin/claude/n16-mimo-decode-parity and claude/n17-mimo-mtp-heads-fa
             (git show into a temporary directory; nothing merged) with their drivers' options
  mimoloader NAME=<llama.cpp checkout> ...: per pin, the tensor names llama_model_mimo2::load_arch_tensors creates (parsed
             from src/models/mimo2.cpp + src/llama-arch.cpp), and per MiMo file the tensors the loader does not create
             and the required ones it lacks (by reading; nothing loaded)

Usage: python3 <this> pins|metashard|qwenmtp|glmmtp|matrices|listing|mimopins|mimo|mimotiny ;
       python3 <this> mimoloader NAME=<checkout> ...
"""
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HDR = os.path.join(ROOT, "third_party", "gguf-headers")
QMTP, GQ4K, GMTP, MIMO = ("qwen3.8-flash-next-mtp", "glm-5.3-flash-ggml-org-q4_k", "glm-5.3-flash-ggml-org-mtp",
                          "discover-mimo")
QTRUNK, GUNS = "qwen3.8-flash-next-ud-q4_k_xl", "glm-5.3-flash-ud-q4_k_xl"


def analyzer():
    spec = importlib.util.spec_from_file_location("gguf_headers_analyze",
                                                  os.path.join(ROOT, "tools", "gguf_headers", "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model(A, name):
    r = A.load_request(os.path.join(HDR, name))
    assert all(ok for _, ok in r["integrity"]), name
    return r, A.Model([f["summary"] for f in r["files"]])


def pins():
    for d in (QMTP, GQ4K, GMTP):
        with open(os.path.join(HDR, d, "PIN.json")) as f:
            p = json.load(f)
        print("==", d, p["status"], p["repo"], p["requested_revision"], "->", p["revision"], "gated", p.get("gated"),
              p.get("license_tags"), "fetched_utc", p.get("fetched_utc"), "workflow_sha", p.get("workflow_sha"))
        print("   bytes_fetched", p["bytes_fetched"], "committed_bytes (gz)", p["committed_bytes"], "files in request",
              len(p["files"]), "max_header_bytes", p["max_header_bytes"])
        for f in p["files"]:
            print(f"   {os.path.basename(f['path'])}: size {f['size']}, lfs {f['lfs_sha256'][:12]}, header "
                  f"{f['header_bytes']}, fetched {f['bytes_fetched']} in {len(f['range_requests'])}, gz "
                  f"{f['header_gz_bytes']}, size_check(PIN) {f['size_check']!r}, etag_ok {f['etag_matches_lfs_sha256']}, "
                  f"commit_ok {f['repo_commit_matches']}")
    with open(os.path.join(HDR, MIMO, "LISTING.json")) as f:
        lst = json.load(f)
    print("==", MIMO, "fetched_utc", lst["fetched_utc"], "request", lst["request_path"], "sha256",
          lst["request_sha256"][:12])


def metashard():
    with open(os.path.join(HDR, GQ4K, "PIN.json")) as f:
        p = json.load(f)
    rec = p["files"][0]
    with open(os.path.join(HDR, GQ4K, rec["header_gz"]), "rb") as f:
        hdr = gzip.decompress(f.read())
    pad = rec["size"] - len(hdr)
    print(f"{os.path.basename(rec['path'])}: header {len(hdr)} + {pad} zero bytes = {len(hdr) + pad} "
          f"(file {rec['size']})")
    got = hashlib.sha256(hdr + b"\0" * pad).hexdigest()
    print(f"   sha256(header + zeros) {got}\n   PIN.json lfs_sha256    {rec['lfs_sha256']}\n"
          f"   equal: {got == rec['lfs_sha256']}")


def _template_sha(m):
    t = m.v("tokenizer.chat_template")
    return hashlib.sha256(t.encode("utf-8", "surrogateescape")).hexdigest() if t is not None else None


def qwenmtp():
    A = analyzer()
    r, _ = model(A, QMTP)
    _, trunk = model(A, QTRUNK)
    with open(os.path.join(ROOT, "third_party", "hf", "qwen3.8-flash-next-chat", "chat_template.jinja"), "rb") as f:
        ven = hashlib.sha256(f.read()).hexdigest()
    print(f"vendored HF template sha256 {ven[:16]}, UD-Q4_K_XL embedded {_template_sha(trunk)[:16]}")
    files = {}
    for f in r["files"]:
        m = A.Model([f["summary"]])
        n = os.path.basename(f["rec"]["path"])
        files[n] = (f, m)
        g = {t["name"]: (t["type"], t["shape"], t["nbytes"]) for t in m.tensors if not t["name"].startswith("blk.")}
        cr = m.a("attention.compress_ratios")
        print(f"{n}: {len(m.tensors)} tensors, {len(m.kv)} KV, general.name {m.v('general.name')!r}, size_label "
              f"{m.v('general.size_label')!r}, file_type {m.v('general.file_type')}, nextn_shared_target_tensors "
              f"{m.a('nextn_shared_target_tensors')}, compress_ratios len {len(cr)} [48] = {cr[48]}, non-layer "
              f"{g if g else 'none'}, template {(_template_sha(m) or 'absent')[:16]} (= vendored: "
              f"{_template_sha(m) == ven})")
    for q in ("BF16", "Q4_K_M", "Q8_0"):
        (fa, a), (fb, b) = files[f"mtp-Qwen3.8-Flash-Next-{q}.gguf"], files[f"mtp-Qwen3.8-Flash-Next-shared-{q}.gguf"]
        only_a, only_b = sorted(a.names - b.names), sorted(b.names - a.names)
        same = all(a.shapes[t] == b.shapes[t] for t in a.names & b.names)
        types_a = {t["name"]: t["type"] for t in a.tensors}
        types_b = {t["name"]: t["type"] for t in b.tensors}
        diff_types = sorted(t for t in a.names & b.names if types_a[t] != types_b[t])
        kv_a, kv_b = set(a.kv), set(b.kv)
        gone = sum(t["nbytes"] for t in a.tensors if t["name"] in only_a)
        print(f"{q}: plain-only tensors {only_a}, shared-only {only_b}, common shapes equal {same}, common types differ "
              f"{diff_types}; KV plain-only {sorted(kv_a - kv_b)}, shared-only {sorted(kv_b - kv_a)}; file sizes "
              f"{fa['rec']['size']} - {fb['rec']['size']} = {fa['rec']['size'] - fb['rec']['size']} B, plain-only "
              f"tensor bytes {gone} (+ header difference {fa['rec']['header_bytes'] - fb['rec']['header_bytes']} B)")


def glmmtp():
    A = analyzer()
    _, uns = model(A, GUNS)
    _, trunk = model(A, GQ4K)
    r, _ = model(A, GMTP)
    print(f"ggml-org Q4_K trunk: block_count {trunk.a('block_count')}, nextn_predict_layers "
          f"{trunk.a('nextn_predict_layers')}, layers with tensors "
          f"{sorted({int(n.split('.')[1]) for n in trunk.names if n.startswith('blk.')})[-3:]} (last 3), nextn tensors "
          f"{sorted(n for n in trunk.names if '.nextn.' in n)}, tensors {len(trunk.tensors)}")
    ref = {n[len('blk.45.'):]: t for n, t in ((t["name"], t) for t in uns.tensors) if n.startswith("blk.45.")}
    print(f"unsloth UD-Q4_K_XL blk.45: {len(ref)} tensors")
    params = lambda ts: sum(int(__import__('math').prod(t['shape'])) for t in ts)  # noqa: E731
    for f in r["files"]:
        m = A.Model([f["summary"]])
        mine = {t["name"][len("blk.45."):]: t for t in m.tensors if t["name"].startswith("blk.45.")}
        shapes = all(mine[k]["shape"] == ref[k]["shape"] for k in mine.keys() & ref.keys())
        types = collections.Counter(t["type"] for t in mine.values())
        print(f"{os.path.basename(f['rec']['path'])}: blk.45 {len(mine)} tensors, names equal to unsloth blk.45: "
              f"{set(mine) == set(ref)} (only here {sorted(set(mine) - set(ref))}, only unsloth "
              f"{sorted(set(ref) - set(mine))}), shapes equal {shapes}, types {dict(types)}, params blk.45 "
              f"{params(mine.values())}, all tensors {params(m.tensors)}")
    print(f"unsloth blk.45 params {params(ref.values())}; unsloth all {params(uns.tensors)} - ggml-org trunk "
          f"{params(trunk.tensors)} = {params(uns.tensors) - params(trunk.tensors)}")


def matrices():
    A = analyzer()
    _, trunk = model(A, GQ4K)
    agg = collections.Counter()
    for t in trunk.tensors:
        if len(t["shape"]) >= 2 and "_exps" not in t["name"] and t["type"] != "Q8_0":
            agg[(".".join(t["name"].split(".")[2:]) if t["name"].startswith("blk.") else t["name"], t["type"],
                 json.dumps(t["shape"]))] += 1
    for (n, ty, sh), c in sorted(agg.items()):
        print(f"   {c:3d} x {n} {ty} {sh}")


def listing():
    with open(os.path.join(HDR, MIMO, "LISTING.json")) as f:
        lst = json.load(f)
    by_sha = collections.defaultdict(list)
    for x in lst["repos"]:
        print(f"{x['repo']}: {x['status']}, revision {x.get('revision')}, http {x.get('http_status')}, arch "
              f"{x.get('gguf_architecture')}, total params {x.get('gguf_total_params')}, template sha256 "
              f"{(x.get('gguf_chat_template_sha256') or '-')[:16]}, lastModified {x.get('lastModified')}")
        for g in x.get("gguf_files") or []:
            print(f"    {g['path']}  {g['size']}  {g['lfs_sha256']}")
            by_sha[g["lfs_sha256"]].append(f"{x['repo']}:{g['path']}")
    for sha, where in sorted(by_sha.items()):
        if len(where) > 1:
            print(f"same LFS sha256 {sha[:16]}: {where}")
    s = lst["searches"][0]
    print(f"search {s['search']!r}: {s['n_results']} repos, unsloth among them: "
          f"{[r['id'] for r in s['results'] if r['id'].lower().startswith('unsloth/')]}")


MMX, MMTP, MQP, MKP = ("mimo-v2.6-flash-mopd-ggml-org-mxfp4", "mimo-v2.6-flash-mopd-ggml-org-mtp",
                       "mimo-v2.6-flash-mopd-quantaplanta-mxfp4", "mimo-v2.6-flash-mopd-kernelpool-mxfp4")
MIMO_ALL = (MMX, MMTP, MQP, MKP)


def mimopins():
    for d in MIMO_ALL:
        with open(os.path.join(HDR, d, "PIN.json")) as f:
            p = json.load(f)
        print("==", d, p["status"], p["repo"], p["requested_revision"], "->", p["revision"], "gated", p.get("gated"),
              p.get("license_tags"), "fetched_utc", p.get("fetched_utc"), "workflow_sha", p.get("workflow_sha"))
        print("   bytes_fetched", p["bytes_fetched"], "committed_bytes (gz)", p["committed_bytes"])
        for f in p["files"]:
            print(f"   {os.path.basename(f['path'])}: size {f['size']}, lfs {f['lfs_sha256'][:12]}, header "
                  f"{f['header_bytes']}, fetched {f['bytes_fetched']} in {len(f['range_requests'])}, gz "
                  f"{f['header_gz_bytes']}, size_check(PIN) {f['size_check']!r}, etag_ok {f['etag_matches_lfs_sha256']}, "
                  f"commit_ok {f['repo_commit_matches']}")
    with open(os.path.join(HDR, MMX, "PIN.json")) as f:
        rec = json.load(f)["files"][0]
    with open(os.path.join(HDR, MMX, rec["header_gz"]), "rb") as f:
        hdr = gzip.decompress(f.read())
    pad = rec["size"] - len(hdr)
    print(f"{os.path.basename(rec['path'])}: header {len(hdr)} + {pad} zero bytes = file {rec['size']}; sha256 equal to "
          f"LFS: {hashlib.sha256(hdr + bytes(pad)).hexdigest() == rec['lfs_sha256']}")


def _mimo_models(A):
    return {d: model(A, d)[1] for d in MIMO_ALL}


def _layer(name):
    return int(name.split(".")[1]) if name.startswith("blk.") else None


def mimo():
    A = analyzer()
    ms = _mimo_models(A)
    for d, m in ms.items():
        g = {k: m.v(k) for k in sorted(m.kv) if k.startswith("general.")}
        lay = [x for x in map(_layer, m.names) if x is not None]
        print(f"{d}: {len(m.tensors)} tensors, {len(m.kv)} KV, layers {min(lay)}-{max(lay)}, nextn_predict_layers "
              f"{m.a('nextn_predict_layers')}, general {json.dumps(g)[:700]}")
    trunks = {d: ms[d] for d in (MMX, MQP, MKP)}
    keys = sorted(set().union(*(set(m.arch_keys()) for m in trunks.values())))
    print("arch KV that differ between the three MXFP4 files (value or presence):")
    for k in keys:
        vals = [trunks[d].arch_keys().get(k, "(absent)") for d in trunks]
        if not all(json.dumps(v) == json.dumps(vals[0]) for v in vals):
            print(f"   {k}: " + " | ".join(f"{d.split('-mopd-')[1]} " + (
                f"[{len(v)}] first 48 equal to ggml-org's: {v[:48] == vals[0][:48]}, after: {v[48:]}"
                if isinstance(v, list) and isinstance(vals[0], list) else json.dumps(v)[:60]) for d, v in zip(trunks, vals)))
    other = sorted(k for k in set().union(*(set(m.kv) for m in trunks.values())) if not k.startswith(("general.", "mimo2.")))
    for k in other:
        vals = [trunks[d].kv.get(k) for d in trunks]
        ids = [json.dumps((v or {}).get("value", (v or {}).get("sha256"))) if v else "(absent)" for v in vals]
        if len(set(ids)) > 1:
            print(f"   {k}: " + " | ".join(f"{d.split('-mopd-')[1]} {i[:40]}" for d, i in zip(trunks, ids)))
    tt = {d: {t["name"]: t for t in m.tensors if (_layer(t["name"]) is None or _layer(t["name"]) < 48)}
          for d, m in trunks.items()}
    base = tt[MMX]
    for d in (MQP, MKP):
        o = tt[d]
        shapes = all(o[n]["shape"] == base[n]["shape"] for n in base.keys() & o.keys())
        diff = collections.Counter((base[n]["type"], o[n]["type"]) for n in base.keys() & o.keys()
                                   if base[n]["type"] != o[n]["type"])
        print(f"trunk tensors (layers 0-47 + non-layer) {d} vs ggml-org: names equal {set(o) == set(base)} (only here "
              f"{sorted(set(o) - set(base))[:6]}, only ggml-org {sorted(set(base) - set(o))[:6]}), shapes equal {shapes}, "
              f"type pairs (ggml-org, this) that differ {dict(diff)}")
    for d in (MMX, MQP, MKP):
        c = collections.Counter(t["type"] for n, t in tt[d].items() if "_exps" not in n and len(t["shape"]) >= 2)
        print(f"{d}: non-expert matrices by type {dict(c)}; non-layer {sorted((n, t['type']) for n, t in tt[d].items() if _layer(n) is None)}")
    for d in (MMTP, MQP, MKP):
        m = ms[d]
        lay = {i: {n[len(f'blk.{i}.'):]: (t["type"], t["shape"]) for n, t in ((t["name"], t) for t in m.tensors)
                   if n.startswith(f"blk.{i}.")} for i in (48, 49, 50)}
        ref = {i: {n[len(f'blk.{i}.'):]: (t["type"], t["shape"]) for n, t in ((t["name"], t) for t in ms[MQP].tensors)
                   if n.startswith(f"blk.{i}.")} for i in (48, 49, 50)}
        print(f"{d}: NextN layers " + "; ".join(
            f"blk.{i} {len(lay[i])} tensors, names = QuantaPlanta's {set(lay[i]) == set(ref[i])}, shapes = "
            f"{all(lay[i][k][1] == ref[i][k][1] for k in lay[i].keys() & ref[i].keys())}, types "
            f"{dict(collections.Counter(v[0] for v in lay[i].values()))}" for i in (48, 49, 50)))
        print(f"   only here {sorted(set(lay[48]) - set(ref[48]))}, only QuantaPlanta {sorted(set(ref[48]) - set(lay[48]))}")
    for d, m in ms.items():
        print(f"{d}: embedded chat template sha256 {_template_sha(m)}")


def mimotiny():
    import subprocess
    import tempfile
    A = analyzer()
    gp = os.environ.get("LLAMA_CPP_GGUF_PY")
    if not gp:
        print("NOT RUN: set LLAMA_CPP_GGUF_PY")
        return
    sys.path.insert(0, gp)
    ms = _mimo_models(A)
    d = tempfile.mkdtemp()
    tiny = {}
    for ref, label, kws in (
            ("origin/claude/n16-mimo-decode-parity", "N16 n16.gguf (--fused-qkv --sinks swa --value-scale 0.75 --mtp)",
             [dict(fused_qkv=True, sinks="swa", value_scale=0.75, mtp=True)]),
            ("claude/n17-mimo-mtp-heads-fa", "N17 n17.gguf + n17hd.gguf (--mtp-layers 3; --head-dims 192,128)",
             [dict(fused_qkv=True, sinks="swa", value_scale=0.75, mtp_layers=3),
              dict(fused_qkv=True, sinks="swa", value_scale=0.75, head_dims=(192, 128))])):
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", ref], capture_output=True, text=True).stdout.strip()
        src = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:tools/tinygen/mimo2_tiny.py"], capture_output=True,
                             text=True, check=True).stdout
        mp = os.path.join(d, f"mimo2_tiny_{len(tiny)}.py")
        with open(mp, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location(f"mimo2_tiny_{len(tiny)}", mp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, kw in enumerate(kws):
            path = os.path.join(d, f"t{len(tiny)}-{k}.gguf")
            mod.build(path, **kw)
            h, hb = A.G.read_file_header(path)
            tiny[f"{label} [{k}] @{sha[:7]}"] = A.Model([A.G.summarize(h, hb)])
    for lab, t in tiny.items():
        print(f"tiny {lab}: block_count {t.a('block_count')}, nextn {t.a('nextn_predict_layers')}, K/V "
              f"{t.a('attention.key_length')}/{t.a('attention.value_length')}, n_rot {t.a('rope.dimension_count')}")
    for name in (MMX, MMTP, MQP, MKP):
        br = A.section_branches(ms[name], "mimo2", tiny, {})
        print(f"== {name}")
        for b in br:
            print(f"   [{'tiny OK' if b['exercised_by_tiny'] else 'NOT IN N16/N17 TINY'}] {b['id']}: real {b['real']}; "
                  f"not exercised {b['not_exercised']}")


def _mimo2_loader_names(root):
    """{(scope, name): required} of llama_model_mimo2::load_arch_tensors; scope global / trunk / nextn"""
    import re
    with open(os.path.join(root, "src", "llama-arch.cpp"), errors="replace") as f:
        names = dict(re.findall(r'\{\s*(LLM_TENSOR_\w+)\s*,\s*"([^"]+)"\s*\}', f.read()))
    with open(os.path.join(root, "src", "models", "mimo2.cpp"), errors="replace") as f:
        text = f.read()
    body = text[text.index("::load_arch_tensors"):text.index("::build_arch_graph")]
    out, depth_nextn = {}, None
    depth = 0
    for ln in body.splitlines():
        if "if (is_nextn) {" in ln:
            depth_nextn = depth + 1
        m = re.search(r'create_tensor\(tn\((LLM_TENSOR_\w+),\s*"(\w+)"(,\s*i)?\)[^;]*\},\s*([^;]*)\);', ln)
        if m:
            scope = "global" if not m.group(3) else ("nextn" if depth_nextn and depth >= depth_nextn else "trunk")
            nm = names[m.group(1)].replace("blk.%d.", "") + "." + m.group(2)
            out[(scope, nm)] = "TENSOR_NOT_REQUIRED" not in m.group(4)
        if "create_tensor_qkv(" in ln:
            out[("trunk", "attn_qkv|attn_q+attn_k+attn_v")] = True
        depth += ln.count("{") - ln.count("}")
        if depth_nextn and depth < depth_nextn:
            depth_nextn = None
    return out


def mimoloader(srcs):
    A = analyzer()
    ms = _mimo_models(A)
    for spec in srcs:
        pin, root = spec.split("=", 1)
        rules = _mimo2_loader_names(root)
        print(f"[{pin}] load_arch_tensors creates: " + ", ".join(
            f"{sc}:{n}{'' if req else ' (opt)'}" for (sc, n), req in sorted(rules.items())))
        for d, m in ms.items():
            n_layer = (m.a("block_count") or 0) - (m.a("nextn_predict_layers") or 0)
            unknown, missing = [], []
            per = collections.defaultdict(set)
            for t in m.tensors:
                i = _layer(t["name"])
                if i is None:
                    if ("global", t["name"]) not in rules:
                        unknown.append(t["name"])
                    continue
                suf = t["name"][len(f"blk.{i}."):]
                per[i].add(suf)
                scope = "nextn" if i >= n_layer else "trunk"
                qkv = suf.split(".")[0] in ("attn_qkv", "attn_q", "attn_k", "attn_v")
                if not qkv and (scope, suf) not in rules and ("trunk", suf) not in rules:
                    unknown.append(t["name"])
            for i, have in sorted(per.items()):
                scope = "nextn" if i >= n_layer else "trunk"
                need = [n for (sc, n), req in rules.items() if req and sc in ({"trunk", "nextn"} if scope == "nextn"
                                                                                 else {"trunk"})]
                for n in need:
                    if n == "attn_qkv|attn_q+attn_k+attn_v":
                        if "attn_qkv.weight" not in have and not {"attn_q.weight", "attn_k.weight", "attn_v.weight"} <= have:
                            missing.append(f"blk.{i}.attn_qkv")
                    elif n not in have:
                        missing.append(f"blk.{i}.{n}")
            glob_missing = [n for (sc, n), req in rules.items() if sc == "global" and req and n not in m.names]
            print(f"   {d}: layers {min(per)}-{max(per)} (n_layer {n_layer}), tensors the loader does not create "
                  f"{unknown}, required ones absent {missing + glob_missing}")


if __name__ == "__main__":
    if sys.argv[1] == "mimoloader":
        mimoloader(sys.argv[2:])
    else:
        {"pins": pins, "metashard": metashard, "qwenmtp": qwenmtp, "glmmtp": glmmtp, "matrices": matrices,
         "listing": listing, "mimopins": mimopins, "mimo": mimo, "mimotiny": mimotiny}[sys.argv[1]]()
