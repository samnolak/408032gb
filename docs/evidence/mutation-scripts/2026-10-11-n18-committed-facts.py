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

Usage: python3 <this> pins|metashard|qwenmtp|glmmtp|matrices|listing
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


if __name__ == "__main__":
    {"pins": pins, "metashard": metashard, "qwenmtp": qwenmtp, "glmmtp": glmmtp, "matrices": matrices,
     "listing": listing}[sys.argv[1]]()
