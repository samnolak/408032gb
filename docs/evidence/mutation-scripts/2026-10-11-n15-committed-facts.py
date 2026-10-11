#!/usr/bin/env python3
"""N15 round 2: facts printed from the committed third_party/gguf-headers files (bot commit 6cce416), offline.

  pins       per request: status, repo, revision, bytes fetched / committed, per file size, LFS sha256, header bytes,
             Range requests, gzip bytes, the size check PIN.json recorded, etag / commit checks
  tensors    per layer kind: are the real file's tensor names those of a tinygen layer (needs LLAMA_CPP_GGUF_PY + numpy)
  listing    the discovery listing: counts, and the GLM / MiMo repos the evidence names
  keys       the keys load_arch_hparams reads without a default at each given llama.cpp checkout, looked up in the files

Usage: python3 <this> pins|tensors|listing|experts|matrices|metashard ; python3 <this> keys NAME=<llama.cpp checkout> ...
"""
import collections
import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HDR = os.path.join(ROOT, "third_party", "gguf-headers")
GLM, QWEN = "glm-5.3-flash-ud-q4_k_xl", "qwen3.8-flash-next-ud-q4_k_xl"


def analyzer():
    spec = importlib.util.spec_from_file_location("gguf_headers_analyze",
                                                  os.path.join(ROOT, "tools", "gguf_headers", "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pins():
    for d in (GLM, QWEN):
        with open(os.path.join(HDR, d, "PIN.json")) as f:
            p = json.load(f)
        print("==", d, p["status"], p["repo"], p["requested_revision"], "->", p["revision"], "gated", p.get("gated"),
              p.get("license_tags"), "fetched_utc", p.get("fetched_utc"))
        print("   bytes_fetched", p["bytes_fetched"], "committed_bytes (gz)", p["committed_bytes"], "gguf files in repo",
              len(p["gguf_listing"]))
        for f in p["files"]:
            print(f"   {os.path.basename(f['path'])}: size {f['size']}, lfs {f['lfs_sha256'][:12]}, header {f['header_bytes']}, "
                  f"fetched {f['bytes_fetched']} in {len(f['range_requests'])}, gz {f['header_gz_bytes']}, "
                  f"size_check(PIN) {f['size_check']!r}, etag_ok {f['etag_matches_lfs_sha256']}, "
                  f"commit_ok {f['repo_commit_matches']}")


def tensors():
    A = analyzer()
    gp = os.environ.get("LLAMA_CPP_GGUF_PY")
    for n, tiny, var in ((GLM, "glm5next", "--mtp"), (QWEN, "qwen4exp", "--mtp")):
        r = A.load_request(os.path.join(HDR, n))
        m = A.Model([f["summary"] for f in r["files"]])
        d = tempfile.mkdtemp()
        tm, why = A.build_tiny(tiny, gp, d)
        if why:
            print(n, why)
            continue
        t = tm[var]

        def kinds(model):
            n_trunk = (model.a("block_count") or 0) - (model.a("nextn_predict_layers") or 0)
            per, glob_ = collections.defaultdict(set), set()
            for name in model.names:
                if name.startswith("blk."):
                    per[int(name.split(".")[1])].add(".".join(name.split(".")[2:]))
                else:
                    glob_.add(name)
            sig = collections.defaultdict(set)
            for i, s in per.items():
                sig[frozenset(s)].add("mtp" if i >= n_trunk else i)
            return glob_, sig

        gr, sr = kinds(m)
        gt, st = kinds(t)
        print(f"== {n} vs tiny {tiny} {var}: non-layer tensors real-only {sorted(gr - gt)}, tiny-only {sorted(gt - gr)}")
        for s, layers in sorted(sr.items(), key=lambda kv: sorted(str(x) for x in kv[1])):
            lay = sorted(layers, key=lambda x: (isinstance(x, str), x))
            head = f"   real layers {lay[:6]}{'...' if len(lay) > 6 else ''} ({len(lay)}):"
            if s in st:
                print(head, "same tensor names as a tiny layer")
            else:
                best = min(st, key=lambda u: len(u ^ s))
                print(head, f"no identical tiny layer; real-only {sorted(s - best)}, tiny-only {sorted(best - s)}")


def keys(srcs):
    """For each NAME=PATH llama.cpp checkout: the keys load_arch_hparams of glm5-next / qwen4exp reads without a default
    (ml.get_key / get_key_or_arr / get_arr calls whose last argument is not false; one call per line), each looked up in
    the real file's KV under the architecture prefix."""
    import re
    A = analyzer()
    models = {"glm5-next": A.Model([f["summary"] for f in A.load_request(os.path.join(HDR, GLM))["files"]]),
              "qwen4exp": A.Model([f["summary"] for f in A.load_request(os.path.join(HDR, QWEN))["files"]])}
    for spec in srcs:
        pin, root = spec.split("=", 1)
        with open(os.path.join(root, "src", "llama-arch.cpp")) as f:
            names = dict(re.findall(r'\{\s*(LLM_KV_\w+)\s*,\s*"([^"]+)"\s*\}', f.read()))
        for arch, src in (("glm5-next", "glm5-next.cpp"), ("qwen4exp", "qwen4exp.cpp")):
            with open(os.path.join(root, "src", "models", src)) as f:
                text = f.read()
            body = text[text.index("::load_arch_hparams"):text.index("::load_arch_tensors")]
            req = [m.group(2) for m in re.finditer(r"ml\.(get_key|get_key_or_arr|get_arr|get_arr_n)\((LLM_KV_\w+)([^;]*)\);", body)
                   if not m.group(3).strip().endswith("false")]
            missing = [(k, names.get(k, "?").replace("%s", arch)) for k in req
                       if names.get(k, "?").replace("%s", arch) not in models[arch].kv]
            print(f"{pin} {arch}: {len(req)} keys read without a default, missing in the real file: {missing}")


def experts():
    """ggml type of each routed-expert projection, counted over the layers (from the committed summaries)."""
    import glob
    for d in (GLM, QWEN):
        types = {}
        for p in sorted(glob.glob(os.path.join(HDR, d, "*.summary.json"))):
            with open(p) as f:
                for t in json.load(f)["tensors"]:
                    if "_exps" in t["name"]:
                        k = t["name"].split(".")[2]
                        types.setdefault(k, collections.Counter())[t["type"]] += 1
        print(d, {k: dict(sorted(v.items())) for k, v in sorted(types.items())})


def matrices():
    """Non-expert tensors with 2+ dimensions whose type is not Q8_0, grouped by name without the block index."""
    import glob
    for d in (GLM, QWEN):
        c, shapes = collections.Counter(), {}
        for p in sorted(glob.glob(os.path.join(HDR, d, "*.summary.json"))):
            with open(p) as f:
                for t in json.load(f)["tensors"]:
                    if "_exps" in t["name"] or len(t["shape"]) < 2 or t["type"] == "Q8_0":
                        continue
                    name = ".".join(t["name"].split(".")[2:]) if t["name"].startswith("blk.") else t["name"]
                    c[(name, t["type"])] += 1
                    shapes.setdefault((name, t["type"]), t["shape"])
        print(d)
        for (name, typ), n in sorted(c.items()):
            print(f"   {n:3d} x {name} {typ} {shapes[(name, typ)]}")


def metashard():
    """sha256 of each committed metadata-only first shard header + its zero padding to 32, against PIN.json's LFS
    sha256 of the whole file."""
    import gzip
    import hashlib
    for d in (GLM, QWEN):
        with open(os.path.join(HDR, d, "PIN.json")) as f:
            rec = json.load(f)["files"][0]
        with open(os.path.join(HDR, d, rec["header_gz"]), "rb") as f:
            hdr = gzip.decompress(f.read())
        whole = hdr + b"\0" * (-len(hdr) % 32)
        got = hashlib.sha256(whole).hexdigest()
        print(f"{os.path.basename(rec['path'])}: header {len(hdr)} + {len(whole) - len(hdr)} zero bytes = {len(whole)} "
              f"(file {rec['size']})")
        print(f"   sha256(header + zeros) {got}")
        print(f"   PIN.json lfs_sha256    {rec['lfs_sha256']}")
        print(f"   equal: {got == rec['lfs_sha256']}")


def listing():
    with open(os.path.join(HDR, "discover-glm-mimo", "LISTING.json")) as f:
        lst = json.load(f)
    print("fetched", lst["fetched_utc"], "request", lst["request_path"])
    for s in lst["searches"]:
        arch = collections.Counter(str(r["gguf_architecture"]) for r in s["results"])
        print(f"search {s['search']!r}: {s['n_results']} repos, expanded {s['expanded']}, matches "
              f"{sum(1 for r in s['results'] if r['arch_match'])}, with file listing "
              f"{sum(1 for r in s['results'] if 'gguf_files' in r)}; architectures {dict(sorted(arch.items()))}")
    want = ("unsloth/GLM-5.3-Flash-GGUF", "ggml-org/GLM-5.3-Flash-GGUF", "ggml-org/MiMo-V2.6-Flash-MOPD-GGUF",
            "ggml-org/MiMo-V2.6-Flash-RL-GGUF")
    for s in lst["searches"]:
        for r in s["results"]:
            if r["id"] not in want:
                continue
            print(f"{r['id']} @{r['sha']} arch {r['gguf_architecture']} lastModified {r['lastModified']} params "
                  f"{r['gguf_total_params']} ctx {r['gguf_context_length']} chat_template sha256 {r['gguf_chat_template_sha256']}")
            groups = collections.OrderedDict()
            for x in r.get("gguf_files") or []:
                k = x["path"].split("/")[0]
                g = groups.setdefault(k, [0, 0])
                g[0] += 1
                g[1] += x["size"] or 0
            for k, (nf, b) in groups.items():
                print(f"   {k}: {nf} file(s), {b} B = {b / 2 ** 30:.2f} GiB")


if __name__ == "__main__":
    if sys.argv[1] == "keys":
        keys(sys.argv[2:])
    else:
        {"pins": pins, "tensors": tensors, "listing": listing, "experts": experts, "matrices": matrices,
         "metashard": metashard}[sys.argv[1]]()
