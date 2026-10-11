#!/usr/bin/env python3
"""Read-only check of the GLM-5.3-Flash UD-Q4_K_XL GGUF shards on a host (epyc-ai): sizes, sha256, GGUF header.

Why: the six files on epyc-ai (AGENTS.md "Weights on host epyc-ai") are not yet trusted. AGENTS.md calls their
architecture `glm5next`; the Hugging Face files of unsloth/GLM-5.3-Flash-GGUF say `glm5-next` (N15,
docs/evidence/2026-10-11-n15-gguf-headers.md on branch claude/n15-gguf-headers). This tool tells which one the host has.

Pins (embedded below, copied from third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/PIN.json at commit 67bd5ba of
claude/n15-gguf-headers; repo unsloth/GLM-5.3-Flash-GGUF, revision a38483c8cd5df544f53d70fb281afe97369d5ab6):
  size, lfs_sha256 (= sha256 of the whole file), header_bytes and header_sha256 (sha256 of the first header_bytes bytes:
  the GGUF header up to the data section) of each shard, and header facts of shard 1 (architecture, expert groups, chat
  template sha256 of the UTF-8 bytes).

Safety: the weights are only opened for reading ("rb"); nothing is written outside --out, which must be a new or empty
directory outside --dir; no network access (stdlib only: os, hashlib, struct, json, argparse, concurrent.futures).
Run it as `python3 -I` (tools/hostcheck/glm_gguf_check.sh does) so no module from the current directory is imported.

Usage:
  tools/hostcheck/glm_gguf_check.sh --dir /mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL --out ~/glmcheck
  options: --jobs N (parallel sha256, default 2), --no-hash (sizes and headers only), --pins FILE (other pins, tests)
Exit codes: 0 every shard matches size and sha256; 1 a shard is missing or differs; 3 --no-hash and nothing differs
(sha256 not checked: not a pass); 2 usage or I/O error.
Outputs in --out: report.json, report.txt (also printed), sha256sum.txt (sha256sum -c format, computed hashes).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import sys
import time

PIN_SOURCE = ("claude/n15-gguf-headers@67bd5ba:third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/PIN.json "
              "(unsloth/GLM-5.3-Flash-GGUF @ a38483c8cd5df544f53d70fb281afe97369d5ab6)")
PINS = {
    "source": PIN_SOURCE,
    "files": [
        {"name": "GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf", "size": 9429984,
         "sha256": "b59c030090f012b38213bc8c4f78ce10767f2693d730d0dbcd3dc16a5f69e2bc",
         "header_bytes": 9429956, "header_sha256": "0c90bc17e2486d3ed599dfda7e2b244bdded27c48d0680569077d741c11b1016"},
        {"name": "GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf", "size": 49198039200,
         "sha256": "d3ecb6ff3957a99878f9a0352676a0913748a431a9a4a1b1ffa5ade9b8947a74",
         "header_bytes": 24608, "header_sha256": "a1e0c64d7bf1be6528c0945b06e58b9259b38ab710a3da89a93ce156022cdc1a"},
        {"name": "GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf", "size": 49988626304,
         "sha256": "328073c004e7c5395b208247feed26089db247b94dd8fa4ab0f2569d86c1d770",
         "header_bytes": 21376, "header_sha256": "34fc5c898088996d8e1ee0d76a9c10df8fa6dc7c30273612823427bf892fc317"},
        {"name": "GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf", "size": 48700701344,
         "sha256": "6803ab7effa6da4a0b02c9c7865952fdd6c5b2e10e53903348da369e90bd13f9",
         "header_bytes": 20256, "header_sha256": "7176860b6d2fdc38f36bd3cbfac4055c223201623370b97267f316c420a9fc5f"},
        {"name": "GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf", "size": 49026026752,
         "sha256": "852f3df9dacde3d196796f9fe738468e80c8701555ff0a8c30bc010fb8974dd8",
         "header_bytes": 21632, "header_sha256": "1b4a3a705d62bdf4fa36910ad01243a6843d351daa4deea4302001340a394ba1"},
        {"name": "GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf", "size": 2784497888,
         "sha256": "c6ba510fafc1e12cc0addbc361a329c0c592ab96980098e7e25e107d8faa983e",
         "header_bytes": 1248, "header_sha256": "dd1a523a0a9dcbad7cd6754fe69682cd94177f1a51f937721fc21f8ea57f019e"},
    ],
    # header facts of shard 1 at the pin (N15 summary.json; the template sha256 is of its UTF-8 bytes, 10648 B)
    "expect_first": {
        "general.architecture": "glm5-next",
        "expert_group_count": 1,
        "expert_group_used_count": 1,
        "chat_template_sha256": "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724",
        "split.count": 6,
    },
}
# sha256 of the vendored zai-org template (third_party/hf/glm-5.3-flash-chat/chat_template.jinja, N14): for comparison only
ZAI_TEMPLATE_SHA256 = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"

GGUF_MAGIC = b"GGUF"
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
T_STRING, T_ARRAY = 8, 9
TYPE_NAMES = {0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32", 6: "float32", 7: "bool",
              8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64"}
MAX_STRING = 1 << 30
MAX_COUNT = 1 << 40
KEEP_ARRAY = 4096          # numeric arrays up to this length are kept in the report; longer ones are summarised


class Reader:
    """Sequential little-endian reader over a binary file opened read-only; counts the bytes consumed."""

    def __init__(self, f):
        self.f, self.pos = f, 0

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("file ends inside the GGUF header (at byte %d, wanted %d more)" % (self.pos + len(b), n))
        self.pos += n
        return b

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        n = self.u("Q")
        if n > MAX_STRING:
            raise ValueError("implausible string length %d at byte %d" % (n, self.pos - 8))
        return self.take(n)


def read_header(path):
    """Parse the GGUF header of `path` (read-only): returns dict with version, n_tensors, n_kv, kv (key -> value or
    array summary), header_bytes (= data offset, as N15 counts it) and the chat template bytes' sha256."""
    with open(path, "rb") as f:
        r = Reader(f)
        if r.take(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        version = r.u("I")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version %d" % version)
        n_tensors, n_kv = r.u("q"), r.u("q")
        if not (0 <= n_tensors <= MAX_COUNT and 0 <= n_kv <= MAX_COUNT):
            raise ValueError("implausible counts n_tensors=%d n_kv=%d" % (n_tensors, n_kv))
        kv, types, template = {}, {}, None
        for _ in range(n_kv):
            key = r.string().decode("utf-8", errors="replace")
            if key in kv:
                raise ValueError("duplicate key %r" % key)
            vt = r.u("I")
            types[key] = TYPE_NAMES.get(vt, str(vt))
            if vt in SCALAR:
                kv[key] = r.u(SCALAR[vt])
            elif vt == T_STRING:
                raw = r.string()
                if key == "tokenizer.chat_template":
                    template = raw
                    kv[key] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                else:
                    kv[key] = raw.decode("utf-8", errors="replace")
            elif vt == T_ARRAY:
                et, count = r.u("I"), r.u("Q")
                if count > MAX_COUNT:
                    raise ValueError("implausible array length %d for %s" % (count, key))
                h = hashlib.sha256()
                if et in SCALAR:
                    raw = r.take(count * struct.calcsize("<" + SCALAR[et]))
                    h.update(raw)
                    if count <= KEEP_ARRAY:
                        kv[key] = list(struct.unpack("<%d%s" % (count, SCALAR[et]), raw))
                        continue
                elif et == T_STRING:
                    for _ in range(count):
                        s = r.string()
                        h.update(struct.pack("<Q", len(s)))
                        h.update(s)
                else:
                    raise ValueError("array element type %d of %s is not allowed" % (et, key))
                kv[key] = {"elem_type": TYPE_NAMES.get(et, str(et)), "count": count, "sha256": h.hexdigest()}
            else:
                raise ValueError("unknown GGUF value type %d for %s" % (vt, key))
        alignment = kv.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("bad general.alignment %r" % (alignment,))
        for _ in range(n_tensors):
            r.string()
            nd = r.u("I")
            if nd > 4:
                raise ValueError("tensor with %d dimensions" % nd)
            r.take(8 * nd + 4 + 8)
        infos_end = r.pos
    data_offset = (infos_end + alignment - 1) // alignment * alignment if n_tensors > 0 else infos_end
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "kv": kv, "types": types,
            "alignment": alignment, "infos_end": infos_end, "header_bytes": data_offset,
            "chat_template_sha256": hashlib.sha256(template).hexdigest() if template is not None else None,
            "chat_template_bytes": len(template) if template is not None else None}


def facts(h):
    """The header facts this check reports (architecture-prefixed keys looked up under the file's own architecture)."""
    kv = h["kv"]
    arch = kv.get("general.architecture")
    a = (arch + ".") if isinstance(arch, str) else "?."
    clamp = kv.get(a + "swiglu_clamp_exp")
    return {
        "general.architecture": arch,
        "general.name": kv.get("general.name"),
        "general.file_type": kv.get("general.file_type"),
        "expert_count": kv.get(a + "expert_count"),
        "expert_used_count": kv.get(a + "expert_used_count"),
        "expert_group_count": kv.get(a + "expert_group_count"),
        "expert_group_used_count": kv.get(a + "expert_group_used_count"),
        "block_count": kv.get(a + "block_count"),
        "swiglu_clamp_exp_values": sorted(set(clamp)) if isinstance(clamp, list) else clamp,
        "split.no": kv.get("split.no"),
        "split.count": kv.get("split.count"),
        "split.tensors.count": kv.get("split.tensors.count"),
        "n_tensors": h["n_tensors"],
        "n_kv": h["n_kv"],
        "chat_template_sha256": h["chat_template_sha256"],
        "chat_template_bytes": h["chat_template_bytes"],
        "header_bytes": h["header_bytes"],
    }


def sha256_prefix(path, n, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = n
        while left > 0:
            b = f.read(min(bufsize, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def sha256_file(path, bufsize=8 << 20):
    h = hashlib.sha256()
    t0 = time.monotonic()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
            n += len(b)
    return path, h.hexdigest(), n, time.monotonic() - t0


def prepare_out(out, wdir):
    out_r, dir_r = os.path.realpath(out), os.path.realpath(wdir)
    if out_r == dir_r or out_r.startswith(dir_r.rstrip(os.sep) + os.sep):
        raise ValueError("--out %s is inside --dir %s: nothing may be written next to the weights" % (out, wdir))
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise ValueError("--out %s exists and is not an empty directory" % out)
    os.makedirs(out, exist_ok=True)


def check(wdir, pins, jobs=2, do_hash=True, log=print):
    files = pins["files"]
    present = sorted(os.listdir(wdir)) if os.path.isdir(wdir) else None
    if present is None:
        raise ValueError("--dir %s is not a directory" % wdir)
    pinned = {p["name"] for p in files}
    other = [n for n in present if n.endswith(".gguf") and n not in pinned]
    rows = []
    for p in files:
        path = os.path.join(wdir, p["name"])
        row = {"name": p["name"], "path": path, "pin_size": p["size"], "pin_sha256": p["sha256"]}
        if not os.path.isfile(path):
            row.update(status="MISSING")
            rows.append(row)
            continue
        row["size"] = os.stat(path).st_size
        row["size_ok"] = row["size"] == p["size"]
        try:
            h = read_header(path)
            row["header"] = facts(h)
            row["header_sha256"] = sha256_prefix(path, h["header_bytes"])
            row["header_ok"] = (h["header_bytes"] == p["header_bytes"] and row["header_sha256"] == p["header_sha256"])
        except (OSError, ValueError, struct.error) as e:
            row["header_error"] = "%s: %s" % (type(e).__name__, e)
            row["header_ok"] = False
        rows.append(row)
    todo = [r for r in rows if r.get("status") != "MISSING"]
    if do_hash and todo:
        log("hashing %d file(s), %d job(s) ..." % (len(todo), jobs))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
            for path, digest, n, secs in ex.map(sha256_file, [r["path"] for r in todo]):
                r = next(x for x in todo if x["path"] == path)
                r.update(sha256=digest, hashed_bytes=n, hash_seconds=round(secs, 1))
                log("  %s  %s  (%d bytes, %.1f s)" % (digest, r["name"], n, secs))
    for r in rows:
        if r.get("status") == "MISSING":
            continue
        if do_hash:
            r["status"] = "MATCH" if (r["size_ok"] and r["sha256"] == r["pin_sha256"]) else "MISMATCH"
        else:
            r["status"] = "SIZE+HEADER OK (sha256 not checked)" if (r["size_ok"] and r["header_ok"]) else "MISMATCH"
    first = next((r for r in rows if r["name"].endswith("-00001-of-00006.gguf") and "header" in r), None)
    expect = []
    for k, want in pins.get("expect_first", {}).items():
        got = first["header"].get(k) if first else None
        expect.append({"fact": k, "expected": want, "found": got, "ok": got == want})
    if first:
        expect.append({"fact": "chat template == vendored zai-org template (N14), informational",
                       "expected": ZAI_TEMPLATE_SHA256, "found": first["header"].get("chat_template_sha256"),
                       "ok": first["header"].get("chat_template_sha256") == ZAI_TEMPLATE_SHA256, "informational": True})
    if any(r["status"] in ("MISSING", "MISMATCH") for r in rows):
        verdict, code = "FAIL: %d of %d shard(s) missing or different from the pin" % (
            sum(r["status"] in ("MISSING", "MISMATCH") for r in rows), len(rows)), 1
    elif not do_hash:
        verdict, code = "NOT VERIFIED: sizes and headers match the pin, sha256 not computed (--no-hash)", 3
    else:
        verdict, code = "PASS: all %d shards match the pinned size and sha256" % len(rows), 0
    return {"tool": "tools/hostcheck/glm_gguf_check.py", "pins_source": pins.get("source"), "dir": os.path.abspath(wdir),
            "hashed": do_hash, "files": rows, "other_gguf_in_dir": other, "header_expectations": expect,
            "verdict": verdict}, code


def render(rep):
    L = ["glm_gguf_check: %s" % rep["verdict"], "pins: %s" % rep["pins_source"], "dir: %s" % rep["dir"], ""]
    for r in rep["files"]:
        L.append("%-10s %s" % (r["status"].split(" ")[0], r["name"]))
        if r["status"] == "MISSING":
            continue
        L.append("           size %d (pin %d)%s" % (r["size"], r["pin_size"], "" if r["size_ok"] else "  DIFFERS"))
        if "sha256" in r:
            L.append("           sha256 %s%s" % (r["sha256"], "" if r["sha256"] == r["pin_sha256"] else
                                                   "  DIFFERS (pin %s)" % r["pin_sha256"]))
        if "header_error" in r:
            L.append("           header: %s" % r["header_error"])
        else:
            hd = r["header"]
            L.append("           header %s; architecture %r, split %s/%s, n_tensors %s" % (
                "== pin" if r["header_ok"] else "DIFFERS from pin", hd["general.architecture"], hd["split.no"],
                hd["split.count"], hd["n_tensors"]))
    L.append("")
    L.append("header facts of shard 1 (expected = the pinned Hugging Face file):")
    for e in rep["header_expectations"]:
        L.append("  %-4s %s: found %r, expected %r%s" % ("ok" if e["ok"] else "DIFF", e["fact"], e["found"], e["expected"],
                                                         " (informational)" if e.get("informational") else ""))
    if rep["other_gguf_in_dir"]:
        L.append("other .gguf files in the directory (not checked): %s" % ", ".join(rep["other_gguf_in_dir"]))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory with the six GLM-5.3-Flash-UD-Q4_K_XL-*-of-00006.gguf")
    ap.add_argument("--out", required=True, help="new or empty output directory, outside --dir")
    ap.add_argument("--jobs", type=int, default=2, help="files hashed in parallel (default 2)")
    ap.add_argument("--no-hash", action="store_true", help="sizes and headers only (exit 3, never a pass)")
    ap.add_argument("--pins", help="JSON with the same layout as PINS (tests); default: the embedded N15 pins")
    a = ap.parse_args(argv)
    try:
        pins = PINS
        if a.pins:
            with open(a.pins) as f:
                pins = json.load(f)
        prepare_out(a.out, a.dir)
        rep, code = check(a.dir, pins, a.jobs, not a.no_hash, log=lambda s: print(s, flush=True))
    except (OSError, ValueError) as e:
        print("glm_gguf_check: error: %s" % e, file=sys.stderr)
        return 2
    text = render(rep)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(a.out, "sha256sum.txt"), "w") as f:
        f.write("".join("%s  %s\n" % (r["sha256"], r["path"]) for r in rep["files"] if "sha256" in r))
    sys.stdout.write(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
