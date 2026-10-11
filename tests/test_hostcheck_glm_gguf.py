"""tools/hostcheck/glm_gguf_check.py on SYNTHETIC GGUF shards written here (stdlib only, no GPU, no network).

EVERY FILE HERE IS SYNTHETIC TEST DATA, NOT EVIDENCE: six small GGUF v3 shards named like the epyc-ai files, with a
header like the host's (architecture string 'glm5next', the converter dialect AGENTS.md names) or like the pinned
Hugging Face file ('glm5-next'), and pins computed from these files. The check against the real epyc-ai files is NOT RUN.
Run: python3 -m unittest tests.test_hostcheck_glm_gguf -v   (from the repo root)
"""
import ast
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "hostcheck", "glm_gguf_check.py")
WRAPPER = os.path.join(ROOT, "tools", "hostcheck", "glm_gguf_check.sh")
sys.path.insert(0, os.path.dirname(TOOL))
import glm_gguf_check as gc  # noqa: E402

TEMPLATE = "[gMASK]<sop>{% for m in messages %}SYNTHETIC {{ m.content }}{% endfor %}"


def _s(b):
    return struct.pack("<Q", len(b)) + b


def kv_bytes(key, vtype, value):
    out = _s(key.encode()) + struct.pack("<I", vtype)
    if vtype == 8:
        return out + _s(value.encode())
    if vtype == 9:
        et, vals = value
        out += struct.pack("<IQ", et, len(vals))
        if et == 8:
            return out + b"".join(_s(v.encode()) for v in vals)
        return out + struct.pack("<%d%s" % (len(vals), gc.SCALAR[et]), *vals)
    return out + struct.pack("<" + gc.SCALAR[vtype], value)


def write_shard(path, kvs, tensors=(), data=b""):
    """GGUF v3: kvs = [(key, type, value)], tensors = [(name, dims, ggml_type, nbytes)]; data appended after padding."""
    head = b"GGUF" + struct.pack("<Iqq", 3, len(tensors), len(kvs)) + b"".join(kv_bytes(*kv) for kv in kvs)
    off = 0
    for name, dims, tt, nb in tensors:
        head += _s(name.encode()) + struct.pack("<I", len(dims)) + struct.pack("<%dq" % len(dims), *dims)
        head += struct.pack("<iQ", tt, off)
        off += (nb + 31) // 32 * 32
    if tensors:
        head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head + data)
    return len(head)


def make_set(d, arch="glm5next", template=TEMPLATE, groups=1):
    os.makedirs(d, exist_ok=True)
    pins = {"source": "SYNTHETIC test pins", "files": [], "expect_first": {
        "general.architecture": arch, "expert_group_count": groups, "expert_group_used_count": 1,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(), "split.count": 6}}
    for i in range(1, 7):
        name = "GLM-5.3-Flash-UD-Q4_K_XL-%05d-of-00006.gguf" % i
        split = [("split.no", 2, i - 1), ("split.count", 2, 6), ("split.tensors.count", 5, 5)]
        if i == 1:
            kvs = [("general.architecture", 8, arch), ("general.name", 8, "SYNTHETIC GLM"),
                   (arch + ".expert_count", 4, 288), (arch + ".expert_group_count", 4, groups),
                   (arch + ".expert_group_used_count", 4, 1), (arch + ".swiglu_clamp_exp", 9, (6, [10.0] * 46)),
                   ("tokenizer.ggml.tokens", 9, (8, ["a", "b", "SYNTHETIC"])),
                   ("tokenizer.chat_template", 8, template)] + split
            hb = write_shard(os.path.join(d, name), kvs)
        else:
            data = bytes((i * 7 + k) % 251 for k in range(256))
            hb = write_shard(os.path.join(d, name), split, [("blk.%d.ffn_up_exps.weight" % i, [32, 2], 0, 256)], data)
        p = os.path.join(d, name)
        with open(p, "rb") as f:
            body = f.read()
        pins["files"].append({"name": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                              "header_bytes": hb, "header_sha256": hashlib.sha256(body[:hb]).hexdigest()})
    return pins


def snapshot(d):
    out = {}
    for dp, _, fns in os.walk(d):
        for fn in fns:
            p = os.path.join(dp, fn)
            st = os.stat(p)
            with open(p, "rb") as f:
                out[os.path.relpath(p, d)] = (st.st_size, st.st_mtime_ns, hashlib.sha256(f.read()).hexdigest())
    return out


class HostCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hostcheck-SYNTHETIC-")
        self.dir = os.path.join(self.tmp, "models", "UD-Q4_K_XL")
        self.pins = make_set(self.dir)
        self.pins_path = os.path.join(self.tmp, "pins-SYNTHETIC.json")
        with open(self.pins_path, "w") as f:
            json.dump(self.pins, f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self, *extra, out="out", pins=True):
        out = os.path.join(self.tmp, out)
        argv = ["--dir", self.dir, "--out", out] + (["--pins", self.pins_path] if pins else []) + list(extra)
        p = subprocess.run([sys.executable, "-I", TOOL] + argv, capture_output=True, text=True, timeout=120)
        rep = None
        if os.path.isfile(os.path.join(out, "report.json")):
            with open(os.path.join(out, "report.json")) as f:
                rep = json.load(f)
        return p.returncode, p.stdout + p.stderr, rep, out

    def test_matching_set_passes_and_nothing_outside_out_changes(self):
        before = snapshot(self.tmp)
        code, text, rep, out = self.run_tool("--jobs", "3")
        self.assertEqual(code, 0, text)
        self.assertTrue(rep["verdict"].startswith("PASS"), rep["verdict"])
        self.assertEqual([r["status"] for r in rep["files"]], ["MATCH"] * 6)
        self.assertTrue(all(r["header_ok"] for r in rep["files"]))
        self.assertTrue(all(e["ok"] for e in rep["header_expectations"] if not e.get("informational")))
        after = snapshot(self.tmp)
        new = sorted(set(after) - set(before))
        self.assertEqual(new, ["out/report.json", "out/report.txt", "out/sha256sum.txt"])
        self.assertEqual({k: v for k, v in after.items() if k in before}, before)   # weights untouched
        with open(os.path.join(out, "sha256sum.txt")) as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 6)
        p = subprocess.run(["sha256sum", "-c", os.path.join(out, "sha256sum.txt")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_header_facts_of_the_first_shard(self):
        code, text, rep, _ = self.run_tool("--no-hash")
        h = rep["files"][0]["header"]
        self.assertEqual(h["general.architecture"], "glm5next")
        self.assertEqual(h["expert_group_count"], 1)
        self.assertEqual(h["expert_count"], 288)
        self.assertEqual(h["swiglu_clamp_exp_values"], [10.0])
        self.assertEqual(h["chat_template_sha256"], hashlib.sha256(TEMPLATE.encode()).hexdigest())
        self.assertEqual(h["chat_template_bytes"], len(TEMPLATE.encode()))
        self.assertEqual((h["split.no"], h["split.count"]), (0, 6))
        self.assertEqual(rep["files"][1]["header"]["n_tensors"], 1)
        self.assertEqual(code, 3, text)          # --no-hash is never a pass
        self.assertTrue(rep["verdict"].startswith("NOT VERIFIED"))

    def test_one_flipped_data_byte_is_a_mismatch_with_the_header_intact(self):
        p = os.path.join(self.dir, self.pins["files"][3]["name"])
        with open(p, "r+b") as f:
            f.seek(-10, 2)
            b = f.read(1)
            f.seek(-10, 2)
            f.write(bytes([b[0] ^ 1]))
        code, text, rep, _ = self.run_tool()
        self.assertEqual(code, 1, text)
        r = rep["files"][3]
        self.assertEqual(r["status"], "MISMATCH")
        self.assertTrue(r["size_ok"] and r["header_ok"])
        self.assertIn("DIFFERS (pin", text)

    def test_missing_shard_and_other_files(self):
        os.remove(os.path.join(self.dir, self.pins["files"][5]["name"]))
        open(os.path.join(self.dir, "mtp-SYNTHETIC.gguf"), "wb").close()
        code, text, rep, _ = self.run_tool()
        self.assertEqual(code, 1, text)
        self.assertEqual(rep["files"][5]["status"], "MISSING")
        self.assertEqual(rep["other_gguf_in_dir"], ["mtp-SYNTHETIC.gguf"])

    def test_embedded_pins_against_host_dialect_files(self):
        """Default pins (the Hugging Face file): sizes differ and the 'glm5next' architecture string is reported."""
        code, text, rep, _ = self.run_tool("--no-hash", pins=False)
        self.assertEqual(code, 1, text)
        self.assertEqual(rep["pins_source"], gc.PIN_SOURCE)
        arch = next(e for e in rep["header_expectations"] if e["fact"] == "general.architecture")
        self.assertEqual((arch["found"], arch["expected"], arch["ok"]), ("glm5next", "glm5-next", False))

    def test_out_must_be_new_or_empty_and_outside_the_weights(self):
        before = snapshot(self.tmp)
        code, text, rep, _ = self.run_tool(out=os.path.join("models", "UD-Q4_K_XL", "out"))
        self.assertEqual(code, 2, text)
        self.assertIn("inside --dir", text)
        self.assertEqual(snapshot(self.tmp), before)
        os.makedirs(os.path.join(self.tmp, "busy"))
        open(os.path.join(self.tmp, "busy", "x"), "w").close()
        code, text, rep, _ = self.run_tool(out="busy")
        self.assertEqual(code, 2, text)
        self.assertIn("not an empty directory", text)

    def test_not_a_gguf_file(self):
        p = os.path.join(self.dir, self.pins["files"][0]["name"])
        with open(p, "r+b") as f:
            f.write(b"XXXX")
        code, text, rep, _ = self.run_tool("--no-hash")
        self.assertEqual(code, 1, text)
        self.assertIn("bad magic", rep["files"][0]["header_error"])

    def test_wrapper_runs_isolated(self):
        out = os.path.join(self.tmp, "wout")
        p = subprocess.run(["bash", WRAPPER, "--dir", self.dir, "--out", out, "--pins", self.pins_path],
                           capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(WRAPPER) as f:
            self.assertIn('exec python3 -I "$HERE/glm_gguf_check.py"', f.read())


class Static(unittest.TestCase):
    def test_embedded_pins(self):
        f = gc.PINS["files"]
        self.assertEqual([x["name"] for x in f],
                         ["GLM-5.3-Flash-UD-Q4_K_XL-%05d-of-00006.gguf" % i for i in range(1, 7)])
        self.assertEqual(sum(x["size"] for x in f), 199707321472)       # 185.99 GiB of files (N15 PIN.json)
        for x in f:
            self.assertRegex(x["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(x["header_sha256"], r"^[0-9a-f]{64}$")
            self.assertLess(x["header_bytes"], x["size"])
        self.assertEqual(gc.PINS["expect_first"]["general.architecture"], "glm5-next")

    def test_stdlib_only_no_network(self):
        with open(TOOL) as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add((n.module or "").split(".")[0])
        self.assertLessEqual(mods, {"argparse", "concurrent", "hashlib", "json", "os", "struct", "sys", "time"}, mods)
        with open(WRAPPER) as fh:
            sh = fh.read()
        for word in ("curl", "wget", "http", "rm ", "dd ", "mv "):
            self.assertNotIn(word, sh)

    def test_weights_are_opened_read_only(self):
        with open(TOOL) as fh:
            src = fh.read()
        opens = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"]
        modes = []
        for n in opens:
            m = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else "r"
            target = ast.unparse(n.args[0])
            modes.append((target, m))
        writes = [t for t, m in modes if "w" in m or "a" in m or "+" in m]
        # every write goes to a path joined under a.out
        self.assertTrue(writes)
        for t in writes:
            self.assertTrue(t.startswith("os.path.join(a.out"), t)


if __name__ == "__main__":
    unittest.main()
