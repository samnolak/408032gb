"""tools/verify/pack.py: verification-pack assembler, tested on SYNTHETIC artifacts produced by the real tools.

EVERY FIXTURE HERE IS SYNTHETIC TEST DATA, NOT EVIDENCE. The fixtures are generated at test time in a temporary
directory whose name contains "SYNTHETIC", by calling the real tools so their formats cannot drift:
  * tools/bench/run_matrix.py and tools/bench/probes.py against tools/bench/mock_server.py (in-process);
    "unmocked" copies (is_mock_server false, owned_by relabelled, banner replaced by a SYNTHETIC banner) exercise
    the PASS path; the untouched mock runs exercise the "mock -> NOT RUN" rule;
  * tools/gguf_quant_audit.audit() on header-only GGUF files written here;
  * tools/quality/logprob_compare.py and tools/kv/slot_persist.py against small stdlib stub servers in this file;
  * tools/bench/vs_roofline.py --json on the generated matrix runs;
  * scripts/diag/summarize.py --json on copies of tests/fixtures/diag (themselves SYNTHETIC fixtures);
  * scripts/offline/weights.py manifest (build_manifest on a SYNTHETIC HF tree listing) and verify on small files.
The small-scale criteria (ctx_tokens 4096 instead of 262144, ...) are a SYNTHETIC criteria file; the pack marks
every deviation from tools/verify/criteria.default.json in verdict.json and VERDICT.md.
Run: python3 -m unittest tests.test_verify_pack -v   (from the repo root)
"""
import contextlib
import hashlib
import http.server
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (("tools", "verify"), ("tools", "bench"), ("tools",), ("tools", "kv"), ("tools", "quality"), ("scripts", "diag"),
           ("scripts", "offline")):
    sys.path.insert(0, os.path.join(ROOT, *_p))
import pack as vp  # noqa: E402  (tools/verify/pack.py)
import gguf_quant_audit as qa  # noqa: E402
import logprob_compare  # noqa: E402
import mock_server  # noqa: E402
import probes  # noqa: E402
import run_matrix  # noqa: E402
import runlog  # noqa: E402
import slot_persist  # noqa: E402
import summarize  # noqa: E402  (scripts/diag/summarize.py)
import vs_roofline  # noqa: E402
import weights  # noqa: E402  (scripts/offline/weights.py)

for _m, _d in ((vp, ("tools", "verify")), (run_matrix, ("tools", "bench")), (summarize, ("scripts", "diag")),
               (slot_persist, ("tools", "kv")), (logprob_compare, ("tools", "quality"))):
    assert os.path.dirname(os.path.abspath(_m.__file__)) == os.path.join(ROOT, *_d), _m.__file__

# SYNTHETIC small-scale criteria: the same rules at 4096 tokens so the mock runs stay fast.
SMALL = {"ctx_tokens": 4096, "decode_min_output_tokens": 32, "agent_loop_min_prompt_tokens": 3000,
         "kv_min_prompt_tokens": 3000, "decode_target_tok_s": 1000.0}
DESC_ENGINE = "SYNTHETIC test engine: tools/bench/mock_server.py, NOT EVIDENCE"
GGUF_NAME = "SYNTHETIC-model-Q4_K.gguf"
DESC_MODEL = "SYNTHETIC test model %s, NOT EVIDENCE" % GGUF_NAME
MODEL_ID = "SYNTHETIC-mock-model"
PREFILL = 0.1          # mock seconds per 1000 uncached prompt tokens
DECODE_OK = 0.0125     # <= 80 tok/s: above the 50 floor, below the qwen3.8 roofline bound (~100 at 4K)
DECODE_SLOW = 0.025    # <= 40 tok/s: below the 50 floor
FIX = None             # module temp dir
G = {}                 # generated artifacts: name -> path


# ---------------------------------------------------------------------------- stub servers (stdlib)
def _serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def _reply(h, code, obj):
    out = json.dumps(obj).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(out)))
    h.end_headers()
    h.wfile.write(out)


def logprob_handler(flip=False):
    """/v1/completions with top_logprobs; deterministic per prompt. flip=True swaps the two best tokens."""
    vocab = ["t%d" % i for i in range(40)]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            h = hashlib.sha256(body["prompt"].encode()).digest()
            logits = [((h[i % 32] * (i + 3)) % 89) / 7.0 + i * 1e-4 for i in range(len(vocab))]
            if flip:
                o = sorted(range(len(vocab)), key=lambda i: -logits[i])
                logits[o[0]], logits[o[1]] = logits[o[1]], logits[o[0]]
            m = max(logits)
            lse = m + math.log(sum(math.exp(x - m) for x in logits))
            lp = sorted(((vocab[i], logits[i] - lse) for i in range(len(vocab))), key=lambda x: -x[1])
            top = dict(lp[:int(body.get("logprobs") or 5)])
            best = lp[0][0]
            _reply(self, 200, {"choices": [{"index": 0, "text": best, "finish_reason": "length", "logprobs": {
                "tokens": [best], "token_logprobs": [top[best]], "top_logprobs": [top]}}]})
    return H


def slot_handler(reuse=True):
    """llama-server-like /completion + /slots/0?action=save|restore|erase. Timings are SYNTHETIC: 2 ms per prompt token."""
    st = {"cache": None, "files": {}}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            u = urlparse(self.path)
            if u.path == "/completion":
                n = len(body["prompt"].split())
                processed = 1 if (reuse and st["cache"] == body["prompt"]) else n
                st["cache"] = body["prompt"]
                return _reply(self, 200, {"content": "SYNTHETIC continuation of %d words" % n, "tokens_cached": n,
                                          "timings": {"prompt_n": processed, "prompt_ms": processed * 2.0,
                                                      "predicted_n": body["n_predict"], "predicted_ms": 5.0}})
            action = parse_qs(u.query).get("action", [""])[0]
            if action == "erase":
                st["cache"] = None
                return _reply(self, 200, {"id_slot": 0, "n_erased": 1})
            if action == "save":
                st["files"][body["filename"]] = st["cache"]
                return _reply(self, 200, {"id_slot": 0, "filename": body["filename"], "n_saved": 1,
                                          "n_written": 123456, "timings": {"save_ms": 1.0}})
            if action == "restore":
                st["cache"] = st["files"].get(body["filename"])
                return _reply(self, 200, {"id_slot": 0, "n_restored": 1, "n_read": 123456,
                                          "timings": {"restore_ms": 0.5}})
            return _reply(self, 404, {"error": "not found"})
    return H


# ---------------------------------------------------------------------------- fixture generation
def quiet(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    return code, buf.getvalue()


def _type_id(name):
    return next(k for k, v in qa.GGML_TYPES.items() if v[0] == name)


def write_gguf(path, tensors):
    """Header-only GGUF v3 (the auditor never reads tensor data). tensors: (name, dims, type name). SYNTHETIC."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    out += s("general.architecture") + struct.pack("<I", 8) + s("synthetic-arch")
    for i, (name, dims, tname) in enumerate(tensors):
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", _type_id(tname), i * 4096)
    with open(path, "wb") as f:
        f.write(out)


GOOD_TENSORS = [("token_embd.weight", [4096, 1000], "Q8_0"), ("output.weight", [4096, 1000], "Q6_K"),
                ("blk.0.attn_q.weight", [4096, 4096], "Q8_0"), ("blk.0.attn_norm.weight", [4096], "F32"),
                ("blk.1.ffn_gate_exps.weight", [4096, 2048, 64], "Q4_K"),
                ("blk.1.ffn_up_exps.weight", [4096, 2048, 64], "Q4_K"),
                ("blk.1.ffn_down_exps.weight", [2048, 4096, 64], "Q5_K"),
                ("blk.1.ffn_up_shexp.weight", [4096, 2048], "Q8_0")]


def audit_json(name, tensors, **kw):
    gguf = os.path.join(FIX, "gguf-" + name, GGUF_NAME)
    os.makedirs(os.path.dirname(gguf))
    write_gguf(gguf, tensors)
    p = os.path.join(FIX, "gen", "quant_audit.%s.SYNTHETIC.json" % name)
    with open(p, "w") as f:                     # == tools/gguf_quant_audit.py --json (json.dump indent=1 + newline)
        json.dump(qa.audit(gguf, **kw), f, indent=1)
        f.write("\n")
    return p


def unmock(src, dst):
    """Relabel a mock run as non-mock (SYNTHETIC) so the PASS path can be exercised with real-format data."""
    shutil.copytree(src, dst)
    mp, sp, mdp = (os.path.join(dst, n) for n in ("meta.json", "summary.json", "summary.md"))
    with open(mp) as f:
        meta = json.load(f)
    meta["is_mock_server"] = False
    for m in meta["models_response"]["json"]["data"]:
        m["owned_by"] = "SYNTHETIC-relabelled-not-mock"
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(sp) as f:
        summ = json.load(f)
    summ["is_mock_server"] = False
    with open(sp, "w") as f:
        json.dump(summ, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(mdp) as f:
        md = f.read()
    assert vp.MOCK_BANNER in md
    md = md.replace("**MOCK SERVER RUN: these numbers test the harness only; they are not engine evidence.**",
                    "**SYNTHETIC TEST FIXTURE: mock output relabelled as non-mock to exercise the PASS path; NOT EVIDENCE.**")
    with open(mdp, "w") as f:
        f.write(md)
    return dst


def hw_json(src_fixture, name, synthetic_flag):
    d = os.path.join(FIX, "diag-" + name)
    shutil.copytree(os.path.join(ROOT, "tests", "fixtures", "diag", src_fixture), d)
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    man["synthetic"] = synthetic_flag
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    out = os.path.join(FIX, "gen", "hw.%s.SYNTHETIC.json" % name)
    code, _ = quiet(summarize.main, [d, "--output", os.path.join(d, "HARDWARE_FACTS.draft.md"), "--json", out])
    assert code == 0
    return out


def roofline_json(run_dir, name, spec="qwen3.8-flash-next.json", hit="1.0"):
    code, out = quiet(vs_roofline.main, [run_dir, "--spec", os.path.join(ROOT, "models", "specs", spec),
                                         "--hit", hit, "--strategy", "pp", "--json"])
    assert code == 0
    p = os.path.join(FIX, "gen", "vs_roofline.%s.SYNTHETIC.json" % name)
    with open(p, "w") as f:
        f.write(out)
    return p


def weights_fixture():
    """SYNTHETIC weights dir + manifest (weights.build_manifest on a fake HF tree) + `weights.py verify` stdout."""
    dest = os.path.join(FIX, "weights-SYNTHETIC")
    os.makedirs(dest)
    entries = []
    for name, lfs in ((GGUF_NAME, True), ("README.md", False)):
        data = ("SYNTHETIC bytes of %s, NOT EVIDENCE\n" % name).encode() * 64
        with open(os.path.join(dest, name), "wb") as f:
            f.write(data)
        e = {"type": "file", "path": name, "size": len(data), "oid": hashlib.sha1(data).hexdigest()}
        if lfs:
            e["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
        entries.append(e)
    man = weights.build_manifest("SYNTHETIC/not-a-repo", "0" * 40, entries, "*")
    G["wman"] = os.path.join(FIX, "gen", "weights.manifest.SYNTHETIC.json")
    with open(G["wman"], "w") as f:                 # == weights.py manifest -o (json.dump indent=1)
        json.dump(man, f, indent=1)
    for key in ("wlog", "wlog_bad"):
        if key == "wlog_bad":
            with open(os.path.join(dest, GGUF_NAME), "ab") as f:
                f.write(b"corrupted")
        code, out = quiet(weights.main, ["verify", G["wman"], "--dest", dest])
        assert code == (0 if key == "wlog" else 1), (key, code, out)
        G[key] = os.path.join(FIX, "gen", "weights_verify.%s.SYNTHETIC.txt" % key)
        with open(G[key], "w") as f:
            f.write(out)


def setUpModule():
    global FIX
    FIX = tempfile.mkdtemp(prefix="verify-pack-SYNTHETIC-")
    try:
        _generate()
    except BaseException:
        shutil.rmtree(FIX, ignore_errors=True)    # unittest skips tearDownModule when setUpModule fails
        raise


def _generate():
    gen = os.path.join(FIX, "gen")
    os.makedirs(gen)
    srv = mock_server.MockServer(model_id=MODEL_ID, prefill_s_per_1k=PREFILL, decode_s_per_token=DECODE_OK).start()
    G["mock_url"] = srv.url
    try:
        common = ["--url", srv.url, "--engine-desc", DESC_ENGINE, "--model-desc", DESC_MODEL, "--no-nvidia-smi"]
        G["mock_matrix"] = os.path.join(gen, "matrix-filled-SYNTHETIC")
        code, _ = quiet(run_matrix.main, common + [
            "--inputs", "4096", "--outputs", "32", "--states", "cold,warm", "--concurrency", "1", "--repeats", "3",
            "--length-mode", "total", "--scenarios", "matrix,agent", "--agent-target-tokens", "3000",
            "--agent-tool-tokens", "300,600", "--out-dir", G["mock_matrix"]])
        assert code == 0, code
        G["mock_probes"] = os.path.join(gen, "probes-SYNTHETIC")
        code, _ = quiet(probes.main, common + ["--contexts", "4096", "--length-mode", "total", "--answer-tokens", "256",
                                               "--out-dir", G["mock_probes"]])
        assert code == 0, code
        srv.reset(model_id=MODEL_ID, prefill_s_per_1k=PREFILL, decode_s_per_token=0.001, nondeterministic=True)
        G["mock_probes_nondet"] = os.path.join(gen, "probes-nondet-SYNTHETIC")
        code, _ = quiet(probes.main, common + ["--probes", "json,tools,determinism", "--out-dir", G["mock_probes_nondet"]])
        assert code == 0, code                    # determinism is info-only: probes.py still exits 0
        srv.reset(model_id=MODEL_ID, prefill_s_per_1k=PREFILL, decode_s_per_token=0.001)
        G["mock_short"] = os.path.join(gen, "matrix-short-SYNTHETIC")
        code, _ = quiet(run_matrix.main, common + [
            "--inputs", "1024", "--outputs", "32", "--states", "cold,warm", "--concurrency", "1", "--repeats", "3",
            "--length-mode", "total", "--scenarios", "matrix", "--out-dir", G["mock_short"]])
        assert code == 0, code
        srv.reset(model_id=MODEL_ID, prefill_s_per_1k=PREFILL, decode_s_per_token=DECODE_SLOW)
        G["mock_slow"] = os.path.join(gen, "matrix-slow-SYNTHETIC")
        code, _ = quiet(run_matrix.main, common + [
            "--inputs", "4096", "--outputs", "32", "--states", "cold", "--concurrency", "1", "--repeats", "3",
            "--length-mode", "total", "--scenarios", "matrix", "--out-dir", G["mock_slow"]])
        assert code == 0, code
        # real tools against the MOCK url: their artifacts carry no mock flag, the pack links them by URL
        corpus = os.path.join(FIX, "corpus-SYNTHETIC.txt")
        with open(corpus, "w") as f:
            f.write(" ".join("SYNTHETIC corpus sentence number %d for prefix cuts." % i for i in range(400)))
        G["corpus"] = corpus
        lp_srv, lp_url = _serve(logprob_handler())
        G["lp_mock"] = os.path.join(gen, "logprob_compare.vs-mock.SYNTHETIC.json")
        quiet(logprob_compare.main, ["--ref", lp_url, "--cand", srv.url, "--corpus", corpus, "--prefixes", "5",
                                     "--out", G["lp_mock"]])
        prompt = os.path.join(FIX, "slot-prompt-SYNTHETIC.txt")
        with open(prompt, "w") as f:
            f.write(" ".join("w%d" % i for i in range(3200)))
        G["slot_prompt"] = prompt
        G["slot_mock"] = os.path.join(gen, "slot_persist.vs-mock.SYNTHETIC.json")
        quiet(slot_persist.main, ["--url", srv.url, "--prompt-file", prompt, "--gen", "8", "--repeats", "1",
                                  "--timeout", "10", "--out", G["slot_mock"]])
    finally:
        srv.stop()
    G["roofline_mock"] = roofline_json(G["mock_matrix"], "mock")
    for k in ("matrix", "probes", "short", "slow", "probes_nondet"):
        G[k] = unmock(G["mock_" + k], os.path.join(gen, "unmocked", os.path.basename(G["mock_" + k])))
    G["roofline"] = roofline_json(G["matrix"], "ok")
    G["roofline_slow"] = roofline_json(G["slow"], "slow")
    G["roofline_glm"] = roofline_json(G["matrix"], "glm-bound-below-measured", spec="glm-5.3-flash.json", hit="0.7")
    weights_fixture()
    # logprob compare: identical engines (top-1 = 1.0) and a flipped candidate (top-1 = 0.0)
    cand_srv, cand_url = _serve(logprob_handler())
    flip_srv, flip_url = _serve(logprob_handler(flip=True))
    G["lp_ok"] = os.path.join(gen, "logprob_compare.ok.SYNTHETIC.json")
    G["lp_flip"] = os.path.join(gen, "logprob_compare.flip.SYNTHETIC.json")
    G["lp_same"] = os.path.join(gen, "logprob_compare.same.SYNTHETIC.json")
    assert quiet(logprob_compare.main, ["--ref", lp_url, "--cand", cand_url, "--corpus", G["corpus"], "--out", G["lp_ok"]])[0] == 0
    assert quiet(logprob_compare.main, ["--ref", lp_url, "--cand", flip_url, "--corpus", G["corpus"], "--out", G["lp_flip"]])[0] == 0
    assert quiet(logprob_compare.main, ["--ref", lp_url, "--cand", lp_url, "--corpus", G["corpus"], "--out", G["lp_same"]])[0] == 0
    for s in (lp_srv, cand_srv, flip_srv):
        s.shutdown()
        s.server_close()
    # KV slot persistence
    for name, reuse in (("ok", True), ("noreuse", False)):
        s, url = _serve(slot_handler(reuse))
        G["slot_" + name] = os.path.join(gen, "slot_persist.%s.SYNTHETIC.json" % name)
        assert quiet(slot_persist.main, ["--url", url, "--prompt-file", G["slot_prompt"], "--gen", "8", "--repeats", "3",
                                         "--out", G["slot_" + name]])[0] == 0
        s.shutdown()
        s.server_close()
    # quant audits
    bad = [t if t[0] != "blk.1.ffn_up_exps.weight" else (t[0], t[1], "IQ3_S") for t in GOOD_TENSORS]
    low_dense = [t if t[0] != "blk.0.attn_q.weight" else (t[0], t[1], "Q3_K") for t in GOOD_TENSORS]
    G["quant_ok"] = audit_json("ok", GOOD_TENSORS)
    G["quant_bad"] = audit_json("bad", bad)
    G["quant_bad_weak"] = audit_json("bad-weak", bad, min_expert_bpw=3.0)
    G["quant_ok_weak"] = audit_json("ok-weak", GOOD_TENSORS, min_expert_bpw=3.0)
    G["quant_dense_warn"] = audit_json("dense-warn", low_dense, dense_warn_only=True)
    # hardware facts
    G["hw_ok"] = hw_json("synthetic-full", "flag-false", False)
    G["hw_flagged"] = hw_json("synthetic-full", "flag-true", True)
    G["hw_nonroot"] = hw_json("synthetic-nonroot", "nonroot", False)
    with open(os.path.join(FIX, "criteria-SYNTHETIC-small.json"), "w") as f:
        json.dump(SMALL, f)
    G["small"] = os.path.join(FIX, "criteria-SYNTHETIC-small.json")


def tearDownModule():
    if FIX:
        shutil.rmtree(FIX, ignore_errors=True)


PASSING = {"matrix": "matrix", "probes": "probes", "quant": "quant_ok", "logprob": "lp_ok", "slot": "slot_ok",
           "roofline": "roofline", "hw": "hw_ok", "wman": "wman", "wlog": "wlog"}
BENCH_IDS = ["single_configuration", "openai_api", "context_probes_full_length", "output_after_full_prompt",
             "agent_api_probes", "decode_filled_ctx_floor", "prefix_cache", "agent_loop"]


class PackCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="case-SYNTHETIC-", dir=FIX)

    def make_runs(self, **parts):
        """parts: role -> G key (or list of G keys for matrix/probes). Builds a --runs directory."""
        runs = os.path.join(self.tmp, "runs")
        os.makedirs(runs)
        with open(os.path.join(runs, "SYNTHETIC_FIXTURE_README.txt"), "w") as f:
            f.write("SYNTHETIC TEST FIXTURE - NOT EVIDENCE (tests/test_verify_pack.py)\n")
        for role, keys in parts.items():
            if keys is None:
                continue
            for key in ([keys] if isinstance(keys, str) else keys):
                src = G[key]
                if role in ("matrix", "probes"):
                    shutil.copytree(src, os.path.join(runs, role, os.path.basename(src)))
                else:
                    sub = os.path.join(runs, "hw") if role == "hw" else runs
                    os.makedirs(sub, exist_ok=True)
                    shutil.copy(src, os.path.join(sub, os.path.basename(src)))
        return runs

    def build(self, runs, criteria="small", name="pack"):
        out = os.path.join(self.tmp, name)
        argv = ["--runs", runs, "--out", out] + (["--criteria", G["small"]] if criteria == "small" else [])
        code, text = quiet(vp.main, argv)
        with open(os.path.join(out, "verdict.json")) as f:
            v = json.load(f)
        return code, v, out, text

    @staticmethod
    def by_id(v):
        return {r["id"]: r for r in v["criteria_results"]}


class Criteria(unittest.TestCase):
    def test_default_criteria_hold_the_customer_numbers(self):
        crit, dev = vp.load_criteria()
        self.assertEqual(dev, {})
        self.assertEqual(crit["ctx_tokens"], 262144)
        self.assertEqual(crit["decode_floor_tok_s"], 50.0)
        self.assertEqual(crit["decode_target_tok_s"], 80.0)
        self.assertEqual(crit["min_bpw"], 4.0)
        self.assertEqual(crit["decode_min_output_tokens"], 512)
        self.assertTrue(0.9 < crit["top1_agreement_min"] <= 1.0)
        self.assertTrue(crit["_notes"]["top1_agreement_min"].startswith("PROVISIONAL"))
        self.assertEqual(set(crit["mandatory"]), set(vp.CRITERIA_IDS) - {"decode_filled_ctx_target", "determinism",
                                                                          "manual_quality_eval",
                                                                          "manual_operational_faults"})

    def test_unknown_criteria_key_is_an_error(self):
        p = os.path.join(FIX, "criteria-SYNTHETIC-typo.json")
        with open(p, "w") as f:
            json.dump({"decode_flor_tok_s": 1.0}, f)
        with self.assertRaises(ValueError):
            vp.load_criteria(p)


class Verdicts(PackCase):
    def test_all_missing_every_criterion_not_run(self):
        runs = self.make_runs()
        for crit in ("default", "small"):
            code, v, out, _ = self.build(runs, crit, "pack-" + crit)
            self.assertEqual(code, 3)
            self.assertEqual(v["overall"], "NOT RUN")
            for r in v["criteria_results"]:
                self.assertEqual(r["status"], "NOT RUN", r["id"])
            for fn in ("MANIFEST.sha256", "env.txt", "verdict.json", "VERDICT.md", "criteria.json",
                       "artifacts/SYNTHETIC_FIXTURE_README.txt"):
                self.assertTrue(os.path.isfile(os.path.join(out, fn)), fn)
            with open(os.path.join(out, "env.txt")) as f:
                env = f.read()
            for cmd in ("$ git rev-parse HEAD", "$ git status --short", "$ uname -a", "$ python3 --version",
                        "$ nvidia-smi -L"):
                self.assertIn(cmd, env)
            self.assertTrue("nvidia-smi: not present" in env or "GPU" in env)
            with open(os.path.join(out, "VERDICT.md")) as f:
                self.assertIn("**Итог: NOT RUN**", f.read())

    def test_full_synthetic_passing_set_passes(self):
        runs = self.make_runs(**PASSING)
        code, v, out, text = self.build(runs)
        res = self.by_id(v)
        for r in v["criteria_results"]:
            if r["mandatory"]:
                self.assertEqual(r["status"], "PASS", "%s: %s" % (r["id"], r["reason"]))
                self.assertTrue(r["evidence"], r["id"])
        self.assertEqual(v["overall"], "PASS", v["overall_reason"])
        self.assertEqual(code, 0)
        # the optional target (1000 tok/s in the SYNTHETIC criteria) fails without blocking the overall PASS
        self.assertEqual(res["decode_filled_ctx_target"]["status"], "FAIL")
        self.assertFalse(res["decode_filled_ctx_target"]["mandatory"])
        self.assertEqual(res["manual_quality_eval"]["status"], "NOT RUN")
        floor = res["decode_filled_ctx_floor"]
        self.assertTrue(50.0 <= floor["value"] <= 80.0, floor["value"])
        self.assertTrue(floor["evidence"][0].startswith("artifacts/matrix/matrix-filled-SYNTHETIC/"))
        self.assertEqual(floor["v"], "V4")
        self.assertLessEqual(res["prefix_cache"]["value"], 0.10)
        self.assertEqual(res["exact_routing_logprobs"]["value"], 1.0)
        self.assertLessEqual(res["decode_vs_roofline"]["value"], 1.0)
        self.assertTrue(v["synthetic_markers"])
        self.assertFalse(v["criteria_is_default"])
        self.assertIn("ctx_tokens", v["criteria_deviations"])
        with open(os.path.join(out, "VERDICT.md")) as f:
            md = f.read()
        self.assertIn("**Итог: PASS**", md)
        self.assertIn("пометки SYNTHETIC", md)
        self.assertIn("Критерии отличаются", md)
        # the same artifacts against the customer's default criteria are not a PASS (4K is not 262K)
        code_d, v_d, _, _ = self.build(runs, "default", "pack-default")
        self.assertNotEqual(v_d["overall"], "PASS")
        self.assertEqual(self.by_id(v_d)["decode_filled_ctx_floor"]["reason_code"], "not_filled")

    def test_mock_sourced_artifacts_are_not_run_mock(self):
        runs = self.make_runs(matrix="mock_matrix", probes="mock_probes", quant="quant_ok", logprob="lp_mock",
                              slot="slot_mock", roofline="roofline_mock", hw="hw_ok")
        code, v, _, _ = self.build(runs)
        res = self.by_id(v)
        for cid in BENCH_IDS + ["decode_filled_ctx_target", "determinism", "decode_vs_roofline",
                                "exact_routing_logprobs", "kv_slot_persist"]:
            self.assertEqual(res[cid]["status"], "NOT RUN", cid)
            self.assertEqual(res[cid]["reason_code"], "mock", "%s: %s" % (cid, res[cid]["reason"]))
            self.assertTrue(res[cid]["reason"].startswith("mock"), cid)
        self.assertEqual(res["quant_min_bpw"]["status"], "PASS")
        self.assertEqual(v["overall"], "NOT RUN")
        self.assertEqual(code, 3)
        self.assertTrue(all(a["mock"] for a in v["artifacts"] if a["kind"] in ("matrix", "probes")))

    def test_one_mock_run_among_real_ones_blocks_the_criterion(self):
        parts = dict(PASSING, matrix=["matrix", "mock_short"])
        _, v, _, _ = self.build(self.make_runs(**parts))
        res = self.by_id(v)
        self.assertEqual(res["decode_filled_ctx_floor"]["reason_code"], "mock")
        self.assertEqual(v["overall"], "NOT RUN")

    def test_decode_below_floor_fails(self):
        parts = dict(PASSING, matrix="slow", roofline="roofline_slow")
        code, v, _, _ = self.build(self.make_runs(**parts))
        r = self.by_id(v)["decode_filled_ctx_floor"]
        self.assertEqual(r["status"], "FAIL", r["reason"])
        self.assertEqual(r["reason_code"], "threshold")
        self.assertLess(r["value"], 50.0)
        self.assertEqual(v["overall"], "FAIL")
        self.assertEqual(code, 1)

    def test_short_context_decode_does_not_count(self):
        parts = dict(PASSING, matrix="short", roofline=None)
        code, v, _, _ = self.build(self.make_runs(**parts))
        res = self.by_id(v)
        r = res["decode_filled_ctx_floor"]
        self.assertEqual(r["status"], "NOT RUN", r["reason"])
        self.assertEqual(r["reason_code"], "not_filled")
        self.assertIsNone(r["value"])
        short = r["details"]["short_context_cells_not_counted"]
        self.assertTrue(short and all(c["decode_p50"] > 50.0 for c in short))   # fast, but at 1024 tokens
        self.assertEqual(res["prefix_cache"]["reason_code"], "not_filled")
        self.assertNotEqual(v["overall"], "PASS")

    def test_truncated_prompt_at_full_target_is_not_run(self):
        """SYNTHETIC 'truncating server': the cell asked for 4096 tokens, usage says the prompt was cut to 2000.
        raw.jsonl and summary.json are edited consistently, so only the measured-fill rule can stop a PASS."""
        runs = self.make_runs(**PASSING)
        d = os.path.join(runs, "matrix", "matrix-filled-SYNTHETIC")
        rows = runlog.load_raw(os.path.join(d, "raw.jsonl"))
        for r in rows:
            if r.get("phase") == "measure":
                r["result"]["usage"]["prompt_tokens"] = 2000
                r["metrics"]["prompt_tokens"] = 2000
        with open(os.path.join(d, "raw.jsonl"), "w") as f:
            f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        with open(os.path.join(d, "summary.json")) as f:
            s = json.load(f)
        for c in s["cells"]:
            c["metrics"]["prompt_tokens"] = runlog.describe([2000] * c["n_ok"])
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(s, f, indent=2, sort_keys=True)
        _, v, _, _ = self.build(runs)
        res = self.by_id(v)
        r = res["decode_filled_ctx_floor"]
        self.assertEqual(r["status"], "NOT RUN", r["reason"])
        self.assertEqual(r["reason_code"], "not_filled")
        self.assertIn("prompt not filled", r["reason"])
        for cid in ("prefix_cache", "decode_vs_roofline"):
            self.assertEqual(res[cid]["status"], "NOT RUN", cid)
            self.assertEqual(res[cid]["reason_code"], "not_filled", cid)
        self.assertEqual(v["overall"], "NOT RUN")

    def test_strict_prompt_basis_needs_the_prompt_alone_to_fill_the_window(self):
        crit = os.path.join(self.tmp, "criteria-SYNTHETIC-prompt-basis.json")
        with open(crit, "w") as f:
            json.dump(dict(SMALL, decode_filled_context_basis="prompt"), f)
        runs = self.make_runs(**PASSING)
        out = os.path.join(self.tmp, "pack-prompt-basis")
        code, _ = quiet(vp.main, ["--runs", runs, "--out", out, "--criteria", crit])
        with open(os.path.join(out, "verdict.json")) as f:
            v = json.load(f)
        r = self.by_id(v)["decode_filled_ctx_floor"]
        self.assertEqual(r["status"], "NOT RUN", r["reason"])         # prompt 4064 + output 32: only the sum fills 4096
        self.assertEqual(r["reason_code"], "not_filled")
        self.assertEqual(code, 3)

    def test_quant_audit_below_4_bpw_fails(self):
        for key, status in (("quant_bad", "FAIL"), ("quant_bad_weak", "FAIL"), ("quant_dense_warn", "FAIL"),
                            ("quant_ok_weak", "NOT RUN")):
            self.tmp = tempfile.mkdtemp(prefix="case-SYNTHETIC-", dir=FIX)
            code, v, _, _ = self.build(self.make_runs(**dict(PASSING, quant=key)))
            r = self.by_id(v)["quant_min_bpw"]
            self.assertEqual(r["status"], status, "%s: %s" % (key, r["reason"]))
            if status == "FAIL":
                self.assertEqual(v["overall"], "FAIL", key)
                self.assertLess(r["value"], 4.0, key)
            else:
                self.assertEqual(v["overall"], "NOT RUN", key)
        with open(G["quant_bad_weak"]) as f:
            self.assertEqual(json.load(f)["verdict"], "PASS")   # the tool passes it at 3.0; the pack does not

    def test_summary_edited_against_raw_is_not_run(self):
        runs = self.make_runs(**PASSING)
        sp = os.path.join(runs, "matrix", "matrix-filled-SYNTHETIC", "summary.json")
        with open(sp) as f:
            s = json.load(f)
        for c in s["cells"]:
            c["metrics"]["decode_tok_s"]["p50"] = 999.0
        with open(sp, "w") as f:
            json.dump(s, f, indent=2, sort_keys=True)
        _, v, _, _ = self.build(runs)
        r = self.by_id(v)["decode_filled_ctx_floor"]
        self.assertEqual(r["status"], "NOT RUN")
        self.assertEqual(r["reason_code"], "inconsistent")
        self.assertEqual(v["overall"], "NOT RUN")

    def test_logprob_and_kv_failures(self):
        cases = (("logprob", "lp_flip", "exact_routing_logprobs", "FAIL"),
                 ("logprob", "lp_same", "exact_routing_logprobs", "NOT RUN"),
                 ("slot", "slot_noreuse", "kv_slot_persist", "FAIL"))
        for role, key, cid, status in cases:
            self.tmp = tempfile.mkdtemp(prefix="case-SYNTHETIC-", dir=FIX)
            _, v, _, _ = self.build(self.make_runs(**dict(PASSING, **{role: key})))
            r = self.by_id(v)[cid]
            self.assertEqual(r["status"], status, "%s: %s" % (key, r["reason"]))
            self.assertNotEqual(v["overall"], "PASS", key)

    def test_weights_roofline_and_nondeterminism(self):
        cases = ((dict(PASSING, wlog="wlog_bad"), "weights_integrity", "FAIL", "failed", "FAIL"),
                 (dict(PASSING, wlog=None), "weights_integrity", "NOT RUN", "missing", "NOT RUN"),
                 (dict(PASSING, roofline="roofline_glm"), "decode_vs_roofline", "NOT RUN", "invalid", "NOT RUN"),
                 (dict(PASSING, probes=["probes", "probes_nondet"]), "determinism", "FAIL", "threshold", "PASS"))
        for parts, cid, status, code, overall in cases:
            self.tmp = tempfile.mkdtemp(prefix="case-SYNTHETIC-", dir=FIX)
            _, v, _, _ = self.build(self.make_runs(**parts))
            r = self.by_id(v)[cid]
            self.assertEqual((r["status"], r["reason_code"]), (status, code), "%s: %s" % (cid, r["reason"]))
            self.assertEqual(v["overall"], overall, "%s: %s" % (cid, v["overall_reason"]))
        with open(G["roofline_glm"]) as f:
            self.assertTrue(all(row["ratio"] > 1.0 for row in json.load(f)))   # measured above the bound
        # format nuance the pack must tolerate: probes.py rewrites "pass" after logging the probe_result row
        d = G["probes_nondet"]
        with open(os.path.join(d, "summary.json")) as f:
            det = [r for r in json.load(f)["results"] if r["probe"] == "determinism"][0]
        raw = runlog.load_raw(os.path.join(d, "raw.jsonl"))
        det_raw = [r for r in raw if r.get("phase") == "probe_result" and r.get("probe") == "determinism"][0]
        self.assertEqual((det["pass"], det_raw["pass"]), (False, True))

    def test_hardware_facts_synthetic_flag_and_unknown_facts(self):
        for key, code in (("hw_flagged", "synthetic"), ("hw_nonroot", "insufficient")):
            self.tmp = tempfile.mkdtemp(prefix="case-SYNTHETIC-", dir=FIX)
            _, v, _, _ = self.build(self.make_runs(**dict(PASSING, hw=key)))
            r = self.by_id(v)["hardware_facts"]
            self.assertEqual(r["status"], "NOT RUN", key)
            self.assertEqual(r["reason_code"], code, "%s: %s" % (key, r["reason"]))

    def test_unparseable_artifact_blocks_overall_pass(self):
        runs = self.make_runs(**PASSING)
        with open(os.path.join(runs, "broken-SYNTHETIC.json"), "w") as f:
            f.write("{not json")
        _, v, _, _ = self.build(runs)
        self.assertTrue(all(r["status"] == "PASS" for r in v["criteria_results"] if r["mandatory"]))
        self.assertEqual(v["overall"], "NOT RUN")
        self.assertIn("broken-SYNTHETIC.json", v["overall_reason"])


class Manifest(PackCase):
    def test_check_detects_modified_missing_and_extra_files(self):
        _, v, out, _ = self.build(self.make_runs(**PASSING))
        self.assertEqual(quiet(vp.main, ["--check", out])[0], 0)
        target = os.path.join(out, "artifacts", "matrix", "matrix-filled-SYNTHETIC", "summary.json")
        with open(target, "rb") as f:
            orig = f.read()
        with open(target, "wb") as f:
            f.write(orig.replace(b'"n_ok": 3', b'"n_ok": 4', 1))
        code, text = quiet(vp.main, ["--check", out])
        self.assertEqual(code, 1)
        self.assertIn("MODIFIED artifacts/matrix/matrix-filled-SYNTHETIC/summary.json", text)
        with open(target, "wb") as f:
            f.write(orig)
        self.assertEqual(quiet(vp.main, ["--check", out])[0], 0)
        extra = os.path.join(out, "artifacts", "planted-SYNTHETIC.txt")
        with open(extra, "w") as f:
            f.write("x")
        code, text = quiet(vp.main, ["--check", out])
        self.assertEqual(code, 1)
        self.assertIn("EXTRA artifacts/planted-SYNTHETIC.txt", text)
        os.remove(extra)
        os.remove(os.path.join(out, "VERDICT.md"))
        code, text = quiet(vp.main, ["--check", out])
        self.assertEqual(code, 1)
        self.assertIn("MISSING VERDICT.md", text)

    def test_manifest_lists_every_file_with_its_sha256(self):
        _, _, out, _ = self.build(self.make_runs(**PASSING))
        with open(os.path.join(out, "MANIFEST.sha256")) as f:
            lines = [l.split("  ", 1) for l in f.read().splitlines()]
        listed = {p: d for d, p in lines}
        files = vp.list_files(out)
        self.assertEqual(set(listed), set(files) - {"MANIFEST.sha256"})
        for p, d in listed.items():
            with open(os.path.join(out, p), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), d, p)

    def test_wrapper_script_build_and_check(self):
        runs = self.make_runs(**PASSING)
        out = os.path.join(self.tmp, "pack-cli")
        sh = os.path.join(ROOT, "scripts", "verify_pack.sh")
        p = subprocess.run(["bash", sh, "--runs", runs, "--out", out, "--criteria", G["small"]],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("overall: PASS", p.stdout)
        p = subprocess.run(["bash", sh, "--check", out], capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = subprocess.run(["bash", sh, "--runs", runs, "--out", out], capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 2)                       # never overwrites an existing pack
        self.assertIn("not empty", p.stderr)


if __name__ == "__main__":
    unittest.main()
