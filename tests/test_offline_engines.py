"""scripts/offline/engines.py and build_engine.py --offline-kit: the R1 engines reach a server without internet as
one-commit git bundles (+ pinned wheels). Everything here runs on local fake upstream repositories; the "network" URLs
in the pins handed to build_engine are unreachable on purpose, so a passing build proves the kit alone was used."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "offline"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engines"))
import build_engine as be  # noqa: E402
import engines as ek  # noqa: E402

GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True,
                          text=True).stdout.strip()


def call(main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue() + err.getvalue()


class Kit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kit-")
        self.repos = {}
        for name in ("eng", "ggml"):
            r = os.path.join(self.tmp, "up-" + name)
            git("init", "-q", r)
            for i in range(3):                       # history behind the pin: the kit carries only the pinned commit
                with open(os.path.join(r, "f.txt"), "w") as f:
                    f.write("%s %d\n" % (name, i))
                git("-C", r, "add", "f.txt")
                git("-C", r, "commit", "-q", "-m", "c%d" % i)
            self.repos[name] = (r, git("-C", r, "rev-parse", "HEAD"))
        (er, es), (gr, gs) = self.repos["eng"], self.repos["ggml"]
        self.pins = {"engines": {"eng": {"repo": er, "sha": es, "why": "test", "cmake": ["-DX=${ARCH}"],
                                         "targets": ["x"], "bins": ["bin/x"],
                                         "deps": {"ggml": {"repo": gr, "sha": gs, "why": "test"}}}}}
        self.pins_path = self.write_pins(self.pins, "pins.json")
        self.kit = os.path.join(self.tmp, "kit")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_pins(self, pins, name):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            json.dump(pins, f)
        return p

    def pack(self, *extra):
        return call(ek.main, ["--pins", self.pins_path, "pack", "--out", self.kit, *extra])

    def verify(self):
        return call(ek.main, ["--pins", self.pins_path, "verify", self.kit])

    def manifest(self):
        with open(os.path.join(self.kit, "manifest.json")) as f:
            return json.load(f)

    def test_pack_writes_one_commit_bundles_and_a_manifest(self):
        rc, out = self.pack()
        self.assertEqual(rc, 0, out)
        m = self.manifest()
        self.assertEqual(m["schema"], ek.SCHEMA)
        src = m["engines"]["eng"]["sources"]
        self.assertEqual({k: v["sha"] for k, v in src.items()},
                         {"eng": self.repos["eng"][1], "eng-dep-ggml": self.repos["ggml"][1]})
        for key, v in src.items():
            path = os.path.join(self.kit, v["file"])
            self.assertEqual(v["file"], os.path.join("sources", key + ".bundle"))
            self.assertEqual(v["sha256"], ek.sha256_file(path))
            self.assertEqual(v["ref"], "refs/pins/" + key)
            heads = subprocess.run(["git", "bundle", "list-heads", path], capture_output=True, text=True).stdout.split()
            self.assertEqual(heads, [v["sha"], "refs/pins/" + key])
        self.assertFalse(os.path.exists(os.path.join(self.kit, ".work")))
        rc, out = self.pack()
        self.assertEqual(rc, 2, "packing into a non-empty directory must be refused")

    def test_verify_accepts_the_kit_and_rejects_tampering(self):
        self.assertEqual(self.pack()[0], 0)
        rc, out = self.verify()
        self.assertEqual(rc, 0, out)
        self.assertIn("verify: ok", out)
        bundle = os.path.join(self.kit, "sources", "eng.bundle")
        with open(bundle, "ab") as f:                 # one byte appended: the sha256 check fails
            f.write(b"x")
        rc, out = self.verify()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL source eng sha256", out)

    def test_verify_rejects_a_bundle_of_another_commit_even_with_a_matching_hash(self):
        self.assertEqual(self.pack()[0], 0)
        r, sha = self.repos["eng"]
        # the root commit has no parents: the fetch succeeds, so only the commit comparison can reject it
        other = git("-C", r, "rev-list", "--max-parents=0", "HEAD")
        work = os.path.join(self.tmp, "w")
        git("init", "-q", work)
        git("-C", work, "fetch", "-q", "--depth", "1", r, other)
        git("-C", work, "update-ref", "refs/pins/eng", other)
        bundle = os.path.join(self.kit, "sources", "eng.bundle")
        os.remove(bundle)
        git("-C", work, "bundle", "create", bundle, "refs/pins/eng")
        m = self.manifest()                           # an attacker can also rewrite the manifest's sha256
        m["engines"]["eng"]["sources"]["eng"].update(sha256=ek.sha256_file(bundle), size=os.path.getsize(bundle))
        with open(os.path.join(self.kit, "manifest.json"), "w") as f:
            json.dump(m, f)
        rc, out = self.verify()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL source eng commit", out)
        self.assertIn("the bundle gives %s, pinned %s" % (other, sha), out)
        self.assertIn("ok   source eng sha256", out)        # the rewritten hash alone does not reveal it

    def test_verify_rejects_a_kit_for_other_pins(self):
        self.assertEqual(self.pack()[0], 0)
        pins = json.loads(json.dumps(self.pins))
        pins["engines"]["eng"]["sha"] = "0" * 40
        self.pins_path = self.write_pins(pins, "pins-moved.json")
        rc, out = self.verify()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL pins eng", out)

    def fake_toolchain(self):
        bin_ = os.path.join(self.tmp, "bin")
        os.makedirs(bin_, exist_ok=True)
        for name, body in (("cmake", 'if [ "$1" = "--build" ]; then mkdir -p "$2/bin"; printf "#!/bin/sh\\n" > "$2/bin/x"; '
                                     'chmod +x "$2/bin/x"; fi; exit 0'),
                           ("cuobjdump", 'echo "ELF file    1: x.1.sm_89.cubin"'),
                           ("nvcc", 'echo "Cuda compilation tools, release 13.0, V13.0.88"')):
            p = os.path.join(bin_, name)
            with open(p, "w") as f:
                f.write("#!/usr/bin/env bash\n" + body + "\n")
            os.chmod(p, 0o755)
        return bin_

    def build(self, pins_path, kit=None):
        bin_ = self.fake_toolchain()
        old = os.environ["PATH"]
        os.environ["PATH"] = bin_ + os.pathsep + old
        try:
            rc, out = call(be.main, ["--engine", "eng", "--pins", pins_path, "--nvcc", os.path.join(bin_, "nvcc"),
                                     "--generator", "make", "--src-root", os.path.join(self.tmp, "src"),
                                     "--build-root", os.path.join(self.tmp, "build"),
                                     *(["--offline-kit", kit] if kit else [])])
        finally:
            os.environ["PATH"] = old
        info = os.path.join(self.tmp, "build", "eng", "build-info.json")
        if not os.path.exists(info):
            return rc, out, None
        with open(info) as f:
            return rc, out, json.load(f)

    def offline_pins(self):
        """The same commits, but upstream URLs that cannot be reached: only the kit can provide them."""
        pins = json.loads(json.dumps(self.pins))
        pins["engines"]["eng"]["repo"] = "https://invalid.example/eng"
        pins["engines"]["eng"]["deps"]["ggml"]["repo"] = "https://invalid.example/ggml"
        return self.write_pins(pins, "pins-offline.json")

    def test_build_from_the_kit_without_the_upstream(self):
        self.assertEqual(self.pack()[0], 0)
        offline = self.offline_pins()
        rc, out, info = self.build(offline)
        self.assertNotEqual(rc, 0, "without the kit the unreachable upstream must fail the fetch")
        shutil.rmtree(os.path.join(self.tmp, "src"))
        shutil.rmtree(os.path.join(self.tmp, "build"))
        rc, out, info = self.build(offline, self.kit)
        self.assertEqual(rc, 0, (out, info))
        self.assertEqual(info["status"], "ok")
        self.assertEqual(info["checked_out"], self.repos["eng"][1])
        self.assertEqual(info["deps_checked_out"], {"ggml": self.repos["ggml"][1]})
        self.assertEqual(info["offline_kit"], os.path.abspath(self.kit))
        with open(os.path.join(self.tmp, "src", "eng", "f.txt")) as f:
            self.assertEqual(f.read(), "eng 2\n")                 # the pinned tree, not an older one

    def test_build_refuses_a_kit_packed_for_other_commits(self):
        self.assertEqual(self.pack()[0], 0)
        pins = json.loads(json.dumps(self.pins))
        pins["engines"]["eng"]["deps"]["ggml"]["sha"] = "1" * 40
        rc, out, info = self.build(self.write_pins(pins, "pins-dep-moved.json"), self.kit)
        self.assertEqual(rc, 2, out)
        self.assertIn("packed for other commits", out)
        self.assertIsNone(info)

    def test_dependency_checkout_is_checked_against_its_pin(self):
        self.assertEqual(self.pack()[0], 0)
        # a tampered dependency bundle (another commit under the pinned ref name, verify not run first); the root
        # commit has no parents, so the fetch itself succeeds and only the checkout check can catch it
        r, _ = self.repos["ggml"]
        other = git("-C", r, "rev-list", "--max-parents=0", "HEAD")
        work = os.path.join(self.tmp, "w")
        git("init", "-q", work)
        git("-C", work, "fetch", "-q", "--depth", "1", r, other)
        git("-C", work, "update-ref", "refs/pins/eng-dep-ggml", other)
        bundle = os.path.join(self.kit, "sources", "eng-dep-ggml.bundle")
        os.remove(bundle)
        git("-C", work, "bundle", "create", bundle, "refs/pins/eng-dep-ggml")
        rc, out, info = self.build(self.offline_pins(), self.kit)
        self.assertEqual(rc, 1, out)
        self.assertEqual(info["status"], "failed: dependency ggml checked out %s, pinned %s"
                         % (other, self.repos["ggml"][1]))
        self.assertEqual(info["checked_out"], self.repos["eng"][1])

    def test_plans(self):
        pins = {"engines": {"tx": {"repo": "https://example.invalid/tx", "sha": "a" * 40, "why": "t",
                                   "kind": "torch-extension", "targets": ["tx_ext"], "bins": ["ext/tx_ext.so"],
                                   "python": {"torch": "2.14.1", "cuda": "13.0",
                                              "install": ["torch==2.14.1", "setuptools>=77"]}}}}
        p = self.write_pins(pins, "pins-tx.json")
        rc, out = call(ek.main, ["--pins", p, "pack", "--out", self.kit, "--python-version", "3.12", "--plan"])
        self.assertEqual(rc, 0, out)
        self.assertIn("pip download --only-binary=:all: --python-version 3.12 --platform manylinux_2_28_x86_64 "
                      "--platform manylinux_2_27_x86_64", out)
        self.assertIn("--platform manylinux2014_x86_64", out)         # ninja ships manylinux2014 / 2_17 wheels only
        self.assertIn("--platform manylinux_2_18_x86_64", out)         # nvidia-nccl-cu13 2.30.7 ships 2_18 only
        self.assertEqual(out.count("--platform manylinux_2_"), 24)    # every glibc 2.5 .. 2.28
        self.assertIn("--platform manylinux1_x86_64 --implementation cp -d %s torch==2.14.1 'setuptools>=77'"
                      % os.path.join(self.kit, "wheels", "tx"), out)
        rc, out = call(ek.main, ["--pins", p, "pack", "--out", self.kit, "--python-version", "3.12", "--plan",
                                 "--platform", "manylinux_2_35_x86_64"])
        self.assertIn("--platform manylinux_2_35_x86_64 --implementation cp", out)
        self.assertNotIn("manylinux1", out)
        self.assertIn("bundle create %s refs/pins/tx" % os.path.join(self.kit, "sources", "tx.bundle"), out)
        self.assertFalse(os.path.exists(self.kit), "--plan must not create anything")
        rc, out = call(ek.main, ["--pins", p, "install-python", self.kit, "--engine", "tx", "--venv", "/v", "--plan"])
        self.assertEqual(rc, 0, out)
        self.assertIn("/v/bin/pip install --no-index --find-links %s torch==2.14.1 'setuptools>=77'"
                      % os.path.join(os.path.abspath(self.kit), "wheels", "tx"), out)
        rc, out = call(ek.main, ["--pins", self.pins_path, "install-python", self.kit, "--engine", "eng", "--venv", "/v"])
        self.assertEqual(rc, 2, "a cmake engine has no Python list")

    def test_kit_plan_of_build_engine_fetches_from_the_bundle(self):
        self.assertEqual(self.pack()[0], 0)
        rc, out = call(be.main, ["--engine", "eng", "--pins", self.offline_pins(), "--plan", "--nvcc", "/c/bin/nvcc",
                                 "--src-root", "/w/src", "--build-root", "/w/build", "--offline-kit", self.kit])
        self.assertEqual(rc, 0, out)
        k = os.path.abspath(self.kit)
        self.assertIn("git -C /w/src/eng fetch -q %s/sources/eng.bundle refs/pins/eng" % k, out)
        self.assertIn("git -C /w/src/eng-dep-ggml fetch -q %s/sources/eng-dep-ggml.bundle refs/pins/eng-dep-ggml" % k, out)
        self.assertNotIn("invalid.example", out)
        self.assertLess(out.index("/w/src/eng " + self.repos["eng"][1]), out.index("/w/src/eng fetch"))


if __name__ == "__main__":
    unittest.main()
