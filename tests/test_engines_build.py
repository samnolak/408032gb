"""scripts/engines/build_engine.py and tools/engines/pins.json (ROADMAP R1.4, track N1): pins are well-formed, the plan
pins the exact commits and passes the architecture, and the verification logic rejects a wrong build. Nothing is
compiled here (the compile itself runs in .github/workflows/engines-sm89.yml and on the server)."""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engines"))
import build_engine as be  # noqa: E402

SHA = re.compile(r"^[0-9a-f]{40}$")


class Pins(unittest.TestCase):
    def test_pins_are_full_shas_on_github_with_cuda_arch(self):
        pins = be.load_pins()
        self.assertEqual(set(pins), {"llama.cpp", "ik_llama.cpp", "strata"})
        for name, p in pins.items():
            self.assertRegex(p["sha"], SHA, name)
            self.assertTrue(p["repo"].startswith("https://github.com/"), name)
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=${ARCH}", p["cmake"], name)
            self.assertTrue(p["targets"] and p["bins"], name)
            self.assertTrue(p["why"], name)
            for d in (p.get("deps") or {}).values():
                self.assertRegex(d["sha"], SHA, name)

    def test_llama_cpp_pin_has_the_expert_cache(self):
        # research/R2: --moe-cache-mib arrived in d6cf9acb2, multi-GPU in c811cb8f0; the pin must contain both
        llama = os.environ.get("LLAMA_CPP_FULL_CLONE")
        if not llama:
            self.skipTest("set LLAMA_CPP_FULL_CLONE=<llama.cpp clone with history> to check ancestry")
        sha = be.load_pins()["llama.cpp"]["sha"]
        for c in ("d6cf9acb2", "c811cb8f0"):
            r = subprocess.run(["git", "-C", llama, "merge-base", "--is-ancestor", c, sha])
            self.assertEqual(r.returncode, 0, c)


class Plan(unittest.TestCase):
    def plan_text(self, engine, *extra):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = be.main(["--engine", engine, "--plan", "--arch", "89", "--nvcc", "/usr/local/cuda-13.0/bin/nvcc",
                          "--src-root", "/w/src", "--build-root", "/w/build", "--generator", "ninja", *extra])
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_plan_fetches_the_pinned_commit_and_passes_the_arch(self):
        for name, p in be.load_pins().items():
            t = self.plan_text(name)
            self.assertIn("fetch -q --depth 1 %s %s" % (p["repo"], p["sha"]), t)
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", t)
            self.assertIn("-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc", t)
            self.assertIn("-DCUDAToolkit_ROOT=/usr/local/cuda-13.0", t)
            self.assertIn("--target " + " ".join(p["targets"]), t)
            self.assertNotIn("${", t)

    def test_strata_builds_against_its_own_ggml_pin(self):
        t = self.plan_text("strata")
        dep = be.load_pins()["strata"]["deps"]["ggml"]
        self.assertIn("fetch -q --depth 1 %s %s" % (dep["repo"], dep["sha"]), t)
        self.assertIn("-DSTRATA_GGML_DIR=/w/src/strata-dep-ggml", t)
        self.assertLess(t.index(dep["sha"]), t.index("cmake -G Ninja"))

    def test_extra_cmake_is_appended_and_unknown_engine_is_usage_error(self):
        self.assertIn("-DGGML_CUDA_NO_VMM=ON", self.plan_text("llama.cpp", "--extra-cmake=-DGGML_CUDA_NO_VMM=ON"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(be.main(["--engine", "nope", "--plan"]), 2)

    def test_unsubstituted_variable_is_an_error(self):
        with self.assertRaises(ValueError):
            be.subst(["-DX=${NOPE}"], {"ARCH": "89"})


class Verification(unittest.TestCase):
    """main() with a fake toolchain: the pinned commit is checked out, but the binaries or the cubins are wrong."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.repo = os.path.join(d, "upstream")
        os.makedirs(self.repo)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@e")
        for c in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "x"]):
            subprocess.run(["git", "-C", self.repo, *c], check=True, env=env, capture_output=True)
        self.sha = subprocess.run(["git", "-C", self.repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        self.bin = os.path.join(d, "bin")
        os.makedirs(self.bin)

    def tearDown(self):
        self.tmp.cleanup()

    def tool(self, name, body):
        p = os.path.join(self.bin, name)
        with open(p, "w") as f:
            f.write("#!/usr/bin/env bash\n" + body + "\n")
        os.chmod(p, 0o755)
        return p

    def run_build(self, cubin_arch, make_bin=True):
        # fake cmake: configure does nothing; build creates bin/x; fake cuobjdump reports one cubin arch
        self.tool("cmake", 'if [ "$1" = "--build" ]; then mkdir -p "$2/bin"; %s; fi; exit 0'
                  % ('printf "#!/bin/sh\\n" > "$2/bin/x"; chmod +x "$2/bin/x"' if make_bin else "true"))
        self.tool("cuobjdump", 'echo "ELF file    1: x.1.%s.cubin"' % cubin_arch)
        nvcc = self.tool("nvcc", 'echo "Cuda compilation tools, release 13.4, V13.4.92"')
        pins = os.path.join(self.tmp.name, "pins.json")
        with open(pins, "w") as f:
            json.dump({"engines": {"fake": {"repo": self.repo, "sha": self.sha, "why": "test",
                                            "cmake": ["-DCMAKE_CUDA_ARCHITECTURES=${ARCH}"], "targets": ["x"],
                                            "bins": ["bin/x"]}}}, f)
        old = os.environ["PATH"]
        os.environ["PATH"] = self.bin + os.pathsep + old
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = be.main(["--engine", "fake", "--pins", pins, "--nvcc", nvcc, "--generator", "make",
                              "--src-root", os.path.join(self.tmp.name, "src"),
                              "--build-root", os.path.join(self.tmp.name, "build")])
        finally:
            os.environ["PATH"] = old
        with open(os.path.join(self.tmp.name, "build", "fake", "build-info.json")) as f:
            return rc, json.load(f)

    def test_ok_build_records_sha_and_cubins(self):
        rc, info = self.run_build("sm_89")
        self.assertEqual(rc, 0, info)
        self.assertEqual(info["status"], "ok")
        self.assertEqual(info["checked_out"], self.sha)
        self.assertEqual(info["cubin_archs"], {"bin/x": ["sm_89"]})
        with open(os.path.join(self.tmp.name, "build", "fake", "bin", "x"), "rb") as f:
            self.assertEqual(info["files_sha256"], {"bin/x": hashlib.sha256(f.read()).hexdigest()})
        self.assertIn("13.4", info["nvcc_version"])

    def test_wrong_arch_fails(self):
        rc, info = self.run_build("sm_86")
        self.assertEqual(rc, 1)
        self.assertIn("no sm_89 cubin", info["status"])

    def test_missing_binary_fails(self):
        rc, info = self.run_build("sm_89", make_bin=False)
        self.assertEqual(rc, 1)
        self.assertIn("missing binaries", info["status"])


if __name__ == "__main__":
    unittest.main()
