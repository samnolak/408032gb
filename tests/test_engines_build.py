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
        self.assertEqual(set(pins), {"llama.cpp", "ik_llama.cpp", "strata", "exllamav3"})
        for name, p in pins.items():
            self.assertRegex(p["sha"], SHA, name)
            self.assertTrue(p["repo"].startswith("https://github.com/"), name)
            if p.get("kind") == "torch-extension":
                self.assertNotIn("cmake", p, name)
                self.assertRegex(p["python"]["torch"], r"^\d+\.\d+\.\d+$", name)
                self.assertRegex(p["python"]["cuda"], r"^\d+\.\d+$", name)
                self.assertIn("torch==" + p["python"]["torch"], p["python"]["install"], name)
            else:
                self.assertNotIn("kind", p, name)
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
            if p.get("kind") == "torch-extension":
                continue                                   # test_torch_extension_plan
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

    def test_torch_extension_plan(self):
        p = be.load_pins()["exllamav3"]
        t = self.plan_text("exllamav3", "--python", "/w/venv/bin/python", "--jobs", "4")
        lines = t.splitlines()
        check = [x for x in lines if "torch.__version__" in x]
        self.assertEqual(len(check), 1, t)
        self.assertTrue(check[0].startswith("/w/venv/bin/python -c "), check[0])
        self.assertTrue(check[0].endswith(" %s %s" % (p["python"]["torch"], p["python"]["cuda"])), check[0])
        wheel = [x for x in lines if " pip wheel " in x]
        self.assertEqual(len(wheel), 1, t)
        for part in ("env TORCH_CUDA_ARCH_LIST=8.9 ", "CUDA_HOME=/usr/local/cuda-13.0 ",
                     "PYTORCH_NVCC=/usr/local/cuda-13.0/bin/nvcc ", "MAX_JOBS=4 ",
                     "LIBRARY_PATH=/w/build/exllamav3-linkshim ",
                     "/w/venv/bin/python -m pip wheel --no-deps --no-build-isolation -w /w/build/exllamav3/dist "
                     "/w/src/exllamav3"):
            self.assertIn(part, wheel[0])
        shim = [x for x in lines if "libcudart.so.[0-9]*" in x]
        self.assertEqual(len(shim), 1, t)
        self.assertTrue(shim[0].endswith(" /usr/local/cuda-13.0 /w/build/exllamav3-linkshim"), shim[0])
        self.assertLess(t.index("torch.__version__"), t.index("libcudart.so.[0-9]*"))
        self.assertLess(t.index("libcudart.so.[0-9]*"), t.index(" pip wheel "))
        self.assertLess(t.index(" pip wheel "), t.index("/w/build/exllamav3/ext exllamav3_ext"))
        self.assertNotIn("cmake", t)
        self.assertNotIn("${", t)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(be.main(["--engine", "exllamav3", "--plan", "--extra-cmake=-DX=1"]), 2)

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


FAKE_PYTHON = r'''case "$2" in *torch.__version__*) echo "torch 2.14.1+cu130 cuda 13.0"; exit %(torch_rc)d;; esac
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  env > "%(tmp)s/wheel-env.txt"
  shift 6
  exec "%(real)s" -c "%(wheel)s" "$1" "%(so)s"
fi
exec "%(real)s" "$@"'''
# what the fake "pip wheel" writes: a wheel with one extension module (the real build compiles it with nvcc)
FAKE_WHEEL = ("import os, sys, zipfile; d = sys.argv[1]; os.makedirs(d, exist_ok=True); "
              "z = zipfile.ZipFile(os.path.join(d, 'fake-0.1-cp312-cp312-linux_x86_64.whl'), 'w'); "
              "z.writestr(sys.argv[2], b'SYNTHETIC ELF'); z.writestr('fake/__init__.py', ''); z.close()")


class TorchExtensionBuild(unittest.TestCase):
    """main() for a "kind": "torch-extension" pin with a fake interpreter: torch check, wheel, unpack, cubins."""

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

    def run_build(self, torch_ok=True, so_name="fake_ext.cpython-312-x86_64-linux-gnu.so", cubin="sm_89"):
        py = self.tool("python", FAKE_PYTHON % {"torch_rc": 0 if torch_ok else 1, "tmp": self.tmp.name,
                                                "real": sys.executable, "wheel": FAKE_WHEEL, "so": so_name})
        self.tool("cuobjdump", 'echo "ELF file    1: x.1.%s.cubin"' % cubin)
        nvcc = self.tool("nvcc", 'echo "Cuda compilation tools, release 13.0, V13.0.88"')
        pins = os.path.join(self.tmp.name, "pins.json")
        with open(pins, "w") as f:
            json.dump({"engines": {"fake": {"repo": self.repo, "sha": self.sha, "why": "test", "kind": "torch-extension",
                                            "python": {"torch": "2.14.1", "cuda": "13.0", "install": []},
                                            "targets": ["fake_ext"], "bins": ["ext/fake_ext.so"]}}}, f)
        old = os.environ["PATH"]
        os.environ["PATH"] = self.bin + os.pathsep + old
        try:
            with redirect_stdout(io.StringIO()):
                rc = be.main(["--engine", "fake", "--pins", pins, "--nvcc", nvcc, "--python", py, "--jobs", "3",
                              "--src-root", os.path.join(self.tmp.name, "src"),
                              "--build-root", os.path.join(self.tmp.name, "build")])
        finally:
            os.environ["PATH"] = old
        with open(os.path.join(self.tmp.name, "build", "fake", "build-info.json")) as f:
            return rc, json.load(f)

    def test_ok_build_unpacks_the_extension_and_hashes_the_wheel(self):
        lib = os.path.join(self.tmp.name, "lib")         # CUDA_HOME = <tmp> (two levels above the fake nvcc)
        os.makedirs(lib)
        open(os.path.join(lib, "libcudart.so.13"), "wb").close()   # pip CUDA layout: no unversioned link name
        rc, info = self.run_build()
        self.assertEqual(rc, 0, info)
        self.assertEqual(info["status"], "ok")
        self.assertEqual(info["kind"], "torch-extension")
        self.assertEqual(info["python_pin"]["torch"], "2.14.1")
        self.assertEqual(info["bins"], {"ext/fake_ext.so": True})
        self.assertEqual(info["cubin_archs"]["ext/fake_ext.so"], ["sm_89"])
        self.assertEqual(sorted(info["files_sha256"]), ["dist/fake-0.1-cp312-cp312-linux_x86_64.whl",
                                                        "ext/fake_ext.so"])
        with open(os.path.join(self.tmp.name, "build", "fake", "ext", "fake_ext.so"), "rb") as f:
            self.assertEqual(f.read(), b"SYNTHETIC ELF")
        with open(os.path.join(self.tmp.name, "wheel-env.txt")) as f:
            env = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
        self.assertEqual((env["TORCH_CUDA_ARCH_LIST"], env["MAX_JOBS"]), ("8.9", "3"))
        self.assertEqual(env["PYTORCH_NVCC"], os.path.join(self.bin, "nvcc"))
        shim = os.path.join(self.tmp.name, "build", "fake-linkshim")
        self.assertEqual(env["LIBRARY_PATH"], shim)
        self.assertEqual(os.readlink(os.path.join(shim, "libcudart.so")), os.path.join(lib, "libcudart.so.13"))

    def test_wrong_torch_stops_before_the_compile(self):
        rc, info = self.run_build(torch_ok=False)
        self.assertEqual(rc, 1)
        self.assertEqual(info["status"], "failed at: check torch 2.14.1 with CUDA 13.0")
        self.assertEqual(info["steps"][-1]["step"], "check torch 2.14.1 with CUDA 13.0")
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "wheel-env.txt")))

    def test_wheel_without_the_extension_or_wrong_arch_fails(self):
        rc, info = self.run_build(so_name="other.cpython-312-x86_64-linux-gnu.so")
        self.assertEqual(rc, 1)
        self.assertEqual(info["status"], "failed at: unpack fake_ext")
        rc, info = self.run_build(cubin="sm_86")
        self.assertEqual(rc, 1)
        self.assertIn("no sm_89 cubin", info["status"])


if __name__ == "__main__":
    unittest.main()
