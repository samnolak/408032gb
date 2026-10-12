#!/usr/bin/env python3
"""Build one R1 engine at its pinned commit for one CUDA architecture (ROADMAP R1.4, track N1).

The same script runs in CI (.github/workflows/engines-sm89.yml: compile only, the runner has no GPU) and on the server.
Pins, CMake flags, targets and expected binaries are in tools/engines/pins.json.

  python3 scripts/engines/build_engine.py --engine llama.cpp --plan                # print the commands, run nothing
  python3 scripts/engines/build_engine.py --engine llama.cpp --arch 89 \
      --nvcc /usr/local/cuda-13.0/bin/nvcc --src-root ../engines-src --build-root ../engines-build

Steps: fetch the pinned commit (and pinned dependencies) with a shallow fetch by SHA, or, on a host without internet,
from the one-commit git bundles of an offline kit (--offline-kit, made by scripts/offline/engines.py pack); configure
with CMake (Ninja when available); build the listed targets; check that the expected binaries exist; list the cubins
of every built binary and shared library with cuobjdump and require code for sm_<arch>; write <build>/build-info.json
(engine, repository, SHA, nvcc version, flags, binaries, cubin architectures, sha256 of every built binary and library).
Exit 0 = built and verified, 1 = a step failed, 2 = usage.
A build here says nothing about speed or correctness on the GPU: that is R1 on the server.
"""
import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PINS = os.path.join(ROOT, "tools", "engines", "pins.json")


def load_pins(path=PINS):
    with open(path) as f:
        return json.load(f)["engines"]


# Offline kit (scripts/offline/engines.py): one-commit git bundles <kit>/sources/<key>.bundle with the ref
# refs/pins/<key>, key = <engine> or <engine>-dep-<dep>. Such a bundle carries the pinned commit without its parents, so
# the checkout first marks that commit as a shallow boundary (.git/shallow); git then accepts the fetch, and the commit
# hash proves the content.
KIT_REF = "refs/pins/%s"
SHALLOW = "import os, sys; open(os.path.join(sys.argv[1], '.git', 'shallow'), 'w').write(sys.argv[2] + '\\n')"


def kit_key(name, dep=None):
    return name if dep is None else "%s-dep-%s" % (name, dep)


def kit_bundle(kit, key):
    return os.path.join(kit, "sources", key + ".bundle")


def fetch_cmds(repo, sha, dest, kit=None, key=None):
    if kit:
        return [["git", "init", "-q", dest],
                [sys.executable, "-c", SHALLOW, dest, sha],
                ["git", "-C", dest, "fetch", "-q", kit_bundle(kit, key), KIT_REF % key],
                ["git", "-C", dest, "checkout", "-q", "--detach", "FETCH_HEAD"]]
    return [["git", "init", "-q", dest],
            ["git", "-C", dest, "fetch", "-q", "--depth", "1", repo, sha],
            ["git", "-C", dest, "checkout", "-q", "--detach", "FETCH_HEAD"]]


def subst(flags, values):
    out = []
    for f in flags:
        for k, v in values.items():
            f = f.replace("${%s}" % k, v)
        if "${" in f:
            raise ValueError("unsubstituted variable in %r" % f)
        out.append(f)
    return out


# torch-extension builds (pins.json "kind": "torch-extension", e.g. ExLlamaV3): the extension's ABI is tied to the
# torch version and its CUDA, so the build environment is checked against the pin before compiling
TORCH_CHECK = ("import sys, torch; v, c = torch.__version__.split('+')[0], torch.version.cuda or ''; "
               "print('torch', torch.__version__, 'cuda', c or None); "
               "sys.exit(0 if v == sys.argv[1] and (c + '.').startswith(sys.argv[2] + '.') else 1)")
UNPACK = ("import glob, os, sys, zipfile; w = sorted(glob.glob(os.path.join(sys.argv[1], '*.whl'))); "
          "assert len(w) == 1, 'expected one wheel, found %r' % w; z = zipfile.ZipFile(w[0]); "
          "n = [x for x in z.namelist() if x.startswith(sys.argv[3] + '.') and x.endswith('.so')]; "
          "assert len(n) == 1, 'expected one %s*.so in the wheel, found %r' % (sys.argv[3], n); "
          "os.makedirs(sys.argv[2], exist_ok=True); "
          "open(os.path.join(sys.argv[2], sys.argv[3] + '.so'), 'wb').write(z.read(n[0])); print(w[0], '->', n[0])")

# torch's cpp_extension links with -L<CUDA_HOME>/lib64 (or lib) -lcudart; the pip CUDA 13 runtime ships only
# libcudart.so.13, so a shim directory with the unversioned link name goes on LIBRARY_PATH (the extension still
# records the soname libcudart.so.13, the one torch loads). Nothing is created when the toolkit has libcudart.so.
LINK_SHIM = ("import glob, os, sys; root, shim = sys.argv[1], sys.argv[2]; "
             "have = [d for d in ('lib64', 'lib') if os.path.exists(os.path.join(root, d, 'libcudart.so'))]; "
             "found = sorted(glob.glob(os.path.join(root, 'lib*', 'libcudart.so.[0-9]*'))); "
             "dst = os.path.join(shim, 'libcudart.so'); "
             "print('libcudart.so present in', os.path.join(root, have[0])) if have else "
             "print('no libcudart.so.N under', root, '(the link will fail)') if not found else "
             "(os.makedirs(shim, exist_ok=True), os.path.lexists(dst) and os.remove(dst), os.symlink(found[0], dst), "
             "print(dst, '->', found[0]))")


def torch_extension_steps(pin, arch, nvcc, cuda_root, src, build, jobs, python):
    """Build the pinned Python package as a wheel with its CUDA extension for one architecture, then unpack the
    extension to <build>/ext/<target>.so for the cubin check. The torch/CUDA pair comes from pins.json."""
    py = pin["python"]
    target = pin["targets"][0]
    shim = build + "-linkshim"                   # outside the build tree: not a build product, not hashed
    env = ["env", "TORCH_CUDA_ARCH_LIST=%d.%d" % divmod(arch, 10), "CUDA_HOME=" + cuda_root, "PYTORCH_NVCC=" + nvcc,
           "MAX_JOBS=%d" % jobs, "LIBRARY_PATH=" + shim] + ["%s=%s" % kv for kv in sorted((py.get("env") or {}).items())]
    return [("check torch %s with CUDA %s" % (py["torch"], py["cuda"]), [python, "-c", TORCH_CHECK, py["torch"], py["cuda"]]),
            ("link shim for libcudart", [python, "-c", LINK_SHIM, cuda_root, shim]),
            ("build wheel", env + [python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                                   "-w", os.path.join(build, "dist"), src]),
            ("unpack %s" % target, [python, "-c", UNPACK, os.path.join(build, "dist"), os.path.join(build, "ext"),
                                    target])]


def plan(name, pin, arch, nvcc, src_root, build_root, jobs, generator, extra=(), python=None, kit=None):
    """The list of (description, argv) steps for one engine."""
    src = os.path.join(src_root, name)
    build = os.path.join(build_root, name)
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc))) if os.path.exists(nvcc) else \
        os.path.dirname(os.path.dirname(nvcc))
    values = {"ARCH": str(arch), "NVCC": nvcc, "CUDA_ROOT": cuda_root}
    steps = []
    for dep, d in sorted((pin.get("deps") or {}).items()):
        ddir = os.path.join(src_root, "%s-dep-%s" % (name, dep))
        values["DEP_" + dep.upper()] = ddir
        steps += [("fetch dependency %s @ %s" % (dep, d["sha"][:12]), c)
                  for c in fetch_cmds(d["repo"], d["sha"], ddir, kit, kit_key(name, dep))]
    steps += [("fetch %s @ %s" % (name, pin["sha"][:12]), c)
              for c in fetch_cmds(pin["repo"], pin["sha"], src, kit, kit_key(name))]
    if pin.get("kind") == "torch-extension":
        if extra:
            raise ValueError("--extra-cmake does not apply to a torch-extension build")
        steps += torch_extension_steps(pin, arch, nvcc, cuda_root, src, build, jobs, python or sys.executable)
        return steps, src, build
    gen = ["-G", "Ninja"] if generator == "ninja" else []
    steps.append(("configure", ["cmake", *gen, "-S", src, "-B", build, *subst(pin["cmake"], values), *extra]))
    steps.append(("build", ["cmake", "--build", build, "-j", str(jobs), "--target", *pin["targets"]]))
    return steps, src, build


def run(argv, log):
    log.write("$ %s\n" % " ".join(shlex.quote(a) for a in argv))
    log.flush()
    p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode


def cubin_archs(path, cuobjdump):
    """sm_XX architectures of the ELF cubins embedded in a binary or library (empty if none or no cuobjdump)."""
    if not cuobjdump:
        return None
    p = subprocess.run([cuobjdump, "--list-elf", path], capture_output=True, text=True)
    archs = set()
    for line in p.stdout.splitlines():
        for tok in line.replace(".", " ").split():
            if tok.startswith("sm_") and tok[3:].rstrip("a").isdigit():
                archs.add(tok)
    return sorted(archs)


def built_files(build):
    out = []
    for dirpath, _, files in os.walk(build):
        if "CMakeFiles" in dirpath:
            continue
        for f in files:
            p = os.path.join(dirpath, f)
            if (".so" in f or f.endswith(".whl") or os.access(p, os.X_OK)) and os.path.isfile(p) \
                    and not f.endswith((".sh", ".py", ".cmake")):
                out.append(p)
    return sorted(out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nvcc_version(nvcc):
    try:
        return subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--arch", type=int, default=89)
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--src-root", default=os.path.join(ROOT, "..", "engines-src"))
    ap.add_argument("--build-root", default=os.path.join(ROOT, "..", "engines-build"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--generator", choices=["auto", "ninja", "make"], default="auto")
    ap.add_argument("--pins", default=PINS)
    ap.add_argument("--extra-cmake", action="append", default=[],
                    help="extra -D flag appended to the pinned ones (recorded in build-info.json); repeatable")
    ap.add_argument("--python", default=sys.executable,
                    help="torch-extension builds: the interpreter with the pinned torch, setuptools, wheel and ninja")
    ap.add_argument("--offline-kit", default=None, metavar="KIT",
                    help="fetch the pinned sources from the git bundles of scripts/offline/engines.py pack (no network)")
    ap.add_argument("--plan", action="store_true", help="print the commands and exit without running anything")
    a = ap.parse_args(argv)
    pins = load_pins(a.pins)
    if a.engine not in pins:
        print("unknown engine %r; known: %s" % (a.engine, ", ".join(sorted(pins))), file=sys.stderr)
        return 2
    pin = pins[a.engine]
    gen = a.generator if a.generator != "auto" else ("ninja" if shutil.which("ninja") else "make")
    src_root, build_root = os.path.abspath(a.src_root), os.path.abspath(a.build_root)
    kit = os.path.abspath(a.offline_kit) if a.offline_kit else None
    if kit:
        try:
            with open(os.path.join(kit, "manifest.json")) as f:
                kit_pin = json.load(f)["engines"][a.engine]
        except (OSError, ValueError, KeyError) as e:
            print("offline kit %s has no entry for %s: %s" % (kit, a.engine, e), file=sys.stderr)
            return 2
        want = {kit_key(a.engine): pin["sha"]}
        want.update({kit_key(a.engine, dep): d["sha"] for dep, d in (pin.get("deps") or {}).items()})
        got = {k: v.get("sha") for k, v in (kit_pin.get("sources") or {}).items()}
        if got != want:
            print("offline kit %s was packed for other commits: %s, pins.json wants %s" % (kit, got, want),
                  file=sys.stderr)
            return 2
    try:
        steps, src, build = plan(a.engine, pin, a.arch, a.nvcc, src_root, build_root, a.jobs, gen, a.extra_cmake,
                                 a.python, kit)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if a.plan:
        print("# plan for %s @ %s, sm_%d (nothing is run)" % (a.engine, pin["sha"], a.arch))
        for desc, argv_ in steps:
            print("# %s\n%s" % (desc, " ".join(shlex.quote(x) for x in argv_)))
        return 0
    os.makedirs(build_root, exist_ok=True)
    os.makedirs(src_root, exist_ok=True)
    log_path = os.path.join(build_root, "%s.log" % a.engine)
    info = {"engine": a.engine, "repo": pin["repo"], "sha": pin["sha"], "deps": pin.get("deps") or {}, "arch": a.arch,
            "nvcc": a.nvcc, "nvcc_version": nvcc_version(a.nvcc), "generator": gen, "extra_cmake": a.extra_cmake,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "steps": [], "status": "running"}
    if pin.get("kind") == "torch-extension":
        info.update(kind="torch-extension", python=a.python, python_pin=pin["python"], generator=None)
    if kit:
        info["offline_kit"] = kit
    with open(log_path, "w") as log:
        for desc, argv_ in steps:
            if desc.startswith("fetch") and argv_[1] == "init" and os.path.isdir(os.path.join(argv_[-1], ".git")):
                rc = 0                                              # already initialised; the fetch below re-pins
            else:
                rc = run(argv_, log)
            info["steps"].append({"step": desc, "rc": rc})
            print("%-40s rc=%d" % (desc, rc), flush=True)
            if rc != 0:
                info["status"] = "failed at: " + desc
                break
    if info["status"] == "running":
        head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        info["checked_out"] = head
        dep_heads = {dep: subprocess.run(["git", "-C", os.path.join(src_root, "%s-dep-%s" % (a.engine, dep)), "rev-parse",
                                          "HEAD"], capture_output=True, text=True).stdout.strip()
                     for dep in sorted(pin.get("deps") or {})}
        info["deps_checked_out"] = dep_heads
        bad_deps = ["%s checked out %s, pinned %s" % (dep, h, pin["deps"][dep]["sha"])
                    for dep, h in dep_heads.items() if h != pin["deps"][dep]["sha"]]
        missing = [b for b in pin["bins"] if not os.path.isfile(os.path.join(build, b))]
        cuobjdump = shutil.which("cuobjdump") or (os.path.join(os.path.dirname(a.nvcc), "cuobjdump")
                                                 if os.path.isfile(os.path.join(os.path.dirname(a.nvcc), "cuobjdump"))
                                                 else None)
        archs, files = {}, {}
        for f in built_files(build):
            files[os.path.relpath(f, build)] = sha256_file(f)
            got = cubin_archs(f, cuobjdump)
            if got:
                archs[os.path.relpath(f, build)] = got
        info["files_sha256"] = files     # scripts/r1/run_r1.py (engine_build_check) re-hashes them before a run
        info.update({"bins": {b: os.path.isfile(os.path.join(build, b)) for b in pin["bins"]},
                     "cubin_archs": archs if cuobjdump else "cuobjdump not found"})
        want = "sm_%d" % a.arch
        if head != pin["sha"]:
            info["status"] = "failed: checked out %s, pinned %s" % (head, pin["sha"])
        elif bad_deps:
            info["status"] = "failed: dependency " + "; ".join(bad_deps)
        elif missing:
            info["status"] = "failed: missing binaries " + ", ".join(missing)
        elif cuobjdump and not any(want in v for v in archs.values()):
            info["status"] = "failed: no %s cubin in any built binary or library" % want
        else:
            info["status"] = "ok" if cuobjdump else "ok (cubins not checked: no cuobjdump)"
    info["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    info["log"] = log_path
    os.makedirs(build, exist_ok=True)
    with open(os.path.join(build, "build-info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in ("engine", "sha", "arch", "status")}))
    if info["status"].startswith("ok"):
        for f, v in sorted((info.get("cubin_archs") or {}).items() if isinstance(info.get("cubin_archs"), dict) else []):
            print("cubins %-40s %s" % (f, " ".join(v)))
        return 0
    with open(log_path) as f:
        tail = f.readlines()[-40:]
    sys.stderr.write("".join(tail))
    return 1


if __name__ == "__main__":
    sys.exit(main())
