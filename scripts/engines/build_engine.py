#!/usr/bin/env python3
"""Build one R1 engine at its pinned commit for one CUDA architecture (ROADMAP R1.4, track N1).

The same script runs in CI (.github/workflows/engines-sm89.yml: compile only, the runner has no GPU) and on the server.
Pins, CMake flags, targets and expected binaries are in tools/engines/pins.json.

  python3 scripts/engines/build_engine.py --engine llama.cpp --plan                # print the commands, run nothing
  python3 scripts/engines/build_engine.py --engine llama.cpp --arch 89 \
      --nvcc /usr/local/cuda-13.0/bin/nvcc --src-root ../engines-src --build-root ../engines-build

Steps: fetch the pinned commit (and pinned dependencies) with a shallow fetch by SHA; configure with CMake (Ninja when
available); build the listed targets; check that the expected binaries exist; list the cubins of every built binary
and shared library with cuobjdump and require code for sm_<arch>; write <build>/build-info.json (engine, repository,
SHA, nvcc version, flags, binaries, cubin architectures). Exit 0 = built and verified, 1 = a step failed, 2 = usage.
A build here says nothing about speed or correctness on the GPU: that is R1 on the server.
"""
import argparse
import datetime
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


def fetch_cmds(repo, sha, dest):
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


def plan(name, pin, arch, nvcc, src_root, build_root, jobs, generator, extra=()):
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
        steps += [("fetch dependency %s @ %s" % (dep, d["sha"][:12]), c) for c in fetch_cmds(d["repo"], d["sha"], ddir)]
    steps += [("fetch %s @ %s" % (name, pin["sha"][:12]), c) for c in fetch_cmds(pin["repo"], pin["sha"], src)]
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
            if (".so" in f or os.access(p, os.X_OK)) and os.path.isfile(p) and not f.endswith((".sh", ".py", ".cmake")):
                out.append(p)
    return sorted(out)


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
    ap.add_argument("--plan", action="store_true", help="print the commands and exit without running anything")
    a = ap.parse_args(argv)
    pins = load_pins(a.pins)
    if a.engine not in pins:
        print("unknown engine %r; known: %s" % (a.engine, ", ".join(sorted(pins))), file=sys.stderr)
        return 2
    pin = pins[a.engine]
    gen = a.generator if a.generator != "auto" else ("ninja" if shutil.which("ninja") else "make")
    src_root, build_root = os.path.abspath(a.src_root), os.path.abspath(a.build_root)
    steps, src, build = plan(a.engine, pin, a.arch, a.nvcc, src_root, build_root, a.jobs, gen, a.extra_cmake)
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
        missing = [b for b in pin["bins"] if not os.path.isfile(os.path.join(build, b))]
        cuobjdump = shutil.which("cuobjdump") or (os.path.join(os.path.dirname(a.nvcc), "cuobjdump")
                                                 if os.path.isfile(os.path.join(os.path.dirname(a.nvcc), "cuobjdump"))
                                                 else None)
        archs = {}
        for f in built_files(build):
            got = cubin_archs(f, cuobjdump)
            if got:
                archs[os.path.relpath(f, build)] = got
        info.update({"bins": {b: os.path.isfile(os.path.join(build, b)) for b in pin["bins"]},
                     "cubin_archs": archs if cuobjdump else "cuobjdump not found"})
        want = "sm_%d" % a.arch
        if head != pin["sha"]:
            info["status"] = "failed: checked out %s, pinned %s" % (head, pin["sha"])
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
