#!/usr/bin/env python3
"""Offline kit of the R1 engines for an inference server without internet (ROADMAP R1.4; companion of weights.py).

scripts/engines/build_engine.py fetches the pinned sources from GitHub, which the server cannot reach. This tool
carries them, and the Python wheels of torch-extension engines, over the customer's PC:

  pack            (PC with internet) for every selected engine of tools/engines/pins.json and each of its pinned
                  dependencies: a shallow fetch of exactly the pinned commit, written as a one-commit git bundle
                  <kit>/sources/<key>.bundle with the ref refs/pins/<key> (key = <engine> or <engine>-dep-<dep>);
                  for "kind": "torch-extension" engines also `pip download` of the pin's python.install list for the
                  server's Python and platform into <kit>/wheels/<engine>/; <kit>/manifest.json lists the commit of
                  every bundle and the size and sha256 of every file
  verify          (server, offline) every file's size and sha256 against the manifest; every bundle fetched into a
                  scratch repository must give exactly its pinned commit (the commit hash then proves the content);
                  the manifest's commits must equal tools/engines/pins.json
  install-python  (server, offline) a venv with the pinned Python list from <kit>/wheels/<engine>/ (--no-index)

Then on the server: python3 scripts/engines/build_engine.py --engine <name> --offline-kit <kit> [--python <venv>/bin/python]
Exit codes: 0 ok, 1 a check or a command failed, 2 usage. Python 3 stdlib + git (+ pip for the wheel steps).
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
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engines"))
import build_engine as be  # noqa: E402  (pins, kit_key, KIT_REF, SHALLOW: one definition of the kit layout)

SCHEMA = "408032gb-engine-kit/1"
# pip download --platform takes exact tags; wheels carry the oldest glibc they support (torch 2.14.1: manylinux_2_28,
# ninja 1.13.2: manylinux2014 / manylinux_2_17, nvidia-nccl-cu13 2.30.7: manylinux_2_18; read from pypi.org on
# 2026-10-10). The default accepts every manylinux tag up to glibc 2.28, i.e. it assumes the server's glibc >= 2.28
# (PROVISIONAL until R0 measures it: ldd --version on the server).
PLATFORMS = ["manylinux_2_%d_x86_64" % v for v in range(28, 4, -1)] + \
            ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources(name, pin):
    """[(key, repo, sha)] of an engine: its pinned dependencies first, then the engine itself."""
    out = [(be.kit_key(name, dep), d["repo"], d["sha"]) for dep, d in sorted((pin.get("deps") or {}).items())]
    return out + [(be.kit_key(name), pin["repo"], pin["sha"])]


def pack_cmds(kit, key, repo, sha):
    work = os.path.join(kit, ".work", key)
    return work, [["git", "init", "-q", work],
                  ["git", "-C", work, "fetch", "-q", "--depth", "1", repo, sha],
                  ["git", "-C", work, "update-ref", be.KIT_REF % key, sha],
                  ["git", "-C", work, "bundle", "create", be.kit_bundle(kit, key), be.KIT_REF % key]]


def wheel_cmd(kit, name, pin, python, python_version, platforms):
    return [python, "-m", "pip", "download", "--only-binary=:all:", "--python-version", python_version,
            *[x for p in platforms for x in ("--platform", p)], "--implementation", "cp",
            "-d", os.path.join(kit, "wheels", name), *pin["python"]["install"]]


def run(argv, quiet=False):
    p = subprocess.run(argv, capture_output=True, text=True)
    if p.returncode and not quiet:
        sys.stderr.write("$ %s\n%s%s" % (shlex.join(argv), p.stdout[-2000:], p.stderr[-2000:]))
    return p


def file_entry(kit, path):
    return {"file": os.path.relpath(path, kit), "size": os.path.getsize(path), "sha256": sha256_file(path)}


def cmd_pack(a, pins):
    names = a.engine or sorted(pins)
    unknown = [n for n in names if n not in pins]
    if unknown:
        print("unknown engines %s; known: %s" % (unknown, ", ".join(sorted(pins))), file=sys.stderr)
        return 2
    kit = os.path.abspath(a.out)
    platforms = a.platform or PLATFORMS
    plan = []
    for name in names:
        for key, repo, sha in sources(name, pins[name]):
            plan += [("source %s @ %s" % (key, sha[:12]), c) for c in pack_cmds(kit, key, repo, sha)[1]]
        if pins[name].get("kind") == "torch-extension" and not a.no_wheels:
            plan.append(("wheels %s for Python %s, %s" % (name, a.python_version, " ".join(platforms)),
                         wheel_cmd(kit, name, pins[name], a.python, a.python_version, platforms)))
    if a.plan:
        print("# offline kit plan -> %s (nothing is run)" % kit)
        for desc, argv in plan:
            print("# %s\n%s" % (desc, shlex.join(argv)))
        return 0
    if os.path.exists(kit) and os.listdir(kit):
        print("%s exists and is not empty: pack into a new directory" % kit, file=sys.stderr)
        return 2
    os.makedirs(os.path.join(kit, "sources"), exist_ok=True)
    manifest = {"schema": SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "pins_sha256": sha256_file(a.pins), "engines": {}}
    for name in names:
        pin = pins[name]
        entry = {"repo": pin["repo"], "sha": pin["sha"], "kind": pin.get("kind", "cmake"), "sources": {}, "wheels": None}
        for key, repo, sha in sources(name, pin):
            work, cmds = pack_cmds(kit, key, repo, sha)
            for argv in cmds:
                if run(argv).returncode:
                    print("pack failed at %s: %s" % (key, shlex.join(argv)), file=sys.stderr)
                    return 1
            got = run(["git", "-C", work, "rev-parse", be.KIT_REF % key]).stdout.strip()
            shutil.rmtree(work)
            if got != sha:
                print("pack %s: the ref points at %s, pinned %s" % (key, got, sha), file=sys.stderr)
                return 1
            entry["sources"][key] = dict(file_entry(kit, be.kit_bundle(kit, key)), sha=sha, repo=repo,
                                         ref=be.KIT_REF % key)
            print("source  %-34s %s  %s" % (key, sha[:12], entry["sources"][key]["sha256"][:16]), flush=True)
        if pin.get("kind") == "torch-extension" and not a.no_wheels:
            argv = wheel_cmd(kit, name, pin, a.python, a.python_version, platforms)
            if run(argv).returncode:
                print("pip download failed for %s" % name, file=sys.stderr)
                return 1
            wdir = os.path.join(kit, "wheels", name)
            entry["wheels"] = {"python_version": a.python_version, "platforms": platforms,
                               "install": pin["python"]["install"],
                               "files": [file_entry(kit, os.path.join(wdir, f)) for f in sorted(os.listdir(wdir))]}
            print("wheels  %-34s %d files, %d bytes" % (name, len(entry["wheels"]["files"]),
                                                        sum(f["size"] for f in entry["wheels"]["files"])), flush=True)
        manifest["engines"][name] = entry
    shutil.rmtree(os.path.join(kit, ".work"), ignore_errors=True)
    with open(os.path.join(kit, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print("kit: %s (%d engines)" % (kit, len(names)))
    return 0


def check_bundle(path, key, sha):
    """Fetch the bundle into a scratch repository (shallow boundary at the pinned commit) -> (ok, detail)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "r")
        for argv in (["git", "init", "-q", repo], [sys.executable, "-c", be.SHALLOW, repo, sha],
                     ["git", "-C", repo, "fetch", "-q", path, be.KIT_REF % key]):
            p = run(argv, quiet=True)
            if p.returncode:
                return False, "fetch failed: " + (p.stderr.strip().splitlines() or ["?"])[-1]
        got = run(["git", "-C", repo, "rev-parse", "FETCH_HEAD"], quiet=True).stdout.strip()
        if got != sha:
            return False, "the bundle gives %s, pinned %s" % (got, sha)
        fsck = run(["git", "-C", repo, "fsck", "--no-dangling"], quiet=True)
        return (fsck.returncode == 0), ("commit %s, fsck ok" % sha[:12] if fsck.returncode == 0 else
                                        "fsck: " + fsck.stderr.strip()[-200:])


def cmd_verify(a, pins):
    kit = os.path.abspath(a.kit)
    try:
        with open(os.path.join(kit, "manifest.json")) as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        print("cannot read %s/manifest.json: %s" % (kit, e), file=sys.stderr)
        return 2
    bad = 0

    def line(ok, what, detail):
        nonlocal bad
        bad += not ok
        print("%-4s %-44s %s" % ("ok" if ok else "FAIL", what, detail))
    if m.get("schema") != SCHEMA:
        line(False, "manifest schema", "%r, expected %r" % (m.get("schema"), SCHEMA))
    for name, e in sorted((m.get("engines") or {}).items()):
        pin = pins.get(name)
        want = {k: s for k, _r, s in sources(name, pin)} if pin else None
        got = {k: v.get("sha") for k, v in (e.get("sources") or {}).items()}
        line(want == got, "pins %s" % name, "kit %s / pins.json %s" % (got, want) if want != got else "= tools/engines/pins.json")
        for key, s in sorted((e.get("sources") or {}).items()):
            path = os.path.join(kit, s["file"])
            if not os.path.isfile(path):
                line(False, "source %s" % key, "missing %s" % s["file"])
                continue
            same = os.path.getsize(path) == s["size"] and sha256_file(path) == s["sha256"]
            line(same, "source %s sha256" % key, s["sha256"][:16] if same else "differs from the manifest")
            ok, detail = check_bundle(path, key, s["sha"])
            line(ok, "source %s commit" % key, detail)
        for w in ((e.get("wheels") or {}).get("files") or []):
            path = os.path.join(kit, w["file"])
            same = os.path.isfile(path) and os.path.getsize(path) == w["size"] and sha256_file(path) == w["sha256"]
            line(same, "wheel %s" % os.path.basename(w["file"]), w["sha256"][:16] if same else "missing or differs")
    left_out = sorted(set(pins) - set(m.get("engines") or {}))
    if left_out:                                      # a kit may carry a subset (pack --engine); say so, do not fail
        print("note pins.json engines not in this kit: %s" % ", ".join(left_out))
    print("verify: %s" % ("ok" if not bad else "%d check(s) failed" % bad))
    return 1 if bad else 0


def cmd_install_python(a, pins):
    kit = os.path.abspath(a.kit)
    pin = pins.get(a.engine)
    if not pin or pin.get("kind") != "torch-extension":
        print("%s is not a torch-extension engine of pins.json" % a.engine, file=sys.stderr)
        return 2
    venv = os.path.abspath(a.venv)
    cmds = [[a.python, "-m", "venv", venv],
            [os.path.join(venv, "bin", "pip"), "install", "--no-index", "--find-links", os.path.join(kit, "wheels", a.engine),
             *pin["python"]["install"]]]
    if a.plan:
        for argv in cmds:
            print(shlex.join(argv))
        return 0
    for argv in cmds:
        if run(argv).returncode:
            return 1
    print("venv: %s (build with: build_engine.py --engine %s --offline-kit %s --python %s/bin/python)"
          % (venv, a.engine, kit, venv))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pins", default=be.PINS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", required=True, help="new, empty kit directory")
    p.add_argument("--engine", action="append", help="engine of pins.json (repeatable; default: all)")
    p.add_argument("--python", default=sys.executable, help="interpreter whose pip downloads the wheels")
    p.add_argument("--python-version", default="%d.%d" % sys.version_info[:2],
                   help="the server's Python (X.Y): the wheels' cpXY tag; UNKNOWN until R0, so set it explicitly")
    p.add_argument("--platform", action="append", help="pip platform tag (repeatable); default: manylinux up to 2_28")
    p.add_argument("--no-wheels", action="store_true", help="sources only")
    p.add_argument("--plan", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("kit")
    i = sub.add_parser("install-python")
    i.add_argument("kit")
    i.add_argument("--engine", required=True)
    i.add_argument("--venv", required=True)
    i.add_argument("--python", default=sys.executable)
    i.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    pins = be.load_pins(a.pins)
    return {"pack": cmd_pack, "verify": cmd_verify, "install-python": cmd_install_python}[a.cmd](a, pins)


if __name__ == "__main__":
    sys.exit(main())
