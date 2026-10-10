#!/usr/bin/env python3
"""Crash-safe, verified store of llama-server slot states for long agent sessions (K3 / R5.2: state on NVMe survives
restarts, so a 262K-token session is restored instead of prefilled again).

Why a store and not just `/slots/{id}?action=save|restore`: llama-server @79e2e74 restores a slot file with a flipped
bit in its state payload without any error, and the corrupted state then changes the generated tokens or yields NaN
log-probs (docs/evidence/2026-10-10-n8-session-store.md); it also has no notion of which model / engine / KV settings
wrote a file, and its own write is not atomic. This store adds, around the documented endpoints only:
  * verification: sha256 and size of every state file recorded at save time and checked before every restore; the
    server's n_restored must equal the recorded token count; anything else -> the file is quarantined, None is returned
    and the caller prefills again (never a silently wrong state);
  * identity: a caller-given dict (model sha256, engine commit, KV type, context...) stored with the state; a restore
    under a different identity is refused (the file is kept for the identity that wrote it);
  * crash consistency: the server writes to a temporary name; the store checks it, fsyncs it, renames it to a
    generation-numbered name and only then commits the per-session manifest (write temp + fsync + rename + fsync dir);
    the previous generation is deleted after the commit. Any crash leaves either the old or the new generation
    restorable; recover() removes temporaries, orphans and dangling manifests;
  * a disk budget with least-recently-used eviction; a session whose save is in progress in this process (including
    the one being saved) is never evicted.
Layout: everything in the server's --slot-save-path directory (the server accepts plain file names only):
  <sid>.<gen>.slot  state files;  <sid>.manifest.json  manifests;  tmp-*.slot  in-flight saves;  quarantine/  rejects.
Single writer per store directory (an fcntl lock file enforces it across processes); one lock per session in-process,
never held while another session's lock is taken (eviction runs after the saving session's lock is released).
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

SID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class StoreError(Exception):
    pass


def _post(url, path, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SessionStore:
    def __init__(self, directory, identity, budget_bytes=None, timeout=3600, hooks=None):
        self.dir = os.path.abspath(directory)
        self.identity = dict(identity)
        self.budget = budget_bytes
        self.timeout = timeout
        self.hooks = hooks or {}          # test-only crash points: name -> callable
        self._locks = {}
        self._saving = set()              # sessions with a save in progress in this process: never evicted
        self.last_reason = None           # why the last restore returned None (observability)
        self._guard = threading.Lock()
        os.makedirs(os.path.join(self.dir, "quarantine"), exist_ok=True)
        self._lockfile = open(os.path.join(self.dir, ".store.lock"), "a+")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lockfile.close()
            raise StoreError(f"another process holds the store lock in {self.dir}")

    def close(self):
        fcntl.flock(self._lockfile, fcntl.LOCK_UN)
        self._lockfile.close()

    # -- helpers --------------------------------------------------------------------------------------------------
    def _lock(self, sid):
        with self._guard:
            return self._locks.setdefault(sid, threading.Lock())

    def _hook(self, name):
        if name in self.hooks:
            self.hooks[name]()

    def _manifest_path(self, sid):
        return os.path.join(self.dir, f"{sid}.manifest.json")

    def manifest(self, sid):
        try:
            with open(self._manifest_path(sid)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _write_manifest(self, sid, m):
        tmp = os.path.join(self.dir, f".{sid}.manifest.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        self._hook("manifest_tmp_written")
        os.rename(tmp, self._manifest_path(sid))
        _fsync_dir(self.dir)

    def _quarantine(self, fname, why):
        src = os.path.join(self.dir, fname)
        if os.path.exists(src):
            # unique name: the same <sid>.<gen>.slot can be rejected more than once and must not overwrite a reject
            dst = os.path.join(self.dir, "quarantine", f"{time.time_ns()}-{uuid.uuid4().hex[:8]}-{fname}")
            os.rename(src, dst)
            with open(dst + ".why", "w") as f:
                f.write(why + "\n")

    def sessions(self):
        out = []
        for name in os.listdir(self.dir):
            if name.endswith(".manifest.json") and not name.startswith("."):
                m = self.manifest(name[:-len(".manifest.json")])
                if m:
                    out.append(m)
        return out

    def used_bytes(self):
        return sum(m["size"] for m in self.sessions())

    # -- save / restore -------------------------------------------------------------------------------------------
    def save(self, url, slot, sid):
        """Save the slot's state as session `sid`. Returns the committed manifest; raises StoreError on failure (the
        previous generation, if any, stays restorable)."""
        if not SID.match(sid):
            raise StoreError(f"bad session id {sid!r}")
        with self._guard:
            self._saving.add(sid)
        try:
            return self._save(url, slot, sid)
        finally:
            with self._guard:
                self._saving.discard(sid)

    def _save(self, url, slot, sid):
        with self._lock(sid):
            tmp = f"tmp-{uuid.uuid4().hex}.slot"
            tmp_path = os.path.join(self.dir, tmp)
            try:
                resp = _post(url, f"/slots/{slot}?action=save", {"filename": tmp}, self.timeout)
            except (urllib.error.URLError, OSError) as e:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise StoreError(f"server save failed: {e}")
            self._hook("after_server_write")
            size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else -1
            if size <= 0 or size != resp.get("n_written"):
                self._quarantine(tmp, f"size {size} != n_written {resp.get('n_written')}")
                raise StoreError(f"server wrote {size} bytes, reported {resp.get('n_written')}")
            digest = _sha256(tmp_path)
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            old = self.manifest(sid)
            gen = (old["gen"] + 1) if old else 1
            final = f"{sid}.{gen}.slot"
            os.rename(tmp_path, os.path.join(self.dir, final))
            _fsync_dir(self.dir)
            self._hook("after_rename")
            now = time.time()
            m = {"sid": sid, "gen": gen, "file": final, "size": size, "sha256": digest, "n_tokens": resp.get("n_saved"),
                 "identity": self.identity, "created": now, "last_used": now}
            self._write_manifest(sid, m)                      # commit point
            self._hook("after_manifest")
            if old and old["file"] != final and os.path.exists(os.path.join(self.dir, old["file"])):
                os.remove(os.path.join(self.dir, old["file"]))
        # evict with no session lock held: two threads saving different sessions must not wait on each other's lock
        self._evict(keep=sid)
        return m

    def restore(self, url, slot, sid):
        """Restore session `sid` into the slot. Returns the server's response, or None when there is nothing valid to
        restore (unknown session, other identity, failed verification, server refusal) - the caller then prefills."""
        with self._lock(sid):
            self.last_reason = None
            m = self.manifest(sid)
            if not m:
                self.last_reason = "unknown session"
                return None
            if m.get("identity") != self.identity:
                self.last_reason = "identity mismatch"         # written by another model/engine/settings: keep it
                return None
            path = os.path.join(self.dir, m["file"])
            why = None
            if not os.path.exists(path):
                why = "file missing"
            elif os.path.getsize(path) != m["size"]:
                why = f"size {os.path.getsize(path)} != {m['size']}"
            elif _sha256(path) != m["sha256"]:
                why = "sha256 mismatch"
            if why is None:
                try:
                    resp = _post(url, f"/slots/{slot}?action=restore", {"filename": m["file"]}, self.timeout)
                except urllib.error.HTTPError as e:            # the server answered: it rejects this file
                    resp, why = None, f"server refused: {e}"
                except (urllib.error.URLError, OSError) as e:  # no answer: transient, the file is not judged
                    raise StoreError(f"server unreachable during restore: {e}")
                if resp is not None and resp.get("n_restored") != m["n_tokens"]:
                    why = f"n_restored {resp.get('n_restored')} != {m['n_tokens']}"
            if why is not None:
                self.last_reason = why
                self._quarantine(m["file"], why)
                os.remove(self._manifest_path(sid))
                _fsync_dir(self.dir)
                try:
                    _post(url, f"/slots/{slot}?action=erase", {}, self.timeout)   # never leave a half-loaded slot
                except (urllib.error.URLError, OSError):
                    pass
                return None
            m["last_used"] = time.time()
            self._write_manifest(sid, m)
            return resp

    def drop(self, sid):
        with self._lock(sid):
            m = self.manifest(sid)
            if m:
                os.remove(self._manifest_path(sid))
                _fsync_dir(self.dir)
                p = os.path.join(self.dir, m["file"])
                if os.path.exists(p):
                    os.remove(p)

    def _evict(self, keep):
        if self.budget is None:
            return
        with self._guard:
            busy = set(self._saving) | {keep}
        ms = sorted((m for m in self.sessions() if m["sid"] not in busy), key=lambda m: m["last_used"])
        used = self.used_bytes()
        for m in ms:
            if used <= self.budget:
                break
            self.drop(m["sid"])
            used -= m["size"]

    # -- recovery -------------------------------------------------------------------------------------------------
    def recover(self, deep=False):
        """After a crash: delete in-flight temporaries and orphaned state files, drop manifests whose file is gone or
        (deep) whose sha256 does not match. Returns what was done."""
        done = {"tmp_removed": [], "orphans_removed": [], "manifests_dropped": []}
        referenced = set()
        for m in self.sessions():
            p = os.path.join(self.dir, m["file"])
            bad = (not os.path.exists(p) or os.path.getsize(p) != m["size"] or
                   (deep and _sha256(p) != m["sha256"]))
            if bad:
                if os.path.exists(p):
                    self._quarantine(m["file"], "recover: verification failed")
                os.remove(self._manifest_path(m["sid"]))
                done["manifests_dropped"].append(m["sid"])
            else:
                referenced.add(m["file"])
        for name in os.listdir(self.dir):
            p = os.path.join(self.dir, name)
            if name.startswith("tmp-") and name.endswith(".slot") or (name.startswith(".") and name.endswith(".tmp")):
                os.remove(p)
                done["tmp_removed"].append(name)
            elif name.endswith(".slot") and name not in referenced:
                os.remove(p)
                done["orphans_removed"].append(name)
        _fsync_dir(self.dir)
        return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the server's --slot-save-path")
    ap.add_argument("--identity", required=True, help="JSON object: model sha256, engine commit, KV type, ...")
    ap.add_argument("--budget-gib", type=float)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("save", "restore"):
        p = sub.add_parser(c)
        p.add_argument("--url", required=True)
        p.add_argument("--slot", type=int, default=0)
        p.add_argument("--sid", required=True)
    sub.add_parser("list")
    r = sub.add_parser("recover")
    r.add_argument("--deep", action="store_true")
    a = ap.parse_args(argv)
    st = SessionStore(a.dir, json.loads(a.identity), int(a.budget_gib * 2**30) if a.budget_gib else None)
    try:
        if a.cmd == "save":
            print(json.dumps(st.save(a.url, a.slot, a.sid)))
        elif a.cmd == "restore":
            res = st.restore(a.url, a.slot, a.sid)
            print(json.dumps(res))
            return 0 if res is not None else 1
        elif a.cmd == "list":
            print(json.dumps(st.sessions(), indent=1))
        else:
            print(json.dumps(st.recover(deep=a.deep)))
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
