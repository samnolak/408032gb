"""tools/kv/session_store.py against the stand-in llama-server (tests/fixtures/srvcheck/stub_llama_server.py), started as a
process with --slot-save-path = the store directory, exactly like the real server. Real-server runs: N8 evidence."""
import os
import stat
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "kv"))
sys.path.insert(0, os.path.join(ROOT, "tools", "srvcheck"))
import hybrid_server_checks as hc  # noqa: E402
import session_store as ss  # noqa: E402

STUB = os.path.join(ROOT, "tests", "fixtures", "srvcheck", "stub_llama_server.py")
IDENT = {"model_sha256": "a" * 64, "engine": "llama.cpp 79e2e74", "ctk": "f32"}
PROMPT = list(range(3, 63))


class Crash(Exception):
    pass


class Store(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bin_dir = tempfile.TemporaryDirectory()
        cls.binary = os.path.join(cls.bin_dir.name, "llama-server")
        with open(cls.binary, "w") as f:
            f.write(f"#!/bin/sh\nexec {sys.executable} {STUB} \"$@\"\n")
        os.chmod(cls.binary, os.stat(cls.binary).st_mode | stat.S_IEXEC)

    @classmethod
    def tearDownClass(cls):
        cls.bin_dir.cleanup()

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.servers, self.stores = [], []
        os.environ.pop("STUB_FAULT", None)

    def tearDown(self):
        os.environ.pop("STUB_FAULT", None)
        for s in self.servers:
            s.stop()
        for st in self.stores:
            try:
                st.close()
            except ValueError:
                pass

    def server(self, fault=""):
        os.environ["STUB_FAULT"] = fault
        s = hc.Server(self.binary, "m.gguf", ["--slot-save-path", self.dir], os.path.join(self.dir, "srv.log"))
        os.environ.pop("STUB_FAULT", None)
        self.servers.append(s)
        return s

    def store(self, identity=IDENT, **kw):
        st = ss.SessionStore(self.dir, identity, **kw)
        self.stores.append(st)
        return st

    def restart(self, fault=""):
        self.servers.pop().stop()
        return self.server(fault)

    def slot_files(self):
        return sorted(n for n in os.listdir(self.dir) if n.endswith(".slot"))

    def saved_session(self, sid="s1", fault=""):
        srv = self.server(fault)
        ref = hc.complete(srv.url, PROMPT, 6)
        m = self.store().save(srv.url, 0, sid)
        return srv, ref, m

    def test_round_trip_across_a_restart(self):
        srv, ref, m = self.saved_session()
        self.assertEqual((m["gen"], m["n_tokens"]), (1, len(PROMPT) + 5))
        srv = self.restart()
        resp = self.stores[0].restore(srv.url, 0, "s1")
        self.assertEqual(resp["n_restored"], m["n_tokens"])
        again = hc.complete(srv.url, PROMPT, 6)
        self.assertEqual(again["tokens"], ref["tokens"])
        self.assertEqual(again["prompt_n"], 4)                  # restored up to the checkpoint, not prefilled again

    def corrupt_and_restore(self, mutate):
        srv, ref, m = self.saved_session()
        path = os.path.join(self.dir, m["file"])
        with open(path, "rb") as f:
            data = bytearray(f.read())
        mutate(data)
        with open(path, "wb") as f:
            f.write(bytes(data))
        srv = self.restart()
        st = self.stores[0]
        self.assertIsNone(st.restore(srv.url, 0, "s1"))
        self.assertIsNone(st.manifest("s1"))
        self.assertEqual(self.slot_files(), [])
        self.assertEqual(len([n for n in os.listdir(os.path.join(self.dir, "quarantine")) if n.endswith(".slot")]), 1)
        self.assertEqual(hc.complete(srv.url, PROMPT, 6)["prompt_n"], len(PROMPT))   # prefilled, nothing half-loaded
        return st.last_reason

    def test_flipped_byte_is_refused_before_the_server_sees_it(self):
        def flip(d):
            i = len(d) // 2
            d[i] = ord("7") if d[i] != ord("7") else ord("3")
        self.assertEqual(self.corrupt_and_restore(flip), "sha256 mismatch")

    def test_truncated_file_is_refused(self):
        self.assertTrue(self.corrupt_and_restore(lambda d: d.__delitem__(slice(len(d) - 5, len(d)))).startswith("size"))

    def test_other_identity_is_refused_and_the_file_kept(self):
        srv, _, m = self.saved_session()
        self.stores[0].close()
        self.stores.pop()
        other = self.store(dict(IDENT, ctk="q8_0"))
        self.assertIsNone(other.restore(srv.url, 0, "s1"))
        self.assertEqual(other.last_reason, "identity mismatch")
        self.assertEqual(self.slot_files(), [m["file"]])

    def test_server_refusal_quarantines(self):
        srv, _, _ = self.saved_session()
        srv = self.restart("restore_refuse")
        st = self.stores[0]
        self.assertIsNone(st.restore(srv.url, 0, "s1"))
        self.assertTrue(st.last_reason.startswith("server refused"))
        self.assertEqual(self.slot_files(), [])

    def test_restore_count_mismatch_quarantines(self):
        srv, _, _ = self.saved_session()
        srv = self.restart("restore_partial")
        st = self.stores[0]
        self.assertIsNone(st.restore(srv.url, 0, "s1"))
        self.assertTrue(st.last_reason.startswith("n_restored"))
        # the server had loaded the state: the store must erase the slot, not leave a half-trusted state in it
        self.assertEqual(hc.complete(srv.url, PROMPT, 6)["prompt_n"], len(PROMPT))

    def test_short_write_keeps_the_previous_generation(self):
        srv, ref, m1 = self.saved_session()
        srv = self.restart("save_short")
        hc.complete(srv.url, PROMPT + [70, 71], 3)
        with self.assertRaises(ss.StoreError):
            self.stores[0].save(srv.url, 0, "s1")
        self.assertEqual(self.stores[0].manifest("s1")["gen"], 1)
        self.assertEqual(self.slot_files(), [m1["file"]])
        srv = self.restart()
        self.assertIsNotNone(self.stores[0].restore(srv.url, 0, "s1"))

    def test_crash_at_every_save_step_leaves_a_restorable_store(self):
        for point, expect_gen in (("after_server_write", 1), ("after_rename", 1), ("manifest_tmp_written", 1),
                                  ("after_manifest", 2)):
            with self.subTest(point=point):
                self.tearDown()
                self.setUp()
                srv, _, m1 = self.saved_session()
                self.stores[0].close()
                self.stores.pop()

                def boom():
                    raise Crash(point)
                st = self.store(hooks={point: boom})
                hc.complete(srv.url, PROMPT + [80, 81, 82], 3)
                with self.assertRaises(Crash):
                    st.save(srv.url, 0, "s1")
                st.close()
                self.stores.pop()
                st = self.store()                                  # the process came back
                done = st.recover(deep=True)
                m = st.manifest("s1")
                self.assertEqual(m["gen"], expect_gen, done)
                self.assertEqual(self.slot_files(), [m["file"]], done)   # no temporaries, no orphans
                self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".tmp")])
                srv = self.restart()
                self.assertEqual(st.restore(srv.url, 0, "s1")["n_restored"], m["n_tokens"])

    def test_budget_evicts_least_recently_used(self):
        srv = self.server()
        hc.complete(srv.url, PROMPT, 3)
        st = self.store()
        m1 = st.save(srv.url, 0, "s1")
        st.budget = int(m1["size"] * 1.5)
        hc.complete(srv.url, PROMPT + [90], 3)
        st.save(srv.url, 0, "s2")
        self.assertEqual([m["sid"] for m in st.sessions()], ["s2"])
        self.assertIsNone(st.restore(srv.url, 0, "s1"))
        self.assertEqual(st.last_reason, "unknown session")

    def test_session_just_saved_is_never_evicted(self):
        srv = self.server()
        hc.complete(srv.url, PROMPT, 3)
        st = self.store(budget_bytes=10)                       # smaller than any state file
        m = st.save(srv.url, 0, "s1")
        self.assertEqual(st.manifest("s1")["file"], m["file"])
        self.assertEqual(self.slot_files(), [m["file"]])

    def test_concurrent_saves_under_a_tight_budget_do_not_deadlock(self):
        import threading
        srv0, srv1 = self.server(), self.server()            # two servers share the store directory
        for srv in (srv0, srv1):
            hc.complete(srv.url, PROMPT, 3)
        st = self.store()
        size = st.save(srv0.url, 0, "warm")["size"]
        st.budget = size                                      # room for about one session: every save evicts
        errors = []

        def worker(srv, sid):
            try:
                for _ in range(15):
                    st.save(srv.url, 0, sid)
            except Exception as e:  # recorded
                errors.append(repr(e))
        ts = [threading.Thread(target=worker, args=(srv0, "a"), daemon=True),
              threading.Thread(target=worker, args=(srv1, "b"), daemon=True)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        self.assertFalse(any(t.is_alive() for t in ts), "deadlock: a saving thread did not finish")
        self.assertEqual(errors, [])
        st.recover(deep=True)
        self.assertLessEqual(len(st.sessions()), 2)

    def test_repeated_rejects_of_the_same_file_name_are_all_kept(self):
        # audit M11: quarantine names were <seconds>-<file>, so a second reject of <sid>.1.slot overwrote the first
        srv = self.server()
        st = self.store()
        names = []
        for _ in range(3):
            hc.complete(srv.url, PROMPT, 3)
            m = st.save(srv.url, 0, "s1")
            names.append(m["file"])
            with open(os.path.join(self.dir, m["file"]), "r+b") as f:
                f.seek(10)
                b = f.read(1)
                f.seek(10)
                f.write(bytes([b[0] ^ 0x40]))
            self.assertIsNone(st.restore(srv.url, 0, "s1"))
        self.assertEqual(set(names), {"s1.1.slot"})                     # the same name every time
        q = [n for n in os.listdir(os.path.join(self.dir, "quarantine")) if n.endswith(".slot")]
        self.assertEqual(len(q), 3, q)

    def test_session_being_saved_by_another_thread_is_not_evicted(self):
        # audit M12: eviction excluded only the caller's own session
        import threading
        srv_a, srv_b = self.server(), self.server()
        for srv in (srv_a, srv_b):
            hc.complete(srv.url, PROMPT, 3)
        size = self.store().save(srv_a.url, 0, "warm")["size"]
        self.stores[0].drop("warm")
        entered, release = threading.Event(), threading.Event()

        def pause():
            if threading.current_thread().name == "saver-a":
                entered.set()
                release.wait(30)
        st = self.stores[0]
        st.hooks = {"after_manifest": pause}
        st.budget = size                                      # room for one session
        ta = threading.Thread(target=st.save, args=(srv_a.url, 0, "a"), name="saver-a", daemon=True)
        ta.start()
        self.assertTrue(entered.wait(30))                     # a's manifest is committed, its save still in progress
        tb = threading.Thread(target=st.save, args=(srv_b.url, 0, "b"), daemon=True)
        tb.start()
        tb.join(15)                                           # fixed store: b's save does not touch a
        release.set()
        ta.join(30)
        tb.join(30)                                           # old store: b waited for a's lock, then dropped a
        self.assertFalse(ta.is_alive() or tb.is_alive())
        self.assertIsNotNone(st.manifest("a"), "b's eviction removed a while a was being saved")

    def test_new_generation_replaces_the_old_file(self):
        srv, _, m1 = self.saved_session()
        hc.complete(srv.url, PROMPT + [91, 92], 3)
        m2 = self.stores[0].save(srv.url, 0, "s1")
        self.assertEqual((m2["gen"], self.slot_files()), (2, [m2["file"]]))
        self.assertNotEqual(m1["file"], m2["file"])

    def test_second_writer_is_refused(self):
        self.store()
        with self.assertRaises(ss.StoreError):
            ss.SessionStore(self.dir, IDENT)

    def test_unreachable_server_does_not_judge_the_file(self):
        srv, _, m = self.saved_session()
        url = srv.url
        self.servers.pop().stop()
        with self.assertRaises(ss.StoreError):
            self.stores[0].restore(url, 0, "s1")
        self.assertEqual(self.stores[0].manifest("s1")["file"], m["file"])
        self.assertEqual(self.slot_files(), [m["file"]])

    def test_bad_session_id(self):
        srv = self.server()
        with self.assertRaises(ss.StoreError):
            self.store().save(srv.url, 0, "../x")


if __name__ == "__main__":
    unittest.main()
