"""N6: the agentic evaluation stand (tools/eval, N3) run THROUGH the API gateway (tools/gateway) against the scripted
mock agent server, as the product will be evaluated (clients only see the gateway, DEPLOY.md section 3).

Checks that the gateway is transparent for the stand: every SYNTHETIC task passes through it exactly as directly,
the request bodies the backend receives are byte-for-byte the same JSON (tool calls, tool results, tool_call_id),
streaming timings are still recorded, auth failures surface as request errors (not as model failures), and every
request appears in the gateway's JSONL log. Mock only: nothing here says anything about a real engine (NOT RUN).
"""
import json
import logging
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "gateway"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import gateway as gw  # noqa: E402
import test_eval as te  # noqa: E402   (reuses its mock server, task set and correct/wrong scripts)

KEY = "SYNTHETIC-test-key-not-a-secret"
MOCK_ID = re.compile(r"call_\d+_\d+_\d+")   # mock_agent_server ids embed its global request counter


def normalized(body):
    """The request body as canonical JSON with mock-generated tool_call ids renamed by order of first appearance
    inside this body (the counter differs between two runs; the pairing of calls and results must not)."""
    names = {}
    return MOCK_ID.sub(lambda m: names.setdefault(m.group(0), "ID%d" % len(names)), json.dumps(body, sort_keys=True))


def id_pairing_ok(body):
    """Every tool message answers a tool call made by an earlier assistant message in the same conversation."""
    seen = set()
    for m in body.get("messages") or []:
        for c in m.get("tool_calls") or []:
            seen.add(c.get("id"))
        if m.get("role") == "tool" and m.get("tool_call_id") not in seen:
            return False
    return True


def setUpModule():
    te.setUpModule()
    logging.getLogger("gateway").setLevel(logging.ERROR)


def tearDownModule():
    te.tearDownModule()


class ViaGateway(te.RunCase):
    def start_gateway(self, **kw):
        log = os.path.join(self.tmp, "gw-%d.jsonl" % self.n)
        g = gw.Gateway(gw.GatewayConfig(te.SRV.url, listen_port=0, log_file=log, **kw)).start()
        self.addCleanup(g.stop)
        return g, log

    def bodies(self):
        return [normalized(b) for b in te.SRV.requests]

    def test_stand_through_gateway_equals_direct(self):
        direct = self.run_eval(["--repeats", "1"], te.scripts("correct"))
        self.assertEqual(direct["code"], 0, direct["stdout"] + direct["stderr"])
        direct_bodies = self.bodies()

        g, log = self.start_gateway()
        via = self.run_eval(["--repeats", "1"], te.scripts("correct"), url=g.url)
        self.assertEqual(via["code"], 0, via["stdout"] + via["stderr"])
        self.assertEqual({k: v["status"] for k, v in via["by_id"].items()},
                         {k: v["status"] for k, v in direct["by_id"].items()})
        self.assertEqual(set(v["status"] for v in via["by_id"].values()), {"pass"})
        self.assertEqual(self.bodies(), direct_bodies)              # the backend saw the same requests
        self.assertTrue(any(m.get("role") == "tool" for b in te.SRV.requests for m in b.get("messages") or []),
                        "the task set must exercise tool calls through the gateway")
        self.assertTrue(all(id_pairing_ok(b) for b in te.SRV.requests))
        req = [x for x in via["raw"] if x["kind"] == "request"]
        self.assertTrue(all(x["metrics"]["ttft_ms"] is not None for x in req))
        self.assertTrue(via["meta"]["is_mock_server"])              # mock detection survives the proxy
        with open(log) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        gen = [r for r in rows if r.get("path") == "/v1/chat/completions"]
        self.assertEqual(len(gen), len(te.SRV.requests))
        self.assertEqual({r.get("status") for r in gen}, {200})

    def test_wrong_answers_still_fail_through_gateway(self):
        g, _ = self.start_gateway()
        via = self.run_eval(["--repeats", "1"], te.scripts("wrong"), url=g.url)
        self.assertEqual(set(v["status"] for v in via["by_id"].values()), {"fail"})

    def test_auth_failure_is_a_request_error_not_a_model_failure(self):
        g, _ = self.start_gateway(api_keys=[KEY])
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            no_key = self.run_eval(["--repeats", "1", "--only", "reasoning"], te.scripts("correct"), url=g.url)
            os.environ["OPENAI_API_KEY"] = KEY
            with_key = self.run_eval(["--repeats", "1", "--only", "reasoning"], te.scripts("correct"), url=g.url)
        finally:
            if old is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old
        statuses = {v["status"] for v in no_key.get("by_id", {}).values()}
        self.assertNotIn("pass", statuses)
        self.assertNotIn("fail", statuses, "an auth error must not be scored as a wrong answer")
        self.assertEqual({v["status"] for v in with_key["by_id"].values()}, {"pass"})


if __name__ == "__main__":
    unittest.main()
