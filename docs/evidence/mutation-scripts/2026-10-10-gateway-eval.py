"""Mutation check for tests/test_eval_via_gateway.py: each mutant edits a copy of tools/gateway/gateway.py."""
import os, shutil, subprocess, sys, tempfile
ROOT = sys.argv[1]
ANCHOR = "        body, ctx.injected = apply_defaults(raw, obj, gw.cfg.set_defaults)\n"
MUT = {
  "G1 drop tool_call_id": ANCHOR + "        body = json.dumps({**obj, 'messages': [{k: v for k, v in m.items() if k != 'tool_call_id'} for m in obj.get('messages') or []]}).encode()\n",
  "G2 drop tools": ANCHOR + "        body = json.dumps({k: v for k, v in obj.items() if k != 'tools'}).encode()\n",
  "G3 force non-stream": ANCHOR + "        body = json.dumps({**obj, 'stream': False}).encode() if ctx.path == '/v1/chat/completions' else body\n",
  "G4 drop tool messages": ANCHOR + "        body = json.dumps({**obj, 'messages': [m for m in obj.get('messages') or [] if m.get('role') != 'tool']}).encode()\n",
  "G5 inject temperature": ANCHOR + "        body = json.dumps({**obj, 'temperature': 0.7}).encode()\n",
}
def run(tree):
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_eval_via_gateway"], cwd=tree,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]
res = {}
for name, repl in [("baseline", None)] + list(MUT.items()):
    tree = tempfile.mkdtemp(dir=os.environ.get("SP"))
    for d in ("tools", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(tree, d), ignore=shutil.ignore_patterns("__pycache__"))
    g = os.path.join(tree, "tools", "gateway", "gateway.py")
    s = open(g).read()
    if repl:
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, repl)
        open(g, "w").write(s)
    rc, last = run(tree)
    verdict = ("ok" if rc == 0 else "BROKEN") if repl is None else ("caught" if rc != 0 else "SURVIVED")
    print("%-24s rc=%d %-8s %s" % (name, rc, verdict, last), flush=True)
    shutil.rmtree(tree)
