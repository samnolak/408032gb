"""N26a mutation check: break one rule at a time in the worktree, run the tests, restore the file (always)."""
import os, re, subprocess, sys
WT = "/home/user/wt-n26a"
SR, RR, GW = "scripts/r1/safety_rules.py", "scripts/r1/run_r1.py", "tools/gateway/gateway.py"
MUT = [
 ("M1 R-NP: explicit -np not required", SR, '''        if not given:
            out.append(''', '''        if False:
            out.append('''),
 ("M2 R-NP: rs_rollback_rule.check() not used", SR, "        probs = rs.check(argv, env)\n", "        probs = []\n"),
 ("M3 R-NP: 'any speculation' part dropped", SR, "        if spec and not probs and slots != 1:", "        if False:"),
 ("M4 R-GLM-TEMPLATE: --chat-template-file not required", SR, "        if not files:\n            out.append(\"GLM with --jinja", "        if False:\n            out.append(\"GLM with --jinja"),
 ("M5 R-GLM-TEMPLATE: path not compared", SR, "            if os.path.normpath(f) != want:", "            if False:"),
 ("M6 R-GLM-TEMPLATE: sha256 not compared", SR, "                if got != sha:", "                if False:"),
 ("M7 R-MTP-FILES: -shared- accepted", SR, '            if "-shared-" in base:', '            if False:'),
 ("M8 R-MTP-FILES: any basename accepted", SR, "            elif not QWEN_MTP_FILE.match(base):", "            elif False:"),
 ("M9 R-MTP-FILES: -hfd accepted", SR, "        if hf:\n", "        if False:\n"),
 ("M10 R-MIMO-MTP: rule off", SR, '    if "mimo" not in model_text(cfg):\n        return []', '    if True:\n        return []'),
 ("M11 R-WEIGHTS: rule off", SR, "    if is_mock(cfg):\n        return []\n    m = cfg", "    if True:\n        return []\n    m = cfg"),
 ("M12 R-SYNTH: rule off", SR, "    out = []\n    m = cfg.get(\"model\") if isinstance(cfg.get(\"model\"), dict) else {}\n    marked", "    return []\n    m = cfg.get(\"model\") if isinstance(cfg.get(\"model\"), dict) else {}\n    marked"),
 ("M13 run_r1: violations not refused", RR, '        if self.safety["violations"]:\n            raise ConfigError(', '        if False:\n            raise ConfigError('),
 ("M14 run_r1: weights gate open", RR, "        if c.mock:\n            return None\n        st = self.step_state.get(\"weights_verify\")", "        return None\n        st = self.step_state.get(\"weights_verify\")"),
 ("M15 run_r1: --only engine_start does not add weights_verify", RR, '            selected.add("weights_verify")', '            pass'),
 ("M16 gateway: null not dropped", GW, '            raw, obj, ctx.dropped = drop_null_reasoning(raw, obj)', '            pass'),
 ("M17 gateway: whole body re-serialised (bytes not kept)", GW, '        body = "".join(out).encode("utf-8")\n', '        body = json.dumps(obj).encode("utf-8")\n'),
]
TESTS = ["tests.test_r1_orchestration", "tests.test_r1_rs_rollback_rule", "tests.test_gateway",
         "tests.test_eval_via_gateway", "tests.test_r1_tools_smoke", "tests.test_r1_safety_rules"]
rows = []
for name, rel, a, b in MUT:
    path = os.path.join(WT, rel)
    orig = open(path).read()
    assert orig.count(a) == 1, (name, orig.count(a))
    try:
        open(path, "w").write(orig.replace(a, b))
        p = subprocess.run([sys.executable, "-m", "unittest"] + TESTS, cwd=WT, capture_output=True, text=True,
                           timeout=1200)
        out = p.stdout + p.stderr
        fails = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(([\w.]+)\)", out, re.M)))
        summary = [l for l in out.splitlines() if re.match(r"^(Ran |OK|FAILED)", l)]
        rows.append((name, p.returncode, " ".join(summary), fails))
    finally:
        open(path, "w").write(orig)
    print("%-60s rc=%d %s" % (name, rows[-1][1], rows[-1][2]))
    for t, q in rows[-1][3][:4]:
        print("      %s (%s)" % (t, q))
    sys.stdout.flush()
