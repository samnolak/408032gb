#!/usr/bin/env python3
"""Safety rules of the recommended R1 configuration against silently wrong results: rules 1-11 of section (v) of
docs/reassessment/PHASE-CLOSE-2026-10-11.md, accepted by the owner. Configuration guards only: no upstream patch is
applied to the R1 builds (patches/llama.cpp/* stay outside them).

  python3 scripts/r1/safety_rules.py scripts/r1/configs/<name>.json ...    # exit 1 on any violation, 2 on a bad file

check(cfg, repo_root) -> list of Violation(rule, message); empty = the config may run. cfg is a scripts/r1 config, either
as written (its "vars" are expanded here, ${REPO} = repo_root; built-ins of run_r1.py such as ${MODEL} stay as written)
or as resolved by scripts/r1/run_r1.py, which calls check() on the fully substituted config before it plans anything
and refuses a violating config (exit 2, no run directory, no step runs).

Rules encoded here (ids in the messages; labels are those of the rule's basis in the PHASE-CLOSE file):
  R-NP            rule 1. Every llama-server argv (llama.cpp or ik_llama.cpp: an argv element whose basename is
                  llama-server; the server argv is what follows it) passes -np/--parallel explicitly: the default of
                  llama.cpp's server is -np auto = 4 slots (scripts/r1/rs_rollback_rule.py). With any speculation
                  (--spec-type other than none, a draft model -md/--model-draft/-hfd/... even without --spec-type, the
                  draft presets, --mtp/--dflash/-mtp) the slot count is exactly 1. The recurrent-rollback part reuses
                  rs_rollback_rule.check(); the broader "any speculation" part is the accepted wording of rule 1.
  R-GLM-TEMPLATE  rule 2. A GLM config (name or model.desc contains "glm", any case) whose engine argv has --jinja passes
                  --chat-template-file ${REPO}/templates/glm-5.3-flash/chat_template.jinja after --jinja, no
                  --chat-template, and records model.chat_template_sha256, which must equal the sha256 of that file on
                  disk. Any config with --chat-template-file must record model.chat_template_sha256 of the file it
                  names (the sha256 then goes to the run artifacts).
  R-MTP-FILES     rule 4. A Qwen config (name or desc contains "qwen"): every llama-server draft model file has the
                  basename mtp-Qwen3.8-Flash-Next-{Q8_0,BF16,Q4_K_M}.gguf (no "-shared-" file), goes with an explicit
                  --spec-type draft-mtp, and no Hugging Face draft download (-hfd/-hfrd/--spec-draft-hf/--hf-repo-draft,
                  LLAMA_ARG_SPEC_DRAFT_HF_REPO), no --mtp/--dflash download flag and no draft preset is used.
  R-MIMO-MTP      rule 5. A MiMo config (name or desc contains "mimo") with any speculation is refused: the MTP fix of
                  N21 (patches/llama.cpp/0001-mtp-heads-slot-replay.patch) is not in the R1 builds. Speculation here
                  also counts mode.spec_k > 0 and an engine "--spec N" (N > 0) / "--mtp" argument anywhere in
                  engine.argv or engine.config_files.*.args (Strata).
  R-WEIGHTS       rule 8. A non-mock config has model.weights_manifest (weights_verify then must pass before the engine
                  starts: scripts/r1/run_r1.py) or, with no manifest, a non-empty model.weights_unverified_reason that
                  marks every run of it as unverified; not both; weights_verify is not disabled when a manifest is set.
                  tools/gguf_quant_audit.py (quant_audit) is unchanged.
  R-SYNTH         rule 10 (the pack side is tools/verify/pack.py). A config whose model.desc lacks the SYNTHETIC marker
                  must not point the pack at a criteria file with "synthetic_run": true; a "mock": true config must
                  carry the marker and must not start a llama-server (so the mock exemptions of R-WEIGHTS cannot be
                  borrowed by a real engine). tests/test_r1_safety_rules.py asserts that mock.json is the only
                  committed config that declares a synthetic model.
Rules 3 (gateway drops "reasoning_content": null: tools/gateway/gateway.py), 6, 7, 9 and 11 are operational: see
scripts/r1/README.md and the _notes of the configs.
Python 3 stdlib only.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load("rs_rollback_rule", os.path.join(HERE, "rs_rollback_rule.py"))

Violation = collections.namedtuple("Violation", "rule message")
RULES = {
    "R-NP": "rule 1: explicit -np in every llama-server argv; -np 1 with any speculation",
    "R-GLM-TEMPLATE": "rule 2: GLM with --jinja uses templates/glm-5.3-flash/chat_template.jinja, sha256 recorded",
    "R-MTP-FILES": "rule 4: Qwen MTP only from the plain mtp-...-{Q8_0,BF16,Q4_K_M}.gguf files with -md",
    "R-MIMO-MTP": "rule 5: no speculation for MiMo (N21 patch not in the R1 builds)",
    "R-WEIGHTS": "rule 8: a weights manifest, or an explicit unverified reason",
    "R-SYNTH": "rule 10: only a SYNTHETIC-marked model may be declared synthetic; mock only for the mock",
}
GLM_TEMPLATE = os.path.join("templates", "glm-5.3-flash", "chat_template.jinja")
SERVER_NAMES = ("llama-server", "llama-server.exe")
QWEN_MTP_FILE = re.compile(r"^mtp-Qwen3\.8-Flash-Next-(Q8_0|BF16|Q4_K_M)\.gguf$")     # rule 4 (N18)
DRAFT_FILE_FLAGS = ["-md", "--spec-draft-model", "--model-draft"]                    # rs_rollback_rule.DRAFT_FLAGS
DRAFT_HF_FLAGS = ["-hfd", "-hfrd", "--spec-draft-hf", "--hf-repo-draft"]
DOWNLOAD_SPEC_FLAGS = ["--mtp", "--dflash"]       # common/arg.cpp:3096-3109@79e2e74 (LLAMA_EXAMPLE_DOWNLOAD)
IK_MTP_FLAGS = ["-mtp", "--multi-token-prediction"]   # ik common/common.cpp:2353-2356@5194a9e (legacy, refused there)
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRUE_WORDS = ("true", "yes", "1", "on")


def truthy(v):
    """run_r1.py's flag(): a bool, or one of true/yes/1/on (any case)."""
    return v is True or (isinstance(v, str) and v.lower() in TRUE_WORDS)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------- config helpers
def resolve(cfg, repo_root):
    """A copy of cfg with ${NAME} expanded from cfg["vars"] and REPO (unknown names stay as written; keys starting
    with "_" are free text). A config already resolved by run_r1.py has no "vars" and no ${...} left."""
    table = {k: str(v) for k, v in (cfg.get("vars") or {}).items() if isinstance(v, (str, int, float))}
    table["REPO"] = repo_root

    def sub(s, depth=0):
        if depth > 20:
            return s
        out = VAR_RE.sub(lambda m: table.get(m.group(1), m.group(0)), s)
        return sub(out, depth + 1) if out != s and "${" in out else out

    def walk(o):
        if isinstance(o, dict):
            return {k: (v if k.startswith("_") else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return sub(o) if isinstance(o, str) else o

    return walk({k: v for k, v in cfg.items() if k != "vars"})


def model_text(cfg):
    m = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return ("%s %s" % (cfg.get("name") or "", m.get("desc") or "")).lower()


def is_mock(cfg):
    return truthy(cfg.get("mock", False))


def engine(cfg):
    e = cfg.get("engine")
    return e if isinstance(e, dict) else {}


def str_argv(v):
    return [str(a) for a in v] if isinstance(v, list) else []


def server_argvs(cfg):
    """The llama-server argument lists of engine.argv: what follows an element whose basename is llama-server."""
    argv = str_argv(engine(cfg).get("argv"))
    for i, a in enumerate(argv):
        if os.path.basename(a) in SERVER_NAMES:
            return [argv[i + 1:]]
    return []


def engine_env(cfg):
    env = engine(cfg).get("env")
    return {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {}


def server_spec_types(argv, env):
    """--spec-type values (ik's TYPE:k=v,... stage syntax reduced to TYPE), 'none' dropped."""
    return [t.split(":", 1)[0] for t in rs.spec_types(argv, env) if t.split(":", 1)[0] not in ("", "none")]


def server_speculation(argv, env):
    why = ["--spec-type %s" % t for t in server_spec_types(argv, env)]
    if rs.draft_model(argv, env):
        why.append("a draft model (%s)" % ", ".join(
            [a for a in argv if a.split("=", 1)[0] in rs.DRAFT_FLAGS + rs.DRAFT_PRESETS]
            + [k for k in rs.DRAFT_ENV if env.get(k)]))
    why += [f for f in DOWNLOAD_SPEC_FLAGS + IK_MTP_FLAGS if f in argv]
    return why


def _spec_args(args):
    out = []
    for i, a in enumerate(args):
        name, eq, val = a.partition("=")
        if name == "--spec":
            v = val if eq else (args[i + 1] if i + 1 < len(args) else "")
            try:
                if float(v) > 0:
                    out.append("--spec %s" % v)
            except ValueError:
                out.append("--spec %s (not a number)" % v)
        elif name == "--mtp":
            out.append("--mtp")
    return out


def speculation(cfg):
    """Every sign of speculative decoding in the config (empty: none)."""
    e, env = engine(cfg), engine_env(cfg)
    why = []
    for argv in server_argvs(cfg):
        why += server_speculation(argv, env)
    argv = str_argv(e.get("argv"))
    srv = server_argvs(cfg)
    if not srv:
        why += _spec_args(argv)
    files = e.get("config_files") if isinstance(e.get("config_files"), dict) else {}
    for name, body in sorted(files.items()):
        if isinstance(body, dict):
            why += ["%s in engine.config_files[%s].args" % (w, name) for w in _spec_args(str_argv(body.get("args")))]
    mode = cfg.get("mode") if isinstance(cfg.get("mode"), dict) else {}
    try:
        if float(mode.get("spec_k", 0) or 0) > 0:
            why.append("mode.spec_k = %s" % mode.get("spec_k"))
    except (TypeError, ValueError):
        why.append("mode.spec_k = %r (not a number)" % (mode.get("spec_k"),))
    return list(dict.fromkeys(why))


# ---------------------------------------------------------------------------- rules
def rule_np(cfg, repo_root):
    out, env = [], engine_env(cfg)
    for argv in server_argvs(cfg):
        given = rs._values(argv, ["-np", "--parallel"])
        if not given:
            out.append("llama-server argv has no explicit -np/--parallel: llama.cpp's default is -np auto = 4 slots "
                       "(common/arg.cpp:1406@79e2e74); give -np 1")
            continue
        try:
            slots = rs.n_slots(argv, env)
        except ValueError:
            out.append("-np/--parallel %r is not an integer" % given[-1])
            continue
        probs = rs.check(argv, env)
        out += ["%s (scripts/r1/rs_rollback_rule.py)" % p for p in probs]
        spec = server_speculation(argv, env)
        if spec and not probs and slots != 1:
            out.append("speculation (%s) needs exactly one slot: -np 1, not %d (rule 1 as accepted: any speculation)"
                       % ("; ".join(spec), slots))
    return out


def rule_glm_template(cfg, repo_root):
    out = []
    argv = str_argv(engine(cfg).get("argv"))
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    want = os.path.normpath(os.path.join(repo_root, GLM_TEMPLATE))
    files = rs._values(argv, ["--chat-template-file"])
    glm = "glm" in model_text(cfg)
    server = bool(server_argvs(cfg))
    env = engine_env(cfg)
    # llama-server @79e2e74 turns jinja on by default (common/common.h:646 use_jinja = true; common/arg.cpp:1401-1403
    # turn it off only for the completion and mtmd examples), so a GLM server without --jinja still renders the
    # built-in GGUF template (N26 audit F1): the template file and an explicit --jinja are required for every GLM server
    if glm and server:
        if "--jinja" not in argv:
            out.append("GLM llama-server/ik argv needs an explicit --jinja followed by --chat-template-file "
                       "${REPO}/%s (jinja is on by default at the R1 pin, so leaving --jinja out still uses the "
                       "built-in GGUF template)" % GLM_TEMPLATE)
        if "--no-jinja" in argv:
            out.append("GLM argv passes --no-jinja: the GLM chat format needs our jinja template file")
        for k in ("LLAMA_ARG_JINJA", "LLAMA_ARG_NO_JINJA", "LLAMA_ARG_CHAT_TEMPLATE", "LLAMA_ARG_CHAT_TEMPLATE_FILE"):
            if k in env:
                out.append("GLM engine.env sets %s, which can override the template flags of argv" % k)
    if glm and (server or "--jinja" in argv):
        if not files:
            out.append("GLM needs --chat-template-file ${REPO}/%s (the built-in GGUF template leaks the "
                       "previous turn's reasoning: risk matrix item 1, N15/N23)" % GLM_TEMPLATE)
        for f in files:
            if os.path.normpath(f) != want:
                out.append("GLM --chat-template-file is %s, not ${REPO}/%s = %s" % (f, GLM_TEMPLATE, want))
        idx = [i for i, a in enumerate(argv) if a == "--chat-template-file" or a.startswith("--chat-template-file=")]
        if idx and "--jinja" in argv and idx[0] < argv.index("--jinja"):
            out.append("--chat-template-file must come after --jinja (llama.cpp --help: only common templates are "
                       "accepted unless --jinja is set before it)")
        if rs._values(argv, ["--chat-template"]):
            out.append("GLM argv also passes --chat-template, which replaces the template file")
    if files:
        sha = model.get("chat_template_sha256")
        if not (isinstance(sha, str) and SHA256_RE.match(sha)):
            out.append("--chat-template-file is given but model.chat_template_sha256 (64 lower-case hex) is not: %r"
                       % (sha,))
        else:
            for f in files:
                path = f if os.path.isabs(f) else os.path.join(repo_root, f)
                try:
                    got = sha256_file(path)
                except OSError as e:
                    out.append("cannot read the chat template %s: %s" % (f, e))
                    continue
                if got != sha:
                    out.append("sha256 of %s is %s, model.chat_template_sha256 says %s" % (f, got, sha))
    return out


def rule_mtp_files(cfg, repo_root):
    if "qwen" not in model_text(cfg):
        return []
    out, env = [], engine_env(cfg)
    for argv in server_argvs(cfg):
        files = rs._values(argv, DRAFT_FILE_FLAGS)
        if env.get("LLAMA_ARG_SPEC_DRAFT_MODEL") and not files:
            files = [env["LLAMA_ARG_SPEC_DRAFT_MODEL"]]
        for f in files:
            base = os.path.basename(f)
            if "-shared-" in base:
                out.append("Qwen MTP file %s is a -shared- file: it does not load on either pin (token_embd created "
                           "with flags 0, N18)" % base)
            elif not QWEN_MTP_FILE.match(base):
                out.append("Qwen draft model %s is not one of mtp-Qwen3.8-Flash-Next-{Q8_0,BF16,Q4_K_M}.gguf" % base)
        if files and "draft-mtp" not in server_spec_types(argv, env):
            out.append("a Qwen MTP file goes with an explicit --spec-type draft-mtp (got %s)"
                       % (server_spec_types(argv, env) or "no --spec-type"))
        hf = rs._values(argv, DRAFT_HF_FLAGS) + ([env["LLAMA_ARG_SPEC_DRAFT_HF_REPO"]]
                                                 if env.get("LLAMA_ARG_SPEC_DRAFT_HF_REPO") else [])
        if hf:
            out.append("Qwen draft from Hugging Face (%s): use -md with a verified local mtp-...gguf file" % ", ".join(hf))
        for fl in DOWNLOAD_SPEC_FLAGS:
            if fl in argv:
                out.append("%s: -hf ... %s does not find the MTP/ folder (N18); use -md FILE --spec-type draft-mtp"
                           % (fl, fl))
        for p in rs.DRAFT_PRESETS:
            if p in argv:
                out.append("draft preset %s is not a Qwen MTP file" % p)
    return out


def rule_mimo_mtp(cfg, repo_root):
    if "mimo" not in model_text(cfg):
        return []
    spec = speculation(cfg)
    if spec:
        return ["MiMo with speculation (%s): the MTP fix of N21 is not in the R1 builds; run without speculation"
                % "; ".join(spec)]
    return []


def rule_weights(cfg, repo_root):
    if is_mock(cfg):
        return []
    m = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    man, why = m.get("weights_manifest"), m.get("weights_unverified_reason")
    out = []
    if why is not None and not (isinstance(why, str) and why.strip()):
        out.append("model.weights_unverified_reason must be a non-empty string, got %r" % (why,))
        why = None
    if not man and not why:
        out.append("model.weights_manifest is null and there is no model.weights_unverified_reason: give the sha256 "
                   "manifest (scripts/offline/weights.py) or mark every run of this config unverified with a reason")
    if man and why:
        out.append("both model.weights_manifest and model.weights_unverified_reason are set: a run with a manifest "
                   "must verify the weights; drop the reason")
    steps = cfg.get("steps") if isinstance(cfg.get("steps"), dict) else {}
    wv = steps.get("weights_verify") if isinstance(steps.get("weights_verify"), dict) else {}
    if man and not truthy(wv.get("enabled", True)):
        out.append("steps.weights_verify is disabled while model.weights_manifest is set: the engine would never "
                   "start (run_r1.py starts it only after weights_verify passed)")
    return out


_pack = None


def synthetic_marked(desc):
    global _pack
    if _pack is None:
        _pack = _load("_safety_pack", os.path.join(REPO, "tools", "verify", "pack.py"))
    return _pack.synthetic_marked(desc)


def rule_synth(cfg, repo_root):
    out = []
    m = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    marked = synthetic_marked(m.get("desc"))
    if is_mock(cfg):
        if not marked:
            out.append("a \"mock\": true config must say SYNTHETIC in model.desc")
        if server_argvs(cfg):
            out.append("a \"mock\": true config must not start a llama-server (the mock exemption is for "
                       "tools/bench/mock_server.py)")
    steps = cfg.get("steps") if isinstance(cfg.get("steps"), dict) else {}
    pk = steps.get("pack") if isinstance(steps.get("pack"), dict) else {}
    crit = pk.get("criteria")
    if crit and not marked:
        path = crit if os.path.isabs(crit) else os.path.join(repo_root, crit)
        try:
            with open(path, encoding="utf-8") as f:
                synthetic = json.load(f).get("synthetic_run") is True
        except (OSError, ValueError, AttributeError) as e:
            out.append("cannot read steps.pack.criteria %s: %s" % (crit, e))
            synthetic = False
        if synthetic:
            out.append("steps.pack.criteria %s declares \"synthetic_run\": true but model.desc has no SYNTHETIC marker"
                       % crit)
    return out


CHECKS = [("R-NP", rule_np), ("R-GLM-TEMPLATE", rule_glm_template), ("R-MTP-FILES", rule_mtp_files),
          ("R-MIMO-MTP", rule_mimo_mtp), ("R-WEIGHTS", rule_weights), ("R-SYNTH", rule_synth)]


def check(cfg, repo_root=REPO):
    """Violations of the R1 safety rules (empty list: the config may run)."""
    if not isinstance(cfg, dict):
        return [Violation("R-CONFIG", "the config is not a JSON object")]
    c = resolve(cfg, repo_root)
    out = []
    for rule, fn in CHECKS:
        out += [Violation(rule, msg) for msg in fn(c, repo_root)]
    return out


def report(cfg, repo_root=REPO):
    """check() plus the facts a run records (artifacts/orchestrator/safety_rules.json)."""
    c = resolve(cfg, repo_root) if isinstance(cfg, dict) else {}
    m = c.get("model") if isinstance(c.get("model"), dict) else {}
    templates = []
    for f in rs._values(str_argv(engine(c).get("argv")), ["--chat-template-file"]):
        path = f if os.path.isabs(f) else os.path.join(repo_root, f)
        templates.append({"path": f, "sha256": sha256_file(path) if os.path.isfile(path) else None,
                          "expected_sha256": m.get("chat_template_sha256")})
    return {"_label": "R1 safety rules (scripts/r1/safety_rules.py; docs/reassessment/PHASE-CLOSE-2026-10-11.md "
                      "section (v), rules 1-11) checked before any step",
            "rules": RULES, "violations": [v._asdict() for v in check(cfg, repo_root)],
            "speculation": speculation(c), "slots": [rs._values(a, ["-np", "--parallel"]) for a in server_argvs(c)],
            "chat_templates": templates,
            "weights": {"manifest": m.get("weights_manifest"), "unverified_reason": m.get("weights_unverified_reason")}}


def main(paths):
    if not paths:
        print("usage: safety_rules.py CONFIG.json ...", file=sys.stderr)
        return 2
    bad = err = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            print("%s: cannot read: %s" % (p, e))
            err += 1
            continue
        vs = check(cfg)
        if vs:
            bad += 1
            for v in vs:
                print("%s: VIOLATES %s: %s" % (p, v.rule, v.message))
        else:
            print("%s: ok (%s)" % (p, ", ".join(rule for rule, _ in CHECKS)))
    return 2 if err else (1 if bad else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
