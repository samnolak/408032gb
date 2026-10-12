#!/usr/bin/env python3
"""N10 probe (gap 3, N7 finding 7; written for the evidence audit of bd715da): trace N7's own glm5-next measurement at
79e2e74 with dump-tensors instead of llama-server.

N7 (docs/evidence/2026-10-10-n7-server-hybrid-cpu.md, finding 7 and its attribution table, on branch
claude/n7-server-hybrid-cpu) reports for the tiny glm5-next with the server defaults (f16 KV cache, flash attention)
"cold 256-token prompt vs the first run: first differing token 3, |d| at step 0 0.416, max |d| up to it 0.86"
(tools/srvcheck/n7_report.py attr: the cold answer of check prefix_extend of each configuration against the gate's,
max |difference of log-prob| over the top-5 ids both list, step by step up to the first differing token). The
256-token prompt of check prefix_extend is prompt + the configuration's OWN first answer + tail
(tools/srvcheck/hybrid_server_checks.py@090ca71:168-171 check_prefix_extend: nxt = prompt + first["tokens"] + tail;
:333-335 prompt = 200 and tail = 24 ids of random.Random(seed 7); :156-157 ids(rng, n) = rng.randrange(3, 96)).

This script:
1. rebuilds prompt and tail, reads both configurations' answers from N7's result files and prints where the two first
   answers and hence the two 256-token prompts differ;
2. runs dump-tensors (llama.cpp at 79e2e74, N7's glm5next.gguf) with the server's flags (-c 1024 -b 2048 -ub 512 -t 2;
   --no-warmup is not a common argument, dump-tensors sets params.warmup = false itself; "defaults" = no -ctk / -ctv /
   -fa flag, as the server ran; "f32" = -ctk f32 -ctv f32 -fa off, the gate) and the server's call plan
   (llama-server @79e2e74 splits a prompt of a model with recurrent memory into n - 4 and 4 tokens to place a context
   checkpoint, tools/server/server-context.cpp:3865-3878, 3986-4004; then one token per call), teacher-forced with the
   answers:
     F  = prompt (200) + the gate's first answer, both precisions;
     C  = the gate's 256-token prompt + the gate's cold answer, both precisions;
     D  = the defaults' 256-token prompt + the defaults' cold answer, defaults only;
3. checks that the dumps reproduce the server: |server top-5 log-prob - dump log-softmax| on every step whose history
   the run shares with the server answer;
4. compares the two precisions on the same tokens (F, C): |difference of log-softmax| per step, greedy tokens, and every
   MoE expert set (ffn_moe_topk-<il>) and DSA selection (kq_mask_dsa-<il>) that differs, by position and layer (for a
   MoE set: llama.cpp's own selection scores and margin in both precisions, ffn_moe_probs_biased-<il>); and
   recomputes N7's step-0 number from the dumps (C at f32 against D at the defaults: different prompts) next to the
   same-prompt one (C against C).
Usage: 2026-10-10-n10-n7-trace.py <llama.cpp @79e2e74> <tools/g0 build with dump-tensors> <N7 artifacts dir
       (run_n7.sh: glm5next.gguf, glm5next-gate.json, glm5next-defaults.json)> <work dir>
"""
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys

import numpy as np

PIN = "79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6"
PREFIXES = "ffn_moe_topk,ffn_moe_probs,kq_mask_dsa,result_"
FLAGS = {"f32": ["-ctk", "f32", "-ctv", "f32", "-fa", "off"], "defaults": []}
SERVER = ["-c", "1024", "-b", "2048", "-ub", "512", "-t", "2"]


def ids(rng, n, lo=3, hi=96):
    return [rng.randrange(lo, hi) for _ in range(n)]


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def run(build, model, work, name, toks, n_prompt, prec):
    d = pathlib.Path(work, f"{name}-{prec}")
    calls = [str(n_prompt - 4), "4"] + ["1"] * (len(toks) - n_prompt)
    env = dict(os.environ, DUMP_DIR=str(d), DUMP_TOKENS=",".join(map(str, toks)), DUMP_PER_GRAPH="1",
               DUMP_PREFIXES=PREFIXES, DUMP_CALLS=",".join(calls), DUMP_N_RS_SEQ="0")
    log = pathlib.Path(work, f"{name}-{prec}.log")
    with open(log, "w") as f:
        cmd = [str(pathlib.Path(build, "dump-tensors")), "-m", str(model)] + SERVER + FLAGS[prec] + ["-lv", "4"]
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text(errors="replace")
    fa = [ln.split("llama_context:", 1)[-1].strip() for ln in text.splitlines()
          if "llama_context: flash_attn" in ln][:1]
    fa += [ln.split(" I ", 1)[-1].strip() for ln in text.splitlines() if "Flash Attention" in ln][:1]
    kv = [ln.split("llama_kv_cache:", 1)[-1].strip() for ln in text.splitlines() if "llama_kv_cache: size" in ln][:1]
    man = json.loads((d / "manifest.json").read_text())
    print(f"run {name}-{prec}: rc={rc}, {len(toks)} tokens, calls {n_prompt - 4} + 4 + {len(toks) - n_prompt} x 1, "
          f"{len(man['graphs'])} graphs; log: {' -> '.join(fa) or '?'} | {kv[0] if kv else '?'}")
    return d


def dump(d, g, name):
    p = d / f"g{g:03d}"
    m = json.loads((p / f"{name}__0.json").read_text())
    dt = np.float32 if m["type"] == "f32" else np.int32
    return np.fromfile(p / f"{name}__0.bin", dtype=dt).reshape(m["ne"][::-1])


def has(d, g, name):
    return (d / f"g{g:03d}" / f"{name}__0.json").exists()


def log_softmax(d):
    n_vocab = json.loads((d / "logits.json").read_text())["n_vocab"]
    x = np.fromfile(d / "logits.bin", dtype=np.float32).reshape(-1, n_vocab).astype(np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def choices(d):
    """{(kind, layer, position): chosen experts / kept cells} over every graph of a run, and {(layer, position): the
    MoE selection scores} (ffn_moe_probs_biased: sigmoid(logit) + exp_probs_b, what ggml_top_k ranks)."""
    man = json.loads((d / "manifest.json").read_text())
    out, scores = {}, {}
    for g, gi in enumerate(man["graphs"]):
        call = man["calls"][gi["call"]]
        j = g - call["graphs"][0]
        n = min(man["n_ubatch"], call["n"] - j * man["n_ubatch"])
        pos0 = call["pos0"] + j * man["n_ubatch"]
        for il in range(64):
            if has(d, g, f"ffn_moe_topk-{il}"):
                e = np.sort(dump(d, g, f"ffn_moe_topk-{il}").reshape(n, -1), 1)
                sc = dump(d, g, f"ffn_moe_probs_biased-{il}").reshape(n, -1)
                for r in range(n):
                    out[("MoE", il, pos0 + r)] = tuple(e[r].tolist())
                    scores[(il, pos0 + r)] = sc[r].astype(np.float64)
            if has(d, g, f"kq_mask_dsa-{il}"):
                m = dump(d, g, f"kq_mask_dsa-{il}").reshape(n, -1)
                for r in range(n):
                    out[("DSA", il, pos0 + r)] = tuple(np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0].tolist())
    return out, scores


def top5_metric(la, lb):
    """N7's |d| of one step: max |difference| over the top-5 ids both distributions list."""
    ta, tb = set(np.argsort(-la)[:5].tolist()), set(np.argsort(-lb)[:5].tolist())
    return max([abs(la[i] - lb[i]) for i in ta & tb] or [0.0])


def main(llama, build, n7, work):
    head = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"llama.cpp {head}")
    if head != PIN:
        sys.exit(f"needs {PIN}")
    model = pathlib.Path(n7, "glm5next.gguf")
    print(f"model {model.name} sha256 {hashlib.sha256(model.read_bytes()).hexdigest()[:16]}")
    pathlib.Path(work).mkdir(parents=True, exist_ok=True)
    gate = json.loads(pathlib.Path(n7, "glm5next-gate.json").read_text())["prefix_extend"]
    dflt = json.loads(pathlib.Path(n7, "glm5next-defaults.json").read_text())["prefix_extend"]
    rng = random.Random(7)
    prompt, tail = ids(rng, 200), ids(rng, 24)
    gp = prompt + gate["first"]["tokens"] + tail
    dp = prompt + dflt["first"]["tokens"] + tail
    fd = first_diff(gate["first"]["tokens"], dflt["first"]["tokens"])
    print(f"N7 first answers (200-token prompt): gate {gate['first']['tokens']}")
    print(f"                                  defaults {dflt['first']['tokens']}")
    print(f"  first differing answer token: {fd}; n_prompt of the cold requests: gate {gate['cold']['n_prompt']}, "
          f"defaults {dflt['cold']['n_prompt']}; the two 256-token prompts differ at "
          f"{sum(a != b for a, b in zip(gp, dp))} positions, the first at {first_diff(gp, dp)}")
    F = {p: run(build, model, work, "F", prompt + gate["first"]["tokens"][:31], 200, p) for p in ("f32", "defaults")}
    C = {p: run(build, model, work, "C", gp + gate["cold"]["tokens"][:31], 256, p) for p in ("f32", "defaults")}
    D = run(build, model, work, "D", dp + dflt["cold"]["tokens"][:31], 256, "defaults")
    ls = {("F", p): log_softmax(F[p]) for p in F}
    ls.update({("C", p): log_softmax(C[p]) for p in C})
    ls[("D", "defaults")] = log_softmax(D)
    # 3. the dumps against the server's own top-5 log-probs, on the steps whose history the run shares
    print("dump log-softmax against N7's server answers (steps with the same history):")
    for key, srv, row0, ok in ((("F", "f32"), gate["first"], 199, 32),
                               (("F", "defaults"), dflt["first"], 199, fd + 1),
                               (("C", "f32"), gate["cold"], 255, 32),
                               (("D", "defaults"), dflt["cold"], 255, 32)):
        err = max(abs(v - ls[key][row0 + s][int(t)]) for s in range(ok) for t, v in srv["top"][s].items())
        same = all(int(ls[key][row0 + s].argmax()) == srv["tokens"][s] for s in range(ok))
        print(f"  {key[0]}-{key[1]:8s} vs server {'gate' if key[1] == 'f32' else 'defaults'} "
              f"{'first' if key[0] == 'F' else 'cold'} answer, steps 0..{ok - 1}: max |d log-prob| {err:.3e}, "
              f"greedy tokens equal: {same}")
    # 4. the two precisions on the same tokens
    for name, d, row0 in (("F", F, 199), ("C", C, 255)):
        a, b = ls[(name, "defaults")], ls[(name, "f32")]
        dd = np.abs(a - b).max(1)
        steps = dd[row0:row0 + 32]
        greedy = [s for s in range(32) if a[row0 + s].argmax() != b[row0 + s].argmax()]
        t5 = [top5_metric(a[row0 + s], b[row0 + s]) for s in range(32)]
        print(f"{name}: defaults vs f32 on the same tokens: max |d log-softmax| over the prompt rows "
              f"{dd[:row0].max():.3e}; per answer step: step 0 {steps[0]:.3e} (top-5 metric {t5[0]:.3e}), max "
              f"{steps.max():.3e} at step {int(steps.argmax())} (top-5 metric max {max(t5):.3e}); greedy tokens "
              f"differing at steps {greedy}")
        if greedy:
            s0 = greedy[0]
            print(f"   |d log-softmax| (top-5 metric) at answer steps {max(0, s0 - 2)}..{min(31, s0 + 1)}: "
                  + ", ".join(f"{steps[s]:.3e} ({t5[s]:.3e})" for s in range(max(0, s0 - 2), min(32, s0 + 2))))
        (ca, sa), (cb, sb) = choices(d["defaults"]), choices(d["f32"])
        diff = sorted((k for k in ca if ca[k] != cb.get(k)), key=lambda k: (k[2], k[1], k[0]))
        print(f"   discrete choices differing: {len(diff)} of {len(ca)} "
              f"(MoE {sum(k[0] == 'MoE' for k in diff)}, DSA {sum(k[0] == 'DSA' for k in diff)})")
        for k in diff[:8]:
            extra = f": {list(ca[k])} vs {list(cb[k])}" if k[0] == "MoE" else ""
            print(f"     {k[0]} L{k[1]} pos {k[2]} (step {k[2] - row0 if k[2] >= row0 else 'prompt'}){extra}")
            if k[0] == "MoE":
                n_used = len(ca[k])
                for tag, sc in (("defaults", sa[(k[1], k[2])]), ("f32", sb[(k[1], k[2])])):
                    o = np.argsort(-sc, kind="stable")[:n_used + 1]
                    print(f"         {tag:8s} best {n_used + 1}: " + ", ".join(f"{int(e)} {sc[e]:.9f}" for e in o)
                          + f"; margin k/k+1 {sc[o[n_used - 1]] - sc[o[n_used]]:.3e}")
        if len(diff) > 8:
            print(f"     ... {len(diff) - 8} more")
        if diff:
            p = diff[0][2]
            print(f"   |d log-softmax| at the rows before / at / after the first differing position {p}: "
                  + ", ".join(f"{dd[r]:.3e}" for r in range(max(0, p - 2), min(len(dd), p + 3))))
    print("N7's step-0 number from the dumps (top-5 metric): C-f32 vs D-defaults (each configuration's own prompt) "
          f"{top5_metric(ls[('D', 'defaults')][255], ls[('C', 'f32')][255]):.3e}; C-defaults vs C-f32 (the same "
          f"prompt) {top5_metric(ls[('C', 'defaults')][255], ls[('C', 'f32')][255]):.3e}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
