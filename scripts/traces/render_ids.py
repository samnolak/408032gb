#!/usr/bin/env python3
"""Render conversations (scripts/traces/build_agent_prompts.py) with the model's own chat template and tokenize them
through a running llama-server, writing one <name>.ids file per conversation (token ids, comma separated) - the exact
input tools/g0/expert_trace.cpp and expert_probe.cpp replay.

Uses only documented llama-server endpoints (tools/server/README.md @ec7630a): POST /apply-template {"messages"} ->
{"prompt"}; POST /tokenize {"content", "add_special": true, "parse_special": true} -> {"tokens"}. add_special=true
tokenizes the rendered prompt the way the server does for a chat request (BOS only if the model's vocab asks for it).
Usage: python3 scripts/traces/render_ids.py --url http://127.0.0.1:8080 --prompts prompts/ --out ids/ [--max-tokens 6000]
"""
import argparse
import glob
import hashlib
import json
import os
import urllib.request


def post(url, path, body):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=6000, help="error if a rendered conversation is longer")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for path in sorted(glob.glob(os.path.join(a.prompts, "*.json"))):
        if os.path.basename(path) == "INDEX.json":
            continue
        with open(path, encoding="utf-8") as f:
            conv = json.load(f)
        prompt = post(a.url, "/apply-template", {"messages": conv["messages"]})["prompt"]
        tokens = post(a.url, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
        if len(tokens) > a.max_tokens:
            raise SystemExit(f"{conv['name']}: {len(tokens)} tokens > --max-tokens {a.max_tokens}")
        ids = ",".join(str(t) for t in tokens)
        with open(os.path.join(a.out, conv["name"] + ".ids"), "w") as f:
            f.write(ids)
        manifest.append({"name": conv["name"], "n_tokens": len(tokens),
                         "rendered_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "ids_sha256": hashlib.sha256(ids.encode()).hexdigest(), "rendered_head": prompt[:160]})
        print(f"{conv['name']}: {len(tokens)} tokens")
    with open(os.path.join(a.out, "RENDER.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
