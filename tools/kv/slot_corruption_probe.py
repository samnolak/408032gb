#!/usr/bin/env python3
"""Does llama-server notice a corrupted slot file, and what does a corrupted state do to the answer? (N8)

Starts a real llama-server (any build; CPU is enough) on a model, greedily completes a token-id prompt, saves the slot
through tools/kv/session_store.py, then for each corruption (one bit flipped at a fraction of the file, or the file
truncated) restores the damaged file twice:
  raw    POST /slots/0?action=restore directly (what llama-server alone does), then the same prompt again: accepted or
         refused, prompt tokens re-processed, same tokens as the reference, max |delta log-prob|, null (NaN) log-probs;
  store  SessionStore.restore: must refuse (sha256 / size) before the server sees the file, then the prompt is prefilled.
Prints one line per case and writes a JSON report. Exit code 0 only if the store refused every corrupted file and
restored the intact one.
Usage: python3 tools/kv/slot_corruption_probe.py --server build/bin/llama-server --model m.gguf --out report.json
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "srvcheck"))
import hybrid_server_checks as hc  # noqa: E402
import session_store as ss  # noqa: E402


def flip(data, frac, mask, word_high_byte):
    d = bytearray(data)
    off = int(len(d) * frac)
    if word_high_byte:
        off = off - off % 4 + 3          # highest byte of an aligned 32-bit word (sign/exponent of an f32)
    d[off] ^= mask
    return bytes(d), off


def answer_vs(ref, out):
    nulls = sum(1 for st in out["top"] for v in st.values() if v is None)
    c = hc.compare(out, ref, 1e-3)
    return {"prompt_n": out["prompt_n"], "same_tokens": c["same_tokens"], "first_diff": c["first_diff"],
            "max_abs_dlogprob": c["max_abs_dlogprob"], "null_logprobs": nulls}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default="-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup")
    ap.add_argument("--fractions", default="0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--prompt-len", type=int, default=200)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--workdir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    work = a.workdir or tempfile.mkdtemp(prefix="n8-")
    slots = os.path.join(work, "slots")
    os.makedirs(slots, exist_ok=True)
    extra = a.extra.split() + ["--slot-save-path", slots]
    prompt = hc.ids(random.Random(5), a.prompt_len)
    ident = {"model": os.path.basename(a.model), "extra": a.extra}
    store = ss.SessionStore(slots, ident)
    report = {"model": a.model, "server": a.server, "server_args": extra, "cases": []}
    srv = hc.Server(a.server, a.model, extra, os.path.join(work, "save.log"))
    try:
        hc.erase(srv.url)
        ref = hc.complete(srv.url, prompt, a.gen)
        m = store.save(srv.url, 0, "probe")
    finally:
        srv.stop()
    path = os.path.join(slots, m["file"])
    with open(path, "rb") as f:
        good = f.read()
    shutil.copy(path, os.path.join(work, "intact.slot"))
    cases = [("intact", good, None)]
    for fr in [float(x) for x in a.fractions.split(",")]:
        cases.append((f"bit6@{fr}", *flip(good, fr, 0x40, False)))
        cases.append((f"exp@{fr}", *flip(good, fr, 0x40, True)))
    cases.append(("truncated-1000", good[:-1000], len(good) - 1000))
    ok = True
    srv = hc.Server(a.server, a.model, extra, os.path.join(work, "restore.log"))
    try:
        for name, data, off in cases:
            row = {"case": name, "offset": off, "bytes": len(data)}
            raw_name = "raw-" + name.replace("@", "_") + ".slot"
            with open(os.path.join(slots, raw_name), "wb") as f:
                f.write(data)
            hc.erase(srv.url)
            try:
                r = hc.post(srv.url, "/slots/0?action=restore", {"filename": raw_name})
                row["raw_restore"] = {"accepted": True, "n_restored": r.get("n_restored")}
            except Exception as e:  # recorded: the server refused
                row["raw_restore"] = {"accepted": False, "error": str(e)[:120]}
            row["raw_answer"] = answer_vs(ref, hc.complete(srv.url, prompt, a.gen))
            os.remove(os.path.join(slots, raw_name))
            # the store's copy of the session gets the same damage
            with open(path, "wb") as f:
                f.write(data)
            hc.erase(srv.url)
            res = store.restore(srv.url, 0, "probe")
            row["store_restore"] = {"accepted": res is not None, "reason": store.last_reason}
            row["store_answer"] = answer_vs(ref, hc.complete(srv.url, prompt, a.gen))
            if (name == "intact") != (res is not None) or not row["store_answer"]["same_tokens"]:
                ok = False
            report["cases"].append(row)
            print("%-16s raw: %-8s prompt_n %3s same_tokens %-5s max|d| %-9s nulls %-3s | store: %-8s %-16s prompt_n %3s same %s"
                  % (name, "accepted" if row["raw_restore"]["accepted"] else "REFUSED", row["raw_answer"]["prompt_n"],
                     row["raw_answer"]["same_tokens"], "%.3g" % row["raw_answer"]["max_abs_dlogprob"]
                     if row["raw_answer"]["max_abs_dlogprob"] is not None else "-", row["raw_answer"]["null_logprobs"],
                     "restored" if res is not None else "refused", store.last_reason or "",
                     row["store_answer"]["prompt_n"], row["store_answer"]["same_tokens"]), flush=True)
            # put the intact session back for the next case
            if store.manifest("probe") is None:
                srv2_url = srv.url
                with open(os.path.join(work, "intact.slot"), "rb") as f:
                    data_ok = f.read()
                with open(os.path.join(slots, "rebuild.slot"), "wb") as f:
                    f.write(data_ok)
                hc.erase(srv2_url)
                hc.post(srv2_url, "/slots/0?action=restore", {"filename": "rebuild.slot"})
                os.remove(os.path.join(slots, "rebuild.slot"))
                m = store.save(srv2_url, 0, "probe")
                path = os.path.join(slots, m["file"])
            else:
                with open(path, "wb") as f:
                    f.write(good)
    finally:
        srv.stop()
        store.close()
    report["store_refused_every_corruption_and_restored_intact"] = ok
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
