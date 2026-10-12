"""Prefix stability of a chat template over a growing conversation (B3, gap 6 of docs/reassessment/CORRECTNESS.md).

Two views, both on rendered text (bytes of the UTF-8 string; token-level effects need the real tokenizer):

history_checks   render(conv[:k]) without the generation prompt against render(conv[:k+1]): is the shorter a byte prefix
                 of the longer? If not, where is the first differing byte and in which earlier message does it fall.

loop_checks      what llama-server's slot holds against what the next request sends, in an agent loop where the model
                 itself produced every assistant turn. A request is sent whenever the conversation ends with a user or
                 tool message and the next message is the assistant's. The slot then holds
                     P_k + O_k                (prompt of request k + the model's output, which ends with an end-of-turn
                                               token that is sampled but not decoded)
                 where O_k is the assistant turn exactly as the template renders it in history ("ideal output": the
                 model writes the format the template shows it). The next request's prompt P_j must start with
                 P_k + O_k for the server to reuse everything; otherwise reuse stops at the first differing byte and,
                 for a hybrid/recurrent model, falls back to the last context checkpoint at or before it
                 (tools/server/server-context.cpp:3726-3771 @79e2e74).
                 O_k is defined only if P_k is a prefix of render(conv[:k+1]) (the generation prompt is the start of
                 the rendered turn); if it is not, that is reported as its own break.
"""


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def message_starts(render, conv, upto, **kw):
    """Byte offset where each of conv[:upto] starts in render(conv[:upto]), from the stable prefixes (best effort):
    start[i] = len(render(conv[:i])) when that is a prefix of the full render, else None."""
    full = render(conv[:upto], add_generation_prompt=False, **kw)
    starts = []
    for i in range(upto):
        try:
            s = render(conv[:i], add_generation_prompt=False, **kw) if i else ""
        except Exception:  # noqa: BLE001 - templates may refuse prefixes (e.g. Qwen: no user query yet)
            s = None
        starts.append(len(s) if s is not None and full.startswith(s) else None)
    return full, starts


def owner(starts, pos):
    """Index of the message whose rendered span holds byte pos (None if unknown)."""
    best = None
    for i, s in enumerate(starts):
        if s is not None and s <= pos:
            best = i
    return best


def nbytes(text, pos):
    """UTF-8 byte offset of character offset pos."""
    return len(text[:pos].encode("utf-8"))


def snippet(text, pos, before=30, after=50):
    return text[max(0, pos - before):pos] + "‖" + text[pos:pos + after]


def _try(render, msgs, agp, kw):
    try:
        return render(msgs, add_generation_prompt=agp, **kw), None
    except Exception as e:  # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, str(e).strip().splitlines()[-1] if str(e).strip() else "")


def history_checks(render, conv, **kw):
    out = []
    for k in range(1, len(conv)):
        a, ea = _try(render, conv[:k], False, kw)
        b, eb = _try(render, conv[:k + 1], False, kw)
        row = {"k": k, "added_role": conv[k]["role"]}
        if ea or eb:
            row.update({"stable": None, "error": ea or eb})
        elif b.startswith(a):
            row.update({"stable": True, "len": len(a)})
        else:
            d = lcp(a, b)
            _, starts = message_starts(render, conv, k, **kw)
            row.update({"stable": False, "len": len(a), "first_diff": d, "first_diff_bytes": nbytes(a, d),
                        "in_message": owner(starts, d),
                        "old": snippet(a, d), "new": snippet(b, d)})
        out.append(row)
    return out


def request_points(conv):
    """Indices k such that the server is asked to generate conv[k] (an assistant turn after a user/tool message)."""
    return [k for k in range(1, len(conv)) if conv[k]["role"] == "assistant" and conv[k - 1]["role"] in ("user", "tool")]


def loop_checks(render, conv, render_out=None, **kw):
    """render_out: the renderer that defines what the model writes (default: render). With render = llama-server and
    render_out = the reference renderer, the slot holds the server's prompt followed by the turn as a model trained on
    the reference template writes it."""
    render_out = render_out or render
    pts = request_points(conv)
    rows = []
    for idx, k in enumerate(pts):
        p, err = _try(render, conv[:k], True, kw)
        p_out, err1 = _try(render_out, conv[:k], True, kw)
        hist, err2 = _try(render_out, conv[:k + 1], False, kw)
        row = {"request": idx, "k": k}
        if err or err1 or err2:
            row["error"] = err or err1 or err2
            rows.append(row)
            continue
        row["prompt_len"] = len(p)
        row["prompt_equals_reference"] = p == p_out
        if not hist.startswith(p_out):
            d = lcp(p_out, hist)
            row.update({"gen_prompt_is_prefix": False, "first_diff": d, "first_diff_bytes": nbytes(p_out, d),
                        "old": snippet(p_out, d), "new": snippet(hist, d)})
            rows.append(row)
            continue
        out = hist[len(p_out):]
        row["gen_prompt_is_prefix"] = True
        row["output_len"] = len(out)
        held = p + out
        if idx + 1 < len(pts):
            nxt, err3 = _try(render, conv[:pts[idx + 1]], True, kw)
            if err3:
                row["error"] = err3
                rows.append(row)
                continue
            d = lcp(held, nxt)
            row["held_len"] = len(held)
            row["next_prompt_len"] = len(nxt)
            row["reused_bytes"] = d
            row["extends"] = d == len(held)
            if d < len(held):
                _, starts = message_starts(render, conv, pts[idx + 1], **kw)
                row.update({"first_diff": d, "first_diff_bytes": nbytes(held, d), "held_bytes": len(held.encode()),
                            "in_message": owner(starts, d),
                            "in_message_role": None, "old": snippet(held, d), "new": snippet(nxt, d)})
                if row["in_message"] is not None:
                    row["in_message_role"] = conv[row["in_message"]]["role"]
        rows.append(row)
    return rows
