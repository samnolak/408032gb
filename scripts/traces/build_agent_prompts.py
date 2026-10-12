#!/usr/bin/env python3
"""Deterministic coding-agent style conversations built from this repository's own files, for routing traces of real
MoE models (tools/g0/expert_trace.cpp, expert_probe.cpp).

What this is and is not: the TEXT is real (source code, tests, Russian and English docs of this repository); the
conversations are CONSTRUCTED (system prompt of a coding agent, a task, the file content as a tool result, a follow-up),
not recorded customer sessions. Routing traces from them characterise a real model on agent-shaped, code-heavy input;
they do not replace traces of the customer's own sessions (EXPERIMENTS.md M1).

Output: <out>/<name>.json, each {"name", "lang", "messages": [...]} in OpenAI chat format (roles system/user/assistant),
plus <out>/INDEX.json with the file list, sizes and the sha256 of every source file used.
Usage: python3 scripts/traces/build_agent_prompts.py --root . --out prompts/ [--n 16] [--max-chars 7000]
"""
import argparse
import hashlib
import json
import os

SYSTEM = ("You are a careful coding agent working inside the git repository 408032gb (local LLM inference research). "
          "You can call tools: read_file(path), run(command), apply_patch(diff). Think briefly, then act. Keep answers "
          "short and precise; never invent file contents or test results.")

TASKS_EN = [
    "Review {path} for correctness bugs. List each bug with the line and a minimal fix.",
    "Write unit tests for the most important function in {path}. Use unittest and keep them deterministic.",
    "Explain what {path} does to a new team member in five bullet points, then name one risky assumption.",
    "Refactor the longest function in {path} into smaller functions without changing behaviour. Show the patch.",
    "Find any place in {path} where an error is silently ignored and propose how to surface it.",
    "Add type hints and a docstring to the public functions of {path}. Show only the changed lines.",
]
TASKS_RU = [
    "Проверь {path} на ошибки корректности. Для каждой ошибки укажи строку и минимальное исправление.",
    "Кратко объясни, что делает {path}, и предложи один тест, который ловит самую вероятную регрессию.",
    "Найди в {path} места, где число или вывод не подтверждены источником, и предложи формулировку с меткой.",
]
FOLLOW_EN = "Now apply the change you consider most important and show the exact patch."
FOLLOW_RU = "Теперь внеси самое важное изменение и покажи точный патч."

# code-heavy first (what a coding agent mostly reads), then Russian docs; all paths relative to --root
SOURCES = [
    ("tools/g0/cachesim.py", "en"), ("tools/roofline.py", "en"), ("tools/gateway/gateway.py", "en"),
    ("tools/bench/run_matrix.py", "en"), ("ref/qwen4exp.py", "en"), ("ref/dsa.py", "en"),
    ("tools/g0/expert_trace.cpp", "en"), ("tools/kv/slot_persist.py", "en"), ("tools/eval/run_eval.py", "en"),
    ("scripts/r1/run_r1.py", "en"), ("tools/verify/pack.py", "en"), ("tests/test_gateway.py", "en"),
    ("docs/reassessment/ADR-012-reassessment.md", "ru"), ("docs/reassessment/ROADMAP.md", "ru"),
    ("ref/kda.py", "ru"), ("tools/gguf_quant_audit.py", "ru"),
]


def fence(path):
    ext = os.path.splitext(path)[1]
    return {".py": "python", ".cpp": "cpp", ".md": "markdown", ".sh": "bash", ".json": "json"}.get(ext, "")


def build(root, n, max_chars):
    convs, index = [], []
    for i, (path, lang) in enumerate(SOURCES[:n]):
        full = os.path.join(root, path)
        with open(full, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8")[:max_chars]
        tasks = TASKS_RU if lang == "ru" else TASKS_EN
        task = tasks[i % len(tasks)].format(path=path)
        tool_reply = (f"read_file({path!r}) returned {len(raw)} bytes" +
                      (f" (first {max_chars} characters shown)" if len(raw) > max_chars else "") +
                      f":\n```{fence(path)}\n{text}\n```")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"I will read {path} first." if lang == "en" else f"Сначала прочитаю {path}."},
            {"role": "user", "content": tool_reply + "\n\n" + (FOLLOW_RU if lang == "ru" else FOLLOW_EN)},
        ]
        name = f"c{i:02d}_{lang}_{os.path.basename(path).replace('.', '_')}"
        convs.append({"name": name, "lang": lang, "source": path, "messages": messages})
        index.append({"name": name, "source": path, "lang": lang, "source_bytes": len(raw),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "chars_used": len(text)})
    return convs, index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=len(SOURCES))
    ap.add_argument("--max-chars", type=int, default=7000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    convs, index = build(a.root, a.n, a.max_chars)
    for c in convs:
        with open(os.path.join(a.out, c["name"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "INDEX.json"), "w") as f:
        json.dump({"system": SYSTEM, "conversations": index, "max_chars": a.max_chars}, f, indent=1)
    print(f"{len(convs)} conversations -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
