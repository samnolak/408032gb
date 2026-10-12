"""Conversations for the N14 chat-template probes (gap 6 of docs/reassessment/CORRECTNESS.md).

Messages are kept in the form Hugging Face templates expect (tool-call `arguments` as a dict); `to_openai()` gives the
OpenAI wire form llama-server takes (`arguments` as a JSON string, as /v1/chat/completions clients send it).
The texts are invented; they exercise template branches, not model behaviour.
"""
import copy
import json

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files under a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory."}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file; returns at most max_lines lines.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "max_lines": {"type": "integer", "minimum": 1}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search files for a regular expression («поиск»).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}},
            "ignore_case": {"type": "boolean"}},
            "required": ["pattern", "paths"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "options": {"type": "object", "properties": {"mode": {"type": "string"}, "backup": {"type": "boolean"}}}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite.",
        "parameters": {"type": "object", "properties": {
            "verbose": {"type": "boolean"}, "timeout_s": {"type": "number"}, "only": {"type": ["string", "null"]}},
            "required": ["verbose"]}}},
]

SYSTEM = "You are a coding agent. Work in the repository /work. Be brief."


def tc(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def tool(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# A realistic agent session that grows one message at a time (user -> assistant with tool calls -> tool results ->
# ... -> answer -> user -> ...). Every assistant turn carries reasoning_content.
AGENT_SESSION = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "The test suite fails. Find the failing test and fix it."},
    {"role": "assistant", "reasoning_content": "I should look at the layout of the repository first.",
     "content": "", "tool_calls": [tc("call_1", "list_files", {"path": "/work"})]},
    tool("call_1", "src/\ntests/\nREADME.md"),
    {"role": "assistant", "reasoning_content": "Two things to check: the tests and where `parse` is used.",
     "content": "Reading the tests and searching for the parser.",
     "tool_calls": [tc("call_2", "read_file", {"path": "/work/tests/test_parse.py", "max_lines": 200}),
                    tc("call_3", "grep", {"pattern": "def parse", "paths": ["src", "tests"], "ignore_case": False})]},
    tool("call_2", "def test_parse():\n    assert parse('1,2') == [1, 2]\n"),
    tool("call_3", "src/parse.py:3:def parse(s):"),
    {"role": "assistant", "reasoning_content": "parse() splits on ';' but the test uses ','. Fix the separator.\n",
     "content": "The separator is wrong; fixing it.",
     "tool_calls": [tc("call_4", "write_file", {
         "path": "/work/src/parse.py",
         "content": "def parse(s):\n    return [int(x) for x in s.split(',')]  # «запятая»\n",
         "options": {"mode": "overwrite", "backup": True}})]},
    tool("call_4", "ok: 2 lines written"),
    {"role": "assistant", "reasoning_content": "Run the tests to confirm.",
     "content": "", "tool_calls": [tc("call_5", "run_tests", {"verbose": True, "timeout_s": 1.5, "only": None})]},
    tool("call_5", "1 passed in 0.01s"),
    {"role": "assistant", "reasoning_content": "All green; summarise.",
     "content": "Fixed: `parse` now splits on ','. The test passes."},
    {"role": "user", "content": "Thanks. Now add a test for an empty string."},
    {"role": "assistant", "reasoning_content": "An empty string should give an empty list; the current code raises.",
     "content": "", "tool_calls": [tc("call_6", "read_file", {"path": "/work/src/parse.py"})]},
    tool("call_6", "def parse(s):\n    return [int(x) for x in s.split(',')]\n"),
    {"role": "assistant", "reasoning_content": "Handle the empty case and add the test.",
     "content": "Done: `parse('')` returns `[]`, test added."},
    {"role": "user", "content": "Bye."},
]

CASES = {
    "system_user": {"messages": [{"role": "system", "content": "You are terse."},
                                 {"role": "user", "content": "Hi there"}]},
    "multi_turn": {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "  And 3+3?  "},
        {"role": "assistant", "content": "\n6\n"},
        {"role": "user", "content": "Привет, мир"}]},
    "no_system": {"messages": [{"role": "user", "content": "Hi"}]},
    "reasoning": {"messages": [
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "reasoning_content": "Rayleigh scattering.", "content": "Short wavelengths scatter more."},
        {"role": "user", "content": "And sunsets?"},
        {"role": "assistant", "reasoning_content": "  Longer path.\n", "content": "Long path, blue scattered out."},
        {"role": "user", "content": "Thanks"}]},
    "reasoning_mixed": {"messages": [  # second assistant turn without reasoning_content
        {"role": "user", "content": "u1"},
        {"role": "assistant", "reasoning_content": "R1", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"}]},
    "inline_think": {"messages": [  # reasoning embedded in content, as a client that never splits it would send it
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>R1</think>a1"},
        {"role": "user", "content": "u2"}]},
    "reasoning_empty": {"messages": [  # second assistant turn with an empty reasoning_content
        {"role": "user", "content": "u1"},
        {"role": "assistant", "reasoning_content": "R1", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "reasoning_content": "", "content": "a2"},
        {"role": "user", "content": "u3"}]},
    "tools": {"tools": True, "messages": AGENT_SESSION[:11]},
    # ends with an assistant message: llama-server prefills it (continue_final_message, --prefill-assistant default on)
    "tools_ends_with_assistant": {"tools": True, "messages": AGENT_SESSION[:12]},
    "agent_session": {"tools": True, "messages": AGENT_SESSION},
}


def without_reasoning_at(conv, k):
    """The session with no reasoning_content on message k: what a client sends back for a turn whose reasoning was
    empty (llama-server omits an empty reasoning_content, common/chat.cpp to_json_oaicompat @79e2e74)."""
    out = copy.deepcopy(conv)
    out[k].pop("reasoning_content", None)
    return out


# the turn at index 9 (run_tests) with an empty reasoning, i.e. the model wrote an empty think block
AGENT_SESSION_EMPTY_REASONING = without_reasoning_at(AGENT_SESSION, 9)


def to_openai(messages):
    """OpenAI wire form: tool-call arguments as a JSON string (ensure_ascii False, default separators)."""
    out = copy.deepcopy(messages)
    for m in out:
        for call in m.get("tool_calls") or []:
            args = call["function"]["arguments"]
            if not isinstance(args, str):
                call["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
    return out


# N23 (corrected GLM template, templates/glm-5.3-flash/chat_template.jinja): conversations aimed at the loop-scope bug
# of llama.cpp's Jinja engine at 79e2e74 (a {% set %} in a {% for %} body survives into the next iteration) - assistant
# turns without reasoning, with an empty one (llama-server drops "" from the message, common/chat.cpp:202-204), with
# None, with reasoning inline in content, in tool loops and over several user rounds. Texts are invented.
def _asst(content="", reasoning=..., calls=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not ...:
        m["reasoning_content"] = reasoning
    if calls:
        m["tool_calls"] = calls
    return m


AGENT_LOOP_MIXED = [  # two user rounds, parallel calls; reasoning present / empty / missing / whitespace
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Count the TODO markers in src and tests."},
    _asst("", "Search both trees at once.", [tc("c1", "grep", {"pattern": "TODO", "paths": ["src"], "ignore_case": True}),
                                            tc("c2", "grep", {"pattern": "TODO", "paths": ["tests"], "ignore_case": True})]),
    tool("c1", "src/a.py:1:# TODO\nsrc/b.py:7:# todo"),
    tool("c2", ""),
    _asst("", "", [tc("c3", "read_file", {"path": "/work/src/b.py", "max_lines": 10})]),
    tool("c3", "x = 1\n# todo: rename\n"),
    _asst("Checking the README too.", ..., [tc("c4", "read_file", {"path": "/work/README.md"})]),
    tool("c4", "No TODO here («нет»)."),
    _asst("", "Run the TODO tests.", [tc("c5", "run_tests", {"verbose": False, "timeout_s": 3, "only": "todo"})]),
    tool("c5", "0 passed"),
    _asst("Two markers, both in src.", "Done: a.py and b.py."),
    {"role": "user", "content": "Remove them."},
    _asst("", ..., [tc("c6", "write_file", {"path": "/work/src/a.py", "content": "", "options": {"mode": "w"}})]),
    tool("c6", "ok"),
    _asst("", "  \n", [tc("c7", "write_file", {"path": "/work/src/b.py", "content": "x = 1\n"})]),
    tool("c7", "ok"),
    _asst("Both removed.", ...),
    {"role": "user", "content": "Thanks."},
]

N23_CASES = {
    "missing_first": {"messages": [  # first assistant turn without reasoning, second with
        {"role": "user", "content": "u1"}, _asst("a1"), {"role": "user", "content": "u2"},
        _asst("a2", "R2"), {"role": "user", "content": "u3"}]},
    "missing_after_two": {"messages": [  # R1, R2, then a turn without: which one would leak?
        {"role": "user", "content": "u1"}, _asst("a1", "R1"), {"role": "user", "content": "u2"},
        _asst("a2", "R2"), {"role": "user", "content": "u3"}, _asst("a3"), {"role": "user", "content": "u4"}]},
    # reasoning_content None (JSON null): llama-server @79e2e74 answers HTTP 500 "type must be string, but is null"
    # (common/chat.cpp:422-423 reads it as a string), with either template; jinja2 renders an empty think block
    "reasoning_null": {"messages": [
        {"role": "user", "content": "u1"}, _asst("a1", "R1"), {"role": "user", "content": "u2"},
        _asst("a2", None), {"role": "user", "content": "u3"}]},
    "reasoning_whitespace_then_missing": {"messages": [
        {"role": "user", "content": "u1"}, _asst("a1", " \n "), {"role": "user", "content": "u2"},
        _asst("a2"), {"role": "user", "content": "u3"}]},
    "inline_then_missing": {"messages": [  # reasoning only inline in content, then a turn without any
        {"role": "user", "content": "u1"}, _asst("<think>R1</think>a1"), {"role": "user", "content": "u2"},
        _asst("a2"), {"role": "user", "content": "u3"}]},
    "inline_empty_then_missing": {"messages": [
        {"role": "user", "content": "u1"}, _asst("<think></think>a1"), {"role": "user", "content": "u2"},
        _asst("a2"), {"role": "user", "content": "u3"}]},
    "tools_reasoning_empty": {"tools": True, "messages": [  # tool loop: R1, "", then missing
        {"role": "user", "content": "List and read."},
        _asst("", "R1", [tc("t1", "list_files", {"path": "/work"})]), tool("t1", "a.txt"),
        _asst("", "", [tc("t2", "read_file", {"path": "/work/a.txt"})]), tool("t2", "hello"),
        _asst("", ..., [tc("t3", "read_file", {"path": "/work/b.txt"})]), tool("t3", "missing"),
        _asst("Read a.txt; b.txt is missing.", ...), {"role": "user", "content": "ok"}]},
    "tools_no_reasoning_at_all": {"tools": True, "messages": [
        {"role": "user", "content": "List."}, _asst("", ..., [tc("t1", "list_files", {"path": "/"})]),
        tool("t1", "work/"), _asst("One directory.", ...), {"role": "user", "content": "ok"}]},
    "agent_loop_mixed": {"tools": True, "messages": AGENT_LOOP_MIXED},
    "agent_loop_mixed_open": {"tools": True, "messages": AGENT_LOOP_MIXED[:15]},  # ends with a tool result
    "agent_session_empty_reasoning": {"tools": True, "messages": AGENT_SESSION_EMPTY_REASONING},
}
