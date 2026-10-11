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
