"""Deterministic tool simulator for tool_use tasks (stdlib only).

A task never contains code. Its "tool_simulator" object holds data only:

  {"responses": [                                  # canned results, first match wins
      {"name": "get_order", "arguments": {"order_id": "A-17"}, "match": "exact", "result": {...}},
      {"name": "get_order", "arguments": {"order_id": "A-17"}, "match": "subset",
       "results": [first, second, ...]}            # successive results for repeated matching calls
   ],
   "builtins": {"calculator": {"impl": "calculator"},
                "stock": {"impl": "lookup", "key_arg": "sku", "table": {"X-1": {...}}},
                "search_kb": {"impl": "search", "query_arg": "query", "documents": [{"id": .., "text": ..}]}},
   "default_result": {"error": "..."}}             # when nothing matches (optional)

Argument matching: "exact" (default) = the parsed JSON arguments equal the canned object (integral floats
equal ints); "subset" = every canned key is present with an equal value. Results that are not strings are
sent to the model as JSON text. Builtin implementations (BUILTINS) are small, safe Python functions that live
here, in the stand; tasks only choose one by name and pass data.
"""
from __future__ import annotations

import ast
import json
import math
import operator

from ev_schema import json_equal

MAX_POW = 64


class ToolError(Exception):
    pass


def args_match(call_args, want, mode="exact"):
    if not isinstance(call_args, dict):
        return False
    if mode == "exact":
        return json_equal(call_args, want)
    if mode == "subset":
        return all(k in call_args and json_equal(call_args[k], v) for k, v in want.items())
    raise ValueError("unknown match mode %r" % mode)


# ---------------------------------------------------------------------------- builtins
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_arith(expr):
    """Evaluate + - * / // % ** and parentheses over int/float literals; anything else is rejected."""
    if not isinstance(expr, str) or len(expr) > 500:
        raise ToolError("expression must be a string of at most 500 characters")
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")      # '^' is read as a power
    except SyntaxError as e:
        raise ToolError("syntax error: %s" % e.msg)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(b) > MAX_POW or abs(a) > 1e6):
                raise ToolError("exponent too large")
            try:
                return _BIN[type(node.op)](a, b)
            except ZeroDivisionError:
                raise ToolError("division by zero")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ToolError("unsupported element %s" % type(node).__name__)
    v = ev(tree)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ToolError("result is not finite")
        v = round(v, 10)
        if v.is_integer() and abs(v) < 1e15:
            v = int(v)
    return v


def b_calculator(args, cfg):
    arg = cfg.get("expression_arg", "expression")
    if arg not in args:
        raise ToolError("missing argument %r" % arg)
    return {"result": safe_arith(args[arg])}


def b_lookup(args, cfg):
    key_arg = cfg.get("key_arg", "id")
    if key_arg not in args:
        raise ToolError("missing argument %r" % key_arg)
    key = str(args[key_arg]).strip()
    table = cfg.get("table") or {}
    if key not in table:
        if cfg.get("case_insensitive"):
            for k, v in table.items():
                if k.casefold() == key.casefold():
                    return v
        return {"error": "not found: %s" % key}
    return table[key]


def b_search(args, cfg):
    q_arg = cfg.get("query_arg", "query")
    if q_arg not in args or not isinstance(args[q_arg], str):
        raise ToolError("missing string argument %r" % q_arg)
    terms = [t for t in args[q_arg].casefold().split() if t]
    hits = []
    for d in cfg.get("documents") or []:
        text = str(d.get("text", "")).casefold()
        if terms and all(t in text for t in terms):
            hits.append({"id": d.get("id"), "text": d.get("text")})
    return {"results": hits[: int(cfg.get("max_results", 3))]}


BUILTINS = {"calculator": b_calculator, "lookup": b_lookup, "search": b_search}


# ---------------------------------------------------------------------------- simulator
class ToolSimulator:
    def __init__(self, spec, tools):
        self.spec = spec or {}
        self.declared = {t["function"]["name"] for t in (tools or []) if isinstance(t, dict) and "function" in t}
        self.hits = {}                     # canned index -> number of times used

    def call(self, name, arguments):
        """Execute one call. Returns (result_text, record)."""
        rec = {"name": name, "arguments_raw": arguments, "args": None, "args_valid": False,
               "source": None, "matched": None}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else (
                {} if not arguments else arguments)
        except ValueError:
            args = None
        if isinstance(args, dict):
            rec["args"], rec["args_valid"] = args, True
        if name not in self.declared:
            rec["source"] = "unknown_tool"
            return json.dumps({"error": "unknown tool %r" % name}), rec
        if not rec["args_valid"]:
            rec["source"] = "invalid_arguments"
            return json.dumps({"error": "arguments are not a valid JSON object"}), rec
        for i, r in enumerate(self.spec.get("responses") or []):
            if r.get("name") == name and args_match(args, r.get("arguments") or {}, r.get("match", "exact")):
                n = self.hits.get(i, 0)
                self.hits[i] = n + 1
                if "results" in r:
                    seq = r["results"]
                    result = seq[min(n, len(seq) - 1)]
                else:
                    result = r.get("result")
                rec["source"], rec["matched"] = "canned", i
                return (result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)), rec
        b = (self.spec.get("builtins") or {}).get(name)
        if b:
            rec["source"] = "builtin:" + b["impl"]
            try:
                out = BUILTINS[b["impl"]](args, b)
            except ToolError as e:
                out = {"error": str(e)}
            return json.dumps(out, ensure_ascii=False), rec
        rec["source"] = "default"
        d = self.spec.get("default_result", {"error": "no simulated result for this call"})
        return (d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)), rec


def spec_problems(spec, tools):
    """Static problems of a simulator spec (unknown builtin, response for an undeclared tool, ...)."""
    out = []
    declared = {t.get("function", {}).get("name") for t in (tools or []) if isinstance(t, dict)}
    for i, r in enumerate(spec.get("responses") or []):
        if r.get("name") not in declared:
            out.append("tool_simulator.responses[%d]: tool %r is not declared in tools" % (i, r.get("name")))
        if r.get("match", "exact") not in ("exact", "subset"):
            out.append("tool_simulator.responses[%d]: match must be exact or subset" % i)
        if ("result" in r) == ("results" in r):
            out.append("tool_simulator.responses[%d]: give exactly one of result / results" % i)
        if "results" in r and not (isinstance(r["results"], list) and r["results"]):
            out.append("tool_simulator.responses[%d]: results must be a non-empty list" % i)
    for name, b in (spec.get("builtins") or {}).items():
        if name not in declared:
            out.append("tool_simulator.builtins: tool %r is not declared in tools" % name)
        if not isinstance(b, dict) or b.get("impl") not in BUILTINS:
            out.append("tool_simulator.builtins[%r]: impl must be one of %s" % (name, sorted(BUILTINS)))
    extra = set(spec) - {"responses", "builtins", "default_result"}
    if extra:
        out.append("tool_simulator: unknown keys %s" % sorted(extra))
    return out
