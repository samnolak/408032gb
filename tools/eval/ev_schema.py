"""A small JSON Schema subset validator (stdlib only) for structured_output checks.

Supported keywords (draft 2020-12 semantics for these keywords; anything else is REJECTED by
schema_problems() so a task never silently relies on an unchecked keyword):

  type (string or list)  enum  const
  object : properties  required  additionalProperties (bool or schema)  minProperties  maxProperties
  array  : items (one schema for every element)  minItems  maxItems  uniqueItems
  string : minLength  maxLength  pattern (Python re.search)
  number : minimum  maximum  exclusiveMinimum  exclusiveMaximum (numbers)  multipleOf
  logic  : allOf  anyOf  oneOf  not
  $ref   : local JSON pointers only ("#", "#/$defs/x", "#/definitions/x")
  annotations (accepted, not validated): $schema $id $comment $defs definitions title description
                                         default examples format

Type rules: booleans are not numbers or integers; an integer is an int or a float with no
fractional part (1.0 is an integer, as in JSON Schema draft 6+).
"""
from __future__ import annotations

import json
import math
import re

VALIDATED = {"type", "enum", "const", "properties", "required", "additionalProperties", "minProperties",
             "maxProperties", "items", "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern",
             "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "allOf", "anyOf", "oneOf",
             "not", "$ref"}
ANNOTATIONS = {"$schema", "$id", "$comment", "$defs", "definitions", "title", "description", "default", "examples",
               "format"}
TYPES = {"null", "boolean", "integer", "number", "string", "array", "object"}


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def type_ok(x, t):
    if t == "null":
        return x is None
    if t == "boolean":
        return isinstance(x, bool)
    if t == "integer":
        if isinstance(x, bool):
            return False
        return isinstance(x, int) or (isinstance(x, float) and math.isfinite(x) and x.is_integer())
    if t == "number":
        return _is_number(x)
    if t == "string":
        return isinstance(x, str)
    if t == "array":
        return isinstance(x, list)
    if t == "object":
        return isinstance(x, dict)
    raise ValueError("unknown type %r" % t)


def json_type(x):
    for t in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if type_ok(x, t):
            return t
    return type(x).__name__


def canonical(x):
    """JSON text used for equality: integral floats compare equal to ints, bools stay bools."""
    def norm(v):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return v
        if isinstance(v, float) and math.isfinite(v) and v.is_integer():
            return int(v)
        if isinstance(v, list):
            return [norm(i) for i in v]
        if isinstance(v, dict):
            return {k: norm(i) for k, i in v.items()}
        return v
    return json.dumps(norm(x), sort_keys=True, ensure_ascii=False)


def json_equal(a, b):
    return canonical(a) == canonical(b)


def _sub_schemas(schema, path):
    """Yield (subschema, path) for every place a schema can nest."""
    for key in ("properties", "$defs", "definitions"):
        v = schema.get(key)
        if isinstance(v, dict):
            for name, sub in v.items():
                yield sub, "%s/%s/%s" % (path, key, name)
    for key in ("additionalProperties", "items", "not"):
        v = schema.get(key)
        if isinstance(v, dict):
            yield v, "%s/%s" % (path, key)
    for key in ("allOf", "anyOf", "oneOf"):
        v = schema.get(key)
        if isinstance(v, list):
            for i, sub in enumerate(v):
                yield sub, "%s/%s/%d" % (path, key, i)


def schema_problems(schema, path="#"):
    """List of problems (unsupported keywords, malformed values) in a schema; [] when usable."""
    if isinstance(schema, bool):
        return []
    if not isinstance(schema, dict):
        return ["%s: schema must be an object or a boolean" % path]
    out = []
    for k in schema:
        if k not in VALIDATED and k not in ANNOTATIONS:
            out.append("%s: unsupported keyword %r (supported: %s)" % (path, k, ", ".join(sorted(VALIDATED))))
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        for x in ts:
            if x not in TYPES:
                out.append("%s: unknown type %r" % (path, x))
    if "items" in schema and isinstance(schema["items"], list):
        out.append("%s: tuple-form 'items' (a list) is not supported" % path)
    if "required" in schema and not (isinstance(schema["required"], list)
                                     and all(isinstance(x, str) for x in schema["required"])):
        out.append("%s: 'required' must be a list of strings" % path)
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError) as e:
            out.append("%s: bad pattern: %s" % (path, e))
    ref = schema.get("$ref")
    if ref is not None and not (isinstance(ref, str) and (ref == "#" or ref.startswith("#/"))):
        out.append("%s: only local $ref ('#/...') is supported, got %r" % (path, ref))
    for sub, p in _sub_schemas(schema, path):
        out.extend(schema_problems(sub, p))
    return out


def _resolve_ref(root, ref):
    if ref == "#":
        return root
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError("unresolvable $ref %r" % ref)
    return node


def validate(instance, schema, root=None, path="", depth=0):
    """Return a list of error strings ("<json pointer>: <message>"); [] means valid."""
    if root is None:
        root = schema
    if depth > 64:
        return ["%s: $ref recursion too deep" % (path or "/")]
    where = path or "/"
    if schema is True:
        return []
    if schema is False:
        return ["%s: no value is allowed here (schema false)" % where]
    errs = []
    if "$ref" in schema:
        try:
            target = _resolve_ref(root, schema["$ref"])
        except KeyError as e:
            return ["%s: %s" % (where, e.args[0])]
        errs.extend(validate(instance, target, root, path, depth + 1))
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(type_ok(instance, x) for x in ts):
            errs.append("%s: expected type %s, got %s" % (where, "/".join(ts), json_type(instance)))
            return errs                                  # further keyword errors would only be noise
    if "enum" in schema and not any(json_equal(instance, e) for e in schema["enum"]):
        errs.append("%s: value %s not in enum %s" % (where, canonical(instance)[:80], canonical(schema["enum"])[:200]))
    if "const" in schema and not json_equal(instance, schema["const"]):
        errs.append("%s: value %s != const %s" % (where, canonical(instance)[:80], canonical(schema["const"])[:80]))
    if isinstance(instance, dict):
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            if k not in instance:
                errs.append("%s: missing required property %r" % (where, k))
        for k, v in instance.items():
            p = "%s/%s" % (path, k.replace("~", "~0").replace("/", "~1"))
            if k in props:
                errs.extend(validate(v, props[k], root, p, depth + 1))
            elif "additionalProperties" in schema:
                ap = schema["additionalProperties"]
                if ap is False:
                    errs.append("%s: additional property %r is not allowed" % (where, k))
                elif isinstance(ap, dict):
                    errs.extend(validate(v, ap, root, p, depth + 1))
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errs.append("%s: fewer than %d properties" % (where, schema["minProperties"]))
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            errs.append("%s: more than %d properties" % (where, schema["maxProperties"]))
    if isinstance(instance, list):
        if isinstance(schema.get("items"), (dict, bool)):
            for i, v in enumerate(instance):
                errs.extend(validate(v, schema["items"], root, "%s/%d" % (path, i), depth + 1))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append("%s: fewer than %d items" % (where, schema["minItems"]))
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append("%s: more than %d items" % (where, schema["maxItems"]))
        if schema.get("uniqueItems"):
            seen = set()
            for v in instance:
                c = canonical(v)
                if c in seen:
                    errs.append("%s: items are not unique (%s repeats)" % (where, c[:80]))
                    break
                seen.add(c)
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append("%s: shorter than %d characters" % (where, schema["minLength"]))
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errs.append("%s: longer than %d characters" % (where, schema["maxLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errs.append("%s: %r does not match pattern %r" % (where, instance[:80], schema["pattern"]))
    if _is_number(instance):
        x = instance
        if "minimum" in schema and x < schema["minimum"]:
            errs.append("%s: %s < minimum %s" % (where, x, schema["minimum"]))
        if "maximum" in schema and x > schema["maximum"]:
            errs.append("%s: %s > maximum %s" % (where, x, schema["maximum"]))
        if "exclusiveMinimum" in schema and x <= schema["exclusiveMinimum"]:
            errs.append("%s: %s <= exclusiveMinimum %s" % (where, x, schema["exclusiveMinimum"]))
        if "exclusiveMaximum" in schema and x >= schema["exclusiveMaximum"]:
            errs.append("%s: %s >= exclusiveMaximum %s" % (where, x, schema["exclusiveMaximum"]))
        m = schema.get("multipleOf")
        if m:
            q = x / m
            if not math.isclose(q, round(q), rel_tol=0, abs_tol=1e-9):
                errs.append("%s: %s is not a multiple of %s" % (where, x, m))
    for i, sub in enumerate(schema.get("allOf") or []):
        errs.extend(validate(instance, sub, root, path, depth + 1))
    if "anyOf" in schema:
        if not any(not validate(instance, sub, root, path, depth + 1) for sub in schema["anyOf"]):
            errs.append("%s: matches none of anyOf" % where)
    if "oneOf" in schema:
        n = sum(1 for sub in schema["oneOf"] if not validate(instance, sub, root, path, depth + 1))
        if n != 1:
            errs.append("%s: matches %d of oneOf (exactly 1 required)" % (where, n))
    if "not" in schema and not validate(instance, schema["not"], root, path, depth + 1):
        errs.append("%s: must not match the 'not' schema" % where)
    return errs
