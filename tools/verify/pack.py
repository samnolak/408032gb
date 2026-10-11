#!/usr/bin/env python3
"""Verification-pack assembler (ROADMAP R6.1): copy the operator's tool outputs into one directory, hash every file and
judge each customer acceptance criterion PASS / FAIL / NOT RUN.

Python 3 stdlib only. It imports two stdlib-only modules of this repo so that formulas and tables cannot drift:
tools/bench/runlog.py (per-request metrics, percentiles, mock owner string) and tools/gguf_quant_audit.py (ggml types).

Hard rules
  * a criterion whose artifact is missing, unparseable or internally inconsistent (summary.json != raw.jsonl) is
    NOT RUN, never PASS;
  * any bench artifact produced against tools/bench/mock_server.py (meta.is_mock_server, /v1/models owned_by
    "408032gb-bench-mock", the "MOCK SERVER RUN" banner of summary.md) makes every criterion that uses it NOT RUN with
    reason "mock";
  * decode speed is judged only at the filled-context point: single-stream cells whose target context (input with
    --length-mode total, input + output with --length-mode prompt) is >= ctx_tokens, output >= decode_min_output_tokens,
    and whose measured usage.prompt_tokens + output >= ctx_tokens - ctx_slack_tokens; short-context cells never count;
  * overall PASS only if every mandatory criterion is PASS and every input artifact parsed; FAIL if any mandatory
    criterion is FAIL; NOT RUN otherwise.
  * a run declared synthetic in the criteria file ("synthetic_run": true, N24) is never PASS overall; the criteria that
    judge the model's behaviour (subset of BEHAVIOUR_CRITERIA listed in "synthetic_not_applicable") become NOT RUN with
    reason "not_applicable_synthetic" and their raw verdict is kept and listed - only if every bench run's model
    description carries the positive marker SYNTHETIC (upper case, whole word, not after "non-"/"not "); a probe request
    error, including a rejected response_format (HTTP 400/422), stays FAIL (that is the engine, not the model).

Inputs (--runs DIR, any layout; artifacts are recognised by content, see tools/verify/README.md)
  run dir, meta.json scenario "matrix"            tools/bench/run_matrix.py --out-dir DIR
  run dir, meta.json scenario "probes"            tools/bench/probes.py --out-dir DIR
  JSON with classes / findings / verdict          tools/gguf_quant_audit.py FIRST.gguf --json > FILE
  JSON with ref / cand / top_k / rows             tools/quality/logprob_compare.py ... --out FILE
  JSON with url / prompt_file / gen / runs        tools/kv/slot_persist.py ... --out FILE
  JSON list with measured_decode_p50 / bound      tools/bench/vs_roofline.py RUN_DIR ... --json > FILE
  JSON with synthetic / facts / red_flags         scripts/diag/summarize.py DIAGDIR --json FILE
  JSON with repo / revision / files / total_bytes scripts/offline/weights.py manifest ... -o FILE
  text ending in "verify: N/M files OK"           scripts/offline/weights.py verify MANIFEST --dest DIR > FILE.txt
Everything else is copied, hashed and listed as unclassified.

Output (--out DIR, must not exist or be empty)
  artifacts/       byte copies (or hard links with --hardlink) of everything under --runs
  criteria.json    the effective criteria (defaults merged with --criteria)
  env.txt          git rev-parse HEAD, git status --short, uname -a, python version, nvidia-smi -L (assembling host)
  verdict.json     one entry per criterion: status, reason code, measured value, threshold, evidence paths
  VERDICT.md       the same in Russian
  MANIFEST.sha256  sha256 of every file above (sha256sum format); `--check DIR` re-verifies it

Usage
  python3 tools/verify/pack.py --runs RUNS_DIR --out PACK_DIR [--criteria FILE] [--hardlink]
  python3 tools/verify/pack.py --check PACK_DIR
Exit codes: build 0 = overall PASS, 1 = FAIL, 3 = NOT RUN, 2 = usage or I/O error;
            --check 0 = every file matches, 1 = mismatch / missing / extra / malformed, 2 = no manifest.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools", "bench"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import runlog  # noqa: E402  (tools/bench/runlog.py)
import gguf_quant_audit  # noqa: E402  (tools/gguf_quant_audit.py)

SCHEMA = "408032gb-verify-pack/1"
CRITERIA_SCHEMA = "408032gb-verify-criteria/1"
DEFAULT_CRITERIA = os.path.join(HERE, "criteria.default.json")
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT RUN"
MOCK_BANNER = "MOCK SERVER RUN"
INPUTS = "artifacts"
MANIFEST = "MANIFEST.sha256"
TYPE_BPW = {name: size * 8.0 / block for name, block, size in gguf_quant_audit.GGML_TYPES.values()}
# Criteria that judge what the MODEL does (scores of answers), not what the engine, the API or the tools do. Only these
# may be declared not applicable to a synthetic (random-weight) model: criteria "synthetic_not_applicable" (N24).
BEHAVIOUR_CRITERIA = ("agent_api_probes", "context_probes_full_length", "manual_quality_eval")
# A declared synthetic run needs this positive marker in every bench run's model_desc (N24 audit M1): the upper-case
# word SYNTHETIC as a whole word, not negated by a preceding "non-"/"not " (any case of those); "non-synthetic",
# "NOT SYNTHETIC" or "synthetic" in lower case do not count.
SYNTHETIC_MARK = re.compile(r"(?<![A-Za-z0-9_-])(?<!non-)(?<!Non-)(?<!NON-)(?<!not )(?<!Not )(?<!NOT )SYNTHETIC(?![A-Za-z0-9_])")


def synthetic_marked(desc):
    return bool(SYNTHETIC_MARK.search(str(desc or "")))

# id -> Russian title (VERDICT.md); order = report order
TITLES = [
    ("single_configuration", "Все прогоны стенда — одна конфигурация (модель, квант, движок)"),
    ("openai_api", "OpenAI-совместимый API: /v1/models, потоковые ответы, usage"),
    ("context_probes_full_length", "Контекст реально используется: needle / multihop / codeqa на полной длине"),
    ("output_after_full_prompt", "Модель выдаёт ответ после промпта полной длины"),
    ("agent_api_probes", "Агентное использование API: вызов инструментов и JSON"),
    ("decode_filled_ctx_floor", "Декод single-stream при заполненном контексте ≥ пола заказчика"),
    ("decode_filled_ctx_target", "Декод single-stream при заполненном контексте ≥ желательной цели"),
    ("prefix_cache", "Кэш префиксов: warm TTFT против cold TTFT, сервер сообщает cached tokens"),
    ("agent_loop", "Agent loop до длинного контекста без отказов"),
    ("quant_min_bpw", "Основные веса ≥ 4 bpw (gguf_quant_audit)"),
    ("weights_integrity", "Целостность весов: sha256 каждого файла против манифеста ревизии (weights.py verify)"),
    ("exact_routing_logprobs", "Точная маршрутизация: совпадение top-1 с эталоном (logprob_compare)"),
    ("kv_slot_persist", "Сохранение и восстановление KV-слота (slot_persist)"),
    ("decode_vs_roofline", "Измеренный декод против границы roofline (vs_roofline)"),
    ("hardware_facts", "Факты о железе измерены (scripts/diag/summarize.py)"),
    ("determinism", "Детерминизм при temperature 0 (информационно)"),
    ("manual_operational_faults", "Эксплуатационные отказы F1–F8 (DEPLOY.md §6, вручную)"),
    ("manual_quality_eval", "Оценочный набор заказчика R2.2 (вручную)"),
]
# criterion id -> row of docs/reassessment/VERIFICATION_PACK.md (V1-V11); None = cross-cutting check of this tool
VNUM = {"quant_min_bpw": "V1", "weights_integrity": "V2", "context_probes_full_length": "V3",
        "output_after_full_prompt": "V3", "agent_api_probes": "V3", "decode_filled_ctx_floor": "V4",
        "decode_filled_ctx_target": "V4", "prefix_cache": "V5", "agent_loop": "V5", "exact_routing_logprobs": "V6",
        "kv_slot_persist": "V7", "hardware_facts": "V8", "decode_vs_roofline": "V9", "manual_operational_faults": "V10",
        "manual_quality_eval": "V11", "single_configuration": None, "openai_api": None, "determinism": None}
TITLE = dict(TITLES)
CRITERIA_IDS = [c for c, _ in TITLES]

REASON_RU = {
    "ok": "условие выполнено",
    "missing": "артефакт отсутствует",
    "unparseable": "артефакт не разбирается",
    "inconsistent": "артефакт противоречив (сводка не совпадает с сырыми данными)",
    "mock": "артефакт получен на mock-сервере",
    "mock_unknown": "нельзя установить, был ли сервер mock",
    "synthetic": "артефакт помечен как синтетический",
    "not_filled": "нет замера при заполненном контексте",
    "insufficient": "данных недостаточно (повторы, объём или пороги прогона слабее критериев)",
    "threshold": "измеренное значение не проходит порог",
    "failed": "в прогоне есть отказы или ошибки",
    "mismatch": "прогоны относятся к разным конфигурациям",
    "manual": "проверяется вручную, не автоматически",
    "invalid": "сравнение недействительно: входы границы не соответствуют конфигурации",
    "not_applicable_synthetic": "не применимо: прогон объявлен синтетическим, критерий проверяет поведение модели",
}


# ---------------------------------------------------------------------------- small helpers
def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def close(a, b, rel=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(a)), abs(float(b)))


def fmt(x, nd=1):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return ("%%.%df" % nd) % x
    return str(x)


def fmt_g(x):
    return "n/a" if not is_num(x) else "%.3g" % x


def norm_url(u):
    u = (u or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u.lower()


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------- criteria
def load_criteria(path=None):
    """Defaults merged with an optional user file. Unknown keys are an error (a typo must not silently use a default)."""
    base = load_json(DEFAULT_CRITERIA)
    crit = copy.deepcopy(base)
    if path:
        user = load_json(path)
        if not isinstance(user, dict):
            raise ValueError("criteria file must hold a JSON object")
        unknown = sorted(set(user) - set(base))
        if unknown:
            raise ValueError("unknown criteria keys: %s" % ", ".join(unknown))
        for k, v in user.items():
            if isinstance(base.get(k), dict) and isinstance(v, dict) and k != "_notes":
                crit[k] = dict(base[k], **v)
            else:
                crit[k] = v
    if crit.get("schema") != CRITERIA_SCHEMA:
        raise ValueError("criteria schema must be %r" % CRITERIA_SCHEMA)
    if crit.get("decode_filled_context_basis") not in ("prompt_plus_output", "prompt"):
        raise ValueError("decode_filled_context_basis must be 'prompt_plus_output' or 'prompt'")
    bad = [m for m in crit.get("mandatory", []) if m not in CRITERIA_IDS]
    if bad:
        raise ValueError("unknown criterion ids in 'mandatory': %s" % ", ".join(bad))
    if not isinstance(crit.get("synthetic_run"), bool):
        raise ValueError("synthetic_run must be true or false")
    na = crit.get("synthetic_not_applicable")
    if not isinstance(na, list) or any(c not in BEHAVIOUR_CRITERIA for c in na):
        raise ValueError("synthetic_not_applicable must be a list of model-behaviour criteria (%s); got %r"
                         % (", ".join(BEHAVIOUR_CRITERIA), na))
    tol = crit.get("logprob_tie_tolerance_nats")
    if isinstance(tol, bool) or not is_num(tol) or tol < 0 or tol > 1.0:
        raise ValueError("logprob_tie_tolerance_nats must be a number in [0, 1] (nats); got %r" % (tol,))
    deviations = {k: {"default": base.get(k), "used": crit[k]} for k in sorted(crit)
                  if k not in ("_notes", "schema") and crit[k] != base.get(k)}
    return crit, deviations


# ---------------------------------------------------------------------------- artifacts
class BenchRun:
    """A tools/bench run directory (run_matrix.py or probes.py)."""

    def __init__(self, path, rel):
        self.path, self.rel = path, rel
        self.name = os.path.basename(os.path.normpath(path))
        self.errors = []
        self.meta, self.summary, self.raw, self.md = None, None, [], None
        self.kind = None
        self.cells = {}
        self.load_failed = False
        try:
            self.meta = load_json(os.path.join(path, "meta.json"))
            if not isinstance(self.meta, dict):
                raise ValueError("meta.json is not an object")
            self.kind = self.meta.get("scenario")
        except (OSError, ValueError) as e:
            self.errors.append("meta.json: %s" % e)
            self.meta, self.load_failed = None, True
        try:
            self.summary = load_json(os.path.join(path, "summary.json"))
            if not isinstance(self.summary, dict):
                raise ValueError("summary.json is not an object")
        except (OSError, ValueError) as e:
            self.errors.append("summary.json: %s" % e)
            self.summary, self.load_failed = None, True
        try:
            self.raw = runlog.load_raw(os.path.join(path, "raw.jsonl"))
        except (OSError, ValueError) as e:
            self.errors.append("raw.jsonl: %s" % e)
            self.load_failed = True
        mdp = os.path.join(path, "summary.md")
        if os.path.isfile(mdp):
            with open(mdp, encoding="utf-8", errors="replace") as f:
                self.md = f.read()
        if self.meta is not None and self.summary is not None and not self.errors:
            self._validate()

    @property
    def evidence(self):
        return [self.rel + "/" + n for n in ("meta.json", "summary.json", "raw.jsonl")]

    @property
    def status(self):
        return (self.meta or {}).get("status")

    def _validate(self):
        m, s = self.meta, self.summary
        if m.get("schema") != runlog.SCHEMA or s.get("schema") != runlog.SCHEMA:
            self.errors.append("schema is not %s (meta %r, summary %r)" % (runlog.SCHEMA, m.get("schema"), s.get("schema")))
            return
        if not is_int(m.get("raw_rows")):
            self.errors.append("meta.raw_rows missing (the run did not finish)")
        elif m["raw_rows"] != len(self.raw):
            self.errors.append("meta.raw_rows = %s but raw.jsonl has %d rows" % (m["raw_rows"], len(self.raw)))
        if self.errors:
            return
        try:
            if self.kind == "matrix":
                self._matrix()
            elif self.kind == "probes":
                self._probes()
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            self.errors.append("unexpected structure: %s: %s" % (type(e).__name__, e))

    # ------------------------------------------------------------------ mock
    def mock_reasons(self):
        out = []
        m, s = self.meta or {}, self.summary or {}
        if m.get("is_mock_server") is True:
            out.append("meta.is_mock_server = true")
        if s.get("is_mock_server") is True:
            out.append("summary.is_mock_server = true")
        if runlog.is_mock(m.get("models_response")):
            out.append("/v1/models owned_by = %s" % runlog.MOCK_OWNER)
        if self.md and MOCK_BANNER in self.md:
            out.append("summary.md carries the %r banner" % MOCK_BANNER)
        return out

    def mock_unknown(self):
        m = self.meta or {}
        return m.get("is_mock_server") is None and not ((m.get("models_response") or {}).get("json"))

    # ------------------------------------------------------------------ run_matrix.py
    def _matrix(self):
        lm = (self.meta.get("args") or {}).get("length_mode", "prompt")
        groups = {}
        for row in self.raw:
            if row.get("scenario") == "matrix" and row.get("phase") == "measure":
                k = (row["input"], row["output"], row["state"], row["concurrency"])
                groups.setdefault(k, []).append(row)
        scells = {(c["input"], c["output"], c["state"], c["concurrency"]): c for c in self.summary.get("cells", [])}
        if set(scells) != set(groups):
            self.errors.append("summary.json cells %s != raw.jsonl measure groups %s" % (sorted(scells), sorted(groups)))
            return
        for k, rows in groups.items():
            ms = [runlog.request_metrics(r["result"]) for r in rows]
            ok = [(r, m) for r, m in zip(rows, ms) if m["ok"]]
            short = 0
            for r, m in ok:
                ct = m["completion_tokens"] if m["completion_tokens"] is not None else m["n_chunks"]
                if r.get("requested_max_tokens") and ct < r["requested_max_tokens"]:
                    short += 1
            pts = [m["prompt_tokens"] for _, m in ok]
            c = {"run": self, "input": k[0], "output": k[1], "state": k[2], "concurrency": k[3],
                 "target_ctx": k[0] if lm == "total" else k[0] + k[1], "rows": rows, "metrics": ms,
                 "n": len(rows), "n_ok": len(ok), "n_failed": len(rows) - len(ok), "n_short": short,
                 "prompt_min": (min(pts) if pts and all(p is not None for p in pts) else None),
                 "decode": runlog.describe([m["decode_tok_s"] for _, m in ok]),
                 "ttft": runlog.describe([m["ttft_ms"] for _, m in ok]),
                 "cached": runlog.describe([m["cached_tokens"] for _, m in ok]),
                 "prompt": runlog.describe([m["prompt_tokens"] for _, m in ok])}
            sc = scells[k]
            sm = sc.get("metrics") or {}
            checks = [("n", sc.get("n"), c["n"]), ("n_ok", sc.get("n_ok"), c["n_ok"]),
                      ("n_failed", sc.get("n_failed"), c["n_failed"]),
                      ("n_short_output", sc.get("n_short_output"), c["n_short"])]
            for key, mine in (("decode_tok_s", c["decode"]), ("ttft_ms", c["ttft"]), ("cached_tokens", c["cached"]),
                              ("prompt_tokens", c["prompt"])):
                theirs = sm.get(key) or {}
                for stat in ("n", "p50", "min"):
                    checks.append(("%s.%s" % (key, stat), theirs.get(stat), mine.get(stat)))
            bad = [(name, a, b) for name, a, b in checks if not (a == b or (is_num(a) and is_num(b) and close(a, b)))]
            if bad:
                self.errors.append("summary.json disagrees with raw.jsonl at cell input=%s output=%s %s conc=%s: %s" % (
                    k[0], k[1], k[2], k[3], "; ".join("%s summary=%s raw=%s" % x for x in bad)))
            c["summary_decode_p50"] = (sm.get("decode_tok_s") or {}).get("p50")
            self.cells[k] = c
        turns = ((self.summary.get("agent_loop") or {}).get("turns")) or []
        araw = [r for r in self.raw if r.get("phase") == "agent_turn"]
        if len(turns) != len(araw):
            self.errors.append("agent_loop.turns has %d entries but raw.jsonl has %d agent_turn rows" % (len(turns), len(araw)))
        else:
            for t, r in zip(turns, araw):
                m = runlog.request_metrics(r["result"])
                if bool(t.get("ok")) != m["ok"] or t.get("prompt_tokens") != m["prompt_tokens"]:
                    self.errors.append("agent turn %s: summary (ok=%s, prompt=%s) != raw (ok=%s, prompt=%s)" % (
                        t.get("turn"), t.get("ok"), t.get("prompt_tokens"), m["ok"], m["prompt_tokens"]))
                    break

    # ------------------------------------------------------------------ probes.py
    def _probes(self):
        by_seq = {r.get("seq"): r for r in self.raw}
        pr = [r for r in self.raw if r.get("phase") == "probe_result"]
        results = self.summary.get("results")
        if not isinstance(results, list):
            raise ValueError("summary.results is not a list")
        if len(pr) != len(results):
            self.errors.append("summary.json has %d results but raw.jsonl has %d probe_result rows" % (len(results), len(pr)))
            return
        for res, row in zip(results, pr):
            # "pass" is not compared: probes.py overwrites res["pass"] for json/determinism after the probe_result row
            # was logged (tools/bench/probes.py:358-360,429-431@55db75d); the pack re-judges scores itself anyway
            for key in ("probe", "context", "score", "error", "prompt_tokens"):
                if res.get(key) != row.get(key) and not (is_num(res.get(key)) and is_num(row.get(key))
                                                         and close(res.get(key), row.get(key))):
                    self.errors.append("probe %s@%s: summary %s=%r != raw %r" % (
                        res.get("probe"), res.get("context"), key, res.get(key), row.get(key)))
            answers = []
            for seq in res.get("raw_seqs") or []:
                if seq not in by_seq:
                    self.errors.append("probe %s@%s: raw seq %s missing" % (res.get("probe"), res.get("context"), seq))
                    continue
                answers.append(by_seq[seq])
            res["_answers"] = answers
            if answers and res.get("prompt_tokens") is not None:
                m = runlog.request_metrics(answers[0]["result"])
                if m["prompt_tokens"] != res["prompt_tokens"]:
                    self.errors.append("probe %s@%s: prompt_tokens summary %s != raw usage %s" % (
                        res.get("probe"), res.get("context"), res["prompt_tokens"], m["prompt_tokens"]))


class JsonArtifact:
    def __init__(self, kind, path, rel, data):
        self.kind, self.path, self.rel, self.data = kind, path, rel, data


def classify_json(data):
    if isinstance(data, dict):
        keys = set(data)
        if {"classes", "findings", "verdict", "min_expert_bpw", "min_dense_bpw"} <= keys:
            return "quant_audit"
        if {"ref", "cand", "top_k", "rows", "summary", "errors"} <= keys:
            return "logprob"
        if {"url", "prompt_file", "gen", "runs", "errors", "summary"} <= keys:
            return "slot_persist"
        if {"facts", "red_flags", "synthetic"} <= keys:
            return "hw_facts"
        if {"repo", "revision", "files", "total_bytes"} <= keys:
            return "weights_manifest"
        return None
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data) and all(
            {"run", "mock", "measured_decode_p50", "bound", "ratio"} <= set(r) for r in data):
        return "roofline"
    return None


WV_SUMMARY = re.compile(r"^verify: (\d+)/(\d+) files OK$")
WV_OK = re.compile(r"^OK {7}(.+) (\d+) sha256 ([0-9a-f]{64})$")
WV_OK_SIZE = re.compile(r"^OK-SIZE  (.+) (\d+) \(no sha256 in manifest: not an LFS file\)$")
WV_BAD = re.compile(r"^(MISSING|SIZE|SHA256) +(.+)$")


def parse_weights_verify(path, max_bytes=64 << 20):
    """stdout of `scripts/offline/weights.py verify` (scripts/offline/weights.py:78-101@55db75d), or None."""
    if os.path.getsize(path) > max_bytes:
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f]
    summ = [WV_SUMMARY.match(l) for l in lines if WV_SUMMARY.match(l)]
    if len(summ) != 1:
        return None
    out = {"ok_n": int(summ[0].group(1)), "total": int(summ[0].group(2)), "ok": {}, "ok_size": {}, "bad": [], "other": []}
    for l in lines:
        m = WV_OK.match(l)
        if m:
            out["ok"][m.group(1)] = {"size": int(m.group(2)), "sha256": m.group(3)}
            continue
        m = WV_OK_SIZE.match(l)
        if m:
            out["ok_size"][m.group(1)] = {"size": int(m.group(2))}
            continue
        m = WV_BAD.match(l)
        if m:
            out["bad"].append(l)
            continue
        if l.strip() and not WV_SUMMARY.match(l):
            out["other"].append(l)
    return out


class Inputs:
    def __init__(self, root):
        self.root = root
        self.matrix, self.probes, self.other_runs = [], [], []
        self.art = {k: [] for k in ("quant_audit", "logprob", "slot_persist", "roofline", "hw_facts",
                                    "weights_manifest", "weights_verify")}
        self.unparseable = []     # (rel, error)
        self.unclassified = []    # rel
        self.files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            reld = os.path.relpath(dirpath, root).replace(os.sep, "/")
            reld = "" if reld == "." else reld
            consumed = set()
            if "meta.json" in filenames and ("raw.jsonl" in filenames or "summary.json" in filenames):
                run = BenchRun(dirpath, (INPUTS + "/" + reld).rstrip("/"))
                consumed = {"meta.json", "raw.jsonl", "summary.json", "summary.md", "metrics_before.txt",
                            "metrics_after.txt"}
                if run.kind == "matrix":
                    self.matrix.append(run)
                elif run.kind == "probes":
                    self.probes.append(run)
                elif run.meta is None:
                    self.unparseable.append((run.rel, "; ".join(run.errors)))
                else:
                    self.other_runs.append(run)
                    self.unclassified.append(run.rel + " (bench run, scenario %r: not judged)" % run.kind)
            for fn in sorted(filenames):
                rel = INPUTS + "/" + (reld + "/" if reld else "") + fn
                self.files.append(rel)
                if fn in consumed:
                    continue
                if fn.lower().endswith(".json"):
                    try:
                        data = load_json(os.path.join(dirpath, fn))
                    except (OSError, ValueError) as e:
                        self.unparseable.append((rel, "%s: %s" % (type(e).__name__, e)))
                        continue
                    kind = classify_json(data)
                    if kind:
                        self.art[kind].append(JsonArtifact(kind, os.path.join(dirpath, fn), rel, data))
                    else:
                        self.unclassified.append(rel)
                    continue
                wv = parse_weights_verify(os.path.join(dirpath, fn)) if fn.lower().endswith((".txt", ".log")) else None
                if wv is not None:
                    self.art["weights_verify"].append(JsonArtifact("weights_verify", os.path.join(dirpath, fn), rel, wv))
                else:
                    self.unclassified.append(rel)
        for run in self.matrix + self.probes:
            if run.errors:
                self.unparseable.append((run.rel, "; ".join(run.errors)))

    def mock_urls(self):
        out = set()
        for run in self.matrix + self.probes:
            if run.mock_reasons():
                out.add(norm_url((run.meta or {}).get("server_url")))
        out.discard("")
        return out


# ---------------------------------------------------------------------------- results
def res(status, code, reason, measured=None, value=None, threshold=None, evidence=(), details=None):
    return {"status": status, "reason_code": code, "reason": reason, "measured": measured, "value": value,
            "threshold": threshold, "evidence": list(evidence), "details": details or {}}


def bench_gate(runs, kind, tool):
    """None when the runs are usable; otherwise a NOT RUN result. Order: missing, unparseable, mock, mock unknown, fatal."""
    if not runs:
        return res(NOT_RUN, "missing", "no %s run directory under --runs (produce it with %s)" % (kind, tool))
    ev = [p for r in runs for p in r.evidence]
    broken = [r for r in runs if r.errors]
    if broken:
        return res(NOT_RUN, "unparseable" if any(r.load_failed for r in broken) else "inconsistent",
                   "; ".join("%s: %s" % (r.rel, "; ".join(r.errors)) for r in broken), evidence=ev)
    mock = [(r, r.mock_reasons()) for r in runs]
    mock = [(r, why) for r, why in mock if why]
    if mock:
        return res(NOT_RUN, "mock", "mock: " + "; ".join("%s (%s)" % (r.rel, ", ".join(why)) for r, why in mock),
                   evidence=ev)
    unk = [r for r in runs if r.mock_unknown()]
    if unk:
        return res(NOT_RUN, "mock_unknown", "cannot tell whether the server was the mock (no is_mock_server, no "
                   "/v1/models snapshot): " + ", ".join(r.rel for r in unk), evidence=ev)
    bad = [r for r in runs if r.status not in ("ok", "completed_with_failures", "interrupted")]
    if bad:
        return res(NOT_RUN, "unparseable", "run status not usable: " + ", ".join(
            "%s status=%s%s" % (r.rel, r.status, (" fatal=" + str(r.meta.get("fatal"))) if r.meta.get("fatal") else "")
            for r in bad), evidence=ev)
    return None


def multi_gate(*groups):
    """Gate over several run kinds that are present; None if all usable and at least one present."""
    present = [(runs, kind, tool) for runs, kind, tool in groups if runs]
    if not present:
        return res(NOT_RUN, "missing", "none of: " + ", ".join("%s (%s)" % (k, t) for _, k, t in groups))
    for runs, kind, tool in present:
        g = bench_gate(runs, kind, tool)
        if g:
            return g
    return None


def cell_tag(c):
    return "%s input=%s output=%s %s" % (c["run"].rel, c["input"], c["output"], c["state"])


# ---------------------------------------------------------------------------- criteria: bench
def crit_single_configuration(inp, crit):
    runs = inp.matrix + inp.probes
    g = multi_gate((inp.matrix, "run_matrix", "tools/bench/run_matrix.py"), (inp.probes, "probes", "tools/bench/probes.py"))
    if g:
        return g
    ev = [r.rel + "/meta.json" for r in runs]
    empty = [r.rel for r in runs if not (r.meta.get("engine_desc") or "").strip() or not (r.meta.get("model_desc") or "").strip()]
    vals = {k: sorted({str(r.meta.get(k)) for r in runs}) for k in ("model", "engine_desc", "model_desc")}
    details = {"values": vals}
    names = [os.path.basename(a.data.get("files", [""])[0]) for a in inp.art["quant_audit"] if a.data.get("files")]
    text = " ".join(vals["model_desc"] + vals["engine_desc"])
    unnamed = [n for n in names if n and n not in text]
    if unnamed:
        details["warning"] = ("quant audit file(s) %s not named in --model-desc/--engine-desc; the pack cannot link the "
                              "audit to the served model" % ", ".join(unnamed))
    diff = [k for k, v in vals.items() if len(v) > 1]
    if diff:
        return res(FAIL, "mismatch", "runs differ in " + "; ".join("%s: %s" % (k, " | ".join(vals[k])) for k in diff),
                   evidence=ev, details=details)
    if empty:
        return res(NOT_RUN, "insufficient", "--engine-desc / --model-desc not given in: " + ", ".join(empty) +
                   " (the configuration cannot be identified)", evidence=ev, details=details)
    return res(PASS, "ok", "%d run(s), one configuration" % len(runs),
               measured="model=%s; engine=%s; model_desc=%s" % (vals["model"][0], vals["engine_desc"][0], vals["model_desc"][0]),
               evidence=ev, details=details)


def crit_openai_api(inp, crit):
    runs = inp.matrix + inp.probes
    g = multi_gate((inp.matrix, "run_matrix", "tools/bench/run_matrix.py"), (inp.probes, "probes", "tools/bench/probes.py"))
    if g:
        return g
    ev = [p for r in runs for p in r.evidence]
    fails, n_ok, n_stream = [], 0, 0
    for r in runs:
        st = (r.meta.get("models_response") or {}).get("status")
        if st != 200:
            fails.append("%s: GET /v1/models status %s" % (r.rel, st))
        for row in r.raw:
            d = row.get("result")
            if not isinstance(d, dict) or row.get("scenario") == "setup":
                continue
            m = runlog.request_metrics(d)
            if not m["ok"]:
                continue
            n_ok += 1
            if d.get("events"):
                n_stream += 1
            if m["prompt_tokens"] is None or m["completion_tokens"] is None:
                fails.append("%s seq %s: usage.prompt_tokens/completion_tokens not reported" % (r.rel, row.get("seq")))
    if fails:
        return res(FAIL, "failed", "; ".join(fails[:5]) + (" (+%d more)" % (len(fails) - 5) if len(fails) > 5 else ""),
                   evidence=ev)
    if n_stream == 0:
        return res(FAIL, "failed", "no successful streamed generation request", evidence=ev)
    return res(PASS, "ok", "/v1/models 200; streamed requests with usage", measured="%d ok requests, %d streamed, usage in all" % (
        n_ok, n_stream), value=n_stream, evidence=ev)


def cell_context(c, crit):
    """(target, measured) context of a matrix cell under crit["decode_filled_context_basis"]:
    prompt_plus_output (default; EXPERIMENTS.md:19, BENCH_PROTOCOL.md:128 --length-mode total): the window is filled
    when prompt + forced output reaches ctx_tokens; prompt: the prompt alone must reach ctx_tokens."""
    if crit["decode_filled_context_basis"] == "prompt":
        target = c["target_ctx"] - c["output"]
        measured = c["prompt_min"]
    else:
        target = c["target_ctx"]
        measured = None if c["prompt_min"] is None else c["prompt_min"] + c["output"]
    return target, measured


def measured_filled(c, crit):
    m = cell_context(c, crit)[1]
    return m is not None and m >= crit["ctx_tokens"] - crit["ctx_slack_tokens"]


def filled_target_cells(runs, crit, min_output=0):
    cells, short = [], []
    for r in runs:
        for c in r.cells.values():
            if c["concurrency"] != 1:
                continue
            if cell_context(c, crit)[0] >= crit["ctx_tokens"] and c["output"] >= min_output:
                cells.append(c)
            else:
                short.append(c)
    return cells, short


def crit_decode(inp, crit, thr_key):
    g = bench_gate(inp.matrix, "run_matrix", "tools/bench/run_matrix.py")
    if g:
        return g
    ctx, slack, min_out = crit["ctx_tokens"], crit["ctx_slack_tokens"], crit["decode_min_output_tokens"]
    thr = float(crit[thr_key])
    ev = [p for r in inp.matrix for p in r.evidence]
    cells, short = filled_target_cells(inp.matrix, crit, min_out)
    basis = crit["decode_filled_context_basis"]
    info = {"basis": basis, "short_context_cells_not_counted": [
        {"cell": cell_tag(c), "target_context": cell_context(c, crit)[0], "decode_p50": c["decode"].get("p50")}
        for c in short]}
    if not cells:
        return res(NOT_RUN, "not_filled", "no single-stream cell at filled context (%s >= %d, output >= %d); "
                   "cells present: %s" % (basis, ctx, min_out, ", ".join(
                       "%s (%s %s)" % (cell_tag(c), basis, cell_context(c, crit)[0]) for c in short) or "none"),
                   threshold=thr, evidence=ev, details=info)
    fail, nr, vals = [], [], []
    for c in cells:
        t = cell_tag(c)
        if c["n_failed"]:
            fail.append("%s: %d of %d requests failed" % (t, c["n_failed"], c["n"]))
        if c["n_short"]:
            fail.append("%s: %d short outputs (decode length not forced)" % (t, c["n_short"]))
        if c["n_ok"] < crit["min_repeats"]:
            nr.append("%s: %d ok repeats < %d" % (t, c["n_ok"], crit["min_repeats"]))
        measured = cell_context(c, crit)[1]
        if measured is None:
            nr.append("%s: usage.prompt_tokens not reported" % t)
        elif measured < ctx - slack:
            nr.append("%s: measured %s %d < %d (prompt not filled)" % (t, crit["decode_filled_context_basis"], measured,
                                                                       ctx - slack))
        elif c["decode"].get("p50") is not None:
            vals.append((c["decode"]["p50"], t))
        else:
            nr.append("%s: decode not measurable" % t)
    info["cells"] = [{"cell": cell_tag(c), "decode_p50": c["decode"].get("p50"), "decode_min": c["decode"].get("min"),
                      "prompt_min": c["prompt_min"], "n_ok": c["n_ok"]} for c in cells]
    below = [(v, t) for v, t in vals if v < thr]
    worst = min(vals) if vals else None
    meas = "worst cell p50 %s tok/s (%s)" % (fmt(worst[0]), worst[1]) if worst else None
    if below:
        return res(FAIL, "threshold", "; ".join("%s: decode p50 %.2f < %.1f tok/s" % (t, v, thr) for v, t in below),
                   measured=meas, value=worst[0], threshold=thr, evidence=ev, details=info)
    if fail:
        return res(FAIL, "failed", "; ".join(fail), measured=meas, value=worst[0] if worst else None, threshold=thr,
                   evidence=ev, details=info)
    if nr:
        code = "not_filled" if any("not filled" in x for x in nr) else "insufficient"
        return res(NOT_RUN, code, "; ".join(nr), measured=meas, value=worst[0] if worst else None, threshold=thr,
                   evidence=ev, details=info)
    return res(PASS, "ok", "%d filled-context cell(s), every decode p50 >= %.1f tok/s" % (len(vals), thr),
               measured=meas, value=worst[0], threshold=thr, evidence=ev, details=info)


def crit_prefix_cache(inp, crit):
    g = bench_gate(inp.matrix, "run_matrix", "tools/bench/run_matrix.py")
    if g:
        return g
    thr = float(crit["prefix_cache_ttft_ratio_max"])
    ev = [p for r in inp.matrix for p in r.evidence]
    cells, _ = filled_target_cells(inp.matrix, crit)
    pairs = {}
    for c in cells:
        pairs.setdefault((c["run"].rel, c["input"], c["output"]), {})[c["state"]] = c
    pairs = {k: v for k, v in pairs.items() if "cold" in v and "warm" in v}
    if not pairs:
        return res(NOT_RUN, "not_filled", "no cold+warm single-stream pair at filled context (target context >= %d)"
                   % crit["ctx_tokens"], threshold=thr, evidence=ev)
    fail, nr, ratios, det = [], [], [], []
    for (rel, i, o), d in sorted(pairs.items()):
        cold, warm = d["cold"], d["warm"]
        t = "%s input=%s output=%s" % (rel, i, o)
        if cold["n_failed"] or warm["n_failed"]:
            fail.append("%s: failed requests (cold %d, warm %d)" % (t, cold["n_failed"], warm["n_failed"]))
        if not (measured_filled(cold, crit) and measured_filled(warm, crit)):
            nr.append("%s: measured context below %d (prompt not filled)" % (t, crit["ctx_tokens"] - crit["ctx_slack_tokens"]))
            continue
        tc, tw = cold["ttft"].get("p50"), warm["ttft"].get("p50")
        if not tc or tw is None:
            nr.append("%s: TTFT not measured" % t)
            continue
        ratio = tw / tc
        ratios.append((ratio, t))
        det.append({"pair": t, "cold_ttft_ms_p50": tc, "warm_ttft_ms_p50": tw, "ratio": ratio,
                    "warm_cached_tokens_p50": warm["cached"].get("p50"), "warm_cached_tokens_n": warm["cached"].get("n")})
        if not warm["cached"].get("n"):
            fail.append("%s: server does not report cached tokens (usage.prompt_tokens_details / timings.cache_n)" % t)
        elif not warm["cached"].get("p50"):
            fail.append("%s: warm cached tokens p50 = 0" % t)
        if ratio > thr:
            fail.append("%s: warm/cold TTFT p50 = %.3f > %.2f" % (t, ratio, thr))
    worst = max(ratios) if ratios else None
    meas = "worst warm/cold TTFT p50 = %.4f (%s)" % worst if worst else None
    if fail:
        return res(FAIL, "threshold" if any(">" in x for x in fail) else "failed", "; ".join(fail), measured=meas,
                   value=worst[0] if worst else None, threshold=thr, evidence=ev, details={"pairs": det})
    if nr:
        return res(NOT_RUN, "not_filled" if any("not filled" in x for x in nr) else "insufficient", "; ".join(nr),
                   measured=meas, threshold=thr, evidence=ev, details={"pairs": det})
    return res(PASS, "ok", "%d pair(s) at filled context, cached tokens reported" % len(ratios), measured=meas,
               value=worst[0], threshold=thr, evidence=ev, details={"pairs": det})


def crit_agent_loop(inp, crit):
    g = bench_gate(inp.matrix, "run_matrix", "tools/bench/run_matrix.py")
    if g:
        return g
    need = int(crit["agent_loop_min_prompt_tokens"])
    ev = [p for r in inp.matrix for p in r.evidence]
    loops = [(r, r.summary["agent_loop"]["turns"]) for r in inp.matrix
             if ((r.summary.get("agent_loop") or {}).get("turns"))]
    if not loops:
        return res(NOT_RUN, "missing", "no agent loop in the matrix runs (run_matrix.py --scenarios agent)",
                   threshold=need, evidence=ev)
    fail, best, det = [], None, []
    for r, turns in loops:
        bad = [t for t in turns if not t.get("ok")]
        reached = max([t.get("prompt_tokens") or 0 for t in turns if t.get("ok")] or [0])
        det.append({"run": r.rel, "turns": len(turns), "failed_turns": len(bad), "max_prompt_tokens": reached,
                    "ttft_ms_last_ok": next((t.get("ttft_ms") for t in reversed(turns) if t.get("ok")), None)})
        if bad:
            fail.append("%s: %d of %d turns failed (first: turn %s)" % (r.rel, len(bad), len(turns), bad[0].get("turn")))
        best = reached if best is None else max(best, reached)
    meas = "max prompt tokens reached %s" % best
    if fail:
        return res(FAIL, "failed", "; ".join(fail), measured=meas, value=best, threshold=need, evidence=ev,
                   details={"loops": det})
    if best < need:
        return res(NOT_RUN, "insufficient", "agent loop reached %d prompt tokens < %d (raise --agent-target-tokens)" % (
            best, need), measured=meas, value=best, threshold=need, evidence=ev, details={"loops": det})
    return res(PASS, "ok", "no failed turns, reached %d >= %d prompt tokens" % (best, need), measured=meas, value=best,
               threshold=need, evidence=ev, details={"loops": det})


def probe_results(runs, names):
    out = []
    for r in runs:
        for x in r.summary.get("results", []):
            if x.get("probe") in names:
                out.append((r, x))
    return out


def crit_context_probes(inp, crit):
    g = bench_gate(inp.probes, "probes", "tools/bench/probes.py")
    if g:
        return g
    ctx, slack = crit["ctx_tokens"], crit["ctx_slack_tokens"]
    req = list(crit["long_probes_required"])
    thr = crit["probe_thresholds"]
    ev = [p for r in inp.probes for p in r.evidence]
    qual = {p: [] for p in req}
    for r, x in probe_results(inp.probes, req):
        if is_int(x.get("context")) and x["context"] >= ctx:
            qual[x["probe"]].append((r, x))
    fail, nr, scores, errs = [], [], {}, []
    missing = [p for p in req if not qual[p]]
    if missing:
        nr.append("not run at context >= %d: %s" % (ctx, ", ".join(missing)))
    for p in req:
        for r, x in qual[p]:
            t = "%s %s@%s" % (r.rel, p, x["context"])
            at = (r.meta.get("args") or {}).get("answer_tokens")
            if x.get("error"):
                fail.append("%s: error %s" % (t, str(x["error"])[:200]))
                errs.append(fail[-1])
                continue
            if not is_int(at) or at > crit["probe_answer_tokens_max"]:
                nr.append("%s: --answer-tokens %s not in [0, %d]" % (t, at, crit["probe_answer_tokens_max"]))
                continue
            need = ctx - at - slack
            pt = x.get("prompt_tokens")
            if pt is None:
                nr.append("%s: usage.prompt_tokens not reported" % t)
                continue
            if pt < need:
                nr.append("%s: prompt_tokens %d < %d (prompt not filled or truncated)" % (t, pt, need))
                continue
            sc = x.get("score")
            if sc is None:
                nr.append("%s: no score" % t)
                continue
            scores[p] = min(scores.get(p, sc), sc)
            if sc < thr[p]:
                fail.append("%s: score %.3f < %.2f" % (t, sc, thr[p]))
    meas = ", ".join("%s %.3f" % (p, scores[p]) for p in req if p in scores) or None
    th = {p: thr[p] for p in req}
    if fail:
        return res(FAIL, "threshold" if any("score" in x for x in fail) else "failed", "; ".join(fail), measured=meas,
                   threshold=th, evidence=ev, details={"scores_min": scores, "request_errors": errs})
    if nr:
        return res(NOT_RUN, "not_filled" if any("not filled" in x or "not run at" in x for x in nr) else "insufficient",
                   "; ".join(nr), measured=meas, threshold=th, evidence=ev, details={"scores_min": scores})
    return res(PASS, "ok", "all of %s pass at context >= %d with the prompt filled" % (", ".join(req), ctx),
               measured=meas, threshold=th, evidence=ev, details={"scores_min": scores})


def crit_output_after_full_prompt(inp, crit):
    g = multi_gate((inp.probes, "probes", "tools/bench/probes.py"), (inp.matrix, "run_matrix", "tools/bench/run_matrix.py"))
    if g:
        return g
    ctx, slack = crit["ctx_tokens"], crit["ctx_slack_tokens"]
    ev = [p for r in inp.probes + inp.matrix for p in r.evidence]
    checked, fail, skipped = 0, [], []
    for r, x in probe_results(inp.probes, set(crit["long_probes_required"])):
        if not (is_int(x.get("context")) and x["context"] >= ctx):
            continue
        at = (r.meta.get("args") or {}).get("answer_tokens")
        at = at if is_int(at) and at <= crit["probe_answer_tokens_max"] else 0
        for row in x.get("_answers") or []:
            m = runlog.request_metrics(row["result"])
            t = "%s %s@%s seq %s" % (r.rel, x["probe"], x["context"], row.get("seq"))
            if not m["ok"]:
                fail.append("%s: request failed: %s" % (t, str(row["result"].get("error"))[:160]))
                continue
            if m["prompt_tokens"] is None or m["prompt_tokens"] < ctx - at - slack:
                skipped.append(t)
                continue
            checked += 1
            if not _has_output(row["result"], m):
                fail.append("%s: no output after a %d-token prompt" % (t, m["prompt_tokens"]))
    cells, _ = filled_target_cells(inp.matrix, crit)
    for c in cells:
        for row, m in zip(c["rows"], c["metrics"]):
            t = "%s seq %s" % (cell_tag(c), row.get("seq"))
            if not m["ok"]:
                fail.append("%s: request failed: %s" % (t, str(row["result"].get("error"))[:160]))
                continue
            occ = None if m["prompt_tokens"] is None else (
                m["prompt_tokens"] if crit["decode_filled_context_basis"] == "prompt" else m["prompt_tokens"] + c["output"])
            if occ is None or occ < ctx - slack:
                skipped.append(t)
                continue
            checked += 1
            if not _has_output(row["result"], m):
                fail.append("%s: no output after a %d-token prompt" % (t, m["prompt_tokens"]))
    det = {"requests_checked": checked, "not_filled_skipped": skipped[:20]}
    if fail:
        return res(FAIL, "failed", "; ".join(fail[:5]), measured="%d checked, %d without output or failed" % (checked, len(fail)),
                   evidence=ev, details=det)
    if not checked:
        return res(NOT_RUN, "not_filled", "no request with a prompt filled to the context (>= %d tokens incl. output budget)"
                   % (ctx - slack), evidence=ev, details=det)
    return res(PASS, "ok", "every filled-context request produced output", measured="%d of %d requests produced output" % (
        checked, checked), value=checked, evidence=ev, details=det)


def _has_output(d, m):
    n = m["completion_tokens"] if m["completion_tokens"] is not None else m["n_chunks"]
    chars = (d.get("content_chars") or 0) + (d.get("reasoning_chars") or 0) + len(d.get("tool_calls") or [])
    return bool(n and n >= 1 and chars > 0)


def crit_agent_api_probes(inp, crit):
    g = bench_gate(inp.probes, "probes", "tools/bench/probes.py")
    if g:
        return g
    req = list(crit["agent_probes_required"])
    thr = crit["probe_thresholds"]
    ev = [p for r in inp.probes for p in r.evidence]
    found = {p: [] for p in req}
    for r, x in probe_results(inp.probes, req):
        found[x["probe"]].append((r, x))
    fail, scores, errs = [], {}, []
    for p in req:
        for r, x in found[p]:
            t = "%s %s" % (r.rel, p)
            sc = x.get("score")
            if x.get("error"):
                fail.append("%s: error %s" % (t, str(x["error"])[:200]))
                errs.append(fail[-1])
            elif sc is None or sc < thr[p]:
                fail.append("%s: score %s < %.2f" % (t, fmt(sc, 3), thr[p]))
            elif p == "json" and not (x.get("details") or {}).get("all_attempted_valid"):
                fail.append("%s: not every attempted JSON answer is valid" % t)
            if p == "json" and (x.get("details") or {}).get("response_format_accepted") is False:
                # N24 audit M2: a rejected response_format (probes.py: HTTP 400/422) is an API defect whatever the score
                fail.append("%s: the server rejected response_format json_object (API defect)" % t)
                errs.append(fail[-1])
            if sc is not None:
                scores[p] = min(scores.get(p, sc), sc)
    meas = ", ".join("%s %.3f" % (p, scores[p]) for p in req if p in scores) or None
    th = {p: thr[p] for p in req}
    if fail:
        return res(FAIL, "threshold", "; ".join(fail), measured=meas, threshold=th, evidence=ev,
                   details={"request_errors": errs})
    missing = [p for p in req if not found[p]]
    if missing:
        return res(NOT_RUN, "missing", "probe(s) not run: " + ", ".join(missing), measured=meas, threshold=th, evidence=ev)
    return res(PASS, "ok", "all of %s pass" % ", ".join(req), measured=meas, threshold=th, evidence=ev)


def crit_determinism(inp, crit):
    g = bench_gate(inp.probes, "probes", "tools/bench/probes.py")
    if g:
        return g
    ev = [p for r in inp.probes for p in r.evidence]
    xs = probe_results(inp.probes, {"determinism"})
    if not xs:
        return res(NOT_RUN, "missing", "determinism probe not run", evidence=ev)
    bad = [(r, x) for r, x in xs if x.get("error") or not (x.get("details") or {}).get("identical")]
    if bad:
        return res(FAIL, "threshold", "; ".join("%s: identical=%s first_diff_char=%s error=%s" % (
            r.rel, (x.get("details") or {}).get("identical"), (x.get("details") or {}).get("first_diff_char"),
            x.get("error")) for r, x in bad), measured="%d of %d identical" % (len(xs) - len(bad), len(xs)), evidence=ev)
    return res(PASS, "ok", "identical output twice at temperature 0", measured="%d of %d identical" % (len(xs), len(xs)),
               evidence=ev)


# ---------------------------------------------------------------------------- criteria: standalone JSON artifacts
def crit_quant(inp, crit):
    arts = inp.art["quant_audit"]
    thr = float(crit["min_bpw"])
    if not arts:
        return res(NOT_RUN, "missing", "no tools/gguf_quant_audit.py --json output under --runs", threshold=thr)
    ev = [a.rel for a in arts]
    fail, nr, det = [], [], []
    for a in arts:
        d = a.data
        classes = d.get("classes")
        if not isinstance(classes, dict) or not isinstance(d.get("findings"), list):
            nr.append("%s: classes/findings malformed" % a.rel)
            continue
        low, unknown_t, minbpw = [], [], None
        for cname, st in sorted(classes.items()):
            for tname in sorted((st or {}).get("types") or {}):
                bpw = TYPE_BPW.get(tname)
                if bpw is None:
                    unknown_t.append("%s:%s" % (cname, tname))
                    continue
                minbpw = bpw if minbpw is None else min(minbpw, bpw)
                if bpw < thr:
                    low.append("%s %s (%.3f bpw)" % (cname, tname, bpw))
        lvl = [f for f in d["findings"] if f.get("level") in ("FAIL", "WARN")]
        exp = classes.get("routed_experts") or {}
        det.append({"file": a.rel, "gguf": (d.get("files") or [None])[0], "architecture": d.get("architecture"),
                    "tool_verdict": d.get("verdict"), "routed_experts_bpw": exp.get("bpw"),
                    "min_type_bpw": minbpw, "findings": len(d["findings"]), "low_types": low})
        if low:
            fail.append("%s: tensor types below %.1f bpw in the class histogram: %s" % (a.rel, thr, ", ".join(low)))
        if unknown_t or d.get("unknown_types"):
            fail.append("%s: unknown ggml types %s" % (a.rel, unknown_t or d.get("unknown_types")))
        if lvl:
            fail.append("%s: %d finding(s) below the audit floor, e.g. %s %s %s bpw" % (
                a.rel, len(lvl), lvl[0].get("tensor"), lvl[0].get("type"), lvl[0].get("bpw")))
        if d.get("verdict") != "PASS" and not (low or lvl or unknown_t or d.get("unknown_types")):
            fail.append("%s: tool verdict %s" % (a.rel, d.get("verdict")))
        if not (is_num(d.get("min_expert_bpw")) and is_num(d.get("min_dense_bpw"))) or \
                d["min_expert_bpw"] < thr or d["min_dense_bpw"] < thr:
            nr.append("%s: audit ran with --min-expert-bpw %s / --min-dense-bpw %s, weaker than %.1f; rerun with the "
                      "defaults" % (a.rel, d.get("min_expert_bpw"), d.get("min_dense_bpw"), thr))
    meas = "; ".join("%s: experts %s bpw, lowest type %s bpw" % (x["gguf"], fmt(x["routed_experts_bpw"], 3),
                                                                fmt(x["min_type_bpw"], 3)) for x in det) or None
    vals = [x["min_type_bpw"] for x in det if x["min_type_bpw"] is not None]
    value = min(vals) if vals else None
    if fail:
        return res(FAIL, "threshold", "; ".join(fail), measured=meas, value=value, threshold=thr, evidence=ev,
                   details={"audits": det})
    if nr:
        return res(NOT_RUN, "insufficient", "; ".join(nr), measured=meas, value=value, threshold=thr, evidence=ev,
                   details={"audits": det})
    return res(PASS, "ok", "every tensor type >= %.1f bpw, no findings" % thr, measured=meas, value=value,
               threshold=thr, evidence=ev, details={"audits": det})


def near_tie(row, tol):
    """A top-1 disagreement that numerical noise alone can explain (N24, V6 tie tolerance): both greedy tokens are in
    both top-K lists and ref_gap + cand_gap <= tol, where ref_gap = lp_ref(r1) - lp_ref(c1) and cand_gap =
    lp_cand(c1) - lp_cand(r1) (tools/quality/logprob_compare.py compare()). Their sum is the change of the logit
    difference l(r1) - l(c1) between the engines, at most 2 * max_t |lp_ref(t) - lp_cand(t)|. Rows without the gaps
    (files written before N24, or a token outside the other top-K) are never excused."""
    if row.get("top1_agree") or tol <= 0:
        return False
    g, h = row.get("ref_gap"), row.get("cand_gap")
    if not (is_num(g) and is_num(h)) or isinstance(g, bool) or isinstance(h, bool):
        return False
    return g >= 0 and h >= 0 and g + h <= tol


def crit_logprob(inp, crit):
    arts = inp.art["logprob"]
    thr = float(crit["top1_agreement_min"])
    tol = float(crit["logprob_tie_tolerance_nats"])
    if not arts:
        return res(NOT_RUN, "missing", "no tools/quality/logprob_compare.py --out file under --runs", threshold=thr)
    ev = [a.rel for a in arts]
    mock = inp.mock_urls()
    fail, nr, det, worst = [], [], [], None
    for a in arts:
        d = a.data
        ref, cand = norm_url(d.get("ref")), norm_url(d.get("cand"))
        if ref in mock or cand in mock:
            return res(NOT_RUN, "mock", "mock: %s compares against %s, the URL of a mock bench run" % (
                a.rel, d.get("ref") if ref in mock else d.get("cand")), threshold=thr, evidence=ev)
        if ref and ref == cand:
            nr.append("%s: ref and cand are the same endpoint (%s)" % (a.rel, d.get("ref")))
            continue
        rows, errs, s = d.get("rows") or [], d.get("errors") or [], d.get("summary")
        if len(errs) > crit["logprob_max_errors"]:
            nr.append("%s: %d prefix(es) failed (e.g. %s); the comparison is incomplete" % (
                a.rel, len(errs), str(errs[0])[:160]))
            continue
        if not isinstance(s, dict) or not rows:
            nr.append("%s: no successful prefixes" % a.rel)
            continue
        n = len(rows)
        top1 = sum(1 for r in rows if r.get("top1_agree")) / float(n)
        if s.get("n") != n or not close(s.get("top1_agree"), top1):
            nr.append("%s: summary (n=%s, top1=%s) != rows (n=%d, top1=%.6f)" % (a.rel, s.get("n"), s.get("top1_agree"), n, top1))
            continue
        if n < crit["logprob_min_prefixes"]:
            nr.append("%s: %d prefixes < %d" % (a.rel, n, crit["logprob_min_prefixes"]))
            continue
        ties = [r.get("prefix") for r in rows if near_tie(r, tol)]
        eff = (sum(1 for r in rows if r.get("top1_agree")) + len(ties)) / float(n)
        det.append({"file": a.rel, "ref": d.get("ref"), "cand": d.get("cand"), "ref_model": d.get("ref_model"),
                    "cand_model": d.get("cand_model"), "n": n, "top_k": d.get("top_k"), "top1_agree": top1,
                    "near_tie_flips": len(ties), "near_tie_prefixes": ties[:50], "top1_agree_with_ties": eff,
                    "kl_topk_median": s.get("kl_topk_median"), "kl_topk_p95": s.get("kl_topk_p95"),
                    "max_abs_dlogprob_p99": s.get("max_abs_dlogprob_p99"),
                    "max_abs_dlogprob_max": s.get("max_abs_dlogprob_max")})
        worst = eff if worst is None else min(worst, eff)
        if eff < thr:
            fail.append("%s: top-1 agreement %.4f (%.4f strict + %d near-tie flip(s) within %.3g nats) < %.3f over %d "
                        "prefixes" % (a.rel, eff, top1, len(ties), tol, thr, n))
    meas = "; ".join("top-1 %.4f over %d prefixes (strict %.4f, near-tie flips %d at tolerance %.3g nats; KL top-K "
                     "median %s; max |d logprob| p99 %s)" % (x["top1_agree_with_ties"], x["n"], x["top1_agree"],
                                                              x["near_tie_flips"], tol, fmt(x["kl_topk_median"], 4),
                                                              fmt_g(x["max_abs_dlogprob_p99"])) for x in det) or None
    if fail:
        return res(FAIL, "threshold", "; ".join(fail), measured=meas, value=worst, threshold=thr, evidence=ev,
                   details={"comparisons": det})
    if nr:
        return res(NOT_RUN, "insufficient", "; ".join(nr), measured=meas, value=worst, threshold=thr, evidence=ev,
                   details={"comparisons": det})
    return res(PASS, "ok", "top-1 agreement >= %.3f (near-tie flips within %.3g nats count as agreement)" % (thr, tol),
               measured=meas, value=worst, threshold=thr, evidence=ev,
               details={"comparisons": det, "note": "ref/cand engines are identified only by URL and model id in the "
                                                    "artifact; the operator states them in tools/verify/README.md terms"})


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def crit_kv(inp, crit):
    arts = inp.art["slot_persist"]
    frac = float(crit["kv_restore_max_fraction_of_prefill"])
    if not arts:
        return res(NOT_RUN, "missing", "no tools/kv/slot_persist.py --out file under --runs", threshold=frac)
    ev = [a.rel for a in arts]
    mock = inp.mock_urls()
    fail, nr, det, worst = [], [], [], None
    for a in arts:
        d = a.data
        if norm_url(d.get("url")) in mock:
            return res(NOT_RUN, "mock", "mock: %s ran against %s, the URL of a mock bench run" % (a.rel, d.get("url")),
                       threshold=frac, evidence=ev)
        runs, errs, s = d.get("runs") or [], d.get("errors") or [], d.get("summary")
        if errs:
            fail.append("%s: %d repeat(s) failed, e.g. %s" % (a.rel, len(errs), str(errs[0])[:200]))
            continue
        if not runs or not isinstance(s, dict):
            nr.append("%s: no runs" % a.rel)
            continue
        try:
            mine = {"cold_prompt_ms_p50": _median([r["cold"]["prompt_ms"] for r in runs]),
                    "cold_prompt_n_p50": _median([r["cold"]["prompt_n"] for r in runs]),
                    "restore_s_p50": _median([r["restore"]["seconds"] for r in runs]),
                    "warm_prompt_ms_p50": _median([r["warm"]["prompt_ms"] for r in runs]),
                    "warm_prompt_n_p50": _median([r["warm"]["prompt_n"] for r in runs]),
                    "saved_bytes": _median([(r["save"]["response"] or {}).get("n_written") for r in runs]),
                    "all_continuations_identical": all(r["same_continuation"] for r in runs)}
        except (KeyError, TypeError) as e:
            nr.append("%s: runs malformed (%s)" % (a.rel, e))
            continue
        bad = [k for k, v in mine.items() if not (s.get(k) == v or (is_num(s.get(k)) and is_num(v) and close(s.get(k), v)))]
        if bad or s.get("runs") != len(runs):
            nr.append("%s: summary disagrees with runs at %s" % (a.rel, ", ".join(bad) or "runs"))
            continue
        cold_ms, cold_n = mine["cold_prompt_ms_p50"], mine["cold_prompt_n_p50"]
        warm_ms, warm_n, rest = mine["warm_prompt_ms_p50"], mine["warm_prompt_n_p50"], mine["restore_s_p50"]
        item = dict(mine, file=a.rel, url=d.get("url"), runs=len(runs))
        det.append(item)
        if len(runs) < crit["kv_min_runs"]:
            nr.append("%s: %d runs < %d" % (a.rel, len(runs), crit["kv_min_runs"]))
        if not mine["all_continuations_identical"]:
            fail.append("%s: continuation after restore differs from the cold continuation" % a.rel)
        if mine["saved_bytes"] is None:
            nr.append("%s: server did not report n_written for the saved slot" % a.rel)
        elif mine["saved_bytes"] <= 0:
            fail.append("%s: saved slot file is empty (n_written %s)" % (a.rel, mine["saved_bytes"]))
        if None in (cold_ms, cold_n, warm_ms, warm_n, rest):
            nr.append("%s: timings missing (cold prompt_ms/prompt_n, warm prompt_ms/prompt_n, restore seconds)" % a.rel)
            continue
        if cold_n < crit["kv_min_prompt_tokens"]:
            nr.append("%s: cold prompt %d tokens < %d (state too short to test)" % (a.rel, cold_n, crit["kv_min_prompt_tokens"]))
        if warm_n > crit["kv_warm_reprocess_max_fraction"] * cold_n:
            fail.append("%s: warm request re-processed %s of %s tokens (> %.0f%%): restored state not reused" % (
                a.rel, warm_n, cold_n, 100 * crit["kv_warm_reprocess_max_fraction"]))
        restored = rest + warm_ms / 1000.0
        ratio = restored / (cold_ms / 1000.0) if cold_ms else None
        item["restore_over_prefill"] = ratio
        if ratio is None:
            nr.append("%s: cold prefill time is 0" % a.rel)
        else:
            worst = ratio if worst is None else max(worst, ratio)
            if ratio > frac:
                fail.append("%s: restore %.3f s + warm prefill %.3f s = %.3f of the cold prefill %.3f s (> %.2f)" % (
                    a.rel, rest, warm_ms / 1000.0, ratio, cold_ms / 1000.0, frac))
    meas = "; ".join("cold %s tok: restore+warm %s s vs prefill %s s (ratio %s), %s bytes" % (
        fmt(x["cold_prompt_n_p50"], 0), fmt(x["restore_s_p50"] + (x["warm_prompt_ms_p50"] or 0) / 1000.0, 3)
        if x["restore_s_p50"] is not None else "n/a", fmt((x["cold_prompt_ms_p50"] or 0) / 1000.0, 3),
        fmt(x.get("restore_over_prefill"), 3), fmt(x["saved_bytes"], 0)) for x in det) or None
    if fail:
        return res(FAIL, "threshold", "; ".join(fail), measured=meas, value=worst, threshold=frac, evidence=ev,
                   details={"slots": det})
    if nr:
        return res(NOT_RUN, "insufficient", "; ".join(nr), measured=meas, value=worst, threshold=frac, evidence=ev,
                   details={"slots": det})
    return res(PASS, "ok", "restore + warm prefill <= %.2f of the cold prefill, continuation identical" % frac,
               measured=meas, value=worst, threshold=frac, evidence=ev, details={"slots": det})


def crit_roofline(inp, crit):
    arts = inp.art["roofline"]
    thr = float(crit["roofline_ratio_max"])
    if not arts:
        return res(NOT_RUN, "missing", "no tools/bench/vs_roofline.py --json output under --runs", threshold=thr)
    ev = [a.rel for a in arts]
    rows = [(a, r) for a in arts for r in a.data]
    mock = [(a, r) for a, r in rows if r.get("mock")]
    if mock:
        return res(NOT_RUN, "mock", "mock: %s has rows of mock runs (%s)" % (
            mock[0][0].rel, ", ".join(sorted({str(r.get("run")) for _, r in mock}))), threshold=thr, evidence=ev)
    g = bench_gate(inp.matrix, "run_matrix", "tools/bench/run_matrix.py")
    if g:
        g["reason"] = "matrix runs needed to locate the filled-context cells: " + g["reason"]
        return g
    ev += [p for r in inp.matrix for p in r.evidence]
    cells, _ = filled_target_cells(inp.matrix, crit, crit["decode_min_output_tokens"])
    if not cells:
        return res(NOT_RUN, "not_filled", "no filled-context single-stream matrix cell to compare with the bound",
                   threshold=thr, evidence=ev)
    fail, nr, det, worst = [], [], [], None
    for c in cells:
        t = cell_tag(c)
        if not measured_filled(c, crit):
            nr.append("%s: measured context below %d (prompt not filled)" % (t, crit["ctx_tokens"] - crit["ctx_slack_tokens"]))
            continue
        match = [(a, r) for a, r in rows if r.get("run") == c["run"].name and r.get("input") == c["input"]
                 and r.get("output") == c["output"] and r.get("state") == c["state"]]
        if not match:
            nr.append("%s: no vs_roofline row (run %s)" % (t, c["run"].name))
            continue
        for a, r in match:
            md, bound, ratio = r.get("measured_decode_p50"), r.get("bound"), r.get("ratio")
            if not (md == c["summary_decode_p50"] or (is_num(md) and is_num(c["summary_decode_p50"])
                                                      and close(md, c["summary_decode_p50"]))):
                nr.append("%s: %s measured %s != summary.json p50 %s" % (t, a.rel, md, c["summary_decode_p50"]))
                continue
            if not is_num(bound) or bound <= 0 or not is_num(md) or not is_num(ratio) or not close(ratio, md / bound):
                nr.append("%s: %s bound/ratio not usable (bound %s, ratio %s)" % (t, a.rel, bound, ratio))
                continue
            det.append({"cell": t, "measured_decode_p50": md, "bound": bound, "ratio": ratio, "ctx": r.get("ctx"),
                        "file": a.rel})
            worst = ratio if worst is None else max(worst, ratio)
            if ratio > thr:
                fail.append("%s: measured/bound %.3f > %.2f: the roofline inputs (hit share, bpw, bandwidths) are wrong "
                            "for this configuration; fix the inputs and rerun vs_roofline.py" % (t, ratio, thr))
    meas = "; ".join("%s: %.1f / %.1f tok/s = %.3f" % (x["cell"], x["measured_decode_p50"], x["bound"], x["ratio"])
                     for x in det) or None
    note = ("vs_roofline.py --json does not record --spec/--hit/--strategy/--expert-bpw/--dense-bpw; the bound's inputs "
            "must be stated next to the pack")
    if fail:   # invalid comparison, not a measured failure (VERIFICATION_PACK.md:23@d89e633 V9)
        return res(NOT_RUN, "invalid", "; ".join(fail), measured=meas, value=worst, threshold=thr, evidence=ev,
                   details={"rows": det, "note": note})
    if nr:
        return res(NOT_RUN, "not_filled" if any("not filled" in x for x in nr) else "insufficient", "; ".join(nr),
                   measured=meas, value=worst, threshold=thr, evidence=ev, details={"rows": det, "note": note})
    return res(PASS, "ok", "every filled-context cell compared, measured/bound <= %.2f" % thr, measured=meas,
               value=worst, threshold=thr, evidence=ev, details={"rows": det, "note": note})


def crit_hw(inp, crit):
    arts = inp.art["hw_facts"]
    if not arts:
        return res(NOT_RUN, "missing", "no scripts/diag/summarize.py --json output under --runs")
    ev = [a.rel for a in arts]
    synth = [a for a in arts if a.data.get("synthetic") is not False]
    if synth:
        return res(NOT_RUN, "synthetic", "synthetic: %s (summarize.py 'synthetic' = %s: the diag input is a test fixture)"
                   % (", ".join(a.rel for a in synth), ", ".join(str(a.data.get("synthetic")) for a in synth)), evidence=ev)
    facts, flags = [], []
    for a in arts:
        for f in a.data.get("facts") or []:
            if isinstance(f, dict):
                facts.append(dict(f, file=a.rel))
        for fl in a.data.get("red_flags") or []:
            if isinstance(fl, dict):
                flags.append(dict(fl, file=a.rel))
    missing, found = [], {}
    for pat in crit["hw_required_facts"]:
        hits = [f for f in facts if re.fullmatch(pat, str(f.get("fact"))) and f.get("label") in ("CONFIRMED", "PROVISIONAL")]
        if hits:
            found[pat] = hits
        else:
            missing.append(pat)
    det = {"red_flags": [{"flag": fl.get("flag"), "source": fl.get("source"), "file": fl["file"]} for fl in flags],
           "required": {p: [{"fact": f["fact"], "value": f.get("value"), "label": f.get("label"), "file": f["file"]}
                            for f in found.get(p, [])][:8] for p in crit["hw_required_facts"]}}
    hosts = sorted({str(f.get("value")) for f in facts if f.get("fact") == "hostname"})
    clients = sorted({str(((r.meta or {}).get("host") or {}).get("hostname")) for r in inp.matrix + inp.probes
                      if ((r.meta or {}).get("host") or {}).get("hostname")})
    det["hostnames"] = {"diag": hosts, "bench_client": clients, "match": bool(set(hosts) & set(clients)),
                        "note": "informational: the bench client may run on another machine, and the Proxmox host "
                                "and the VM have different names (VERIFICATION_PACK.md V8 asks a person to confirm)"}
    counts = []
    for f in facts:
        if f.get("fact") == "GPU count (nvidia-smi)" and f.get("label") == "CONFIRMED":
            try:
                counts.append(int(str(f.get("value")).strip()))
            except ValueError:
                pass
    key = {}
    for name in ("GPU count (nvidia-smi)", "memory channels populated", "AVX-512 (any avx512* flag)"):
        v = next((f for f in facts if f.get("fact") == name and f.get("label") != "UNKNOWN"), None)
        if v:
            key[name] = "%s (%s)" % (v.get("value"), v.get("label"))
    meas = "; ".join("%s = %s" % kv for kv in key.items()) + "; red flags: %d" % len(flags)
    exp = crit["hw_expected_gpu_count"]
    wrong = [c for c in counts if c != exp]
    if wrong:
        return res(FAIL, "threshold", "measured GPU count %s != expected %d" % (wrong, exp), measured=meas,
                   value=wrong[0], threshold=exp, evidence=ev, details=det)
    if missing:
        return res(NOT_RUN, "insufficient", "facts not measured (UNKNOWN or absent): " + ", ".join(missing), measured=meas,
                   threshold=exp, evidence=ev, details=det)
    return res(PASS, "ok", "all required facts measured; red flags listed for review (not judged automatically)",
               measured=meas, value=counts[0] if counts else None, threshold=exp, evidence=ev, details=det)


def crit_weights(inp, crit):
    mans, logs = inp.art["weights_manifest"], inp.art["weights_verify"]
    ev = [a.rel for a in mans + logs]
    if not mans or not logs:
        return res(NOT_RUN, "missing", "need both the weights manifest (scripts/offline/weights.py manifest ... -o FILE) "
                   "and the stdout of `weights.py verify MANIFEST --dest DIR` saved as .txt; found %d manifest(s), %d "
                   "verify log(s)" % (len(mans), len(logs)), evidence=ev)
    fail, nr, det = [], [], []
    ok, ok_size = {}, {}
    for a in logs:
        d = a.data
        if d["bad"]:
            fail.append("%s: %d file(s) failed verification, e.g. %s" % (a.rel, len(d["bad"]), d["bad"][0][:200]))
        if d["ok_n"] != d["total"]:
            fail.append("%s: verify: %d/%d files OK" % (a.rel, d["ok_n"], d["total"]))
        if d["other"]:
            nr.append("%s: unrecognised lines, e.g. %r" % (a.rel, d["other"][0][:160]))
        if len(d["ok"]) + len(d["ok_size"]) != d["ok_n"]:
            nr.append("%s: %d OK lines but the summary says %d" % (a.rel, len(d["ok"]) + len(d["ok_size"]), d["ok_n"]))
        ok.update(d["ok"])
        ok_size.update(d["ok_size"])
    total = 0
    for a in mans:
        m = a.data
        if not re.fullmatch(r"[0-9a-f]{40}", str(m.get("revision"))):
            nr.append("%s: revision %r is not a 40-hex commit" % (a.rel, m.get("revision")))
        files = m.get("files") or []
        if not files:
            nr.append("%s: no files" % a.rel)
        for f in files:
            path, sha, size = f.get("path"), f.get("sha256"), f.get("size")
            total += size or 0
            if sha:
                got = ok.get(path)
                if got is None:
                    nr.append("%s: %s not verified with sha256 in any verify log" % (a.rel, path))
                elif got["sha256"] != sha or (size is not None and got["size"] != size):
                    nr.append("%s: %s verified as sha256 %s / %d bytes, manifest says %s / %s (log from another "
                              "manifest?)" % (a.rel, path, got["sha256"], got["size"], sha, size))
            elif path not in ok_size and path not in ok:
                nr.append("%s: %s (no sha256 in manifest) not in any verify log" % (a.rel, path))
        det.append({"manifest": a.rel, "repo": m.get("repo"), "revision": m.get("revision"), "files": len(files)})
    names = {os.path.basename(f.get("path", "")) for a in mans for f in (a.data.get("files") or [])}
    audited = [os.path.basename((a.data.get("files") or [""])[0]) for a in inp.art["quant_audit"] if a.data.get("files")]
    details = {"manifests": det}
    if audited and not all(n in names for n in audited):
        details["warning"] = "quant audit file(s) %s are not in the weights manifest(s)" % ", ".join(
            n for n in audited if n not in names)
    meas = "; ".join("%s@%s: %d files" % (x["repo"], (x["revision"] or "")[:12], x["files"]) for x in det) + \
        ", %.2f GB" % (total / 1e9)
    if fail:
        return res(FAIL, "failed", "; ".join(fail), measured=meas, evidence=ev, details=details)
    if nr:
        return res(NOT_RUN, "inconsistent", "; ".join(nr[:5]) + (" (+%d more)" % (len(nr) - 5) if len(nr) > 5 else ""),
                   measured=meas, evidence=ev, details=details)
    return res(PASS, "ok", "every manifest file verified (size + sha256) in the verify log(s)", measured=meas,
               evidence=ev, details=details)


def crit_manual_faults(inp, crit):
    return res(NOT_RUN, "manual", "operational faults F1-F8 (docs/reassessment/DEPLOY.md section 6) are judged by a person "
               "from literal logs in docs/evidence/R5-faults-<date>.md; this pack does not judge them")


def crit_manual(inp, crit):
    return res(NOT_RUN, "manual", "the customer's evaluation set (docs/reassessment/R2-EVAL-SET.md) is graded by people "
               "and by the customer's own harness; this pack does not judge it")


EVAL = {
    "single_configuration": crit_single_configuration,
    "openai_api": crit_openai_api,
    "context_probes_full_length": crit_context_probes,
    "output_after_full_prompt": crit_output_after_full_prompt,
    "agent_api_probes": crit_agent_api_probes,
    "decode_filled_ctx_floor": lambda i, c: crit_decode(i, c, "decode_floor_tok_s"),
    "decode_filled_ctx_target": lambda i, c: crit_decode(i, c, "decode_target_tok_s"),
    "prefix_cache": crit_prefix_cache,
    "agent_loop": crit_agent_loop,
    "quant_min_bpw": crit_quant,
    "exact_routing_logprobs": crit_logprob,
    "kv_slot_persist": crit_kv,
    "decode_vs_roofline": crit_roofline,
    "hardware_facts": crit_hw,
    "determinism": crit_determinism,
    "weights_integrity": crit_weights,
    "manual_operational_faults": crit_manual_faults,
    "manual_quality_eval": crit_manual,
}


def synthetic_declaration(inp, crit):
    """N24 rule. None when the criteria do not declare a synthetic run; otherwise a dict: applied (bool), why, and the
    bench runs whose model description lacks the SYNTHETIC word (the declaration must match the artifacts: a real-model
    run cannot hide a behaviour FAIL by a flag in the criteria file)."""
    if not crit.get("synthetic_run"):
        return None
    runs = inp.matrix + inp.probes
    unmarked = [r.rel for r in runs if not synthetic_marked((r.meta or {}).get("model_desc"))]
    if not runs:
        return {"declared": True, "applied": False, "unmarked_runs": [],
                "why": "no matrix/probes run to check the declaration against; nothing declared not applicable"}
    if unmarked:
        return {"declared": True, "applied": False, "unmarked_runs": unmarked,
                "why": "declared synthetic, but the model description of %d run(s) has no positive SYNTHETIC marker: %s; the "
                       "behaviour criteria are judged as usual" % (len(unmarked), ", ".join(unmarked[:5]))}
    return {"declared": True, "applied": True, "unmarked_runs": [],
            "why": "declared synthetic in the criteria file and every bench run's model description says SYNTHETIC"}


def apply_synthetic(r, crit, decl):
    """Turn a PASS or a score FAIL of a model-behaviour criterion into NOT RUN 'not_applicable_synthetic', keeping the
    raw verdict. NOT RUN results (missing, mock, unparseable, ...) and FAILs with probe request errors are kept."""
    if not (decl and decl["applied"]) or r["id"] not in crit["synthetic_not_applicable"]:
        return r, None
    if r["status"] not in (PASS, FAIL) or (r["status"] == FAIL and (r.get("details") or {}).get("request_errors")):
        return r, None
    raw = {"id": r["id"], "raw_status": r["status"], "raw_reason_code": r["reason_code"], "raw_reason": r["reason"]}
    out = dict(r, status=NOT_RUN, reason_code="not_applicable_synthetic",
               reason="not applicable to the declared synthetic model (%s judges model behaviour); raw verdict %s: %s"
                      % (r["id"], r["status"], r["reason"]),
               details=dict(r.get("details") or {}, synthetic_raw=raw))
    return out, raw


def evaluate(inputs_dir, crit):
    inp = Inputs(inputs_dir)
    decl = synthetic_declaration(inp, crit)
    results, not_applicable = [], []
    for cid in CRITERIA_IDS:
        try:
            r = EVAL[cid](inp, crit)
        except Exception as e:  # noqa: BLE001 - a pack bug must never become a PASS
            r = res(NOT_RUN, "unparseable", "evaluator error: %s: %s" % (type(e).__name__, e))
        r = dict({"id": cid, "v": VNUM[cid], "title_ru": TITLE[cid], "mandatory": cid in crit["mandatory"]}, **r)
        r, raw = apply_synthetic(r, crit, decl)
        if raw:
            not_applicable.append(raw)
        results.append(r)
    mand = [r for r in results if r["mandatory"]]
    if not mand:
        overall, why = NOT_RUN, "no mandatory criteria configured"
    elif any(r["status"] == FAIL for r in mand):
        overall, why = FAIL, "mandatory criteria FAIL: " + ", ".join(r["id"] for r in mand if r["status"] == FAIL)
    elif all(r["status"] == PASS for r in mand) and not inp.unparseable:
        overall, why = PASS, "every mandatory criterion is PASS"
    else:
        nr = [r["id"] for r in mand if r["status"] != PASS]
        why = ("mandatory criteria NOT RUN: " + ", ".join(nr)) if nr else ""
        if inp.unparseable:
            why = (why + "; " if why else "") + "unparseable inputs: " + ", ".join(p for p, _ in inp.unparseable)
        overall = NOT_RUN
    if decl and overall == PASS:
        overall, why = NOT_RUN, "declared synthetic run: never acceptance evidence (every mandatory criterion PASS)"
    elif decl and overall == NOT_RUN:
        why = (why + "; " if why else "") + "declared synthetic run"
    markers = []
    for p in inp.files:
        if "synthetic" in p.lower():
            markers.append(p)
    for r in inp.matrix + inp.probes:
        for k in ("engine_desc", "model_desc"):
            v = (r.meta or {}).get(k)
            if v and "synthetic" in str(v).lower():
                markers.append("%s meta.%s = %s" % (r.rel, k, v))
    artifacts = ([{"path": r.rel, "kind": r.kind or "unknown_run", "mock": bool(r.mock_reasons()), "errors": r.errors}
                  for r in inp.matrix + inp.probes + inp.other_runs]
                 + [{"path": a.rel, "kind": k} for k, arts in inp.art.items() for a in arts])
    return {"overall": overall, "overall_reason": why, "criteria_results": results, "artifacts": artifacts,
            "unparseable": [{"path": p, "error": e} for p, e in inp.unparseable], "unclassified": inp.unclassified,
            "synthetic_markers": markers,
            "synthetic_run": dict(decl, not_applicable=not_applicable) if decl else {"declared": False}}


# ---------------------------------------------------------------------------- environment, manifest
def _cmd(cmd, cwd=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        return out.rstrip("\n") + ("\n[exit %d]" % p.returncode if p.returncode else "")
    except (OSError, subprocess.SubprocessError) as e:
        return "[not run: %s: %s]" % (type(e).__name__, e)


def env_text():
    lines = ["# env.txt: environment of the host that ASSEMBLED this pack, %s" % utc_now(),
             "# The server's own environment is in each run's meta.json (git, host, nvidia_smi) and in the diag archive.",
             "# repo: %s" % REPO, ""]
    for title, cmd in (("git rev-parse HEAD", ["git", "rev-parse", "HEAD"]),
                       ("git status --short", ["git", "status", "--short"]),
                       ("uname -a", ["uname", "-a"])):
        lines += ["$ " + title, _cmd(cmd, cwd=REPO) if shutil.which(cmd[0]) else "%s: not present" % cmd[0], ""]
    lines += ["$ python3 --version", "Python %s (%s)" % (sys.version.replace("\n", " "), sys.executable), ""]
    if shutil.which("nvidia-smi"):
        lines += ["$ nvidia-smi -L", _cmd(["nvidia-smi", "-L"]), ""]
    else:
        lines += ["$ nvidia-smi -L", "nvidia-smi: not present", ""]
    return "\n".join(lines)


def list_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            out.append(rel)
    return sorted(out)


def write_manifest(pack):
    lines = ["%s  %s" % (sha256_file(os.path.join(pack, rel)), rel) for rel in list_files(pack) if rel != MANIFEST]
    with open(os.path.join(pack, MANIFEST), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def check(pack):
    mpath = os.path.join(pack, MANIFEST)
    if not os.path.isfile(mpath):
        print("check: %s not found" % mpath)
        return 2
    problems, listed = [], set()
    with open(mpath, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            m = re.match(r"^([0-9a-f]{64})  (.+)$", line)
            if not m:
                problems.append("MALFORMED line %d: %r" % (n, line[:120]))
                continue
            digest, rel = m.group(1), m.group(2)
            if rel.startswith("/") or ".." in rel.split("/") or rel == MANIFEST:
                problems.append("UNSAFE path on line %d: %s" % (n, rel))
                continue
            listed.add(rel)
            p = os.path.join(pack, *rel.split("/"))
            if not os.path.isfile(p):
                problems.append("MISSING %s" % rel)
            elif sha256_file(p) != digest:
                problems.append("MODIFIED %s" % rel)
    for rel in list_files(pack):
        if rel != MANIFEST and rel not in listed:
            problems.append("EXTRA %s (not in the manifest)" % rel)
    for p in problems:
        print("check: " + p)
    vpath = os.path.join(pack, "verdict.json")
    overall = None
    try:
        overall = load_json(vpath).get("overall")
    except (OSError, ValueError, AttributeError):
        problems.append("verdict.json unreadable")
        print("check: verdict.json unreadable")
    print("check: %s: %d files listed, %d problem(s); manifest sha256 %s; recorded overall verdict: %s" % (
        pack, len(listed), len(problems), sha256_file(mpath), overall))
    return 1 if problems else 0


# ---------------------------------------------------------------------------- VERDICT.md
def md_cell(s):
    return str(s if s is not None else "—").replace("|", "\\|").replace("\n", " ")


def render_md(v):
    st_ru = {PASS: "PASS", FAIL: "FAIL", NOT_RUN: "NOT RUN"}
    L = ["# Verification pack: вердикт по критериям приёмки", "",
         "**Итог: %s** — %s" % (st_ru[v["overall"]], md_cell(v["overall_reason"])), ""]
    if v["synthetic_markers"]:
        L += ["> **ВНИМАНИЕ: во входных данных есть пометки SYNTHETIC — это тестовые данные, а не доказательство.** "
              "Пометки: %s" % md_cell(", ".join(v["synthetic_markers"][:8]) + (" …" if len(v["synthetic_markers"]) > 8 else "")), ""]
    sr = v.get("synthetic_run") or {}
    if sr.get("declared"):
        L += ["> **Прогон объявлен СИНТЕТИЧЕСКИМ в файле критериев (`synthetic_run: true`): итог никогда не PASS.** %s." % (
            "Правило применено" if sr.get("applied") else "Правило НЕ применено")
              + " %s" % md_cell(sr.get("why"))]
        for x in sr.get("not_applicable") or []:
            L.append("> - `%s`: не применимо к синтетической модели; исходный вердикт **%s** (%s): %s" % (
                x["id"], x["raw_status"], x["raw_reason_code"], md_cell(x["raw_reason"][:300])))
        L.append("")
    if v["criteria_deviations"]:
        L += ["> **Критерии отличаются от `tools/verify/criteria.default.json`: вердикт НЕ относится к требованиям заказчика "
              "в их исходном виде.** Изменено: %s" % md_cell(", ".join(
                  "%s: %s → %s" % (k, json.dumps(d["default"], ensure_ascii=False), json.dumps(d["used"], ensure_ascii=False))
                  for k, d in v["criteria_deviations"].items())), ""]
    if v["unparseable"]:
        L += ["> **Неразборчивые входные артефакты (итог не может быть PASS):** %s" % md_cell(
            "; ".join("%s: %s" % (u["path"], u["error"][:200]) for u in v["unparseable"])), ""]
    L += ["Собрано `tools/verify/pack.py` %s (UTC). Правила: PASS ставится только по артефакту из этого пакета, который "
          "разобран и сверен с сырыми данными; отсутствующий, неразборчивый или противоречивый артефакт и прогон на "
          "mock-сервере — NOT RUN, никогда не PASS. Декод засчитывается только при заполненном контексте. Итог PASS — "
          "только если все обязательные критерии PASS. Пути — относительно корня пакета; контрольные суммы — "
          "`MANIFEST.sha256`, проверка: `scripts/verify_pack.sh --check <пакет>`." % v["created_utc"], "",
          "| # | V | критерий | обязательный | статус | измерено | порог | доказательство | причина |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in enumerate(v["criteria_results"], 1):
        ev = r["evidence"]
        evs = ("`%s`" % ev[0] + (" (+%d)" % (len(ev) - 1) if len(ev) > 1 else "")) if ev else "—"
        thr = r["threshold"]
        thr = json.dumps(thr, ensure_ascii=False) if isinstance(thr, dict) else thr
        L.append("| %d | %s | %s (`%s`) | %s | **%s** | %s | %s | %s | %s: %s |" % (
            n, r["v"] or "—", md_cell(r["title_ru"]), r["id"], "да" if r["mandatory"] else "нет", st_ru[r["status"]],
            md_cell(r["measured"]), md_cell(thr), evs, md_cell(REASON_RU.get(r["reason_code"], r["reason_code"])),
            md_cell(r["reason"][:300])))
    L += ["", "V — строка `docs/reassessment/VERIFICATION_PACK.md` (V1–V11); «—» — сквозная проверка сборщика.",
          "", "## Подробности по критериям", ""]
    for r in v["criteria_results"]:
        L += ["### `%s` — %s" % (r["id"], r["status"]), "",
              "- критерий: %s" % r["title_ru"],
              "- причина (%s): %s" % (REASON_RU.get(r["reason_code"], r["reason_code"]), md_cell(r["reason"])),
              "- измерено: %s" % md_cell(r["measured"]),
              "- доказательство: %s" % (", ".join("`%s`" % e for e in r["evidence"]) or "нет"), ""]
    L += ["## Входные артефакты", "", "| путь | тип | mock |", "|---|---|---|"]
    for a in v["artifacts"]:
        L.append("| `%s` | %s | %s |" % (a["path"], a["kind"], {True: "да", False: "нет"}.get(a.get("mock"), "—")))
    if v["unclassified"]:
        L += ["", "Скопированы без разбора (тип не распознан): %s" % ", ".join("`%s`" % p for p in v["unclassified"][:40])
              + (" …" if len(v["unclassified"]) > 40 else "")]
    L += ["", "## Что этот пакет не проверяет автоматически", "",
          "- оценочный набор заказчика (R2.2, `docs/reassessment/R2-EVAL-SET.md`) и субъективное качество ответов — вручную;",
          "- что `ref` в `logprob_compare` — действительно эталонный движок со всеми экспертами на тех же весах: в "
          "артефакте есть только URL и id моделей;",
          "- что аудит битности относится к той же модели, что обслуживал сервер: связь только через `--model-desc` "
          "(предупреждение в `single_configuration`);",
          "- входные параметры границы roofline (`--spec`, `--hit`, `--strategy`, bpw): `vs_roofline.py --json` их не пишет;",
          "- красные флаги диагностики железа — перечислены в `verdict.json` (`hardware_facts.details.red_flags`), "
          "оцениваются человеком;",
          "- что все прогоны сделаны на одном запуске сервера: проверяется только совпадение модели и описаний.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------- build
def copy_inputs(src, dst, hardlink):
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        rel = os.path.relpath(dirpath, src)
        target = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target, exist_ok=True)
        for fn in sorted(filenames):
            s, d = os.path.join(dirpath, fn), os.path.join(target, fn)
            if not os.path.isfile(s):
                continue
            if hardlink:
                try:
                    os.link(s, d)
                    n += 1
                    continue
                except OSError:
                    pass
            shutil.copy2(s, d)
            n += 1
    return n


def build(runs, out, criteria_path=None, hardlink=False):
    runs, out = os.path.abspath(runs), os.path.abspath(out)
    if not os.path.isdir(runs):
        raise ValueError("--runs %s is not a directory" % runs)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise ValueError("--out %s exists and is not empty (a pack is never overwritten)" % out)
    if os.path.commonpath([runs, out]) in (runs, out):
        raise ValueError("--out and --runs must not contain each other")
    crit, deviations = load_criteria(criteria_path)
    os.makedirs(out, exist_ok=True)
    inputs = os.path.join(out, INPUTS)
    os.makedirs(inputs)
    n_files = copy_inputs(runs, inputs, hardlink)
    with open(os.path.join(out, "criteria.json"), "w", encoding="utf-8") as f:
        json.dump(crit, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(out, "env.txt"), "w", encoding="utf-8") as f:
        f.write(env_text())
    v = evaluate(inputs, crit)
    verdict = {"schema": SCHEMA, "created_utc": utc_now(), "runs_dir": runs, "n_input_files": n_files,
               "copy_mode": "hardlink" if hardlink else "copy",
               "pack_tool": {"path": "tools/verify/pack.py", "sha256": sha256_file(os.path.abspath(__file__))},
               "criteria_file": os.path.abspath(criteria_path) if criteria_path else DEFAULT_CRITERIA,
               "criteria_is_default": not deviations, "criteria_deviations": deviations}
    verdict.update(v)
    with open(os.path.join(out, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(out, "VERDICT.md"), "w", encoding="utf-8") as f:
        f.write(render_md(verdict))
    n = write_manifest(out)
    return verdict, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="directory with the operator's tool outputs (any layout)")
    ap.add_argument("--out", help="pack directory to create (must not exist or be empty)")
    ap.add_argument("--criteria", help="JSON with thresholds (default tools/verify/criteria.default.json)")
    ap.add_argument("--hardlink", action="store_true", help="hard-link inputs instead of copying (falls back to copy)")
    ap.add_argument("--check", metavar="PACK_DIR", help="re-verify MANIFEST.sha256 of an existing pack")
    a = ap.parse_args(argv)
    if a.check:
        if a.runs or a.out:
            ap.error("--check takes no --runs/--out")
        return check(a.check)
    if not (a.runs and a.out):
        ap.error("--runs and --out are required (or --check PACK_DIR)")
    try:
        v, n = build(a.runs, a.out, a.criteria, a.hardlink)
    except (ValueError, OSError) as e:
        print("verify_pack: error: %s" % e, file=sys.stderr)
        return 2
    for r in v["criteria_results"]:
        print("%-8s %-27s %s%s" % (r["status"], r["id"], "" if r["mandatory"] else "(optional) ",
                                   (r["reason"] or "")[:150]))
    print("overall: %s (%s); %d files in %s/%s" % (v["overall"], v["overall_reason"][:200], n, a.out, MANIFEST))
    return {PASS: 0, FAIL: 1, NOT_RUN: 3}[v["overall"]]


if __name__ == "__main__":
    sys.exit(main())
