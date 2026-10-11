#!/usr/bin/env python3
"""N14 probe: model facts of the vendored config.json files, with the line each value sits on.
Run from the repository root: python3 docs/evidence/mutation-scripts/2026-10-11-n14-config-facts.py"""
import json
import pathlib
import re

ROOT = pathlib.Path(".")
for d, keys in (
        ("third_party/hf/glm-5.3-flash-chat", ["architectures", "n_group", "topk_group", "swiglu_limit", "index_topk",
                                              "first_k_dense_replace", "num_hidden_layers", "vocab_size",
                                              "max_position_embeddings", "vision_config", "quantization_config",
                                              "quant_method", "fmt", "activation_scheme", "weight_block_size"]),
        ("third_party/hf/qwen3.8-flash-next-chat", ["architectures", "language_model_only", "vision_config",
                                                   "num_hidden_layers", "vocab_size", "max_position_embeddings",
                                                   "quantization_config"])):
    p = ROOT / d / "config.json"
    pin = json.loads((ROOT / d / "PIN.json").read_text())
    print("== %s (%s @ %s)" % (p, pin["repo"], pin["revision"]))
    lines = p.read_text().splitlines()
    for k in keys:
        hits = [(i + 1, ln.strip()) for i, ln in enumerate(lines) if re.search(r'"%s"\s*:' % re.escape(k), ln)]
        print("  %-24s %s" % (k, "; ".join("%d: %s" % h for h in hits) if hits else "ABSENT"))
    c = json.loads(p.read_text())
    t = c.get("text_config", {})
    print("  architectures=%s text_config.model_type=%s top-level keys=%s" % (
        c.get("architectures"), t.get("model_type"), sorted(c)))
