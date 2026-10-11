#!/usr/bin/env python3
"""Tiny random-weight qwen4exp / glm5-next GGUF whose vocabulary has exactly one tokenization for every text.

The live B3 probe (run_n14.py, section live) forces each assistant turn byte for byte with a grammar and then asks
whether the next request's prompt starts with what the slot holds. The --text-vocab vocabulary of N11
(tools/tinygen/text_vocab.py on branch claude/n11-r1-tools-cpu-server @6680448) has merges (" a".." z", "\\n\\n"),
so a grammar-forced turn may be sampled as different tokens than the same bytes tokenize to in the next prompt, and the
probe would measure that instead of the template. This wrapper keeps N11's generator and replaces the vocabulary by:
  ids 0, 1, 2   <unk>, <s> (BOS), </s> (EOS)       (as text_vocab.py)
  ids 3..258    the 256 single bytes (GPT-2 byte-to-unicode characters, as text_vocab.py)
  id 259        one merge of the bytes 0x01 0x02, which never occur in the probe's texts (llama.cpp's BPE loader
                requires a non-empty string array of merges: src/llama-vocab.cpp:2110-2127 @79e2e74)
so every text tokenizes to its bytes. Special tokens of the real templates (<|im_end|>, <|user|>, ...) are plain bytes
here; their token-level behaviour needs the real tokenizer (UNKNOWN in N14).

Usage: bytes_vocab_model.py <N11 tools/tinygen dir> qwen4exp|glm5next <out.gguf>
       (PYTHONPATH must hold gguf-py of the llama.cpp checkout, as for the generators)
       N11's tinygen: git archive claude/n11-r1-tools-cpu-server tools/tinygen | tar -x -C <dir>
"""
import importlib
import sys


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in ("qwen4exp", "glm5next"):
        sys.exit(__doc__)
    tinygen, arch, out = sys.argv[1:]
    sys.path.insert(0, tinygen)
    tv = importlib.import_module("text_vocab")
    b2u = tv.bytes_to_unicode()

    def bytes_only_vocab():
        tokens = ["<unk>", "<s>", "</s>"] + [b2u[b] for b in range(256)]
        merges = ["%s %s" % (b2u[1], b2u[2])]
        tokens.append(b2u[1] + b2u[2])
        types = [2, 3, 3] + [1] * (len(tokens) - 3)
        return tokens, types, merges

    tv.byte_bpe_vocab = bytes_only_vocab
    tv.N_TEXT_VOCAB = len(bytes_only_vocab()[0])
    gen = importlib.import_module(arch + "_tiny")
    cfg = dict(gen.DEFAULT, n_vocab=tv.N_TEXT_VOCAB, ctx=16384)
    if arch == "glm5next":
        cfg["n_layer"] = 12  # the generator's CLI default: 3 DSA layers (3, 7, 11), as N7/N11
    gen.build(out, cfg=cfg, seed=0, text_vocab=True)
    print("wrote %s (n_vocab %d)" % (out, tv.N_TEXT_VOCAB))


if __name__ == "__main__":
    main()
